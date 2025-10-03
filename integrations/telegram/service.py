"""Telegram bot service for GAJA server."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass(slots=True)
class TelegramConfig:
    """Configuration holder for Telegram integration."""

    enabled: bool = False
    bot_token: Optional[str] = None
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    default_user_id: Optional[str] = "1"
    allowed_chat_ids: set[str] = field(default_factory=set)
    chat_user_map: dict[str, str] = field(default_factory=dict)
    send_typing_action: bool = True

    def resolve_token(self) -> Optional[str]:
        if self.bot_token:
            return self.bot_token.strip()
        env_value = os.getenv(self.bot_token_env or "TELEGRAM_BOT_TOKEN")
        if env_value:
            return env_value.strip()
        return None

    def is_chat_allowed(self, chat_id: str) -> bool:
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids

    def resolve_user_id(self, chat_id: str) -> Optional[str]:
        if chat_id in self.chat_user_map:
            return self.chat_user_map[chat_id]
        return self.default_user_id


def load_telegram_config(raw: dict[str, Any]) -> TelegramConfig:
    """Create TelegramConfig from raw config dict."""

    enabled = bool(raw.get("enabled", False))
    token = raw.get("bot_token")
    token_env = raw.get("bot_token_env", "TELEGRAM_BOT_TOKEN")

    allowed_raw = raw.get("allowed_chat_ids", []) or []
    allowed_ids = {str(item) for item in allowed_raw if str(item).strip()}

    mapping_raw = raw.get("chat_user_map", {}) or {}
    mapping: dict[str, str] = {}
    for key, value in mapping_raw.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            mapping[key_str] = value_str

    default_user_id = raw.get("default_user_id")
    if default_user_id is not None:
        default_user_id = str(default_user_id).strip() or None

    return TelegramConfig(
        enabled=enabled,
        bot_token=token.strip() if isinstance(token, str) and token.strip() else None,
        bot_token_env=token_env,
        default_user_id=default_user_id,
        allowed_chat_ids=allowed_ids,
        chat_user_map=mapping,
        send_typing_action=bool(raw.get("send_typing_action", True)),
    )


class TelegramBotService:
    """Asynchronous Telegram bot bridge for GAJA."""

    def __init__(self, server_app: Any, config: TelegramConfig):
        self.server_app = server_app
        self.config = config
        self._application: Any = None
        self._polling_task: Optional[asyncio.Task] = None
        self._startup_lock = asyncio.Lock()
        self._chat_action_typing: Any = None
        self._stop_event: asyncio.Event | None = None
        self._pending_clarifications: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start Telegram polling if enabled and configured."""
        async with self._startup_lock:
            if self._application or not self.config.enabled:
                if not self.config.enabled:
                    logger.info("Telegram integration disabled; skipping startup")
                return

            token = self.config.resolve_token()
            if not token:
                logger.warning("Telegram bot token missing; integration disabled")
                return

            try:
                from telegram import Update
                from telegram.constants import ChatAction
                from telegram.ext import (
                    ApplicationBuilder,
                    CommandHandler,
                    ContextTypes,
                    MessageHandler,
                    filters,
                )
            except ImportError as exc:
                logger.error(
                    "python-telegram-bot is not installed; add it to requirements to enable Telegram integration: %s",
                    exc,
                )
                return

            self._chat_action_typing = getattr(ChatAction, "TYPING", None)

            self._application = ApplicationBuilder().token(token).build()

            self._application.add_handler(CommandHandler("start", self._handle_start))
            self._application.add_handler(
                MessageHandler(filters.TEXT & (~filters.COMMAND), self._handle_text)
            )

            self._stop_event = asyncio.Event()

            async def _polling_loop() -> None:
                try:
                    await self._application.initialize()
                    await self._application.start()
                    await self._application.updater.start_polling(
                        allowed_updates=Update.ALL_TYPES,
                        drop_pending_updates=True,
                    )
                    logger.info("Telegram bot polling started")
                    if self._stop_event:
                        await self._stop_event.wait()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(f"Telegram polling stopped due to error: {exc}")
                finally:
                    await self._shutdown_application()

            self._polling_task = asyncio.create_task(
                _polling_loop(), name="telegram-bot-polling"
            )

    async def stop(self) -> None:
        """Stop Telegram polling."""
        if self._application:
            with suppress(Exception):
                await self._application.updater.stop()

        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

    async def _shutdown_application(self) -> None:
        if not self._application:
            return
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()
        with suppress(Exception):
            await self._application.updater.stop()
        with suppress(Exception):
            await self._application.stop()
        with suppress(Exception):
            await self._application.shutdown()
        self._application = None
        logger.info("Telegram bot polling stopped")

    async def _handle_start(self, update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if not message:
            return
        chat_id = str(message.chat_id)
        if not self.config.is_chat_allowed(chat_id):
            await message.reply_text("Access denied.")
            return
        await message.reply_text("Hej! Wyślij wiadomość, a przekażę ją do GAJA.")

    async def _handle_text(self, update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if not message or not message.text:
            return

        chat_id = str(message.chat_id)
        if not self.config.is_chat_allowed(chat_id):
            logger.warning(f"Blocked message from unauthorized chat {chat_id}")
            await message.reply_text("Brak uprawnień do korzystania z tego bota.")
            return

        user_id = self.config.resolve_user_id(chat_id)
        if not user_id:
            logger.error("Cannot resolve GAJA user for chat %s", chat_id)
            await message.reply_text("Konfiguracja bota nie przypisuje użytkownika do tego czatu.")
            return

        query = message.text.strip()
        if not query:
            await message.reply_text("Wiadomość jest pusta.")
            return

        if self.config.send_typing_action and self._chat_action_typing:
            try:
                await message.chat.send_action(self._chat_action_typing)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Failed to send typing action: {exc}")

        ai_context: dict[str, Any] = {
            "user_id": user_id,
            "source": "telegram",
            "telegram_chat_id": chat_id,
        }

        pending = self._pending_clarifications.pop(chat_id, None)
        original_query = pending.get("original_query") if pending else query

        db_manager = getattr(self.server_app, "db_manager", None)
        if db_manager:
            try:
                history = await db_manager.get_user_history(user_id, limit=20)
                ai_context["history"] = history
            except Exception as exc:
                logger.debug(f"Unable to fetch history for user {user_id}: {exc}")

        ai_module = getattr(self.server_app, "ai_module", None)
        if not ai_module:
            await message.reply_text("Moduł AI nie jest dostępny.")
            return

        response_payload: Any = None
        try:
            if pending:
                ai_context.update(
                    {
                        "is_clarification_response": True,
                        "original_query": original_query,
                        "clarification_answer": query,
                        "pending_clarification": pending,
                    }
                )
            else:
                ai_context.setdefault("original_query", query)

            response_payload = await ai_module.process_query(query, ai_context)
        except Exception as exc:
            logger.error(f"Telegram AI query failed: {exc}")
            await message.reply_text("Przepraszam, wystąpił błąd podczas przetwarzania wiadomości.")
            return

        if await self._handle_possible_clarification(chat_id, message, response_payload, original_query):
            return

        if db_manager:
            await self._persist_history(db_manager, user_id, original_query, query, response_payload)

        text_response = self._normalize_ai_response(response_payload)
        if not text_response:
            text_response = "(brak odpowiedzi)"

        try:
            await message.reply_text(text_response)
        except Exception as exc:
            logger.error(f"Failed to send Telegram reply: {exc}")

    def _normalize_ai_response(self, payload: Any) -> str:
        """Normalize AI module response to plain text."""
        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    return self._normalize_ai_response(json.loads(stripped))
                except Exception:
                    return stripped
            return stripped

        if not isinstance(payload, dict):
            return str(payload) if payload is not None else ""

        inner = payload.get("response")
        if isinstance(inner, dict):
            if "text" in inner and inner.get("text"):
                return str(inner["text"])
            if "response" in inner and isinstance(inner["response"], str):
                return inner["response"]
            try:
                return json.dumps(inner, ensure_ascii=False)
            except Exception:
                return str(inner)

        if isinstance(inner, str):
            stripped = inner.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        candidate = parsed.get("text") or parsed.get("response")
                        if candidate:
                            return str(candidate)
                except Exception:
                    return stripped
            return stripped

        if payload.get("text"):
            return str(payload.get("text"))

        return ""

    async def _handle_possible_clarification(
        self,
        chat_id: str,
        message: Any,
        payload: Any,
        original_query: str,
    ) -> bool:
        """Handle clarification flow; returns True if handled."""

        if not isinstance(payload, dict):
            return False

        if payload.get("type") != "clarification_request":
            return False

        clarification = payload.get("clarification_data") or {}
        question = (
            clarification.get("question")
            or payload.get("question")
            or payload.get("message")
            or "Potrzebuję dodatkowych informacji."
        )

        stored_original = original_query or clarification.get("original_query") or ""
        if chat_id in self._pending_clarifications:
            previous = self._pending_clarifications[chat_id]
            if previous.get("original_query"):
                stored_original = previous["original_query"]

        self._pending_clarifications[chat_id] = {
            "original_query": stored_original or original_query,
            "clarification_data": clarification,
            "question": question,
        }

        try:
            await message.reply_text(question)
        except Exception as exc:
            logger.error(f"Failed to send clarification prompt: {exc}")
        return True

    async def _persist_history(
        self,
        db_manager: Any,
        user_id: str,
        original_query: str,
        current_query: str,
        payload: Any,
    ) -> None:
        """Persist Telegram conversation to history."""

        try:
            query_text = original_query or current_query
            if original_query and current_query and original_query.strip() != current_query.strip():
                query_text = f"{original_query}\n[telegram doprecyzowanie: {current_query}]"

            response_json = payload
            if isinstance(payload, dict):
                try:
                    response_json = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    response_json = str(payload)
            else:
                response_json = str(payload)

            await db_manager.save_interaction(user_id, query_text, response_json)
        except Exception as exc:
            logger.debug(f"Failed to persist Telegram history: {exc}")
