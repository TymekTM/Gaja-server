"""Telegram bot service for GAJA server."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import ssl
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, timezone, time
from pathlib import Path
from typing import Any, Optional

from zoneinfo import ZoneInfo

import httpx

from loguru import logger

_WEEKDAY_ALIASES: dict[int, tuple[str, ...]] = {
    0: ("poniedzialek", "pon"),
    1: ("wtorek", "wt"),
    2: ("sroda", "sr", "srod"),
    3: ("czwartek", "czw"),
    4: ("piatek", "pt"),
    5: ("sobota", "sob"),
    6: ("niedziela", "nd", "ndz", "niedz"),
}

_WEEKDAY_LABELS: dict[int, str] = {
    0: "poniedziałek",
    1: "wtorek",
    2: "środa",
    3: "czwartek",
    4: "piątek",
    5: "sobota",
    6: "niedziela",
}

_MONTH_LABELS: dict[int, str] = {
    1: "stycznia",
    2: "lutego",
    3: "marca",
    4: "kwietnia",
    5: "maja",
    6: "czerwca",
    7: "lipca",
    8: "sierpnia",
    9: "września",
    10: "października",
    11: "listopada",
    12: "grudnia",
}

_POLISH_ASCII_MAP = str.maketrans(
    {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "a",
        "Ć": "c",
        "Ę": "e",
        "Ł": "l",
        "Ń": "n",
        "Ó": "o",
        "Ś": "s",
        "Ź": "z",
        "Ż": "z",
    }
)


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
    timetable_group: str = "1"
    daily_brief_defaults: dict[str, Any] = field(default_factory=dict)
    daily_brief_storage_path: str = "user_data/telegram_daily_briefs.json"

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

    daily_brief_raw = raw.get("daily_brief", {}) or {}
    daily_brief_storage = str(
        daily_brief_raw.get("storage_path", "user_data/telegram_daily_briefs.json")
    ).strip() or "user_data/telegram_daily_briefs.json"

    daily_brief_defaults: dict[str, Any] = {}
    for key in (
        "enabled",
        "time",
        "name",
        "weather_location",
        "include_schedule",
        "include_fun",
        "send_if_weekend",
    ):
        if key in daily_brief_raw:
            daily_brief_defaults[key] = daily_brief_raw[key]

    return TelegramConfig(
        enabled=enabled,
        bot_token=token.strip() if isinstance(token, str) and token.strip() else None,
        bot_token_env=token_env,
        default_user_id=default_user_id,
        allowed_chat_ids=allowed_ids,
        chat_user_map=mapping,
        send_typing_action=bool(raw.get("send_typing_action", True)),
        timetable_group=str(raw.get("timetable_group", "1")).strip() or "1",
        daily_brief_defaults=daily_brief_defaults,
        daily_brief_storage_path=daily_brief_storage,
    )


@dataclass(slots=True)
class TimetableEntry:
    """Represents a single schedule entry."""

    start: datetime
    end: datetime
    subject: str
    location: Optional[str] = None
    teacher: Optional[str] = None
    group_label: Optional[str] = None
    raw_description: Optional[str] = None

    def format_line(self) -> str:
        """Return human readable summary for Telegram message."""

        time_part = f"{self.start:%H:%M}-{self.end:%H:%M}"
        details = self.subject.strip() if self.subject else "Zajęcia"

        extras: list[str] = []
        if self.location:
            extras.append(f"sala {self.location.strip()}")
        if self.teacher:
            extras.append(f"z {self.teacher.strip()}")
        if not extras and self.group_label:
            extras.append(self.group_label.strip())
        if extras:
            details = f"{details} ({', '.join(extras)})"

        return f"{time_part} - {details}"


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
        self._timetable_cache: dict[str, tuple[datetime, list["TimetableEntry"]]] = {}
        self._timetable_raw_cache: tuple[datetime, str] | None = None
        self._timetable_group = (self.config.timetable_group or "1").strip() or "1"
        try:
            self._timetable_zone = ZoneInfo("Europe/Warsaw")
        except Exception:  # pragma: no cover - fallback when tzdata missing
            self._timetable_zone = ZoneInfo("UTC")
        self._timetable_cache_ttl = timedelta(minutes=30)
        self._timetable_url = (
            "https://plan.polsl.pl/plan.php?type=2&id=343261191&cvsfile=true&wd=1"
        )
        self._timetable_http_client: httpx.AsyncClient | None = None
        self._timetable_max_attempts = 4
        self._timetable_retry_delay = 1.5
        self._timetable_use_http_fallback = False
        self._timetable_use_legacy_tls = False

        base_daily_defaults = {
            "enabled": False,
            "time": "08:00",
            "name": "",
            "weather_location": "",
            "include_schedule": True,
            "include_fun": False,
            "send_if_weekend": True,
            "preferred_variant": 2,
            "subject_overrides": [],
        }
        if self.config.daily_brief_defaults:
            for key, value in self.config.daily_brief_defaults.items():
                if value is not None:
                    base_daily_defaults[key] = value
        base_daily_defaults["time"] = self._sanitize_brief_time(
            str(base_daily_defaults.get("time", "08:00"))
        )
        base_daily_defaults["enabled"] = bool(base_daily_defaults.get("enabled", False))
        base_daily_defaults["include_schedule"] = bool(
            base_daily_defaults.get("include_schedule", True)
        )
        base_daily_defaults["include_fun"] = bool(
            base_daily_defaults.get("include_fun", False)
        )
        base_daily_defaults["send_if_weekend"] = bool(
            base_daily_defaults.get("send_if_weekend", True)
        )

        self._daily_brief_defaults = base_daily_defaults
        self._daily_brief_lock = asyncio.Lock()
        root_dir = Path(__file__).resolve().parent.parent.parent
        storage_path = Path(self.config.daily_brief_storage_path).expanduser()
        if not storage_path.is_absolute():
            storage_path = root_dir / storage_path
        self._daily_brief_storage_path = storage_path
        self._daily_briefs_config: dict[str, dict[str, Any]] = self._load_daily_brief_settings()
        self._daily_brief_tasks: dict[str, asyncio.Task] = {}
        self._weather_module: Any | None = None

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
            self._application.add_handler(CommandHandler("plan", self._handle_plan_command))
            self._application.add_handler(
                CommandHandler("brief", self._handle_daily_brief_command)
            )
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

            self._restore_daily_brief_tasks()

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

        await self._reset_timetable_http_client()

        for task in list(self._daily_brief_tasks.values()):
            task.cancel()
            with suppress(Exception):
                await task
        self._daily_brief_tasks.clear()

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

        await self._reset_timetable_http_client()

    async def _handle_start(self, update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if not message:
            return
        chat_id = str(message.chat_id)
        if not self.config.is_chat_allowed(chat_id):
            await message.reply_text("Access denied.")
            return
        await message.reply_text("Hej! Wyślij wiadomość, a przekażę ją do GAJA.")

    async def _handle_plan_command(self, update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if not message:
            return

        chat_id = str(message.chat_id)
        if not self.config.is_chat_allowed(chat_id):
            await message.reply_text("Brak uprawnień do korzystania z tego bota.")
            return

        args = getattr(context, "args", None) if context else None
        args_text = " ".join(args) if args else ""
        day_selector = self._detect_day_selector(args_text, allow_without_keyword=True) or "tomorrow"
        await self._send_timetable_reply(message, day_selector)

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

        day_selector = self._detect_day_selector(query)
        if day_selector:
            await self._send_timetable_reply(message, day_selector)
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
        original_query = (pending.get("original_query") if pending else query) or query

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

    def _normalize_for_matching(self, text: str) -> str:
        if not text:
            return ""
        translated = text.translate(_POLISH_ASCII_MAP)
        normalized = unicodedata.normalize("NFKD", translated)
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        ascii_text = stripped.encode("ascii", "ignore").decode("ascii")
        return ascii_text.lower()

    def _sanitize_brief_variant(self, value: Any) -> int:
        try:
            variant = int(value)
        except (TypeError, ValueError):
            variant = 2
        return max(0, min(variant, 3))

    def _normalize_subject_override_entry(self, item: Any) -> Optional[dict[str, str]]:
        if not isinstance(item, dict):
            return None
        pattern = str(item.get("pattern") or "").strip()
        if not pattern:
            return None
        name = str(item.get("name") or item.get("label") or "").strip()
        room = str(item.get("room") or item.get("location") or "").strip()
        notes = str(item.get("notes") or "").strip()
        return {
            "pattern": pattern,
            "name": name,
            "room": room,
            "notes": notes,
        }

    def _normalize_subject_overrides(self, raw: Any) -> list[dict[str, str]]:
        if not raw:
            return []
        overrides: list[dict[str, str]] = []
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw:
                normalized = self._normalize_subject_override_entry(item)
                if normalized:
                    overrides.append(normalized)
        return overrides

    def _get_subject_overrides(self, settings: dict[str, Any]) -> list[dict[str, str]]:
        overrides = settings.get("subject_overrides") or []
        if not isinstance(overrides, list):
            overrides = self._normalize_subject_overrides(overrides)
        return overrides

    def _find_subject_override_index(
        self, overrides: list[dict[str, str]], pattern: str
    ) -> Optional[int]:
        if not pattern:
            return None
        normalized_pattern = self._normalize_for_matching(pattern)
        for idx, override in enumerate(overrides):
            current_pattern = self._normalize_for_matching(override.get("pattern", ""))
            if current_pattern == normalized_pattern:
                return idx
        return None

    async def _set_subject_override(
        self, chat_id: str, pattern: str, name: str, room: str, notes: str
    ) -> None:
        settings = self._ensure_brief_settings(chat_id)
        overrides = self._get_subject_overrides(settings)
        normalized_entry = self._normalize_subject_override_entry(
            {"pattern": pattern, "name": name, "room": room, "notes": notes}
        )
        if not normalized_entry:
            return
        index = self._find_subject_override_index(overrides, normalized_entry["pattern"])
        if index is None:
            overrides.append(normalized_entry)
        else:
            overrides[index] = normalized_entry
        settings["subject_overrides"] = overrides
        await self._save_daily_brief_settings()
        if settings.get("enabled"):
            self._schedule_daily_brief_task(chat_id)

    async def _remove_subject_override(self, chat_id: str, pattern: str) -> bool:
        settings = self._ensure_brief_settings(chat_id)
        overrides = self._get_subject_overrides(settings)
        index = self._find_subject_override_index(overrides, pattern)
        if index is None:
            return False
        overrides.pop(index)
        settings["subject_overrides"] = overrides
        await self._save_daily_brief_settings()
        if settings.get("enabled"):
            self._schedule_daily_brief_task(chat_id)
        return True

    def _format_subject_overrides(self, settings: dict[str, Any]) -> str:
        overrides = self._get_subject_overrides(settings)
        if not overrides:
            return "(brak mapowań)"
        lines = []
        for item in overrides:
            parts = [item.get("pattern") or "?"]
            if item.get("name"):
                parts.append(f"→ {item['name']}")
            if item.get("room"):
                parts.append(f"sala {item['room']}")
            if item.get("notes"):
                parts.append(f"notatka: {item['notes']}")
            lines.append(" | ".join(part for part in parts if part))
        return "\n".join(lines)

    def _detect_day_selector(self, text: str, allow_without_keyword: bool = False) -> Optional[str]:
        normalized = self._normalize_for_matching(text)
        if not normalized:
            return None

        plan_match = re.search(r"\b(plan|plan lekcji|rozklad|rozklad zajec|schedule)\b", normalized)
        # Allow explicit /plan command arguments or forced detection to skip keyword requirement.
        if not plan_match and not text.startswith("/") and not allow_without_keyword:
            return None

        weekday_idx = self._match_weekday(normalized)
        if weekday_idx is not None:
            return f"weekday:{weekday_idx}"

        if re.search(r"\bpojutrze\b", normalized) or "day after" in normalized:
            return "day_after"
        if re.search(r"\bjutro\b", normalized) or "tomorrow" in normalized:
            return "tomorrow"
        if re.search(r"\bdzisiaj\b", normalized) or re.search(r"\bdzis\b", normalized) or "today" in normalized:
            return "today"

        if plan_match or allow_without_keyword:
            return "tomorrow"
        return None

    def _match_weekday(self, normalized_text: str) -> Optional[int]:
        for weekday, aliases in _WEEKDAY_ALIASES.items():
            for alias in aliases:
                if re.search(rf"\b{alias}\b", normalized_text):
                    return weekday
        return None

    def _sanitize_brief_time(self, value: str) -> str:
        if not value:
            return "08:00"
        match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?", value.strip())
        if not match:
            return "08:00"
        hour = max(0, min(23, int(match.group(1))))
        minute = int(match.group(2) or 0)
        minute = max(0, min(59, minute))
        return f"{hour:02d}:{minute:02d}"

    def _default_brief_settings(self) -> dict[str, Any]:
        settings = dict(self._daily_brief_defaults)
        settings.setdefault("enabled", False)
        settings.setdefault("time", "08:00")
        settings.setdefault("name", "")
        settings.setdefault("weather_location", "")
        settings.setdefault("include_schedule", True)
        settings.setdefault("include_fun", False)
        settings.setdefault("send_if_weekend", True)
        settings.setdefault("preferred_variant", 2)
        settings.setdefault("subject_overrides", [])
        settings.setdefault("last_sent_at", None)
        settings.setdefault("last_sent_date", None)

        settings["time"] = self._sanitize_brief_time(str(settings.get("time", "08:00")))
        settings["name"] = str(settings.get("name") or "").strip()
        settings["weather_location"] = str(settings.get("weather_location") or "").strip()
        settings["enabled"] = bool(settings.get("enabled", False))
        settings["include_schedule"] = bool(settings.get("include_schedule", True))
        settings["include_fun"] = bool(settings.get("include_fun", False))
        settings["send_if_weekend"] = bool(settings.get("send_if_weekend", True))
        settings["preferred_variant"] = self._sanitize_brief_variant(settings.get("preferred_variant", 2))
        settings["subject_overrides"] = self._normalize_subject_overrides(
            settings.get("subject_overrides")
        )
        settings["last_sent_at"] = settings.get("last_sent_at")
        settings["last_sent_date"] = settings.get("last_sent_date")
        return settings

    def _normalize_brief_settings(self, raw: Any) -> dict[str, Any]:
        settings = self._default_brief_settings()
        if not isinstance(raw, dict):
            return settings

        if "enabled" in raw:
            settings["enabled"] = bool(raw.get("enabled"))
        if "include_schedule" in raw:
            settings["include_schedule"] = bool(raw.get("include_schedule"))
        if "include_fun" in raw:
            settings["include_fun"] = bool(raw.get("include_fun"))
        if "send_if_weekend" in raw:
            settings["send_if_weekend"] = bool(raw.get("send_if_weekend"))
        if "time" in raw:
            settings["time"] = self._sanitize_brief_time(str(raw.get("time", "08:00")))
        if "name" in raw:
            settings["name"] = str(raw.get("name") or "").strip()
        if "weather_location" in raw:
            settings["weather_location"] = str(raw.get("weather_location") or "").strip()
        if raw.get("last_sent_at"):
            settings["last_sent_at"] = str(raw.get("last_sent_at"))
        if raw.get("last_sent_date"):
            settings["last_sent_date"] = str(raw.get("last_sent_date"))
        if "preferred_variant" in raw:
            settings["preferred_variant"] = self._sanitize_brief_variant(
                raw.get("preferred_variant")
            )
        if "subject_overrides" in raw:
            settings["subject_overrides"] = self._normalize_subject_overrides(
                raw.get("subject_overrides")
            )
        return settings

    def _load_daily_brief_settings(self) -> dict[str, dict[str, Any]]:
        path = self._daily_brief_storage_path
        try:
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Unable to load Telegram daily brief settings: {exc}")
            return {}

        if not isinstance(data, dict):
            logger.warning("Telegram daily brief settings file had unexpected structure")
            return {}

        result: dict[str, dict[str, Any]] = {}
        for chat_id, raw_settings in data.items():
            result[str(chat_id)] = self._normalize_brief_settings(raw_settings)
        return result

    def _ensure_brief_settings(self, chat_id: str) -> dict[str, Any]:
        if chat_id not in self._daily_briefs_config:
            self._daily_briefs_config[chat_id] = self._default_brief_settings()
        return self._daily_briefs_config[chat_id]

    def _restore_daily_brief_tasks(self) -> None:
        for chat_id, settings in self._daily_briefs_config.items():
            if settings.get("enabled"):
                self._schedule_daily_brief_task(chat_id)

    def _schedule_daily_brief_task(self, chat_id: str) -> None:
        self._cancel_daily_brief_task(chat_id)
        settings = self._daily_briefs_config.get(chat_id)
        if not settings or not settings.get("enabled"):
            return
        task = asyncio.create_task(
            self._daily_brief_loop(chat_id), name=f"telegram-daily-brief-{chat_id}"
        )
        self._daily_brief_tasks[chat_id] = task

    def _cancel_daily_brief_task(self, chat_id: str) -> None:
        task = self._daily_brief_tasks.pop(chat_id, None)
        if task:
            task.cancel()

    def _calculate_next_brief_run(self, settings: dict[str, Any]) -> datetime:
        now = self._current_time()
        sanitized_time = str(settings.get("time", "08:00"))
        hour, minute = sanitized_time.split(":")
        hour_int = int(hour)
        minute_int = int(minute)

        def adjust(candidate: datetime) -> datetime:
            if settings.get("send_if_weekend", True):
                return candidate
            weekday = candidate.weekday()
            if weekday < 5:
                return candidate
            days_ahead = (7 - weekday) % 7 or 1
            if weekday == 5:  # Saturday -> +2 days
                days_ahead = 2
            candidate = candidate + timedelta(days=days_ahead)
            return candidate.replace(hour=hour_int, minute=minute_int, second=0, microsecond=0)

        target = now.replace(hour=hour_int, minute=minute_int, second=0, microsecond=0)
        target = adjust(target)

        if target <= now:
            if settings.get("last_sent_date") != now.date().isoformat():
                candidate = adjust(now + timedelta(seconds=5))
                if candidate > now:
                    return candidate
            next_day = now + timedelta(days=1)
            next_day = next_day.replace(hour=hour_int, minute=minute_int, second=0, microsecond=0)
            target = adjust(next_day)
        return target

    async def _daily_brief_loop(self, chat_id: str) -> None:
        try:
            while True:
                settings = self._daily_briefs_config.get(chat_id)
                if not settings or not settings.get("enabled"):
                    break

                next_run = self._calculate_next_brief_run(settings)
                wait_seconds = max((next_run - self._current_time()).total_seconds(), 0)
                try:
                    await asyncio.sleep(wait_seconds)
                except asyncio.CancelledError:
                    raise

                settings = self._daily_briefs_config.get(chat_id)
                if not settings or not settings.get("enabled"):
                    continue

                try:
                    messages = await self._generate_daily_brief_messages(
                        chat_id, settings, count=1
                    )
                    if not messages:
                        continue
                    if not self._application or not getattr(self._application, "bot", None):
                        logger.warning("Telegram application bot not ready for daily brief send")
                        await asyncio.sleep(5)
                        continue
                    for message_text in messages:
                        await self._application.bot.send_message(
                            chat_id=chat_id, text=message_text
                        )
                        await asyncio.sleep(0.5)
                    now = self._current_time()
                    settings["last_sent_at"] = now.isoformat()
                    settings["last_sent_date"] = now.date().isoformat()
                    await self._save_daily_brief_settings()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(f"Failed to send daily brief to chat {chat_id}: {exc}")
                    await asyncio.sleep(10)
        finally:
            self._daily_brief_tasks.pop(chat_id, None)

    async def _generate_daily_brief_messages(
        self, chat_id: str, settings: dict[str, Any], *, count: int
    ) -> list[str]:
        components = await self._collect_brief_components(chat_id, settings)
        messages: list[str] = []
        if not components:
            return messages

        seed = random.randint(100000, 999999)
        target = max(1, count)
        preferred_variant = self._sanitize_brief_variant(
            settings.get("preferred_variant", 2)
        )
        variants_to_try: list[int] = [preferred_variant]
        if target > 1:
            variants_to_try.extend(v for v in range(4) if v != preferred_variant)

        for variant in variants_to_try:
            rendered = await self._render_daily_brief_text(
                chat_id, components, variant, seed
            )
            if rendered:
                normalized = rendered.strip()
                if normalized and normalized not in messages:
                    messages.append(normalized)
            if len(messages) >= target:
                break

        if not messages:
            fallback = self._render_brief_fallback(
                components, preferred_variant, seed
            )
            if fallback:
                messages.append(fallback)

        return messages[:target]

    async def _collect_brief_components(
        self, chat_id: str, settings: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        now = self._current_time()
        name = settings.get("name") or ""
        location = settings.get("weather_location") or ""

        weather_line: Optional[str] = None
        if location:
            weather_line = await self._build_weather_line(chat_id, location)

        schedule_lines: list[str] = []
        schedule_details: list[dict[str, Any]] = []
        if settings.get("include_schedule", True):
            schedule_lines, schedule_details = await self._prepare_schedule_brief(
                now.date(), settings
            )

        fun_line: Optional[str] = None
        if settings.get("include_fun", False):
            fun_line = self._choose_fun_line()

        return {
            "name": name,
            "date_text": self._format_polish_date(now),
            "timestamp_iso": now.isoformat(),
            "weather_line": weather_line,
            "schedule_lines": schedule_lines,
            "schedule_details": schedule_details,
            "fun_line": fun_line,
            "chat_id": chat_id,
        }

    async def _prepare_schedule_brief(
        self, target_date: date, settings: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        overrides = self._get_subject_overrides(settings)
        include_weekend = settings.get("send_if_weekend", True)
        weekend = target_date.weekday() >= 5

        if weekend and not include_weekend:
            return ["Weekend — żadnych zaplanowanych zajęć."], []

        try:
            entries = await self._get_timetable_entries(target_date)
        except httpx.HTTPError as exc:
            logger.error(f"Unable to download timetable for brief: {exc}")
            return ["Nie udało się pobrać planu zajęć na dziś."], []
        except Exception as exc:  # pragma: no cover - defensive log
            logger.error(f"Unexpected timetable error for brief: {exc}")
            return ["Wystąpił błąd podczas pobierania planu zajęć."], []

        if weekend and not entries:
            return ["Weekend — żadnych zaplanowanych zajęć."], []

        if not entries:
            return ["Plan dnia: brak zaplanowanych zajęć."], []

        sorted_entries = sorted(entries, key=lambda item: item.start)
        lines: list[str] = ["Plan dnia:"]
        details: list[dict[str, Any]] = []
        for entry in sorted_entries:
            summary, detail = self._summarize_schedule_entry(entry, overrides)
            lines.append(summary)
            details.append(detail)
        return lines, details

    def _summarize_schedule_entry(
        self, entry: "TimetableEntry", overrides: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        subject_raw = (entry.subject or "Zajęcia").strip()
        override = self._match_subject_override(subject_raw, overrides, (entry.location or ""))
        name = (override.get("name") if override else "") or subject_raw

        room_override = (override.get("room") if override else "") or ""
        location = (entry.location or "").strip()
        room = room_override or location

        extras: list[str] = []
        if room:
            extras.append(f"sala {room}")
        if entry.teacher:
            extras.append(f"z {entry.teacher.strip()}")

        description = name
        if extras:
            description = f"{description} ({', '.join(extras)})"

        note = (override.get("notes") if override else "") or ""
        if note:
            description = f"{description} — {note}"

        time_range = f"{entry.start:%H:%M}-{entry.end:%H:%M}"
        summary = f"{time_range}: {description}"

        detail = {
            "time_range": time_range,
            "start": entry.start.isoformat(),
            "end": entry.end.isoformat(),
            "subject": name,
            "subject_raw": subject_raw,
            "room": room,
            "location": location,
            "teacher": entry.teacher or "",
            "group": entry.group_label or "",
            "notes": note,
        }
        return summary, detail

    def _match_subject_override(
        self,
        subject_text: str,
        overrides: list[dict[str, str]],
        location_text: str = "",
    ) -> Optional[dict[str, str]]:
        if not overrides:
            return None
        normalized_subject = self._normalize_for_matching(subject_text)
        normalized_location = self._normalize_for_matching(location_text)
        for override in overrides:
            pattern = override.get("pattern") or ""
            if not pattern:
                continue
            if self._normalize_for_matching(pattern) in normalized_subject:
                return override
            if normalized_location and self._normalize_for_matching(pattern) in normalized_location:
                return override
        return None

    async def _render_daily_brief_text(
        self,
        chat_id: str,
        components: dict[str, Any],
        variant: int,
        seed: int,
    ) -> Optional[str]:
        ai_text = await self._call_ai_for_brief(chat_id, components, variant, seed)
        if ai_text:
            return ai_text
        return self._render_brief_fallback(components, variant, seed)

    async def _call_ai_for_brief(
        self,
        chat_id: str,
        components: dict[str, Any],
        variant: int,
        seed: int,
    ) -> Optional[str]:
        ai_module = getattr(self.server_app, "ai_module", None)
        if not ai_module:
            return None

        user_id = self.config.resolve_user_id(chat_id) or "0"
        tone_variants = [
            "energetyczny i motywujący",
            "spokojny i wspierający",
            "dynamiczny i rzeczowy",
            "z humorem i lekkością",
        ]
        tone = tone_variants[variant % len(tone_variants)]

        prompt_payload = {
            "seed": seed,
            "variant": variant,
            "tone": tone,
            "components": {
                "name": components.get("name"),
                "date_text": components.get("date_text"),
                "weather_line": components.get("weather_line"),
                "schedule_lines": components.get("schedule_lines"),
                "schedule_details": components.get("schedule_details"),
                "fun_line": components.get("fun_line"),
            },
        }

        prompt = (
            "Stwórz krótki poranny brief po polsku dla użytkownika.\n"
            "- Użyj tylko danych z JSON-u i nie dopowiadaj własnych faktów.\n"
            "- Zachowaj ton: {tone}.\n"
            "- Zwrot ma się składać z 2-4 zdań, bez list, nagłówków ani numeracji.\n"
            "- Jeśli `weather_line` ma wartość, wpleć ją naturalnie w jednym zdaniu.\n"
            "- Jeśli `schedule_lines` zawiera dane, streść je w jednym zdaniu, nie kopiuj dosłownie listy.\n"
            "- Wykorzystaj `schedule_details` (czas, sala, nazwa) aby przygotować zwięzłe podsumowanie zajęć.\n"
            "- Nie używaj słów typu 'Wariant' ani oznaczeń markdown.\n"
            "- Zakończ krótką zachętą lub dobrym życzeniem."
        ).format(tone=tone)

        query = (
            f"{prompt}\nDANE_JSON = {json.dumps(prompt_payload, ensure_ascii=False)}"
        )

        context = {
            "user_id": user_id,
            "source": "telegram_daily_brief",
            "telegram_chat_id": chat_id,
            "brief_variant": variant,
        }

        try:
            response = await ai_module.process_query(query, context)
            text = self._normalize_ai_response(response)
            if text:
                cleaned = self._postprocess_ai_brief(text)
                if cleaned:
                    return cleaned
        except Exception as exc:
            logger.error(f"AI brief generation failed: {exc}")
        return None

    def _render_brief_fallback(
        self, components: dict[str, Any], variant: int, seed: int
    ) -> str:
        random.seed(seed + variant)
        name = components.get("name") or ""
        greeting_options = [
            "Dzień dobry",
            "Hej",
            "Witaj",
            "Cześć",
        ]
        greeting = random.choice(greeting_options)
        if name:
            greeting_sentence = f"{greeting} {name}!"
        else:
            greeting_sentence = f"{greeting}!"

        sentences: list[str] = [greeting_sentence]

        date_text = components.get("date_text")
        if date_text:
            date_prefixes = [
                "Jest",
                "Mamy",
                "Na zegarze",
                "Dzisiejsza data to",
            ]
            sentences.append(f"{random.choice(date_prefixes)} {date_text}.")

        weather_line = components.get("weather_line")
        if weather_line:
            sentences.append(self._ensure_sentence(weather_line))

        schedule_details = components.get("schedule_details") or []
        if schedule_details:
            fragments = []
            for detail in schedule_details[:3]:
                fragment = detail.get("subject", detail.get("subject_raw", "zajęcia"))
                time_range = detail.get("time_range")
                room = detail.get("room")
                note = detail.get("notes")
                if time_range:
                    fragment = f"{time_range} {fragment}"
                if room:
                    fragment = f"{fragment} (sala {room})"
                if note:
                    fragment = f"{fragment} — {note}"
                fragments.append(fragment)
            if fragments:
                schedule_intro_options = [
                    "Dziś czekają Cię",
                    "W planie masz",
                    "Przed Tobą",
                    "Zaplanowane są",
                ]
                sentences.append(
                    f"{random.choice(schedule_intro_options)} "
                    + "; ".join(fragments)
                    + "."
                )

        fun_line = components.get("fun_line")
        if fun_line:
            sentences.append(self._ensure_sentence(fun_line))
        else:
            closing_options = [
                "Powodzenia – działaj swoim tempem.",
                "Złap oddech między zadaniami i trzymaj się dzielnie.",
                "Niech ten dzień przyniesie Ci sporo satysfakcji.",
                "Pamiętaj o drobnej przerwie dla siebie.",
            ]
            sentences.append(random.choice(closing_options))

        random.seed()
        return " ".join(sentence.strip() for sentence in sentences if sentence).strip()

    def _ensure_sentence(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        if stripped[-1] not in ".!?":
            stripped += "."
        return stripped

    def _postprocess_ai_brief(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        lines = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("wariant"):
                continue
            if line.startswith(('*', '-', '•', '→')):
                line = line.lstrip('*-•→ ').strip()
            lines.append(line)

        if not lines:
            return ""

        text_joined = " ".join(lines)
        sentences = re.split(r"(?<=[.!?])\s+", text_joined)
        filtered = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            filtered.append(sentence)
            if len(filtered) >= 4:
                break

        final_text = " ".join(filtered)
        return final_text.strip()

    def _format_polish_date(self, dt: datetime) -> str:
        weekday = _WEEKDAY_LABELS.get(dt.weekday(), dt.strftime("%A"))
        month = _MONTH_LABELS.get(dt.month, dt.strftime("%B"))
        return f"{weekday.capitalize()} {dt.day} {month} {dt.year}"

    async def _build_weather_line(self, chat_id: str, location: str) -> str | None:
        try:
            if self._weather_module is None:
                from modules.weather_module import WeatherModule

                self._weather_module = WeatherModule()
                await self._weather_module.initialize()

            user_id = self.config.resolve_user_id(chat_id) or "0"
            result = await self._weather_module.safe_execute(
                "get_weather", {"location": location}, user_id
            )
        except Exception as exc:
            logger.debug(f"Weather retrieval failed for {location}: {exc}")
            return f"Nie udało się pobrać pogody dla {location}."

        if not isinstance(result, dict) or not result.get("success"):
            error_text = ""
            if isinstance(result, dict):
                error_text = result.get("error") or result.get("message") or ""
            detail = f": {error_text}" if error_text else ""
            return f"Nie udało się pobrać pogody dla {location}{detail}."

        data = result.get("data") or {}
        loc_value = data.get("location")
        if isinstance(loc_value, dict):
            location_label = loc_value.get("name") or loc_value.get("label") or location
        elif isinstance(loc_value, str):
            location_label = loc_value
        else:
            location_label = location

        current = data.get("current") or {}
        description = current.get("description") or data.get("description")
        temperature = current.get("temperature") or data.get("temperature")
        feels_like = current.get("feels_like") or current.get("feelslike") or data.get("feels_like")

        temp_fragment = ""
        if isinstance(temperature, (int, float)):
            temp_fragment = f", około {temperature:.1f}°C"
        elif isinstance(feels_like, (int, float)):
            temp_fragment = f", odczuwalnie {feels_like:.1f}°C"

        if description:
            description_text = str(description).strip().lower()
            return f"Pogoda w {location_label}: {description_text}{temp_fragment}."

        if temp_fragment:
            return f"Temperatura w {location_label}{temp_fragment}."

        return f"Warunki pogodowe w {location_label} są dziś stabilne."

    async def _build_schedule_lines(
        self, target_date: date, settings: dict[str, Any]
    ) -> list[str]:
        lines, _ = await self._prepare_schedule_brief(target_date, settings)
        return lines

    def _choose_fun_line(self) -> str:
        options = [
            "Może krótki spacer po zajęciach? Głowa odpocznie, a ty złapiesz świeże powietrze.",
            "Pamiętaj o szklance wody zanim usiądziesz do komputera.",
            "Dodaj dziś coś miłego do listy zadań — choćby kawę z dobrą muzyką.",
            "Zrób krótką przerwę na rozciąganie, plecy ci podziękują!",
            "Jeśli znajdziesz wolną chwilę, obejrzyj zabawny klip — śmiech dobrze rozpoczyna dzień.",
        ]
        return random.choice(options)

    def _format_brief_help(self, chat_id: str) -> str:
        status = self._format_brief_status(chat_id)
        instructions = (
            "Dostępne polecenia:\n"
            "/brief on – włącz codzienny brief\n"
            "/brief off – wyłącz codzienny brief\n"
            "/brief time HH:MM – ustaw godzinę wysyłki\n"
            "/brief name Imię – ustaw powitanie\n"
            "/brief weather Miasto – ustaw lokalizację pogody (\"off\" aby wyłączyć)\n"
            "/brief plan on/off – włącz lub wyłącz plan zajęć\n"
            "/brief fun on/off – kontrola sekcji humorystycznej\n"
            "/brief subject list – pokaż aktualne mapowania przedmiotów\n"
            "/brief subject set wzorzec|nazwa|[sala]|[notatka] – dodaj/edytuj mapowanie\n"
            "/brief subject remove wzorzec – usuń mapowanie\n"
            "/brief variant 0-3 – wybierz ton wygenerowanej wiadomości\n"
            "/brief now – wyślij brief od razu\n"
            "/brief status – pokaż ustawienia\n"
            "Zaplanowany brief wysyła jedną wiadomość generowaną przez AI (wariant 2).\n"
            "Wzorzec dopasowuje fragment nazwy przedmiotu lub sali (bez polskich znaków)."
        )
        return f"{status}\n\n{instructions}"

    def _format_brief_status(self, chat_id: str) -> str:
        settings = self._ensure_brief_settings(chat_id)
        status = "włączony" if settings.get("enabled") else "wyłączony"
        weather_info = settings.get("weather_location") or "wyłączona"
        fun_info = "tak" if settings.get("include_fun") else "nie"
        plan_info = "tak" if settings.get("include_schedule") else "nie"
        weekend_info = "tak" if settings.get("send_if_weekend", True) else "nie"
        overrides_count = len(self._get_subject_overrides(settings))
        variant = self._sanitize_brief_variant(settings.get("preferred_variant", 2))
        last_sent = self._format_timestamp(settings.get("last_sent_at"))
        next_run = self._format_next_run_text(chat_id)

        lines = [
            f"Stan: {status}",
            f"Godzina: {settings.get('time', '08:00')}",
            f"Imię: {settings.get('name') or 'nie ustawiono'}",
            f"Pogoda: {weather_info}",
            f"Plan zajęć: {plan_info}",
            f"Weekendowe wysyłki: {weekend_info}",
            f"Wiadomości na wysyłkę: 1 (wariant {variant})",
            f"Humor/sugestie: {fun_info}",
            f"Mapowania przedmiotów: {overrides_count}",
            "Generowanie treści: AI (personalizowany wariant)",
            f"Ostatni brief: {last_sent}",
            next_run,
        ]
        if overrides_count:
            overrides_text = self._format_subject_overrides(settings)
            lines.append("")
            lines.append("Mapowania:\n" + overrides_text)
        return "\n".join(lines)

    def _format_timestamp(self, value: Optional[str]) -> str:
        if not value:
            return "-"
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(self._timetable_zone)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(value)

    def _format_next_run_text(self, chat_id: str) -> str:
        settings = self._daily_briefs_config.get(chat_id)
        if not settings or not settings.get("enabled"):
            return "Następny brief: wyłączony."
        next_run = self._calculate_next_brief_run(settings)
        date_text = self._format_polish_date(next_run)
        return f"Następny brief: {date_text} o {next_run:%H:%M}."

    async def _save_daily_brief_settings(self) -> None:
        async with self._daily_brief_lock:
            data_to_save = {
                chat_id: settings for chat_id, settings in self._daily_briefs_config.items()
            }

            def _write() -> None:
                path = self._daily_brief_storage_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data_to_save, fh, indent=2, ensure_ascii=False)

            await asyncio.to_thread(_write)

    async def _handle_daily_brief_command(self, update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if not message:
            return

        chat_id = str(message.chat_id)
        if not self.config.is_chat_allowed(chat_id):
            await message.reply_text("Brak uprawnień do korzystania z tej funkcji.")
            return

        args = getattr(context, "args", None) if context else None
        args = args or []

        if not args:
            await message.reply_text(self._format_brief_help(chat_id))
            return

        command = args[0].lower()
        payload = " ".join(args[1:]).strip()
        settings = self._ensure_brief_settings(chat_id)

        if command in {"on", "start"}:
            changed = await self._update_daily_brief_settings(chat_id, {"enabled": True})
            next_run = self._format_next_run_text(chat_id)
            text = "Briefing codzienny został włączony." if changed else "Briefing był już włączony."
            await message.reply_text(f"{text}\n{next_run}")
            return

        if command in {"off", "stop"}:
            changed = await self._update_daily_brief_settings(chat_id, {"enabled": False})
            text = "Briefing został wyłączony." if changed else "Briefing był już wyłączony."
            await message.reply_text(text)
            return

        if command == "time":
            if not payload:
                await message.reply_text("Podaj godzinę w formacie HH:MM, np. /brief time 07:30")
                return
            sanitized = self._sanitize_brief_time(payload)
            await self._update_daily_brief_settings(chat_id, {"time": sanitized})
            await message.reply_text(
                f"Godzina briefu ustawiona na {sanitized}.\n{self._format_next_run_text(chat_id)}"
            )
            return

        if command == "name":
            if not payload:
                await message.reply_text("Podaj imię, np. /brief name Ania")
                return
            await self._update_daily_brief_settings(chat_id, {"name": payload})
            await message.reply_text(f"Zapisano imię: {payload}.")
            return

        if command == "weather":
            if not payload or payload.lower() in {"off", "brak", "none"}:
                await self._update_daily_brief_settings(chat_id, {"weather_location": ""})
                await message.reply_text("Sekcja pogody została wyłączona.")
                return
            await self._update_daily_brief_settings(chat_id, {"weather_location": payload})
            await message.reply_text(f"Pogoda będzie pobierana dla lokalizacji: {payload}.")
            return

        if command == "plan":
            if payload.lower() in {"off", "false", "nie"}:
                await self._update_daily_brief_settings(chat_id, {"include_schedule": False})
                await message.reply_text("Plan zajęć został pominięty w briefie.")
            elif payload.lower() in {"on", "true", "tak"}:
                await self._update_daily_brief_settings(chat_id, {"include_schedule": True})
                await message.reply_text("Plan zajęć będzie dołączany do briefu.")
            else:
                await message.reply_text(
                    "Użyj /brief plan on lub /brief plan off, aby kontrolować plan zajęć."
                )
            return

        if command in {"fun", "humor"}:
            if payload.lower() in {"off", "false", "nie"}:
                await self._update_daily_brief_settings(chat_id, {"include_fun": False})
                await message.reply_text("Sekcja humorystyczna została wyłączona.")
            elif payload.lower() in {"on", "true", "tak"}:
                await self._update_daily_brief_settings(chat_id, {"include_fun": True})
                await message.reply_text("Dodam codziennie drobny żart lub sugestię.")
            else:
                await message.reply_text(
                    "Użyj /brief fun on lub /brief fun off, aby kontrolować sekcję humorystyczną."
                )
            return

        if command == "subject":
            await self._handle_subject_command(chat_id, message, payload)
            return

        if command == "variant":
            if not payload:
                await message.reply_text("Podaj wariant 0-3, np. /brief variant 2")
                return
            try:
                variant_value = self._sanitize_brief_variant(int(payload))
            except ValueError:
                await message.reply_text("Wariant musi być liczbą 0-3.")
                return
            await self._update_daily_brief_settings(
                chat_id, {"preferred_variant": variant_value}
            )
            await message.reply_text(f"Ustawiono wariant AI na {variant_value}.")
            return

        if command in {"weekend"}:
            if payload.lower() in {"off", "false", "nie"}:
                await self._update_daily_brief_settings(chat_id, {"send_if_weekend": False})
                await message.reply_text("Brief nie będzie wysyłany w weekendy.")
            elif payload.lower() in {"on", "true", "tak"}:
                await self._update_daily_brief_settings(chat_id, {"send_if_weekend": True})
                await message.reply_text("Brief będzie wysyłany także w weekendy.")
            else:
                await message.reply_text(
                    "Użyj /brief weekend on lub /brief weekend off, aby sterować weekendami."
                )
            return

        if command == "status":
            await message.reply_text(self._format_brief_status(chat_id))
            return

        if command == "now":
            messages = await self._generate_daily_brief_messages(
                chat_id, settings, count=1
            )
            if not messages:
                await message.reply_text("Nie udało się zbudować briefu — spróbuj ponownie później.")
                return
            for text in messages:
                await message.reply_text(text)
            now = self._current_time()
            await self._update_daily_brief_settings(
                chat_id,
                {
                    "last_sent_at": now.isoformat(),
                    "last_sent_date": now.date().isoformat(),
                },
                reschedule=True,
            )
            return

        if command == "help":
            await message.reply_text(self._format_brief_help(chat_id))
            return

        await message.reply_text(
            "Nie rozpoznaję tej opcji. Użyj /brief help, aby zobaczyć dostępne polecenia."
        )

    async def _update_daily_brief_settings(
        self, chat_id: str, updates: dict[str, Any], *, reschedule: bool = True
    ) -> bool:
        settings = self._ensure_brief_settings(chat_id)
        changed = False
        for key, value in updates.items():
            if key == "time":
                value = self._sanitize_brief_time(str(value))
            elif key in {"enabled", "include_schedule", "include_fun", "send_if_weekend"}:
                value = bool(value)
            elif key in {"name", "weather_location"}:
                value = str(value or "").strip()
            elif key in {"last_sent_at", "last_sent_date"}:
                value = str(value) if value else None
            elif key == "preferred_variant":
                value = self._sanitize_brief_variant(value)
            elif key == "subject_overrides":
                continue

            if settings.get(key) != value:
                settings[key] = value
                changed = True

        if changed:
            await self._save_daily_brief_settings()

        if reschedule:
            if settings.get("enabled"):
                self._schedule_daily_brief_task(chat_id)
            else:
                self._cancel_daily_brief_task(chat_id)

        return changed

    async def _handle_subject_command(self, chat_id: str, message: Any, payload: str) -> None:
        if not payload:
            await message.reply_text(self._subject_command_help())
            return

        action, _, rest = payload.partition(" ")
        action = action.lower()
        rest = rest.strip()

        if action in {"list", "ls"}:
            settings = self._ensure_brief_settings(chat_id)
            overrides_text = self._format_subject_overrides(settings)
            await message.reply_text(f"Mapowania przedmiotów:\n{overrides_text}")
            return

        if action in {"set", "add"}:
            if not rest or "|" not in rest:
                await message.reply_text(self._subject_command_help())
                return
            parts = [part.strip() for part in rest.split("|")]
            if len(parts) < 2:
                await message.reply_text(self._subject_command_help())
                return
            pattern = parts[0]
            name = parts[1]
            if not pattern or not name:
                await message.reply_text(self._subject_command_help())
                return
            room = parts[2] if len(parts) > 2 else ""
            notes = parts[3] if len(parts) > 3 else ""
            await self._set_subject_override(chat_id, pattern, name, room, notes)
            await message.reply_text(
                f"Zapisano mapowanie: {pattern} → {name}{f' (sala {room})' if room else ''}."
            )
            return

        if action in {"remove", "del", "delete"}:
            if not rest:
                await message.reply_text(self._subject_command_help())
                return
            removed = await self._remove_subject_override(chat_id, rest)
            if removed:
                await message.reply_text(f"Usunięto mapowanie dla wzorca: {rest}.")
            else:
                await message.reply_text(f"Nie znaleziono mapowania dla wzorca: {rest}.")
            return

        await message.reply_text(self._subject_command_help())

    def _subject_command_help(self) -> str:
        return (
            "Użycie mapowania przedmiotów:\n"
            "/brief subject list – pokaż obecne mapowania\n"
            "/brief subject set wzorzec|nazwa|[sala]|[notatka] – zapisz mapowanie (np. /brief subject set JA|Język angielski (C1)||lektorat gr.19)\n"
            "/brief subject remove wzorzec – usuń mapowanie"
        )

    async def _send_timetable_reply(self, message: Any, day_selector: str) -> None:
        target_date, label = self._resolve_target_date(day_selector)
        header = f"Plan (grupa {self._timetable_group}) na {label} ({target_date.isoformat()}):"

        if target_date.weekday() >= 5:
            try:
                await message.reply_text("Jutro nie ma zajęć")
            except Exception as exc:
                logger.error(f"Failed to send weekend timetable info: {exc}")
            return

        status_message: Any | None = None
        try:
            status_message = await message.reply_text("Pobieram plan, proszę czekać...")
        except Exception as exc:
            logger.debug(f"Unable to send timetable progress message: {exc}")

        try:
            entries = await self._get_timetable_entries(target_date)
        except httpx.HTTPError as exc:
            logger.exception("Failed to download timetable: {}", exc)
            await message.reply_text("Nie udało się pobrać planu zajęć. Spróbuj ponownie później.")
            await self._delete_message_safely(status_message)
            return
        except Exception as exc:  # pragma: no cover - defensive log
            logger.error(f"Unexpected timetable error: {exc}")
            await message.reply_text("Wystąpił błąd przy przetwarzaniu planu zajęć.")
            await self._delete_message_safely(status_message)
            return

        if not entries:
            await message.reply_text(f"{header}\nBrak zaplanowanych zajęć dla wybranego dnia.")
            await self._delete_message_safely(status_message)
            return

        # Sort entries just in case.
        entries = sorted(entries, key=lambda item: item.start)
        lines = [header] + [entry.format_line() for entry in entries]
        try:
            await message.reply_text("\n".join(lines))
        finally:
            await self._delete_message_safely(status_message)

    def _resolve_target_date(self, selector: str) -> tuple[date, str]:
        selector = (selector or "tomorrow").lower()
        now = self._current_time()

        if selector == "today":
            return now.date(), "dziś"
        if selector == "day_after":
            return (now + timedelta(days=2)).date(), "pojutrze"
        if selector.startswith("weekday:"):
            try:
                weekday = int(selector.split(":", 1)[1])
            except (ValueError, IndexError):
                weekday = now.weekday()
            target = self._next_date_for_weekday(weekday, now)
            label = _WEEKDAY_LABELS.get(weekday, "wybrany dzień")
            return target, label
        # Default -> tomorrow
        return (now + timedelta(days=1)).date(), "jutro"

    def _current_time(self) -> datetime:
        return datetime.now(self._timetable_zone)

    def _next_date_for_weekday(self, weekday: int, reference: datetime) -> date:
        weekday = weekday % 7
        delta_days = (weekday - reference.weekday()) % 7
        return (reference + timedelta(days=delta_days)).date()

    async def _get_timetable_entries(self, target_date: date) -> list[TimetableEntry]:
        now = self._current_time()
        cache_key = f"{target_date.isoformat()}::{self._timetable_group}"
        cached = self._timetable_cache.get(cache_key)
        if cached and (now - cached[0]) < self._timetable_cache_ttl:
            return cached[1]

        ics_text = await self._get_timetable_text(now)
        entries_by_date = self._parse_timetable_by_date(ics_text, self._timetable_group)
        day_entries = entries_by_date.get(target_date, [])
        self._timetable_cache[cache_key] = (now, day_entries)
        return day_entries

    async def _get_timetable_text(self, now: datetime) -> str:
        if self._timetable_raw_cache:
            fetched_at, cached_text = self._timetable_raw_cache
            if (now - fetched_at) < self._timetable_cache_ttl:
                return cached_text

        last_error: Exception | None = None
        attempt = 1
        while attempt <= self._timetable_max_attempts:
            client = await self._ensure_timetable_http_client()
            try:
                response = await client.get(self._resolve_timetable_request_url())
                response.raise_for_status()
                text = response.text
                if "BEGIN:VCALENDAR" not in text.upper():
                    raise ValueError("Timetable response did not contain ICS data")
                break
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_error = exc
                logger.warning(
                    f"Timetable download failed (attempt {attempt}/{self._timetable_max_attempts}): {exc}"
                )
                fallback_applied = False
                if isinstance(exc, httpx.ConnectError):
                    fallback_applied = self._apply_timetable_connect_fallback(exc)
                await self._reset_timetable_http_client()
                if not fallback_applied:
                    attempt += 1
                if attempt > self._timetable_max_attempts:
                    raise
                await asyncio.sleep(self._timetable_retry_delay * attempt)
                continue
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    f"Timetable download returned unexpected content (attempt {attempt}/{self._timetable_max_attempts})"
                )
                await self._reset_timetable_http_client()
                attempt += 1
                if attempt > self._timetable_max_attempts:
                    raise
                await asyncio.sleep(self._timetable_retry_delay * attempt)
                continue
            except Exception:
                await self._reset_timetable_http_client()
                raise
        else:  # pragma: no cover - defensive guard
            if last_error:
                raise last_error
            raise RuntimeError("Unable to download timetable")

        self._timetable_raw_cache = (now, text)
        self._timetable_cache.clear()
        return text

    async def _ensure_timetable_http_client(self) -> httpx.AsyncClient:
        if self._timetable_http_client is None:
            timeout = httpx.Timeout(60.0, connect=15.0, read=60.0, write=60.0)
            verify: bool | ssl.SSLContext = True
            if not self._timetable_use_http_fallback and self._timetable_use_legacy_tls:
                verify = self._build_legacy_tls_context()
            transport = httpx.AsyncHTTPTransport(http2=False, verify=verify)
            self._timetable_http_client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; GAJA Telegram Bot/1.0)",
                    "Accept": "text/calendar, text/plain;q=0.9, */*;q=0.8",
                    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "close",
                    "Referer": "https://plan.polsl.pl/",
                },
                transport=transport,
            )
        return self._timetable_http_client

    async def _reset_timetable_http_client(self) -> None:
        client = self._timetable_http_client
        if client is None:
            return
        self._timetable_http_client = None
        with suppress(Exception):
            await client.aclose()

    def _resolve_timetable_request_url(self) -> str:
        if self._timetable_use_http_fallback and self._timetable_url.startswith("https://"):
            return "http://" + self._timetable_url[len("https://") :]
        return self._timetable_url

    def _build_legacy_tls_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        with suppress(AttributeError):
            context.minimum_version = ssl.TLSVersion.TLSv1
        for option_name in ("OP_NO_TLSv1", "OP_NO_TLSv1_1"):
            option = getattr(ssl, option_name, 0)
            if option:
                context.options &= ~option
        with suppress(ssl.SSLError):
            context.set_ciphers("DEFAULT:@SECLEVEL=1")
        return context

    def _apply_timetable_connect_fallback(self, exc: httpx.ConnectError) -> bool:
        if not self._timetable_use_legacy_tls:
            self._timetable_use_legacy_tls = True
            logger.warning(
                f"Timetable download: enabling legacy TLS compatibility mode due to connect error: {exc}"
            )
            return True
        if (not self._timetable_use_http_fallback) and self._timetable_url.startswith("https://"):
            self._timetable_use_http_fallback = True
            logger.warning(
                "Timetable download: falling back to HTTP due to persistent TLS failures"
            )
            return True
        return False

    def _parse_timetable_by_date(self, ics_text: str, group: str) -> dict[date, list[TimetableEntry]]:
        entries: dict[date, list[TimetableEntry]] = {}
        if not ics_text:
            return entries

        lines = self._unfold_ics_lines(ics_text)
        current_event: dict[str, tuple[dict[str, str], str]] | None = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.upper() == "BEGIN:VEVENT":
                current_event = {}
                continue
            if line.upper() == "END:VEVENT":
                if current_event:
                    entry = self._event_to_entry(current_event, group)
                    if entry:
                        day_key = entry.start.astimezone(self._timetable_zone).date()
                        entries.setdefault(day_key, []).append(entry)
                current_event = None
                continue

            if current_event is None:
                continue

            name, params, value = self._parse_ics_property(line)
            if not name:
                continue
            current_event[name] = (params, value)

        return entries

    def _parse_ics_property(self, line: str) -> tuple[str, dict[str, str], str]:
        colon_idx = line.find(":")
        equals_idx = line.find("=")
        if colon_idx == -1 and equals_idx == -1:
            return "", {}, line

        separator = None
        if colon_idx != -1:
            before_colon = line[:colon_idx]
            if equals_idx == -1 or colon_idx < equals_idx or ";" in before_colon:
                separator = ":"
        if separator is None:
            if equals_idx != -1:
                separator = "="
            else:
                separator = ":"

        before, value = line.split(separator, 1)
        parts = before.split(";")
        name_raw = parts[0].strip()
        if "=" in name_raw:
            name_raw = name_raw.split("=", 1)[0].strip()
        name = name_raw.upper()
        params: dict[str, str] = {}
        for param in parts[1:]:
            if "=" not in param:
                continue
            key, val = param.split("=", 1)
            params[key.strip().upper()] = val.strip()
        return name, params, value.strip()

    def _event_to_entry(
        self,
        event: dict[str, tuple[dict[str, str], str]],
        group: str,
    ) -> Optional[TimetableEntry]:
        dtstart = event.get("DTSTART")
        dtend = event.get("DTEND")
        summary = event.get("SUMMARY")
        if not dtstart or not dtend or not summary:
            return None

        try:
            start = self._parse_ics_datetime(dtstart[1], dtstart[0])
            end = self._parse_ics_datetime(dtend[1], dtend[0])
        except ValueError as exc:
            logger.debug(f"Skipping timetable event due to datetime parse error: {exc}")
            return None
        subject = self._unescape_ics_value(summary[1])

        description = None
        if "DESCRIPTION" in event:
            description = self._unescape_ics_value(event["DESCRIPTION"][1])
        location = None
        if "LOCATION" in event:
            location = self._unescape_ics_value(event["LOCATION"][1])

        extra_fields = [subject, description or "", location or ""]
        if "CATEGORIES" in event:
            extra_fields.append(self._unescape_ics_value(event["CATEGORIES"][1]))

        if not self._matches_group(extra_fields, group):
            return None

        teacher = self._extract_teacher(description, subject)
        group_label = self._extract_group_label(extra_fields)

        return TimetableEntry(
            start=start,
            end=end,
            subject=subject,
            location=location,
            teacher=teacher,
            group_label=group_label,
            raw_description=description,
        )

    def _parse_ics_datetime(self, value: str, params: dict[str, str]) -> datetime:
        value = (value or "").strip()
        tz = self._timetable_zone
        tzid = params.get("TZID") if params else None
        if tzid:
            try:
                tz = ZoneInfo(tzid)
            except Exception:  # pragma: no cover - fallback when tz not available
                tz = self._timetable_zone

        def _parse_with_format(fmt: str) -> datetime | None:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                return None

        if value.endswith("Z"):
            naive = _parse_with_format("%Y%m%dT%H%M%SZ")
            if naive:
                return naive.replace(tzinfo=timezone.utc).astimezone(tz)
        if "T" in value:
            naive = _parse_with_format("%Y%m%dT%H%M%S")
            if naive:
                return naive.replace(tzinfo=tz)

        # DATE only
        naive_date = _parse_with_format("%Y%m%d")
        if naive_date:
            return datetime.combine(naive_date.date(), time.min, tz)

        raise ValueError(f"Unsupported ICS datetime value: {value}")

    def _matches_group(self, fields: list[str], group: str) -> bool:
        if not group:
            return True

        clean_group = re.sub(r"[^0-9a-z]", "", group.lower())
        if not clean_group:
            return True

        patterns = [
            rf"\bgr(?:upa|\.?)\W*{clean_group}\b",
            rf"\bg\W*{clean_group}\b",
            rf"\bgroup\W*{clean_group}\b",
            rf"\b{clean_group}\W*gr(?:upa|\.?)\b",
        ]

        contains_any_group_marker = False
        for field in fields:
            normalized = self._normalize_for_matching(field)
            if not normalized:
                continue

            if re.search(r"\bgr(?:upa|\.?)\W*[0-9]", normalized) or re.search(r"\bgroup\W*[0-9]", normalized):
                contains_any_group_marker = True
            for pattern in patterns:
                if re.search(pattern, normalized):
                    return True

        # If no explicit group marker was found in any field, treat entry as relevant.
        return not contains_any_group_marker

    def _extract_group_label(self, fields: list[str]) -> Optional[str]:
        for field in fields:
            normalized = self._normalize_for_matching(field)
            if not normalized:
                continue
            match = re.search(r"gr(?:upa|\.?)\W*([0-9]+)", normalized)
            if match:
                return f"gr. {match.group(1)}"
        return None

    def _extract_teacher(self, description: Optional[str], subject: str) -> Optional[str]:
        sources = []
        if description:
            sources.extend(description.splitlines())
        sources.append(subject)

        teacher_patterns = [
            r"prowadzą[cś]y:?(.*)",
            r"prowadzacy:?(.*)",
            r"wykładowca:?(.*)",
            r"wykladowca:?(.*)",
            r"ćwiczący:?(.*)",
            r"cwiczacy:?(.*)",
            r"z\s+dr\.?\s+(.+)",
            r"z\s+mgr\.?\s+(.+)",
            r"z\s+prof\.?\s+(.+)",
        ]

        for source in sources:
            cleaned = source.strip()
            if not cleaned:
                continue
            normalized = self._normalize_for_matching(cleaned)
            for pattern in teacher_patterns:
                match = re.search(pattern, normalized)
                if match:
                    extracted = match.group(1).strip()
                    if not extracted:
                        continue
                    return self._restore_original_case(extracted, cleaned)

        return None

    def _restore_original_case(self, match_fragment: str, original_line: str) -> str:
        if not match_fragment:
            return original_line.strip()
        fragment_normalized = self._normalize_for_matching(match_fragment)
        line_normalized = self._normalize_for_matching(original_line)
        idx = line_normalized.find(fragment_normalized)
        if idx == -1:
            return match_fragment.strip()
        return original_line[idx : idx + len(match_fragment)].strip()

    def _unfold_ics_lines(self, ics_text: str) -> list[str]:
        raw_lines = ics_text.splitlines()
        unfolded: list[str] = []
        buffer: str | None = None
        for line in raw_lines:
            if line.startswith(" ") or line.startswith("\t"):
                if buffer is not None:
                    buffer += line[1:]
                continue
            if buffer is not None:
                unfolded.append(buffer)
            buffer = line
        if buffer is not None:
            unfolded.append(buffer)
        return unfolded

    def _unescape_ics_value(self, value: str) -> str:
        if value is None:
            return ""
        result = value.replace("\\\\", "\\")
        result = result.replace("\\n", "\n")
        result = result.replace("\\,", ",")
        result = result.replace("\\;", ";")
        return result.strip()

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

    async def _delete_message_safely(self, telegram_message: Any) -> None:
        if not telegram_message:
            return
        with suppress(Exception):
            await telegram_message.delete()

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
