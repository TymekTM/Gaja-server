"""Telegram bot service for GAJA server."""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, timezone, time
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
        timetable_group=str(raw.get("timetable_group", "1")).strip() or "1",
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

        await self._reset_timetable_http_client()

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
