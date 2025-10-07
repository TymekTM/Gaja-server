import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("httpx")

from integrations.telegram.service import TelegramBotService, TelegramConfig


class _DummyBot:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        self.messages.append((chat_id, text))


@pytest.mark.asyncio
async def test_rescheduling_daily_brief_avoids_duplicate_tasks(tmp_path):
    config = TelegramConfig(
        enabled=True,
        bot_token="dummy",
        daily_brief_storage_path=str(tmp_path / "telegram_daily_briefs.json"),
    )
    service = TelegramBotService(SimpleNamespace(ai_module=None), config)

    chat_id = "123"
    settings = service._ensure_brief_settings(chat_id)
    settings["enabled"] = True

    service._application = SimpleNamespace(bot=_DummyBot())

    async def _fake_generate(chat: str, current_settings: dict[str, Any], *, count: int):
        current_settings["enabled"] = False
        return ["brief"]

    service._generate_daily_brief_messages = _fake_generate  # type: ignore[assignment]
    service._calculate_next_brief_run = (  # type: ignore[assignment]
        lambda current_settings: service._current_time() + timedelta(milliseconds=1)
    )

    service._schedule_daily_brief_task(chat_id)
    service._schedule_daily_brief_task(chat_id)
    service._schedule_daily_brief_task(chat_id)

    await asyncio.sleep(0.05)

    assert service._application.bot.messages == [(chat_id, "brief")]
    assert service._daily_brief_tasks == {}

