import asyncio
from types import SimpleNamespace

import pytest

from core import app_paths
from integrations.telegram.service import TelegramBotService, TelegramConfig
from server_main import ServerApp


@pytest.fixture(autouse=True)
def clear_app_paths_cache():
    app_paths.get_data_root.cache_clear()
    yield
    app_paths.get_data_root.cache_clear()


@pytest.mark.asyncio
async def test_forward_voice_result_to_telegram_invokes_service(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    async def fake_forward(user_id: str, query: str, response: str) -> None:
        calls.append((user_id, query, response))

    app = ServerApp()
    app.telegram_service = SimpleNamespace(forward_voice_interaction=fake_forward)

    await app._forward_voice_result_to_telegram("client1", "hej", "odpowiedź")

    assert calls == [("client1", "hej", "odpowiedź")]


@pytest.mark.asyncio
async def test_forward_voice_interaction_sends_message(tmp_path, monkeypatch):
    monkeypatch.setenv("GAJA_DATA_DIR", str(tmp_path / "gaja_data"))

    config = TelegramConfig(
        enabled=True,
        chat_user_map={"555": "1"},
        allowed_chat_ids=set(),
        default_user_id="1",
        daily_brief_storage_path="telegram_bridge_test.json",
    )

    service = TelegramBotService(server_app=None, config=config)
    bot_calls: list[tuple[str, str]] = []

    class StubBot:
        async def send_message(self, chat_id: str, text: str) -> None:
            bot_calls.append((chat_id, text))

    service._application = SimpleNamespace(bot=StubBot())

    await service.forward_voice_interaction("1", "co u Ciebie?", "Wszystko dobrze!")

    assert bot_calls, "Voice interaction was not forwarded"
    chat_id, text = bot_calls[0]
    assert chat_id == "555"
    assert "Voice" in text
    assert "Wszystko dobrze" in text
