import json
from collections import deque
from pathlib import Path
import importlib.util
import pytest

from modules.ai_module import generate_response

BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "modules"
USER_ID = 2

def _load_module(name: str):
    path = MODULES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"modules.{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Spec load failed for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_fast_tool_path_weather_summary(monkeypatch):
    """Symuluje warunki fast tool path wykonując manualnie weather tool i
    podmieniając pierwszą odpowiedź modelu aby wymusić pojedynczy tool call.
    """
    weather_mod = _load_module("weather_module")

    # Uzyskaj dane mock (test_mode) dla spójności
    weather_result = await weather_mod.execute_function(
        "get_weather", {"location": "Warszawa", "test_mode": True}, USER_ID
    )
    assert weather_result.get("success")

    # Przygotuj sztuczną odpowiedź modelu z jednym tool call-em
    async def fake_chat_with_providers(model, messages, functions=None, function_calling_system=None, **kwargs):  # noqa: D401
        # Budujemy streszczenie identyczne logicznie z tym co robi fast path w chat_openai
        data = weather_result.get("data", {})
        loc = data.get("location", {}) if isinstance(data, dict) else {}
        curr = data.get("current", {}) if isinstance(data, dict) else {}
        forecast = data.get("forecast", []) if isinstance(data, dict) else []
        location_text = loc.get("name") or "(lokalizacja)"
        desc = curr.get("description", "")
        temp = curr.get("temperature")
        feels = curr.get("feels_like")
        summary = f"Pogoda w {location_text}: {desc}, {temp}°C (odczuwalna {feels}°C)."
        if forecast and isinstance(forecast, list):
            first = forecast[0]
            if isinstance(first, dict):
                summary += f" Dziś min {first.get('min_temp')}°C / max {first.get('max_temp')}°C."
        return {
            "message": {"content": summary},
            "tool_calls_executed": 1,
            "tool_call_details": [
                {"name": "weather_get_weather", "arguments": {"location": "Warszawa"}, "result": weather_result}
            ],
            "fast_tool_path": True,
        }

    async def fake_execute_function(fn_name, fn_args, conversation_history=None):
        # Zwracamy dokładnie strukturę jak weather_result aby fast path mógł ją sparsować
        return weather_result

    import modules.ai_module as aim
    monkeypatch.setattr(aim, 'chat_with_providers', fake_chat_with_providers)
    # Nie potrzebujemy faktycznego function_calling_system ponieważ fast path jest już zasymulowany

    history = deque([{"role": "user", "content": "Jaka jest pogoda w Warszawie?"}])
    # Ustaw tryb fast
    monkeypatch.setenv('GAJA_FAST_TOOL_MODE', '1')

    response_json = await generate_response(
        conversation_history=history,
        tools_info="weather",
        detected_language="pl",
        language_confidence=1.0,
        modules={"weather": weather_mod},
        use_function_calling=True,
        user_name="Tester"
    )

    # Fast path zwraca string JSON z kluczem fast_tool_path
    assert isinstance(response_json, str)
    parsed = json.loads(response_json)
    assert parsed.get('fast_tool_path') is True, parsed
    text = parsed.get('text','')
    assert 'Pogoda w' in text and 'Warszawa' in text and '°C' in text, text
    assert any(key in text for key in ['min', 'max']), text


@pytest.mark.asyncio
async def test_fast_tool_path_skips_on_error(monkeypatch):
    """Gdy success=False fast path nie powinien aktywować się."""
    # Zbuduj fałszywy tool_result z success False
    fake_tool_result = {"success": False, "data": {}}

    async def fake_chat_with_providers(model, messages, functions=None, function_calling_system=None, **kwargs):
        return {
            "message": {"content": "(fallback)"},
            "tool_calls_executed": 1,
            "tool_call_details": [
                {"name": "weather_get_weather", "arguments": {"location": "X"}, "result": fake_tool_result}
            ],
        }

    import modules.ai_module as aim
    monkeypatch.setattr(aim, 'chat_with_providers', fake_chat_with_providers)
    history = deque([{"role": "user", "content": "Jaka jest pogoda?"}])
    monkeypatch.setenv('GAJA_FAST_TOOL_MODE', '1')
    response_json = await generate_response(
        conversation_history=history,
        tools_info="weather",
        detected_language="pl",
        language_confidence=1.0,
        modules={},
        use_function_calling=True,
        user_name="Tester"
    )
    assert isinstance(response_json, str)
    parsed = json.loads(response_json)
    assert not parsed.get('fast_tool_path'), parsed
