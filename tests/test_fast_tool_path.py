import os, json, asyncio, sys
from collections import deque
import pytest

# We import the generate_response function directly
from modules.ai_module import generate_response, LatencyTracer

class DummyFunctionCallingSystem:
    def convert_modules_to_functions(self):
        return [{
            "name": "weather_get_weather",
            "description": "Get weather info",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
        }]

@pytest.mark.asyncio
async def test_fast_weather_tool_path(monkeypatch):
    os.environ['GAJA_FAST_TOOL_MODE'] = '1'

    # Build conversation history with a simple user query
    history = deque([
        {"role": "user", "content": "Jaka jest pogoda w Warszawie?"}
    ])

    # Prepare fake first model call response that includes a tool call executed
    async def fake_chat_with_providers(model, messages, functions=None, function_calling_system=None, tracer=None, **kwargs):
        # Simulate provider returning tool call details already executed (our patched system would do that)
        return {
            "message": {"content": "Pogoda w Warszawie: Słonecznie, 21°C (odczuwalna 20°C). Dziś min 15°C / max 23°C."},
            "tool_calls_executed": 1,
            "tool_call_details": [
                {
                    "name": "weather_get_weather",
                    "result": {
                        "data": {
                            "location": {"name": "Warszawa"},
                            "current": {"description": "Słonecznie", "temperature": 21, "feels_like": 20},
                            "forecast": [{"min_temp": 15, "max_temp": 23}]
                        }
                    }
                }
            ],
            "fast_tool_path": True  # This is the key field that was missing!
        }

    # Monkeypatch chat_with_providers used inside generate_response
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    import modules.ai_module as aim
    monkeypatch.setattr(aim, 'chat_with_providers', fake_chat_with_providers)

    # Also monkeypatch function calling system retrieval to avoid real imports
    class DummyFCS:  # minimal stub
        def convert_modules_to_functions(self):
            return [{"name": "weather_get_weather", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}]
    def fake_get_function_calling_system():
        return DummyFCS()
    monkeypatch.setitem(os.environ, 'GAJA_LATENCY_TRACE', '0')  # disable tracer file writes
    # Patch inside function when imported lazily
    import types
    fake_module = types.SimpleNamespace(get_function_calling_system=fake_get_function_calling_system)
    monkeypatch.setitem(sys.modules, 'core.function_calling_system', fake_module)

    # Run generate_response
    result_json = await generate_response(
        conversation_history=history,
        tools_info="",
        detected_language="pl",
        language_confidence=1.0,
        modules={},
        use_function_calling=True,
        user_name="Tester"
    )

    parsed = json.loads(result_json)
    assert parsed.get('fast_tool_path') is True, parsed
    assert 'Pogoda' in parsed.get('text','')
    assert parsed.get('tools_used') == 1

