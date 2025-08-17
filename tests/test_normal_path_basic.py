import os, json, asyncio, sys
from collections import deque
import pytest
from modules.ai_module import generate_response

@pytest.mark.asyncio
async def test_normal_path_without_tools(monkeypatch):
    # Force no fast tool path and no functions
    os.environ['GAJA_FAST_TOOL_MODE'] = '1'  # irrelevant since no tools
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    async def fake_chat_with_providers(model, messages, **kwargs):
        return {"message": {"content": "Hello user this is a simple reply."}}

    import modules.ai_module as aim
    monkeypatch.setattr(aim, 'chat_with_providers', fake_chat_with_providers)

    history = deque([
        {"role": "user", "content": "Say hello"}
    ])

    result_json = await generate_response(
        conversation_history=history,
        tools_info="",
        detected_language="pl",
        language_confidence=1.0,
        modules={},
        use_function_calling=False,
        user_name="Tester"
    )

    parsed = json.loads(result_json)
    assert parsed.get('text','').startswith('Hello user'), parsed
    assert 'fast_tool_path' not in parsed
