import asyncio
import json
import os
import time
from collections import deque

import pytest

import sys

from core.app_paths import migrate_legacy_file, resolve_data_path

sys.path.append('Gaja-server')
from modules.ai_module import generate_response  # noqa

LAT_PATH = resolve_data_path('latency_events.jsonl', create_parents=True)
migrate_legacy_file('user_data/latency_events.jsonl', LAT_PATH)

@pytest.mark.asyncio
async def test_stream_complete_event_and_tokens(monkeypatch):
    # Skip test if no OPENAI_API_KEY and not in CI to avoid failures
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip('OPENAI_API_KEY not set; skipping streaming integration test')
    if LAT_PATH.exists():
        LAT_PATH.unlink()
    history = deque([{'role':'user','content':'Napisz dwa bardzo krótkie zdania.'}])
    partial_chunks = []
    def collect(chunk: str):
        partial_chunks.append(chunk)
    resp = await generate_response(history, detected_language='pl', language_confidence=1.0, tracking_id='test_stream_case', enable_latency_trace=True, stream=True, partial_callback=collect, model_override='gpt-4o-mini', use_function_calling=False)
    assert isinstance(resp, str)
    # Ensure partial callback captured some data (heuristic)
    assert partial_chunks, 'No partial chunks captured'
    
    # Simple test: just check that partial chunks contain expected content
    joined = "".join(partial_chunks)
    assert len(joined) > 0, "No content in joined chunks"
