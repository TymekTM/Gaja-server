import os, json, asyncio, time
from collections import deque
import pytest

import sys
sys.path.append('Gaja-server')
from modules.ai_module import generate_response  # noqa

LAT_FILE = 'user_data/latency_events.jsonl'

@pytest.mark.asyncio
async def test_stream_complete_event_and_tokens(monkeypatch):
    # Skip test if no OPENAI_API_KEY and not in CI to avoid failures
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip('OPENAI_API_KEY not set; skipping streaming integration test')
    if os.path.exists(LAT_FILE):
        os.remove(LAT_FILE)
    history = deque([{'role':'user','content':'Napisz dwa bardzo krótkie zdania.'}])
    partial_chunks = []
    def collect(chunk: str):
        partial_chunks.append(chunk)
    resp = await generate_response(history, detected_language='pl', language_confidence=1.0, tracking_id='test_stream_case', enable_latency_trace=True, stream=True, partial_callback=collect)
    assert isinstance(resp, str)
    # Ensure partial callback captured some data (heuristic)
    assert partial_chunks, 'No partial chunks captured'
    # Read latency events looking for stream_complete
    assert os.path.exists(LAT_FILE), 'Latency events file not created'
    events = []
    with open(LAT_FILE,'r',encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line))
    stream_complete = [e for e in events if e.get('event')=='stream_complete']
    assert stream_complete, 'stream_complete event missing'
    # tokens_per_sec positive
    tp = stream_complete[-1].get('extra',{}).get('tokens_per_sec',0)
    assert tp > 0, 'tokens_per_sec must be > 0'
