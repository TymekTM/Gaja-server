import asyncio
import base64
import json
import os
import time

import pytest

from core.app_paths import migrate_legacy_file, resolve_data_path

from websockets import connect as ws_connect


@pytest.mark.asyncio
@pytest.mark.integration
async def test_time_to_first_audio_chunk_stub(monkeypatch):
    """Measure time-to-first-audio-token using stub streaming.

    Preconditions:
      - Server must be running (this test treats server as external dependency)
      - Env GAJA_AUDIO_STREAM_STUB=1 enables early tts_chunk emission
    The test connects via WebSocket, sends a simple query, waits for first
    'tts_chunk' message and records latency. It then waits for final 'ai_response'.
    The test asserts that latency to first chunk is significantly lower than a
    conservative upper bound (e.g. 4 seconds) to catch regressions where early
    streaming stops working.
    """

    # Ensure stub streaming is enabled for the running server (best‑effort)
    monkeypatch.setenv('GAJA_AUDIO_STREAM_STUB', '1')

    host = os.getenv('SERVER_HOST', '127.0.0.1')
    port = int(os.getenv('SERVER_PORT', '8001'))
    uri = f"ws://{host}:{port}/ws/test_user_latency"

    # Allow skipping if server not running to avoid false negatives in unit-only runs
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
        except OSError:
            pytest.skip(f"Server not reachable at {host}:{port}; skipping streaming latency test")

    async with ws_connect(uri) as ws:
        # Expect handshake response first
        handshake_raw = await asyncio.wait_for(ws.recv(), timeout=5)
        hs = json.loads(handshake_raw)
        assert hs.get('type') == 'handshake_response'

        # Send query
        query_payload = {
            "type": "query",
            "data": {"query": "Jaka jest pogoda?"}
        }
        await ws.send(json.dumps(query_payload))

        t_start = time.perf_counter()
        first_chunk_latency = None
        received_ai_response = False

        # Collect messages until ai_response received (cap to 30s)
        end_deadline = t_start + 30
        while time.perf_counter() < end_deadline and not received_ai_response:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            mtype = msg.get('type')
            if mtype == 'tts_chunk' and first_chunk_latency is None:
                # Basic sanity on chunk
                data = msg.get('data') or {}
                chunk_b64 = data.get('chunk')
                assert chunk_b64, 'Empty audio chunk'
                # Validate base64 integrity (doesn't raise)
                base64.b64decode(chunk_b64)
                first_chunk_latency = time.perf_counter() - t_start
            elif mtype == 'ai_response':
                received_ai_response = True
                response_payload = msg.get('data') or {}
                # Optional: validate structure
                assert 'response' in response_payload

        assert received_ai_response, 'Did not receive ai_response within timeout'
        assert first_chunk_latency is not None, 'No early tts_chunk received (stub may be disabled)'

        # Assert a reasonable upper bound (adjust if environment is slower)
        # We expect stub chunk << full TTS generation, so 4s is conservative.
        assert first_chunk_latency < 4.0, f"First audio chunk latency too high: {first_chunk_latency:.2f}s"

        # Store metric for later aggregation (append JSONL)
        metrics_path = resolve_data_path('latency_audio_metrics.jsonl', create_parents=True)
        migrate_legacy_file('user_data/latency_audio_metrics.jsonl', metrics_path)
        with metrics_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({
                'metric': 'time_to_first_audio_chunk_stub',
                'latency_s': first_chunk_latency,
                'timestamp': time.time()
            }, ensure_ascii=False) + '\n')
