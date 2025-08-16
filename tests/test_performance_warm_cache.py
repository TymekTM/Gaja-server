import sys, asyncio, json, os, time
from collections import deque, defaultdict
import pytest

# Adjust path
if 'Gaja-server' not in sys.path:
    sys.path.append('Gaja-server')

from modules.ai_module import generate_response  # noqa

LAT_FILE = 'user_data/latency_events.jsonl'

async def _single_call(tag: str):
    hist = deque([{'role':'user','content':'Powiedz jedno zdanie powitania.'}])
    start = time.perf_counter()
    resp = await generate_response(hist, detected_language='pl', language_confidence=1.0, tracking_id=tag, enable_latency_trace=True)
    total = (time.perf_counter() - start) * 1000
    return total

def _extract_stage(tracking_id: str):
    if not os.path.exists(LAT_FILE):
        return None, None
    with open(LAT_FILE,'r',encoding='utf-8') as f:
        evs = [json.loads(l) for l in f if tracking_id in l]
    if not evs:
        return None, None
    # provider duration
    start_ev = next((e for e in evs if e['event']=='provider_request_start'), None)
    end_ev = next((e for e in evs if e['event']=='provider_request_end'), None)
    prov_ms = (end_ev['t_rel_ms'] - start_ev['t_rel_ms']) if start_ev and end_ev else None
    spb = next((e for e in evs if e['event']=='system_prompt_built'), None)
    spb_ms = spb.get('extra',{}).get('build_ms') if spb else None
    return prov_ms, spb_ms

@pytest.mark.perf
@pytest.mark.ai
async def test_warm_cache_reduces_prompt_and_total_time():
    # clean latency file
    if os.path.exists(LAT_FILE):
        os.remove(LAT_FILE)
    cold_total = await _single_call('perf_test_cold')
    warm_total = await _single_call('perf_test_warm')
    # extract
    prov_cold, spb_cold = _extract_stage('perf_test_cold')
    prov_warm, spb_warm = _extract_stage('perf_test_warm')
    # Assertions (allow some fluctuation but warm should be less or equal)
    assert warm_total <= cold_total * 1.05, f"Warm total not improved: cold={cold_total:.1f} warm={warm_total:.1f}"
    if spb_cold is not None and spb_warm is not None:
        assert spb_warm <= spb_cold, f"System prompt build not cached: cold={spb_cold} warm={spb_warm}"
