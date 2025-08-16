import sys, asyncio, time, json, statistics, math, os
from collections import deque, defaultdict
sys.path.append('Gaja-server')
from modules.ai_module import generate_response  # noqa

LATENCY_FILE = 'user_data/latency_events.jsonl'

def pctl(data, p):
    if not data:
        return 0.0
    k = (len(data)-1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[f]*(c-k)
    d1 = data[c]*(k-f)
    return d0+d1

async def one_call(i):
    hist = deque([{'role':'user','content':'Powiedz jedno zdanie powitania.'}])
    tid = f'perf_harness_{i}'
    start = time.perf_counter()
    r = await generate_response(hist, detected_language='pl', language_confidence=1.0, tracking_id=tid, enable_latency_trace=True)
    total_ms = (time.perf_counter() - start) * 1000
    return tid, total_ms, len(r)

async def main(warmup=1, measured=3):
    # truncate old latency events for clean run
    if os.path.exists(LATENCY_FILE):
        os.remove(LATENCY_FILE)
    totals = []
    tids = []
    for i in range(warmup + measured):
        tid, total_ms, rlen = await one_call(i)
        print(f'Call {i} total_ms={total_ms:.1f} resp_len={rlen}')
        if i >= warmup:
            totals.append(total_ms)
            tids.append(tid)
    totals_sorted = sorted(totals)
    summary = {
        'count': len(totals),
        'avg_ms': sum(totals)/len(totals) if totals else 0,
        'p50_ms': pctl(totals_sorted, 0.50),
        'p95_ms': pctl(totals_sorted, 0.95),
        'p99_ms': pctl(totals_sorted, 0.99),
        'min_ms': min(totals) if totals else 0,
        'max_ms': max(totals) if totals else 0,
    }
    # stage breakdown from latency events
    stage_totals = defaultdict(list)
    if os.path.exists(LATENCY_FILE):
        with open(LATENCY_FILE,'r',encoding='utf-8') as f:
            events = [json.loads(l) for l in f]
        # group by tracking_id
        by_tid = defaultdict(list)
        for ev in events:
            by_tid[ev['tracking_id']].append(ev)
        for tid, evs in by_tid.items():
            evs.sort(key=lambda e: e['t_rel_ms'])
            start = evs[0]['t_rel_ms'] if evs else 0
            end = evs[-1]['t_rel_ms'] if evs else 0
            total = end - start
            stage_totals['total_trace_ms'].append(total)
            # measure provider_request duration if start/end present
            s = next((e for e in evs if e['event']=='provider_request_start'), None)
            e = next((e for e in evs if e['event']=='provider_request_end'), None)
            if s and e:
                stage_totals['provider_request_ms'].append(e['t_rel_ms'] - s['t_rel_ms'])
            b = next((e for e in evs if e['event']=='system_prompt_built'), None)
            if b:
                stage_totals['system_prompt_build_ms'].append(b.get('extra',{}).get('build_ms',0))
    stage_summary = {}
    for k,v in stage_totals.items():
        v_sorted = sorted(v)
        stage_summary[k] = {
            'avg': sum(v)/len(v),
            'p95': pctl(v_sorted, 0.95),
            'min': min(v),
            'max': max(v)
        }
    result = {'totals': summary, 'stages': stage_summary}
    print('SUMMARY:', json.dumps(result, ensure_ascii=False, indent=2))
    with open('user_data/perf_harness_summary.json','w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)

if __name__ == '__main__':
    asyncio.run(main())
