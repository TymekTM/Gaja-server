import sys, asyncio, time, json, math, statistics, random
from collections import deque, defaultdict

sys.path.append('Gaja-server')

from core.app_paths import migrate_legacy_file, resolve_data_path
from modules.ai_module import generate_response  # noqa

LATENCY_PATH = resolve_data_path('latency_events.jsonl', create_parents=True)
migrate_legacy_file('user_data/latency_events.jsonl', LATENCY_PATH)
OUT_PATH = resolve_data_path('perf_streaming_summary.json', create_parents=True)
migrate_legacy_file('user_data/perf_streaming_summary.json', OUT_PATH)


def pct(data, p):
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

async def one_call(i, stream=True):
    hist = deque([{'role':'user','content':'Napisz krótkie zdanie powitania.'}])
    tid = f'stream_harness_{i}'
    start = time.perf_counter()
    r = await generate_response(hist, detected_language='pl', language_confidence=1.0, tracking_id=tid, enable_latency_trace=True, stream=stream)
    total_ms = (time.perf_counter() - start) * 1000
    return tid, total_ms, len(r)

async def main(warmup=1, measured=5):
    # Optionally truncate old events
    if LATENCY_PATH.exists():
        LATENCY_PATH.unlink()
    totals = []
    tids = []
    for i in range(warmup + measured):
        tid, total_ms, rlen = await one_call(i, stream=True)
        print(f'[stream] Call {i} total_ms={total_ms:.1f} resp_len={rlen}')
        if i >= warmup:
            totals.append(total_ms)
            tids.append(tid)
    # Load events and compute streaming specific metrics
    events = []
    if LATENCY_PATH.exists():
        with LATENCY_PATH.open('r',encoding='utf-8') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    by_tid = defaultdict(list)
    for ev in events:
        if ev.get('tracking_id') in tids:
            by_tid[ev['tracking_id']].append(ev)
    first_token_latencies = []
    provider_latencies = []
    tokens_per_sec = []
    approx_tokens_list = []
    chars_list = []
    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: e['t_rel_ms'])
        start_ev = next((e for e in evs if e['event']=='provider_request_start'), None)
        first_token_ev = next((e for e in evs if e['event']=='stream_first_token'), None)
        complete_ev = next((e for e in evs if e['event']=='stream_complete'), None)
        end_ev = next((e for e in evs if e['event']=='provider_request_end'), None)
        if start_ev and first_token_ev:
            first_token_latencies.append(first_token_ev['t_rel_ms'] - start_ev['t_rel_ms'])
        if start_ev and end_ev:
            provider_latencies.append(end_ev['t_rel_ms'] - start_ev['t_rel_ms'])
        if complete_ev:
            extra = complete_ev.get('extra',{})
            if 'tokens_per_sec' in extra:
                tokens_per_sec.append(extra['tokens_per_sec'])
            if 'approx_tokens' in extra:
                approx_tokens_list.append(extra['approx_tokens'])
            if 'chars' in extra:
                chars_list.append(extra['chars'])
    # Correlations
    def pearson(x,y):
        if len(x) < 2 or len(x)!=len(y):
            return 0.0
        mx = statistics.mean(x); my = statistics.mean(y)
        num = sum((a-mx)*(b-my) for a,b in zip(x,y))
        denx = math.sqrt(sum((a-mx)**2 for a in x)); deny = math.sqrt(sum((b-my)**2 for b in y))
        if denx==0 or deny==0:
            return 0.0
        return num/(denx*deny)
    pearson_tokens_latency = pearson(approx_tokens_list, provider_latencies) if approx_tokens_list and provider_latencies else 0.0
    pearson_chars_latency = pearson(chars_list, provider_latencies) if chars_list and provider_latencies else 0.0

    # Spearman (rank) correlation
    def spearman(x,y):
        if len(x) < 2 or len(x)!=len(y):
            return 0.0
        # rank with average for ties
        def ranks(vals):
            sorted_vals = sorted((v,i) for i,v in enumerate(vals))
            ranks: list[float] = [0.0]*len(vals)
            i=0
            while i < len(sorted_vals):
                j=i
                while j < len(sorted_vals) and sorted_vals[j][0]==sorted_vals[i][0]:
                    j+=1
                avg_rank = (i + j - 1)/2 + 1
                for k in range(i,j):
                    ranks[sorted_vals[k][1]] = avg_rank
                i=j
            return ranks
        rx = ranks(x); ry = ranks(y)
        return pearson(rx, ry)

    spearman_tokens_latency = spearman(approx_tokens_list, provider_latencies) if approx_tokens_list and provider_latencies else 0.0
    spearman_chars_latency = spearman(chars_list, provider_latencies) if chars_list and provider_latencies else 0.0

    # Bootstrap confidence intervals for pearson tokens_provider
    def bootstrap_ci(x,y, corr_fn, iters=500, alpha=0.05):
        if len(x) < 3:
            return (0.0,0.0)
        pairs = list(zip(x,y))
        vals = []
        for _ in range(iters):
            sample = [random.choice(pairs) for __ in range(len(pairs))]
            sx, sy = zip(*sample)
            vals.append(corr_fn(sx, sy))
        vals.sort()
        lo_idx = max(0, int((alpha/2)*len(vals))-1)
        hi_idx = min(len(vals)-1, int((1-alpha/2)*len(vals))-1)
        return (vals[lo_idx], vals[hi_idx])

    pearson_tokens_ci = bootstrap_ci(approx_tokens_list, provider_latencies, pearson) if approx_tokens_list and provider_latencies else (0.0,0.0)

    totals_sorted = sorted(totals)
    summary = {
        'count': len(totals),
        'avg_total_ms': sum(totals)/len(totals) if totals else 0,
        'p50_total_ms': pct(totals_sorted, 0.50),
        'p95_total_ms': pct(totals_sorted, 0.95),
        'first_token_avg_ms': sum(first_token_latencies)/len(first_token_latencies) if first_token_latencies else 0,
        'provider_latency_avg_ms': sum(provider_latencies)/len(provider_latencies) if provider_latencies else 0,
        'tokens_per_sec_avg': sum(tokens_per_sec)/len(tokens_per_sec) if tokens_per_sec else 0,
        'approx_tokens_avg': sum(approx_tokens_list)/len(approx_tokens_list) if approx_tokens_list else 0,
        'pearson_tokens_provider_ms': round(pearson_tokens_latency,3),
        'pearson_chars_provider_ms': round(pearson_chars_latency,3),
        'spearman_tokens_provider_ms': round(spearman_tokens_latency,3),
        'spearman_chars_provider_ms': round(spearman_chars_latency,3),
        'pearson_tokens_ci95': [round(pearson_tokens_ci[0],3), round(pearson_tokens_ci[1],3)],
    }
    with OUT_PATH.open('w',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print('STREAMING SUMMARY:', json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
