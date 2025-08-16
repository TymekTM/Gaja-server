import json, os, time

PERF_FILE = 'user_data/perf_harness_summary.json'
PERF_STREAM_FILE = 'user_data/perf_streaming_summary.json'
REPORT_JSON = 'user_data/test_performance_report.json'
REPORT_MD = 'user_data/test_performance_report.md'

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def main():
    perf = load_json(PERF_FILE)
    stream_perf = load_json(PERF_STREAM_FILE)
    if not perf and not stream_perf:
        print('No performance summaries found.')
        return
    existing = load_json(REPORT_JSON) or {}
    if perf:
        existing['latest_run'] = perf
    if stream_perf:
        existing['latest_stream_run'] = stream_perf
    existing['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(REPORT_JSON,'w',encoding='utf-8') as f:
        json.dump(existing,f,ensure_ascii=False,indent=2)
    md_lines = [
        '# Performance Report (Auto-Updated)',
        '',
        f"Updated: {existing['updated_at']}"
    ]
    if perf:
        md_lines += [
            '## Totals (Non-Streaming)',
            json.dumps(perf['totals'], ensure_ascii=False, indent=2),
            '',
            '## Stages (Non-Streaming)',
            json.dumps(perf['stages'], ensure_ascii=False, indent=2),
            ''
        ]
    if stream_perf:
        md_lines += [
            '## Streaming Summary',
            json.dumps(stream_perf, ensure_ascii=False, indent=2),
            ''
        ]
    md_lines += [
        '## Notes',
        '- singleton + prompt cache active',
        '- streaming metrics: first_token_avg_ms, tokens_per_sec_avg, correlation tokens vs provider latency',
    ]
    with open(REPORT_MD,'w',encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print('Performance report updated.')

if __name__ == '__main__':
    main()
