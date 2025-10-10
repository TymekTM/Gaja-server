import json
import time

from core.app_paths import migrate_legacy_file, resolve_data_path

PERF_PATH = resolve_data_path('perf_harness_summary.json', create_parents=True)
migrate_legacy_file('user_data/perf_harness_summary.json', PERF_PATH)
PERF_STREAM_PATH = resolve_data_path('perf_streaming_summary.json', create_parents=True)
migrate_legacy_file('user_data/perf_streaming_summary.json', PERF_STREAM_PATH)
REPORT_JSON_PATH = resolve_data_path('test_performance_report.json', create_parents=True)
migrate_legacy_file('user_data/test_performance_report.json', REPORT_JSON_PATH)
REPORT_MD_PATH = resolve_data_path('test_performance_report.md', create_parents=True)
migrate_legacy_file('user_data/test_performance_report.md', REPORT_MD_PATH)

def load_json(path):
    if not path.exists():
        return None
    with path.open('r',encoding='utf-8') as f:
        return json.load(f)

def main():
    perf = load_json(PERF_PATH)
    stream_perf = load_json(PERF_STREAM_PATH)
    if not perf and not stream_perf:
        print('No performance summaries found.')
        return
    existing = load_json(REPORT_JSON_PATH) or {}
    if perf:
        existing['latest_run'] = perf
    if stream_perf:
        existing['latest_stream_run'] = stream_perf
    existing['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with REPORT_JSON_PATH.open('w',encoding='utf-8') as f:
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
    with REPORT_MD_PATH.open('w',encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print('Performance report updated.')

if __name__ == '__main__':
    main()
