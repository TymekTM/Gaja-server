#!/usr/bin/env python3
"""Simple performance benchmark for GAJA server.

Metrics captured:
- Cold startup time until health endpoint responsive.
- Health endpoint latency distribution over N probes.
- Optional /docs endpoint first-byte latency.

Usage: python perf/benchmark.py [--host 127.0.0.1] [--port 8001] [--probes 50] [--startup-timeout 40]
"""
from __future__ import annotations
import argparse, subprocess, sys, time, statistics, json, os, pathlib
from typing import List, Dict

try:
    import requests
except ImportError:
    print("requests required. pip install requests")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
START_SCRIPT = ROOT / 'start.py'


def wait_for_health(host: str, port: int, timeout: float) -> float | None:
    url = f"http://{host}:{port}/health"
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - start
        except Exception:
            pass
        time.sleep(0.5)
    return None


def measure_latency(host: str, port: int, probes: int) -> Dict:
    url = f"http://{host}:{port}/health"
    samples: List[float] = []
    for _ in range(probes):
        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=5)
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                samples.append(elapsed)
        except Exception:
            pass
        time.sleep(0.2)
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "min_ms": round(min(samples), 2),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(sorted(samples)[int(0.95 * (len(samples)-1))], 2),
        "max_ms": round(max(samples), 2),
        "mean_ms": round(statistics.mean(samples), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8001)
    ap.add_argument('--probes', type=int, default=30)
    ap.add_argument('--startup-timeout', type=float, default=40.0)
    ap.add_argument('--no-start', action='store_true', help='Assume server already running')
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault('GAJA_TEST_MODE', '1')  # lighter auth path

    proc = None
    startup_time = None
    if not args.no_start:
        # Launch server (console mode) in background
        proc = subprocess.Popen([sys.executable, str(START_SCRIPT)], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Non-blocking read while waiting for health
        startup_time = wait_for_health(args.host, args.port, args.startup_timeout)
    else:
        startup_time = wait_for_health(args.host, args.port, args.startup_timeout)

    health_latency = measure_latency(args.host, args.port, args.probes)

    # Terminate process if we started it
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    result = {
        "startup_seconds": startup_time,
        "health_latency": health_latency,
        "probes": args.probes,
    }
    print(json.dumps(result, indent=2))

    # Also store to perf/results.json
    out_dir = ROOT / 'perf'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
