"""Simple heuristic benchmark comparing gpt-5-nano vs gpt-5-mini.

Usage:
  OPENAI_API_KEY=... GAJA_USE_GPT5_MINI=1 python -m tests.model_benchmark

It sends a small fixed prompt suite and measures:
 - latency
 - token length (approx)
 - presence of required keywords

Decides if premium model is "SIGNIFICANTLY BETTER" by:
 1. At least 30% average length increase (proxy for richness) AND
 2. >=1 additional keyword hits across suite.

Adjust heuristics as needed. Real semantic eval would require external judges.
"""
from __future__ import annotations

import os
import time
import json
import statistics as stats
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import requests

BASE_URL = os.getenv("GAJA_BASE_URL", "http://127.0.0.1:8001")
MODEL_PRIMARY = os.getenv("GAJA_MODEL", "gpt-5-nano")
MODEL_PREMIUM = os.getenv("GAJA_MODEL_PREMIUM", "gpt-5-mini")

PROMPTS = [
    {
        "name": "creativity",
        "prompt": "Wymyśl trzy nieoczywiste zastosowania AI w edukacji w punktach.",
        "keywords": ["1", "2", "3"],
    },
    {
        "name": "reasoning",
        "prompt": "W jednym zdaniu porównaj energie odnawialne i paliwa kopalne.",
        "keywords": ["energia", "paliwa"],
    },
    {
        "name": "concise",
        "prompt": "Stwórz haiku o programowaniu w Pythonie.",
        "keywords": ["Python"],
    },
]

@dataclass
class Result:
    model: str
    name: str
    text: str
    latency: float
    length: int
    keyword_hits: int


def _login() -> str:
    resp = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"email": os.getenv("GAJA_TEST_EMAIL", "demo@mail.com"), "password": os.getenv("GAJA_TEST_PASSWORD", "demo123")},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def _query(token: str, prompt: str) -> Tuple[str, float]:
    start = time.time()
    r = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": prompt, "context": {"user_id": "2"}},
        timeout=60,
    )
    latency = time.time() - start
    r.raise_for_status()
    data = r.json()
    raw = data.get("response", "")
    # Attempt to extract text field if JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "text" in parsed:
            raw = parsed["text"]
    except Exception:
        pass
    return raw, latency


def run_suite(model: str, token: str) -> List[Result]:
    out: List[Result] = []
    for item in PROMPTS:
    text, latency = _query(token, item["prompt"])
    length = len(text.split())
    keyword_hits = sum(1 for kw in item["keywords"] if kw.lower() in text.lower())
    out.append(Result(model=model, name=item["name"], text=text, latency=latency, length=length, keyword_hits=keyword_hits))
    return out


def summarize(results: List[Result]) -> Dict[str, Any]:
    latencies = [r.latency for r in results]
    lengths = [r.length for r in results]
    kw_hits = [r.keyword_hits for r in results]
    return {
        "model": results[0].model if results else "?",
        "avg_latency": stats.mean(latencies),
        "avg_length": stats.mean(lengths),
        "total_keyword_hits": sum(kw_hits),
    }


def main():
    token = _login()
    print(f"Logged in. Testing models: {MODEL_PRIMARY} vs {MODEL_PREMIUM}")
    primary_results = run_suite(MODEL_PRIMARY, token)
    primary_summary = summarize(primary_results)
    print("Primary:", json.dumps(primary_summary, ensure_ascii=False, indent=2))

    if os.getenv("GAJA_USE_GPT5_MINI") not in {"1", "true", "True"}:
        print("Premium model disabled via env. Skipping premium comparison.")
        return

    premium_results = run_suite(MODEL_PREMIUM, token)
    premium_summary = summarize(premium_results)
    print("Premium:", json.dumps(premium_summary, ensure_ascii=False, indent=2))

    # Decision heuristic
    length_gain = (premium_summary["avg_length"] - primary_summary["avg_length"]) / max(primary_summary["avg_length"], 1)
    keyword_gain = premium_summary["total_keyword_hits"] - primary_summary["total_keyword_hits"]
    significantly_better = length_gain >= 0.30 and keyword_gain >= 1
    decision = {
        "length_gain_pct": round(length_gain * 100, 1),
        "keyword_gain": keyword_gain,
        "significantly_better": significantly_better,
        "recommend_upgrade": bool(significantly_better),
    }
    print("Decision:", json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
