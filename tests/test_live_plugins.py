"""Live integration tests for GAJA server plugins using real AI model.

These tests intentionally DO NOT mock network calls. They require:
- Running server instance (or will start one via uvicorn import)
- Valid OPENAI_API_KEY in environment (or configured in server_config.json)

Set BASE_URL env var to override default http://127.0.0.1:8001

Pytest markers:
  slow: network + model invocations
  ai: requires AI key

Run:
  pytest -k live_plugins -m "ai" -s
"""
from __future__ import annotations

import os
import time
import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
import requests

TEST_PORT = int(os.getenv("GAJA_TEST_PORT", "8001"))
BASE_URL = os.getenv("GAJA_BASE_URL", f"http://127.0.0.1:{TEST_PORT}")
LOGIN_EMAIL = os.getenv("GAJA_TEST_EMAIL", "demo@mail.com")
LOGIN_PASSWORD = os.getenv("GAJA_TEST_PASSWORD", "demo123")  # In test mode password is ignored
MODEL_PRIMARY = os.getenv("GAJA_MODEL", "gpt-5-nano")
MODEL_PREMIUM = os.getenv("GAJA_MODEL_PREMIUM", "gpt-5-mini")

SERVER_STARTED = False
SERVER_START_LOCK = threading.Lock()
OPENAI_KEY_PRESENT = bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture(autouse=True)
def _skip_if_no_key(request):
    """Automatically skip ai-marked tests if OPENAI_API_KEY missing.

    This prevents hard failures on environments without secrets.
    """
    if 'ai' in request.keywords and not OPENAI_KEY_PRESENT:
        pytest.skip("OPENAI_API_KEY not set; skipping AI integration tests")


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _next_free_port(start: int) -> int:
    port = start
    while port < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:  # free
                return port
        port += 1
    raise RuntimeError("No free port found in range")


def _ensure_server():
    """Ensure single test-mode server with test endpoints is running.

    Always starts in GAJA_TEST_MODE=1 on an isolated port to avoid
    interference with any manually running instance.
    """
    global SERVER_STARTED, TEST_PORT, BASE_URL

    if SERVER_STARTED:
        # Quick probe for presence of test token endpoint
        try:
            r = requests.post(f"{BASE_URL}/api/v1/auth/_test_token", json={"email": LOGIN_EMAIL}, timeout=1)
            if r.status_code != 404:
                return
        except Exception:
            pass
        # Endpoint missing -> treat as stale foreign server; choose new port
        SERVER_STARTED = False

    with SERVER_START_LOCK:
        if SERVER_STARTED:
            return
        # Pick a free port starting from env or 8100 to avoid 8001 conflicts
        base_start = int(os.getenv("GAJA_TEST_PORT", "8100"))
        TEST_PORT = _next_free_port(base_start)
        BASE_URL = f"http://127.0.0.1:{TEST_PORT}"
        os.environ["GAJA_TEST_PORT"] = str(TEST_PORT)
        os.environ["GAJA_BASE_URL"] = BASE_URL
        os.environ["GAJA_TEST_MODE"] = "1"

        server_path = Path(__file__).resolve().parent.parent
        main_file = server_path / "server_main.py"
        if not main_file.exists():
            pytest.skip("Server main file not found; cannot auto-start server")

        def _run():
            import uvicorn  # local import after env set
            uvicorn.run("server_main:app", host="127.0.0.1", port=TEST_PORT, log_level="warning")

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        for _ in range(120):
            if _port_open("127.0.0.1", TEST_PORT):
                # Probe test token endpoint to ensure new code loaded
                try:
                    pr = requests.post(f"{BASE_URL}/api/v1/auth/_test_token", json={"email": LOGIN_EMAIL}, timeout=1)
                    if pr.status_code in (200, 404, 403):
                        SERVER_STARTED = True
                        break
                except Exception:
                    pass
            time.sleep(0.25)
        if not SERVER_STARTED:
            pytest.skip(f"Could not start server on port {TEST_PORT}")


@pytest.fixture(scope="session")
def auth_token() -> str:
    _ensure_server()
    # Clear lock if test mode enabled
    if os.getenv("GAJA_TEST_MODE") in {"1", "true", "True"}:
        try:
            requests.post(
                f"{BASE_URL}/api/v1/auth/_unlock_test_user",
                json={"email": LOGIN_EMAIL},
                timeout=5,
            )
        except Exception:
            pass
    # Retry loop in case of transient lock
    resp = None  # type: ignore
    use_test_token = os.getenv("GAJA_TEST_MODE") in {"1", "true", "True"}
    for attempt in range(12):
        resp = requests.post(
            f"{BASE_URL}/api/v1/auth/_test_token",
            json={"email": LOGIN_EMAIL},
            timeout=3,
        )
        if resp.status_code == 404:
            # Stale server without test endpoint; restart on new port
            globals()["SERVER_STARTED"] = False
            _ensure_server()
            time.sleep(0.5)
            continue
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("token")
            if token:
                return token
        time.sleep(0.5)
    if resp is None:
        raise AssertionError("Login response is None")
    assert resp.status_code == 200, getattr(resp, "text", "")
    data = resp.json()
    return data.get("token", "")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.ai
@pytest.mark.slow
def test_health(auth_token: str):
    r = requests.get(f"{BASE_URL}/api/v1/healthz", headers=_auth_headers(auth_token), timeout=5)
    assert r.status_code == 200, r.text
    js = r.json()
    assert js.get("status") == "healthy"


def test_auth_token_generation(auth_token: str):
    """Basic sanity test (not marked as ai) to verify auth token issuance works
    even when OPENAI_API_KEY is absent (AI tests will be skipped)."""
    assert isinstance(auth_token, str) and len(auth_token) > 20


@pytest.mark.ai
@pytest.mark.slow
def test_list_plugins(auth_token: str):
    r = requests.get(f"{BASE_URL}/api/v1/plugins", headers=_auth_headers(auth_token), timeout=10)
    assert r.status_code == 200, r.text
    plugins = r.json()
    assert isinstance(plugins, list)
    # Ensure core expected plugins discovered
    plugin_slugs = {p['slug'] for p in plugins}
    expected = {"weather_module", "search_module"}
    assert expected & plugin_slugs, f"Missing expected plugins in {plugin_slugs}"


@pytest.mark.ai
@pytest.mark.slow
def test_enable_all_plugins_and_query(auth_token: str):
    # Get plugin list
    r = requests.get(f"{BASE_URL}/api/v1/plugins", headers=_auth_headers(auth_token), timeout=10)
    assert r.status_code == 200
    for p in r.json():
        if not p.get("enabled"):
            tr = requests.patch(
                f"{BASE_URL}/api/v1/plugins/{p['slug']}",
                headers=_auth_headers(auth_token),
                json={"enabled": True},
                timeout=10,
            )
            assert tr.status_code == 200, tr.text

    # Fire an AI query that should allow function calling
    prompt = "Jaka jest pogoda w Warszawie? A potem wyszukaj wiadomości o technologii."
    qr = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers=_auth_headers(auth_token),
        json={"query": prompt, "context": {"user_id": "2"}},
        timeout=60,
    )
    assert qr.status_code == 200, qr.text
    data = qr.json()
    assert data.get("success")
    assert "response" in data


@pytest.mark.ai
@pytest.mark.slow
def test_ai_query_primary_model(auth_token: str):
    prompt = "W jednym zdaniu wyjaśnij czym jest polska flaga."
    qr = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers=_auth_headers(auth_token),
        json={"query": prompt, "context": {"user_id": "2"}},
        timeout=45,
    )
    assert qr.status_code == 200
    out = qr.json()
    assert out.get("success")
    body = out.get("response")
    assert body


@pytest.mark.ai
@pytest.mark.slow
@pytest.mark.parametrize("premium", [False, True])
def test_model_comparison(auth_token: str, premium: bool):
    """Compare gpt-5-nano vs (optionally) gpt-5-mini for a quality heuristic.
    Premium test only runs if premium flag True and OPENAI key is present.
    """
    if premium and os.getenv("GAJA_USE_GPT5_MINI") not in {"1", "true", "True"}:
        pytest.skip("Premium model not enabled via GAJA_USE_GPT5_MINI env")

    model = MODEL_PREMIUM if premium else MODEL_PRIMARY
    # Use direct low-level call if available later; for now rely on default model setting
    prompt = "Podaj trzy krótkie, kreatywne zastosowania sztucznej inteligencji w edukacji w formie listy."  # noqa: E501
    qr = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers=_auth_headers(auth_token),
        json={"query": prompt, "context": {"user_id": "2", "force_model": model}},
        timeout=60,
    )
    assert qr.status_code == 200
    out = qr.json()
    assert out.get("success")
    resp_txt = out.get("response")
    # Normalization safeguard: unwrap dict variants to text
    if isinstance(resp_txt, dict):
        if "text" in resp_txt:
            resp_txt = resp_txt.get("text")
        elif "response" in resp_txt and isinstance(resp_txt.get("response"), str):
            resp_txt = resp_txt.get("response")
        else:
            resp_txt = json.dumps(resp_txt, ensure_ascii=False)
    assert isinstance(resp_txt, str) and resp_txt.strip()
    # Heuristic quality metric: either bullet markers/numbering OR at least 3 sentence-like segments
    markers = any(b in resp_txt for b in ["1.", "-", "•"])
    # Count rough sentences (split on '.' and filter length)
    sentence_count = len([s for s in resp_txt.split('.') if s.strip()])
    assert markers or sentence_count >= 3, resp_txt


@pytest.mark.ai
@pytest.mark.slow
def test_weather_plugin_direct(auth_token: str):
    # Ensure weather plugin is enabled
    r = requests.get(f"{BASE_URL}/api/v1/plugins", headers=_auth_headers(auth_token), timeout=10)
    assert r.status_code == 200
    weather_enabled = any(p['slug']== 'weather_module' and p['enabled'] for p in r.json())
    if not weather_enabled:
        tr = requests.patch(
            f"{BASE_URL}/api/v1/plugins/weather_module",
            headers=_auth_headers(auth_token),
            json={"enabled": True},
            timeout=10,
        )
        assert tr.status_code == 200
    # Query AI specifically about weather to trigger function
    prompt = "Sprawdź pogodę dla Krakowa i odpowiedz z temperaturą."
    qr = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers=_auth_headers(auth_token),
        json={"query": prompt, "context": {"user_id": "2"}},
        timeout=60,
    )
    assert qr.status_code == 200
    out = qr.json()
    assert out.get("success")


@pytest.mark.ai
@pytest.mark.slow
def test_search_plugin_direct(auth_token: str):
    r = requests.get(f"{BASE_URL}/api/v1/plugins", headers=_auth_headers(auth_token), timeout=10)
    assert r.status_code == 200
    search_enabled = any(p['slug']== 'search_module' and p['enabled'] for p in r.json())
    if not search_enabled:
        tr = requests.patch(
            f"{BASE_URL}/api/v1/plugins/search_module",
            headers=_auth_headers(auth_token),
            json={"enabled": True},
            timeout=10,
        )
        assert tr.status_code == 200
    prompt = "Wyszukaj najnowsze informacje o Python 3.13 i streść w dwóch zdaniach."
    qr = requests.post(
        f"{BASE_URL}/api/v1/ai/query",
        headers=_auth_headers(auth_token),
        json={"query": prompt, "context": {"user_id": "2"}},
        timeout=60,
    )
    assert qr.status_code == 200
    out = qr.json()
    assert out.get("success")


# Additional plugin tests can be appended following same pattern for new modules.
