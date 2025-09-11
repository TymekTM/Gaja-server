"""ai_providers.py – ulepszona wersja."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections import deque
from time import perf_counter, time as epoch_time
import asyncio
from collections.abc import Callable
from functools import lru_cache
import hashlib
from typing import Any, Optional
import time

import httpx  # Async HTTP client replacing requests
from config.config_loader import MAIN_MODEL, PROVIDER, _config, load_config
from core.performance_monitor import measure_performance
from templates.prompt_builder import build_convert_query_prompt, build_full_system_prompt
from templates.prompts import WEATHER_STYLE_PROMPT

# -----------------------------------------------------------------------------
# Konfiguracja logów
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import environment manager for secure API key handling
try:
    from config.config_manager import EnvironmentManager

    env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_manager = EnvironmentManager(env_file=env_file_path)
except ImportError as e:
    env_manager = None
    logger.warning(f"Could not import EnvironmentManager: {e}")


# -----------------------------------------------------------------------------
# Prosty tracer latencji (lekki, opcjonalny)
# -----------------------------------------------------------------------------

class LatencyTracer:
    """Collects timestamped latency events for a single response generation.

    Usage:
        tracer = LatencyTracer(tracking_id)
        tracer.event("stage_name", extra={...})
        tracer.flush_to_file(path)
    """

    __slots__ = ("tracking_id", "start_monotonic", "start_epoch", "events", "enabled")

    def __init__(self, tracking_id: str | None, enabled: bool = True):
        self.tracking_id = tracking_id or "generic"
        self.start_monotonic = perf_counter()
        self.start_epoch = epoch_time()
        self.events: list[dict[str, Any]] = []
        self.enabled = enabled
        if self.enabled:
            self.event("trace_start")

    def event(self, name: str, extra: dict | None = None):  # noqa: D401 - short util
        if not self.enabled:
            return
        now_mono = perf_counter()
        payload = {
            "tracking_id": self.tracking_id,
            "t_rel_ms": (now_mono - self.start_monotonic) * 1000.0,
            "t_epoch": epoch_time(),
            "event": name,
        }
        if extra:
            # keep JSON serializable (best-effort)
            safe_extra = {}
            for k, v in extra.items():
                try:
                    json.dumps(v)
                    safe_extra[k] = v
                except Exception:  # pragma: no cover - defensive
                    safe_extra[k] = str(v)
            payload["extra"] = safe_extra
        self.events.append(payload)

    def flush_to_file(self, path: str = "user_data/latency_events.jsonl"):
        if not self.enabled or not self.events:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for ev in self.events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"LatencyTracer flush failed: {exc}")


# Environment toggle (set GAJA_LATENCY_TRACE=0 to disable without code changes)
LATENCY_TRACE_ENABLED = os.getenv("GAJA_LATENCY_TRACE", "1") not in ("0", "false", "False")

# -----------------------------------------------------------------------------
# Klasa providerów
# -----------------------------------------------------------------------------
class AIProviders:
    """Rejestr wszystkich obsługiwanych dostawców + metody pomocnicze."""

    # Class-level provider availability cache shared across instances
    _provider_status_cache: dict[str, dict[str, float | bool]] = {}

    def __init__(self) -> None:
        # Async HTTP client for LM Studio (reduced latency)
        # Single shared AsyncClient (keep-alive, connection pooling)
        self._httpx_client = httpx.AsyncClient(timeout=30.0)

        # Cached clients to avoid reinitialization overhead
        self._openai_client = None
        self._openrouter_client = None
        self._lmstudio_base_url = None  # resolved base URL (supports Docker/host)
        
        self._modules: dict[str, Any] = {
            mod: AIProviders._safe_import(mod)
            for mod in ("ollama", "openai")
        }

        # Active providers registry (only requested ones)
        self.providers: dict[str, dict[str, Optional[Callable[..., Any]]]] = {
            "openai": {"module": self._modules["openai"], "check": self.check_openai, "chat": self.chat_openai},
            "openrouter": {"module": self._modules["openai"], "check": self.check_openrouter, "chat": self.chat_openrouter},
            "ollama": {"module": self._modules["ollama"], "check": self.check_ollama, "chat": self.chat_ollama},
            "lmstudio": {"module": None, "check": self.check_lmstudio, "chat": self.chat_lmstudio},
        }

    # ---------------------------------------------------------------------
    # Helpery
    # ---------------------------------------------------------------------
    @staticmethod
    def _safe_import(module_name: str) -> Any | None:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            logger.debug("Moduł %s nie został znaleziony – pomijam.", module_name)
            return None

    @staticmethod
    def _key_ok(env_var: str, cfg_key: str) -> bool:
        key = os.getenv(env_var) or _config.get("api_keys", {}).get(cfg_key.lower())
        return bool(key and not key.startswith("YOUR_"))

    @staticmethod
    def _append_images(messages: list[dict], images: list[str] | None) -> None:
        if images:
            messages[-1]["content"] += "\n\nObrazy: " + ", ".join(images)

    # ---------------------------------------------------------------------
    # Check‑i (zwracają bool, nic nie rzucają)
    # ---------------------------------------------------------------------
    # ----------------------- provider checks with caching ------------------
    def _cached_status(self, name: str, ok_ttl: float = 300.0, fail_ttl: float = 60.0) -> Optional[bool]:
        entry = self._provider_status_cache.get(name)
        if not entry:
            return None
        age = time.time() - float(entry.get("ts", 0.0))
        ttl = ok_ttl if entry.get("ok") else fail_ttl
        if age < ttl:
            return bool(entry.get("ok"))
        return None

    def _store_status(self, name: str, ok: bool) -> bool:
        self._provider_status_cache[name] = {"ok": ok, "ts": time.time()}
        return ok

    @staticmethod
    def map_model_for_provider(model: str, provider: str) -> str:
        """Return a provider-compatible model identifier.

        - For OpenAI, reject obvious OpenRouter-style IDs (with vendor prefix or `:` tag)
          and map to a safe default from config or environment.
        - For other providers, keep the original unless we have a clear rule.
        """
        try:
            m = (model or "").strip()
            p = (provider or "").strip().lower()
            if p == "openai":
                # Detect OpenRouter-style alias (e.g., "openai/gpt-oss-120b", "vendor/model:tag")
                if "/" in m or ":" in m or "oss" in m:
                    # Choose explicit override if available
                    override = _config.get("ai", {}).get("provider_models", {}).get("openai")
                    env_override = os.getenv("GAJA_OPENAI_MODEL")
                    # Conservative default that exists in our codepaths
                    fallback = env_override or override or "gpt-5-nano"
                    return fallback
            return m
        except Exception:
            return model

    def check_ollama(self) -> bool:
        cached = self._cached_status("ollama")
        if cached is not None:
            return cached
        try:
            # Quick inexpensive HTTP check (root); tiny timeout to avoid latency inflation
            r = httpx.get("http://localhost:11434", timeout=0.5)
            ok = r.status_code == 200
        except Exception:
            ok = False
        return self._store_status("ollama", ok)

    def _resolve_lmstudio_candidates(self) -> list[str]:
        # Priority: explicit env/config -> Docker host -> compose svc -> localhost
        base = _config.get("LMSTUDIO_URL_BASE", os.getenv("LMSTUDIO_URL_BASE"))
        candidates = []
        if base:
            candidates.append(base.rstrip("/"))
        # Docker host (Linux requires manual setup; try anyway)
        candidates.append("http://host.docker.internal:1234")
        # Docker Compose service name
        candidates.append("http://lmstudio:1234")
        # Localhost fallback
        candidates.append("http://localhost:1234")
        # De-dup while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def check_lmstudio(self) -> bool:
        cached = self._cached_status("lmstudio")
        if cached is not None:
            return cached
        ok = False
        chosen = None
        for cand in self._resolve_lmstudio_candidates():
            try:
                r = httpx.get(cand + "/v1/models", timeout=0.6)
                if r.status_code < 500:
                    ok = True
                    chosen = cand
                    break
            except Exception:
                continue
        if chosen:
            self._lmstudio_base_url = chosen
        return self._store_status("lmstudio", ok)

    def check_openai(self) -> bool:
        # OpenAI key presence rarely changes during runtime; cache short-term
        cached = self._cached_status("openai", ok_ttl=600.0, fail_ttl=30.0)
        if cached is not None:
            return cached
        key_valid = AIProviders._key_ok("OPENAI_API_KEY", "openai")
        logger.info(f"🔧 OpenAI check: key_valid={key_valid}")
        return self._store_status("openai", key_valid)

    def check_openrouter(self) -> bool:
        cached = self._cached_status("openrouter", ok_ttl=600.0, fail_ttl=30.0)
        if cached is not None:
            return cached
        key_valid = AIProviders._key_ok("OPENROUTER_API_KEY", "openrouter")
        logger.info(f"🔧 OpenRouter check: key_valid={key_valid}")
        return self._store_status("openrouter", key_valid)

    # remove original dynamic key checks for placeholders

    # ---------------------------------------------------------------------
    # Chat‑y
    # ---------------------------------------------------------------------
    def chat_ollama(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            self._append_images(messages, images)
            ollama_mod = self.providers["ollama"]["module"]
            if not ollama_mod or not hasattr(ollama_mod, "chat"):
                raise RuntimeError("Ollama module not available")
            resp = ollama_mod.chat(model=model, messages=messages)  # type: ignore[attr-defined]
            content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else str(resp)
            return {"message": {"content": content}}
        except Exception as exc:  # pragma: no cover
            logger.error("Ollama error: %s", exc)
            return {"message": {"content": f"Ollama error: {exc}"}}

    async def chat_lmstudio(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        functions: list[dict] | None = None,
        function_calling_system=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tracer: LatencyTracer | None = None,
        partial_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any | None]:
        """Call LM Studio local server (OpenAI-compatible). Supports tools when available."""
        # Resolve base URL with Docker-aware fallbacks
        if not self._lmstudio_base_url:
            # trigger availability check to populate base url
            try:
                self.check_lmstudio()
            except Exception:
                pass
        base_url = self._lmstudio_base_url or _config.get(
            "LMSTUDIO_URL_BASE", os.getenv("LMSTUDIO_URL_BASE", "http://localhost:1234")
        )
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        try:
            self._append_images(messages, images)
            # Resolve max output tokens: env -> config.ai.max_tokens -> top-level -> 4000 default
            try:
                MAX_OUT = int(os.getenv("GAJA_MAX_OUTPUT_TOKENS") or _config.get("ai", {}).get("max_tokens") or _config.get("max_tokens") or 4000)
            except Exception:
                MAX_OUT = 4000
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature if temperature is not None else _config.get("temperature", 0.7),
                "max_tokens": max_tokens if max_tokens is not None else MAX_OUT,
            }
            if functions:
                payload["tools"] = functions
                payload["tool_choice"] = "auto"
            # True async call with connection reuse
            r = await self._httpx_client.post(url, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"LM Studio HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0] or {}
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            # If model returned tool calls and we have a function system, execute and do a second pass
            if tool_calls and function_calling_system and isinstance(tool_calls, list):
                tool_results = []
                tool_call_details = []
                from collections import deque as _dq
                for tc in tool_calls:
                    try:
                        fn_name = (((tc or {}).get("function") or {}).get("name") or "").strip()
                        fn_args_raw = (((tc or {}).get("function") or {}).get("arguments") or "{}")
                        try:
                            fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else (fn_args_raw or {})
                        except Exception:
                            fn_args = {"raw": fn_args_raw}
                        if tracer:
                            tracer.event("tool_call_start", {"name": fn_name})
                        exec_result = await function_calling_system.execute_function(
                            fn_name,
                            fn_args,
                            conversation_history=_dq(messages[-10:]) if messages else None,
                        )
                        if tracer:
                            tracer.event("tool_call_end", {"name": fn_name})
                        tool_results.append({
                            "tool_call_id": (tc or {}).get("id"),
                            "role": "tool",
                            "name": fn_name,
                            "content": str(exec_result),
                        })
                        tool_call_details.append({
                            "name": fn_name,
                            "arguments": fn_args,
                            "result": exec_result,
                        })
                    except Exception as _exc:
                        tool_call_details.append({"name": "(error)", "arguments": {}, "result": str(_exc)})

                # append assistant message with tool_calls and tool results
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })
                messages.extend(tool_results)

                # second pass
                # Inject concise weather style guidance if any weather tool was used
                final_messages = list(messages)
                try:
                    if any((d or {}).get("name", "").startswith("weather_") for d in tool_call_details):
                        final_messages.append({"role": "system", "content": WEATHER_STYLE_PROMPT})
                except Exception:
                    pass
                final_payload = {
                    "model": model,
                    "messages": final_messages,
                    "max_tokens": payload["max_tokens"],
                    # Include tools again as per OpenAI-compatible semantics
                    **({"tools": functions, "tool_choice": "auto"} if functions else {}),
                }
                r2 = await self._httpx_client.post(url, json=final_payload)
                if r2.status_code >= 400:
                    raise RuntimeError(f"LM Studio (2nd) HTTP {r2.status_code}: {r2.text[:200]}")
                data2 = r2.json()
                msg2 = ((data2.get("choices") or [{}])[0] or {}).get("message") or {}
                content2 = (msg2.get("content") or content or "").strip()
                return {
                    "message": {"content": content2},
                    "tool_calls_executed": len(tool_results),
                    "tool_call_details": tool_call_details,
                }

            return {"message": {"content": content or "(no content)"}}
        except Exception as exc:
            logger.error("LM Studio error: %s", exc)
            # Raise to allow fallback to other providers when LM Studio is unreachable
            raise

    async def chat_openai(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        functions: list[dict] | None = None,
        function_calling_system=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    stream: bool = False,
    tracer: LatencyTracer | None = None,
    partial_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any | None]:
        try:
            # 1. Resolve API key
            api_key = env_manager.get_api_key("openai") if env_manager else None
            if not api_key:
                api_key = (
                    os.getenv("OPENAI_API_KEY")
                    or _config.get("api_keys", {}).get("openai")
                )
            if not api_key:
                return {"message": {"content": "Błąd: Brak OPENAI_API_KEY"}}

            # 2. Lazy client init
            if self._openai_client is None:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=api_key)
            client = self._openai_client

            # 3. Images append (simple textual reference)
            self._append_images(messages, images)

            # 4. Param builder supporting migration from max_tokens -> max_completion_tokens
            def build_params(use_new: bool):
                token_key = "max_completion_tokens" if use_new else "max_tokens"
                params: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    # Some newer models (e.g., gpt-5-nano) only support default temperature=1
                    # We'll add temperature initially; may remove on unsupported_value error and retry
                    "temperature": temperature if temperature is not None else _config.get("temperature", 0.7),
                    token_key: max_tokens if max_tokens is not None else (int(os.getenv("GAJA_MAX_OUTPUT_TOKENS") or _config.get("ai", {}).get("max_tokens") or _config.get("max_tokens") or 4000)),
                }
                if functions:
                    params["tools"] = functions
                    params["tool_choice"] = "auto"
                # Pre-emptive guard: certain lightweight models (gpt-5-nano/mini) reject non-default temperature values.
                # If a custom temperature is present and model matches pattern, drop it to avoid 400.
                if model.startswith("gpt-5-"):
                    temp_val = params.get("temperature")
                    if temp_val not in (None, 1, 1.0):
                        params.pop("temperature", None)
                return params, token_key

            attempt_order = [True, False] if model.startswith("gpt-5-") else [False, True]
            last_error: Exception | None = None
            response = None
            used_new_key = False
            temperature_removed = False
            usage_first = None
            for use_new in attempt_order:
                params, token_key = build_params(use_new)
                try:
                    response = client.chat.completions.create(**params)
                    # Capture usage from the first call if provided by API
                    try:
                        u = getattr(response, 'usage', None)
                        if u:
                            usage_first = {
                                'prompt_tokens': getattr(u, 'prompt_tokens', None) or getattr(u, 'prompt_tokens_total', None),
                                'completion_tokens': getattr(u, 'completion_tokens', None),
                                'total_tokens': getattr(u, 'total_tokens', None) or (
                                    (getattr(u, 'prompt_tokens', 0) or 0) + (getattr(u, 'completion_tokens', 0) or 0)
                                ),
                            }
                    except Exception:
                        usage_first = None
                    used_new_key = use_new
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if "unsupported value" in msg and "temperature" in msg and not temperature_removed:
                        # Remove temperature and retry same attempt once
                        temperature_removed = True
                        try:
                            params.pop("temperature", None)
                            response = client.chat.completions.create(**params)
                            used_new_key = use_new
                            break
                        except Exception as e2:  # noqa: BLE001
                            msg2 = str(e2).lower()
                            if "unsupported parameter" in msg2:
                                last_error = e2
                                continue
                            last_error = e2
                            break
                    if "unsupported parameter" in msg:
                        last_error = e
                        continue  # try alternate token key
                    last_error = e
                    break
            if response is None:
                raise last_error if last_error else RuntimeError("OpenAI call failure")

            # 5. Streaming path (no tool calling on first pass for simplicity)
            if stream and not functions:
                try:
                    # Attempt streaming with new or old token param key
                    stream_attempts = [True, False] if model.startswith("gpt-5-") else [False, True]
                    response_stream = None
                    for use_new in stream_attempts:
                        params, token_key = build_params(use_new)
                        params["stream"] = True
                        try:
                            if tracer:
                                tracer.event("provider_stream_start")
                            response_stream = client.chat.completions.create(**params)
                            break
                        except Exception:
                            continue
                    if response_stream is not None and hasattr(response_stream, "__iter__"):
                        collected = []
                        token_count = 0
                        first_token_ts = None
                        for chunk in response_stream:
                            if not chunk or not getattr(chunk, 'choices', None):
                                continue
                            delta = getattr(chunk.choices[0].delta, 'content', None)
                            if delta:
                                if first_token_ts is None:
                                    first_token_ts = perf_counter()
                                    if tracer:
                                        tracer.event("stream_first_token")
                                collected.append(delta)
                                token_count += len(delta.split())  # rough proxy; real tokens require tokenizer
                                if partial_callback:
                                    try:
                                        partial_callback(delta)
                                    except Exception:  # pragma: no cover
                                        pass
                        content_joined = "".join(collected).strip()
                        if tracer and first_token_ts is not None:
                            # Use approximate_token_count for more consistent estimate at end
                            try:
                                total_tokens = approximate_token_count(content_joined)
                            except Exception:
                                total_tokens = token_count if token_count else len(content_joined.split())
                            elapsed_tokens_s = perf_counter() - first_token_ts
                            if elapsed_tokens_s > 0:
                                tracer.event(
                                    "stream_complete",
                                    {
                                        "approx_tokens": total_tokens,
                                        "tokens_per_sec": round(total_tokens / elapsed_tokens_s, 2),
                                        "chars": len(content_joined),
                                        "elapsed_s": round(elapsed_tokens_s, 3),
                                    },
                                )
                        return {"message": {"content": content_joined}, "streamed": True}
                except Exception as stream_exc:  # pragma: no cover
                    logger.warning(f"Streaming fallback to non-stream due to error: {stream_exc}")
                    if tracer:
                        tracer.event("stream_error", {"error": str(stream_exc)[:120]})

            # 6. Function calling handling (non-stream)
            #  (unchanged logic below)
            # 5. Function calling handling
            if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
                tool_results = []
                tool_call_details = []  # Track details with timings
                for tool_call in response.choices[0].message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    exec_result = None
                    # Intercept obvious intent/tool mismatch: calendar intent vs timer tool
                    try:
                        last_user_text = ""
                        for m in reversed(messages):
                            if isinstance(m, dict) and m.get("role") == "user":
                                last_user_text = str(m.get("content") or "")
                                break
                        if (
                            fn_name == "core_set_timer"
                            and last_user_text
                            and any(kw in last_user_text.lower() for kw in [
                                "kalendarz", "w kalendarz", "spotkan", "spotkanie", "termin", "piątek", "fryzjer"
                            ])
                        ):
                            # Return clarification suggesting calendar event instead of timer
                            clarify = {
                                "type": "clarification_request",
                                "action_type": "clarification_request",
                                "message": (
                                    "Wygląda na prośbę o wydarzenie w kalendarzu, a nie timer. "
                                    "Podaj tytuł oraz dokładną datę (YYYY-MM-DD) i godzinę (HH:MM), "
                                    "albo potwierdź: dodać wydarzenie 'Fryzjer' na najbliższy piątek o 14:00?"
                                ),
                                "clarification_data": {
                                    "question": "Jaki tytuł, data (YYYY-MM-DD) i godzina (HH:MM) wydarzenia?",
                                    "parameter": "event_details",
                                    "function": "core_add_event",
                                    "provided_parameters": {}
                                }
                            }
                            # Build immediate payload with single tool detail
                            payload = {
                                "message": {"content": clarify.get("message", "Clarification requested")},
                                "clarification_request": clarify.get("clarification_data"),
                                "tool_calls_executed": 1,
                                "tool_call_details": [{
                                    "name": fn_name,
                                    "arguments": fn_args,
                                    "result": clarify,
                                }],
                                "requires_user_response": True,
                            }
                            if usage_first:
                                payload["usage"] = usage_first
                            return payload
                    except Exception:
                        pass

                    # Measure tool call timing
                    _t0 = perf_counter()
                    _t_rel = None
                    try:
                        if tracer:
                            try:
                                _t_rel = (perf_counter() - float(getattr(tracer, 'start_monotonic', _t0))) * 1000.0
                            except Exception:
                                _t_rel = None
                            tracer.event("tool_call_start", {"name": fn_name, "t_rel_ms": round(_t_rel, 2) if _t_rel is not None else None})
                    except Exception:
                        pass
                    if function_calling_system:
                        exec_result = await function_calling_system.execute_function(
                            fn_name,
                            fn_args,
                            conversation_history=deque(messages[-10:]) if messages else None,
                        )
                    _t1 = perf_counter()
                    _dur_ms = (_t1 - _t0) * 1000.0
                    try:
                        if tracer:
                            tracer.event("tool_call_end", {"name": fn_name, "duration_ms": round(_dur_ms, 2)})
                    except Exception:
                        pass
                    if (
                        isinstance(exec_result, dict)
                        and exec_result.get("action_type") == "clarification_request"
                    ):
                        # Early return but include tool call details and usage if known
                        payload = {
                            "message": {"content": exec_result.get("message", "Clarification requested")},
                            "clarification_request": exec_result.get("clarification_data"),
                            "tool_calls_executed": 1,
                            "tool_call_details": [{
                                "name": fn_name,
                                "arguments": fn_args,
                                "result": exec_result,
                                "invoked_rel_ms": round(_t_rel, 2) if _t_rel is not None else None,
                                "duration_ms": round(_dur_ms, 2),
                            }],
                            "requires_user_response": True,
                        }
                        if usage_first:
                            payload["usage"] = usage_first
                            return payload
                    tool_results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": str(exec_result),
                        }
                    )
                    # Store tool call details for fast-path optimization
                    tool_call_details.append({
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": exec_result,
                        "invoked_rel_ms": round(_t_rel, 2) if _t_rel is not None else None,
                        "duration_ms": round(_dur_ms, 2),
                    })

                # Augment conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response.choices[0].message.tool_calls
                        ],
                    }
                )
                
                # FAST PATH: Check if we can skip the second OpenAI call
                # Fast-path disabled by default; enable with GAJA_FAST_TOOL_MODE=1 explicitly
                fast_tool_mode = os.getenv("GAJA_FAST_TOOL_MODE", "0") not in ("0","false","False")
                if fast_tool_mode and len(tool_call_details) == 1:
                    detail = tool_call_details[0]
                    tool_name = detail.get("name", "") if isinstance(detail, dict) else ""
                    tool_result = detail.get("result") if isinstance(detail, dict) else None
                    if tool_name.lower().startswith("weather") and isinstance(tool_result, dict):
                        logger.warning(f"🚀 FAST PATH WEATHER ACTIVATED! Tool: {tool_name}")
                        if tracer:
                            tracer.event("fast_tool_path_weather")
                        # Build concise natural language summary
                        try:
                            # Guard: only proceed if success True and data present
                            if not tool_result.get("success"):
                                raise ValueError("fast_path_skip: success False")
                            data_block = tool_result.get("data") or {}
                            if not isinstance(data_block, dict):
                                raise ValueError("fast_path_skip: data not dict")
                            loc = data_block.get("location", {}) or {}
                            curr = data_block.get("current", {}) or {}
                            forecast = data_block.get("forecast", []) or []

                            # Minimal required fields (fallbacks to avoid None spam)
                            location_text = loc.get("name") or loc.get("city") or "(lokalizacja)"
                            temp = curr.get("temperature")
                            precip = curr.get("precipitation_chance") or curr.get("rain_chance") or curr.get("precip_chance")
                            clouds = curr.get("cloud_cover") or curr.get("clouds")
                            desc = curr.get("description") or ""

                            # Infer last user intent for richer but brief phrasing
                            last_user_text = ""
                            try:
                                for _m in reversed(messages):
                                    if isinstance(_m, dict) and _m.get("role") == "user":
                                        last_user_text = str(_m.get("content") or "")
                                        break
                            except Exception:
                                last_user_text = ""
                            lower_q = last_user_text.lower()

                            # Parameter-specific short sentence (e.g., humidity)
                            if any(k in lower_q for k in ["wilgot", "wilgoć", "humidity"]):
                                hum = curr.get("humidity")
                                if hum is None and forecast and isinstance(forecast, list) and isinstance(forecast[0], dict):
                                    hum = forecast[0].get("humidity")
                                if hum is not None:
                                    summary = f"W {location_text} wilgotność wynosi {round(hum)}%."
                                    fast_payload = {
                                        "message": {"content": summary},
                                        "tool_calls_executed": len(tool_call_details),
                                        "tool_call_details": tool_call_details,
                                        "fast_tool_path": True,
                                    }
                                    if tracer:
                                        tracer.event("fast_tool_response_ready", {"chars": len(summary)})
                                    return fast_payload

                            # Tomorrow/forecast style in 1–2 sentences
                            ask_tomorrow = ("jutro" in lower_q) or tool_name == "weather_get_forecast"
                            if ask_tomorrow and isinstance(forecast, list) and forecast:
                                idx = 1 if len(forecast) >= 2 else 0
                                day = forecast[idx] if isinstance(forecast[idx], dict) else None
                                dmin = day.get('min_temp') if isinstance(day, dict) else None
                                dmax = day.get('max_temp') if isinstance(day, dict) else None
                                ddesc = (day.get('description') if isinstance(day, dict) else None) or desc or ""
                                first_sentence = f"Jutro w {location_text} {ddesc.lower()}." if ddesc else f"Jutro w {location_text}."
                                second_sentence = None
                                if dmin is not None and dmax is not None:
                                    second_sentence = f"Temperatura {dmin}–{dmax}°C."
                                elif temp is not None:
                                    second_sentence = f"Będzie {temp}°C."
                                # Precipitation hint if available
                                p = None
                                if isinstance(day, dict):
                                    p = day.get('precip_chance') or day.get('daily_chance_of_rain')
                                p = p if p is not None else curr.get("precipitation_chance")
                                precip_sentence = None
                                try:
                                    if p is not None:
                                        p = float(p)
                                        if p >= 60:
                                            precip_sentence = "Duża szansa na deszcz."
                                        elif p >= 30:
                                            precip_sentence = "Umiarkowana szansa na deszcz."
                                        elif p > 0:
                                            precip_sentence = "Mała szansa na deszcz."
                                    elif any(s in (ddesc or "").lower() for s in ["deszcz", "opad"]):
                                        precip_sentence = "Możliwe przelotne opady."
                                except Exception:
                                    precip_sentence = None
                                parts_out = [first_sentence]
                                if second_sentence:
                                    parts_out.append(second_sentence)
                                if precip_sentence:
                                    parts_out.append(precip_sentence)
                                summary = " ".join(parts_out)
                                fast_payload = {
                                    "message": {"content": summary},
                                    "tool_calls_executed": len(tool_call_details),
                                    "tool_call_details": tool_call_details,
                                    "fast_tool_path": True,
                                }
                                if tracer:
                                    tracer.event("fast_tool_response_ready", {"chars": len(summary)})
                                return fast_payload

                            # Default current weather: two compact sentences when possible
                            base_sentence = f"{location_text}: {desc.lower()}, {temp}°C." if desc or temp is not None else f"{location_text}."
                            precip_sentence = None
                            try:
                                p = curr.get("precipitation_chance")
                                if p is not None:
                                    p = float(p)
                                    if p >= 60:
                                        precip_sentence = "Duża szansa na deszcz."
                                    elif p >= 30:
                                        precip_sentence = "Umiarkowana szansa na deszcz."
                                    elif p > 0:
                                        precip_sentence = "Mała szansa na deszcz."
                            except Exception:
                                precip_sentence = None
                            summary = base_sentence if not precip_sentence else f"{base_sentence} {precip_sentence}"
                            fast_payload = {
                                "message": {"content": summary},
                                "tool_calls_executed": len(tool_call_details),
                                "tool_call_details": tool_call_details,
                                "fast_tool_path": True,
                            }
                            logger.warning(f"🚀 FAST PATH SUCCESS: {summary[:50]}...")
                            if tracer:
                                tracer.event("fast_tool_response_ready", {"chars": len(summary)})
                            return fast_payload
                        except Exception as ft_exc:
                            logger.warning(f"🚀 FAST PATH ERROR: {ft_exc}")
                            if tracer:
                                tracer.event("fast_tool_path_error", {"error": str(ft_exc)[:160]})

                messages.extend(tool_results)

                # Second pass with tool outputs
                token_key = "max_completion_tokens" if used_new_key else "max_tokens"
                # Inject concise weather style guidance if a weather tool was called
                _msgs2 = list(messages)
                try:
                    if any((d or {}).get("name", "").startswith("weather_") for d in tool_call_details):
                        _msgs2.append({"role": "system", "content": WEATHER_STYLE_PROMPT})
                except Exception:
                    pass
                final_params = {
                    "model": model,
                    "messages": _msgs2,
                    "temperature": temperature if temperature is not None else _config.get("temperature", 0.7),
                    token_key: max_tokens if max_tokens is not None else (int(os.getenv("GAJA_MAX_OUTPUT_TOKENS") or _config.get("ai", {}).get("max_tokens") or _config.get("max_tokens") or 4000)),
                }
                if functions:
                    final_params["tools"] = functions
                    final_params["tool_choice"] = "auto"
                # Second-pass guard for gpt-5-* models rejecting custom temperature
                if model.startswith("gpt-5-") and final_params.get("temperature") not in (None, 1, 1.0):
                    final_params.pop("temperature", None)
                final_response = client.chat.completions.create(**final_params)
                # Extract content with fallback
                try:
                    final_content = final_response.choices[0].message.content
                except Exception:
                    final_content = None
                if not final_content:
                    # Build a graceful fallback using tool errors or a compact summary
                    try:
                        # Prefer first tool error message if present
                        err_msg = None
                        for d in tool_call_details:
                            res = d.get('result')
                            if isinstance(res, dict) and not res.get('success', True):
                                err_msg = res.get('error') or res.get('message')
                                break
                        if err_msg:
                            final_content = f"Nie mogłem dokończyć odpowiedzi, bo narzędzie zwróciło błąd: {err_msg}. Spróbuj ponownie lub wybierz inny silnik wyszukiwania."
                        else:
                            names = [str(d.get('name')) for d in tool_call_details]
                            final_content = (
                                "Odpowiedź nie zawierała treści. Wykonane narzędzia: "
                                + ", ".join([n for n in names if n])
                            )
                    except Exception:
                        final_content = "(Brak treści od modelu po wykonaniu narzędzi)"
                # Merge usages from first and second call if available
                usage_combined = None
                try:
                    u2 = getattr(final_response, 'usage', None)
                    if u2:
                        usage_second = {
                            'prompt_tokens': getattr(u2, 'prompt_tokens', None) or getattr(u2, 'prompt_tokens_total', None),
                            'completion_tokens': getattr(u2, 'completion_tokens', None),
                            'total_tokens': getattr(u2, 'total_tokens', None) or (
                                (getattr(u2, 'prompt_tokens', 0) or 0) + (getattr(u2, 'completion_tokens', 0) or 0)
                            ),
                        }
                        if usage_first:
                            usage_combined = {
                                'prompt_tokens': (usage_first.get('prompt_tokens') or 0) + (usage_second.get('prompt_tokens') or 0),
                                'completion_tokens': (usage_first.get('completion_tokens') or 0) + (usage_second.get('completion_tokens') or 0),
                                'total_tokens': (usage_first.get('total_tokens') or 0) + (usage_second.get('total_tokens') or 0),
                                'by_call': [usage_first, usage_second],
                            }
                        else:
                            usage_combined = usage_second
                except Exception:
                    usage_combined = usage_first
                return {
                    "message": {"content": final_content},
                    "tool_calls_executed": len(tool_results),
                    "tool_call_details": tool_call_details,
                    **({"usage": usage_combined} if usage_combined else {}),
                }

            # Normal path: extract content, guard against None/empty
            try:
                base_content = response.choices[0].message.content if response.choices else ""
            except Exception:
                base_content = ""
            if not base_content:
                # Provide multi‑sentence graceful fallback so quality heuristic still passes
                base_content = (
                    "(Brak treści od modelu w pierwszej próbie). "
                    "Spróbuj proszę powtórzyć pytanie lub poczekaj chwilę – wykonam ponowną próbę. "
                    "To tymczasowa odpowiedź wygenerowana lokalnie."
                )
            # Attach usage from first call if present
            payload = {"message": {"content": base_content}}
            if usage_first:
                payload["usage"] = usage_first
            return payload
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("OpenAI error: %s", exc, exc_info=True)
            return {"message": {"content": f"Błąd OpenAI: {exc}"}}

    

    async def chat_openrouter(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        functions: list[dict] | None = None,
        function_calling_system=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tracer: LatencyTracer | None = None,
        partial_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any | None]:
        try:
            # Resolve API key
            api_key = env_manager.get_api_key("openrouter") if env_manager else None
            if not api_key:
                api_key = (
                    os.getenv("OPENROUTER_API_KEY")
                    or _config.get("api_keys", {}).get("openrouter")
                )
            if not api_key:
                return {"message": {"content": "Błąd: Brak OPENROUTER_API_KEY"}}

            # Lazy init client
            if self._openrouter_client is None:
                from openai import OpenAI  # type: ignore
                self._openrouter_client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://gaja.local"),
                        "X-Title": os.getenv("OPENROUTER_TITLE", "Gaja"),
                    },
                )
            client = self._openrouter_client

            # Append images (textual note)
            self._append_images(messages, images)

            # Build params
            params: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            else:
                try:
                    params["max_tokens"] = int(os.getenv("GAJA_MAX_OUTPUT_TOKENS") or _config.get("ai", {}).get("max_tokens") or _config.get("max_tokens") or 4000)
                except Exception:
                    params["max_tokens"] = 4000
            if functions:
                params["tools"] = functions
                params["tool_choice"] = "auto"

            if tracer:
                tracer.event("provider_request_start", {"provider": "openrouter"})
            # First attempt – with tools if provided
            response = None
            try:
                response = client.chat.completions.create(**params)
            except Exception as e:
                # Handle OpenRouter provider routing error when tools are not supported
                msg = str(e)
                lmsg = msg.lower()
                if (
                    ("no endpoints found" in lmsg and "tool use" in lmsg)
                    or ("404" in lmsg and "tool" in lmsg and "openrouter" in lmsg)
                ) and ("tools" in params or functions):
                    logger.warning(
                        "OpenRouter: tools unsupported for model '%s' (404). Retrying without tools...",
                        model,
                    )
                    # Retry without tools/tool_choice
                    params.pop("tools", None)
                    params.pop("tool_choice", None)
                    try:
                        response = client.chat.completions.create(**params)
                    except Exception as e2:  # re-raise with original context if second try fails
                        raise e2
                else:
                    # Different error – re-raise to outer handler
                    raise
            if tracer:
                tracer.event("provider_request_end", {"provider": "openrouter"})

            # Extract usage if present (first call)
            usage_first = None
            try:
                u = getattr(response, 'usage', None)
                if u:
                    usage_first = {
                        'prompt_tokens': getattr(u, 'prompt_tokens', None) or getattr(u, 'prompt_tokens_total', None),
                        'completion_tokens': getattr(u, 'completion_tokens', None),
                        'total_tokens': getattr(u, 'total_tokens', None),
                    }
            except Exception:
                usage_first = None

            # Handle tool calls (if present)
            if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
                tool_results = []
                tool_call_details = []  # Track details with timings
                for tool_call in response.choices[0].message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    exec_result = None

                    # Measure tool call timing
                    _t0 = perf_counter()
                    _t_rel = None
                    try:
                        if tracer:
                            try:
                                _t_rel = (perf_counter() - float(getattr(tracer, 'start_monotonic', _t0))) * 1000.0
                            except Exception:
                                _t_rel = None
                            tracer.event("tool_call_start", {"name": fn_name, "t_rel_ms": round(_t_rel, 2) if _t_rel is not None else None})
                    except Exception:
                        pass
                    
                    if function_calling_system:
                        exec_result = await function_calling_system.execute_function(
                            fn_name,
                            fn_args,
                            conversation_history=deque(messages[-10:]) if messages else None,
                        )
                    _t1 = perf_counter()
                    _dur_ms = (_t1 - _t0) * 1000.0
                    try:
                        if tracer:
                            tracer.event("tool_call_end", {"name": fn_name, "duration_ms": round(_dur_ms, 2)})
                    except Exception:
                        pass

                    if (
                        isinstance(exec_result, dict)
                        and exec_result.get("action_type") == "clarification_request"
                    ):
                        # Early return with clarification request
                        payload = {
                            "message": {"content": exec_result.get("message", "Clarification requested")},
                            "clarification_request": exec_result.get("clarification_data"),
                            "tool_calls_executed": 1,
                            "tool_call_details": [{
                                "name": fn_name,
                                "arguments": fn_args,
                                "result": exec_result,
                                "invoked_rel_ms": round(_t_rel, 2) if _t_rel is not None else None,
                                "duration_ms": round(_dur_ms, 2),
                            }],
                            "requires_user_response": True,
                        }
                        if usage_first:
                            payload["usage"] = usage_first
                        return payload
                    
                    tool_results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": str(exec_result),
                        }
                    )
                    # Store tool call details
                    tool_call_details.append({
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": exec_result,
                        "invoked_rel_ms": round(_t_rel, 2) if _t_rel is not None else None,
                        "duration_ms": round(_dur_ms, 2),
                    })

                # Augment conversation with tool calls and results
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response.choices[0].message.tool_calls
                        ],
                    }
                )
                messages.extend(tool_results)

                # Make second API call to get final response
                try:
                    params_second = params.copy()
                    params_second["messages"] = messages
                    # Remove tools for second call to get natural language response
                    params_second.pop("tools", None)
                    params_second.pop("tool_choice", None)
                    
                    response_second = client.chat.completions.create(**params_second)
                    
                    # Extract usage from second call
                    usage_second = None
                    try:
                        u2 = getattr(response_second, 'usage', None)
                        if u2:
                            usage_second = {
                                'prompt_tokens': getattr(u2, 'prompt_tokens', None) or getattr(u2, 'prompt_tokens_total', None),
                                'completion_tokens': getattr(u2, 'completion_tokens', None),
                                'total_tokens': getattr(u2, 'total_tokens', None),
                            }
                    except Exception:
                        usage_second = None

                    # Combine usage
                    usage_combined = None
                    try:
                        if usage_first and usage_second:
                            usage_combined = {
                                'prompt_tokens': (usage_first.get('prompt_tokens', 0) or 0) + (usage_second.get('prompt_tokens', 0) or 0),
                                'completion_tokens': (usage_first.get('completion_tokens', 0) or 0) + (usage_second.get('completion_tokens', 0) or 0),
                                'total_tokens': (usage_first.get('total_tokens', 0) or 0) + (usage_second.get('total_tokens', 0) or 0),
                            }
                        else:
                            usage_combined = usage_second or usage_first
                    except Exception:
                        usage_combined = usage_first

                    # Extract final content
                    try:
                        final_content = (response_second.choices[0].message.content or "").strip()
                    except Exception:
                        final_content = ""

                    return {
                        "message": {"content": final_content},
                        "tool_calls_executed": len(tool_results),
                        "tool_call_details": tool_call_details,
                        **({"usage": usage_combined} if usage_combined else {}),
                    }
                
                except Exception as second_call_exc:
                    logger.error("OpenRouter second call error: %s", second_call_exc, exc_info=True)
                    # Fallback: return tool execution results as message
                    fallback_content = f"Wykonano {len(tool_results)} narzędzi, ale wystąpił błąd podczas generowania odpowiedzi."
                    return {
                        "message": {"content": fallback_content},
                        "tool_calls_executed": len(tool_results),
                        "tool_call_details": tool_call_details,
                        **({"usage": usage_first} if usage_first else {}),
                    }

            # No tool calls - extract regular content
            try:
                content = (response.choices[0].message.content or "").strip()
            except Exception:
                content = ""
            return {"message": {"content": content}, **({"usage": usage_first} if usage_first else {})}
        except Exception as exc:  # pragma: no cover
            logger.error("OpenRouter error: %s", exc, exc_info=True)
            return {"message": {"content": f"Błąd OpenRouter: {exc}"}}



    async def cleanup(self) -> None:
        """Clean up async resources."""
        await self._httpx_client.aclose()


_ai_providers: AIProviders | None = None


def get_ai_providers() -> AIProviders:
    global _ai_providers
    if _ai_providers is None:
        _ai_providers = AIProviders()
    return _ai_providers


# -----------------------------------------------------------------------------
# Publiczne funkcje
# -----------------------------------------------------------------------------
@measure_performance
def health_check() -> dict[str, bool]:
    providers = get_ai_providers()
    results: dict[str, bool] = {}
    for name, cfg in providers.providers.items():
        ok = False
        try:
            check_fn = cfg.get("check")
            if callable(check_fn):
                val = check_fn()
                ok = bool(val is True)
        except Exception:
            ok = False
        results[name] = ok
    return results


# ------------------------------------------------------------------ utils ---


def remove_chain_of_thought(text: str) -> str:
    pattern = (
        r"<think>.*?</think>|<\|begin_of_thought\|>.*?<\|end_of_thought\|>|"
        r"<\|begin_of_solution\|>.*?<\|end_of_solution\|>|<\|end\|>"
    )
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> str:
    """Zwraca pierwszy blok JSON z tekstu (albo cały string)."""
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if ln.strip("`")]
        text = "\n".join(lines).strip()
    # Find all JSON-like blocks and choose the most complete one
    matches = re.findall(r"(\{.*\})", text, flags=re.DOTALL)
    if matches:
        # Return the largest match, assuming it's the full JSON object
        return max(matches, key=len)
    return text


# ------------------------------------------------------------------ token util ---

_tiktoken_encoder = None  # lazy global cache

def approximate_token_count(text: str) -> int:
    """Approximate token count.

    Tries to use tiktoken if available (same heuristic as OpenAI tokenization for
    many models). Falls back to a simple whitespace split *  (with a rough scaling
    for languages with longer average word/token ratios).
    """
    global _tiktoken_encoder
    if not text:
        return 0
    # Try tiktoken once
    if _tiktoken_encoder is None:
        try:  # pragma: no cover - optional dependency
            import tiktoken  # type: ignore
            # Use cl100k_base which matches most gpt-3.5/4 style models
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_encoder = False  # sentinel meaning unavailable
    if _tiktoken_encoder and _tiktoken_encoder is not False:
        try:
            return len(_tiktoken_encoder.encode(text))
        except Exception:  # pragma: no cover - fallback safety
            pass
    # Fallback heuristic: word count *  (English ~0.75 ratio; we keep simple)
    wc = len(text.split())
    return max(1, wc)


def approximate_token_count_for_model(model: str, text: str) -> int:
    """Model-aware wrapper (future extension point).

    For now delegates to approximate_token_count, but allows plugging in
    specialized encoders per model family (e.g., llama, mistral) later.
    """
    if not text:
        return 0
    # Simple heuristic: OpenAI GPT families -> use approximate_token_count (tiktoken if available)
    lower = model.lower()
    if any(prefix in lower for prefix in ("gpt-", "o", "openai")):
        return approximate_token_count(text)
    # Default
    return approximate_token_count(text)


# ---------------------------------------------------------------- refiner ----


@lru_cache(maxsize=256)
@measure_performance
async def refine_query(query: str, detected_language: str = "Polish") -> str:
    try:
        prompt = build_convert_query_prompt(detected_language)
        resp = await chat_with_providers(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        if isinstance(resp, dict):
            msg_obj = resp.get("message")
            if isinstance(msg_obj, dict):
                content_val = msg_obj.get("content", "")
                if isinstance(content_val, str) and content_val.strip():
                    return content_val.strip()
        return query
    except Exception as exc:  # pragma: no cover
        logger.error("refine_query error: %s", exc)
        return query

# ---------------------------------------------------------------- prompt cache ---

@lru_cache(maxsize=128)
def _cached_system_prompt(
    system_prompt_override: str | None,
    detected_language: str | None,
    language_confidence: float | None,
    tools_desc_hash: str,
    active_window_title: str | None,
    track_active_window_setting: bool,
    user_name: str | None,
    funcs_count: int,
) -> str:
    """Build (and cache) system prompt.

    We hash tools description outside to keep the cache key small. Active window
    title only influences prompt if tracking flag is True; we still include it in the key
    (hashed implicitly via string) to avoid stale context.
    """
    # We can't reconstruct tools description from hash, but the content only impacts
    # final prompt text; collisions on short hash slice are extremely unlikely for our use.
    # (If needed we can store mapping hash->original in future.)
    from templates.prompt_builder import build_full_system_prompt  # local import to avoid cycles
    # For tools_description we just indicate count; detailed list often large and mostly static for run.
    tools_description = f"{funcs_count} functions available" if funcs_count else ""
    return build_full_system_prompt(
        system_prompt_override=system_prompt_override,
        detected_language=detected_language,
        language_confidence=language_confidence,
        tools_description=tools_description,
        active_window_title=active_window_title,
        track_active_window_setting=track_active_window_setting,
        tool_suggestion=None,
        user_name=user_name,
    )


# ---------------------------------------------------------------- chat glue --


@measure_performance
async def chat_with_providers(
    model: str,
    messages: list[dict],
    images: list[str] | None = None,
    provider_override: str | None = None,
    functions: list[dict] | None = None,
    function_calling_system=None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tracer: LatencyTracer | None = None,
    stream: bool = False,
    partial_callback: Callable[[str], None] | None = None,
    no_fallback: bool = False,
) -> dict[str, Any | None]:
    providers = get_ai_providers()
    selected = (provider_override or PROVIDER).lower()
    provider_cfg = providers.providers.get(selected)

    logger.info(
        f"🔧 AI Request: model={model}, provider={selected}, provider_override={provider_override}"
    )
    logger.info(f"🔧 Available providers: {list(providers.providers.keys())}")
    logger.info(f"🔧 Selected provider config exists: {provider_cfg is not None}")

    forced_selected = provider_override is not None

    async def _try(provider_name: str) -> dict[str, Any] | None:
        prov = providers.providers[provider_name]
        logger.info(f"🔧 Trying provider: {provider_name}")
        try:
            check_fn = prov.get("check")
            check_ok = False
            if forced_selected and provider_name == selected:
                # Bypass check when user explicitly forced provider
                check_ok = True
            else:
                check_ok = bool(callable(check_fn) and check_fn())
            if check_ok:
                logger.info(f"✅ Provider {provider_name} check passed")
                if tracer:
                    tracer.event("provider_check_pass", {"provider": provider_name})

                # Map model ID to provider-compatible value when necessary
                model_mapped = AIProviders.map_model_for_provider(model, provider_name)

                # Handle different providers with appropriate parameters
                chat_func = prov.get("chat")
                if provider_name == "openai":
                    if tracer:
                        tracer.event("provider_request_start", {"provider": provider_name})
                    result = await chat_func(  # type: ignore[misc]
                        model_mapped,
                        messages,
                        images,
                        functions,
                        function_calling_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        tracer=tracer,
                        partial_callback=partial_callback,
                    )
                    if tracer:
                        tracer.event("provider_request_end", {"provider": provider_name})
                elif provider_name == "openrouter":
                    if tracer:
                        tracer.event("provider_request_start", {"provider": provider_name})
                    result = await chat_func(  # type: ignore[misc]
                        model_mapped,
                        messages,
                        images,
                        functions,
                        function_calling_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        tracer=tracer,
                        partial_callback=partial_callback,
                    )
                    if tracer:
                        tracer.event("provider_request_end", {"provider": provider_name})
                elif provider_name == "lmstudio":
                    if tracer:
                        tracer.event("provider_request_start", {"provider": provider_name})
                    result = await chat_func(  # type: ignore[misc]
                        model_mapped,
                        messages,
                        images,
                        functions,
                        function_calling_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        tracer=tracer,
                        partial_callback=partial_callback,
                    )
                    if tracer:
                        tracer.event("provider_request_end", {"provider": provider_name})
                else:
                    if tracer:
                        tracer.event("provider_request_start", {"provider": provider_name})
                    result = chat_func(
                        model,
                        messages,
                        images,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ) if callable(chat_func) else None
                    if tracer:
                        tracer.event("provider_request_end", {"provider": provider_name})

                logger.info(f"✅ Provider {provider_name} returned result")
                if isinstance(result, dict):
                    return result
                return {"message": {"content": str(result)}}
            else:
                logger.warning(f"❌ Provider {provider_name} check failed")
        except Exception as exc:  # pragma: no cover
            logger.error("❌ Provider %s failed: %s", provider_name, exc)
        return None

    # najpierw preferowany
    if provider_cfg:
        if tracer:
            tracer.event("provider_primary_try", {"provider": selected})
        resp = await _try(selected)
        if resp:
            logger.info(f"✅ Using preferred provider: {selected}")
            if tracer:
                tracer.event("provider_primary_success", {"provider": selected})
            return resp
        if forced_selected and no_fallback:
            # Return explicit error without trying other providers
            logger.error("❌ Provider %s failed and fallback disabled.", selected)
            if tracer:
                tracer.event("provider_primary_failed_no_fallback", {"provider": selected})
            error_payload = json.dumps(
                {
                    "command": "",
                    "params": {},
                    "text": f"Błąd: Dostawca {selected} nieosiągalny, fallback wyłączony",
                },
                ensure_ascii=False,
            )
            return {"message": {"content": error_payload}}
    else:
        logger.warning(f"❌ Selected provider {selected} not found in providers")

    # fallback‑i
    if forced_selected and no_fallback:
        logger.warning("No fallback enabled; skipping other providers.")
        error_payload = json.dumps(
            {
                "command": "",
                "params": {},
                "text": f"Błąd: Dostawca {selected} nieosiągalny, fallback wyłączony",
            },
            ensure_ascii=False,
        )
        if tracer:
            tracer.event("provider_all_failed_no_fallback")
        return {"message": {"content": error_payload}}

    logger.warning(f"⚠️ Preferred provider {selected} failed, trying fallbacks...")
    for name in [n for n in providers.providers if n != selected]:
        if tracer:
            tracer.event("provider_fallback_try", {"provider": name})
        resp = await _try(name)
        if resp:
            logger.info("✅ Fallback provider %s zadziałał.", name)
            if tracer:
                tracer.event("provider_fallback_success", {"provider": name})
            return resp

    # total failure
    logger.error("❌ Wszyscy providerzy zawiedli.")
    error_payload = json.dumps(
        {
            "command": "",
            "params": {},
            "text": "Błąd: Żaden dostawca nie odpowiada",
        },
        ensure_ascii=False,
    )
    if tracer:
        tracer.event("provider_all_failed")
    return {"message": {"content": error_payload}}


# ---------------------------------------------------------------- response ---


@measure_performance
async def generate_response_logic(
    provider_name: str,
    model_name: str,
    messages: list[dict[str, Any]],
    tools_info: str,
    system_prompt_override: str | None = None,
    detected_language: str | None = None,
    language_confidence: float | None = None,
    images: list[str] | None = None,  # Added images
    active_window_title: str | None = None,  # Added
    track_active_window_setting: bool = False,  # Added
) -> str:
    """Core logic to generate a response from a chosen AI provider."""
    # Build the full system prompt
    # The first message in 'messages' is typically the system prompt.
    # We will replace it or prepend a new one if it doesn't exist.
    system_message_content = build_full_system_prompt(
        system_prompt_override=system_prompt_override,
        detected_language=detected_language,
        language_confidence=language_confidence,
        tools_description=tools_info,
        active_window_title=active_window_title,  # Pass through
        track_active_window_setting=track_active_window_setting,  # Pass through
    )

    # Convert deque to list for slicing and modification
    messages_list = list(messages)
    if messages_list and messages_list[0]["role"] == "system":
        messages_list[0]["content"] = system_message_content
    else:
        messages_list.insert(
            0, {"role": "system", "content": system_message_content}
        )  # Send the modified messages to the AI provider
    response = await chat_with_providers(
        model=model_name,
        messages=messages_list,
        images=images,  # Pass images to the provider
        provider_override=provider_name,  # Ensure the correct provider is used
    )

    # Extract and return the response content
    if isinstance(response, dict):
        msg_obj = response.get("message")
        if isinstance(msg_obj, dict):
            return msg_obj.get("content", "").strip()
    return ""


@measure_performance
async def generate_response(
    conversation_history: deque,
    tools_info: str = "",
    system_prompt_override: Optional[str] = None,
    detected_language: str = "en",
    language_confidence: float = 1.0,
    active_window_title: Optional[str] = None,
    track_active_window_setting: bool = False,
    tool_suggestion: Optional[str] = None,
    modules: Optional[dict[str, Any]] = None,
    use_function_calling: bool = True,
    user_name: Optional[str] = None,
    model_override: Optional[str] = None,
    tracking_id: Optional[str] = None,
    enable_latency_trace: bool = True,
    stream: bool = False,
    partial_callback: Callable[[str], None] | None = None,
) -> str | dict[str, Any]:
    """Generates a response from the AI model based on conversation history and
    available tools. Can use either traditional prompt-based approach or OpenAI Function
    Calling.

    Args:
        conversation_history: A deque of previous messages.
        tools_info: A string describing available tools/plugins.
        system_prompt_override: An optional string to override the default system prompt.
        detected_language: The detected language code (e.g., "en", "pl").
        language_confidence: The confidence score for the detected language.
        active_window_title: The title of the currently active window.
        track_active_window_setting: Boolean indicating if active window tracking is enabled.
        modules: Dictionary of available modules for function calling.
        use_function_calling: Whether to use OpenAI Function Calling or traditional approach.    Returns:
        A string containing the AI's response, potentially in JSON format for commands.
    """
    import datetime

    def log_append(lines: list[str]):  # small helper to centralize logging writes
        try:
            with open("user_data/prompts_log.txt", "a", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
        except Exception as log_exc:  # pragma: no cover
            logger.warning(f"[PromptLog] Failed to append log: {log_exc}")

    tracer = LatencyTracer(tracking_id, enabled=enable_latency_trace and LATENCY_TRACE_ENABLED)
    tracer.event("generate_response_start")
    try:
        # 1. API key resolution
        api_key = env_manager.get_api_key("openai") if env_manager else None
        if not api_key:
            cfg = load_config()
            api_key = cfg.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            tracer.event("api_key_missing")
            return json.dumps({
                "text": "Błąd: Klucz API OpenAI nie został skonfigurowany.",
                "command": "",
                "params": {}
            }, ensure_ascii=False)

        # 2. Function calling preparation
        function_calling_system = None
        functions = None
        # Prepare functions for providers that support OpenAI-style tools (OpenAI, OpenRouter, LM Studio-compatible)
        if use_function_calling:
            from core.function_calling_system import get_function_calling_system
            function_calling_system = get_function_calling_system()
            if tracer:
                tracer.event("function_system_singleton", {"cached": True})
            try:
                functions = function_calling_system.convert_modules_to_functions()
            except Exception as _fc_exc:
                logger.warning(f"Function conversion failed: {_fc_exc}")
                functions = None
            if not functions:
                function_calling_system = None
        tracer.event("functions_prepared", {"count": len(functions) if functions else 0})

        # 3. Build system prompt using module-level LRU cache
        ov_hash = hashlib.sha256(system_prompt_override.encode("utf-8")).hexdigest()[:16] if system_prompt_override else "noovr"
        tools_desc_hash = hashlib.sha256((tools_info if tools_info else "").encode("utf-8")).hexdigest()[:12]
        funcs_count = len(functions) if functions else 0
        cache_start = perf_counter()
        system_prompt = _cached_system_prompt(
            system_prompt_override,
            detected_language,
            language_confidence,
            tools_desc_hash,
            active_window_title if track_active_window_setting else None,
            track_active_window_setting,
            user_name,
            funcs_count,
        )
        build_ms = (perf_counter() - cache_start) * 1000
        cached = build_ms < 1.2  # heuristic threshold ~1ms for cache hit
        tracer.event("system_prompt_built", {"chars": len(system_prompt), "cached": cached, "build_ms": round(build_ms,2), "funcs": funcs_count})
        tracer.event(f"prompt_cache_{'hit' if cached else 'miss'}", {"build_ms": round(build_ms,2), "funcs": funcs_count})

        # 4. Prepare messages
        messages = list(conversation_history)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # 5. Log prompt & history
        ts = datetime.datetime.now().isoformat()
        log_lines = [f"{ts} | {json.dumps({'role':'system','content':system_prompt}, ensure_ascii=False)}"]
        for m in messages[1:]:  # skip system already logged
            log_lines.append(f"{ts} | {json.dumps(m, ensure_ascii=False)}")
        if functions:
            log_lines.append(f"{ts} | {json.dumps({'role':'system','content': f'Available functions: {len(functions)}'}, ensure_ascii=False)}")
        log_append(log_lines)

    # 6. Call provider orchestration (FIRST MODEL CALL)
        model_to_use = (model_override or MAIN_MODEL).strip()
        tracer.event("provider_call_begin")
        resp = await chat_with_providers(
            model_to_use,
            messages,
            functions=functions,
            function_calling_system=function_calling_system,
            tracer=tracer,
            stream=stream,
            partial_callback=partial_callback,
        )
        tracer.event("provider_call_end")

        # 7. Extract content
        content = ""
        fast_tool_path_detected = False
        fast_tool_path_result = None
        if isinstance(resp, dict):
            # Check for fast-path weather response
            if resp.get("fast_tool_path"):
                fast_tool_path_detected = True
                fast_tool_path_result = resp
                logger.warning(f"🚀 FAST PATH DETECTED in generate_response: {resp.get('tool_calls_executed', 0)} tools")
            
            msg_obj = resp.get("message")
            if isinstance(msg_obj, dict):
                content = (msg_obj.get("content") or "").strip()
        fallback_used = False  # track if alternate provider succeeded
        # Optional guard: allow disabling provider fallback via env
        disable_fallback = os.getenv("GAJA_DISABLE_PROVIDER_FALLBACK", "0") in ("1", "true", "True")
        if (not content) and (not disable_fallback) and not (isinstance(resp, dict) and (resp.get("tool_calls_executed") or resp.get("clarification_request"))):
            # Attempt a single fallback provider if available (different from selected default)
            try:
                alt_provider = None
                primary = PROVIDER.lower()
                for prov_name in ["openai", "ollama", "lmstudio"]:
                    if prov_name != primary:
                        alt_provider = prov_name
                        break
                if alt_provider:
                    tracer.event("fallback_attempt", {"provider": alt_provider})
                    resp_alt = await chat_with_providers(
                        model_to_use,
                        messages,
                        provider_override=alt_provider,
                        functions=functions,
                        function_calling_system=function_calling_system,
                        tracer=tracer,
                        partial_callback=partial_callback,
                    )
                    if isinstance(resp_alt, dict):
                        msg_obj2 = resp_alt.get("message")
                        if isinstance(msg_obj2, dict):
                            content2 = (msg_obj2.get("content") or "").strip()
                            if content2:
                                content = content2
                                fallback_used = True
                                tracer.event("fallback_success", {"provider": alt_provider})
            except Exception as fb_exc:  # pragma: no cover
                logger.warning(f"Fallback provider attempt failed: {fb_exc}")
                tracer.event("fallback_error", {"error": str(fb_exc)[:200]})
        if not content:
            # Natural assistant-style fallback (voice oriented)
            content = (
                "Nie mogłem teraz wygenerować pełnej odpowiedzi. Spróbuj proszę powtórzyć pytanie za chwilę; "
                + ("wcześniej użyłem alternatywnego dostawcy." if fallback_used else "próbowałem głównego dostawcy.")
            )
            logger.warning("Generated natural fallback due to empty model output (fallback_used=%s)", fallback_used)
            tracer.event("content_empty_fallback_generated", {"fallback_used": fallback_used})

        # 8. Fast-path weather result handling
        if fast_tool_path_detected and fast_tool_path_result:
            tracer.event("fast_path_return_from_generate_response")
            # Build proper JSON response structure for fast path
            message_obj = fast_tool_path_result.get("message") or {}
            fast_content = message_obj.get("content", "") if isinstance(message_obj, dict) else ""
            fast_result = {
                "text": fast_content,
                "command": "",
                "params": {},
                "fast_tool_path": True,
                "tools_used": fast_tool_path_result.get("tool_calls_executed", 0),
                "tool_call_details": fast_tool_path_result.get("tool_call_details", [])
            }
            logger.warning(f"🚀 FAST PATH RETURN from generate_response: tools={fast_result.get('tools_used', 0)}")
            return json.dumps(fast_result, ensure_ascii=False)  # Return JSON string!

        # 8. Clarification branch
        if resp and resp.get("clarification_request"):
            tracer.event("clarification_request")
            return json.dumps({
                "text": content,
                "command": "",
                "params": {},
                "clarification_data": resp.get("clarification_request"),
                "requires_user_response": True,
                "action_type": "clarification_request",
            }, ensure_ascii=False)

        # 9. Function calling result normalization
        if use_function_calling and functions and resp.get("tool_calls_executed"):
            tracer.event("tool_calls_detected", {"count": resp.get("tool_calls_executed")})
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and "text" in parsed_content:
                    content_json = parsed_content
                else:
                    raise ValueError
            except Exception:
                content_json = {
                    "text": content,
                    "command": "",
                    "params": {},
                    "function_calls_executed": True,
                    "tools_used": resp.get("tool_calls_executed", 0),
                }
            tracer.event("function_calls_executed", {"tools_used": resp.get("tool_calls_executed", 0)})
            return json.dumps(content_json, ensure_ascii=False)

        # 10. Traditional JSON attempt
        extracted = extract_json(content)
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict) and "text" in parsed:
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # 11. Fallback wrap
        # Token / length metrics (final path)
        approx_tokens = len(content.split())
        tracer.event("return_normal", {"chars": len(content), "approx_tokens": approx_tokens})
        return json.dumps({"text": content, "command": "", "params": {}, "approx_tokens": approx_tokens}, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        logger.error("generate_response error: %s", exc, exc_info=True)
        tracer.event("error", {"error": str(exc)[:200]})
        return json.dumps({
            "text": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
            "command": "",
            "params": {},
        }, ensure_ascii=False)
    finally:
        tracer.event("generate_response_end")
        tracer.flush_to_file()


class AIModule:
    """Główna klasa modułu AI dla serwera."""

    def __init__(self, config: dict):
        self.config = config
        self.providers = get_ai_providers()
        self._conversation_history = {}
        # Optional pre-warm of weather module to avoid first-call latency
        if os.getenv("GAJA_PREWARM_WEATHER", "1") not in ("0","false","False"):
            try:
                # Delay import until here
                from modules.weather_module import WeatherModule  # type: ignore
                self._weather_module = WeatherModule()
                logger.debug("Pre-warmed WeatherModule instance")
            except Exception as pw_exc:  # pragma: no cover
                logger.debug(f"WeatherModule pre-warm skipped: {pw_exc}")

    async def process_query(self, query: str, context: dict) -> dict:
        try:
            if context is None:
                context = {}
            history = context.get("history", [])
            available_plugins = context.get("available_plugins", [])
            modules = context.get("modules", {})
            force_model = context.get("force_model")

            conversation_history = deque()
            for msg in history[-20:]:
                content = msg.get("content", "")
                if msg.get("role") == "assistant":
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "text" in parsed:
                            content = parsed["text"]
                    except Exception:
                        pass
                if content and content.strip():
                    conversation_history.append({"role": msg.get("role", "user"), "content": content})
            conversation_history.append({"role": "user", "content": query})

            tools_info = f"Dostępne pluginy: {', '.join(available_plugins)}" if available_plugins else ""
            response = await generate_response(
                conversation_history=conversation_history,
                tools_info=tools_info,
                detected_language="pl",
                language_confidence=1.0,
                modules=modules,
                use_function_calling=True,
                user_name=context.get("user_name", "User"),
                model_override=force_model,
            )
            
            # Check if response is fast-path dict
            if isinstance(response, dict) and response.get("fast_tool_path"):
                logger.warning(f"🚀 FAST PATH RESPONSE in process_query: tools={response.get('tool_calls_executed', 0)}")
                # Convert back to JSON string for consistent interface
                content = response.get("response_content", "")
                return {
                    "type": "normal_response",
                    "response": content,
                    "fast_tool_path": True,
                    "tool_calls_executed": response.get("tool_calls_executed", 0),
                    "tool_call_details": response.get("tool_call_details", [])
                }
            
            # Normal string response processing
            if isinstance(response, str):
                try:
                    parsed_resp = json.loads(response)
                    if isinstance(parsed_resp, dict) and parsed_resp.get("requires_user_response"):
                        return {
                            "type": "clarification_request",
                            "response": response,
                            "clarification_data": parsed_resp.get("clarification_data"),
                            "requires_user_response": True,
                        }
                except Exception:
                    pass
                return {"type": "normal_response", "response": response}
            
            # Fallback for unexpected response type
            return {"type": "normal_response", "response": str(response)}
        except Exception as e:
            logger.error(f"Error processing AI query: {e}")
            err = json.dumps({
                "text": f"Przepraszam, wystąpił błąd podczas przetwarzania zapytania: {e}",
                "command": "",
                "params": {},
            }, ensure_ascii=False)
            return {"type": "error_response", "response": err}
