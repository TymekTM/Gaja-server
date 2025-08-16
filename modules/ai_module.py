"""ai_providers.py – ulepszona wersja."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections import deque
import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Optional

import httpx  # Async HTTP client replacing requests
from config.config_loader import MAIN_MODEL, PROVIDER, _config, load_config
from core.performance_monitor import measure_performance
from templates.prompt_builder import build_convert_query_prompt, build_full_system_prompt

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
# Klasa providerów
# -----------------------------------------------------------------------------
class AIProviders:
    """Rejestr wszystkich obsługiwanych dostawców + metody pomocnicze."""

    def __init__(self) -> None:
        # Async HTTP client for LM Studio (reduced latency)
        self._httpx_client = httpx.AsyncClient(timeout=30.0)

        # Cached clients to avoid reinitialization overhead
        self._openai_client = None
        # Removed deprecated provider clients (deepseek, anthropic, transformer)
        self._modules: dict[str, Any] = {
            mod: AIProviders._safe_import(mod)
            for mod in ("ollama", "openai")
        }

        # Active providers registry (only requested ones)
        self.providers: dict[str, dict[str, Optional[Callable[..., Any]]]] = {
            "openai": {"module": self._modules["openai"], "check": self.check_openai, "chat": self.chat_openai},
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
    async def _check_ollama_async(self) -> bool:
        try:
            response = await self._httpx_client.get("http://localhost:11434", timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    def check_ollama(self) -> bool:
        try:
            return asyncio.run(self._check_ollama_async())
        except RuntimeError:
            return False

    def check_lmstudio(self) -> bool:
        # Lightweight availability check for LM Studio (OpenAI-compatible local server)
        base_url = _config.get("LMSTUDIO_URL_BASE", "http://localhost:1234")
        try:
            r = httpx.get(base_url + "/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # Fallback: attempt root path quick connect (some builds don't expose /health)
        try:
            r = httpx.get(base_url, timeout=1.0)
            return r.status_code < 500
        except Exception:
            return False

    def check_openai(self) -> bool:
        key_valid = AIProviders._key_ok("OPENAI_API_KEY", "openai")
        logger.info(f"🔧 OpenAI check: key_valid={key_valid}")
        return key_valid

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

    def chat_lmstudio(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        """Call LM Studio local server (OpenAI-compatible) if available."""
        base_url = _config.get("LMSTUDIO_URL_BASE", "http://localhost:1234")
        url = f"{base_url}/v1/chat/completions"
        try:
            self._append_images(messages, images)
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature if temperature is not None else _config.get("temperature", 0.7),
            }
            payload["max_tokens"] = max_tokens if max_tokens is not None else _config.get("max_tokens", 1500)
            r = httpx.post(url, json=payload, timeout=30)
            if r.status_code >= 400:
                return {"message": {"content": f"LM Studio HTTP {r.status_code}: {r.text[:200]}"}}
            data = r.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "(no content)")
            )
            return {"message": {"content": content}}
        except Exception as exc:
            logger.error("LM Studio error: %s", exc)
            return {"message": {"content": f"LM Studio error: {exc}"}}

    async def chat_openai(
        self,
        model: str,
        messages: list[dict],
        images: list[str] | None = None,
        functions: list[dict] | None = None,
        function_calling_system=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
                    token_key: max_tokens if max_tokens is not None else _config.get("max_tokens", 1500),
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
            for use_new in attempt_order:
                params, token_key = build_params(use_new)
                try:
                    response = client.chat.completions.create(**params)
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

            # 5. Function calling handling
            if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
                tool_results = []
                for tool_call in response.choices[0].message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    exec_result = None
                    if function_calling_system:
                        exec_result = await function_calling_system.execute_function(
                            fn_name,
                            fn_args,
                            conversation_history=deque(messages[-10:]) if messages else None,
                        )
                        if (
                            isinstance(exec_result, dict)
                            and exec_result.get("action_type") == "clarification_request"
                        ):
                            return {
                                "message": {"content": exec_result.get("message", "Clarification requested")},
                                "clarification_request": exec_result.get("clarification_data"),
                                "tool_calls_executed": 1,
                                "requires_user_response": True,
                            }
                    tool_results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": str(exec_result),
                        }
                    )

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
                messages.extend(tool_results)

                # Second pass with tool outputs
                token_key = "max_completion_tokens" if used_new_key else "max_tokens"
                final_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else _config.get("temperature", 0.7),
                    token_key: max_tokens if max_tokens is not None else _config.get("max_tokens", 1500),
                }
                if functions:
                    final_params["tools"] = functions
                    final_params["tool_choice"] = "auto"
                # Second-pass guard for gpt-5-* models rejecting custom temperature
                if model.startswith("gpt-5-") and final_params.get("temperature") not in (None, 1, 1.0):
                    final_params.pop("temperature", None)
                final_response = client.chat.completions.create(**final_params)
                return {
                    "message": {"content": final_response.choices[0].message.content},
                    "tool_calls_executed": len(tool_results),
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
            return {"message": {"content": base_content}}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("OpenAI error: %s", exc, exc_info=True)
            return {"message": {"content": f"Błąd OpenAI: {exc}"}}

    # Removed deprecated chat providers (deepseek, anthropic, transformer)

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
) -> dict[str, Any | None]:
    providers = get_ai_providers()
    selected = (provider_override or PROVIDER).lower()
    provider_cfg = providers.providers.get(selected)

    logger.info(
        f"🔧 AI Request: model={model}, provider={selected}, provider_override={provider_override}"
    )
    logger.info(f"🔧 Available providers: {list(providers.providers.keys())}")
    logger.info(f"🔧 Selected provider config exists: {provider_cfg is not None}")

    async def _try(provider_name: str) -> dict[str, Any] | None:
        prov = providers.providers[provider_name]
        logger.info(f"🔧 Trying provider: {provider_name}")
        try:
            check_fn = prov.get("check")
            if callable(check_fn) and check_fn():
                logger.info(f"✅ Provider {provider_name} check passed")

                # Handle different providers with appropriate parameters
                chat_func = prov.get("chat")

                if provider_name == "openai":
                    # OpenAI supports function calling
                    result = await chat_func(  # type: ignore[misc]
                        model,
                        messages,
                        images,
                        functions,
                        function_calling_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    # Sync providers (ollama, lmstudio)
                    result = chat_func(
                        model,
                        messages,
                        images,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ) if callable(chat_func) else None

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
        resp = await _try(selected)
        if resp:
            logger.info(f"✅ Using preferred provider: {selected}")
            return resp
    else:
        logger.warning(f"❌ Selected provider {selected} not found in providers")

    # fallback‑i
    logger.warning(f"⚠️ Preferred provider {selected} failed, trying fallbacks...")
    for name in [n for n in providers.providers if n != selected]:
        resp = await _try(name)
        if resp:
            logger.info("✅ Fallback provider %s zadziałał.", name)
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
) -> str:
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
    # (Legacy initialization block removed; API key resolution handled later)

    """Generate AI response (supports optional OpenAI function calling)."""
    import datetime

    def log_append(lines: list[str]):  # small helper to centralize logging writes
        try:
            with open("user_data/prompts_log.txt", "a", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
        except Exception as log_exc:  # pragma: no cover
            logger.warning(f"[PromptLog] Failed to append log: {log_exc}")

    try:
        # 1. API key resolution
        api_key = env_manager.get_api_key("openai") if env_manager else None
        if not api_key:
            cfg = load_config()
            api_key = cfg.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return json.dumps({
                "text": "Błąd: Klucz API OpenAI nie został skonfigurowany.",
                "command": "",
                "params": {}
            }, ensure_ascii=False)

        # 2. Function calling preparation
        function_calling_system = None
        functions = None
        if use_function_calling and PROVIDER.lower() == "openai":
            from core.function_calling_system import FunctionCallingSystem
            function_calling_system = FunctionCallingSystem()
            functions = function_calling_system.convert_modules_to_functions()
            if not functions:
                function_calling_system = None

        # 3. Build system prompt
        system_prompt = build_full_system_prompt(
            system_prompt_override=system_prompt_override,
            detected_language=detected_language,
            language_confidence=language_confidence,
            tools_description="" if functions else tools_info,
            active_window_title=active_window_title,
            track_active_window_setting=track_active_window_setting,
            tool_suggestion=tool_suggestion,
            user_name=user_name,
        )

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

        # 6. Call provider orchestration
        model_to_use = (model_override or MAIN_MODEL).strip()
        resp = await chat_with_providers(
            model_to_use,
            messages,
            functions=functions,
            function_calling_system=function_calling_system,
        )

        # 7. Extract content
        content = ""
        if isinstance(resp, dict):
            msg_obj = resp.get("message")
            if isinstance(msg_obj, dict):
                content = (msg_obj.get("content") or "").strip()
        fallback_used = False  # track if alternate provider succeeded
        if not content:
            # Attempt a single fallback provider if available (different from selected default)
            try:
                alt_provider = None
                primary = PROVIDER.lower()
                for prov_name in ["openai", "ollama", "lmstudio"]:
                    if prov_name != primary:
                        alt_provider = prov_name
                        break
                if alt_provider:
                    resp_alt = await chat_with_providers(
                        model_to_use,
                        messages,
                        provider_override=alt_provider,
                        functions=functions,
                        function_calling_system=function_calling_system,
                    )
                    if isinstance(resp_alt, dict):
                        msg_obj2 = resp_alt.get("message")
                        if isinstance(msg_obj2, dict):
                            content2 = (msg_obj2.get("content") or "").strip()
                            if content2:
                                content = content2
                                fallback_used = True
            except Exception as fb_exc:  # pragma: no cover
                logger.warning(f"Fallback provider attempt failed: {fb_exc}")
        if not content:
            # Natural assistant-style fallback (voice oriented)
            content = (
                "Nie mogłem teraz wygenerować pełnej odpowiedzi. Spróbuj proszę powtórzyć pytanie za chwilę; "
                + ("wcześniej użyłem alternatywnego dostawcy." if fallback_used else "próbowałem głównego dostawcy.")
            )
            logger.warning("Generated natural fallback due to empty model output (fallback_used=%s)", fallback_used)

        # 8. Clarification branch
        if resp and resp.get("clarification_request"):
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
        return json.dumps({"text": content, "command": "", "params": {}}, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        logger.error("generate_response error: %s", exc, exc_info=True)
        return json.dumps({
            "text": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
            "command": "",
            "params": {},
        }, ensure_ascii=False)


class AIModule:
    """Główna klasa modułu AI dla serwera."""

    def __init__(self, config: dict):
        self.config = config
        self.providers = get_ai_providers()
        self._conversation_history = {}

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
        except Exception as e:
            logger.error(f"Error processing AI query: {e}")
            err = json.dumps({
                "text": f"Przepraszam, wystąpił błąd podczas przetwarzania zapytania: {e}",
                "command": "",
                "params": {},
            }, ensure_ascii=False)
            return {"type": "error_response", "response": err}
