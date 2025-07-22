"""ai_providers.py – ulepszona wersja."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import httpx  # Async HTTP client replacing requests
from config_loader import MAIN_MODEL, PROVIDER, _config, load_config
from performance_monitor import measure_performance
from prompt_builder import build_convert_query_prompt, build_full_system_prompt

# -----------------------------------------------------------------------------
# Konfiguracja logów
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Lazy import for transformers to speed up startup
pipeline = None

# Import environment manager for secure API key handling
try:
    from config_manager import EnvironmentManager

    env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_manager = EnvironmentManager(env_file=env_file_path)
except ImportError as e:
    env_manager = None
    logger.warning(f"Could not import EnvironmentManager: {e}")


def _load_pipeline():
    global pipeline
    if pipeline is None:
        try:
            from transformers import pipeline as _pipeline

            pipeline = _pipeline
        except ImportError:
            pipeline = None
            logger.warning(
                "⚠️  transformers nie jest dostępny - "
                "będzie automatycznie doinstalowany przy pierwszym użyciu"
            )
    return pipeline


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
        self._deepseek_client = (
            None  # Dynamiczny import modułów – brakujące biblioteki ≠ twardy crash
        )
        self._modules: dict[str, Any] = {
            mod: AIProviders._safe_import(mod)
            for mod in ("ollama", "openai", "anthropic")
        }

        self.providers: dict[str, dict[str, Callable[..., Any] | None]] = {
            "ollama": {
                "module": self._modules["ollama"],
                "check": self.check_ollama,
                "chat": self.chat_ollama,
            },
            "lmstudio": {
                "module": None,  # REST‑only – klucz zostawiamy dla spójności
                "check": self.check_lmstudio,
                "chat": self.chat_lmstudio,
            },
            "openai": {
                "module": self._modules["openai"],
                "check": self.check_openai,
                "chat": self.chat_openai,
            },
            "deepseek": {
                "module": None,
                "check": self.check_deepseek,
                "chat": self.chat_deepseek,
            },
            "anthropic": {
                "module": self._modules["anthropic"],
                "check": self.check_anthropic,
                "chat": self.chat_anthropic,
            },
            "transformer": {
                "module": None,
                "check": lambda: True,
                "chat": self.chat_transformer,
            },
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
    async def check_ollama(self) -> bool:
        try:
            response = await self._httpx_client.get(
                "http://localhost:11434", timeout=5.0
            )
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def check_lmstudio(self) -> bool:
        # Force disable LMStudio to use OpenAI instead
        return False
        # try:
        #     url = _config.get("LMSTUDIO_URL", "http://localhost:1234/v1/models")
        #     response = await self._httpx_client.get(url, timeout=5.0)
        #     return response.status_code == 200
        # except httpx.RequestError:
        #     return False

    def check_openai(self) -> bool:
        key_valid = AIProviders._key_ok("OPENAI_API_KEY", "openai")
        logger.info(f"🔧 OpenAI check: key_valid={key_valid}")
        return key_valid

    def check_deepseek(self) -> bool:
        return AIProviders._key_ok("DEEPSEEK_API_KEY", "deepseek")

    def check_anthropic(self) -> bool:
        return AIProviders._key_ok("ANTHROPIC_API_KEY", "anthropic")

    # ---------------------------------------------------------------------
    # Chat‑y
    # ---------------------------------------------------------------------
    def chat_ollama(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            self._append_images(messages, images)
            ollama_mod = self.providers["ollama"]["module"]
            resp = ollama_mod.chat(model=model, messages=messages)
            return {"message": {"content": resp["message"]["content"]}}
        except Exception as exc:  # pragma: no cover
            logger.error("Ollama error: %s", exc)
            return None

    def chat_lmstudio(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": (
                    temperature
                    if temperature is not None
                    else _config.get("temperature", 0.7)
                ),
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            self._append_images(messages, images)
            url = _config.get(
                "LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions"
            )
            r = self._lmstudio_session.post(url, json=payload, timeout=30)
            data = r.json()
            return {"message": {"content": data["choices"][0]["message"]["content"]}}
        except Exception as exc:
            logger.error("LM Studio error: %s", exc)
            return {"message": {"content": f"LM Studio error: {exc}"}}

    async def chat_openai(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        functions: list[dict | None] = None,
        function_calling_system=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            # Use environment manager for secure API key handling
            api_key = None
            if env_manager:
                api_key = env_manager.get_api_key("openai")

            # Fallback to config file or environment variable
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY") or _config.get(
                    "api_keys", {}
                ).get("openai")

            if not api_key:
                raise ValueError("Brak OPENAI_API_KEY.")

            if self._openai_client is None:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=api_key)

            client = self._openai_client
            self._append_images(messages, images)

            # Prepare parameters for OpenAI API call
            params = {
                "model": model,
                "messages": messages,
                "temperature": (
                    temperature
                    if temperature is not None
                    else _config.get("temperature", 0.7)
                ),
                "max_tokens": (
                    max_tokens
                    if max_tokens is not None
                    else _config.get("max_tokens", 1500)
                ),
            }

            # Add tools (functions) if provided
            if functions:
                params["tools"] = functions
                params["tool_choice"] = "auto"

            resp = client.chat.completions.create(**params)
            # Handle function calls
            if resp.choices[0].message.tool_calls:
                # Execute function calls and collect results
                tool_results = []
                for tool_call in resp.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse function arguments for {function_name}: {e}"
                        )
                        function_args = {}

                    if function_calling_system:
                        result = await function_calling_system.execute_function(
                            function_name,
                            function_args,
                            conversation_history=(
                                deque(messages[-10:]) if messages else None
                            ),  # Pass recent conversation
                        )

                        # Handle special clarification requests
                        if (
                            isinstance(result, dict)
                            and result.get("action_type") == "clarification_request"
                        ):
                            # This is a clarification request - return it directly to be handled by WebSocket
                            return {
                                "message": {
                                    "content": result.get(
                                        "message", "Clarification requested"
                                    )
                                },
                                "clarification_request": result.get(
                                    "clarification_data"
                                ),
                                "tool_calls_executed": 1,
                                "requires_user_response": True,
                            }

                        tool_results.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(result),
                            }
                        )

                # Add tool calls to conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in resp.choices[0].message.tool_calls
                        ],
                    }
                )

                # Add tool results
                messages.extend(tool_results)

                # Get final response with tool results
                final_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": (
                        temperature
                        if temperature is not None
                        else _config.get("temperature", 0.7)
                    ),
                    "max_tokens": (
                        max_tokens
                        if max_tokens is not None
                        else _config.get("max_tokens", 1500)
                    ),
                }

                final_resp = client.chat.completions.create(**final_params)
                return {
                    "message": {"content": final_resp.choices[0].message.content},
                    "tool_calls_executed": len(tool_results),
                }
            else:
                return {"message": {"content": resp.choices[0].message.content}}
        except Exception as exc:
            logger.error("OpenAI error: %s", exc, exc_info=True)
            return {"message": {"content": f"Błąd OpenAI: {exc}"}}

    def chat_deepseek(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            api_key = os.getenv("DEEPSEEK_API_KEY") or _config["api_keys"]["deepseek"]
            if not api_key:
                raise ValueError("Brak DEEPSEEK_API_KEY.")
            # DeepSeek jest w 100 % OpenAI‑compatible
            if self._deepseek_client is None:
                from openai import OpenAI  # type: ignore

                self._deepseek_client = OpenAI(
                    api_key=api_key, base_url="https://api.deepseek.com"
                )

            client = self._deepseek_client
            self._append_images(messages, images)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(
                    temperature
                    if temperature is not None
                    else _config.get("temperature", 0.7)
                ),
                max_tokens=(
                    max_tokens
                    if max_tokens is not None
                    else _config.get("max_tokens", 1500)
                ),
            )
            return {"message": {"content": resp.choices[0].message.content}}
        except Exception as exc:
            logger.error("DeepSeek error: %s", exc)
            return None

    def chat_anthropic(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY") or _config["api_keys"]["anthropic"]
            if not api_key:
                raise ValueError("Brak ANTHROPIC_API_KEY.")
            from anthropic import Anthropic  # type: ignore

            client = Anthropic(api_key=api_key)
            self._append_images(messages, images)
            resp = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens if max_tokens is not None else 1000,
                temperature=(
                    temperature
                    if temperature is not None
                    else _config.get("temperature", 0.7)
                ),
            )
            return {"message": {"content": resp.content[0].text}}
        except Exception as exc:
            logger.error("Anthropic error: %s", exc)
            return None

    def chat_transformer(
        self,
        model: str,
        messages: list[dict],
        images: list[str | None] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any | None]:
        try:
            self._append_images(messages, images)
            pl = _load_pipeline()
            if pl is None:
                return None
            generator = pl("text-generation", model=model)
            gen_kwargs = {"max_length": max_tokens or 512, "do_sample": True}
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            resp = generator(messages[-1]["content"], **gen_kwargs)
            return {"message": {"content": resp[0]["generated_text"]}}
        except Exception as exc:  # pragma: no cover
            logger.error("Transformers error: %s", exc)
            return None

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
    return {name: cfg["check"]() for name, cfg in providers.providers.items()}


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
        return (
            resp["message"]["content"].strip()
            if resp and resp.get("message", {}).get("content")
            else query
        )
    except Exception as exc:  # pragma: no cover
        logger.error("refine_query error: %s", exc)
        return query


# ---------------------------------------------------------------- chat glue --


@measure_performance
async def chat_with_providers(
    model: str,
    messages: list[dict],
    images: list[str | None] = None,
    provider_override: str | None = None,
    functions: list[dict | None] = None,
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

    async def _try(provider_name: str) -> dict[str, Any | None]:
        prov = providers.providers[provider_name]
        logger.info(f"🔧 Trying provider: {provider_name}")
        try:
            if prov["check"]():
                logger.info(f"✅ Provider {provider_name} check passed")

                # Handle different providers with appropriate parameters
                chat_func = prov["chat"]

                if provider_name == "openai":
                    # OpenAI supports function calling
                    result = await chat_func(
                        model,
                        messages,
                        images,
                        functions,
                        function_calling_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif provider_name in ["deepseek"]:
                    # Other async providers that don't support function calling yet
                    result = await chat_func(
                        model,
                        messages,
                        images,
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
                    )

                logger.info(f"✅ Provider {provider_name} returned result")
                return result
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
    for name in providers.providers:
        if name == selected:
            continue
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
    images: list[str | None] = None,  # Added images
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
    return (
        response["message"]["content"].strip()
        if response and response.get("message", {}).get("content")
        else ""
    )


@measure_performance
async def generate_response(
    conversation_history: deque,
    tools_info: str = "",
    system_prompt_override: str = None,
    detected_language: str = "en",
    language_confidence: float = 1.0,
    active_window_title: str = None,
    track_active_window_setting: bool = False,
    tool_suggestion: str = None,
    modules: dict[str, Any] = None,
    use_function_calling: bool = True,
    user_name: str = None,
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

    try:
        # Use environment manager for secure API key handling
        api_key = None
        if env_manager:
            api_key = env_manager.get_api_key("openai")

        # Fallback to config file or environment variable
        if not api_key:
            config = load_config()  # Use imported load_config
            api_keys = config.get("api_keys", {})  # Get the nested api_keys dictionary
            api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.error("OpenAI API key not found in configuration.")
            return '{"text": "Błąd: Klucz API OpenAI nie został skonfigurowany.", "command": "", "params": {}}'

        # Initialize function calling system if enabled and plugins available
        function_calling_system = None
        functions = None

        if use_function_calling and PROVIDER.lower() == "openai":
            # Get functions directly from plugin_manager
            from server.function_calling_system import FunctionCallingSystem

            # Initialize function calling system
            function_calling_system = FunctionCallingSystem()

            # Get functions from plugin manager
            functions = function_calling_system.convert_modules_to_functions()

            if functions:
                logger.info(f"Function calling enabled with {len(functions)} functions")
                logger.debug(
                    f"Available functions: {[f['function']['name'] for f in functions]}"
                )
            else:
                logger.warning("No functions available for function calling")
                function_calling_system = None

            # Use standard system prompt for function calling
            system_prompt = build_full_system_prompt(
                system_prompt_override=system_prompt_override,
                detected_language=detected_language,
                language_confidence=language_confidence,
                tools_description="",  # Functions are handled separately
                active_window_title=active_window_title,
                track_active_window_setting=track_active_window_setting,
                tool_suggestion=tool_suggestion,
                user_name=user_name,
            )
        else:
            # Traditional prompt-based approach
            system_prompt = build_full_system_prompt(
                system_prompt_override=system_prompt_override,
                detected_language=detected_language,
                language_confidence=language_confidence,
                tools_description=tools_info,
                active_window_title=active_window_title,
                track_active_window_setting=track_active_window_setting,
                tool_suggestion=tool_suggestion,
                user_name=user_name,
            )

        # --- PROMPT LOGGING ---
        try:
            import json

            timestamp = datetime.datetime.now().isoformat()

            with open("user_data/prompts_log.txt", "a", encoding="utf-8") as f:
                # Log the system prompt
                system_prompt_msg = {"role": "system", "content": system_prompt}
                f.write(
                    f"{timestamp} | {json.dumps(system_prompt_msg, ensure_ascii=False)}\n"
                )

                # Log conversation history
                for msg in list(conversation_history):
                    if msg.get("role") != "system":
                        f.write(
                            f"{timestamp} | {json.dumps(msg, ensure_ascii=False)}\n"
                        )

                # Log available functions if using function calling
                if functions:
                    functions_msg = {
                        "role": "system",
                        "content": f"Available functions: {len(functions)}",
                    }
                    f.write(
                        f"{timestamp} | {json.dumps(functions_msg, ensure_ascii=False)}\n"
                    )
        except Exception as log_exc:
            logger.warning(f"[PromptLog] Failed to log prompt: {log_exc}")

        # Convert deque to list for slicing and modification
        messages = list(conversation_history)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(
                0, {"role": "system", "content": system_prompt}
            )  # Make API call with or without functions
        resp = await chat_with_providers(
            MAIN_MODEL,
            messages,
            functions=functions,
            function_calling_system=function_calling_system,
        )

        # --- RAW API RESPONSE LOGGING ---
        try:
            raw_content = (
                resp.get("message", {}).get("content", "").strip() if resp else ""
            )
            import datetime
            import json

            with open("user_data/prompts_log.txt", "a", encoding="utf-8") as f:
                raw_api_msg = {"role": "assistant_api_raw", "content": raw_content}
                f.write(
                    f"{datetime.datetime.now().isoformat()} | {json.dumps(raw_api_msg, ensure_ascii=False)}\n"
                )

                # Log if function calls were executed
                if resp and resp.get("tool_calls_executed"):
                    tool_calls_msg = {
                        "role": "system",
                        "content": f"Tool calls executed: {resp['tool_calls_executed']}",
                    }
                    f.write(
                        f"{datetime.datetime.now().isoformat()} | {json.dumps(tool_calls_msg, ensure_ascii=False)}\n"
                    )
        except Exception as log_exc:
            logger.warning(f"[RawAPI Log] Failed to log raw API response: {log_exc}")

        content = resp["message"]["content"].strip() if resp else ""
        if not content:
            raise ValueError("Empty response.")

        # Check for clarification request first
        if resp and resp.get("clarification_request"):
            # This is a clarification request - return it specially formatted
            clarification_data = resp.get("clarification_request")
            return json.dumps(
                {
                    "text": content,
                    "command": "",
                    "params": {},
                    "clarification_data": clarification_data,
                    "requires_user_response": True,
                    "action_type": "clarification_request",
                },
                ensure_ascii=False,
            )

        # If using function calling, return the content directly as it should be a natural response
        if use_function_calling and functions and resp.get("tool_calls_executed"):
            # Content is already a natural language response from function calling
            # Check if it's already JSON formatted, if not wrap it
            try:
                # Try to parse as JSON to see if it's already formatted
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and "text" in parsed_content:
                    # It's already a proper JSON response, return as is
                    return content
                else:
                    # It's JSON but not in our expected format, wrap it
                    return json.dumps(
                        {
                            "text": content,
                            "command": "",
                            "params": {},
                            "function_calls_executed": True,
                        },
                        ensure_ascii=False,
                    )
            except json.JSONDecodeError:
                # Content is not JSON, wrap it in our expected format
                logger.debug(f"Content is not JSON, wrapping: {content[:100]}...")
                return json.dumps(
                    {
                        "text": content,
                        "command": "",
                        "params": {},
                        "function_calls_executed": True,
                    },
                    ensure_ascii=False,
                )

        # Traditional JSON parsing for non-function calling responses
        parsed = extract_json(content)
        try:
            result = json.loads(parsed)
            if isinstance(result, dict) and "text" in result:
                return json.dumps(result, ensure_ascii=False)
        except Exception:
            pass

        # fallback: zawinąć surowy tekst
        return json.dumps(
            {"text": content, "command": "", "params": {}}, ensure_ascii=False
        )
    except Exception as exc:  # pragma: no cover
        logger.error("generate_response error: %s", exc, exc_info=True)
        return json.dumps(
            {
                "text": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
                "command": "",
                "params": {},
            },
            ensure_ascii=False,
        )


# -----------------------------------------------------------------------------
# Klasa AIModule
# -----------------------------------------------------------------------------


class AIModule:
    """Główna klasa modułu AI dla serwera."""

    def __init__(self, config: dict):
        self.config = config
        self.providers = get_ai_providers()
        self._conversation_history = {}

    async def process_query(self, query: str, context: dict) -> dict:
        """Przetwarza zapytanie użytkownika i zwraca odpowiedź AI."""
        import json

        try:
            print(f"DEBUG: process_query called with context: {context}")
            print(f"DEBUG: context type: {type(context)}")

            if context is None:
                print("DEBUG: context is None, creating empty dict")
                context = {}

            user_id = context.get("user_id", "anonymous")
            history = context.get("history", [])
            available_plugins = context.get("available_plugins", [])
            modules = context.get("modules", {})

            logger.info(f"AI Context - user_id: {user_id}")
            logger.info(f"AI Context - available_plugins: {available_plugins}")
            logger.info(
                f"AI Context - modules: {list(modules.keys()) if modules else 'None'}"
            )
            logger.info(f"AI Context - modules content: {modules}")

            # Convert history to deque format for generate_response
            conversation_history = deque()

            # Add history from database
            logger.info(f"Processing {len(history)} messages from history for context")
            for msg in history[-20:]:  # Last 20 messages for better context
                content = msg["content"]

                # If the content is a JSON string (from assistant), extract the text
                if msg["role"] == "assistant":
                    try:
                        parsed_content = json.loads(content)
                        if (
                            isinstance(parsed_content, dict)
                            and "text" in parsed_content
                        ):
                            content = parsed_content["text"]
                    except (json.JSONDecodeError, KeyError):
                        # If parsing fails, use content as-is
                        pass

                # Only add non-empty messages
                if content and content.strip():
                    conversation_history.append(
                        {"role": msg["role"], "content": content}
                    )

            # Add current query
            conversation_history.append({"role": "user", "content": query})

            logger.info(
                f"Conversation history prepared with {len(conversation_history)} messages"
            )

            # Build tools description
            tools_info = ""
            if available_plugins:
                tools_info = f"Dostępne pluginy: {', '.join(available_plugins)}"  # Use the same generate_response function as in main ai_module.py
            response = await generate_response(
                conversation_history=conversation_history,
                tools_info=tools_info,
                detected_language="pl",
                language_confidence=1.0,
                modules=modules,
                use_function_calling=True,  # Enable function calling
                user_name=context.get("user_name", "User"),
            )

            # Check if response contains clarification request
            try:
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict):
                    # If it has clarification request data, return structured response
                    if parsed_response.get("requires_user_response"):
                        return {
                            "type": "clarification_request",
                            "response": response,
                            "clarification_data": parsed_response.get(
                                "clarification_data"
                            ),
                            "requires_user_response": True,
                        }
            except (json.JSONDecodeError, TypeError):
                # Response is not JSON, continue normally
                pass

            # Normal response
            return {"type": "normal_response", "response": response}

        except Exception as e:
            logger.error(f"Error processing AI query: {e}")
            error_response = json.dumps(
                {
                    "text": f"Przepraszam, wystąpił błąd podczas przetwarzania zapytania: {str(e)}",
                    "command": "",
                    "params": {},
                },
                ensure_ascii=False,
            )
            return {"type": "error_response", "response": error_response}
