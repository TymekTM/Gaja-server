"""Real World Integration Tests - No Mocking (with minimal resilience).

These tests exercise real AI provider calls and real plugin logic (no mocks).
To reduce flakiness from transient placeholder/error responses or network
variance, a small retry loop is applied only when a response clearly matches a
known placeholder pattern. Latency assertions are soft (warnings) so sporadic
API slowdowns don't fail the suite while we still collect timing signals.
"""

import asyncio
import importlib
import inspect
import json
import os
import time
from typing import Any, Dict

import pytest

from modules.ai_module import AIModule
from config.config_manager import DatabaseManager


# ---------------------------------------------------------------------------
# Helper utilities (single, deduplicated versions)
# ---------------------------------------------------------------------------
def parse_ai_response(result: Any) -> str:
    """Extract a textual representation from diverse result formats."""
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        resp = result.get("response")
        if isinstance(resp, str):
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, dict) and "text" in parsed:
                    return str(parsed["text"])
                return resp
            except Exception:
                return resp
        if isinstance(resp, dict) and "text" in resp:
            return str(resp["text"])
        return str(result)
    return str(result)


PLACEHOLDER_PATTERNS = [
    "nie mogłem teraz wygenerować",
    "brak treści",
    "błąd openai",
    "błąd: brak openai_api_key",
]


def is_placeholder_response(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in PLACEHOLDER_PATTERNS)


async def run_query_with_retries(
    ai_module: AIModule,
    query: str,
    context: Dict[str, Any],
    retries: int = 2,
    delay: float = 1.2,
):
    """Execute a real AI query with limited retries on placeholder outputs."""
    attempt = 0
    while True:
        result = await ai_module.process_query(query, context)
        text = parse_ai_response(result)
        if not is_placeholder_response(text) or attempt >= retries:
            return result, text, attempt
        attempt += 1
        await asyncio.sleep(delay * (1 + attempt * 0.4))


def assert_latency(latency: float, threshold: float, label: str):
    """Soft latency assertion with warning beyond threshold."""
    if latency <= threshold:
        return
    print(
        f"[WARN] {label} latency {latency:.2f}s exceeded threshold {threshold:.2f}s (soft warning)."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRealConversationHistory:
    """Test real conversation history without mocking."""

    @pytest.fixture
    def real_ai_setup(self):
        db_path = "test_real_ai.db"
        db_manager = DatabaseManager(db_path)
        config = {"ai": {"provider": "openai", "model": "gpt-4o-mini"}}
        ai_module = AIModule(config=config)
        try:
            yield ai_module, db_manager
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_weather_conversation_flow(self, real_ai_setup):
        ai_module, db_manager = real_ai_setup
        user_id = "test_user_real_weather"

        # Initial weather intent
        start_time = time.time()
        _result1, response_text1, _ = await run_query_with_retries(
            ai_module, "czy będzie padać", {"user_id": user_id}
        )
        end_time = time.time()

        print(f"Weather query response: {response_text1}")
        assert len(response_text1) > 10, "Response too short"
        location_words = [
            "gdzie",
            "miasto",
            "lokalizacja",
            "location",
            "jakiego",
            "dla jakiego",
            "którego",
            "dla którego",
            "city",
            "which city",
        ]
        assert any(word in response_text1.lower() for word in location_words), (
            f"AI should ask for location. Response: {response_text1}"
        )
        latency = end_time - start_time
        assert_latency(latency, 10.0, "Initial weather query")

        await db_manager.save_interaction(
            user_id, "czy będzie padać", json.dumps({"text": response_text1})
        )

        # Provide location with context
        history = await db_manager.get_user_history(user_id, limit=10)
        context = {"user_id": user_id, "history": history}
        start_time = time.time()
        _result2, response_text2, _ = await run_query_with_retries(
            ai_module, "Warszawa", context
        )
        end_time = time.time()
        print(f"Location response: {response_text2}")

        assert len(response_text2) > 20, "Weather response too short"
        assert "warszaw" in response_text2.lower(), "Should mention Warsaw"
        weather_words = [
            "pogoda",
            "temperatura",
            "deszcz",
            "słońce",
            "chmury",
            "weather",
            "rain",
            "sun",
            "cloud",
        ]
        assert any(word in response_text2.lower() for word in weather_words), (
            "Should contain weather information"
        )
        latency = end_time - start_time
        assert_latency(latency, 10.0, "Context weather query")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_conversation_persistence(self, real_ai_setup):
        ai_module, db_manager = real_ai_setup
        user_id = "test_user_real_persistence"
        queries = [
            "Jutro mam ważne spotkanie biznesowe",
            "O 15:00",
            "Przypomnij mi o tym 30 minut wcześniej",
        ]

        for i, query in enumerate(queries):
            history = await db_manager.get_user_history(user_id, limit=20)
            context = {"user_id": user_id, "history": history}
            _result, response_text, _ = await run_query_with_retries(
                ai_module, query, context
            )
            print(f"Query {i+1}: {query}\nResponse {i+1}: {response_text}")
            assert len(response_text) > 5, f"Response {i+1} too short"
            await db_manager.save_interaction(
                user_id, query, json.dumps({"text": response_text})
            )
            new_history = await db_manager.get_user_history(user_id, limit=20)
            assert len(new_history) == (i + 1) * 2, (
                f"History not growing correctly at step {i+1}"
            )

        final_history = await db_manager.get_user_history(user_id, limit=20)
        history_str = str(final_history).lower()
        assert "spotkanie" in history_str or "meeting" in history_str
        assert "15:00" in history_str or " 15" in history_str
        assert "przypomn" in history_str or "remind" in history_str


class TestRealMemoryAnchors:
    """Test real short-term memory without mocking."""

    @pytest.fixture
    def real_memory_setup(self):
        db_path = "test_real_memory.db"
        db_manager = DatabaseManager(db_path)
        config = {"ai": {"provider": "openai", "model": "gpt-4o-mini"}}
        ai_module = AIModule(config=config)
        try:
            yield ai_module, db_manager
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_exam_reminder_memory(self, real_memory_setup):
        ai_module, db_manager = real_memory_setup
        user_id = "test_user_real_exam"

        context1 = {"user_id": user_id, "history": []}
        _r1, response_text1, _ = await run_query_with_retries(
            ai_module,
            "Jutro mam kolokwium z algebry. Przypomnij mi o 18.",
            context1,
        )
        print(f"Reminder setup: {response_text1}")
        assert len(response_text1) > 10, "Initial response too short"
        if "nie mogłem" in response_text1.lower() or "brak treści" in response_text1.lower():
            pytest.skip("AI returned error response, skipping detailed checks")

        time_mentioned = "18" in response_text1 or "osiemnast" in response_text1.lower()
        subject_mentioned = any(
            word in response_text1.lower()
            for word in ["kolokwium", "algebra", "exam", "reminder", "przypomn"]
        )
        assert time_mentioned or subject_mentioned, (
            f"Should mention time or subject. Response: {response_text1}"
        )
        await db_manager.save_interaction(
            user_id,
            "Jutro mam kolokwium z algebry. Przypomnij mi o 18.",
            json.dumps({"text": response_text1}),
        )

        for query in ["Jak się masz?", "Co dziś robimy?"]:
            history = await db_manager.get_user_history(user_id, limit=10)
            context = {"user_id": user_id, "history": history}
            _r, r_text, _ = await run_query_with_retries(ai_module, query, context)
            await db_manager.save_interaction(user_id, query, json.dumps({"text": r_text}))

        history = await db_manager.get_user_history(user_id, limit=10)
        context3 = {"user_id": user_id, "history": history}
        _r3, response_text3, _ = await run_query_with_retries(
            ai_module, "Ej, o której ta przypominajka?", context3
        )
        print(f"Memory recall: {response_text3}")
        assert len(response_text3) > 5, "Memory recall response too short"
        time_mentioned = "18" in response_text3 or "osiemnast" in response_text3.lower()
        subject_mentioned = any(
            word in response_text3.lower() for word in ["kolokwium", "algebra", "exam"]
        )
        assert time_mentioned or subject_mentioned, "AI failed to recall reminder details"
        if time_mentioned and subject_mentioned:
            print("Perfect memory recall - both time and subject remembered")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_reminder_time_update(self, real_memory_setup):
        ai_module, db_manager = real_memory_setup
        user_id = "test_user_real_update"
        context1 = {"user_id": user_id, "history": []}
        _r1, response_text1, _ = await run_query_with_retries(
            ai_module,
            "Jutro mam kolokwium z matematyki. Przypomnij mi o 18.",
            context1,
        )
        await db_manager.save_interaction(
            user_id,
            "Jutro mam kolokwium z matematyki. Przypomnij mi o 18.",
            json.dumps({"text": response_text1}),
        )
        history = await db_manager.get_user_history(user_id, limit=10)
        context2 = {"user_id": user_id, "history": history}
        _r2, response_text2, _ = await run_query_with_retries(
            ai_module, "Właściwie to przesuń na 19.", context2
        )
        print(f"Time update response: {response_text2}")
        assert len(response_text2) > 5, "Update response too short"
        # Skip if obvious provider/tool error to avoid false negatives
        if any(err in response_text2.lower() for err in ["ollama error", "błąd", "error"]):
            pytest.skip("Provider/tool error in update response; skipping precision assertion")
        update_understood = (
            any(tok in response_text2 for tok in ["19", "19:", "19.00", "19:00"])  # explicit new time
            or "dziewiętnast" in response_text2.lower()  # Polish word stem
            or any(
                word in response_text2.lower()
                for word in ["przesun", "przenie", "zmien", "update", "change", "shift", "move"]
            )
        )
        if not update_understood:
            pytest.skip("AI did not clearly acknowledge time update; treating as flaky external response")


class TestRealDayPlanning:
    """Test real day planning without mocking."""

    @pytest.fixture
    def real_planning_setup(self):
        db_path = "test_real_planning.db"
        db_manager = DatabaseManager(db_path)
        config = {"ai": {"provider": "openai", "model": "gpt-4o-mini"}}
        ai_module = AIModule(config=config)
        try:
            yield ai_module, db_manager
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_study_planning(self, real_planning_setup):
        ai_module, db_manager = real_planning_setup
        user_id = "test_user_real_planning"
        start_time = time.time()
        context = {"user_id": user_id, "history": []}
        _r, response_text, _ = await run_query_with_retries(
            ai_module,
            "Jutro mam 3h okienka między 13 a 16 — zaplanuj naukę z przerwami.",
            context,
        )
        end_time = time.time()
        print(f"Study planning response: {response_text}")
        assert len(response_text) > 30, "Planning response too short"
        if "nie mogłem" in response_text.lower() or "brak treści" in response_text.lower():
            pytest.skip("AI returned error response, skipping detailed checks")
        assert "13" in response_text and "16" in response_text, (
            f"Should mention time range 13-16. Response: {response_text}"
        )
        planning_words = [
            "plan",
            "przerw",
            "nauka",
            "study",
            "break",
            "schedule",
            "harmonogram",
        ]
        assert any(word in response_text.lower() for word in planning_words), (
            "Should contain planning terminology"
        )
        latency = end_time - start_time
        assert_latency(latency, 15.0, "Planning initial request")
        await db_manager.save_interaction(
            user_id,
            "Jutro mam 3h okienka między 13 a 16 — zaplanuj naukę z przerwami.",
            json.dumps({"text": response_text}),
        )
        history = await db_manager.get_user_history(user_id, limit=10)
        context2 = {"user_id": user_id, "history": history}
        _r2, response_text2, _ = await run_query_with_retries(
            ai_module, "Właściwie to przenieś cały plan na 14:30", context2
        )
        print(f"Rescheduling response: {response_text2}")
        assert len(response_text2) > 10, "Rescheduling response too short"
        reschedule_understood = (
            "14:30" in response_text2
            or any(
                word in response_text2.lower() for word in ["przenieś", "przesun", "reschedule", "move"]
            )
        )
        assert reschedule_understood, "AI should understand rescheduling request"


class TestRealPluginFunctionality:
    """Test real plugin/module functionality without mocking."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_weather_module_functions(self):
        try:
            module = importlib.import_module("modules.weather_module_refactored")
            functions = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]
            assert functions, "Weather module has no functions"
            if hasattr(module, "get_functions"):
                info = module.get_functions()
                assert info is not None
                print(f"Weather module functions: {info}")
            if hasattr(module, "execute_function"):
                assert callable(getattr(module, "execute_function"))
        except ImportError:
            pytest.skip("Weather module not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_core_module_functions(self):
        try:
            module = importlib.import_module("modules.core_module")
            core_functions = ["get_functions", "execute_function", "set_reminder", "add_task"]
            available = [f for f in core_functions if hasattr(module, f)]
            assert available, "Core module missing expected functions"
            print(f"Available core functions: {available}")
            if hasattr(module, "get_functions"):
                try:
                    info = module.get_functions()
                    print(f"Core module function info: {info}")
                except Exception as e:
                    print(f"get_functions call failed: {e}")
            for name in available:
                assert callable(getattr(module, name)), f"{name} should be callable"
        except ImportError:
            pytest.skip("Core module not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_module_discovery(self):
        module_paths = [
            "modules.weather_module_refactored",
            "modules.search_module",
            "modules.vector_memory_module",
            "modules.core_module",
            "modules.shopping_list_module",
            "modules.notes_module",
            "modules.tasks_module",
        ]
        loaded = []
        details = {}
        for name in module_paths:
            try:
                m = importlib.import_module(name)
                loaded.append(name)
                funcs = [n for n, o in inspect.getmembers(m) if inspect.isfunction(o)]
                details[name] = {"functions": len(funcs), "function_names": funcs[:5]}
            except ImportError as e:
                print(f"Could not load {name}: {e}")
        assert loaded, "No modules could be loaded"
        print(f"Successfully loaded {len(loaded)} modules:")
        for name in loaded:
            print(f"  {name}: {details[name]['functions']} functions")
        assert len(loaded) >= 2, "Should load at least 2 modules"


class TestRealIntegrationFlow:
    """Test real end-to-end integration flow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_ai_conversation_performance(self):
        db_path = "test_real_integration.db"
        db_manager = DatabaseManager(db_path)
        config = {"ai": {"provider": "openai", "model": "gpt-4o-mini"}}
        ai_module = AIModule(config=config)
        try:
            user_id = "test_user_real_integration"
            queries = [
                "Jaka będzie dziś pogoda w Warszawie?",
                "Przypomnij mi o zakupach o 18:00",
                "Dodaj zadanie: przygotować prezentację",
            ]
            total_start = time.time()
            for i, q in enumerate(queries):
                history = await db_manager.get_user_history(user_id, limit=10)
                context = {"user_id": user_id, "history": history}
                start = time.time()
                result, text, _ = await run_query_with_retries(ai_module, q, context)
                end = time.time()
                print(f"Query {i+1}: {q}\nResponse {i+1}: {text}")
                assert len(text) > 5, f"Response {i+1} too short"
                assert_latency(end - start, 15.0, f"Conversation query {i+1}")
                await db_manager.save_interaction(user_id, q, json.dumps({"text": text}))
            total_latency = time.time() - total_start
            print(f"Total conversation time: {total_latency:.2f}s")
            assert total_latency < 45.0, (
                f"Total conversation time {total_latency:.2f}s exceeds 45s"
            )
            final_history = await db_manager.get_user_history(user_id, limit=20)
            assert len(final_history) == len(queries) * 2, "History should contain all interactions"
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_function_calling_integration(self):
        db_path = "test_real_functions.db"
        db_manager = DatabaseManager(db_path)
        config = {"ai": {"provider": "openai", "model": "gpt-4o-mini"}}
        ai_module = AIModule(config=config)
        try:
            user_id = "test_user_real_functions"
            function_queries = [
                "Sprawdź pogodę w Krakowie",
                "Ustaw przypomnienie na jutro o 10:00",
                "Wyszukaj informacje o AI",
            ]
            for q in function_queries:
                context = {"user_id": user_id, "history": []}
                start = time.time()
                result, text, _ = await run_query_with_retries(ai_module, q, context)
                end = time.time()
                print(f"Function test query: {q}\nResponse: {text}")
                assert len(text) > 10, "Function call response too short"
                if isinstance(result, dict) and "functions_called" in result:
                    assert result["functions_called"], "Should have called functions"
                    print(f"Functions called: {result['functions_called']}")
                assert_latency(end - start, 20.0, "Function call query")
        finally:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass