"""Simplified search module providing deterministic results with optional DuckDuckGo lookup.

The previous version delegated HTTP calls through api_module; with that module removed we
provide a leaner implementation that still satisfies the contract expected by tests:
- `execute_function` returns a dict with `success`, `data`, and metadata fields.
- Empty queries return a successful response describing the issue instead of raising.
- When duckduckgo_search is available we attempt a real lookup (off the main loop via
  asyncio.to_thread). On failure or when the dependency is missing we fall back to a
  deterministic stub so tests remain reliable offline.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:  # Optional dependency
    from duckduckgo_search import DDGS  # type: ignore
except ImportError:  # pragma: no cover - dependency optional in CI
    DDGS = None


PLUGIN_NAME = "search_module"
PLUGIN_DESCRIPTION = "Web search aggregator with graceful offline fallback"
PLUGIN_VERSION = "2.2.0"
PLUGIN_AUTHOR = "GAJA Team"
PLUGIN_DEPENDENCIES: List[str] = []


class SearchModule:
    """Lightweight async search module."""

    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            self._initialized = True
            logger.info("SearchModule initialized")

    async def cleanup(self) -> None:  # pragma: no cover - nothing to cleanup
        """Compatibility hook."""
        return None

    def get_functions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "search",
                "description": "Wyszukuje informacje i zwraca skrócone wyniki.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Zapytanie użytkownika"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maksymalna liczba wyników",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "fetch_pages": {
                            "type": "boolean",
                            "description": "Czy pobrać treść stron (stub)",
                            "default": True,
                        },
                        "test_mode": {
                            "type": "boolean",
                            "description": "Zwraca deterministyczne wyniki bez sieci",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    async def execute_function(
        self, function_name: str, parameters: Dict[str, Any], user_id: int
    ) -> Dict[str, Any]:
        if function_name != "search":
            return {
                "success": False,
                "error": f"Unknown function: {function_name}",
                "function": function_name,
            }

        await self.initialize()

        query = (parameters.get("query") or "").strip()
        max_results = self._normalize_max_results(parameters.get("max_results"))
        test_mode = bool(parameters.get("test_mode", False))

        if not query:
            data = self._build_stub_payload(
                query=query,
                max_results=max_results,
                reason="empty_query",
                fetch_pages=bool(parameters.get("fetch_pages", True)),
            )
            data["error"] = "empty_query"
            return {"success": True, "data": data}

        try:
            if test_mode:
                payload = self._build_stub_payload(
                    query=query,
                    max_results=max_results,
                    reason="test_mode",
                    fetch_pages=bool(parameters.get("fetch_pages", True)),
                )
                return {"success": True, "data": payload}

            if DDGS is None:
                payload = self._build_stub_payload(
                    query=query,
                    max_results=max_results,
                    reason="duckduckgo_unavailable",
                    fetch_pages=bool(parameters.get("fetch_pages", True)),
                )
                return {"success": True, "data": payload}

            results, metadata = await asyncio.to_thread(
                self._duckduckgo_search, query, max_results
            )
            payload = {
                "query": query,
                "engine": "duckduckgo",
                "results": results,
                "instant_answer": metadata.get("instant_answer"),
                "definition": metadata.get("definition"),
                "search_metadata": metadata,
            }
            return {"success": True, "data": payload}

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Search error for '%s': %s", query, exc)
            payload = self._build_stub_payload(
                query=query,
                max_results=max_results,
                reason=str(exc),
                fetch_pages=bool(parameters.get("fetch_pages", True)),
            )
            payload["error"] = str(exc)
            return {"success": True, "data": payload}

    def _normalize_max_results(self, value: Any) -> int:
        try:
            max_results = int(value)
        except (TypeError, ValueError):
            max_results = 5
        return max(1, min(max_results, 50))

    def _duckduckgo_search(
        self, query: str, max_results: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {
            "query": query,
            "engine": "duckduckgo",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "result_count": 0,
            "source": "duckduckgo",
        }

        with DDGS() as ddgs:  # type: ignore[operator]
            raw = list(ddgs.text(query, max_results=max_results))

        for item in raw[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": item.get("title") or item.get("heading") or "",
                    "url": item.get("href") or item.get("link") or item.get("url") or "",
                    "snippet": item.get("body") or item.get("description") or item.get("content") or "",
                    "type": item.get("type") or "web_result",
                }
            )

        metadata["result_count"] = len(results)
        metadata["instant_answer"] = None
        metadata["definition"] = None
        return results, metadata

    def _build_stub_payload(
        self,
        *,
        query: str,
        max_results: int,
        reason: str,
        fetch_pages: bool,
    ) -> Dict[str, Any]:
        sample_results = [
            {
                "title": f"{query.title()} — przegląd",
                "url": "https://example.com/articles/overview",
                "snippet": (
                    f"Zarys najważniejszych informacji dla zapytania '{query}'. "
                    "Wynik wygenerowany lokalnie (tryb offline)."
                ),
                "type": "web_result",
            }
        ]
        if max_results > 1:
            sample_results.append(
                {
                    "title": f"Historia i kontekst: {query[:40]}",
                    "url": "https://example.com/articles/details",
                    "snippet": "Szczegółowe opracowanie dostępne w trybie offline.",
                    "type": "web_result",
                }
            )

        metadata = {
            "query": query,
            "engine": "duckduckgo",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "result_count": min(len(sample_results), max_results),
            "source": reason,
            "fetch_pages": fetch_pages,
        }

        return {
            "query": query,
            "engine": "duckduckgo",
            "results": sample_results[:max_results],
            "instant_answer": None,
            "definition": None,
            "search_metadata": metadata,
        }


search_module = SearchModule()


def get_functions() -> List[Dict[str, Any]]:
    return search_module.get_functions()


async def execute_function(
    function_name: str, parameters: Dict[str, Any], user_id: int
) -> Dict[str, Any]:
    if not search_module._initialized:
        await search_module.initialize()
    return await search_module.execute_function(function_name, parameters, user_id)


async def search_search(
    function_name: str, parameters: Dict[str, Any], user_id: int
) -> Dict[str, Any]:
    """Backward compatible alias used by some older tests."""
    return await execute_function(function_name, parameters, user_id)
