import logging
import re
from datetime import datetime
from typing import Any

from .api_module import get_api_module

logger = logging.getLogger(__name__)


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Zwraca listę dostępnych funkcji w pluginie."""
    return [
        {
            "name": "search",
            "description": "Wyszukuje informacje w internecie",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Zapytanie do wyszukania",
                    },
                    "engine": {
                        "type": "string",
                        "description": "Silnik wyszukiwania",
                        "enum": ["duckduckgo", "google", "bing"],
                        "default": "duckduckgo",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Tryb testowy (używa mock danych)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_news",
            "description": "Wyszukuje najnowsze wiadomości",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Temat wiadomości do wyszukania",
                    },
                    "language": {
                        "type": "string",
                        "description": "Język wiadomości",
                        "default": "pl",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Tryb testowy (używa mock danych)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Wykonuje funkcję pluginu."""
    search_module = SearchModule()
    await search_module.initialize()

    try:
        if function_name == "search":
            query = parameters.get("query")
            engine = parameters.get("engine", "duckduckgo")
            max_results = parameters.get("max_results", 10)
            test_mode = parameters.get("test_mode", False)

            if test_mode:
                # Zwróć mock dane
                mock_data = search_module._get_mock_search_data(query, max_results)
                return {
                    "success": True,
                    "data": mock_data,
                    "message": f"Wyszukano informacje dla: {query} (tryb testowy)",
                    "test_mode": True,
                }

            result = await search_module.search(
                user_id, query, engine, max_results=max_results
            )
            return {
                "success": True,
                "data": result,
                "message": f"Wyszukano informacje dla: {query}",
            }

        elif function_name == "search_news":
            query = parameters.get("query")
            language = parameters.get("language", "pl")
            max_results = parameters.get("max_results", 5)
            test_mode = parameters.get("test_mode", False)

            # Sprawdź czy jest tryb testowy lub brak klucza API
            api_key = await search_module._get_user_api_key(user_id, "newsapi")
            if not api_key or test_mode:
                # Zwróć mock dane
                mock_data = search_module._get_mock_news_data(query, max_results)
                return {
                    "success": True,
                    "data": mock_data,
                    "message": f"Znaleziono najnowsze wiadomości dla: {query} (tryb testowy)",
                    "test_mode": True,
                }

            result = await search_module.search_news(
                user_id, query, language, max_results, api_key
            )
            return {
                "success": True,
                "data": result,
                "message": f"Znaleziono najnowsze wiadomości dla: {query}",
            }

        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}

    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Ensure cleanup
        await search_module.cleanup()


class SearchModule:
    """Moduł wyszukiwania informacji w internecie."""

    def __init__(self):
        self.api_module = None
        self.search_engines = {
            "duckduckgo": self._search_duckduckgo,
            "google": self._search_google,
            "bing": self._search_bing,
        }

    async def initialize(self):
        """Inicjalizuje moduł wyszukiwania."""
        self.api_module = await get_api_module()
        logger.info("SearchModule initialized")

    async def search(
        self,
        user_id: int,
        query: str,
        engine: str = "duckduckgo",
        api_key: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Wykonuje wyszukiwanie w internecie.

        Args:
            user_id: ID użytkownika
            query: Zapytanie wyszukiwania
            engine: Silnik wyszukiwania
            api_key: Klucz API (jeśli wymagany)
            max_results: Maksymalna liczba wyników

        Returns:
            Wyniki wyszukiwania
        """
        if not self.api_module:
            await self.initialize()

        if engine not in self.search_engines:
            return {
                "error": f"Nieobsługiwany silnik wyszukiwania: {engine}",
                "available_engines": list(self.search_engines.keys()),
            }

        try:
            search_func = self.search_engines[engine]
            results = await search_func(user_id, query, api_key, max_results)

            # Dodaj metadane wyszukiwania
            results["search_metadata"] = {
                "query": query,
                "engine": engine,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
            }

            return results

        except Exception as e:
            logger.error(f"Search error with {engine}: {e}")
            return {
                "error": f"Błąd wyszukiwania: {str(e)}",
                "query": query,
                "engine": engine,
            }

    async def _search_duckduckgo(
        self,
        user_id: int,
        query: str,
        api_key: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Wyszukiwanie za pomocą DuckDuckGo."""

        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

        response = await self.api_module.get(user_id, url, params=params)

        if response.get("status") != 200:
            return {"error": "Błąd połączenia z DuckDuckGo", "details": response}

        data = response.get("data", {})

        # Przetwórz wyniki DuckDuckGo
        results: dict[str, Any] = {
            "engine": "duckduckgo",
            "query": query,
            "results": [],
            "instant_answer": None,
            "definition": None,
        }

        # Instant Answer (jeśli jest)
        if data.get("Answer"):
            results["instant_answer"] = {
                "answer": data["Answer"],
                "type": data.get("AnswerType", ""),
                "source": data.get("AbstractSource", ""),
            }

        # Definicja (jeśli jest)
        if data.get("Definition"):
            results["definition"] = {
                "definition": data["Definition"],
                "source": data.get("DefinitionSource", ""),
                "url": data.get("DefinitionURL", ""),
            }

        # Abstrakty
        if data.get("Abstract"):
            results["results"].append(
                {
                    "title": data.get("Heading", "Informacja"),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                    "type": "abstract",
                }
            )

        # Related Topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results["results"].append(
                    {
                        "title": (
                            topic.get("Text", "").split(" - ")[0]
                            if " - " in topic.get("Text", "")
                            else "Powiązany temat"
                        ),
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "type": "related_topic",
                    }
                )

        return results

    async def _search_google(
        self,
        user_id: int,
        query: str,
        api_key: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Wyszukiwanie za pomocą Google Custom Search API."""

        if not api_key:
            return {
                "error": "Google Search wymaga klucza API",
                "help": "Ustaw klucz Google Custom Search API",
            }

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "q": query,
            "num": min(max_results, 10),  # Google pozwala max 10 na zapytanie
            "gl": "pl",  # Lokalizacja - Polska
            "hl": "pl",  # Język
            "safe": "medium",
        }

        response = await self.api_module.get(user_id, url, params=params)

        if response.get("status") != 200:
            return {"error": "Błąd połączenia z Google Search API", "details": response}

        data = response.get("data", {})

        results = {
            "engine": "google",
            "query": query,
            "results": [],
            "search_information": data.get("searchInformation", {}),
            "spelling": data.get("spelling", {}),
        }

        # Przetwórz wyniki Google
        for item in data.get("items", []):
            result = {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "display_link": item.get("displayLink", ""),
                "type": "web_result",
            }

            # Dodaj obrazek jeśli jest
            if "pagemap" in item and "cse_image" in item["pagemap"]:
                images = item["pagemap"]["cse_image"]
                if images:
                    result["image"] = images[0].get("src")

            results["results"].append(result)

        return results

    async def _search_bing(
        self,
        user_id: int,
        query: str,
        api_key: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Wyszukiwanie za pomocą Bing Search API."""

        if not api_key:
            return {
                "error": "Bing Search wymaga klucza API",
                "help": "Ustaw klucz Bing Search API",
            }

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": min(max_results, 50),
            "offset": 0,
            "mkt": "pl-PL",
            "safesearch": "Moderate",
        }

        response = await self.api_module.get(
            user_id, url, headers=headers, params=params
        )

        if response.get("status") != 200:
            return {"error": "Błąd połączenia z Bing Search API", "details": response}

        data = response.get("data", {})

        results = {
            "engine": "bing",
            "query": query,
            "results": [],
            "web_pages": data.get("webPages", {}),
            "query_context": data.get("queryContext", {}),
        }
        # Przetwórz wyniki Bing
        web_pages = data.get("webPages", {})
        for item in web_pages.get("value", []):
            result = {
                "title": item.get("name", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("url", ""),
                "display_url": item.get("displayUrl", ""),
                "date_last_crawled": item.get("dateLastCrawled", ""),
                "type": "web_result",
            }

            results["results"].append(result)

        return results

    async def search_news(
        self,
        user_id: int,
        query: str | None = None,
        language: str = "pl",
        max_results: int = 5,
        api_key: str | None = None,
        category: str | None = None,
        country: str = "pl",
    ) -> dict[str, Any]:
        """Wyszukuje wiadomości.

        Args:
            user_id: ID użytkownika
            query: Zapytanie (opcjonalne)
            category: Kategoria wiadomości
            country: Kod kraju
            api_key: Klucz NewsAPI

        Returns:
            Wyniki wyszukiwania wiadomości
        """
        if not api_key:
            return {
                "error": "Wyszukiwanie wiadomości wymaga klucza NewsAPI",
                "help": "Ustaw klucz API z https://newsapi.org",
            }

        if not self.api_module:
            await self.initialize()

        return await self.api_module.get_news(
            user_id=user_id,
            api_key=api_key,
            query=query,
            category=category,
            country=country,
        )

    async def search_images(
        self, user_id: int, query: str, api_key: str | None = None, engine: str = "bing"
    ) -> dict[str, Any]:
        """Wyszukuje obrazy.

        Args:
            user_id: ID użytkownika
            query: Zapytanie wyszukiwania
            api_key: Klucz API
            engine: Silnik wyszukiwania obrazów

        Returns:
            Wyniki wyszukiwania obrazów
        """
        if engine == "bing" and api_key:
            url = "https://api.bing.microsoft.com/v7.0/images/search"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {
                "q": query,
                "count": 20,
                "offset": 0,
                "mkt": "pl-PL",
                "safeSearch": "Moderate",
            }

            if not self.api_module:
                await self.initialize()

            response = await self.api_module.get(
                user_id, url, headers=headers, params=params
            )

            if response.get("status") != 200:
                return {"error": "Błąd wyszukiwania obrazów", "details": response}

            data = response.get("data", {})
            images = []

            for item in data.get("value", []):
                images.append(
                    {
                        "title": item.get("name", ""),
                        "url": item.get("webSearchUrl", ""),
                        "thumbnail_url": item.get("thumbnailUrl", ""),
                        "content_url": item.get("contentUrl", ""),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "size": item.get("contentSize", ""),
                        "host_page_url": item.get("hostPageUrl", ""),
                    }
                )

            return {
                "engine": "bing_images",
                "query": query,
                "images": images,
                "total_estimated_matches": data.get("totalEstimatedMatches", 0),
            }

        else:
            return {
                "error": f"Nieobsługiwany silnik obrazów: {engine}",
                "supported_engines": ["bing"],
            }

    def extract_search_intent(self, query: str) -> dict[str, Any]:
        """Analizuje intencję zapytania wyszukiwania.

        Args:
            query: Zapytanie użytkownika

        Returns:
            Analiza intencji
        """
        query_lower = query.lower()

        # Wzorce dla różnych typów zapytań
        patterns = {
            "weather": r"pogoda|temperatura|deszcz|słońce|prognoza pogody",
            "news": r"wiadomości|aktualności|newsy|co się dzieje",
            "definition": r"co to jest|co oznacza|definicja|znaczenie",
            "how_to": r"jak|w jaki sposób|tutorial|instrukcja",
            "when": r"kiedy|o której|w którym roku|data",
            "where": r"gdzie|w którym miejscu|lokalizacja|adres",
            "who": r"kto|kim jest|biografia",
            "why": r"dlaczego|z jakiego powodu|przyczyna",
            "price": r"cena|koszt|ile kosztuje|za ile",
            "review": r"opinia|recenzja|test|ocena",
        }

        intent = {
            "query": query,
            "type": "general",
            "keywords": query_lower.split(),
            "suggested_engine": "duckduckgo",
            "filters": [],
        }

        # Wykryj typ zapytania
        for intent_type, pattern in patterns.items():
            if re.search(pattern, query_lower):
                intent["type"] = intent_type
                break

        # Sugeruj silnik na podstawie typu
        if intent["type"] == "news":
            intent["suggested_engine"] = "google"
        elif intent["type"] == "weather":
            intent["suggested_engine"] = "duckduckgo"
        elif intent["type"] in ["definition", "how_to"]:
            intent["suggested_engine"] = "duckduckgo"

        # Wyciągnij potencjalne filtry
        if "ostatni" in query_lower or "najnowszy" in query_lower:
            intent["filters"].append("recent")

        if any(word in query_lower for word in ["polski", "polska", "polskie"]):
            intent["filters"].append("polish")

        return intent

    async def smart_search(
        self, user_id: int, query: str, user_api_keys: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Inteligentne wyszukiwanie z automatycznym wyborem silnika.

        Args:
            user_id: ID użytkownika
            query: Zapytanie
            user_api_keys: Klucze API użytkownika

        Returns:
            Wyniki inteligentnego wyszukiwania
        """
        if user_api_keys is None:
            user_api_keys = {}

        # Analizuj intencję
        intent = self.extract_search_intent(query)

        # Wybierz najlepszy silnik
        engine = intent["suggested_engine"]

        # Sprawdź dostępność API key
        if engine == "google" and "google_search" not in user_api_keys:
            engine = "duckduckgo"  # Fallback

        api_key = user_api_keys.get(f"{engine}_search") or user_api_keys.get(
            "google_search"
        )

        # Wykonaj wyszukiwanie
        results = await self.search(
            user_id=user_id, query=query, engine=engine, api_key=api_key
        )

        # Dodaj analizę intencji do wyników
        results["intent_analysis"] = intent
        return results

    async def _get_user_api_key(self, user_id: int, provider: str) -> str | None:
        """Pobiera klucz API użytkownika dla danego providera.

        Args:
            user_id: ID użytkownika
            provider: Provider (newsapi, google_search, bing_search)

        Returns:
            Klucz API lub None
        """
        try:
            from config_manager import get_database_manager

            db_manager = get_database_manager()
            return db_manager.get_user_api_key(user_id, provider)
        except Exception as e:
            logger.error(
                f"Error getting API key for user {user_id}, provider {provider}: {e}"
            )
            return None

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "session") and self.session:
            await self.session.close()
            self.session = None
        logger.debug("SearchModule cleanup completed")

    def _get_mock_search_data(self, query: str, max_results: int) -> dict[str, Any]:
        """Zwraca przykładowe wyniki wyszukiwania (mock) dla testów."""
        return {
            "query": query,
            "results": [
                {
                    "title": f'Przykładowy wynik 1 dla "{query}"',
                    "url": "https://example.com/result1",
                    "snippet": f'To jest przykładowy opis wyników wyszukiwania dla zapytania "{query}". Zawiera różne informacje związane z tym tematem.',
                    "displayed_url": "example.com/result1",
                    "rank": 1,
                },
                {
                    "title": f"Informacje o {query} - Wikipedia",
                    "url": "https://pl.wikipedia.org/wiki/example",
                    "snippet": f"Artykuł z Wikipedii o {query}. Zawiera szczegółowe informacje, historię i najnowsze dane.",
                    "displayed_url": "pl.wikipedia.org/wiki/example",
                    "rank": 2,
                },
                {
                    "title": f"Najnowsze informacje o {query}",
                    "url": "https://news.example.com/latest",
                    "snippet": f"Najnowsze wiadomości i aktualizacje dotyczące {query}. Regularnie aktualizowane informacje.",
                    "displayed_url": "news.example.com/latest",
                    "rank": 3,
                },
            ][
                :max_results
            ],  # Ogranicz do żądanej liczby wyników
            "total_results": max_results,
            "search_time": 0.15,
            "engine": "duckduckgo_mock",
            "test_mode": True,
        }

    def _get_mock_news_data(self, query: str, max_results: int) -> dict[str, Any]:
        """Zwraca przykładowe wyniki wiadomości (mock) dla testów."""
        from datetime import datetime, timedelta

        # Generuj przykładowe daty
        now = datetime.now()
        dates = [
            (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(max_results)
        ]

        return {
            "query": query,
            "articles": [
                {
                    "title": f"Ważne wiadomości o {query}",
                    "description": f"Najnowsze informacje dotyczące {query}. To jest przykładowy opis artykułu prasowego.",
                    "url": "https://news.example.com/article1",
                    "published_at": (
                        dates[0] if dates else now.strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    "source": {
                        "name": "Example News",
                        "url": "https://news.example.com",
                    },
                    "author": "Jan Kowalski",
                    "category": "general",
                },
                {
                    "title": f"Analiza sytuacji związanej z {query}",
                    "description": f"Szczegółowa analiza i komentarz ekspertów na temat {query}.",
                    "url": "https://analysis.example.com/article2",
                    "published_at": (
                        dates[1]
                        if len(dates) > 1
                        else now.strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    "source": {
                        "name": "Expert Analysis",
                        "url": "https://analysis.example.com",
                    },
                    "author": "Anna Nowak",
                    "category": "business",
                },
                {
                    "title": f"Wydarzenia związane z {query}",
                    "description": f"Relacja z ostatnich wydarzeń i ich wpływ na {query}.",
                    "url": "https://events.example.com/article3",
                    "published_at": (
                        dates[2]
                        if len(dates) > 2
                        else now.strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    "source": {
                        "name": "Event Reporter",
                        "url": "https://events.example.com",
                    },
                    "author": "Piotr Wiśniewski",
                    "category": "politics",
                },
            ][
                :max_results
            ],  # Ogranicz do żądanej liczby wyników
            "total_results": max_results,
            "language": "pl",
            "test_mode": True,
        }


# Globalna instancja
_search_module = None


async def get_search_module() -> SearchModule:
    """Pobiera globalną instancję modułu wyszukiwania."""
    global _search_module
    if _search_module is None:
        _search_module = SearchModule()
        await _search_module.initialize()
    return _search_module
