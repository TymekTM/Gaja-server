import asyncio
import logging
import urllib.parse
from typing import Any

import aiohttp
from config_manager import get_database_manager

logger = logging.getLogger(__name__)


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Zwraca listę dostępnych funkcji w pluginie."""
    return [
        {
            "name": "make_api_request",
            "description": "Wykonuje zapytanie do zewnętrznego API",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Metoda HTTP (GET, POST, PUT, DELETE)",
                    },
                    "url": {"type": "string", "description": "URL API"},
                    "headers": {"type": "object", "description": "Nagłówki HTTP"},
                    "params": {"type": "object", "description": "Parametry URL"},
                    "json_data": {"type": "object", "description": "Dane JSON w body"},
                },
                "required": ["method", "url"],
            },
        }
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Wykonuje funkcję pluginu."""
    api_module = APIModule()
    await api_module.initialize()

    try:
        if function_name == "make_api_request":
            method = parameters.get("method", "GET")
            url = parameters.get("url")
            headers = parameters.get("headers", {})
            params = parameters.get("params", {})
            json_data = parameters.get("json_data")

            result = await api_module.make_request(
                user_id=user_id,
                method=method,
                url=url,
                headers=headers,
                params=params,
                json_data=json_data,
            )

            return {
                "success": True,
                "data": result,
                "message": f"Wykonano zapytanie do {url}",
            }

        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}

    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        return {"success": False, "error": str(e)}
    finally:
        await api_module.cleanup()


class APIModule:
    """Moduł do obsługi zewnętrznych API."""

    def __init__(self):
        self.session = None
        self.db_manager = get_database_manager()

    async def initialize(self):
        """Inicjalizuje sesję HTTP."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("APIModule initialized")

    async def cleanup(self):
        """Czyści zasoby."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("APIModule cleaned up")

    async def make_request(
        self,
        user_id: int,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        data: Any = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Wykonuje zapytanie HTTP z logowaniem użycia API.

        Args:
            user_id: ID użytkownika
            method: Metoda HTTP
            url: URL zapytania
            headers: Nagłówki
            params: Parametry URL
            data: Dane w body
            json_data: Dane JSON w body

        Returns:
            Odpowiedź API
        """
        await self.initialize()

        start_time = asyncio.get_event_loop().time()
        success = False
        error_message = None
        response_data = None

        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
            ) as response:
                response_data = {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url),
                }

                # Sprawdź content-type
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    response_data["data"] = await response.json()
                else:
                    response_data["data"] = await response.text()

                success = response.status < 400

                if not success:
                    error_message = f"HTTP {response.status}: {response.reason}"

        except Exception as e:
            error_message = str(e)
            logger.error(f"API request failed: {error_message}")
            response_data = {"error": error_message}

        finally:
            # Loguj użycie API
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            # Wyciągnij provider z URL
            parsed_url = urllib.parse.urlparse(url)
            api_provider = parsed_url.netloc or "unknown"
            endpoint = parsed_url.path or url

            self.db_manager.log_api_usage(
                user_id=user_id,
                api_provider=api_provider,
                endpoint=endpoint,
                method=method.upper(),
                success=success,
                response_time=response_time,
                error_message=error_message,
                metadata={
                    "url": url,
                    "status_code": (
                        response_data.get("status") if response_data else None
                    ),
                },
            )

        return response_data

    async def get(self, user_id: int, url: str, **kwargs) -> dict[str, Any]:
        """Wykonuje zapytanie GET."""
        return await self.make_request(user_id, "GET", url, **kwargs)

    async def post(self, user_id: int, url: str, **kwargs) -> dict[str, Any]:
        """Wykonuje zapytanie POST."""
        return await self.make_request(user_id, "POST", url, **kwargs)

    async def put(self, user_id: int, url: str, **kwargs) -> dict[str, Any]:
        """Wykonuje zapytanie PUT."""
        return await self.make_request(user_id, "PUT", url, **kwargs)

    async def delete(self, user_id: int, url: str, **kwargs) -> dict[str, Any]:
        """Wykonuje zapytanie DELETE."""
        return await self.make_request(user_id, "DELETE", url, **kwargs)

    async def call_openai_api(
        self,
        user_id: int,
        user_api_key: str,
        messages: list[dict[str, str]],
        model: str = "gpt-4.1-nano",
        **kwargs,
    ) -> dict[str, Any]:
        """Wywołuje OpenAI API.

        Args:
            user_id: ID użytkownika
            user_api_key: Klucz API użytkownika
            messages: Lista wiadomości
            model: Model do użycia
            **kwargs: Dodatkowe parametry

        Returns:
            Odpowiedź z OpenAI API
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {user_api_key}",
            "Content-Type": "application/json",
        }

        payload = {"model": model, "messages": messages, **kwargs}

        response = await self.post(user_id, url, headers=headers, json_data=payload)

        # Dodatkowe logowanie dla OpenAI
        if response.get("status") == 200 and "data" in response:
            data = response["data"]
            if "usage" in data:
                usage = data["usage"]
                self.db_manager.log_api_usage(
                    user_id=user_id,
                    api_provider="api.openai.com",
                    endpoint="/v1/chat/completions",
                    method="POST",
                    tokens_used=usage.get("total_tokens", 0),
                    success=True,
                    metadata={
                        "model": model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                )

        return response

    async def call_anthropic_api(
        self,
        user_id: int,
        user_api_key: str,
        messages: list[dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        **kwargs,
    ) -> dict[str, Any]:
        """Wywołuje Anthropic API.

        Args:
            user_id: ID użytkownika
            user_api_key: Klucz API użytkownika
            messages: Lista wiadomości
            model: Model do użycia
            **kwargs: Dodatkowe parametry

        Returns:
            Odpowiedź z Anthropic API
        """
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": user_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Konwertuj format wiadomości dla Anthropic
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            **{k: v for k, v in kwargs.items() if k != "max_tokens"},
        }

        if system_message:
            payload["system"] = system_message

        response = await self.post(user_id, url, headers=headers, json_data=payload)

        # Dodatkowe logowanie dla Anthropic
        if response.get("status") == 200 and "data" in response:
            data = response["data"]
            if "usage" in data:
                usage = data["usage"]
                self.db_manager.log_api_usage(
                    user_id=user_id,
                    api_provider="api.anthropic.com",
                    endpoint="/v1/messages",
                    method="POST",
                    tokens_used=usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                    success=True,
                    metadata={
                        "model": model,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    },
                )

        return response

    async def search_web(
        self,
        user_id: int,
        query: str,
        api_key: str | None = None,
        engine: str = "google",
    ) -> dict[str, Any]:
        """Wyszukuje w internecie.

        Args:
            user_id: ID użytkownika
            query: Zapytanie
            api_key: Klucz API do wyszukiwarki
            engine: Silnik wyszukiwania

        Returns:
            Wyniki wyszukiwania
        """
        if engine == "google" and api_key:
            # Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": api_key, "q": query, "num": 10}
            return await self.get(user_id, url, params=params)

        elif engine == "duckduckgo":
            # DuckDuckGo Instant Answer API (darmowe)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            return await self.get(user_id, url, params=params)

        else:
            return {"error": f"Unsupported search engine: {engine}", "status": 400}

    async def get_weather(
        self, user_id: int, location: str, api_key: str, provider: str = "openweather"
    ) -> dict[str, Any]:
        """Pobiera prognozę pogody.

        Args:
            user_id: ID użytkownika
            location: Lokalizacja
            api_key: Klucz API
            provider: Provider pogodowy

        Returns:
            Dane pogodowe
        """
        if provider == "openweather":
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"q": location, "appid": api_key, "units": "metric", "lang": "pl"}
            return await self.get(user_id, url, params=params)

        else:
            return {"error": f"Unsupported weather provider: {provider}", "status": 400}

    async def translate_text(
        self,
        user_id: int,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        provider: str = "google",
    ) -> dict[str, Any]:
        """Tłumaczy tekst.

        Args:
            user_id: ID użytkownika
            text: Tekst do tłumaczenia
            target_lang: Język docelowy
            source_lang: Język źródłowy
            provider: Provider tłumaczeń

        Returns:
            Przetłumaczony tekst
        """
        if provider == "google":
            # Google Translate API (wymaga klucza)
            url = "https://translation.googleapis.com/language/translate/v2"
            # To wymaga właściwej autoryzacji Google Cloud
            return {
                "error": "Google Translate API requires proper setup",
                "status": 501,
            }

        else:
            return {
                "error": f"Unsupported translation provider: {provider}",
                "status": 400,
            }

    async def get_news(
        self,
        user_id: int,
        api_key: str,
        query: str | None = None,
        category: str | None = None,
        country: str = "pl",
    ) -> dict[str, Any]:
        """Pobiera wiadomości.

        Args:
            user_id: ID użytkownika
            api_key: Klucz API NewsAPI
            query: Zapytanie
            category: Kategoria wiadomości
            country: Kod kraju

        Returns:
            Wiadomości
        """
        if query:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": api_key,
                "language": "pl",
                "sortBy": "publishedAt",
                "pageSize": 20,
            }
        else:
            url = "https://newsapi.org/v2/top-headlines"
            params = {"apiKey": api_key, "country": country, "pageSize": 20}
            if category:
                params["category"] = category

        return await self.get(user_id, url, params=params)

    async def get_usage_stats(self, user_id: int, days: int = 30) -> dict[str, Any]:
        """Pobiera statystyki użycia API dla użytkownika.

        Args:
            user_id: ID użytkownika
            days: Liczba dni wstecz

        Returns:
            Statystyki użycia
        """
        usage_records = self.db_manager.get_user_api_usage(user_id, days)

        stats = {
            "total_requests": len(usage_records),
            "successful_requests": sum(1 for r in usage_records if r.success),
            "failed_requests": sum(1 for r in usage_records if not r.success),
            "total_tokens": sum(r.tokens_used for r in usage_records),
            "total_cost": sum(r.cost for r in usage_records),
            "providers": {},
            "daily_usage": {},
        }

        # Grupuj po providerach
        for record in usage_records:
            provider = record.api_provider
            if provider not in stats["providers"]:
                stats["providers"][provider] = {"requests": 0, "tokens": 0, "cost": 0.0}

            stats["providers"][provider]["requests"] += 1
            stats["providers"][provider]["tokens"] += record.tokens_used
            stats["providers"][provider]["cost"] += record.cost

            # Grupuj po dniach
            day = record.created_at.strftime("%Y-%m-%d")
            if day not in stats["daily_usage"]:
                stats["daily_usage"][day] = {"requests": 0, "tokens": 0, "cost": 0.0}

            stats["daily_usage"][day]["requests"] += 1
            stats["daily_usage"][day]["tokens"] += record.tokens_used
            stats["daily_usage"][day]["cost"] += record.cost

        return stats


# Globalna instancja
_api_module = None


async def get_api_module() -> APIModule:
    """Pobiera globalną instancję modułu API."""
    global _api_module
    if _api_module is None:
        _api_module = APIModule()
        await _api_module.initialize()
    return _api_module


async def cleanup_api_module():
    """Czyści globalną instancję modułu API."""
    global _api_module
    if _api_module:
        await _api_module.cleanup()
        _api_module = None
