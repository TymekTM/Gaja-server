import logging
import re
from datetime import datetime
from typing import Any, Optional

import aiohttp
from .api_module import get_api_module

logger = logging.getLogger(__name__)


# Plugin metadata (new default search plugin)
PLUGIN_NAME = "search_module"
PLUGIN_DESCRIPTION = "Web search + scraping to provide AI-ready context (universal)"
PLUGIN_VERSION = "2.1.0"
PLUGIN_AUTHOR = "GAJA Team"
PLUGIN_DEPENDENCIES: list[str] = []  # Keep deps light; Scrapling used lazily


def get_functions() -> list[dict[str, Any]]:
    """Deklaracja funkcji publicznych pluginu."""
    return [
        {
            "name": "search",
            "description": (
                "Wyszukuje w internecie, pobiera strony, wyciąga kluczowe treści i zwraca AI-"
                "gotowy kontekst (Scrapling + fallbacki)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Zapytanie do wyszukania"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników do pobrania",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "fetch_pages": {
                        "type": "boolean",
                        "description": "Czy pobierać i streszczać treści ze stron",
                        "default": True,
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Tryb testowy (zwraca mocki bez sieci)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        }
    ]


async def search_search(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Główny punkt wejścia modułu zgodny z interfejsem pluginów."""
    mod = SearchModule()
    await mod.initialize()

    try:
        return await mod.execute_function(function_name, parameters, user_id)
    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        return {"success": False, "error": str(e)}
    finally:
        await mod.cleanup()


class SearchModule:
    """Nowy moduł wyszukiwania: biblioteka duckduckgo-search + Scrapling + fallbacki.

    Główna ścieżka:
      1) Próba użycia 'duckduckgo_search' (jeśli dostępne).
      2) Fallback: DDG Instant Answer API (bez klucza).
      3) Fallback: HTML scraping DDG wyników.
      4) Fallback: Wikipedia (opensearch + summary REST API).

    Następnie: pobranie i streszczenie treści stron przez Scrapling (jeśli fetch_pages=True).
    """

    SEARCH_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,*;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive'
    }

    def __init__(self) -> None:
        self.api_module = None

    async def initialize(self) -> None:
        try:
            from .api_module import get_api_module
            self.api_module = await get_api_module()
            logger.info("SearchModule v2 initialized successfully")
        except Exception as e:
            logger.warning(f"SearchModule initialization failed, will use fallback mode: {e}")
            self.api_module = None

    async def search(
        self,
        user_id: int,
        query: str,
        max_results: int = 10,
        fetch_pages: bool = True,
    ) -> dict[str, Any]:
        if not self.api_module:
            await self.initialize()

        # Initialize variables
        engine_used = None
        items: list[dict[str, Any]] = []
        
        # For certain queries, try Wikipedia first (language-neutral)
        if self._should_try_wikipedia_first(query):
            wiki_pack = await self._fallback_wikipedia(user_id, query)
            if wiki_pack and wiki_pack.get("results"):
                logger.debug(f"Using Wikipedia results for query: {query}")
                engine_used = "wikipedia"
                for w in wiki_pack.get("results", [])[:max_results]:
                    items.append({
                        "title": w.get("title", "Wikipedia"),
                        "url": w.get("url"),
                        "snippet": w.get("snippet") or w.get("extract") or "",
                        "type": "web_result",
                    })

        # Detect "current office" intent (e.g., current president of X) and prepare targeted candidates
        instant_answer: Optional[str] = None
        office_intent = self._detect_current_office_query(query)
        if not items and office_intent:
            # 1) Try direct Wikipedia titles for the office page (to get reliable context)
            title_candidates = self._wikipedia_title_candidates(office_intent)
            for title, lang in title_candidates:
                try:
                    summ = await self._wikipedia_summary(user_id, lang, title)
                except Exception:
                    summ = None
                if summ:
                    engine_used = "wikipedia"
                    items.append({
                        "title": summ.get("title", title),
                        "url": summ.get("url"),
                        "snippet": summ.get("extract") or "",
                        "type": "wiki_summary",
                    })
                    # Do not break; collect a couple of good items
                    if len(items) >= max_results:
                        break
            # 2) Try to extract the incumbent name via DDG person page
            if not instant_answer:
                name_guess, person_url, person_snippet = await self._answer_current_office_via_ddg(user_id, office_intent)
                if name_guess:
                    instant_answer = name_guess
                    if person_url:
                        items.insert(0, {"title": name_guess, "url": person_url, "snippet": person_snippet or "", "type": "person"})

        # Ensure at least one synthetic item if we have an instant answer but no sources yet
        if instant_answer and not items:
            items.append({
                "title": instant_answer,
                "url": "",
                "snippet": "",
                "type": "instant_answer"
            })

        # 1) Try duckduckgo_search library (optional dependency) if no Wikipedia results
        if not items:
            engine_used = None
            try:
                from duckduckgo_search import DDGS  # type: ignore

                try:
                    with DDGS() as ddgs:
                        # Select region based on query language for better relevance
                        lang = self._detect_language(query)
                        region = "pl-pl" if lang == "pl" else ("us-en" if lang == "en" else "wt-wt")
                        q_clean = self._normalize_query(query)
                        try:
                            results = list(
                                ddgs.text(q_clean, region=region, safesearch="moderate", max_results=max_results * 2)
                            )
                        except TypeError:
                            results = list(ddgs.text(q_clean, max_results=max_results * 2))
                except Exception:
                    results = []

                # Convert and filter results
                raw_items = []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    url = r.get("href") or r.get("link") or r.get("url")
                    title = r.get("title") or r.get("name") or ""
                    snippet = r.get("body") or r.get("snippet") or r.get("text") or ""
                    if url and title:
                        raw_items.append({"title": title, "url": url, "snippet": snippet, "type": "web_result"})
                
                # Filter and score results
                items = self._filter_and_score_results(raw_items, query)
                if items:
                    engine_used = "duckduckgo_search_lib"
            except Exception as ddg_import_err:
                logger.debug(f"duckduckgo_search not available or failed: {ddg_import_err}")

        # 2) DuckDuckGo Instant Answer API if library path empty
        if not items:
            ddg_api = await self._search_duckduckgo_api(user_id, query, max_results)
            if isinstance(ddg_api, dict) and (ddg_api.get("results") or ddg_api.get("instant_answer")):
                engine_used = ddg_api.get("engine", "duckduckgo")
                items = self._filter_and_score_results(ddg_api.get("results", []) or [], query)

        # 3) HTML fallback (DDG results page)
        if not items:
            html_fb = await self._ddg_html_search_fallback(user_id, query, max_results)
            if html_fb and html_fb.get("results"):
                engine_used = html_fb.get("engine", "duckduckgo_html")
                items = self._filter_and_score_results(html_fb.get("results", []) or [], query)

        # 4) Wikipedia fallback (useful for definitions)
        wiki_pack = None
        if not items:
            wiki_pack = await self._fallback_wikipedia(user_id, query)
            if wiki_pack and wiki_pack.get("results"):
                engine_used = "wikipedia"
                # synthesize items as URL-bearing results
                witems = []
                for w in wiki_pack.get("results", [])[:max_results]:
                    witems.append({
                        "title": w.get("title", "Wikipedia"),
                        "url": w.get("url"),
                        "snippet": w.get("snippet") or w.get("extract") or "",
                        "type": "web_result",
                    })
                items = witems

        # If still nothing, but we have an instant answer, proceed with it
        if not items and instant_answer:
            items = [{
                "title": instant_answer,
                "url": "",
                "snippet": "",
                "type": "instant_answer"
            }]

        # If still nothing, return a structured error
        if not items:
            return {"error": "Brak wyników wyszukiwania", "engine": engine_used or "unknown", "query": query}

        # Deduplicate and trim to max_results
        seen = set()
        unique_items: list[dict[str, Any]] = []
        for it in items:
            url = it.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            unique_items.append(it)
            if len(unique_items) >= max_results:
                break

        # Optionally fetch and summarize pages
        summaries: list[dict[str, Any]] = []
        context_chunks: list[dict[str, Any]] = []
        if fetch_pages:
            summaries = await self._summarize_with_scrapling(user_id, unique_items, max_results, query=query)
            for s in summaries:
                context_chunks.append(
                    {
                        "title": s.get("title") or "",
                        "content": s.get("summary") or "",
                        "source_url": s.get("source_url"),
                    }
                )
        else:
            for it in unique_items:
                context_chunks.append(
                    {
                        "title": it.get("title") or "",
                        "content": it.get("snippet") or "",
                        "source_url": it.get("url"),
                    }
                )

        # Create comprehensive context text with key information for AI
        key_info = []
        for i, c in enumerate(context_chunks[:5]):  # Top 5 most relevant for better AI context
            content = c.get('content', '').strip()
            title = c.get('title', '').strip()
            url = c.get('source_url', '').strip()
            
            # Filter out meaningless content
            if content and len(content) > 10:
                # Add generic source credibility indicators
                source_type = ""
                url_l = url.lower()
                if "wikipedia.org" in url_l:
                    source_type = "[Wikipedia] "
                elif url_l.endswith(".gov") or "/gov/" in url_l or ".gov/" in url_l:
                    source_type = "[Official] "
                elif any(d in url_l for d in [
                    'reuters.com','bbc.com','apnews.com','cnn.com','nytimes.com','theguardian.com',
                    'aljazeera.com','wsj.com','bloomberg.com']):
                    source_type = "[News] "
                
                if title and title.lower() not in content.lower():
                    key_info.append(f"{source_type}{title}: {content}")
                else:
                    key_info.append(f"{source_type}{content}")

        # Fallback: if no content-based key info, synthesize from titles/urls
        if not key_info and unique_items:
            synthesized = []
            for it in unique_items[:3]:
                t = (it.get('title') or '').strip()
                u = (it.get('url') or '').strip()
                if t or u:
                    synthesized.append(f"{t} — {u}")
            if synthesized:
                key_info = ["Sources:"] + synthesized

        # If we have an instant answer (e.g., current office holder), surface it first
        if 'instant_answer' in locals() and instant_answer and (not key_info or instant_answer not in "\n".join(key_info)):
            key_info.insert(0, f"Answer: {instant_answer}")
        
        # Add current date context
        context_text = ""
        if key_info:
            now = datetime.now()
            month_year = now.strftime("%B %Y")
            context_text = f"Key information (as of {month_year}):\n\n" + "\n\n".join(key_info)
        else:
            context_text = "Brak trafnych wyników wyszukiwania."

        # Return simplified response with only essential data
        response = {
            "query": query,
            "engine": engine_used or "duckduckgo",
            "key_info": context_text,
            # Maintain compatibility: return both 'sources' (top 3) and 'results' (up to max_results)
            "results": [
                {
                    "title": item.get('title', ''),
                    "url": item.get('url', ''),
                    "snippet": item.get('snippet', '')
                }
                for item in unique_items  # full deduped list up to max_results
            ],
            "sources": [
                {
                    "title": item.get('title', ''),
                    "url": item.get('url', ''),
                    "snippet": item.get('snippet', '')[:200]  # Limit snippet length
                }
                for item in unique_items[:3]  # Only top 3 sources
            ],
            "timestamp": datetime.now().isoformat()
        }
        if instant_answer:
            response["instant_answer"] = instant_answer
        
        # Add detailed data only if requested or for debugging
        if fetch_pages:
            response["detailed_summaries"] = [s for s in summaries if s.get('summary') and len(s.get('summary', '')) > 10]
        
        return response

    async def execute_function(self, function_name: str, parameters: dict[str, Any], user_id: int) -> dict[str, Any]:
        """Instance-level dispatcher for FunctionCallingSystem compatibility."""
        try:
            if function_name != "search":
                return {"success": False, "error": f"Unknown function: {function_name}"}

            query = (
                parameters.get("query")
                or parameters.get("q")
                or parameters.get("text")
                or parameters.get("topic")
            )
            if not query or not isinstance(query, str) or not query.strip():
                return {"success": False, "error": "Brak zapytania (query)"}

            max_results = parameters.get("max_results", 10)
            try:
                max_results_int = int(max_results)
            except Exception:
                max_results_int = 10
            if max_results_int < 1:
                max_results_int = 1
            if max_results_int > 50:
                max_results_int = 50

            fetch_pages = bool(parameters.get("fetch_pages", True))
            test_mode = bool(parameters.get("test_mode", False))

            if test_mode:
                data = self._get_mock_search_data(query, max_results_int)
            else:
                data = await self.search(user_id, query, max_results=max_results_int, fetch_pages=fetch_pages)

            success = False
            if isinstance(data, dict) and not data.get("error"):
                # Check for any meaningful content
                key_info = data.get("key_info")
                if key_info and isinstance(key_info, str) and key_info.strip():
                    success = True
                elif data.get("sources") and len(data.get("sources", [])) > 0:
                    success = True  
                elif data.get("context") or data.get("summaries"):
                    success = True
                elif data.get("results") and len(data.get("results", [])) > 0:
                    success = True
                elif data.get("instant_answer"):
                    success = True

            # Expose test_mode flag for tests
            return {"success": success, "data": data, "test_mode": test_mode, "message": f"Zebrano kontekst dla: {query}"}
        except Exception as e:
            logger.error(f"Error in SearchModule.execute_function: {e}")
            return {"success": False, "error": str(e)}

    async def _search_duckduckgo_api(self, user_id: int, query: str, max_results: int) -> dict[str, Any]:
        """DuckDuckGo Instant Answer API minimal adapter."""
        url = "https://api.duckduckgo.com/"
        q_clean = self._normalize_query(query)
        params = {"q": q_clean or query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        
        # Try API module first, then direct HTTP
        resp_data = None
        if self.api_module:
            try:
                resp = await self.api_module.get(user_id, url, params=params)
                if resp.get("status") == 200:
                    resp_data = resp.get("data")
            except Exception as e:
                logger.warning(f"DuckDuckGo API call via api_module failed: {e}")
        
        # Direct HTTP fallback
        if not resp_data:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(headers=self.SEARCH_HEADERS, timeout=timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            resp_data = await response.json()
            except Exception as e:
                logger.warning(f"Direct DuckDuckGo API call failed: {e}")
                return {"error": f"DDG API error: {e}", "engine": "duckduckgo"}
        
        if not resp_data:
            return {"error": "DDG API error", "engine": "duckduckgo"}
        
        data = resp_data or {}
        results: list[dict[str, Any]] = []

        if isinstance(data, dict):
            if data.get("Abstract"):
                results.append(
                    {
                        "title": data.get("Heading", "Informacja"),
                        "snippet": data.get("Abstract", ""),
                        "url": data.get("AbstractURL", ""),
                        "type": "abstract",
                    }
                )
            for topic in (data.get("RelatedTopics") or [])[:max_results]:
                if isinstance(topic, dict) and (topic.get("FirstURL") or topic.get("Icon") or topic.get("Text")):
                    results.append(
                        {
                            "title": (topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", "Powiązany temat")),
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                            "type": "related_topic",
                        }
                    )
        return {"engine": "duckduckgo", "query": query, "results": results}

    async def _ddg_html_search_fallback(self, user_id: int, query: str, max_results: int) -> Optional[dict[str, Any]]:
        """Fallback: prosty parser wyników HTML DuckDuckGo (bez JS)."""
        try:
            import urllib.parse as _u

            q_clean = self._normalize_query(query)
            url = f"https://duckduckgo.com/html/?q={_u.quote(q_clean or query)}&kl=wt-wt&kp=1"
            html = None
            
            # Try API module first, then direct HTTP
            if self.api_module:
                try:
                    resp = await self.api_module.get(user_id, url)
                    if resp.get("status") == 200:
                        html = resp.get("data")
                except Exception as e:
                    logger.warning(f"DuckDuckGo HTML search via api_module failed: {e}")
            
            # Direct HTTP fallback
            if not html:
                try:
                    timeout = aiohttp.ClientTimeout(total=15)
                    async with aiohttp.ClientSession(headers=self.SEARCH_HEADERS, timeout=timeout) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                html = await response.text()
                except Exception as e:
                    logger.warning(f"Direct DuckDuckGo HTML search failed: {e}")
                    return None
            
            if not html:
                return None
            # Bardzo uproszczone regexy do wyciągania wyników.
            # Szukamy <a class="result__a" href="...">Tytuł</a>
            links = re.findall(r'<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>', html, re.I)
            snippets = re.findall(r'<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>', html, re.I)

            out = []
            for i, (href, title_html) in enumerate(links[:max_results]):
                title = re.sub(r"<[^>]+>", "", title_html).strip()
                out.append({
                    "title": title,
                    "snippet": re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else "",
                    "url": href,
                    "type": "web_result",
                })
            if out:
                return {"engine": "duckduckgo_html", "query": query, "results": out}
        except Exception:
            return None
        return None

    async def _summarize_with_scrapling(
        self, user_id: int, items: list[dict[str, Any]], max_results: int, query: str = ""
    ) -> list[dict[str, Any]]:
        selector_cls = None
        try:
            from scrapling.parser import Selector as _Selector  # type: ignore
            selector_cls = _Selector
        except Exception as e:
            logger.debug(f"Scrapling not available, fallback to raw text clean: {e}")
            selector_cls = None

        summaries: list[dict[str, Any]] = []
        for it in items[:max_results]:
            url = it.get("url")
            title_hint = it.get("title") or ""
            if not url:
                continue
            
            try:
                # Try API module first, then direct HTTP
                html = None
                status = None
                
                if self.api_module:
                    try:
                        resp = await self.api_module.get(user_id, url)
                        status = resp.get("status")
                        if status and int(status) < 400:
                            html = resp.get("data") or ""
                    except Exception as e:
                        logger.debug(f"API module failed for {url}: {e}")
                
                # Direct HTTP fallback
                if not html:
                    try:
                        timeout = aiohttp.ClientTimeout(total=15)
                        async with aiohttp.ClientSession(headers=self.SEARCH_HEADERS, timeout=timeout) as session:
                            async with session.get(url) as response:
                                if response.status < 400:
                                    html = await response.text()
                                    status = response.status
                                else:
                                    status = response.status
                    except Exception as e:
                        logger.debug(f"Direct HTTP failed for {url}: {e}")
                        status = "error"
                
                if not html or (status and int(status) >= 400):
                    summaries.append(
                        {
                            "title": title_hint,
                            "summary": f"Nie udało się pobrać strony (status: {status})",
                            "source_url": url,
                            "status": f"http_{status}" if status else "error",
                        }
                    )
                    continue
                if not isinstance(html, str) or len(html) < 50:  # Lowered threshold
                    summaries.append(
                        {
                            "title": title_hint,
                            "summary": it.get("snippet", "Brak treści lub strona pusta"),  # Use snippet as fallback
                            "source_url": url,
                            "status": "limited_content",
                        }
                    )
                    continue

                page = None
                if selector_cls is not None:
                    try:
                        page = selector_cls(html)
                    except Exception as e:
                        logger.debug(f"Scrapling parsing failed for {url}: {e}")
                        page = None

                title = self._extract_title(page) or title_hint
                text = self._extract_main_text(page) if page else self._clean_text(html)
                
                # Enhanced summarization - more context for AI
                if len(text) > 100:
                    summary = self._enhanced_summarize(text, query=query)  # Use actual query
                else:
                    summary = it.get("snippet", text) or "Brak wystarczającej treści"

                summaries.append(
                    {"title": title, "summary": summary, "source_url": url, "status": "ok" if text else "limited"}
                )
            except Exception as e:
                logger.debug(f"Summarization failed for {url}: {e}")
                summaries.append(
                    {
                        "title": title_hint,
                        "summary": "Błąd podczas pobierania lub przetwarzania strony.",
                        "source_url": url,
                        "status": "error",
                    }
                )
        return summaries

    def _extract_title(self, page: Any) -> Optional[str]:
        try:
            if not page:
                return None
            h1 = page.css_first("article h1::text") or page.css_first("main h1::text")
            if h1:
                return str(h1)
            t = None
            try:
                t = page.css_first("title::text")
                if t:
                    return str(t).strip()
            except Exception:
                pass
            h1 = page.css_first("h1::text")
            if h1:
                return str(h1)
        except Exception:
            return None
        return None

    def _extract_main_text(self, page: Any) -> str:
        try:
            candidates = [
                "article",
                "main",
                "[role=main]",
                "div[itemprop=articleBody]",
                ".post-content",
                ".entry-content",
                ".content",
                ".article",
                ".post",
                ".story",
                ".markdown-body",
                "#content",
                "#main",
                "body",
            ]
            best_text = ""
            for sel in candidates:
                try:
                    el = page.css_first(sel)
                    if not el:
                        continue
                    txt = el.text if hasattr(el, "text") else str(el)
                    txt = self._clean_text(txt)
                    if len(txt) > len(best_text):
                        best_text = txt
                except Exception:
                    continue
            return best_text
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        import re as _re
        return _re.sub(r"\s+", " ", text or "").strip()

    def _enhanced_summarize(self, text: str, query: str = "", max_sentences: int = 8, max_chars: int = 1200) -> str:
        """Enhanced summarization with query context for better AI understanding."""
        if not text:
            return ""
        
        import re as _re
        
        # Clean and normalize text
        text = _re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = _re.split(r'(?<=[\.!?])\s+', text)
        
        # Score sentences based on query relevance
        scored_sentences = []
        query_words = set(query.lower().split()) if query else set()
        
        for sentence in sentences:
            if not sentence.strip() or len(sentence) < 10:
                continue
                
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on query word matches
            for word in query_words:
                if len(word) > 2 and word in sentence_lower:
                    score += 2
            
            # Boost sentences with generic civic/identity terms (language-agnostic)
            key_terms = [
                'president', 'prime minister', 'government', 'parliament', 'election', 'minister',
                'biography', 'defined', 'definition'
            ]
            for term in key_terms:
                if term in sentence_lower:
                    score += 1
            
            # Boost sentences with dates (current context)
            if _re.search(r'202[0-9]', sentence):
                score += 3
                
            # Boost sentences with names (proper nouns)
            if _re.search(r'[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+ [A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+', sentence):
                score += 1
                
            scored_sentences.append((score, sentence))
        
        # Sort by score and select best sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        summary_parts = []
        total_chars = 0
        
        for score, sentence in scored_sentences:
            if len(summary_parts) >= max_sentences:
                break
            if total_chars + len(sentence) > max_chars and summary_parts:
                break
            
            summary_parts.append(sentence)
            total_chars += len(sentence)
        
        if not summary_parts:
            return text[:max_chars]
        
        return " ".join(summary_parts)

    def _simple_summarize(self, text: str, max_sentences: int = 5, max_chars: int = 800) -> str:
        if not text:
            return ""
        import re as _re
        parts = _re.split(r"(?<=[\.!?])\s+", text)
        summary_parts = []
        total = 0
        for p in parts:
            if not p:
                continue
            if len(summary_parts) >= max_sentences:
                break
            if total + len(p) > max_chars and summary_parts:
                break
            summary_parts.append(p)
            total += len(p)
        if not summary_parts:
            return text[:max_chars]
        return " ".join(summary_parts)

    

    def _should_try_wikipedia_first(self, query: str) -> bool:
        """Check if query should prefer Wikipedia results."""
        query_lower = query.lower().strip()
        
        # Wikipedia-first patterns
        wikipedia_patterns = [
            'kto jest',
            'who is',
            'prezydent',
            'president',
            'minister',
            'premier',
            'prime minister',
            'co to jest',
            'what is',
            'definicja',
            'definition',
            'biografia',
            'biography'
        ]
        
        return any(pattern in query_lower for pattern in wikipedia_patterns)

    def _detect_language(self, text: str) -> str:
        """Very light heuristic language detection (pl/en)."""
        t = (text or "").lower()
        if any(ch in t for ch in "ąćęłńóśźż"):
            return "pl"
        if any(w in t for w in [" kto ", " jest ", " prezydent", " polsk", " rp", " aktualn", "premier"]):
            return "pl"
        if any(w in t for w in [" who ", " president", " current", " prime minister", " usa", " uk", " poland"]):
            return "en"
        return "en"

    def _detect_current_office_query(self, query: str) -> Optional[dict[str, str]]:
        q = (query or "").lower()
        data: dict[str, str] = {}
        # Polish
        if "prezydent" in q or "president" in q:
            data["office"] = "president"
        elif "premier" in q or "prime minister" in q:
            data["office"] = "prime_minister"
        else:
            return None
        # Subject/country
        if any(s in q for s in [" polsk", " rp", " rzeczpospolitej"]):
            data["subject"] = "poland"
        elif "usa" in q or "stany" in q or " united states" in q:
            data["subject"] = "united states"
        elif "uk" in q or "united kingdom" in q or "wielkiej brytanii" in q:
            data["subject"] = "united kingdom"
        elif "india" in q or "indie" in q:
            data["subject"] = "india"
        else:
            # Try extract last token after "of" or simple country word
            # keep generic if no clear subject
            data["subject"] = ""
        data["lang"] = self._detect_language(query)
        return data

    def _normalize_query(self, query: str) -> str:
        """Normalize query by removing filler words like 'current/aktualny' and years."""
        import re as _re
        q = (query or "").strip()
        # Remove year-like tokens
        q = _re.sub(r"\b(19|20)\d{2}\b", "", q)
        # Remove common filler phrases (pl/en)
        patterns = [
            r"\bkto jest\b",
            r"\bkim jest\b",
            r"\baktualn\w*\b",
            r"\bobecny\w*\b",
            r"\bteraz\b",
            r"\bcurrent\b",
            r"\bwho is\b",
            r"\bthe\b",
        ]
        for p in patterns:
            q = _re.sub(p, " ", q, flags=_re.I)
        # Collapse whitespace
        q = _re.sub(r"\s+", " ", q)
        return q.strip()

    def _wikipedia_title_candidates(self, intent: dict[str, str]) -> list[tuple[str, str]]:
        """Generate Wikipedia title candidates (title, lang) for office page."""
        office = intent.get("office", "")
        subject = intent.get("subject", "")
        lang = intent.get("lang", "en")
        candidates: list[tuple[str, str]] = []
        if office == "president" and subject in ("poland", ""):
            candidates.append(("President of Poland", "en"))
            candidates.append(("Prezydent Rzeczypospolitej Polskiej", "pl"))
        elif office == "prime_minister" and subject in ("poland", ""):
            candidates.append(("Prime Minister of Poland", "en"))
            candidates.append(("Prezes Rady Ministrów", "pl"))
        # United States
        if office == "president" and subject == "united states":
            candidates = [("President of the United States", "en")]
        if office == "prime_minister" and subject == "united kingdom":
            candidates = [("Prime Minister of the United Kingdom", "en")]
        # India
        if office == "president" and subject == "india":
            candidates = [("President of India", "en")]
        # Ensure language preferred candidates first
        # Stable order: move preferred language to front where possible
        if lang:
            preferred = [(t, l) for (t, l) in candidates if l == lang]
            others = [(t, l) for (t, l) in candidates if l != lang]
            candidates = preferred + others
        return candidates

    async def _answer_current_office_via_ddg(self, user_id: int, intent: dict[str, str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Try to guess incumbent's name using DDG results (Wikipedia person page)."""
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return (None, None, None)

        office = intent.get("office", "")
        subject = intent.get("subject", "")
        lang = intent.get("lang", "en")
        # Build queries
        if lang == "pl":
            base_q = "prezydent polski" if office == "president" else "premier polski"
        else:
            if office == "president":
                base_q = f"president of {subject or 'poland'}"
            else:
                base_q = f"prime minister of {subject or 'poland'}"
        queries = [base_q + " site:wikipedia.org", base_q + " wikipedia", base_q + " incumbent site:wikipedia.org"]
        region = "pl-pl" if lang == "pl" else "us-en"

        name_guess = None
        url_guess = None
        snippet_guess = None
        try:
            with DDGS() as ddgs:
                for q in queries:
                    try:
                        results = list(ddgs.text(q, region=region, safesearch="moderate", max_results=5))
                    except TypeError:
                        results = list(ddgs.text(q, max_results=5))
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        url = r.get("href") or r.get("link") or r.get("url") or ""
                        title = r.get("title") or r.get("name") or ""
                        snippet = r.get("body") or r.get("snippet") or r.get("text") or ""
                        if "wikipedia.org/wiki/" in (url or ""):
                            # try parse "Name – Wikipedia"
                            name = title.split(" – ")[0].split(" - ")[0].strip()
                            # Heuristic: looks like a person if it has at least 2 words with capital initials
                            if name and len(name.split()) >= 2 and name[0].isupper():
                                name_guess = name
                                url_guess = url
                                snippet_guess = snippet
                                return (name_guess, url_guess, snippet_guess)
        except Exception:
            return (None, None, None)
        return (name_guess, url_guess, snippet_guess)
    
    def _filter_and_score_results(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Filter and score results based on relevance and quality."""
        if not results or not query:
            return results
        
        query_lower = query.lower().strip()
        scored_results = []
        
        for result in results:
            title = (result.get('title') or '').lower()
            snippet = (result.get('snippet') or '').lower()
            url = (result.get('url') or '').lower()
            
            # Skip clearly irrelevant results
            if self._is_irrelevant_result(title, snippet, url, query_lower):
                continue
            
            score = 0
            
            # General quality indicators
            if 'wikipedia' in url:
                score += 40
            if any(domain in url for domain in ['gov.', '.gov/', '.edu', '.org/']):
                score += 20

            # Strongly boost office queries for relevant keywords
            if any(k in query_lower for k in ['prezydent', 'president', 'premier', 'prime minister']):
                if any(k in (title + ' ' + snippet) for k in ['prezydent', 'president', 'premier', 'prime minister']):
                    score += 40
                # Penalize dictionary/time/synonym sites for such queries
                if any(d in url for d in ['synonim.net', 'sjp.pwn.pl', 'wiktionary.org', 'time.is']):
                    score -= 60
            
            # Penalize low-quality indicators
            if any(word in title + snippet for word in ['error', '404', 'not found', 'bonjour']):
                score -= 20
            if len(snippet.strip()) < 20:
                score -= 10
                
            # Check query word presence
            query_words = query_lower.split()
            title_snippet = title + ' ' + snippet
            matching_words = sum(1 for word in query_words if word in title_snippet)
            score += matching_words * 5
            
            if score > -10:  # Only keep results with reasonable score
                result['_score'] = score
                scored_results.append(result)
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x.get('_score', 0), reverse=True)
        
        # Clean up score field before returning
        for result in scored_results:
            result.pop('_score', None)
            
        return scored_results[:8]  # Return max 8 best results
    
    def _is_irrelevant_result(self, title: str, snippet: str, url: str, query: str) -> bool:
        """Check if a result is clearly irrelevant to the query."""
        # Skip results with suspicious patterns
        irrelevant_patterns = [
            'infirmiers.com',  # Medical forums
            'zhidao.baidu.com',  # Chinese Q&A sites
            'baidu.com',
            'zhihu.com',
            'weibo.com',
            'qq.com',
            'bilibili.com',
            'error',
            '404',
            'not found',
            'page not found'
        ]
        
        content = title + ' ' + snippet + ' ' + url
        
        for pattern in irrelevant_patterns:
            if pattern in content:
                return True
                
        # If query is about an office holder, drop dictionaries/synonyms/time pages
        if any(w in query for w in ['prezydent', 'president', 'premier', 'prime minister']):
            for bad in ['synonim.net', 'sjp.pwn.pl', 'wiktionary.org', 'time.is']:
                if bad in url:
                    return True
        
        # Drop pages clearly in CJK scripts for non-CJK queries
        import re as _re
        if not any(w in query for w in ['中国', '中文', '日本', '日本語', '한국', '한국어']):
            if _re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', title + ' ' + snippet):
                return True
        # Do not apply country-specific filtering; keep universal
        
        return False
    
    def extract_search_intent(self, query: str) -> dict[str, Any]:
        """Prosta analiza intencji zapytania (kompatybilność z poprzednią wersją)."""
        q = (query or "").lower()
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
            "keywords": q.split(),
            "suggested_engine": "duckduckgo",
            "filters": [],
        }

        import re as _re
        for itype, pat in patterns.items():
            if _re.search(pat, q):
                intent["type"] = itype
                break

        if intent["type"] == "news":
            intent["suggested_engine"] = "google"
        elif intent["type"] in ("definition", "how_to"):
            intent["suggested_engine"] = "duckduckgo"

        if "ostatni" in q or "najnowszy" in q:
            intent["filters"].append("recent")
        if any(w in q for w in ["polski", "polska", "polskie"]):
            intent["filters"].append("polish")

        return intent

    async def _wikipedia_opensearch(self, user_id: int, lang: str, query: str) -> Optional[str]:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        # Use normalized query to improve matching
        q_norm = self._normalize_query(query) if hasattr(self, '_normalize_query') else query
        params = {"action": "opensearch", "search": q_norm or query, "limit": 1, "namespace": 0, "format": "json"}
        headers = {
            "User-Agent": "GAJA-Assistant/1.0 (contact: admin@example.com)",
            "Accept-Language": f"{lang},en;q=0.8,*;q=0.6",
        }
        # Try via API module first
        if self.api_module:
            try:
                resp = await self.api_module.get(user_id, url, params=params, headers=headers)
                if resp.get("status") == 200:
                    data = resp.get("data")
                    if isinstance(data, list) and len(data) >= 2 and data[1]:
                        return data[1][0]
            except Exception as e:
                logger.warning(f"Wikipedia opensearch via API module failed: {e}")
        # Direct HTTP fallback
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, params=params) as r:
                    if r.status == 200:
                        data = await r.json()
                        if isinstance(data, list) and len(data) >= 2 and data[1]:
                            return data[1][0]
        except Exception as e:
            logger.debug(f"Wikipedia opensearch direct HTTP failed: {e}")
        return None

    async def _wikipedia_summary(self, user_id: int, lang: str, title: str) -> Optional[dict[str, Any]]:
        import urllib.parse as _u
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{_u.quote(title)}"
        headers = {
            "User-Agent": "GAJA-Assistant/1.0 (contact: admin@example.com)",
            "Accept-Language": f"{lang},en;q=0.8,*;q=0.6",
        }
        # Try via API module first
        if self.api_module:
            try:
                resp = await self.api_module.get(user_id, url, headers=headers)
                if resp.get("status") == 200:
                    data = resp.get("data") or {}
                    extract = data.get("extract") or data.get("description") or ""
                    page_url = (
                        ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
                        or data.get("canonicalurl")
                        or data.get("pageid")
                    )
                    return {"title": data.get("title", title), "extract": extract, "url": page_url}
            except Exception as e:
                logger.warning(f"Wikipedia summary via API module failed: {e}")
        # Direct HTTP fallback
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as r:
                    if r.status == 200:
                        data = await r.json()
                        extract = data.get("extract") or data.get("description") or ""
                        page_url = (
                            ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
                            or data.get("canonicalurl")
                            or data.get("pageid")
                        )
                        return {"title": data.get("title", title), "extract": extract, "url": page_url}
        except Exception as e:
            logger.debug(f"Wikipedia summary direct HTTP failed: {e}")
        return None

    async def _fallback_wikipedia(self, user_id: int, query: str) -> Optional[dict[str, Any]]:
        # Prefer English first for universality, then try Polish
        lang_order = ["en", "pl"]
        for lang in lang_order:
            try:
                title = await self._wikipedia_opensearch(user_id, lang, query)
                if not title:
                    continue
                summ = await self._wikipedia_summary(user_id, lang, title)
                if not summ:
                    continue
                return {
                    "engine": "wikipedia",
                    "query": query,
                    "results": [
                        {
                            "title": summ.get("title", title),
                            "snippet": summ.get("extract", ""),
                            "url": summ.get("url"),
                            "type": "wiki_summary",
                        }
                    ],
                    "instant_answer": {
                        "answer": summ.get("extract", ""),
                        "source": "wikipedia",
                        "url": summ.get("url"),
                    },
                }
            except Exception:
                continue
        return None

    async def cleanup(self) -> None:
        # Nothing persistent to cleanup here; API module is shared
        logger.debug("SearchModule v2 cleanup completed")

    def _get_mock_search_data(self, query: str, max_results: int) -> dict[str, Any]:
        results = []
        for i in range(1, max_results + 1):
            results.append(
                {
                    "title": f'Wynik {i} dla "{query}"',
                    "url": f"https://example.com/{i}",
                    "snippet": f"Przykładowy opis wyniku {i} dla zapytania '{query}'.",
                    "type": "web_result",
                }
            )
        # Syntetyczne streszczenia bez pobierania stron
        summaries = [
            {
                "title": r["title"],
                "summary": f"Skrót najważniejszych informacji z: {r['title']}.",
                "source_url": r["url"],
                "status": "ok",
            }
            for r in results
        ]
        return {
            "engine": "duckduckgo_mock",
            "query": query,
            "results": results,
            "summaries": summaries,
            "context": [
                {"title": s["title"], "content": s["summary"], "source_url": s["source_url"]}
                for s in summaries
            ],
            "context_text": "\n\n".join([f"{s['title']}: {s['summary']}" for s in summaries[:5]]),
            "search_metadata": {
                "query": query,
                "engine": "duckduckgo_mock",
                "timestamp": datetime.now().isoformat(),
            },
        }
