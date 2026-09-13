"""
Ultron - Web Search Providers (Firecrawl, DDGS, Wikipedia)

Unified interface for multiple search providers with fallback chain.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

log = get_logger("tools.websearch_providers")

_TIMEOUT = 8
_MAX_RESULTS = 5

# Small in-memory cache so repeat queries don't re-hit the network.
_SEARCH_CACHE = {}
_SEARCH_CACHE_TTL = 900  # 15 minutes


class SearchProvider:
    """Base search provider"""
    name = "base"
    requires_key = False
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        return True


class DDGSProvider(SearchProvider):
    """DuckDuckGo search (free, no key needed)"""
    name = "ddgs"
    requires_key = False
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        
        try:
            with DDGS() as ddgs:
                # Use html backend which is more stable
                results = list(ddgs.text(query, max_results=max_results, backend="html"))
        except Exception as e:
            log.warning(f"DDGS html backend failed, trying lite: {e}")
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results, backend="lite"))
            except Exception as e2:
                log.error(f"DDGS failed: {e2}")
                return []
        
        return [{
            "title": r.get("title", r.get("h", "")),
            "url": r.get("href", r.get("u", "")),
            "snippet": r.get("body", r.get("a", "")),
            "source": "ddgs"
        } for r in results]


class FirecrawlProvider(SearchProvider):
    """Firecrawl search (requires FIRECRAWL_API_KEY)"""
    name = "firecrawl"
    requires_key = True
    
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        
        try:
            # No scrapeOptions: plain search is much faster than full markdown scrape.
            resp = requests.post(
                f"{self.base_url}/search",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"query": query, "limit": max_results},
                timeout=_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("data", [])[:max_results]:
                results.append({
                    "title": item.get("title", item.get("url", "")),
                    "url": item.get("url", ""),
                    "snippet": (item.get("description", "") or item.get("markdown", ""))[:300],
                    "source": "firecrawl"
                })
            return results
        except Exception as e:
            log.warning(f"Firecrawl search failed: {e}")
            return []


class WikipediaProvider(SearchProvider):
    """Wikipedia fallback (always available)"""
    name = "wikipedia"
    requires_key = False
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": max_results
                },
                headers={"User-Agent": "Ultron/1.0 (https://github.com/ultron)"},
                timeout=_TIMEOUT
            )
            resp.raise_for_status()
            
            results = []
            for item in resp.json().get("query", {}).get("search", []):
                title = item.get("title", "")
                results.append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "snippet": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                    "source": "wikipedia"
                })
            return results
        except Exception as e:
            log.warning(f"Wikipedia search failed: {e}")
            return []


# Provider chain - order matters for fallback (lazy initialization)
_PROVIDER_CLASSES = [
    ("firecrawl", FirecrawlProvider),
    ("ddgs", DDGSProvider),
    ("wikipedia", WikipediaProvider),
]

_PROVIDER_INSTANCES = {}


def get_provider_instance(name: str, cls) -> SearchProvider:
    """Lazy initialization of provider instances."""
    if name not in _PROVIDER_INSTANCES:
        _PROVIDER_INSTANCES[name] = cls()
    return _PROVIDER_INSTANCES[name]


def get_available_providers() -> List[SearchProvider]:
    """Get list of available providers in priority order (lazy init)."""
    available = []
    for name, cls in _PROVIDER_CLASSES:
        provider = get_provider_instance(name, cls)
        if provider.is_available():
            available.append(provider)
            log.info(f"Web search provider available: {name}")
        else:
            log.debug(f"Web search provider unavailable: {name} (key missing)")
    return available


def multi_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search with fast fallback chain: Firecrawl -> DDGS -> Wikipedia.
    Returns the FIRST provider that yields usable results (no slow merging).
    Results are cached in-memory for 15 minutes.
    """
    import time as _time

    try:
        max_results = int(max_results)
    except (ValueError, TypeError):
        max_results = 5

    cache_key = f"{query.strip().lower()}:{max_results}"
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and (_time.time() - cached[0]) < _SEARCH_CACHE_TTL:
        return cached[1]

    providers = get_available_providers()
    if not providers:
        log.error("No web search providers available!")
        return []

    seen_urls = set()
    for provider in providers:
        try:
            results = provider.search(query, max_results)
            deduped = []
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduped.append(r)
            if deduped:
                _SEARCH_CACHE[cache_key] = (_time.time(), deduped[:max_results])
                return deduped[:max_results]
        except Exception as e:
            log.warning(f"Provider {provider.name} failed: {e}")
            continue

    return []


def format_search_results(results: List[Dict[str, Any]], query: str) -> str:
    """Format results for LLM context."""
    if not results:
        return f"Web search for '{query}' returned no results."
    
    lines = [f"Web search results for \"{query}\" (using {len(results)} sources):"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    URL: {r.get('url', 'No URL')}")
        lines.append(f"    Source: {r.get('source', 'unknown')}")
        lines.append(f"    Snippet: {r.get('snippet', 'No snippet')[:300]}")
        lines.append("")
    
    lines.append("Use these results to answer. Cite sources with [n].")
    return "\n".join(lines)