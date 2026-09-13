"""
Ultron - web search & scan tool.

1. Searches the web (keyless DuckDuckGo via the `ddgs`/`duckduckgo_search`
   package, with a lite-scrape fallback).
2. Fetches the top result pages and extracts readable text.
3. Returns compact snippets + source URLs for the LLM to synthesize.
"""
import re
import time
import hashlib
import json
from html.parser import HTMLParser
from functools import lru_cache

import requests

from utils.logger import get_logger
from database import db

log = get_logger("tools.websearch")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT = 8
_MAX_FETCH_CHARS = 4000  # per page we pass to the LLM
_CACHE_TTL = 3600  # 1 hour cache


class _TextExtractor(HTMLParser):
    """Strip HTML tags and pull out readable visible text."""

    _BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "section", "article", "br"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0
        self._block_open = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        if tag in self._BLOCK and not self._skip:
            self.parts.append("\n")
            self._block_open = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self):
        raw = " ".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n", raw)
        return raw.strip()


def _extract_text(html):
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _cache_key(query: str, max_results: int) -> str:
    return hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()


def _get_cached_search(query: str, max_results: int):
    """Check if we have cached search results."""
    key = _cache_key(query, max_results)
    row = db.query_one("SELECT value FROM web_search_cache WHERE key = ? AND expires > datetime('now')", (key,))
    if row:
        try:
            return json.loads(row["value"])
        except Exception:
            pass
    return None


def _set_cached_search(query: str, max_results: int, results: list):
    """Cache search results."""
    key = _cache_key(query, max_results)
    expires = f"datetime('now', '+{int(_CACHE_TTL/60)} minutes')"
    db.execute(
        "INSERT OR REPLACE INTO web_search_cache (key, value, expires) VALUES (?, ?, datetime('now', '+60 minutes'))",
        (key, json.dumps(results)),
    )


def _fetch_page(url):
    """Fetch a URL and return its readable text (truncated)."""
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        return text[:_MAX_FETCH_CHARS]
    except Exception:
        return ""


def _search(query, max_results, attempts=2, settings=None):
    """Return a list of {title, href, body} via Tavily → Brave → DDGS → Wikipedia → lite."""
    # Check cache first
    cached = _get_cached_search(query, max_results)
    if cached:
        log.debug("Web search cache hit for: %s", query[:50])
        return cached

    # 1) Try Tavily (best quality, API key required)
    try:
        tavily_results = _search_tavily(query, max_results, settings)
        if tavily_results:
            _set_cached_search(query, max_results, tavily_results)
            return tavily_results
    except Exception as exc:
        log.debug("Tavily search failed: %s", exc)

    # 2) Try Brave Search (API key required)
    try:
        brave_results = _search_brave(query, max_results, settings)
        if brave_results:
            _set_cached_search(query, max_results, brave_results)
            return brave_results
    except Exception as exc:
        log.debug("Brave search failed: %s", exc)

    # 3) Fallback to DuckDuckGo (keyless)
    backends = ["lite", "html"]  # prefer lite for speed
    for attempt in range(attempts):
        backend = backends[attempt % len(backends)]
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend=backend))
            if results:
                _set_cached_search(query, max_results, results)
                return results
        except Exception as exc:
            log.debug("DDGS attempt %d (%s) failed: %s", attempt + 1, backend, exc)
        time.sleep(0.4 * (attempt + 1))

    # 4) Wikipedia fallback
    try:
        wiki = _search_wikipedia(query, max_results)
        if wiki:
            _set_cached_search(query, max_results, wiki)
            return wiki
    except Exception as exc:
        log.debug("Wikipedia fallback failed: %s", exc)

    # 5) Lite scrape fallback
    try:
        lite_results = _search_lite(query, max_results)
        if lite_results:
            _set_cached_search(query, max_results, lite_results)
            return lite_results
    except Exception as exc:
        log.debug("Lite search fallback failed: %s", exc)
    return []


def _search_tavily(query, max_results, settings=None):
    """Search via Tavily API (requires TAVILY_API_KEY)."""
    import os
    api_key = ""
    if settings:
        api_key = settings.get("tavily_key") or ""
    if not api_key:
        api_key = os.getenv("TAVILY_API_KEY") or ""
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        # Tavily's direct answer is the best source — inject it as the first result
        answer = (data.get("answer") or "").strip()
        if answer:
            results.append({
                "title": "Direct Answer",
                "href": "https://tavily.com",
                "body": answer,
            })
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("url", ""),
                "body": r.get("content", ""),
            })
        return results
    except Exception as e:
        log.debug("Tavily search failed: %s", e)
        return []


def _search_brave(query, max_results, settings=None):
    """Search via Brave Search API (requires BRAVE_API_KEY)."""
    import os
    api_key = ""
    if settings:
        api_key = settings.get("brave_key") or ""
    if not api_key:
        api_key = os.getenv("BRAVE_API_KEY") or ""
    if not api_key:
        return []
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("web", {}).get("results", []):
        results.append({
            "title": r.get("title", ""),
            "href": r.get("url", ""),
            "body": r.get("description", ""),
        })
    return results


def _search_wikipedia(query, max_results):
    """Fallback search engine: the keyless, reliable Wikipedia API."""
    import requests as _requests

    resp = _requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
            "utf8": 1,
        },
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for s in (resp.json().get("query", {}).get("search", []) or []):
        title = s.get("title", "")
        out.append(
            {
                "title": title,
                "href": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "body": s.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
            }
        )
    return out


def _search_lite(query, max_results):
    """Fallback: scrape lite.duckduckgo.com directly."""
    resp = requests.get(
        "https://lite.duckduckgo.com/lite/",
        params={"q": query},
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = []
    for href in re.findall(r'href="//duckduckgo\.com/l/\?uddg=([^"]+)"', resp.text)[:max_results]:
        from urllib.parse import unquote

        results.append({"title": "", "href": unquote(href), "body": ""})
    return results


def web_search(query, max_results=5, fetch_pages=False, settings=None):
    """
    Search + scan the web for `query`.

    By default answers come straight from the engine's title + snippet
    (fast, never blocked by anti-bot 403s). When `fetch_pages` is True, or a
    result has no snippet, the page is fetched and extracted for real text.

    Returns a list of dicts: {title, url, snippet, text}.
    """
    try:
        max_results = int(max_results)
        results = _search(query, max_results=max_results, settings=settings)
    except Exception as exc:
        log.warning("Web search failed: %s", exc)
        return []

    items = []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        title = r.get("title") or url
        snippet = (r.get("body") or "").strip()
        if not url.startswith("http"):
            continue
        # Skip page fetch for Tavily direct answers (not real URLs)
        is_direct_answer = (title == "Direct Answer")
        text = ""
        if not is_direct_answer and (fetch_pages or not snippet):
            try:
                text = _fetch_page(url)
            except Exception:
                log.debug("Could not fetch %s", url)
        items.append({"title": title, "url": url, "snippet": snippet, "text": text})
        if len(items) >= max_results:
            break
    return items


def format_context(items, query):
    """Render scanned results as a compact context block for the LLM."""
    if not items:
        return f"Web search for '{query}' returned no usable results."
    blocks = [f"Web search snippet results for \"{query}\":"]
    for i, item in enumerate(items, 1):
        body = item["text"] or item["snippet"] or "No extractable content."
        blocks.append(f"[{i}] {item['title']}\nURL: {item['url']}\n{body[:900]}")
    blocks.append(
        "Answer the user using ONLY these results when relevant, cite the [n] sources, "
        "and say clearly if the results are insufficient."
    )
    return "\n\n".join(blocks)


def init_web_search_cache():
    """Initialize the web search cache table."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS web_search_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires TEXT NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_web_search_cache_expires ON web_search_cache(expires)")
