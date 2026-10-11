"""Module 5: find evidence on trusted sources only."""
from dataclasses import dataclass
from urllib.parse import urlparse

import logging
import re

import requests

from . import config

log = logging.getLogger("truth_rail.search")

_TIME_WORDS = re.compile(
    r"(last date|deadline|due date|till|until|from tomorrow|from today|starting|extended|"
    r"new rule|new scheme|current|latest|\b(19|20)\d{2}\b|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)",
    re.I)


def is_time_sensitive(claim):
    return bool(_TIME_WORDS.search(claim))


@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    published: str = ""


def is_trusted(url):
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if any(host == d or host.endswith("." + d) for d in config.TRUSTED_DOMAINS):
        return True
    return any(host.endswith(s) for s in config.TRUSTED_SUFFIXES)


class EvidenceRetriever:
    def __init__(self, search_fn=None, max_results=4):
        self.max_results = max_results
        self.search_fn = search_fn or self._default_search

    def search(self, claim):
        raw = []
        queries = [claim, f"fact check {claim}"]
        if is_time_sensitive(claim):
            queries.append(f"{claim} latest update extended deadline")
        queries.append(f"{claim} site:pib.gov.in OR site:who.int OR site:factcheck.afp.com OR site:boomlive.in")
        for q in queries:
            try:
                raw.extend(self.search_fn(q))
            except Exception as e:
                log.warning("Search failed for %r: %s", q[:60], e)
                continue
            trusted = [e for e in raw if is_trusted(e.url)]
            if len(trusted) >= self.max_results and not is_time_sensitive(claim):
                break
        seen, out = set(), []
        for e in raw:
            if is_trusted(e.url) and e.url not in seen:
                seen.add(e.url)
                out.append(e)
        return out[: self.max_results + (2 if is_time_sensitive(claim) else 0)]

    def probe(self, query="Hot lemon water cures cancer fact check"):
        """For the self-test: returns (results, trusted results, error text)."""
        try:
            raw = self.search_fn(query)
        except Exception as e:
            return 0, 0, f"{type(e).__name__}: {str(e)[:200]}"
        return len(raw), len([e for e in raw if is_trusted(e.url)]), ""

    def _default_search(self, query):
        if config.TAVILY_API_KEY:
            return self._tavily(query)
        return self._ddg(query)

    @staticmethod
    def _tavily(query):
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json={"query": query, "max_results": 10, "search_depth": "basic"},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return [Evidence(r.get("title", ""), r.get("url", ""), r.get("content", ""),
                         r.get("published_date", "") or "")
                for r in resp.json().get("results", [])]

    @staticmethod
    def _ddg(query):
        from ddgs import DDGS
        with DDGS() as d:
            return [Evidence(r.get("title", ""), r.get("href", ""), r.get("body", ""))
                    for r in d.text(query, max_results=10)]
