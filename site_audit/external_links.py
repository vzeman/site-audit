"""External (cross-domain) link analysis.

GEO models give credit to pages that cite their sources. This module
turns the crawler's per-page external link list into:

* per-page **citation density** (external links per 1000 words)
* per-page **distinct outbound domains**
* site-wide **top cited domains** (with anchor-text samples)
* an **authority profile** of cited domains using a coarse heuristic
  (gov / edu / known authoritative TLDs, plus our own count of how
  often the site cites them)
* optional **broken outbound link detection** via cached HEAD requests

The HEAD check is opt-in (``--check-external``) because it can hit
hundreds of third-party domains; results are cached in the same
``HttpCache`` so repeated runs are free.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests

LOG = logging.getLogger(__name__)


_AUTH_TLDS = {"gov", "edu", "ac.uk", "edu.au", "gov.uk"}
_KNOWN_AUTH_DOMAINS = {
    "wikipedia.org", "nih.gov", "nature.com", "sciencedirect.com",
    "pubmed.ncbi.nlm.nih.gov", "who.int", "europa.eu", "harvard.edu",
    "mit.edu", "stanford.edu", "ox.ac.uk", "cam.ac.uk", "github.com",
    "arxiv.org", "ieee.org", "acm.org", "researchgate.net",
}


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _tld_authority_score(domain: str) -> float:
    """Coarse 0–1 authority hint from TLD + known-domain list.

    Not a rank — just a signal that pages citing these are usually
    showing better source-discipline than ones linking to random blogs.
    """
    if not domain:
        return 0.0
    if domain in _KNOWN_AUTH_DOMAINS:
        return 1.0
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in _AUTH_TLDS:
            return 0.8
    if parts and parts[-1] in {"gov", "edu", "mil"}:
        return 0.8
    return 0.0


@dataclass
class ExternalLinksResult:
    per_page: list[dict]                  # [{url, external_count, distinct_domains, citation_density, authority_share, ...}]
    top_domains: list[dict]               # site-wide most-cited domains
    broken_links: list[dict] = field(default_factory=list)
    citation_density_summary: dict = field(default_factory=dict)


def analyze(
    pages,                                              # list of PageInfo
    page_word_counts: dict[str, int],                   # url -> word_count
    pages_with_external: list[tuple[str, list[tuple[str, str]]]],
    check_links: bool = False,
    http_cache=None,
    max_workers: int = 8,
    timeout: float = 12.0,
) -> ExternalLinksResult:
    per_page: list[dict] = []
    domain_counts: Counter[str] = Counter()
    domain_anchors: dict[str, Counter[str]] = defaultdict(Counter)
    domain_pages: dict[str, set[str]] = defaultdict(set)

    densities: list[float] = []
    for url, externals in pages_with_external:
        if not externals:
            per_page.append({
                "url": url,
                "external_count": 0,
                "distinct_domains": 0,
                "citation_density": 0.0,
                "authority_share": 0.0,
                "domains": [],
            })
            densities.append(0.0)
            continue

        unique_domains: set[str] = set()
        auth_hits = 0
        for tgt, anchor in externals:
            d = _domain_of(tgt)
            if not d:
                continue
            unique_domains.add(d)
            domain_counts[d] += 1
            if anchor:
                domain_anchors[d][anchor.strip().lower()[:80]] += 1
            domain_pages[d].add(url)
            if _tld_authority_score(d) >= 0.8:
                auth_hits += 1

        words = max(page_word_counts.get(url, 0), 1)
        density = round(1000.0 * len(externals) / words, 3)
        densities.append(density)
        per_page.append({
            "url": url,
            "external_count": len(externals),
            "distinct_domains": len(unique_domains),
            "citation_density": density,
            "authority_share": round(auth_hits / max(1, len(externals)), 3),
            "domains": sorted(unique_domains)[:10],
        })

    per_page.sort(key=lambda r: r["external_count"], reverse=True)

    top_domains_payload = []
    for d, count in domain_counts.most_common(50):
        top_domains_payload.append({
            "domain": d,
            "link_count": count,
            "pages_citing": len(domain_pages[d]),
            "authority_score": round(_tld_authority_score(d), 2),
            "top_anchors": [a for a, _ in domain_anchors[d].most_common(3)],
        })

    broken_links: list[dict] = []
    if check_links:
        # gather candidate (url, anchor) pairs once, dedup by URL
        unique_urls: dict[str, str] = {}
        for _, externals in pages_with_external:
            for tgt, anchor in externals:
                if tgt not in unique_urls:
                    unique_urls[tgt] = anchor
        if unique_urls:
            broken_links = _check_links(list(unique_urls.items()), http_cache=http_cache, max_workers=max_workers, timeout=timeout)

    if densities:
        import statistics
        density_summary = {
            "median": round(statistics.median(densities), 3),
            "mean": round(statistics.fmean(densities), 3),
            "p90": round(sorted(densities)[int(len(densities) * 0.9)], 3) if densities else 0.0,
            "pages_with_zero_external": sum(1 for d in densities if d == 0),
        }
    else:
        density_summary = {}

    return ExternalLinksResult(
        per_page=per_page,
        top_domains=top_domains_payload,
        broken_links=broken_links,
        citation_density_summary=density_summary,
    )


def _check_links(
    targets: list[tuple[str, str]],
    http_cache=None,
    max_workers: int = 8,
    timeout: float = 12.0,
) -> list[dict]:
    """HEAD-check unique external URLs; return only the broken ones.

    We treat anything in [400, 599] as broken. 4xx domains often
    redirect 200, so we follow redirects and use the final status.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "site-audit/0.2 (+https://github.com/vzeman/site-audit) link-check",
        "Accept": "*/*",
    })

    def _check(url: str, anchor: str) -> Optional[dict]:
        if http_cache is not None:
            cached = http_cache.get(url)
            if cached is not None:
                if 400 <= cached.status < 600:
                    return {"url": url, "anchor": anchor, "status": cached.status, "from_cache": True}
                return None
        try:
            r = session.head(url, allow_redirects=True, timeout=timeout)
            if r.status_code == 405 or r.status_code == 403:
                # some servers reject HEAD; fall back to GET
                r = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
                r.close()
        except requests.RequestException as exc:
            return {"url": url, "anchor": anchor, "status": None, "error": str(exc)[:120]}
        if http_cache is not None and r.status_code < 400:
            try:
                http_cache.put(url, r.status_code, dict(r.headers), b"")
            except Exception:
                pass
        if 400 <= r.status_code < 600:
            return {"url": url, "anchor": anchor, "status": r.status_code, "from_cache": False}
        return None

    broken: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_check, url, anchor): url for url, anchor in targets}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 50 == 0:
                LOG.info("  external link-check: %d / %d", done, len(futs))
            try:
                res = fut.result()
            except Exception:
                continue
            if res is not None:
                broken.append(res)
    broken.sort(key=lambda r: (r.get("status") or 999, r.get("url", "")))
    return broken


def to_payload(result: ExternalLinksResult, top_n: int = 50) -> dict:
    return {
        "citation_density_summary": result.citation_density_summary,
        "per_page": result.per_page[:top_n * 4],   # surface more pages here than other tables
        "top_domains": result.top_domains[:top_n],
        "broken_links": result.broken_links,
    }
