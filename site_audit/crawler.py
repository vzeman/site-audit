"""Polite, sitemap-first crawler with persistent HTTP cache.

The crawler is deliberately conservative: we honour ``robots.txt``, cap
in-flight requests, and throttle per-host. Discovery happens in two
passes:

1. **Sitemap pass** — fetch ``robots.txt``, follow any ``Sitemap:`` URLs
   it advertises, plus the conventional ``/sitemap.xml`` path. This
   typically gives us 90% of the site's indexed URLs in one shot.
2. **Crawl pass** — for any seed URL not covered by sitemaps, fall back
   to a BFS frontier capped by ``max_pages``.

Every fetched response is stored in the HTTP cache, keyed by URL. Cached
responses are reused unless ``--no-cache`` is passed.
"""

from __future__ import annotations

import collections
import gzip
import io
import logging
import random
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable, Optional, Set
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

# `curl_cffi` mimics a real browser's TLS fingerprint, which lets us past
# Cloudflare / Shopify / DataDome / PerimeterX bot challenges that
# fingerprint `python-requests` and serve 429s no matter how polite we
# are. We use it when available and fall back to plain `requests`
# otherwise so the package keeps working without the extra wheel.
try:
    from curl_cffi import requests as _cffi  # type: ignore
    _HAS_CFFI = True
except Exception:  # pragma: no cover - optional dep
    _cffi = None
    _HAS_CFFI = False

from .cache import HttpCache

LOG = logging.getLogger(__name__)


# Many CDNs (Cloudflare, Shopify Bot Manager, ...) treat anything that
# doesn't look like a real browser as bot traffic and serve a 429
# challenge page. We borrow Chrome's UA + accept headers so we can audit
# sites behind those defenses without forcing the user to do anything.
# Add ?bot to the query string of the homepage if you want to test
# without this masking.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 site-audit/+https://github.com/vzeman/site-audit"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # NOTE: deliberately omit "br" — `requests` can't decompress brotli
    # without the `brotli` extra, and some CDNs (Shopify) ignore the
    # accept-encoding hint and brotli-encode anyway.
    "Accept-Encoding": "gzip, deflate",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

DEFAULT_EXCLUDE_PATTERNS = [
    r"/wp-admin",
    r"/wp-login",
    r"/wp-json",
    r"/cgi-bin",
    r"/xmlrpc\.php",
    r"\.(jpg|jpeg|png|gif|svg|webp|ico|pdf|zip|gz|tar|mp3|mp4|avi|mov|wmv|css|js|woff2?|ttf|eot|json|xml)$",
    r"/feed/?$",
    r"/rss/?$",
    r"/comments/feed",
    r"\?replytocom=",
    r"/page/\d+/",
    r"/cdn-cgi/",
]


@dataclass
class CrawlConfig:
    domain: str
    max_pages: int = 2000
    max_workers: int = 8
    request_delay: float = 0.0  # extra delay per worker between requests
    timeout: float = 20.0
    follow_subdomains: bool = False
    respect_robots: bool = True
    use_cache: bool = True
    exclude_patterns: list = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    seed_paths: list = field(default_factory=lambda: ["/"])
    user_agent: str = USER_AGENT


@dataclass
class FetchResult:
    url: str
    status: int
    body: str
    content_type: str
    from_cache: bool
    outlinks: list = field(default_factory=list)         # same-site (target_url, anchor_text)
    external_links: list = field(default_factory=list)   # cross-site (target_url, anchor_text)


def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes consistently."""
    url, _ = urldefrag(url)
    return url


def _starting_url(domain: str) -> str:
    if domain.startswith("http://") or domain.startswith("https://"):
        return domain.rstrip("/")
    return f"https://{domain.rstrip('/')}"


class Crawler:
    def __init__(self, config: CrawlConfig, http_cache: HttpCache):
        self.config = config
        self.cache = http_cache
        self.base_url = _starting_url(config.domain)
        parsed = urlparse(self.base_url)
        self.host = parsed.netloc
        self._exclude_re = [re.compile(p) for p in config.exclude_patterns]
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        if _HAS_CFFI:
            # impersonate a real Chrome to bypass TLS-fingerprint bot detection
            self._session = _cffi.Session(impersonate="chrome124")
            self._session.headers.update({"User-Agent": config.user_agent})
        else:
            self._session = requests.Session()
            self._session.headers.update({**DEFAULT_HEADERS, "User-Agent": config.user_agent})
        self._using_cffi = _HAS_CFFI

    # --- public API ----------------------------------------------------

    def discover_and_crawl(self) -> list[FetchResult]:
        # Warm the session: many CDNs (Cloudflare bot challenge, Shopify Bot
        # Manager) 429 the first request from a fresh client until the
        # homepage has been visited and a session cookie has been issued.
        self._warm_session()
        sitemap_urls = self._discover_via_sitemaps()
        seeds = [normalize_url(urljoin(self.base_url, p)) for p in self.config.seed_paths]
        frontier_seed: Set[str] = set()
        for u in sitemap_urls:
            if self._allowed(u):
                frontier_seed.add(u)
        for u in seeds:
            if self._allowed(u):
                frontier_seed.add(u)
        if not frontier_seed:
            frontier_seed.add(self.base_url)

        return self._bfs(frontier_seed)

    def _warm_session(self) -> None:
        """Fetch the homepage so CDNs hand us their bot-challenge cookie."""
        r = self._request_with_retry(self.base_url + "/")
        if r is not None and r.status_code < 400:
            LOG.debug("session warmed at %s (status %d)", r.url, r.status_code)

    # --- robots & sitemaps --------------------------------------------

    def _load_robots(self) -> Optional[urllib.robotparser.RobotFileParser]:
        if not self.config.respect_robots:
            return None
        if self._robots is not None:
            return self._robots
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{self.base_url}/robots.txt"
        r = self._request_with_retry(robots_url)
        if r is not None and r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            rp.parse([])
        self._robots = rp
        return rp

    def _sitemaps_from_robots(self) -> list[str]:
        rp = self._load_robots()
        if rp is None:
            return []
        try:
            return list(rp.site_maps() or [])
        except Exception:
            return []

    def _discover_via_sitemaps(self) -> list[str]:
        candidates: list[str] = list(self._sitemaps_from_robots())
        candidates.append(f"{self.base_url}/sitemap.xml")
        candidates.append(f"{self.base_url}/sitemap_index.xml")

        seen_sitemaps: Set[str] = set()
        urls: Set[str] = set()
        queue = collections.deque(dict.fromkeys(candidates))

        while queue:
            sm = queue.popleft()
            if sm in seen_sitemaps:
                continue
            seen_sitemaps.add(sm)
            r = self._request_with_retry(sm)
            if r is None or r.status_code != 200:
                continue
            content = r.content
            if sm.endswith(".gz") or r.headers.get("Content-Type", "").endswith("gzip"):
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass
            try:
                root = ET.fromstring(content)
            except Exception as exc:
                LOG.warning("sitemap parse failed for %s: %s", sm, exc)
                continue

            tag = root.tag.split("}", 1)[-1]
            if tag == "sitemapindex":
                for loc in root.iter():
                    # Only the unprefixed <loc>; image:loc / video:loc carry
                    # asset URLs, not page URLs.
                    if loc.tag.endswith("}loc") or loc.tag == "loc":
                        if loc.text and not loc.tag.endswith("image}loc") and "image" not in loc.tag and "video" not in loc.tag:
                            queue.append(loc.text.strip())
            elif tag == "urlset":
                for loc in root.iter():
                    if loc.tag.endswith("}loc") or loc.tag == "loc":
                        # skip <image:loc>, <video:loc> — those are media URLs
                        if "image" in loc.tag or "video" in loc.tag:
                            continue
                        if loc.text:
                            urls.add(normalize_url(loc.text.strip()))

        LOG.info("Sitemap discovery: %d URLs across %d sitemaps", len(urls), len(seen_sitemaps))
        return sorted(urls)

    # --- BFS crawl -----------------------------------------------------

    def _allowed(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        if not self._same_site(parsed.netloc):
            return False
        for rx in self._exclude_re:
            if rx.search(url):
                return False
        rp = self._load_robots()
        if rp is not None:
            try:
                if not rp.can_fetch(self.config.user_agent, url):
                    return False
            except Exception:
                pass
        return True

    def _same_site(self, netloc: str) -> bool:
        if netloc == self.host:
            return True
        # treat "x.com" and "www.x.com" as the same site by default
        host_stripped = self.host[4:] if self.host.startswith("www.") else self.host
        netloc_stripped = netloc[4:] if netloc.startswith("www.") else netloc
        if host_stripped == netloc_stripped:
            return True
        if self.config.follow_subdomains:
            base_root = ".".join(host_stripped.split(".")[-2:])
            netloc_root = ".".join(netloc_stripped.split(".")[-2:])
            return base_root == netloc_root
        return False

    def _bfs(self, seeds: Iterable[str]) -> list[FetchResult]:
        seen: Set[str] = set()
        frontier: collections.deque[str] = collections.deque()
        for u in seeds:
            n = normalize_url(u)
            if n not in seen:
                seen.add(n)
                frontier.append(n)

        results: list[FetchResult] = []
        max_pages = self.config.max_pages
        active: dict = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            while frontier and len(results) < max_pages:
                while frontier and len(active) < self.config.max_workers and len(active) + len(results) < max_pages:
                    url = frontier.popleft()
                    fut = pool.submit(self._fetch, url)
                    active[fut] = url

                if not active:
                    break

                done_futures = []
                for fut in as_completed(list(active.keys()), timeout=None):
                    done_futures.append(fut)
                    break  # process one at a time so we can refill the pool

                for fut in done_futures:
                    url = active.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as exc:
                        LOG.warning("fetch failed %s: %s", url, exc)
                        continue
                    if result is None:
                        continue
                    results.append(result)

                    if "html" in result.content_type:
                        page_outlinks: list[tuple[str, str]] = []
                        page_external: list[tuple[str, str]] = []
                        for link, anchor in self._extract_links(result.url, result.body):
                            try:
                                netloc = urlparse(link).netloc
                            except Exception:
                                continue
                            if not netloc:
                                continue
                            if self._same_site(netloc):
                                if link == result.url:
                                    continue
                                page_outlinks.append((link, anchor))
                                if link in seen:
                                    continue
                                if not self._allowed(link):
                                    continue
                                seen.add(link)
                                frontier.append(link)
                            else:
                                page_external.append((link, anchor))
                        result.outlinks = page_outlinks
                        result.external_links = page_external

                    if len(results) % 25 == 0:
                        LOG.info("crawled %d / queue %d / cache %s",
                                 len(results), len(frontier),
                                 "hit" if result.from_cache else "miss")

        LOG.info("Crawl finished: %d pages", len(results))
        return results

    def _extract_links(self, base_url: str, body: str) -> list[tuple[str, str]]:
        """Return list of (absolute_url, anchor_text)."""
        try:
            soup = BeautifulSoup(body, "html.parser")
        except Exception:
            return []
        out: list[tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            absolute = urljoin(base_url, href)
            absolute = normalize_url(absolute)
            anchor = " ".join(a.get_text(" ").split())
            if not anchor:
                # Fall back to title attribute or aria-label so image links aren't blank
                anchor = (a.get("title") or a.get("aria-label") or "").strip()
            out.append((absolute, anchor[:200]))
        return out

    # --- single fetch --------------------------------------------------

    def _request_with_retry(self, url: str, method: str = "GET") -> Optional[requests.Response]:
        """GET/HEAD with exponential backoff on 429/503 (bot-challenge pages).

        We're persistent on retries because Shopify / Cloudflare bot-managed
        sites issue lots of transient 429s under burst traffic — they almost
        always recover within ~15 s if we slow down.
        """
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                if method == "HEAD":
                    r = self._session.head(url, timeout=self.config.timeout, allow_redirects=True)
                else:
                    r = self._session.get(url, timeout=self.config.timeout, allow_redirects=True)
            except Exception as exc:
                LOG.warning("%s %s failed: %s", method, url, exc)
                return None
            if r.status_code in (429, 503):
                # honour Retry-After when present, else exponential + jitter
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = min(int(retry_after), 60)
                else:
                    # 1.5 ** n: 1.5, 2.25, 3.4, 5.1, 7.6, 11.4 seconds
                    delay = (1.5 ** (attempt + 1)) + random.uniform(0, 1.5)
                LOG.info("  %d on %s — backing off %.1fs (attempt %d/%d)",
                         r.status_code, url, delay, attempt + 1, max_attempts)
                time.sleep(delay)
                continue
            return r
        LOG.warning("giving up on %s after %d 429/503 retries", url, max_attempts)
        return r  # last response (still 429)

    def _fetch(self, url: str) -> Optional[FetchResult]:
        if self.config.use_cache:
            cached = self.cache.get(url)
            if cached and 200 <= cached.status < 400:
                return FetchResult(
                    url=url,
                    status=cached.status,
                    body=cached.text,
                    content_type=(cached.content_type or "").lower(),
                    from_cache=True,
                )

        if self.config.request_delay > 0:
            time.sleep(self.config.request_delay)

        r = self._request_with_retry(url)
        if r is None:
            return None

        final_url = normalize_url(r.url)
        ctype = r.headers.get("Content-Type", "").lower()
        body_bytes = r.content

        if self.config.use_cache and 200 <= r.status_code < 400 and "html" in ctype:
            self.cache.put(final_url, r.status_code, dict(r.headers), body_bytes)

        if "html" not in ctype:
            return None
        if r.status_code >= 400:
            return None

        try:
            text = body_bytes.decode(r.encoding or "utf-8", errors="replace")
        except LookupError:
            text = body_bytes.decode("utf-8", errors="replace")

        return FetchResult(
            url=final_url,
            status=r.status_code,
            body=text,
            content_type=ctype,
            from_cache=False,
        )
