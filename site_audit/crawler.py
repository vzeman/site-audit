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
from datetime import date, datetime, timedelta
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
    r"/feed/?$",
    r"/rss/?$",
    r"/comments/feed",
    r"\?replytocom=",
    r"/page/\d+/",
    r"/cdn-cgi/",
]

# Applied to ``urlparse(url).path`` (so query strings/fragments don't mask
# the extension, e.g. ``/img/foo.jpg?v=2`` still fails). Catches every common
# binary/asset extension we never want to treat as a crawled page.
_NON_HTML_EXTENSIONS = re.compile(
    r"\.(?:jpe?g|png|gif|svg|webp|avif|bmp|ico|heic|tiff?|"   # images
    r"pdf|zip|gz|tgz|tar|7z|rar|"                              # archives / docs
    r"mp3|mp4|m4a|m4v|avi|mov|wmv|webm|ogg|wav|flac|"          # audio/video
    r"css|js|mjs|map|"                                          # web assets
    r"woff2?|ttf|otf|eot|"                                      # fonts
    r"json|xml|rss|atom|csv|tsv|"                               # data
    r"exe|dmg|pkg|deb|rpm|apk|msi|"                             # installers
    r"doc|docx|xls|xlsx|ppt|pptx)"                              # office
    r"(?:$|/)",                                                # boundary
    re.IGNORECASE,
)


@dataclass
class CrawlConfig:
    domain: str
    max_pages: int = 10000
    max_workers: int = 8
    request_delay: float = 0.0  # extra delay per worker between requests
    timeout: float = 20.0
    follow_subdomains: bool = False
    respect_robots: bool = True
    use_cache: bool = True
    crawl_discovered_links: bool = True
    strip_header_footer: bool = False
    content_include_classes: list = field(default_factory=list)
    content_exclude_classes: list = field(default_factory=list)
    exclude_patterns: list = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    include_patterns: list = field(default_factory=list)
    sitemap_urls: list = field(default_factory=list)
    sitemap_include_patterns: list = field(default_factory=list)
    sitemap_exclude_patterns: list = field(default_factory=list)
    sitemap_lastmod_after: Optional[str] = None
    sitemap_lastmod_within_days: Optional[int] = None
    seed_paths: list = field(default_factory=lambda: ["/"])
    user_agent: str = USER_AGENT


@dataclass
class FetchResult:
    url: str
    status: int
    body: str
    content_type: str
    from_cache: bool
    content_length_bytes: int = 0
    x_robots_tag: str = ""                               # raw X-Robots-Tag header (lowercased)
    outlinks: list = field(default_factory=list)         # same-site (target_url, anchor_text)
    external_links: list = field(default_factory=list)   # cross-site (target_url, anchor_text)
    error: str = ""
    requested_url: str = ""
    redirect_target_url: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    redirect_hop_count: int = 0


@dataclass
class SitemapEntry:
    url: str
    source_sitemaps: list[str] = field(default_factory=list)
    lastmod: str = ""


def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes consistently."""
    url, _ = urldefrag(url)
    return url


def _response_redirect_chain(response, requested_url: str, final_url: str) -> list[str]:
    history = list(getattr(response, "history", []) or [])
    if not history and requested_url == final_url:
        return []
    chain: list[str] = []
    for item in history:
        item_url = normalize_url(getattr(item, "url", "") or "")
        if item_url and (not chain or chain[-1] != item_url):
            chain.append(item_url)
    if not chain and requested_url != final_url:
        chain.append(requested_url)
    if final_url and (not chain or chain[-1] != final_url):
        chain.append(final_url)
    return chain


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
        self._include_re = [re.compile(p) for p in config.include_patterns]
        self._sitemap_include_re = [re.compile(p) for p in config.sitemap_include_patterns]
        self._sitemap_exclude_re = [re.compile(p) for p in config.sitemap_exclude_patterns]
        self._sitemap_lastmod_cutoff = self._lastmod_cutoff()
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        self.sitemap_entries: list[dict] = []
        self.sitemap_urls_seen: list[str] = []
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
        """Fetch the homepage so CDNs hand us their bot-challenge cookie.

        Also detect canonical-host redirects (apex → www, http → https) and
        switch ``self.base_url`` and ``self.host`` to the post-redirect form
        — otherwise everything keyed by the original base_url misses cache
        and the same-site filter rejects internal links.

        We try cache before hitting the network: a previous run on the same
        domain has already paid the redirect cost, and re-warming over the
        wire just risks tripping bot detection on subsequent fetches.
        """
        canonical_url: Optional[str] = None

        # 1) cache lookup for either the apex or www form of the homepage
        if self.config.use_cache:
            for guess in [self.base_url + "/", self.base_url, self.base_url.replace("://", "://www.") + "/"]:
                cached = self.cache.get(guess)
                if cached and 200 <= cached.status < 400:
                    canonical_url = guess
                    LOG.debug("warm session served from cache: %s", guess)
                    break

        # 2) fall back to the network when cache misses
        final_url: Optional[str] = canonical_url
        if final_url is None:
            r = self._request_with_retry(self.base_url + "/")
            if r is None or r.status_code <= 0 or r.status_code >= 400:
                return
            final_url = str(r.url)
            if self.config.use_cache and "html" in (r.headers.get("Content-Type") or "").lower():
                try:
                    self.cache.put(normalize_url(final_url), r.status_code, dict(r.headers), r.content)
                except Exception:
                    pass

        try:
            final = urlparse(final_url)
        except Exception:
            return
        if not final.netloc:
            return
        canonical_base = f"{final.scheme}://{final.netloc}".rstrip("/")
        if canonical_base != self.base_url.rstrip("/"):
            LOG.info("canonical host: %s → %s", self.base_url, canonical_base)
            self.base_url = canonical_base
            self.host = final.netloc
        LOG.debug("session warmed at %s", final_url)

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
        if self.config.sitemap_urls:
            candidates: list[str] = list(self.config.sitemap_urls)
        else:
            candidates = list(self._sitemaps_from_robots())
            candidates.append(f"{self.base_url}/sitemap.xml")
            candidates.append(f"{self.base_url}/sitemap_index.xml")

        seen_sitemaps: Set[str] = set()
        urls: Set[str] = set()
        entries: dict[str, SitemapEntry] = {}
        queue = collections.deque(dict.fromkeys(sm for sm in candidates if self._sitemap_allowed(sm)))
        self.sitemap_entries = []
        self.sitemap_urls_seen = []

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
                            child_sitemap = loc.text.strip()
                            if self._sitemap_allowed(child_sitemap):
                                queue.append(child_sitemap)
            elif tag == "urlset":
                for url_node in root:
                    if self._local_name(url_node.tag) != "url":
                        continue
                    loc_text = ""
                    lastmod_text = ""
                    for child in url_node:
                        child_name = self._local_name(child.tag)
                        if child_name == "loc" and "image" not in child.tag and "video" not in child.tag:
                            loc_text = (child.text or "").strip()
                        elif child_name == "lastmod":
                            lastmod_text = (child.text or "").strip()
                    if not loc_text:
                        continue
                    url = normalize_url(loc_text)
                    if self._url_pattern_allowed(url) and self._lastmod_allowed(lastmod_text):
                        urls.add(url)
                        entry = entries.setdefault(url, SitemapEntry(url=url))
                        if sm not in entry.source_sitemaps:
                            entry.source_sitemaps.append(sm)
                        if lastmod_text and not entry.lastmod:
                            entry.lastmod = lastmod_text

        self.sitemap_urls_seen = sorted(seen_sitemaps)
        self.sitemap_entries = [
            {
                "url": entry.url,
                "source_sitemaps": sorted(entry.source_sitemaps),
                "lastmod": entry.lastmod,
            }
            for entry in sorted(entries.values(), key=lambda item: item.url)
        ]
        LOG.info("Sitemap discovery: %d URLs across %d sitemaps", len(urls), len(seen_sitemaps))
        return sorted(urls)

    def _lastmod_cutoff(self) -> Optional[date]:
        if self.config.sitemap_lastmod_after:
            parsed = self._parse_sitemap_date(self.config.sitemap_lastmod_after)
            if parsed is None:
                raise ValueError(
                    f"invalid sitemap_lastmod_after date: {self.config.sitemap_lastmod_after!r}"
                )
            return parsed
        if self.config.sitemap_lastmod_within_days is not None:
            return date.today() - timedelta(days=int(self.config.sitemap_lastmod_within_days))
        return None

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.split("}", 1)[-1]

    @staticmethod
    def _parse_sitemap_date(raw: str) -> Optional[date]:
        text = (raw or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                return None

    def _lastmod_allowed(self, lastmod_text: str) -> bool:
        if self._sitemap_lastmod_cutoff is None:
            return True
        parsed = self._parse_sitemap_date(lastmod_text)
        return parsed is not None and parsed >= self._sitemap_lastmod_cutoff

    def _sitemap_allowed(self, sitemap_url: str) -> bool:
        if self._sitemap_include_re and not any(rx.search(sitemap_url) for rx in self._sitemap_include_re):
            return False
        if any(rx.search(sitemap_url) for rx in self._sitemap_exclude_re):
            return False
        return True

    def _url_pattern_allowed(self, url: str) -> bool:
        if self._include_re and not any(rx.search(url) for rx in self._include_re):
            return False
        if any(rx.search(url) for rx in self._exclude_re):
            return False
        return True

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
        if _NON_HTML_EXTENSIONS.search(parsed.path or ""):
            return False
        if not self._url_pattern_allowed(url):
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
        seen: Set[str] = set()                # URLs we've enqueued (request-time)
        result_urls: Set[str] = set()         # URLs we've already kept (post-redirect)
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
                    # Skip duplicates that arise when several request URLs
                    # redirect to the same canonical (with/without trailing
                    # slash, with/without www, redirected query params).
                    if result.url in result_urls:
                        # Still mark the pre-redirect URL as seen so we
                        # don't try it again.
                        seen.add(url)
                        continue
                    result_urls.add(result.url)
                    seen.add(result.url)
                    results.append(result)

                    if 200 <= result.status < 400 and "html" in result.content_type:
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
                                if not self.config.crawl_discovered_links:
                                    continue
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

        For metadata-style fetches (robots.txt, sitemap.xml) we check the
        HTTP cache first so re-runs on the same domain stay offline.
        """
        if method == "GET" and self.config.use_cache:
            cached = self.cache.get(url)
            if cached and 200 <= cached.status < 400:
                resp = requests.Response()
                resp.status_code = cached.status
                resp._content = cached.body
                resp.url = url
                resp.headers.update(cached.headers or {})
                return resp

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                if method == "HEAD":
                    r = self._session.head(url, timeout=self.config.timeout, allow_redirects=True)
                else:
                    r = self._session.get(url, timeout=self.config.timeout, allow_redirects=True)
            except Exception as exc:
                LOG.warning("%s %s failed: %s", method, url, exc)
                resp = requests.Response()
                resp.status_code = 0
                resp.url = url
                resp._content = b""
                resp.headers = {}
                message = str(exc).lower()
                resp.reason = "timed_out" if "timeout" in message or "timed out" in message else "fetch_failed"
                return resp
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
            # Cache successful GET responses transparently so re-runs (and
            # other helpers like sitemap discovery) can stay offline.
            if (
                method == "GET"
                and self.config.use_cache
                and 200 <= r.status_code < 400
            ):
                try:
                    self.cache.put(url, r.status_code, dict(r.headers), r.content)
                except Exception:
                    pass
            return r
        LOG.warning("giving up on %s after %d 429/503 retries", url, max_attempts)
        return r  # last response (still 429)

    def _fetch(self, url: str) -> Optional[FetchResult]:
        if self.config.use_cache:
            cached = self.cache.get(url)
            if cached and 200 <= cached.status < 400:
                cached_content_type = (cached.content_type or "").lower()
                if "html" not in cached_content_type:
                    return None
                cached_headers = cached.headers or {}
                # case-insensitive lookup for X-Robots-Tag (cache may
                # preserve original casing — be defensive).
                xrt = ""
                for k, v in cached_headers.items():
                    if k.lower() == "x-robots-tag":
                        xrt = (v or "").lower()
                        break
                return FetchResult(
                    url=cached.canonical_url or url,
                    status=cached.status,
                    body=self._prepare_html_body(cached.text),
                    content_type=cached_content_type,
                    from_cache=True,
                    content_length_bytes=len(cached.body or b""),
                    x_robots_tag=xrt,
                    requested_url=url,
                    redirect_target_url=cached.canonical_url or "",
                    redirect_chain=[url, cached.canonical_url] if cached.canonical_url and cached.canonical_url != url else [],
                    redirect_hop_count=1 if cached.canonical_url and cached.canonical_url != url else 0,
                )

        if self.config.request_delay > 0:
            time.sleep(self.config.request_delay)

        r = self._request_with_retry(url)
        if r is None:
            return None

        final_url = normalize_url(r.url)
        ctype = r.headers.get("Content-Type", "").lower()
        body_bytes = r.content

        redirect_chain = _response_redirect_chain(r, url, final_url)
        redirect_hop_count = max(len(redirect_chain) - 1, 0)

        if r.status_code <= 0:
            return FetchResult(
                url=final_url,
                status=0,
                body="",
                content_type=ctype,
                from_cache=False,
                content_length_bytes=0,
                error=getattr(r, "reason", "") or "fetch_failed",
                requested_url=url,
                redirect_target_url=final_url if url != final_url else "",
                redirect_chain=redirect_chain,
                redirect_hop_count=redirect_hop_count,
            )

        if self.config.use_cache and 200 <= r.status_code < 400 and "html" in ctype:
            self.cache.put(final_url, r.status_code, dict(r.headers), body_bytes)
            # Also cache under the original request URL when it differs
            # (e.g. apex → www redirect) so a future fetch of the same
            # request URL hits cache directly. Store canonical_url so the
            # cache hit returns the same URL as a live fetch would.
            if url != final_url:
                self.cache.put(url, r.status_code, dict(r.headers), body_bytes,
                               canonical_url=final_url)

        if r.status_code >= 400:
            return FetchResult(
                url=final_url,
                status=r.status_code,
                body="",
                content_type=ctype,
                from_cache=False,
                content_length_bytes=len(body_bytes or b""),
                x_robots_tag=(r.headers.get("X-Robots-Tag", "") or "").lower(),
                requested_url=url,
                redirect_target_url=final_url if url != final_url else "",
                redirect_chain=redirect_chain,
                redirect_hop_count=redirect_hop_count,
            )
        if "html" not in ctype:
            return None

        try:
            text = body_bytes.decode(r.encoding or "utf-8", errors="replace")
        except LookupError:
            text = body_bytes.decode("utf-8", errors="replace")

        return FetchResult(
            url=final_url,
            status=r.status_code,
            body=self._prepare_html_body(text),
            content_type=ctype,
            from_cache=False,
            content_length_bytes=len(body_bytes or b""),
            x_robots_tag=(r.headers.get("X-Robots-Tag", "") or "").lower(),
            requested_url=url,
            redirect_target_url=final_url if url != final_url else "",
            redirect_chain=redirect_chain,
            redirect_hop_count=redirect_hop_count,
        )

    def _prepare_html_body(self, body: str) -> str:
        if not (
            self.config.strip_header_footer
            or self.config.content_include_classes
            or self.config.content_exclude_classes
        ) or not body:
            return body
        try:
            soup = BeautifulSoup(body, "html.parser")
        except Exception:
            return body
        if self.config.strip_header_footer:
            for tag in soup.find_all(["header", "footer"]):
                tag.decompose()
        if self.config.content_exclude_classes:
            excluded = set(self.config.content_exclude_classes)
            for tag in soup.find_all(True):
                if self._tag_has_any_class(tag, excluded):
                    tag.decompose()
        if self.config.content_include_classes:
            included = set(self.config.content_include_classes)
            selected = []
            selected_ids = set()
            for tag in soup.find_all(True):
                if not self._tag_has_any_class(tag, included):
                    continue
                if any(id(parent) in selected_ids for parent in tag.parents):
                    continue
                selected.append(tag)
                selected_ids.add(id(tag))
            scoped = BeautifulSoup("<html><body></body></html>", "html.parser")
            if soup.head:
                scoped.html.insert(0, soup.head.extract())
            target = scoped.body or scoped
            for tag in selected:
                target.append(tag.extract())
            soup = scoped
        return str(soup)

    @staticmethod
    def _tag_has_any_class(tag, class_names: set[str]) -> bool:
        if not getattr(tag, "attrs", None):
            return False
        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        return any(cls in class_names for cls in classes)
