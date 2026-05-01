"""Extract clean title + body text + structural signals from raw HTML.

We try ``trafilatura`` first because it routinely beats hand-rolled
heuristics on real-world pages, then fall back to a BeautifulSoup
strip when trafilatura returns nothing.

Beyond the body text we also pull a few structural signals that
downstream analyses want to see:

* heading text (H1–H3) — used by keyword-coverage to auto-mine queries
* counts of lists / tables — feeds the answerability score
* JSON-LD types — detects FAQPage, HowTo, Article, Product schema
* outbound links to external domains — used as a citation signal

Returning ``None`` for the body signals the caller to skip the page.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

try:  # trafilatura is the preferred extractor
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:  # pragma: no cover - optional dep
    _HAS_TRAFILATURA = False


@dataclass
class ExtractedPage:
    url: str
    title: str
    description: str
    body: str
    word_count: int
    language: Optional[str]
    headings: list[str] = field(default_factory=list)  # H2 + H3 text, in order (kept for backwards compat)
    h1: str = ""
    h1_count: int = 0                         # number of <h1> elements on the page
    headers_rich: list[dict] = field(default_factory=list)  # every H1-H6 as {"level": int, "text": str, "order": int}
    list_count: int = 0
    table_count: int = 0
    schema_types: list[str] = field(default_factory=list)  # JSON-LD @type values
    external_link_count: int = 0
    has_dates: bool = False
    stat_count: int = 0  # numbers with units / percentages
    paragraphs: list[str] = field(default_factory=list)  # body broken into clean paragraph blocks
    paragraph_link_counts: list[tuple[int, int]] = field(default_factory=list)  # (internal, external) per paragraph, aligned with .paragraphs
    noindex: bool = False  # set when meta robots / X-Robots-Tag asks search engines not to index this URL
    noindex_source: str = ""  # "meta" | "header" | "" — diagnostic only
    link_quality: dict = field(default_factory=dict)  # per-page link quality counters (total, has_text, has_title, image_only, empty, …)
    link_audit_rows: list[dict] = field(default_factory=list)  # one row per <a href>: anchor + flags, used for site-level aggregation


_WHITESPACE_RE = re.compile(r"\s+")
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b",
    re.I,
)
_STAT_RE = re.compile(
    r"\b\d+(?:[,.]\d+)*\s*(?:%|percent|million|billion|thousand|users?|customers?|clients?|"
    r"\$|€|£|kg|kilograms?|grams?|hours?|minutes?|days?|weeks?|months?|years?)\b",
    re.I,
)


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name}) or soup.find(
        "meta", attrs={"property": name}
    )
    if tag and tag.get("content"):
        return _clean(html.unescape(tag["content"]))
    return ""


def _title_from_html(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return _clean(html.unescape(soup.title.string))
    h1 = soup.find("h1")
    if h1:
        return _clean(h1.get_text(" "))
    return ""


def _strip_to_text(html_body: str) -> str:
    soup = BeautifulSoup(html_body, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        tag.decompose()
    return _clean(soup.get_text(" "))


def _extract_paragraphs(
    html_body: str,
    page_url: str = "",
    min_chars: int = 120,
    max_chars: int = 1200,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Pull paragraph-sized blocks + per-paragraph (internal, external) link counts.

    We prefer ``<p>`` and ``<li>`` tags inside the main content; we filter
    nav/footer/aside upfront. Each block is normalized to single-space
    whitespace, length-bounded, and de-duplicated. Blocks shorter than
    ``min_chars`` are usually nav fragments, so they're dropped.

    Link counts: every ``<a href>`` directly inside the paragraph element is
    classified — same registrable host = internal, different host = external,
    on-page anchors / mailto / javascript / tel are ignored. Useful both as
    an editorial signal (link spam vs. orphaned text) and as a filter for
    the paragraph-link-recommendations stage (don't suggest links into
    already-saturated paragraphs).
    """
    soup = BeautifulSoup(html_body, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]):
        tag.decompose()

    try:
        host = urlparse(page_url).netloc.lower() if page_url else ""
    except Exception:
        host = ""
    host_root = host[4:] if host.startswith("www.") else host

    blocks: list[str] = []
    counts: list[tuple[int, int]] = []
    seen: set[str] = set()
    for el in soup.find_all(["p", "li", "blockquote", "h2", "h3", "h4"]):
        text = _clean(el.get_text(" "))
        if not text:
            continue
        if len(text) < min_chars:
            continue
        text = text[:max_chars]
        key = text[:200].lower()
        if key in seen:
            continue
        seen.add(key)

        internal = 0
        external = 0
        for a in el.find_all("a", href=True):
            href = (a["href"] or "").strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            if href.startswith(("http://", "https://")):
                try:
                    other = urlparse(href).netloc.lower()
                except Exception:
                    continue
                other_root = other[4:] if other.startswith("www.") else other
                if not other_root:
                    continue
                if host_root and other_root == host_root:
                    internal += 1
                else:
                    external += 1
            else:
                internal += 1
        blocks.append(text)
        counts.append((internal, external))
    return blocks, counts


def _extract_schema_types(soup: BeautifulSoup) -> list[str]:
    types: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            # malformed JSON-LD blocks are common; ignore them
            continue
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict):
                t = item.get("@type")
                if isinstance(t, list):
                    types.extend(str(x) for x in t)
                elif t:
                    types.append(str(t))
                graph = item.get("@graph")
                if isinstance(graph, list):
                    for g in graph:
                        if isinstance(g, dict):
                            t = g.get("@type")
                            if isinstance(t, list):
                                types.extend(str(x) for x in t)
                            elif t:
                                types.append(str(t))
    return types


def _extract_headings(soup: BeautifulSoup) -> tuple[str, list[str], list[dict], int]:
    """Return (h1_text_first, h2_h3_simple_list, all_headers_rich, h1_count).

    The simple list is kept for callers that already use it; the rich list
    has every H1-H6 in document order, with level + position. ``h1_count``
    lets analysers flag pages with multiple H1s (an SEO anti-pattern that
    Google still tolerates but most CMS templates avoid).
    """
    h1_tags = soup.find_all("h1")
    h1 = _clean(h1_tags[0].get_text(" ")) if h1_tags else ""
    simple: list[str] = []
    rich: list[dict] = []
    order = 0
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = _clean(tag.get_text(" "))
        if not text or len(text) > 200:
            continue
        level = int(tag.name[1])
        rich.append({"level": level, "text": text, "order": order})
        order += 1
        if level in (2, 3):
            simple.append(text)
    return h1, simple, rich, len(h1_tags)


# `noindex` is honoured from two channels:
#   1. <meta name="robots" content="noindex">  (also "googlebot", "bingbot", …)
#      Multi-valued: "noindex,nofollow" or "max-snippet:0, noindex".
#   2. X-Robots-Tag HTTP header (same syntax, optional bot prefix).
# We split on commas and on whitespace + colon so all reasonable forms parse.
_NOINDEX_TOKEN_RE = re.compile(r"\bnoindex\b", re.IGNORECASE)
_INDEX_BOTS = ("robots", "googlebot", "bingbot", "slurp", "duckduckbot", "baiduspider", "yandex")


def _has_noindex_directive(value: str) -> bool:
    return bool(_NOINDEX_TOKEN_RE.search(value or ""))


def _detect_noindex(soup: BeautifulSoup, header_value: str = "") -> tuple[bool, str]:
    """Return (is_noindex, source). Source is "meta" / "header" / ""."""
    # X-Robots-Tag may name a specific bot: "googlebot: noindex". We treat
    # any bot directive that contains "noindex" as a hard noindex signal.
    if header_value and _has_noindex_directive(header_value):
        return True, "header"
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("http-equiv") or "").strip().lower()
        if name not in _INDEX_BOTS:
            continue
        content = (tag.get("content") or "").strip()
        if _has_noindex_directive(content):
            return True, "meta"
    return False, ""


def _link_quality(soup: BeautifulSoup, page_url: str) -> tuple[dict, list[dict]]:
    """Walk every <a href> on the page and tally anchor-quality signals.

    Returns ``(counters, audit_rows)`` where ``audit_rows`` has one entry
    per link with the effective anchor + booleans for site-level aggregation:

    * ``has_text``  — the anchor element contains visible text.
    * ``has_title`` — the ``title`` attribute is set.
    * ``has_aria``  — ``aria-label`` is set (used for icon-only links).
    * ``is_image_only`` — anchor wraps an ``<img>`` and has no visible text.
    * ``img_alt`` — the image's ``alt`` (empty / missing = bad anchor for
      assistive tech and search engines).

    Skips intra-page anchors / ``mailto:`` / ``tel:`` / ``javascript:``.
    """
    try:
        host = urlparse(page_url).netloc.lower()
    except Exception:
        host = ""
    host_root = host[4:] if host.startswith("www.") else host

    total = 0
    has_text = 0
    has_title = 0
    has_aria = 0
    image_only = 0
    empty_link = 0
    internal = 0
    external = 0
    img_no_alt = 0
    rows: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        total += 1
        text = a.get_text(strip=True)
        title_attr = (a.get("title") or "").strip()
        aria_label = (a.get("aria-label") or "").strip()
        img = a.find("img")
        is_img = img is not None
        img_alt = (img.get("alt", "") or "").strip() if is_img else ""
        if text:
            has_text += 1
        if title_attr:
            has_title += 1
        if aria_label:
            has_aria += 1
        if is_img and not text:
            image_only += 1
            if not img_alt:
                img_no_alt += 1
        if not text and not title_attr and not aria_label and not (is_img and img_alt):
            empty_link += 1
        # internal vs external by host
        if href.startswith(("http://", "https://")):
            try:
                other = urlparse(href).netloc.lower()
            except Exception:
                other = ""
            other_root = other[4:] if other.startswith("www.") else other
            if other_root == host_root:
                internal += 1
            else:
                external += 1
        else:
            internal += 1
        # Effective anchor (used by linkbuilding analysis)
        eff = text or title_attr or aria_label or img_alt or ""
        if eff:
            rows.append({
                "anchor": eff[:200],
                "has_text": bool(text),
                "has_title": bool(title_attr),
                "is_image_only": is_img and not text,
                "is_internal": (href.startswith(("/", "#", "?")) or
                                (href.startswith(("http://", "https://")) and host_root and
                                 (urlparse(href).netloc[4:] if urlparse(href).netloc.startswith("www.") else urlparse(href).netloc) == host_root)),
            })

    counters = {
        "total": total,
        "internal": internal,
        "external": external,
        "has_text": has_text,
        "has_title": has_title,
        "has_aria": has_aria,
        "image_only": image_only,
        "image_no_alt": img_no_alt,
        "empty_link": empty_link,
    }
    return counters, rows


def _count_external_links(soup: BeautifulSoup, page_url: str) -> int:
    try:
        host = urlparse(page_url).netloc.lower()
    except Exception:
        host = ""
    host_root = host[4:] if host.startswith("www.") else host
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith(("http://", "https://")):
            continue
        try:
            other = urlparse(href).netloc.lower()
        except Exception:
            continue
        other_root = other[4:] if other.startswith("www.") else other
        if other_root and other_root != host_root:
            count += 1
    return count


def extract(
    url: str,
    html_body: str,
    max_chars: int = 4000,
    x_robots_tag: str = "",
) -> Optional[ExtractedPage]:
    """Return cleaned text + structural signals, or None when unusable.

    If ``x_robots_tag`` (the HTTP header) or a ``<meta name="robots">``
    in the HTML asks search engines not to index this URL, we still
    extract the content but flag it as ``noindex=True``. The pipeline
    drops noindex pages from the analysis corpus.
    """
    if not html_body:
        return None

    soup = BeautifulSoup(html_body, "html.parser")
    is_noindex, noindex_source = _detect_noindex(soup, x_robots_tag)
    title = _title_from_html(soup)
    description = _meta(soup, "description") or _meta(soup, "og:description")
    lang_attr = soup.find("html")
    language = lang_attr.get("lang") if lang_attr and lang_attr.has_attr("lang") else None

    body_text: str = ""
    if _HAS_TRAFILATURA:
        try:
            extracted = trafilatura.extract(
                html_body,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
                url=url,
            )
            body_text = _clean(extracted or "")
        except Exception:
            body_text = ""

    if len(body_text) < 200:
        body_text = _strip_to_text(html_body)

    if not body_text:
        return None

    truncated = body_text[:max_chars]
    word_count = len(body_text.split())

    if not title:
        title = " ".join(body_text.split()[:12])

    h1, headings, headers_rich, h1_count = _extract_headings(soup)
    list_count = len(soup.find_all(["ul", "ol"]))
    table_count = len(soup.find_all("table"))
    schema_types = _extract_schema_types(soup)
    external_link_count = _count_external_links(soup, url)
    has_dates = bool(_DATE_RE.search(body_text))
    stat_count = len(_STAT_RE.findall(body_text))
    paragraphs, paragraph_link_counts = _extract_paragraphs(html_body, url)
    link_quality, link_audit_rows = _link_quality(soup, url)

    return ExtractedPage(
        url=url,
        title=title,
        description=description,
        body=truncated,
        word_count=word_count,
        language=language,
        headings=headings,
        h1=h1,
        h1_count=h1_count,
        headers_rich=headers_rich,
        list_count=list_count,
        table_count=table_count,
        schema_types=schema_types,
        external_link_count=external_link_count,
        has_dates=has_dates,
        stat_count=stat_count,
        paragraphs=paragraphs,
        paragraph_link_counts=paragraph_link_counts,
        noindex=is_noindex,
        noindex_source=noindex_source,
        link_quality=link_quality,
        link_audit_rows=link_audit_rows,
    )
