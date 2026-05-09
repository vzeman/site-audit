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
import os
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_DISABLE_TRAFILATURA = os.getenv("SITE_AUDIT_DISABLE_TRAFILATURA", "").lower() in {"1", "true", "yes"}
_TRAFILATURA_MAX_CHARS = _env_int("SITE_AUDIT_TRAFILATURA_MAX_CHARS", 400_000)


@dataclass
class ExtractedPage:
    url: str
    title: str
    description: str
    body: str
    word_count: int
    language: Optional[str]
    canonical_url: str = ""
    robots_content: str = ""
    og_title: str = ""
    og_description: str = ""
    og_image: str = ""
    twitter_card: str = ""
    twitter_title: str = ""
    twitter_description: str = ""
    headings: list[str] = field(default_factory=list)  # H2 + H3 text, in order (kept for backwards compat)
    h1: str = ""
    h1_count: int = 0                         # number of <h1> elements on the page
    headers_rich: list[dict] = field(default_factory=list)  # every H1-H6 as {"level": int, "text": str, "order": int}
    list_count: int = 0
    table_count: int = 0
    schema_types: list[str] = field(default_factory=list)  # JSON-LD @type values
    schema_blocks: list[dict] = field(default_factory=list)  # parsed JSON-LD diagnostics
    external_link_count: int = 0
    has_dates: bool = False
    date_published: str = ""
    date_modified: str = ""
    date_candidates: list[dict] = field(default_factory=list)
    stat_count: int = 0  # numbers with units / percentages
    paragraphs: list[str] = field(default_factory=list)  # body broken into clean paragraph blocks
    paragraph_link_counts: list[tuple[int, int]] = field(default_factory=list)  # (internal, external) per paragraph, aligned with .paragraphs
    noindex: bool = False  # set when meta robots / X-Robots-Tag asks search engines not to index this URL
    noindex_source: str = ""  # "meta" | "header" | "" — diagnostic only
    link_quality: dict = field(default_factory=dict)  # per-page link quality counters (total, has_text, has_title, image_only, empty, …)
    link_audit_rows: list[dict] = field(default_factory=list)  # one row per <a href>: anchor + flags, used for site-level aggregation
    media_items: list[dict] = field(default_factory=list)  # images/video/audio/iframes with accessibility-relevant attributes
    conversion_signals: dict = field(default_factory=dict)  # CTA/form/contact signals for conversion analysis


_WHITESPACE_RE = re.compile(r"\s+")
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b",
    re.I,
)
_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
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


def _schema_type_values(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if x]
    if value:
        return [str(value)]
    return []


def _jsonld_scalar_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("@value", "value", "date"):
            nested = value.get(key)
            if nested:
                return [str(nested)]
    if isinstance(value, list):
        values: list[str] = []
        for child in value:
            values.extend(_jsonld_scalar_values(child))
        return values
    if value:
        return [str(value)]
    return []


def _jsonld_date_values(item, keys: set[str], active_types: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    if isinstance(item, dict):
        item_types = tuple(_schema_type_values(item.get("@type"))) or active_types
        for key, value in item.items():
            key_str = str(key)
            if key_str in keys:
                for scalar in _jsonld_scalar_values(value):
                    values.append((key_str, scalar, item_types[0] if item_types else ""))
            elif isinstance(value, (dict, list)):
                values.extend(_jsonld_date_values(value, keys, item_types))
    elif isinstance(item, list):
        for child in item:
            values.extend(_jsonld_date_values(child, keys, active_types))
    return values


def _date_candidate(value: str, source: str, kind: str) -> dict | None:
    clean_value = _clean(html.unescape(value or ""))
    if not clean_value:
        return None
    if _ISO_DATE_PREFIX_RE.match(clean_value):
        date_value = clean_value[:10]
    else:
        match = _DATE_RE.search(clean_value)
        if not match:
            return None
        date_value = match.group(0)
    return {"date": date_value, "source": source, "kind": kind}


def _extract_date_candidates(soup: BeautifulSoup, body_text: str) -> tuple[str, str, list[dict]]:
    """Collect publication/update date hints from metadata, schema, and body text.

    Values are intentionally kept as short strings; the freshness analyzer owns
    parsing and ageing so tests can pass a deterministic ``today`` date.
    """
    candidates: list[dict] = []
    published_names = {
        "article:published_time",
        "date",
        "datepublished",
        "dc.date",
        "dc.date.issued",
        "pubdate",
        "publishdate",
    }
    modified_names = {
        "article:modified_time",
        "last-modified",
        "lastmod",
        "datemodified",
        "dc.date.modified",
        "updated_time",
    }

    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or tag.get("itemprop") or "").strip().lower()
        content = tag.get("content") or ""
        if name in published_names:
            candidate = _date_candidate(content, f"meta:{name}", "published")
            if candidate:
                candidates.append(candidate)
        elif name in modified_names:
            candidate = _date_candidate(content, f"meta:{name}", "modified")
            if candidate:
                candidates.append(candidate)

    for time_tag in soup.find_all("time"):
        raw = time_tag.get("datetime") or time_tag.get_text(" ")
        candidate = _date_candidate(raw, "time", "visible")
        if candidate:
            candidates.append(candidate)

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for key, value, schema_type in _jsonld_date_values(data, {"datePublished", "dateModified", "dateCreated", "uploadDate"}):
            kind = "modified" if key == "dateModified" else "published"
            source = f"jsonld:{schema_type}.{key}" if schema_type else f"jsonld:{key}"
            candidate = _date_candidate(value, source, kind)
            if candidate:
                candidates.append(candidate)

    for match in _DATE_RE.finditer(body_text or ""):
        candidates.append({"date": match.group(0), "source": "body", "kind": "visible"})
        if len(candidates) >= 25:
            break

    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate["date"], candidate["source"], candidate["kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    date_published = next((c["date"] for c in deduped if c["kind"] == "published"), "")
    date_modified = next((c["date"] for c in deduped if c["kind"] == "modified"), "")
    return date_published, date_modified, deduped


def _title_from_html(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return _clean(html.unescape(soup.title.string))
    h1 = soup.find("h1")
    if h1:
        return _clean(h1.get_text(" "))
    return ""


def _canonical_url(soup: BeautifulSoup) -> str:
    tag = soup.find("link", rel=lambda value: value and "canonical" in [str(v).lower() for v in (value if isinstance(value, list) else [value])])
    href = (tag.get("href") or "").strip() if tag else ""
    return _clean(html.unescape(href))


def _robots_content(soup: BeautifulSoup) -> str:
    values: list[str] = []
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("http-equiv") or "").strip().lower()
        if name in _INDEX_BOTS and tag.get("content"):
            values.append(_clean(html.unescape(tag["content"])))
    return ", ".join(values)


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


def _schema_types_from_item(item) -> list[str]:
    types: list[str] = []
    if isinstance(item, dict):
        types.extend(_schema_type_values(item.get("@type")))
        graph = item.get("@graph")
        if isinstance(graph, list):
            for graph_item in graph:
                types.extend(_schema_types_from_item(graph_item))
    elif isinstance(item, list):
        for child in item:
            types.extend(_schema_types_from_item(child))
    return types


def _schema_keys_from_item(item) -> list[str]:
    keys: set[str] = set()
    if isinstance(item, dict):
        keys.update(str(k) for k in item.keys())
        graph = item.get("@graph")
        if isinstance(graph, list):
            for graph_item in graph:
                keys.update(_schema_keys_from_item(graph_item))
    elif isinstance(item, list):
        for child in item:
            keys.update(_schema_keys_from_item(child))
    return sorted(keys)


def _schema_names_from_item(item) -> list[str]:
    names: list[str] = []
    if isinstance(item, dict):
        item_types = set(_schema_type_values(item.get("@type")))
        if item_types & {"Organization", "LocalBusiness", "Corporation", "NGO", "EducationalOrganization"}:
            for key in ("name", "legalName", "alternateName"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    names.append(value.strip())
                elif isinstance(value, list):
                    names.extend(str(v).strip() for v in value if str(v).strip())
        graph = item.get("@graph")
        if isinstance(graph, list):
            for graph_item in graph:
                names.extend(_schema_names_from_item(graph_item))
    elif isinstance(item, list):
        for child in item:
            names.extend(_schema_names_from_item(child))
    return names


def _extract_schema_data(soup: BeautifulSoup) -> tuple[list[str], list[dict]]:
    types: list[str] = []
    blocks: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception as exc:
            blocks.append({
                "format": "json-ld",
                "valid": False,
                "types": [],
                "error": str(exc),
            })
            continue
        block_types = _schema_types_from_item(data)
        types.extend(block_types)
        blocks.append({
            "format": "json-ld",
            "valid": True,
            "types": sorted(set(block_types)),
            "keys": _schema_keys_from_item(data),
            "names": sorted(set(_schema_names_from_item(data))),
            "error": "",
        })
    return sorted(set(types)), blocks


def _extract_schema_types(soup: BeautifulSoup) -> list[str]:
    types, _ = _extract_schema_data(soup)
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


def _media_items(soup: BeautifulSoup) -> list[dict]:
    """Extract lightweight media accessibility signals from HTML."""
    items: list[dict] = []

    for img in soup.find_all("img"):
        parent_link = img.find_parent("a")
        link_text = _clean(parent_link.get_text(" ")) if parent_link else ""
        items.append({
            "type": "image",
            "src": _clean(html.unescape(img.get("src") or img.get("data-src") or "")),
            "alt": _clean(html.unescape(img.get("alt") or "")),
            "alt_present": img.has_attr("alt"),
            "title": _clean(html.unescape(img.get("title") or "")),
            "aria_label": _clean(html.unescape(img.get("aria-label") or "")),
            "role": _clean(img.get("role") or "").lower(),
            "aria_hidden": str(img.get("aria-hidden") or "").lower() == "true",
            "in_link": parent_link is not None,
            "link_text": link_text,
            "link_title": _clean(html.unescape(parent_link.get("title") or "")) if parent_link else "",
            "link_aria_label": _clean(html.unescape(parent_link.get("aria-label") or "")) if parent_link else "",
        })

    for video in soup.find_all("video"):
        tracks = [
            _clean(track.get("kind") or "").lower()
            for track in video.find_all("track")
            if _clean(track.get("kind") or "")
        ]
        items.append({
            "type": "video",
            "src": _clean(html.unescape(video.get("src") or "")),
            "title": _clean(html.unescape(video.get("title") or "")),
            "aria_label": _clean(html.unescape(video.get("aria-label") or "")),
            "track_kinds": tracks,
            "has_captions": any(kind in {"captions", "subtitles"} for kind in tracks),
        })

    for audio in soup.find_all("audio"):
        parent = audio.parent
        nearby_text = _clean(parent.get_text(" "))[:500].lower() if parent else ""
        items.append({
            "type": "audio",
            "src": _clean(html.unescape(audio.get("src") or "")),
            "title": _clean(html.unescape(audio.get("title") or "")),
            "aria_label": _clean(html.unescape(audio.get("aria-label") or "")),
            "has_transcript_hint": "transcript" in nearby_text,
        })

    for iframe in soup.find_all("iframe"):
        items.append({
            "type": "iframe",
            "src": _clean(html.unescape(iframe.get("src") or "")),
            "title": _clean(html.unescape(iframe.get("title") or "")),
            "aria_label": _clean(html.unescape(iframe.get("aria-label") or "")),
        })

    return items


_CTA_RE = re.compile(
    r"\b("
    r"buy|order|checkout|cart|pricing|price|quote|estimate|demo|book|schedule|"
    r"contact|call|start|signup|sign\s*up|register|subscribe|download|apply|"
    r"request|consultation|get\s+started|learn\s+more"
    r")\b",
    re.I,
)
_PRIMARY_CTA_RE = re.compile(
    r"\b("
    r"buy|order|checkout|quote|estimate|demo|book|schedule|contact|call|"
    r"signup|sign\s*up|subscribe|apply|get\s+started"
    r")\b",
    re.I,
)


def _conversion_signals(soup: BeautifulSoup, page_url: str) -> dict:
    """Extract lightweight CTA, form, and direct-contact signals."""
    try:
        host = urlparse(page_url).netloc.lower()
    except Exception:
        host = ""
    host_root = host[4:] if host.startswith("www.") else host

    ctas: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for tag in soup.find_all(["a", "button", "input"]):
        tag_name = tag.name or ""
        input_type = (tag.get("type") or "").strip().lower()
        if tag_name == "input" and input_type not in {"button", "submit", "image"}:
            continue
        href = (tag.get("href") or "").strip()
        if href.startswith(("javascript:", "#")):
            continue
        text = _clean(tag.get_text(" ") or tag.get("value") or tag.get("aria-label") or tag.get("title") or "")
        if not text and href.startswith(("tel:", "mailto:")):
            text = href.split(":", 1)[0]
        if not text:
            continue
        text = text[:120]
        is_primary = bool(_PRIMARY_CTA_RE.search(text) or href.startswith(("tel:", "mailto:")))
        is_cta = is_primary or bool(_CTA_RE.search(text))
        if not is_cta:
            continue
        key = (text.lower(), href)
        if key in seen:
            continue
        seen.add(key)
        destination = "internal"
        if href.startswith(("http://", "https://")):
            other = urlparse(href).netloc.lower()
            other_root = other[4:] if other.startswith("www.") else other
            destination = "internal" if host_root and other_root == host_root else "external"
        elif href.startswith("mailto:"):
            destination = "email"
        elif href.startswith("tel:"):
            destination = "phone"
        ctas.append({
            "text": text,
            "element": tag_name,
            "href": href,
            "destination": destination,
            "primary": is_primary,
        })

    forms: list[dict] = []
    for form in soup.find_all("form"):
        fields = []
        for field in form.find_all(["input", "select", "textarea"]):
            field_type = (field.get("type") or field.name or "").strip().lower()
            if field_type in {"hidden", "submit", "button", "reset", "image"}:
                continue
            label = field.get("name") or field.get("id") or field.get("placeholder") or field_type
            fields.append(_clean(str(label))[:80])
        submit = form.find(["button", "input"], attrs={"type": re.compile(r"submit|button", re.I)})
        if submit is None:
            submit = form.find("button")
        submit_text = _clean(submit.get_text(" ") or submit.get("value") or submit.get("aria-label") or "") if submit else ""
        forms.append({
            "action": (form.get("action") or "").strip(),
            "method": (form.get("method") or "get").strip().lower(),
            "field_count": len(fields),
            "fields": fields[:20],
            "has_submit": bool(submit),
            "submit_text": submit_text[:120],
        })

    contact_links = [
        (a.get("href") or "").strip()
        for a in soup.find_all("a", href=True)
        if (a.get("href") or "").strip().startswith(("tel:", "mailto:"))
    ]
    return {
        "cta_count": len(ctas),
        "primary_cta_count": sum(1 for cta in ctas if cta["primary"]),
        "ctas": ctas[:50],
        "form_count": len(forms),
        "form_field_count": sum(form["field_count"] for form in forms),
        "forms": forms[:20],
        "contact_link_count": len(contact_links),
        "contact_links": contact_links[:20],
    }


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
    canonical_url = _canonical_url(soup)
    robots_content = _robots_content(soup)
    og_title = _meta(soup, "og:title")
    og_description = _meta(soup, "og:description")
    og_image = _meta(soup, "og:image")
    twitter_card = _meta(soup, "twitter:card")
    twitter_title = _meta(soup, "twitter:title")
    twitter_description = _meta(soup, "twitter:description")
    lang_attr = soup.find("html")
    language = lang_attr.get("lang") if lang_attr and lang_attr.has_attr("lang") else None

    body_text: str = ""
    if (
        _HAS_TRAFILATURA
        and not _DISABLE_TRAFILATURA
        and (_TRAFILATURA_MAX_CHARS <= 0 or len(html_body) <= _TRAFILATURA_MAX_CHARS)
    ):
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
    schema_types, schema_blocks = _extract_schema_data(soup)
    external_link_count = _count_external_links(soup, url)
    date_published, date_modified, date_candidates = _extract_date_candidates(soup, body_text)
    has_dates = bool(date_candidates)
    stat_count = len(_STAT_RE.findall(body_text))
    paragraphs, paragraph_link_counts = _extract_paragraphs(html_body, url)
    link_quality, link_audit_rows = _link_quality(soup, url)
    media_items = _media_items(soup)
    conversion_signals = _conversion_signals(soup, url)

    return ExtractedPage(
        url=url,
        title=title,
        description=description,
        body=truncated,
        word_count=word_count,
        language=language,
        canonical_url=canonical_url,
        robots_content=robots_content,
        og_title=og_title,
        og_description=og_description,
        og_image=og_image,
        twitter_card=twitter_card,
        twitter_title=twitter_title,
        twitter_description=twitter_description,
        headings=headings,
        h1=h1,
        h1_count=h1_count,
        headers_rich=headers_rich,
        list_count=list_count,
        table_count=table_count,
        schema_types=schema_types,
        schema_blocks=schema_blocks,
        external_link_count=external_link_count,
        has_dates=has_dates,
        date_published=date_published,
        date_modified=date_modified,
        date_candidates=date_candidates,
        stat_count=stat_count,
        paragraphs=paragraphs,
        paragraph_link_counts=paragraph_link_counts,
        noindex=is_noindex,
        noindex_source=noindex_source,
        link_quality=link_quality,
        link_audit_rows=link_audit_rows,
        media_items=media_items,
        conversion_signals=conversion_signals,
    )
