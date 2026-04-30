"""Extract clean title + body text from raw HTML.

We try ``trafilatura`` first because it routinely beats hand-rolled
heuristics on real-world pages, then fall back to a BeautifulSoup
strip when trafilatura returns nothing. Returning ``None`` for the
body signals the caller to skip the page (login walls, JS-only apps).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional

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


_WHITESPACE_RE = re.compile(r"\s+")


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


def extract(url: str, html_body: str, max_chars: int = 4000) -> Optional[ExtractedPage]:
    """Return cleaned text + metadata, or None when the page is unusable."""
    if not html_body:
        return None

    soup = BeautifulSoup(html_body, "html.parser")
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
        # last-resort: first 12 words of the body
        title = " ".join(body_text.split()[:12])

    return ExtractedPage(
        url=url,
        title=title,
        description=description,
        body=truncated,
        word_count=word_count,
        language=language,
    )
