"""Page type and template classification.

This module is intentionally heuristic and cheap: it uses signals the
extractor already collects (URL path, title/H1, schema types, headings,
word count, lists/tables, and paragraph count) to classify every page into
an editorial page type and a reusable template family.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from .extractor import ExtractedPage


@dataclass
class PageTypeRow:
    url: str
    title: str
    page_type: str
    template_family: str
    template_signature: str
    confidence: float
    signals: list[str]


@dataclass
class PageTypeReport:
    summary: dict
    type_counts: list[dict]
    template_counts: list[dict]
    per_page: list[dict]


_ARTICLE_SCHEMA = {"Article", "NewsArticle", "BlogPosting", "TechArticle", "Report"}
_PRODUCT_SCHEMA = {"Product", "Service"}
_FAQ_SCHEMA = {"FAQPage", "QAPage", "HowTo"}
_LOCAL_SCHEMA = {"LocalBusiness", "Organization"}

_BLOG_RE = re.compile(r"/(blog|news|articles?|insights?|resources?|press)/", re.I)
_PRODUCT_RE = re.compile(r"/(products?|solutions?|pricing|shop|store)/", re.I)
_SERVICE_RE = re.compile(r"/(services?|agency|consulting)/", re.I)
_DOCS_RE = re.compile(r"/(docs?|documentation|guides?|manual|help|support|kb|learn)/", re.I)
_FAQ_RE = re.compile(r"/(faq|faqs|questions|how-to|howto)/", re.I)
_CONTACT_RE = re.compile(r"/(contact|locations?|book|demo|request)/?$", re.I)
_ABOUT_RE = re.compile(r"/(about|team|company|careers?)/?$", re.I)
_LEGAL_RE = re.compile(r"/(privacy|terms|legal|cookies?|gdpr|impressum)/?$", re.I)
_LISTING_RE = re.compile(r"/(category|categories|tag|tags|archive|author|topics?|collections?)/", re.I)
_SEARCH_RE = re.compile(r"/(search|find)/?$", re.I)

_QUESTION_RE = re.compile(r"\b(what|why|how|when|where|who|which|can|should|does|do|is|are)\b", re.I)


def _path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def _text_blob(page: ExtractedPage) -> str:
    parts = [page.title, page.h1, page.description, " ".join(page.headings or [])]
    return " ".join(part for part in parts if part).lower()


def _schema_set(page: ExtractedPage) -> set[str]:
    return {str(schema_type) for schema_type in (page.schema_types or [])}


def _is_home(path: str) -> bool:
    return path in ("", "/")


def _bucket(value: int, cuts: tuple[int, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"le{cut}"
    return f"gt{cuts[-1]}"


def _score_candidates(page: ExtractedPage) -> dict[str, list[str]]:
    path = _path(page.url)
    blob = _text_blob(page)
    schema = _schema_set(page)
    headings = page.headers_rich or []
    paragraphs = page.paragraphs or []
    candidates: dict[str, list[str]] = defaultdict(list)

    if _is_home(path):
        candidates["home"].append("root_url")
    if schema & _ARTICLE_SCHEMA:
        candidates["article"].append("article_schema")
    if "BlogPosting" in schema:
        candidates["blog_post"].append("blog_schema")
    if _BLOG_RE.search(path):
        candidates["blog_post"].append("blog_url")
        if schema & _ARTICLE_SCHEMA:
            candidates["blog_post"].append("article_in_blog_section")
    if schema & _PRODUCT_SCHEMA:
        candidates["product"].append("product_or_service_schema")
    if _PRODUCT_RE.search(path):
        candidates["product"].append("product_url")
    if _SERVICE_RE.search(path) or "service" in blob:
        candidates["service"].append("service_url_or_text")
    if schema & _FAQ_SCHEMA or _FAQ_RE.search(path):
        candidates["faq"].append("faq_howto_signal")
    if _DOCS_RE.search(path):
        candidates["docs"].append("docs_url")
    if _CONTACT_RE.search(path) or "contact" in blob:
        candidates["contact"].append("contact_signal")
    if _ABOUT_RE.search(path) or any(word in blob for word in ("about us", "our team", "company")):
        candidates["about"].append("about_signal")
    if _LEGAL_RE.search(path):
        candidates["legal"].append("legal_url")
    if _SEARCH_RE.search(path):
        candidates["search"].append("search_url")
    if _LISTING_RE.search(path):
        candidates["listing"].append("listing_url")

    question_headings = sum(
        1 for heading in headings
        if _QUESTION_RE.search(str(heading.get("text", "")))
    )
    if question_headings >= 3:
        candidates["faq"].append("question_headings")
    if page.list_count >= 3 and page.word_count < 900 and len(paragraphs) <= 5:
        candidates["listing"].append("many_lists_low_body")
    if page.table_count >= 1 and any(word in blob for word in ("pricing", "compare", "specification", "plans")):
        candidates["product"].append("commercial_table")
    if page.has_dates and page.word_count >= 500 and not candidates.get("product"):
        candidates["article"].append("dated_longform")
    if page.word_count >= 700 and len(paragraphs) >= 4 and not _is_home(path):
        candidates["article"].append("longform_body")
    if schema & _LOCAL_SCHEMA and _CONTACT_RE.search(path):
        candidates["contact"].append("local_business_contact")

    return candidates


def classify_page(page: ExtractedPage) -> PageTypeRow:
    """Classify one extracted page."""
    candidates = _score_candidates(page)
    priority = [
        "home", "contact", "legal", "search", "faq", "product", "service",
        "docs", "blog_post", "article", "listing", "about",
    ]
    weighted = {
        page_type: len(signals) + (1 if page_type in {"home", "contact", "legal", "search"} else 0)
        for page_type, signals in candidates.items()
    }
    page_type = "other"
    if weighted:
        page_type = max(weighted, key=lambda name: (weighted[name], -priority.index(name) if name in priority else -99))
    signals = sorted(set(candidates.get(page_type, [])))
    confidence = min(0.95, 0.45 + 0.15 * len(signals)) if signals else 0.35
    template_family = _template_family(page_type)
    signature = template_signature(page, page_type, template_family)
    return PageTypeRow(
        url=page.url,
        title=page.title,
        page_type=page_type,
        template_family=template_family,
        template_signature=signature,
        confidence=round(confidence, 2),
        signals=signals,
    )


def _template_family(page_type: str) -> str:
    if page_type == "home":
        return "home_template"
    if page_type in {"article", "blog_post", "docs", "faq"}:
        return "content_template"
    if page_type in {"product", "service"}:
        return "commercial_template"
    if page_type in {"listing", "search"}:
        return "listing_template"
    if page_type in {"contact", "about"}:
        return "company_template"
    if page_type == "legal":
        return "legal_template"
    return "generic_template"


def template_signature(page: ExtractedPage, page_type: str, template_family: str) -> str:
    """Build a stable structural fingerprint for template grouping."""
    h1_count = page.h1_count or (1 if page.h1 else 0)
    headers = page.headers_rich or []
    header_levels = Counter(int(h.get("level", 0)) for h in headers if h.get("level"))
    schema_bucket = "+".join(sorted(_schema_set(page))[:3]) or "no_schema"
    return "|".join([
        template_family,
        page_type,
        schema_bucket,
        f"h1:{_bucket(h1_count, (0, 1, 2))}",
        f"h2:{_bucket(header_levels.get(2, 0), (0, 2, 6))}",
        f"h3:{_bucket(header_levels.get(3, 0), (0, 4, 10))}",
        f"lists:{_bucket(page.list_count, (0, 1, 4))}",
        f"tables:{_bucket(page.table_count, (0, 1, 3))}",
        f"paras:{_bucket(len(page.paragraphs or []), (0, 3, 8))}",
        f"words:{_bucket(page.word_count, (250, 750, 1500))}",
    ])


def analyze(pages: Iterable[ExtractedPage]) -> PageTypeReport:
    page_list = list(pages)
    rows = [classify_page(page) for page in page_list]
    type_counts_counter = Counter(row.page_type for row in rows)
    template_counts_counter = Counter(row.template_family for row in rows)
    signature_groups: dict[str, list[PageTypeRow]] = defaultdict(list)
    for row in rows:
        signature_groups[row.template_signature].append(row)

    per_page = [
        {
            "url": row.url,
            "title": row.title,
            "page_type": row.page_type,
            "template_family": row.template_family,
            "template_signature": row.template_signature,
            "confidence": row.confidence,
            "signals": row.signals,
        }
        for row in rows
    ]
    per_page.sort(key=lambda row: (row["page_type"], row["url"]))

    total_pages = len(rows)
    type_counts = [
        {"page_type": page_type, "pages": count, "share": count / total_pages if total_pages else 0.0}
        for page_type, count in type_counts_counter.most_common()
    ]
    template_counts = [
        {
            "template_family": family,
            "pages": count,
            "share": count / total_pages if total_pages else 0.0,
            "signatures": sum(
                1 for signature, group in signature_groups.items()
                if group and group[0].template_family == family
            ),
        }
        for family, count in template_counts_counter.most_common()
    ]
    summary = {
        "total_pages": total_pages,
        "page_type_count": len(type_counts_counter),
        "template_family_count": len(template_counts_counter),
        "template_signature_count": len(signature_groups),
        "dominant_page_type": type_counts[0]["page_type"] if type_counts else "",
        "dominant_template_family": template_counts[0]["template_family"] if template_counts else "",
    }
    return PageTypeReport(
        summary=summary,
        type_counts=type_counts,
        template_counts=template_counts,
        per_page=per_page,
    )


def to_payload(report: PageTypeReport) -> dict:
    return {
        "summary": report.summary,
        "type_counts": report.type_counts,
        "template_counts": report.template_counts,
        "per_page": report.per_page,
    }
