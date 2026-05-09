"""Structured-data health analysis.

The extractor keeps lightweight JSON-LD diagnostics per page. This module
turns those page-level signals into a report payload and comparison metrics:
coverage, invalid blocks, type mix, and common missing recommended fields.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from .extractor import ExtractedPage
from .page_types import classify_page


_RECOMMENDED_PROPS: dict[str, set[str]] = {
    "Article": {"headline", "datePublished", "author"},
    "NewsArticle": {"headline", "datePublished", "author"},
    "BlogPosting": {"headline", "datePublished", "author"},
    "FAQPage": {"mainEntity"},
    "QAPage": {"mainEntity"},
    "HowTo": {"name", "step"},
    "Product": {"name"},
    "Organization": {"name", "url"},
    "LocalBusiness": {"name", "address"},
    "BreadcrumbList": {"itemListElement"},
    "VideoObject": {"name", "uploadDate"},
}


@dataclass
class StructuredDataReport:
    summary: dict
    top_types: list[dict]
    invalid_blocks: list[dict]
    missing_recommended: list[dict]
    per_page: list[dict]
    opportunities: list[dict]
    clusters: list[dict]
    references: list[dict]


_GOOGLE_REFERENCES = [
    {
        "label": "Google structured data intro and supported features",
        "url": "https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data",
    },
    {
        "label": "Google helpful, reliable, people-first content",
        "url": "https://developers.google.com/search/docs/fundamentals/creating-helpful-content",
    },
]

_SCHEMA_GUIDES = {
    "Article": "https://developers.google.com/search/docs/appearance/structured-data/article",
    "BlogPosting": "https://developers.google.com/search/docs/appearance/structured-data/article",
    "NewsArticle": "https://developers.google.com/search/docs/appearance/structured-data/article",
    "FAQPage": "https://developers.google.com/search/docs/appearance/structured-data/faqpage",
    "QAPage": "https://developers.google.com/search/docs/appearance/structured-data/qapage",
    "HowTo": "https://developers.google.com/search/docs/appearance/structured-data/how-to",
    "Product": "https://developers.google.com/search/docs/appearance/structured-data/product-snippet",
    "Organization": "https://developers.google.com/search/docs/appearance/structured-data/organization",
    "LocalBusiness": "https://developers.google.com/search/docs/appearance/structured-data/local-business",
    "BreadcrumbList": "https://developers.google.com/search/docs/appearance/structured-data/breadcrumb",
    "VideoObject": "https://developers.google.com/search/docs/appearance/structured-data/video",
    "SoftwareApplication": "https://developers.google.com/search/docs/appearance/structured-data/software-app",
    "Review": "https://developers.google.com/search/docs/appearance/structured-data/review-snippet",
}

_QUESTION_RE = re.compile(r"\b(what|why|how|when|where|who|which|can|should|does|do|is|are)\b", re.I)
_HOWTO_RE = re.compile(r"\b(how to|steps?|step-by-step|tutorial|guide|setup|configure|install)\b", re.I)
_PRODUCT_RE = re.compile(r"\b(product|pricing|plans?|features?|software|app|platform|tool)\b", re.I)
_REVIEW_RE = re.compile(r"\b(review|rating|stars?|pros and cons|best|top)\b", re.I)
_PRICE_RE = re.compile(r"[$€£]\s?\d+|\b\d+(?:\.\d+)?\s?(?:usd|eur|gbp|dollars?|euros?)\b", re.I)


def _top_level_keys(block: dict) -> set[str]:
    if not isinstance(block, dict):
        return set()
    return set(block.get("keys") or [])


def _missing_recommended(types: list[str], keys: set[str]) -> list[dict]:
    missing: list[dict] = []
    for schema_type in types:
        wanted = _RECOMMENDED_PROPS.get(schema_type)
        if not wanted:
            continue
        absent = sorted(prop for prop in wanted if prop not in keys)
        if absent:
            missing.append({"type": schema_type, "missing": absent})
    return missing


def _path_depth(url: str) -> int:
    return len([part for part in (urlparse(url).path or "/").split("/") if part])


def _heading_blob(page: ExtractedPage) -> str:
    return " ".join([page.h1, *list(page.headings or [])]).strip()


def _page_search_context(search_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not search_payload:
        return out

    def add(url: str, row: dict) -> None:
        if not url:
            return
        key = url.rstrip("/")
        ctx = out.setdefault(key, {"traffic": 0, "keywords": 0, "intents": Counter(), "top_keywords": [], "cluster": "", "cluster_label": ""})
        ctx["traffic"] += _to_int(row.get("traffic"))
        ctx["keywords"] += 1
        for intent in row.get("intents") or []:
            ctx["intents"][str(intent)] += 1
        if row.get("keyword") and len(ctx["top_keywords"]) < 8:
            ctx["top_keywords"].append({"keyword": row.get("keyword"), "traffic": _to_int(row.get("traffic")), "position": _to_int(row.get("position"))})
        if row.get("cluster_label"):
            ctx["cluster_label"] = row.get("cluster_label")
        if row.get("cluster") is not None:
            ctx["cluster"] = row.get("cluster")

    for row in search_payload.get("organic_keywords") or []:
        add(row.get("matched_url") or row.get("url") or "", row)
    for row in search_payload.get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        key = url.rstrip("/")
        ctx = out.setdefault(key, {"traffic": 0, "keywords": 0, "intents": Counter(), "top_keywords": [], "cluster": "", "cluster_label": ""})
        ctx["traffic"] = max(_to_int(row.get("traffic")), ctx["traffic"])
        ctx["keywords"] = max(_to_int(row.get("keywords")), ctx["keywords"])
        if row.get("top_keyword") and not ctx["top_keywords"]:
            ctx["top_keywords"].append({"keyword": row.get("top_keyword"), "traffic": _to_int(row.get("traffic")), "position": _to_int(row.get("top_keyword_position"))})
        if row.get("cluster_label"):
            ctx["cluster_label"] = row.get("cluster_label")
        if row.get("cluster") is not None:
            ctx["cluster"] = row.get("cluster")
    return out


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _valid_keys(page: ExtractedPage) -> set[str]:
    keys: set[str] = set()
    for block in page.schema_blocks or []:
        if block.get("valid"):
            keys.update(_top_level_keys(block))
    return keys


def _candidate_schemas(page: ExtractedPage, page_type: str, keyword_intents: list[str]) -> list[dict]:
    heading_blob = _heading_blob(page)
    body_blob = f"{page.title} {page.description} {heading_blob} {page.body}".lower()
    candidates: list[dict] = []

    def add(schema_type: str, reason: str, required: list[str], evidence: dict[str, bool], signals: list[str], primary: bool = False) -> None:
        candidates.append({
            "schema_type": schema_type,
            "reason": reason,
            "required_evidence": required,
            "evidence": evidence,
            "source_signals": signals,
            "primary": primary,
        })

    if page_type in {"article", "blog_post", "docs"} or page.word_count >= 700 or page.has_dates:
        add(
            "Article" if page_type != "blog_post" else "BlogPosting",
            "Long-form or dated informational content should expose article metadata.",
            ["headline or title", "published or modified date", "author or organization"],
            {
                "headline or title": bool(page.title or page.h1),
                "published or modified date": bool(page.date_published or page.date_modified or page.has_dates),
                "author or organization": bool(any("Organization" in t for t in page.schema_types or [])),
            },
            ["page_type:" + page_type, "word_count" if page.word_count >= 700 else "", "date" if page.has_dates else ""],
            primary=page_type in {"article", "blog_post"},
        )
    question_headings = sum(1 for h in (page.headers_rich or []) if _QUESTION_RE.search(str(h.get("text") or "")))
    if page_type == "faq" or question_headings >= 2:
        add(
            "FAQPage",
            "Question-led sections indicate FAQ content that may qualify when answers are visible on the page.",
            ["visible questions", "visible answers for each question"],
            {
                "visible questions": question_headings >= 2,
                "visible answers for each question": len(page.paragraphs or []) >= question_headings,
            },
            ["question_headings", "page_type:" + page_type],
            primary=page_type == "faq",
        )
    if _HOWTO_RE.search(body_blob) and (page.list_count or question_headings):
        add(
            "HowTo",
            "Step-by-step language or guide structure suggests procedural content.",
            ["process name", "ordered steps"],
            {
                "process name": bool(page.title or page.h1),
                "ordered steps": bool(page.list_count or len(page.paragraphs or []) >= 3),
            },
            ["how_to_language", "lists" if page.list_count else ""],
        )
    if page_type in {"product", "service"} or _PRODUCT_RE.search(body_blob):
        schema_type = "SoftwareApplication" if re.search(r"\b(software|app|platform|tool|api)\b", body_blob) else "Product"
        add(
            schema_type,
            "Commercial product/service pages should identify the product or application being offered.",
            ["name", "description", "offer or pricing evidence"],
            {
                "name": bool(page.title or page.h1),
                "description": bool(page.description or page.word_count >= 120),
                "offer or pricing evidence": bool(_PRICE_RE.search(body_blob) or re.search(r"\b(pricing|plans?|quote|demo)\b", body_blob)),
            },
            ["commercial_page", "intent:" + ",".join(keyword_intents[:3]) if keyword_intents else ""],
            primary=page_type in {"product", "service"},
        )
    if page_type in {"home", "about", "contact"}:
        add(
            "Organization",
            "Company-level pages should identify the organization and canonical URL.",
            ["organization name", "canonical website URL"],
            {
                "organization name": bool(page.title or page.h1),
                "canonical website URL": bool(page.canonical_url or page.url),
            },
            ["company_page"],
            primary=page_type in {"home", "about"},
        )
    if page_type == "contact" and (page.conversion_signals or {}).get("contact_link_count"):
        add(
            "LocalBusiness",
            "Contact pages with direct contact details can support local business markup when address details are present.",
            ["business name", "phone/email", "address if local"],
            {
                "business name": bool(page.title or page.h1),
                "phone/email": bool((page.conversion_signals or {}).get("contact_link_count")),
                "address if local": bool(re.search(r"\b(address|street|city|zip|postal)\b", body_blob)),
            },
            ["contact_links"],
        )
    if _path_depth(page.url) >= 2:
        add(
            "BreadcrumbList",
            "Deep URLs should expose breadcrumb hierarchy when visible navigation supports it.",
            ["visible or inferable breadcrumb path"],
            {"visible or inferable breadcrumb path": True},
            ["url_depth"],
        )
    if any(item.get("type") in {"video", "iframe"} for item in page.media_items or []):
        add(
            "VideoObject",
            "Video or embedded media was detected on the page.",
            ["video name", "thumbnail or embed URL", "upload date when available"],
            {
                "video name": bool(page.title or page.h1),
                "thumbnail or embed URL": True,
                "upload date when available": bool(page.has_dates),
            },
            ["video_media"],
        )
    if _REVIEW_RE.search(body_blob):
        add(
            "Review",
            "Review/rating language was detected; only add review markup when review content is first-party and visible.",
            ["review subject", "rating/review text"],
            {
                "review subject": bool(page.title or page.h1),
                "rating/review text": bool(re.search(r"\b\d(?:\.\d)?\s*/\s*5\b|\bstars?\b|rating", body_blob)),
            },
            ["review_language"],
        )
    return candidates


def _opportunities(page_list: list[ExtractedPage], search_payload: dict | None) -> tuple[list[dict], list[dict]]:
    search = _page_search_context(search_payload)
    rows: list[dict] = []
    cluster_stats: dict[str, dict] = {}
    for page in page_list:
        page_type = classify_page(page).page_type
        ctx = search.get(page.url.rstrip("/"), {})
        intents = [intent for intent, _ in (ctx.get("intents") or Counter()).most_common(4)]
        existing = set(page.schema_types or [])
        keys = _valid_keys(page)
        invalid = [block for block in page.schema_blocks or [] if not block.get("valid")]
        candidates = _candidate_schemas(page, page_type, intents)
        candidate_types = {candidate["schema_type"] for candidate in candidates}
        for schema_type in sorted(existing):
            if schema_type in _RECOMMENDED_PROPS and schema_type not in candidate_types:
                props = sorted(_RECOMMENDED_PROPS[schema_type])
                candidates.append({
                    "schema_type": schema_type,
                    "reason": "Existing structured data is present but recommended properties should be completed.",
                    "required_evidence": props,
                    "evidence": {prop: prop in keys for prop in props},
                    "source_signals": ["existing_schema"],
                    "primary": False,
                })

        for block in invalid:
            rows.append({
                "url": page.url,
                "title": page.title,
                "schema_type": "JSON-LD",
                "recommendation_type": "fix_invalid_jsonld",
                "priority": "high",
                "reason": "Invalid JSON-LD blocks are ignored and can hide otherwise useful schema.",
                "required_evidence": ["valid JSON-LD syntax", "schema object with @type"],
                "present_evidence": [],
                "missing_evidence": ["valid JSON-LD syntax"],
                "target_url": page.url,
                "page_type": page_type,
                "keyword_intents": intents,
                "traffic": _to_int(ctx.get("traffic")),
                "guideline_url": _GOOGLE_REFERENCES[0]["url"],
                "source_signals": ["invalid_jsonld"],
                "existing_schema_types": sorted(existing),
                "invalid_diagnostics": [{"format": block.get("format", "json-ld"), "error": block.get("error", "")}],
            })

        for candidate in candidates:
            schema_type = candidate["schema_type"]
            evidence = candidate["evidence"]
            missing_evidence = [name for name, present in evidence.items() if not present]
            missing_props = sorted(prop for prop in _RECOMMENDED_PROPS.get(schema_type, set()) if prop not in keys)
            if schema_type not in existing:
                rec_type = "add_primary_schema" if candidate.get("primary") else "add_supporting_schema"
                priority = "high" if candidate.get("primary") else "medium"
            elif missing_evidence or missing_props:
                rec_type = "complete_schema_fields"
                priority = "medium"
            else:
                continue
            rows.append({
                "url": page.url,
                "title": page.title,
                "schema_type": schema_type,
                "recommendation_type": rec_type,
                "priority": priority,
                "reason": candidate["reason"],
                "required_evidence": candidate["required_evidence"],
                "present_evidence": [name for name, present in evidence.items() if present],
                "missing_evidence": missing_evidence,
                "missing_recommended_properties": missing_props,
                "target_url": page.url,
                "page_type": page_type,
                "keyword_intents": intents,
                "traffic": _to_int(ctx.get("traffic")),
                "top_keywords": ctx.get("top_keywords") or [],
                "guideline_url": _SCHEMA_GUIDES.get(schema_type, _GOOGLE_REFERENCES[0]["url"]),
                "google_reference": "Google Search Central structured data documentation",
                "source_signals": [s for s in candidate["source_signals"] if s],
                "existing_schema_types": sorted(existing),
                "invalid_diagnostics": [],
            })

        cluster = str(ctx.get("cluster_label") or ctx.get("cluster") or page_type or "unclustered")
        stat = cluster_stats.setdefault(cluster, {
            "cluster": cluster,
            "pages": 0,
            "traffic": 0,
            "pages_with_schema": 0,
            "invalid_blocks": 0,
            "opportunities": 0,
            "schema_types": Counter(),
        })
        stat["pages"] += 1
        stat["traffic"] += _to_int(ctx.get("traffic"))
        stat["pages_with_schema"] += 1 if existing else 0
        stat["invalid_blocks"] += len(invalid)
        stat["schema_types"].update(existing)

    by_url = Counter(row["url"] for row in rows)
    for cluster, stat in cluster_stats.items():
        stat["opportunities"] = sum(by_url.get(page.url, 0) for page in page_list if str(search.get(page.url.rstrip("/"), {}).get("cluster_label") or search.get(page.url.rstrip("/"), {}).get("cluster") or classify_page(page).page_type or "unclustered") == cluster)

    clusters = []
    for stat in cluster_stats.values():
        pages = max(1, stat["pages"])
        clusters.append({
            "cluster": stat["cluster"],
            "pages": stat["pages"],
            "traffic": stat["traffic"],
            "pages_with_schema": stat["pages_with_schema"],
            "schema_coverage": stat["pages_with_schema"] / pages,
            "invalid_blocks": stat["invalid_blocks"],
            "opportunities": stat["opportunities"],
            "top_schema_types": [{"type": t, "pages": c} for t, c in stat["schema_types"].most_common(8)],
        })
    rows.sort(key=lambda r: ({"high": 2, "medium": 1}.get(r.get("priority"), 0), _to_int(r.get("traffic"))), reverse=True)
    clusters.sort(key=lambda r: (_to_int(r.get("traffic")), _to_int(r.get("opportunities"))), reverse=True)
    return rows[:500], clusters[:120]


def analyze(pages: Iterable[ExtractedPage], search_payload: dict | None = None) -> StructuredDataReport:
    page_list = list(pages)
    type_counts: Counter[str] = Counter()
    invalid_blocks: list[dict] = []
    missing_rows: list[dict] = []
    per_page: list[dict] = []
    pages_with_schema = 0
    pages_with_invalid = 0
    valid_blocks = 0
    invalid_count = 0

    for page in page_list:
        blocks = list(page.schema_blocks or [])
        types = sorted(set(page.schema_types or []))
        valid_page_blocks = [block for block in blocks if block.get("valid")]
        invalid_page_blocks = [block for block in blocks if not block.get("valid")]
        if types or valid_page_blocks:
            pages_with_schema += 1
        if invalid_page_blocks:
            pages_with_invalid += 1
        valid_blocks += len(valid_page_blocks)
        invalid_count += len(invalid_page_blocks)
        type_counts.update(types)

        keys: set[str] = set()
        for block in valid_page_blocks:
            keys.update(_top_level_keys(block))
        missing = _missing_recommended(types, keys)
        if missing:
            missing_rows.append({
                "url": page.url,
                "title": page.title,
                "types": types,
                "missing": missing,
            })
        for block in invalid_page_blocks:
            invalid_blocks.append({
                "url": page.url,
                "title": page.title,
                "format": block.get("format", "json-ld"),
                "error": block.get("error", ""),
            })

        per_page.append({
            "url": page.url,
            "title": page.title,
            "types": types,
            "valid_blocks": len(valid_page_blocks),
            "invalid_blocks": len(invalid_page_blocks),
            "missing_recommended": missing,
        })

    total_pages = len(page_list)
    summary = {
        "total_pages": total_pages,
        "pages_with_schema": pages_with_schema,
        "schema_coverage": pages_with_schema / total_pages if total_pages else 0.0,
        "valid_jsonld_blocks": valid_blocks,
        "invalid_jsonld_blocks": invalid_count,
        "pages_with_invalid_jsonld": pages_with_invalid,
        "schema_type_count": len(type_counts),
        "pages_missing_schema": max(0, total_pages - pages_with_schema),
    }
    top_types = [
        {"type": schema_type, "pages": count}
        for schema_type, count in type_counts.most_common()
    ]
    per_page.sort(key=lambda row: (-row["invalid_blocks"], not row["missing_recommended"], row["url"]))
    opportunities, clusters = _opportunities(page_list, search_payload)
    summary.update({
        "schema_opportunities": len(opportunities),
        "high_priority_schema_opportunities": sum(1 for row in opportunities if row.get("priority") == "high"),
        "schema_opportunity_clusters": len(clusters),
    })
    return StructuredDataReport(
        summary=summary,
        top_types=top_types,
        invalid_blocks=invalid_blocks[:200],
        missing_recommended=missing_rows[:200],
        per_page=per_page,
        opportunities=opportunities,
        clusters=clusters,
        references=_GOOGLE_REFERENCES,
    )


def to_payload(report: StructuredDataReport) -> dict:
    return {
        "summary": report.summary,
        "top_types": report.top_types,
        "invalid_blocks": report.invalid_blocks,
        "missing_recommended": report.missing_recommended,
        "per_page": report.per_page,
        "opportunities": report.opportunities,
        "clusters": report.clusters,
        "references": report.references,
    }
