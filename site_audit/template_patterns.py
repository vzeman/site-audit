"""Mine structural template patterns from high-performing pages."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from .analyzer import PageInfo
from .extractor import ExtractedPage
from .paragraph_impact import _match_page, _page_lookup, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_QUESTION_RE = re.compile(r"\b(what|why|how|when|where|who|which|can|should|does|do|is|are)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|alternatives?|pricing|plans?|features?)\b", re.I)
_EXAMPLE_RE = re.compile(r"\b(example|examples|case stud(?:y|ies)|template|workflow)\b", re.I)
_USE_CASE_RE = re.compile(r"\b(use cases?|solutions?|industries|roles?)\b", re.I)
_INTEGRATION_RE = re.compile(r"\b(integrations?|connectors?|api|apps?)\b", re.I)


@dataclass(frozen=True)
class _Feature:
    key: str
    label: str
    category: str
    description: str
    recommendation: str
    predicate: Callable[[dict], bool]


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _share(part: float, whole: float) -> float:
    return float(part / whole) if whole else 0.0


def _directory(url: str, fallback: str = "") -> str:
    parsed = urlparse(url)
    parts = [p for p in (parsed.path or "/").split("/") if p]
    if parts:
        return parts[0].lower()
    return fallback or "/"


def _header_text(page: ExtractedPage) -> str:
    parts = [page.h1, *list(page.headings or [])]
    return " ".join(part for part in parts if part)


def _page_type_rows(page_types: dict | None) -> dict[str, dict]:
    rows = {}
    for row in (page_types or {}).get("per_page") or []:
        url = str(row.get("url") or "")
        if url:
            rows[url] = row
    return rows


def _search_lookup(pages: list[PageInfo], search_payload: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    out: dict[int, dict] = defaultdict(lambda: {
        "traffic": 0,
        "keywords": 0,
        "traffic_value": 0.0,
        "top_keyword": "",
        "top_position": 0,
        "top3_keywords": 0,
    })

    for row in (search_payload or {}).get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        traffic = _to_int(row.get("traffic"))
        if traffic >= out[page_i]["traffic"]:
            out[page_i]["traffic"] = traffic
            out[page_i]["top_keyword"] = row.get("top_keyword") or out[page_i]["top_keyword"]
            out[page_i]["top_position"] = _to_int(row.get("top_keyword_position") or row.get("position"))
        out[page_i]["keywords"] = max(out[page_i]["keywords"], _to_int(row.get("keywords") or row.get("keywords_total")))
        out[page_i]["traffic_value"] = max(out[page_i]["traffic_value"], _to_float(row.get("value") or row.get("traffic_value")))

    keyword_seen: dict[int, set[str]] = defaultdict(set)
    for row in (search_payload or {}).get("organic_keywords") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        keyword = str(row.get("keyword") or "").strip().lower()
        if keyword:
            keyword_seen[page_i].add(keyword)
        position = _to_int(row.get("position"))
        if 0 < position <= 3:
            out[page_i]["top3_keywords"] += 1
        traffic = _to_int(row.get("traffic"))
        if traffic > out[page_i]["traffic"] and not out[page_i]["top_keyword"]:
            out[page_i]["top_keyword"] = row.get("keyword") or ""

    for page_i, keywords in keyword_seen.items():
        out[page_i]["keywords"] = max(out[page_i]["keywords"], len(keywords))
    return out


def _link_lookup(pages: list[PageInfo], linkgraph: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    out: dict[int, dict] = defaultdict(lambda: {
        "in_degree": 0,
        "out_degree": 0,
        "click_depth": None,
        "authority_score": 0.0,
        "pagerank": 0.0,
    })
    if not isinstance(linkgraph, dict):
        return out
    for source in ("page_link_counts", "top_authority_pages", "top_hits_authorities", "top_hits_hubs"):
        for row in linkgraph.get(source) or []:
            page_i = _match_page(row.get("url") or "", lookup)
            if page_i is None:
                continue
            out[page_i]["in_degree"] = max(out[page_i]["in_degree"], _to_int(row.get("in_degree")))
            out[page_i]["out_degree"] = max(out[page_i]["out_degree"], _to_int(row.get("out_degree")))
            if row.get("click_depth") is not None:
                depth = _to_int(row.get("click_depth"))
                current = out[page_i]["click_depth"]
                out[page_i]["click_depth"] = depth if current is None else min(current, depth)
            out[page_i]["authority_score"] = max(out[page_i]["authority_score"], _to_float(row.get("authority_score")))
            out[page_i]["pagerank"] = max(out[page_i]["pagerank"], _to_float(row.get("pagerank")))
    return out


FEATURES: tuple[_Feature, ...] = (
    _Feature(
        "article_schema",
        "Article schema",
        "schema",
        "Page uses Article, BlogPosting, NewsArticle, or TechArticle schema.",
        "Add article-style schema with headline, date, author, and canonical URL.",
        lambda r: bool(set(r["schema_types"]) & {"Article", "BlogPosting", "NewsArticle", "TechArticle"}),
    ),
    _Feature(
        "faq_schema",
        "FAQ schema",
        "schema",
        "Page uses FAQPage or QAPage schema.",
        "Add FAQ schema for answer-style sections when the page has real question/answer content.",
        lambda r: bool(set(r["schema_types"]) & {"FAQPage", "QAPage"}),
    ),
    _Feature(
        "howto_schema",
        "HowTo schema",
        "schema",
        "Page uses HowTo schema for procedural content.",
        "Add HowTo schema to pages that explain a repeatable process step by step.",
        lambda r: "HowTo" in set(r["schema_types"]),
    ),
    _Feature(
        "question_headings",
        "Question-led headings",
        "section",
        "Page has at least two question-style headings.",
        "Add a short FAQ or objection-handling section with question-led H2/H3 headings.",
        lambda r: r["question_headings"] >= 2,
    ),
    _Feature(
        "comparison_table",
        "Comparison/data table",
        "section",
        "Page contains at least one HTML table.",
        "Add a concise comparison, pricing, feature, or summary table where it helps scanning.",
        lambda r: r["table_count"] >= 1,
    ),
    _Feature(
        "scannable_lists",
        "Scannable lists",
        "section",
        "Page contains multiple ordered or unordered lists.",
        "Break dense sections into bullet lists for features, steps, benefits, or criteria.",
        lambda r: r["list_count"] >= 2,
    ),
    _Feature(
        "deep_h2_outline",
        "Deep H2 outline",
        "structure",
        "Page has at least four H2 sections.",
        "Expand the page template with clear H2 sections covering use cases, proof, objections, and next steps.",
        lambda r: r["h2_count"] >= 4,
    ),
    _Feature(
        "subsection_depth",
        "H3 subsection depth",
        "structure",
        "Page has at least four H3 subsections.",
        "Add H3-level subsections under broad H2s so long pages expose more specific search intents.",
        lambda r: r["h3_count"] >= 4,
    ),
    _Feature(
        "longform_depth",
        "Long-form depth",
        "structure",
        "Page has at least 1000 body words.",
        "Add enough unique explanatory copy, examples, and evidence to match the strongest pages in this template.",
        lambda r: r["word_count"] >= 1000,
    ),
    _Feature(
        "multi_paragraph_body",
        "Multi-section body",
        "structure",
        "Page has at least six extracted paragraph blocks.",
        "Split the body into multiple focused sections instead of one short generic block.",
        lambda r: r["paragraph_count"] >= 6,
    ),
    _Feature(
        "concise_intro",
        "Concise intro",
        "structure",
        "First paragraph is present and no longer than 90 words.",
        "Open with a direct, specific intro before moving into details.",
        lambda r: 0 < r["first_paragraph_words"] <= 90,
    ),
    _Feature(
        "proof_stats",
        "Proof statistics",
        "evidence",
        "Page contains at least two numbers with units, money, counts, or percentages.",
        "Add current stats, counts, benchmarks, or result metrics near claims that need proof.",
        lambda r: r["stat_count"] >= 2,
    ),
    _Feature(
        "external_citations",
        "External citations",
        "evidence",
        "Page links out to at least two external references.",
        "Add selective citations to authoritative sources for factual or market claims.",
        lambda r: r["external_link_count"] >= 2,
    ),
    _Feature(
        "examples_section",
        "Examples/case section",
        "section",
        "Headings mention examples, cases, templates, or workflows.",
        "Add concrete examples, templates, workflows, or mini case studies to make the page less generic.",
        lambda r: bool(_EXAMPLE_RE.search(r["heading_text"])),
    ),
    _Feature(
        "use_case_section",
        "Use-case section",
        "section",
        "Headings mention use cases, solutions, industries, or roles.",
        "Add a use-case section that maps the template to concrete audiences or jobs.",
        lambda r: bool(_USE_CASE_RE.search(r["heading_text"])),
    ),
    _Feature(
        "integration_section",
        "Integration/API section",
        "section",
        "Headings mention integrations, connectors, APIs, or apps.",
        "Add integration or API coverage when the topic depends on ecosystem fit.",
        lambda r: bool(_INTEGRATION_RE.search(r["heading_text"])),
    ),
    _Feature(
        "comparison_heading",
        "Comparison heading",
        "section",
        "Headings mention comparison, alternatives, pricing, plans, or features.",
        "Add a comparison/pricing/features section for commercial evaluation intent.",
        lambda r: bool(_COMPARE_RE.search(r["heading_text"])),
    ),
    _Feature(
        "primary_cta",
        "Primary CTA",
        "conversion",
        "Page has at least one primary CTA.",
        "Add one clear primary CTA that matches the search intent and page type.",
        lambda r: r["primary_cta_count"] >= 1,
    ),
    _Feature(
        "lead_capture",
        "Lead capture path",
        "conversion",
        "Page has a form, phone link, or email link.",
        "Add a low-friction contact, demo, quote, or signup path on pages with commercial intent.",
        lambda r: r["form_count"] >= 1 or r["contact_link_count"] >= 1,
    ),
    _Feature(
        "internal_link_hub",
        "Internal link hub",
        "internal_links",
        "Page links to at least four internal pages.",
        "Add contextual internal links to related pages, comparisons, docs, and conversion pages.",
        lambda r: r["out_degree"] >= 4,
    ),
    _Feature(
        "internally_promoted",
        "Internally promoted",
        "internal_links",
        "Page has at least three inbound internal links.",
        "Promote this template from relevant hubs, feature pages, and high-authority pages.",
        lambda r: r["in_degree"] >= 3,
    ),
    _Feature(
        "shallow_click_depth",
        "Shallow click depth",
        "internal_links",
        "Page is reachable within two clicks when depth is known.",
        "Move important pages closer to the home/hub navigation path.",
        lambda r: r["click_depth"] is not None and r["click_depth"] <= 2,
    ),
)


def _build_rows(
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    page_types: dict | None,
    search_payload: dict | None,
    linkgraph: dict | None,
) -> list[dict]:
    pt_rows = _page_type_rows(page_types)
    search = _search_lookup(pages, search_payload)
    links = _link_lookup(pages, linkgraph)

    rows: list[dict] = []
    for i, page in enumerate(pages):
        if i >= len(extracted_pages):
            continue
        ext = extracted_pages[i]
        pt = pt_rows.get(page.url, {})
        headers = list(ext.headers_rich or [])
        h2_count = sum(1 for h in headers if _to_int(h.get("level")) == 2)
        h3_count = sum(1 for h in headers if _to_int(h.get("level")) == 3)
        heading_text = _header_text(ext)
        first_para = (ext.paragraphs or [""])[0] if (ext.paragraphs or []) else ""
        conversion = ext.conversion_signals or {}
        link = links.get(i, {})
        traffic = _to_int(search.get(i, {}).get("traffic"))
        keywords = _to_int(search.get(i, {}).get("keywords"))
        authority = _to_float(link.get("authority_score")) + _to_float(link.get("pagerank")) * 100
        search_score = float(traffic) + math.sqrt(max(0, keywords)) * 8.0
        fallback_score = authority + min(30.0, _to_int(link.get("in_degree")) * 3.0)
        row = {
            "page_index": i,
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "directory": _directory(page.url, page.section),
            "page_type": str(pt.get("page_type") or "other"),
            "template_family": str(pt.get("template_family") or "generic_template"),
            "template_signature": str(pt.get("template_signature") or ""),
            "traffic": traffic,
            "keywords": keywords,
            "traffic_value": _to_float(search.get(i, {}).get("traffic_value")),
            "top_keyword": str(search.get(i, {}).get("top_keyword") or ""),
            "top3_keywords": _to_int(search.get(i, {}).get("top3_keywords")),
            "search_score": search_score,
            "fallback_score": fallback_score,
            "performance_score": search_score if search_score > 0 else fallback_score,
            "word_count": int(ext.word_count or page.word_count or 0),
            "paragraph_count": len(ext.paragraphs or []),
            "first_paragraph_words": len(_tokens(first_para)),
            "h2_count": h2_count,
            "h3_count": h3_count,
            "heading_count": len(headers),
            "question_headings": sum(1 for h in headers if _QUESTION_RE.search(str(h.get("text") or ""))),
            "heading_text": heading_text,
            "list_count": int(ext.list_count or 0),
            "table_count": int(ext.table_count or 0),
            "schema_types": list(ext.schema_types or []),
            "external_link_count": int(ext.external_link_count or 0),
            "stat_count": int(ext.stat_count or 0),
            "has_dates": bool(ext.has_dates or ext.date_published or ext.date_modified),
            "media_count": len(ext.media_items or []),
            "cta_count": _to_int(conversion.get("cta_count")),
            "primary_cta_count": _to_int(conversion.get("primary_cta_count")),
            "form_count": _to_int(conversion.get("form_count")),
            "contact_link_count": _to_int(conversion.get("contact_link_count")),
            "in_degree": _to_int(link.get("in_degree")),
            "out_degree": _to_int(link.get("out_degree")),
            "click_depth": link.get("click_depth"),
            "authority_score": _to_float(link.get("authority_score")),
        }
        features = [feature.key for feature in FEATURES if feature.predicate(row)]
        row["features"] = features
        rows.append(row)
    return rows


def _top_weak_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda r: (float(r.get("performance_score") or 0), _to_int(r.get("traffic"))), reverse=True)
    n = len(ordered)
    bucket = max(2, min(max(2, n // 3), n // 2))
    top = ordered[:bucket]
    weak = ordered[-bucket:]
    if top and weak and float(top[-1].get("performance_score") or 0) <= float(weak[0].get("performance_score") or 0):
        return [], []
    return top, weak


def _avg(rows: list[dict], key: str = "performance_score") -> float:
    if not rows:
        return 0.0
    return float(sum(float(r.get(key) or 0.0) for r in rows) / len(rows))


def _confidence(sample_size: int, top_rate: float, weak_rate: float, observed_lift: float, has_search: bool) -> float:
    contrast = max(0.0, top_rate - weak_rate)
    size_component = min(0.25, sample_size / 40.0)
    lift_component = min(0.2, max(0.0, observed_lift) * 0.08)
    base = 0.34 + size_component + contrast * 0.32 + lift_component
    if not has_search:
        base -= 0.12
    return round(max(0.0, min(0.95, base)), 3)


def _sample(row: dict) -> dict:
    return {
        "url": row["url"],
        "title": row["title"],
        "traffic": int(row.get("traffic") or 0),
        "keywords": int(row.get("keywords") or 0),
        "top_keyword": row.get("top_keyword") or "",
        "score": round(float(row.get("performance_score") or 0.0), 2),
        "features": row.get("features") or [],
    }


def _segment_key(row: dict, mode: str) -> tuple[str, str, str]:
    page_type = str(row.get("page_type") or "other")
    if mode == "page_type_directory":
        directory = str(row.get("directory") or "/")
        return mode, page_type, directory
    return mode, page_type, "*"


def _mine_segments(rows: list[dict], *, min_segment_size: int, top_n: int) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []

    by_dir: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_type: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_dir[_segment_key(row, "page_type_directory")].append(row)
        by_type[_segment_key(row, "page_type")].append(row)

    eligible_dir_page_types = {
        key[1]
        for key, group in by_dir.items()
        if len(group) >= min_segment_size
    }
    candidate_segments = [
        (key, group)
        for key, group in by_dir.items()
        if len(group) >= min_segment_size
    ]
    candidate_segments.extend(
        (key, group)
        for key, group in by_type.items()
        if len(group) >= min_segment_size and key[1] not in eligible_dir_page_types
    )

    patterns: list[dict] = []
    comparisons: list[dict] = []
    seen_patterns: set[tuple[str, str, str, str]] = set()
    has_search = any(float(row.get("search_score") or 0.0) > 0 for row in rows)

    feature_map = {feature.key: feature for feature in FEATURES}
    for key, segment_rows in candidate_segments:
        mode, page_type, directory = key
        top, weak = _top_weak_split(segment_rows)
        if len(top) < 2 or len(weak) < 2:
            continue
        top_features = Counter(feature for row in top for feature in row.get("features", []))
        weak_features = Counter(feature for row in weak for feature in row.get("features", []))
        segment_features = Counter(feature for row in segment_rows for feature in row.get("features", []))
        common_top = []
        for feature_key, count in top_features.items():
            feature = feature_map.get(feature_key)
            if not feature:
                continue
            top_rate = _share(count, len(top))
            weak_rate = _share(weak_features.get(feature_key, 0), len(weak))
            if top_rate >= 0.5:
                common_top.append({
                    "feature_key": feature_key,
                    "label": feature.label,
                    "category": feature.category,
                    "top_rate": round(top_rate, 3),
                    "weak_rate": round(weak_rate, 3),
                    "contrast": round(top_rate - weak_rate, 3),
                })
        common_top.sort(key=lambda r: (r["contrast"], r["top_rate"]), reverse=True)

        comparisons.append({
            "segment": {
                "mode": mode,
                "page_type": page_type,
                "directory": directory,
                "sample_size": len(segment_rows),
            },
            "top_pages": [_sample(row) for row in top[:8]],
            "weak_pages": [_sample(row) for row in weak[:8]],
            "avg_top_score": round(_avg(top), 2),
            "avg_weak_score": round(_avg(weak), 2),
            "common_top_features": common_top[:12],
        })

        for feature in FEATURES:
            pattern_key = (mode, page_type, directory, feature.key)
            if pattern_key in seen_patterns:
                continue
            top_present = [row for row in top if feature.key in row.get("features", [])]
            weak_present = [row for row in weak if feature.key in row.get("features", [])]
            weak_missing = [row for row in weak if feature.key not in row.get("features", [])]
            present = [row for row in segment_rows if feature.key in row.get("features", [])]
            absent = [row for row in segment_rows if feature.key not in row.get("features", [])]
            if not top_present or not weak_missing or len(present) < 2 or not absent:
                continue
            top_rate = _share(len(top_present), len(top))
            weak_rate = _share(len(weak_present), len(weak))
            contrast = top_rate - weak_rate
            avg_present = _avg(present)
            avg_absent = _avg(absent)
            observed_lift = (avg_present - avg_absent) / max(1.0, avg_absent)
            if top_rate < 0.5:
                continue
            if contrast < 0.34 and observed_lift < 0.25:
                continue
            conf = _confidence(len(segment_rows), top_rate, weak_rate, observed_lift, has_search)
            if conf < 0.45:
                continue
            seen_patterns.add(pattern_key)
            affected = sorted(weak_missing, key=lambda r: (float(r.get("performance_score") or 0), r.get("url") or ""))[:12]
            patterns.append({
                "pattern_id": f"template_{len(patterns) + 1}",
                "feature_key": feature.key,
                "label": feature.label,
                "category": feature.category,
                "description": feature.description,
                "recommendation": feature.recommendation,
                "page_type": page_type,
                "directory": directory,
                "segment_mode": mode,
                "sample_size": len(segment_rows),
                "top_sample_size": len(top),
                "weak_sample_size": len(weak),
                "top_rate": round(top_rate, 3),
                "weak_rate": round(weak_rate, 3),
                "segment_rate": round(_share(segment_features.get(feature.key, 0), len(segment_rows)), 3),
                "observed_lift": round(max(-0.99, min(9.99, observed_lift)), 3),
                "lift_score_difference": round(avg_present - avg_absent, 2),
                "avg_score_with_feature": round(avg_present, 2),
                "avg_score_without_feature": round(avg_absent, 2),
                "confidence": conf,
                "sample_urls": [_sample(row) for row in sorted(top_present, key=lambda r: float(r.get("performance_score") or 0), reverse=True)[:8]],
                "affected_weak_pages": [_sample(row) for row in affected],
            })

    patterns.sort(key=lambda r: (r["confidence"], r["observed_lift"], r["sample_size"]), reverse=True)
    comparisons.sort(key=lambda r: (r["avg_top_score"] - r["avg_weak_score"], r["segment"]["sample_size"]), reverse=True)
    return patterns[:top_n], comparisons[:80], rows


def _recommendations(patterns: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        for page in pattern.get("affected_weak_pages") or []:
            rows.append({
                "url": page.get("url"),
                "title": page.get("title"),
                "page_type": pattern.get("page_type"),
                "directory": pattern.get("directory"),
                "missing_pattern": pattern.get("label"),
                "feature_key": pattern.get("feature_key"),
                "category": pattern.get("category"),
                "recommendation": pattern.get("recommendation"),
                "confidence": pattern.get("confidence"),
                "observed_lift": pattern.get("observed_lift"),
                "traffic": page.get("traffic", 0),
                "keywords": page.get("keywords", 0),
                "top_keyword": page.get("top_keyword", ""),
            })
    rows.sort(key=lambda r: (float(r.get("confidence") or 0), float(r.get("observed_lift") or 0)), reverse=True)
    return rows[:300]


def build_template_patterns(
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    *,
    page_types: dict | None = None,
    search_payload: dict | None = None,
    linkgraph: dict | None = None,
    min_segment_size: int = 4,
    top_n: int = 120,
) -> dict:
    """Return template success patterns and weak-page additions."""
    if not pages or not extracted_pages:
        return {
            "summary": {"status": "no_pages", "patterns": 0, "recommendations": 0},
            "patterns": [],
            "comparisons": [],
            "recommendations": [],
            "feature_catalog": [],
        }

    rows = _build_rows(pages, extracted_pages, page_types, search_payload, linkgraph)
    if not rows:
        return {
            "summary": {"status": "no_pages", "patterns": 0, "recommendations": 0},
            "patterns": [],
            "comparisons": [],
            "recommendations": [],
            "feature_catalog": [],
        }

    has_search = any(float(row.get("search_score") or 0.0) > 0 for row in rows)
    has_link_scores = any(float(row.get("fallback_score") or 0.0) > 0 for row in rows)
    if not has_search and not has_link_scores:
        return {
            "summary": {
                "status": "insufficient_performance_data",
                "total_pages": len(rows),
                "patterns": 0,
                "recommendations": 0,
                "min_segment_size": min_segment_size,
            },
            "patterns": [],
            "comparisons": [],
            "recommendations": [],
            "feature_catalog": [
                {"key": f.key, "label": f.label, "category": f.category, "description": f.description}
                for f in FEATURES
            ],
        }

    patterns, comparisons, rows = _mine_segments(rows, min_segment_size=min_segment_size, top_n=top_n)
    recs = _recommendations(patterns)
    feature_counts = Counter(pattern["feature_key"] for pattern in patterns)
    page_type_counts = Counter(pattern["page_type"] for pattern in patterns)
    directories = Counter(pattern["directory"] for pattern in patterns)
    status = "ok" if patterns else "no_patterns"
    summary = {
        "status": status,
        "model": "template_patterns_v1",
        "total_pages": len(rows),
        "scored_pages": sum(1 for row in rows if float(row.get("performance_score") or 0.0) > 0),
        "performance_source": "search" if has_search else "internal_link_authority",
        "patterns": len(patterns),
        "recommendations": len(recs),
        "segments_compared": len(comparisons),
        "min_segment_size": min_segment_size,
        "page_types_with_patterns": len(page_type_counts),
        "directories_with_patterns": len(directories),
        "median_confidence": round(sorted([p["confidence"] for p in patterns])[len(patterns) // 2], 3) if patterns else 0.0,
        "top_observed_lift": patterns[0]["observed_lift"] if patterns else 0.0,
    }
    return {
        "summary": summary,
        "patterns": patterns,
        "comparisons": comparisons,
        "recommendations": recs,
        "feature_catalog": [
            {
                "key": feature.key,
                "label": feature.label,
                "category": feature.category,
                "description": feature.description,
                "pattern_count": feature_counts.get(feature.key, 0),
            }
            for feature in FEATURES
        ],
        "top_page_types": [
            {"page_type": page_type, "patterns": count}
            for page_type, count in page_type_counts.most_common(20)
        ],
        "top_directories": [
            {"directory": directory, "patterns": count}
            for directory, count in directories.most_common(20)
        ],
    }
