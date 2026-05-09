"""E-E-A-T style trust and evidence signal audit."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from urllib.parse import urlparse
from typing import Any

from .analyzer import PageInfo
from .extractor import ExtractedPage
from .page_types import classify_page
from .paragraph_impact import _match_page, _page_lookup, _to_int

_AUTHOR_RE = re.compile(r"\b(by|author|written by|reviewed by|edited by|expert reviewer)\b", re.I)
_CASE_RE = re.compile(r"\b(case stud(?:y|ies)|customer story|testimonial|success story|customer|client)\b", re.I)
_PROOF_RE = re.compile(r"\b(data|study|research|benchmark|survey|report|tested|we tested|our analysis|according to)\b", re.I)
_COMMERCIAL_RE = re.compile(r"\b(price|pricing|plan|quote|demo|trial|guarantee|sla|security|compliance|integration)\b", re.I)
_REVIEW_RE = re.compile(r"\b(review|rating|stars?|g2|capterra|trustpilot)\b", re.I)
_SCREENSHOT_RE = re.compile(r"\b(screenshot|screen shot|dashboard|demo|walkthrough)\b", re.I)

_REFERENCES = [
    {
        "label": "Google helpful, reliable, people-first content",
        "url": "https://developers.google.com/search/docs/fundamentals/creating-helpful-content",
    },
    {
        "label": "Google link best practices",
        "url": "https://developers.google.com/search/docs/crawling-indexing/links-crawlable",
    },
    {
        "label": "Google structured data intro",
        "url": "https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data",
    },
]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _search_lookup(pages: list[PageInfo], search_payload: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    out: dict[int, dict] = defaultdict(lambda: {"traffic": 0, "keywords": 0, "cluster": "", "cluster_label": "", "top_keywords": []})
    for row in (search_payload or {}).get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        out[page_i]["traffic"] = max(out[page_i]["traffic"], _to_int(row.get("traffic")))
        out[page_i]["keywords"] = max(out[page_i]["keywords"], _to_int(row.get("keywords")))
        if row.get("cluster_label"):
            out[page_i]["cluster_label"] = row.get("cluster_label")
        if row.get("cluster") is not None:
            out[page_i]["cluster"] = row.get("cluster")
        if row.get("top_keyword") and len(out[page_i]["top_keywords"]) < 5:
            out[page_i]["top_keywords"].append({"keyword": row.get("top_keyword"), "traffic": _to_int(row.get("traffic"))})
    for row in (search_payload or {}).get("organic_keywords") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        out[page_i]["traffic"] += _to_int(row.get("traffic"))
        out[page_i]["keywords"] += 1
        if row.get("cluster_label"):
            out[page_i]["cluster_label"] = row.get("cluster_label")
        if row.get("cluster") is not None:
            out[page_i]["cluster"] = row.get("cluster")
        if row.get("keyword") and len(out[page_i]["top_keywords"]) < 5:
            out[page_i]["top_keywords"].append({"keyword": row.get("keyword"), "traffic": _to_int(row.get("traffic")), "position": _to_int(row.get("position"))})
    return out


def _path_bucket(url: str) -> str:
    parts = [p for p in (urlparse(url).path or "/").split("/") if p]
    return parts[0] if parts else "/"


def _text(page: ExtractedPage) -> str:
    return " ".join([page.title, page.description, page.h1, " ".join(page.headings or []), page.body or ""])


def _has_author(page: ExtractedPage) -> bool:
    if _AUTHOR_RE.search(_text(page)):
        return True
    return any(t in {"Person", "Author"} for t in (page.schema_types or []))


def _has_reviewer(page: ExtractedPage) -> bool:
    return bool(re.search(r"\b(reviewed by|expert reviewer|medically reviewed|fact checked)\b", _text(page), re.I))


def _media_proof(page: ExtractedPage) -> int:
    count = 0
    for item in page.media_items or []:
        haystack = f"{item.get('src', '')} {item.get('alt', '')} {item.get('title', '')}"
        if _SCREENSHOT_RE.search(haystack):
            count += 1
        elif item.get("type") in {"image", "video", "iframe"}:
            count += 1
    return count


def _component_scores(page: ExtractedPage, page_type: str) -> tuple[dict, dict, list[str]]:
    body = _text(page)
    conversion = page.conversion_signals or {}
    valid_blocks = [b for b in page.schema_blocks or [] if b.get("valid")]
    invalid_blocks = [b for b in page.schema_blocks or [] if not b.get("valid")]
    schema_types = set(page.schema_types or [])
    has_author = _has_author(page)
    has_reviewer = _has_reviewer(page)
    has_date = bool(page.date_published or page.date_modified or page.has_dates)
    citation_count = int(page.external_link_count or 0)
    stat_count = int(page.stat_count or 0)
    case_count = len(_CASE_RE.findall(body))
    proof_language = len(_PROOF_RE.findall(body))
    commercial_facts = len(_COMMERCIAL_RE.findall(body))
    review_mentions = len(_REVIEW_RE.findall(body))
    media_count = _media_proof(page)
    contact_links = _to_int(conversion.get("contact_link_count"))

    components = {
        "authorship": min(20.0, (8 if has_author else 0) + (4 if has_reviewer else 0) + (4 if has_date else 0) + (4 if schema_types & {"Person", "Organization"} else 0)),
        "evidence": min(25.0, min(8, citation_count * 2) + min(7, stat_count * 1.5) + min(5, proof_language) + min(5, media_count * 1.5)),
        "experience": min(15.0, min(6, case_count * 2) + min(5, review_mentions * 1.5) + min(4, media_count)),
        "transparency": min(15.0, (5 if contact_links else 0) + (5 if page_type in {"about", "contact", "home"} else 0) + (5 if schema_types & {"Organization", "LocalBusiness"} else 0)),
        "schema": min(15.0, (10 if valid_blocks or schema_types else 0) - min(8, len(invalid_blocks) * 4) + (5 if schema_types & {"Article", "BlogPosting", "FAQPage", "Product", "Organization", "SoftwareApplication"} else 0)),
        "commercial_proof": min(10.0, min(5, commercial_facts) + min(3, case_count) + min(2, review_mentions)),
    }
    components = {k: round(max(0.0, v), 2) for k, v in components.items()}
    facts = {
        "has_author": has_author,
        "has_reviewer": has_reviewer,
        "has_date": has_date,
        "citation_count": citation_count,
        "stat_count": stat_count,
        "case_study_mentions": case_count,
        "proof_language_mentions": proof_language,
        "commercial_fact_mentions": commercial_facts,
        "review_mentions": review_mentions,
        "media_proof_count": media_count,
        "contact_link_count": contact_links,
        "valid_schema_blocks": len(valid_blocks),
        "invalid_schema_blocks": len(invalid_blocks),
        "schema_types": sorted(schema_types),
    }
    missing = []
    if page_type in {"article", "blog_post", "docs"} and not has_author:
        missing.append("Add a visible author, reviewer, or editorial owner with credentials.")
    if page_type in {"article", "blog_post", "docs"} and not has_date:
        missing.append("Add visible published or updated dates for time-sensitive information.")
    if citation_count < 2 and (page.word_count or 0) >= 500:
        missing.append("Add selective citations to authoritative sources for factual claims.")
    if stat_count < 2 and _PROOF_RE.search(body):
        missing.append("Add concrete numbers, benchmarks, or measured results for proof-heavy claims.")
    if page_type in {"product", "service"} and case_count < 1:
        missing.append("Add verifiable customer proof such as a case study, testimonial, or named use case.")
    if media_count < 1 and page_type in {"product", "service", "docs"}:
        missing.append("Add screenshots, product images, or demo media that show firsthand experience.")
    if not valid_blocks and page_type in {"article", "blog_post", "product", "service", "faq"}:
        missing.append("Add valid structured data that matches the visible page purpose.")
    if invalid_blocks:
        missing.append("Fix invalid JSON-LD before adding more schema.")
    return components, facts, missing


def build_trust_signals(
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    *,
    search_payload: dict | None = None,
    top_n: int = 500,
) -> dict:
    if not pages or not extracted_pages:
        return {"summary": {"status": "no_pages", "total_pages": 0}, "pages": [], "clusters": [], "missing_evidence": [], "references": _REFERENCES}

    search = _search_lookup(pages, search_payload)
    rows = []
    for i, page in enumerate(pages):
        if i >= len(extracted_pages):
            continue
        ext = extracted_pages[i]
        page_type = classify_page(ext).page_type
        ctx = search.get(i, {})
        components, facts, missing = _component_scores(ext, page_type)
        score = round(sum(components.values()), 2)
        cluster = str(ctx.get("cluster_label") or ctx.get("cluster") or page.section or _path_bucket(page.url) or page_type)
        rows.append({
            "url": page.url,
            "title": page.title,
            "page_type": page_type,
            "section": page.section,
            "cluster": cluster,
            "traffic": _to_int(ctx.get("traffic")),
            "keywords": _to_int(ctx.get("keywords")),
            "top_keywords": ctx.get("top_keywords") or [],
            "trust_score": score,
            "components": components,
            "facts": facts,
            "missing_signals": missing,
            "priority": "high" if (_to_int(ctx.get("traffic")) >= 20 and score < 60) or (page_type in {"product", "service"} and score < 55) else ("medium" if score < 60 else "low"),
        })

    clusters = []
    missing_rows = []
    for cluster, group in defaultdict(list, {k: [r for r in rows if r["cluster"] == k] for k in {r["cluster"] for r in rows}}).items():
        ordered = sorted(group, key=lambda r: (_to_int(r.get("traffic")), _to_float(r.get("trust_score"))), reverse=True)
        leaders = ordered[: max(1, min(3, len(ordered)))]
        leader_components = Counter()
        for leader in leaders:
            for key, value in (leader.get("components") or {}).items():
                leader_components[key] += float(value)
        benchmark = {key: round(value / len(leaders), 2) for key, value in leader_components.items()}
        avg_score = sum(_to_float(r.get("trust_score")) for r in group) / max(1, len(group))
        clusters.append({
            "cluster": cluster,
            "pages": len(group),
            "traffic": sum(_to_int(r.get("traffic")) for r in group),
            "avg_trust_score": round(avg_score, 2),
            "leader_score": leaders[0]["trust_score"] if leaders else 0,
            "benchmark_components": benchmark,
            "leader_examples": [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "trust_score": r["trust_score"],
                    "traffic": r["traffic"],
                    "strong_signals": [k for k, v in (r.get("components") or {}).items() if v >= {"authorship": 12, "evidence": 12, "experience": 7, "transparency": 8, "schema": 10, "commercial_proof": 5}.get(k, 99)],
                }
                for r in leaders
            ],
        })
        for row in group:
            if row["priority"] == "low" or not row["missing_signals"]:
                continue
            examples = [leader for leader in leaders if leader["url"] != row["url"]][:2]
            for signal in row["missing_signals"][:6]:
                missing_rows.append({
                    "url": row["url"],
                    "title": row["title"],
                    "cluster": cluster,
                    "page_type": row["page_type"],
                    "traffic": row["traffic"],
                    "trust_score": row["trust_score"],
                    "priority": row["priority"],
                    "missing_signal": signal,
                    "recommendation": signal,
                    "stronger_examples": [
                        {"url": e["url"], "title": e["title"], "trust_score": e["trust_score"], "traffic": e["traffic"]}
                        for e in examples
                    ],
                })

    rows.sort(key=lambda r: ({"high": 2, "medium": 1, "low": 0}.get(r["priority"], 0), _to_int(r.get("traffic")), -_to_float(r.get("trust_score"))), reverse=True)
    clusters.sort(key=lambda r: (_to_int(r.get("traffic")), _to_float(r.get("leader_score"))), reverse=True)
    missing_rows.sort(key=lambda r: ({"high": 2, "medium": 1}.get(r["priority"], 0), _to_int(r.get("traffic"))), reverse=True)
    total = len(rows)
    avg_score = sum(_to_float(r.get("trust_score")) for r in rows) / total if total else 0.0
    component_avgs = {}
    for key in ("authorship", "evidence", "experience", "transparency", "schema", "commercial_proof"):
        component_avgs[key] = round(sum(_to_float((r.get("components") or {}).get(key)) for r in rows) / total, 2) if total else 0.0
    return {
        "summary": {
            "status": "ok",
            "model": "trust_signals_v1",
            "total_pages": total,
            "avg_trust_score": round(avg_score, 2),
            "high_priority_pages": sum(1 for r in rows if r["priority"] == "high"),
            "medium_priority_pages": sum(1 for r in rows if r["priority"] == "medium"),
            "missing_evidence_items": len(missing_rows),
            "clusters": len(clusters),
            "component_averages": component_avgs,
        },
        "pages": rows[:top_n],
        "clusters": clusters[:160],
        "missing_evidence": missing_rows[:500],
        "references": _REFERENCES,
        "interpretation": {
            "score": "0-100 sum of visible authorship, evidence, experience, transparency, schema, and commercial proof signals.",
            "recommendations": "Recommendations request verifiable additions only; do not add unsupported author, review, customer, or performance claims.",
        },
    }
