"""Traffic-weighted freshness impact by page section."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable

import numpy as np

from .analyzer import PageInfo
from .paragraph_impact import _heading_for_paragraph, _match_page, _page_lookup, _to_int

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PRODUCT_RE = re.compile(
    r"\b(legacy|deprecated|outdated|old UI|old interface|classic|version\s+\d+(?:\.\d+)?|v\d+(?:\.\d+)?|"
    r"beta|as of\s+(?:19|20)\d{2}|current(?:ly)?|latest|newest)\b",
    re.I,
)

_VOLATILE_PATTERNS: list[tuple[str, str, int]] = [
    ("pricing", r"\b(price|pricing|cost|plans?|subscription|per user|license)\b", 22),
    ("comparison", r"\b(vs|versus|compare|comparison|alternative|alternatives|competitors?|best)\b", 20),
    ("integration", r"\b(integration|integrate|api|webhook|zapier|slack|salesforce|hubspot)\b", 18),
    ("how_to", r"\b(how to|setup|set up|configure|install|tutorial|guide|steps?)\b", 16),
    ("product_claim", r"\b(product|feature|release|workflow|automation|software|tool|platform)\b", 12),
]

_BUCKET_RISK = {
    "fresh": 0,
    "aging": 9,
    "stale": 30,
    "very_stale": 46,
    "unknown": 26,
    "future": 18,
}

SUPERFICIAL_WARNING = (
    "Do not change publication or modified dates unless the page was materially reviewed; "
    "update facts, screenshots, examples, integrations, pricing, or citations first."
)


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _cluster_label_lookup(cluster_summaries) -> dict[int, str]:
    out: dict[int, str] = {}
    for summary in cluster_summaries or []:
        cid = getattr(summary, "cluster_id", None)
        if cid is None:
            continue
        keywords = getattr(summary, "keywords", []) or []
        label = ", ".join(k.get("keyword", "") for k in keywords[:4] if k.get("keyword"))
        out[int(cid)] = label or f"cluster {cid}"
    return out


def _cluster_for(index: int, cluster_labels: list[int]) -> int:
    return int(cluster_labels[index]) if index < len(cluster_labels) else 0


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if len(s.strip()) >= 25]


def _snippet_for_match(text: str, regex: re.Pattern[str], *, limit: int = 4) -> list[dict]:
    rows: list[dict] = []
    for sentence in _sentences(text):
        match = regex.search(sentence)
        if not match:
            continue
        rows.append({"match": match.group(0), "snippet": sentence[:260]})
        if len(rows) >= limit:
            break
    return rows


def _stale_year_evidence(text: str, current_year: int, *, limit: int = 4) -> list[dict]:
    rows: list[dict] = []
    for sentence in _sentences(text):
        years = sorted({int(m.group(0)) for m in _YEAR_RE.finditer(sentence)})
        stale_years = [year for year in years if year <= current_year - 2]
        if not stale_years:
            continue
        rows.append({"match": ", ".join(str(y) for y in stale_years), "snippet": sentence[:260]})
        if len(rows) >= limit:
            break
    return rows


def _topic_class(text: str) -> tuple[str, int, list[str]]:
    haystack = text.lower()
    hits: list[tuple[str, int]] = []
    for label, pattern, score in _VOLATILE_PATTERNS:
        if re.search(pattern, haystack, re.I):
            hits.append((label, score))
    if not hits:
        return "general", 4, []
    hits.sort(key=lambda item: item[1], reverse=True)
    return hits[0][0], hits[0][1], [label for label, _ in hits[:4]]


def _search_context(pages: list[PageInfo], search_payload: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    contexts: dict[int, dict] = defaultdict(lambda: {
        "traffic": 0,
        "keywords": 0,
        "top_keyword": "",
        "keyword_rows": [],
    })
    payload = search_payload or {}
    for row in payload.get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        ctx = contexts[page_i]
        traffic = _to_int(row.get("traffic"))
        if traffic >= int(ctx["traffic"]):
            ctx["traffic"] = traffic
            ctx["top_keyword"] = row.get("top_keyword") or ctx["top_keyword"]
        ctx["keywords"] = max(int(ctx["keywords"]), _to_int(row.get("keywords")))
        if row.get("top_keyword"):
            ctx["keyword_rows"].append({
                "keyword": row.get("top_keyword"),
                "traffic": traffic,
                "position": _to_int(row.get("top_keyword_position")),
                "source": "top_page",
            })
    for row in payload.get("organic_keywords") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        ctx = contexts[page_i]
        traffic = _to_int(row.get("traffic"))
        ctx["traffic"] = max(int(ctx["traffic"]), traffic)
        keyword = row.get("keyword") or ""
        if keyword and (not ctx["top_keyword"] or traffic > _to_int((ctx["keyword_rows"] or [{}])[0].get("traffic"))):
            ctx["top_keyword"] = keyword
        ctx["keyword_rows"].append({
            "keyword": keyword,
            "traffic": traffic,
            "position": _to_int(row.get("position")),
            "serp_features": row.get("serp_features") or [],
            "source": "organic_keyword",
        })
    for ctx in contexts.values():
        ctx["keyword_rows"].sort(key=lambda r: _to_int(r.get("traffic")), reverse=True)
        ctx["keyword_rows"] = ctx["keyword_rows"][:12]
    return contexts


def _keyword_volatility(keyword_rows: list[dict]) -> tuple[int, list[str]]:
    score = 0
    evidence: list[str] = []
    for row in keyword_rows[:8]:
        keyword = str(row.get("keyword") or "")
        if not keyword:
            continue
        topic, topic_score, labels = _topic_class(keyword)
        if topic != "general":
            score += min(8, topic_score // 2)
            evidence.append(f"{keyword} ({topic})")
        pos = _to_int(row.get("position"))
        if 4 <= pos <= 20 and _to_int(row.get("traffic")) > 0:
            score += 3
            evidence.append(f"{keyword} ranks position {pos}")
        for feature in row.get("serp_features") or []:
            if str(feature).lower() in {"ai_overview", "top_stories", "reviews", "shopping", "video", "question"}:
                score += 2
                evidence.append(f"{keyword} SERP feature {feature}")
    return min(18, score), evidence[:8]


def _date_evidence(freshness_row: dict) -> list[str]:
    bucket = freshness_row.get("bucket") or "unknown"
    date_value = freshness_row.get("date") or "missing date"
    source = freshness_row.get("date_source") or "no date source"
    age = freshness_row.get("age_days")
    if age is None:
        return [f"{date_value} · {bucket} · {source}"]
    return [f"{date_value} · {int(age):,} days old · {bucket} · {source}"]


def _recommendation(row: dict) -> str:
    topic = row.get("topic_class") or ""
    if row.get("stale_year_evidence"):
        if topic == "pricing":
            return "Replace dated year-specific pricing or plan claims and verify current offers before updating the visible date."
        if topic == "integration":
            return "Replace dated year-specific integration or API claims and verify current steps before updating the visible date."
        if topic == "comparison":
            return "Replace dated year-specific comparison claims and recheck current competitors before updating the visible date."
        return "Replace dated year-specific claims and verify the current facts before updating the visible date."
    if topic == "pricing":
        return "Review pricing, plan, and packaging details; update screenshots or tables only after validating current offers."
    if topic == "integration":
        return "Verify current integration steps, API names, permissions, and screenshots; then update the modified date."
    if topic == "comparison":
        return "Recheck competitor/product claims, alternatives, rankings, and examples; cite what changed."
    if topic == "how_to":
        return "Retest the workflow and refresh steps, UI labels, screenshots, and prerequisites."
    if row.get("freshness_bucket") == "unknown":
        return "Perform a substantive review and then add a visible publish or modified date."
    return "Refresh examples, citations, screenshots, and product claims; document unchanged facts explicitly."


def _score_section(
    *,
    text: str,
    title: str,
    heading: str,
    freshness_row: dict,
    search_ctx: dict,
    current_year: int,
) -> dict:
    combined_topic_text = " ".join([
        title or "",
        heading or "",
        search_ctx.get("top_keyword") or "",
        " ".join(str(k.get("keyword") or "") for k in search_ctx.get("keyword_rows") or []),
        text[:500],
    ])
    topic_class, topic_score, topic_labels = _topic_class(combined_topic_text)
    keyword_score, keyword_evidence = _keyword_volatility(search_ctx.get("keyword_rows") or [])
    stale_years = _stale_year_evidence(text, current_year)
    product_evidence = _snippet_for_match(text, _PRODUCT_RE)
    bucket = freshness_row.get("bucket") or "unknown"
    age_days = freshness_row.get("age_days")
    date_risk = _BUCKET_RISK.get(bucket, 20)
    if isinstance(age_days, int) and age_days > 365:
        date_risk += min(14, (age_days - 365) / 90)
    if bucket == "unknown" and topic_score >= 16:
        date_risk += 8
    if re.search(r"\b(latest|current(?:ly)?|newest|202[0-9])\b", text, re.I) and bucket in {"stale", "very_stale"}:
        date_risk += 10
    risk = _clip(
        date_risk
        + topic_score * 0.7
        + keyword_score
        + min(18, len(stale_years) * 7)
        + min(14, len(product_evidence) * 6)
    )
    traffic = _to_int(search_ctx.get("traffic"))
    traffic_weight = 1.0 + min(3.4, math.log10(max(traffic, 0) + 1))
    priority = round(risk * traffic_weight, 2)
    return {
        "freshness_risk": round(risk, 2),
        "priority_score": priority,
        "topic_class": topic_class,
        "topic_labels": topic_labels,
        "keyword_volatility_score": keyword_score,
        "keyword_evidence": keyword_evidence,
        "stale_year_evidence": stale_years,
        "product_evidence": product_evidence,
        "date_evidence": _date_evidence(freshness_row),
    }


def build_freshness_impact(
    pages: list[PageInfo],
    extracted_pages: list,
    freshness_payload: dict | None,
    *,
    search_payload: dict | None = None,
    paragraph_records: list[tuple[int, int, str, Any]] | None = None,
    cluster_labels: Iterable[int] | None = None,
    cluster_summaries=None,
    coords: np.ndarray | None = None,
    today: date | None = None,
    top_n: int = 800,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "sections": [], "pages": [], "clusters": [], "scatter": {"points": []}}
    freshness_rows = {
        row.get("url"): row
        for row in (freshness_payload or {}).get("per_page") or []
        if row.get("url")
    }
    if not freshness_rows:
        return {"summary": {"status": "no_freshness_data", "pages": len(pages)}, "sections": [], "pages": [], "clusters": [], "scatter": {"points": []}}

    search_context = _search_context(pages, search_payload)
    cluster_labels_list = list(cluster_labels) if cluster_labels is not None else [0] * len(pages)
    cluster_names = _cluster_label_lookup(cluster_summaries)
    current_year = (today or date.today()).year
    by_page_paras: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for page_i, para_i, text, _ in paragraph_records or []:
        by_page_paras[int(page_i)].append((int(para_i), str(text or "")))

    sections: list[dict] = []
    page_agg: dict[str, dict] = {}
    cluster_agg: dict[int, dict] = {}

    for page_i, page in enumerate(pages):
        if page_i >= len(extracted_pages):
            continue
        ext = extracted_pages[page_i]
        fresh = freshness_rows.get(page.url) or {}
        if not fresh:
            continue
        search_ctx = search_context.get(page_i, {})
        cid = _cluster_for(page_i, cluster_labels_list)
        cluster_label = cluster_names.get(cid, f"cluster {cid}")
        paragraphs = by_page_paras.get(page_i)
        if not paragraphs:
            body = getattr(ext, "body", "") or ""
            paragraphs = [(0, body[:1200] if body else page.title)]
        for para_i, text in paragraphs:
            if not text or len(text.split()) < 8:
                continue
            heading = _heading_for_paragraph(ext, para_i)
            scored = _score_section(
                text=text,
                title=page.title,
                heading=heading,
                freshness_row=fresh,
                search_ctx=search_ctx,
                current_year=current_year,
            )
            traffic = _to_int(search_ctx.get("traffic"))
            row = {
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "cluster": cid,
                "cluster_label": cluster_label,
                "heading": heading or getattr(ext, "h1", "") or page.title,
                "paragraph_index": int(para_i),
                "excerpt": text[:420],
                "freshness_bucket": fresh.get("bucket") or "unknown",
                "freshness_date": fresh.get("date") or "",
                "freshness_age_days": fresh.get("age_days"),
                "date_source": fresh.get("date_source") or "",
                "date_kind": fresh.get("date_kind") or "",
                "traffic": traffic,
                "keywords": _to_int(search_ctx.get("keywords")),
                "top_keyword": search_ctx.get("top_keyword") or "",
                **scored,
            }
            row["recommended_update_type"] = _recommendation(row)
            row["superficial_update_warning"] = SUPERFICIAL_WARNING
            sections.append(row)

            p = page_agg.setdefault(page.url, {
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "cluster": cid,
                "cluster_label": cluster_label,
                "traffic": traffic,
                "top_keyword": search_ctx.get("top_keyword") or "",
                "freshness_bucket": row["freshness_bucket"],
                "freshness_date": row["freshness_date"],
                "freshness_age_days": row["freshness_age_days"],
                "max_freshness_risk": 0.0,
                "max_priority_score": 0.0,
                "stale_sections": 0,
                "top_heading": "",
            })
            if row["freshness_risk"] >= 50:
                p["stale_sections"] += 1
            if row["priority_score"] >= float(p["max_priority_score"]):
                p["max_priority_score"] = row["priority_score"]
                p["max_freshness_risk"] = row["freshness_risk"]
                p["top_heading"] = row["heading"]

            c = cluster_agg.setdefault(cid, {
                "cluster": cid,
                "label": cluster_label,
                "pages": set(),
                "sections": 0,
                "stale_sections": 0,
                "risk_sum": 0.0,
                "max_priority_score": 0.0,
                "traffic_at_risk": 0,
                "traffic_urls": set(),
                "topic_classes": Counter(),
            })
            c["pages"].add(page.url)
            c["sections"] += 1
            c["risk_sum"] += float(row["freshness_risk"])
            c["topic_classes"][row["topic_class"]] += 1
            c["max_priority_score"] = max(float(c["max_priority_score"]), float(row["priority_score"]))
            if row["freshness_risk"] >= 50:
                c["stale_sections"] += 1
                if page.url not in c["traffic_urls"]:
                    c["traffic_at_risk"] += traffic
                    c["traffic_urls"].add(page.url)

    sections.sort(key=lambda r: (float(r.get("priority_score", 0.0)), float(r.get("freshness_risk", 0.0))), reverse=True)
    pages_payload = sorted(page_agg.values(), key=lambda r: float(r.get("max_priority_score", 0.0)), reverse=True)
    clusters: list[dict] = []
    for raw in cluster_agg.values():
        sections_count = max(int(raw["sections"]), 1)
        topics = raw["topic_classes"].most_common(3)
        clusters.append({
            "cluster": raw["cluster"],
            "label": raw["label"],
            "pages": len(raw["pages"]),
            "sections": raw["sections"],
            "stale_sections": raw["stale_sections"],
            "avg_freshness_risk": round(float(raw["risk_sum"]) / sections_count, 2),
            "max_priority_score": round(float(raw["max_priority_score"]), 2),
            "traffic_at_risk": raw["traffic_at_risk"],
            "top_topic_classes": [{"topic_class": label, "sections": count} for label, count in topics],
        })
    clusters.sort(key=lambda r: (float(r["avg_freshness_risk"]), _to_int(r["traffic_at_risk"])), reverse=True)

    scatter_points: list[dict] = []
    if coords is not None:
        page_by_url = {row["url"]: row for row in pages_payload}
        for i, page in enumerate(pages):
            if i >= len(coords):
                continue
            row = page_by_url.get(page.url)
            if not row:
                continue
            scatter_points.append({
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "cluster": row.get("cluster"),
                "cluster_label": row.get("cluster_label"),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "traffic": _to_int(row.get("traffic")),
                "top_keyword": row.get("top_keyword", ""),
                "freshness_bucket": row.get("freshness_bucket", "unknown"),
                "freshness_risk": float(row.get("max_freshness_risk", 0.0)),
                "priority_score": float(row.get("max_priority_score", 0.0)),
                "freshness_date": row.get("freshness_date", ""),
                "freshness_age_days": row.get("freshness_age_days"),
                "top_heading": row.get("top_heading", ""),
            })

    risky_pages = [row for row in pages_payload if float(row.get("max_freshness_risk", 0.0)) >= 50]
    traffic_at_risk = sum(_to_int(row.get("traffic")) for row in risky_pages)
    summary = {
        "status": "ok",
        "model": "freshness_impact_v1",
        "pages": len(pages_payload),
        "sections": len(sections),
        "high_risk_sections": sum(1 for row in sections if float(row.get("freshness_risk", 0.0)) >= 60),
        "high_impact_sections": sum(1 for row in sections if float(row.get("priority_score", 0.0)) >= 120),
        "traffic_at_risk": traffic_at_risk,
        "avg_freshness_risk": round(sum(float(r.get("freshness_risk", 0.0)) for r in sections) / max(len(sections), 1), 2),
        "clusters": len(clusters),
        "superficial_update_warning": SUPERFICIAL_WARNING,
    }
    return {
        "summary": summary,
        "sections": sections[:top_n],
        "pages": pages_payload[:top_n],
        "clusters": clusters[:top_n],
        "scatter": {"points": scatter_points, "shown": len(scatter_points)},
        "interpretation": {
            "freshness_risk": "0-100 section risk from page dates, stale year claims, old product/version language, volatile topic class, and volatile ranking keywords.",
            "priority_score": "Freshness risk multiplied by organic traffic weight so high-impact stale sections rise above low-traffic stale content.",
            "warning": SUPERFICIAL_WARNING,
        },
    }
