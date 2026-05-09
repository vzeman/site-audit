"""Heading-section impact maps built from paragraph and keyword signals."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable
from urllib.parse import urlparse

from .analyzer import PageInfo
from .entities import extract_entities_from_text
from .paragraph_impact import _normalize_url, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
_GENERIC_HEADINGS = {
    "overview",
    "learn more",
    "features",
    "benefits",
    "conclusion",
    "summary",
    "faq",
    "faqs",
    "why choose us",
    "get started",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "") if len(m.group(0)) > 1}


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _heading_key(url: str, order: int, text: str) -> str:
    return f"{_normalize_url(url)}#{int(order)}#{' '.join((text or '').lower().split())}"


def _paragraph_key(url: str, para_i: Any) -> str:
    return f"{_normalize_url(url)}#{_to_int(para_i)}"


def _text_key(url: str, text: str) -> tuple[str, str]:
    return (_normalize_url(url), " ".join((text or "").lower().split()))


def _directory(url: str, fallback: str = "/") -> str:
    path = urlparse(url).path or "/"
    parts = [p for p in path.split("/") if p]
    return f"/{parts[0]}/" if parts else (fallback or "/")


def _headers_for_page(page: PageInfo, ext) -> list[dict]:
    rows: list[dict] = []
    for i, header in enumerate(getattr(ext, "headers_rich", []) or []):
        text = " ".join(str(header.get("text") or "").split())
        if not text:
            continue
        rows.append({
            "level": max(1, min(6, _to_int(header.get("level")) or 1)),
            "order": _to_int(header.get("order")) or i + 1,
            "text": text,
        })
    if not rows:
        fallback = " ".join(str(getattr(ext, "h1", "") or page.title or page.url).split())
        rows.append({"level": 1, "order": 0, "text": fallback, "synthetic": True})
    rows.sort(key=lambda h: int(h.get("order", 0)))
    return rows


def _heading_for_paragraph(headers: list[dict], para_i: int, para_count: int) -> dict:
    if not headers:
        return {}
    if len(headers) == 1 or para_count <= 1:
        return headers[0]
    max_order = max(1, max(_to_int(h.get("order")) for h in headers))
    para_ratio = (int(para_i) + 0.5) / max(1, para_count)
    chosen = headers[0]
    for header in headers:
        if (_to_int(header.get("order")) / max_order) <= para_ratio:
            chosen = header
        else:
            break
    return chosen


def _freshness_lookup(freshness_payload: dict | None) -> dict[str, dict]:
    return {
        _normalize_url(row.get("url") or ""): row
        for row in (freshness_payload or {}).get("per_page") or []
    }


def _impact_lookup(paragraph_impact: dict | None) -> dict[str, dict]:
    return {
        _paragraph_key(row.get("url") or "", row.get("paragraph_index")): row
        for row in (paragraph_impact or {}).get("top_paragraphs") or []
    }


def _keyword_lookups(keyword_attribution: dict | None) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_para: dict[str, dict] = defaultdict(lambda: {"traffic": 0, "keywords": []})
    by_heading: dict[tuple[str, str], dict] = defaultdict(lambda: {"traffic": 0, "keywords": []})
    for row in (keyword_attribution or {}).get("keywords") or []:
        keyword = row.get("keyword") or ""
        traffic = _to_int(row.get("traffic"))
        item = {
            "keyword": keyword,
            "traffic": traffic,
            "position": _to_int(row.get("position")),
            "status": row.get("status") or "",
        }
        if row.get("best_paragraph_index") is not None:
            key = _paragraph_key(row.get("url") or "", row.get("best_paragraph_index"))
            by_para[key]["traffic"] += traffic
            if len(by_para[key]["keywords"]) < 12:
                by_para[key]["keywords"].append(item)
        if row.get("best_heading"):
            key2 = _text_key(row.get("url") or "", row.get("best_heading") or "")
            by_heading[key2]["traffic"] += traffic
            if len(by_heading[key2]["keywords"]) < 12:
                by_heading[key2]["keywords"].append(item)
    for payload in list(by_para.values()) + list(by_heading.values()):
        payload["keywords"].sort(key=lambda r: int(r.get("traffic", 0)), reverse=True)
    return dict(by_para), dict(by_heading)


def _keyword_overlap(heading: str, keywords: list[dict]) -> float:
    heading_tokens = _tokens(heading)
    if not heading_tokens or not keywords:
        return 0.0
    scores: list[float] = []
    weights: list[float] = []
    for row in keywords[:8]:
        kw_tokens = _tokens(row.get("keyword") or "")
        if not kw_tokens:
            continue
        scores.append(len(heading_tokens & kw_tokens) / len(kw_tokens))
        weights.append(max(1.0, float(row.get("traffic", 0) or 0)))
    if not scores:
        return 0.0
    total = sum(weights) or 1.0
    return sum(score * weight for score, weight in zip(scores, weights)) / total


def _labels(issue_codes: list[str]) -> list[str]:
    label_map = {
        "no_body": "No supporting body",
        "low_support": "Low support",
        "rename": "Rename opportunity",
        "expand": "Expand opportunity",
        "freshness": "Freshness risk",
    }
    return [label_map.get(code, code.replace("_", " ")) for code in issue_codes]


def build_heading_impact(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, Any]],
    *,
    paragraph_impact: dict | None = None,
    keyword_attribution: dict | None = None,
    freshness: dict | None = None,
    cluster_labels: Iterable[int] | None = None,
    top_n: int = 1000,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "rows": [], "per_page": []}

    impact_by_para = _impact_lookup(paragraph_impact)
    keywords_by_para, keywords_by_heading = _keyword_lookups(keyword_attribution)
    freshness_by_url = _freshness_lookup(freshness)
    clusters = list(cluster_labels) if cluster_labels is not None else []

    paragraphs_by_page: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for page_i, para_i, text, _ in paragraph_records or []:
        paragraphs_by_page[int(page_i)].append((int(para_i), text))
    for rows in paragraphs_by_page.values():
        rows.sort(key=lambda item: item[0])

    section_rows: dict[str, dict] = {}
    heading_by_text: dict[tuple[str, str], str] = {}

    for page_i, page in enumerate(pages):
        ext = extracted_pages[page_i] if page_i < len(extracted_pages) else None
        headers = _headers_for_page(page, ext)
        norm_url = _normalize_url(page.url)
        fresh = freshness_by_url.get(norm_url) or {}
        for header in headers:
            key = _heading_key(page.url, int(header["order"]), header["text"])
            heading_by_text.setdefault(_text_key(page.url, header["text"]), key)
            section_rows[key] = {
                "heading_key": key,
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "directory": _directory(page.url, page.section),
                "cluster": int(clusters[page_i]) if page_i < len(clusters) else None,
                "level": int(header["level"]),
                "order": int(header["order"]),
                "heading": header["text"],
                "synthetic": bool(header.get("synthetic")),
                "paragraph_count": 0,
                "word_count": 0,
                "paragraph_indices": [],
                "paragraph_impact_score_sum": 0.0,
                "paragraph_impact_score_max": 0.0,
                "paragraph_impact_rows": 0,
                "impact_traffic": 0.0,
                "keyword_traffic": 0.0,
                "keywords": [],
                "_text": [],
                "freshness_bucket": fresh.get("bucket", ""),
                "freshness_age_days": fresh.get("age_days"),
            }

        page_paras = paragraphs_by_page.get(page_i, [])
        para_count = len(page_paras)
        for para_i, text in page_paras:
            header = _heading_for_paragraph(headers, para_i, para_count)
            if not header:
                continue
            key = _heading_key(page.url, int(header["order"]), header["text"])
            row = section_rows[key]
            row["paragraph_count"] += 1
            row["word_count"] += _word_count(text)
            row["paragraph_indices"].append(int(para_i))
            row["_text"].append(text)
            para_key = _paragraph_key(page.url, para_i)
            impact = impact_by_para.get(para_key) or {}
            if impact:
                score = float(impact.get("impact_score", 0.0) or 0.0)
                row["paragraph_impact_score_sum"] += score
                row["paragraph_impact_score_max"] = max(float(row["paragraph_impact_score_max"]), score)
                row["paragraph_impact_rows"] += 1
                row["impact_traffic"] += float(impact.get("attributed_traffic", 0.0) or 0.0)
            kw = keywords_by_para.get(para_key) or {}
            if kw:
                row["keyword_traffic"] += float(kw.get("traffic", 0.0) or 0.0)
                row["keywords"].extend(kw.get("keywords") or [])

    for text_key, payload in keywords_by_heading.items():
        key = heading_by_text.get(text_key)
        if not key:
            continue
        row = section_rows[key]
        row["keyword_traffic"] = max(float(row.get("keyword_traffic", 0.0)), float(payload.get("traffic", 0.0) or 0.0))
        row["keywords"].extend(payload.get("keywords") or [])

    rows: list[dict] = []
    for row in section_rows.values():
        keyword_by_name: dict[str, dict] = {}
        for kw in row.get("keywords") or []:
            name = str(kw.get("keyword") or "").lower()
            if not name:
                continue
            current = keyword_by_name.get(name)
            if current is None or _to_int(kw.get("traffic")) > _to_int(current.get("traffic")):
                keyword_by_name[name] = kw
        keywords = sorted(keyword_by_name.values(), key=lambda r: _to_int(r.get("traffic")), reverse=True)[:10]
        text = " ".join(row.pop("_text", []) or [])
        entities = Counter(extract_entities_from_text(text))
        top_entities = [{"entity": entity, "mentions": count} for entity, count in entities.most_common(8)]
        traffic = max(float(row.get("impact_traffic", 0.0)), float(row.get("keyword_traffic", 0.0)))
        keyword_count = len(keywords)
        avg_impact = (
            float(row.get("paragraph_impact_score_sum", 0.0)) / max(1, int(row.get("paragraph_impact_rows", 0)))
        )
        heading_overlap = _keyword_overlap(row["heading"], keywords)
        freshness_bucket = row.get("freshness_bucket") or ""
        years = [int(y) for y in _YEAR_RE.findall(text)]
        stale_year = bool(years and max(years) <= date.today().year - 3)
        support_score = min(
            100.0,
            min(35.0, row["paragraph_count"] * 10.0)
            + min(30.0, row["word_count"] / 12.0)
            + min(20.0, len(entities) * 4.0)
            + min(15.0, avg_impact / 4.0),
        )
        if freshness_bucket in {"stale", "very_stale", "unknown"} or stale_year:
            support_score = max(0.0, support_score - 8.0)
        demand_score = min(
            100.0,
            math.log1p(max(traffic, 0.0)) * 18.0
            + min(24.0, keyword_count * 6.0)
            + min(24.0, float(row.get("paragraph_impact_score_max", 0.0)) / 3.0),
        )
        issue_codes: list[str] = []
        if row["paragraph_count"] == 0 or row["word_count"] < 25:
            issue_codes.append("no_body")
        if demand_score >= 30 and support_score < 45:
            issue_codes.append("low_support")
        heading_text_key = " ".join(str(row["heading"]).lower().split())
        if demand_score >= 25 and keywords and (heading_overlap < 0.22 or heading_text_key in _GENERIC_HEADINGS):
            issue_codes.append("rename")
        if demand_score >= 35 and row["word_count"] < 140:
            issue_codes.append("expand")
        if freshness_bucket in {"stale", "very_stale", "unknown"} or stale_year:
            issue_codes.append("freshness")
        issue_codes = list(dict.fromkeys(issue_codes))
        opportunity_score = max(0.0, demand_score - support_score * 0.7)
        row.update({
            "keyword_count": keyword_count,
            "keywords": keywords,
            "attributed_traffic": round(traffic, 2),
            "entity_count": len(entities),
            "top_entities": top_entities,
            "average_impact_score": round(avg_impact, 2),
            "heading_keyword_overlap": round(heading_overlap, 4),
            "support_score": round(support_score, 2),
            "demand_score": round(demand_score, 2),
            "opportunity_score": round(opportunity_score, 2),
            "issue_codes": issue_codes,
            "issue_labels": _labels(issue_codes),
            "recommended_action": (
                "rename" if "rename" in issue_codes
                else "expand" if "expand" in issue_codes or "low_support" in issue_codes
                else "refresh" if "freshness" in issue_codes
                else "maintain"
            ),
            "is_high_demand": bool(demand_score >= 35),
            "is_low_support": bool(demand_score >= 30 and support_score < 45),
            "has_stale_year": stale_year,
        })
        rows.append(row)

    rows.sort(key=lambda r: (float(r.get("attributed_traffic", 0.0)), float(r.get("opportunity_score", 0.0))), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    per_page: list[dict] = []
    by_url: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_url[row["url"]].append(row)
    for page in pages:
        page_rows = sorted(by_url.get(page.url, []), key=lambda r: int(r.get("order", 0)))
        if not page_rows:
            continue
        per_page.append({
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "heading_count": len(page_rows),
            "attributed_traffic": round(sum(float(r.get("attributed_traffic", 0.0)) for r in page_rows), 2),
            "keyword_count": sum(int(r.get("keyword_count", 0)) for r in page_rows),
            "opportunity_count": sum(1 for r in page_rows if r.get("issue_codes")),
            "headings": page_rows,
        })
    per_page.sort(key=lambda r: (float(r.get("attributed_traffic", 0.0)), int(r.get("opportunity_count", 0))), reverse=True)

    issue_counts = Counter(issue for row in rows for issue in row.get("issue_codes", []))
    summary = {
        "status": "ok" if rows else "no_headings",
        "model": "heading_impact_v1",
        "pages": len(pages),
        "pages_with_heading_metrics": len(per_page),
        "headings": len(rows),
        "headings_with_demand": sum(1 for r in rows if float(r.get("attributed_traffic", 0.0)) > 0 or int(r.get("keyword_count", 0)) > 0),
        "high_demand_headings": sum(1 for r in rows if r.get("is_high_demand")),
        "low_support_high_demand": sum(1 for r in rows if r.get("is_low_support")),
        "rename_opportunities": issue_counts.get("rename", 0),
        "expand_opportunities": issue_counts.get("expand", 0),
        "no_body_headings": issue_counts.get("no_body", 0),
        "attributed_traffic": round(sum(float(r.get("attributed_traffic", 0.0)) for r in rows), 2),
        "issue_counts": dict(issue_counts.most_common()),
    }

    strongest = sorted(rows, key=lambda r: (float(r.get("attributed_traffic", 0.0)), float(r.get("support_score", 0.0))), reverse=True)[:120]
    weakest = sorted(rows, key=lambda r: (float(r.get("opportunity_score", 0.0)), float(r.get("demand_score", 0.0))), reverse=True)[:120]
    return {
        "summary": summary,
        "rows": rows[:top_n],
        "strongest_headings": strongest,
        "weakest_headings": weakest,
        "per_page": per_page[:500],
        "interpretation": {
            "support_score": "Section support from paragraph count, word count, entity coverage, impact scores, and freshness.",
            "demand_score": "Search demand attached to the heading from attributed traffic, ranking keywords, and paragraph impact.",
            "opportunity_score": "Demand minus support. High values indicate rename, expand, or refresh opportunities.",
        },
    }
