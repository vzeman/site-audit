"""Weak paragraph and content-decay diagnostics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from .analyzer import PageInfo
from .paragraph_impact import _heading_for_paragraph, _normalize_url, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
_GENERIC_PHRASES = (
    "learn more",
    "click here",
    "read more",
    "contact us",
    "get started",
    "find out more",
    "for more information",
    "this article",
    "our solution",
)
_BOILERPLATE_RE = re.compile(
    r"\b(cookie|privacy policy|terms(?: and conditions)?|all rights reserved|copyright|"
    r"newsletter|subscribe|unsubscribe|gdpr|consent|powered by)\b",
    re.I,
)


def _key(url: str, paragraph_index: Any) -> str:
    return f"{_normalize_url(url)}#{_to_int(paragraph_index)}"


def _text_fingerprint(text: str) -> str:
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _directory(url: str, fallback: str = "/") -> str:
    path = urlparse(url).path or "/"
    parts = [p for p in path.split("/") if p]
    if parts:
        return f"/{parts[0]}/"
    return fallback or "/"


def _freshness_lookup(freshness_payload: dict | None) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in (freshness_payload or {}).get("per_page") or []:
        rows[_normalize_url(row.get("url") or "")] = row
    return rows


def _impact_lookup(paragraph_impact: dict | None) -> dict[str, dict]:
    return {
        _key(row.get("url") or "", row.get("paragraph_index")): row
        for row in (paragraph_impact or {}).get("top_paragraphs") or []
    }


def _ablation_lookup(semantic_ablation: dict | None) -> dict[str, dict]:
    return {
        _key(row.get("url") or "", row.get("paragraph_index")): row
        for row in (semantic_ablation or {}).get("rows") or []
    }


def _keyword_lookup(keyword_attribution: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"traffic": 0, "keywords": [], "statuses": Counter()})
    for row in (keyword_attribution or {}).get("keywords") or []:
        para_i = row.get("best_paragraph_index")
        if para_i is None:
            continue
        key = _key(row.get("url") or "", para_i)
        traffic = _to_int(row.get("traffic"))
        out[key]["traffic"] += traffic
        out[key]["statuses"][row.get("status") or "unknown"] += 1
        if len(out[key]["keywords"]) < 8:
            out[key]["keywords"].append({
                "keyword": row.get("keyword") or "",
                "traffic": traffic,
                "position": _to_int(row.get("position")),
                "status": row.get("status") or "",
            })
    for row in out.values():
        row["statuses"] = dict(row["statuses"])
        row["keywords"].sort(key=lambda r: int(r.get("traffic", 0)), reverse=True)
    return dict(out)


def _density_lookup(paragraph_density_rows: Iterable[Any]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for row in paragraph_density_rows or []:
        page_i = getattr(row, "page_index", None)
        para_i = getattr(row, "paragraph_index", None)
        if page_i is None or para_i is None:
            continue
        out[(int(page_i), int(para_i))] = {
            "words": int(getattr(row, "words", 0) or 0),
            "internal_links": int(getattr(row, "internal", 0) or 0),
            "external_links": int(getattr(row, "external", 0) or 0),
            "density_per_100w": float(getattr(row, "density_per_100", 0.0) or 0.0),
        }
    return out


def _traffic_by_url(search_payload: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        norm = _normalize_url(url)
        out[norm] = max(out.get(norm, 0), _to_int(row.get("traffic")))
    return out


def _quality_scores(severity: float, traffic: float) -> dict:
    current = round(max(0.0, 100.0 - severity), 1)
    potential = round(min(100.0, current + min(38.0, severity * 0.48)), 1)
    recoverable = round(float(traffic) * min(0.35, severity / 280.0), 2) if traffic > 0 else 0.0
    return {
        "current_quality_score": current,
        "potential_quality_score": potential,
        "estimated_recoverable_traffic": recoverable,
    }


def _action(issue_types: list[str], content_kind: str) -> str:
    issue_set = set(issue_types)
    if content_kind == "template" and "duplicate" in issue_set:
        return "remove"
    if "off_topic" in issue_set:
        return "move"
    if "duplicate" in issue_set:
        return "merge"
    if "stale" in issue_set:
        return "update"
    if "missing_evidence" in issue_set:
        return "cite"
    if "missing_internal_link" in issue_set or "excessive_links" in issue_set:
        return "link"
    return "rewrite"


def _action_label(action: str) -> str:
    return {
        "rewrite": "Rewrite",
        "remove": "Remove",
        "merge": "Merge",
        "update": "Update",
        "link": "Link",
        "cite": "Cite",
        "move": "Move",
    }.get(action, action)


def build_weak_paragraphs(
    pages: list[PageInfo],
    page_embeddings: np.ndarray,
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    *,
    search_payload: dict | None = None,
    paragraph_impact: dict | None = None,
    semantic_ablation: dict | None = None,
    keyword_attribution: dict | None = None,
    paragraph_density_rows: Iterable[Any] = (),
    freshness: dict | None = None,
    cluster_labels: Iterable[int] | None = None,
    today: date | None = None,
    top_n: int = 700,
) -> dict:
    total = len(paragraph_records or [])
    if not paragraph_records:
        return {"summary": {"status": "no_paragraphs", "total_paragraphs": 0}, "rows": [], "per_page": []}

    today_date = today or date.today()
    fingerprints = Counter(_text_fingerprint(text) for _, _, text, _ in paragraph_records)
    frequent_floor = max(3, math.ceil(max(len(pages), 1) * 0.04))
    impact_by_key = _impact_lookup(paragraph_impact)
    ablation_by_key = _ablation_lookup(semantic_ablation)
    keyword_by_key = _keyword_lookup(keyword_attribution)
    density_by_key = _density_lookup(paragraph_density_rows)
    freshness_by_url = _freshness_lookup(freshness)
    page_traffic = _traffic_by_url(search_payload)
    clusters = list(cluster_labels) if cluster_labels is not None else []

    rows: list[dict] = []
    by_page: dict[int, list[dict]] = defaultdict(list)

    for page_i, para_i, text, emb in paragraph_records:
        if page_i >= len(pages):
            continue
        page = pages[page_i]
        ext = extracted_pages[page_i] if page_i < len(extracted_pages) else None
        norm_url = _normalize_url(page.url)
        key = _key(page.url, para_i)
        fp = _text_fingerprint(text)
        dup_count = fingerprints.get(fp, 0)
        token_list = _tokens(text)
        word_count = len(token_list)
        unique_ratio = len(set(token_list)) / max(word_count, 1)
        paragraph_lower = (text or "").lower()
        generic_hits = [p for p in _GENERIC_PHRASES if p in paragraph_lower]
        has_boilerplate_terms = bool(_BOILERPLATE_RE.search(text or ""))
        content_kind = "template" if has_boilerplate_terms or (dup_count >= frequent_floor and word_count <= 80) else "main"

        issue_types: list[str] = []
        reasons: list[str] = []
        severity = 0.0

        if word_count < 25:
            issue_types.append("thin")
            severity += 15
            reasons.append(f"Only {word_count} words, so the block is too thin to carry a clear answer.")

        if generic_hits or (word_count >= 35 and unique_ratio < 0.42):
            issue_types.append("generic")
            severity += 14
            if generic_hits:
                reasons.append(f"Generic CTA/vague wording appears: {', '.join(generic_hits[:3])}.")
            else:
                reasons.append("Low unique-word ratio suggests repeated or generic phrasing.")

        if dup_count >= frequent_floor:
            issue_types.append("duplicate")
            severity += 24 if content_kind == "main" else 10
            reasons.append(f"Exact paragraph text appears on {dup_count} pages.")

        page_vec = np.asarray(page_embeddings[page_i], dtype=np.float32) if page_i < len(page_embeddings) else np.zeros(0, dtype=np.float32)
        para_vec = np.asarray(emb, dtype=np.float32)
        page_similarity = float(np.clip(para_vec @ page_vec, -1.0, 1.0)) if page_vec.size and para_vec.size else 0.0
        ablation = ablation_by_key.get(key) or {}
        if ablation.get("classification") == "noise_candidate":
            issue_types.append("off_topic")
            severity += 42
            reasons.append("Semantic ablation says removing this paragraph would improve page-topic alignment.")
        elif page_similarity and page_similarity < 0.48 and content_kind == "main":
            issue_types.append("off_topic")
            severity += 26
            reasons.append(f"Paragraph/page semantic similarity is low ({page_similarity:.2f}).")

        years = [int(y) for y in _YEAR_RE.findall(text or "")]
        fresh = freshness_by_url.get(norm_url) or {}
        stale_bucket = fresh.get("bucket") in {"stale", "very_stale", "unknown", "future"}
        if years and max(years) <= today_date.year - 3:
            issue_types.append("stale")
            severity += 22
            reasons.append(f"Newest visible year is {max(years)}, which looks stale.")
        elif stale_bucket and content_kind == "main" and word_count >= 40:
            issue_types.append("stale")
            severity += 14 if fresh.get("bucket") == "unknown" else 18
            reasons.append(f"Page freshness bucket is {fresh.get('bucket')}.")

        density = density_by_key.get((int(page_i), int(para_i))) or {}
        internal_links = int(density.get("internal_links", 0))
        external_links = int(density.get("external_links", 0))
        link_density = float(density.get("density_per_100w", 0.0))
        if link_density >= 5.0 and internal_links + external_links >= 2:
            issue_types.append("excessive_links")
            severity += 20
            reasons.append(f"Link density is {link_density:.1f} links per 100 words.")

        impact = impact_by_key.get(key) or {}
        impact_components = impact.get("components") or {}
        kw = keyword_by_key.get(key) or {}
        traffic_opportunity = max(
            float(impact.get("attributed_traffic", 0.0) or 0.0),
            float(kw.get("traffic", 0.0) or 0.0),
        )
        if not traffic_opportunity:
            traffic_opportunity = page_traffic.get(norm_url, 0) * 0.08 if page_traffic.get(norm_url, 0) and issue_types else 0.0

        statuses = kw.get("statuses") or {}
        if statuses.get("weak_paragraph") or statuses.get("unmatched"):
            issue_types.append("intent_mismatch")
            severity += 30
            reasons.append("Ranking keyword attribution found weak or unmatched visible paragraph support.")
        elif traffic_opportunity > 0 and (
            float(impact_components.get("semantic", 1.0) or 0.0) < 0.35
            or float(impact_components.get("keyword_overlap", 1.0) or 0.0) < 0.18
        ):
            issue_types.append("intent_mismatch")
            severity += 20
            reasons.append("Paragraph has traffic opportunity but weak keyword semantic/lexical fit.")

        has_number = bool(re.search(r"\d", text or ""))
        if traffic_opportunity >= 5 and word_count >= 45 and not has_number and external_links == 0:
            issue_types.append("missing_evidence")
            severity += 16
            reasons.append("Traffic-carrying paragraph has no numbers, dates, or outbound citation support.")

        if traffic_opportunity >= 5 and internal_links == 0 and word_count >= 80:
            issue_types.append("missing_internal_link")
            severity += 15
            reasons.append("Long traffic-carrying paragraph has no internal links.")

        if not issue_types:
            continue

        deduped_issue_types = list(dict.fromkeys(issue_types))
        severity = _clip(severity + math.log1p(max(traffic_opportunity, 0.0)) * 3.0, 0.0, 100.0)
        action = _action(deduped_issue_types, content_kind)
        quality = _quality_scores(severity, traffic_opportunity)
        row = {
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "directory": _directory(page.url, page.section),
            "cluster": int(clusters[page_i]) if page_i < len(clusters) else None,
            "paragraph_index": int(para_i),
            "heading": _heading_for_paragraph(ext, para_i) if ext is not None else "",
            "paragraph_excerpt": text[:360],
            "word_count": word_count,
            "content_kind": content_kind,
            "editorial_recommendation": content_kind == "main",
            "issue_types": deduped_issue_types,
            "primary_issue": deduped_issue_types[0],
            "reasons": reasons[:5],
            "recommended_action": action,
            "recommended_action_label": _action_label(action),
            "severity_score": round(severity, 2),
            "severity": "high" if severity >= 70 else ("medium" if severity >= 45 else "low"),
            "traffic_opportunity": round(traffic_opportunity, 2),
            "page_traffic": int(page_traffic.get(norm_url, 0)),
            "estimated_recoverable_traffic": quality["estimated_recoverable_traffic"],
            "current_quality_score": quality["current_quality_score"],
            "potential_quality_score": quality["potential_quality_score"],
            "duplicate_count": int(dup_count),
            "page_similarity": round(page_similarity, 4),
            "ablation_classification": ablation.get("classification", ""),
            "alignment_delta": ablation.get("alignment_delta"),
            "freshness_bucket": fresh.get("bucket", ""),
            "freshness_age_days": fresh.get("age_days"),
            "internal_links": internal_links,
            "external_links": external_links,
            "link_density_per_100w": round(link_density, 2),
            "attributed_keywords": kw.get("keywords", []),
        }
        rows.append(row)
        by_page[int(page_i)].append(row)

    rows.sort(key=lambda r: (float(r.get("traffic_opportunity", 0.0)), float(r.get("severity_score", 0.0))), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    issue_counts = Counter(issue for row in rows for issue in row.get("issue_types", []))
    action_counts = Counter(row.get("recommended_action") for row in rows)
    directory_counts = Counter(row.get("directory") or "/" for row in rows)
    cluster_counts = Counter(str(row.get("cluster")) for row in rows if row.get("cluster") is not None)

    per_page: list[dict] = []
    for page_i, page_rows in by_page.items():
        page = pages[page_i]
        page_rows.sort(key=lambda r: (float(r.get("severity_score", 0.0)), float(r.get("traffic_opportunity", 0.0))), reverse=True)
        per_page.append({
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "directory": _directory(page.url, page.section),
            "cluster": int(clusters[page_i]) if page_i < len(clusters) else None,
            "page_traffic": int(page_rows[0].get("page_traffic", 0) if page_rows else 0),
            "weak_count": len(page_rows),
            "template_count": sum(1 for r in page_rows if r.get("content_kind") == "template"),
            "main_content_count": sum(1 for r in page_rows if r.get("content_kind") == "main"),
            "max_severity_score": max(float(r.get("severity_score", 0.0)) for r in page_rows),
            "traffic_opportunity": round(sum(float(r.get("traffic_opportunity", 0.0)) for r in page_rows), 2),
            "blocks": [
                {
                    "paragraph_index": r["paragraph_index"],
                    "severity": r["severity"],
                    "severity_score": r["severity_score"],
                    "content_kind": r["content_kind"],
                    "issue_types": r["issue_types"],
                    "recommended_action": r["recommended_action"],
                    "excerpt": r["paragraph_excerpt"],
                }
                for r in page_rows[:24]
            ],
        })
    per_page.sort(key=lambda r: (float(r.get("traffic_opportunity", 0.0)), float(r.get("max_severity_score", 0.0))), reverse=True)

    summary = {
        "status": "ok" if rows else "no_weak_paragraphs",
        "model": "weak_paragraphs_v1",
        "total_paragraphs": total,
        "flagged_rows": len(rows),
        "main_content_rows": sum(1 for r in rows if r.get("content_kind") == "main"),
        "template_rows": sum(1 for r in rows if r.get("content_kind") == "template"),
        "high_severity_rows": sum(1 for r in rows if r.get("severity") == "high"),
        "traffic_rows": sum(1 for r in rows if float(r.get("traffic_opportunity", 0.0)) > 0),
        "total_traffic_opportunity": round(sum(float(r.get("traffic_opportunity", 0.0)) for r in rows), 2),
        "estimated_recoverable_traffic": round(sum(float(r.get("estimated_recoverable_traffic", 0.0)) for r in rows), 2),
        "issue_counts": dict(issue_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "directories": [{"key": k, "count": v} for k, v in directory_counts.most_common(40)],
        "clusters": [{"key": k, "count": v} for k, v in cluster_counts.most_common(40)],
    }
    return {
        "summary": summary,
        "rows": rows[:top_n],
        "per_page": per_page[:500],
        "interpretation": {
            "severity_score": "Heuristic editorial risk score from thinness, genericness, duplication, off-topic ablation, stale signals, link density, missing evidence, and ranking-intent mismatch.",
            "traffic_opportunity": "Estimated organic traffic attached to the paragraph from paragraph impact, keyword attribution, or page-level search data fallback.",
            "estimated_recoverable_traffic": "A directional before/after estimate used to prioritize edits when ranking keyword data is available.",
            "template_rows": "Boilerplate/template paragraphs are separated so repeated legal, cookie, and footer copy does not pollute main-content recommendations.",
        },
    }
