"""Score internal links by the paragraph or section context they appear in."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .extractor import ExtractedPage

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "") if len(m.group(0)) > 1}


def _safe_float(value) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _find_paragraph(page: ExtractedPage, context: str) -> tuple[int | None, str, float]:
    paragraphs = list(page.paragraphs or [])
    if not paragraphs:
        return None, "", 0.0
    context_clean = re.sub(r"\s+", " ", context or "").strip()
    context_tokens = _tokens(context_clean)
    best_i = None
    best_score = 0.0
    for i, paragraph in enumerate(paragraphs):
        para_clean = re.sub(r"\s+", " ", paragraph or "").strip()
        if context_clean and (context_clean in para_clean or para_clean in context_clean):
            return i, para_clean[:280], 1.0
        para_tokens = _tokens(para_clean)
        score = len(context_tokens & para_tokens) / max(1, len(context_tokens)) if context_tokens else 0.0
        if score > best_score:
            best_i = i
            best_score = score
    if best_i is None or best_score <= 0:
        return None, "", 0.0
    return best_i, re.sub(r"\s+", " ", paragraphs[best_i]).strip()[:280], round(best_score, 4)


def _impact_lookup(paragraph_impact: dict | None) -> tuple[dict[tuple[str, int], dict], float]:
    rows = (paragraph_impact or {}).get("top_paragraphs") or []
    lookup = {}
    max_impact = 0.0
    for row in rows:
        key = (row.get("url") or "", _safe_int(row.get("paragraph_index")))
        lookup[key] = row
        max_impact = max(max_impact, _safe_float(row.get("impact_score")))
    return lookup, max_impact


def build_contextual_link_impact(
    extracted_pages: list[ExtractedPage],
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    page_embeddings: np.ndarray | None,
    *,
    linkgraph: dict | None = None,
    paragraph_impact: dict | None = None,
) -> dict:
    linkgraph = linkgraph or {}
    anchor_payload = linkgraph.get("anchor_relevance") or {}
    links = anchor_payload.get("links") or []
    if not links:
        return {"summary": {"status": "no_links", "total_links": 0}, "links": [], "source_pages": []}

    page_by_url = {p.url: p for p in extracted_pages}
    page_idx = {p.url: i for i, p in enumerate(extracted_pages)}
    para_emb_by_key = {(int(page_i), int(para_i)): vec for page_i, para_i, _, vec in paragraph_records or []}
    impact_by_key, max_impact = _impact_lookup(paragraph_impact)
    removal_by_edge = {
        (row.get("source_url"), row.get("target_url")): row
        for row in (linkgraph.get("link_removal_simulation") or {}).get("links", [])
    }
    authority_by_url = {
        row.get("url"): row
        for row in (linkgraph.get("traffic_weighted_pagerank") or {}).get("pages", [])
        if row.get("url")
    }

    rows = []
    for link in links:
        source_url = link.get("source_url") or ""
        target_url = link.get("target_url") or ""
        source = page_by_url.get(source_url)
        target_i = page_idx.get(target_url)
        para_i = None
        paragraph_excerpt = ""
        paragraph_fit = 0.0
        if source is not None:
            para_i, paragraph_excerpt, paragraph_fit = _find_paragraph(source, link.get("context") or "")
        semantic_context = 0.0
        source_i = page_idx.get(source_url)
        if para_i is not None and source_i is not None and target_i is not None and page_embeddings is not None:
            vec = para_emb_by_key.get((source_i, para_i))
            if vec is not None:
                semantic_context = max(0.0, min(1.0, (float(np.clip(vec @ page_embeddings[target_i], -1.0, 1.0)) + 1.0) / 2.0))
        context_relevance = max(semantic_context, _safe_float(link.get("context_overlap")), paragraph_fit)
        pimpact = impact_by_key.get((source_url, para_i if para_i is not None else -1), {})
        paragraph_impact_score = _safe_float(pimpact.get("impact_score"))
        paragraph_impact_component = (paragraph_impact_score / max_impact * 100.0) if max_impact else 0.0
        removal = removal_by_edge.get((source_url, target_url), {})
        target_auth = authority_by_url.get(target_url) or {}
        authority_component = max(_safe_float(removal.get("removal_loss_score")), _safe_float(target_auth.get("weighted_pagerank_percentile")) * 100.0)
        anchor_component = _safe_float(link.get("score"))
        placement = removal.get("placement") or ""
        if placement == "template_navigation" or not paragraph_excerpt:
            context_type = "template"
        elif context_relevance >= 0.62:
            context_type = "main_content"
        else:
            context_type = "weak_context"
        score = (
            context_relevance * 35.0
            + paragraph_impact_component * 0.25
            + authority_component * 0.25
            + anchor_component * 0.15
        )
        row = {
            **link,
            "paragraph_index": para_i,
            "paragraph_excerpt": paragraph_excerpt,
            "paragraph_match": round(paragraph_fit, 4),
            "contextual_similarity": round(context_relevance, 4),
            "paragraph_impact_score": round(paragraph_impact_score, 2),
            "structural_authority_score": round(authority_component, 2),
            "contextual_link_impact": round(max(0.0, min(100.0, score)), 2),
            "context_type": context_type,
            "target_traffic": _safe_int(target_auth.get("traffic")),
            "recommended_action": (
                "Protect or reuse this high-context main-content link pattern."
                if context_type == "main_content" and score >= 65
                else "Move this link into a more relevant paragraph or improve the surrounding copy."
                if context_type == "weak_context"
                else "Treat this as template/navigation support, not a contextual editorial link."
            ),
        }
        rows.append(row)

    rows.sort(key=lambda r: _safe_float(r.get("contextual_link_impact")), reverse=True)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_url"]].append(row)
    source_pages = []
    for source_url, source_rows in by_source.items():
        source_pages.append({
            "source_url": source_url,
            "source_title": source_rows[0].get("source_title") or source_url,
            "strongest_outbound_links": source_rows[:10],
            "avg_contextual_impact": round(sum(_safe_float(r.get("contextual_link_impact")) for r in source_rows) / max(1, len(source_rows)), 2),
            "main_content_links": sum(1 for r in source_rows if r.get("context_type") == "main_content"),
            "template_links": sum(1 for r in source_rows if r.get("context_type") == "template"),
        })
    source_pages.sort(key=lambda r: (r["avg_contextual_impact"], r["main_content_links"]), reverse=True)
    main_links = sum(1 for r in rows if r["context_type"] == "main_content")
    weak_links = sum(1 for r in rows if r["context_type"] == "weak_context")
    template_links = sum(1 for r in rows if r["context_type"] == "template")
    return {
        "summary": {
            "status": "ok",
            "model": "contextual_link_impact_v1",
            "total_links": len(rows),
            "avg_contextual_impact": round(sum(_safe_float(r.get("contextual_link_impact")) for r in rows) / max(1, len(rows)), 2),
            "main_content_links": main_links,
            "weak_context_links": weak_links,
            "template_links": template_links,
            "high_impact_contextual_links": sum(1 for r in rows if r["context_type"] == "main_content" and _safe_float(r.get("contextual_link_impact")) >= 65),
        },
        "links": rows,
        "top_contextual_links": [r for r in rows if r["context_type"] == "main_content"][:300],
        "weak_context_links": [r for r in rows if r["context_type"] == "weak_context"][:300],
        "source_pages": source_pages[:200],
        "interpretation": {
            "contextual_link_impact": "Blend of paragraph-target contextual similarity, paragraph impact score, structural authority, and anchor relevance.",
            "context_type": "main_content links are matched to relevant paragraphs; template links lack paragraph context or are classified as navigation/template edges.",
        },
    }
