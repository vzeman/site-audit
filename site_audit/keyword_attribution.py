"""Attribute ranking-keyword demand to headings and paragraphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .analyzer import PageInfo
from .paragraph_impact import (
    _heading_for_paragraph,
    _keyword_overlap,
    _match_page,
    _page_lookup,
    _semantic_component,
    _to_int,
    _tokens,
)


def _candidate_score(similarity: float, overlap: float) -> float:
    return 0.72 * _semantic_component(similarity) + 0.28 * overlap


def _heading_candidates(page: PageInfo, ext) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []

    def add(text: str, kind: str, level: int = 0, order: int = 0) -> None:
        clean = " ".join(str(text or "").split())
        if not clean:
            return
        key = clean.lower()
        if key in seen:
            return
        seen.add(key)
        rows.append({"text": clean, "kind": kind, "level": int(level), "order": int(order)})

    add(page.title, "title", 0, 0)
    add(getattr(ext, "h1", "") if ext is not None else "", "h1", 1, 1)
    for h in getattr(ext, "headers_rich", []) or []:
        add(h.get("text") or "", f"h{_to_int(h.get('level'))}", _to_int(h.get("level")), _to_int(h.get("order")))
    return rows


def _keyword_rows(pages: list[PageInfo], search_payload: dict, *, max_keywords_per_page: int) -> list[dict]:
    lookup = _page_lookup(pages)
    by_key: dict[tuple[int, str], dict] = {}

    def put(page_i: int | None, row: dict) -> None:
        if page_i is None:
            return
        keyword = str(row.get("keyword") or "").strip()
        if not keyword:
            return
        key = (page_i, keyword.lower())
        current = by_key.get(key)
        if current is None or _to_int(row.get("traffic")) > _to_int(current.get("traffic")):
            by_key[key] = {**row, "page_index": page_i}

    for row in search_payload.get("organic_keywords") or []:
        put(_match_page(row.get("matched_url") or row.get("url") or "", lookup), {
            "keyword": row.get("keyword") or "",
            "traffic": _to_int(row.get("traffic")),
            "volume": _to_int(row.get("volume")),
            "position": _to_int(row.get("position")),
            "country": row.get("country") or "",
            "source": "organic_keyword",
        })
    for row in search_payload.get("top_pages") or []:
        put(_match_page(row.get("matched_url") or row.get("url") or "", lookup), {
            "keyword": row.get("top_keyword") or "",
            "traffic": _to_int(row.get("traffic")),
            "volume": _to_int(row.get("top_keyword_volume")),
            "position": _to_int(row.get("top_keyword_position")),
            "country": row.get("top_keyword_country") or "",
            "source": "top_page",
        })

    by_page: dict[int, list[dict]] = defaultdict(list)
    for row in by_key.values():
        by_page[int(row["page_index"])].append(row)
    out: list[dict] = []
    for page_rows in by_page.values():
        page_rows.sort(key=lambda r: (_to_int(r.get("traffic")), _to_int(r.get("volume"))), reverse=True)
        out.extend(page_rows[:max_keywords_per_page])
    return out


def build_keyword_attribution(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    search_payload: dict,
    *,
    embedder=None,
    max_keywords_per_page: int = 40,
    top_n: int = 1000,
) -> dict:
    if not paragraph_records:
        return {"summary": {"status": "no_paragraphs", "keyword_rows": 0}, "keywords": [], "headings": [], "paragraphs": []}
    if not search_payload or not ((search_payload.get("top_pages") or []) or (search_payload.get("organic_keywords") or [])):
        return {"summary": {"status": "no_search_data", "keyword_rows": 0}, "keywords": [], "headings": [], "paragraphs": []}
    if embedder is None:
        return {"summary": {"status": "missing_embedder", "keyword_rows": 0}, "keywords": [], "headings": [], "paragraphs": []}

    keywords = _keyword_rows(pages, search_payload, max_keywords_per_page=max_keywords_per_page)
    if not keywords:
        return {"summary": {"status": "no_matched_keywords", "keyword_rows": 0}, "keywords": [], "headings": [], "paragraphs": []}

    keyword_texts = [str(row["keyword"]) for row in keywords]
    keyword_embs = embedder.encode(keyword_texts, batch_size=64, show_progress=False)
    by_page_paras: dict[int, list[tuple[int, int, str, np.ndarray]]] = defaultdict(list)
    for rec in paragraph_records:
        by_page_paras[int(rec[0])].append(rec)

    heading_texts: list[str] = []
    heading_refs: list[tuple[int, dict]] = []
    for page_i, page in enumerate(pages):
        if page_i >= len(extracted_pages):
            continue
        if not any(int(k["page_index"]) == page_i for k in keywords):
            continue
        for heading in _heading_candidates(page, extracted_pages[page_i]):
            heading_refs.append((page_i, heading))
            heading_texts.append(heading["text"])
    heading_embs = embedder.encode(heading_texts, batch_size=64, show_progress=False) if heading_texts else np.zeros((0, 0), dtype=np.float32)
    headings_by_page: dict[int, list[tuple[dict, np.ndarray]]] = defaultdict(list)
    for i, (page_i, heading) in enumerate(heading_refs):
        headings_by_page[page_i].append((heading, heading_embs[i]))

    rows: list[dict] = []
    heading_rollup: dict[tuple[str, str], dict] = {}
    paragraph_rollup: dict[tuple[str, int], dict] = {}

    for k_i, kw in enumerate(keywords):
        page_i = int(kw["page_index"])
        if page_i >= len(pages):
            continue
        page = pages[page_i]
        keyword = str(kw["keyword"])
        kw_emb = np.asarray(keyword_embs[k_i], dtype=np.float32)
        kw_tokens = _tokens(keyword)

        best_heading: dict[str, Any] = {}
        best_heading_score = -1.0
        for heading, h_emb in headings_by_page.get(page_i, []):
            sim = float(np.clip(kw_emb @ np.asarray(h_emb, dtype=np.float32), -1.0, 1.0))
            overlap = _keyword_overlap(_tokens(heading["text"]), keyword)
            score = _candidate_score(sim, overlap)
            if score > best_heading_score:
                best_heading_score = score
                best_heading = {**heading, "similarity": round(sim, 4), "overlap": round(overlap, 4), "score": round(score, 4)}

        best_para: dict[str, Any] = {}
        best_para_score = -1.0
        for _, para_i, text, emb in by_page_paras.get(page_i, []):
            sim = float(np.clip(kw_emb @ np.asarray(emb, dtype=np.float32), -1.0, 1.0))
            overlap = len(_tokens(text) & kw_tokens) / len(kw_tokens) if kw_tokens else 0.0
            score = _candidate_score(sim, overlap)
            if score > best_para_score:
                heading = _heading_for_paragraph(extracted_pages[page_i], para_i) if page_i < len(extracted_pages) else ""
                best_para_score = score
                best_para = {
                    "paragraph_index": int(para_i),
                    "paragraph_excerpt": text[:320],
                    "heading": heading,
                    "similarity": round(sim, 4),
                    "overlap": round(overlap, 4),
                    "score": round(score, 4),
                }

        status = "matched"
        if best_para_score < 0.45 and best_heading_score < 0.45:
            status = "unmatched"
        elif best_para_score < 0.45:
            status = "weak_paragraph"
        elif best_heading_score < 0.45:
            status = "weak_heading"

        traffic = _to_int(kw.get("traffic"))
        row = {
            "keyword": keyword,
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "traffic": traffic,
            "volume": _to_int(kw.get("volume")),
            "position": _to_int(kw.get("position")),
            "country": kw.get("country") or "",
            "source": kw.get("source") or "",
            "status": status,
            "best_heading": best_heading.get("text", ""),
            "best_heading_kind": best_heading.get("kind", ""),
            "best_heading_score": round(max(best_heading_score, 0.0), 4),
            "best_heading_similarity": best_heading.get("similarity", 0.0),
            "best_paragraph_index": best_para.get("paragraph_index"),
            "best_paragraph_excerpt": best_para.get("paragraph_excerpt", ""),
            "best_paragraph_heading": best_para.get("heading", ""),
            "best_paragraph_score": round(max(best_para_score, 0.0), 4),
            "best_paragraph_similarity": best_para.get("similarity", 0.0),
            "attributed_traffic": traffic,
        }
        rows.append(row)

        if best_heading.get("text"):
            h_key = (page.url, best_heading["text"])
            group = heading_rollup.setdefault(h_key, {
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "heading": best_heading["text"],
                "kind": best_heading.get("kind", ""),
                "traffic": 0,
                "keyword_count": 0,
                "keywords": [],
            })
            group["traffic"] += traffic
            group["keyword_count"] += 1
            if len(group["keywords"]) < 8:
                group["keywords"].append({"keyword": keyword, "traffic": traffic, "position": row["position"]})

        if best_para.get("paragraph_index") is not None:
            p_key = (page.url, int(best_para["paragraph_index"]))
            group = paragraph_rollup.setdefault(p_key, {
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "paragraph_index": int(best_para["paragraph_index"]),
                "paragraph_excerpt": best_para.get("paragraph_excerpt", ""),
                "heading": best_para.get("heading", ""),
                "traffic": 0,
                "keyword_count": 0,
                "keywords": [],
            })
            group["traffic"] += traffic
            group["keyword_count"] += 1
            if len(group["keywords"]) < 8:
                group["keywords"].append({"keyword": keyword, "traffic": traffic, "position": row["position"]})

    rows.sort(key=lambda r: (int(r.get("traffic", 0)), float(r.get("best_paragraph_score", 0.0))), reverse=True)
    headings = sorted(heading_rollup.values(), key=lambda r: (int(r.get("traffic", 0)), int(r.get("keyword_count", 0))), reverse=True)
    paragraphs = sorted(paragraph_rollup.values(), key=lambda r: (int(r.get("traffic", 0)), int(r.get("keyword_count", 0))), reverse=True)
    summary = {
        "status": "ok",
        "model": "keyword_attribution_v1",
        "keyword_rows": len(rows),
        "matched_keywords": sum(1 for r in rows if r["status"] == "matched"),
        "weak_heading": sum(1 for r in rows if r["status"] == "weak_heading"),
        "weak_paragraph": sum(1 for r in rows if r["status"] == "weak_paragraph"),
        "unmatched_keywords": sum(1 for r in rows if r["status"] == "unmatched"),
        "attributed_traffic": sum(int(r.get("traffic", 0)) for r in rows),
        "heading_targets": len(headings),
        "paragraph_targets": len(paragraphs),
        "provider": (search_payload.get("meta", {}) or {}).get("provider_label") or (search_payload.get("summary", {}) or {}).get("provider_label") or "Search",
    }
    return {
        "summary": summary,
        "keywords": rows[:top_n],
        "headings": headings[:500],
        "paragraphs": paragraphs[:500],
        "interpretation": {
            "status": "matched means both heading and paragraph fit the keyword; weak_heading or weak_paragraph means one side needs reinforcement; unmatched means neither visible section labels nor paragraph text strongly support the query.",
            "attributed_traffic": "Keyword traffic assigned to the best matching heading and paragraph on the ranking URL.",
        },
    }
