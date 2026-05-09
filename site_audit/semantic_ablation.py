"""Semantic ablation for paragraph importance.

For each paragraph, estimate how page/topic alignment changes if that
paragraph is virtually removed from the page's paragraph centroid. Positive
delta means the paragraph carries topic alignment; negative delta means the
page becomes more focused without it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .analyzer import PageInfo
from .paragraph_impact import _keyword_weight, _search_context


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm <= 0 else arr / norm


def _centroid(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(0, dtype=np.float32)
    return _normalize(np.mean(np.stack([np.asarray(v, dtype=np.float32) for v in vectors]), axis=0))


def _score_label(delta: float, self_alignment: float) -> str:
    if delta >= 0.015 or (delta > 0.006 and self_alignment >= 0.72):
        return "topic_carrier"
    if delta <= -0.015:
        return "noise_candidate"
    return "neutral"


def _row_priority(row: dict) -> float:
    delta = abs(float(row.get("alignment_delta", 0.0)))
    self_alignment = max(0.0, float(row.get("self_alignment", 0.0)))
    return delta * 100 + self_alignment


def build_semantic_ablation(
    pages: list[PageInfo],
    page_embeddings: np.ndarray,
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    search_payload: dict | None,
    *,
    embedder=None,
    top_n: int = 500,
    max_keywords_per_page: int = 24,
) -> dict:
    total_paragraphs = len(paragraph_records or [])
    if not paragraph_records:
        return {"summary": {"status": "no_paragraphs", "total_paragraphs": 0, "scored_paragraphs": 0}, "rows": []}

    by_page: dict[int, list[tuple[int, int, str, np.ndarray]]] = defaultdict(list)
    for rec in paragraph_records:
        by_page[int(rec[0])].append(rec)

    contexts: dict[int, dict] = {}
    search_summary: dict[str, Any] = {"provider": "page_embedding", "pages_with_search_data": 0, "keyword_rows": 0}
    if search_payload:
        contexts, search_summary = _search_context(pages, search_payload, max_keywords_per_page=max_keywords_per_page)

    keyword_texts: list[str] = []
    keyword_index: dict[str, int] = {}
    for ctx in contexts.values():
        for row in ctx.get("keywords") or []:
            keyword = str(row.get("keyword") or "").strip()
            if keyword and keyword.lower() not in keyword_index:
                keyword_index[keyword.lower()] = len(keyword_texts)
                keyword_texts.append(keyword)
    keyword_embs = (
        embedder.encode(keyword_texts, batch_size=64, show_progress=False)
        if keyword_texts and embedder is not None
        else np.zeros((0, 0), dtype=np.float32)
    )

    rows: list[dict] = []
    for page_i, records in by_page.items():
        if page_i >= len(pages) or page_i >= len(page_embeddings):
            continue
        if len(records) < 2:
            continue
        page = pages[page_i]
        ext = extracted_pages[page_i] if page_i < len(extracted_pages) else None
        para_embs = [np.asarray(r[3], dtype=np.float32) for r in records]
        full_centroid = _centroid(para_embs)
        if not full_centroid.size:
            continue

        ctx = contexts.get(page_i) or {}
        keyword_rows = ctx.get("keywords") or []
        target = None
        target_label = "page embedding"
        best_keywords: list[dict] = []
        if keyword_rows and keyword_embs.size:
            target_vectors = []
            weights = []
            for row in keyword_rows:
                keyword = str(row.get("keyword") or "")
                idx = keyword_index.get(keyword.lower())
                if idx is None or idx >= len(keyword_embs):
                    continue
                target_vectors.append(np.asarray(keyword_embs[idx], dtype=np.float32))
                weights.append(_keyword_weight(row))
            if target_vectors:
                weights_arr = np.asarray(weights, dtype=np.float32)
                stacked = np.stack(target_vectors)
                target = _normalize(np.average(stacked, axis=0, weights=weights_arr))
                target_label = "ranking keywords"
                best_keywords = [
                    {
                        "keyword": row.get("keyword") or "",
                        "traffic": int(row.get("traffic") or 0),
                        "volume": int(row.get("volume") or 0),
                        "position": int(row.get("position") or 0),
                    }
                    for row in sorted(keyword_rows, key=_keyword_weight, reverse=True)[:5]
                ]
        if target is None:
            target = _normalize(page_embeddings[page_i])

        full_alignment = float(np.clip(full_centroid @ target, -1.0, 1.0))
        for local_idx, (src_page_i, para_i, text, emb) in enumerate(records):
            remaining = [para_embs[i] for i in range(len(para_embs)) if i != local_idx]
            if not remaining:
                continue
            without_centroid = _centroid(remaining)
            without_alignment = float(np.clip(without_centroid @ target, -1.0, 1.0))
            delta = full_alignment - without_alignment
            self_alignment = float(np.clip(np.asarray(emb, dtype=np.float32) @ target, -1.0, 1.0))
            label = _score_label(delta, self_alignment)
            rows.append({
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "paragraph_index": int(para_i),
                "paragraph_excerpt": text[:360],
                "word_count": len(text.split()),
                "target": target_label,
                "page_traffic": int(ctx.get("traffic", 0) or 0),
                "top_keywords": best_keywords,
                "full_alignment": round(full_alignment, 4),
                "without_alignment": round(without_alignment, 4),
                "alignment_delta": round(delta, 4),
                "self_alignment": round(self_alignment, 4),
                "ablation_score": round(delta * 100, 3),
                "classification": label,
                "classification_label": {
                    "topic_carrier": "Topic carrier",
                    "noise_candidate": "Noise candidate",
                    "neutral": "Neutral",
                }[label],
                "has_dates": bool(getattr(ext, "has_dates", False)),
            })

    rows.sort(key=_row_priority, reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    topic_carriers = sum(1 for r in rows if r["classification"] == "topic_carrier")
    noise_candidates = sum(1 for r in rows if r["classification"] == "noise_candidate")
    summary = {
        "status": "ok" if rows else "no_scored_pages",
        "model": "semantic_ablation_v1",
        "total_paragraphs": total_paragraphs,
        "scored_paragraphs": len(rows),
        "scored_pages": len({r["url"] for r in rows}),
        "topic_carriers": topic_carriers,
        "noise_candidates": noise_candidates,
        "neutral": len(rows) - topic_carriers - noise_candidates,
        **search_summary,
    }
    return {
        "summary": summary,
        "rows": rows[:top_n],
        "topic_carriers": [r for r in rows if r["classification"] == "topic_carrier"][:top_n],
        "noise_candidates": [r for r in rows if r["classification"] == "noise_candidate"][:top_n],
        "interpretation": {
            "alignment_delta": "Full page paragraph-centroid alignment minus alignment after removing the paragraph. Positive means removing it hurts topical alignment; negative means removing it improves alignment.",
            "topic_carrier": "A paragraph whose removal lowers alignment enough to mark it as carrying the topic.",
            "noise_candidate": "A paragraph whose removal improves alignment enough to mark it as possible topical noise.",
        },
    }
