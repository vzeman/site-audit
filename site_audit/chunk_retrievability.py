"""Model query answerability at retrieval-chunk granularity.

This analysis deliberately reuses paragraph and query embeddings computed by
keyword coverage. It must not call an embedder: chunks are word-count-weighted
means of existing paragraph vectors, then L2-normalized.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .paragraph_impact import _heading_for_paragraph

MIN_CHUNK_WORDS = 120
MAX_CHUNK_WORDS = 250
DEFAULT_STRONG_SIMILARITY = 0.65
RECOMMENDATION_LIMIT = 20

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _stable_slug(*parts: object) -> str:
    base = "-".join(str(part or "") for part in parts)
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if len(slug) > 80:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:80].strip('-')}-{digest}"
    return slug or hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


# Positional map for the pipeline's paragraph-record tuples:
# (page_index, paragraph_index, text, embedding). Tuples carry no
# url/title/heading — those derive from pages[page_index] / the extractor,
# so those keys use pos=None (dict records only).
def _record_value(record: Any, key: str, pos: int | None, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    if pos is None:
        return default
    try:
        return record[pos]
    except (IndexError, TypeError):
        return default


def _paragraph_index(record: Any) -> int:
    value = _record_value(record, "paragraph_index", 1, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _record_text(record: Any) -> str:
    return str(_record_value(record, "text", 2, "") or "")


def _record_heading(record: Any) -> str:
    return str(_record_value(record, "heading", None, "") or "")


def build_chunks(paragraph_records_for_page: Iterable[Any]) -> list[dict]:
    """Greedily merge consecutive paragraphs into retrieval windows.

    Paragraph order is the source of truth. A heading change always starts a
    new chunk; paragraphs above 250 words stand alone because they already
    exceed the retrieval-window target. Chunks target ``<= MAX_CHUNK_WORDS``
    words; ``MIN_CHUNK_WORDS`` is a soft minimum — a leftover chunk below it
    is merged back into the previous chunk when both share a heading,
    otherwise it is kept as-is.
    """
    records = list(paragraph_records_for_page or [])
    chunks: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            chunks.append(current)
            current = None

    for record in records:
        heading = _record_heading(record)
        para_index = _paragraph_index(record)
        words = _word_count(_record_text(record))
        if words <= 0:
            continue
        if words > MAX_CHUNK_WORDS:
            flush()
            chunks.append({
                "heading": heading,
                "paragraph_indexes": [para_index],
                "word_count": words,
            })
            continue
        if current is None:
            current = {
                "heading": heading,
                "paragraph_indexes": [para_index],
                "word_count": words,
            }
            continue
        if heading != current["heading"]:
            flush()
            current = {
                "heading": heading,
                "paragraph_indexes": [para_index],
                "word_count": words,
            }
            continue
        if int(current["word_count"]) + words <= MAX_CHUNK_WORDS:
            current["paragraph_indexes"].append(para_index)
            current["word_count"] = int(current["word_count"]) + words
        else:
            flush()
            current = {
                "heading": heading,
                "paragraph_indexes": [para_index],
                "word_count": words,
            }

    flush()

    # Soft minimum: fold sub-MIN_CHUNK_WORDS leftovers into the previous
    # chunk when both sit under the same heading; keep them otherwise.
    merged: list[dict] = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and int(chunk["word_count"]) < MIN_CHUNK_WORDS
            and chunk["heading"] == prev["heading"]
        ):
            prev["paragraph_indexes"].extend(chunk["paragraph_indexes"])
            prev["word_count"] = int(prev["word_count"]) + int(chunk["word_count"])
            continue
        merged.append(chunk)
    return merged


def _coerce_records(
    pages_or_records: Any,
    paragraph_embeddings: Any,
) -> tuple[list[Any], list[Any], list[dict], np.ndarray | None]:
    pages: list[Any] = []
    extracted_pages: list[Any] = []
    records_raw: list[Any]

    if isinstance(pages_or_records, dict):
        pages = list(pages_or_records.get("pages") or [])
        extracted_pages = list(pages_or_records.get("extracted_pages") or [])
        records_raw = list(pages_or_records.get("paragraph_records") or pages_or_records.get("records") or [])
    else:
        records_raw = list(pages_or_records or [])

    if paragraph_embeddings is None:
        embedded = [_record_value(record, "embedding", 3) for record in records_raw]
        if not embedded or any(value is None for value in embedded):
            embeddings = None
        else:
            embeddings = np.asarray(embedded, dtype=np.float32)
    else:
        embeddings = np.asarray(paragraph_embeddings, dtype=np.float32)

    records: list[dict] = []
    for pos, record in enumerate(records_raw):
        page_index_value = _record_value(record, "page_index", 0, None)
        try:
            page_index = int(page_index_value)
        except (TypeError, ValueError):
            page_index = None
        page = pages[page_index] if page_index is not None and 0 <= page_index < len(pages) else None
        ext = extracted_pages[page_index] if page_index is not None and 0 <= page_index < len(extracted_pages) else None
        para_index = _paragraph_index(record)
        heading = _record_heading(record)
        if not heading and ext is not None:
            heading = _heading_for_paragraph(ext, para_index)
        records.append({
            "page_index": page_index,
            "url": str(_record_value(record, "url", None, "") or getattr(page, "url", "") or ""),
            "title": str(_record_value(record, "title", None, "") or getattr(page, "title", "") or ""),
            "paragraph_index": para_index,
            "text": _record_text(record),
            "heading": heading,
            "record_pos": pos,
            "word_count": _word_count(_record_text(record)),
        })

    return pages, extracted_pages, records, embeddings


def _chunk_embedding(chunk: dict, records_for_page: list[dict], embeddings: np.ndarray) -> np.ndarray:
    # Select by record position rather than a paragraph_index dict so a
    # duplicated paragraph_index cannot silently drop a vector.
    wanted = {int(idx) for idx in chunk.get("paragraph_indexes") or []}
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for record in records_for_page:
        if int(record["paragraph_index"]) not in wanted:
            continue
        vectors.append(embeddings[int(record["record_pos"])].astype(np.float32, copy=False))
        weights.append(float(max(1, int(record.get("word_count") or 0))))
    if not vectors:
        return np.zeros((embeddings.shape[1],), dtype=np.float32)
    avg = np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights, dtype=np.float32))
    return _l2_normalize(avg.astype(np.float32, copy=False))


def _query_importance(row: dict) -> float:
    fields = (
        "impressions", "query_impressions", "volume", "search_volume",
        "query_volume", "top_keyword_volume", "clicks", "query_clicks",
        "traffic", "query_traffic",
    )
    values: list[float] = []
    for field in fields:
        try:
            values.append(float(row.get(field) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _query_has_demand(row: dict) -> bool:
    return _query_importance(row) > 0


def _page_matters(row: dict) -> bool:
    if row.get("top_traffic_page"):
        return True
    try:
        return float(row.get("page_traffic") or 0.0) > 0
    except (TypeError, ValueError):
        return False


def _mapped_url(row: dict) -> str:
    return str(row.get("best_url") or row.get("url") or row.get("matched_url") or "")


def _mapped_title(row: dict) -> str:
    return str(row.get("best_title") or row.get("title") or row.get("page_title") or "")


def _round_score(value: float) -> float:
    return float(round(float(value), 3))


def _select_query_rows(query_rows: Iterable[dict], max_queries: int) -> tuple[list[dict], int]:
    rows = [dict(row, _source_index=i) for i, row in enumerate(query_rows or []) if row.get("query")]
    ordered = sorted(rows, key=lambda row: (-_query_importance(row), int(row["_source_index"])))
    truncated = max(0, len(ordered) - max_queries)
    return ordered[:max_queries], truncated


def _unavailable(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "summary": {"status": "unavailable", "reason": reason},
        "model": {
            "strong_similarity": DEFAULT_STRONG_SIMILARITY,
            "split_margin": 0.05,
            "chunk_min_words": MIN_CHUNK_WORDS,
            "chunk_max_words": MAX_CHUNK_WORDS,
        },
        "queries": [],
        "pages": [],
        "recommendations": [],
    }


def build_chunk_retrievability(
    pages_or_records: Any,
    paragraph_embeddings: Any,
    query_rows: Iterable[dict] | None,
    query_embeddings: Any,
    *,
    strong: float = DEFAULT_STRONG_SIMILARITY,
    split_margin: float = 0.05,
    max_queries: int = 200,
) -> dict:
    """Score whether each mapped page has a single retrievable answer chunk."""
    all_query_rows = list(query_rows or [])
    selected_rows, truncated = _select_query_rows(all_query_rows, max_queries)
    if not selected_rows:
        return _unavailable("keyword coverage rows unavailable")
    if query_embeddings is None or len(query_embeddings) == 0:
        return _unavailable("query embeddings unavailable")

    _, _, records, embeddings = _coerce_records(pages_or_records, paragraph_embeddings)
    if not records:
        return _unavailable("paragraph records unavailable")
    if embeddings is None or len(embeddings) != len(records):
        return _unavailable("paragraph embeddings unavailable")

    q_embs = np.asarray(query_embeddings, dtype=np.float32)
    if len(q_embs) < len(all_query_rows):
        return _unavailable("query embeddings do not align with coverage rows")

    records_by_url: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_url[record["url"]].append(record)

    chunks_by_url: dict[str, list[dict]] = {}
    chunk_embeddings_by_url: dict[str, np.ndarray] = {}
    for url, page_records in records_by_url.items():
        page_records.sort(key=lambda record: int(record["paragraph_index"]))
        chunks = build_chunks(page_records)
        chunks_by_url[url] = chunks
        if chunks:
            chunk_embeddings_by_url[url] = np.vstack([
                _chunk_embedding(chunk, page_records, embeddings)
                for chunk in chunks
            ]).astype(np.float32)

    query_results: list[dict] = []
    page_rollups: dict[str, dict] = {}
    counts: dict[str, int] = {"retrievable": 0, "split_answer": 0, "missing": 0}

    for row in selected_rows:
        source_index = int(row["_source_index"])
        query = str(row.get("query") or "")
        url = _mapped_url(row)
        chunks = chunks_by_url.get(url) or []
        chunk_embs = chunk_embeddings_by_url.get(url)
        q = _l2_normalize(q_embs[source_index].astype(np.float32, copy=False))

        best_score = 0.0
        second_score = 0.0
        best_chunk: dict = {}
        second_chunk: dict = {}
        if chunk_embs is not None and len(chunks):
            sims = np.clip(chunk_embs @ q, -1.0, 1.0)
            order = np.argsort(-sims)
            best_idx = int(order[0])
            best_score = float(sims[best_idx])
            best_chunk = chunks[best_idx]
            if len(order) > 1:
                second_idx = int(order[1])
                second_score = float(sims[second_idx])
                second_chunk = chunks[second_idx]

        if best_score >= strong:
            status = "retrievable"
        elif (
            best_chunk
            and second_chunk
            and best_score < strong
            and best_score - second_score <= split_margin + 1e-6
            and best_score >= strong - 0.1
            and second_score >= strong - 0.1
        ):
            status = "split_answer"
        else:
            status = "missing"
        counts[status] += 1

        result = {
            "query": query,
            "source": row.get("source", ""),
            "url": url,
            "title": _mapped_title(row),
            "status": status,
            "best_similarity": _round_score(best_score),
            "second_similarity": _round_score(second_score),
            "chunk_heading": best_chunk.get("heading", "") if best_chunk else "",
            "paragraph_indexes": list(best_chunk.get("paragraph_indexes") or []),
            "second_heading": second_chunk.get("heading", "") if status == "split_answer" and second_chunk else "",
            "second_paragraph_indexes": list(second_chunk.get("paragraph_indexes") or []) if status == "split_answer" and second_chunk else [],
            "chunk_word_count": int(best_chunk.get("word_count") or 0) if best_chunk else 0,
            "query_importance": _round_score(_query_importance(row)),
            "has_demand": _query_has_demand(row),
            "top_traffic_page": _page_matters(row),
        }
        query_results.append(result)

        rollup = page_rollups.setdefault(url, {
            "url": url,
            "title": result["title"],
            "queries": 0,
            "retrievable": 0,
            "split_answer": 0,
            "missing": 0,
            "weakest_queries": [],
        })
        rollup["queries"] += 1
        rollup[status] += 1
        if status != "retrievable":
            rollup["weakest_queries"].append({
                "query": query,
                "status": status,
                "best_similarity": result["best_similarity"],
                "chunk_heading": result["chunk_heading"],
                "second_heading": result["second_heading"],
            })

    pages: list[dict] = []
    for rollup in page_rollups.values():
        total = max(1, int(rollup["queries"]))
        rollup["retrievable_share"] = round(float(rollup["retrievable"]) / total, 3)
        rollup["split_answer_share"] = round(float(rollup["split_answer"]) / total, 3)
        rollup["weakest_queries"] = sorted(
            rollup["weakest_queries"],
            key=lambda item: (float(item.get("best_similarity") or 0.0), item.get("query", "")),
        )[:5]
        pages.append(rollup)
    pages.sort(key=lambda row: (row["retrievable_share"], -row["queries"], row["url"]))

    recommendations, rec_truncated = _build_recommendations(query_results)
    summary = {
        "status": "ok",
        "queries": len(query_results),
        "retrievable": counts["retrievable"],
        "split_answer": counts["split_answer"],
        "missing": counts["missing"],
        "retrievable_share": round(counts["retrievable"] / max(1, len(query_results)), 3),
        "pages": len(pages),
        "recommendations": len(recommendations),
    }
    if truncated:
        summary["truncated"] = truncated
    if rec_truncated:
        summary["recommendations_truncated"] = rec_truncated

    return {
        "available": True,
        "summary": summary,
        "model": {
            # Calibrated to keyword_coverage.match_queries_to_paragraphs:
            # 0.65 is the existing paragraph similarity floor for a plausible
            # answer, so a chunk must clear the same cosine threshold.
            "strong_similarity": float(strong),
            "split_margin": float(split_margin),
            "split_floor": round(float(strong) - 0.1, 3),
            "chunk_min_words": MIN_CHUNK_WORDS,
            "chunk_max_words": MAX_CHUNK_WORDS,
            "max_queries": int(max_queries),
            "embedding_source": "precomputed paragraph and keyword-coverage query embeddings",
            "calibration": "strong defaults to 0.65, matching the keyword coverage paragraph similarity floor.",
        },
        "queries": query_results,
        "pages": pages,
        "recommendations": recommendations,
    }


def _build_recommendations(rows: list[dict]) -> tuple[list[dict], int]:
    candidates: list[dict] = []
    for row in rows:
        status = row.get("status")
        has_demand = bool(row.get("has_demand"))
        page_matters = bool(row.get("top_traffic_page"))
        if status == "split_answer" and (has_demand or page_matters):
            query = str(row.get("query") or "")
            heading_a = str(row.get("chunk_heading") or "the best chunk")
            heading_b = str(row.get("second_heading") or "the second chunk")
            url = str(row.get("url") or "")
            best = float(row.get("best_similarity") or 0.0)
            candidates.append({
                "id": f"geo-chunk-{_stable_slug(query, url)}",
                "status": status,
                "url": url,
                "query": query,
                "text": (
                    f'For "{query}", the answer is split between "{heading_a}" and "{heading_b}" on {url}. '
                    f"Consolidate into one self-contained passage under a single H2 phrased as the question; "
                    f"best chunk currently scores {best:.2f}."
                ),
                "effort": "medium",
                "best_similarity": round(best, 2),
                "heading_a": heading_a,
                "heading_b": heading_b,
            })
        elif status == "missing" and has_demand:
            query = str(row.get("query") or "")
            url = str(row.get("url") or "")
            best = float(row.get("best_similarity") or 0.0)
            candidates.append({
                "id": f"geo-chunk-{_stable_slug(query, url)}",
                "status": status,
                "url": url,
                "query": query,
                "text": (
                    f'"{url}" has no chunk that answers "{query}" (best {best:.2f}). '
                    "Add a 40–90 word passage under an H2 phrased as the question, "
                    "stating the answer in the first sentence."
                ),
                "effort": "medium",
                "best_similarity": round(best, 2),
            })

    candidates.sort(key=lambda rec: (rec["status"] != "missing", rec["best_similarity"], rec["id"]))
    return candidates[:RECOMMENDATION_LIMIT], max(0, len(candidates) - RECOMMENDATION_LIMIT)
