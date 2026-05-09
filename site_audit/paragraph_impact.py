"""Estimate which paragraphs carry the most organic-search value.

This is an attribution model, not a claim that a paragraph directly caused
rankings. It blends observable signals we already collect: keyword demand,
paragraph/keyword semantic fit, lexical overlap, heading alignment, paragraph
position, link context, freshness, and page traffic.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any
from urllib.parse import unquote, urlparse, urlunparse

import numpy as np

from .analyzer import PageInfo

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.I)


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower()
    path = unquote(parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower() or "https", host, path, "", "", ""))


def _url_keys(url: str) -> list[str]:
    norm = _normalize_url(url)
    if not norm:
        return []
    parsed = urlparse(norm)
    host = parsed.netloc
    path = parsed.path or "/"
    keys = [norm, path.rstrip("/") or "/"]
    if host.startswith("www."):
        keys.append(urlunparse((parsed.scheme, host[4:], path, "", "", "")))
    else:
        keys.append(urlunparse((parsed.scheme, f"www.{host}", path, "", "", "")))
    return list(dict.fromkeys(keys))


def _page_lookup(pages: list[PageInfo]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, page in enumerate(pages):
        for key in _url_keys(page.url):
            lookup.setdefault(key, i)
    return lookup


def _match_page(url: str, lookup: dict[str, int]) -> int | None:
    for key in _url_keys(url):
        if key in lookup:
            return lookup[key]
    return None


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "") if len(m.group(0)) > 1}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _semantic_component(sim: float) -> float:
    # Similarities from normalized multilingual embeddings are often high.
    # This maps weak topical relation to ~0 and very strong relation to ~1.
    return _clip((sim - 0.35) / 0.5)


def _keyword_weight(row: dict) -> float:
    traffic = _to_int(row.get("traffic"))
    volume = _to_int(row.get("volume"))
    position = _to_int(row.get("position") or row.get("top_keyword_position"))
    # Use traffic first, then volume fallback. Position improves confidence.
    demand = max(1.0, float(traffic), math.sqrt(max(volume, 0)))
    if position > 0:
        demand *= 1.0 + min(0.7, 1.0 / math.sqrt(position))
    return demand


def _keyword_overlap(text_tokens: set[str], keyword: str) -> float:
    kw_tokens = _tokens(keyword)
    if not kw_tokens:
        return 0.0
    return len(text_tokens & kw_tokens) / len(kw_tokens)


def _heading_for_paragraph(ext, para_i: int) -> str:
    """Best-effort parent heading.

    The extractor stores paragraph order and header order separately, but not
    DOM parentage. We approximate with relative order so the UI can still show
    the likely section label without requiring a new extraction format.
    """
    headers = list(getattr(ext, "headers_rich", []) or [])
    if not headers:
        return getattr(ext, "h1", "") or ""
    paragraphs = list(getattr(ext, "paragraphs", []) or [])
    if len(paragraphs) <= 1:
        return headers[0].get("text", "") or getattr(ext, "h1", "") or ""
    max_order = max(1, max(_to_int(h.get("order")) for h in headers))
    para_ratio = (para_i + 0.5) / max(1, len(paragraphs))
    chosen = headers[0]
    for header in sorted(headers, key=lambda h: _to_int(h.get("order"))):
        if (_to_int(header.get("order")) / max_order) <= para_ratio:
            chosen = header
        else:
            break
    return str(chosen.get("text") or getattr(ext, "h1", "") or "")


def _search_context(
    pages: list[PageInfo],
    search_payload: dict,
    *,
    max_keywords_per_page: int,
) -> tuple[dict[int, dict], dict]:
    lookup = _page_lookup(pages)
    provider = str((search_payload.get("meta", {}) or {}).get("provider_label") or (search_payload.get("summary", {}) or {}).get("provider_label") or "Search")
    contexts: dict[int, dict] = defaultdict(lambda: {
        "traffic": 0,
        "keywords_total": 0,
        "top_keyword": "",
        "keywords": [],
        "provider": provider,
    })

    for row in search_payload.get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        ctx = contexts[page_i]
        traffic = _to_int(row.get("traffic"))
        if traffic > ctx["traffic"]:
            ctx["traffic"] = traffic
        ctx["keywords_total"] = max(ctx["keywords_total"], _to_int(row.get("keywords")))
        top_keyword = str(row.get("top_keyword") or "")
        if top_keyword and not ctx["top_keyword"]:
            ctx["top_keyword"] = top_keyword
        if top_keyword:
            ctx["keywords"].append({
                "keyword": top_keyword,
                "traffic": max(traffic, 1),
                "volume": _to_int(row.get("top_keyword_volume")),
                "position": _to_int(row.get("top_keyword_position")),
                "source": "top_page",
            })

    for row in search_payload.get("organic_keywords") or []:
        keyword = str(row.get("keyword") or "")
        if not keyword:
            continue
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        contexts[page_i]["keywords"].append({
            "keyword": keyword,
            "traffic": _to_int(row.get("traffic")),
            "volume": _to_int(row.get("volume")),
            "position": _to_int(row.get("position")),
            "country": row.get("country") or "",
            "intents": row.get("intents") or [],
            "source": "keyword",
        })

    # Deduplicate and cap by demand to keep the per-page scoring bounded.
    for ctx in contexts.values():
        by_keyword: dict[str, dict] = {}
        for row in ctx["keywords"]:
            key = str(row.get("keyword") or "").strip().lower()
            if not key:
                continue
            current = by_keyword.get(key)
            if current is None or _keyword_weight(row) > _keyword_weight(current):
                by_keyword[key] = row
        rows = sorted(by_keyword.values(), key=_keyword_weight, reverse=True)[:max_keywords_per_page]
        ctx["keywords"] = rows
    summary = {
        "provider": provider,
        "pages_with_search_data": len(contexts),
        "total_search_traffic": sum(int(c.get("traffic", 0)) for c in contexts.values()),
        "keyword_rows": sum(len(c.get("keywords") or []) for c in contexts.values()),
    }
    return dict(contexts), summary


def build_paragraph_impact(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    search_payload: dict,
    *,
    embedder=None,
    top_n: int = 500,
    max_keywords_per_page: int = 24,
) -> dict:
    """Return impact summary + top paragraph rows for report rendering."""
    total_paragraphs = len(paragraph_records or [])
    if not paragraph_records:
        return {
            "summary": {"status": "no_paragraphs", "total_paragraphs": 0, "scored_paragraphs": 0},
            "top_paragraphs": [],
            "per_page": [],
        }
    if not search_payload or not ((search_payload.get("top_pages") or []) or (search_payload.get("organic_keywords") or [])):
        return {
            "summary": {"status": "no_search_data", "total_paragraphs": total_paragraphs, "scored_paragraphs": 0},
            "top_paragraphs": [],
            "per_page": [],
        }
    if embedder is None:
        return {
            "summary": {"status": "missing_embedder", "total_paragraphs": total_paragraphs, "scored_paragraphs": 0},
            "top_paragraphs": [],
            "per_page": [],
        }

    contexts, search_summary = _search_context(pages, search_payload, max_keywords_per_page=max_keywords_per_page)
    keyword_texts: list[str] = []
    keyword_index: dict[str, int] = {}
    for ctx in contexts.values():
        for row in ctx.get("keywords") or []:
            keyword = str(row.get("keyword") or "").strip()
            if keyword and keyword.lower() not in keyword_index:
                keyword_index[keyword.lower()] = len(keyword_texts)
                keyword_texts.append(keyword)

    keyword_embs = embedder.encode(keyword_texts, batch_size=64, show_progress=False) if keyword_texts else np.zeros((0, 0), dtype=np.float32)
    max_traffic = max([int(c.get("traffic", 0)) for c in contexts.values()] + [0])

    by_page_rows: dict[int, list[dict]] = defaultdict(list)
    rows: list[dict] = []

    for page_i, para_i, text, emb in paragraph_records:
        ctx = contexts.get(page_i)
        if not ctx or not ctx.get("keywords"):
            continue
        page = pages[page_i]
        ext = extracted_pages[page_i] if page_i < len(extracted_pages) else None
        para_count = len(getattr(ext, "paragraphs", []) or []) if ext is not None else 0
        text_tokens = _tokens(text)
        heading = _heading_for_paragraph(ext, para_i) if ext is not None else ""
        heading_tokens = _tokens(" ".join([heading, page.title or ""]))
        keyword_rows = ctx.get("keywords") or []
        weights = np.asarray([_keyword_weight(row) for row in keyword_rows], dtype=np.float32)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(keyword_rows), dtype=np.float32)

        sims: list[float] = []
        overlaps: list[float] = []
        heading_matches: list[float] = []
        for row in keyword_rows:
            keyword = str(row.get("keyword") or "")
            idx = keyword_index.get(keyword.lower())
            if idx is not None and keyword_embs.size and idx < len(keyword_embs):
                sims.append(float(np.clip(np.asarray(emb, dtype=np.float32) @ keyword_embs[idx], -1.0, 1.0)))
            else:
                sims.append(0.0)
            overlaps.append(_keyword_overlap(text_tokens, keyword))
            heading_matches.append(_keyword_overlap(heading_tokens, keyword))

        sim_arr = np.asarray(sims, dtype=np.float32)
        overlap_arr = np.asarray(overlaps, dtype=np.float32)
        heading_arr = np.asarray(heading_matches, dtype=np.float32)
        semantic = float(np.average([_semantic_component(v) for v in sim_arr], weights=weights))
        overlap = float(np.average(overlap_arr, weights=weights))
        heading_match = float(np.average(heading_arr, weights=weights))
        best_idx = int(np.argmax(sim_arr)) if len(sim_arr) else 0
        best_keyword = keyword_rows[best_idx] if keyword_rows else {}

        position = 1.0 if para_count <= 1 else 1.0 - (para_i / max(1, para_count - 1))
        position_component = 0.4 + 0.6 * _clip(position)
        link_counts = (getattr(ext, "paragraph_link_counts", []) or [])
        internal_links = external_links = 0
        if para_i < len(link_counts):
            internal_links, external_links = link_counts[para_i]
        link_component = min(1.0, (int(internal_links) * 1.0 + int(external_links) * 0.5) / 3.0)
        freshness_component = 1.0 if getattr(ext, "has_dates", False) else 0.45
        page_traffic = int(ctx.get("traffic", 0))
        demand_component = math.log1p(page_traffic) / math.log1p(max_traffic) if max_traffic > 0 else 0.0

        relevance = (
            0.44 * semantic
            + 0.20 * overlap
            + 0.12 * heading_match
            + 0.08 * position_component
            + 0.08 * link_component
            + 0.04 * freshness_component
            + 0.04 * demand_component
        )
        relevance = _clip(relevance)

        row = {
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "paragraph_index": int(para_i),
            "paragraph_excerpt": text[:360],
            "heading": heading,
            "word_count": len(text.split()),
            "page_traffic": page_traffic,
            "page_keywords": int(ctx.get("keywords_total", 0)),
            "top_page_keyword": ctx.get("top_keyword") or "",
            "best_keyword": best_keyword.get("keyword") or "",
            "best_keyword_traffic": _to_int(best_keyword.get("traffic")),
            "best_keyword_volume": _to_int(best_keyword.get("volume")),
            "best_keyword_position": _to_int(best_keyword.get("position")),
            "best_keyword_similarity": round(float(sim_arr[best_idx]) if len(sim_arr) else 0.0, 4),
            "matched_keywords": [
                {
                    "keyword": r.get("keyword") or "",
                    "traffic": _to_int(r.get("traffic")),
                    "volume": _to_int(r.get("volume")),
                    "position": _to_int(r.get("position")),
                    "similarity": round(float(sim_arr[i]), 4),
                    "overlap": round(float(overlap_arr[i]), 4),
                }
                for i, r in sorted(enumerate(keyword_rows), key=lambda item: (float(sim_arr[item[0]]), _keyword_weight(item[1])), reverse=True)[:5]
            ],
            "components": {
                "semantic": round(semantic, 4),
                "keyword_overlap": round(overlap, 4),
                "heading_match": round(heading_match, 4),
                "position": round(position_component, 4),
                "link_context": round(link_component, 4),
                "freshness": round(freshness_component, 4),
                "page_demand": round(demand_component, 4),
            },
            "internal_links": int(internal_links),
            "external_links": int(external_links),
            "relevance_score": round(relevance * 100, 2),
            "_raw_relevance": relevance,
        }
        rows.append(row)
        by_page_rows[page_i].append(row)

    for page_i, page_rows in by_page_rows.items():
        ctx = contexts.get(page_i) or {}
        page_traffic = int(ctx.get("traffic", 0))
        total = sum(float(r.get("_raw_relevance", 0.0)) for r in page_rows) or 1.0
        for row in page_rows:
            share = float(row.get("_raw_relevance", 0.0)) / total
            attributed = page_traffic * share
            row["traffic_share"] = round(share, 4)
            row["attributed_traffic"] = round(attributed, 2)
            row["impact_score"] = round(float(row["relevance_score"]) * math.log1p(max(attributed, 0.0)), 2)

    rows.sort(key=lambda r: (float(r.get("impact_score", 0.0)), float(r.get("attributed_traffic", 0.0))), reverse=True)
    scored = len(rows)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        if scored and rank <= max(1, math.ceil(scored * 0.10)):
            row["impact_tier"] = "high"
        elif scored and rank <= max(1, math.ceil(scored * 0.30)):
            row["impact_tier"] = "medium"
        else:
            row["impact_tier"] = "low"
        row.pop("_raw_relevance", None)

    per_page = []
    for page_i, page_rows in by_page_rows.items():
        page = pages[page_i]
        sorted_rows = sorted(page_rows, key=lambda r: float(r.get("impact_score", 0.0)), reverse=True)
        per_page.append({
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "page_traffic": int((contexts.get(page_i) or {}).get("traffic", 0)),
            "scored_paragraphs": len(page_rows),
            "top_paragraphs": [
                {
                    "paragraph_index": r["paragraph_index"],
                    "paragraph_excerpt": r["paragraph_excerpt"],
                    "impact_score": r["impact_score"],
                    "attributed_traffic": r["attributed_traffic"],
                    "best_keyword": r["best_keyword"],
                }
                for r in sorted_rows[:5]
            ],
        })
    per_page.sort(key=lambda r: (int(r.get("page_traffic", 0)), len(r.get("top_paragraphs") or [])), reverse=True)

    summary = {
        "status": "ok" if rows else "no_matching_search_pages",
        "model": "paragraph_impact_v1",
        "total_paragraphs": total_paragraphs,
        "scored_paragraphs": scored,
        "top_rows": min(top_n, scored),
        "high_impact_count": sum(1 for r in rows if r.get("impact_tier") == "high"),
        "medium_impact_count": sum(1 for r in rows if r.get("impact_tier") == "medium"),
        "low_impact_count": sum(1 for r in rows if r.get("impact_tier") == "low"),
        "attributed_traffic": round(sum(float(r.get("attributed_traffic", 0.0)) for r in rows), 2),
        **search_summary,
    }
    return {
        "summary": summary,
        "top_paragraphs": rows[:top_n],
        "per_page": per_page[:500],
        "interpretation": {
            "impact_score": "Relevance score multiplied by estimated traffic attribution. Higher means the paragraph appears more valuable to protect or improve.",
            "attributed_traffic": "Page organic traffic distributed across paragraphs by relative paragraph relevance. This is an estimate, not observed analytics.",
            "components": "semantic, keyword_overlap, heading_match, position, link_context, freshness, and page_demand are normalized 0-1 signals.",
        },
    }
