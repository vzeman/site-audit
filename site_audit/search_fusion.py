"""Combine multiple search-provider payloads into one semantic map."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ahrefs import AhrefsAnalysis, _entity_alignment, _normalize_url, _semantic_map
from .analyzer import PageInfo


def build_combined_search_analysis(
    provider_payloads: list[dict],
    pages: list[PageInfo],
    embeddings: np.ndarray,
    *,
    extracted_pages: Optional[list] = None,
    paragraph_records: Optional[list] = None,
    linkbuilding: Optional[dict] = None,
    embedder=None,
    semantic_sample_cap: int = 500,
) -> AhrefsAnalysis:
    payloads = [_tagged_payload(p) for p in provider_payloads if _has_search_rows(p)]
    if not payloads:
        return AhrefsAnalysis(payload={}, semantic_rows=[], semantic_embeddings=None)
    if len(payloads) == 1:
        return AhrefsAnalysis(payload=payloads[0], semantic_rows=[], semantic_embeddings=None)

    top_pages = _merge_top_pages(payloads)
    keywords = _merge_keywords(payloads)
    query_pages = _merge_query_pages(payloads)
    clusters = _merge_clusters(payloads)
    semantic_points, semantic_rows, semantic_embeddings = _semantic_map(
        pages,
        embeddings,
        top_pages,
        keywords,
        extracted_pages or [],
        paragraph_records or [],
        linkbuilding or {},
        embedder=embedder,
        sample_cap=semantic_sample_cap,
    )
    primary = payloads[0]
    combined = {
        **primary,
        "meta": {
            "provider": "combined",
            "provider_label": "Combined search sources",
            "status": "ok",
            "cache_status": "mixed",
            "providers": [_provider_summary(p) for p in payloads],
        },
        "summary": _combined_summary(payloads, top_pages, keywords, clusters),
        "top_pages": top_pages,
        "organic_keywords": keywords,
        "query_pages": query_pages,
        "clusters": clusters,
        "semantic_map": {
            "points": semantic_points,
            "shown": len(semantic_points),
            "entity_types": ["page", "page_title", "keyword", "header", "paragraph", "link_title"],
            "providers": [_provider_summary(p) for p in payloads],
        },
        "entity_alignment": _entity_alignment(semantic_rows, semantic_embeddings),
        "provider_payloads": {
            (p.get("meta", {}) or {}).get("provider", "search"): _compact_provider_payload(p)
            for p in payloads
        },
    }
    return AhrefsAnalysis(payload=combined, semantic_rows=semantic_rows, semantic_embeddings=semantic_embeddings)


def _has_search_rows(payload: dict) -> bool:
    if not payload:
        return False
    summary = payload.get("summary") or {}
    return bool(
        payload.get("top_pages")
        or payload.get("organic_keywords")
        or payload.get("query_pages")
        or summary.get("top_pages")
        or summary.get("organic_keywords")
    )


def _tagged_payload(payload: dict) -> dict:
    meta = dict(payload.get("meta") or {})
    summary = dict(payload.get("summary") or {})
    provider = (summary.get("provider") or meta.get("provider") or "search").lower()
    label = summary.get("provider_label") or meta.get("provider_label") or provider
    out = dict(payload)
    out["meta"] = {**meta, "provider": provider, "provider_label": label}
    out["summary"] = {**summary, "provider": provider, "provider_label": label}
    out["top_pages"] = [{**row, "provider": provider, "provider_label": label} for row in (payload.get("top_pages") or [])]
    out["organic_keywords"] = [{**row, "provider": provider, "provider_label": label} for row in (payload.get("organic_keywords") or [])]
    out["query_pages"] = [
        {**row, "source": row.get("source") or provider, "provider": provider, "provider_label": label}
        for row in (payload.get("query_pages") or [])
    ]
    return out


def _merge_top_pages(payloads: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for payload in payloads:
        for row in payload.get("top_pages") or []:
            url = row.get("matched_url") or row.get("url") or ""
            if not url:
                continue
            current = by_url.get(url)
            if current is None or _num(row.get("traffic")) > _num(current.get("traffic")):
                by_url[url] = dict(row)
    return sorted(by_url.values(), key=lambda r: _num(r.get("traffic")), reverse=True)


def _merge_keywords(payloads: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for payload in payloads:
        rows.extend(dict(row) for row in (payload.get("organic_keywords") or []))
    rows.sort(key=lambda r: max(_num(r.get("traffic")), _num(r.get("paid_cost")), _num(r.get("volume")) / 100), reverse=True)
    return rows


def _merge_query_pages(payloads: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for payload in payloads:
        for row in payload.get("query_pages") or []:
            query = str(row.get("query") or row.get("keyword") or "").strip().lower()
            url = _normalize_url(str(row.get("matched_url") or row.get("url") or "").strip())
            if not query or not url:
                continue
            key = (query, url)
            current = by_key.get(key)
            if current is None or _prefer_query_page(row, current):
                by_key[key] = dict(row)
    rows = list(by_key.values())
    rows.sort(key=lambda r: _num(r.get("impressions")), reverse=True)
    return rows


def _prefer_query_page(candidate: dict, current: dict) -> bool:
    candidate_source = str(candidate.get("source") or candidate.get("provider") or "").lower()
    current_source = str(current.get("source") or current.get("provider") or "").lower()
    if candidate_source == "gsc" and current_source != "gsc":
        return True
    if current_source == "gsc" and candidate_source != "gsc":
        return False
    return _num(candidate.get("impressions")) > _num(current.get("impressions"))


def _merge_clusters(payloads: list[dict]) -> list[dict]:
    by_cluster: dict[str, dict] = {}
    for payload in payloads:
        provider = (payload.get("meta") or {}).get("provider", "search")
        label = (payload.get("meta") or {}).get("provider_label", provider)
        for row in payload.get("clusters") or []:
            key = str(row.get("cluster") if row.get("cluster") is not None else row.get("key", ""))
            if not key:
                continue
            current = by_cluster.setdefault(key, {
                "key": key,
                "cluster": row.get("cluster") if row.get("cluster") is not None else row.get("key"),
                "label": row.get("label") or f"cluster {key}",
                "traffic": 0,
                "keyword_traffic": 0,
                "paid_traffic": 0,
                "value_usd": 0.0,
                "pages": 0,
                "matched_pages": 0,
                "keyword_rows": 0,
                "top_keywords": [],
                "providers": [],
            })
            current["traffic"] += int(_num(row.get("traffic")))
            current["keyword_traffic"] += int(_num(row.get("keyword_traffic")))
            current["paid_traffic"] += int(_num(row.get("paid_traffic")))
            current["value_usd"] = round(float(current.get("value_usd", 0.0)) + _num(row.get("value_usd")), 2)
            current["pages"] = max(int(current.get("pages", 0)), int(_num(row.get("pages"))))
            current["matched_pages"] = max(int(current.get("matched_pages", 0)), int(_num(row.get("matched_pages"))))
            current["keyword_rows"] += int(_num(row.get("keyword_rows")))
            current["providers"].append({"provider": provider, "label": label, "traffic": int(_num(row.get("traffic")))})
            seen = {kw.get("keyword") for kw in current["top_keywords"] if isinstance(kw, dict)}
            for kw in row.get("top_keywords") or []:
                keyword = kw.get("keyword") if isinstance(kw, dict) else str(kw)
                if keyword and keyword not in seen and len(current["top_keywords"]) < 12:
                    current["top_keywords"].append(kw if isinstance(kw, dict) else {"keyword": keyword})
                    seen.add(keyword)
    rows = list(by_cluster.values())
    rows.sort(key=lambda r: int(r.get("traffic", 0)), reverse=True)
    return rows


def _combined_summary(payloads: list[dict], top_pages: list[dict], keywords: list[dict], clusters: list[dict]) -> dict:
    start_date = next((str((p.get("summary") or {}).get("start_date")) for p in payloads if (p.get("summary") or {}).get("start_date")), "")
    end_date = next((str((p.get("summary") or {}).get("end_date")) for p in payloads if (p.get("summary") or {}).get("end_date")), "")
    return {
        "provider": "combined",
        "provider_label": "Combined search sources",
        "start_date": start_date,
        "end_date": end_date,
        "providers": len(payloads),
        "top_pages": len(top_pages),
        "organic_keywords": len(keywords),
        "top_pages_traffic": int(sum(_num(r.get("traffic")) for r in top_pages)),
        "matched_top_pages": sum(1 for r in top_pages if r.get("matched")),
        "matched_traffic": int(sum(_num(r.get("traffic")) for r in top_pages if r.get("matched"))),
        "paid_cost": round(sum(_num(r.get("paid_cost")) for r in keywords), 2),
        "paid_conversions": round(sum(_num(r.get("paid_conversions")) for r in keywords), 2),
        "paid_conversion_value": round(sum(_num(r.get("paid_conversion_value")) for r in keywords), 2),
        "traffic_clusters": len(clusters),
        "provider_breakdown": [_provider_summary(p) for p in payloads],
    }


def _provider_summary(payload: dict) -> dict:
    meta = payload.get("meta") or {}
    summary = payload.get("summary") or {}
    provider = summary.get("provider") or meta.get("provider") or "search"
    return {
        "provider": provider,
        "label": summary.get("provider_label") or meta.get("provider_label") or provider,
        "status": meta.get("status", ""),
        "cache_status": meta.get("cache_status", ""),
        "top_pages": int(summary.get("top_pages") or len(payload.get("top_pages") or [])),
        "keywords": int(summary.get("organic_keywords") or len(payload.get("organic_keywords") or [])),
        "traffic": int(summary.get("top_pages_traffic") or summary.get("total_clicks") or summary.get("paid_cost") or 0),
    }


def _compact_provider_payload(payload: dict) -> dict:
    return {
        "meta": payload.get("meta") or {},
        "summary": payload.get("summary") or {},
        "top_pages": (payload.get("top_pages") or [])[:200],
        "organic_keywords": (payload.get("organic_keywords") or [])[:500],
        "query_pages": (payload.get("query_pages") or [])[:500],
    }


def _num(value) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
