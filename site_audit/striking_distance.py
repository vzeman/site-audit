"""Striking-distance keyword opportunities from query+page rankings."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

from .ahrefs import _match_page, _page_lookup
from .analyzer import PageInfo
from .ctr_curve import estimate_clicks_gain, expected_ctr


MAX_PAGE_GROUPS = 100
MAX_QUERIES_PER_PAGE = 20


def build_striking_distance(
    search_payload: dict | None,
    pages: list[PageInfo] | None,
    *,
    min_position: float = 4.0,
    max_position: float = 20.0,
    min_impressions: int = 10,
    target_position: float = 3.0,
    max_rows: int = 200,
) -> dict:
    """Build a modeled list of keywords close enough to improve."""
    query_pages = list((search_payload or {}).get("query_pages") or [])
    if not query_pages:
        return {
            "available": False,
            "reason": "No search provider supplied query+page ranking rows.",
            "rows": [],
            "pages": [],
            "summary": {"status": "unavailable"},
            "model": _model(search_payload or {}, target_position),
        }

    lookup = _page_lookup(pages or [])
    scored: list[dict] = []
    for row in query_pages:
        if _is_branded(row):
            continue
        position = _to_float(row.get("position"))
        impressions = _to_float(row.get("impressions"))
        if position < min_position or position > max_position or impressions < min_impressions:
            continue
        current_ctr = expected_ctr(position)
        target_ctr = expected_ctr(target_position)
        gain = estimate_clicks_gain(impressions, position, target_position)
        if gain <= 0:
            continue
        scored.append(_opportunity_row(
            row,
            pages or [],
            lookup,
            position=position,
            impressions=impressions,
            current_ctr=current_ctr,
            target_ctr=target_ctr,
            estimated_gain=gain,
        ))

    scored.sort(key=lambda r: r["estimated_clicks_gain"], reverse=True)
    emitted = scored[:max_rows]
    start_date, end_date = _window_dates(search_payload or {})
    return {
        "available": True,
        "summary": {
            "status": "ok",
            "opportunities": len(scored),
            "shown": len(emitted),
            "total_modeled_click_gain": round(sum(_to_float(r.get("estimated_clicks_gain")) for r in scored), 2),
            "start_date": start_date,
            "end_date": end_date,
            "min_position": min_position,
            "max_position": max_position,
            "min_impressions": min_impressions,
        },
        "model": _model(search_payload or {}, target_position),
        "rows": emitted,
        "pages": _group_pages(scored),
    }


def _opportunity_row(
    row: dict,
    pages: list[PageInfo],
    lookup: dict[str, int],
    *,
    position: float,
    impressions: float,
    current_ctr: float,
    target_ctr: float,
    estimated_gain: float,
) -> dict:
    url = str(row.get("matched_url") or row.get("url") or "")
    page_index = _match_page(url, lookup)
    page = pages[page_index] if page_index is not None else None
    matched_url = page.url if page else url
    title = _page_attr(page, "title") or row.get("page_title") or row.get("title") or matched_url
    word_count = _page_attr(page, "word_count")
    return {
        "query": row.get("query") or row.get("keyword") or "",
        "url": matched_url,
        "source_url": row.get("url") or "",
        "page_title": title,
        "cluster": row.get("cluster"),
        "cluster_label": row.get("cluster_label", ""),
        "word_count": word_count,
        "clicks": round(_to_float(row.get("clicks")), 2),
        "impressions": round(impressions, 2),
        "ctr": row.get("ctr"),
        "position": round(position, 2),
        "expected_ctr_current": round(current_ctr, 4),
        "expected_ctr_target": round(target_ctr, 4),
        "estimated_clicks_gain": round(estimated_gain, 2),
        "source": row.get("source") or row.get("provider") or "search",
    }


def _group_pages(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"queries": [], "total_estimated_gain": 0.0})
    for row in rows:
        url = row.get("url") or ""
        group = grouped[url]
        group["url"] = url
        group["title"] = row.get("page_title") or url
        group["cluster"] = row.get("cluster")
        group["word_count"] = row.get("word_count")
        group["total_estimated_gain"] += _to_float(row.get("estimated_clicks_gain"))
        group["queries"].append(row)
    pages = list(grouped.values())
    for page in pages:
        page["total_estimated_gain"] = round(_to_float(page.get("total_estimated_gain")), 2)
        page["queries"].sort(key=lambda r: _to_float(r.get("estimated_clicks_gain")), reverse=True)
        page["queries"] = page["queries"][:MAX_QUERIES_PER_PAGE]
    pages.sort(key=lambda p: _to_float(p.get("total_estimated_gain")), reverse=True)
    return pages[:MAX_PAGE_GROUPS]


def _model(search_payload: dict, target_position: float) -> dict:
    return {
        "target_position": target_position,
        "curve": "site-audit-v1",
        "period_days": _period_days(search_payload),
    }


def _window_dates(search_payload: dict) -> tuple[str, str]:
    summary = search_payload.get("summary") or {}
    params = (search_payload.get("meta") or {}).get("params") or {}
    return (
        str(summary.get("start_date") or params.get("start_date") or ""),
        str(summary.get("end_date") or params.get("end_date") or ""),
    )


def _period_days(search_payload: dict) -> int | None:
    start, end = _window_dates(search_payload)
    if not start or not end:
        return None
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        return None
    return max(1, (end_date - start_date).days + 1)


def _is_branded(row: dict) -> bool:
    intents = {str(intent).lower() for intent in (row.get("intents") or [])}
    return "branded" in intents or bool(row.get("is_branded"))


def _page_attr(page, name: str):
    if page is None:
        return None
    if isinstance(page, dict):
        return page.get(name)
    return getattr(page, name, None)


def _to_float(value) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
