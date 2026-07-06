"""CTR-vs-position anomaly detection for title/meta rewrite candidates."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Any

from .ahrefs import _match_page, _page_lookup, _url_keys
from .analyzer import PageInfo
from .ctr_curve import expected_ctr


MAX_PAGE_GROUPS = 100
MAX_QUERIES_PER_PAGE = 20
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "best",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}


def build_ctr_anomalies(
    search_payload: dict | None,
    pages: list[PageInfo] | None,
    metadata_payload: dict | None = None,
    *,
    max_position: float = 10.0,
    min_impressions: int = 100,
    underperformance_ratio: float = 0.6,
    max_rows: int = 100,
) -> dict:
    """Build CTR underperformance rows from GSC query+page data."""
    payload = search_payload or {}
    query_pages = list(payload.get("query_pages") or [])
    measured_rows = [row for row in query_pages if _has_measured_gsc_ctr(row)]
    model = _model(payload, max_position, min_impressions, underperformance_ratio)
    if not measured_rows:
        return {
            "available": False,
            "reason": "No GSC-sourced query+page rows with measured CTR were available.",
            "rows": [],
            "pages": [],
            "recommendations": [],
            "summary": {"status": "unavailable"},
            "model": model,
        }

    page_list = list(pages or [])
    lookup = _page_lookup(page_list)
    metadata_lookup = _metadata_lookup(metadata_payload or {})
    serp_features = _serp_features_by_query(payload)
    scored: list[dict] = []
    for row in measured_rows:
        position = _to_float(row.get("position"))
        impressions = _to_float(row.get("impressions"))
        if position <= 0 or position > max_position or impressions < min_impressions:
            continue
        actual_ctr = _to_float(row.get("ctr"))
        modeled_ctr = expected_ctr(position)
        if actual_ctr >= underperformance_ratio * modeled_ctr:
            continue
        missed_clicks = max(0.0, (modeled_ctr - actual_ctr) * impressions)
        if missed_clicks <= 0:
            continue
        scored.append(_anomaly_row(
            row,
            page_list,
            lookup,
            metadata_lookup,
            serp_features,
            position=position,
            impressions=impressions,
            actual_ctr=actual_ctr,
            expected=modeled_ctr,
            missed_clicks=missed_clicks,
        ))

    scored.sort(key=lambda r: _to_float(r.get("missed_clicks")), reverse=True)
    emitted = scored[:max_rows]
    start_date, end_date = _window_dates(payload)
    groups = _group_pages(scored)
    return {
        "available": True,
        "summary": {
            "status": "ok",
            "anomalies": len(scored),
            "shown": len(emitted),
            "total_missed_clicks": round(sum(_to_float(r.get("missed_clicks")) for r in scored), 2),
            "start_date": start_date,
            "end_date": end_date,
            "max_position": max_position,
            "min_impressions": min_impressions,
            "underperformance_ratio": underperformance_ratio,
        },
        "model": model,
        "rows": emitted,
        "pages": groups,
        "recommendations": _recommendation_inputs(groups, model),
    }


def _anomaly_row(
    row: dict,
    pages: list[PageInfo],
    lookup: dict[str, int],
    metadata_lookup: dict[str, dict],
    serp_features: dict[str, list],
    *,
    position: float,
    impressions: float,
    actual_ctr: float,
    expected: float,
    missed_clicks: float,
) -> dict:
    raw_url = str(row.get("matched_url") or row.get("url") or "")
    page_index = _match_page(raw_url, lookup)
    page = pages[page_index] if page_index is not None else None
    url = _page_attr(page, "url") or raw_url
    metadata = _match_metadata(url, metadata_lookup) or {}
    query = str(row.get("query") or row.get("keyword") or "")
    title = _page_attr(page, "title") or metadata.get("title") or row.get("page_title") or row.get("title") or ""
    description = _page_attr(page, "description") or metadata.get("description") or ""
    features = list(row.get("serp_features") or serp_features.get(_query_key(query)) or [])
    query_terms_in_title = _query_terms_in_title(query, title)
    title_length = len(title or "")
    cause = _probable_cause(
        title=title,
        title_length=title_length,
        query_terms_in_title=query_terms_in_title,
        metadata=metadata,
        serp_features=features,
    )
    return {
        "query": query,
        "url": url,
        "source_url": row.get("url") or "",
        "page_title": title,
        "current_title": title,
        "meta_description": description,
        "title_length": title_length,
        "query_terms_in_title": query_terms_in_title,
        "probable_cause": cause,
        "position": round(position, 2),
        "impressions": round(impressions, 2),
        "clicks": round(_to_float(row.get("clicks")), 2),
        "actual_ctr": round(actual_ctr, 4),
        "expected_ctr": round(expected, 4),
        "missed_clicks": round(missed_clicks, 2),
        "source": row.get("source") or row.get("provider") or "gsc",
        "serp_features": features,
        "cluster": row.get("cluster"),
        "cluster_label": row.get("cluster_label", ""),
    }


def _group_pages(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"worst_queries": [], "total_missed_clicks": 0.0})
    for row in rows:
        url = row.get("url") or ""
        group = grouped[url]
        group["url"] = url
        group["title"] = row.get("page_title") or url
        group["cluster"] = row.get("cluster")
        group["cluster_label"] = row.get("cluster_label", "")
        group["total_missed_clicks"] += _to_float(row.get("missed_clicks"))
        group["worst_queries"].append(row)
    pages = list(grouped.values())
    for page in pages:
        page["total_missed_clicks"] = round(_to_float(page.get("total_missed_clicks")), 2)
        page["worst_queries"].sort(key=lambda r: _to_float(r.get("missed_clicks")), reverse=True)
        page["worst_queries"] = page["worst_queries"][:MAX_QUERIES_PER_PAGE]
    pages.sort(key=lambda p: _to_float(p.get("total_missed_clicks")), reverse=True)
    return pages[:MAX_PAGE_GROUPS]


def _recommendation_inputs(pages: list[dict], model: dict) -> list[dict]:
    period = _period_label(model.get("period_days"))
    out: list[dict] = []
    for page in pages:
        queries = page.get("worst_queries") or []
        if not queries:
            continue
        top = queries[0]
        missed = _to_float(page.get("total_missed_clicks"))
        query = str(top.get("query") or "")
        position = _to_float(top.get("position"))
        actual = _to_float(top.get("actual_ctr"))
        expected = _to_float(top.get("expected_ctr"))
        cause = str(top.get("probable_cause") or "unclear")
        url = str(page.get("url") or top.get("url") or "")
        out.append({
            "title": f'Title underperforms position #{position:.0f} for "{query}"',
            "action": (
                f"Rewrite title/meta of {url}: CTR is {actual:.1%} vs {expected:.1%} expected "
                f"at position {position:.0f} — ~{missed:.0f} missed clicks/{period} "
                f"across {len(queries)} underperforming quer{'y' if len(queries) == 1 else 'ies'}. "
                f'Probable cause: {cause}. Include "{query}" phrasing in the title (<=65 chars).'
            ),
            "url": url,
            "query": query,
            "position": round(position, 2),
            "actual_ctr": round(actual, 4),
            "expected_ctr": round(expected, 4),
            "missed_clicks": round(missed, 2),
            "estimated_clicks_gain": round(missed, 2),
            "probable_cause": cause,
            "period": period,
            "current_title": top.get("current_title") or top.get("page_title") or "",
            "worst_queries": queries[:5],
        })
    return out


def _probable_cause(
    *,
    title: str,
    title_length: int,
    query_terms_in_title: bool,
    metadata: dict,
    serp_features: list,
) -> str:
    if not query_terms_in_title:
        return "title_missing_query_terms"
    if title_length > 65:
        return "title_too_long_truncated"
    issues = set(metadata.get("issues") or [])
    if issues.intersection({"missing_description", "duplicate_description"}):
        return "description_missing_or_duplicate"
    if serp_features:
        return "serp_feature_competition"
    return "unclear"


def _metadata_lookup(metadata_payload: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in metadata_payload.get("per_page") or []:
        for key in _url_keys(row.get("url") or ""):
            lookup[key] = row
    return lookup


def _match_metadata(url: str, lookup: dict[str, dict]) -> dict | None:
    for key in _url_keys(url):
        if key in lookup:
            return lookup[key]
    return None


def _serp_features_by_query(search_payload: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for row in search_payload.get("organic_keywords") or []:
        features = list(row.get("serp_features") or [])
        if not features:
            continue
        query = row.get("query") or row.get("keyword") or ""
        if query:
            out[_query_key(query)] = features
    return out


def _query_terms_in_title(query: str, title: str) -> bool:
    terms = [term for term in _tokens(query) if term not in _QUERY_STOPWORDS]
    if not terms:
        terms = _tokens(query)
    title_terms = set(_tokens(title))
    return bool(terms) and all(term in title_terms for term in terms)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _has_measured_gsc_ctr(row: dict) -> bool:
    source = str(row.get("source") or row.get("provider") or "").lower()
    return source == "gsc" and row.get("ctr") not in (None, "")


def _model(search_payload: dict, max_position: float, min_impressions: int, underperformance_ratio: float) -> dict:
    return {
        "curve": "site-audit-v1",
        "max_position": max_position,
        "min_impressions": min_impressions,
        "underperformance_ratio": underperformance_ratio,
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


def _period_label(period_days: object) -> str:
    days = _to_float(period_days)
    return f"{days:.0f} days" if days else "period"


def _query_key(query: object) -> str:
    return " ".join(str(query or "").strip().lower().split())


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
