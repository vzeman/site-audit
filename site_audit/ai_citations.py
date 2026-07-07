"""Google AI Overview citation analysis from DataForSEO ranked keywords."""

from __future__ import annotations

from typing import Iterable

from .ahrefs import _match_page, _page_lookup, _url_keys
from .analyzer import PageInfo, section_for_url

COVERAGE_OWN_DOMAIN_ITEMS_ONLY = "own_domain_items_only"
UNAVAILABLE_NO_DATAFORSEO = "DataForSEO payload is unavailable."
UNAVAILABLE_NO_CITATIONS = "DataForSEO payload has no AI Overview citation rows."
COVERAGE_NOTE = (
    "DataForSEO ranked_keywords is scoped to the audited domain's own SERP items. "
    "Opportunities are shown only when those returned rows explicitly include an AI Overview item type."
)


def build_ai_citations(
    dataforseo_payload: dict | None,
    search_payload: dict | None,
    pages: Iterable[PageInfo],
    freshness_payload: dict | None = None,
) -> dict:
    """Build a report payload for Google AI Overview citations."""
    page_list = list(pages)
    citations = list((dataforseo_payload or {}).get("ai_overview_citations") or [])
    coverage = _coverage_payload(dataforseo_payload)
    if not isinstance(dataforseo_payload, dict) or "ai_overview_citations" not in dataforseo_payload:
        # A payload without the citation key means DataForSEO never ran
        # (provider disabled) — distinct from a run that found no citations.
        return _unavailable(UNAVAILABLE_NO_DATAFORSEO, coverage)
    if not citations:
        opportunities = _opportunities(page_list, dataforseo_payload, search_payload, citations)
        if not opportunities:
            return _unavailable(UNAVAILABLE_NO_CITATIONS, coverage)
        return {
            "available": True,
            "reason": "",
            "coverage": coverage["coverage"],
            "coverage_note": coverage["note"],
            "summary": {
                "cited_pages": 0,
                "citing_queries": 0,
                "citing_query_volume": 0,
                "top_traffic_pages": len(((search_payload or {}).get("top_pages") or [])[:20]),
                "top_traffic_pages_cited": 0,
                "top_traffic_pages_cited_share": 0.0,
            },
            "cited_pages": [],
            "at_risk": [],
            "opportunities": opportunities,
        }

    lookup = _page_lookup(page_list)
    search_lookup = _search_context_lookup(search_payload, dataforseo_payload)
    cited_pages = _cited_pages(citations, page_list, lookup, search_lookup)
    at_risk = _at_risk_pages(cited_pages, freshness_payload)
    opportunities = _opportunities(page_list, dataforseo_payload, search_payload, citations)
    summary = _summary(cited_pages, citations, search_payload)
    return {
        "available": True,
        "reason": "",
        "coverage": coverage["coverage"],
        "coverage_note": coverage["note"],
        "summary": summary,
        "cited_pages": cited_pages,
        "at_risk": at_risk,
        "opportunities": opportunities,
    }


def _unavailable(reason: str, coverage: dict) -> dict:
    return {
        "available": False,
        "reason": reason,
        "coverage": coverage["coverage"],
        "coverage_note": coverage["note"],
        "summary": {
            "cited_pages": 0,
            "citing_queries": 0,
            "citing_query_volume": 0,
            "top_traffic_pages": 0,
            "top_traffic_pages_cited": 0,
            "top_traffic_pages_cited_share": 0.0,
        },
        "cited_pages": [],
        "at_risk": [],
        "opportunities": [],
    }


def _coverage_payload(dataforseo_payload: dict | None) -> dict:
    coverage = (dataforseo_payload or {}).get("ai_overview_coverage") or {}
    return {
        "coverage": str(coverage.get("coverage") or COVERAGE_OWN_DOMAIN_ITEMS_ONLY),
        "note": str(coverage.get("note") or COVERAGE_NOTE),
    }


def _safe_int(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _url_key_set(url: str) -> set[str]:
    return set(_url_keys(str(url or "")))


def _store_context(out: dict[str, dict], url: str, context: dict) -> None:
    if not url:
        return
    for key in _url_keys(url):
        current = out.get(key)
        if current is None or _safe_float(context.get("traffic")) > _safe_float(current.get("traffic")):
            out[key] = dict(context)


def _search_context_lookup(search_payload: dict | None, dataforseo_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for payload in (dataforseo_payload or {}, search_payload or {}):
        for row in payload.get("top_pages") or []:
            url = row.get("matched_url") or row.get("url") or ""
            _store_context(out, url, {
                "title": row.get("title") or row.get("page_title") or "",
                "cluster": row.get("cluster_label") or row.get("cluster") or row.get("section") or "",
                "section": row.get("section") or section_for_url(str(url)),
                "traffic": row.get("traffic", 0),
            })
        for row in payload.get("organic_keywords") or []:
            url = row.get("matched_url") or row.get("url") or ""
            _store_context(out, url, {
                "title": row.get("page_title") or row.get("title") or "",
                "cluster": row.get("cluster_label") or row.get("cluster") or row.get("section") or "",
                "section": row.get("section") or section_for_url(str(url)),
                "traffic": row.get("traffic", 0),
            })
    return out


def _page_context(url: str, pages: list[PageInfo], lookup: dict[str, int], search_lookup: dict[str, dict]) -> dict:
    page_index = _match_page(url, lookup)
    page = pages[page_index] if page_index is not None else None
    search = next((search_lookup.get(key) for key in _url_keys(url) if search_lookup.get(key)), {})
    canonical_url = page.url if page else url
    return {
        "url": canonical_url,
        "source_url": url,
        "matched": page is not None,
        "title": page.title if page else str(search.get("title") or ""),
        "section": page.section if page else str(search.get("section") or section_for_url(url)),
        "cluster": str(search.get("cluster") or ""),
        "traffic": _safe_int(search.get("traffic")),
    }


def _cited_pages(
    citations: list[dict],
    pages: list[PageInfo],
    lookup: dict[str, int],
    search_lookup: dict[str, dict],
) -> list[dict]:
    by_url: dict[str, dict] = {}
    for citation in citations:
        url = str(citation.get("url") or "")
        keyword = str(citation.get("keyword") or "")
        if not url or not keyword:
            continue
        context = _page_context(url, pages, lookup, search_lookup)
        row = by_url.setdefault(context["url"], {
            **context,
            "queries": [],
            "query_count": 0,
            "total_volume": 0,
        })
        query = {
            "keyword": keyword,
            "search_volume": _safe_int(citation.get("search_volume")),
            "keyword_difficulty": _safe_int(citation.get("keyword_difficulty")),
            "rank_absolute": _safe_int(citation.get("rank_absolute")),
            "serp_title": str(citation.get("serp_title") or ""),
        }
        if not any(existing.get("keyword") == keyword for existing in row["queries"]):
            row["queries"].append(query)
            row["query_count"] += 1
            row["total_volume"] += query["search_volume"]
    rows = list(by_url.values())
    for row in rows:
        row["queries"].sort(key=lambda q: (_safe_int(q.get("search_volume")), q.get("keyword") or ""), reverse=True)
    rows.sort(key=lambda r: (_safe_int(r.get("total_volume")), _safe_int(r.get("query_count"))), reverse=True)
    return rows


def _freshness_lookup(freshness_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (freshness_payload or {}).get("per_page") or []:
        url = str(row.get("url") or "")
        if row.get("bucket") not in {"stale", "very_stale"}:
            continue
        for key in _url_keys(url):
            out[key] = row
    return out


def _at_risk_pages(cited_pages: list[dict], freshness_payload: dict | None) -> list[dict]:
    freshness = _freshness_lookup(freshness_payload)
    out: list[dict] = []
    for page in cited_pages:
        stale = next((freshness.get(key) for key in _url_keys(str(page.get("url") or "")) if freshness.get(key)), None)
        if not stale:
            continue
        top_query = (page.get("queries") or [{}])[0]
        out.append({
            **page,
            "bucket": stale.get("bucket") or "",
            "age_days": stale.get("age_days"),
            "date": stale.get("date") or "",
            "top_keyword": top_query.get("keyword") or "",
            "top_keyword_volume": _safe_int(top_query.get("search_volume")),
        })
    out.sort(key=lambda r: (_safe_int(r.get("top_keyword_volume")), r.get("url") or ""), reverse=True)
    return out


def _cited_keywords(citations: list[dict]) -> set[str]:
    # Keyword-level: a keyword whose AI Overview already cites ANY page of
    # the site is not an opportunity, even for another ranking page.
    return {
        keyword
        for citation in citations
        if (keyword := str(citation.get("keyword") or "").strip().lower())
    }


def _opportunity_rows(dataforseo_payload: dict | None, search_payload: dict | None) -> list[dict]:
    rows: list[dict] = []
    for payload in (dataforseo_payload or {}, search_payload or {}):
        for row in payload.get("query_pages") or []:
            rows.append(row)
        for row in payload.get("organic_keywords") or []:
            rows.append(row)
    return rows


def _row_keyword(row: dict) -> str:
    return str(row.get("query") or row.get("keyword") or "").strip()


def _row_url(row: dict) -> str:
    return str(row.get("matched_url") or row.get("url") or "").strip()


def _row_position(row: dict) -> float:
    return _safe_float(row.get("position") or row.get("rank") or row.get("rank_group"))


def _opportunities(
    pages: list[PageInfo],
    dataforseo_payload: dict | None,
    search_payload: dict | None,
    citations: list[dict],
) -> list[dict]:
    lookup = _page_lookup(pages)
    cited = _cited_keywords(citations)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in _opportunity_rows(dataforseo_payload, search_payload):
        if row.get("has_ai_overview") is not True:
            continue
        keyword = _row_keyword(row)
        url = _row_url(row)
        position = _row_position(row)
        if not keyword or not url or position <= 0 or position > 20:
            continue
        if keyword.lower() in cited:
            continue
        page_index = _match_page(url, lookup)
        page = pages[page_index] if page_index is not None else None
        canonical_url = page.url if page else url
        key = (keyword.lower(), canonical_url)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "keyword": keyword,
            "url": canonical_url,
            "title": page.title if page else str(row.get("page_title") or row.get("title") or ""),
            "position": position,
            "search_volume": _safe_int(row.get("volume") or row.get("impressions") or row.get("search_volume")),
            "traffic": _safe_int(row.get("traffic") or row.get("clicks")),
            "cluster": str(row.get("cluster_label") or row.get("cluster") or ""),
        })
    out.sort(key=lambda r: (_safe_int(r.get("search_volume")), -_safe_float(r.get("position"))), reverse=True)
    return out[:100]


def _summary(cited_pages: list[dict], citations: list[dict], search_payload: dict | None) -> dict:
    volume_by_keyword: dict[str, int] = {}
    for citation in citations:
        keyword = str(citation.get("keyword") or "").strip()
        if not keyword:
            continue
        volume_by_keyword[keyword] = max(volume_by_keyword.get(keyword, 0), _safe_int(citation.get("search_volume")))
    top_pages = sorted(
        list((search_payload or {}).get("top_pages") or []),
        key=lambda r: _safe_int(r.get("traffic")),
        reverse=True,
    )[:20]
    cited_keys = set()
    for row in cited_pages:
        cited_keys.update(_url_key_set(str(row.get("url") or "")))
        cited_keys.update(_url_key_set(str(row.get("source_url") or "")))
    cited_top = 0
    for row in top_pages:
        url = str(row.get("matched_url") or row.get("url") or "")
        if cited_keys & _url_key_set(url):
            cited_top += 1
    return {
        "cited_pages": len(cited_pages),
        "citing_queries": len(volume_by_keyword),
        "citing_query_volume": sum(volume_by_keyword.values()),
        "top_traffic_pages": len(top_pages),
        "top_traffic_pages_cited": cited_top,
        "top_traffic_pages_cited_share": round(cited_top / len(top_pages), 4) if top_pages else 0.0,
    }
