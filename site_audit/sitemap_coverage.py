"""Sitemap coverage matrix for crawled and analyzed URLs."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


ACTION_BY_STATUS = {
    "sitemap_indexable": "No sitemap action needed for this URL.",
    "sitemap_not_fetched": "Increase crawl limits or inspect robots/rules if this sitemap URL should be audited.",
    "sitemap_non_indexable": "Remove non-indexable URLs from XML sitemaps or fix indexability when the page should rank.",
    "crawled_not_in_sitemap": "Add this indexable URL to the XML sitemap if it is an SEO landing page.",
}

SITEMAP_REDIRECT_ISSUE = "3xx_redirect_in_sitemap"
SITEMAP_4XX_ISSUE = "4xx_page_in_sitemap"
SITEMAP_5XX_ISSUE = "5xx_page_in_sitemap"
SITEMAP_NOINDEX_ISSUE = "noindex_page_in_sitemap"
SITEMAP_NON_CANONICAL_ISSUE = "non_canonical_page_in_sitemap"


def analyze(
    sitemap_entries: Iterable[dict],
    fetched: Iterable,
    extraction_rows: list[dict],
    indexability: dict | None = None,
) -> dict:
    sitemap_by_url = {row.get("url", ""): row for row in sitemap_entries if row.get("url")}
    fetched_by_url = {getattr(row, "url", ""): row for row in fetched}
    fetched_by_requested_url = {
        getattr(row, "requested_url", ""): row
        for row in fetched
        if getattr(row, "requested_url", "")
    }
    extraction_by_url = {row.get("url", ""): row for row in extraction_rows if row.get("url")}
    indexability_by_url = {
        row.get("url", ""): row
        for row in (indexability or {}).get("per_page", [])
        if row.get("url")
    }
    urls = sorted(set(sitemap_by_url) | set(fetched_by_url))
    rows: list[dict] = []
    issues: list[dict] = []
    status_counts: Counter[str] = Counter()

    for url in urls:
        sitemap_row = sitemap_by_url.get(url, {})
        fetched_row = fetched_by_url.get(url) or fetched_by_requested_url.get(url)
        extraction_row = extraction_by_url.get(url, {})
        indexability_row = indexability_by_url.get(url, {})
        in_sitemap = url in sitemap_by_url
        crawled = fetched_row is not None
        indexability_status = indexability_row.get("indexability_status") or (
            "indexable" if extraction_row.get("status") == "analyzed" else extraction_row.get("reason", "")
        )
        coverage_status = _coverage_status(in_sitemap, crawled, indexability_status)
        redirect_status_codes = [_safe_int(code) for code in getattr(fetched_row, "redirect_status_codes", []) or []]
        redirect_target_url = getattr(fetched_row, "redirect_target_url", "") or ""
        sitemap_issue_types = []
        if in_sitemap and any(300 <= code < 400 for code in redirect_status_codes):
            sitemap_issue_types.append(SITEMAP_REDIRECT_ISSUE)
        http_status = extraction_row.get("http_status") or getattr(fetched_row, "status", "")
        if in_sitemap and 400 <= _safe_int(http_status) < 500:
            sitemap_issue_types.append(SITEMAP_4XX_ISSUE)
        if in_sitemap and 500 <= _safe_int(http_status) < 600:
            sitemap_issue_types.append(SITEMAP_5XX_ISSUE)
        if in_sitemap and indexability_status == "noindex":
            sitemap_issue_types.append(SITEMAP_NOINDEX_ISSUE)
        canonical_url = extraction_row.get("canonical_url", "")
        if in_sitemap and _canonical_differs(url, canonical_url):
            sitemap_issue_types.append(SITEMAP_NON_CANONICAL_ISSUE)
        status_counts[coverage_status] += 1
        row = {
            "url": url,
            "title": extraction_row.get("title") or indexability_row.get("title", ""),
            "in_sitemap": in_sitemap,
            "source_sitemaps": sitemap_row.get("source_sitemaps", []),
            "lastmod": sitemap_row.get("lastmod", ""),
            "crawled": crawled,
            "http_status": http_status,
            "canonical_url": canonical_url,
            "redirect_target_url": redirect_target_url,
            "redirect_status_codes": redirect_status_codes,
            "redirect_hop_count": _safe_int(getattr(fetched_row, "redirect_hop_count", 0)),
            "extraction_status": extraction_row.get("status", ""),
            "indexability_status": indexability_status,
            "coverage_status": coverage_status,
            "sitemap_issue_types": sitemap_issue_types,
            "indexability_issues": indexability_row.get("issues", []),
            "recommended_action": ACTION_BY_STATUS.get(coverage_status, "Review sitemap coverage for this URL."),
        }
        rows.append(row)
        for issue_type in sitemap_issue_types:
            issues.append({
                "url": url,
                "title": row["title"],
                "issue": issue_type,
                "in_sitemap": in_sitemap,
                "crawled": crawled,
                "http_status": row["http_status"],
                "redirect_target_url": row["redirect_target_url"],
                "redirect_status_codes": row["redirect_status_codes"],
                "source_sitemaps": row["source_sitemaps"],
                "recommended_action": _sitemap_issue_action(issue_type),
            })
        if coverage_status != "sitemap_indexable":
            issues.append({
                "url": url,
                "title": row["title"],
                "issue": coverage_status,
                "in_sitemap": in_sitemap,
                "crawled": crawled,
                "http_status": row["http_status"],
                "indexability_status": indexability_status,
                "source_sitemaps": row["source_sitemaps"],
                "recommended_action": row["recommended_action"],
            })

    sitemap_total = len(sitemap_by_url)
    fetched_sitemap = sum(1 for row in rows if row["in_sitemap"] and row["crawled"])
    indexable_sitemap = status_counts.get("sitemap_indexable", 0)
    return {
        "summary": {
            "status": "ok" if rows else "no_urls",
            "total_sitemap_urls": sitemap_total,
            "total_crawled_urls": len(fetched_by_url),
            "fetched_sitemap_urls": fetched_sitemap,
            "sitemap_not_fetched": status_counts.get("sitemap_not_fetched", 0),
            "sitemap_non_indexable": status_counts.get("sitemap_non_indexable", 0),
            "sitemap_indexable": indexable_sitemap,
            "3xx_redirect_in_sitemap": sum(
                1 for row in rows if SITEMAP_REDIRECT_ISSUE in (row.get("sitemap_issue_types") or [])
            ),
            "4xx_page_in_sitemap": sum(
                1 for row in rows if SITEMAP_4XX_ISSUE in (row.get("sitemap_issue_types") or [])
            ),
            "5xx_page_in_sitemap": sum(
                1 for row in rows if SITEMAP_5XX_ISSUE in (row.get("sitemap_issue_types") or [])
            ),
            "noindex_page_in_sitemap": sum(
                1 for row in rows if SITEMAP_NOINDEX_ISSUE in (row.get("sitemap_issue_types") or [])
            ),
            "non_canonical_page_in_sitemap": sum(
                1 for row in rows if SITEMAP_NON_CANONICAL_ISSUE in (row.get("sitemap_issue_types") or [])
            ),
            "crawled_not_in_sitemap": status_counts.get("crawled_not_in_sitemap", 0),
            "sitemap_fetch_coverage_share": fetched_sitemap / sitemap_total if sitemap_total else 0.0,
            "sitemap_indexable_share": indexable_sitemap / sitemap_total if sitemap_total else 0.0,
        },
        "status_counts": dict(status_counts),
        "rows": rows,
        "issues": issues,
        "interpretation": {
            "why_it_matters": "XML sitemaps should point search engines at canonical, indexable URLs. Gaps show wasted crawl hints or important pages missing from sitemap coverage.",
            "how_to_use": "First remove or fix non-indexable sitemap URLs, then add valuable crawled indexable pages that are missing from sitemaps.",
        },
    }


def _coverage_status(in_sitemap: bool, crawled: bool, indexability_status: str) -> str:
    if in_sitemap and not crawled:
        return "sitemap_not_fetched"
    if in_sitemap and indexability_status and indexability_status != "indexable":
        return "sitemap_non_indexable"
    if in_sitemap:
        return "sitemap_indexable"
    return "crawled_not_in_sitemap"


def _sitemap_issue_action(issue_type: str) -> str:
    if issue_type == SITEMAP_REDIRECT_ISSUE:
        return "Update the sitemap URL to the final destination instead of a redirecting URL."
    if issue_type == SITEMAP_4XX_ISSUE:
        return "Restore the URL or remove the 4XX page from XML sitemaps."
    if issue_type == SITEMAP_5XX_ISSUE:
        return "Fix the server error before keeping this URL in XML sitemaps."
    if issue_type == SITEMAP_NOINDEX_ISSUE:
        return "Remove noindex URLs from XML sitemaps or make the page indexable if it should rank."
    if issue_type == SITEMAP_NON_CANONICAL_ISSUE:
        return "Update XML sitemaps to list the canonical URL instead of a non-canonical URL."
    return "Review and fix this sitemap issue."


def _canonical_differs(url: str, canonical_url: str) -> bool:
    if not canonical_url:
        return False
    return url.rstrip("/") != canonical_url.rstrip("/")


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0
