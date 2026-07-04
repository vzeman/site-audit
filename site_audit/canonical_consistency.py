"""Canonical URL consistency audit."""

from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse


ACTION_BY_ISSUE = {
    "missing_canonical": "Add a self-referencing canonical or the intended canonical target.",
    "canonical_external_host": "Confirm the external canonical is intentional; otherwise point it to the canonical URL on this domain.",
    "canonical_non_self": "Use a self-referencing canonical on unique SEO pages, or confirm this is a deliberate duplicate consolidation.",
    "canonical_points_to_4xx": "Point canonical tags only at live 2xx URLs or restore the canonical target.",
    "canonical_points_to_5xx": "Fix the canonical target server error or point the canonical tag at a stable live URL.",
    "canonical_points_to_redirect": "Point canonical tags directly at the final destination URL instead of a redirecting URL.",
    "non_canonical_page_specified_as_canonical_one": "Point the canonical tag at the final self-canonical URL instead of a non-canonical target.",
    "canonical_target_not_crawled": "Make sure the canonical target is crawlable and included in the audit scope.",
    "canonical_target_non_indexable": "Point canonical tags only at indexable URLs or fix the target page indexability.",
    "canonical_target_shared": "Review whether multiple pages should consolidate to the same canonical target.",
}


def analyze(extraction_rows: list[dict], indexability: dict | None = None) -> dict:
    page_rows = [row for row in extraction_rows if row.get("url")]
    urls = {row.get("url", "") for row in page_rows}
    normalized_urls = {_normalize_url(url) for url in urls}
    redirect_by_normalized_requested_url = {
        _normalize_url(row.get("requested_url", "")): row.get("redirect_target_url", "")
        for row in page_rows
        if row.get("requested_url") and row.get("redirect_target_url")
    }
    indexability_by_url = {
        row.get("url", ""): row
        for row in (indexability or {}).get("per_page", [])
        if row.get("url")
    }
    indexability_by_normalized_url = {
        _normalize_url(url): row
        for url, row in indexability_by_url.items()
    }
    canonical_counts = Counter(
        _normalize_url(row.get("canonical_url", ""))
        for row in page_rows
        if row.get("canonical_url")
    )
    page_by_normalized_url = {
        _normalize_url(row.get("url", "")): row
        for row in page_rows
        if row.get("url")
    }
    rows: list[dict] = []
    issues: list[dict] = []
    issue_counts: Counter[str] = Counter()

    for page in page_rows:
        url = page.get("url", "")
        canonical_url = page.get("canonical_url", "")
        normalized_canonical = _normalize_url(canonical_url)
        target = indexability_by_url.get(canonical_url) or indexability_by_normalized_url.get(normalized_canonical)
        target_page = page_by_normalized_url.get(normalized_canonical, {})
        canonical_redirect_target = redirect_by_normalized_requested_url.get(normalized_canonical, "")
        issue_keys = _issues_for_page(
            url,
            canonical_url,
            normalized_urls,
            indexability_by_url,
            indexability_by_normalized_url,
            canonical_counts,
            redirect_by_normalized_requested_url,
            page_by_normalized_url,
        )
        issue_counts.update(issue_keys)
        row = {
            "url": url,
            "title": page.get("title", ""),
            "canonical_url": canonical_url,
            "canonical_target_normalized": normalized_canonical,
            "canonical_target_http_status": (target or {}).get("http_status", ""),
            "canonical_target_indexability_status": (target or {}).get("indexability_status", ""),
            "canonical_target_canonical_url": target_page.get("canonical_url", ""),
            "canonical_redirect_target_url": canonical_redirect_target,
            "http_status": page.get("http_status", ""),
            "extraction_status": page.get("status", ""),
            "indexability_status": (indexability_by_url.get(url) or {}).get("indexability_status", ""),
            "canonical_status": "needs_review" if issue_keys else "ok",
            "canonical_issue_count": len(issue_keys),
            "issues": issue_keys,
            "recommended_action": _recommended_action(issue_keys),
        }
        rows.append(row)
        for issue in issue_keys:
            issues.append({
                "url": url,
                "title": row["title"],
                "issue": issue,
                "canonical_url": canonical_url,
                "canonical_target_http_status": row["canonical_target_http_status"],
                "canonical_target_indexability_status": row["canonical_target_indexability_status"],
                "canonical_target_canonical_url": row["canonical_target_canonical_url"],
                "canonical_redirect_target_url": row["canonical_redirect_target_url"],
                "http_status": row["http_status"],
                "indexability_status": row["indexability_status"],
                "recommended_action": ACTION_BY_ISSUE.get(issue, "Review canonical configuration."),
            })

    rows.sort(key=lambda row: (row["canonical_issue_count"], row["url"]), reverse=True)
    total = len(rows)
    pages_with_issues = sum(1 for row in rows if row["issues"])
    return {
        "summary": {
            "status": "ok" if rows else "no_pages",
            "total_pages": total,
            "pages_with_canonical_issues": pages_with_issues,
            "canonical_issue_share": pages_with_issues / total if total else 0.0,
            "missing_canonical": issue_counts.get("missing_canonical", 0),
            "canonical_external_host": issue_counts.get("canonical_external_host", 0),
            "canonical_non_self": issue_counts.get("canonical_non_self", 0),
            "canonical_points_to_4xx": issue_counts.get("canonical_points_to_4xx", 0),
            "canonical_points_to_5xx": issue_counts.get("canonical_points_to_5xx", 0),
            "canonical_points_to_redirect": issue_counts.get("canonical_points_to_redirect", 0),
            "non_canonical_page_specified_as_canonical_one": issue_counts.get("non_canonical_page_specified_as_canonical_one", 0),
            "canonical_target_not_crawled": issue_counts.get("canonical_target_not_crawled", 0),
            "canonical_target_non_indexable": issue_counts.get("canonical_target_non_indexable", 0),
            "canonical_target_shared": issue_counts.get("canonical_target_shared", 0),
        },
        "issue_counts": dict(issue_counts),
        "rows": rows,
        "issues": issues,
        "interpretation": {
            "why_it_matters": "Canonical tags tell search engines which URL should consolidate duplicate signals and rank. Incorrect canonicals can remove good pages from competition or point signals at weak targets.",
            "how_to_use": "Fix missing and external canonicals first, then review non-self canonicals and shared canonical targets on important SEO pages.",
        },
    }


def _issues_for_page(
    url: str,
    canonical_url: str,
    normalized_crawled_urls: set[str],
    indexability_by_url: dict[str, dict],
    indexability_by_normalized_url: dict[str, dict],
    canonical_counts: Counter[str],
    redirect_by_normalized_requested_url: dict[str, str],
    page_by_normalized_url: dict[str, dict],
) -> list[str]:
    if not canonical_url:
        return ["missing_canonical"]
    issues: list[str] = []
    if not _same_host(url, canonical_url):
        issues.append("canonical_external_host")
    if _normalize_url(url) != _normalize_url(canonical_url):
        issues.append("canonical_non_self")
    normalized_canonical = _normalize_url(canonical_url)
    if normalized_canonical not in normalized_crawled_urls:
        issues.append("canonical_target_not_crawled")
    target = indexability_by_url.get(canonical_url) or indexability_by_normalized_url.get(normalized_canonical)
    target_status = _safe_int((target or {}).get("http_status"))
    if 400 <= target_status < 500:
        issues.append("canonical_points_to_4xx")
    if 500 <= target_status < 600:
        issues.append("canonical_points_to_5xx")
    redirect_target_url = redirect_by_normalized_requested_url.get(normalized_canonical, "")
    if redirect_target_url and _normalize_url(redirect_target_url) != normalized_canonical:
        issues.append("canonical_points_to_redirect")
    target_page = page_by_normalized_url.get(normalized_canonical, {})
    target_canonical = target_page.get("canonical_url", "")
    if target_canonical and _normalize_url(target_canonical) != normalized_canonical:
        issues.append("non_canonical_page_specified_as_canonical_one")
    if (
        normalized_canonical != _normalize_url(url)
        and target
        and target.get("indexability_status")
        and target.get("indexability_status") != "indexable"
    ):
        issues.append("canonical_target_non_indexable")
    if canonical_counts.get(_normalize_url(canonical_url), 0) > 1 and _normalize_url(url) != _normalize_url(canonical_url):
        issues.append("canonical_target_shared")
    return list(dict.fromkeys(issues))


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _recommended_action(issues: list[str]) -> str:
    if not issues:
        return "No canonical action needed based on the current crawl."
    return ACTION_BY_ISSUE.get(issues[0], "Review canonical configuration.")


def _same_host(url: str, canonical_url: str) -> bool:
    parsed = urlparse(url)
    canonical = urlparse(canonical_url)
    if not canonical.netloc:
        return True
    left = parsed.netloc.lower().removeprefix("www.")
    right = canonical.netloc.lower().removeprefix("www.")
    return left == right


def _normalize_url(url: str) -> str:
    parsed = urlparse(url or "")
    if not parsed.scheme or not parsed.netloc:
        return (url or "").rstrip("/")
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{netloc}{path}"
