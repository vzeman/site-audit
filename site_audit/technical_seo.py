"""Shared technical SEO page and issue model.

This module does not run new audits by itself. It normalizes existing
technical signals into stable per-page and per-issue exports that later
technical SEO analyzers can extend.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

from .technical_issue_catalog import TECHNICAL_ISSUE_BY_KEY, TECHNICAL_ISSUE_CATALOG


_SEVERITY_WEIGHT = {"high": 100.0, "medium": 55.0, "low": 25.0}
_GOOGLEBOT_HTML_LIMIT_BYTES = 2 * 1024 * 1024
_METADATA_SEVERITY = {
    "missing_title": "high",
    "missing_canonical": "high",
    "canonical_external_host": "high",
    "missing_description": "medium",
    "duplicate_title": "medium",
    "duplicate_description": "medium",
    "short_title": "low",
    "long_title": "low",
    "short_description": "low",
    "long_description": "low",
    "incomplete_open_graph": "low",
    "missing_twitter_card": "low",
    "noindex": "high",
    "canonical_duplicate": "medium",
}


def build_technical_seo(
    pages: Iterable,
    *,
    indexability: dict | None = None,
    metadata_quality: dict | None = None,
    performance: dict | None = None,
    canonical_consistency: dict | None = None,
    history_changes: dict | None = None,
    linkgraph: dict | None = None,
    search_payload: dict | None = None,
    page_types: dict | None = None,
) -> dict:
    page_rows = [_base_page_row(page) for page in pages]
    by_url = {row["url"]: row for row in page_rows if row.get("url")}
    metadata = _lookup_rows((metadata_quality or {}).get("per_page") or [])
    perf = _lookup_rows((performance or {}).get("per_page") or [])
    canonical = _lookup_rows((canonical_consistency or {}).get("rows") or [])
    history = _lookup_rows((history_changes or {}).get("changes") or [])
    links = _lookup_rows((linkgraph or {}).get("page_link_counts") or [])
    search = _search_lookup(search_payload)
    types = _lookup_rows((page_types or {}).get("per_page") or [])
    skipped = _lookup_rows(((indexability or {}).get("skipped") or []) + ((indexability or {}).get("noindex_pages") or []))

    for url, row in by_url.items():
        _merge_page_signals(row, metadata.get(url), perf.get(url), canonical.get(url), history.get(url), links.get(url), search.get(url), types.get(url), skipped.get(url))

    for url, skip in skipped.items():
        if url in by_url:
            continue
        row = _base_skipped_row(skip)
        _merge_page_signals(row, metadata.get(url), perf.get(url), canonical.get(url), history.get(url), links.get(url), search.get(url), types.get(url), skip)
        by_url[url] = row

    rows = list(by_url.values())
    issues = []
    for row in rows:
        issues.extend(_issues_for_row(row))
    issue_counts = Counter(issue["issue_type"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    category_counts = Counter(issue["category"] for issue in issues)
    issue_urls = {issue["url"] for issue in issues}
    for row in rows:
        row_issues = [issue for issue in issues if issue["url"] == row["url"]]
        row["technical_issue_count"] = len(row_issues)
        row["technical_severity_score"] = round(sum(_SEVERITY_WEIGHT.get(issue["severity"], 0.0) * float(issue["confidence"] or 0.0) for issue in row_issues), 2)
        row["technical_high_issues"] = sum(1 for issue in row_issues if issue["severity"] == "high")
        row["technical_status"] = "needs_review" if row_issues else "clean"

    rows.sort(key=lambda row: (float(row.get("technical_severity_score") or 0.0), _safe_int(row.get("traffic"))), reverse=True)
    issues.sort(key=lambda issue: (float(issue.get("impact_score") or 0.0), _safe_int(issue.get("traffic"))), reverse=True)
    total = len(rows)
    return {
        "summary": {
            "status": "ok" if total else "no_pages",
            "model": "technical_seo_v1",
            "total_pages": total,
            "pages_with_issues": len(issue_urls),
            "issue_share": len(issue_urls) / total if total else 0.0,
            "total_issues": len(issues),
            "catalog_issue_types": len(TECHNICAL_ISSUE_CATALOG),
            "high_issues": severity_counts.get("high", 0),
            "medium_issues": severity_counts.get("medium", 0),
            "low_issues": severity_counts.get("low", 0),
            "traffic_at_risk": sum(_safe_int(row.get("traffic")) for row in rows if row["url"] in issue_urls),
        },
        "issue_counts": dict(issue_counts),
        "category_counts": dict(category_counts),
        "severity_counts": dict(severity_counts),
        "pages": rows,
        "issues": issues,
        "issue_catalog": TECHNICAL_ISSUE_CATALOG,
        "interpretation": {
            "technical_severity_score": "Weighted count of technical SEO issues on the URL. High severity issues count more than medium/low issues.",
            "impact_score": "Issue-level prioritization score combining severity, confidence, and search traffic where available.",
        },
    }


def _base_page_row(page) -> dict:
    return {
        "url": getattr(page, "url", "") or "",
        "title": getattr(page, "title", "") or "",
        "section": getattr(page, "section", "") or "",
        "word_count": _safe_int(getattr(page, "word_count", 0)),
        "language": getattr(page, "language", "") or "",
        "indexability_status": "indexable",
        "http_status": "",
        "canonical_url": "",
        "robots_content": "",
        "noindex_source": "",
        "nofollow": False,
        "nofollow_source": "",
        "metadata_issues": [],
        "traffic": 0,
        "keywords": 0,
        "top_keyword": "",
        "page_type": "",
        "template_family": "",
        "template_signature": "",
        "fix_scope": "page",
    }


def _base_skipped_row(row: dict) -> dict:
    return {
        "url": row.get("url") or "",
        "title": row.get("title") or "",
        "section": "",
        "word_count": "",
        "language": "",
        "indexability_status": row.get("reason") or "skipped",
        "http_status": row.get("http_status", ""),
        "canonical_url": "",
        "robots_content": "",
        "noindex_source": row.get("noindex_source") or row.get("source", ""),
        "nofollow": bool(row.get("nofollow")),
        "nofollow_source": row.get("nofollow_source", ""),
        "metadata_issues": [],
        "traffic": 0,
        "keywords": 0,
        "top_keyword": "",
        "page_type": "",
        "template_family": "",
        "template_signature": "",
        "fix_scope": "page",
    }


def _merge_page_signals(
    row: dict,
    metadata: dict | None,
    perf: dict | None,
    canonical: dict | None,
    history: dict | None,
    links: dict | None,
    search: dict | None,
    page_type: dict | None,
    skipped: dict | None,
) -> None:
    if metadata:
        row["title"] = row.get("title") or metadata.get("title", "")
        row["canonical_url"] = metadata.get("canonical_url", "")
        row["robots_content"] = metadata.get("robots_content", "")
        row["noindex_source"] = metadata.get("noindex_source", row.get("noindex_source", ""))
        row["nofollow"] = bool(metadata.get("nofollow"))
        row["nofollow_source"] = metadata.get("nofollow_source", "")
        row["metadata_issues"] = list(metadata.get("issues") or [])
    if perf:
        row["http_status"] = perf.get("status", row.get("http_status", ""))
        row["content_type"] = perf.get("content_type", "")
        row["html_weight_bytes"] = perf.get("html_weight_bytes", "")
        row["estimated_weight_bytes"] = perf.get("estimated_weight_bytes", "")
        row["weight_bucket"] = perf.get("weight_bucket", "")
        row["resource_tag_count"] = perf.get("resource_tag_count", "")
        row["render_blocking_count"] = perf.get("render_blocking_count", "")
        row["image_count"] = perf.get("image_count", "")
        row["script_count"] = perf.get("script_count", "")
        row["mixed_content_url_count"] = perf.get("mixed_content_url_count", 0)
        row["mixed_content_urls"] = list(perf.get("mixed_content_urls") or [])
    if canonical:
        row["canonical_url"] = row.get("canonical_url") or canonical.get("canonical_url", "")
        row["canonical_target_http_status"] = canonical.get("canonical_target_http_status", "")
        row["canonical_target_indexability_status"] = canonical.get("canonical_target_indexability_status", "")
        row["canonical_target_canonical_url"] = canonical.get("canonical_target_canonical_url", "")
        row["canonical_redirect_target_url"] = canonical.get("canonical_redirect_target_url", "")
        row["canonical_issues"] = list(canonical.get("issues") or [])
    if history:
        row["previous_canonical_url"] = history.get("canonical_before", "")
        row["canonical_changed"] = "canonical" in (history.get("changed_fields") or [])
        row["previous_indexability_status"] = history.get("indexability_before", "")
        row["current_indexability_status"] = history.get("indexability_after", "")
    if links:
        row["in_degree"] = _safe_int(links.get("in_degree"))
        row["out_degree"] = _safe_int(links.get("out_degree"))
        row["click_depth"] = links.get("click_depth", "")
        row["internal_http_link_count"] = _safe_int(links.get("internal_http_link_count"))
        row["internal_http_links"] = list(links.get("internal_http_links") or [])
        row["internal_https_link_count"] = _safe_int(links.get("internal_https_link_count"))
        row["internal_https_links"] = list(links.get("internal_https_links") or [])
    if search:
        row["traffic"] = _safe_int(search.get("traffic"))
        row["keywords"] = _safe_int(search.get("keywords"))
        row["top_keyword"] = search.get("top_keyword", "")
    if page_type:
        row["page_type"] = page_type.get("page_type", "")
        row["template_family"] = page_type.get("template_family", "")
        row["template_signature"] = page_type.get("template_signature", "")
        row["fix_scope"] = "template" if page_type.get("template_family") else "page"
    if skipped:
        reason = skipped.get("reason") or row.get("indexability_status") or "skipped"
        row["indexability_status"] = "noindex" if reason == "noindex" else reason
        row["http_status"] = skipped.get("http_status", row.get("http_status", ""))
        row["noindex_source"] = skipped.get("noindex_source") or skipped.get("source") or row.get("noindex_source", "")
        row["nofollow"] = bool(skipped.get("nofollow", row.get("nofollow", False)))
        row["nofollow_source"] = skipped.get("nofollow_source", row.get("nofollow_source", ""))


def _issues_for_row(row: dict) -> list[dict]:
    issues: list[dict] = []
    for issue_type in _http_status_issue_types(row):
        issues.append(_issue(row, "internal_pages", issue_type, "high", 0.98, _recommendation(issue_type)))
    status = str(row.get("indexability_status") or "")
    if status == "timed_out":
        issues.append(_issue(row, "internal_pages", "timed_out", "high", 0.98, _recommendation("timed_out")))
    elif status == "noindex":
        issues.append(_issue(row, "indexability", "noindex_page", "medium", 0.95, _recommendation("noindex_page")))
    elif status and status != "indexable":
        issues.append(_issue(row, "indexability", status, "high", 0.95, _recommendation(status)))
    if status == "noindex" and row.get("noindex_source") == "meta+header":
        issues.append(_issue(row, "indexability", "noindex_in_html_and_http_header", "medium", 0.94, _recommendation("noindex_in_html_and_http_header")))
    if status == "noindex" and row.get("nofollow"):
        issues.append(_issue(row, "indexability", "noindex_and_nofollow_page", "low", 0.86, _recommendation("noindex_and_nofollow_page")))
    if status == "noindex" and not row.get("nofollow"):
        issues.append(_issue(row, "indexability", "noindex_follow_page", "low", 0.86, _recommendation("noindex_follow_page")))
    for issue_type in row.get("metadata_issues") or []:
        issues.append(_issue(row, "metadata", issue_type, _METADATA_SEVERITY.get(issue_type, "medium"), 0.9, _recommendation(issue_type)))
    if row.get("nofollow") and row.get("nofollow_source") == "meta+header":
        issues.append(_issue(row, "indexability", "nofollow_in_html_and_http_header", "medium", 0.94, _recommendation("nofollow_in_html_and_http_header")))
    if row.get("nofollow"):
        issues.append(_issue(row, "indexability", "nofollow_page", "medium", 0.9, _recommendation("nofollow_page")))
    for issue_type in row.get("canonical_issues") or []:
        if issue_type in {"canonical_points_to_4xx", "canonical_points_to_5xx", "canonical_points_to_redirect"}:
            issues.append(_issue(row, "indexability", issue_type, "high", 0.96, _recommendation(issue_type)))
        elif issue_type == "non_canonical_page_specified_as_canonical_one":
            issues.append(_issue(row, "indexability", issue_type, "medium", 0.9, _recommendation(issue_type)))
        elif issue_type in {"canonical_from_http_to_https", "canonical_from_https_to_http"}:
            issues.append(_issue(row, "indexability", issue_type, "low", 0.86, _recommendation(issue_type)))
    if row.get("canonical_changed"):
        issues.append(_issue(row, "indexability", "canonical_url_changed", "low", 0.82, _recommendation("canonical_url_changed")))
    if row.get("previous_indexability_status") == "indexable" and row.get("current_indexability_status") not in {"", "indexable"}:
        issues.append(_issue(row, "indexability", "indexable_page_became_non_indexable", "low", 0.84, _recommendation("indexable_page_became_non_indexable")))
    if row.get("previous_indexability_status") == "noindex" and row.get("current_indexability_status") == "indexable":
        issues.append(_issue(row, "indexability", "noindex_page_became_indexable", "low", 0.84, _recommendation("noindex_page_became_indexable")))
    if _is_self_canonical(row) and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "indexable_canonical_url_has_no_incoming_internal_links", "high", 0.92, _recommendation("indexable_canonical_url_has_no_incoming_internal_links")))
    if status == "indexable" and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "indexable_orphan_page_has_no_incoming_internal_links", "high", 0.92, _recommendation("indexable_orphan_page_has_no_incoming_internal_links")))
    if status == "indexable" and _url_scheme(row.get("url", "")) == "https" and _safe_int(row.get("internal_http_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_https_page_has_internal_links_to_http", "high", 0.94, _recommendation("indexable_https_page_has_internal_links_to_http")))
    if _safe_int(row.get("html_weight_bytes")) > _GOOGLEBOT_HTML_LIMIT_BYTES:
        issues.append(_issue(row, "indexability", "page_size_exceeds_googlebot_s_2_mb_crawl_limit", "high", 0.9, _recommendation("page_size_exceeds_googlebot_s_2_mb_crawl_limit")))
    if row.get("weight_bucket") == "very_heavy":
        issues.append(_issue(row, "performance", "very_heavy_page", "medium", 0.72, "Reduce page weight, heavy images, scripts, and fonts."))
    elif row.get("weight_bucket") == "heavy":
        issues.append(_issue(row, "performance", "heavy_page", "low", 0.68, "Review page weight and resource count."))
    if _safe_int(row.get("render_blocking_count")) > 0:
        issues.append(_issue(row, "performance", "render_blocking_resources", "low", 0.72, "Defer non-critical scripts and reduce blocking stylesheets."))
    if _safe_int(row.get("mixed_content_url_count")) > 0:
        issues.append(_issue(row, "internal_pages", "https_http_mixed_content", "medium", 0.92, _recommendation("https_http_mixed_content")))
    return issues


def _http_status_issue_types(row: dict) -> list[str]:
    status = _safe_int(row.get("http_status"))
    if status == 404:
        return ["404_page", "4xx_page"]
    if 400 <= status < 500:
        return ["4xx_page"]
    if status == 500:
        return ["500_page", "5xx_page"]
    if 500 <= status < 600:
        return ["5xx_page"]
    return []


def _issue(row: dict, category: str, issue_type: str, severity: str, confidence: float, recommendation: str) -> dict:
    traffic = _safe_int(row.get("traffic"))
    impact = _SEVERITY_WEIGHT.get(severity, 25.0) * confidence * (1.0 + min(3.0, traffic / 1000.0))
    catalog = TECHNICAL_ISSUE_BY_KEY.get(issue_type, {})
    return {
        "url": row.get("url", ""),
        "title": row.get("title", ""),
        "category": category,
        "issue_type": issue_type,
        "issue_name": catalog.get("name", issue_type.replace("_", " ")),
        "importance": catalog.get("importance", severity),
        "severity": severity,
        "confidence": round(confidence, 2),
        "traffic": traffic,
        "keywords": _safe_int(row.get("keywords")),
        "page_type": row.get("page_type", ""),
        "template_family": row.get("template_family", ""),
        "fix_scope": row.get("fix_scope", "page"),
        "impact_score": round(impact, 2),
        "recommendation": recommendation,
    }


def _recommendation(issue_type: str) -> str:
    return {
        "noindex": "Decide whether this URL should be indexed; remove noindex only when it is a canonical search landing page.",
        "noindex_page": "Decide whether this URL should be indexed; remove noindex only when it is a canonical search landing page.",
        "canonical_duplicate": "Remove this non-canonical URL from internal links and sitemaps, or change its canonical if it should be indexable.",
        "404_page": "Restore the page, redirect it to a relevant live URL, or remove internal links and sitemap references.",
        "4xx_page": "Fix the client error or remove internal links and sitemap references to this URL.",
        "500_page": "Fix the server error so the URL returns a stable 2xx response, or remove it from crawlable SEO surfaces.",
        "5xx_page": "Investigate server-side failures and make the URL reliably crawlable before keeping it in SEO surfaces.",
        "timed_out": "Reduce response time, fix blocking behavior, or remove the URL from crawlable SEO surfaces.",
        "https_http_mixed_content": "Serve every embedded resource on the HTTPS page over HTTPS or remove the insecure resource.",
        "canonical_points_to_4xx": "Update the canonical tag to a live 2xx URL, restore the canonical target, or remove the broken canonical.",
        "canonical_points_to_5xx": "Fix the canonical target server error or update the canonical tag to a stable live 2xx URL.",
        "canonical_points_to_redirect": "Change the canonical tag to the final destination URL and avoid canonicalizing through redirects.",
        "non_canonical_page_specified_as_canonical_one": "Update the canonical tag to point directly at the final self-canonical URL.",
        "canonical_from_http_to_https": "Use the HTTPS URL as the crawled and internally linked version so the canonical does not need to consolidate from HTTP.",
        "canonical_from_https_to_http": "Update the canonical tag to the HTTPS URL and avoid consolidating secure pages to HTTP.",
        "canonical_url_changed": "Review the canonical change against the previous snapshot and confirm the new canonical target is intentional.",
        "indexable_page_became_non_indexable": "Review the before/after snapshot and restore indexability if this URL should remain eligible for search.",
        "noindex_page_became_indexable": "Review the before/after snapshot and confirm this formerly noindex URL should now be indexable.",
        "indexable_canonical_url_has_no_incoming_internal_links": "Add at least one crawlable internal link to this canonical URL from a relevant page.",
        "indexable_orphan_page_has_no_incoming_internal_links": "Add crawlable internal links to this orphan page from relevant navigation, hub, or content pages.",
        "indexable_https_page_has_internal_links_to_http": "Update internal links on this HTTPS page so they point directly to HTTPS URLs.",
        "page_size_exceeds_googlebot_s_2_mb_crawl_limit": "Reduce the HTML document below 2 MB by trimming inline markup, scripts, styles, or excessive embedded data.",
        "nofollow_in_html_and_http_header": "Remove duplicate nofollow directives from either the HTML meta robots tag or the X-Robots-Tag header unless both are intentional.",
        "nofollow_page": "Review whether this page should prevent link discovery; remove the nofollow directive when internal links should pass crawl signals.",
        "noindex_in_html_and_http_header": "Remove duplicate noindex directives from either the HTML meta robots tag or the X-Robots-Tag header unless both are intentional.",
        "noindex_and_nofollow_page": "Confirm this page should be excluded from indexing and that links on it should not be followed.",
        "noindex_follow_page": "Confirm this page should be excluded from indexing while allowing crawlers to follow links from it.",
        "empty_embedding_text": "Add crawlable main content or remove the URL from SEO surfaces.",
        "unusable": "Review extraction/crawlability and ensure the page has readable HTML main content.",
        "missing_title": "Add a unique descriptive title.",
        "short_title": "Expand the title to describe the primary topic.",
        "long_title": "Shorten the title while preserving the primary query intent.",
        "missing_description": "Add a concise meta description aligned with the page intent.",
        "duplicate_title": "Rewrite the title so it is unique to this URL.",
        "duplicate_description": "Rewrite the meta description so it is unique to this URL.",
        "missing_canonical": "Add a self-referencing canonical or the correct canonical target.",
        "canonical_external_host": "Verify that canonicalizing to an external host is intentional.",
        "incomplete_open_graph": "Complete OpenGraph title and description for share previews.",
        "missing_twitter_card": "Add Twitter/X card metadata if social previews matter.",
    }.get(issue_type, "Review and fix this technical SEO issue.")


def _lookup_rows(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        url = row.get("url") or row.get("matched_url")
        if url:
            out[str(url)] = row
    return out


def _is_self_canonical(row: dict) -> bool:
    if row.get("indexability_status") != "indexable":
        return False
    url = row.get("url", "")
    canonical_url = row.get("canonical_url", "")
    if not url or not canonical_url:
        return False
    return _normalize_url(url) == _normalize_url(canonical_url)


def _normalize_url(url: str) -> str:
    parsed = urlparse(url or "")
    if not parsed.scheme or not parsed.netloc:
        return (url or "").rstrip("/")
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{netloc}{path}"


def _url_scheme(url: str) -> str:
    return urlparse(url or "").scheme.lower()


def _search_lookup(payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url")
        if not url:
            continue
        out[str(url)] = {
            "traffic": row.get("traffic", 0),
            "keywords": row.get("keywords", 0),
            "top_keyword": row.get("top_keyword", ""),
        }
    return out


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0
