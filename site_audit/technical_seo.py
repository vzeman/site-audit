"""Shared technical SEO page and issue model.

This module does not run new audits by itself. It normalizes existing
technical signals into stable per-page and per-issue exports that later
technical SEO analyzers can extend.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable
from urllib.parse import urlparse

from .technical_issue_catalog import TECHNICAL_ISSUE_BY_KEY, TECHNICAL_ISSUE_CATALOG


_SEVERITY_WEIGHT = {"high": 100.0, "medium": 55.0, "low": 25.0}
_GOOGLEBOT_HTML_LIMIT_BYTES = 2 * 1024 * 1024
_MAX_REDIRECT_HOPS = 5
_LOW_WORD_COUNT_THRESHOLD = 200
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
    header_analysis: dict | None = None,
    content_quality: dict | None = None,
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
    headers = _lookup_rows((header_analysis or {}).get("per_page") or [])
    quality = _lookup_rows((content_quality or {}).get("per_page") or (content_quality or {}).get("rows") or [])
    index_rows = _lookup_rows((indexability or {}).get("per_page") or [])
    skipped = _lookup_rows(((indexability or {}).get("skipped") or []) + ((indexability or {}).get("noindex_pages") or []))

    for url, row in by_url.items():
        _merge_page_signals(row, metadata.get(url), perf.get(url), canonical.get(url), history.get(url), links.get(url), index_rows.get(url), search.get(url), types.get(url), headers.get(url), quality.get(url), skipped.get(url))

    for url, skip in skipped.items():
        if url in by_url:
            continue
        row = _base_skipped_row(skip)
        _merge_page_signals(row, metadata.get(url), perf.get(url), canonical.get(url), history.get(url), links.get(url), index_rows.get(url), search.get(url), types.get(url), headers.get(url), quality.get(url), skip)
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
    indexability: dict | None,
    search: dict | None,
    page_type: dict | None,
    header: dict | None,
    quality: dict | None,
    skipped: dict | None,
) -> None:
    if metadata:
        row["title"] = row.get("title") or metadata.get("title", "")
        row["title_length"] = _safe_int(metadata.get("title_length"))
        row["description"] = metadata.get("description", row.get("description", ""))
        row["description_length"] = _safe_int(metadata.get("description_length"))
        row["canonical_url"] = metadata.get("canonical_url", "")
        row["robots_content"] = metadata.get("robots_content", "")
        row["noindex_source"] = metadata.get("noindex_source", row.get("noindex_source", ""))
        row["nofollow"] = bool(metadata.get("nofollow"))
        row["nofollow_source"] = metadata.get("nofollow_source", "")
        row["meta_refresh_redirect"] = bool(metadata.get("meta_refresh_redirect"))
        row["meta_refresh_target_url"] = metadata.get("meta_refresh_target_url", "")
        row["title_tag_count"] = _safe_int(metadata.get("title_tag_count"))
        row["meta_description_tag_count"] = _safe_int(metadata.get("meta_description_tag_count"))
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
        row["previous_title"] = history.get("title_before", "")
        row["current_title"] = history.get("title_after", "")
        row["title_changed"] = "title" in (history.get("changed_fields") or [])
        row["previous_canonical_url"] = history.get("canonical_before", "")
        row["canonical_changed"] = "canonical" in (history.get("changed_fields") or [])
        row["previous_description"] = history.get("description_before", "")
        row["current_description"] = history.get("description_after", "")
        row["description_changed"] = "description" in (history.get("changed_fields") or [])
        row["previous_indexability_status"] = history.get("indexability_before", "")
        row["current_indexability_status"] = history.get("indexability_after", "")
        row["previous_h1"] = history.get("h1_before", "")
        row["current_h1"] = history.get("h1_after", "")
        row["h1_changed"] = "h1" in (history.get("changed_fields") or [])
        row["previous_serp_title"] = history.get("serp_title_before", "")
        row["current_serp_title"] = history.get("serp_title_after", "")
        row["serp_title_changed"] = "serp_title" in (history.get("changed_fields") or [])
        row["previous_word_count"] = _safe_int(history.get("word_count_before"))
        row["current_word_count"] = _safe_int(history.get("word_count_after"))
        row["word_count_changed"] = "word_count" in (history.get("changed_fields") or [])
        row["previous_redirect_target_url"] = history.get("redirect_target_before", "")
        row["current_redirect_target_url"] = history.get("redirect_target_after", "")
        row["redirect_target_changed"] = "redirect_target" in (history.get("changed_fields") or [])
    if links:
        row["in_degree"] = _safe_int(links.get("in_degree"))
        row["out_degree"] = _safe_int(links.get("out_degree"))
        row["click_depth"] = links.get("click_depth", "")
        row["raw_internal_link_count"] = _safe_int(links.get("raw_internal_link_count"))
        row["internal_http_link_count"] = _safe_int(links.get("internal_http_link_count"))
        row["internal_http_links"] = list(links.get("internal_http_links") or [])
        row["internal_https_link_count"] = _safe_int(links.get("internal_https_link_count"))
        row["internal_https_links"] = list(links.get("internal_https_links") or [])
        row["internal_link_counts_available"] = any(
            key in links
            for key in ("raw_internal_link_count", "internal_http_link_count", "internal_https_link_count")
        )
        row["broken_internal_link_count"] = _safe_int(links.get("broken_internal_link_count"))
        row["broken_internal_links"] = list(links.get("broken_internal_links") or [])
        row["redirect_internal_link_count"] = _safe_int(links.get("redirect_internal_link_count"))
        row["redirect_internal_links"] = list(links.get("redirect_internal_links") or [])
        row["incoming_nofollow_internal_link_count"] = _safe_int(links.get("incoming_nofollow_internal_link_count"))
        row["incoming_dofollow_internal_link_count"] = _safe_int(links.get("incoming_dofollow_internal_link_count"))
        row["incoming_nofollow_internal_links"] = list(links.get("incoming_nofollow_internal_links") or [])
        row["incoming_dofollow_internal_links"] = list(links.get("incoming_dofollow_internal_links") or [])
        row["outgoing_nofollow_internal_link_count"] = _safe_int(links.get("outgoing_nofollow_internal_link_count"))
        row["outgoing_nofollow_internal_links"] = list(links.get("outgoing_nofollow_internal_links") or [])
    if indexability:
        row["requested_url"] = indexability.get("requested_url", "")
        row["redirect_target_url"] = indexability.get("redirect_target_url", "")
        row["redirect_chain"] = list(indexability.get("redirect_chain") or [])
        row["redirect_hop_count"] = _safe_int(indexability.get("redirect_hop_count"))
        row["redirect_status_codes"] = [_safe_int(code) for code in (indexability.get("redirect_status_codes") or [])]
        row["meta_refresh_redirect"] = bool(indexability.get("meta_refresh_redirect", row.get("meta_refresh_redirect", False)))
        row["meta_refresh_target_url"] = indexability.get("meta_refresh_target_url", row.get("meta_refresh_target_url", ""))
        row["title_tag_count"] = _safe_int(indexability.get("title_tag_count", row.get("title_tag_count", 0)))
        row["meta_description_tag_count"] = _safe_int(indexability.get("meta_description_tag_count", row.get("meta_description_tag_count", 0)))
    if search:
        row["traffic"] = _safe_int(search.get("traffic"))
        row["keywords"] = _safe_int(search.get("keywords"))
        row["top_keyword"] = search.get("top_keyword", "")
        row["serp_title"] = search.get("serp_title", "")
    if page_type:
        row["page_type"] = page_type.get("page_type", "")
        row["template_family"] = page_type.get("template_family", "")
        row["template_signature"] = page_type.get("template_signature", "")
        row["fix_scope"] = "template" if page_type.get("template_family") else "page"
    if header:
        row["h1"] = header.get("h1", "")
        row["h1_count"] = _safe_int(header.get("h1_count"))
        row["header_count"] = _safe_int(header.get("header_count"))
    if quality:
        row["ai_content_level"] = quality.get("ai_content_level", "")
        row["ai_content_score"] = _safe_float(quality.get("ai_content_score", quality.get("ai_content_probability", 0.0)))
        row["ai_content_probability"] = _safe_float(quality.get("ai_content_probability", quality.get("ai_content_score", 0.0)))
    if skipped:
        reason = skipped.get("reason") or row.get("indexability_status") or "skipped"
        row["indexability_status"] = "noindex" if reason == "noindex" else reason
        row["http_status"] = skipped.get("http_status", row.get("http_status", ""))
        row["noindex_source"] = skipped.get("noindex_source") or skipped.get("source") or row.get("noindex_source", "")
        row["nofollow"] = bool(skipped.get("nofollow", row.get("nofollow", False)))
        row["nofollow_source"] = skipped.get("nofollow_source", row.get("nofollow_source", ""))
        row["word_count"] = _safe_int(skipped.get("word_count", row.get("word_count", 0)))
        row["requested_url"] = skipped.get("requested_url", row.get("requested_url", ""))
        row["redirect_target_url"] = skipped.get("redirect_target_url", row.get("redirect_target_url", ""))
        row["redirect_chain"] = list(skipped.get("redirect_chain") or row.get("redirect_chain", []) or [])
        row["redirect_hop_count"] = _safe_int(skipped.get("redirect_hop_count", row.get("redirect_hop_count", 0)))
        row["redirect_status_codes"] = [_safe_int(code) for code in (skipped.get("redirect_status_codes") or row.get("redirect_status_codes", []) or [])]
        row["meta_refresh_redirect"] = bool(skipped.get("meta_refresh_redirect", row.get("meta_refresh_redirect", False)))
        row["meta_refresh_target_url"] = skipped.get("meta_refresh_target_url", row.get("meta_refresh_target_url", ""))
        row["title_tag_count"] = _safe_int(skipped.get("title_tag_count", row.get("title_tag_count", 0)))
        row["meta_description_tag_count"] = _safe_int(skipped.get("meta_description_tag_count", row.get("meta_description_tag_count", 0)))


def _issues_for_row(row: dict) -> list[dict]:
    issues: list[dict] = []
    for issue_type in _http_status_issue_types(row):
        issues.append(_issue(row, "internal_pages", issue_type, "high", 0.98, _recommendation(issue_type)))
    status = str(row.get("indexability_status") or "")
    if status == "timed_out":
        issues.append(_issue(row, "internal_pages", "timed_out", "high", 0.98, _recommendation("timed_out")))
    elif status == "redirect_loop":
        issues.append(_issue(row, "redirects", "redirect_loop", "high", 0.96, _recommendation("redirect_loop")))
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
    if _is_redirected_fetch(row) and _safe_int(row.get("http_status")) >= 400:
        issues.append(_issue(row, "redirects", "broken_redirect", "high", 0.96, _recommendation("broken_redirect")))
    if _safe_int(row.get("redirect_hop_count")) > _MAX_REDIRECT_HOPS:
        issues.append(_issue(row, "redirects", "redirect_chain_too_long", "high", 0.94, _recommendation("redirect_chain_too_long")))
    if _safe_int(row.get("redirect_hop_count")) > 1:
        issues.append(_issue(row, "redirects", "redirect_chain", "low", 0.82, _recommendation("redirect_chain")))
    if 302 in (row.get("redirect_status_codes") or []):
        issues.append(_issue(row, "redirects", "302_redirect", "medium", 0.9, _recommendation("302_redirect")))
    if any(300 <= _safe_int(code) < 400 for code in (row.get("redirect_status_codes") or [])):
        issues.append(_issue(row, "redirects", "3xx_redirect", "medium", 0.88, _recommendation("3xx_redirect")))
    if _is_redirected_fetch(row) and _url_scheme(row.get("requested_url", "")) == "https" and _url_scheme(row.get("redirect_target_url", "")) == "http":
        issues.append(_issue(row, "redirects", "https_to_http_redirect", "medium", 0.9, _recommendation("https_to_http_redirect")))
    if _is_redirected_fetch(row) and _url_scheme(row.get("requested_url", "")) == "http" and _url_scheme(row.get("redirect_target_url", "")) == "https":
        issues.append(_issue(row, "redirects", "http_to_https_redirect", "low", 0.82, _recommendation("http_to_https_redirect")))
    if row.get("meta_refresh_redirect"):
        issues.append(_issue(row, "redirects", "meta_refresh_redirect", "low", 0.82, _recommendation("meta_refresh_redirect")))
    if row.get("redirect_target_changed"):
        issues.append(_issue(row, "redirects", "redirect_target_changed", "low", 0.8, _recommendation("redirect_target_changed")))
    if status == "indexable" and _safe_int(row.get("meta_description_tag_count")) > 1:
        issues.append(_issue(row, "content", "indexable_multiple_meta_description_tags", "high", 0.92, _recommendation("indexable_multiple_meta_description_tags")))
    if status != "indexable" and _safe_int(row.get("meta_description_tag_count")) > 1:
        issues.append(_issue(row, "content", "not_indexable_multiple_meta_description_tags", "medium", 0.86, _recommendation("not_indexable_multiple_meta_description_tags")))
    if status == "indexable" and _safe_int(row.get("title_tag_count")) > 1:
        issues.append(_issue(row, "content", "indexable_multiple_title_tags", "high", 0.92, _recommendation("indexable_multiple_title_tags")))
    if status != "indexable" and _safe_int(row.get("title_tag_count")) > 1:
        issues.append(_issue(row, "content", "not_indexable_multiple_title_tags", "medium", 0.86, _recommendation("not_indexable_multiple_title_tags")))
    if status == "indexable" and "h1_count" in row and (_safe_int(row.get("h1_count")) == 0 or not str(row.get("h1") or "").strip()):
        issues.append(_issue(row, "content", "indexable_h1_tag_missing_or_empty", "medium", 0.9, _recommendation("indexable_h1_tag_missing_or_empty")))
    if status != "indexable" and "h1_count" in row and (_safe_int(row.get("h1_count")) == 0 or not str(row.get("h1") or "").strip()):
        issues.append(_issue(row, "content", "not_indexable_h1_tag_missing_or_empty", "low", 0.82, _recommendation("not_indexable_h1_tag_missing_or_empty")))
    if status == "indexable" and _safe_int(row.get("h1_count")) > 1:
        issues.append(_issue(row, "content", "indexable_multiple_h1_tags", "low", 0.82, _recommendation("indexable_multiple_h1_tags")))
    if status == "indexable" and row.get("h1_changed"):
        issues.append(_issue(row, "content", "indexable_h1_tag_changed", "low", 0.8, _recommendation("indexable_h1_tag_changed")))
    if status == "indexable" and row.get("description_changed"):
        issues.append(_issue(row, "content", "indexable_meta_description_changed", "low", 0.8, _recommendation("indexable_meta_description_changed")))
    if status == "indexable" and _titles_do_not_match(row.get("title", ""), row.get("serp_title", "")):
        issues.append(_issue(row, "content", "indexable_page_and_serp_titles_do_not_match", "low", 0.82, _recommendation("indexable_page_and_serp_titles_do_not_match")))
    if status == "indexable" and row.get("serp_title_changed"):
        issues.append(_issue(row, "content", "indexable_serp_title_changed", "low", 0.8, _recommendation("indexable_serp_title_changed")))
    if status == "indexable" and row.get("title_changed"):
        issues.append(_issue(row, "content", "indexable_title_tag_changed", "low", 0.8, _recommendation("indexable_title_tag_changed")))
    if status == "indexable" and _has_high_ai_content(row):
        issues.append(_issue(row, "content", "indexable_pages_have_high_ai_content_levels", "low", 0.82, _recommendation("indexable_pages_have_high_ai_content_levels")))
    if status == "indexable" and row.get("word_count_changed"):
        issues.append(_issue(row, "content", "indexable_word_count_changed", "low", 0.8, _recommendation("indexable_word_count_changed")))
    if status == "indexable" and 0 < _safe_int(row.get("word_count")) < _LOW_WORD_COUNT_THRESHOLD:
        issues.append(_issue(row, "content", "indexable_low_word_count", "medium", 0.86, _recommendation("indexable_low_word_count")))
    if status != "indexable" and 0 < _safe_int(row.get("word_count")) < _LOW_WORD_COUNT_THRESHOLD:
        issues.append(_issue(row, "content", "not_indexable_low_word_count", "low", 0.8, _recommendation("not_indexable_low_word_count")))
    if (
        status == "indexable"
        and (
            "missing_description" in (row.get("metadata_issues") or [])
            or ("meta_description_tag_count" in row and _safe_int(row.get("meta_description_tag_count")) == 0)
        )
    ):
        issues.append(_issue(row, "content", "indexable_meta_description_tag_missing_or_empty", "medium", 0.9, _recommendation("indexable_meta_description_tag_missing_or_empty")))
    if (
        status != "indexable"
        and (
            "missing_description" in (row.get("metadata_issues") or [])
            or ("meta_description_tag_count" in row and _safe_int(row.get("meta_description_tag_count")) == 0)
        )
    ):
        issues.append(_issue(row, "content", "not_indexable_meta_description_tag_missing_or_empty", "medium", 0.86, _recommendation("not_indexable_meta_description_tag_missing_or_empty")))
    if (
        status == "indexable"
        and (
            "long_description" in (row.get("metadata_issues") or [])
            or _safe_int(row.get("description_length")) > 160
        )
    ):
        issues.append(_issue(row, "content", "indexable_meta_description_too_long", "medium", 0.86, _recommendation("indexable_meta_description_too_long")))
    if (
        status != "indexable"
        and (
            "long_description" in (row.get("metadata_issues") or [])
            or _safe_int(row.get("description_length")) > 160
        )
    ):
        issues.append(_issue(row, "content", "not_indexable_meta_description_too_long", "low", 0.8, _recommendation("not_indexable_meta_description_too_long")))
    if (
        status == "indexable"
        and (
            "short_description" in (row.get("metadata_issues") or [])
            or 0 < _safe_int(row.get("description_length")) < 50
        )
    ):
        issues.append(_issue(row, "content", "indexable_meta_description_too_short", "medium", 0.86, _recommendation("indexable_meta_description_too_short")))
    if (
        status != "indexable"
        and (
            "short_description" in (row.get("metadata_issues") or [])
            or 0 < _safe_int(row.get("description_length")) < 50
        )
    ):
        issues.append(_issue(row, "content", "not_indexable_meta_description_too_short", "low", 0.8, _recommendation("not_indexable_meta_description_too_short")))
    if (
        status == "indexable"
        and (
            "missing_title" in (row.get("metadata_issues") or [])
            or not str(row.get("title") or "").strip()
            or ("title_tag_count" in row and _safe_int(row.get("title_tag_count")) == 0)
        )
    ):
        issues.append(_issue(row, "content", "indexable_title_tag_missing_or_empty", "high", 0.94, _recommendation("indexable_title_tag_missing_or_empty")))
    if (
        status != "indexable"
        and (
            "missing_title" in (row.get("metadata_issues") or [])
            or not str(row.get("title") or "").strip()
            or ("title_tag_count" in row and _safe_int(row.get("title_tag_count")) == 0)
        )
    ):
        issues.append(_issue(row, "content", "not_indexable_title_tag_missing_or_empty", "medium", 0.86, _recommendation("not_indexable_title_tag_missing_or_empty")))
    if (
        status == "indexable"
        and (
            "long_title" in (row.get("metadata_issues") or [])
            or _safe_int(row.get("title_length")) > 65
        )
    ):
        issues.append(_issue(row, "content", "indexable_title_too_long", "medium", 0.86, _recommendation("indexable_title_too_long")))
    if (
        status == "indexable"
        and (
            "short_title" in (row.get("metadata_issues") or [])
            or 0 < _safe_int(row.get("title_length")) < 20
        )
    ):
        issues.append(_issue(row, "content", "indexable_title_too_short", "medium", 0.86, _recommendation("indexable_title_too_short")))
    if _is_self_canonical(row) and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "indexable_canonical_url_has_no_incoming_internal_links", "high", 0.92, _recommendation("indexable_canonical_url_has_no_incoming_internal_links")))
    if status == "indexable" and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "indexable_orphan_page_has_no_incoming_internal_links", "high", 0.92, _recommendation("indexable_orphan_page_has_no_incoming_internal_links")))
    if status != "indexable" and "in_degree" in row and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "not_indexable_orphan_page_has_no_incoming_internal_links", "medium", 0.88, _recommendation("not_indexable_orphan_page_has_no_incoming_internal_links")))
    if status == "indexable" and _url_scheme(row.get("url", "")) == "https" and _safe_int(row.get("internal_http_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_https_page_has_internal_links_to_http", "high", 0.94, _recommendation("indexable_https_page_has_internal_links_to_http")))
    if status != "indexable" and _url_scheme(row.get("url", "")) == "https" and _safe_int(row.get("internal_http_link_count")) > 0:
        issues.append(_issue(row, "links", "not_indexable_https_page_has_internal_links_to_http", "medium", 0.9, _recommendation("not_indexable_https_page_has_internal_links_to_http")))
    if status == "indexable" and _url_scheme(row.get("url", "")) == "http" and _safe_int(row.get("internal_https_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_http_page_has_internal_links_to_https", "low", 0.86, _recommendation("indexable_http_page_has_internal_links_to_https")))
    if status != "indexable" and _url_scheme(row.get("url", "")) == "http" and _safe_int(row.get("internal_https_link_count")) > 0:
        issues.append(_issue(row, "links", "not_indexable_http_page_has_internal_links_to_https", "low", 0.82, _recommendation("not_indexable_http_page_has_internal_links_to_https")))
    if status == "indexable" and _safe_int(row.get("broken_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_page_has_links_to_broken_page", "high", 0.94, _recommendation("indexable_page_has_links_to_broken_page")))
    if status != "indexable" and _safe_int(row.get("broken_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "not_indexable_page_has_links_to_broken_page", "medium", 0.9, _recommendation("not_indexable_page_has_links_to_broken_page")))
    if status == "indexable" and _safe_int(row.get("redirect_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_page_has_links_to_redirect", "medium", 0.9, _recommendation("indexable_page_has_links_to_redirect")))
    if status != "indexable" and _safe_int(row.get("redirect_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "not_indexable_page_has_links_to_redirect", "low", 0.82, _recommendation("not_indexable_page_has_links_to_redirect")))
    if status == "indexable" and _is_redirected_fetch(row) and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "indexable_redirected_page_has_no_incoming_internal_links", "medium", 0.88, _recommendation("indexable_redirected_page_has_no_incoming_internal_links")))
    if status != "indexable" and _is_redirected_fetch(row) and "in_degree" in row and _safe_int(row.get("in_degree")) == 0:
        issues.append(_issue(row, "links", "not_indexable_redirected_page_has_no_incoming_internal_links", "low", 0.82, _recommendation("not_indexable_redirected_page_has_no_incoming_internal_links")))
    if status == "indexable" and row.get("internal_link_counts_available") and _safe_int(row.get("raw_internal_link_count")) == 0:
        issues.append(_issue(row, "links", "indexable_page_has_no_outgoing_links", "high", 0.9, _recommendation("indexable_page_has_no_outgoing_links")))
    if status != "indexable" and row.get("internal_link_counts_available") and _safe_int(row.get("raw_internal_link_count")) == 0:
        issues.append(_issue(row, "links", "not_indexable_page_has_no_outgoing_links", "medium", 0.86, _recommendation("not_indexable_page_has_no_outgoing_links")))
    if (
        status == "indexable"
        and _safe_int(row.get("incoming_nofollow_internal_link_count")) > 0
        and _safe_int(row.get("incoming_dofollow_internal_link_count")) == 0
    ):
        issues.append(_issue(row, "links", "indexable_page_has_nofollow_incoming_internal_links_only", "medium", 0.9, _recommendation("indexable_page_has_nofollow_incoming_internal_links_only")))
    if (
        status != "indexable"
        and _safe_int(row.get("incoming_nofollow_internal_link_count")) > 0
        and _safe_int(row.get("incoming_dofollow_internal_link_count")) == 0
    ):
        issues.append(_issue(row, "links", "not_indexable_page_has_nofollow_incoming_internal_links_only", "low", 0.82, _recommendation("not_indexable_page_has_nofollow_incoming_internal_links_only")))
    if (
        status == "indexable"
        and _safe_int(row.get("incoming_nofollow_internal_link_count")) > 0
        and _safe_int(row.get("incoming_dofollow_internal_link_count")) > 0
    ):
        issues.append(_issue(row, "links", "indexable_page_has_nofollow_and_dofollow_incoming_internal_links", "low", 0.84, _recommendation("indexable_page_has_nofollow_and_dofollow_incoming_internal_links")))
    if (
        status != "indexable"
        and _safe_int(row.get("incoming_nofollow_internal_link_count")) > 0
        and _safe_int(row.get("incoming_dofollow_internal_link_count")) > 0
    ):
        issues.append(_issue(row, "links", "not_indexable_page_has_nofollow_and_dofollow_incoming_internal_links", "low", 0.82, _recommendation("not_indexable_page_has_nofollow_and_dofollow_incoming_internal_links")))
    if status == "indexable" and _safe_int(row.get("outgoing_nofollow_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "indexable_page_has_nofollow_outgoing_internal_links", "low", 0.84, _recommendation("indexable_page_has_nofollow_outgoing_internal_links")))
    if status != "indexable" and _safe_int(row.get("outgoing_nofollow_internal_link_count")) > 0:
        issues.append(_issue(row, "links", "not_indexable_page_has_nofollow_outgoing_internal_links", "low", 0.82, _recommendation("not_indexable_page_has_nofollow_outgoing_internal_links")))
    if status == "indexable" and _safe_int(row.get("incoming_dofollow_internal_link_count")) == 1:
        issues.append(_issue(row, "links", "indexable_page_has_only_one_dofollow_incoming_internal_link", "low", 0.82, _recommendation("indexable_page_has_only_one_dofollow_incoming_internal_link")))
    if status != "indexable" and _safe_int(row.get("incoming_dofollow_internal_link_count")) == 1:
        issues.append(_issue(row, "links", "not_indexable_page_has_only_one_dofollow_incoming_internal_link", "low", 0.8, _recommendation("not_indexable_page_has_only_one_dofollow_incoming_internal_link")))
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
        "broken_redirect": "Update the redirect so it resolves to a live 2XX destination or remove links and sitemap references to the redirecting URL.",
        "redirect_chain_too_long": "Collapse the redirect path so the requested URL redirects directly to the final destination.",
        "redirect_chain": "Collapse multi-hop redirects so the requested URL points directly to the final destination.",
        "redirect_loop": "Fix the redirect rules so the URL resolves to a final destination instead of cycling between URLs.",
        "302_redirect": "Review whether the temporary redirect should be a permanent 301/308 redirect for SEO consolidation.",
        "3xx_redirect": "Review redirecting URLs and update internal links or sitemap references to the final destination where appropriate.",
        "https_to_http_redirect": "Change the redirect target to HTTPS so secure URLs do not downgrade users or crawl signals to HTTP.",
        "http_to_https_redirect": "Update internal links and sitemap URLs to the HTTPS destination so crawlers do not need the HTTP redirect hop.",
        "meta_refresh_redirect": "Replace the meta refresh with a server-side redirect or direct internal links to the target URL.",
        "redirect_target_changed": "Review the changed redirect destination and confirm the new target is intentional.",
        "indexable_multiple_meta_description_tags": "Keep one meta description tag per indexable page and remove duplicate description tags from the template.",
        "not_indexable_multiple_meta_description_tags": "Keep one meta description tag on non-indexable pages that remain in crawl paths, or remove duplicate tags from the template.",
        "indexable_multiple_title_tags": "Keep one title tag per indexable page and remove duplicate title tags from the template.",
        "not_indexable_multiple_title_tags": "Keep one title tag on non-indexable pages that remain in crawl paths, or remove duplicate title tags from the template.",
        "indexable_h1_tag_missing_or_empty": "Add one clear H1 heading to the indexable page.",
        "not_indexable_h1_tag_missing_or_empty": "Review whether the non-indexable page still needs a clear H1 if it remains useful to users or internal crawl paths.",
        "indexable_multiple_h1_tags": "Keep one primary H1 heading and demote extra H1s to lower heading levels.",
        "indexable_h1_tag_changed": "Review the H1 change and confirm the new heading still matches the page intent.",
        "indexable_meta_description_changed": "Review the meta description change and confirm the new snippet still matches search intent.",
        "indexable_page_and_serp_titles_do_not_match": "Review the SERP title Google is showing and align the page title when the rewrite is not intentional.",
        "indexable_serp_title_changed": "Review the changed SERP title and confirm the displayed search result still matches page intent.",
        "indexable_title_tag_changed": "Review the title tag change and confirm it still targets the intended search demand.",
        "indexable_pages_have_high_ai_content_levels": "Review pages with high AI-content scores and add original evidence, expert detail, and brand-specific value.",
        "indexable_word_count_changed": "Review the word count change and confirm the page still satisfies its search intent.",
        "indexable_low_word_count": "Review whether the indexable page has enough crawlable main content to satisfy its search intent.",
        "not_indexable_low_word_count": "Review whether the non-indexable page needs more crawlable content if it remains useful to users or may become indexable later.",
        "indexable_meta_description_tag_missing_or_empty": "Add one concise meta description tag to the indexable page.",
        "not_indexable_meta_description_tag_missing_or_empty": "Review whether the non-indexable page still needs a meta description; add one if it appears in crawl paths, previews, or future indexing plans.",
        "indexable_meta_description_too_long": "Shorten the meta description so it is concise enough for search snippets.",
        "not_indexable_meta_description_too_long": "Shorten the meta description if this non-indexable page still appears in previews, internal search, or future indexing plans.",
        "indexable_meta_description_too_short": "Expand the meta description so it communicates the page value in search snippets.",
        "not_indexable_meta_description_too_short": "Expand the meta description if this non-indexable page still appears in previews, internal search, or future indexing plans.",
        "indexable_title_tag_missing_or_empty": "Add one descriptive title tag to the indexable page.",
        "not_indexable_title_tag_missing_or_empty": "Review whether the non-indexable page still needs a title tag; add one if it remains in crawl paths, previews, or future indexing plans.",
        "indexable_title_too_long": "Shorten the title tag so the main topic and differentiator fit cleanly in search results.",
        "indexable_title_too_short": "Expand the title tag with a clear topic and differentiator while keeping it concise.",
        "indexable_canonical_url_has_no_incoming_internal_links": "Add at least one crawlable internal link to this canonical URL from a relevant page.",
        "indexable_orphan_page_has_no_incoming_internal_links": "Add crawlable internal links to this orphan page from relevant navigation, hub, or content pages.",
        "not_indexable_orphan_page_has_no_incoming_internal_links": "Review whether this non-indexable page still needs internal discovery, then add links or keep it intentionally isolated.",
        "indexable_https_page_has_internal_links_to_http": "Update internal links on this HTTPS page so they point directly to HTTPS URLs.",
        "not_indexable_https_page_has_internal_links_to_http": "Update internal links on this non-indexable HTTPS page so they point directly to HTTPS URLs.",
        "indexable_http_page_has_internal_links_to_https": "Prefer serving and linking the HTTPS source page directly instead of relying on HTTP pages that link into HTTPS.",
        "not_indexable_http_page_has_internal_links_to_https": "Review HTTP non-indexable pages that link to HTTPS URLs and migrate or remove the HTTP source when it is obsolete.",
        "indexable_page_has_links_to_broken_page": "Update or remove internal links that point to broken 4XX/5XX URLs.",
        "not_indexable_page_has_links_to_broken_page": "Update or remove broken internal links from this non-indexable page if it remains part of the crawl path.",
        "indexable_page_has_links_to_redirect": "Update internal links to point directly at the final destination URL instead of the redirecting URL.",
        "not_indexable_page_has_links_to_redirect": "Update redirected internal links from this non-indexable page when it remains part of crawlable navigation or content.",
        "indexable_redirected_page_has_no_incoming_internal_links": "Add direct internal links to the final redirected URL or remove obsolete redirected URL references.",
        "not_indexable_redirected_page_has_no_incoming_internal_links": "Review whether this redirected non-indexable URL should receive direct internal links or be removed from crawl paths.",
        "indexable_page_has_no_outgoing_links": "Add relevant crawlable internal links from this page to useful destination pages.",
        "not_indexable_page_has_no_outgoing_links": "Review whether this non-indexable page should remain a crawl dead end; add relevant internal links if it remains useful.",
        "indexable_page_has_nofollow_incoming_internal_links_only": "Add at least one dofollow internal link to this page or remove nofollow from appropriate incoming internal links.",
        "not_indexable_page_has_nofollow_incoming_internal_links_only": "Review why this non-indexable page only receives nofollow internal links and make that treatment intentional.",
        "indexable_page_has_nofollow_and_dofollow_incoming_internal_links": "Review mixed incoming internal link directives and keep nofollow only where the link should not pass crawl signals.",
        "not_indexable_page_has_nofollow_and_dofollow_incoming_internal_links": "Review mixed incoming internal link directives to this non-indexable page and keep nofollow only where intentional.",
        "indexable_page_has_nofollow_outgoing_internal_links": "Review outgoing nofollow internal links and remove nofollow when internal destinations should receive crawl signals.",
        "not_indexable_page_has_nofollow_outgoing_internal_links": "Review outgoing nofollow internal links on this non-indexable page and keep them only where intentional.",
        "indexable_page_has_only_one_dofollow_incoming_internal_link": "Add more relevant dofollow internal links so this indexable page is not dependent on a single crawl path.",
        "not_indexable_page_has_only_one_dofollow_incoming_internal_link": "Review whether this non-indexable page needs more internal discovery or should remain lightly linked.",
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


def _is_redirected_fetch(row: dict) -> bool:
    requested_url = row.get("requested_url", "")
    redirect_target_url = row.get("redirect_target_url", "")
    return bool(requested_url and redirect_target_url and _normalize_url(requested_url) != _normalize_url(redirect_target_url))


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
            "serp_title": row.get("top_keyword_title") or row.get("serp_title") or "",
        }
    return out


def _title_fingerprint(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _titles_do_not_match(page_title: str, serp_title: str) -> bool:
    page = _title_fingerprint(page_title)
    serp = _title_fingerprint(serp_title)
    return bool(page and serp and page != serp)


def _has_high_ai_content(row: dict) -> bool:
    level = str(row.get("ai_content_level") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if level in {"high", "very_high"}:
        return True
    return max(_safe_float(row.get("ai_content_score")), _safe_float(row.get("ai_content_probability"))) >= 0.8


def _safe_float(value) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0
