"""Resource status analysis for page-linked assets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


_LARGE_CSS_BYTES = 150_000
_RESOURCE_ISSUE_SAMPLE_LIMIT = 300


@dataclass
class ResourceStatusReport:
    summary: dict
    issues_by_type: dict[str, int]
    per_page: list[dict]
    resources_with_issues: list[dict]


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _resource_type(value: str) -> str:
    resource_type = str(value or "").lower()
    if resource_type in {"script", "js"}:
        return "javascript"
    if resource_type in {"stylesheet", "style"}:
        return "css"
    return resource_type or "unknown"


def _cached_resource_meta(url: str, http_cache=None, meta_cache: dict[str, dict | None] | None = None) -> dict | None:
    if http_cache is None:
        return None
    if meta_cache is not None and url in meta_cache:
        return meta_cache[url]
    if hasattr(http_cache, "get_metadata"):
        meta = http_cache.get_metadata(url)
    else:
        cached = http_cache.get(url)
        meta = None
        if cached is not None:
            meta = {
                "status": getattr(cached, "status", ""),
                "body_size_bytes": len(getattr(cached, "body", b"") or b""),
            }
    if meta_cache is not None:
        meta_cache[url] = meta
    return meta


def _linked_resources(
    page_url: str,
    body: str,
    http_cache=None,
    meta_cache: dict[str, dict | None] | None = None,
) -> list[dict]:
    soup = BeautifulSoup(body or "", "html.parser")
    resources: list[dict] = []
    for tag in soup.find_all("script"):
        src = str(tag.get("src") or "").strip()
        if not src:
            continue
        absolute_url = urljoin(page_url, src)
        meta = _cached_resource_meta(absolute_url, http_cache, meta_cache)
        resources.append({
            "type": "javascript",
            "src": absolute_url,
            "http_status": (meta or {}).get("status", ""),
            "size_bytes": (meta or {}).get("body_size_bytes", ""),
        })
    for tag in soup.find_all("link"):
        rel = tag.get("rel") or []
        if isinstance(rel, str):
            rel = rel.split()
        if "stylesheet" not in {str(item).lower() for item in rel}:
            continue
        href = str(tag.get("href") or "").strip()
        if not href:
            continue
        absolute_url = urljoin(page_url, href)
        meta = _cached_resource_meta(absolute_url, http_cache, meta_cache)
        resources.append({
            "type": "css",
            "src": absolute_url,
            "http_status": (meta or {}).get("status", ""),
            "size_bytes": (meta or {}).get("body_size_bytes", ""),
        })
    return resources


def _resource_items(fetch, http_cache=None, meta_cache: dict[str, dict | None] | None = None) -> list[dict]:
    explicit = getattr(fetch, "resource_items", None)
    if explicit is not None:
        return [dict(item) for item in explicit]
    return _linked_resources(
        getattr(fetch, "url", "") or "",
        getattr(fetch, "body", "") or "",
        http_cache,
        meta_cache,
    )


def _resource_issues(item: dict, page_url: str = "") -> list[str]:
    issues: list[str] = []
    resource_type = _resource_type(item.get("type") or item.get("resource_type") or "")
    status = _safe_int(item.get("http_status", item.get("status", 0)))
    size_bytes = _safe_int(item.get("size_bytes", item.get("file_size_bytes", item.get("content_length_bytes", 0))))
    redirect_target_url = str(item.get("redirect_target_url") or "")
    if resource_type in {"javascript", "script", "js"} and (item.get("broken") or status >= 400):
        issues.append("javascript_broken")
    if resource_type == "css" and (item.get("broken") or status >= 400):
        issues.append("css_broken")
    if resource_type == "css" and size_bytes > _LARGE_CSS_BYTES:
        issues.append("css_file_size_too_large")
    src = str(item.get("src") or item.get("url") or "")
    if resource_type == "javascript" and urlparse(page_url or "").scheme.lower() == "https" and urlparse(src).scheme.lower() == "http":
        issues.append("https_page_links_to_http_javascript")
    if resource_type == "css" and urlparse(page_url or "").scheme.lower() == "https" and urlparse(src).scheme.lower() == "http":
        issues.append("https_page_links_to_http_css")
    if resource_type == "javascript" and (item.get("redirected") or redirect_target_url or 300 <= status < 400):
        issues.append("javascript_redirects")
    if resource_type == "css" and (item.get("redirected") or redirect_target_url or 300 <= status < 400):
        issues.append("css_redirects")
    return issues


def analyze(
    fetched_pages: Iterable,
    *,
    http_cache=None,
    resource_issue_sample_limit: int = _RESOURCE_ISSUE_SAMPLE_LIMIT,
) -> ResourceStatusReport:
    fetched_list = list(fetched_pages)
    issues_by_type: Counter[str] = Counter()
    resource_type_counts: Counter[str] = Counter()
    per_page: list[dict] = []
    resources_with_issues: list[dict] = []
    pages_with_issues = 0
    resource_meta_cache: dict[str, dict | None] = {}

    for fetch in fetched_list:
        page_url = getattr(fetch, "url", "") or ""
        page_title = getattr(fetch, "title", "") or ""
        resources = _resource_items(fetch, http_cache, resource_meta_cache)
        page_issues: Counter[str] = Counter()
        page_resource_type_counts: Counter[str] = Counter()
        for idx, item in enumerate(resources):
            resource_type = _resource_type(item.get("type") or item.get("resource_type") or "")
            resource_type_counts[resource_type] += 1
            page_resource_type_counts[resource_type] += 1
            issues = _resource_issues(item, page_url)
            if not issues:
                continue
            page_issues.update(issues)
            issues_by_type.update(issues)
            if len(resources_with_issues) < resource_issue_sample_limit:
                resources_with_issues.append({
                    "url": page_url,
                    "title": page_title,
                    "index": idx,
                    "type": resource_type,
                    "src": item.get("src") or item.get("url") or "",
                    "http_status": item.get("http_status", item.get("status", "")),
                    "redirect_target_url": item.get("redirect_target_url", ""),
                    "size_bytes": item.get("size_bytes", item.get("file_size_bytes", item.get("content_length_bytes", ""))),
                    "issues": issues,
                })
        if page_issues:
            pages_with_issues += 1
        per_page.append({
            "url": page_url,
            "title": page_title,
            "resource_count": len(resources),
            "javascript_count": page_resource_type_counts.get("javascript", 0),
            "css_count": page_resource_type_counts.get("css", 0),
            "issues": dict(page_issues),
            "issue_count": sum(page_issues.values()),
        })

    total_pages = len(fetched_list)
    summary = {
        "total_pages": total_pages,
        "pages_with_issues": pages_with_issues,
        "issue_share": pages_with_issues / total_pages if total_pages else 0.0,
        "total_resources": sum(resource_type_counts.values()),
        "total_javascript": resource_type_counts.get("javascript", 0),
        "total_css": resource_type_counts.get("css", 0),
        "broken_javascript": issues_by_type.get("javascript_broken", 0),
        "broken_css": issues_by_type.get("css_broken", 0),
        "large_css": issues_by_type.get("css_file_size_too_large", 0),
        "https_pages_linking_to_http_javascript": issues_by_type.get("https_page_links_to_http_javascript", 0),
        "https_pages_linking_to_http_css": issues_by_type.get("https_page_links_to_http_css", 0),
        "redirected_javascript": issues_by_type.get("javascript_redirects", 0),
        "redirected_css": issues_by_type.get("css_redirects", 0),
        "unique_resource_urls_checked": len(resource_meta_cache),
        "resource_issue_sample_limit": resource_issue_sample_limit,
    }
    per_page.sort(key=lambda row: (-row["issue_count"], -row["resource_count"], row["url"]))
    return ResourceStatusReport(
        summary=summary,
        issues_by_type=dict(issues_by_type),
        per_page=per_page,
        resources_with_issues=resources_with_issues,
    )


def to_payload(report: ResourceStatusReport) -> dict:
    return {
        "summary": report.summary,
        "issues_by_type": report.issues_by_type,
        "per_page": report.per_page,
        "resources_with_issues": report.resources_with_issues,
    }
