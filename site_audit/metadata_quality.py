"""SERP metadata quality analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from .extractor import ExtractedPage


@dataclass
class MetadataQualityReport:
    summary: dict
    issues_by_type: dict[str, int]
    per_page: list[dict]


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _same_host(url: str, canonical_url: str) -> bool:
    if not canonical_url:
        return True
    parsed = urlparse(url)
    canonical = urlparse(canonical_url)
    if not canonical.netloc:
        return True
    left = parsed.netloc.lower().removeprefix("www.")
    right = canonical.netloc.lower().removeprefix("www.")
    return left == right


def _title_issue(title: str) -> str:
    length = len(title or "")
    if length == 0:
        return "missing_title"
    if length < 20:
        return "short_title"
    if length > 65:
        return "long_title"
    return ""


def _description_issue(description: str) -> str:
    length = len(description or "")
    if length == 0:
        return "missing_description"
    if length < 50:
        return "short_description"
    if length > 160:
        return "long_description"
    return ""


def _open_graph_values(page: ExtractedPage) -> dict[str, str]:
    return {
        "og_title": page.og_title or "",
        "og_description": page.og_description or "",
        "og_image": page.og_image or "",
        "og_url": getattr(page, "og_url", "") or "",
    }


def _open_graph_missing_fields(page: ExtractedPage) -> list[str]:
    values = _open_graph_values(page)
    required = ("og_title", "og_description")
    return [field for field in required if not values[field]]


def _has_open_graph_tags(page: ExtractedPage) -> bool:
    return any(_open_graph_values(page).values())


def _twitter_values(page: ExtractedPage) -> dict[str, str]:
    return {
        "twitter_card": page.twitter_card or "",
        "twitter_title": page.twitter_title or "",
        "twitter_description": page.twitter_description or "",
    }


def _twitter_missing_fields(page: ExtractedPage) -> list[str]:
    values = _twitter_values(page)
    required = ("twitter_card", "twitter_title", "twitter_description")
    return [field for field in required if not values[field]]


def _has_twitter_tags(page: ExtractedPage) -> bool:
    return any(_twitter_values(page).values())


def analyze(pages: Iterable[ExtractedPage]) -> MetadataQualityReport:
    page_list = list(pages)
    title_counts = Counter(_norm(page.title) for page in page_list if _norm(page.title))
    desc_counts = Counter(_norm(page.description) for page in page_list if _norm(page.description))
    issues_by_type: Counter[str] = Counter()
    rows: list[dict] = []

    for page in page_list:
        issues: list[str] = []
        for issue in (_title_issue(page.title), _description_issue(page.description)):
            if issue:
                issues.append(issue)
        if title_counts[_norm(page.title)] > 1:
            issues.append("duplicate_title")
        if desc_counts[_norm(page.description)] > 1:
            issues.append("duplicate_description")
        if not page.canonical_url:
            issues.append("missing_canonical")
        elif not _same_host(page.url, page.canonical_url):
            issues.append("canonical_external_host")
        og_missing_fields = _open_graph_missing_fields(page)
        if _has_open_graph_tags(page) and og_missing_fields:
            issues.append("incomplete_open_graph")
        twitter_missing_fields = _twitter_missing_fields(page)
        if _has_twitter_tags(page) and twitter_missing_fields:
            issues.append("incomplete_twitter_card")
        if not _has_twitter_tags(page):
            issues.append("missing_twitter_card")
        if page.noindex:
            issues.append("noindex")

        issues_by_type.update(issues)
        rows.append({
            "url": page.url,
            "title": page.title,
            "title_length": len(page.title or ""),
            "description": page.description,
            "description_length": len(page.description or ""),
            "canonical_url": page.canonical_url,
            "robots_content": page.robots_content,
            "nofollow": page.nofollow,
            "nofollow_source": page.nofollow_source,
            "meta_refresh_redirect": page.meta_refresh_redirect,
            "meta_refresh_target_url": page.meta_refresh_target_url,
            "title_tag_count": page.title_tag_count,
            "meta_description_tag_count": page.meta_description_tag_count,
            **_open_graph_values(page),
            "og_tag_count": sum(1 for value in _open_graph_values(page).values() if value),
            "og_missing_fields": og_missing_fields,
            "og_complete": not og_missing_fields,
            **_twitter_values(page),
            "twitter_tag_count": sum(1 for value in _twitter_values(page).values() if value),
            "twitter_missing_fields": twitter_missing_fields,
            "twitter_complete": not twitter_missing_fields,
            "issues": issues,
        })

    total = len(page_list)
    issue_pages = sum(1 for row in rows if row["issues"])
    summary = {
        "total_pages": total,
        "pages_with_issues": issue_pages,
        "issue_share": issue_pages / total if total else 0.0,
        "missing_title": issues_by_type.get("missing_title", 0),
        "missing_description": issues_by_type.get("missing_description", 0),
        "duplicate_title_pages": issues_by_type.get("duplicate_title", 0),
        "duplicate_description_pages": issues_by_type.get("duplicate_description", 0),
        "missing_canonical": issues_by_type.get("missing_canonical", 0),
        "canonical_external_host": issues_by_type.get("canonical_external_host", 0),
        "incomplete_open_graph": issues_by_type.get("incomplete_open_graph", 0),
        "incomplete_twitter_card": issues_by_type.get("incomplete_twitter_card", 0),
        "missing_twitter_card": issues_by_type.get("missing_twitter_card", 0),
    }
    rows.sort(key=lambda row: (-len(row["issues"]), row["url"]))
    return MetadataQualityReport(
        summary=summary,
        issues_by_type=dict(issues_by_type),
        per_page=rows,
    )


def to_payload(report: MetadataQualityReport) -> dict:
    return {
        "summary": report.summary,
        "issues_by_type": report.issues_by_type,
        "per_page": report.per_page,
    }
