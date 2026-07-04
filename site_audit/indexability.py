"""Indexability and crawlability funnel analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass
class IndexabilityReport:
    summary: dict
    status_counts: dict[str, int]
    skipped: list[dict]
    noindex_pages: list[dict]
    per_page: list[dict]
    issues: list[dict]
    issue_counts: dict[str, int]


ACTION_BY_ISSUE = {
    "noindex": "Confirm the page should be excluded from Google. If it is a strategic SEO page, remove the noindex directive and recrawl.",
    "unusable": "Make the page return readable HTML with a crawlable title and body, or remove it from SEO crawl surfaces.",
    "empty_embedding_text": "Add crawlable main content so the page can be evaluated for topical relevance and internal linking.",
    "canonical_duplicate": "Keep the canonical target in the SEO corpus and remove duplicate internal links/sitemap entries that point at this non-canonical URL.",
    "timed_out": "Reduce server latency or crawl-blocking behavior so the URL responds within the audit timeout.",
    "non_2xx_status": "Fix the HTTP response or update internal links and sitemaps so SEO pages resolve cleanly.",
    "skipped": "Review why extraction skipped this URL and decide whether it should be part of the SEO corpus.",
}


def _clean(value) -> str:
    if value is None:
        return ""
    return str(value)


def _row_issue_keys(row: dict) -> list[str]:
    keys: list[str] = []
    reason = _clean(row.get("reason"))
    status = _clean(row.get("status"))
    http_status = int(row.get("http_status") or 0)
    if reason == "noindex":
        keys.append("noindex")
    elif status != "analyzed":
        keys.append(reason or "skipped")
    if http_status and not 200 <= http_status < 400:
        keys.append("non_2xx_status")
    return list(dict.fromkeys(keys))


def _indexability_status(row: dict) -> str:
    if row.get("status") == "analyzed":
        return "indexable"
    if row.get("reason") == "noindex":
        return "noindex"
    return "not_analyzed"


def _recommended_action(issue_keys: list[str]) -> str:
    if not issue_keys:
        return "No action needed for indexability based on the current crawl."
    return ACTION_BY_ISSUE.get(issue_keys[0], ACTION_BY_ISSUE["skipped"])


def analyze(fetched: Iterable, extraction_rows: list[dict], analyzed_urls: set[str]) -> IndexabilityReport:
    fetched_list = list(fetched)
    status_counts = Counter(str(getattr(row, "status", 0) or 0) for row in fetched_list)
    fetched_by_url = {getattr(row, "url", ""): row for row in fetched_list}
    skipped = [row for row in extraction_rows if row.get("status") != "analyzed"]
    noindex_pages = [row for row in extraction_rows if row.get("reason") == "noindex"]
    extracted_ok = sum(1 for row in extraction_rows if row.get("status") in {"analyzed", "skipped"})
    per_page: list[dict] = []
    issues: list[dict] = []
    issue_counts: Counter[str] = Counter()

    for row in extraction_rows:
        fetched_row = fetched_by_url.get(row.get("url", ""))
        http_status = int(row.get("http_status") or getattr(fetched_row, "status", 0) or 0)
        normalized = {
            "url": row.get("url", ""),
            "title": row.get("title", ""),
            "http_status": http_status,
            "content_type": row.get("content_type") or getattr(fetched_row, "content_type", ""),
            "extraction_status": row.get("status", ""),
            "reason": row.get("reason", ""),
            "indexability_status": _indexability_status(row),
            "canonical_url": row.get("canonical_url", ""),
            "robots_content": row.get("robots_content", ""),
            "x_robots_tag": row.get("x_robots_tag") or getattr(fetched_row, "x_robots_tag", ""),
            "noindex_source": row.get("noindex_source") or row.get("source", ""),
            "nofollow": bool(row.get("nofollow")),
            "nofollow_source": row.get("nofollow_source", ""),
            "requested_url": row.get("requested_url", ""),
            "redirect_target_url": row.get("redirect_target_url", ""),
            "language": row.get("language", ""),
            "word_count": row.get("word_count", ""),
        }
        issue_keys = _row_issue_keys({**row, "http_status": http_status})
        normalized["issues"] = issue_keys
        normalized["recommended_action"] = _recommended_action(issue_keys)
        per_page.append(normalized)
        for key in issue_keys:
            issue_counts[key] += 1
            issues.append({
                "url": normalized["url"],
                "title": normalized["title"],
                "issue": key,
                "http_status": http_status,
                "indexability_status": normalized["indexability_status"],
                "reason": normalized["reason"],
                "canonical_url": normalized["canonical_url"],
                "noindex_source": normalized["noindex_source"],
                "nofollow": normalized["nofollow"],
                "nofollow_source": normalized["nofollow_source"],
                "requested_url": normalized["requested_url"],
                "redirect_target_url": normalized["redirect_target_url"],
                "x_robots_tag": normalized["x_robots_tag"],
                "recommended_action": ACTION_BY_ISSUE.get(key, ACTION_BY_ISSUE["skipped"]),
            })

    reason_counts = Counter(row.get("reason", "unknown") for row in skipped)
    total_fetched = len(fetched_list)
    analyzed_count = len(analyzed_urls)
    summary = {
        "fetched_pages": total_fetched,
        "extracted_pages": extracted_ok,
        "analyzed_pages": analyzed_count,
        "skipped_pages": len(skipped),
        "noindex_pages": len(noindex_pages),
        "indexable_share": analyzed_count / total_fetched if total_fetched else 0.0,
        "noindex_share": len(noindex_pages) / total_fetched if total_fetched else 0.0,
        "unusable_pages": reason_counts.get("unusable", 0),
        "empty_embedding_pages": reason_counts.get("empty_embedding_text", 0),
        "indexable_pages": analyzed_count,
        "non_indexable_pages": len(skipped),
        "pages_with_indexability_issues": sum(1 for row in per_page if row.get("issues")),
        "issue_count": sum(issue_counts.values()),
    }
    per_page.sort(key=lambda row: (0 if row.get("issues") else 1, row.get("url", "")))
    return IndexabilityReport(
        summary=summary,
        status_counts=dict(status_counts),
        skipped=skipped[:500],
        noindex_pages=noindex_pages[:500],
        per_page=per_page,
        issues=issues,
        issue_counts=dict(issue_counts),
    )


def to_payload(report: IndexabilityReport) -> dict:
    return {
        "summary": report.summary,
        "status_counts": report.status_counts,
        "skipped": report.skipped,
        "noindex_pages": report.noindex_pages,
        "per_page": report.per_page,
        "issues": report.issues,
        "issue_counts": report.issue_counts,
        "interpretation": {
            "why_it_matters": "Indexability issues explain which crawled URLs were excluded from the semantic SEO corpus and why Google may not be able to rank them.",
            "how_to_use": "Start with pages marked noindex or non-2xx. Keep intentional exclusions, but fix strategic landing pages before judging content gaps or competitor coverage.",
        },
    }
