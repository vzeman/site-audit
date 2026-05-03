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


def analyze(fetched: Iterable, extraction_rows: list[dict], analyzed_urls: set[str]) -> IndexabilityReport:
    fetched_list = list(fetched)
    status_counts = Counter(str(getattr(row, "status", 0) or 0) for row in fetched_list)
    skipped = [row for row in extraction_rows if row.get("status") != "analyzed"]
    noindex_pages = [row for row in extraction_rows if row.get("reason") == "noindex"]
    extracted_ok = sum(1 for row in extraction_rows if row.get("status") in {"analyzed", "skipped"})

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
    }
    return IndexabilityReport(
        summary=summary,
        status_counts=dict(status_counts),
        skipped=skipped[:500],
        noindex_pages=noindex_pages[:500],
    )


def to_payload(report: IndexabilityReport) -> dict:
    return {
        "summary": report.summary,
        "status_counts": report.status_counts,
        "skipped": report.skipped,
        "noindex_pages": report.noindex_pages,
    }
