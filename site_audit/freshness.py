"""Content freshness analysis.

Turns extractor date hints into page-level freshness diagnostics: how many
pages expose dates, which pages are stale, and which content lacks any usable
publication/update signal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from statistics import median
from typing import Iterable, Optional

from .extractor import ExtractedPage


@dataclass
class FreshnessReport:
    summary: dict
    buckets: dict[str, int]
    stale_pages: list[dict]
    per_page: list[dict]


_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
)


def _parse_date(value: str) -> Optional[date]:
    raw = (value or "").strip()
    if not raw:
        return None
    if len(raw) >= 10 and raw[4:5] == "-" and raw[7:8] == "-":
        raw = raw[:10]
    for fmt in _FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    try:
        return parsedate_to_datetime(raw).date()
    except Exception:
        return None


def _best_candidate(page: ExtractedPage) -> tuple[Optional[date], str, str]:
    candidates = list(getattr(page, "date_candidates", []) or [])
    if not candidates:
        for value, kind, source in (
            (getattr(page, "date_modified", ""), "modified", "field:date_modified"),
            (getattr(page, "date_published", ""), "published", "field:date_published"),
        ):
            if value:
                candidates.append({"date": value, "kind": kind, "source": source})

    parsed: list[tuple[date, str, str]] = []
    for candidate in candidates:
        parsed_date = _parse_date(str(candidate.get("date", "")))
        if parsed_date is None:
            continue
        parsed.append((
            parsed_date,
            str(candidate.get("kind") or "visible"),
            str(candidate.get("source") or ""),
        ))
    if not parsed:
        return None, "", ""

    modified = [item for item in parsed if item[1] == "modified"]
    if modified:
        return max(modified, key=lambda item: item[0])
    return max(parsed, key=lambda item: item[0])


def _bucket(age_days: Optional[int], stale_days: int, very_stale_days: int) -> str:
    if age_days is None:
        return "unknown"
    if age_days < 0:
        return "future"
    if age_days <= 90:
        return "fresh"
    if age_days <= stale_days:
        return "aging"
    if age_days <= very_stale_days:
        return "stale"
    return "very_stale"


def analyze(
    pages: Iterable[ExtractedPage],
    today: Optional[date] = None,
    stale_days: int = 365,
    very_stale_days: int = 730,
) -> FreshnessReport:
    page_list = list(pages)
    today_date = today or date.today()
    bucket_counts: Counter[str] = Counter()
    per_page: list[dict] = []
    ages: list[int] = []
    dated_values: list[date] = []

    for page in page_list:
        best_date, date_kind, date_source = _best_candidate(page)
        age_days: Optional[int] = None
        issues: list[str] = []
        if best_date is None:
            issues.append("missing_date")
        else:
            age_days = (today_date - best_date).days
            dated_values.append(best_date)
            if age_days >= 0:
                ages.append(age_days)
            if age_days < 0:
                issues.append("future_date")
            elif age_days > very_stale_days:
                issues.append("very_stale")
            elif age_days > stale_days:
                issues.append("stale")

        bucket = _bucket(age_days, stale_days, very_stale_days)
        bucket_counts[bucket] += 1
        per_page.append({
            "url": page.url,
            "title": page.title,
            "date": best_date.isoformat() if best_date else "",
            "date_kind": date_kind,
            "date_source": date_source,
            "age_days": age_days,
            "bucket": bucket,
            "issues": issues,
        })

    total = len(page_list)
    stale_pages = [
        row for row in per_page
        if row["bucket"] in {"stale", "very_stale", "unknown", "future"}
    ]
    stale_pages.sort(key=lambda row: (
        row["bucket"] != "unknown",
        -(row["age_days"] if row["age_days"] is not None else 10**9),
        row["url"],
    ))
    per_page.sort(key=lambda row: (
        row["bucket"] != "unknown",
        -(row["age_days"] if row["age_days"] is not None else 10**9),
        row["url"],
    ))

    pages_with_date = len(dated_values)
    stale_count = bucket_counts["stale"] + bucket_counts["very_stale"]
    summary = {
        "total_pages": total,
        "pages_with_date": pages_with_date,
        "date_coverage": pages_with_date / total if total else 0.0,
        "missing_dates": bucket_counts["unknown"],
        "pages_stale": stale_count,
        "stale_share": stale_count / total if total else 0.0,
        "pages_very_stale": bucket_counts["very_stale"],
        "future_dates": bucket_counts["future"],
        "median_age_days": int(median(ages)) if ages else None,
        "newest_date": max(dated_values).isoformat() if dated_values else "",
        "oldest_date": min(dated_values).isoformat() if dated_values else "",
        "stale_days": stale_days,
        "very_stale_days": very_stale_days,
    }
    return FreshnessReport(
        summary=summary,
        buckets=dict(bucket_counts),
        stale_pages=stale_pages[:200],
        per_page=per_page,
    )


def to_payload(report: FreshnessReport) -> dict:
    return {
        "summary": report.summary,
        "buckets": report.buckets,
        "stale_pages": report.stale_pages,
        "per_page": report.per_page,
    }
