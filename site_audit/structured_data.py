"""Structured-data health analysis.

The extractor keeps lightweight JSON-LD diagnostics per page. This module
turns those page-level signals into a report payload and comparison metrics:
coverage, invalid blocks, type mix, and common missing recommended fields.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .extractor import ExtractedPage


_RECOMMENDED_PROPS: dict[str, set[str]] = {
    "Article": {"headline", "datePublished", "author"},
    "NewsArticle": {"headline", "datePublished", "author"},
    "BlogPosting": {"headline", "datePublished", "author"},
    "FAQPage": {"mainEntity"},
    "QAPage": {"mainEntity"},
    "HowTo": {"name", "step"},
    "Product": {"name"},
    "Organization": {"name", "url"},
    "LocalBusiness": {"name", "address"},
    "BreadcrumbList": {"itemListElement"},
    "VideoObject": {"name", "uploadDate"},
}


@dataclass
class StructuredDataReport:
    summary: dict
    top_types: list[dict]
    invalid_blocks: list[dict]
    missing_recommended: list[dict]
    per_page: list[dict]


def _top_level_keys(block: dict) -> set[str]:
    if not isinstance(block, dict):
        return set()
    return set(block.get("keys") or [])


def _missing_recommended(types: list[str], keys: set[str]) -> list[dict]:
    missing: list[dict] = []
    for schema_type in types:
        wanted = _RECOMMENDED_PROPS.get(schema_type)
        if not wanted:
            continue
        absent = sorted(prop for prop in wanted if prop not in keys)
        if absent:
            missing.append({"type": schema_type, "missing": absent})
    return missing


def analyze(pages: Iterable[ExtractedPage]) -> StructuredDataReport:
    page_list = list(pages)
    type_counts: Counter[str] = Counter()
    invalid_blocks: list[dict] = []
    missing_rows: list[dict] = []
    per_page: list[dict] = []
    pages_with_schema = 0
    pages_with_invalid = 0
    valid_blocks = 0
    invalid_count = 0

    for page in page_list:
        blocks = list(page.schema_blocks or [])
        types = sorted(set(page.schema_types or []))
        valid_page_blocks = [block for block in blocks if block.get("valid")]
        invalid_page_blocks = [block for block in blocks if not block.get("valid")]
        if types or valid_page_blocks:
            pages_with_schema += 1
        if invalid_page_blocks:
            pages_with_invalid += 1
        valid_blocks += len(valid_page_blocks)
        invalid_count += len(invalid_page_blocks)
        type_counts.update(types)

        keys: set[str] = set()
        for block in valid_page_blocks:
            keys.update(_top_level_keys(block))
        missing = _missing_recommended(types, keys)
        if missing:
            missing_rows.append({
                "url": page.url,
                "title": page.title,
                "types": types,
                "missing": missing,
            })
        for block in invalid_page_blocks:
            invalid_blocks.append({
                "url": page.url,
                "title": page.title,
                "format": block.get("format", "json-ld"),
                "error": block.get("error", ""),
            })

        per_page.append({
            "url": page.url,
            "title": page.title,
            "types": types,
            "valid_blocks": len(valid_page_blocks),
            "invalid_blocks": len(invalid_page_blocks),
            "missing_recommended": missing,
        })

    total_pages = len(page_list)
    summary = {
        "total_pages": total_pages,
        "pages_with_schema": pages_with_schema,
        "schema_coverage": pages_with_schema / total_pages if total_pages else 0.0,
        "valid_jsonld_blocks": valid_blocks,
        "invalid_jsonld_blocks": invalid_count,
        "pages_with_invalid_jsonld": pages_with_invalid,
        "schema_type_count": len(type_counts),
        "pages_missing_schema": max(0, total_pages - pages_with_schema),
    }
    top_types = [
        {"type": schema_type, "pages": count}
        for schema_type, count in type_counts.most_common()
    ]
    per_page.sort(key=lambda row: (-row["invalid_blocks"], not row["missing_recommended"], row["url"]))
    return StructuredDataReport(
        summary=summary,
        top_types=top_types,
        invalid_blocks=invalid_blocks[:200],
        missing_recommended=missing_rows[:200],
        per_page=per_page,
    )


def to_payload(report: StructuredDataReport) -> dict:
    return {
        "summary": report.summary,
        "top_types": report.top_types,
        "invalid_blocks": report.invalid_blocks,
        "missing_recommended": report.missing_recommended,
        "per_page": report.per_page,
    }
