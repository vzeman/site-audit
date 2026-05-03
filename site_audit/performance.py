"""Lightweight offline performance signals.

This module intentionally avoids browser automation or network fetches. It
uses only the HTML responses already collected by the crawler, so the output
is a cheap set of page-weight and render-blocking heuristics rather than lab
performance metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Iterable

from bs4 import BeautifulSoup


_IMAGE_WEIGHT = 120_000
_SCRIPT_WEIGHT = 35_000
_STYLE_WEIGHT = 20_000
_FONT_WEIGHT = 40_000
_PRELOAD_WEIGHT = 20_000


@dataclass
class PerformanceReport:
    summary: dict
    buckets: dict
    per_page: list[dict]
    top_heavy_pages: list[dict]
    render_blocking_pages: list[dict]


def _html_weight(fetch) -> int:
    byte_count = int(getattr(fetch, "content_length_bytes", 0) or 0)
    if byte_count:
        return byte_count
    return len((getattr(fetch, "body", "") or "").encode("utf-8"))


def _has_rel(tag, value: str) -> bool:
    rel = tag.get("rel") or []
    if isinstance(rel, str):
        rel = rel.split()
    return value.lower() in {str(item).lower() for item in rel}


def _is_font_link(tag) -> bool:
    href = str(tag.get("href") or "").lower()
    as_attr = str(tag.get("as") or "").lower()
    type_attr = str(tag.get("type") or "").lower()
    return as_attr == "font" or "font/" in type_attr or any(ext in href for ext in (".woff", ".woff2", ".ttf", ".otf", ".eot"))


def _is_blocking_stylesheet(tag) -> bool:
    if not _has_rel(tag, "stylesheet") or tag.has_attr("disabled"):
        return False
    media = str(tag.get("media") or "").strip().lower()
    return not media or media in {"all", "screen"}


def _is_blocking_script(tag) -> bool:
    if not tag.get("src"):
        return False
    script_type = str(tag.get("type") or "").strip().lower()
    if script_type == "module":
        return False
    return not (tag.has_attr("async") or tag.has_attr("defer"))


def _bucket(estimated_weight: int) -> str:
    if estimated_weight < 500_000:
        return "light"
    if estimated_weight < 1_500_000:
        return "moderate"
    if estimated_weight < 3_000_000:
        return "heavy"
    return "very_heavy"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _row(fetch) -> dict:
    body = getattr(fetch, "body", "") or ""
    soup = BeautifulSoup(body, "html.parser")
    scripts = soup.find_all("script")
    styles = soup.find_all("style")
    links = soup.find_all("link")
    images = soup.find_all("img")

    script_count = len(scripts)
    external_script_count = sum(1 for tag in scripts if tag.get("src"))
    inline_script_count = script_count - external_script_count
    stylesheet_count = sum(1 for tag in links if _has_rel(tag, "stylesheet"))
    inline_style_count = len(styles)
    style_attr_count = len(soup.select("[style]"))
    font_count = sum(1 for tag in links if _is_font_link(tag))
    preload_count = sum(1 for tag in links if _has_rel(tag, "preload"))
    blocking_css_count = sum(1 for tag in links if _is_blocking_stylesheet(tag))
    blocking_script_count = sum(1 for tag in scripts if _is_blocking_script(tag))
    html_weight_bytes = _html_weight(fetch)
    estimated_weight_bytes = (
        html_weight_bytes
        + len(images) * _IMAGE_WEIGHT
        + external_script_count * _SCRIPT_WEIGHT
        + stylesheet_count * _STYLE_WEIGHT
        + font_count * _FONT_WEIGHT
        + preload_count * _PRELOAD_WEIGHT
    )

    return {
        "url": getattr(fetch, "url", ""),
        "status": int(getattr(fetch, "status", 0) or 0),
        "content_type": getattr(fetch, "content_type", "") or "",
        "content_size_bytes": int(getattr(fetch, "content_length_bytes", 0) or html_weight_bytes),
        "html_weight_bytes": html_weight_bytes,
        "estimated_weight_bytes": estimated_weight_bytes,
        "weight_bucket": _bucket(estimated_weight_bytes),
        "image_count": len(images),
        "script_count": script_count,
        "external_script_count": external_script_count,
        "inline_script_count": inline_script_count,
        "stylesheet_count": stylesheet_count,
        "inline_style_count": inline_style_count,
        "style_attr_count": style_attr_count,
        "font_count": font_count,
        "preload_count": preload_count,
        "resource_tag_count": len(images) + script_count + stylesheet_count + font_count + preload_count,
        "render_blocking_css_count": blocking_css_count,
        "render_blocking_script_count": blocking_script_count,
        "render_blocking_count": blocking_css_count + blocking_script_count,
    }


def analyze(fetched_pages: Iterable) -> PerformanceReport:
    rows = [_row(fetch) for fetch in fetched_pages]
    bucket_counts = Counter(row["weight_bucket"] for row in rows)
    statuses = Counter(str(row["status"]) for row in rows)
    html_weights = [float(row["html_weight_bytes"]) for row in rows]
    estimated_weights = [float(row["estimated_weight_bytes"]) for row in rows]
    total_pages = len(rows)
    pages_with_blocking = sum(1 for row in rows if row["render_blocking_count"] > 0)
    heavy_pages = sum(1 for row in rows if row["weight_bucket"] in {"heavy", "very_heavy"})

    summary = {
        "total_pages": total_pages,
        "status_counts": dict(sorted(statuses.items())),
        "total_content_size_bytes": sum(int(row["content_size_bytes"]) for row in rows),
        "total_html_weight_bytes": sum(int(row["html_weight_bytes"]) for row in rows),
        "median_html_weight_bytes": int(median(html_weights)) if html_weights else 0,
        "p90_html_weight_bytes": int(_percentile(html_weights, 0.9)),
        "median_estimated_weight_bytes": int(median(estimated_weights)) if estimated_weights else 0,
        "p90_estimated_weight_bytes": int(_percentile(estimated_weights, 0.9)),
        "avg_resource_tags_per_page": round(sum(row["resource_tag_count"] for row in rows) / total_pages, 2) if total_pages else 0.0,
        "total_images": sum(row["image_count"] for row in rows),
        "total_scripts": sum(row["script_count"] for row in rows),
        "total_stylesheets": sum(row["stylesheet_count"] for row in rows),
        "total_fonts": sum(row["font_count"] for row in rows),
        "total_preloads": sum(row["preload_count"] for row in rows),
        "pages_with_render_blocking": pages_with_blocking,
        "render_blocking_share": pages_with_blocking / total_pages if total_pages else 0.0,
        "heavy_pages": heavy_pages,
        "heavy_page_share": heavy_pages / total_pages if total_pages else 0.0,
    }

    return PerformanceReport(
        summary=summary,
        buckets={bucket: int(bucket_counts.get(bucket, 0)) for bucket in ("light", "moderate", "heavy", "very_heavy")},
        per_page=sorted(rows, key=lambda row: (-row["estimated_weight_bytes"], row["url"])),
        top_heavy_pages=sorted(rows, key=lambda row: (-row["estimated_weight_bytes"], row["url"]))[:25],
        render_blocking_pages=[
            row for row in sorted(rows, key=lambda row: (-row["render_blocking_count"], -row["estimated_weight_bytes"], row["url"]))
            if row["render_blocking_count"] > 0
        ][:25],
    )


def to_payload(report: PerformanceReport) -> dict:
    return {
        "summary": report.summary,
        "buckets": report.buckets,
        "top_heavy_pages": report.top_heavy_pages,
        "render_blocking_pages": report.render_blocking_pages,
        "per_page": report.per_page,
    }
