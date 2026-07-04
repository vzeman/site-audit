"""Lightweight offline performance signals.

This module intentionally avoids browser automation or network fetches. It
uses only the HTML responses already collected by the crawler, so the output
is a cheap set of page-weight and render-blocking heuristics rather than lab
performance metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from statistics import median
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


_IMAGE_WEIGHT = 120_000
_SCRIPT_WEIGHT = 35_000
_STYLE_WEIGHT = 20_000
_FONT_WEIGHT = 40_000
_PRELOAD_WEIGHT = 20_000
_MOBILE_VIEWPORT_WIDTH = 390
_MIN_READABLE_FONT_PX = 12.0
_CSS_URL_RE = re.compile(r"url\(\s*['\"]?(http://[^)'\"\s]+)", re.I)
_FIXED_WIDTH_RE = re.compile(r"(?<![\w-])(width|min-width)\s*:\s*(\d+(?:\.\d+)?)px\b", re.I)
_FONT_SIZE_RE = re.compile(r"(?<![\w-])font-size\s*:\s*(\d+(?:\.\d+)?)(px|em|rem|%)(?![\w-])", re.I)
_WIDTH_ATTR_RE = re.compile(r"\s*(\d+(?:\.\d+)?)(?:px)?\s*", re.I)
_RESOURCE_LINK_RELS = {
    "apple-touch-icon",
    "icon",
    "manifest",
    "modulepreload",
    "preload",
    "stylesheet",
}


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


def _srcset_urls(value: str) -> list[str]:
    urls: list[str] = []
    for candidate in value.split(","):
        url = candidate.strip().split(" ", 1)[0].strip()
        if url:
            urls.append(url)
    return urls


def _resource_urls(soup: BeautifulSoup) -> list[str]:
    urls: list[str] = []
    attr_specs = [
        ("img", "src"),
        ("script", "src"),
        ("iframe", "src"),
        ("frame", "src"),
        ("embed", "src"),
        ("audio", "src"),
        ("video", "src"),
        ("video", "poster"),
        ("source", "src"),
        ("track", "src"),
        ("object", "data"),
        ("input", "src"),
    ]
    for tag_name, attr in attr_specs:
        for tag in soup.find_all(tag_name):
            value = str(tag.get(attr) or "").strip()
            if value:
                urls.append(value)
    for tag_name in ("img", "source"):
        for tag in soup.find_all(tag_name):
            urls.extend(_srcset_urls(str(tag.get("srcset") or "")))
    for tag in soup.find_all("link"):
        rel_attr = tag.get("rel") or []
        if isinstance(rel_attr, str):
            rel_attr = rel_attr.split()
        rel = {str(item).lower() for item in rel_attr}
        if rel.intersection(_RESOURCE_LINK_RELS):
            href = str(tag.get("href") or "").strip()
            if href:
                urls.append(href)
    for tag in soup.find_all("style"):
        urls.extend(_CSS_URL_RE.findall(tag.get_text(" ") or ""))
    for tag in soup.select("[style]"):
        urls.extend(_CSS_URL_RE.findall(str(tag.get("style") or "")))
    return urls


def _mixed_content_urls(page_url: str, soup: BeautifulSoup) -> list[str]:
    if urlparse(page_url).scheme.lower() != "https":
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for raw_url in _resource_urls(soup):
        absolute = urljoin(page_url, raw_url)
        if urlparse(absolute).scheme.lower() != "http" or absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def _oversized_fixed_widths(soup: BeautifulSoup) -> list[dict]:
    issues: list[dict] = []
    for tag in soup.select("[style]"):
        for prop, value in _FIXED_WIDTH_RE.findall(str(tag.get("style") or "")):
            width = int(float(value))
            if width > _MOBILE_VIEWPORT_WIDTH:
                issues.append({"source": "inline_style", "tag": tag.name, "property": prop.lower(), "width_px": width})
    for tag in soup.find_all("style"):
        for prop, value in _FIXED_WIDTH_RE.findall(tag.get_text(" ") or ""):
            width = int(float(value))
            if width > _MOBILE_VIEWPORT_WIDTH:
                issues.append({"source": "style_block", "tag": "style", "property": prop.lower(), "width_px": width})
    for tag in soup.find_all(attrs={"width": True}):
        match = _WIDTH_ATTR_RE.fullmatch(str(tag.get("width") or ""))
        if not match:
            continue
        width = int(float(match.group(1)))
        if width > _MOBILE_VIEWPORT_WIDTH:
            issues.append({"source": "width_attribute", "tag": tag.name, "property": "width", "width_px": width})
    return issues[:25]


def _plugin_elements(soup: BeautifulSoup) -> list[dict]:
    plugins: list[dict] = []
    for tag in soup.find_all(["applet", "embed", "object"]):
        source = str(tag.get("src") or tag.get("data") or tag.get("code") or "").strip()
        plugins.append({
            "tag": tag.name,
            "type": str(tag.get("type") or "").strip(),
            "source": source,
        })
    return plugins[:25]


def _font_size_px(value: str, unit: str) -> float:
    size = float(value)
    unit = unit.lower()
    if unit == "px":
        return size
    if unit in {"em", "rem"}:
        return size * 16.0
    if unit == "%":
        return size * 16.0 / 100.0
    return 0.0


def _small_font_sizes(soup: BeautifulSoup) -> list[dict]:
    issues: list[dict] = []
    for tag in soup.select("[style]"):
        for value, unit in _FONT_SIZE_RE.findall(str(tag.get("style") or "")):
            px = _font_size_px(value, unit)
            if 0 < px < _MIN_READABLE_FONT_PX:
                issues.append({"source": "inline_style", "tag": tag.name, "font_size": f"{value}{unit}", "font_size_px": round(px, 2)})
    for tag in soup.find_all("style"):
        for value, unit in _FONT_SIZE_RE.findall(tag.get_text(" ") or ""):
            px = _font_size_px(value, unit)
            if 0 < px < _MIN_READABLE_FONT_PX:
                issues.append({"source": "style_block", "tag": "style", "font_size": f"{value}{unit}", "font_size_px": round(px, 2)})
    return issues[:25]


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
    page_url = getattr(fetch, "url", "") or ""
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
    mixed_content_urls = _mixed_content_urls(page_url, soup)
    content_sizing_issues = _oversized_fixed_widths(soup)
    plugin_elements = _plugin_elements(soup)
    small_font_issues = _small_font_sizes(soup)
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
        "url": page_url,
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
        "mixed_content_url_count": len(mixed_content_urls),
        "mixed_content_urls": mixed_content_urls[:25],
        "content_sized_correctly": not bool(content_sizing_issues),
        "content_width_exceeds_viewport": bool(content_sizing_issues),
        "max_fixed_width_px": max((issue["width_px"] for issue in content_sizing_issues), default=0),
        "content_sizing_issues": content_sizing_issues,
        "plugin_element_count": len(plugin_elements),
        "plugin_elements": plugin_elements,
        "small_font_size_count": len(small_font_issues),
        "small_font_size_issues": small_font_issues,
    }


def analyze(fetched_pages: Iterable) -> PerformanceReport:
    rows = [_row(fetch) for fetch in fetched_pages]
    bucket_counts = Counter(row["weight_bucket"] for row in rows)
    statuses = Counter(str(row["status"]) for row in rows)
    html_weights = [float(row["html_weight_bytes"]) for row in rows]
    estimated_weights = [float(row["estimated_weight_bytes"]) for row in rows]
    total_pages = len(rows)
    pages_with_blocking = sum(1 for row in rows if row["render_blocking_count"] > 0)
    pages_with_mixed_content = sum(1 for row in rows if row["mixed_content_url_count"] > 0)
    pages_with_content_sizing_issues = sum(1 for row in rows if row["content_width_exceeds_viewport"])
    pages_with_plugins = sum(1 for row in rows if row["plugin_element_count"] > 0)
    pages_with_small_font_sizes = sum(1 for row in rows if row["small_font_size_count"] > 0)
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
        "pages_with_mixed_content": pages_with_mixed_content,
        "mixed_content_share": pages_with_mixed_content / total_pages if total_pages else 0.0,
        "total_mixed_content_urls": sum(row["mixed_content_url_count"] for row in rows),
        "pages_with_content_sizing_issues": pages_with_content_sizing_issues,
        "content_sizing_issue_share": pages_with_content_sizing_issues / total_pages if total_pages else 0.0,
        "pages_with_plugins": pages_with_plugins,
        "plugin_usage_share": pages_with_plugins / total_pages if total_pages else 0.0,
        "pages_with_small_font_sizes": pages_with_small_font_sizes,
        "small_font_size_share": pages_with_small_font_sizes / total_pages if total_pages else 0.0,
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
