"""Media accessibility analysis."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import urlparse

from .extractor import ExtractedPage


@dataclass
class MediaAccessibilityReport:
    summary: dict
    issues_by_type: dict[str, int]
    per_page: list[dict]
    media_with_issues: list[dict]


_GENERIC_ALT_RE = re.compile(r"^(?:image|img|photo|picture|graphic|icon|logo|banner)(?:\s+\d+)?$", re.I)


def _image_filename(src: str) -> str:
    path = urlparse(src or "").path or src or ""
    name = PurePosixPath(path).stem
    return re.sub(r"[-_]+", " ", name).strip().lower()


def _is_decorative(item: dict) -> bool:
    role = (item.get("role") or "").lower()
    return bool(item.get("aria_hidden")) or role in {"presentation", "none"}


def _image_issues(item: dict) -> list[str]:
    issues: list[str] = []
    alt = (item.get("alt") or "").strip()
    alt_present = bool(item.get("alt_present"))
    decorative = _is_decorative(item)
    if not alt_present and not decorative:
        issues.append("image_missing_alt")
    if alt_present and not alt and item.get("in_link"):
        if not (item.get("link_text") or item.get("link_title") or item.get("link_aria_label")):
            issues.append("linked_image_empty_alt")
    if alt:
        if len(alt) > 125:
            issues.append("image_long_alt")
        if _GENERIC_ALT_RE.match(alt):
            issues.append("image_generic_alt")
        filename = _image_filename(item.get("src") or "")
        if filename and alt.lower() == filename:
            issues.append("image_filename_alt")
    return issues


def _media_issues(item: dict) -> list[str]:
    media_type = item.get("type")
    if media_type == "image":
        return _image_issues(item)
    if media_type == "video" and not item.get("has_captions"):
        return ["video_missing_captions"]
    if media_type == "audio" and not item.get("has_transcript_hint"):
        return ["audio_missing_transcript"]
    if media_type == "iframe" and not (item.get("title") or item.get("aria_label")):
        return ["iframe_missing_title"]
    return []


def analyze(pages: Iterable[ExtractedPage]) -> MediaAccessibilityReport:
    page_list = list(pages)
    issues_by_type: Counter[str] = Counter()
    per_page: list[dict] = []
    media_with_issues: list[dict] = []
    media_type_counts: Counter[str] = Counter()
    pages_with_media = 0
    pages_with_issues = 0
    decorative_images = 0

    for page in page_list:
        items = list(page.media_items or [])
        if items:
            pages_with_media += 1
        page_issues: Counter[str] = Counter()
        for idx, item in enumerate(items):
            media_type = str(item.get("type") or "unknown")
            media_type_counts[media_type] += 1
            if media_type == "image" and _is_decorative(item):
                decorative_images += 1
            issues = _media_issues(item)
            if not issues:
                continue
            page_issues.update(issues)
            issues_by_type.update(issues)
            media_with_issues.append({
                "url": page.url,
                "title": page.title,
                "index": idx,
                "type": media_type,
                "src": item.get("src", ""),
                "alt": item.get("alt", ""),
                "issues": issues,
            })
        if page_issues:
            pages_with_issues += 1
        per_page.append({
            "url": page.url,
            "title": page.title,
            "media_count": len(items),
            "image_count": sum(1 for item in items if item.get("type") == "image"),
            "video_count": sum(1 for item in items if item.get("type") == "video"),
            "audio_count": sum(1 for item in items if item.get("type") == "audio"),
            "iframe_count": sum(1 for item in items if item.get("type") == "iframe"),
            "issues": dict(page_issues),
            "issue_count": sum(page_issues.values()),
        })

    total_pages = len(page_list)
    summary = {
        "total_pages": total_pages,
        "pages_with_media": pages_with_media,
        "pages_with_issues": pages_with_issues,
        "issue_share": pages_with_issues / total_pages if total_pages else 0.0,
        "total_media": sum(media_type_counts.values()),
        "total_images": media_type_counts.get("image", 0),
        "decorative_images": decorative_images,
        "images_missing_alt": issues_by_type.get("image_missing_alt", 0),
        "linked_images_empty_alt": issues_by_type.get("linked_image_empty_alt", 0),
        "images_long_alt": issues_by_type.get("image_long_alt", 0),
        "images_generic_alt": issues_by_type.get("image_generic_alt", 0),
        "images_filename_alt": issues_by_type.get("image_filename_alt", 0),
        "total_videos": media_type_counts.get("video", 0),
        "videos_missing_captions": issues_by_type.get("video_missing_captions", 0),
        "total_audio": media_type_counts.get("audio", 0),
        "audio_missing_transcript": issues_by_type.get("audio_missing_transcript", 0),
        "total_iframes": media_type_counts.get("iframe", 0),
        "iframes_missing_title": issues_by_type.get("iframe_missing_title", 0),
    }
    per_page.sort(key=lambda row: (-row["issue_count"], -row["media_count"], row["url"]))
    return MediaAccessibilityReport(
        summary=summary,
        issues_by_type=dict(issues_by_type),
        per_page=per_page,
        media_with_issues=media_with_issues[:300],
    )


def to_payload(report: MediaAccessibilityReport) -> dict:
    return {
        "summary": report.summary,
        "issues_by_type": report.issues_by_type,
        "per_page": report.per_page,
        "media_with_issues": report.media_with_issues,
    }
