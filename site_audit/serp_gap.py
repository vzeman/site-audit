"""Standalone SERP semantic content-gap workflow.

This command intentionally lives outside the main audit pipeline. It starts
from an existing project report, expands selected pages into SERP competitors,
uses the domain cache under ``projects/<domain>/cache/serp_gap``, and writes an
independent report under ``projects/<domain>/serp_gap/report``.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import requests

from .analyzer import PageInfo, section_for_url
from .cache import HttpCache, content_hash, domain_slug
from .competitive_analysis import (
    CompetitiveTarget,
    CompetitorPage,
    _serp_items,
    build_serp_paragraph_gap,
)
from .competitive_analysis import _fetch_dataforseo_serp as fetch_dataforseo_serp
from .competitive_analysis import CompetitiveAutoConfig
from .ahrefs import AhrefsConfig, build_analysis as build_ahrefs_analysis, fetch_snapshot as fetch_ahrefs_snapshot
from .embedder import DEFAULT_MODEL, Embedder
from .extractor import ExtractedPage, extract
from .scatter import project


USER_AGENT = "site-audit-serp-gap/0.1 (+https://github.com/vzeman/site-audit)"

_IGNORED_SERP_HOSTS = {
    "x.com",
    "twitter.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
    "tiktok.com",
    "reddit.com",
    "play.google.com",
    "apps.apple.com",
}


@dataclass
class SerpGapConfig:
    domain: str
    projects_root: Path = Path("projects")
    model: str = DEFAULT_MODEL
    urls: list[str] = field(default_factory=list)
    url_include_patterns: list[str] = field(default_factory=list)
    url_exclude_patterns: list[str] = field(default_factory=list)
    keyword_source: str = "auto"
    keywords: list[str] = field(default_factory=list)
    keywords_file: Path | None = None
    keywords_per_page: int = 3
    results_per_keyword: int = 5
    max_pages: int = 20
    max_competitor_pages: int = 100
    max_paragraphs_per_page: int = 80
    provider: str = "auto"
    country: str | None = None
    language: str | None = None
    min_ranking_position: int = 1
    max_ranking_position: int = 30
    min_impressions: int = 0
    min_traffic: float = 0.0
    use_h1_keyword: bool = False
    include_serp_keyword_suggestions: bool = False
    max_serp_keyword_suggestions: int = 8
    use_ahrefs_metrics: bool = False
    ahrefs_refresh: bool = False
    ahrefs_date: str | None = None
    ahrefs_country: str | None = None
    ahrefs_mode: str = "subdomains"
    ahrefs_top_pages_limit: int = 1000
    ahrefs_keywords_limit: int = 1000
    refresh_serp: bool = False
    refresh_competitors: bool = False
    budget_usd: float | None = None
    dry_run: bool = False


def run(config: SerpGapConfig) -> dict:
    project_dir = config.projects_root / domain_slug(config.domain)
    base_report_dir = project_dir / "report"
    if not base_report_dir.exists():
        return {
            "status": "missing_base_report",
            "message": f"No existing audit report found at {base_report_dir}.",
        }

    pages = _load_pages(base_report_dir / "pages.json")
    if not pages:
        return {"status": "no_pages", "message": "Existing report has no pages.json rows."}

    cache_dir = project_dir / "cache" / "serp_gap"
    report_dir = project_dir / "serp_gap" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_pages, skipped_pages = _select_pages(pages, config)
    search_payload = _load_search_payload(base_report_dir)
    ahrefs_payload = {}
    if (config.use_ahrefs_metrics or config.keyword_source == "ahrefs") and not config.dry_run:
        ahrefs_payload = _load_or_fetch_ahrefs_payload(config.domain, project_dir, pages, config)
        search_payload = _merge_search_payloads(search_payload, ahrefs_payload)
    keyword_metrics = _keyword_metrics_lookup(search_payload, ahrefs_payload)
    manual_keywords = _load_manual_keywords(config)
    keyword_rows, skipped_keywords = _select_keywords(
        selected_pages,
        search_payload,
        manual_keywords,
        config,
    )
    _enrich_keyword_rows(keyword_rows, keyword_metrics)
    plan = _plan(keyword_rows, cache_dir, config)
    if plan.get("budget_status") == "over_budget":
        payload = {
            "status": "budget_exceeded",
            "domain": config.domain,
            "summary": plan,
            "selected_pages": [_page_payload(p) for p in selected_pages],
            "selected_keywords": keyword_rows,
            "skipped_pages": skipped_pages,
            "skipped_keywords": skipped_keywords,
        }
        _write_outputs(payload, report_dir)
        return payload
    if config.dry_run:
        payload = {
            "status": "dry_run",
            "domain": config.domain,
            "summary": plan,
            "selected_pages": [_page_payload(p) for p in selected_pages],
            "selected_keywords": keyword_rows,
            "skipped_pages": skipped_pages,
            "skipped_keywords": skipped_keywords,
        }
        _write_outputs(payload, report_dir)
        return payload

    provider = _resolve_provider(config.provider)
    if provider == "serper" and not _serper_key():
        return {
            "status": "missing_serper_api_key",
            "message": "Set SERPER_API_KEY or SERPER_DEV_API_KEY, or use --provider dataforseo.",
            "summary": plan,
        }

    embedder = Embedder(config.model)
    own_cache = HttpCache(project_dir / "cache" / "http.sqlite")
    competitor_cache = HttpCache(cache_dir / "competitors.sqlite")

    page_results: list[dict] = []
    all_competitor_urls: set[str] = set()
    serp_url_rankings: dict[str, dict] = {}
    overview_rows: list[dict] = []
    overview_texts: list[str] = []
    overview_keywords_seen: set[str] = set()
    overview_urls_seen: set[str] = set()
    for page in selected_pages:
        page_keywords = [row for row in keyword_rows if _same_url(row["url"], page.url)]
        if not page_keywords:
            continue
        own_ext = _fetch_and_extract(page.url, own_cache, refresh=False)
        if own_ext is None:
            skipped_pages.append({"url": page.url, "reason": "own page fetch/extract failed"})
            continue

        if page.url not in overview_urls_seen:
            overview_urls_seen.add(page.url)
            overview_rows.append({
                "entity_type": "url",
                "source": "ours",
                "text": own_ext.title or own_ext.h1 or page.url,
                "url": page.url,
                "domain": urlparse(page.url).netloc,
            })
            overview_texts.append(" ".join([own_ext.title, own_ext.h1, page.url]).strip() or page.url)
            if own_ext.title:
                overview_rows.append({"entity_type": "title", "source": "ours", "text": own_ext.title, "url": page.url})
                overview_texts.append(own_ext.title)
            if own_ext.h1:
                overview_rows.append({"entity_type": "h1", "source": "ours", "text": own_ext.h1, "url": page.url})
                overview_texts.append(own_ext.h1)
            for header in own_ext.headers_rich[:40]:
                text = str(header.get("text") or "").strip()
                if not text:
                    continue
                overview_rows.append({
                    "entity_type": _heading_entity_type(header.get("level")),
                    "source": "ours",
                    "text": text,
                    "level": header.get("level"),
                    "url": page.url,
                })
                overview_texts.append(text)
            for para in (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]:
                if not para.strip():
                    continue
                overview_rows.append({
                    "entity_type": "paragraph",
                    "source": "ours",
                    "text": para[:300],
                    "url": page.url,
                })
                overview_texts.append(para)

        page_blocks: list[dict] = []
        page_keyword_keys = {row["keyword"].strip().lower() for row in page_keywords}
        keyword_index = 0
        while keyword_index < len(page_keywords):
            kw = page_keywords[keyword_index]
            keyword_index += 1
            keyword_key = kw["keyword"].strip().lower()
            if keyword_key and keyword_key not in overview_keywords_seen:
                overview_keywords_seen.add(keyword_key)
                overview_rows.append({
                    "entity_type": "keyword",
                    "source": "keyword",
                    "text": kw["keyword"],
                    "url": kw.get("url", page.url),
                    "clicks": kw.get("clicks", 0),
                    "impressions": kw.get("impressions", 0),
                    "position": kw.get("position", ""),
                    "traffic": kw.get("traffic", 0),
                    "volume": kw.get("volume", 0),
                })
                overview_texts.append(kw["keyword"])

            serp = _fetch_serp(kw["keyword"], provider, cache_dir, config)
            serp_meta = serp.get("meta") or {}
            if serp_meta.get("status") != "ok":
                page_blocks.append({
                    "keyword": kw,
                    "status": serp_meta.get("status", "serp_error"),
                    "message": serp_meta.get("message", ""),
                    "competitors": [],
                })
                continue
            if config.include_serp_keyword_suggestions and not str(kw.get("source", "")).startswith("serp_"):
                for suggestion in _serp_keyword_suggestion_rows(page, kw, serp, config):
                    suggestion_key = suggestion["keyword"].strip().lower()
                    if not suggestion_key or suggestion_key in page_keyword_keys:
                        continue
                    _apply_keyword_metrics(suggestion, keyword_metrics)
                    page_keyword_keys.add(suggestion_key)
                    page_keywords.append(suggestion)
                    keyword_rows.append(suggestion)
            _add_serp_url_rankings(
                serp_url_rankings,
                config.domain,
                kw,
                serp,
                top_n=10,
            )
            targets = _select_targets_with_budget(
                _targets_from_serp(config.domain, kw["keyword"], serp, config),
                all_competitor_urls,
                config,
            )
            if not targets:
                page_blocks.append({
                    "keyword": kw,
                    "status": "no_competitor_targets",
                    "competitors": [],
                })
                continue
            for target in targets:
                all_competitor_urls.add(target.competitor_url)
            competitor_pages = [
                _competitor_page(target, competitor_cache, embedder, config)
                for target in targets
            ]
            for competitor_page in competitor_pages:
                competitor_url = competitor_page.target.competitor_url
                if competitor_url in overview_urls_seen:
                    continue
                overview_urls_seen.add(competitor_url)
                overview_rows.append({
                    "entity_type": "url",
                    "source": "competitor",
                    "domain": urlparse(competitor_url).netloc,
                    "url": competitor_url,
                    "rank": competitor_page.target.rank,
                    "text": competitor_page.title or competitor_url,
                })
                overview_texts.append(" ".join([competitor_page.title, competitor_url]).strip() or competitor_url)
                if competitor_page.title:
                    overview_rows.append({
                        "entity_type": "title",
                        "source": "competitor",
                        "domain": urlparse(competitor_url).netloc,
                        "url": competitor_url,
                        "rank": competitor_page.target.rank,
                        "text": competitor_page.title,
                    })
                    overview_texts.append(competitor_page.title)
                if competitor_page.h1:
                    overview_rows.append({
                        "entity_type": "h1",
                        "source": "competitor",
                        "domain": urlparse(competitor_url).netloc,
                        "url": competitor_url,
                        "rank": competitor_page.target.rank,
                        "text": competitor_page.h1,
                    })
                    overview_texts.append(competitor_page.h1)
                for header in competitor_page.headers_rich[:40]:
                    text = str(header.get("text") or "").strip()
                    if not text:
                        continue
                    overview_rows.append({
                        "entity_type": _heading_entity_type(header.get("level")),
                        "source": "competitor",
                        "domain": urlparse(competitor_url).netloc,
                        "url": competitor_url,
                        "rank": competitor_page.target.rank,
                        "level": header.get("level"),
                        "text": text,
                    })
                    overview_texts.append(text)
                for para in competitor_page.paragraphs[:config.max_paragraphs_per_page]:
                    if not para.strip():
                        continue
                    overview_rows.append({
                        "entity_type": "paragraph",
                        "source": "competitor",
                        "domain": urlparse(competitor_url).netloc,
                        "url": competitor_url,
                        "rank": competitor_page.target.rank,
                        "text": para[:300],
                    })
                    overview_texts.append(para)
            gap = _build_gap(page, kw, own_ext, competitor_pages, embedder, config)
            gap["serp"] = {
                "provider": provider,
                "cache_status": serp_meta.get("cache_status", ""),
                "targets": [
                    {"url": t.competitor_url, "rank": t.rank}
                    for t in targets
                ],
            }
            page_blocks.append(gap)

        page_results.append({
            "url": page.url,
            "title": page.title,
            "h1": own_ext.h1,
            "keywords": page_keywords,
            "analyses": page_blocks,
        })

    payload = {
        "status": "ok",
        "domain": config.domain,
        "provider": provider,
        "summary": _summary(page_results, selected_pages, keyword_rows, plan),
        "selected_pages": [_page_payload(p) for p in selected_pages],
        "selected_keywords": keyword_rows,
        "skipped_pages": skipped_pages,
        "skipped_keywords": skipped_keywords,
        "ahrefs": {
            "meta": ahrefs_payload.get("meta", {}) if ahrefs_payload else {},
            "summary": ahrefs_payload.get("summary", {}) if ahrefs_payload else {},
        },
        "serp_url_rankings": _serp_url_ranking_rows(serp_url_rankings),
        "overview_scatter": _overview_scatter(overview_rows, overview_texts, embedder),
        "pages": page_results,
    }
    _write_outputs(payload, report_dir)
    return payload


def _load_pages(path: Path) -> list[PageInfo]:
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    pages = []
    for row in rows:
        url = str(row.get("url") or "")
        if not url:
            continue
        pages.append(PageInfo(
            url=url,
            title=str(row.get("title") or ""),
            description=str(row.get("description") or ""),
            section=str(row.get("section") or section_for_url(url)),
            word_count=int(row.get("word_count") or 0),
            language=row.get("language"),
        ))
    return pages


def _select_pages(pages: list[PageInfo], config: SerpGapConfig) -> tuple[list[PageInfo], list[dict]]:
    selected = []
    skipped = []
    for raw_url in config.urls:
        url = raw_url if "://" in raw_url else f"https://{raw_url}"
        existing = next((p for p in pages if _same_url(p.url, url)), None)
        selected.append(existing or PageInfo(
            url=url,
            title="",
            description="",
            section=section_for_url(url),
            word_count=0,
            language=None,
        ))
        if len(selected) >= config.max_pages:
            return selected, skipped
    for page in pages:
        if any(_same_url(page.url, p.url) for p in selected):
            continue
        if config.url_include_patterns and not any(_pattern_match(page.url, p) for p in config.url_include_patterns):
            skipped.append({"url": page.url, "reason": "outside URL pattern"})
            continue
        if config.url_exclude_patterns and any(_pattern_match(page.url, p) for p in config.url_exclude_patterns):
            skipped.append({"url": page.url, "reason": "excluded URL pattern"})
            continue
        selected.append(page)
        if len(selected) >= config.max_pages:
            break
    return selected, skipped


def _pattern_match(url: str, pattern: str) -> bool:
    parsed = urlparse(url)
    targets = [url, parsed.path or "/"]
    for target in targets:
        if fnmatch.fnmatch(target, pattern):
            return True
    if not _looks_like_regex(pattern):
        return False
    if pattern.startswith("re:"):
        pattern = pattern[3:]
    try:
        return re.search(pattern, url) is not None or re.search(pattern, parsed.path or "/") is not None
    except re.error:
        return False


def _looks_like_regex(pattern: str) -> bool:
    return pattern.startswith("re:") or any(ch in pattern for ch in ("^", "$", "(", ")", "[", "]", "\\", "+", "|"))


def _load_search_payload(report_dir: Path) -> dict:
    for name in ("search.json", "ahrefs.json", "gsc.json", "dataforseo.json"):
        path = report_dir / name
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("organic_keywords") or payload.get("top_pages"):
                    return payload
            except json.JSONDecodeError:
                continue
    return {}


def _load_or_fetch_ahrefs_payload(domain: str, project_dir: Path, pages: list[PageInfo], config: SerpGapConfig) -> dict:
    snapshot = fetch_ahrefs_snapshot(
        domain,
        project_dir / "cache",
        AhrefsConfig(
            enabled=True,
            date=config.ahrefs_date,
            country=str(config.ahrefs_country or "").lower() or None,
            mode=config.ahrefs_mode,
            top_pages_limit=config.ahrefs_top_pages_limit,
            keywords_limit=config.ahrefs_keywords_limit,
            refresh=config.ahrefs_refresh,
            semantic_sample_cap=0,
        ),
    )
    analysis = build_ahrefs_analysis(
        snapshot,
        pages,
        np.zeros((len(pages), 0), dtype=np.float32),
        semantic_sample_cap=0,
    )
    return analysis.payload


def _merge_search_payloads(*payloads: dict) -> dict:
    rows = []
    provider_payloads = []
    for payload in payloads:
        if not payload:
            continue
        provider = str((payload.get("meta") or {}).get("provider") or (payload.get("summary") or {}).get("provider") or "search")
        for row in payload.get("organic_keywords") or []:
            rows.append({**row, "provider": row.get("provider") or provider})
        provider_payloads.append({
            "provider": provider,
            "meta": payload.get("meta", {}),
            "summary": payload.get("summary", {}),
        })
    if not rows:
        return next((p for p in payloads if p), {})
    return {
        "meta": {"provider": "combined", "provider_label": "Combined Search Metrics"},
        "organic_keywords": rows,
        "provider_payloads": provider_payloads,
    }


def _metric_url_key(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{parsed.netloc.lower().removeprefix('www.')}{path}"


def _keyword_metrics_lookup(*payloads: dict) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for payload in payloads:
        if not payload:
            continue
        provider = str((payload.get("meta") or {}).get("provider") or (payload.get("summary") or {}).get("provider") or "search")
        for row in payload.get("organic_keywords") or []:
            keyword = str(row.get("keyword") or row.get("query") or "").strip().lower()
            url = str(row.get("matched_url") or row.get("url") or "").strip()
            if not keyword or not url:
                continue
            metrics = {
                "position": _safe_float(row.get("position")),
                "impressions": _safe_int(row.get("impressions")),
                "clicks": _safe_int(row.get("clicks")),
                "traffic": _safe_float(row.get("traffic")),
                "volume": _safe_int(row.get("volume")),
                "metrics_source": row.get("provider") or provider,
                "metrics_url": url,
            }
            for key in ((_metric_url_key(url), keyword), ("", keyword)):
                current = out.get(key)
                if current is None or (
                    _safe_float(metrics.get("traffic")) > _safe_float(current.get("traffic"))
                    or _safe_int(metrics.get("impressions")) > _safe_int(current.get("impressions"))
                ):
                    out[key] = metrics
    return out


def _apply_keyword_metrics(row: dict, lookup: dict[tuple[str, str], dict]) -> None:
    key = (_metric_url_key(str(row.get("url") or "")), str(row.get("keyword") or "").strip().lower())
    metrics = lookup.get(key)
    if not metrics:
        metrics = lookup.get(("", str(row.get("keyword") or "").strip().lower()))
    if not metrics:
        return
    for field in ("position", "impressions", "clicks", "traffic", "volume"):
        if not row.get(field):
            row[field] = metrics.get(field, row.get(field))
    row["metrics_source"] = metrics.get("metrics_source", "")
    row["metrics_url"] = metrics.get("metrics_url", "")


def _enrich_keyword_rows(rows: list[dict], lookup: dict[tuple[str, str], dict]) -> None:
    for row in rows:
        _apply_keyword_metrics(row, lookup)


def _load_manual_keywords(config: SerpGapConfig) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if config.keywords:
        out["*"] = [k.strip() for k in config.keywords if k.strip()]
    if config.keywords_file and config.keywords_file.is_file():
        for raw in config.keywords_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= 2:
                out.setdefault(parts[0], []).append(parts[1])
            elif parts:
                out.setdefault("*", []).append(parts[0])
    return out


def _select_keywords(
    pages: list[PageInfo],
    search_payload: dict,
    manual_keywords: dict[str, list[str]],
    config: SerpGapConfig,
) -> tuple[list[dict], list[dict]]:
    rows = []
    skipped = []
    search_rows = list(search_payload.get("organic_keywords") or [])
    for page in pages:
        candidates: list[dict] = []
        for kw in manual_keywords.get("*", []):
            candidates.append(_keyword_row(page, kw, "manual", synthetic=False))
        for url_key, kws in manual_keywords.items():
            if url_key == "*":
                continue
            if _same_url(url_key, page.url) or _pattern_match(page.url, url_key):
                for kw in kws:
                    candidates.append(_keyword_row(page, kw, "file", synthetic=False))
        if config.keyword_source in {"auto", "gsc", "ahrefs", "dataforseo", "google_ads"}:
            for row in search_rows:
                matched = row.get("matched_url") or row.get("url") or ""
                keyword = str(row.get("keyword") or row.get("query") or "")
                if not keyword or not matched or not _same_url(matched, page.url):
                    continue
                provider = str(row.get("provider") or (search_payload.get("meta") or {}).get("provider") or "search")
                if config.keyword_source != "auto" and provider != config.keyword_source:
                    continue
                pos = _safe_float(row.get("position"))
                if pos and (pos < config.min_ranking_position or pos > config.max_ranking_position):
                    skipped.append({"url": page.url, "keyword": keyword, "reason": "outside ranking position range"})
                    continue
                impressions = _safe_int(row.get("impressions"))
                traffic = _safe_float(row.get("traffic"))
                if impressions < config.min_impressions or traffic < config.min_traffic:
                    skipped.append({"url": page.url, "keyword": keyword, "reason": "below demand threshold"})
                    continue
                candidates.append(_keyword_row(page, keyword, provider, row=row))
        if config.use_h1_keyword or config.keyword_source == "h1":
            title = page.title.strip()
            if title:
                candidates.append(_keyword_row(page, title, "h1", synthetic=True))
        candidates = _dedupe_keywords(candidates)
        candidates.sort(key=_keyword_priority, reverse=True)
        if not candidates:
            skipped.append({"url": page.url, "reason": "no ranking keywords"})
        rows.extend(candidates[:config.keywords_per_page])
    return rows, skipped


def _keyword_row(page: PageInfo, keyword: str, source: str, row: dict | None = None, synthetic: bool = False) -> dict:
    row = row or {}
    return {
        "url": page.url,
        "page_title": page.title,
        "keyword": keyword,
        "source": source,
        "synthetic": synthetic,
        "position": _safe_float(row.get("position")),
        "impressions": _safe_int(row.get("impressions")),
        "clicks": _safe_int(row.get("clicks")),
        "traffic": _safe_float(row.get("traffic")),
        "volume": _safe_int(row.get("volume")),
    }


def _dedupe_keywords(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (row["url"], row["keyword"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _serp_keyword_suggestion_rows(
    page: PageInfo,
    parent_keyword: dict,
    payload: dict,
    config: SerpGapConfig,
) -> list[dict]:
    rows = []
    limit = max(0, int(config.max_serp_keyword_suggestions or 0))
    if not limit:
        return rows
    for source, keyword in _extract_serp_keyword_suggestions(payload):
        row = _keyword_row(page, keyword, source, synthetic=True)
        row["parent_keyword"] = parent_keyword.get("keyword", "")
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _extract_serp_keyword_suggestions(payload: dict) -> list[tuple[str, str]]:
    suggestions: list[tuple[str, str]] = []
    raw = payload.get("raw") or {}
    for item in raw.get("peopleAlsoAsk") or []:
        text = str(item.get("question") or item.get("title") or "").strip()
        if text:
            suggestions.append(("serp_people_also_ask", text))
    for item in raw.get("peopleAlsoSearch") or raw.get("relatedSearches") or []:
        text = str(item.get("query") if isinstance(item, dict) else item or "").strip()
        if text:
            suggestions.append(("serp_people_also_search", text))
    for item in _serp_items(payload):
        item_type = item.get("type")
        if item_type == "people_also_ask":
            for child in item.get("items") or []:
                text = str((child or {}).get("title") or "").strip()
                if text:
                    suggestions.append(("serp_people_also_ask", text))
        elif item_type == "people_also_search":
            for child in item.get("items") or []:
                text = str(child.get("title") if isinstance(child, dict) else child or "").strip()
                if text:
                    suggestions.append(("serp_people_also_search", text))
    seen = set()
    out = []
    for source, keyword in suggestions:
        normalized = re.sub(r"\s+", " ", keyword).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append((source, normalized))
    return out


def _keyword_priority(row: dict) -> float:
    position = _safe_float(row.get("position")) or 20.0
    demand = max(_safe_int(row.get("impressions")), _safe_int(row.get("volume")), int(_safe_float(row.get("traffic")) * 20))
    opportunity = min(max(position - 1.0, 0.0), 30.0) / 30.0
    source_weight = {"gsc": 1.2, "ahrefs": 1.0, "dataforseo": 1.0, "google_ads": 0.85, "manual": 1.4, "file": 1.4, "h1": 0.5}.get(str(row.get("source")), 0.8)
    return source_weight * (np.log1p(demand) + opportunity)


def _plan(keyword_rows: list[dict], cache_dir: Path, config: SerpGapConfig) -> dict:
    serp_calls = len(keyword_rows)
    estimated_competitors = min(
        len(keyword_rows) * config.results_per_keyword,
        config.max_competitor_pages,
    )
    cached_serp = sum(1 for row in keyword_rows if _any_serp_cache(cache_dir, row["keyword"]))
    # Serper and DataForSEO costs vary by plan; this is intentionally an
    # estimate used for budget gating, not billing.
    estimated_cost = max(0, serp_calls - cached_serp) * 0.001
    return {
        "pages_selected": len({row["url"] for row in keyword_rows}),
        "keywords_selected": len(keyword_rows),
        "serp_api_calls": serp_calls,
        "serp_api_calls_after_cache": max(0, serp_calls - cached_serp),
        "competitor_urls_estimated": estimated_competitors,
        "embedding_texts_estimated": estimated_competitors * config.max_paragraphs_per_page,
        "estimated_cost_usd": round(estimated_cost, 4),
        "budget_usd": config.budget_usd,
        "budget_status": "ok" if config.budget_usd is None or estimated_cost <= config.budget_usd else "over_budget",
    }


def _any_serp_cache(cache_dir: Path, keyword: str) -> bool:
    key = content_hash(keyword.lower())
    return any(cache_dir.glob(f"serp/**/{key}.json"))


def _resolve_provider(provider: str) -> str:
    if provider != "auto":
        return provider
    return "serper" if _serper_key() else "dataforseo"


def _serper_key() -> str:
    return os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_DEV_API_KEY") or ""


def _fetch_serp(keyword: str, provider: str, cache_dir: Path, config: SerpGapConfig) -> dict:
    if provider == "dataforseo":
        auto_config = CompetitiveAutoConfig(
            enabled=True,
            results_per_keyword=max(config.results_per_keyword * 3, config.results_per_keyword + 10),
            refresh_serp=config.refresh_serp,
        )
        if config.country:
            if config.country.isdigit():
                auto_config.location_code = int(config.country)
            else:
                auto_config.location_name = config.country
        if config.language:
            auto_config.language_code = config.language
        return fetch_dataforseo_serp(keyword, cache_dir / "serp_dataforseo", auto_config)
    if provider != "serper":
        return {"meta": {"status": "unsupported_provider", "message": provider}, "raw": {}}
    key = content_hash(keyword.lower())
    country = config.country or "us"
    language = config.language or "en"
    path = cache_dir / "serp" / "serper" / country / language / f"{key}.json"
    if path.is_file() and not config.refresh_serp:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("meta", {})["cache_status"] = "hit"
        return payload
    api_key = _serper_key()
    if not api_key:
        return {"meta": {"status": "missing_api_key", "message": "Set SERPER_API_KEY."}, "raw": {}}
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": keyword, "gl": country, "hl": language, "num": max(config.results_per_keyword * 3, config.results_per_keyword + 10)},
        timeout=60,
    )
    if resp.status_code >= 400:
        return {"meta": {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}, "raw": {}}
    raw = resp.json()
    payload = {"meta": {"status": "ok", "cache_status": "miss", "provider": "serper", "fetched_at": time.time()}, "raw": raw}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _targets_from_serp(domain: str, keyword: str, payload: dict, config: SerpGapConfig) -> list[CompetitiveTarget]:
    out = []
    seen = set()
    for row in _serp_result_rows(payload):
        url = row.get("url") or ""
        if not url or _is_own_url(url, domain) or _is_ignored_serp_url(url) or url in seen:
            continue
        seen.add(url)
        out.append(CompetitiveTarget(keyword, url, keyword, _safe_int(row.get("rank"))))
        if len(out) >= config.results_per_keyword:
            break
    return out


def _select_targets_with_budget(
    targets: list[CompetitiveTarget],
    known_competitor_urls: set[str],
    config: SerpGapConfig,
) -> list[CompetitiveTarget]:
    selected = []
    projected_new_urls = set(known_competitor_urls)
    for target in targets:
        is_known = target.competitor_url in known_competitor_urls
        if not is_known and len(projected_new_urls) >= config.max_competitor_pages:
            continue
        selected.append(target)
        projected_new_urls.add(target.competitor_url)
        if len(selected) >= config.results_per_keyword:
            break
    return selected


def _serp_result_rows(payload: dict) -> list[dict]:
    provider = (payload.get("meta") or {}).get("provider") or ""
    rows = []
    if provider == "serper" or "organic" in (payload.get("raw") or {}):
        for item in (payload.get("raw") or {}).get("organic") or []:
            rows.append({"url": item.get("link"), "rank": item.get("position"), "title": item.get("title", "")})
    else:
        for item in _serp_items(payload):
            if item.get("type") == "organic":
                rows.append({
                    "url": item.get("url"),
                    "rank": item.get("rank_group") or item.get("rank_absolute"),
                    "title": item.get("title", ""),
                })
    return rows


def _add_serp_url_rankings(
    rankings: dict[str, dict],
    domain: str,
    keyword: str | dict,
    payload: dict,
    top_n: int = 10,
) -> None:
    keyword_text = str(keyword.get("keyword") if isinstance(keyword, dict) else keyword or "")
    keyword_metrics = keyword if isinstance(keyword, dict) else {}
    keyword_seen: set[str] = set()
    for row in _serp_result_rows(payload):
        rank = _safe_int(row.get("rank"))
        if rank <= 0 or rank > top_n:
            continue
        url = _canonical_serp_url(str(row.get("url") or ""))
        if not url or _is_ignored_serp_url(url) or url in keyword_seen:
            continue
        keyword_seen.add(url)
        host = urlparse(url).netloc.lower()
        item = rankings.setdefault(url, {
            "url": url,
            "domain": host,
            "is_selected_domain": _is_own_url(url, domain),
            "top10_count": 0,
            "best_rank": rank,
            "rank_sum": 0,
            "impressions": 0,
            "clicks": 0,
            "traffic": 0.0,
            "volume": 0,
            "keywords": [],
        })
        item["top10_count"] += 1
        item["best_rank"] = min(int(item.get("best_rank") or rank), rank)
        item["rank_sum"] += rank
        item["impressions"] += _safe_int(keyword_metrics.get("impressions", 0))
        item["clicks"] += _safe_int(keyword_metrics.get("clicks", 0))
        item["traffic"] += _safe_float(keyword_metrics.get("traffic", 0.0))
        item["volume"] += _safe_int(keyword_metrics.get("volume", 0))
        item["keywords"].append({
            "keyword": keyword_text,
            "rank": rank,
            "impressions": _safe_int(keyword_metrics.get("impressions", 0)),
            "clicks": _safe_int(keyword_metrics.get("clicks", 0)),
            "traffic": round(_safe_float(keyword_metrics.get("traffic", 0.0)), 4),
            "source_position": _safe_float(keyword_metrics.get("position", 0.0)),
            "volume": _safe_int(keyword_metrics.get("volume", 0)),
            "source": keyword_metrics.get("source", ""),
        })


def _serp_url_ranking_rows(rankings: dict[str, dict]) -> list[dict]:
    rows = []
    for item in rankings.values():
        count = max(1, int(item.get("top10_count") or 0))
        keywords = sorted(item.get("keywords") or [], key=lambda row: (_safe_int(row.get("rank")), row.get("keyword", "")))
        rows.append({
            "url": item.get("url", ""),
            "domain": item.get("domain", ""),
            "is_selected_domain": bool(item.get("is_selected_domain")),
            "top10_count": count,
            "best_rank": int(item.get("best_rank") or 0),
            "average_rank": round(float(item.get("rank_sum") or 0) / count, 2),
            "impressions": int(item.get("impressions") or 0),
            "clicks": int(item.get("clicks") or 0),
            "traffic": round(float(item.get("traffic") or 0.0), 4),
            "volume": int(item.get("volume") or 0),
            "keywords": keywords,
        })
    return sorted(rows, key=lambda row: (
        -int(row.get("top10_count") or 0),
        int(row.get("best_rank") or 999),
        float(row.get("average_rank") or 999),
        row.get("domain", ""),
        row.get("url", ""),
    ))


def _canonical_serp_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return parsed._replace(fragment="", path=parsed.path or "/").geturl()


def _heading_entity_type(level) -> str:
    value = _safe_int(level)
    if 1 <= value <= 6:
        return f"h{value}"
    return "header"


def _semantic_dedupe_key(row: dict, text: str | None = None) -> tuple[str, str, str, str]:
    normalized_text = re.sub(
        r"\s+",
        " ",
        str(text if text is not None else row.get("_dedupe_text") or row.get("text") or ""),
    ).strip().lower()
    return (
        str(row.get("url") or ""),
        str(row.get("source") or ""),
        str(row.get("entity_type") or ""),
        normalized_text,
    )


def _dedupe_semantic_row_texts(rows: list[dict], texts: list[str]) -> tuple[list[dict], list[str], int]:
    seen = set()
    deduped_rows: list[dict] = []
    deduped_texts: list[str] = []
    removed = 0
    for row, text in zip(rows, texts):
        key = _semantic_dedupe_key(row, text)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped_rows.append({k: v for k, v in row.items() if k != "_dedupe_text"})
        deduped_texts.append(text)
    return deduped_rows, deduped_texts, removed


def _dedupe_semantic_matrix(meta: list[dict], matrix: np.ndarray) -> tuple[list[dict], np.ndarray, int]:
    seen = set()
    indexes: list[int] = []
    deduped_meta: list[dict] = []
    removed = 0
    for i, row in enumerate(meta[:len(matrix)]):
        key = _semantic_dedupe_key(row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        indexes.append(i)
        deduped_meta.append({k: v for k, v in row.items() if k != "_dedupe_text"})
    if not indexes:
        return [], matrix[:0], removed
    return deduped_meta, matrix[indexes], removed


def _is_ignored_serp_url(url: str) -> bool:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower().removeprefix("www.")
    return any(host == ignored or host.endswith(f".{ignored}") for ignored in _IGNORED_SERP_HOSTS)


def _competitor_page(
    target: CompetitiveTarget,
    cache: HttpCache,
    embedder: Embedder,
    config: SerpGapConfig,
) -> CompetitorPage:
    ext = _fetch_and_extract(target.competitor_url, cache, refresh=config.refresh_competitors)
    if ext is None:
        return CompetitorPage(
            target=target,
            title="",
            paragraphs=[],
            paragraph_embeddings=np.zeros((0, 0), dtype=np.float32),
            structural_gaps=[],
            answerability=0.0,
            paragraph_count=0,
            error="competitor fetch/extract failed",
        )
    paragraphs = (ext.paragraphs or [])[:config.max_paragraphs_per_page]
    embs = embedder.encode(paragraphs, batch_size=64).astype(np.float32) if paragraphs else np.zeros((0, 0), dtype=np.float32)
    return CompetitorPage(
        target=target,
        title=ext.title or target.competitor_url,
        paragraphs=paragraphs,
        paragraph_embeddings=embs,
        structural_gaps=[],
        answerability=0.0,
        paragraph_count=len(paragraphs),
        h1=ext.h1,
        headers_rich=ext.headers_rich,
    )


def _fetch_and_extract(url: str, cache: HttpCache, refresh: bool) -> ExtractedPage | None:
    cached = None if refresh else cache.get(url)
    if cached and 200 <= cached.status < 300:
        body = cached.text
    else:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
        except requests.RequestException:
            return None
        if resp.status_code >= 400:
            return None
        cache.put(url, resp.status_code, dict(resp.headers), resp.content, canonical_url=resp.url)
        body = resp.text
    return extract(url, body, max_chars=16000)


def _build_gap(
    page: PageInfo,
    keyword: dict,
    own_ext: ExtractedPage,
    competitor_pages: list[CompetitorPage],
    embedder: Embedder,
    config: SerpGapConfig,
) -> dict:
    own_paragraphs = (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
    own_embeddings = embedder.encode(own_paragraphs, batch_size=64).astype(np.float32) if own_paragraphs else np.zeros((0, 0), dtype=np.float32)
    gap = build_serp_paragraph_gap(
        query=keyword["keyword"],
        cluster=keyword["keyword"],
        our_url=page.url,
        our_title=page.title,
        our_paragraphs=own_paragraphs,
        our_paragraph_embeddings=own_embeddings,
        competitor_pages=competitor_pages,
    )
    gap["keyword"] = keyword
    gap["scatter"] = _scatter(keyword, own_ext, own_embeddings, competitor_pages, embedder)
    gap["competitor_pages"] = [
        {
            "url": cp.target.competitor_url,
            "rank": cp.target.rank,
            "title": cp.title,
            "paragraph_count": cp.paragraph_count,
            "error": cp.error,
        }
        for cp in competitor_pages
    ]
    return gap


def _scatter(
    keyword: dict,
    own_ext: ExtractedPage,
    own_para_embeddings: np.ndarray,
    competitor_pages: list[CompetitorPage],
    embedder: Embedder,
) -> dict:
    texts = []
    meta = []

    def add_text(row: dict, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        meta.append({**row, "text": text})
        texts.append(text)

    add_text({
        "entity_type": "keyword",
        "source": "keyword",
        "url": own_ext.url,
        "impressions": _safe_int(keyword.get("impressions")),
        "clicks": _safe_int(keyword.get("clicks")),
        "traffic": _safe_float(keyword.get("traffic")),
        "volume": _safe_int(keyword.get("volume")),
        "position": _safe_float(keyword.get("position")),
    }, keyword.get("keyword", ""))
    add_text({"entity_type": "title", "source": "ours", "url": own_ext.url}, own_ext.title)
    add_text({"entity_type": "h1", "source": "ours", "url": own_ext.url}, own_ext.h1)
    for header in own_ext.headers_rich[:40]:
        add_text({
            "entity_type": _heading_entity_type(header.get("level")),
            "source": "ours",
            "level": header.get("level"),
            "url": own_ext.url,
        }, header.get("text", ""))
    for cp in competitor_pages:
        if cp.error:
            continue
        domain = urlparse(cp.target.competitor_url).netloc
        add_text({
            "entity_type": "title",
            "source": "competitor",
            "domain": domain,
            "url": cp.target.competitor_url,
            "rank": cp.target.rank,
        }, cp.title)
        add_text({
            "entity_type": "h1",
            "source": "competitor",
            "domain": domain,
            "url": cp.target.competitor_url,
            "rank": cp.target.rank,
        }, cp.h1)
        for header in cp.headers_rich[:40]:
            add_text({
                "entity_type": _heading_entity_type(header.get("level")),
                "source": "competitor",
                "domain": domain,
                "url": cp.target.competitor_url,
                "rank": cp.target.rank,
                "level": header.get("level"),
            }, header.get("text", ""))
    base_embs = embedder.encode(texts, batch_size=64).astype(np.float32) if texts else np.zeros((0, 0), dtype=np.float32)
    embs = [base_embs]
    for i, para in enumerate(own_ext.paragraphs or []):
        if i >= len(own_para_embeddings):
            break
        meta.append({
            "entity_type": "paragraph",
            "source": "ours",
            "text": para[:300],
            "_dedupe_text": para,
            "url": own_ext.url,
        })
    if len(own_para_embeddings):
        embs.append(own_para_embeddings[:len(own_ext.paragraphs or [])])
    for cp in competitor_pages:
        if cp.error:
            continue
        for i, para in enumerate(cp.paragraphs[:60]):
            if i >= len(cp.paragraph_embeddings):
                break
            meta.append({
                "entity_type": "paragraph",
                "source": "competitor",
                "domain": urlparse(cp.target.competitor_url).netloc,
                "url": cp.target.competitor_url,
                "rank": cp.target.rank,
                "text": para[:300],
                "_dedupe_text": para,
            })
        if len(cp.paragraph_embeddings):
            embs.append(cp.paragraph_embeddings[:60])
    if not embs:
        return {"points": [], "shown": 0}
    matrix = np.vstack([e for e in embs if len(e)]).astype(np.float32)
    if len(matrix) != len(meta):
        meta = meta[:len(matrix)]
    meta, matrix, duplicates_removed = _dedupe_semantic_matrix(meta, matrix)
    if not len(matrix):
        return {"points": [], "shown": 0, "total": 0, "duplicates_removed": duplicates_removed}
    keyword_vec = matrix[0] if len(matrix) else None
    labels, coords = project(matrix, num_clusters=min(30, max(2, len(matrix) // 4)))
    points = []
    for i, row in enumerate(meta):
        similarity = float(np.clip(matrix[i] @ keyword_vec, -1.0, 1.0)) if keyword_vec is not None else 0.0
        points.append({
            **row,
            "x": round(float(coords[i, 0]), 5),
            "y": round(float(coords[i, 1]), 5),
            "cluster": int(labels[i]),
            "keyword_similarity": round(similarity, 4),
            "keyword_distance": round(1.0 - similarity, 4),
        })
    return {"points": points[:1600], "shown": min(len(points), 1600), "total": len(points), "duplicates_removed": duplicates_removed}


def _overview_scatter(rows: list[dict], texts: list[str], embedder: Embedder) -> dict:
    usable = [(row, text) for row, text in zip(rows, texts) if str(text or "").strip()]
    if not usable:
        return {"points": [], "shown": 0, "total": 0}
    rows = [row for row, _ in usable]
    texts = [text for _, text in usable]
    rows, texts, duplicates_removed = _dedupe_semantic_row_texts(rows, texts)
    if not rows:
        return {"points": [], "shown": 0, "total": 0, "duplicates_removed": duplicates_removed}
    matrix = embedder.encode(texts, batch_size=64).astype(np.float32)
    keyword_indexes = [i for i, row in enumerate(rows) if row.get("entity_type") == "keyword"]
    keyword_matrix = matrix[keyword_indexes] if keyword_indexes else np.zeros((0, matrix.shape[1]), dtype=np.float32)
    keyword_labels = [str(rows[i].get("text") or "") for i in keyword_indexes]
    if keyword_indexes:
        weights = np.array([
            _safe_float(rows[i].get("impressions"))
            or _safe_float(rows[i].get("volume"))
            or _safe_float(rows[i].get("traffic"))
            or _safe_float(rows[i].get("clicks"))
            or 1.0
            for i in keyword_indexes
        ], dtype=np.float32)
        centroid = np.average(keyword_matrix, axis=0, weights=weights)
        norm = float(np.linalg.norm(centroid))
        if norm:
            centroid = centroid / norm
        centroid_text = f"Demand-weighted keyword centroid ({len(keyword_indexes)} keywords)"
        rows.append({
            "entity_type": "keyword_centroid",
            "source": "keyword",
            "text": centroid_text,
            "keyword_count": len(keyword_indexes),
            "impressions": sum(_safe_int(rows[i].get("impressions")) for i in keyword_indexes),
            "clicks": sum(_safe_int(rows[i].get("clicks")) for i in keyword_indexes),
            "traffic": round(sum(_safe_float(rows[i].get("traffic")) for i in keyword_indexes), 4),
            "volume": sum(_safe_int(rows[i].get("volume")) for i in keyword_indexes),
        })
        matrix = np.vstack([matrix, centroid.astype(np.float32)])
    labels, coords = project(matrix, num_clusters=min(30, max(2, len(matrix) // 3)))
    points = []
    for i, row in enumerate(rows):
        nearest_keyword = ""
        similarity = 0.0
        if row.get("entity_type") == "keyword":
            nearest_keyword = str(row.get("text") or "")
            similarity = 1.0
        elif row.get("entity_type") == "keyword_centroid":
            nearest_keyword = "Keyword centroid"
            similarity = 1.0
        elif len(keyword_matrix):
            similarities = keyword_matrix @ matrix[i]
            best = int(np.argmax(similarities))
            similarity = float(np.clip(similarities[best], -1.0, 1.0))
            nearest_keyword = keyword_labels[best]
        points.append({
            **row,
            "x": round(float(coords[i, 0]), 5),
            "y": round(float(coords[i, 1]), 5),
            "cluster": int(labels[i]),
            "nearest_keyword": nearest_keyword,
            "keyword_similarity": round(similarity, 4),
            "keyword_distance": round(1.0 - similarity, 4),
        })
    return {"points": points[:1600], "shown": min(len(points), 1600), "total": len(points), "duplicates_removed": duplicates_removed}


def _summary(page_results: list[dict], selected_pages: list[PageInfo], keyword_rows: list[dict], plan: dict) -> dict:
    analyses = [a for p in page_results for a in p.get("analyses", [])]
    summaries = [a.get("summary") or {} for a in analyses if a.get("status") == "ok"]
    selected_urls = {str(p.get("url") or "") for p in page_results if p.get("url")}
    competitor_urls_attempted = set()
    competitor_urls_downloaded = set()
    review_paragraphs = 0
    for analysis in analyses:
        if analysis.get("status") != "ok":
            continue
        for cp in analysis.get("competitor_pages") or []:
            url = str(cp.get("url") or "")
            if not url:
                continue
            competitor_urls_attempted.add(url)
            if not cp.get("error") and int(cp.get("paragraph_count") or 0) > 0:
                competitor_urls_downloaded.add(url)
        review_rows = analysis.get("off_intent_paragraphs") or analysis.get("own_paragraphs_to_review") or []
        review_paragraphs += len(review_rows)
    return {
        **plan,
        "pages_analyzed": len(page_results),
        "pages_selected": len(selected_pages),
        "keywords_selected": len(keyword_rows),
        "serp_api_calls": len(keyword_rows),
        "urls_downloaded": len(selected_urls) + len(competitor_urls_downloaded),
        "selected_domain_urls_downloaded": len(selected_urls),
        "competitor_urls_downloaded": len(competitor_urls_downloaded),
        "competitor_urls_attempted": len(competitor_urls_attempted),
        "serp_clusters": len(summaries),
        "missing_topics": sum(int(s.get("missing", 0)) for s in summaries),
        "partial_topics": sum(int(s.get("partial", 0)) for s in summaries),
        "off_intent_paragraphs": sum(int(s.get("off_intent_paragraphs", 0)) for s in summaries),
        "review_paragraphs": review_paragraphs,
    }


def _write_outputs(payload: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "serp_gap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(report_dir / "serp_gap.csv", _csv_rows(payload))
    (report_dir / "index.html").write_text(_html(payload), encoding="utf-8")


def _csv_rows(payload: dict) -> list[dict]:
    rows = []
    for page in payload.get("pages") or []:
        for analysis in page.get("analyses") or []:
            for topic in analysis.get("topics") or []:
                rows.append({
                    "page_url": page.get("url", ""),
                    "keyword": (analysis.get("keyword") or {}).get("keyword", analysis.get("query", "")),
                    "topic": topic.get("label", ""),
                    "coverage": topic.get("coverage", ""),
                    "priority": topic.get("priority", ""),
                    "competitor_prevalence": topic.get("competitor_prevalence", ""),
                    "our_best_similarity": topic.get("our_best_similarity", ""),
                    "example": ((topic.get("examples") or [{}])[0]).get("paragraph", ""),
                })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    template = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SERP Semantic Gap</title>
<style>
:root{--ink:#17202a;--muted:#5d6d7e;--line:#d7dee8;--soft:#f5f7fa;--panel:#fff;--ours:#176a35;--comp:#2d5b9a;--kw:#8a4b00;--missing:#b42318;--partial:#9a6700;--covered:#176a35;--shadow:0 1px 3px rgba(22,34,51,.08)}
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:14px;line-height:1.45}a{color:#1b5dbf;text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1440px;margin:0 auto;padding:24px}.topbar{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:18px}.title h1{font-size:28px;line-height:1.1;margin:0 0 8px}.title p{margin:0;color:var(--muted);max-width:820px}.summary{display:grid;grid-template-columns:repeat(7,minmax(112px,1fr));gap:10px;margin:18px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;box-shadow:var(--shadow)}.metric b{display:block;font-size:22px;line-height:1.1}.metric span{color:var(--muted);font-size:12px}.page-section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:18px 0 28px;box-shadow:var(--shadow);overflow:hidden}.page-head{padding:18px 20px;border-bottom:1px solid var(--line);background:#fff}.page-head h2{font-size:21px;margin:0 0 6px}.url{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.keyword-card{padding:20px;border-top:1px solid var(--line)}.keyword-card:first-of-type{border-top:0}.keyword-grid{display:grid;grid-template-columns:minmax(460px,1.2fr) minmax(420px,.8fr);gap:18px;align-items:start}.keyword-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}.keyword-head h3{font-size:19px;margin:0}.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:3px 8px;background:#fff;font-size:12px;color:var(--muted)}.chip.missing{color:var(--missing);border-color:#f1b4ad;background:#fff7f6}.chip.partial{color:var(--partial);border-color:#e8cf85;background:#fff9e8}.chip.covered{color:var(--covered);border-color:#a8d5b6;background:#f1faf4}.panel{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.panel h4{margin:0;padding:11px 12px;border-bottom:1px solid var(--line);font-size:14px;background:var(--soft)}.panel-body{padding:12px}.scatter-wrap{position:relative}.scatter{width:100%;height:390px;display:block;background:#fbfcfe}.scatter-point{cursor:pointer}.scatter-point:focus{outline:none;stroke:#111;stroke-width:2.4}.scatter-tooltip{display:none;position:absolute;left:12px;right:12px;bottom:12px;max-height:250px;overflow:auto;background:linear-gradient(180deg,#fff,#f8fbff);border:1px solid #b9c7d8;border-radius:10px;box-shadow:0 14px 34px rgba(22,34,51,.22);padding:0;z-index:2}.scatter-tooltip.open{display:block}.tip-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:11px 12px;border-bottom:1px solid #e3e9f2;background:#f4f7fb}.tip-title{font-weight:750;font-size:13px;color:#142033}.tip-sub{margin-top:2px;color:#637083;font-size:11px}.tip-close{border:0;background:#e7edf5;border-radius:6px;padding:2px 8px;cursor:pointer}.tip-body{padding:11px 12px}.tip-badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.tip-badge{display:inline-flex;align-items:center;border:1px solid #d5deea;border-radius:999px;padding:3px 7px;background:#fff;font-size:11px;color:#405166}.tip-badge.ours{border-color:#9bd0ae;color:#176a35;background:#f1faf4}.tip-badge.competitor{border-color:#acc3e5;color:#2d5b9a;background:#f3f7ff}.tip-badge.keyword{border-color:#e7c889;color:#8a4b00;background:#fff9e8}.tip-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:9px}.tip-field{border:1px solid #edf1f5;border-radius:7px;background:#fff;padding:6px 7px;min-width:0}.tip-field span{display:block;color:#768395;font-size:10px;text-transform:uppercase;letter-spacing:.04em}.tip-field strong{display:block;color:#1b2838;font-size:12px;overflow-wrap:anywhere}.tip-text{border-left:3px solid #8fb2df;background:#f7faff;border-radius:6px;padding:8px 9px;color:#263445;font-size:12px;line-height:1.45}.tip-explain{color:#637083;font-size:12px;margin-bottom:9px}.scatter-controls{position:absolute;top:10px;right:10px;display:flex;gap:5px;z-index:3}.scatter-controls button{border:1px solid #c9d3df;background:#fff;color:#263445;border-radius:6px;padding:3px 8px;font-size:12px;line-height:1;box-shadow:0 1px 3px rgba(22,34,51,.12);cursor:pointer}.scatter-controls button:hover{background:#f1f5f9}.scatter.is-panning{cursor:grabbing}.scatter{cursor:grab}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.tables{display:grid;grid-template-columns:1fr;gap:14px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid #edf1f5;text-align:left;vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#fbfcfe}.topic-label{font-weight:600}.coverage-missing{color:var(--missing);font-weight:700}.coverage-partial{color:var(--partial);font-weight:700}.coverage-covered{color:var(--covered);font-weight:700}.cluster-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.cluster{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fff}.cluster strong{display:block;margin-bottom:5px}.bar{height:7px;background:#e9eef5;border-radius:999px;overflow:hidden;margin:8px 0}.bar span{display:block;height:100%;background:#5f8cc9}.muted{color:var(--muted)}.empty{padding:18px;color:var(--muted)}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}.competitors li,.review li{margin:0 0 8px}.competitors,.review{padding-left:18px;margin:0}.mini{font-size:12px;color:var(--muted)}@media(max-width:980px){.wrap{padding:14px}.summary{grid-template-columns:repeat(2,1fr)}.keyword-grid,.two-col{grid-template-columns:1fr}.scatter{height:320px}.topbar{display:block}}
  :root {
    --audit-bg: #f5efe6;
    --audit-panel: #fffdfa;
    --audit-panel-soft: #fff8ef;
    --audit-line: #eadfce;
    --audit-text: #241f19;
    --audit-muted: #7f776f;
    --audit-accent: #ff8a1f;
    --audit-accent-dark: #b85a00;
    --audit-accent-soft: #fff0df;
    --audit-green: #1f9d66;
    --audit-red: #cf5060;
    --audit-blue: #4766ff;
    --audit-shadow: 0 18px 48px rgba(61, 43, 18, 0.08);
    --audit-radius: 26px;
  }
  body {
    color: var(--audit-text);
    background:
      radial-gradient(circle at 8% 0%, rgba(255, 138, 31, 0.16), transparent 30%),
      radial-gradient(circle at 88% 10%, rgba(71, 102, 255, 0.10), transparent 26%),
      linear-gradient(135deg, #fbf6ee 0%, var(--audit-bg) 54%, #efe2d2 100%);
  }
  .wrap { max-width: 1500px; margin-left: 300px; padding: 28px 28px 72px; }
  .topbar {
    position: relative;
    overflow: hidden;
    padding: 30px 32px;
    margin-bottom: 28px;
    border: 1px solid var(--audit-line);
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(255, 253, 250, 0.96), rgba(255, 240, 223, 0.82));
    box-shadow: var(--audit-shadow);
  }
  .topbar::before {
    content: "";
    position: absolute;
    right: -90px;
    top: -110px;
    width: 330px;
    height: 330px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(255, 138, 31, 0.28), rgba(255, 138, 31, 0));
    pointer-events: none;
  }
  .title h1 {
    position: relative;
    font-size: clamp(2.35rem, 5vw, 4.75rem);
    line-height: 0.95;
    letter-spacing: -0.055em;
    color: var(--audit-text);
  }
  .title p, .mini, .muted, .url { color: var(--audit-muted); }
  .section-note {
    border: 1px solid var(--audit-line);
    border-radius: 14px;
    background: var(--audit-panel-soft);
    color: var(--audit-muted);
    font-size: 0.82rem;
    line-height: 1.45;
    margin: 0 0 12px;
    padding: 10px 12px;
  }
  #status {
    min-width: 230px;
    text-align: right;
    white-space: nowrap;
    overflow-wrap: normal;
  }
  .metric, .page-section, .panel, .cluster {
    background: var(--audit-panel);
    border-color: var(--audit-line);
    border-radius: var(--audit-radius);
    box-shadow: var(--audit-shadow);
  }
  .page-head, .panel h4, th { background: var(--audit-panel-soft); }
  .keyword-card, .page-head, .panel h4, th, td { border-color: var(--audit-line); }
  .chip, .tip-badge, .tip-field, .scatter-controls button {
    border-color: var(--audit-line);
    border-radius: 999px;
  }
  .scatter-tooltip {
    border-color: var(--audit-line);
    border-radius: 18px;
    background: linear-gradient(180deg, var(--audit-panel), #fff8ef);
    box-shadow: 0 24px 56px rgba(61, 43, 18, 0.16);
  }
  .tip-head { background: var(--audit-accent-soft); border-color: var(--audit-line); }
  .tip-title { color: var(--audit-text); }
  .tip-text { border-left-color: var(--audit-accent); background: #fffdfa; color: var(--audit-text); }
  .keyword-grid { display: block; }
  .keyword-grid > .panel { width: 100%; }
  .tables {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 14px;
    margin-top: 14px;
  }
  .scatter-filters {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
  }
  .scatter-filter-group {
    margin-bottom: 10px;
  }
  .scatter-filter-group .scatter-filters {
    margin-top: 5px;
  }
  .scatter-filter {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--audit-line);
    border-radius: 999px;
    background: #fffdfa;
    color: var(--audit-muted);
    cursor: pointer;
    font-size: 0.76rem;
    padding: 5px 9px;
    user-select: none;
  }
  .scatter-filter input {
    width: 13px;
    height: 13px;
    margin: 0;
    accent-color: var(--audit-accent);
  }
  .serp-ranking-list {
    display: grid;
    gap: 10px;
  }
  .serp-ranking-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 12px;
    align-items: start;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdfa;
    padding: 10px 12px;
  }
  .serp-ranking-url {
    min-width: 0;
    overflow-wrap: anywhere;
    font-weight: 750;
  }
  .serp-ranking-domain {
    color: var(--audit-muted);
    font-size: 0.76rem;
    margin-top: 3px;
  }
  .serp-ranking-stats {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: flex-end;
    min-width: 160px;
  }
  .serp-ranking-keywords {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    grid-column: 1 / -1;
  }
  .serp-ranking-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border: 1px solid var(--audit-line);
    border-radius: 999px;
    background: var(--audit-panel-soft);
    color: var(--audit-muted);
    font-size: 0.74rem;
    padding: 3px 7px;
  }
  .serp-ranking-chip strong {
    color: var(--audit-text);
  }
  .serp-ranking-chart {
    display: grid;
    gap: 10px;
    margin-bottom: 14px;
  }
  .serp-ranking-chart-row {
    display: grid;
    grid-template-columns: minmax(180px, 0.82fr) minmax(220px, 1.4fr) 128px;
    gap: 10px;
    align-items: center;
  }
  .serp-ranking-chart-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 750;
  }
  .serp-ranking-chart-track {
    position: relative;
    height: 18px;
    overflow: hidden;
    border-radius: 999px;
    background: #f2e7d9;
  }
  .serp-ranking-chart-bar {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--audit-accent), #ffc06f);
  }
  .serp-ranking-chart-meta {
    color: var(--audit-muted);
    font-size: 0.76rem;
    text-align: right;
    white-space: nowrap;
  }
  .serp-url-graph {
    width: 100%;
    height: 560px;
    display: block;
    margin: 2px 0 16px;
    border: 1px solid var(--audit-line);
    border-radius: 18px;
    background: #fffdfa;
  }
  .serp-url-graph .graph-edge {
    stroke-linecap: round;
    fill: none;
    opacity: 0.72;
    mix-blend-mode: multiply;
    transition: opacity 120ms ease, stroke 120ms ease, stroke-width 120ms ease;
  }
  .serp-url-graph-wrap.has-active .graph-edge:not(.is-active) {
    stroke: #c9c1b6;
    stroke-width: 1.1px;
    opacity: 0.16;
    mix-blend-mode: normal;
  }
  .serp-url-graph .graph-edge.is-active {
    opacity: 1;
    stroke-width: 7px;
    filter: drop-shadow(0 2px 5px rgba(61, 43, 18, 0.26));
  }
  .serp-url-graph .graph-node {
    stroke: #fffdfa;
    stroke-width: 2;
    transition: opacity 120ms ease, stroke 120ms ease, stroke-width 120ms ease;
  }
  .serp-url-graph-wrap.has-active .graph-node:not(.is-active) {
    opacity: 0.34;
  }
  .serp-url-graph .graph-node.is-active {
    opacity: 1;
    stroke: var(--audit-text);
    stroke-width: 3;
  }
  .serp-url-graph .graph-label {
    fill: var(--audit-text);
    font-size: 11px;
    font-weight: 750;
    pointer-events: none;
  }
  .serp-url-graph .graph-meta {
    fill: var(--audit-muted);
    font-size: 10px;
    pointer-events: none;
  }
  .serp-url-graph .traffic-legend text {
    fill: var(--audit-muted);
    font-size: 10px;
  }
  .serp-url-graph-wrap {
    position: relative;
  }
  .graph-tooltip {
    display: none;
    position: absolute;
    left: 14px;
    top: 14px;
    z-index: 4;
    max-width: 360px;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: linear-gradient(180deg, var(--audit-panel), #fff8ef);
    box-shadow: 0 18px 42px rgba(61, 43, 18, 0.16);
    padding: 12px;
    color: var(--audit-text);
    pointer-events: none;
  }
  .graph-tooltip.open { display: block; }
  .graph-tooltip h5 {
    margin: 0 0 6px;
    font-size: 0.86rem;
  }
  .graph-tooltip .metric-line {
    color: var(--audit-muted);
    font-size: 0.76rem;
    margin-top: 3px;
  }
  .topic-chart {
    display: grid;
    gap: 9px;
    margin-bottom: 12px;
  }
  .topic-chart-row {
    display: grid;
    grid-template-columns: minmax(180px, 0.9fr) minmax(220px, 1.5fr) 64px;
    gap: 10px;
    align-items: center;
  }
  .topic-chart-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 700;
  }
  .topic-chart-track {
    height: 11px;
    overflow: hidden;
    border-radius: 999px;
    background: #f2e7d9;
  }
  .topic-chart-bar {
    display: block;
    height: 100%;
    border-radius: 999px;
  }
  .topic-chart-bar.coverage-missing { background: var(--audit-red); }
  .topic-chart-bar.coverage-partial { background: var(--audit-accent); }
  .topic-chart-bar.coverage-covered { background: var(--audit-green); }
  a { color: var(--audit-accent-dark); }
  button { transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease; }
  button:hover { transform: translateY(-1px); }
  .report-sidebar {
    position: fixed;
    inset: 18px auto 18px 18px;
    z-index: 50;
    width: 260px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid var(--audit-line);
    border-radius: 24px;
    background: rgba(255, 253, 250, 0.94);
    box-shadow: var(--audit-shadow);
    backdrop-filter: blur(16px);
  }
  .report-sidebar-header {
    padding: 18px 18px 12px;
    border-bottom: 1px solid var(--audit-line);
  }
  .report-sidebar-title {
    display: block;
    color: var(--audit-text);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .report-sidebar-subtitle {
    display: block;
    margin-top: 4px;
    color: var(--audit-muted);
    font-size: 0.75rem;
    line-height: 1.3;
  }
  .report-nav {
    display: flex;
    flex: 1;
    flex-direction: column;
    gap: 4px;
    overflow-y: auto;
    padding: 10px;
  }
  .report-nav-button {
    width: 100%;
    border: 0;
    border-radius: 14px;
    background: transparent;
    color: var(--audit-muted);
    cursor: pointer;
    padding: 9px 10px;
    text-align: left;
    text-decoration: none;
    transform: none !important;
  }
  .report-nav-button:hover {
    background: var(--audit-panel-soft);
    color: var(--audit-text);
    transform: none;
  }
  .report-nav-button.is-active {
    background: var(--audit-accent-soft);
    color: var(--audit-accent-dark);
    font-weight: 800;
  }
  .report-nav-label {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.84rem;
    line-height: 1.15;
  }
  .report-nav-button.nav-keyword {
    margin-left: 10px;
    width: calc(100% - 10px);
    padding: 6px 10px;
  }
  .report-nav-button.nav-keyword .report-nav-label {
    font-size: 0.78rem;
  }
  @media(max-width:1180px) {
    .report-sidebar { position: static; width: auto; margin: 14px; }
    .wrap { margin-left: 0; padding: 14px; }
    #status { min-width: 0; text-align: left; white-space: normal; }
    .tables, .topic-chart-row { grid-template-columns: 1fr; }
  }
</style>
<body>
<aside class="report-sidebar">
  <div class="report-sidebar-header">
    <span class="report-sidebar-title">SERP Gap</span>
    <span class="report-sidebar-subtitle">Page and keyword sections</span>
  </div>
  <nav class="report-nav" id="report-nav"></nav>
</aside>
<div class="wrap">
  <div class="topbar"><div class="title"><h1>SERP Semantic Gap</h1><p>Semantic comparison of selected audited pages against live SERP competitors. Each page section contains keyword-level scatterplots, topic clusters, competitor relationships, and editorial gaps.</p></div><div class="url" id="status"></div></div>
  <div class="summary" id="summary"></div>
  <div id="overview"></div>
  <div id="app"></div>
</div>
<script>
const data = __DATA__;
const app = document.getElementById('app');
const overviewEl = document.getElementById('overview');
const summaryEl = document.getElementById('summary');
const statusEl = document.getElementById('status');
const navEl = document.getElementById('report-nav');
const colors = {keyword:'#8a4b00', ours:'#176a35', competitor:'#2d5b9a', title:'#7b3fb2', h1:'#b65f00', header:'#68788b', paragraph:'#2d5b9a'};
const ownDomain = data.domain || 'selected domain';
function esc(s){return String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function urlLink(url,label=url){return url?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label||url)}</a>`:'';}
function sectionNote(text){return `<div class="section-note">${esc(text)}</div>`;}
function n(v){return Number(v||0).toLocaleString();}
function pct(v){return Math.round(Number(v||0)*100)+'%';}
const domainPalette=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#be123c','#4f46e5','#0f766e','#a16207','#7c3aed','#15803d'];
function hashString(s){let h=0;for(let i=0;i<String(s||'').length;i++)h=(h*31+String(s).charCodeAt(i))>>>0;return h;}
function domainColor(domain){return domainPalette[hashString(domain)%domainPalette.length];}
function urlDomain(url){try{return new URL(url).hostname;}catch(_){return '';}}
function pointDomain(p){return p.domain||urlDomain(p.url||'');}
function sourceColor(p){if(p.entity_type==='keyword'||p.entity_type==='keyword_centroid')return colors.keyword;const domain=pointDomain(p);if(domain)return domainColor(domain);if(p.entity_type==='title')return colors.title;if(/^h[1-6]$/.test(p.entity_type||''))return colors.h1;if(p.entity_type==='header')return colors.header;return colors.competitor;}
function metrics(){const s=data.summary||{};return [['Pages',s.pages_analyzed||0],['Keywords',s.keywords_selected||0],['SERP calls',s.serp_api_calls_after_cache ?? s.serp_api_calls ?? 0],['URLs downloaded',s.urls_downloaded||0],['Missing topics',s.missing_topics||0],['Partial topics',s.partial_topics||0],['Review paragraphs',s.review_paragraphs ?? s.off_intent_paragraphs ?? 0]];}
statusEl.textContent = `${data.domain || ''} · ${data.provider || ''} · ${data.status || ''}`;
summaryEl.innerHTML = metrics().map(([label,value])=>`<div class="metric"><b>${n(value)}</b><span>${esc(label)}</span></div>`).join('');
function pointLabel(d){if(d.type==='keyword_centroid')return 'Keyword centroid';if(d.type==='keyword')return 'Keyword';if(d.type==='url'&&d.source==='ours')return `${ownDomain} URL`;if(d.type==='url'&&d.source==='competitor')return 'Competitor URL';if(d.type==='url')return 'URL';if(d.type==='title')return 'Page title';if(/^h[1-6]$/.test(d.type||''))return `${String(d.type).toUpperCase()} heading`;if(d.type==='header')return 'Header';if(d.source==='ours')return `${ownDomain} paragraph`;if(d.source==='competitor')return 'Competitor paragraph';return d.type||'Point';}
function pointDetail(p){const type=p.entity_type||'point';return{type,source:p.source||'',cluster:p.cluster??'',domain:p.domain||'',rank:p.rank||'',url:p.url||'',text:String(p.text||'').slice(0,520),nearest_keyword:p.nearest_keyword||'',keyword_similarity:p.keyword_similarity??'',keyword_distance:p.keyword_distance??'',impressions:p.impressions??'',clicks:p.clicks??'',traffic:p.traffic??'',volume:p.volume??'',position:p.position??'',keyword_count:p.keyword_count??''};}
function pointTooltip(p){const d=pointDetail(p);return [pointLabel(d),d.text,d.impressions!==''&&Number(d.impressions)>0&&`Impressions: ${d.impressions}`,d.clicks!==''&&Number(d.clicks)>0&&`Clicks: ${d.clicks}`,d.traffic!==''&&Number(d.traffic)>0&&`Traffic: ${d.traffic}`,d.keyword_distance!==''&&`Keyword distance: ${d.keyword_distance}`,d.keyword_similarity!==''&&`Keyword similarity: ${d.keyword_similarity}`,d.domain&&`Domain: ${d.domain}`,d.rank&&`SERP rank: ${d.rank}`,d.cluster!==''&&`Cluster: ${d.cluster}`].filter(Boolean).join('\\n');}
function pointDetailHtml(d){const sourceClass=d.type==='keyword'||d.type==='keyword_centroid'?'keyword':d.source==='ours'?'ours':d.source==='competitor'?'competitor':'';const badges=[];badges.push(`<span class="tip-badge ${sourceClass}">${esc(d.source==='ours'?ownDomain:(d.source||d.type))}</span>`);if(Number(d.keyword_count||0)>0)badges.push(`<span class="tip-badge">${n(d.keyword_count)} keywords</span>`);if(Number(d.impressions||0)>0)badges.push(`<span class="tip-badge">${n(d.impressions)} impressions</span>`);if(Number(d.clicks||0)>0)badges.push(`<span class="tip-badge">${n(d.clicks)} clicks</span>`);if(Number(d.traffic||0)>0)badges.push(`<span class="tip-badge">${n(d.traffic)} traffic</span>`);if(Number(d.volume||0)>0)badges.push(`<span class="tip-badge">vol ${n(d.volume)}</span>`);if(d.nearest_keyword&&d.type!=='keyword')badges.push(`<span class="tip-badge">nearest ${esc(d.nearest_keyword)}</span>`);if(d.keyword_distance!==''&&d.keyword_distance!==undefined)badges.push(`<span class="tip-badge">distance ${esc(d.keyword_distance)}</span>`);if(d.keyword_similarity!==''&&d.keyword_similarity!==undefined)badges.push(`<span class="tip-badge">similarity ${esc(d.keyword_similarity)}</span>`);if(d.domain)badges.push(`<span class="tip-badge">${esc(d.domain)}</span>`);if(d.rank)badges.push(`<span class="tip-badge">rank ${esc(d.rank)}</span>`);if(d.cluster!==''&&d.cluster!==undefined)badges.push(`<span class="tip-badge">cluster ${esc(d.cluster)}</span>`);return `<div class="tip-head"><div><div class="tip-title">${esc(pointLabel(d))}</div>${d.url?`<div class="tip-sub">${urlLink(d.url)}</div>`:''}</div><button class="tip-close" type="button" aria-label="Close tooltip">x</button></div><div class="tip-body"><div class="tip-badges">${badges.join('')}</div><div class="tip-text">${esc(d.text||'(no text captured)')}</div></div>`;}
function clusterSummary(points){const keywordDemand=new Map();for(const p of points||[]){if(p.entity_type!=='keyword')continue;const name=String(p.text||'').trim();if(!name)continue;keywordDemand.set(name,{traffic:Number(p.traffic||0),impressions:Number(p.impressions||0),volume:Number(p.volume||0),clicks:Number(p.clicks||0)});}const groups=new Map();for(const p of points||[]){const id=p.cluster ?? 0;const g=groups.get(id)||{id,total:0,ours:0,competitor:0,keyword:0,centroid:0,headers:0,samples:[],keywordNames:new Set(),impactTraffic:0,impactImpressions:0,impactVolume:0,impactClicks:0};g.total++;if(p.entity_type==='keyword'){g.keyword++;const name=String(p.text||'').trim();if(name)g.keywordNames.add(name);}else if(p.entity_type==='keyword_centroid'){g.centroid++;}else if(p.source==='ours')g.ours++;else if(p.source==='competitor')g.competitor++;if(['title','header'].includes(p.entity_type)||/^h[1-6]$/.test(p.entity_type||''))g.headers++;if(p.entity_type!=='keyword_centroid'&&g.samples.length<3&&p.text)g.samples.push(p.text);const nearest=String(p.nearest_keyword||'').trim();if(nearest&&nearest!=='Keyword centroid')g.keywordNames.add(nearest);groups.set(id,g);}const clusters=[...groups.values()].map(g=>{for(const name of g.keywordNames){const demand=keywordDemand.get(name);if(!demand)continue;g.impactTraffic+=demand.traffic;g.impactImpressions+=demand.impressions;g.impactVolume+=demand.volume;g.impactClicks+=demand.clicks;}g.impactScore=g.impactTraffic||g.impactImpressions||g.impactVolume||g.impactClicks||0;g.keywords=[...g.keywordNames];delete g.keywordNames;return g;});return clusters.sort((a,b)=>b.impactScore-a.impactScore||b.total-a.total).slice(0,8);}
function pointSize(p){if(p.entity_type==='keyword'||p.entity_type==='keyword_centroid'){const demand=Number(p.impressions||0)||Number(p.volume||0);if(demand>0)return Math.max(p.entity_type==='keyword_centroid'?12:8,Math.min(p.entity_type==='keyword_centroid'?28:22,7+Math.sqrt(demand)/2.8));return p.entity_type==='keyword_centroid'?12:8;}if(p.clicks)return Math.max(5,Math.min(14,4+Math.sqrt(Number(p.clicks)||0)));if(p.rank)return Math.max(4.5,12-Number(p.rank||10)*0.65);if(p.entity_type==='url')return 8;if(p.entity_type==='title'||/^h[1-6]$/.test(p.entity_type||''))return 7;if(p.entity_type==='header')return 5.8;return 3.9;}
function markerSvg(p, xRaw, yRaw, color, stroke, tip, detail){const x=Number(xRaw),y=Number(yRaw);const type=p.entity_type||'paragraph';const domain=pointDomain(p);const size=pointSize(p);const attrs=`class="scatter-point" tabindex="0" fill="${color}" stroke="${stroke}" stroke-width="1.2" opacity=".86" aria-label="${esc(tip)}" data-tooltip="${esc(tip)}" data-detail="${esc(detail)}" data-entity="${esc(type)}" data-domain="${esc(domain)}"`;const title=`<title>${esc(tip)}</title>`;if(type==='keyword_centroid'){const r=size*.95;const pts=[];for(let i=0;i<6;i++){const a=-Math.PI/2+i*Math.PI/3;pts.push(`${(x+Math.cos(a)*r).toFixed(1)},${(y+Math.sin(a)*r).toFixed(1)}`);}return `<polygon ${attrs} stroke-width="2" points="${pts.join(' ')}">${title}</polygon>`;}if(type==='keyword'){return `<polygon ${attrs} points="${x},${y-size} ${x+size},${y} ${x},${y+size} ${x-size},${y}">${title}</polygon>`;}if(type==='title'||/^h[1-6]$/.test(type)){return `<polygon ${attrs} points="${x},${y-size} ${x+size},${y+size*.85} ${x-size},${y+size*.85}">${title}</polygon>`;}if(type==='header'){const s=size*1.7;return `<rect ${attrs} x="${x-s/2}" y="${y-s/2}" width="${s}" height="${s}" rx="1.5">${title}</rect>`;}return `<circle ${attrs} cx="${x}" cy="${y}" r="${size}">${title}</circle>`;}
function entityLabel(type){return ({keyword_centroid:'keyword centroid',keyword:'keywords',url:'URLs',title:'titles',h1:'H1s',h2:'H2s',h3:'H3s',h4:'H4s',h5:'H5s',h6:'H6s',header:'unclassified headings',paragraph:'paragraphs'}[type]||type);}
function entityFilters(points){const order=['keyword_centroid','keyword','url','title','h1','h2','h3','h4','h5','h6','header','paragraph'];const orderIndex=type=>{const index=order.indexOf(type);return index<0?999:index;};const types=[...new Set((points||[]).map(p=>p.entity_type||'paragraph'))].sort((a,b)=>orderIndex(a)-orderIndex(b));return `<div class="scatter-filters" aria-label="Visible entity types">${types.map(type=>`<label class="scatter-filter"><input type="checkbox" data-entity-filter="${esc(type)}" checked> ${esc(entityLabel(type))}</label>`).join('')}</div>`;}
function domainFilters(points){const domains=[...new Set((points||[]).map(pointDomain).filter(Boolean))].sort();if(!domains.length)return '';return `<div class="scatter-filter-group"><div class="mini">Domains</div><div class="scatter-filters" aria-label="Visible domains">${domains.map(domain=>`<label class="scatter-filter"><input type="checkbox" data-domain-filter="${esc(domain)}" checked> <i class="dot" style="background:${domainColor(domain)}"></i>${esc(domain)}</label>`).join('')}</div></div>`;}
function scatterSvg(points){if(!points||!points.length)return '<div class="empty">No scatter data available for this keyword.</div>';const w=820,h=390,pad=26;const xs=points.map(p=>+p.x),ys=points.map(p=>+p.y);let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);if(minX===maxX){minX-=1;maxX+=1}if(minY===maxY){minY-=1;maxY+=1}const sx=x=>pad+(x-minX)/(maxX-minX)*(w-pad*2);const sy=y=>h-pad-(y-minY)/(maxY-minY)*(h-pad*2);const marks=points.map(p=>{const x=sx(+p.x).toFixed(1),y=sy(+p.y).toFixed(1);const stroke=p.source==='ours'?'#0b3d1e':'#fff';const tip=pointTooltip(p);const detail=JSON.stringify(pointDetail(p));return markerSvg(p,x,y,sourceColor(p),stroke,tip,detail);}).join('');const domains=[...new Set(points.map(pointDomain).filter(Boolean))];const domainLegend=domains.slice(0,8).map(d=>`<span><i class="dot" style="background:${domainColor(d)}"></i>${esc(d)}</span>`).join('');const unclassifiedLegend=points.some(p=>p.entity_type==='header')?'<span>■ unclassified headings</span>':'';const centroidLegend=points.some(p=>p.entity_type==='keyword_centroid')?'<span>hexagon = demand-weighted keyword centroid</span>':'';return `${entityFilters(points)}${domainFilters(points)}<div class="scatter-wrap"><div class="scatter-controls" aria-label="Scatterplot zoom controls"><button type="button" data-zoom="in" title="Zoom in">+</button><button type="button" data-zoom="out" title="Zoom out">−</button><button type="button" data-zoom="reset" title="Reset zoom">Reset</button></div><svg class="scatter" viewBox="0 0 ${w} ${h}" data-base-viewbox="0 0 ${w} ${h}" role="img" aria-label="Semantic scatterplot. Wheel to zoom, drag to pan, double-click to reset, click or focus dots for point explanations."><rect x="0" y="0" width="${w}" height="${h}" fill="#fbfcfe"/><line x1="${pad}" x2="${w-pad}" y1="${h-pad}" y2="${h-pad}" stroke="#d7dee8"/><line x1="${pad}" x2="${pad}" y1="${pad}" y2="${h-pad}" stroke="#d7dee8"/>${marks}</svg><div class="scatter-tooltip" role="dialog" aria-live="polite" aria-label="Scatter point details"></div></div><div class="legend"><span><i class="dot" style="background:${colors.keyword};transform:rotate(45deg)"></i>keyword diamond size = impressions or Ahrefs volume</span>${centroidLegend}<span>▲ title/H1-H6</span>${unclassifiedLegend}${domainLegend}<span class="muted">Wheel to zoom, drag to pan, double-click to reset, click a dot for details.</span></div>`;}
function topicChart(topics){if(!topics||!topics.length)return '<div class="empty">No topic relation chart available.</div>';const maxSeen=Math.max(1,...topics.map(t=>Number(t.competitor_coverage||0)));return `<div class="topic-chart">${topics.slice(0,12).map(t=>{const seen=Number(t.competitor_coverage||0);const sim=Number(t.our_best_similarity||0);const width=Math.max(5,Math.min(100,Math.round((seen/maxSeen)*100)));return `<div class="topic-chart-row"><div class="topic-chart-label" title="${esc(t.label||'')}">${esc(t.label||'Untitled topic')}</div><div class="topic-chart-track" title="Seen on ${seen} competitor pages; ${esc(ownDomain)} similarity ${sim.toFixed(2)}"><span class="topic-chart-bar coverage-${esc(t.coverage||'partial')}" style="width:${width}%"></span></div><div class="mini">${esc(t.coverage||'')}</div></div>`;}).join('')}</div>`;}
function topicRows(topics, limit=12){if(!topics||!topics.length)return '<tr><td colspan="6" class="muted">No topics classified.</td></tr>';return topics.slice(0,limit).map(t=>{const ex=(t.examples||[])[0]||{};return `<tr><td class="coverage-${esc(t.coverage)}">${esc(t.coverage)}</td><td>${esc(t.priority)}</td><td><div class="topic-label">${esc(t.label)}</div><div class="mini">${esc(ex.paragraph||'')}</div></td><td>${esc(t.competitor_coverage)}/${esc(t.competitor_urls?.length||'')}</td><td>${esc(t.our_best_similarity)}</td><td>${urlLink(ex.url)}</td></tr>`}).join('');}
function clusterImpactChart(points){const clusters=clusterSummary(points);if(!clusters.length)return '<div class="empty">No topic impact data available.</div>';const maxImpact=Math.max(1,...clusters.map(c=>Number(c.impactScore||0)));return `<div class="topic-chart">${clusters.map(c=>{const width=Math.max(5,Math.round((Number(c.impactScore||0)/maxImpact)*100));const metrics=[Number(c.impactTraffic||0)>0&&`${n(c.impactTraffic)} traffic`,Number(c.impactImpressions||0)>0&&`${n(c.impactImpressions)} impr`,Number(c.impactVolume||0)>0&&`vol ${n(c.impactVolume)}`].filter(Boolean).join(' · ')||'No demand metric';const label=(c.keywords||[]).slice(0,3).join(', ')||`Cluster ${c.id}`;return `<div class="topic-chart-row"><div class="topic-chart-label" title="${esc(label)}">Cluster ${esc(c.id)} · ${esc(label)}</div><div class="topic-chart-track" title="${esc(metrics)}"><span class="topic-chart-bar coverage-missing" style="width:${width}%"></span></div><div class="mini">${esc(metrics)}</div></div>`;}).join('')}</div>`;}
function clusterCards(points){const clusters=clusterSummary(points);if(!clusters.length)return '<div class="empty">No semantic clusters available.</div>';const maxImpact=Math.max(1,...clusters.map(c=>Number(c.impactScore||0)));return '<div class="cluster-list">'+clusters.map(c=>`<div class="cluster"><strong>Cluster ${esc(c.id)} · ${n(c.total)} points</strong><div class="mini">${esc(ownDomain)} ${n(c.ours)} · Competitor ${n(c.competitor)} · Headers ${n(c.headers)} · Keywords ${n(c.keyword)}${c.centroid?` · Centroid ${n(c.centroid)}`:''}</div><div class="mini">Impact ${Number(c.impactTraffic||0)>0?`${n(c.impactTraffic)} traffic`:Number(c.impactImpressions||0)>0?`${n(c.impactImpressions)} impressions`:Number(c.impactVolume||0)>0?`vol ${n(c.impactVolume)}`:'unavailable'}</div><div class="bar"><span style="width:${Math.min(100,Math.round((Number(c.impactScore||0)/maxImpact)*100))}%"></span></div><div class="mini">${esc(c.samples.join(' / '))}</div></div>`).join('')+'</div>';}
function competitorList(rows){if(!rows||!rows.length)return '<div class="empty">No competitors fetched.</div>';return '<ol class="competitors">'+rows.map(c=>`<li>${urlLink(c.url,c.title||c.url)}<div class="mini">Rank ${esc(c.rank||'')} · Paragraphs ${esc(c.paragraph_count||0)}${c.error?' · '+esc(c.error):''}</div></li>`).join('')+'</ol>';}
function reviewList(rows){if(!rows||!rows.length)return `<div class="empty">No high-distance ${esc(ownDomain)} paragraphs were flagged for this keyword. This means the analyzed paragraphs stayed close enough to either the target keyword vector or the SERP topic space.</div>`;return '<ul class="review">'+rows.map(p=>`<li><b>${esc(p.similarity_to_serp_topics)}</b> <span class="mini">similarity · ${esc(p.review_reason||'review candidate')}</span><br>${esc(p.paragraph)}</li>`).join('')+'</ul>';}
function sharedKeywordNames(a,b){const aKeywords=new Set((a.keywords||[]).map(k=>String(k.keyword||'')));return (b.keywords||[]).map(k=>String(k.keyword||'')).filter(k=>aKeywords.has(k));}
function rankForKeyword(row,keyword){const match=(row.keywords||[]).find(k=>String(k.keyword||'')===String(keyword||''));return Number(match?.rank||0);}
function serpTrafficScore(a,b,keywords){if(!keywords.length)return 0;const total=keywords.reduce((sum,keyword)=>{const ar=rankForKeyword(a,keyword),br=rankForKeyword(b,keyword);return sum+Math.max(0,11-ar)+Math.max(0,11-br);},0);return total/(keywords.length*20);}
function trafficColor(score){const t=Math.max(0,Math.min(1,Number(score)||0));const low=[88,123,181],mid=[151,121,77],high=[194,91,30];const mix=(a,b,x)=>Math.round(a+(b-a)*x);const left=t<.5?low:mid;const right=t<.5?mid:high;const x=t<.5?t*2:(t-.5)*2;return `rgb(${mix(left[0],right[0],x)} ${mix(left[1],right[1],x)} ${mix(left[2],right[2],x)})`;}
function serpUrlGraph(rows){if(!rows||rows.length<2)return '<div class="empty">Not enough URLs for a co-ranking graph.</div>';const nodes=rows.slice(0,18).map(row=>({...row,domain:row.domain||urlDomain(row.url||'')||row.url})).sort((a,b)=>String(a.domain).localeCompare(String(b.domain))||Number(b.top10_count||0)-Number(a.top10_count||0));const edges=[];for(let i=0;i<nodes.length;i++){for(let j=i+1;j<nodes.length;j++){const shared=sharedKeywordNames(nodes[i],nodes[j]);if(shared.length){const traffic=serpTrafficScore(nodes[i],nodes[j],shared);edges.push({a:i,b:j,weight:shared.length,traffic,keywords:shared});}}}if(!edges.length)return '<div class="empty">No URLs share selected keywords.</div>';const w=920,h=560,cx=w/2,cy=h/2,leafR=205,domainR=118,bundleR=34;const maxCount=Math.max(1,...nodes.map(row=>Number(row.top10_count||0)));const domains=[...new Set(nodes.map(n=>n.domain))];const domainAngles=new Map(domains.map((domain,i)=>[domain,-Math.PI/2+(i/domains.length)*Math.PI*2]));const groups=new Map;nodes.forEach(node=>{const group=groups.get(node.domain)||[];group.push(node);groups.set(node.domain,group);});const positions=nodes.map((node,index)=>{const group=groups.get(node.domain)||[node];const groupIndex=group.indexOf(node);const base=domainAngles.get(node.domain)||0;const spread=Math.min(0.46,0.09*Math.max(group.length-1,0));const angle=base+(group.length<=1?0:-spread/2+(groupIndex/(group.length-1))*spread);return{index,angle,x:cx+Math.cos(angle)*leafR,y:cy+Math.sin(angle)*leafR,dx:cx+Math.cos(base)*domainR,dy:cy+Math.sin(base)*domainR,bx:cx+Math.cos(base)*bundleR,by:cy+Math.sin(base)*bundleR};});const edgeSvg=edges.map(edge=>{const a=positions[edge.a],b=positions[edge.b];const d=`M${a.x.toFixed(1)},${a.y.toFixed(1)} C${a.dx.toFixed(1)},${a.dy.toFixed(1)} ${a.bx.toFixed(1)},${a.by.toFixed(1)} ${cx},${cy} S${b.dx.toFixed(1)},${b.dy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;const title=`${nodes[edge.a].url} ↔ ${nodes[edge.b].url}: ${edge.weight} shared keyword${edge.weight===1?'':'s'} (${edge.keywords.join(', ')}); traffic proxy ${(edge.traffic*100).toFixed(0)}% from SERP positions`;return `<path class="graph-edge" d="${d}" stroke="${trafficColor(edge.traffic)}" stroke-width="${(0.9+edge.weight*1.5).toFixed(1)}"><title>${esc(title)}</title></path>`;}).join('');const domainMarks=domains.map(domain=>{const angle=domainAngles.get(domain)||0;const x=cx+Math.cos(angle)*domainR,y=cy+Math.sin(angle)*domainR;return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${domainColor(domain)}" opacity=".76"><title>${esc(domain)}</title></circle>`;}).join('');const nodeSvg=nodes.map((node,i)=>{const p=positions[i];const count=Number(node.top10_count||0);const size=7+Math.sqrt(count/maxCount)*10;const title=`${node.url}\\n${count} top-10 appearance${count===1?'':'s'}\\nBest rank #${node.best_rank||''}\\nKeywords: ${(node.keywords||[]).map(k=>`${k.keyword} #${k.rank}`).join(', ')}`;const labelX=p.x+(p.x>=cx?size+5:-size-5);const anchor=p.x>=cx?'start':'end';return `<a href="${esc(node.url||'#')}" target="_blank" rel="noopener noreferrer"><circle class="graph-node" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${size.toFixed(1)}" fill="${domainColor(node.domain)}"><title>${esc(title)}</title></circle></a><text class="graph-label" x="${labelX.toFixed(1)}" y="${(p.y-2).toFixed(1)}" text-anchor="${anchor}">${esc(node.domain).slice(0,26)}</text><text class="graph-meta" x="${labelX.toFixed(1)}" y="${(p.y+11).toFixed(1)}" text-anchor="${anchor}">${n(count)} top-10 · best #${esc(node.best_rank||'')}</text>`;}).join('');const defs='<defs><linearGradient id="trafficGradient" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="'+trafficColor(0.08)+'"></stop><stop offset="50%" stop-color="'+trafficColor(0.5)+'"></stop><stop offset="100%" stop-color="'+trafficColor(1)+'"></stop></linearGradient></defs>';const legend=`<g class="traffic-legend" transform="translate(26 ${h-34})"><text x="0" y="-8">Connection color: SERP-position traffic proxy</text><rect x="0" y="0" width="150" height="8" rx="4" fill="url(#trafficGradient)"></rect><text x="0" y="23">lower</text><text x="150" y="23" text-anchor="end">higher</text></g>`;return `<svg class="serp-url-graph" viewBox="0 0 ${w} ${h}" role="img" aria-label="Hierarchical edge bundling chart for URL co-ranking. Nodes are ranking URLs grouped by domain. Curved connections join URLs that rank for the same selected keywords; thicker lines mean more shared keywords and warmer colors mean stronger SERP-position traffic proxy."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${defs}${edgeSvg}${domainMarks}${nodeSvg}${legend}</svg><div class="mini">Bundled co-ranking graph: nodes are URLs grouped by domain; curved connections mean URLs rank for the same selected keyword; line width shows shared keyword count; warmer color means stronger SERP-position traffic proxy.</div>`;}
function serpRankingChart(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 URL chart available.</div>';const maxCount=Math.max(1,...rows.map(row=>Number(row.top10_count||0)));return `<div class="serp-ranking-chart" aria-label="Top-10 URL relevance chart">${rows.slice(0,12).map(row=>{const count=Number(row.top10_count||0);const width=Math.max(6,Math.round((count/maxCount)*100));const title=`${row.url} · ${count} top-10 appearances · best #${row.best_rank||''} · avg #${row.average_rank||''}`;return `<div class="serp-ranking-chart-row"><div class="serp-ranking-chart-label" title="${esc(row.url||'')}">${urlLink(row.url,row.domain||row.url)}</div><div class="serp-ranking-chart-track" title="${esc(title)}"><span class="serp-ranking-chart-bar" style="width:${width}%"></span></div><div class="serp-ranking-chart-meta">${n(count)} top-10 · best #${esc(row.best_rank||'')}</div></div>`;}).join('')}</div>`;}
function serpRankingList(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 SERP URLs available.</div>';return `<div class="serp-ranking-list">${rows.map((row,index)=>{const keywords=(row.keywords||[]).map(k=>`<span class="serp-ranking-chip"><strong>#${esc(k.rank)}</strong> ${esc(k.keyword)}</span>`).join('');return `<div class="serp-ranking-row"><div><div class="serp-ranking-url">${index+1}. ${urlLink(row.url)}</div><div class="serp-ranking-domain">${esc(row.domain||'')}${row.is_selected_domain?' · selected domain':''}</div></div><div class="serp-ranking-stats"><span class="chip covered">${n(row.top10_count)} top-10</span><span class="chip">Best #${esc(row.best_rank||'')}</span><span class="chip">Avg #${esc(row.average_rank||'')}</span></div><div class="serp-ranking-keywords">${keywords}</div></div>`;}).join('')}</div>`;}
function hasDemandMetrics(row){return Number(row?.impressions||0)>0||Number(row?.clicks||0)>0||Number(row?.traffic||0)>0||Number(row?.volume||0)>0;}
function keywordMetrics(k){const parts=[`#${esc(k.rank)}`];if(hasDemandMetrics(k)){if(Number(k.impressions||0)>0)parts.push(`${n(k.impressions)} impr`);if(Number(k.clicks||0)>0)parts.push(`${n(k.clicks)} clicks`);if(Number(k.traffic||0)>0)parts.push(`${n(k.traffic)} traffic`);if(Number(k.volume||0)>0)parts.push(`vol ${n(k.volume)}`);}else{parts.push('no demand metrics');}if(Number(k.source_position||0))parts.push(`source pos ${Number(k.source_position).toFixed(1)}`);if(k.source)parts.push(esc(k.source));return parts.join(' · ');}
function graphKeywordRows(keywords){return (keywords||[]).map(k=>`<div class="metric-line"><b>${esc(k.keyword)}</b> · ${keywordMetrics(k)}</div>`).join('');}
function demandMetricLine(row){if(hasDemandMetrics(row)){const parts=[];if(Number(row.impressions||0)>0)parts.push(`Impressions: ${n(row.impressions)}`);if(Number(row.clicks||0)>0)parts.push(`Clicks: ${n(row.clicks)}`);if(Number(row.traffic||0)>0)parts.push(`Traffic: ${n(row.traffic)}`);if(Number(row.volume||0)>0)parts.push(`Volume: ${n(row.volume)}`);return `<div class="metric-line">${parts.join(' · ')}</div>`;}return '<div class="metric-line">Demand metrics unavailable: this run used manual keywords and SERP suggestions without GSC/Ahrefs click, impression, traffic, or volume data.</div>';}
function aggregateDemandRows(rows){return (rows||[]).reduce((out,row)=>({impressions:Number(out.impressions||0)+Number(row.impressions||0),clicks:Number(out.clicks||0)+Number(row.clicks||0),traffic:Number(out.traffic||0)+Number(row.traffic||0),volume:Number(out.volume||0)+Number(row.volume||0)}),{});}
function graphTooltipHtml(detail){if(detail.kind==='connection'){return `<h5>Shared keyword connection</h5><div>${urlLink(detail.url_a,detail.domain_a)} ↔ ${urlLink(detail.url_b,detail.domain_b)}</div><div class="metric-line">SERP-position proxy: ${Math.round(Number(detail.traffic_proxy||0)*100)}% · Shared keywords: ${n(detail.shared_count||0)}</div>${demandMetricLine(aggregateDemandRows(detail.keywords))}${graphKeywordRows(detail.keywords)}`;}return `<h5>${esc(detail.domain||'URL')}</h5><div>${urlLink(detail.url)}</div><div class="metric-line">Top-10 keywords: ${n(detail.top10_count||0)} · Best SERP position: #${esc(detail.best_rank||'')} · Avg SERP position: #${esc(detail.average_rank||'')}</div>${demandMetricLine(detail)}${graphKeywordRows(detail.keywords)}`;}
function serpUrlGraph(rows){if(!rows||rows.length<2)return '<div class="empty">Not enough URLs for a co-ranking graph.</div>';const nodes=rows.slice(0,18).map(row=>({...row,domain:row.domain||urlDomain(row.url||'')||row.url})).sort((a,b)=>String(a.domain).localeCompare(String(b.domain))||Number(b.top10_count||0)-Number(a.top10_count||0));const edges=[];for(let i=0;i<nodes.length;i++){for(let j=i+1;j<nodes.length;j++){const shared=sharedKeywordNames(nodes[i],nodes[j]);if(shared.length){const traffic=serpTrafficScore(nodes[i],nodes[j],shared);const keywords=shared.map(keyword=>{const a=(nodes[i].keywords||[]).find(k=>String(k.keyword||'')===keyword)||{};const b=(nodes[j].keywords||[]).find(k=>String(k.keyword||'')===keyword)||{};return{keyword,rank:`${a.rank||''} / ${b.rank||''}`,impressions:Math.max(Number(a.impressions||0),Number(b.impressions||0)),clicks:Math.max(Number(a.clicks||0),Number(b.clicks||0)),traffic:Math.max(Number(a.traffic||0),Number(b.traffic||0)),source_position:Number(a.source_position||b.source_position||0),volume:Math.max(Number(a.volume||0),Number(b.volume||0))};});edges.push({a:i,b:j,weight:shared.length,traffic,keywords});}}}if(!edges.length)return '<div class="empty">No URLs share selected keywords.</div>';const w=920,h=560,cx=w/2,cy=h/2,leafR=205,domainR=118,bundleR=34;const maxCount=Math.max(1,...nodes.map(row=>Number(row.top10_count||0)));const domains=[...new Set(nodes.map(n=>n.domain))];const domainAngles=new Map(domains.map((domain,i)=>[domain,-Math.PI/2+(i/domains.length)*Math.PI*2]));const groups=new Map;nodes.forEach(node=>{const group=groups.get(node.domain)||[];group.push(node);groups.set(node.domain,group);});const positions=nodes.map((node,index)=>{const group=groups.get(node.domain)||[node];const groupIndex=group.indexOf(node);const base=domainAngles.get(node.domain)||0;const spread=Math.min(0.46,0.09*Math.max(group.length-1,0));const angle=base+(group.length<=1?0:-spread/2+(groupIndex/(group.length-1))*spread);return{index,angle,x:cx+Math.cos(angle)*leafR,y:cy+Math.sin(angle)*leafR,dx:cx+Math.cos(base)*domainR,dy:cy+Math.sin(base)*domainR,bx:cx+Math.cos(base)*bundleR,by:cy+Math.sin(base)*bundleR};});const edgeSvg=edges.map(edge=>{const a=positions[edge.a],b=positions[edge.b];const d=`M${a.x.toFixed(1)},${a.y.toFixed(1)} C${a.dx.toFixed(1)},${a.dy.toFixed(1)} ${a.bx.toFixed(1)},${a.by.toFixed(1)} ${cx},${cy} S${b.dx.toFixed(1)},${b.dy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;const detail={kind:'connection',url_a:nodes[edge.a].url,url_b:nodes[edge.b].url,domain_a:nodes[edge.a].domain,domain_b:nodes[edge.b].domain,shared_count:edge.weight,traffic_proxy:edge.traffic,keywords:edge.keywords};return `<path class="graph-edge" tabindex="0" data-graph-edge data-nodes=",${edge.a},${edge.b}," data-graph-detail="${esc(JSON.stringify(detail))}" d="${d}" stroke="${trafficColor(edge.traffic)}" stroke-width="${(0.9+edge.weight*1.5).toFixed(1)}"></path>`;}).join('');const domainMarks=domains.map(domain=>{const angle=domainAngles.get(domain)||0;const x=cx+Math.cos(angle)*domainR,y=cy+Math.sin(angle)*domainR;return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${domainColor(domain)}" opacity=".76"><title>${esc(domain)}</title></circle>`;}).join('');const nodeSvg=nodes.map((node,i)=>{const p=positions[i];const count=Number(node.top10_count||0);const size=7+Math.sqrt(count/maxCount)*10;const labelX=p.x+(p.x>=cx?size+5:-size-5);const anchor=p.x>=cx?'start':'end';const detail={kind:'url',url:node.url,domain:node.domain,top10_count:node.top10_count,best_rank:node.best_rank,average_rank:node.average_rank,impressions:node.impressions,clicks:node.clicks,traffic:node.traffic,keywords:node.keywords||[]};return `<a href="${esc(node.url||'#')}" target="_blank" rel="noopener noreferrer"><circle class="graph-node" tabindex="0" data-graph-node data-node-index="${i}" data-graph-detail="${esc(JSON.stringify(detail))}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${size.toFixed(1)}" fill="${domainColor(node.domain)}"></circle></a><text class="graph-label" x="${labelX.toFixed(1)}" y="${(p.y-2).toFixed(1)}" text-anchor="${anchor}">${esc(node.domain).slice(0,26)}</text><text class="graph-meta" x="${labelX.toFixed(1)}" y="${(p.y+11).toFixed(1)}" text-anchor="${anchor}">${n(count)} top-10 · best #${esc(node.best_rank||'')}</text>`;}).join('');const defs='<defs><linearGradient id="trafficGradient" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="'+trafficColor(0.08)+'"></stop><stop offset="50%" stop-color="'+trafficColor(0.5)+'"></stop><stop offset="100%" stop-color="'+trafficColor(1)+'"></stop></linearGradient></defs>';const legend=`<g class="traffic-legend" transform="translate(26 ${h-34})"><text x="0" y="-8">Connection color: SERP-position traffic proxy</text><rect x="0" y="0" width="150" height="8" rx="4" fill="url(#trafficGradient)"></rect><text x="0" y="23">lower</text><text x="150" y="23" text-anchor="end">higher</text></g>`;return `<div class="serp-url-graph-wrap"><svg class="serp-url-graph" viewBox="0 0 ${w} ${h}" role="img" aria-label="Hierarchical edge bundling chart for URL co-ranking. Nodes are ranking URLs grouped by domain. Curved connections join URLs that rank for the same selected keywords; thicker lines mean more shared keywords and warmer colors mean stronger SERP-position traffic proxy."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${defs}${edgeSvg}${domainMarks}${nodeSvg}${legend}</svg><div class="graph-tooltip" role="tooltip"></div></div><div class="mini">Bundled co-ranking graph: nodes are URLs grouped by domain; curved connections mean URLs rank for the same selected keyword; line width shows shared keyword count; warmer color means stronger SERP-position traffic proxy.</div>`;}
function serpRankingList(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 SERP URLs available.</div>';return `<div class="serp-ranking-list">${rows.map((row,index)=>{const keywords=(row.keywords||[]).map(k=>`<span class="serp-ranking-chip"><strong>#${esc(k.rank)}</strong> ${esc(k.keyword)} · ${keywordMetrics(k)}</span>`).join('');const demandChips=hasDemandMetrics(row)?`${Number(row.impressions||0)>0?`<span class="chip">${n(row.impressions)} impr</span>`:''}${Number(row.clicks||0)>0?`<span class="chip">${n(row.clicks)} clicks</span>`:''}${Number(row.traffic||0)>0?`<span class="chip">${n(row.traffic)} traffic</span>`:''}`:'<span class="chip">Demand metrics unavailable</span>';return `<div class="serp-ranking-row"><div><div class="serp-ranking-url">${index+1}. ${urlLink(row.url)}</div><div class="serp-ranking-domain">${esc(row.domain||'')}${row.is_selected_domain?' · selected domain':''}</div></div><div class="serp-ranking-stats"><span class="chip covered">${n(row.top10_count)} top-10</span><span class="chip">Best #${esc(row.best_rank||'')}</span><span class="chip">Avg #${esc(row.average_rank||'')}</span>${demandChips}</div><div class="serp-ranking-keywords">${keywords}</div></div>`;}).join('')}</div>`;}
function keywordId(pageIndex, keywordIndex){return `keyword-${pageIndex}-${keywordIndex}`;}
function overviewSection(){const points=data.overview_scatter?.points||[];const rankings=data.serp_url_rankings||[];return `<section class="page-section" id="overview-section"><div class="page-head"><h2>Keyword and Content Semantic Map</h2><div class="mini">All selected keywords, processed URLs, titles, headings, and paragraphs in one shared vector space.</div></div><div class="keyword-card"><div class="panel"><h4>Top-10 URLs Across Selected Keywords</h4><div class="panel-body">${sectionNote('Use this section to see which URLs repeatedly win across the selected keywords. The graph connects URLs that rank for the same keyword; stronger connections usually mean those URLs compete for the same search intent. The table adds the available SERP, impression, click, traffic, and rank metrics for each URL.')} ${serpUrlGraph(rankings)}${serpRankingChart(rankings)}${serpRankingList(rankings)}</div></div><div class="panel" style="margin-top:14px"><h4>All Keywords, URLs, and Content</h4><div class="panel-body">${sectionNote('This map places every selected keyword, ranking URL, page title, H1-H6 heading, and paragraph into one semantic space. Color identifies the domain, shape identifies the entity type, and size reflects available clicks or SERP position. The hexagon is a demand-weighted centroid of all selected keywords. Points near it are aligned with the combined keyword demand center.')} ${scatterSvg(points)}</div></div><div class="panel" style="margin-top:14px"><h4>Topic Traffic Impact</h4><div class="panel-body">${sectionNote('This chart estimates which aggregate semantic clusters carry the most keyword demand. Clusters are sorted by available keyword traffic first, then impressions, volume, or clicks when traffic is unavailable. Treat it as directional: it shows which topic groups deserve more attention before editing page text.')} ${clusterImpactChart(points)}</div></div><div class="panel" style="margin-top:14px"><h4>Aggregate Semantic Clusters</h4><div class="panel-body">${sectionNote('These clusters summarize the shared vector space above across all selected keywords and processed URLs. Use them to spot broad topic groups where competitors, headings, or selected-domain paragraphs dominate the aggregate map before drilling into individual keyword sections.')} ${clusterCards(points)}</div></div></div></section>`;}
function keywordCard(a, pageIndex, keywordIndex){const s=a.summary||{};const points=a.scatter?.points||[];const reviewRows=(a.off_intent_paragraphs&&a.off_intent_paragraphs.length)?a.off_intent_paragraphs:a.own_paragraphs_to_review;return `<div class="keyword-card" id="${keywordId(pageIndex, keywordIndex)}"><div class="keyword-head"><div><h3>${esc(a.query || a.keyword?.keyword || '')}</h3><div class="mini">Status ${esc(a.status)} · Competitors ${esc(a.competitors||a.competitor_pages?.length||0)} · Scatter points ${esc(a.scatter?.shown||0)}</div></div><div class="chips"><span class="chip missing">Missing ${n(s.missing||0)}</span><span class="chip partial">Partial ${n(s.partial||0)}</span><span class="chip covered">Covered ${n(s.covered||0)}</span></div></div><div class="keyword-grid"><div class="panel"><h4>Semantic Scatterplot</h4><div class="panel-body">${sectionNote('This chart compares the keyword with the analyzed title, headings, and paragraphs from each URL. Use it to spot whether the visible page structure clusters around the target keyword and whether competitor content covers nearby semantic territory that the selected domain does not.')} ${scatterSvg(points)}</div></div><div class="tables"><div class="panel"><h4>Semantic Clusters</h4><div class="panel-body">${sectionNote('Clusters group nearby vectors into themes. A cluster with many competitor points and few selected-domain points usually indicates a topic area competitors cover more deeply.')} ${clusterCards(points)}</div></div><div class="panel"><h4>Competitor SERP</h4><div class="panel-body">${sectionNote('These are the ranking competitor pages fetched from the SERP for this keyword after ignored hosts such as social/video platforms are skipped. Review them when validating missing topics or unusual scatter positions.')} ${competitorList(a.competitor_pages)}</div></div></div></div><div class="two-col" style="margin-top:14px"><div class="panel"><h4>Topic Relations</h4><div class="panel-body">${sectionNote('Topic relations summarize competitor paragraph themes and compare them with the nearest selected-domain paragraph. Missing means competitors share a topic that was not found close enough on the selected domain; partial means the selected domain is related but weaker or thinner than the SERP set.')} ${topicChart(a.topics)}</div><table><thead><tr><th>Coverage</th><th>Priority</th><th>Topic and Example</th><th>Seen</th><th>${esc(ownDomain)} sim</th><th>Example URL</th></tr></thead><tbody>${topicRows(a.topics,18)}</tbody></table></div><div class="panel"><h4>${esc(ownDomain)} Paragraphs To Review</h4><div class="panel-body">${sectionNote('These paragraphs are review candidates because their vectors are far from the target keyword and/or weakly connected to the SERP topic space. Do not remove them automatically: first check whether each paragraph supports a different necessary intent. If it should support this keyword, add clearer keyword-related context, connect it to a missing/partial topic, or move unrelated information to a better page.')} ${reviewList(reviewRows)}</div></div></div></div>`;}
function pageSection(page, index){return `<section class="page-section" id="page-${index}"><div class="page-head"><h2>${index+1}. ${esc(page.title || page.url)}</h2><div class="url">${urlLink(page.url)}</div>${page.h1?`<div class="mini">H1: ${esc(page.h1)}</div>`:''}</div>${(page.analyses||[]).map((analysis, keywordIndex)=>keywordCard(analysis, index, keywordIndex)).join('') || '<div class="empty">No keyword analyses for this page.</div>'}</section>`;}
function buildNav(){if(!navEl)return;navEl.innerHTML=`<button type="button" class="report-nav-button" data-target="overview-section"><span class="report-nav-label">Keyword and content map</span></button>`+(data.pages||[]).map((page,pageIndex)=>`<button type="button" class="report-nav-button" data-target="page-${pageIndex}"><span class="report-nav-label">${esc(page.title||page.url||`Page ${pageIndex+1}`)}</span></button>${(page.analyses||[]).map((analysis,keywordIndex)=>`<button type="button" class="report-nav-button nav-keyword" data-target="${keywordId(pageIndex,keywordIndex)}"><span class="report-nav-label">${esc(analysis.query||analysis.keyword?.keyword||`Keyword ${keywordIndex+1}`)}</span></button>`).join('')}`).join('');const buttons=[...navEl.querySelectorAll('.report-nav-button')];const sections=buttons.map(button=>document.getElementById(button.dataset.target||'')).filter(Boolean);buttons.forEach(button=>button.addEventListener('click',()=>{const target=document.getElementById(button.dataset.target||'');if(target)target.scrollIntoView({block:'start',behavior:'smooth'});}));function update(){let active=0;for(let i=0;i<sections.length;i++){if(sections[i].getBoundingClientRect().top<160)active=i;}buttons.forEach((button,i)=>{const selected=i===active;button.classList.toggle('is-active',selected);button.setAttribute('aria-current',selected?'page':'false');});}document.addEventListener('scroll',update,{passive:true});update();}

function bindSerpUrlGraphInteractions(){document.querySelectorAll('.serp-url-graph-wrap').forEach(wrap=>{const tooltip=wrap.querySelector('.graph-tooltip');const clear=()=>{wrap.classList.remove('has-active');wrap.querySelectorAll('.is-active').forEach(el=>el.classList.remove('is-active'));tooltip?.classList.remove('open');};function show(el,event){let detail={};try{detail=JSON.parse(el.getAttribute('data-graph-detail')||'{}');}catch(_){}wrap.classList.add('has-active');wrap.querySelectorAll('.is-active').forEach(active=>active.classList.remove('is-active'));el.classList.add('is-active');if(el.matches('[data-graph-node]')){const index=el.getAttribute('data-node-index');wrap.querySelectorAll(`[data-graph-edge][data-nodes*=",${index},"]`).forEach(edge=>edge.classList.add('is-active'));}else if(el.matches('[data-graph-edge]')){(el.getAttribute('data-nodes')||'').split(',').filter(Boolean).forEach(index=>wrap.querySelector(`[data-graph-node][data-node-index="${index}"]`)?.classList.add('is-active'));}if(tooltip){tooltip.innerHTML=graphTooltipHtml(detail);tooltip.classList.add('open');position(event);}}function position(event){if(!tooltip||!event)return;const rect=wrap.getBoundingClientRect();const x=Math.min(Math.max(10,event.clientX-rect.left+14),Math.max(10,rect.width-380));const y=Math.min(Math.max(10,event.clientY-rect.top+14),Math.max(10,rect.height-190));tooltip.style.left=`${x}px`;tooltip.style.top=`${y}px`;}wrap.querySelectorAll('[data-graph-node],[data-graph-edge]').forEach(el=>{el.addEventListener('mouseenter',event=>show(el,event));el.addEventListener('mousemove',position);el.addEventListener('focus',event=>show(el,event));el.addEventListener('mouseleave',clear);el.addEventListener('blur',clear);});});}

function bindScatterInteractions(){document.querySelectorAll('.scatter-wrap').forEach(wrap=>{const svg=wrap.querySelector('svg.scatter');const tooltip=wrap.querySelector('.scatter-tooltip');if(!svg||!tooltip)return;const base=(svg.getAttribute('data-base-viewbox')||'0 0 820 390').split(/\\s+/).map(Number);let vb={x:base[0],y:base[1],w:base[2],h:base[3]};const setVb=()=>svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);function zoomAt(factor,cx=base[2]/2,cy=base[3]/2){const nx=cx-(cx-vb.x)*factor;const ny=cy-(cy-vb.y)*factor;vb={x:nx,y:ny,w:vb.w*factor,h:vb.h*factor};setVb();}function pointFromEvent(event){const rect=svg.getBoundingClientRect();return{x:vb.x+(event.clientX-rect.left)/Math.max(rect.width,1)*vb.w,y:vb.y+(event.clientY-rect.top)/Math.max(rect.height,1)*vb.h};}function show(point){let detail=null;try{detail=JSON.parse(point.getAttribute('data-detail')||'{}');}catch(_){detail={type:'point',explanation:point.getAttribute('data-tooltip')||point.getAttribute('aria-label')||''};}tooltip.innerHTML=pointDetailHtml(detail);tooltip.classList.add('open');tooltip.querySelector('.tip-close')?.addEventListener('click',event=>{event.stopPropagation();tooltip.classList.remove('open');});}wrap.querySelectorAll('.scatter-point').forEach(point=>{point.addEventListener('click',event=>{event.stopPropagation();show(point);});point.addEventListener('focus',()=>show(point));});svg.addEventListener('wheel',event=>{event.preventDefault();const p=pointFromEvent(event);zoomAt(event.deltaY<0?0.82:1.22,p.x,p.y);},{passive:false});let drag=null;svg.addEventListener('mousedown',event=>{if(event.target.classList?.contains('scatter-point'))return;drag={x:event.clientX,y:event.clientY,vx:vb.x,vy:vb.y};svg.classList.add('is-panning');});window.addEventListener('mousemove',event=>{if(!drag)return;const rect=svg.getBoundingClientRect();vb.x=drag.vx-(event.clientX-drag.x)/Math.max(rect.width,1)*vb.w;vb.y=drag.vy-(event.clientY-drag.y)/Math.max(rect.height,1)*vb.h;setVb();});window.addEventListener('mouseup',()=>{drag=null;svg.classList.remove('is-panning');});svg.addEventListener('dblclick',()=>{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();});wrap.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',event=>{event.stopPropagation();const action=button.getAttribute('data-zoom');if(action==='in')zoomAt(0.78);else if(action==='out')zoomAt(1.28);else{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();}}));});document.addEventListener('click',event=>{const target=event.target;if(target?.closest?.('.scatter-tooltip')||target?.closest?.('.scatter-point'))return;document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});}
function bindScatterFilters(){document.querySelectorAll('.panel-body').forEach(panel=>{const entityFilters=[...panel.querySelectorAll('[data-entity-filter]')];const domainFilters=[...panel.querySelectorAll('[data-domain-filter]')];const filters=[...entityFilters,...domainFilters];const points=[...panel.querySelectorAll('.scatter-point')];const tooltip=panel.querySelector('.scatter-tooltip');if(!filters.length||!points.length)return;function apply(){const visibleEntities=new Set(entityFilters.filter(input=>input.checked).map(input=>input.dataset.entityFilter));const visibleDomains=new Set(domainFilters.filter(input=>input.checked).map(input=>input.dataset.domainFilter));points.forEach(point=>{const entityVisible=visibleEntities.has(point.dataset.entity||'paragraph');const domain=point.dataset.domain||'';const domainVisible=!domain||visibleDomains.has(domain);const show=entityVisible&&domainVisible;point.style.display=show?'':'none';if(!show&&document.activeElement===point)point.blur();});tooltip?.classList.remove('open');}filters.forEach(input=>input.addEventListener('change',apply));apply();});}
overviewEl.innerHTML = overviewSection();
app.innerHTML = (data.pages||[]).map(pageSection).join('') || '<div class="empty">No analyzed pages in this report.</div>';
buildNav();
bindSerpUrlGraphInteractions();
bindScatterInteractions();
bindScatterFilters();
</script>
</body>
</html>
"""
    return template.replace("__DATA__", data)


def _page_payload(page: PageInfo) -> dict:
    return {
        "url": page.url,
        "title": page.title,
        "section": page.section,
        "word_count": page.word_count,
        "language": page.language,
    }


def _same_url(a: str, b: str) -> bool:
    def norm(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        path = (parsed.path or "/").rstrip("/") or "/"
        return f"{parsed.netloc.lower().removeprefix('www.')}{path}"
    return norm(a) == norm(b)


def _is_own_url(url: str, domain: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    own = urlparse(domain if "://" in domain else f"https://{domain}").netloc.lower().removeprefix("www.")
    return host == own or host.endswith("." + own)


def _safe_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
