"""Standalone SERP semantic content-gap workflow.

This command intentionally lives outside the main audit pipeline. It starts
from an existing project report, expands selected pages into SERP competitors,
uses the domain cache under ``projects/<domain>/cache/serp_gap``, and writes an
independent report under ``projects/<domain>/serp_gap/report``.
"""

from __future__ import annotations

import copy
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

from .agent_workspace import agent_evidence_payload, write_agent_workspace
from .ai_agent import (
    DEFAULT_OPENROUTER_MODEL,
    MissingOpenRouterKey,
    RECOMMENDATION_SCHEMA_DOC,
    cached_workspace_completion,
    parse_recommendation,
    run_harnext_workspace_session,
    validate_recommendation,
    build_agent_client,
    build_editor_brief_messages,
    build_keyword_messages,
    build_language_detection_messages,
    cached_completion,
    fallback_keyword_candidates,
    harnext_status,
    openrouter_api_key,
    parse_language_detection,
    parse_keyword_candidates,
)
from .analyzer import PageInfo, section_for_url
from .cache import HttpCache, content_hash, domain_slug
from .draft_verification import assemble_recommended_blocks, verify_numeric_claims, verify_recommendation
from .answerability import score_page
from .competitive_analysis import (
    CompetitiveTarget,
    CompetitorPage,
    _serp_items,
    build_serp_paragraph_gap,
    structural_diff,
)
from .competitive_analysis import _fetch_dataforseo_serp as fetch_dataforseo_serp
from .competitive_analysis import CompetitiveAutoConfig
from .ahrefs import (
    AHREFS_DOMAIN_RATING_ATTRIBUTION,
    AHREFS_DOMAIN_RATING_LICENSE,
    AhrefsConfig,
    build_analysis as build_ahrefs_analysis,
    fetch_domain_ratings_free,
    fetch_snapshot as fetch_ahrefs_snapshot,
)
from .embedder import DEFAULT_MODEL, Embedder
from .extractor import ExtractedPage, extract
from .page_types import classify_page
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

WINNABILITY_FACTORS = {
    "winnable": 1.0,
    "hard": 0.6,
    "unlikely": 0.25,
    "unknown": 1.0,
}

# Impact scores for the gating actions that must lead every action list.
RETARGET_ACTION_IMPACT = 10000.0
WINNABILITY_GATE_ACTION_IMPACT = 9000.0

# A top-10 result at or below this absolute DR counts as a low-authority
# ("weak") result: if a page this weak ranks, the SERP is reachable.
WEAK_RESULT_DR_MAX = 30.0

# Registrable UGC/forum domains (matched exactly or as a suffix) plus host
# labels that mark community properties (matched as whole labels only, so
# "performance-community.io" does not trip on "community").
_WEAK_RESULT_UGC_DOMAINS = {
    "reddit.com",
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
}

_WEAK_RESULT_HOST_LABELS = {
    "forum",
    "forums",
    "community",
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
    ai_agent: bool = True
    ai_agent_provider: str = "harnext"
    ai_agent_model: str = DEFAULT_OPENROUTER_MODEL
    ai_agent_refresh: bool = False
    ai_agent_max_turns: int = 20


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    own_cache = HttpCache(project_dir / "cache" / "http.sqlite")
    ai_agent = _ai_agent_state(config)

    selected_pages, skipped_pages = _select_pages(pages, config)
    report_root = project_dir / "serp_gap" / "report"
    report_dir = _run_report_dir(report_root, selected_pages, config)
    report_dir.mkdir(parents=True, exist_ok=True)
    search_payload = _load_search_payload(base_report_dir)
    ahrefs_payload = {}
    if (config.use_ahrefs_metrics or config.keyword_source == "ahrefs") and not config.dry_run:
        ahrefs_payload = _load_or_fetch_ahrefs_payload(config.domain, project_dir, pages, config)
        search_payload = _merge_search_payloads(search_payload, ahrefs_payload)
    page_type_lookup = _load_page_type_lookup(base_report_dir)
    language_info = _resolve_serp_language(
        selected_pages,
        search_payload,
        own_cache,
        cache_dir,
        config,
        ai_agent,
    )
    keyword_metrics = _keyword_metrics_lookup(search_payload, ahrefs_payload)
    manual_keywords = _load_manual_keywords(config)
    winnability_lookup = _load_winnability_cache(cache_dir)
    keyword_rows, skipped_keywords = _select_keywords(
        selected_pages,
        search_payload,
        manual_keywords,
        config,
        winnability_lookup,
    )
    ai_keyword_rows, ai_keyword_skips = _ai_agent_keyword_rows(
        selected_pages,
        keyword_rows,
        search_payload,
        own_cache,
        cache_dir,
        config,
        ai_agent,
    )
    if ai_keyword_rows:
        generated_urls = {row.get("url", "") for row in ai_keyword_rows}
        skipped_keywords = [
            row for row in skipped_keywords
            if not (
                row.get("reason") == "no ranking keywords"
                and any(_same_url(row.get("url", ""), url) for url in generated_urls)
            )
        ]
        keyword_rows.extend(ai_keyword_rows)
    skipped_keywords.extend(ai_keyword_skips)
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
            "language_detection": language_info,
            "ai_agent": _ai_agent_payload(ai_agent, []),
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
            "language_detection": language_info,
            "ai_agent": _ai_agent_payload(ai_agent, []),
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
    competitor_cache = HttpCache(cache_dir / "competitors.sqlite")

    page_results: list[dict] = []
    page_competitor_content: dict[str, dict[str, dict]] = {}
    own_exts: dict[str, ExtractedPage] = {}
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

        own_paragraphs_for_page = (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
        own_embeddings_for_page = (
            embedder.encode(own_paragraphs_for_page, batch_size=64).astype(np.float32)
            if own_paragraphs_for_page
            else np.zeros((0, 0), dtype=np.float32)
        )
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
                _competitor_page(target, competitor_cache, embedder, config, own_ext=own_ext)
                for target in targets
            ]
            content_by_url = page_competitor_content.setdefault(page.url, {})
            for cp in competitor_pages:
                if cp.error or cp.target.competitor_url in content_by_url:
                    continue
                content_by_url[cp.target.competitor_url] = {
                    "rank": cp.target.rank,
                    "title": cp.title,
                    "h1": cp.h1,
                    "headings": cp.headers_rich,
                    "paragraphs": cp.paragraphs,
                }
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
            gap = _build_gap(
                page,
                kw,
                own_ext,
                competitor_pages,
                embedder,
                config,
                own_paragraphs=own_paragraphs_for_page,
                own_embeddings=own_embeddings_for_page,
            )
            features = _serp_features(serp, config.domain)
            gap["serp_features"] = features
            gap["paa_coverage"] = _paa_coverage(features, own_paragraphs_for_page, own_embeddings_for_page, embedder)
            serp_rows = _serp_result_rows(serp)
            gap["intent"] = _intent_assessment(
                kw,
                own_ext,
                page_type_lookup.get(_metric_url_key(page.url), {}),
                serp_rows,
                features,
                competitor_pages,
            )
            gap["serp"] = {
                "provider": provider,
                "cache_status": serp_meta.get("cache_status", ""),
                "top10": [
                    {
                        "url": str(row.get("url") or ""),
                        "domain": urlparse(str(row.get("url") or "")).netloc,
                        "rank": _safe_int(row.get("rank")),
                        "title": str(row.get("title") or ""),
                    }
                    for row in serp_rows
                    if 0 < _safe_int(row.get("rank")) <= 10
                ],
                "targets": [
                    {"url": t.competitor_url, "rank": t.rank}
                    for t in targets
                ],
            }
            page_blocks.append(gap)

        own_exts[page.url] = own_ext
        page_results.append({
            "url": page.url,
            "title": page.title,
            "h1": own_ext.h1,
            "keywords": page_keywords,
            "own_content": {
                "headings": [
                    {"order": i, "level": _safe_int(h.get("level")), "text": str(h.get("text") or "").strip()}
                    for i, h in enumerate(own_ext.headers_rich or [])
                    if str(h.get("text") or "").strip()
                ][:60],
                "paragraphs": [
                    {"index": i, "word_count": len(p.split()), "text": p[:500]}
                    for i, p in enumerate(own_paragraphs_for_page)
                ],
                "word_count": own_ext.word_count,
            },
            "analyses": page_blocks,
        })

    domain_rating_meta = _enrich_serp_domain_ratings(
        config.domain,
        serp_url_rankings,
        page_results,
        overview_rows,
        cache_dir,
        refresh=config.refresh_serp,
    )
    _attach_winnability(page_results, keyword_rows, domain_rating_meta.get("own_domain_rating"))
    _save_winnability_cache(cache_dir, keyword_rows)
    aggregate_action_points = _attach_action_points(page_results)
    _attach_ai_editor_briefs(
        page_results,
        cache_dir,
        config,
        ai_agent,
        page_competitor_content=page_competitor_content,
        own_exts=own_exts,
        report_dir=report_dir,
        embedder=embedder,
    )
    payload = {
        "status": "ok",
        "domain": config.domain,
        "provider": provider,
        "summary": _summary(page_results, selected_pages, keyword_rows, plan),
        "selected_pages": [_page_payload(p) for p in selected_pages],
        "selected_keywords": keyword_rows,
        "skipped_pages": skipped_pages,
        "skipped_keywords": skipped_keywords,
        "language_detection": language_info,
        "ai_agent": _ai_agent_payload(ai_agent, page_results),
        "ahrefs": {
            "meta": ahrefs_payload.get("meta", {}) if ahrefs_payload else {},
            "summary": ahrefs_payload.get("summary", {}) if ahrefs_payload else {},
        },
        "domain_ratings": domain_rating_meta,
        "serp_url_rankings": _serp_url_ranking_rows(serp_url_rankings),
        "overview_scatter": _overview_scatter(overview_rows, overview_texts, embedder),
        "action_points": aggregate_action_points,
        "content_briefs": [p.get("content_brief") for p in page_results if p.get("content_brief")],
        "editorial_guidelines": _editorial_guidelines(),
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


def _load_page_type_lookup(report_dir: Path) -> dict[str, dict]:
    path = report_dir / "page_types.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, dict] = {}
    for row in payload.get("per_page") or []:
        url = str(row.get("url") or "")
        if url:
            out[_metric_url_key(url)] = row
    return out


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
    if config.urls:
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


def _url_report_slug(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path = (parsed.path or "/").strip("/") or "home"
    if parsed.query:
        path = f"{path}-{parsed.query}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower() or "home"
    return slug[:90].strip("-") or "home"


def _run_report_dir(report_root: Path, selected_pages: list[PageInfo], config: SerpGapConfig) -> Path:
    explicit_urls = [url for url in config.urls if str(url or "").strip()]
    if len(explicit_urls) == 1:
        return report_root / _url_report_slug(explicit_urls[0])
    if len(explicit_urls) > 1:
        seed = "\n".join(sorted(explicit_urls))
        return report_root / f"multi-url-{content_hash(seed)[:10]}"
    urls = [page.url for page in selected_pages if page.url]
    if len(urls) == 1:
        return report_root / _url_report_slug(urls[0])
    if urls:
        seed = "\n".join(sorted(urls))
        return report_root / f"multi-url-{content_hash(seed)[:10]}"
    scope_parts = [*config.urls, *config.url_include_patterns, *config.keywords]
    seed = "\n".join(scope_parts) or config.domain
    return report_root / f"selection-{content_hash(seed)[:10]}"


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


def _ai_agent_state(config: SerpGapConfig) -> dict:
    state = {
        "enabled": bool(config.ai_agent),
        "provider": config.ai_agent_provider,
        "model": config.ai_agent_model,
        "status": "disabled",
        "language_prompts": 0,
        "keyword_prompts": 0,
        "keyword_fallbacks": 0,
        "editor_briefs": 0,
        "cache_hits": 0,
        "detected_language": "",
        "notes": [],
        "errors": [],
    }
    if not config.ai_agent:
        return state
    if config.dry_run:
        state["status"] = "dry_run"
        state["notes"].append("AI agent calls are skipped during --dry-run; deterministic title/H1 fallbacks may still be used.")
        return state
    if not openrouter_api_key():
        state["status"] = "missing_openrouter_api_key"
        state["notes"].append("Set OPENROUTER_API_KEY in .env to enable AI-authored keyword selection and editor briefs.")
        return state
    if str(config.ai_agent_provider or "").lower() == "harnext":
        ok, detail = harnext_status()
        if not ok:
            config.ai_agent_provider = "openrouter"
            state["provider"] = "openrouter"
            state["notes"].append(
                f"Harnext unavailable, falling back to direct OpenRouter with the same recommendation contract. ({detail})"
            )
        else:
            state["notes"].append(f"Harnext CLI: {detail}")
    state["status"] = "ready"
    return state


def _ai_agent_payload(state: dict, page_results: list[dict]) -> dict:
    payload = {k: v for k, v in state.items() if not k.startswith("_")}
    payload["editor_briefs"] = sum(1 for page in page_results if (page.get("ai_editor_brief") or {}).get("status") == "ok")
    payload["editor_brief_errors"] = sum(1 for page in page_results if (page.get("ai_editor_brief") or {}).get("status") == "error")
    return payload


def _resolve_serp_language(
    pages: list[PageInfo],
    search_payload: dict,
    own_cache: HttpCache,
    cache_dir: Path,
    config: SerpGapConfig,
    state: dict,
) -> dict:
    explicit = _normalize_serp_language(config.language)
    if explicit:
        config.language = explicit
        state["detected_language"] = explicit
        return {
            "status": "explicit",
            "language": explicit,
            "source": "user",
            "message": "SERP language was provided by the user.",
        }
    config.language = None
    evidence = _language_detection_evidence(
        pages,
        list(search_payload.get("organic_keywords") or []),
        own_cache,
        fetch_pages=not config.dry_run,
    )
    fallback = _fallback_language_from_evidence(evidence)
    if config.ai_agent and state.get("status") == "ready":
        try:
            client = build_agent_client(config.ai_agent_provider)
            completion = cached_completion(
                cache_dir,
                kind=f"language-detection-{_language_detection_key(pages)}",
                messages=build_language_detection_messages(evidence),
                client=client,
                model=config.ai_agent_model,
                refresh=config.ai_agent_refresh,
                temperature=0.0,
                timeout=90,
            )
            state["language_prompts"] += 1
            if completion.cache_status == "hit":
                state["cache_hits"] += 1
            detected = parse_language_detection(completion.text)
            language = _normalize_serp_language(detected.get("language_code"))
            if language:
                config.language = language
                state["detected_language"] = language
                return {
                    "status": "detected",
                    "language": language,
                    "language_name": detected.get("language_name", ""),
                    "confidence": detected.get("confidence", 0.0),
                    "reason": detected.get("reason", ""),
                    "source": completion.provider,
                    "cache_status": completion.cache_status,
                    "fallback_language": fallback,
                }
            state.setdefault("errors", []).append("Language detection returned no valid language code.")
        except MissingOpenRouterKey:
            state["status"] = "missing_openrouter_api_key"
            state.setdefault("notes", []).append("Language detection skipped: missing OpenRouter key.")
        except Exception as exc:
            state.setdefault("errors", []).append(f"Language detection failed: {exc}")

    if fallback:
        config.language = fallback
        state["detected_language"] = fallback
        return {
            "status": "fallback",
            "language": fallback,
            "source": "page_html_lang",
            "message": "AI language detection was unavailable; used the audited or extracted page language.",
        }
    return {
        "status": "not_detected",
        "language": "",
        "source": "",
        "message": "No SERP language was provided or detected; provider defaults apply.",
    }


def _language_detection_evidence(
    pages: list[PageInfo],
    search_rows: list[dict],
    own_cache: HttpCache,
    *,
    fetch_pages: bool = True,
) -> dict:
    evidence_pages = []
    existing_language_codes: list[str] = []
    compact_search_rows = []
    for page in pages[:5]:
        extracted = _fetch_and_extract(page.url, own_cache, refresh=False) if fetch_pages else None
        evidence = _agent_page_evidence(page, search_rows, extracted)
        language = _normalize_serp_language(evidence.get("language"))
        if language and language not in existing_language_codes:
            existing_language_codes.append(language)
        evidence_pages.append({
            "url": evidence.get("url", ""),
            "title": evidence.get("title", ""),
            "h1": evidence.get("h1", ""),
            "description": evidence.get("description", ""),
            "language": language,
            "headers": (evidence.get("headers") or [])[:12],
            "paragraphs": (evidence.get("paragraphs") or [])[:8],
        })
        compact_search_rows.extend((evidence.get("search_rows") or [])[:8])
    return {
        "pages": evidence_pages,
        "existing_language_codes": existing_language_codes,
        "search_rows": compact_search_rows[:20],
    }


def _fallback_language_from_evidence(evidence: dict) -> str:
    counts: dict[str, int] = {}
    for code in evidence.get("existing_language_codes") or []:
        language = _normalize_serp_language(code)
        if language:
            counts[language] = counts.get(language, 0) + 2
    for page in evidence.get("pages") or []:
        language = _normalize_serp_language((page or {}).get("language"))
        if language:
            counts[language] = counts.get(language, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _language_detection_key(pages: list[PageInfo]) -> str:
    seed = "\n".join(page.url for page in pages[:10] if page.url) or "selection"
    if len(pages) == 1:
        return _url_report_slug(pages[0].url)
    return content_hash(seed)[:12]


def _normalize_serp_language(value) -> str:
    code = re.sub(r"[^a-zA-Z0-9_-]+", "", str(value or "").strip()).replace("_", "-").lower()
    if not code:
        return ""
    parts = [part for part in code.split("-") if part]
    if not parts:
        return ""
    code = parts[0]
    if not re.fullmatch(r"[a-z]{2,3}", code):
        return ""
    return code


def _ai_agent_keyword_rows(
    pages: list[PageInfo],
    existing_rows: list[dict],
    search_payload: dict,
    own_cache: HttpCache,
    cache_dir: Path,
    config: SerpGapConfig,
    state: dict,
) -> tuple[list[dict], list[dict]]:
    if not config.ai_agent:
        return [], []
    out: list[dict] = []
    skipped: list[dict] = []
    search_rows = list(search_payload.get("organic_keywords") or [])
    client = None
    if state.get("status") == "ready":
        try:
            client = build_agent_client(config.ai_agent_provider)
        except Exception as exc:
            state["status"] = "error"
            state.setdefault("errors", []).append(f"AI agent client init failed: {exc}")
    for page in pages:
        if any(_same_url(row.get("url", ""), page.url) for row in existing_rows):
            continue
        evidence = _agent_page_evidence(page, search_rows)
        source = "ai_agent_fallback"
        keywords: list[str] = []
        if client is not None:
            extracted = _fetch_and_extract(page.url, own_cache, refresh=False)
            if extracted is not None:
                evidence = _agent_page_evidence(page, search_rows, extracted)
            messages = build_keyword_messages(evidence, max_keywords=config.keywords_per_page)
            try:
                completion = cached_completion(
                    cache_dir,
                    kind=f"keyword-selection-{_url_report_slug(page.url)}",
                    messages=messages,
                    client=client,
                    model=config.ai_agent_model,
                    refresh=config.ai_agent_refresh,
                    temperature=0.1,
                    timeout=120,
                )
                state["keyword_prompts"] += 1
                if completion.cache_status == "hit":
                    state["cache_hits"] += 1
                if completion.fallback_from:
                    state.setdefault("notes", []).append(
                        f"Keyword selection for {page.url} used {completion.provider} after {completion.fallback_from}."
                    )
                keywords = parse_keyword_candidates(completion.text, limit=config.keywords_per_page)
                source = "ai_agent"
            except MissingOpenRouterKey:
                state["status"] = "missing_openrouter_api_key"
                state.setdefault("notes", []).append(f"Keyword selection skipped for {page.url}: missing OpenRouter key.")
            except Exception as exc:
                skipped.append({"url": page.url, "reason": f"AI keyword selection failed: {exc}"})
                state.setdefault("errors", []).append(f"Keyword selection failed for {page.url}: {exc}")
        if not keywords:
            keywords = fallback_keyword_candidates(evidence, limit=config.keywords_per_page)
            source = "ai_agent_fallback"
            state["keyword_fallbacks"] += 1
        if not keywords:
            skipped.append({"url": page.url, "reason": "AI agent could not infer target keywords"})
            continue
        for keyword in keywords[:config.keywords_per_page]:
            row = _keyword_row(page, keyword, source, synthetic=True)
            if source == "ai_agent_fallback":
                row["metrics_source"] = "fallback title/H1 suggestion; no API demand metric match"
            else:
                row["metrics_source"] = "AI agent suggestion; no API demand metric match"
            out.append(row)
    return _dedupe_keywords(out), skipped


def _agent_page_evidence(page: PageInfo, search_rows: list[dict], extracted: ExtractedPage | None = None) -> dict:
    matched_rows = []
    for row in search_rows:
        matched = row.get("matched_url") or row.get("url") or ""
        if not matched or not _same_url(matched, page.url):
            continue
        matched_rows.append({
            "keyword": row.get("keyword") or row.get("query") or "",
            "provider": row.get("provider", ""),
            "position": row.get("position", 0),
            "impressions": row.get("impressions", 0),
            "clicks": row.get("clicks", 0),
            "traffic": row.get("traffic", 0),
            "volume": row.get("volume", 0),
        })
    if extracted is None:
        return {
            "url": page.url,
            "title": page.title,
            "h1": "",
            "description": page.description,
            "section": page.section,
            "language": _normalize_serp_language(page.language),
            "headers": [],
            "paragraphs": [],
            "search_rows": matched_rows,
        }
    headers = [
        f"H{header.get('level')}: {header.get('text')}"
        for header in (extracted.headers_rich or [])[:30]
        if str(header.get("text") or "").strip()
    ]
    return {
        "url": page.url,
        "title": extracted.title or page.title,
        "h1": extracted.h1,
        "description": extracted.description or page.description,
        "section": page.section,
        "language": _normalize_serp_language(extracted.language or page.language),
        "headers": headers,
        "paragraphs": [p for p in extracted.paragraphs[:18] if str(p).strip()],
        "search_rows": matched_rows,
    }


def _select_keywords(
    pages: list[PageInfo],
    search_payload: dict,
    manual_keywords: dict[str, list[str]],
    config: SerpGapConfig,
    winnability_lookup: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    # Winnability comes from SERP DR data gathered later in the same run, so a
    # first run selects with the neutral factor 1.0 for every keyword. Bands
    # persisted by previous runs (keyword_winnability.json) re-rank selection
    # from the second run onward.
    rows = []
    skipped = []
    search_rows = list(search_payload.get("organic_keywords") or [])
    for page in pages:
        candidates: list[dict] = []
        for kw in manual_keywords.get("*", []):
            candidates.append(_keyword_row(page, kw, "manual", synthetic=False, winnability_lookup=winnability_lookup))
        for url_key, kws in manual_keywords.items():
            if url_key == "*":
                continue
            if _same_url(url_key, page.url) or _pattern_match(page.url, url_key):
                for kw in kws:
                    candidates.append(_keyword_row(page, kw, "file", synthetic=False, winnability_lookup=winnability_lookup))
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
                candidates.append(_keyword_row(page, keyword, provider, row=row, winnability_lookup=winnability_lookup))
        if config.use_h1_keyword or config.keyword_source == "h1":
            title = page.title.strip()
            if title:
                candidates.append(_keyword_row(page, title, "h1", synthetic=True, winnability_lookup=winnability_lookup))
        candidates = _dedupe_keywords(candidates)
        candidates.sort(key=_keyword_priority, reverse=True)
        if not candidates:
            skipped.append({"url": page.url, "reason": "no ranking keywords"})
        rows.extend(candidates[:config.keywords_per_page])
    return rows, skipped


def _keyword_row(
    page: PageInfo,
    keyword: str,
    source: str,
    row: dict | None = None,
    synthetic: bool = False,
    winnability_lookup: dict[str, dict] | None = None,
) -> dict:
    row = row or {}
    out = {
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
        "intents": list(row.get("intents") or []),
    }
    cached = (winnability_lookup or {}).get(_winnability_cache_key(page.url, keyword)) or {}
    if cached.get("band"):
        out["winnability_band"] = str(cached.get("band"))
        out["winnability_factor"] = _safe_float(cached.get("factor")) or WINNABILITY_FACTORS.get(str(cached.get("band")), 1.0)
    return out


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
    return source_weight * (np.log1p(demand) + opportunity) * _keyword_winnability_factor(row)


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
        language = _normalize_serp_language(config.language)
        if language:
            auto_config.language_code = language
        return fetch_dataforseo_serp(keyword, cache_dir / "serp_dataforseo", auto_config)
    if provider != "serper":
        return {"meta": {"status": "unsupported_provider", "message": provider}, "raw": {}}
    key = content_hash(keyword.lower())
    country = config.country or "us"
    language = _normalize_serp_language(config.language) or "auto"
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
        json={
            **{"q": keyword, "gl": country, "num": max(config.results_per_keyword * 3, config.results_per_keyword + 10)},
            **({"hl": language} if language != "auto" else {}),
        },
        timeout=60,
    )
    if resp.status_code >= 400:
        return {"meta": {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}, "raw": {}}
    raw = resp.json()
    payload = {"meta": {"status": "ok", "cache_status": "miss", "provider": "serper", "language": language, "fetched_at": time.time()}, "raw": raw}
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


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", str(text or "")))


def _snippet_format(answer: str, source: dict | None = None) -> str:
    source = source or {}
    if source.get("table") or source.get("table_items") or source.get("tableRows"):
        return "table"
    if source.get("list") or source.get("items") or source.get("list_items"):
        return "list"
    lines = [line.strip() for line in str(answer or "").splitlines() if line.strip()]
    if len(lines) >= 2 and all(re.match(r"^(?:[-*•]|\d+[.)])\s+", line) for line in lines[:3]):
        return "list"
    return "paragraph"


def _answer_box_payload(title: str, answer: str, url: str, source: dict | None = None) -> dict:
    return {
        "title": str(title or "").strip(),
        "answer": str(answer or "").strip(),
        "url": str(url or "").strip(),
        "format": _snippet_format(answer, source),
        "word_count": _word_count(answer),
    }


def _domain_from_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        return urlparse(raw).netloc.lower().removeprefix("www.")
    if "." in raw and "/" not in raw:
        return raw.lower().removeprefix("www.")
    return ""


def _collect_ai_overview_domains(node) -> list[str]:
    domains: list[str] = []

    def add_domain(value: str) -> None:
        domain = _domain_from_value(value)
        if domain and domain not in domains:
            domains.append(domain)

    def walk(value) -> None:
        if len(domains) >= 10:
            return
        if isinstance(value, dict):
            for key in ("url", "link", "source_url", "domain"):
                add_domain(str(value.get(key) or ""))
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, str):
            for url in re.findall(r"https?://[^\s)>\"]+", value):
                add_domain(url)

    walk(node)
    return domains[:10]


def _ai_overview_payload(node, own_domain: str | None = None) -> dict | None:
    if isinstance(node, list) and node:
        node = {"items": node}
    if not isinstance(node, dict):
        return None
    domains = _collect_ai_overview_domains(node)
    own = _domain_from_value(own_domain or "")
    cites_us: bool | None = None
    if domains and own:
        cites_us = any(domain == own or domain.endswith("." + own) for domain in domains)
    return {
        "present": True,
        "cites_us": cites_us,
        "cited_domains": domains[:10],
    }


def _serp_features(payload: dict, own_domain: str | None = None) -> dict:
    """Extract People Also Ask, related searches, answer box, and AI Overview from SERP payloads."""
    features: dict = {"people_also_ask": [], "related_searches": [], "answer_box": {}, "ai_overview": None}
    raw = payload.get("raw") or {}
    provider = (payload.get("meta") or {}).get("provider") or ""
    own_domain = own_domain or (payload.get("meta") or {}).get("domain") or (payload.get("meta") or {}).get("own_domain")
    if provider == "serper" or "organic" in raw:
        for item in (raw.get("peopleAlsoAsk") or [])[:10]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            features["people_also_ask"].append({
                "question": question,
                "snippet": str(item.get("snippet") or "").strip(),
                "url": str(item.get("link") or "").strip(),
                "title": str(item.get("title") or "").strip(),
            })
        for item in (raw.get("relatedSearches") or [])[:10]:
            query = str((item.get("query") if isinstance(item, dict) else item) or "").strip()
            if query:
                features["related_searches"].append(query)
        answer_box = raw.get("answerBox")
        if isinstance(answer_box, dict):
            answer = str(answer_box.get("answer") or answer_box.get("snippet") or "").strip()
            if answer:
                features["answer_box"] = _answer_box_payload(
                    str(answer_box.get("title") or ""),
                    answer,
                    str(answer_box.get("link") or ""),
                    answer_box,
                )
        for key in ("aiOverview", "ai_overview", "ai_overview_item"):
            overview = _ai_overview_payload(raw.get(key), own_domain)
            if overview is not None:
                features["ai_overview"] = overview
                break
        return features
    for item in _serp_items(payload):
        item_type = str(item.get("type") or "")
        if item_type == "people_also_ask":
            for element in (item.get("items") or [])[:10]:
                if not isinstance(element, dict):
                    continue
                question = str(element.get("title") or "").strip()
                if not question:
                    continue
                snippet = ""
                url = ""
                for expanded in element.get("expanded_element") or []:
                    if isinstance(expanded, dict):
                        snippet = str(expanded.get("description") or "").strip()
                        url = str(expanded.get("url") or "").strip()
                        break
                if len(features["people_also_ask"]) < 10:
                    features["people_also_ask"].append({
                        "question": question,
                        "snippet": snippet,
                        "url": url,
                        "title": "",
                    })
        elif item_type == "related_searches":
            for element in item.get("items") or []:
                query = str(element or "").strip() if not isinstance(element, dict) else str(element.get("title") or "").strip()
                if query and len(features["related_searches"]) < 10:
                    features["related_searches"].append(query)
        elif item_type == "featured_snippet" and not features["answer_box"]:
            answer = str(item.get("description") or "").strip()
            if answer:
                features["answer_box"] = _answer_box_payload(str(item.get("title") or ""), answer, str(item.get("url") or ""), item)
        elif item_type == "ai_overview" and features["ai_overview"] is None:
            features["ai_overview"] = _ai_overview_payload(item, own_domain)
    return features


def _paa_coverage(
    features: dict,
    own_paragraphs: list[str],
    own_embeddings: np.ndarray,
    embedder: Embedder,
) -> list[dict]:
    """Score each People Also Ask question against our paragraphs."""
    questions = [row.get("question") or "" for row in features.get("people_also_ask") or []]
    questions = [q for q in questions if q.strip()]
    if not questions:
        return []
    rows: list[dict] = []
    if not own_paragraphs or not len(own_embeddings):
        return [
            {"question": q, "status": "missing", "best_similarity": 0.0, "best_paragraph_index": None, "best_paragraph": ""}
            for q in questions
        ]
    try:
        question_embeddings = embedder.encode(questions, batch_size=32).astype(np.float32)
    except Exception:
        return []
    for q, q_emb in zip(questions, question_embeddings):
        sims = own_embeddings @ q_emb
        best_i = int(np.argmax(sims))
        best = float(np.clip(sims[best_i], -1.0, 1.0))
        if best >= 0.78:
            status = "covered"
        elif best >= 0.62:
            status = "partial"
        else:
            status = "missing"
        rows.append({
            "question": q,
            "status": status,
            "best_similarity": round(best, 4),
            "best_paragraph_index": best_i,
            "best_paragraph": str(own_paragraphs[best_i] or "")[:240],
        })
    return rows


def _normalize_intent_label(value: str) -> str:
    label = re.sub(r"[_\s]+", "-", str(value or "").strip().lower())
    if label in {"commercial", "commercial-investigation", "commercial-informational"}:
        return "commercial-investigation"
    if label in {"informational", "information"}:
        return "informational"
    if label in {"transactional", "transaction"}:
        return "transactional"
    if label in {"navigational", "navigation", "branded"}:
        return "navigational"
    return label


def _provider_intent(keyword: dict) -> tuple[str, list[str]]:
    intents = [_normalize_intent_label(str(intent)) for intent in keyword.get("intents") or []]
    intents = [intent for intent in intents if intent]
    if not intents:
        return "", []
    priority = ["transactional", "commercial-investigation", "informational", "navigational"]
    for label in priority:
        if label in intents:
            return label, [f"provider_intent:{label}"]
    return intents[0], [f"provider_intent:{intents[0]}"]


def _intent_from_text(text: str) -> tuple[str, str]:
    lower = str(text or "").lower()
    if re.search(r"\b(best|top\s+\d+|vs|versus|compare|comparison|alternative|alternatives|review|reviews)\b", lower):
        return "commercial-investigation", "commercial SERP title/query pattern"
    if re.search(r"\b(price|pricing|cost|buy|order|demo|quote|trial|plan|plans|coupon|discount)\b", lower):
        return "transactional", "transactional SERP title/query pattern"
    if re.search(r"\b(how|what|why|when|where|which|guide|tutorial|examples?|definition|meaning|learn)\b", lower):
        return "informational", "informational SERP title/query pattern"
    return "", ""


def _serp_evidence_intent(keyword: dict, serp_rows: list[dict], features: dict) -> tuple[str, list[str]]:
    votes: dict[str, int] = {}
    evidence: list[str] = []
    for text in [str(keyword.get("keyword") or "")] + [str(row.get("title") or "") for row in serp_rows[:10]]:
        intent, reason = _intent_from_text(text)
        if not intent:
            continue
        votes[intent] = votes.get(intent, 0) + 1
        if len(evidence) < 5:
            evidence.append(f"{reason}: {text[:90]}")
    paa_count = len(features.get("people_also_ask") or [])
    if paa_count:
        votes["informational"] = votes.get("informational", 0) + 1
        evidence.append(f"People Also Ask present ({paa_count} questions)")
    if not votes:
        return "unknown", evidence
    intent = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return intent, evidence


def _page_intent_for_type(page_type: str) -> str:
    if page_type in {"article", "blog_post", "docs", "faq"}:
        return "informational"
    if page_type in {"listing", "search"}:
        return "commercial-investigation"
    if page_type in {"product", "service", "home"}:
        return "transactional"
    if page_type in {"contact", "about"}:
        return "navigational"
    return ""


def _page_intent(page: ExtractedPage, page_type_row: dict) -> tuple[str, list[str]]:
    # The page type is the primary signal: a product page stays transactional
    # even when its title happens to contain "best"/"how" wording. Text
    # patterns are only a fallback for unknown/unmapped page types.
    page_type = str(page_type_row.get("page_type") or "").strip()
    evidence: list[str] = []
    if page_type:
        evidence.append(f"audit_page_type:{page_type}")
    else:
        try:
            classified = classify_page(page)
            page_type = classified.page_type
            evidence.append(f"classified_page_type:{page_type}")
        except Exception:
            page_type = ""
    type_intent = _page_intent_for_type(page_type)
    if type_intent:
        return type_intent, evidence
    text = " ".join([
        str(page.title or ""),
        str(page.h1 or ""),
        str(page.description or ""),
        " ".join(str(h.get("text") or "") for h in (page.headers_rich or [])[:12]),
    ])
    text_intent, reason = _intent_from_text(text)
    if text_intent:
        evidence.append(f"{reason}: target title/H1/headings")
        return text_intent, evidence
    return "unknown", evidence


def _intent_match(serp_intent: str, page_intent: str) -> str:
    if not serp_intent or not page_intent or "unknown" in {serp_intent, page_intent}:
        return "partial"
    if serp_intent == page_intent:
        return "match"
    if {serp_intent, page_intent} <= {"transactional", "commercial-investigation"}:
        return "mismatch"
    if {serp_intent, page_intent} <= {"informational", "commercial-investigation"}:
        return "partial"
    return "mismatch"


def _intent_assessment(
    keyword: dict,
    own_ext: ExtractedPage,
    page_type_row: dict,
    serp_rows: list[dict],
    features: dict,
    competitor_pages: list[CompetitorPage],
) -> dict:
    serp_intent, evidence = _provider_intent(keyword)
    if not serp_intent:
        serp_intent, evidence = _serp_evidence_intent(keyword, serp_rows, features)
    page_intent, page_evidence = _page_intent(own_ext, page_type_row)
    if competitor_pages and len(evidence) < 6:
        top_titles = [
            cp.title for cp in sorted(competitor_pages, key=lambda cp: cp.target.rank or 999)
            if cp.title
        ][:3]
        if top_titles:
            evidence.append("top_competitor_titles:" + " | ".join(title[:80] for title in top_titles))
    return {
        "serp_intent": serp_intent or "unknown",
        "page_intent": page_intent or "unknown",
        "match": _intent_match(serp_intent or "unknown", page_intent or "unknown"),
        "evidence": evidence + page_evidence,
    }


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
            "domain_rating": item.get("domain_rating"),
            "domain_rating_status": item.get("domain_rating_status", ""),
            "domain_rating_source": item.get("domain_rating_source", ""),
            "domain_rating_license": item.get("domain_rating_license", ""),
            "domain_rating_attribution": item.get("domain_rating_attribution", ""),
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


def _domain_rating_for_host(host: str, ratings: dict[str, dict]) -> dict:
    key = str(host or "").lower().removeprefix("www.")
    return ratings.get(key) or {}


def _apply_domain_rating(row: dict, ratings: dict[str, dict]) -> None:
    host = str(row.get("domain") or urlparse(str(row.get("url") or "")).netloc or "").lower()
    rating = _domain_rating_for_host(host, ratings)
    if not rating:
        return
    row["domain_rating"] = rating.get("domain_rating")
    row["domain_rating_status"] = rating.get("status", "")
    row["domain_rating_source"] = rating.get("source", "ahrefs_public_domain_rating_free")
    row["domain_rating_license"] = rating.get("license") or AHREFS_DOMAIN_RATING_LICENSE
    row["domain_rating_attribution"] = rating.get("attribution") or AHREFS_DOMAIN_RATING_ATTRIBUTION


def _domain_rating_value(row: dict) -> float | None:
    raw = row.get("domain_rating")
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _is_ugc_host(url: str) -> bool:
    host = urlparse(str(url or "")).netloc.lower().removeprefix("www.")
    if not host:
        return False
    for domain in _WEAK_RESULT_UGC_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
    labels = set(host.split("."))
    return bool(labels & _WEAK_RESULT_HOST_LABELS)


def _is_low_dr_result(row: dict) -> bool:
    dr = _domain_rating_value(row)
    return dr is not None and dr <= WEAK_RESULT_DR_MAX


def _median(values: list[float]) -> float:
    clean = sorted(values)
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return round((clean[middle - 1] + clean[middle]) / 2, 2)


def _winnability(serp_rows: list[dict], own_dr: float | int | None) -> dict:
    """Estimate whether page-one competition is reachable from free DR data."""
    try:
        own_value = None if own_dr in (None, "") else float(own_dr)
    except (TypeError, ValueError):
        own_value = None
    top_rows = [row for row in serp_rows if 0 < _safe_int(row.get("rank")) <= 10]
    dr_values = [value for value in (_domain_rating_value(row) for row in top_rows) if value is not None]
    if own_value is None or not dr_values:
        return {
            "band": "unknown",
            "factor": WINNABILITY_FACTORS["unknown"],
            "own_dr": own_value,
            "top10_dr_min": None,
            "top10_dr_median": None,
            "within_reach_count": 0,
            "weak_result_present": False,
            "evidence": ["Missing own DR or top-10 DR data; winnability was not used as a gate."],
        }
    min_dr = min(dr_values)
    median_dr = _median(dr_values)
    within_reach = sum(1 for value in dr_values if value <= own_value + 10)
    ugc_rows = [row for row in top_rows if _is_ugc_host(str(row.get("url") or ""))]
    low_dr_rows = [row for row in top_rows if _is_low_dr_result(row)]
    weak_present = bool(ugc_rows or low_dr_rows)
    gap = median_dr - own_value
    if within_reach >= 3 or weak_present:
        band = "winnable"
    elif own_value < min_dr - 30:
        band = "unlikely"
    elif gap > 30:
        # A median gap this large is never "winnable"; with nothing in reach
        # it is effectively hopeless.
        band = "unlikely" if within_reach == 0 else "hard"
    elif gap >= 10:
        band = "hard"
    else:
        band = "winnable"
    evidence = [
        f"Own DR {own_value:g}; top-10 DR min {min_dr:g}; median {median_dr:g}.",
        f"{within_reach} top-10 result(s) are within own DR + 10.",
    ]
    if ugc_rows:
        host = urlparse(str(ugc_rows[0].get("url") or "")).netloc or ugc_rows[0].get("url")
        evidence.append(f"Weak result signal: UGC/forum host {host} ranks in the top 10.")
    if low_dr_rows:
        low_dr = _domain_rating_value(low_dr_rows[0])
        host = urlparse(str(low_dr_rows[0].get("url") or "")).netloc or low_dr_rows[0].get("url")
        evidence.append(f"Weak result signal: low-authority page (DR {low_dr:g}, {host}) ranks in the top 10.")
    return {
        "band": band,
        "factor": WINNABILITY_FACTORS[band],
        "own_dr": round(own_value, 1),
        "top10_dr_min": round(min_dr, 1),
        "top10_dr_median": round(median_dr, 1),
        "within_reach_count": within_reach,
        "weak_result_present": weak_present,
        "evidence": evidence,
    }


def _keyword_winnability_factor(row: dict) -> float:
    raw = row.get("winnability_factor")
    if raw is not None:
        value = _safe_float(raw)
        if value > 0:
            return value
    band = str((row.get("winnability") or {}).get("band") or row.get("winnability_band") or "")
    return WINNABILITY_FACTORS.get(band, 1.0)


def _keyword_demand_score(row: dict) -> float:
    demand = max(
        _safe_int(row.get("impressions")),
        _safe_int(row.get("volume")),
        int(_safe_float(row.get("traffic")) * 20),
        _safe_int(row.get("clicks")) * 50,
    )
    return float(np.log1p(demand)) if demand > 0 else 0.0


def _keyword_similarity_score(a: str, b: str) -> float:
    left = {token for token in re.split(r"\W+", a.lower()) if len(token) > 2}
    right = {token for token in re.split(r"\W+", b.lower()) if len(token) > 2}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _alternative_keyword(analysis: dict, keyword_rows: list[dict]) -> dict:
    current = analysis.get("keyword") or {}
    current_keyword = str(current.get("keyword") or analysis.get("query") or "").strip()
    current_url = str(current.get("url") or "")
    current_demand = _keyword_demand_score(current)
    candidates = []
    for row in keyword_rows:
        keyword = str(row.get("keyword") or "").strip()
        if not keyword or keyword.lower() == current_keyword.lower():
            continue
        if current_url and not _same_url(str(row.get("url") or ""), current_url):
            continue
        factor = _keyword_winnability_factor(row)
        if factor <= WINNABILITY_FACTORS["unlikely"]:
            continue
        band = str((row.get("winnability") or {}).get("band") or row.get("winnability_band") or "")
        verified = band == "winnable"
        demand = _keyword_demand_score(row)
        candidates.append((
            verified,
            factor,
            _keyword_similarity_score(current_keyword, keyword),
            demand <= current_demand,
            demand,
            row,
        ))
    if not candidates:
        return {}
    # Verified-winnable rows beat rows whose winnability was never assessed.
    candidates.sort(key=lambda item: (not item[0], -item[1], -item[2], not item[3], -item[4], str(item[5].get("keyword") or "")))
    verified, _, _, _, _, row = candidates[0]
    return {
        "keyword": row.get("keyword", ""),
        "band": (row.get("winnability") or {}).get("band") or row.get("winnability_band", ""),
        "verified": verified,
        "impressions": row.get("impressions", 0),
        "traffic": row.get("traffic", 0),
        "volume": row.get("volume", 0),
    }


def _winnability_cache_key(url: str, keyword: str) -> str:
    return f"{_metric_url_key(url)}|{str(keyword or '').strip().lower()}"


def _load_winnability_cache(cache_dir: Path) -> dict[str, dict]:
    path = Path(cache_dir) / "keyword_winnability.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_winnability_cache(cache_dir: Path, keyword_rows: list[dict]) -> None:
    """Persist per-keyword winnability bands so the next run's keyword
    selection can deprioritize hard/unlikely keywords before SERPs are fetched."""
    data = _load_winnability_cache(cache_dir)
    changed = False
    for row in keyword_rows:
        band = str(row.get("winnability_band") or "")
        if band not in WINNABILITY_FACTORS or band == "unknown":
            continue
        key = _winnability_cache_key(str(row.get("url") or ""), str(row.get("keyword") or ""))
        entry = {"band": band, "factor": WINNABILITY_FACTORS[band]}
        if data.get(key) != entry:
            data[key] = entry
            changed = True
    if not changed:
        return
    try:
        (Path(cache_dir) / "keyword_winnability.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError:
        pass


def _attach_winnability(page_results: list[dict], keyword_rows: list[dict], own_dr: float | int | None) -> None:
    by_keyword: dict[tuple[str, str], dict] = {}
    for page in page_results:
        for analysis in page.get("analyses") or []:
            if analysis.get("status") != "ok":
                continue
            top10 = (analysis.get("serp") or {}).get("top10") or analysis.get("competitor_pages") or []
            win = _winnability(top10, own_dr)
            analysis["winnability"] = win
            analysis["recommendation_header"] = _recommendation_header(analysis)
            keyword = analysis.get("keyword") or {}
            keyword["winnability"] = win
            keyword["winnability_band"] = win.get("band", "unknown")
            keyword["winnability_factor"] = win.get("factor", 1.0)
            by_keyword[(str(keyword.get("url") or ""), str(keyword.get("keyword") or "").lower())] = win
    for row in keyword_rows:
        win = by_keyword.get((str(row.get("url") or ""), str(row.get("keyword") or "").lower()))
        if not win:
            continue
        row["winnability"] = win
        row["winnability_band"] = win.get("band", "")
        row["winnability_factor"] = win.get("factor", 1.0)
    for page in page_results:
        for analysis in page.get("analyses") or []:
            if analysis.get("status") == "ok" and (analysis.get("winnability") or {}).get("band") == "unlikely":
                analysis["alternative_keyword"] = _alternative_keyword(analysis, keyword_rows)


def _recommendation_header(analysis: dict) -> str:
    win = analysis.get("winnability") or {}
    band = win.get("band")
    notes: list[str] = []
    if band == "unlikely":
        notes.append("Content changes alone are unlikely to reach page 1; build authority or target an easier variant first.")
    elif band == "hard":
        notes.append("Hard SERP: proceed with content improvements, but expect link acquisition or authority gains to matter.")
    overview = (analysis.get("serp_features") or {}).get("ai_overview")
    if isinstance(overview, dict) and overview.get("present") and overview.get("cited_domains") and overview.get("cites_us") is False:
        notes.append("AI Overview is present but does not cite this domain; prioritize direct answers, sourceable facts, and snippet/PAA blocks.")
    return " ".join(notes)


def _enrich_serp_domain_ratings(
    own_domain: str,
    serp_url_rankings: dict[str, dict],
    page_results: list[dict],
    overview_rows: list[dict],
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> dict:
    domains: set[str] = set()
    own_host = urlparse(own_domain if "://" in own_domain else f"https://{own_domain}").netloc.lower().removeprefix("www.")
    if own_host:
        domains.add(own_host)
    for item in serp_url_rankings.values():
        host = str(item.get("domain") or urlparse(str(item.get("url") or "")).netloc or "").lower().removeprefix("www.")
        if host:
            domains.add(host)
    for page in page_results:
        for analysis in page.get("analyses") or []:
            for row in (analysis.get("serp") or {}).get("top10") or []:
                host = str(row.get("domain") or urlparse(str(row.get("url") or "")).netloc or "").lower().removeprefix("www.")
                if host:
                    domains.add(host)
            for cp in analysis.get("competitor_pages") or []:
                host = str(cp.get("domain") or urlparse(str(cp.get("url") or "")).netloc or "").lower().removeprefix("www.")
                if host:
                    domains.add(host)
    for row in overview_rows:
        host = str(row.get("domain") or urlparse(str(row.get("url") or "")).netloc or "").lower().removeprefix("www.")
        if host:
            domains.add(host)
    ratings = fetch_domain_ratings_free(sorted(domains), cache_dir, refresh=refresh)
    for item in serp_url_rankings.values():
        _apply_domain_rating(item, ratings)
    for page in page_results:
        for analysis in page.get("analyses") or []:
            for row in (analysis.get("serp") or {}).get("top10") or []:
                _apply_domain_rating(row, ratings)
            for cp in analysis.get("competitor_pages") or []:
                _apply_domain_rating(cp, ratings)
    for row in overview_rows:
        _apply_domain_rating(row, ratings)
    ok = [row for row in ratings.values() if row.get("status") == "ok"]
    errors = [
        {"domain": domain, "status": row.get("status"), "error": row.get("error", "")}
        for domain, row in ratings.items()
        if row.get("status") != "ok"
    ]
    return {
        "provider": "ahrefs_public_domain_rating_free",
        "attribution": AHREFS_DOMAIN_RATING_ATTRIBUTION,
        "license": AHREFS_DOMAIN_RATING_LICENSE,
        "own_domain": own_host,
        "own_domain_rating": _domain_rating_value(ratings.get(own_host) or {}),
        "domains_requested": len(domains),
        "domains_enriched": len(ok),
        "errors": errors[:20],
    }


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
    own_ext: ExtractedPage | None = None,
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
    structural_gaps: list[dict] = []
    if own_ext is not None:
        try:
            structural_gaps = structural_diff(own_ext, ext)
        except Exception:
            structural_gaps = []
    try:
        answerability = float(score_page(ext).score)
    except Exception:
        answerability = 0.0
    paragraphs = (ext.paragraphs or [])[:config.max_paragraphs_per_page]
    embs = embedder.encode(paragraphs, batch_size=64).astype(np.float32) if paragraphs else np.zeros((0, 0), dtype=np.float32)
    return CompetitorPage(
        target=target,
        title=ext.title or target.competitor_url,
        paragraphs=paragraphs,
        paragraph_embeddings=embs,
        structural_gaps=structural_gaps,
        answerability=answerability,
        paragraph_count=len(paragraphs),
        h1=ext.h1,
        headers_rich=ext.headers_rich,
        content_sequence=ext.content_sequence,
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
    own_paragraphs: list[str] | None = None,
    own_embeddings: np.ndarray | None = None,
) -> dict:
    if own_paragraphs is None:
        own_paragraphs = (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
    if own_embeddings is None:
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
    gap["competitor_pages"] = _competitor_page_profiles(competitor_pages, gap.get("topics") or [])
    gap["content_comparison"] = _content_comparison(page, own_ext, own_paragraphs, competitor_pages, gap.get("topics") or [])
    gap["topic_coverage_matrix"] = _topic_coverage_matrix(page.url, competitor_pages, gap.get("topics") or [])
    gap["paragraph_match_heatmap"] = _paragraph_match_heatmap(own_paragraphs, own_embeddings, competitor_pages)
    gap["content_order_path"] = _content_order_path(keyword["keyword"], own_ext, competitor_pages, embedder)
    gap["recommended_outline"] = _recommended_outline(gap)
    gap["visual_summary"] = _visual_summary(gap["content_comparison"], gap["topic_coverage_matrix"])
    return gap


def _heading_counts(headers: list[dict]) -> dict:
    counts = {f"h{i}": 0 for i in range(1, 7)}
    for header in headers or []:
        try:
            level = int(header.get("level") or 0)
        except (TypeError, ValueError):
            level = 0
        if 1 <= level <= 6:
            counts[f"h{level}"] += 1
    counts["heading_count"] = sum(counts.values())
    counts["h2_h3_count"] = counts["h2"] + counts["h3"]
    counts["h4_h6_count"] = counts["h4"] + counts["h5"] + counts["h6"]
    return counts


def _topic_url_match(topic: dict, url: str) -> bool:
    return any(_same_url(url, topic_url) for topic_url in topic.get("competitor_urls") or [])


def _topic_profile_for_url(url: str, topics: list[dict]) -> dict:
    covered = [topic for topic in topics if _topic_url_match(topic, url)]
    return {
        "topic_count": len(covered),
        "missing_topics_covered": sum(1 for topic in covered if topic.get("coverage") == "missing"),
        "partial_topics_covered": sum(1 for topic in covered if topic.get("coverage") == "partial"),
        "high_priority_topics_covered": sum(1 for topic in covered if topic.get("priority") in {"critical", "high"}),
        "coverage_ratio": round(len(covered) / max(len(topics), 1), 4),
    }


def _competitor_page_profiles(competitor_pages: list[CompetitorPage], topics: list[dict]) -> list[dict]:
    rows = []
    for cp in competitor_pages:
        heading_counts = _heading_counts(cp.headers_rich)
        topic_profile = _topic_profile_for_url(cp.target.competitor_url, topics)
        rows.append({
            "url": cp.target.competitor_url,
            "domain": urlparse(cp.target.competitor_url).netloc,
            "rank": cp.target.rank,
            "title": cp.title,
            "h1": cp.h1,
            "paragraph_count": cp.paragraph_count,
            "error": cp.error,
            **heading_counts,
            **topic_profile,
        })
    return rows


def _content_comparison(
    page: PageInfo,
    own_ext: ExtractedPage,
    own_paragraphs: list[str],
    competitor_pages: list[CompetitorPage],
    topics: list[dict],
) -> dict:
    summary = {
        "total_topics": len(topics),
        "missing_topics": sum(1 for topic in topics if topic.get("coverage") == "missing"),
        "partial_topics": sum(1 for topic in topics if topic.get("coverage") == "partial"),
        "covered_topics": sum(1 for topic in topics if topic.get("coverage") == "covered"),
    }
    own_heading_counts = _heading_counts(own_ext.headers_rich)
    ours = {
        "source": "ours",
        "label": urlparse(page.url).netloc or page.url,
        "url": page.url,
        "domain": urlparse(page.url).netloc,
        "rank": None,
        "title": own_ext.title or page.title,
        "h1": own_ext.h1,
        "paragraph_count": len(own_paragraphs),
        "word_count": own_ext.word_count or page.word_count,
        **own_heading_counts,
        "topic_count": summary["covered_topics"],
        "missing_topics_covered": 0,
        "partial_topics_covered": summary["partial_topics"],
        "high_priority_topics_covered": sum(
            1 for topic in topics
            if topic.get("coverage") == "covered" and topic.get("priority") in {"critical", "high"}
        ),
        "coverage_ratio": round(summary["covered_topics"] / max(len(topics), 1), 4),
        "missing_topics": summary["missing_topics"],
        "partial_topics": summary["partial_topics"],
        "covered_topics": summary["covered_topics"],
    }
    competitors = _competitor_page_profiles([cp for cp in competitor_pages if not cp.error], topics)
    competitors.sort(key=lambda row: (_safe_int(row.get("rank")) or 999, -_safe_int(row.get("topic_count"))))

    def median(values: list[int]) -> float:
        clean = sorted(v for v in values if v is not None)
        if not clean:
            return 0.0
        middle = len(clean) // 2
        if len(clean) % 2:
            return float(clean[middle])
        return round((clean[middle - 1] + clean[middle]) / 2, 2)

    top = competitors[:5]
    benchmark = {
        "competitor_count": len(competitors),
        "median_competitor_paragraphs": median([_safe_int(row.get("paragraph_count")) for row in top]),
        "median_competitor_headings": median([_safe_int(row.get("heading_count")) for row in top]),
        "median_competitor_h2_h3": median([_safe_int(row.get("h2_h3_count")) for row in top]),
        "median_competitor_topics": median([_safe_int(row.get("topic_count")) for row in top]),
        "max_competitor_topics": max([_safe_int(row.get("topic_count")) for row in competitors] or [0]),
    }
    return {
        "summary": summary,
        "ours": ours,
        "competitors": competitors,
        "benchmark": benchmark,
    }


def _topic_coverage_matrix(own_url: str, competitor_pages: list[CompetitorPage], topics: list[dict], limit: int = 14) -> dict:
    competitors = [cp for cp in competitor_pages if not cp.error]
    competitors.sort(key=lambda cp: cp.target.rank or 999)
    columns = [
        {
            "id": "ours",
            "label": urlparse(own_url).netloc or own_url,
            "url": own_url,
            "source": "ours",
            "rank": None,
        }
    ]
    for cp in competitors[:6]:
        columns.append({
            "id": cp.target.competitor_url,
            "label": urlparse(cp.target.competitor_url).netloc or cp.target.competitor_url,
            "url": cp.target.competitor_url,
            "source": "competitor",
            "rank": cp.target.rank,
        })
    priority_order = {"critical": 0, "high": 1, "medium": 2, "covered": 3}
    topic_rows = sorted(
        topics,
        key=lambda topic: (
            priority_order.get(topic.get("priority"), 9),
            topic.get("coverage") == "covered",
            -_safe_float(topic.get("competitor_prevalence")),
            _safe_int(topic.get("best_competitor_rank")) or 999,
            -_safe_int(topic.get("competitor_paragraphs")),
        ),
    )[:limit]
    rows = []
    for topic in topic_rows:
        cells = []
        cells.append({
            "column_id": "ours",
            "status": topic.get("coverage") or "",
            "score": _safe_float(topic.get("our_best_similarity")),
            "paragraph_index": topic.get("our_best_paragraph_index"),
        })
        for cp in competitors[:6]:
            covered = _topic_url_match(topic, cp.target.competitor_url)
            cells.append({
                "column_id": cp.target.competitor_url,
                "status": "covered" if covered else "not_seen",
                "score": 1.0 if covered else 0.0,
                "rank": cp.target.rank,
            })
        rows.append({
            "label": topic.get("label", ""),
            "coverage": topic.get("coverage", ""),
            "priority": topic.get("priority", ""),
            "competitor_coverage": topic.get("competitor_coverage", 0),
            "competitor_prevalence": topic.get("competitor_prevalence", 0),
            "our_best_similarity": topic.get("our_best_similarity", 0),
            "best_competitor_rank": topic.get("best_competitor_rank"),
            "our_best_paragraph": str(topic.get("our_best_paragraph") or "")[:360],
            "examples": [
                {
                    "url": example.get("url", ""),
                    "domain": urlparse(str(example.get("url") or "")).netloc,
                    "rank": example.get("rank", ""),
                    "paragraph": str(example.get("paragraph") or "")[:420],
                }
                for example in (topic.get("examples") or [])[:4]
                if str(example.get("paragraph") or "").strip()
            ],
            "cells": cells,
        })
    return {"columns": columns, "rows": rows}


def _paragraph_match_heatmap(
    own_paragraphs: list[str],
    own_embeddings: np.ndarray,
    competitor_pages: list[CompetitorPage],
    *,
    max_competitors: int = 6,
    max_competitor_paragraphs: int = 80,
) -> dict:
    competitors = [
        cp for cp in competitor_pages
        if not cp.error and len(cp.paragraphs) and len(cp.paragraph_embeddings)
    ]
    competitors.sort(key=lambda cp: cp.target.rank or 999)
    competitors = competitors[:max_competitors]
    columns = [
        {
            "url": cp.target.competitor_url,
            "domain": urlparse(cp.target.competitor_url).netloc,
            "rank": cp.target.rank,
            "title": cp.title,
            "paragraph_count": cp.paragraph_count,
        }
        for cp in competitors
    ]
    rows = []
    if not own_paragraphs or not len(own_embeddings) or not competitors:
        return {"columns": columns, "rows": rows}
    for own_i, own_paragraph in enumerate(own_paragraphs):
        if own_i >= len(own_embeddings):
            break
        paragraph_words = len(str(own_paragraph or "").split())
        cells = []
        scores = []
        for cp in competitors:
            matrix = cp.paragraph_embeddings[:max_competitor_paragraphs]
            if not len(matrix) or matrix.shape[1] != own_embeddings.shape[1]:
                cells.append({
                    "url": cp.target.competitor_url,
                    "status": "no_match",
                    "similarity": 0.0,
                    "rank": _safe_int(cp.target.rank),
                    "rank_weight": 0.0,
                    "rank_impact": 0.0,
                    "paragraph_index": None,
                    "paragraph": "",
                })
                continue
            sims = matrix @ own_embeddings[own_i]
            best_i = int(np.argmax(sims))
            best_score = float(np.clip(sims[best_i], -1.0, 1.0))
            rank = _safe_int(cp.target.rank)
            rank_weight = max(0.0, (11.0 - min(rank, 10)) / 10.0) if rank else 0.0
            rank_impact = max(0.0, best_score) * rank_weight
            scores.append(best_score)
            if best_score >= 0.78:
                status = "strong"
            elif best_score >= 0.62:
                status = "partial"
            else:
                status = "weak"
            cells.append({
                "url": cp.target.competitor_url,
                "status": status,
                "similarity": round(best_score, 4),
                "rank": rank,
                "rank_weight": round(rank_weight, 4),
                "rank_impact": round(rank_impact, 4),
                "paragraph_index": best_i,
                "paragraph": (cp.paragraphs[best_i] if best_i < len(cp.paragraphs) else "")[:360],
            })
        max_similarity = max(scores) if scores else 0.0
        average_similarity = sum(scores) / len(scores) if scores else 0.0
        rank_impacts = [_safe_float(cell.get("rank_impact")) for cell in cells]
        if max_similarity >= 0.78:
            status = "strong"
        elif max_similarity >= 0.62:
            status = "partial"
        else:
            status = "weak"
        rows.append({
            "paragraph_index": own_i,
            "paragraph": str(own_paragraph or "")[:420],
            "word_count": paragraph_words,
            "status": status,
            "max_similarity": round(max_similarity, 4),
            "average_similarity": round(average_similarity, 4),
            "max_rank_impact": round(max(rank_impacts) if rank_impacts else 0.0, 4),
            "cells": cells,
        })
    return {"columns": columns, "rows": rows}


def _content_sequence_items(
    sequence: list[dict],
    paragraphs: list[str],
    headers: list[dict],
    *,
    max_items: int = 70,
) -> list[dict]:
    items: list[dict] = []
    source = sequence or []
    if source:
        for raw in sorted(source, key=lambda row: _safe_int(row.get("order"))):
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            entity_type = str(raw.get("entity_type") or raw.get("type") or "paragraph").lower()
            items.append({
                "order": len(items),
                "entity_type": entity_type if entity_type else "paragraph",
                "level": _safe_int(raw.get("level")),
                "text": text,
            })
            if len(items) >= max_items:
                break
        return items

    for header in headers or []:
        text = str(header.get("text") or "").strip()
        if not text:
            continue
        level = _safe_int(header.get("level"))
        items.append({
            "order": len(items),
            "entity_type": _heading_entity_type(level),
            "level": level,
            "text": text,
        })
        if len(items) >= max_items:
            return items
    for paragraph in paragraphs or []:
        text = str(paragraph or "").strip()
        if not text:
            continue
        items.append({
            "order": len(items),
            "entity_type": "paragraph",
            "level": 0,
            "text": text,
        })
        if len(items) >= max_items:
            break
    return items


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if not len(matrix):
        return matrix
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def _content_order_cluster_label(texts: list[str], cluster_id: int) -> str:
    terms = _topic_terms(" ".join(texts[:8]), limit=5)
    return " ".join(terms) if terms else f"Cluster {cluster_id}"


def _content_path_page_sort_key(page: dict) -> tuple[int, int, str]:
    rank = _safe_int(page.get("rank")) or 9999
    unranked_target = 1 if page.get("source") == "ours" and rank == 9999 else 0
    return (rank, unranked_target, str(page.get("domain") or page.get("url") or ""))


def _content_order_path(
    keyword: str,
    own_ext: ExtractedPage,
    competitor_pages: list[CompetitorPage],
    embedder: Embedder,
    *,
    max_competitors: int = 5,
    max_items_per_page: int = 70,
) -> dict:
    pages: list[dict] = []
    rows: list[dict] = []
    texts: list[str] = []

    def add_page(
        *,
        page_id: str,
        source: str,
        url: str,
        title: str,
        rank: int | None,
        items: list[dict],
    ) -> None:
        clean_items = [item for item in items if str(item.get("text") or "").strip()]
        if not clean_items:
            return
        domain = urlparse(url).netloc
        pages.append({
            "id": page_id,
            "source": source,
            "url": url,
            "domain": domain,
            "title": title,
            "rank": rank,
            "label": own_ext.url if source == "ours" else domain or url,
            "item_count": len(clean_items),
        })
        denominator = max(len(clean_items) - 1, 1)
        for order_index, item in enumerate(clean_items):
            text = str(item.get("text") or "").strip()
            row = {
                "page_id": page_id,
                "source": source,
                "url": url,
                "domain": domain,
                "rank": rank,
                "order_index": order_index,
                "order_position": round(order_index / denominator, 4),
                "entity_type": str(item.get("entity_type") or "paragraph"),
                "level": _safe_int(item.get("level")),
                "text": text[:320],
            }
            rows.append(row)
            texts.append(text)

    add_page(
        page_id="ours",
        source="ours",
        url=own_ext.url,
        title=own_ext.title,
        rank=None,
        items=_content_sequence_items(
            getattr(own_ext, "content_sequence", []),
            own_ext.paragraphs or [],
            own_ext.headers_rich or [],
            max_items=max_items_per_page,
        ),
    )
    competitors = [cp for cp in competitor_pages if not cp.error]
    competitors.sort(key=lambda cp: cp.target.rank or 999)
    for index, cp in enumerate(competitors[:max_competitors], start=1):
        add_page(
            page_id=f"competitor-{index}",
            source="competitor",
            url=cp.target.competitor_url,
            title=cp.title,
            rank=_safe_int(cp.target.rank),
            items=_content_sequence_items(
                getattr(cp, "content_sequence", []),
                cp.paragraphs or [],
                cp.headers_rich or [],
                max_items=max_items_per_page,
            ),
        )

    if len(rows) < 2 or not texts:
        return {
            "pages": sorted(pages, key=_content_path_page_sort_key),
            "items": rows,
            "clusters": [],
            "deviations": [],
            "missing_clusters": [],
            "unmatched_clusters_by_url": [],
            "summary": {
                "page_count": len(pages),
                "item_count": len(rows),
                "order_score": 0.0,
                "deviation_count": 0,
                "missing_cluster_count": 0,
                "unmatched_cluster_count": 0,
            },
        }

    matrix = embedder.encode(texts, batch_size=64).astype(np.float32)
    if not len(matrix):
        return {
            "pages": sorted(pages, key=_content_path_page_sort_key),
            "items": rows,
            "clusters": [],
            "deviations": [],
            "missing_clusters": [],
            "unmatched_clusters_by_url": [],
            "summary": {},
        }
    try:
        labels, coords = project(matrix, num_clusters=max(2, min(18, max(2, len(rows) // 5))))
    except Exception:
        labels = np.zeros(len(rows), dtype=int)
        coords = np.array([[float(i), 0.0] for i in range(len(rows))], dtype=np.float32)

    keyword_similarity = np.zeros(len(rows), dtype=np.float32)
    if keyword:
        keyword_vector = embedder.encode([keyword], batch_size=1).astype(np.float32)
        if len(keyword_vector) and keyword_vector.shape[1] == matrix.shape[1]:
            keyword_similarity = (_normalize_matrix(matrix) @ _normalize_matrix(keyword_vector)[0]).astype(np.float32)

    cluster_data: dict[int, dict] = {}
    for index, row in enumerate(rows):
        cluster_id = int(labels[index]) if index < len(labels) else 0
        row["cluster"] = cluster_id
        row["x"] = round(float(coords[index][0]), 4) if index < len(coords) else 0.0
        row["y"] = round(float(coords[index][1]), 4) if index < len(coords) else 0.0
        row["keyword_similarity"] = round(float(np.clip(keyword_similarity[index], -1.0, 1.0)), 4)
        group = cluster_data.setdefault(cluster_id, {
            "texts": [],
            "ours_positions": [],
            "competitor_positions": [],
            "competitor_page_ids": set(),
            "competitor_domains": set(),
            "page_ids": set(),
            "page_positions": {},
            "page_samples": {},
            "best_rank": 999,
        })
        page_id = str(row.get("page_id") or "")
        if page_id:
            group["page_ids"].add(page_id)
            group["page_positions"].setdefault(page_id, []).append(float(row["order_position"]))
            group["page_samples"].setdefault(page_id, row["text"])
        if len(group["texts"]) < 10:
            group["texts"].append(row["text"])
        if row["source"] == "ours":
            group["ours_positions"].append(float(row["order_position"]))
        else:
            group["competitor_positions"].append(float(row["order_position"]))
            group["competitor_page_ids"].add(row["page_id"])
            if row.get("domain"):
                group["competitor_domains"].add(row["domain"])
            rank = _safe_int(row.get("rank"))
            if rank:
                group["best_rank"] = min(group["best_rank"], rank)

    clusters: list[dict] = []
    deviations: list[dict] = []
    missing_clusters: list[dict] = []
    page_lookup = {str(page.get("id") or ""): page for page in pages}
    unmatched_clusters_by_page: dict[str, dict] = {
        str(page.get("id") or ""): {
            "page_id": str(page.get("id") or ""),
            "source": page.get("source"),
            "url": page.get("url"),
            "domain": page.get("domain"),
            "rank": page.get("rank"),
            "title": page.get("title"),
            "clusters": [],
        }
        for page in pages
        if page.get("id")
    }
    for cluster_id, group in cluster_data.items():
        competitor_positions = group["competitor_positions"]
        ours_positions = group["ours_positions"]
        competitor_mean = _mean(competitor_positions) if competitor_positions else None
        ours_mean = _mean(ours_positions) if ours_positions else None
        best_rank = group["best_rank"] if group["best_rank"] != 999 else None
        cluster = {
            "cluster": cluster_id,
            "label": _content_order_cluster_label(group["texts"], cluster_id),
            "ours_mean_order": round(ours_mean, 4) if ours_mean is not None else None,
            "competitor_mean_order": round(competitor_mean, 4) if competitor_mean is not None else None,
            "competitor_pages": len(group["competitor_page_ids"]),
            "competitor_domains": sorted(group["competitor_domains"])[:5],
            "best_competitor_rank": best_rank,
            "sample_text": str((group["texts"] or [""])[0])[:260],
        }
        clusters.append(cluster)
        if competitor_mean is not None and ours_mean is not None:
            delta = ours_mean - competitor_mean
            if abs(delta) >= 0.22:
                deviations.append({
                    **cluster,
                    "delta": round(delta, 4),
                    "direction": "later" if delta > 0 else "earlier",
                })
        elif competitor_mean is not None and not ours_positions and (len(group["competitor_page_ids"]) >= 2 or (best_rank or 999) <= 3):
            missing_clusters.append(cluster)
        page_ids = [page_id for page_id in group["page_ids"] if page_id in page_lookup]
        if len(page_ids) == 1:
            page_id = page_ids[0]
            positions = group["page_positions"].get(page_id, [])
            page_cluster = {
                "cluster": cluster_id,
                "label": cluster["label"],
                "mean_order": round(_mean(positions), 4) if positions else None,
                "count": len(positions),
                "sample_text": str(group["page_samples"].get(page_id) or cluster["sample_text"] or "")[:260],
            }
            unmatched_clusters_by_page[page_id]["clusters"].append(page_cluster)

    clusters.sort(key=lambda row: (
        -_safe_int(row.get("competitor_pages")),
        _safe_int(row.get("best_competitor_rank")) or 999,
        _safe_float(row.get("competitor_mean_order")),
    ))
    deviations.sort(key=lambda row: (-abs(_safe_float(row.get("delta"))), _safe_int(row.get("best_competitor_rank")) or 999))
    missing_clusters.sort(key=lambda row: (-_safe_int(row.get("competitor_pages")), _safe_int(row.get("best_competitor_rank")) or 999))
    shared_deltas = [abs(_safe_float(row.get("delta"))) for row in deviations]
    order_score = max(0.0, 1.0 - _mean(shared_deltas)) if shared_deltas else 1.0
    unmatched_clusters_by_url = sorted(unmatched_clusters_by_page.values(), key=_content_path_page_sort_key)
    unmatched_cluster_count = 0
    for page_entry in unmatched_clusters_by_url:
        page_entry["clusters"].sort(key=lambda row: (
            row.get("mean_order") is None,
            _safe_float(row.get("mean_order")),
            -_safe_int(row.get("count")),
        ))
        unmatched_cluster_count += len(page_entry["clusters"])
        page_entry["clusters"] = page_entry["clusters"][:8]

    return {
        "pages": sorted(pages, key=_content_path_page_sort_key),
        "items": rows,
        "clusters": clusters[:18],
        "deviations": deviations[:8],
        "missing_clusters": missing_clusters[:8],
        "unmatched_clusters_by_url": [row for row in unmatched_clusters_by_url if row.get("clusters")],
        "summary": {
            "page_count": len(pages),
            "item_count": len(rows),
            "cluster_count": len(clusters),
            "shared_cluster_deviations": len(deviations),
            "missing_cluster_count": len(missing_clusters),
            "unmatched_cluster_count": unmatched_cluster_count,
            "unmatched_cluster_pages": sum(1 for row in unmatched_clusters_by_url if row.get("clusters")),
            "order_score": round(order_score, 4),
        },
    }


def _visual_summary(content_comparison: dict, matrix: dict) -> list[str]:
    summary = content_comparison.get("summary") or {}
    ours = content_comparison.get("ours") or {}
    benchmark = content_comparison.get("benchmark") or {}
    rows = matrix.get("rows") or []
    reasons = []
    missing = _safe_int(summary.get("missing_topics"))
    partial = _safe_int(summary.get("partial_topics"))
    if missing:
        competitor_seen = max([_safe_int(row.get("competitor_coverage")) for row in rows if row.get("coverage") == "missing"] or [0])
        reasons.append(
            f"{missing} SERP topic groups are absent from the target page; the strongest missing group appears on {competitor_seen} ranking page(s)."
        )
    if partial:
        reasons.append(
            f"{partial} topic groups are only partially matched, so the page has related wording but not enough direct answer depth."
        )
    competitor_topics = _safe_float(benchmark.get("median_competitor_topics"))
    our_topics = _safe_float(ours.get("topic_count"))
    if competitor_topics > our_topics:
        reasons.append(
            f"Top ranking pages cover a median of {competitor_topics:g} SERP topic groups versus {our_topics:g} fully covered on the target page."
        )
    competitor_paragraphs = _safe_float(benchmark.get("median_competitor_paragraphs"))
    our_paragraphs = _safe_float(ours.get("paragraph_count"))
    if competitor_paragraphs and our_paragraphs and competitor_paragraphs >= our_paragraphs * 1.25:
        reasons.append(
            f"Top ranking pages use a median of {competitor_paragraphs:g} extracted paragraphs versus {our_paragraphs:g}, suggesting broader answer coverage."
        )
    competitor_h2_h3 = _safe_float(benchmark.get("median_competitor_h2_h3"))
    our_h2_h3 = _safe_float(ours.get("h2_h3_count"))
    if competitor_h2_h3 > our_h2_h3:
        reasons.append(
            f"Competitors expose more H2/H3 structure: median {competitor_h2_h3:g} versus {our_h2_h3:g} on the target page."
        )
    return reasons[:4]


def _topic_terms(label: str, limit: int = 6) -> list[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "your", "you",
        "are", "how", "what", "why", "when", "where", "which", "best", "page",
        "about", "also", "can", "could", "has", "have", "into", "left", "more",
        "right", "should", "than", "then", "their", "they", "was", "were", "will",
    }
    terms = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", str(label or "").lower()):
        if token in stop or token in terms:
            continue
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _clean_topic_label(label: str) -> str:
    parts = []
    for part in re.split(r"[,/]", str(label or "")):
        cleaned = " ".join(_topic_terms(part, limit=4))
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return ", ".join(parts) or str(label or "SERP topic")


def _action_priority_score(action: dict) -> tuple[int, float, int]:
    priority_weight = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return (
        priority_weight.get(str(action.get("priority") or "medium"), 9),
        -_safe_float(action.get("impact_score")),
        _safe_int(action.get("order")),
    )


def _keyword_demand(analysis: dict) -> dict:
    keyword = analysis.get("keyword") or {}
    impressions = _safe_int(keyword.get("impressions"))
    clicks = _safe_int(keyword.get("clicks"))
    traffic = _safe_float(keyword.get("traffic"))
    volume = _safe_int(keyword.get("volume"))
    demand = max(impressions, volume, int(traffic * 20), clicks * 50)
    return {
        "impressions": impressions,
        "clicks": clicks,
        "traffic": round(traffic, 4),
        "volume": volume,
        "score": float(np.log1p(demand)) if demand > 0 else 0.0,
    }


_FILLER_TERMS = (
    "best-in-class",
    "comprehensive",
    "cutting-edge",
    "easy to use",
    "innovative",
    "powerful",
    "robust",
    "seamless",
    "state-of-the-art",
    "user-friendly",
    "world-class",
)


def _editorial_guidelines() -> dict:
    return {
        "principles": [
            "Answer the search intent first; then add proof, examples, caveats, and next steps.",
            "Write for a reader who wants to decide or act, not for keyword repetition.",
            "Use competitor text as intent evidence only. Do not copy phrasing or structure blindly.",
            "Keep useful existing content, but remove or move paragraphs that serve a different intent.",
        ],
        "paragraph_rules": [
            "One paragraph should answer one concrete question or make one concrete point.",
            "Use 40-90 words for normal explanatory paragraphs; split longer paragraphs unless the topic requires legal or technical precision.",
            "Start important sections with a 1-2 sentence direct answer that could stand alone in a SERP snippet or AI answer.",
            "Every paragraph should contain at least one useful detail: condition, number, example, step, comparison, limitation, or decision criterion.",
            "Avoid generic filler adjectives unless they are backed by a concrete feature, workflow, source, or result.",
        ],
        "acceptance_criteria": [
            "A reader can understand the answer without reading competitor pages.",
            "The target keyword and related terms appear naturally in headings, answers, and examples.",
            "The page covers missing and partial SERP topics with original detail from the selected domain.",
            "Low-alignment paragraphs are rewritten, moved, merged, or removed after manual review.",
        ],
        "avoid": [
            "Do not add broad introductory text that repeats the page title.",
            "Do not stuff exact-match keywords into every paragraph.",
            "Do not copy competitor wording or cite competitor claims as your own.",
            "Do not publish vague claims without concrete support.",
            "Do not delete useful facts without moving them to a better location.",
            "Do not rewrite a paragraph for the target keyword if it actually belongs to another intent.",
            "Do not replace specific information with generic brand language.",
        ],
    }


def _paragraph_quality_profile(text: str) -> dict:
    text = str(text or "").strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lower = text.lower()
    filler_terms = [term for term in _FILLER_TERMS if term in lower]
    sentence_count = max(1, len(sentences))
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_words": round(len(words) / sentence_count, 1) if words else 0.0,
        "filler_terms": filler_terms,
        "is_long": len(words) > 120,
        "is_thin": 0 < len(words) < 35,
    }


def _recommended_content_format(keyword: str, label: str) -> str:
    combined = f"{keyword} {label}".lower()
    if re.search(r"\b(price|pricing|cost|plan|plans|fee|fees)\b", combined):
        return "answer block plus comparison table"
    if re.search(r"\b(how|steps|setup|implement|implementation|process)\b", combined):
        return "step-by-step answer block"
    if re.search(r"\b(vs|versus|compare|comparison|alternative|alternatives)\b", combined):
        return "comparison section"
    if re.search(r"\b(what|why|when|which|who)\b", combined):
        return "direct answer block"
    return "concise explanatory section"


def _topic_content_brief(
    *,
    action_type: str,
    page: dict,
    keyword: str,
    label: str,
    terms: list[str],
    topic: dict,
    example: dict,
) -> dict:
    add_topic = action_type == "add_topic"
    heading = label[:1].upper() + label[1:] if label else "SERP topic"
    format_name = _recommended_content_format(keyword, label)
    placement = (
        f"Add an H2/H3 section near the existing page area closest to '{terms[0] if terms else label}'."
        if add_topic
        else "Rewrite the nearest existing paragraph or subsection instead of adding a duplicate section."
    )
    paragraph_plan = [
        f"Open with a 1-2 sentence direct answer for '{keyword}' and the topic '{label}'.",
        "Add 2-4 short paragraphs, each focused on one user question, decision criterion, step, caveat, or example.",
        "Include concrete domain-specific details such as product behavior, process, pricing condition, eligibility, integration, timeline, limitation, or proof.",
        "Use a list or table only when it makes comparison, steps, requirements, or trade-offs easier to scan.",
    ]
    if example.get("paragraph"):
        paragraph_plan.append("Use the competitor example as a checklist for intent coverage, but write original text.")
    acceptance = [
        f"The section clearly covers '{label}' for the search intent behind '{keyword}'.",
        "The section naturally uses related terms: " + (", ".join(terms) if terms else label) + ".",
    ]
    prompt = (
        f"Edit {page.get('url', '')} for the keyword '{keyword}'. Task: {'add' if add_topic else 'strengthen'} "
        f"the topic '{label}' as a {format_name}. {placement} Follow the paragraph rules: one idea per paragraph, "
        "direct answer first, concrete details only, no filler. Use the SERP evidence to infer intent, but write original content."
    )
    return {
        "recommended_heading": heading,
        "recommended_format": format_name,
        "placement": placement,
        "paragraph_plan": paragraph_plan,
        "guidelines_ref": "editorial_guidelines",
        "acceptance_criteria": acceptance,
        "ai_agent_prompt": prompt,
        "source_evidence": {
            "example_url": example.get("url", ""),
            "example_rank": example.get("rank", ""),
            "example_paragraph": example.get("paragraph", ""),
            "competitor_coverage": _safe_int(topic.get("competitor_coverage")),
            "competitor_prevalence": topic.get("competitor_prevalence"),
            "our_best_similarity": topic.get("our_best_similarity"),
            "our_best_paragraph": topic.get("our_best_paragraph", ""),
        },
    }


def _analysis_winnability_factor(analysis: dict) -> float:
    return WINNABILITY_FACTORS.get(str((analysis.get("winnability") or {}).get("band") or "unknown"), 1.0)


def _topic_action(action_type: str, page: dict, analysis: dict, topic: dict, order: int) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "")
    demand = _keyword_demand(analysis)
    examples = topic.get("examples") or []
    example = examples[0] if examples else {}
    competitor_urls = topic.get("competitor_urls") or []
    coverage = str(topic.get("coverage") or "")
    priority = str(topic.get("priority") or ("high" if coverage == "missing" else "medium"))
    label = _clean_topic_label(str(topic.get("label") or "SERP topic"))
    terms = _topic_terms(label)
    if action_type == "add_topic":
        action = "Add a new content block"
        task_summary = f"Create a focused section about {label}."
        instruction = (
            f"Add a concise section to {page.get('url')} for the keyword '{keyword}' covering: {label}. "
            "Start with a direct answer, then add original details, examples, caveats, and decision criteria. "
            f"Use related terms naturally: {', '.join(terms) if terms else label}."
        )
        rationale = "Competitor pages repeatedly cover this topic, but the selected page has no close paragraph-level match."
    else:
        action = "Strengthen an existing content block"
        task_summary = f"Rewrite the closest section so it directly answers {label}."
        instruction = (
            f"Expand or rewrite the closest existing paragraph for '{keyword}' so it directly covers: {label}. "
            "Keep useful existing facts, remove filler, and add clearer examples, constraints, and intent coverage."
        )
        rationale = "The selected page is related to this topic, but competitor coverage is semantically stronger or more direct."
    content_brief = _topic_content_brief(
        action_type=action_type,
        page=page,
        keyword=keyword,
        label=label,
        terms=terms,
        topic=topic,
        example=example,
    )
    return {
        "id": f"{action_type}_{order}",
        "order": order,
        "type": action_type,
        "priority": priority if priority in {"critical", "high", "medium", "low"} else "medium",
        "action": action,
        "task_summary": task_summary,
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "instruction": instruction,
        "rationale": rationale,
        "topic": label,
        "coverage": coverage,
        "suggested_terms": terms,
        "content_brief": content_brief,
        "placement": content_brief["placement"],
        "acceptance_criteria": content_brief["acceptance_criteria"],
        "ai_agent_prompt": content_brief["ai_agent_prompt"],
        "impact_score": round(
            (
                _safe_float(topic.get("competitor_prevalence")) * 100
                + max(0, 11 - _safe_int(topic.get("best_competitor_rank") or 11))
                + demand["score"] * 10
            ) * _analysis_winnability_factor(analysis),
            3,
        ),
        "evidence": {
            "keyword_impressions": demand["impressions"],
            "keyword_clicks": demand["clicks"],
            "keyword_traffic": demand["traffic"],
            "keyword_volume": demand["volume"],
            "competitor_coverage": _safe_int(topic.get("competitor_coverage")),
            "competitor_prevalence": topic.get("competitor_prevalence"),
            "best_competitor_rank": topic.get("best_competitor_rank"),
            "our_best_similarity": topic.get("our_best_similarity"),
            "example_url": example.get("url", ""),
            "example_rank": example.get("rank", ""),
            "example_paragraph": example.get("paragraph", ""),
            "competitor_urls": competitor_urls[:5],
            "our_best_paragraph": topic.get("our_best_paragraph", ""),
        },
    }


def _paragraph_content_brief(page: dict, keyword: str, row: dict, profile: dict) -> dict:
    paragraph_index = row.get("paragraph_index")
    rules = [
        "Decide whether this paragraph belongs on the target page before rewriting it.",
        "If kept, rewrite it around one intent and connect it to the target keyword or a missing/partial SERP topic.",
        "If it answers a different intent, move it to a more relevant page or merge only the useful facts into the correct section.",
        "Split long paragraphs and remove generic claims that do not add a fact, example, condition, step, or caveat.",
    ]
    if profile.get("is_long"):
        rules.append("Split this paragraph because it is long enough to hide multiple ideas.")
    if profile.get("is_thin"):
        rules.append("Add substance or merge it because the paragraph is too thin to stand alone.")
    if profile.get("filler_terms"):
        rules.append("Replace detected filler terms with concrete details: " + ", ".join(profile["filler_terms"]) + ".")
    acceptance = [
        f"The paragraph supports the intent behind '{keyword}' or has been moved away from this page.",
        "The paragraph no longer dilutes the target page with unrelated context.",
    ]
    prompt = (
        f"Review paragraph {paragraph_index} on {page.get('url', '')} for the keyword '{keyword}'. "
        "Classify it as keep, rewrite, move, merge, or remove. If kept, rewrite it as one focused paragraph with concrete information and no filler."
    )
    return {
        "placement": f"Paragraph {paragraph_index} on the target page.",
        "paragraph_rules": rules,
        "acceptance_criteria": acceptance,
        "guidelines_ref": "editorial_guidelines",
        "ai_agent_prompt": prompt,
        "quality_profile": profile,
    }


def _paragraph_action(page: dict, analysis: dict, row: dict, order: int) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "")
    demand = _keyword_demand(analysis)
    similarity = _safe_float(row.get("similarity_to_serp_topics"))
    paragraph = str(row.get("paragraph") or "")
    profile = _paragraph_quality_profile(paragraph)
    content_brief = _paragraph_content_brief(page, keyword, row, profile)
    return {
        "id": f"review_paragraph_{order}",
        "order": order,
        "type": "review_paragraph",
        "priority": "medium" if similarity >= 0.45 else "high",
        "action": "Review or rewrite paragraph",
        "task_summary": f"Review paragraph {row.get('paragraph_index')} for intent drift or filler.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "paragraph_index": row.get("paragraph_index"),
        "instruction": (
            f"Review this paragraph for the keyword '{keyword}'. If it should support this landing page intent, "
            "rewrite it with one concrete point connected to the target keyword and nearby SERP topics. If it serves "
            "a different intent, move it to a better page, merge only the useful facts, or remove it."
        ),
        "rationale": "This paragraph is far from the SERP topic space for the keyword and may dilute topical focus.",
        "content_brief": content_brief,
        "placement": content_brief["placement"],
        "acceptance_criteria": content_brief["acceptance_criteria"],
        "ai_agent_prompt": content_brief["ai_agent_prompt"],
        "impact_score": round((max(0.0, 1.0 - similarity) * 100 + demand["score"] * 10) * _analysis_winnability_factor(analysis), 3),
        "evidence": {
            "keyword_impressions": demand["impressions"],
            "keyword_clicks": demand["clicks"],
            "keyword_traffic": demand["traffic"],
            "keyword_volume": demand["volume"],
            "similarity_to_serp_topics": row.get("similarity_to_serp_topics"),
            "review_reason": row.get("review_reason", ""),
            "paragraph": paragraph,
            "quality_profile": profile,
        },
    }


def _keyword_in_text(keyword: str, text: str) -> bool:
    keyword_norm = re.sub(r"\s+", " ", str(keyword or "")).strip().lower()
    text_norm = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not keyword_norm or not text_norm:
        return False
    if keyword_norm in text_norm:
        return True
    keyword_tokens = {t for t in re.split(r"\W+", keyword_norm) if len(t) > 2}
    if not keyword_tokens:
        return False
    text_tokens = {t for t in re.split(r"\W+", text_norm) if t}
    overlap = len(keyword_tokens & text_tokens) / len(keyword_tokens)
    return overlap >= 0.6


def _title_gap_action(page: dict, analysis: dict, order: int) -> dict | None:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    our_title = str(page.get("title") or "").strip()
    our_h1 = str(page.get("h1") or "").strip()
    competitors = [
        row for row in analysis.get("competitor_pages") or []
        if not row.get("error") and str(row.get("title") or "").strip()
    ]
    competitors.sort(key=lambda row: _safe_int(row.get("rank")) or 999)
    top_titles = [{"rank": row.get("rank"), "title": row.get("title", "")} for row in competitors[:5]]
    competitor_keyword_hits = sum(1 for row in top_titles if _keyword_in_text(keyword, row["title"]))
    keyword_missing = (
        bool(keyword)
        and not _keyword_in_text(keyword, our_title)
        and competitor_keyword_hits >= 3
    )
    title_too_short = bool(our_title) and len(our_title) < 30
    h1_empty = not our_h1
    if not (keyword_missing or title_too_short or h1_empty):
        return None
    demand = _keyword_demand(analysis)
    problems = []
    if keyword_missing:
        problems.append(
            f"Title '{our_title}' does not contain '{keyword}' while {competitor_keyword_hits}/{len(top_titles)} top competitor titles do"
        )
    if title_too_short:
        problems.append(f"Title is only {len(our_title)} characters")
    if h1_empty:
        problems.append("H1 is empty")
    instruction = (
        f"{'; '.join(problems)}. Rewrite the title (<=60 chars) to lead with the primary keyword intent; "
        "align the H1 with the title without duplicating it verbatim."
    )
    return {
        "id": f"rewrite_title_{order}",
        "order": order,
        "type": "rewrite_title",
        "priority": "high" if keyword_missing else "medium",
        "action": "Rewrite title and H1",
        "task_summary": f"Rewrite the title/H1 to match the intent behind '{keyword}'.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": "title and H1",
        "instruction": instruction,
        "rationale": "Title and H1 are the strongest on-page relevance and CTR signals; top-ranking competitors align them with the query.",
        "placement": "Page <title> tag and the main H1.",
        "acceptance_criteria": [
            f"The title contains the primary intent behind '{keyword}' within the first 60 characters.",
            "The H1 matches the title intent without duplicating it word for word.",
        ],
        "ai_agent_prompt": (
            f"Rewrite the title and H1 of {page.get('url', '')} for the keyword '{keyword}'. "
            "Lead with the keyword intent, keep the title under 60 characters, keep the brand suffix if present."
        ),
        "impact_score": round((60 + demand["score"] * 10) * _analysis_winnability_factor(analysis), 3),
        "evidence": {
            "our_title": our_title,
            "our_h1": our_h1,
            "keyword": keyword,
            "competitor_titles": top_titles,
            "keyword_impressions": demand["impressions"],
            "keyword_volume": demand["volume"],
        },
    }


def _depth_action(page: dict, analysis: dict, order: int) -> dict | None:
    comparison = analysis.get("content_comparison") or {}
    ours = comparison.get("ours") or {}
    bench = comparison.get("benchmark") or {}
    our_paragraphs = _safe_int(ours.get("paragraph_count"))
    median_paragraphs = _safe_float(bench.get("median_competitor_paragraphs"))
    if not (median_paragraphs >= 8 and our_paragraphs < 0.6 * median_paragraphs):
        return None
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    demand = _keyword_demand(analysis)
    median_headings = _safe_float(bench.get("median_competitor_headings"))
    delta = max(0, int(round(median_paragraphs - our_paragraphs)))
    instruction = (
        f"Page has {our_paragraphs} paragraphs / {_safe_int(ours.get('heading_count'))} headings; "
        f"top-5 competitor median is {median_paragraphs:g} paragraphs / {median_headings:g} headings. "
        f"Add roughly {delta} paragraphs by implementing the missing-topic tasks rather than padding existing sections."
    )
    return {
        "id": f"expand_depth_{order}",
        "order": order,
        "type": "expand_depth",
        "priority": "medium",
        "action": "Expand page depth",
        "task_summary": "Bring page depth closer to the competitor median via missing-topic sections.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": "page depth",
        "instruction": instruction,
        "rationale": "The page is much thinner than what currently ranks; depth should come from the detected missing topics, not filler.",
        "placement": "New sections from the missing-topic tasks.",
        "acceptance_criteria": [
            "New content comes from missing/partial topic tasks, not generic padding.",
            f"Paragraph count moves toward the competitor median ({median_paragraphs:g}).",
        ],
        "ai_agent_prompt": (
            f"Expand {page.get('url', '')} toward the competitor median depth by implementing the missing-topic sections "
            f"for '{keyword}'. Do not pad existing sections with filler."
        ),
        "impact_score": round((30 + demand["score"] * 8) * _analysis_winnability_factor(analysis), 3),
        "evidence": {"ours": ours, "benchmark": bench},
    }


def _structural_action(page: dict, analysis: dict, pattern: dict, order: int) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    demand = _keyword_demand(analysis)
    competitors = _safe_int(pattern.get("competitors"))
    signal = str(pattern.get("signal") or "structure")
    advice = str(pattern.get("advice") or "")
    instruction = (
        f"{advice} (ours: {pattern.get('ours')}, strongest competitor: {pattern.get('max_theirs')}, "
        f"seen on {competitors} competitor page(s))."
    )
    return {
        "id": f"structural_{order}",
        "order": order,
        "type": "structural",
        "priority": "high" if competitors >= 4 else "medium",
        "action": "Close a structural/GEO gap",
        "task_summary": f"Close the structural gap: {signal}.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": signal,
        "instruction": instruction,
        "rationale": "Ranking competitors consistently use this page structure signal; AI answer engines and rich results favor it.",
        "placement": "Page structure (markup, headings, or content blocks) as described.",
        "acceptance_criteria": [f"The page matches or beats the competitor pattern for: {signal}."],
        "ai_agent_prompt": f"On {page.get('url', '')}: {advice}",
        "impact_score": round((20 + competitors * 8 + demand["score"] * 5) * _analysis_winnability_factor(analysis), 3),
        "evidence": dict(pattern),
    }


def _paa_action(page: dict, analysis: dict, row: dict, order: int, *, top_question: bool) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    demand = _keyword_demand(analysis)
    question = str(row.get("question") or "").strip()
    return {
        "id": f"answer_paa_{order}",
        "order": order,
        "type": "answer_paa",
        "priority": "high" if top_question else "medium",
        "action": "Answer a People Also Ask question",
        "task_summary": f"Answer the PAA question: {question}",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": question,
        "instruction": (
            f"Add a question-form H3 '{question}' with a 40-60 word direct answer first, detail after. "
            "Candidate placement: FAQ block or the nearest related section."
        ),
        "rationale": "Google shows this question for the target keyword and the page has no close answer paragraph.",
        "placement": "FAQ block or nearest related section.",
        "acceptance_criteria": [
            "The first sentence after the heading answers the question completely on its own.",
            "The answer adds at least one concrete fact, condition, step, or example.",
        ],
        "ai_agent_prompt": (
            f"On {page.get('url', '')}, add a question-form section answering '{question}' for the keyword '{keyword}'. "
            "Direct answer first (40-60 words), supporting detail after."
        ),
        "impact_score": round((40 + demand["score"] * 10) * _analysis_winnability_factor(analysis), 3),
        "evidence": dict(row),
    }


def _retarget_or_new_page_action(page: dict, analysis: dict, order: int) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    intent = analysis.get("intent") or {}
    evidence = intent.get("evidence") or []
    instruction = (
        f"SERP intent is {intent.get('serp_intent', 'unknown')} but the selected page reads as "
        f"{intent.get('page_intent', 'unknown')}. Decide whether to retarget this page, create a new page for "
        f"'{keyword}', or choose a keyword whose SERP intent matches the current page before doing paragraph rewrites."
    )
    return {
        "id": f"retarget_or_new_page_{order}",
        "order": order,
        "type": "retarget_or_new_page",
        "priority": "critical",
        "action": "Retarget or create a new page",
        "task_summary": "Resolve SERP intent mismatch before content edits.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": "search intent mismatch",
        "instruction": instruction,
        "rationale": "Paragraph-level edits are unlikely to work when the page type does not match what the SERP rewards.",
        "placement": "Strategy decision before editing this URL.",
        "acceptance_criteria": [
            "The recommendation states whether to retarget this page, create a new page, or proceed with a clear reason.",
            "No paragraph rewrite task is started until the target page/keyword intent match is resolved.",
        ],
        "ai_agent_prompt": (
            f"For {page.get('url', '')}, address the intent mismatch first: SERP intent "
            f"{intent.get('serp_intent', 'unknown')} vs page intent {intent.get('page_intent', 'unknown')}. "
            "State whether to retarget, create a new page, or proceed, then justify the decision from evidence."
        ),
        "impact_score": RETARGET_ACTION_IMPACT,
        "evidence": {"intent": intent, "evidence": evidence},
    }


def _winnability_prerequisite_action(page: dict, analysis: dict, order: int) -> dict:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    win = analysis.get("winnability") or {}
    alternative = analysis.get("alternative_keyword") or {}
    if alternative.get("keyword") and alternative.get("band") == "winnable":
        alt_text = (
            f"Closest lower-difficulty variant from the keyword pool: '{alternative.get('keyword')}' (winnable)."
        )
    elif alternative.get("keyword"):
        alt_text = (
            f"Closest variant from the keyword pool: '{alternative.get('keyword')}' "
            "(winnability not yet assessed for this keyword)."
        )
    else:
        alt_text = "No lower-difficulty variant was found in the current keyword pool."
    instruction = (
        f"Content changes alone are unlikely to reach page 1 for '{keyword}'. {alt_text} "
        "Treat link acquisition or authority building as the prerequisite before investing in detailed rewrite work."
    )
    return {
        "id": f"winnability_prerequisite_{order}",
        "order": order,
        "type": "winnability_prerequisite",
        "priority": "critical",
        "action": "Treat authority as the prerequisite",
        "task_summary": "Content changes alone are unlikely to reach page 1.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": "SERP winnability",
        "instruction": instruction,
        "rationale": "Top-10 Domain Rating evidence shows an authority gap that content edits alone are unlikely to close.",
        "placement": "Recommendation header and prioritization.",
        "acceptance_criteria": [
            "The recommendation names the authority gap before content tasks.",
            "The plan either targets the easier variant keyword or includes link acquisition as a prerequisite.",
        ],
        "ai_agent_prompt": (
            f"For '{keyword}', state that content changes alone are unlikely to reach page 1. "
            "Recommend the easier variant keyword when available and list link acquisition as prerequisite."
        ),
        "impact_score": WINNABILITY_GATE_ACTION_IMPACT,
        "evidence": {"winnability": win, "alternative_keyword": alternative},
    }


def _featured_snippet_action(page: dict, analysis: dict, order: int) -> dict | None:
    keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
    answer_box = (analysis.get("serp_features") or {}).get("answer_box") or {}
    holder_url = str(answer_box.get("url") or "").strip()
    answer = str(answer_box.get("answer") or "").strip()
    if not holder_url or not answer:
        return None
    own_host = urlparse(str(page.get("url") or "")).netloc
    if own_host and _is_own_url(holder_url, own_host):
        return None
    demand = _keyword_demand(analysis)
    snippet_format = str(answer_box.get("format") or _snippet_format(answer) or "paragraph")
    word_count = _safe_int(answer_box.get("word_count")) or _word_count(answer)
    if snippet_format == "list":
        format_instruction = "Match the current list structure with concise, parallel list items immediately under the H2."
    elif snippet_format == "table":
        format_instruction = "Match the current table structure with a compact comparison table immediately under the H2."
    else:
        format_instruction = "Place a 40-55-word direct-answer paragraph immediately under the H2."
    instruction = (
        f"Win the featured snippet for '{keyword}'. Current holder: {holder_url}. "
        f"Current snippet is a {snippet_format} with {word_count} words: \"{answer[:240]}\". "
        f"Add an H2 phrased as the query ('{keyword}') and {format_instruction}"
    )
    acceptance = [
        f"The page contains an H2 phrased as the query: '{keyword}'.",
        f"The block immediately after that H2 uses the same snippet format ({snippet_format}).",
        "Paragraph snippets are 40-55 words; list/table snippets are concise and scannable.",
        "The answer is original, directly answers the query, and uses only sourceable facts.",
    ]
    return {
        "id": f"win_featured_snippet_{order}",
        "order": order,
        "type": "win_featured_snippet",
        "priority": "high",
        "action": "Win the featured snippet",
        "task_summary": f"Create a snippet-ready answer block for '{keyword}'.",
        "target_url": page.get("url", ""),
        "keyword": keyword,
        "topic": "featured snippet",
        "instruction": instruction,
        "rationale": "A competitor owns the answer box for this target keyword; matching the answer format gives the page a concrete snippet target.",
        "placement": f"Immediately under an H2 phrased as '{keyword}'.",
        "acceptance_criteria": acceptance,
        "content_brief": {
            "recommended_heading": keyword,
            "recommended_format": snippet_format,
            "placement": f"Immediately under an H2 phrased as '{keyword}'.",
            "paragraph_plan": [format_instruction, "Use the current holder only as intent/format evidence; write original copy."],
            "acceptance_criteria": acceptance,
            "ai_agent_prompt": (
                f"Add a featured-snippet candidate to {page.get('url', '')} for '{keyword}'. "
                f"Current holder {holder_url} uses {snippet_format} format and {word_count} words. {format_instruction}"
            ),
        },
        "ai_agent_prompt": (
            f"Add a featured-snippet candidate to {page.get('url', '')} for '{keyword}'. "
            f"Current holder {holder_url} uses {snippet_format} format and {word_count} words. {format_instruction}"
        ),
        "impact_score": round((65 + demand["score"] * 12) * _analysis_winnability_factor(analysis), 3),
        "evidence": {
            "holder_url": holder_url,
            "snippet_text": answer,
            "snippet_format": snippet_format,
            "snippet_word_count": word_count,
            "keyword_impressions": demand["impressions"],
            "keyword_volume": demand["volume"],
        },
    }


def _recommended_outline(analysis: dict) -> list[dict]:
    path = analysis.get("content_order_path") or {}
    rows: list[dict] = []
    for cluster in path.get("clusters") or []:
        rows.append({
            "label": cluster.get("label", ""),
            "status": "have" if cluster.get("ours_mean_order") is not None else "add",
            "competitor_pages": _safe_int(cluster.get("competitor_pages")),
            "competitor_mean_order": cluster.get("competitor_mean_order"),
            "sample_text": str(cluster.get("sample_text") or "")[:200],
        })
    have_labels = {row["label"] for row in rows}
    for cluster in path.get("missing_clusters") or []:
        if cluster.get("label") in have_labels:
            continue
        rows.append({
            "label": cluster.get("label", ""),
            "status": "add",
            "competitor_pages": _safe_int(cluster.get("competitor_pages")),
            "competitor_mean_order": cluster.get("competitor_mean_order"),
            "sample_text": str(cluster.get("sample_text") or "")[:200],
        })
    rows.sort(key=lambda row: (
        row.get("competitor_mean_order") is None,
        _safe_float(row.get("competitor_mean_order")),
        -_safe_int(row.get("competitor_pages")),
    ))
    for position, row in enumerate(rows, start=1):
        row["position"] = position
    return rows


def _action_points_for_analysis(page: dict, analysis: dict) -> list[dict]:
    if analysis.get("status") != "ok":
        return []
    actions: list[dict] = []
    leading_actions: list[dict] = []
    order = 1
    if (analysis.get("intent") or {}).get("match") == "mismatch":
        leading_actions.append(_retarget_or_new_page_action(page, analysis, order))
        order += 1
    if (analysis.get("winnability") or {}).get("band") == "unlikely":
        leading_actions.append(_winnability_prerequisite_action(page, analysis, order))
        order += 1
    snippet_action = _featured_snippet_action(page, analysis, order)
    if snippet_action:
        actions.append(snippet_action)
        order += 1
    for topic in (analysis.get("missing_topics") or [])[:8]:
        actions.append(_topic_action("add_topic", page, analysis, topic, order))
        order += 1
    for topic in (analysis.get("weak_topics") or [])[:6]:
        actions.append(_topic_action("strengthen_topic", page, analysis, topic, order))
        order += 1
    review_rows = analysis.get("off_intent_paragraphs") or analysis.get("own_paragraphs_to_review") or []
    for row in review_rows[:6]:
        actions.append(_paragraph_action(page, analysis, row, order))
        order += 1
    structural_rows = [
        pattern for pattern in analysis.get("structural_patterns") or []
        if _safe_int(pattern.get("competitors")) >= 2
    ]
    for pattern in structural_rows[:4]:
        actions.append(_structural_action(page, analysis, pattern, order))
        order += 1
    top_questions = {
        str(row.get("question") or "").strip()
        for row in ((analysis.get("serp_features") or {}).get("people_also_ask") or [])[:4]
    }
    missing_paa = [row for row in analysis.get("paa_coverage") or [] if row.get("status") == "missing"]
    for row in missing_paa[:4]:
        question = str(row.get("question") or "").strip()
        actions.append(_paa_action(page, analysis, row, order, top_question=question in top_questions))
        order += 1
    actions.sort(key=_action_priority_score)
    actions = leading_actions + actions
    for index, action in enumerate(actions, start=1):
        action["order"] = index
    return actions


def _page_content_brief(page: dict) -> dict:
    actions = sorted(page.get("action_points") or [], key=_action_priority_score)
    keywords = []
    for action in actions:
        keyword = str(action.get("keyword") or "").strip()
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    next_actions = []
    for action in actions[:10]:
        next_actions.append({
            "priority": action.get("priority", ""),
            "type": action.get("type", ""),
            "keyword": action.get("keyword", ""),
            "topic": action.get("topic", ""),
            "task_summary": action.get("task_summary") or action.get("action", ""),
            "placement": action.get("placement", ""),
            "impact_score": action.get("impact_score", 0),
        })
    prompt = (
        f"Improve {page.get('url', '')} using the ordered content tasks. Start with critical and high priority items, "
        "add direct answer blocks for missing topics, strengthen partial topics with concrete detail, and review low-alignment paragraphs. "
        "Preserve useful facts, avoid filler, and produce original content that satisfies the user intent behind the listed keywords."
    )
    return {
        "target_url": page.get("url", ""),
        "title": page.get("title", ""),
        "primary_keywords": keywords[:8],
        "total_actions": len(actions),
        "high_priority_actions": sum(1 for a in actions if a.get("priority") in {"critical", "high"}),
        "priority_score": round(sum(_safe_float(a.get("impact_score")) for a in actions[:10]), 3),
        "next_actions": next_actions,
        "paragraph_rules": _editorial_guidelines()["paragraph_rules"],
        "ai_agent_prompt": prompt,
    }


def _action_dedupe_key(action: dict) -> tuple[str, str, str]:
    topic = action.get("topic")
    if topic is None or str(topic).strip() == "":
        topic = action.get("paragraph_index")
    normalized = re.sub(r"\W+", " ", str(topic if topic is not None else "")).strip().lower()
    return (str(action.get("target_url") or ""), str(action.get("type") or ""), normalized)


def _dedupe_actions(actions: list[dict]) -> list[dict]:
    import difflib

    by_key: dict[tuple[str, str], dict] = {}
    dropped: dict[int, int] = {}
    ordered: list[dict] = []
    for action in sorted(actions, key=lambda a: -_safe_float(a.get("impact_score"))):
        key = _action_dedupe_key(action)
        if key in by_key:
            dropped[id(by_key[key])] = dropped.get(id(by_key[key]), 0) + 1
            continue
        by_key[key] = action
        ordered.append(action)
    deduped: list[dict] = []
    for action in ordered:
        duplicate_of = None
        if action.get("type") in {"add_topic", "strengthen_topic"}:
            label = _action_dedupe_key(action)[2]
            for kept in deduped:
                if kept.get("type") not in {"add_topic", "strengthen_topic"}:
                    continue
                if str(kept.get("target_url") or "") != str(action.get("target_url") or ""):
                    continue
                kept_label = _action_dedupe_key(kept)[2]
                if label and kept_label and difflib.SequenceMatcher(None, label, kept_label).ratio() >= 0.85:
                    duplicate_of = kept
                    break
        if duplicate_of is not None:
            dropped[id(duplicate_of)] = dropped.get(id(duplicate_of), 0) + 1
            continue
        deduped.append(action)
    for action in deduped:
        merged = dropped.get(id(action), 0)
        if merged:
            action["merged_duplicates"] = merged
    return deduped


def _attach_action_points(page_results: list[dict]) -> list[dict]:
    aggregate: list[dict] = []
    for page in page_results:
        page_actions: list[dict] = []
        title_added = False
        depth_added = False
        for analysis in page.get("analyses") or []:
            actions = _action_points_for_analysis(page, analysis)
            analysis["action_points"] = actions
            page_actions.extend(actions)
            if analysis.get("status") == "ok":
                if not title_added:
                    title_action = _title_gap_action(page, analysis, len(page_actions) + 1)
                    if title_action is not None:
                        page_actions.append(title_action)
                        title_added = True
                if not depth_added:
                    depth_action = _depth_action(page, analysis, len(page_actions) + 1)
                    if depth_action is not None:
                        page_actions.append(depth_action)
                        depth_added = True
        page_actions = _dedupe_actions(page_actions)
        page["action_points"] = sorted(page_actions, key=_action_priority_score)[:30]
        aggregate.extend(page["action_points"])
    aggregate = _dedupe_actions(aggregate)
    aggregate.sort(key=_action_priority_score)
    out = aggregate[:80]
    for index, action in enumerate(out, start=1):
        action["global_order"] = index
    for page in page_results:
        page["content_brief"] = _page_content_brief(page)
    return out


def _attach_ai_editor_briefs(
    page_results: list[dict],
    cache_dir: Path,
    config: SerpGapConfig,
    state: dict,
    page_competitor_content: dict[str, dict] | None = None,
    own_exts: dict[str, ExtractedPage] | None = None,
    report_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> None:
    if not config.ai_agent:
        return
    if state.get("status") != "ready":
        for page in page_results:
            page["ai_editor_brief"] = {
                "status": state.get("status", "disabled"),
                "message": "; ".join(state.get("notes") or []) or "AI editor brief was not generated.",
            }
        return
    own_exts = own_exts or {}
    page_competitor_content = page_competitor_content or {}
    use_workspace = (
        config.ai_agent_provider == "harnext"
        and report_dir is not None
        and bool(own_exts)
        and harnext_status()[0]
    )
    client = None
    if not use_workspace:
        try:
            client = build_agent_client(config.ai_agent_provider)
        except Exception as exc:
            state["status"] = "error"
            state.setdefault("errors", []).append(f"AI editor client init failed: {exc}")
            for page in page_results:
                page["ai_editor_brief"] = {"status": "error", "message": str(exc)}
            return
    for page in page_results:
        url = page.get("url", "")
        paragraph_count = len((page.get("own_content") or {}).get("paragraphs") or [])
        own_ext = own_exts.get(url)
        own_paragraphs = (
            list(own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
            if own_ext is not None
            else [str(row.get("text") or "") for row in (page.get("own_content") or {}).get("paragraphs") or []]
        )
        try:
            if use_workspace and own_ext is not None:
                _attach_workspace_brief(
                    page,
                    cache_dir,
                    config,
                    state,
                    own_ext=own_ext,
                    competitor_content=page_competitor_content.get(url, {}),
                    report_dir=report_dir,
                    paragraph_count=paragraph_count,
                    own_paragraphs=own_paragraphs,
                    embedder=embedder,
                )
            else:
                _attach_chat_brief(
                    page,
                    cache_dir,
                    config,
                    state,
                    client=client,
                    paragraph_count=paragraph_count,
                    own_paragraphs=own_paragraphs,
                    embedder=embedder,
                )
        except MissingOpenRouterKey:
            state["status"] = "missing_openrouter_api_key"
            message = "Set OPENROUTER_API_KEY in .env to generate the AI editor brief."
            page["ai_editor_brief"] = {"status": "missing_openrouter_api_key", "message": message}
            page["ai_recommendation"] = {"status": "missing_openrouter_api_key", "errors": [], "data": {}}
        except Exception as exc:
            state.setdefault("errors", []).append(f"Editor brief failed for {url}: {exc}")
            page["ai_editor_brief"] = {"status": "error", "message": str(exc)}
            page["ai_recommendation"] = {"status": "error", "errors": [str(exc)], "data": {}}


def _recommended_article_markdown(
    page: dict,
    recommendation: dict,
    own_paragraphs: list[str],
    verification: dict | None = None,
) -> str:
    """Assemble the full recommended article in final reading order, with the
    evidence for why this version should outperform the current page."""
    if not recommendation:
        return ""
    blocks = assemble_recommended_blocks(own_paragraphs, recommendation)
    title = (recommendation.get("title") or {}).get("recommended") or page.get("title") or ""
    h1 = (recommendation.get("h1") or {}).get("recommended") or page.get("h1") or title
    meta = (recommendation.get("meta_description") or {}).get("recommended") or ""

    heading_before: dict[int, dict] = {}
    for row in recommendation.get("outline") or []:
        if not isinstance(row, dict) or str(row.get("status") or "") == "remove":
            continue
        sources = [s for s in row.get("source_paragraphs") or [] if isinstance(s, int) and not isinstance(s, bool)]
        if not sources:
            continue
        first = min(sources)
        if first not in heading_before:
            heading_before[first] = row

    new_sections = recommendation.get("new_sections") or []
    decisions = {
        row.get("index"): row
        for row in recommendation.get("paragraph_decisions") or []
        if isinstance(row, dict)
    }

    lines: list[str] = [
        f"# Recommended article: {page.get('url', '')}",
        "",
        f"- **Title:** {title}",
        f"- **Meta description:** {meta}" if meta else "",
        f"- **H1:** {h1}",
        "",
        "---",
        "",
    ]
    for block in blocks:
        ref = str(block.get("ref") or "")
        source = str(block.get("source") or "")
        if source == "new":
            try:
                section = new_sections[int(ref[1:])]
            except (ValueError, IndexError, TypeError):
                section = {}
            heading = str(section.get("heading") or "").strip()
            draft = str(section.get("draft") or "").strip()
            covers = ", ".join(section.get("covers_paa") or [])
            topic = str(section.get("topic") or "").strip()
            why_bits = [bit for bit in (f"topic: {topic}" if topic else "", f"answers PAA: {covers}" if covers else "") if bit]
            lines.append(f"## {heading}" if heading else "")
            lines.append("")
            if why_bits:
                lines.append(f"*[new section — {'; '.join(why_bits)}]*")
                lines.append("")
            lines.append(draft)
            lines.append("")
            continue
        index = _safe_int(ref[1:]) if ref.startswith("P") else None
        if index is not None and index in heading_before:
            row = heading_before[index]
            level = min(max(_safe_int(row.get("level")) or 2, 2), 4)
            lines.append(f"{'#' * level} {str(row.get('heading') or '').strip()}")
            lines.append("")
        annotation = "rewritten" if source == "rewrite" else ""
        if annotation:
            reason = str((decisions.get(index) or {}).get("reason") or "").strip()
            lines.append(f"*[{annotation}{': ' + reason if reason else ''}]*")
            lines.append("")
        lines.append(str(block.get("text") or "").strip())
        lines.append("")

    removed = [
        f"[P{row.get('index')}] {str(row.get('reason') or '').strip()}"
        for row in recommendation.get("paragraph_decisions") or []
        if isinstance(row, dict) and str(row.get("decision") or "") in {"remove", "move", "merge"}
    ]
    lines.extend(["---", "", "## Why this version should rank better", ""])
    assessment = (recommendation.get("page_assessment") or {}).get("reason") or ""
    if assessment:
        lines.extend([f"- **Target page:** {assessment}", ""])
    summary = (verification or {}).get("summary") or {}
    if summary:
        lines.append(
            "- **Verified topic coverage (re-scored with the same embeddings used for the gap analysis):** "
            f"missing {summary.get('missing_before', 0)} -> {summary.get('missing_after', 0)}, "
            f"partial {summary.get('partial_before', 0)} -> {summary.get('partial_after', 0)}, "
            f"People Also Ask missing {summary.get('paa_missing_before', 0)} -> {summary.get('paa_missing_after', 0)}."
        )
        unresolved = summary.get("unresolved_critical") or []
        if unresolved:
            lines.append(f"- **Still uncovered critical/high topics:** {'; '.join(str(x) for x in unresolved[:10])}")
    unverified = _unverified_number_labels(verification or {})
    if unverified:
        lines.append(f"- **Unverified numeric claims:** {'; '.join(unverified[:10])}")
    if summary or unverified:
        lines.append("")
    for section in new_sections:
        heading = str(section.get("heading") or "").strip()
        covers = ", ".join(section.get("covers_paa") or [])
        if heading:
            lines.append(f"- **New section '{heading}'**{f' answers: {covers}' if covers else ''}.")
    schema_rows = recommendation.get("structured_data") or []
    if schema_rows:
        lines.append(
            "- **Structured data to add:** "
            + "; ".join(f"{row.get('type')} ({str(row.get('reason') or '')[:120]})" for row in schema_rows if isinstance(row, dict))
        )
    if removed:
        lines.append("- **Removed/merged content:** " + " · ".join(removed[:8]))
    title_reason = str((recommendation.get("title") or {}).get("reason") or "").strip()
    if title_reason:
        lines.append(f"- **Title change:** {title_reason}")
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def _recommendation_numeric_text(blocks: list[dict], recommendation: dict) -> str:
    parts: list[str] = []
    for key in ("title", "h1", "meta_description"):
        row = recommendation.get(key) or {}
        if isinstance(row, dict):
            value = str(row.get("recommended") or "").strip()
            if value:
                parts.append(value)
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _numeric_evidence_texts(page: dict, own_paragraphs: list[str] | None = None) -> list[str]:
    """Every text the agent may legitimately source numbers from.

    Mirrors what the agent actually sees: the full own-page paragraphs (own_content
    stores truncated copies) plus the same evidence payload written to the agent
    workspace / chat prompt, so numbers from untouched kept content or any evidence
    field never get flagged as invented.
    """
    texts: list[str] = [str(paragraph or "") for paragraph in own_paragraphs or []]
    texts.append(json.dumps(agent_evidence_payload(page), ensure_ascii=False, sort_keys=True))
    for row in (page.get("own_content") or {}).get("paragraphs") or []:
        if isinstance(row, dict):
            texts.append(str(row.get("text") or ""))
    texts.extend(str(item or "") for item in page.get("keywords") or [])
    for analysis in page.get("analyses") or []:
        texts.append(json.dumps(analysis.get("keyword") or {}, ensure_ascii=False, sort_keys=True))
        features = analysis.get("serp_features") or {}
        texts.append(json.dumps(features, ensure_ascii=False, sort_keys=True))
        for topic in analysis.get("topics") or []:
            for example in topic.get("examples") or []:
                texts.append(str(example.get("paragraph") or ""))
                texts.append(str(example.get("url") or ""))
        for row in ((analysis.get("paragraph_match_heatmap") or {}).get("rows") or []):
            for cell in row.get("cells") or []:
                texts.append(str(cell.get("paragraph") or ""))
        for row in analysis.get("structural_patterns") or []:
            texts.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        for row in analysis.get("competitor_pages") or []:
            texts.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return [text for text in texts if str(text or "").strip()]


def _unverified_number_labels(verification: dict) -> list[str]:
    labels: list[str] = []
    for claim in verification.get("unverified_numbers") or []:
        text = str(claim.get("text") or claim.get("number") or "").strip()
        context = str(claim.get("context") or "").strip()
        if not text:
            continue
        labels.append(f"{text} ({context[:140]})" if context else text)
    return labels


def _verification_repair_prompt(verification: dict, mode: str = "workspace") -> str:
    if mode == "chat":
        coverage_fix = "Add or strengthen sections in your recommendation to cover them."
        closing = " Output the corrected brief and the full recommendation JSON block again."
    else:
        coverage_fix = "Update recommendation.json (add or strengthen sections) to cover them."
        closing = " Modify recommendation.json and brief.md only."
    instructions: list[str] = []
    unresolved = (verification.get("summary") or {}).get("unresolved_critical") or []
    if unresolved:
        instructions.append(
            "Verification: these critical/high SERP topics are still not covered by your recommendation: "
            + "; ".join(str(label) for label in unresolved[:10])
            + ". "
            + coverage_fix
        )
    unverified = _unverified_number_labels(verification)
    if unverified:
        instructions.append(
            "Replace or source these numbers: "
            + "; ".join(unverified[:12])
            + ". Use only numbers present in the Evidence JSON or mark the claim [NEEDS DATA]."
        )
    if not instructions:
        return ""
    return " ".join(instructions) + closing


def _verification_for(
    page: dict,
    recommendation: dict,
    own_paragraphs: list[str],
    embedder: Embedder | None,
) -> dict:
    if not recommendation:
        return {}
    try:
        blocks = assemble_recommended_blocks(own_paragraphs, recommendation)
        verification = (
            verify_recommendation(
                blocks,
                page.get("analyses") or [],
                embed_fn=lambda texts: embedder.encode(texts, batch_size=64).astype(np.float32),
            )
            if embedder is not None
            else {"topics": [], "paa": [], "summary": {}}
        )
        numeric = verify_numeric_claims(
            _recommendation_numeric_text(blocks, recommendation),
            _numeric_evidence_texts(page, own_paragraphs),
        )
        verification["numeric_claims"] = numeric
        verification["unverified_numbers"] = numeric.get("unverified") or []
        return verification
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _attach_workspace_brief(
    page: dict,
    cache_dir: Path,
    config: SerpGapConfig,
    state: dict,
    *,
    own_ext: ExtractedPage,
    competitor_content: dict,
    report_dir: Path,
    paragraph_count: int,
    own_paragraphs: list[str] | None = None,
    embedder: Embedder | None = None,
) -> None:
    url = page.get("url", "")
    own_paragraphs = own_paragraphs or []
    workspace = write_agent_workspace(
        report_dir,
        page,
        own_ext,
        competitor_content,
        schema_doc=RECOMMENDATION_SCHEMA_DOC,
    )
    evidence_text = (workspace / "evidence.json").read_text(encoding="utf-8")
    cache_key = content_hash(
        json.dumps({"evidence": content_hash(evidence_text), "model": config.ai_agent_model}, sort_keys=True)
    )

    def _session(extra_prompt: str = "", max_turns: int | None = None):
        return run_harnext_workspace_session(
            workspace,
            model=config.ai_agent_model,
            max_turns=max_turns or config.ai_agent_max_turns,
            extra_prompt=extra_prompt,
        )

    def _run() -> dict:
        completion = _session()
        recommendation = (completion.raw or {}).get("recommendation") or {}
        errors = validate_recommendation(recommendation, paragraph_count)
        repair_attempted = False
        if errors:
            repair_attempted = True
            repair_prompt = (
                "Your recommendation.json failed validation with these errors:\n- "
                + "\n- ".join(errors[:20])
                + "\nFix recommendation.json only so it satisfies the contract in TASK.md."
            )
            completion = _session(repair_prompt, max_turns=max(6, config.ai_agent_max_turns // 2))
            recommendation = (completion.raw or {}).get("recommendation") or {}
            errors = validate_recommendation(recommendation, paragraph_count)
        verification = {}
        verification_repair_attempted = False
        if not errors:
            verification = _verification_for(page, recommendation, own_paragraphs, embedder)
            repair_prompt = _verification_repair_prompt(verification)
            if repair_prompt:
                verification_repair_attempted = True
                repair_completion = _session(repair_prompt, max_turns=max(6, config.ai_agent_max_turns // 2))
                candidate = (repair_completion.raw or {}).get("recommendation") or {}
                if not validate_recommendation(candidate, paragraph_count):
                    completion = repair_completion
                    recommendation = candidate
                    verification = _verification_for(page, recommendation, own_paragraphs, embedder)
                # An invalid repair candidate is discarded: keep the original valid
                # recommendation and its verification instead of failing the page.
        return {
            "brief": completion.text,
            "recommendation": recommendation,
            "errors": errors,
            "repair_attempted": repair_attempted,
            "verification": verification,
            "verification_repair_attempted": verification_repair_attempted,
            "provider": completion.provider,
            "model": completion.model,
            "workspace": str(workspace),
        }

    payload = cached_workspace_completion(
        cache_dir,
        kind=f"editor-workspace-{_url_report_slug(url)}",
        key=cache_key,
        runner=_run,
        refresh=config.ai_agent_refresh,
    )
    if payload.get("cache_status") == "hit":
        state["cache_hits"] += 1
    errors = payload.get("errors") or []
    page["ai_editor_brief"] = {
        "status": "ok",
        "provider": payload.get("provider", "harnext"),
        "model": payload.get("model", config.ai_agent_model),
        "cache_status": payload.get("cache_status", "miss"),
        "markdown": _clean_ai_markdown(str(payload.get("brief") or "")),
    }
    rec_data = payload.get("recommendation") or {}
    page["ai_recommendation"] = {
        "status": "ok" if not errors else "invalid_recommendation",
        "errors": errors,
        "data": rec_data,
        "repair_attempted": bool(payload.get("repair_attempted")),
        "verification": payload.get("verification") or {},
        "verification_repair_attempted": bool(payload.get("verification_repair_attempted")),
        "workspace": str(Path(payload.get("workspace") or workspace)),
        "article_markdown": (
            _recommended_article_markdown(page, rec_data, own_paragraphs, payload.get("verification") or {})
            if not errors else ""
        ),
    }
    state["editor_briefs"] += 1


def _attach_chat_brief(
    page: dict,
    cache_dir: Path,
    config: SerpGapConfig,
    state: dict,
    *,
    client,
    paragraph_count: int,
    own_paragraphs: list[str] | None = None,
    embedder: Embedder | None = None,
) -> None:
    messages = build_editor_brief_messages(page)
    completion = cached_completion(
        cache_dir,
        kind=f"editor-brief-{_url_report_slug(page.get('url', ''))}",
        messages=messages,
        client=client,
        model=config.ai_agent_model,
        refresh=config.ai_agent_refresh,
        temperature=0.2,
        timeout=180,
    )
    if completion.cache_status == "hit":
        state["cache_hits"] += 1
    if completion.fallback_from:
        state.setdefault("notes", []).append(
            f"Editor brief for {page.get('url', '')} used {completion.provider} after {completion.fallback_from}."
        )
    recommendation = parse_recommendation(completion.text)
    errors = validate_recommendation(recommendation, paragraph_count) if recommendation else ["no recommendation JSON found in completion"]
    brief_markdown = completion.text
    if recommendation:
        brief_markdown = re.sub(r"```json\s*\{.*?\}\s*```", "", brief_markdown, flags=re.DOTALL | re.IGNORECASE).strip()
    verification = {}
    verification_repair_attempted = False
    if not errors:
        verification = _verification_for(page, recommendation, own_paragraphs or [], embedder)
        repair_prompt = _verification_repair_prompt(verification, mode="chat")
        if repair_prompt:
            verification_repair_attempted = True
            repair_messages = messages + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": repair_prompt},
            ]
            repair_completion = cached_completion(
                cache_dir,
                kind=f"editor-brief-repair-{_url_report_slug(page.get('url', ''))}",
                messages=repair_messages,
                client=client,
                model=config.ai_agent_model,
                refresh=config.ai_agent_refresh,
                temperature=0.2,
                timeout=180,
            )
            if repair_completion.cache_status == "hit":
                state["cache_hits"] += 1
            candidate = parse_recommendation(repair_completion.text)
            if candidate and not validate_recommendation(candidate, paragraph_count):
                completion = repair_completion
                recommendation = candidate
                brief_markdown = re.sub(
                    r"```json\s*\{.*?\}\s*```",
                    "",
                    repair_completion.text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
                verification = _verification_for(page, recommendation, own_paragraphs or [], embedder)
            # An invalid repair candidate is discarded: keep the original valid
            # recommendation and its verification instead of failing the page.
    page["ai_editor_brief"] = {
        "status": "ok",
        "provider": completion.provider,
        "model": completion.model,
        "cache_status": completion.cache_status,
        "fallback_from": completion.fallback_from,
        "markdown": _clean_ai_markdown(brief_markdown),
    }
    page["ai_recommendation"] = {
        "status": "ok" if not errors else "invalid_recommendation",
        "errors": errors,
        "data": recommendation,
        "repair_attempted": False,
        "verification": verification,
        "verification_repair_attempted": verification_repair_attempted,
        "workspace": "",
        "article_markdown": (
            _recommended_article_markdown(page, recommendation, own_paragraphs or [], verification)
            if not errors else ""
        ),
    }
    state["editor_briefs"] += 1


def _clean_ai_markdown(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        stripped = line.rstrip()
        key = _line_key(stripped)
        if key and not stripped.lstrip().startswith(("#", "-", "*", "1.", "2.", "3.", "4.", "5.", "6.")):
            if key in seen:
                continue
            seen.add(key)
        if not stripped and out and not out[-1]:
            continue
        out.append(stripped)
    return "\n".join(out).strip()


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
    keyword_url_ridges = _keyword_url_ridges(rows, matrix, keyword_indexes)
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
    return {
        "points": points[:1600],
        "shown": min(len(points), 1600),
        "total": len(points),
        "duplicates_removed": duplicates_removed,
        "keyword_url_ridges": keyword_url_ridges,
    }


def _keyword_url_ridges(rows: list[dict], matrix: np.ndarray, keyword_indexes: list[int], *, max_urls: int = 18) -> dict:
    paragraph_indexes = [
        i for i, row in enumerate(rows)
        if row.get("entity_type") == "paragraph" and str(row.get("url") or "").strip()
    ]
    if not keyword_indexes or not paragraph_indexes or not len(matrix):
        return {"keywords": [], "rows": []}

    norm = _normalize_matrix(matrix)
    keywords = []
    for order, index in enumerate(keyword_indexes):
        row = rows[index]
        keywords.append({
            "order": order,
            "keyword": str(row.get("text") or ""),
            "impressions": _safe_int(row.get("impressions")),
            "clicks": _safe_int(row.get("clicks")),
            "traffic": round(_safe_float(row.get("traffic")), 4),
            "volume": _safe_int(row.get("volume")),
        })

    by_url: dict[str, list[int]] = {}
    url_meta: dict[str, dict] = {}
    for index in paragraph_indexes:
        row = rows[index]
        url = str(row.get("url") or "")
        if not url:
            continue
        by_url.setdefault(url, []).append(index)
        meta = url_meta.setdefault(url, {
            "url": url,
            "domain": row.get("domain") or urlparse(url).netloc,
            "source": row.get("source") or "",
            "rank": row.get("rank") or "",
        })
        if not meta.get("rank") and row.get("rank"):
            meta["rank"] = row.get("rank")

    ridge_rows = []
    keyword_matrix = norm[keyword_indexes]
    for url, indexes in by_url.items():
        if not indexes:
            continue
        para_matrix = norm[indexes]
        cells = []
        row_score = 0.0
        strong_count_total = 0
        for keyword_order, keyword in enumerate(keywords):
            sims = para_matrix @ keyword_matrix[keyword_order]
            if not len(sims):
                continue
            best_local = int(np.argmax(sims))
            best_index = indexes[best_local]
            clean_sims = np.clip(sims, -1.0, 1.0)
            max_similarity = float(clean_sims[best_local])
            top_values = sorted([float(v) for v in clean_sims], reverse=True)[:3]
            top3_similarity = sum(top_values) / len(top_values) if top_values else 0.0
            strong_count = int(np.sum(clean_sims >= 0.72))
            partial_count = int(np.sum(clean_sims >= 0.58))
            strong_count_total += strong_count
            row_score = max(row_score, max_similarity)
            cells.append({
                "keyword": keyword["keyword"],
                "keyword_order": keyword_order,
                "max_similarity": round(max_similarity, 4),
                "top3_similarity": round(top3_similarity, 4),
                "strong_paragraphs": strong_count,
                "partial_paragraphs": partial_count,
                "best_paragraph_index": indexes.index(best_index),
                "best_paragraph": str(rows[best_index].get("text") or "")[:260],
            })
        meta = url_meta.get(url, {})
        ridge_rows.append({
            **meta,
            "paragraph_count": len(indexes),
            "row_score": round(row_score, 4),
            "strong_paragraphs": strong_count_total,
            "cells": cells,
        })

    ridge_rows.sort(key=lambda row: (
        row.get("source") != "ours",
        -_safe_float(row.get("row_score")),
        _safe_int(row.get("rank")) or 999,
        row.get("domain") or row.get("url") or "",
    ))
    return {"keywords": keywords, "rows": ridge_rows[:max_urls]}


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
        "action_points": sum(len(a.get("action_points") or []) for a in analyses if a.get("status") == "ok"),
        "content_briefs": sum(1 for p in page_results if p.get("content_brief")),
        "ai_agent_briefs": sum(1 for p in page_results if (p.get("ai_editor_brief") or {}).get("status") == "ok"),
    }


def _write_outputs(payload: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload["report_dir"] = str(report_dir)
    payload.setdefault("summary", {})["report_dir"] = str(report_dir)
    (report_dir / "serp_gap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(report_dir / "serp_gap.csv", _csv_rows(payload))
    _write_csv(report_dir / "serp_gap_actions.csv", _action_csv_rows(payload))
    (report_dir / "serp_gap_todo.md").write_text(_todo_markdown(payload), encoding="utf-8")
    for page in payload.get("pages") or []:
        article = (page.get("ai_recommendation") or {}).get("article_markdown") or ""
        if article.strip():
            slug = _url_report_slug(str(page.get("url") or ""))
            (report_dir / f"recommended-article-{slug}.md").write_text(article, encoding="utf-8")
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


def _action_csv_rows(payload: dict) -> list[dict]:
    rows = []
    for action in payload.get("action_points") or []:
        brief = action.get("content_brief") or {}
        evidence = action.get("evidence") or {}
        rows.append({
            "global_order": action.get("global_order", ""),
            "priority": action.get("priority", ""),
            "impact_score": action.get("impact_score", ""),
            "target_url": action.get("target_url", ""),
            "keyword": action.get("keyword", ""),
            "type": action.get("type", ""),
            "topic": action.get("topic", ""),
            "task_summary": action.get("task_summary", ""),
            "instruction": action.get("instruction", ""),
            "placement": action.get("placement", ""),
            "recommended_format": brief.get("recommended_format", ""),
            "paragraph_plan": " | ".join(brief.get("paragraph_plan") or brief.get("paragraph_rules") or []),
            "acceptance_criteria": " | ".join(action.get("acceptance_criteria") or []),
            "ai_agent_prompt": action.get("ai_agent_prompt", ""),
            "competitor_coverage": evidence.get("competitor_coverage", ""),
            "best_competitor_rank": evidence.get("best_competitor_rank", ""),
            "our_similarity": evidence.get("our_best_similarity", evidence.get("similarity_to_serp_topics", "")),
            "example_url": evidence.get("example_url", ""),
            "example_paragraph": evidence.get("example_paragraph", evidence.get("paragraph", "")),
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


def _line_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _append_unique(lines: list[str], seen: set[str], line: str = "") -> None:
    if not line:
        if lines and lines[-1] != "":
            lines.append("")
        return
    key = _line_key(line)
    if key in seen:
        return
    seen.add(key)
    lines.append(line)


def _markdown_action_rows(actions: list[dict], limit: int = 25) -> list[dict]:
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for action in sorted(actions or [], key=_action_priority_score):
        if action.get("type") == "review_paragraph":
            continue
        key = (
            str(action.get("type") or ""),
            _line_key(action.get("keyword") or ""),
            _line_key(action.get("topic") or action.get("task_summary") or action.get("instruction") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(action)
        if len(rows) >= limit:
            break
    return rows


def _paragraph_todo_rows(page: dict, limit: int = 10) -> list[dict]:
    by_index: dict[int, dict] = {}
    for analysis in page.get("analyses") or []:
        keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "")
        for row in (analysis.get("paragraph_match_heatmap") or {}).get("rows") or []:
            paragraph_index = _safe_int(row.get("paragraph_index"))
            current = by_index.get(paragraph_index)
            candidate = {
                **row,
                "keyword": keyword,
                "score": _safe_float(row.get("max_similarity")),
            }
            if current is None or candidate["score"] < _safe_float(current.get("score")):
                by_index[paragraph_index] = candidate
    rows = sorted(by_index.values(), key=lambda row: (_safe_float(row.get("score")), _safe_int(row.get("paragraph_index"))))
    return rows[:limit]


def _append_ai_editor_brief_markdown(lines: list[str], seen: set[str], brief: dict, recommendation: dict | None = None) -> None:
    if not brief:
        return
    _append_unique(lines, seen, "### AI Agent TODO")
    verification_summary = ((recommendation or {}).get("verification") or {}).get("summary") or {}
    if verification_summary:
        _append_unique(
            lines,
            seen,
            "- Coverage check: missing {mb} -> {ma}, partial {pb} -> {pa}, PAA missing {qb} -> {qa}".format(
                mb=verification_summary.get("missing_before", 0),
                ma=verification_summary.get("missing_after", 0),
                pb=verification_summary.get("partial_before", 0),
                pa=verification_summary.get("partial_after", 0),
                qb=verification_summary.get("paa_missing_before", 0),
                qa=verification_summary.get("paa_missing_after", 0),
            ),
        )
        unresolved = verification_summary.get("unresolved_critical") or []
        if unresolved:
            _append_unique(lines, seen, f"- Still uncovered critical/high topics: {'; '.join(unresolved[:10])}")
    unverified = _unverified_number_labels((recommendation or {}).get("verification") or {})
    if unverified:
        _append_unique(lines, seen, f"- Unverified numeric claims: {'; '.join(unverified[:10])}")
    status = brief.get("status", "")
    if status == "ok" and brief.get("markdown"):
        meta = [
            brief.get("provider") and f"provider: {brief.get('provider')}",
            brief.get("model") and f"model: {brief.get('model')}",
            brief.get("cache_status") and f"cache: {brief.get('cache_status')}",
            brief.get("fallback_from") and f"fallback: {brief.get('fallback_from')}",
        ]
        if any(meta):
            _append_unique(lines, seen, f"- Agent run: {'; '.join(str(item) for item in meta if item)}")
            _append_unique(lines, seen)
        local_seen: set[str] = set()
        for raw_line in _clean_ai_markdown(str(brief.get("markdown") or "")).splitlines():
            line = raw_line.rstrip()
            key = _line_key(line)
            if key and key in local_seen:
                continue
            if key:
                local_seen.add(key)
            lines.append(line)
        _append_unique(lines, seen)
        return
    message = brief.get("message") or "AI editor brief was not generated."
    _append_unique(lines, seen, f"- Not generated: {status or 'disabled'}")
    _append_unique(lines, seen, f"- Reason: {message}")
    _append_unique(lines, seen)


def _todo_markdown(payload: dict) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    domain = payload.get("domain", "")
    pages = payload.get("pages") or []
    _append_unique(lines, seen, "# SERP Gap TODO")
    _append_unique(lines, seen)
    _append_unique(lines, seen, f"- Domain: {domain}")
    _append_unique(lines, seen, f"- Status: {payload.get('status', '')}")
    report_summary = payload.get("summary") or {}
    if report_summary:
        task_count = sum(len(_markdown_action_rows(page.get("action_points") or [], limit=9999)) for page in pages)
        if not task_count:
            task_count = len(_markdown_action_rows(payload.get("action_points") or [], limit=9999))
        if not task_count:
            task_count = report_summary.get("action_points", 0)
        _append_unique(
            lines,
            seen,
            f"- Scope: {report_summary.get('pages_analyzed', report_summary.get('pages_selected', 0))} page(s), "
            f"{report_summary.get('keywords_selected', 0)} keyword(s), {task_count} content task(s)",
        )
    _append_unique(lines, seen)
    if not pages:
        _append_unique(lines, seen, "No analyzed page tasks were generated.")
        return "\n".join(lines).rstrip() + "\n"

    rules = (payload.get("editorial_guidelines") or {}).get("paragraph_rules") or []
    for page_index, page in enumerate(pages, start=1):
        page_url = page.get("url", "")
        _append_unique(lines, seen, f"## {page_index}. {page.get('title') or page_url}")
        _append_unique(lines, seen, f"- URL: {page_url}")
        keywords = []
        for analysis in page.get("analyses") or []:
            keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        if keywords:
            _append_unique(lines, seen, f"- Target keywords: {', '.join(keywords[:12])}")
        _append_unique(lines, seen)

        _append_ai_editor_brief_markdown(lines, seen, page.get("ai_editor_brief") or {}, page.get("ai_recommendation") or {})

        if rules:
            _append_unique(lines, seen, "### Writing Rules")
            for rule in rules:
                _append_unique(lines, seen, f"- {rule}")
            _append_unique(lines, seen)

        evidence_lines = []
        for analysis in page.get("analyses") or []:
            keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
            for reason in analysis.get("visual_summary") or []:
                evidence_lines.append(f"- {keyword}: {reason}" if keyword else f"- {reason}")
        if evidence_lines:
            _append_unique(lines, seen, "### Why Changes Are Needed")
            for line in evidence_lines[:12]:
                _append_unique(lines, seen, line)
            _append_unique(lines, seen)

        actions = _markdown_action_rows(page.get("action_points") or [], limit=25)
        if actions:
            _append_unique(lines, seen, "### Ordered Content Tasks")
            placement_counts: dict[str, int] = {}
            for action in actions:
                placement = str(action.get("placement") or "").strip()
                if placement:
                    placement_counts[placement] = placement_counts.get(placement, 0) + 1
            default_placement = ""
            if placement_counts:
                default_placement, default_count = max(placement_counts.items(), key=lambda item: item[1])
                if default_count < 2:
                    default_placement = ""
            if default_placement:
                _append_unique(lines, seen, f"- Default placement for repeated rewrite tasks: {default_placement}")
                _append_unique(lines, seen)
            for index, action in enumerate(actions, start=1):
                task_seen: set[str] = set()
                priority = str(action.get("priority") or "medium").upper()
                task = action.get("task_summary") or action.get("action") or action.get("type") or "Content task"
                _append_unique(lines, seen, f"#### {index}. [{priority}] {task}")
                if action.get("keyword"):
                    _append_unique(lines, task_seen, f"- Keyword: {action.get('keyword')}")
                if action.get("topic"):
                    _append_unique(lines, task_seen, f"- Topic: {action.get('topic')}")
                if action.get("instruction"):
                    _append_unique(lines, task_seen, f"- Change: {action.get('instruction')}")
                if action.get("placement"):
                    placement = str(action.get("placement") or "")
                    if placement != default_placement:
                        _append_unique(lines, task_seen, f"- Placement: {placement}")
                acceptance = action.get("acceptance_criteria") or []
                focused_acceptance = acceptance[:1]
                focused_acceptance.extend(
                    item for item in acceptance[1:]
                    if str(item).lower().startswith("the section naturally uses related terms")
                )
                for item in focused_acceptance[:2]:
                    _append_unique(lines, task_seen, f"- Done when: {item}")
                evidence = action.get("evidence") or {}
                evidence_parts = []
                if evidence.get("competitor_coverage"):
                    evidence_parts.append(f"{evidence.get('competitor_coverage')} ranking page(s) cover it")
                if evidence.get("best_competitor_rank"):
                    evidence_parts.append(f"best competitor rank #{evidence.get('best_competitor_rank')}")
                if evidence.get("our_best_similarity") is not None:
                    evidence_parts.append(f"current similarity {evidence.get('our_best_similarity')}")
                if evidence_parts:
                    _append_unique(lines, task_seen, f"- Evidence: {'; '.join(evidence_parts)}")
                if evidence.get("example_url"):
                    _append_unique(lines, task_seen, f"- Example URL: {evidence.get('example_url')}")
                _append_unique(lines, seen)

        paa_lines: list[str] = []
        for analysis in page.get("analyses") or []:
            keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
            for row in analysis.get("paa_coverage") or []:
                if row.get("status") in {"missing", "partial"}:
                    suffix = f" (keyword: {keyword})" if keyword else ""
                    paa_lines.append(f"- [{row.get('status')}] {row.get('question')}{suffix}")
        if paa_lines:
            _append_unique(lines, seen, "### People Also Ask")
            for line in paa_lines:
                _append_unique(lines, seen, line)
            _append_unique(lines, seen)

        paragraph_rows = _paragraph_todo_rows(page)
        if paragraph_rows:
            _append_unique(lines, seen, "### Paragraph Review")
            for row in paragraph_rows:
                index = _safe_int(row.get("paragraph_index")) + 1
                score = _safe_float(row.get("score"))
                keyword = row.get("keyword", "")
                paragraph = str(row.get("paragraph") or "").strip()
                if score >= 0.78:
                    action = "Keep only if it supports the target intent; otherwise move it lower."
                elif score >= 0.62:
                    action = "Rewrite with a clearer direct answer and add missing concrete details."
                else:
                    action = "Rewrite, merge, move, or remove; this paragraph is weakly aligned with top-ranking pages."
                _append_unique(lines, seen, f"- P{index} ({score:.2f} best SERP paragraph match, keyword: {keyword}): {action}")
                if paragraph:
                    _append_unique(lines, seen, f"  Current: {paragraph[:220]}")
            _append_unique(lines, seen)

        _append_unique(lines, seen, "### Do Not")
        for item in [
            "Do not add generic introductions or repeat the page title.",
            "Do not copy competitor wording.",
            "Do not create duplicate sections for the same topic; merge overlapping tasks into one focused section.",
            "Do not publish a paragraph unless it adds a fact, example, condition, step, limitation, comparison, or decision criterion.",
        ]:
            _append_unique(lines, seen, f"- {item}")
        _append_unique(lines, seen)

    return "\n".join(lines).rstrip() + "\n"


def _strip_centroids(node):
    """Remove bulky embedding vectors from a payload copy used for HTML rendering."""
    if isinstance(node, dict):
        node.pop("centroid", None)
        for value in node.values():
            _strip_centroids(value)
    elif isinstance(node, list):
        for item in node:
            _strip_centroids(item)
    return node


def _html(payload: dict) -> str:
    payload = _strip_centroids(copy.deepcopy(payload))
    data = json.dumps(payload, ensure_ascii=False)
    template = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SERP Semantic Gap</title>
<style>
:root{--ink:#17202a;--muted:#5d6d7e;--line:#d7dee8;--soft:#f5f7fa;--panel:#fff;--ours:#176a35;--comp:#2d5b9a;--kw:#8a4b00;--missing:#b42318;--partial:#9a6700;--covered:#176a35;--shadow:0 1px 3px rgba(22,34,51,.08)}
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:14px;line-height:1.45}a{color:#1b5dbf;text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1440px;margin:0 auto;padding:24px}.topbar{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:18px}.title h1{font-size:28px;line-height:1.1;margin:0 0 8px}.title p{margin:0;color:var(--muted);max-width:820px}.summary{display:grid;grid-template-columns:repeat(8,minmax(112px,1fr));gap:10px;margin:18px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;box-shadow:var(--shadow)}.metric b{display:block;font-size:22px;line-height:1.1}.metric span{color:var(--muted);font-size:12px}.page-section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:18px 0 28px;box-shadow:var(--shadow);overflow:hidden}.page-head{padding:18px 20px;border-bottom:1px solid var(--line);background:#fff}.page-head h2{font-size:21px;margin:0 0 6px}.url{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.keyword-card{padding:20px;border-top:1px solid var(--line)}.keyword-card:first-of-type{border-top:0}.keyword-grid{display:grid;grid-template-columns:minmax(460px,1.2fr) minmax(420px,.8fr);gap:18px;align-items:start}.keyword-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}.keyword-head h3{font-size:19px;margin:0}.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:3px 8px;background:#fff;font-size:12px;color:var(--muted)}.chip.missing{color:var(--missing);border-color:#f1b4ad;background:#fff7f6}.chip.partial{color:var(--partial);border-color:#e8cf85;background:#fff9e8}.chip.covered{color:var(--covered);border-color:#a8d5b6;background:#f1faf4}.panel{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.panel h4{margin:0;padding:11px 12px;border-bottom:1px solid var(--line);font-size:14px;background:var(--soft)}.panel-body{padding:12px}.scatter-wrap{position:relative}.scatter{width:100%;height:390px;display:block;background:#fbfcfe}.scatter-point{cursor:pointer}.scatter-point:focus{outline:none;stroke:#111;stroke-width:2.4}.scatter-tooltip{display:none;position:absolute;left:12px;right:12px;bottom:12px;max-height:250px;overflow:auto;background:linear-gradient(180deg,#fff,#f8fbff);border:1px solid #b9c7d8;border-radius:10px;box-shadow:0 14px 34px rgba(22,34,51,.22);padding:0;z-index:2}.scatter-tooltip.open{display:block}.tip-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:11px 12px;border-bottom:1px solid #e3e9f2;background:#f4f7fb}.tip-title{font-weight:750;font-size:13px;color:#142033}.tip-sub{margin-top:2px;color:#637083;font-size:11px}.tip-close{border:0;background:#e7edf5;border-radius:6px;padding:2px 8px;cursor:pointer}.tip-body{padding:11px 12px}.tip-badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.tip-badge{display:inline-flex;align-items:center;border:1px solid #d5deea;border-radius:999px;padding:3px 7px;background:#fff;font-size:11px;color:#405166}.tip-badge.ours{border-color:#9bd0ae;color:#176a35;background:#f1faf4}.tip-badge.competitor{border-color:#acc3e5;color:#2d5b9a;background:#f3f7ff}.tip-badge.keyword{border-color:#e7c889;color:#8a4b00;background:#fff9e8}.tip-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:9px}.tip-field{border:1px solid #edf1f5;border-radius:7px;background:#fff;padding:6px 7px;min-width:0}.tip-field span{display:block;color:#768395;font-size:10px;text-transform:uppercase;letter-spacing:.04em}.tip-field strong{display:block;color:#1b2838;font-size:12px;overflow-wrap:anywhere}.tip-text{border-left:3px solid #8fb2df;background:#f7faff;border-radius:6px;padding:8px 9px;color:#263445;font-size:12px;line-height:1.45}.tip-explain{color:#637083;font-size:12px;margin-bottom:9px}.scatter-controls{position:absolute;top:10px;right:10px;display:flex;gap:5px;z-index:3}.scatter-controls button{border:1px solid #c9d3df;background:#fff;color:#263445;border-radius:6px;padding:3px 8px;font-size:12px;line-height:1;box-shadow:0 1px 3px rgba(22,34,51,.12);cursor:pointer}.scatter-controls button:hover{background:#f1f5f9}.scatter.is-panning{cursor:grabbing}.scatter{cursor:grab}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.tables{display:grid;grid-template-columns:1fr;gap:14px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid #edf1f5;text-align:left;vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#fbfcfe}.topic-label{font-weight:600}.coverage-missing{color:var(--missing);font-weight:700}.coverage-partial{color:var(--partial);font-weight:700}.coverage-covered{color:var(--covered);font-weight:700}.cluster-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.cluster{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fff}.cluster strong{display:block;margin-bottom:5px}.bar{height:7px;background:#e9eef5;border-radius:999px;overflow:hidden;margin:8px 0}.bar span{display:block;height:100%;background:#5f8cc9}.muted{color:var(--muted)}.empty{padding:18px;color:var(--muted)}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}.competitors li,.review li{margin:0 0 8px}.competitors,.review{padding-left:18px;margin:0}.mini{font-size:12px;color:var(--muted)}@media(max-width:980px){.wrap{padding:14px}.summary{grid-template-columns:repeat(2,1fr)}.keyword-grid,.two-col{grid-template-columns:1fr}.scatter{height:320px}.topbar{display:block}}
  .md-body{font-size:13px;line-height:1.55;color:var(--ink)}
  .md-body h3,.md-body h4,.md-body h5,.md-body h6{margin:14px 0 6px;font-size:14px}
  .md-body p{margin:6px 0}
  .md-body ul,.md-body ol{margin:6px 0;padding-left:22px}
  .md-body li{margin:3px 0}
  .md-body code{background:#f0f3f8;border:1px solid #dde4ee;border-radius:4px;padding:1px 4px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
  .md-body pre{background:#f7f9fc;border:1px solid #dde4ee;border-radius:6px;padding:10px;overflow:auto}
  .md-body pre code{background:none;border:0;padding:0}
  :root {
    --audit-bg: #f5f7fb;
    --audit-panel: #ffffff;
    --audit-panel-soft: #f8fafc;
    --audit-line: #dbe3ee;
    --audit-text: #182230;
    --audit-muted: #667085;
    --audit-accent: #2563eb;
    --audit-accent-dark: #1d4ed8;
    --audit-accent-soft: #eef4ff;
    --audit-green: #1f9d66;
    --audit-red: #cf5060;
    --audit-blue: #2563eb;
    --audit-shadow: 0 12px 32px rgba(24, 34, 48, 0.08);
    --audit-radius: 8px;
  }
  body {
    color: var(--audit-text);
    background: var(--audit-bg);
  }
  .wrap { max-width: 1500px; margin-left: 300px; padding: 28px 28px 72px; }
  .topbar {
    position: relative;
    overflow: hidden;
    padding: 30px 32px;
    margin-bottom: 28px;
    border: 1px solid var(--audit-line);
    border-radius: 14px;
    background: var(--audit-panel);
    box-shadow: var(--audit-shadow);
  }
  .topbar::before {
    display: none;
  }
  .title h1 {
    position: relative;
    font-size: 2.15rem;
    line-height: 1.08;
    letter-spacing: 0;
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
  .collapsible-panel > summary {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    list-style: none;
    margin: 0;
    padding: 11px 12px;
    border-bottom: 0;
    background: var(--audit-panel-soft);
    color: var(--audit-text);
    font-size: 14px;
    font-weight: 800;
  }
  .collapsible-panel > summary::-webkit-details-marker { display: none; }
  .collapsible-panel[open] > summary {
    border-bottom: 1px solid var(--audit-line);
  }
  .collapsible-title {
    min-width: 0;
    flex: 1;
  }
  .collapsible-meta {
    color: var(--audit-muted);
    font-size: 0.74rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .collapsible-state {
    border: 1px solid var(--audit-line);
    border-radius: 999px;
    background: #ffffff;
    color: var(--audit-muted);
    font-size: 0.72rem;
    font-weight: 800;
    padding: 3px 8px;
    white-space: nowrap;
  }
  .collapsible-state::after { content: "Open"; }
  .collapsible-panel[open] .collapsible-state::after { content: "Close"; }
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
    background: #fff;
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
  .url-demand-table-wrap {
    overflow-x: auto;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
    margin-top: 14px;
  }
  .url-demand-table {
    min-width: 980px;
    margin: 0;
  }
  .url-demand-table td:first-child,
  .url-demand-table th:first-child {
    min-width: 240px;
  }
  .url-demand-table td.metric-number,
  .url-demand-table th.metric-number {
    text-align: right;
    white-space: nowrap;
  }
  .url-demand-bar {
    min-width: 120px;
  }
  .url-demand-bar span {
    display: block;
    height: 9px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--audit-green), var(--audit-accent));
  }
  .visual-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(300px, 0.82fr);
    gap: 14px;
    align-items: start;
  }
  .visual-card {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: #ffffff;
    padding: 12px;
  }
  .visual-card h5 {
    margin: 0 0 8px;
    color: var(--audit-text);
    font-size: 0.88rem;
  }
  .why-list {
    display: grid;
    gap: 8px;
    margin-bottom: 12px;
  }
  .why-item {
    border-left: 3px solid var(--audit-accent);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    color: var(--audit-text);
    font-size: 0.82rem;
    line-height: 1.42;
    padding: 8px 10px;
  }
  .comparison-chart {
    display: grid;
    gap: 9px;
  }
  .comparison-row {
    display: grid;
    grid-template-columns: minmax(145px, 0.85fr) minmax(150px, 1.05fr) minmax(84px, 0.42fr);
    gap: 8px;
    align-items: center;
  }
  .comparison-row.is-ours .comparison-label {
    color: var(--audit-green);
  }
  .comparison-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 800;
  }
  .comparison-label span {
    display: block;
    color: var(--audit-muted);
    font-size: 0.72rem;
    font-weight: 600;
  }
  .comparison-stack {
    display: grid;
    gap: 5px;
  }
  .comparison-track {
    position: relative;
    height: 9px;
    overflow: hidden;
    border-radius: 999px;
    background: #e9eef5;
  }
  .comparison-bar {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--audit-accent);
  }
  .comparison-bar.paragraphs { background: #2563eb; }
  .comparison-bar.headings { background: #0891b2; }
  .comparison-bar.topics { background: #1f9d66; }
  .comparison-meta {
    color: var(--audit-muted);
    font-size: 0.7rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
    text-align: right;
  }
  .delta-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 12px;
  }
  .delta-box {
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    padding: 9px;
  }
  .delta-box strong {
    display: block;
    color: var(--audit-text);
    font-size: 1.2rem;
    line-height: 1;
  }
  .delta-box span {
    color: var(--audit-muted);
    font-size: 0.72rem;
  }
  .coverage-heatmap-wrap {
    overflow-x: auto;
  }
  .coverage-heatmap {
    min-width: 760px;
    display: grid;
    gap: 6px;
  }
  .heatmap-row,
  .heatmap-head {
    display: grid;
    grid-template-columns: minmax(220px, 1.35fr) repeat(var(--heatmap-cols, 7), minmax(72px, 0.62fr));
    gap: 6px;
    align-items: stretch;
  }
  .heatmap-head {
    color: var(--audit-muted);
    font-size: 0.72rem;
    font-weight: 800;
  }
  .heatmap-head > div {
    min-width: 0;
    overflow: hidden;
    line-height: 1.25;
    overflow-wrap: anywhere;
  }
  .heatmap-topic {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    padding: 8px;
  }
  .heatmap-topic strong {
    display: block;
    overflow: hidden;
    color: var(--audit-text);
    font-size: 0.78rem;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .topic-evidence {
    margin-top: 7px;
    border-top: 1px solid var(--audit-line);
    padding-top: 6px;
  }
  .topic-evidence summary {
    cursor: pointer;
    color: var(--audit-accent-dark);
    font-size: 0.72rem;
    font-weight: 800;
  }
  .topic-snippet {
    margin-top: 7px;
    border-left: 3px solid #b9cdfc;
    padding-left: 7px;
  }
  .topic-snippet b {
    display: block;
    color: var(--audit-text);
    font-size: 0.71rem;
  }
  .topic-snippet p {
    margin: 2px 0 0;
    color: var(--audit-muted);
    font-size: 0.71rem;
    line-height: 1.32;
  }
  .heatmap-cell {
    display: flex;
    min-height: 42px;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: #ffffff;
    color: var(--audit-muted);
    font-size: 0.72rem;
    font-weight: 800;
    text-align: center;
  }
  .heatmap-cell.missing { background: #fff1f3; border-color: #f4b4be; color: var(--audit-red); }
  .heatmap-cell.partial { background: #eef4ff; border-color: #b9cdfc; color: var(--audit-accent-dark); }
  .heatmap-cell.covered { background: #eefbf4; border-color: #a5dabb; color: var(--audit-green); }
  .heatmap-cell.not_seen { background: #f8fafc; color: #98a2b3; }
  .heatmap-cell.ours {
    box-shadow: inset 0 0 0 2px rgba(31, 157, 102, 0.2);
  }
  .heatmap-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 9px;
    color: var(--audit-muted);
    font-size: 0.74rem;
  }
  .heatmap-legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .heatmap-legend i {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    display: inline-block;
  }
  .paragraph-heatmap-wrap {
    overflow-x: auto;
  }
  .paragraph-heatmap {
    min-width: 840px;
    display: grid;
    gap: 6px;
  }
  .paragraph-heatmap .heatmap-head,
  .paragraph-heatmap .paragraph-row {
    display: grid;
    grid-template-columns: minmax(280px, 1.25fr) repeat(var(--paragraph-cols, 6), minmax(104px, 0.58fr));
    gap: 6px;
    align-items: stretch;
  }
  .paragraph-heatmap .heatmap-cell {
    flex-direction: column;
    gap: 2px;
  }
  .paragraph-topic {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    padding: 8px;
  }
  .paragraph-topic strong {
    display: block;
    color: var(--audit-text);
    font-size: 0.78rem;
  }
  .paragraph-topic span {
    display: block;
    margin-top: 3px;
    overflow: hidden;
    color: var(--audit-muted);
    font-size: 0.72rem;
    line-height: 1.25;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .heatmap-cell.strong { background: #eefbf4; border-color: #a5dabb; color: var(--audit-green); }
  .heatmap-cell.weak { background: #fff1f3; border-color: #f4b4be; color: var(--audit-red); }
  .heatmap-cell.no_match { background: #f8fafc; color: #98a2b3; }
  .content-path-card .section-note {
    margin-bottom: 10px;
  }
  .content-path-wrap {
    overflow-x: auto;
  }
  .content-path-chart {
    display: block;
    min-width: 900px;
    width: 100%;
    height: auto;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: #fbfcfe;
  }
  .path-axis {
    stroke: #d7dee8;
    stroke-width: 1;
  }
  .parallel-axis {
    stroke: #cbd5e1;
    stroke-width: 1.4;
  }
  .parallel-axis-label {
    fill: var(--audit-text);
    font-size: 10px;
    font-weight: 800;
  }
  .parallel-axis-meta {
    fill: var(--audit-muted);
    font-size: 9.5px;
  }
  .parallel-topic-line {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2.4;
    opacity: 0.58;
    cursor: pointer;
    transition: opacity 140ms ease, stroke-width 140ms ease, filter 140ms ease;
  }
  .parallel-topic-line.missing-ours {
    stroke-dasharray: 5 4;
  }
  .parallel-topic-marker {
    stroke: #ffffff;
    stroke-width: 1.2;
    cursor: pointer;
    transition: opacity 140ms ease, stroke-width 140ms ease, r 140ms ease, filter 140ms ease;
  }
  .content-path-wrap.has-active .parallel-topic-line,
  .content-path-wrap.has-active .parallel-topic-marker {
    opacity: 0.12;
    filter: grayscale(1);
  }
  .content-path-wrap.has-active .parallel-topic-line.is-active {
    opacity: 1;
    stroke-width: 4.4;
    filter: drop-shadow(0 2px 5px rgba(24, 34, 48, 0.22));
  }
  .content-path-wrap.has-active .parallel-topic-marker.is-active {
    opacity: 1;
    stroke: var(--audit-text);
    stroke-width: 2.2;
    filter: drop-shadow(0 2px 5px rgba(24, 34, 48, 0.18));
  }
  .parallel-topic-line:focus,
  .parallel-topic-marker:focus {
    outline: none;
    stroke: var(--audit-text);
  }
  .path-lane-label {
    fill: var(--audit-text);
    font-size: 11px;
    font-weight: 800;
  }
  .path-lane-meta {
    fill: var(--audit-muted);
    font-size: 10px;
  }
  .path-line {
    fill: none;
    stroke-width: 2.2;
    opacity: 0.68;
  }
  .path-point {
    stroke: #fff;
    stroke-width: 1.2;
    opacity: 0.9;
  }
  .path-point.ours {
    stroke: #0b3d1e;
    stroke-width: 2;
  }
  .path-findings {
    display: grid;
    gap: 8px;
    margin-top: 10px;
  }
  .path-finding {
    border-left: 3px solid var(--audit-accent);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    color: var(--audit-text);
    font-size: 0.78rem;
    line-height: 1.35;
    padding: 8px 10px;
  }
  .path-clusters {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }
  .path-cluster-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    max-width: 260px;
    border: 1px solid var(--audit-line);
    border-radius: 999px;
    background: #ffffff;
    color: var(--audit-muted);
    font-size: 0.72rem;
    padding: 4px 8px;
  }
  .path-cluster-chip i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    flex: 0 0 auto;
  }
  .path-unmatched-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .path-unmatched-card {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: #fffdf9;
    padding: 10px;
  }
  .path-unmatched-card strong {
    display: block;
    color: var(--audit-text);
    font-size: 0.82rem;
    margin-bottom: 2px;
  }
  .path-unmatched-list {
    display: grid;
    gap: 7px;
    margin-top: 8px;
  }
  .path-unmatched-item {
    border-left: 3px solid var(--audit-accent);
    border-radius: 8px;
    background: var(--audit-panel-soft);
    padding: 7px 8px;
  }
  .path-unmatched-item b {
    display: block;
    color: var(--audit-text);
    font-size: 0.76rem;
  }
  .keyword-ridge-wrap {
    overflow-x: auto;
  }
  .keyword-ridge-chart {
    display: block;
    min-width: 920px;
    width: 100%;
    height: auto;
    border: 1px solid var(--audit-line);
    border-radius: 8px;
    background: #fbfcfe;
    margin-top: 10px;
  }
  .keyword-ridge-baseline {
    stroke: #d7dee8;
    stroke-width: 1;
  }
  .keyword-ridge-area {
    fill: rgba(37, 99, 235, 0.16);
    stroke: #2563eb;
    stroke-width: 1.6;
  }
  .keyword-ridge-area.ours {
    fill: rgba(31, 157, 102, 0.18);
    stroke: #176a35;
  }
  .keyword-ridge-point {
    stroke: #ffffff;
    stroke-width: 1.2;
  }
  .keyword-ridge-label {
    fill: var(--audit-text);
    font-size: 10.5px;
    font-weight: 800;
  }
  .keyword-ridge-meta,
  .keyword-ridge-axis {
    fill: var(--audit-muted);
    font-size: 9.5px;
  }
  .keyword-frequency-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }
  .frequency-block {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
    padding: 12px;
  }
  .frequency-block h5 {
    margin: 0 0 8px;
    color: var(--audit-text);
    font-size: 0.86rem;
  }
  .word-cloud {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 10px;
    min-height: 72px;
    margin: 6px 0 10px;
    padding: 10px;
    border-radius: 14px;
    background: #fbf2e7;
  }
  .word-token {
    display: inline-flex;
    align-items: center;
    max-width: 100%;
    color: var(--audit-accent-dark);
    font-weight: 800;
    line-height: 1.05;
    white-space: nowrap;
  }
  .frequency-list {
    display: grid;
    gap: 5px;
  }
  .frequency-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
    color: var(--audit-muted);
    font-size: 0.76rem;
  }
  .frequency-row strong {
    min-width: 0;
    overflow: hidden;
    color: var(--audit-text);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .keyword-metrics-table-wrap {
    overflow-x: auto;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
  }
  .keyword-metrics-table {
    min-width: 1080px;
    margin: 0;
  }
  .keyword-metrics-table td:first-child,
  .keyword-metrics-table th:first-child {
    min-width: 210px;
  }
  .keyword-metrics-table .metric-number {
    text-align: right;
    white-space: nowrap;
  }
  .url-keyword-table {
    width: 100%;
    margin-top: 10px;
    border-collapse: collapse;
    font-size: 0.76rem;
  }
  .url-keyword-table th,
  .url-keyword-table td {
    border-bottom: 1px solid var(--audit-line);
    padding: 6px 7px;
    text-align: left;
    vertical-align: top;
  }
  .url-keyword-table th {
    color: var(--audit-muted);
    font-size: 0.68rem;
    text-transform: uppercase;
  }
  .url-keyword-table td.metric-number,
  .url-keyword-table th.metric-number {
    text-align: right;
    white-space: nowrap;
  }
  .url-keyword-table tr:last-child td {
    border-bottom: 0;
  }
  .action-list {
    display: grid;
    gap: 10px;
  }
  .brief-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
  }
  .brief-card {
    min-width: 0;
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
    padding: 12px;
  }
  .brief-card h5 {
    margin: 0 0 6px;
    color: var(--audit-text);
    font-size: 0.9rem;
  }
  .brief-card ol,
  .brief-card ul,
  .action-card ol,
  .action-card ul {
    margin: 7px 0 0;
    padding-left: 18px;
  }
  .brief-card li,
  .action-card li {
    margin-bottom: 4px;
  }
  .action-card {
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
    padding: 12px;
  }
  .action-card strong {
    display: block;
    color: var(--audit-text);
    margin-bottom: 5px;
  }
  .action-card .instruction {
    color: var(--audit-text);
    line-height: 1.45;
  }
  .action-card .rationale {
    margin-top: 7px;
    color: var(--audit-muted);
    font-size: 0.78rem;
  }
  .brief-label {
    color: var(--audit-muted);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin-top: 8px;
    text-transform: uppercase;
  }
  .prompt-box {
    border-left: 3px solid var(--audit-accent);
    border-radius: 10px;
    background: var(--audit-panel-soft);
    color: var(--audit-text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.76rem;
    line-height: 1.45;
    margin-top: 8px;
    padding: 9px 10px;
    white-space: pre-wrap;
  }
  .evidence-details,
  .diagnostic-details {
    border: 1px solid var(--audit-line);
    border-radius: 16px;
    background: #fffdf9;
    margin-top: 14px;
    overflow: hidden;
  }
  .evidence-details > summary,
  .diagnostic-details > summary {
    cursor: pointer;
    font-weight: 800;
    padding: 11px 13px;
  }
  .evidence-details[open] > summary,
  .diagnostic-details[open] > summary {
    border-bottom: 1px solid var(--audit-line);
    background: var(--audit-panel-soft);
  }
  .diagnostic-body {
    padding: 12px;
  }
  .action-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }
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
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.96);
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
    border-radius: 8px;
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
    overflow-wrap: anywhere;
    word-break: normal;
    white-space: normal;
    font-size: 0.84rem;
    line-height: 1.25;
  }
  .report-nav-button.nav-keyword {
    margin-left: 10px;
    width: calc(100% - 10px);
    padding: 6px 10px;
  }
  .report-nav-button.nav-keyword .report-nav-label {
    font-size: 0.78rem;
    line-height: 1.25;
  }
  @media(max-width:1180px) {
    .report-sidebar { position: static; width: auto; margin: 14px; }
    .wrap { margin-left: 0; padding: 14px; }
    #status { min-width: 0; text-align: left; white-space: normal; }
    .tables, .topic-chart-row, .visual-grid, .comparison-row { grid-template-columns: 1fr; }
    .comparison-meta { text-align: left; }
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
  <div class="topbar"><div class="title"><h1>SERP Content Task Board</h1><p>Prioritized page edits from live SERP competitors, paragraph evidence, and keyword demand. Start with content briefs and action cards; open evidence sections only when validating a task.</p></div><div class="url" id="status"></div></div>
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
function mdInline(s){let t=esc(s);t=t.replace(/`([^`]+)`/g,'<code>$1</code>');t=t.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');t=t.replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<i>$2</i>');return t;}
function mdToHtml(md){
  // All text is escaped via esc() BEFORE any tags are added, so embedded HTML/scripts render inert.
  const lines=String(md||'').replace(/\r\n/g,'\n').split('\n');const out=[];let list=null;let code=false;let codeLines=[];
  const closeList=()=>{if(list){out.push(list==='ul'?'</ul>':'</ol>');list=null;}};
  for(const raw of lines){
    if(code){if(/^```/.test(raw.trim())){out.push(`<pre><code>${esc(codeLines.join('\n'))}</code></pre>`);code=false;codeLines=[];}else{codeLines.push(raw);}continue;}
    const line=raw.trimEnd();const trimmed=line.trim();
    if(/^```/.test(trimmed)){closeList();code=true;codeLines=[];continue;}
    const h=trimmed.match(/^(#{1,4})\s+(.*)$/);
    if(h){closeList();const level=Math.min(h[1].length+2,6);out.push(`<h${level}>${mdInline(h[2])}</h${level}>`);continue;}
    const ul=trimmed.match(/^[-*]\s+(.*)$/);const ol=trimmed.match(/^\d+[.)]\s+(.*)$/);
    if(ul||ol){const kind=ul?'ul':'ol';if(list!==kind){closeList();out.push(kind==='ul'?'<ul>':'<ol>');list=kind;}out.push(`<li>${mdInline((ul||ol)[1])}</li>`);continue;}
    closeList();
    if(!trimmed){continue;}
    out.push(`<p>${mdInline(trimmed)}</p>`);
  }
  if(code&&codeLines.length)out.push(`<pre><code>${esc(codeLines.join('\n'))}</code></pre>`);
  closeList();
  return `<div class="md-body">${out.join('')}</div>`;
}
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
function metrics(){const s=data.summary||{};return [['Pages',s.pages_analyzed||0],['Keywords',s.keywords_selected||0],['Actions',s.action_points||0],['AI briefs',s.ai_agent_briefs||0],['Content briefs',s.content_briefs||0],['Missing topics',s.missing_topics||0],['Partial topics',s.partial_topics||0],['Review paragraphs',s.review_paragraphs ?? s.off_intent_paragraphs ?? 0],['SERP calls',s.serp_api_calls_after_cache ?? s.serp_api_calls ?? 0]];}
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
const frequencyStopwords=new Set('a an and are as at be by can for from has have how if in into is it its more not of on or our page pages that the their this to top use user users with without your you we what when where which who why will all each any same over under also than then these those there here about after before because been being do does did done get got make made many much most some such own other them they his her she he was were would should could may might liveagent live agent'.split(' '));
function frequencyTokens(text){return String(text||'').toLowerCase().replace(/&[a-z]+;/g,' ').replace(/[^a-z0-9]+/g,' ').split(/\s+/).filter(token=>token.length>2&&!frequencyStopwords.has(token)&&!/^\d+$/.test(token));}
function addFrequency(map, term, amount){if(!term)return;map.set(term,(map.get(term)||0)+amount);}
function frequencyRowsFromMap(map,limit=18){return [...map.entries()].map(([term,score])=>({term,score})).sort((a,b)=>b.score-a.score||a.term.localeCompare(b.term)).slice(0,limit);}
function collectFrequency(points,types,weight=1){const map=new Map();for(const p of points||[]){if(!types.includes(p.entity_type))continue;const tokens=frequencyTokens(p.text);for(const token of tokens)addFrequency(map,token,weight);for(let i=0;i<tokens.length-1;i++){if(tokens[i]!==tokens[i+1])addFrequency(map,`${tokens[i]} ${tokens[i+1]}`,weight*1.35);}}return frequencyRowsFromMap(map);}
function weightedFrequency(points){const weights={title:8,h1:7,h2:5,h3:4,h4:3,h5:3,h6:3,paragraph:1};const map=new Map();for(const p of points||[]){const weight=weights[p.entity_type]||0;if(!weight)continue;const tokens=frequencyTokens(p.text);for(const token of tokens)addFrequency(map,token,weight);for(let i=0;i<tokens.length-1;i++){if(tokens[i]!==tokens[i+1])addFrequency(map,`${tokens[i]} ${tokens[i+1]}`,weight*1.35);}}return frequencyRowsFromMap(map,28);}
function wordCloud(rows){if(!rows.length)return '<div class="empty">No repeated keywords found.</div>';const max=Math.max(1,...rows.map(row=>row.score));const min=Math.min(...rows.map(row=>row.score));return `<div class="word-cloud">${rows.map(row=>{const scale=max===min?0.5:(row.score-min)/(max-min);const size=0.78+scale*1.42;return `<span class="word-token" style="font-size:${size.toFixed(2)}rem" title="${esc(row.term)} · ${n(row.score)}">${esc(row.term)}</span>`;}).join('')}</div>`;}
function frequencyList(rows){if(!rows.length)return '';return `<div class="frequency-list">${rows.slice(0,10).map(row=>`<div class="frequency-row"><strong title="${esc(row.term)}">${esc(row.term)}</strong><span>${n(row.score)}</span></div>`).join('')}</div>`;}
function frequencyBlock(title,rows){return `<div class="frequency-block"><h5>${esc(title)}</h5>${wordCloud(rows.slice(0,16))}${frequencyList(rows)}</div>`;}
function keywordFrequencySection(points){const titleRows=collectFrequency(points,['title']);const h1Rows=collectFrequency(points,['h1']);const headingRows=collectFrequency(points,['h2','h3','h4','h5','h6']);const paragraphRows=collectFrequency(points,['paragraph']);const weightedRows=weightedFrequency(points);return `<div class="panel" style="margin-top:14px"><h4>Keyword Frequency Analysis</h4><div class="panel-body">${sectionNote('This analysis counts repeated terms and short phrases in the aggregate content set. Title and H1 terms are weighted more heavily than H2-H6 headings, and paragraph terms have the lowest weight, so the combined word cloud reflects where terms appear in high-impact page elements.')}<div class="frequency-block" style="margin-bottom:14px"><h5>Weighted Content Keyword Cloud</h5>${wordCloud(weightedRows)}${frequencyList(weightedRows)}</div><div class="keyword-frequency-grid">${frequencyBlock('Title Keywords',titleRows)}${frequencyBlock('H1 Keywords',h1Rows)}${frequencyBlock('H2-H6 Keywords',headingRows)}${frequencyBlock('Paragraph Keywords',paragraphRows)}</div></div></div>`;}
function keywordSerpSummary(rankings){const out=new Map();for(const row of rankings||[]){for(const item of row.keywords||[]){const keyword=String(item.keyword||'').trim().toLowerCase();if(!keyword)continue;const current=out.get(keyword)||{top10_urls:0,best_rank:999,best_url:'',domains:new Set()};current.top10_urls+=1;const rank=Number(item.rank||999);if(rank<current.best_rank){current.best_rank=rank;current.best_url=row.url||'';}if(row.domain)current.domains.add(row.domain);out.set(keyword,current);}}return out;}
function keywordMetricRows(){const serp=keywordSerpSummary(data.serp_url_rankings||[]);return (data.selected_keywords||[]).map(row=>{const keyword=String(row.keyword||'').trim();const summary=serp.get(keyword.toLowerCase())||{top10_urls:0,best_rank:0,best_url:'',domains:new Set()};return{...row,keyword,serp_top10_urls:summary.top10_urls||0,serp_best_rank:summary.best_rank===999?0:summary.best_rank,serp_best_url:summary.best_url||'',serp_domains:summary.domains?[...summary.domains].length:0};}).sort((a,b)=>Number(b.traffic||0)-Number(a.traffic||0)||Number(b.impressions||0)-Number(a.impressions||0)||Number(b.volume||0)-Number(a.volume||0)||a.keyword.localeCompare(b.keyword));}
function metricCell(value,decimals=0){const number=Number(value||0);return number?number.toLocaleString(undefined,{maximumFractionDigits:decimals,minimumFractionDigits:decimals&&number%1?decimals:0}):'';}
function keywordMetricsTable(){const rows=keywordMetricRows();if(!rows.length)return '<div class="empty">No selected keywords available.</div>';return `<div class="keyword-metrics-table-wrap"><table class="keyword-metrics-table"><thead><tr><th>Keyword</th><th>Keyword source</th><th>API metrics source</th><th>Analyzed URL</th><th>Matched metrics URL</th><th class="metric-number">Source pos</th><th class="metric-number">Impr</th><th class="metric-number">Clicks</th><th class="metric-number">Traffic</th><th class="metric-number">Volume</th><th class="metric-number">SERP URLs</th><th class="metric-number">Best SERP</th><th>Best ranking URL</th></tr></thead><tbody>${rows.map(row=>{const apiSource=row.metrics_source||(hasDemandMetrics(row)?row.source:'No API metric match');return `<tr><td><strong>${esc(row.keyword)}</strong></td><td>${esc(row.source||'')}</td><td>${esc(apiSource)}</td><td>${urlLink(row.url,row.page_title||row.url)}</td><td>${urlLink(row.metrics_url,row.metrics_url?urlDomain(row.metrics_url)||row.metrics_url:'')}</td><td class="metric-number">${metricCell(row.position,1)}</td><td class="metric-number">${metricCell(row.impressions)}</td><td class="metric-number">${metricCell(row.clicks)}</td><td class="metric-number">${metricCell(row.traffic,2)}</td><td class="metric-number">${metricCell(row.volume)}</td><td class="metric-number">${metricCell(row.serp_top10_urls)}</td><td class="metric-number">${row.serp_best_rank?`#${esc(row.serp_best_rank)}`:''}</td><td>${urlLink(row.serp_best_url,row.serp_best_url?urlDomain(row.serp_best_url)||row.serp_best_url:'')}</td></tr>`;}).join('')}</tbody></table></div>`;}
function keywordMetricsSection(){return `<div class="panel" style="margin-top:14px"><h4>Keyword Metrics From APIs</h4><div class="panel-body">${sectionNote('This table lists each selected keyword and every metric currently available from connected sources. Search Console-style data contributes impressions, clicks, and average position when present; Ahrefs contributes traffic and volume. If no demand metric matches a manual, People Also Ask, or People Also Search keyword, the API metrics source shows No API metric match and the SERP columns still show observed ranking positions from the live SERP fetch.')} ${keywordMetricsTable()}</div></div>`;}
function actionPriorityClass(priority){return priority==='critical'||priority==='high'?'missing':priority==='medium'?'partial':'covered';}
function listHtml(items,type='ul'){if(!items||!items.length)return '';return `<${type}>${items.map(item=>`<li>${esc(item)}</li>`).join('')}</${type}>`;}
function promptBox(text){return text?`<div class="brief-label">AI agent prompt</div><div class="prompt-box">${esc(text)}</div>`:'';}
function collapsiblePanel(title,body,{meta='',open=false,className='',style='margin-top:14px'}={}){const cls=`panel collapsible-panel ${className||''}`.trim();const styleAttr=style?` style="${esc(style)}"`:'';const metaHtml=meta?`<span class="collapsible-meta">${esc(meta)}</span>`:'';return `<details class="${esc(cls)}"${open?' open':''}${styleAttr}><summary><span class="collapsible-title">${esc(title)}</span>${metaHtml}<span class="collapsible-state" aria-hidden="true"></span></summary><div class="panel-body">${body}</div></details>`;}
function diagnosticDetails(title,body,open=false){return `<details class="diagnostic-details"${open?' open':''}><summary>${esc(title)}</summary><div class="diagnostic-body">${body}</div></details>`;}
function actionEvidenceHtml(row){const ev=row.evidence||{};const profile=ev.quality_profile||{};const lines=[ev.keyword_impressions&&`${n(ev.keyword_impressions)} impressions`,ev.keyword_clicks&&`${n(ev.keyword_clicks)} clicks`,ev.keyword_traffic&&`${n(ev.keyword_traffic)} traffic`,ev.keyword_volume&&`vol ${n(ev.keyword_volume)}`,ev.competitor_coverage&&`${n(ev.competitor_coverage)} competitor URLs`,ev.best_competitor_rank&&`best competitor #${esc(ev.best_competitor_rank)}`,ev.our_best_similarity!==undefined&&`selected page sim ${esc(ev.our_best_similarity)}`,ev.similarity_to_serp_topics!==undefined&&`SERP topic sim ${esc(ev.similarity_to_serp_topics)}`,profile.word_count&&`${n(profile.word_count)} words`,profile.avg_sentence_words&&`${esc(profile.avg_sentence_words)} avg sentence words`].filter(Boolean);const example=ev.example_url?`<div class="rationale">Example: ${urlLink(ev.example_url)}${ev.example_paragraph?`<br>${esc(ev.example_paragraph)}`:''}</div>`:'';const paragraph=ev.paragraph?`<div class="rationale">Paragraph: ${esc(ev.paragraph)}</div>`:'';return `<details class="evidence-details"><summary>Evidence and source snippets</summary><div class="diagnostic-body"><div class="mini">${esc(lines.join(' · ')||'No numeric evidence available.')}</div>${example}${paragraph}</div></details>`;}
function actionList(rows,limit=12){if(!rows||!rows.length)return '<div class="empty">No content action points were generated. The analyzed page already covers the main detected SERP topics closely enough for these thresholds.</div>';return `<div class="action-list">${rows.slice(0,limit).map((row,index)=>{const brief=row.content_brief||{};const terms=(row.suggested_terms||[]).map(term=>`<span class="chip">${esc(term)}</span>`).join('');const plan=brief.paragraph_plan||brief.paragraph_rules||[];const acceptance=row.acceptance_criteria||brief.acceptance_criteria||[];return `<div class="action-card"><strong>${n(row.global_order||index+1)}. ${esc(row.task_summary||row.action||row.type)} <span class="chip ${actionPriorityClass(row.priority)}">${esc(row.priority||'medium')}</span></strong><div class="mini">${esc(row.keyword||'')} · ${urlLink(row.target_url)}</div><div class="instruction">${esc(row.instruction||'')}</div>${row.placement?`<div class="brief-label">Placement</div><div class="mini">${esc(row.placement)}</div>`:''}${terms?`<div class="action-meta">${terms}</div>`:''}${plan.length?`<div class="brief-label">Paragraph plan</div>${listHtml(plan,'ol')}`:''}${acceptance.length?`<div class="brief-label">Acceptance criteria</div>${listHtml(acceptance)}`:''}${promptBox(row.ai_agent_prompt||brief.ai_agent_prompt)}${actionEvidenceHtml(row)}</div>`;}).join('')}</div>`;}
function pageBriefCard(brief,index){const actions=brief.next_actions||[];const keywords=(brief.primary_keywords||[]).map(k=>`<span class="chip">${esc(k)}</span>`).join('');return `<div class="brief-card"><h5>${index+1}. ${esc(brief.title||brief.target_url||'Page brief')}</h5><div class="mini">${urlLink(brief.target_url)}</div><div class="action-meta"><span class="chip missing">${n(brief.high_priority_actions||0)} high priority</span><span class="chip">${n(brief.total_actions||0)} tasks</span><span class="chip">score ${esc(brief.priority_score||0)}</span></div>${keywords?`<div class="action-meta">${keywords}</div>`:''}<div class="brief-label">Next tasks</div>${actions.length?`<ol>${actions.slice(0,5).map(a=>`<li><b>${esc(a.priority||'')}</b> ${esc(a.task_summary||a.type||'Task')} ${a.keyword?`<span class="mini">(${esc(a.keyword)})</span>`:''}</li>`).join('')}</ol>`:'<div class="mini">No page-level tasks generated.</div>'}${promptBox(brief.ai_agent_prompt)}</div>`;}
function contentBriefsSection(){const rows=data.content_briefs||[];if(!rows.length)return '';return collapsiblePanel('Page Content Briefs',`${sectionNote('The shortest path from analysis to work: each page brief groups priority tasks, target keywords, paragraph rules, and an AI-agent prompt.')}<div class="brief-grid">${rows.map(pageBriefCard).join('')}</div>`,{meta:`${n(rows.length)} brief${rows.length===1?'':'s'}`});}
function aiAgentStatusSection(){const agent=data.ai_agent||{};if(!agent.enabled)return '';const notes=(agent.notes||[]).concat(agent.errors||[]);const statusRows=[['Status',agent.status||''],['Provider',agent.provider||''],['Model',agent.model||''],['Detected language',agent.detected_language||''],['Language prompts',agent.language_prompts||0],['Keyword prompts',agent.keyword_prompts||0],['Keyword fallbacks',agent.keyword_fallbacks||0],['Editor briefs',agent.editor_briefs||0],['Cache hits',agent.cache_hits||0]];return collapsiblePanel('AI Agent Status',`${sectionNote('Harnext/OpenRouter agent layer for language detection, keyword inference, paragraph-level editor briefs, and final article drafts. Missing prerequisites are reported here; metrics are never fabricated.')}<table><tbody>${statusRows.map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}</tbody></table>${notes.length?`<div class="brief-label">Notes</div>${listHtml(notes)}`:''}`,{meta:agent.status||'',open:agent.status&&agent.status!=='ready'&&agent.status!=='disabled'});}
function aiEditorBriefSection(page){const brief=page.ai_editor_brief||{};if(!brief.status)return '';if(brief.status==='ok'){const meta=[brief.provider,brief.cache_status].filter(Boolean).join(' · ');return collapsiblePanel('AI Agent TODO',`${sectionNote('Generated markdown instructions for an AI coding/content agent. It should be treated as the implementation brief; deterministic task cards remain below as supporting evidence.')}${mdToHtml(brief.markdown||'')}`,{meta,open:false});}return collapsiblePanel('AI Agent TODO',`<div class="empty">${esc(brief.message||'AI editor brief was not generated.')}</div>`,{meta:brief.status||'not generated',open:false});}
function paragraphRulesSection(){const rules=(data.editorial_guidelines||{}).paragraph_rules||[];if(!rules.length)return '';return collapsiblePanel('AI Paragraph Rules',listHtml(rules),{meta:`${n(rules.length)} rule${rules.length===1?'':'s'}`});}
function aggregateActionsSection(){const rows=data.action_points||[];return collapsiblePanel('Content Action Plan For AI Agents',`${sectionNote('Prioritized instructions for editors or an AI agent. Each card includes what to change, where to place it, how paragraphs should be structured, and when the task is done.')} ${actionList(rows,18)}`,{meta:`${n(rows.length)} task${rows.length===1?'':'s'}`});}
function domainRatingValue(row){const raw=row?.domain_rating;if(raw===null||raw===undefined||raw==='')return null;const v=Number(raw);return Number.isFinite(v)&&v>=0?v:null;}
function domainRatingChip(row){const v=domainRatingValue(row);return v===null?'':`<span class="chip" title="Domain Rating by Ahrefs">DR ${v.toFixed(1)}</span>`;}
function domainRatingText(row){const v=domainRatingValue(row);return v===null?'':` · DR ${v.toFixed(1)} by Ahrefs`;}
function domainRatingAttribution(){const meta=data.domain_ratings||{};if(!meta.domains_enriched)return '';const license=meta.license||'http://ahrefs.com/legal/domain-rating-license';return `<div class="mini" style="margin-top:10px">Domain Rating by <a href="https://ahrefs.com/" target="_blank" rel="noopener noreferrer">Ahrefs</a>. <a href="${esc(license)}" target="_blank" rel="noopener noreferrer">License</a>. ${n(meta.domains_enriched)} domain${Number(meta.domains_enriched)===1?'':'s'} enriched.</div>`;}
function competitorList(rows){if(!rows||!rows.length)return '<div class="empty">No competitors fetched.</div>';return '<ol class="competitors">'+rows.map(c=>`<li>${urlLink(c.url,c.title||c.url)}<div class="mini">Rank ${esc(c.rank||'')} · Paragraphs ${esc(c.paragraph_count||0)}${domainRatingText(c)}${c.error?' · '+esc(c.error):''}</div></li>`).join('')+'</ol>';}
function reviewList(rows){if(!rows||!rows.length)return `<div class="empty">No high-distance ${esc(ownDomain)} paragraphs were flagged for this keyword. This means the analyzed paragraphs stayed close enough to either the target keyword vector or the SERP topic space.</div>`;return '<ul class="review">'+rows.map(p=>`<li><b>${esc(p.similarity_to_serp_topics)}</b> <span class="mini">similarity · ${esc(p.review_reason||'review candidate')}</span><br>${esc(p.paragraph)}</li>`).join('')+'</ul>';}
function intentWinnabilityPanel(a){const intent=a.intent||{};const win=a.winnability||{};if(!intent.serp_intent&&!win.band)return '';const evidence=(intent.evidence||[]).concat(win.evidence||[]).slice(0,7);const alt=a.alternative_keyword||{};const header=a.recommendation_header?`<div class="why-item">${esc(a.recommendation_header)}</div>`:'';return `<div class="panel" style="margin-top:14px"><h4>Intent & Winnability</h4><div class="panel-body">${header}<div class="chips"><span class="chip ${intent.match==='mismatch'?'missing':intent.match==='match'?'covered':'partial'}">Intent ${esc(intent.match||'unknown')}</span><span class="chip">SERP ${esc(intent.serp_intent||'unknown')}</span><span class="chip">Page ${esc(intent.page_intent||'unknown')}</span><span class="chip ${win.band==='unlikely'?'missing':win.band==='hard'?'partial':win.band==='winnable'?'covered':''}">Winnability ${esc(win.band||'unknown')}</span>${win.own_dr!==null&&win.own_dr!==undefined?`<span class="chip">Own DR ${esc(win.own_dr)}</span>`:''}${win.top10_dr_median!==null&&win.top10_dr_median!==undefined?`<span class="chip">Top-10 median DR ${esc(win.top10_dr_median)}</span>`:''}</div>${alt.keyword?`<div class="mini" style="margin-top:8px">Alternative keyword: <b>${esc(alt.keyword)}</b>${alt.band?` · ${esc(alt.band)}`:''}</div>`:''}${evidence.length?listHtml(evidence):''}</div></div>`;}
function sharedKeywordNames(a,b){const aKeywords=new Set((a.keywords||[]).map(k=>String(k.keyword||'')));return (b.keywords||[]).map(k=>String(k.keyword||'')).filter(k=>aKeywords.has(k));}
function rankForKeyword(row,keyword){const match=(row.keywords||[]).find(k=>String(k.keyword||'')===String(keyword||''));return Number(match?.rank||0);}
function serpTrafficScore(a,b,keywords){if(!keywords.length)return 0;const total=keywords.reduce((sum,keyword)=>{const ar=rankForKeyword(a,keyword),br=rankForKeyword(b,keyword);return sum+Math.max(0,11-ar)+Math.max(0,11-br);},0);return total/(keywords.length*20);}
function trafficColor(score){const t=Math.max(0,Math.min(1,Number(score)||0));const low=[88,123,181],mid=[151,121,77],high=[194,91,30];const mix=(a,b,x)=>Math.round(a+(b-a)*x);const left=t<.5?low:mid;const right=t<.5?mid:high;const x=t<.5?t*2:(t-.5)*2;return `rgb(${mix(left[0],right[0],x)} ${mix(left[1],right[1],x)} ${mix(left[2],right[2],x)})`;}
function serpUrlGraph(rows){if(!rows||rows.length<2)return '<div class="empty">Not enough URLs for a co-ranking graph.</div>';const nodes=rows.slice(0,18).map(row=>({...row,domain:row.domain||urlDomain(row.url||'')||row.url})).sort((a,b)=>String(a.domain).localeCompare(String(b.domain))||Number(b.top10_count||0)-Number(a.top10_count||0));const edges=[];for(let i=0;i<nodes.length;i++){for(let j=i+1;j<nodes.length;j++){const shared=sharedKeywordNames(nodes[i],nodes[j]);if(shared.length){const traffic=serpTrafficScore(nodes[i],nodes[j],shared);edges.push({a:i,b:j,weight:shared.length,traffic,keywords:shared});}}}if(!edges.length)return '<div class="empty">No URLs share selected keywords.</div>';const w=920,h=560,cx=w/2,cy=h/2,leafR=205,domainR=118,bundleR=34;const maxCount=Math.max(1,...nodes.map(row=>Number(row.top10_count||0)));const domains=[...new Set(nodes.map(n=>n.domain))];const domainAngles=new Map(domains.map((domain,i)=>[domain,-Math.PI/2+(i/domains.length)*Math.PI*2]));const groups=new Map;nodes.forEach(node=>{const group=groups.get(node.domain)||[];group.push(node);groups.set(node.domain,group);});const positions=nodes.map((node,index)=>{const group=groups.get(node.domain)||[node];const groupIndex=group.indexOf(node);const base=domainAngles.get(node.domain)||0;const spread=Math.min(0.46,0.09*Math.max(group.length-1,0));const angle=base+(group.length<=1?0:-spread/2+(groupIndex/(group.length-1))*spread);return{index,angle,x:cx+Math.cos(angle)*leafR,y:cy+Math.sin(angle)*leafR,dx:cx+Math.cos(base)*domainR,dy:cy+Math.sin(base)*domainR,bx:cx+Math.cos(base)*bundleR,by:cy+Math.sin(base)*bundleR};});const edgeSvg=edges.map(edge=>{const a=positions[edge.a],b=positions[edge.b];const d=`M${a.x.toFixed(1)},${a.y.toFixed(1)} C${a.dx.toFixed(1)},${a.dy.toFixed(1)} ${a.bx.toFixed(1)},${a.by.toFixed(1)} ${cx},${cy} S${b.dx.toFixed(1)},${b.dy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;const title=`${nodes[edge.a].url} ↔ ${nodes[edge.b].url}: ${edge.weight} shared keyword${edge.weight===1?'':'s'} (${edge.keywords.join(', ')}); traffic proxy ${(edge.traffic*100).toFixed(0)}% from SERP positions`;return `<path class="graph-edge" d="${d}" stroke="${trafficColor(edge.traffic)}" stroke-width="${(0.9+edge.weight*1.5).toFixed(1)}"><title>${esc(title)}</title></path>`;}).join('');const domainMarks=domains.map(domain=>{const angle=domainAngles.get(domain)||0;const x=cx+Math.cos(angle)*domainR,y=cy+Math.sin(angle)*domainR;return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${domainColor(domain)}" opacity=".76"><title>${esc(domain)}</title></circle>`;}).join('');const nodeSvg=nodes.map((node,i)=>{const p=positions[i];const count=Number(node.top10_count||0);const size=7+Math.sqrt(count/maxCount)*10;const title=`${node.url}\\n${count} top-10 appearance${count===1?'':'s'}\\nBest rank #${node.best_rank||''}\\nKeywords: ${(node.keywords||[]).map(k=>`${k.keyword} #${k.rank}`).join(', ')}`;const labelX=p.x+(p.x>=cx?size+5:-size-5);const anchor=p.x>=cx?'start':'end';return `<a href="${esc(node.url||'#')}" target="_blank" rel="noopener noreferrer"><circle class="graph-node" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${size.toFixed(1)}" fill="${domainColor(node.domain)}"><title>${esc(title)}</title></circle></a><text class="graph-label" x="${labelX.toFixed(1)}" y="${(p.y-2).toFixed(1)}" text-anchor="${anchor}">${esc(node.domain).slice(0,26)}</text><text class="graph-meta" x="${labelX.toFixed(1)}" y="${(p.y+11).toFixed(1)}" text-anchor="${anchor}">${n(count)} top-10 · best #${esc(node.best_rank||'')}${domainRatingText(node)}</text>`;}).join('');const defs='<defs><linearGradient id="trafficGradient" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="'+trafficColor(0.08)+'"></stop><stop offset="50%" stop-color="'+trafficColor(0.5)+'"></stop><stop offset="100%" stop-color="'+trafficColor(1)+'"></stop></linearGradient></defs>';const legend=`<g class="traffic-legend" transform="translate(26 ${h-34})"><text x="0" y="-8">Connection color: SERP-position traffic proxy</text><rect x="0" y="0" width="150" height="8" rx="4" fill="url(#trafficGradient)"></rect><text x="0" y="23">lower</text><text x="150" y="23" text-anchor="end">higher</text></g>`;return `<svg class="serp-url-graph" viewBox="0 0 ${w} ${h}" role="img" aria-label="Hierarchical edge bundling chart for URL co-ranking. Nodes are ranking URLs grouped by domain. Curved connections join URLs that rank for the same selected keywords; thicker lines mean more shared keywords and warmer colors mean stronger SERP-position traffic proxy."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${defs}${edgeSvg}${domainMarks}${nodeSvg}${legend}</svg><div class="mini">Bundled co-ranking graph: nodes are URLs grouped by domain; curved connections mean URLs rank for the same selected keyword; line width shows shared keyword count; warmer color means stronger SERP-position traffic proxy.</div>`;}
function serpRankingChart(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 URL chart available.</div>';const maxCount=Math.max(1,...rows.map(row=>Number(row.top10_count||0)));return `<div class="serp-ranking-chart" aria-label="Top-10 URL relevance chart">${rows.slice(0,12).map(row=>{const count=Number(row.top10_count||0);const width=Math.max(6,Math.round((count/maxCount)*100));const title=`${row.url} · ${count} top-10 appearances · best #${row.best_rank||''} · avg #${row.average_rank||''}${domainRatingText(row)}`;return `<div class="serp-ranking-chart-row"><div class="serp-ranking-chart-label" title="${esc(row.url||'')}">${urlLink(row.url,row.domain||row.url)}</div><div class="serp-ranking-chart-track" title="${esc(title)}"><span class="serp-ranking-chart-bar" style="width:${width}%"></span></div><div class="serp-ranking-chart-meta">${n(count)} top-10 · best #${esc(row.best_rank||'')}${domainRatingText(row)}</div></div>`;}).join('')}</div>`;}
function serpRankingList(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 SERP URLs available.</div>';return `<div class="serp-ranking-list">${rows.map((row,index)=>{const keywords=(row.keywords||[]).map(k=>`<span class="serp-ranking-chip"><strong>#${esc(k.rank)}</strong> ${esc(k.keyword)}</span>`).join('');return `<div class="serp-ranking-row"><div><div class="serp-ranking-url">${index+1}. ${urlLink(row.url)}</div><div class="serp-ranking-domain">${esc(row.domain||'')}${row.is_selected_domain?' · selected domain':''}</div></div><div class="serp-ranking-stats"><span class="chip covered">${n(row.top10_count)} top-10</span><span class="chip">Best #${esc(row.best_rank||'')}</span><span class="chip">Avg #${esc(row.average_rank||'')}</span></div><div class="serp-ranking-keywords">${keywords}</div></div>`;}).join('')}</div>`;}
function hasDemandMetrics(row){return Number(row?.impressions||0)>0||Number(row?.clicks||0)>0||Number(row?.traffic||0)>0||Number(row?.volume||0)>0;}
function keywordMetrics(k){const parts=[`#${esc(k.rank)}`];if(hasDemandMetrics(k)){if(Number(k.impressions||0)>0)parts.push(`${n(k.impressions)} impr`);if(Number(k.clicks||0)>0)parts.push(`${n(k.clicks)} clicks`);if(Number(k.traffic||0)>0)parts.push(`${n(k.traffic)} traffic`);if(Number(k.volume||0)>0)parts.push(`vol ${n(k.volume)}`);}else{parts.push('no demand metrics');}if(Number(k.source_position||0))parts.push(`source pos ${Number(k.source_position).toFixed(1)}`);if(k.source)parts.push(esc(k.source));return parts.join(' · ');}
function graphKeywordRows(keywords){return (keywords||[]).map(k=>`<div class="metric-line"><b>${esc(k.keyword)}</b> · ${keywordMetrics(k)}</div>`).join('');}
function demandMetricLine(row){if(hasDemandMetrics(row)){const parts=[];if(Number(row.impressions||0)>0)parts.push(`Impressions: ${n(row.impressions)}`);if(Number(row.clicks||0)>0)parts.push(`Clicks: ${n(row.clicks)}`);if(Number(row.traffic||0)>0)parts.push(`Traffic: ${n(row.traffic)}`);if(Number(row.volume||0)>0)parts.push(`Volume: ${n(row.volume)}`);return `<div class="metric-line">${parts.join(' · ')}</div>`;}return '<div class="metric-line">Demand metrics unavailable: this run used manual keywords and SERP suggestions without GSC/Ahrefs click, impression, traffic, or volume data.</div>';}
function aggregateDemandRows(rows){return (rows||[]).reduce((out,row)=>({impressions:Number(out.impressions||0)+Number(row.impressions||0),clicks:Number(out.clicks||0)+Number(row.clicks||0),traffic:Number(out.traffic||0)+Number(row.traffic||0),volume:Number(out.volume||0)+Number(row.volume||0)}),{});}
function graphTooltipHtml(detail){if(detail.kind==='connection'){return `<h5>Shared keyword connection</h5><div>${urlLink(detail.url_a,detail.domain_a)} ↔ ${urlLink(detail.url_b,detail.domain_b)}</div><div class="metric-line">SERP-position proxy: ${Math.round(Number(detail.traffic_proxy||0)*100)}% · Shared keywords: ${n(detail.shared_count||0)}</div>${demandMetricLine(aggregateDemandRows(detail.keywords))}${graphKeywordRows(detail.keywords)}`;}return `<h5>${esc(detail.domain||'URL')}</h5><div>${urlLink(detail.url)}</div><div class="metric-line">Top-10 keywords: ${n(detail.top10_count||0)} · Best SERP position: #${esc(detail.best_rank||'')} · Avg SERP position: #${esc(detail.average_rank||'')}${domainRatingText(detail)}</div>${demandMetricLine(detail)}${graphKeywordRows(detail.keywords)}`;}
function serpUrlGraph(rows){if(!rows||rows.length<2)return '<div class="empty">Not enough URLs for a co-ranking graph.</div>';const nodes=rows.slice(0,18).map(row=>({...row,domain:row.domain||urlDomain(row.url||'')||row.url})).sort((a,b)=>String(a.domain).localeCompare(String(b.domain))||Number(b.top10_count||0)-Number(a.top10_count||0));const edges=[];for(let i=0;i<nodes.length;i++){for(let j=i+1;j<nodes.length;j++){const shared=sharedKeywordNames(nodes[i],nodes[j]);if(shared.length){const traffic=serpTrafficScore(nodes[i],nodes[j],shared);const keywords=shared.map(keyword=>{const a=(nodes[i].keywords||[]).find(k=>String(k.keyword||'')===keyword)||{};const b=(nodes[j].keywords||[]).find(k=>String(k.keyword||'')===keyword)||{};return{keyword,rank:`${a.rank||''} / ${b.rank||''}`,impressions:Math.max(Number(a.impressions||0),Number(b.impressions||0)),clicks:Math.max(Number(a.clicks||0),Number(b.clicks||0)),traffic:Math.max(Number(a.traffic||0),Number(b.traffic||0)),source_position:Number(a.source_position||b.source_position||0),volume:Math.max(Number(a.volume||0),Number(b.volume||0))};});edges.push({a:i,b:j,weight:shared.length,traffic,keywords});}}}if(!edges.length)return '<div class="empty">No URLs share selected keywords.</div>';const w=920,h=560,cx=w/2,cy=h/2,leafR=205,domainR=118,bundleR=34;const maxCount=Math.max(1,...nodes.map(row=>Number(row.top10_count||0)));const domains=[...new Set(nodes.map(n=>n.domain))];const domainAngles=new Map(domains.map((domain,i)=>[domain,-Math.PI/2+(i/domains.length)*Math.PI*2]));const groups=new Map;nodes.forEach(node=>{const group=groups.get(node.domain)||[];group.push(node);groups.set(node.domain,group);});const positions=nodes.map((node,index)=>{const group=groups.get(node.domain)||[node];const groupIndex=group.indexOf(node);const base=domainAngles.get(node.domain)||0;const spread=Math.min(0.46,0.09*Math.max(group.length-1,0));const angle=base+(group.length<=1?0:-spread/2+(groupIndex/(group.length-1))*spread);return{index,angle,x:cx+Math.cos(angle)*leafR,y:cy+Math.sin(angle)*leafR,dx:cx+Math.cos(base)*domainR,dy:cy+Math.sin(base)*domainR,bx:cx+Math.cos(base)*bundleR,by:cy+Math.sin(base)*bundleR};});const edgeSvg=edges.map(edge=>{const a=positions[edge.a],b=positions[edge.b];const d=`M${a.x.toFixed(1)},${a.y.toFixed(1)} C${a.dx.toFixed(1)},${a.dy.toFixed(1)} ${a.bx.toFixed(1)},${a.by.toFixed(1)} ${cx},${cy} S${b.dx.toFixed(1)},${b.dy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;const detail={kind:'connection',url_a:nodes[edge.a].url,url_b:nodes[edge.b].url,domain_a:nodes[edge.a].domain,domain_b:nodes[edge.b].domain,shared_count:edge.weight,traffic_proxy:edge.traffic,keywords:edge.keywords};return `<path class="graph-edge" tabindex="0" data-graph-edge data-nodes=",${edge.a},${edge.b}," data-graph-detail="${esc(JSON.stringify(detail))}" d="${d}" stroke="${trafficColor(edge.traffic)}" stroke-width="${(0.9+edge.weight*1.5).toFixed(1)}"></path>`;}).join('');const domainMarks=domains.map(domain=>{const angle=domainAngles.get(domain)||0;const x=cx+Math.cos(angle)*domainR,y=cy+Math.sin(angle)*domainR;return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${domainColor(domain)}" opacity=".76"><title>${esc(domain)}</title></circle>`;}).join('');const nodeSvg=nodes.map((node,i)=>{const p=positions[i];const count=Number(node.top10_count||0);const size=7+Math.sqrt(count/maxCount)*10;const labelX=p.x+(p.x>=cx?size+5:-size-5);const anchor=p.x>=cx?'start':'end';const detail={kind:'url',url:node.url,domain:node.domain,top10_count:node.top10_count,best_rank:node.best_rank,average_rank:node.average_rank,domain_rating:node.domain_rating,impressions:node.impressions,clicks:node.clicks,traffic:node.traffic,keywords:node.keywords||[]};return `<a href="${esc(node.url||'#')}" target="_blank" rel="noopener noreferrer"><circle class="graph-node" tabindex="0" data-graph-node data-node-index="${i}" data-graph-detail="${esc(JSON.stringify(detail))}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${size.toFixed(1)}" fill="${domainColor(node.domain)}"></circle></a><text class="graph-label" x="${labelX.toFixed(1)}" y="${(p.y-2).toFixed(1)}" text-anchor="${anchor}">${esc(node.domain).slice(0,26)}</text><text class="graph-meta" x="${labelX.toFixed(1)}" y="${(p.y+11).toFixed(1)}" text-anchor="${anchor}">${n(count)} top-10 · best #${esc(node.best_rank||'')}</text>`;}).join('');const defs='<defs><linearGradient id="trafficGradient" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="'+trafficColor(0.08)+'"></stop><stop offset="50%" stop-color="'+trafficColor(0.5)+'"></stop><stop offset="100%" stop-color="'+trafficColor(1)+'"></stop></linearGradient></defs>';const legend=`<g class="traffic-legend" transform="translate(26 ${h-34})"><text x="0" y="-8">Connection color: SERP-position traffic proxy</text><rect x="0" y="0" width="150" height="8" rx="4" fill="url(#trafficGradient)"></rect><text x="0" y="23">lower</text><text x="150" y="23" text-anchor="end">higher</text></g>`;return `<div class="serp-url-graph-wrap"><svg class="serp-url-graph" viewBox="0 0 ${w} ${h}" role="img" aria-label="Hierarchical edge bundling chart for URL co-ranking. Nodes are ranking URLs grouped by domain. Curved connections join URLs that rank for the same selected keywords; thicker lines mean more shared keywords and warmer colors mean stronger SERP-position traffic proxy."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${defs}${edgeSvg}${domainMarks}${nodeSvg}${legend}</svg><div class="graph-tooltip" role="tooltip"></div></div><div class="mini">Bundled co-ranking graph: nodes are URLs grouped by domain; curved connections mean URLs rank for the same selected keyword; line width shows shared keyword count; warmer color means stronger SERP-position traffic proxy.</div>`;}
function urlKeywordTable(keywords){if(!keywords||!keywords.length)return '<div class="empty">No keyword rows for this URL.</div>';return `<table class="url-keyword-table"><thead><tr><th class="metric-number">Rank</th><th>Keyword</th><th class="metric-number">Impr</th><th class="metric-number">Clicks</th><th class="metric-number">Traffic</th><th class="metric-number">Volume</th><th class="metric-number">Source pos</th><th>Source</th></tr></thead><tbody>${keywords.map(k=>`<tr><td class="metric-number">#${esc(k.rank||'')}</td><td><strong>${esc(k.keyword||'')}</strong></td><td class="metric-number">${metricCell(k.impressions)}</td><td class="metric-number">${metricCell(k.clicks)}</td><td class="metric-number">${metricCell(k.traffic,2)}</td><td class="metric-number">${metricCell(k.volume)}</td><td class="metric-number">${metricCell(k.source_position,1)}</td><td>${esc(k.source||'')}</td></tr>`).join('')}</tbody></table>`;}
function serpRankingList(rows){if(!rows||!rows.length)return '<div class="empty">No top-10 SERP URLs available.</div>';return `<div class="serp-ranking-list">${rows.map((row,index)=>{const demandChips=hasDemandMetrics(row)?`${Number(row.impressions||0)>0?`<span class="chip">${n(row.impressions)} impr</span>`:''}${Number(row.clicks||0)>0?`<span class="chip">${n(row.clicks)} clicks</span>`:''}${Number(row.traffic||0)>0?`<span class="chip">${n(row.traffic)} traffic</span>`:''}${Number(row.volume||0)>0?`<span class="chip">vol ${n(row.volume)}</span>`:''}`:'<span class="chip">Demand metrics unavailable</span>';return `<div class="serp-ranking-row"><div><div class="serp-ranking-url">${index+1}. ${urlLink(row.url)}</div><div class="serp-ranking-domain">${esc(row.domain||'')}${row.is_selected_domain?' · selected domain':''}</div></div><div class="serp-ranking-stats"><span class="chip covered">${n(row.top10_count)} top-10</span><span class="chip">Best #${esc(row.best_rank||'')}</span><span class="chip">Avg #${esc(row.average_rank||'')}</span>${domainRatingChip(row)}${demandChips}</div>${urlKeywordTable(row.keywords||[])}</div>`;}).join('')}</div>`;}
function urlDemandRows(rows){return (rows||[]).map(row=>({...row,demand_score:Number(row.impressions||0)+Number(row.clicks||0)*50+Number(row.traffic||0)*20+Number(row.volume||0)})).sort((a,b)=>Number(b.demand_score||0)-Number(a.demand_score||0)||Number(b.top10_count||0)-Number(a.top10_count||0)||Number(a.best_rank||999)-Number(b.best_rank||999));}
function urlDemandMetricsSection(rows){const demandRows=urlDemandRows(rows);if(!demandRows.length)return '';const hasAny=demandRows.some(hasDemandMetrics);const maxDemand=Math.max(1,...demandRows.map(row=>Number(row.demand_score||0)));return `<h5 style="margin:14px 0 6px">URL Demand Metrics</h5><div class="url-demand-table-wrap"><table class="url-demand-table"><thead><tr><th>URL</th><th class="metric-number">Top-10</th><th class="metric-number">Best</th><th class="metric-number">Avg</th><th class="metric-number">DR</th><th class="metric-number">Impr</th><th class="metric-number">Clicks</th><th class="metric-number">Traffic</th><th class="metric-number">Volume</th><th>Demand weight</th></tr></thead><tbody>${demandRows.slice(0,18).map(row=>{const width=Math.max(4,Math.round(Number(row.demand_score||0)/maxDemand*100));return `<tr><td>${urlLink(row.url,row.domain||row.url)}${row.is_selected_domain?' <span class="chip covered">selected domain</span>':''}</td><td class="metric-number">${metricCell(row.top10_count)}</td><td class="metric-number">${row.best_rank?`#${esc(row.best_rank)}`:''}</td><td class="metric-number">${row.average_rank?`#${esc(row.average_rank)}`:''}</td><td class="metric-number">${domainRatingValue(row)===null?'':domainRatingValue(row).toFixed(1)}</td><td class="metric-number">${metricCell(row.impressions)}</td><td class="metric-number">${metricCell(row.clicks)}</td><td class="metric-number">${metricCell(row.traffic,2)}</td><td class="metric-number">${metricCell(row.volume)}</td><td class="url-demand-bar" title="${esc(row.demand_score||0)} demand proxy"><span style="width:${hasAny?width:4}%"></span></td></tr>`;}).join('')}</tbody></table></div>${hasAny?'':'<div class="section-note" style="margin-top:10px">Demand metrics unavailable for these URLs because the selected keywords did not have matched GSC/Ahrefs/DataForSEO click, impression, traffic, or volume data in this run.</div>'}`;}
function serpRankSort(a,b){return (Number(a.rank||9999)-Number(b.rank||9999))||(a.source==='ours'?1:0)-(b.source==='ours'?1:0)||String(a.domain||a.url||'').localeCompare(String(b.domain||b.url||''));}
function keywordParagraphRidgeline(ridges){const keywords=ridges?.keywords||[];const rows=(ridges?.rows||[]).slice().sort(serpRankSort).slice(0,16);if(!keywords.length||!rows.length)return '<div class="empty">No keyword-to-paragraph relationship data available.</div>';const w=960,rowH=58,padL=190,padR=26,padT=72,padB=34,h=padT+padB+rows.length*rowH;const x=i=>padL+(keywords.length<=1?0.5:i/(keywords.length-1))*(w-padL-padR);const cellFor=(row,i)=>(row.cells||[]).find(cell=>Number(cell.keyword_order)===i)||{};const axis=keywords.map((kw,i)=>{const xx=x(i);const metric=[Number(kw.impressions||0)>0&&`${n(kw.impressions)} impr`,Number(kw.volume||0)>0&&`vol ${n(kw.volume)}`].filter(Boolean).join(' · ');return `<line class="keyword-ridge-baseline" x1="${xx.toFixed(1)}" x2="${xx.toFixed(1)}" y1="${padT-8}" y2="${h-padB}"></line><text class="keyword-ridge-axis" x="${xx.toFixed(1)}" y="24" text-anchor="middle">${esc(kw.keyword||'').slice(0,20)}</text>${metric?`<text class="keyword-ridge-axis" x="${xx.toFixed(1)}" y="39" text-anchor="middle">${esc(metric)}</text>`:''}`;}).join('');const lanes=rows.map((row,rowIndex)=>{const base=padT+rowIndex*rowH+rowH*0.72;const amp=rowH*0.58;const label=row.source==='ours'?ownDomain:(row.domain||urlDomain(row.url||'')||row.url);const meta=row.source==='ours'?(row.rank?`SERP #${row.rank} · target`:'target · no top-10 rank'):`SERP #${row.rank||'?'} · ${n(row.paragraph_count||0)} paragraphs`;const top=keywords.map((kw,i)=>{const cell=cellFor(row,i);const score=Math.max(0,Math.min(1,Number(cell.top3_similarity||cell.max_similarity||0)));return{x:x(i),y:base-score*amp,score,cell,kw};});const area=top.length===1?`M${(top[0].x-16).toFixed(1)},${base.toFixed(1)} L${top[0].x.toFixed(1)},${top[0].y.toFixed(1)} L${(top[0].x+16).toFixed(1)},${base.toFixed(1)} Z`:`M${top[0].x.toFixed(1)},${base.toFixed(1)} ${top.map(p=>`L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} L${top[top.length-1].x.toFixed(1)},${base.toFixed(1)} Z`;const dots=top.map(point=>{const c=point.cell||{};const title=[label,point.kw.keyword,`top paragraphs similarity ${point.score.toFixed(2)}`,Number(c.strong_paragraphs||0)&&`${n(c.strong_paragraphs)} strong paragraph(s)`,c.best_paragraph].filter(Boolean).join('\\n');return `<circle class="keyword-ridge-point" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${Math.max(3.5,Math.min(8,3.5+point.score*5))}" fill="${point.score>=0.72?'#176a35':point.score>=0.58?'#b65f00':'#68788b'}"><title>${esc(title)}</title></circle>`;}).join('');return `<text class="keyword-ridge-label" x="12" y="${(base-14).toFixed(1)}">${esc(label).slice(0,28)}</text><text class="keyword-ridge-meta" x="12" y="${base.toFixed(1)}">${esc(meta)}</text><line class="keyword-ridge-baseline" x1="${padL}" x2="${w-padR}" y1="${base.toFixed(1)}" y2="${base.toFixed(1)}"></line><path class="keyword-ridge-area ${row.source==='ours'?'ours':''}" d="${area}"><title>${esc(label)} keyword-to-paragraph relation ridge</title></path>${dots}`;}).join('');return `<div class="keyword-ridge-wrap"><svg class="keyword-ridge-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Ridgeline-style keyword-to-paragraph relation chart. URLs are on the Y axis ordered by SERP rank; keywords are on the X axis; ridge height shows how strongly paragraphs on each URL match each keyword."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${axis}${lanes}</svg></div><div class="mini">Rows are URLs ordered by SERP top-10 position. Ridge height and dot size show how strongly that URL's paragraphs match each selected keyword; hover points for the best matching paragraph.</div>`;}
function keywordParagraphRidgelineSection(points){const ridges=data.overview_scatter?.keyword_url_ridges||{};return `<div class="panel" style="margin-top:14px"><h4>Keyword-to-Paragraph Coverage by URL</h4><div class="panel-body">${sectionNote('Ridgeline-style view of selected keyword intent against paragraphs on each URL. Use it to see which ranking pages have paragraph-level support for each keyword and where the target page is thin or off-order.')} ${keywordParagraphRidgeline(ridges)}</div></div>`;}
function pathClusterColor(id){return domainPalette[Math.abs(Number(id||0))%domainPalette.length];}
function contentPathTooltip(item,cluster){const parts=[`${item.source==='ours'?ownDomain:item.domain||'competitor'} · ${item.entity_type||'paragraph'} ${Number(item.order_index||0)+1}`,cluster?.label&&`Cluster: ${cluster.label}`,item.rank&&`SERP #${item.rank}`,item.keyword_similarity!==undefined&&`Keyword sim: ${item.keyword_similarity}`,item.text];return parts.filter(Boolean).join('\\n');}
function orderedPathPages(path){return (path.pages||[]).slice().sort((a,b)=>(Number(a.rank||9999)-Number(b.rank||9999))||(a.source==='ours'?1:0)-(b.source==='ours'?1:0)||String(a.domain||a.url||'').localeCompare(String(b.domain||b.url||'')));}
function parallelClusterRows(path,pages){const items=path.items||[];const pageIndex=new Map(pages.map((p,i)=>[p.id,i]));const clusterMeta=new Map((path.clusters||[]).map(c=>[String(c.cluster),c]));const groups=new Map();for(const item of items){const cid=String(item.cluster??0);const group=groups.get(cid)||{cluster:cid,positions:new Map(),items:[],sources:new Set(),competitorPages:new Set(),bestRank:999};const pageId=item.page_id;if(!pageIndex.has(pageId))continue;const list=group.positions.get(pageId)||[];list.push(Number(item.order_position||0));group.positions.set(pageId,list);group.items.push(item);group.sources.add(item.source||'');if(item.source==='competitor')group.competitorPages.add(pageId);const rank=Number(item.rank||0);if(rank)group.bestRank=Math.min(group.bestRank,rank);groups.set(cid,group);}return [...groups.values()].map(group=>{const meta=clusterMeta.get(group.cluster)||{};const points=[...group.positions.entries()].map(([pageId,values])=>({pageId,pageIndex:pageIndex.get(pageId),order:_meanJs(values),count:values.length})).sort((a,b)=>a.pageIndex-b.pageIndex);const sample=(group.items[0]||{}).text||meta.sample_text||'';return{...group,points,label:meta.label||`Cluster ${group.cluster}`,sample,bestRank:group.bestRank===999?(meta.best_competitor_rank||''):group.bestRank,missingOurs:!group.sources.has('ours')&&group.competitorPages.size>0};}).filter(group=>group.points.length>=2).sort((a,b)=>b.points.length-a.points.length||Number(a.bestRank||999)-Number(b.bestRank||999));}
function _meanJs(values){return values.reduce((sum,value)=>sum+Number(value||0),0)/Math.max(values.length,1);}
function contentPathParallelCoordinates(path){
  const pages=orderedPathPages(path);
  const groups=parallelClusterRows(path,pages).slice(0,18);
  if(!pages.length||!groups.length)return '<div class="empty">No shared content-order clusters available for a parallel-coordinates view.</div>';
  const w=960,h=420,padL=42,padR=34,padT=64,padB=52;
  const xFor=new Map(pages.map((p,i)=>[p.id,padL+(pages.length===1?0.5:i/(pages.length-1))*(w-padL-padR)]));
  const y=pos=>padT+Math.max(0,Math.min(1,Number(pos||0)))*(h-padT-padB);
  const axes=pages.map(p=>{
    const x=xFor.get(p.id)||0;
    const label=p.source==='ours'?ownDomain:(p.domain||p.url||'competitor');
    const meta=p.source==='ours'?(p.rank?`SERP #${p.rank} · target`:'target · no top-10 rank'):`SERP #${p.rank||'?'} · ${n(p.item_count||0)} blocks`;
    return `<line class="parallel-axis" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${padT}" y2="${h-padB}"></line><text class="parallel-axis-label" x="${x.toFixed(1)}" y="24" text-anchor="middle">${esc(label).slice(0,18)}</text><text class="parallel-axis-meta" x="${x.toFixed(1)}" y="39" text-anchor="middle">${esc(meta)}</text>`;
  }).join('');
  const lines=groups.map(group=>{
    const color=pathClusterColor(group.cluster);
    const clusterId=String(group.cluster);
    const d=group.points.map((point,i)=>`${i?'L':'M'}${(xFor.get(point.pageId)||0).toFixed(1)},${y(point.order).toFixed(1)}`).join(' ');
    const title=[group.label,group.bestRank&&`best SERP #${group.bestRank}`,group.missingOurs&&`missing on ${ownDomain}`,group.sample].filter(Boolean).join('\\n');
    const markers=group.points.map(point=>`<circle class="parallel-topic-marker" data-path-cluster="${esc(clusterId)}" tabindex="0" cx="${(xFor.get(point.pageId)||0).toFixed(1)}" cy="${y(point.order).toFixed(1)}" r="${Math.min(7,3.6+point.count)}" fill="${color}" aria-label="${esc(title)}"><title>${esc(title)}</title></circle>`).join('');
    return `<path class="parallel-topic-line ${group.missingOurs?'missing-ours':''}" data-path-cluster="${esc(clusterId)}" tabindex="0" d="${d}" stroke="${color}" aria-label="${esc(title)}"><title>${esc(title)}</title></path>${markers}`;
  }).join('');
  const legend=groups.slice(0,12).map(group=>`<span class="path-cluster-chip" title="${esc(group.sample||'')}"><i style="background:${pathClusterColor(group.cluster)}"></i>${esc(group.label)}</span>`).join('');
  return `<div class="content-path-wrap"><svg class="content-path-chart content-path-parallel" viewBox="0 0 ${w} ${h}" role="img" aria-label="Parallel coordinates chart. URL axes are ordered by SERP top-10 rank, and each colored line connects the same semantic topic cluster at its average paragraph order position on every URL where it appears. Hover, focus, or click a line or node to highlight that topic path."><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>${axes}<text class="parallel-axis-meta" x="8" y="${padT+4}">start</text><text class="parallel-axis-meta" x="8" y="${h-padB}" dominant-baseline="middle">end</text>${lines}</svg></div><div class="path-clusters">${legend}</div><div class="mini">Parallel coordinates: URL axes are ordered by SERP top-10 position; hover, focus, or click a topic line or node to highlight that path and fade unrelated paths. Dashed lines are clusters missing on ${esc(ownDomain)}.</div>`;
}
function contentPathSvg(path){return contentPathParallelCoordinates(path);}
function contentPathFindings(path){const findings=[];for(const row of path.missing_clusters||[])findings.push(`<div class="path-finding"><b>Missing cluster:</b> ${esc(row.label||'Cluster')} appears in ${n(row.competitor_pages||0)} ranking page(s)${row.best_competitor_rank?` as high as SERP #${esc(row.best_competitor_rank)}`:''}, but the target page has no matching path point.</div>`);for(const row of path.deviations||[])findings.push(`<div class="path-finding"><b>Order mismatch:</b> ${esc(row.label||'Cluster')} appears ${esc(row.direction||'later')} on the target page than the competitor median path (${Math.round(Number(row.competitor_mean_order||0)*100)}% vs ${Math.round(Number(row.ours_mean_order||0)*100)}% through the page).</div>`);if(!findings.length)return '<div class="empty">No major missing or reordered semantic clusters were detected for this keyword.</div>';return `<div class="path-findings">${findings.slice(0,8).join('')}</div>`;}
function contentPathUnmatchedClusters(path){const rows=path.unmatched_clusters_by_url||[];if(!rows.length)return '<div class="empty">No URL-only content-order clusters were detected.</div>';return `<div class="path-unmatched-grid">${rows.map(page=>{const label=page.source==='ours'?ownDomain:(page.domain||page.url||'URL');const meta=page.source==='ours'?'target page':`SERP #${page.rank||'?'} competitor`;const clusters=(page.clusters||[]).slice(0,8).map(cluster=>`<div class="path-unmatched-item" title="${esc(cluster.sample_text||'')}"><b>${esc(cluster.label||`Cluster ${cluster.cluster}`)}</b><span class="mini">${cluster.mean_order!==null&&cluster.mean_order!==undefined?`${Math.round(Number(cluster.mean_order||0)*100)}% through page`:''}${cluster.count?` · ${n(cluster.count)} block${Number(cluster.count)===1?'':'s'}`:''}</span><div class="mini">${esc(cluster.sample_text||'').slice(0,160)}</div></div>`).join('');return `<div class="path-unmatched-card"><strong>${esc(label)}</strong><div class="mini">${esc(meta)} · unique clusters not matched with other URLs</div><div class="path-unmatched-list">${clusters}</div></div>`;}).join('')}</div>`;}
function contentOrderPathSection(a){const path=a.content_order_path||{};const s=path.summary||{};return `${sectionNote(`Parallel coordinates compare the order of similar topic clusters across SERP-ranked URLs. Axes are ordered by top-10 SERP position; order score ${Number(s.order_score??0).toFixed(2)}; ${n(s.missing_cluster_count||0)} missing clusters; ${n(s.shared_cluster_deviations||0)} reordered shared clusters; ${n(s.unmatched_cluster_count||0)} URL-only clusters.`)}${contentPathSvg(path)}${contentPathUnmatchedClusters(path)}${contentPathFindings(path)}`;}
function contentComparisonRows(a){const comp=a.content_comparison||{};const ours=comp.ours||null;const competitors=comp.competitors||[];const rows=[ours,...competitors].filter(Boolean).slice(0,7);if(!rows.length)return '<div class="empty">No content comparison data available.</div>';const totalTopics=Number(comp.summary?.total_topics||0);const maxParagraphs=Math.max(1,...rows.map(row=>Number(row.paragraph_count||0)));const maxHeadings=Math.max(1,...rows.map(row=>Number(row.heading_count||0)));const maxTopics=Math.max(1,...rows.map(row=>Number(row.topic_count||0)));return `<div class="comparison-chart" aria-label="SERP rank vs content coverage">${rows.map(row=>{const isOurs=row.source==='ours';const domain=row.domain||urlDomain(row.url||'')||row.label||'';const rank=isOurs?'Target page':`SERP #${row.rank||''}`;const paragraphWidth=Math.max(4,Math.round(Number(row.paragraph_count||0)/maxParagraphs*100));const headingWidth=Math.max(4,Math.round(Number(row.heading_count||0)/maxHeadings*100));const topicWidth=Math.max(4,Math.round(Number(row.topic_count||0)/maxTopics*100));return `<div class="comparison-row ${isOurs?'is-ours':''}"><div class="comparison-label" title="${esc(row.url||'')}">${esc(domain||ownDomain)}<span>${esc(rank)}</span></div><div class="comparison-stack"><div class="comparison-track" title="Topic coverage"><span class="comparison-bar topics" style="width:${topicWidth}%"></span></div><div class="comparison-track" title="Extracted paragraphs"><span class="comparison-bar paragraphs" style="width:${paragraphWidth}%"></span></div><div class="comparison-track" title="Headings"><span class="comparison-bar headings" style="width:${headingWidth}%"></span></div></div><div class="comparison-meta">${n(row.topic_count||0)}/${n(totalTopics)} topics<br>${n(row.paragraph_count||0)} para · ${n(row.heading_count||0)} H</div></div>`;}).join('')}</div><div class="heatmap-legend"><span><i style="background:#1f9d66"></i>topics</span><span><i style="background:#2563eb"></i>paragraphs</span><span><i style="background:#0891b2"></i>headings</span></div>`;}
function contentDeltaBoxes(a){const comp=a.content_comparison||{};const s=comp.summary||a.summary||{};const b=comp.benchmark||{};const ours=comp.ours||{};return `<div class="delta-grid"><div class="delta-box"><strong>${n(s.missing_topics??s.missing??0)}</strong><span>missing topic groups</span></div><div class="delta-box"><strong>${n(s.partial_topics??s.partial??0)}</strong><span>partial topic groups</span></div><div class="delta-box"><strong>${n(b.median_competitor_topics||0)}</strong><span>median competitor topics</span></div></div><div class="mini">Target page: ${n(ours.paragraph_count||0)} paragraphs, ${n(ours.heading_count||0)} headings, ${n(ours.topic_count||0)} fully covered topics.</div>`;}
function visualReasonList(a){const reasons=a.visual_summary||[];if(!reasons.length)return '';return `<div class="why-list">${reasons.map(reason=>`<div class="why-item">${esc(reason)}</div>`).join('')}</div>`;}
function topicExamplesForRow(a,row){const direct=row.examples||[];if(direct.length)return direct;const match=(a.topics||[]).find(t=>String(t.label||'')===String(row.label||''));return match?.examples||[];}
function topicEvidenceHtml(a,row){if(row.coverage==='covered')return '';const examples=topicExamplesForRow(a,row).filter(ex=>String(ex.paragraph||'').trim()).slice(0,3);if(!examples.length)return '';return `<details class="topic-evidence"><summary>Ranking paragraphs missing or weak on ${esc(ownDomain)}</summary>${examples.map(ex=>{const domain=ex.domain||urlDomain(ex.url||'')||'ranking page';const rank=ex.rank?`SERP #${esc(ex.rank)}`:'SERP page';return `<div class="topic-snippet"><b>${esc(domain)} · ${rank}</b><p>${esc(ex.paragraph)}</p>${ex.url?`<div class="mini">${urlLink(ex.url)}</div>`:''}</div>`;}).join('')}</details>`;}
function topicCoverageHeatmap(a){const matrix=a.topic_coverage_matrix||{};const columns=matrix.columns||[];const rows=matrix.rows||[];if(!columns.length||!rows.length)return '<div class="empty">No topic coverage heatmap available.</div>';const header=`<div class="heatmap-head"><div>Topic</div>${columns.map(col=>`<div title="${esc(col.url||'')}">${esc((col.source==='ours'?ownDomain:col.label||'Page')).slice(0,22)}${col.rank?`<br>#${esc(col.rank)}`:''}</div>`).join('')}</div>`;const body=rows.map(row=>`<div class="heatmap-row">${`<div class="heatmap-topic"><strong title="${esc(row.label||'')}">${esc(row.label||'Untitled topic')}</strong><span class="mini">${esc(row.priority||'')} · seen ${n(row.competitor_coverage||0)} · sim ${Number(row.our_best_similarity||0).toFixed(2)}</span>${topicEvidenceHtml(a,row)}</div>`}${(row.cells||[]).map((cell,index)=>{const col=columns[index]||{};const status=cell.status||'not_seen';const label=col.source==='ours'?(status==='covered'?'covered':status==='partial'?'partial':'missing'):(status==='covered'?'seen':'-');const score=col.source==='ours'?` ${Number(cell.score||0).toFixed(2)}`:'';return `<div class="heatmap-cell ${esc(status)} ${col.source==='ours'?'ours':''}" title="${esc(row.label||'')} · ${esc(col.label||'')} · ${esc(status)}">${esc(label)}${esc(score)}</div>`;}).join('')}</div>`).join('');return `<div class="coverage-heatmap-wrap"><div class="coverage-heatmap" style="--heatmap-cols:${columns.length}">${header}${body}</div></div><div class="heatmap-legend"><span><i style="background:#eefbf4;border:1px solid #a5dabb"></i>covered / seen</span><span><i style="background:#eef4ff;border:1px solid #b9cdfc"></i>partial on target page</span><span><i style="background:#fff1f3;border:1px solid #f4b4be"></i>missing on target page</span></div>`;}
function paragraphMatchHeatmap(a){const heat=a.paragraph_match_heatmap||{};const columns=heat.columns||[];const rows=heat.rows||[];if(!columns.length||!rows.length)return '<div class="empty">No paragraph match heatmap available.</div>';const header=`<div class="heatmap-head"><div>${esc(ownDomain)} paragraph</div>${columns.map(col=>`<div title="${esc(col.url||'')}">${esc(col.domain||'Page').slice(0,22)}${col.rank?`<br>#${esc(col.rank)}`:''}</div>`).join('')}</div>`;const body=rows.map(row=>`<div class="paragraph-row"><div class="paragraph-topic"><strong>P${n(Number(row.paragraph_index||0)+1)} · ${esc(row.status||'')} · impact ${Number(row.max_rank_impact||0).toFixed(2)}</strong><span title="${esc(row.paragraph||'')}">${esc(row.paragraph||'')}</span></div>${(row.cells||[]).map((cell,index)=>{const col=columns[index]||{};const status=cell.status||'no_match';const sim=Number(cell.similarity||0);const impact=Number(cell.rank_impact||0);const label=sim?sim.toFixed(2):'-';const title=[`P${Number(row.paragraph_index||0)+1}`,col.domain||col.url||'',`SERP rank ${cell.rank||col.rank||''}`,`similarity ${label}`,`rank impact proxy ${impact.toFixed(3)}`,cell.paragraph||''].filter(Boolean).join(' · ');return `<div class="heatmap-cell ${esc(status)}" title="${esc(title)}">${esc(label)}<br><span class="mini">impact ${impact.toFixed(2)}</span></div>`;}).join('')}</div>`).join('');return `<div class="paragraph-heatmap-wrap"><div class="paragraph-heatmap" style="--paragraph-cols:${columns.length}">${header}${body}</div></div><div class="heatmap-legend"><span><i style="background:#eefbf4;border:1px solid #a5dabb"></i>strong paragraph match</span><span><i style="background:#eef4ff;border:1px solid #b9cdfc"></i>partial match</span><span><i style="background:#fff1f3;border:1px solid #f4b4be"></i>weak match</span><span>Impact = semantic match x top-10 SERP rank weight.</span></div>`;}
function semanticScatterSection(a){const points=a.scatter?.points||[];return `<div class="panel" style="margin-top:14px"><h4>Semantic Scatterplot</h4><div class="panel-body">${sectionNote('Vector space for keyword, target page, competitor titles, headings, and paragraphs. Use filters to isolate entity types and domains.')} ${scatterSvg(points)}</div></div>`;}
function visualComparisonSection(a){const hasComparison=a.content_comparison||a.topic_coverage_matrix||a.paragraph_match_heatmap||a.content_order_path;if(!hasComparison)return '';return `<div class="panel content-comparison" style="margin-top:14px"><h4>Why These Edits Matter</h4><div class="panel-body">${sectionNote('Top ranking page differences show why the recommended edits matter: compare topic coverage, paragraph depth, heading structure, content order, and which SERP pages cover each missing or partial topic.')} ${visualReasonList(a)}<div class="visual-grid"><div class="visual-card"><h5>SERP rank vs content coverage</h5>${contentComparisonRows(a)}</div><div class="visual-card"><h5>Top ranking page differences</h5>${contentDeltaBoxes(a)}</div></div><div class="visual-card coverage-heatmap-card" style="margin-top:14px"><h5>Topic coverage heatmap</h5>${topicCoverageHeatmap(a)}</div><div class="visual-card paragraph-match-card" style="margin-top:14px"><h5>Paragraph match heatmap</h5>${paragraphMatchHeatmap(a)}</div><div class="visual-card content-path-card" style="margin-top:14px"><h5>Content order semantic path</h5>${contentOrderPathSection(a)}</div></div></div>`;}
function keywordChartsSection(a){return `${intentWinnabilityPanel(a)}${serpFeaturesPanel(a)}${semanticScatterSection(a)}${visualComparisonSection(a)}`;}
function keywordId(pageIndex, keywordIndex){return `keyword-${pageIndex}-${keywordIndex}`;}
function overviewSection(){const points=data.overview_scatter?.points||[];const rankings=data.serp_url_rankings||[];const serpEvidence=`<div class="panel"><h4>Top-10 URLs Across Selected Keywords</h4><div class="panel-body">${sectionNote('Repeated winners show which URLs and intents Google currently rewards for the selected keywords.')} ${serpUrlGraph(rankings)}${serpRankingChart(rankings)}${urlDemandMetricsSection(rankings)}${serpRankingList(rankings)}${domainRatingAttribution()}</div></div>${keywordMetricsSection()}`;const semanticEvidence=`${keywordParagraphRidgelineSection(points)}<div class="panel" style="margin-top:14px"><h4>All Keywords, URLs, and Content</h4><div class="panel-body">${sectionNote('Vector-space chart for keywords, URLs, titles, headings, and paragraphs across the selected SERP set.')} ${scatterSvg(points)}</div></div>${keywordFrequencySection(points)}<div class="panel" style="margin-top:14px"><h4>Topic Traffic Impact</h4><div class="panel-body">${sectionNote('Directional demand by aggregate semantic cluster. Use it to decide which topic groups deserve editing effort first.')} ${clusterImpactChart(points)}</div></div><div class="panel" style="margin-top:14px"><h4>Aggregate Semantic Clusters</h4><div class="panel-body">${sectionNote('Broad topic groups across selected keywords and processed URLs.')} ${clusterCards(points)}</div></div>`;return `<section class="page-section" id="overview-section"><div class="page-head"><h2>SERP Content Task Board</h2><div class="mini">Charts first, then tasks: validate SERP and vector-space evidence before editing.</div></div><div class="keyword-card">${serpEvidence}${semanticEvidence}${aiAgentStatusSection()}${contentBriefsSection()}${aggregateActionsSection()}${paragraphRulesSection()}</div></section>`;}
function serpFeaturesPanel(a){const f=a.serp_features||{};const overview=f.ai_overview;const answer=f.answer_box||{};const rows=[];if(answer.answer)rows.push(`<div class="why-item"><b>Featured snippet:</b> ${esc(answer.format||'paragraph')} · ${n(answer.word_count||0)} words · ${urlLink(answer.url||'')}<div class="mini">${esc(answer.answer||'')}</div></div>`);if(overview&&overview.present){const cites=overview.cited_domains||[];const citesLabel=overview.cites_us===true?esc(ownDomain)+' cited':(overview.cites_us===false?esc(ownDomain)+' not cited':'citation status unknown');rows.push(`<div class="why-item"><b>AI Overview:</b> present · ${citesLabel}${cites.length?`<div class="mini">Cited domains: ${cites.map(d=>esc(d)).join(' · ')}</div>`:''}</div>`);}return rows.length?collapsiblePanel('SERP Features',`${sectionNote('Answer-box and AI Overview evidence available from the SERP provider for this keyword.')}<div class="why-list">${rows.join('')}</div>`,{meta:`${n(rows.length)} feature${rows.length===1?'':'s'}`}):'';}
function keywordCard(a, pageIndex, keywordIndex){const s=a.summary||{};const reviewRows=(a.off_intent_paragraphs&&a.off_intent_paragraphs.length)?a.off_intent_paragraphs:a.own_paragraphs_to_review;const semanticEvidence=`<div class="tables"><div class="panel"><h4>Competitor SERP</h4><div class="panel-body">${sectionNote('Ranking pages fetched for this keyword after ignored hosts are skipped.')} ${competitorList(a.competitor_pages)}</div></div></div>`;const actionRows=a.action_points||[];const actionsPanel=collapsiblePanel('Keyword Content Actions',`${sectionNote('Edit these after reviewing the charts. Each task describes paragraph structure, placement, and completion criteria.')} ${actionList(actionRows,10)}`,{meta:`${n(actionRows.length)} task${actionRows.length===1?'':'s'}`});const structRows=(a.structural_patterns||[]).filter(r=>(r.competitors||0)>=1);const outlineRows=a.recommended_outline||[];const outlinePanel=outlineRows.length?collapsiblePanel('Recommended Section Order',`${sectionNote('Section themes ordered by where ranking competitors place them. "add" rows are themes the page does not cover yet.')}<ol style="margin:0;padding-left:20px">${outlineRows.map(r=>`<li style="margin-bottom:6px" title="${esc(r.sample_text||'')}"><span class="chip ${r.status==='have'?'covered':'missing'}">${esc(r.status)}</span> ${esc(r.label||'')} <span class="mini">— seen on ${n(r.competitor_pages||0)} competitor page(s)</span></li>`).join('')}</ol>`,{meta:`${n(outlineRows.filter(r=>r.status==='add').length)} to add`}):'';const paaRows=a.paa_coverage||[];const paaCovered=paaRows.filter(r=>r.status==='covered').length;const relatedQueries=(a.serp_features&&a.serp_features.related_searches)||[];const paaPanel=paaRows.length?collapsiblePanel('People Also Ask Coverage',`${sectionNote('Questions Google shows for this keyword, scored against this page\'s paragraphs. Missing questions are direct content opportunities.')}<table><thead><tr><th>Question</th><th>Status</th><th>Best similarity</th><th>Closest paragraph</th></tr></thead><tbody>${paaRows.map(r=>`<tr><td>${esc(r.question)}</td><td><span class="chip ${esc(r.status)}">${esc(r.status)}</span></td><td>${esc(String(r.best_similarity??''))}</td><td>${esc(r.best_paragraph||'')}</td></tr>`).join('')}</tbody></table>${relatedQueries.length?`<div class="mini" style="margin-top:8px">Related searches: ${relatedQueries.map(q=>esc(q)).join(' · ')}</div>`:''}`,{meta:`${n(paaCovered)}/${n(paaRows.length)} covered`}):'';const structPanel=structRows.length?collapsiblePanel('Structural / GEO Gaps',`${sectionNote('Page-structure signals where ranking competitors beat this page: schema, question headings, statistics, citations, tables, depth.')}<table><thead><tr><th>Signal</th><th>Competitors</th><th>Ours</th><th>Theirs (max)</th><th>Advice</th></tr></thead><tbody>${structRows.map(r=>`<tr><td>${esc(r.signal)}</td><td>${n(r.competitors)}</td><td>${esc(String(r.ours))}</td><td>${esc(String(r.max_theirs))}</td><td>${esc(r.advice)}</td></tr>`).join('')}</tbody></table>`,{meta:`${n(structRows.length)} signal${structRows.length===1?'':'s'}`}):'';const topicPanel=collapsiblePanel('Topic Relations',`${sectionNote('Missing and partial topics are the source evidence for the content tasks above.')} ${topicChart(a.topics)}<table><thead><tr><th>Coverage</th><th>Priority</th><th>Topic and Example</th><th>Seen</th><th>${esc(ownDomain)} sim</th><th>Example URL</th></tr></thead><tbody>${topicRows(a.topics,18)}</tbody></table>`,{meta:`${n((a.topics||[]).length)} topics`,style:''});const reviewPanel=collapsiblePanel(`${ownDomain} Paragraphs To Review`,`${sectionNote('Review candidates for intent drift, thinness, or filler. Keep useful facts, but rewrite, move, merge, or remove weak paragraphs.')} ${reviewList(reviewRows)}`,{meta:`${n((reviewRows||[]).length)} paragraph${(reviewRows||[]).length===1?'':'s'}`,style:''});return `<div class="keyword-card" id="${keywordId(pageIndex, keywordIndex)}"><div class="keyword-head"><div><h3>${esc(a.query || a.keyword?.keyword || '')}</h3><div class="mini">Status ${esc(a.status)} · Competitors ${esc(a.competitors||a.competitor_pages?.length||0)} · Scatter points ${esc(a.scatter?.shown||0)}</div></div><div class="chips"><span class="chip missing">Missing ${n(s.missing||0)}</span><span class="chip partial">Partial ${n(s.partial||0)}</span><span class="chip covered">Covered ${n(s.covered||0)}</span></div></div>${keywordChartsSection(a)}${actionsPanel}${structPanel}${paaPanel}${outlinePanel}<div class="two-col" style="margin-top:14px">${topicPanel}${reviewPanel}</div>${diagnosticDetails('Raw semantic evidence for this keyword',semanticEvidence,false)}</div>`;}
function decisionChipClass(decision){if(decision==='keep')return 'covered';if(decision==='rewrite')return 'partial';return 'missing';}
function recommendationSection(page){const rec=page.ai_recommendation||{};if(!rec.status)return '';if(rec.status!=='ok'&&rec.status!=='invalid_recommendation'){return '';}
const d=rec.data||{};const parts=[];
if(rec.status==='invalid_recommendation'&&(rec.errors||[]).length){parts.push(`<div class="mini" style="color:var(--missing);margin-bottom:8px">Recommendation failed validation: ${rec.errors.slice(0,6).map(e=>esc(e)).join('; ')}</div>`);}
const verif=rec.verification||{};const vs=verif.summary||{};
if((verif.topics||[]).length){const unresolved=vs.unresolved_critical||[];parts.push(`<div style="margin-bottom:10px"><div class="chips" style="margin-bottom:6px"><span class="chip ${vs.missing_after>0?'missing':'covered'}">Missing ${n(vs.missing_before)} → ${n(vs.missing_after)}</span><span class="chip ${vs.partial_after>0?'partial':'covered'}">Partial ${n(vs.partial_before)} → ${n(vs.partial_after)}</span><span class="chip ${vs.paa_missing_after>0?'missing':'covered'}">PAA missing ${n(vs.paa_missing_before)} → ${n(vs.paa_missing_after)}</span></div>${unresolved.length?`<div class="mini" style="color:var(--missing)">Still uncovered critical/high topics: ${unresolved.map(t=>esc(t)).join('; ')}</div>`:''}<details><summary class="mini">Coverage check details</summary><table><thead><tr><th>Keyword</th><th>Topic</th><th>Priority</th><th>Before → After</th><th>Best similarity</th></tr></thead><tbody>${verif.topics.map(r=>`<tr><td>${esc(r.keyword||'')}</td><td>${esc(r.label||'')}</td><td>${esc(r.priority||'')}</td><td><span class="chip ${esc(r.before)}">${esc(r.before)}</span> → <span class="chip ${esc(r.after)}">${esc(r.after)}</span></td><td>${esc(String(r.best_similarity??''))}</td></tr>`).join('')}</tbody></table></details></div>`);}
const unverifiedNums=verif.unverified_numbers||[];if(unverifiedNums.length){parts.push(`<div class="mini" style="color:var(--missing);margin-bottom:10px"><b>Unverified numeric claims:</b> ${unverifiedNums.map(c=>`${esc(c.text||c.number||'')} (${esc(String(c.context||'').slice(0,140))})`).join('; ')}</div>`);}
const titleRec=d.title||{};const h1Rec=d.h1||{};const metaRec=d.meta_description||{};const assessment=d.page_assessment||{};
const headParts=[];
if(assessment.reason)headParts.push(`<div class="mini" style="margin-bottom:6px">Target page check: <b>${assessment.is_right_target_page===false?'wrong target page':'right target page'}</b> — ${esc(assessment.reason||'')}</div>`);
if(titleRec.recommended&&titleRec.recommended!==titleRec.current)headParts.push(`<div class="mini"><b>Title:</b> ${esc(titleRec.recommended)}${titleRec.reason?` <span class="muted">(${esc(titleRec.reason)})</span>`:''}</div>`);
if(h1Rec.recommended)headParts.push(`<div class="mini"><b>H1:</b> ${esc(h1Rec.recommended)}</div>`);
if(metaRec.recommended)headParts.push(`<div class="mini"><b>Meta description:</b> ${esc(metaRec.recommended)}</div>`);
if(headParts.length)parts.push(`<div style="margin-bottom:10px">${headParts.join('')}</div>`);
const outline=d.outline||[];
if(outline.length){parts.push(`<h4 style="margin:10px 0 6px">Recommended outline</h4><table><thead><tr><th>Level</th><th>Heading</th><th>Status</th><th>Maps to topic</th></tr></thead><tbody>${outline.map(r=>`<tr><td>H${esc(String(r.level||2))}</td><td>${esc(r.heading||'')}</td><td><span class="chip ${r.status==='keep'?'covered':(r.status==='remove'?'missing':'partial')}">${esc(r.status||'')}</span></td><td>${esc(r.maps_to_topic||'')}</td></tr>`).join('')}</tbody></table>`);}
const decisions=d.paragraph_decisions||[];
if(decisions.length){parts.push(`<h4 style="margin:14px 0 6px">Paragraph decisions</h4><table><thead><tr><th>Paragraph</th><th>Decision</th><th>Reason</th></tr></thead><tbody>${decisions.map(r=>`<tr><td>[P${esc(String(r.index))}]</td><td><span class="chip ${decisionChipClass(r.decision)}">${esc(r.decision||'')}</span></td><td>${esc(r.reason||'')}${r.rewrite?`<details style="margin-top:4px"><summary class="mini">Replacement text</summary>${mdToHtml(r.rewrite)}</details>`:''}</td></tr>`).join('')}</tbody></table>`);}
const sections=d.new_sections||[];
if(sections.length){parts.push(`<h4 style="margin:14px 0 6px">New sections</h4><table><thead><tr><th>Heading</th><th>Placement</th><th>Format</th><th>Covers PAA</th><th>Draft</th></tr></thead><tbody>${sections.map(r=>`<tr><td>${esc(r.heading||'')}</td><td>${r.placement_after_paragraph===-1?'top':'after [P'+esc(String(r.placement_after_paragraph))+']'}</td><td>${esc(r.format||'')}</td><td>${n((r.covers_paa||[]).length)}</td><td>${r.draft?`<details><summary class="mini">Show draft</summary>${mdToHtml(r.draft)}</details>`:''}</td></tr>`).join('')}</tbody></table>`);}
const schema=d.structured_data||[];const links=d.internal_links||[];
if(schema.length)parts.push(`<div class="mini" style="margin-top:10px"><b>Structured data:</b> ${schema.map(r=>`${esc(r.type||'')} — ${esc(r.reason||'')}`).join(' · ')}</div>`);
if(links.length)parts.push(`<div class="mini" style="margin-top:6px"><b>Internal links:</b> ${links.map(r=>`"${esc(r.anchor||'')}" (${esc(r.from_hint||'')})`).join(' · ')}</div>`);
if(rec.article_markdown){parts.push(`<details style="margin-top:14px" open><summary><b>Final Recommended Article</b> <span class="mini">(full page in final reading order: kept paragraphs, rewrites, new sections — plus why it should rank better; also saved as recommended-article-*.md next to this report)</span></summary>${mdToHtml(rec.article_markdown)}</details>`);}
if(!parts.length)return '';
return collapsiblePanel('AI Page Recommendation',`${sectionNote('Structured, validated recommendation produced by the AI agent from the full computed evidence.')}${parts.join('')}`,{meta:rec.status==='ok'?'validated':'validation errors',open:true});}
function pageSection(page, index){const aiBrief=aiEditorBriefSection(page);const aiRec=recommendationSection(page);return `<section class="page-section" id="page-${index}"><div class="page-head"><h2>${index+1}. ${esc(page.title || page.url)}</h2><div class="url">${urlLink(page.url)}</div>${page.h1?`<div class="mini">H1: ${esc(page.h1)}</div>`:''}</div>${aiRec?`<div class="keyword-card">${aiRec}</div>`:''}${aiBrief?`<div class="keyword-card">${aiBrief}</div>`:''}${(page.analyses||[]).map((analysis, keywordIndex)=>keywordCard(analysis, index, keywordIndex)).join('') || '<div class="empty">No keyword analyses for this page.</div>'}</section>`;}
function buildNav(){if(!navEl)return;navEl.innerHTML=`<button type="button" class="report-nav-button" data-target="overview-section"><span class="report-nav-label">Task board</span></button>`+(data.pages||[]).map((page,pageIndex)=>`<button type="button" class="report-nav-button" data-target="page-${pageIndex}"><span class="report-nav-label">${esc(page.title||page.url||`Page ${pageIndex+1}`)}</span></button>${(page.analyses||[]).map((analysis,keywordIndex)=>`<button type="button" class="report-nav-button nav-keyword" data-target="${keywordId(pageIndex,keywordIndex)}"><span class="report-nav-label">${esc(analysis.query||analysis.keyword?.keyword||`Keyword ${keywordIndex+1}`)}</span></button>`).join('')}`).join('');const buttons=[...navEl.querySelectorAll('.report-nav-button')];const sections=buttons.map(button=>document.getElementById(button.dataset.target||'')).filter(Boolean);buttons.forEach(button=>button.addEventListener('click',()=>{const target=document.getElementById(button.dataset.target||'');if(target)target.scrollIntoView({block:'start',behavior:'smooth'});}));function update(){let active=0;for(let i=0;i<sections.length;i++){if(sections[i].getBoundingClientRect().top<160)active=i;}buttons.forEach((button,i)=>{const selected=i===active;button.classList.toggle('is-active',selected);button.setAttribute('aria-current',selected?'page':'false');});}document.addEventListener('scroll',update,{passive:true});update();}

function bindSerpUrlGraphInteractions(){document.querySelectorAll('.serp-url-graph-wrap').forEach(wrap=>{const tooltip=wrap.querySelector('.graph-tooltip');const clear=()=>{wrap.classList.remove('has-active');wrap.querySelectorAll('.is-active').forEach(el=>el.classList.remove('is-active'));tooltip?.classList.remove('open');};function show(el,event){let detail={};try{detail=JSON.parse(el.getAttribute('data-graph-detail')||'{}');}catch(_){}wrap.classList.add('has-active');wrap.querySelectorAll('.is-active').forEach(active=>active.classList.remove('is-active'));el.classList.add('is-active');if(el.matches('[data-graph-node]')){const index=el.getAttribute('data-node-index');wrap.querySelectorAll(`[data-graph-edge][data-nodes*=",${index},"]`).forEach(edge=>edge.classList.add('is-active'));}else if(el.matches('[data-graph-edge]')){(el.getAttribute('data-nodes')||'').split(',').filter(Boolean).forEach(index=>wrap.querySelector(`[data-graph-node][data-node-index="${index}"]`)?.classList.add('is-active'));}if(tooltip){tooltip.innerHTML=graphTooltipHtml(detail);tooltip.classList.add('open');position(event);}}function position(event){if(!tooltip||!event)return;const rect=wrap.getBoundingClientRect();const x=Math.min(Math.max(10,event.clientX-rect.left+14),Math.max(10,rect.width-380));const y=Math.min(Math.max(10,event.clientY-rect.top+14),Math.max(10,rect.height-190));tooltip.style.left=`${x}px`;tooltip.style.top=`${y}px`;}wrap.querySelectorAll('[data-graph-node],[data-graph-edge]').forEach(el=>{el.addEventListener('mouseenter',event=>show(el,event));el.addEventListener('mousemove',position);el.addEventListener('focus',event=>show(el,event));el.addEventListener('mouseleave',clear);el.addEventListener('blur',clear);});});}

function bindContentPathInteractions(){document.querySelectorAll('.content-path-wrap').forEach(wrap=>{const items=[...wrap.querySelectorAll('[data-path-cluster]')];if(!items.length)return;function clear(){wrap.classList.remove('has-active');items.forEach(el=>el.classList.remove('is-active'));}function activate(cluster){if(!cluster)return;wrap.classList.add('has-active');items.forEach(el=>el.classList.toggle('is-active',el.getAttribute('data-path-cluster')===cluster));}items.forEach(el=>{const cluster=()=>el.getAttribute('data-path-cluster')||'';el.addEventListener('mouseenter',()=>activate(cluster()));el.addEventListener('focus',()=>activate(cluster()));el.addEventListener('click',event=>{event.stopPropagation();activate(cluster());});});wrap.addEventListener('click',event=>{if(!event.target.closest?.('[data-path-cluster]'))clear();});wrap.addEventListener('mousemove',event=>{if(!event.target.closest?.('[data-path-cluster]'))clear();});wrap.addEventListener('mouseleave',clear);wrap.addEventListener('focusout',()=>setTimeout(()=>{if(!wrap.contains(document.activeElement))clear();},0));});}

function bindScatterInteractions(){document.querySelectorAll('.scatter-wrap').forEach(wrap=>{const svg=wrap.querySelector('svg.scatter');const tooltip=wrap.querySelector('.scatter-tooltip');if(!svg||!tooltip)return;let base=(svg.getAttribute('data-base-viewbox')||svg.getAttribute('viewBox')||'0 0 820 390').trim().split(/\s+/).map(Number);if(base.length!==4||base.some(v=>!Number.isFinite(v))||base[2]<=0||base[3]<=0)base=[0,0,820,390];let vb={x:base[0],y:base[1],w:base[2],h:base[3]};const setVb=()=>{if([vb.x,vb.y,vb.w,vb.h].every(Number.isFinite)&&vb.w>0&&vb.h>0)svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);};function zoomAt(factor,cx=base[2]/2,cy=base[3]/2){const nx=cx-(cx-vb.x)*factor;const ny=cy-(cy-vb.y)*factor;const next={x:nx,y:ny,w:vb.w*factor,h:vb.h*factor};if([next.x,next.y,next.w,next.h].every(Number.isFinite)&&next.w>0&&next.h>0){vb=next;setVb();}}function pointFromEvent(event){const rect=svg.getBoundingClientRect();return{x:vb.x+(event.clientX-rect.left)/Math.max(rect.width,1)*vb.w,y:vb.y+(event.clientY-rect.top)/Math.max(rect.height,1)*vb.h};}function show(point){let detail=null;try{detail=JSON.parse(point.getAttribute('data-detail')||'{}');}catch(_){detail={type:'point',explanation:point.getAttribute('data-tooltip')||point.getAttribute('aria-label')||''};}tooltip.innerHTML=pointDetailHtml(detail);tooltip.classList.add('open');tooltip.querySelector('.tip-close')?.addEventListener('click',event=>{event.stopPropagation();tooltip.classList.remove('open');});}wrap.querySelectorAll('.scatter-point').forEach(point=>{point.addEventListener('click',event=>{event.stopPropagation();show(point);});point.addEventListener('focus',()=>show(point));});svg.addEventListener('wheel',event=>{event.preventDefault();const p=pointFromEvent(event);zoomAt(event.deltaY<0?0.82:1.22,p.x,p.y);},{passive:false});let drag=null;svg.addEventListener('mousedown',event=>{if(event.target.classList?.contains('scatter-point'))return;drag={x:event.clientX,y:event.clientY,vx:vb.x,vy:vb.y};svg.classList.add('is-panning');});window.addEventListener('mousemove',event=>{if(!drag)return;const rect=svg.getBoundingClientRect();const nextX=drag.vx-(event.clientX-drag.x)/Math.max(rect.width,1)*vb.w;const nextY=drag.vy-(event.clientY-drag.y)/Math.max(rect.height,1)*vb.h;if(Number.isFinite(nextX)&&Number.isFinite(nextY)){vb.x=nextX;vb.y=nextY;setVb();}});window.addEventListener('mouseup',()=>{drag=null;svg.classList.remove('is-panning');});svg.addEventListener('dblclick',()=>{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();});wrap.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',event=>{event.stopPropagation();const action=button.getAttribute('data-zoom');if(action==='in')zoomAt(0.78);else if(action==='out')zoomAt(1.28);else{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();}}));});document.addEventListener('click',event=>{const target=event.target;if(target?.closest?.('.scatter-tooltip')||target?.closest?.('.scatter-point'))return;document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});}
function bindScatterFilters(){document.querySelectorAll('.panel-body').forEach(panel=>{const entityFilters=[...panel.querySelectorAll('[data-entity-filter]')];const domainFilters=[...panel.querySelectorAll('[data-domain-filter]')];const filters=[...entityFilters,...domainFilters];const points=[...panel.querySelectorAll('.scatter-point')];const tooltip=panel.querySelector('.scatter-tooltip');if(!filters.length||!points.length)return;function apply(){const visibleEntities=new Set(entityFilters.filter(input=>input.checked).map(input=>input.dataset.entityFilter));const visibleDomains=new Set(domainFilters.filter(input=>input.checked).map(input=>input.dataset.domainFilter));points.forEach(point=>{const entityVisible=visibleEntities.has(point.dataset.entity||'paragraph');const domain=point.dataset.domain||'';const domainVisible=!domain||visibleDomains.has(domain);const show=entityVisible&&domainVisible;point.style.display=show?'':'none';if(!show&&document.activeElement===point)point.blur();});tooltip?.classList.remove('open');}filters.forEach(input=>input.addEventListener('change',apply));apply();});}
overviewEl.innerHTML = overviewSection();
app.innerHTML = (data.pages||[]).map(pageSection).join('') || '<div class="empty">No analyzed pages in this report.</div>';
buildNav();
bindSerpUrlGraphInteractions();
bindContentPathInteractions();
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
