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
from .embedder import DEFAULT_MODEL, Embedder
from .extractor import ExtractedPage, extract
from .scatter import project


USER_AGENT = "site-audit-serp-gap/0.1 (+https://github.com/vzeman/site-audit)"


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
    manual_keywords = _load_manual_keywords(config)
    keyword_rows, skipped_keywords = _select_keywords(
        selected_pages,
        search_payload,
        manual_keywords,
        config,
    )
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
    for page in selected_pages:
        page_keywords = [row for row in keyword_rows if _same_url(row["url"], page.url)]
        if not page_keywords:
            continue
        own_ext = _fetch_and_extract(page.url, own_cache, refresh=False)
        if own_ext is None:
            skipped_pages.append({"url": page.url, "reason": "own page fetch/extract failed"})
            continue

        page_blocks: list[dict] = []
        for kw in page_keywords:
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
            targets = _targets_from_serp(config.domain, kw["keyword"], serp, config)
            remaining_slots = max(0, config.max_competitor_pages - len(all_competitor_urls))
            targets = [t for t in targets if t.competitor_url not in all_competitor_urls][:remaining_slots]
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
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


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
            results_per_keyword=config.results_per_keyword,
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
        json={"q": keyword, "gl": country, "hl": language, "num": config.results_per_keyword + 3},
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
    provider = (payload.get("meta") or {}).get("provider") or ""
    rows = []
    if provider == "serper" or "organic" in (payload.get("raw") or {}):
        for item in (payload.get("raw") or {}).get("organic") or []:
            rows.append({"url": item.get("link"), "rank": item.get("position")})
    else:
        for item in _serp_items(payload):
            if item.get("type") == "organic":
                rows.append({"url": item.get("url"), "rank": item.get("rank_group") or item.get("rank_absolute")})
    out = []
    seen = set()
    for row in rows:
        url = row.get("url") or ""
        if not url or _is_own_url(url, domain) or url in seen:
            continue
        seen.add(url)
        out.append(CompetitiveTarget(keyword, url, keyword, _safe_int(row.get("rank"))))
        if len(out) >= config.results_per_keyword:
            break
    return out


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
    gap["scatter"] = _scatter(keyword["keyword"], own_ext, own_embeddings, competitor_pages, embedder)
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
    keyword: str,
    own_ext: ExtractedPage,
    own_para_embeddings: np.ndarray,
    competitor_pages: list[CompetitorPage],
    embedder: Embedder,
) -> dict:
    rows = []
    texts = [keyword, own_ext.title, own_ext.h1, *[h.get("text", "") for h in own_ext.headers_rich[:40]]]
    meta = [
        {"entity_type": "keyword", "source": "keyword", "text": keyword, "url": own_ext.url},
        {"entity_type": "title", "source": "ours", "text": own_ext.title, "url": own_ext.url},
        {"entity_type": "h1", "source": "ours", "text": own_ext.h1, "url": own_ext.url},
        *[
            {"entity_type": "header", "source": "ours", "text": h.get("text", ""), "level": h.get("level"), "url": own_ext.url}
            for h in own_ext.headers_rich[:40]
        ],
    ]
    base_embs = embedder.encode(texts, batch_size=64).astype(np.float32) if texts else np.zeros((0, 0), dtype=np.float32)
    embs = [base_embs]
    for i, para in enumerate(own_ext.paragraphs or []):
        if i >= len(own_para_embeddings):
            break
        meta.append({"entity_type": "paragraph", "source": "ours", "text": para[:300], "url": own_ext.url})
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
            })
        if len(cp.paragraph_embeddings):
            embs.append(cp.paragraph_embeddings[:60])
    if not embs:
        return {"points": [], "shown": 0}
    matrix = np.vstack([e for e in embs if len(e)]).astype(np.float32)
    if len(matrix) != len(meta):
        meta = meta[:len(matrix)]
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
    return {"points": points[:1600], "shown": min(len(points), 1600), "total": len(points)}


def _summary(page_results: list[dict], selected_pages: list[PageInfo], keyword_rows: list[dict], plan: dict) -> dict:
    analyses = [a for p in page_results for a in p.get("analyses", [])]
    summaries = [a.get("summary") or {} for a in analyses if a.get("status") == "ok"]
    return {
        **plan,
        "pages_analyzed": len(page_results),
        "pages_selected": len(selected_pages),
        "keywords_selected": len(keyword_rows),
        "serp_clusters": len(summaries),
        "missing_topics": sum(int(s.get("missing", 0)) for s in summaries),
        "partial_topics": sum(int(s.get("partial", 0)) for s in summaries),
        "off_intent_paragraphs": sum(int(s.get("off_intent_paragraphs", 0)) for s in summaries),
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
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:14px;line-height:1.45}a{color:#1b5dbf;text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1440px;margin:0 auto;padding:24px}.topbar{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:18px}.title h1{font-size:28px;line-height:1.1;margin:0 0 8px}.title p{margin:0;color:var(--muted);max-width:820px}.summary{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin:18px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;box-shadow:var(--shadow)}.metric b{display:block;font-size:22px;line-height:1.1}.metric span{color:var(--muted);font-size:12px}.page-section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:18px 0 28px;box-shadow:var(--shadow);overflow:hidden}.page-head{padding:18px 20px;border-bottom:1px solid var(--line);background:#fff}.page-head h2{font-size:21px;margin:0 0 6px}.url{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.keyword-card{padding:20px;border-top:1px solid var(--line)}.keyword-card:first-of-type{border-top:0}.keyword-grid{display:grid;grid-template-columns:minmax(460px,1.2fr) minmax(420px,.8fr);gap:18px;align-items:start}.keyword-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}.keyword-head h3{font-size:19px;margin:0}.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:3px 8px;background:#fff;font-size:12px;color:var(--muted)}.chip.missing{color:var(--missing);border-color:#f1b4ad;background:#fff7f6}.chip.partial{color:var(--partial);border-color:#e8cf85;background:#fff9e8}.chip.covered{color:var(--covered);border-color:#a8d5b6;background:#f1faf4}.panel{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.panel h4{margin:0;padding:11px 12px;border-bottom:1px solid var(--line);font-size:14px;background:var(--soft)}.panel-body{padding:12px}.scatter-wrap{position:relative}.scatter{width:100%;height:390px;display:block;background:#fbfcfe}.scatter-point{cursor:pointer}.scatter-point:focus{outline:none;stroke:#111;stroke-width:2.4}.scatter-tooltip{display:none;position:absolute;left:12px;right:12px;bottom:12px;max-height:250px;overflow:auto;background:linear-gradient(180deg,#fff,#f8fbff);border:1px solid #b9c7d8;border-radius:10px;box-shadow:0 14px 34px rgba(22,34,51,.22);padding:0;z-index:2}.scatter-tooltip.open{display:block}.tip-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:11px 12px;border-bottom:1px solid #e3e9f2;background:#f4f7fb}.tip-title{font-weight:750;font-size:13px;color:#142033}.tip-sub{margin-top:2px;color:#637083;font-size:11px}.tip-close{border:0;background:#e7edf5;border-radius:6px;padding:2px 8px;cursor:pointer}.tip-body{padding:11px 12px}.tip-badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.tip-badge{display:inline-flex;align-items:center;border:1px solid #d5deea;border-radius:999px;padding:3px 7px;background:#fff;font-size:11px;color:#405166}.tip-badge.ours{border-color:#9bd0ae;color:#176a35;background:#f1faf4}.tip-badge.competitor{border-color:#acc3e5;color:#2d5b9a;background:#f3f7ff}.tip-badge.keyword{border-color:#e7c889;color:#8a4b00;background:#fff9e8}.tip-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:9px}.tip-field{border:1px solid #edf1f5;border-radius:7px;background:#fff;padding:6px 7px;min-width:0}.tip-field span{display:block;color:#768395;font-size:10px;text-transform:uppercase;letter-spacing:.04em}.tip-field strong{display:block;color:#1b2838;font-size:12px;overflow-wrap:anywhere}.tip-text{border-left:3px solid #8fb2df;background:#f7faff;border-radius:6px;padding:8px 9px;color:#263445;font-size:12px;line-height:1.45}.tip-explain{color:#637083;font-size:12px;margin-bottom:9px}.scatter-controls{position:absolute;top:10px;right:10px;display:flex;gap:5px;z-index:3}.scatter-controls button{border:1px solid #c9d3df;background:#fff;color:#263445;border-radius:6px;padding:3px 8px;font-size:12px;line-height:1;box-shadow:0 1px 3px rgba(22,34,51,.12);cursor:pointer}.scatter-controls button:hover{background:#f1f5f9}.scatter.is-panning{cursor:grabbing}.scatter{cursor:grab}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.tables{display:grid;grid-template-columns:1fr;gap:14px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid #edf1f5;text-align:left;vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#fbfcfe}.topic-label{font-weight:600}.coverage-missing{color:var(--missing);font-weight:700}.coverage-partial{color:var(--partial);font-weight:700}.coverage-covered{color:var(--covered);font-weight:700}.cluster-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.cluster{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fff}.cluster strong{display:block;margin-bottom:5px}.bar{height:7px;background:#e9eef5;border-radius:999px;overflow:hidden;margin:8px 0}.bar span{display:block;height:100%;background:#5f8cc9}.muted{color:var(--muted)}.empty{padding:18px;color:var(--muted)}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}.competitors li,.review li{margin:0 0 8px}.competitors,.review{padding-left:18px;margin:0}.mini{font-size:12px;color:var(--muted)}@media(max-width:980px){.wrap{padding:14px}.summary{grid-template-columns:repeat(2,1fr)}.keyword-grid,.two-col{grid-template-columns:1fr}.scatter{height:320px}.topbar{display:block}}
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
  <div id="app"></div>
</div>
<script>
const data = __DATA__;
const app = document.getElementById('app');
const summaryEl = document.getElementById('summary');
const statusEl = document.getElementById('status');
const navEl = document.getElementById('report-nav');
const colors = {keyword:'#8a4b00', ours:'#176a35', competitor:'#2d5b9a', title:'#7b3fb2', h1:'#b65f00', header:'#68788b', paragraph:'#2d5b9a'};
function esc(s){return String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function n(v){return Number(v||0).toLocaleString();}
function pct(v){return Math.round(Number(v||0)*100)+'%';}
const domainPalette=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#be123c','#4f46e5','#0f766e','#a16207','#7c3aed','#15803d'];
function hashString(s){let h=0;for(let i=0;i<String(s||'').length;i++)h=(h*31+String(s).charCodeAt(i))>>>0;return h;}
function domainColor(domain){return domainPalette[hashString(domain)%domainPalette.length];}
function sourceColor(p){if(p.entity_type==='keyword')return colors.keyword;if(p.source==='ours')return colors.ours;if(p.source==='competitor'&&p.domain)return domainColor(p.domain);if(p.entity_type==='title')return colors.title;if(p.entity_type==='h1')return colors.h1;if(p.entity_type==='header')return colors.header;return colors.competitor;}
function metrics(){const s=data.summary||{};return [['Pages',s.pages_analyzed||0],['Keywords',s.keywords_selected||0],['SERP calls',s.serp_api_calls_after_cache ?? s.serp_api_calls ?? 0],['Missing topics',s.missing_topics||0],['Partial topics',s.partial_topics||0],['Review paragraphs',s.off_intent_paragraphs||0]];}
statusEl.textContent = `${data.domain || ''} · ${data.provider || ''} · ${data.status || ''}`;
summaryEl.innerHTML = metrics().map(([label,value])=>`<div class="metric"><b>${n(value)}</b><span>${esc(label)}</span></div>`).join('');
function pointLabel(d){if(d.type==='keyword')return 'Keyword';if(d.type==='title')return 'Page title';if(d.type==='h1')return 'H1 heading';if(d.type==='header')return 'Header';if(d.source==='ours')return 'Our paragraph';if(d.source==='competitor')return 'Competitor paragraph';return d.type||'Point';}
function pointDetail(p){const type=p.entity_type||'point';return{type,source:p.source||'',cluster:p.cluster??'',domain:p.domain||'',rank:p.rank||'',url:p.url||'',text:String(p.text||'').slice(0,520),keyword_similarity:p.keyword_similarity??'',keyword_distance:p.keyword_distance??''};}
function pointTooltip(p){const d=pointDetail(p);return [pointLabel(d),d.text,d.keyword_distance!==''&&`Keyword distance: ${d.keyword_distance}`,d.keyword_similarity!==''&&`Keyword similarity: ${d.keyword_similarity}`,d.domain&&`Domain: ${d.domain}`,d.rank&&`SERP rank: ${d.rank}`,d.cluster!==''&&`Cluster: ${d.cluster}`].filter(Boolean).join('\\n');}
function pointDetailHtml(d){const sourceClass=d.type==='keyword'?'keyword':d.source==='ours'?'ours':d.source==='competitor'?'competitor':'';const badges=[];badges.push(`<span class="tip-badge ${sourceClass}">${esc(d.source||d.type)}</span>`);if(d.keyword_distance!==''&&d.keyword_distance!==undefined)badges.push(`<span class="tip-badge">distance ${esc(d.keyword_distance)}</span>`);if(d.keyword_similarity!==''&&d.keyword_similarity!==undefined)badges.push(`<span class="tip-badge">similarity ${esc(d.keyword_similarity)}</span>`);if(d.domain)badges.push(`<span class="tip-badge">${esc(d.domain)}</span>`);if(d.rank)badges.push(`<span class="tip-badge">rank ${esc(d.rank)}</span>`);if(d.cluster!==''&&d.cluster!==undefined)badges.push(`<span class="tip-badge">cluster ${esc(d.cluster)}</span>`);return `<div class="tip-head"><div><div class="tip-title">${esc(pointLabel(d))}</div>${d.url?`<div class="tip-sub">${esc(d.url)}</div>`:''}</div><button class="tip-close" type="button" aria-label="Close tooltip">x</button></div><div class="tip-body"><div class="tip-badges">${badges.join('')}</div><div class="tip-text">${esc(d.text||'(no text captured)')}</div></div>`;}
function clusterSummary(points){const groups=new Map();for(const p of points||[]){const id=p.cluster ?? 0;const g=groups.get(id)||{id,total:0,ours:0,competitor:0,keyword:0,headers:0,samples:[]};g.total++;if(p.entity_type==='keyword')g.keyword++;else if(p.source==='ours')g.ours++;else if(p.source==='competitor')g.competitor++;if(['h1','header','title'].includes(p.entity_type))g.headers++;if(g.samples.length<3 && p.text)g.samples.push(p.text);groups.set(id,g);}return [...groups.values()].sort((a,b)=>b.total-a.total).slice(0,8);}
function pointSize(p){if(p.clicks)return Math.max(5,Math.min(14,4+Math.sqrt(Number(p.clicks)||0)));if(p.rank)return Math.max(4.5,12-Number(p.rank||10)*0.65);if(p.entity_type==='keyword')return 8;if(p.entity_type==='title'||p.entity_type==='h1')return 7;if(p.entity_type==='header')return 5.8;return 3.9;}
function markerSvg(p, xRaw, yRaw, color, stroke, tip, detail){const x=Number(xRaw),y=Number(yRaw);const type=p.entity_type||'paragraph';const size=pointSize(p);const attrs=`class="scatter-point" tabindex="0" fill="${color}" stroke="${stroke}" stroke-width="1.2" opacity=".86" aria-label="${esc(tip)}" data-tooltip="${esc(tip)}" data-detail="${esc(detail)}"`;const title=`<title>${esc(tip)}</title>`;if(type==='keyword'){return `<polygon ${attrs} points="${x},${y-size} ${x+size},${y} ${x},${y+size} ${x-size},${y}">${title}</polygon>`;}if(type==='title'||type==='h1'){return `<polygon ${attrs} points="${x},${y-size} ${x+size},${y+size*.85} ${x-size},${y+size*.85}">${title}</polygon>`;}if(type==='header'){const s=size*1.7;return `<rect ${attrs} x="${x-s/2}" y="${y-s/2}" width="${s}" height="${s}" rx="1.5">${title}</rect>`;}return `<circle ${attrs} cx="${x}" cy="${y}" r="${size}">${title}</circle>`;}
function scatterSvg(points){if(!points||!points.length)return '<div class="empty">No scatter data available for this keyword.</div>';const w=820,h=390,pad=26;const xs=points.map(p=>+p.x),ys=points.map(p=>+p.y);let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);if(minX===maxX){minX-=1;maxX+=1}if(minY===maxY){minY-=1;maxY+=1}const sx=x=>pad+(x-minX)/(maxX-minX)*(w-pad*2);const sy=y=>h-pad-(y-minY)/(maxY-minY)*(h-pad*2);const marks=points.map(p=>{const x=sx(+p.x).toFixed(1),y=sy(+p.y).toFixed(1);const stroke=p.source==='ours'?'#0b3d1e':'#fff';const tip=pointTooltip(p);const detail=JSON.stringify(pointDetail(p));return markerSvg(p,x,y,sourceColor(p),stroke,tip,detail);}).join('');const competitorDomains=[...new Set(points.filter(p=>p.source==='competitor'&&p.domain).map(p=>p.domain))].slice(0,8);const domainLegend=competitorDomains.map(d=>`<span><i class="dot" style="background:${domainColor(d)}"></i>${esc(d)}</span>`).join('');return `<div class="scatter-wrap"><div class="scatter-controls" aria-label="Scatterplot zoom controls"><button type="button" data-zoom="in" title="Zoom in">+</button><button type="button" data-zoom="out" title="Zoom out">−</button><button type="button" data-zoom="reset" title="Reset zoom">Reset</button></div><svg class="scatter" viewBox="0 0 ${w} ${h}" data-base-viewbox="0 0 ${w} ${h}" role="img" aria-label="Semantic scatterplot. Wheel to zoom, drag to pan, double-click to reset, click or focus dots for point explanations."><rect x="0" y="0" width="${w}" height="${h}" fill="#fbfcfe"/><line x1="${pad}" x2="${w-pad}" y1="${h-pad}" y2="${h-pad}" stroke="#d7dee8"/><line x1="${pad}" x2="${pad}" y1="${pad}" y2="${h-pad}" stroke="#d7dee8"/>${marks}</svg><div class="scatter-tooltip" role="dialog" aria-live="polite" aria-label="Scatter point details"></div></div><div class="legend"><span><i class="dot" style="background:${colors.keyword};transform:rotate(45deg)"></i>keyword diamond</span><span>▲ title/H1</span><span>■ headers</span><span><i class="dot" style="background:${colors.ours}"></i>our content</span>${domainLegend}<span class="muted">Wheel to zoom, drag to pan, double-click to reset, click a dot for details.</span></div>`;}
function topicRows(topics, limit=12){if(!topics||!topics.length)return '<tr><td colspan="6" class="muted">No topics classified.</td></tr>';return topics.slice(0,limit).map(t=>{const ex=(t.examples||[])[0]||{};return `<tr><td class="coverage-${esc(t.coverage)}">${esc(t.coverage)}</td><td>${esc(t.priority)}</td><td><div class="topic-label">${esc(t.label)}</div><div class="mini">${esc(ex.paragraph||'')}</div></td><td>${esc(t.competitor_coverage)}/${esc(t.competitor_urls?.length||'')}</td><td>${esc(t.our_best_similarity)}</td><td>${esc(ex.url||'')}</td></tr>`}).join('');}
function clusterCards(points){const clusters=clusterSummary(points);if(!clusters.length)return '<div class="empty">No semantic clusters available.</div>';return '<div class="cluster-list">'+clusters.map(c=>`<div class="cluster"><strong>Cluster ${esc(c.id)} · ${n(c.total)} points</strong><div class="mini">Own ${n(c.ours)} · Competitor ${n(c.competitor)} · Headers ${n(c.headers)} · Keywords ${n(c.keyword)}</div><div class="bar"><span style="width:${Math.min(100,Math.round((c.competitor/Math.max(c.total,1))*100))}%"></span></div><div class="mini">${esc(c.samples.join(' / '))}</div></div>`).join('')+'</div>';}
function competitorList(rows){if(!rows||!rows.length)return '<div class="empty">No competitors fetched.</div>';return '<ol class="competitors">'+rows.map(c=>`<li><a href="${esc(c.url)}">${esc(c.title||c.url)}</a><div class="mini">Rank ${esc(c.rank||'')} · Paragraphs ${esc(c.paragraph_count||0)}${c.error?' · '+esc(c.error):''}</div></li>`).join('')+'</ol>';}
function reviewList(rows){if(!rows||!rows.length)return '<div class="empty">No own paragraphs were far from the SERP topic space.</div>';return '<ul class="review">'+rows.map(p=>`<li><b>${esc(p.similarity_to_serp_topics)}</b> <span class="mini">similarity</span><br>${esc(p.paragraph)}</li>`).join('')+'</ul>';}
function keywordId(pageIndex, keywordIndex){return `keyword-${pageIndex}-${keywordIndex}`;}
function keywordCard(a, pageIndex, keywordIndex){const s=a.summary||{};const points=a.scatter?.points||[];return `<div class="keyword-card" id="${keywordId(pageIndex, keywordIndex)}"><div class="keyword-head"><div><h3>${esc(a.query || a.keyword?.keyword || '')}</h3><div class="mini">Status ${esc(a.status)} · Competitors ${esc(a.competitors||a.competitor_pages?.length||0)} · Scatter points ${esc(a.scatter?.shown||0)}</div></div><div class="chips"><span class="chip missing">Missing ${n(s.missing||0)}</span><span class="chip partial">Partial ${n(s.partial||0)}</span><span class="chip covered">Covered ${n(s.covered||0)}</span></div></div><div class="keyword-grid"><div class="panel"><h4>Semantic Scatterplot</h4><div class="panel-body">${scatterSvg(points)}</div></div><div class="tables"><div class="panel"><h4>Semantic Clusters</h4><div class="panel-body">${clusterCards(points)}</div></div><div class="panel"><h4>Competitor SERP</h4><div class="panel-body">${competitorList(a.competitor_pages)}</div></div></div></div><div class="two-col" style="margin-top:14px"><div class="panel"><h4>Topic Relations</h4><table><thead><tr><th>Coverage</th><th>Priority</th><th>Topic and Example</th><th>Seen</th><th>Own sim</th><th>Example URL</th></tr></thead><tbody>${topicRows(a.topics,18)}</tbody></table></div><div class="panel"><h4>Own Paragraphs To Review</h4><div class="panel-body">${reviewList(a.off_intent_paragraphs)}</div></div></div></div>`;}
function pageSection(page, index){return `<section class="page-section" id="page-${index}"><div class="page-head"><h2>${index+1}. ${esc(page.title || page.url)}</h2><div class="url">${esc(page.url)}</div>${page.h1?`<div class="mini">H1: ${esc(page.h1)}</div>`:''}</div>${(page.analyses||[]).map((analysis, keywordIndex)=>keywordCard(analysis, index, keywordIndex)).join('') || '<div class="empty">No keyword analyses for this page.</div>'}</section>`;}
function buildNav(){if(!navEl)return;navEl.innerHTML=(data.pages||[]).map((page,pageIndex)=>`<button type="button" class="report-nav-button" data-target="page-${pageIndex}"><span class="report-nav-label">${esc(page.title||page.url||`Page ${pageIndex+1}`)}</span></button>${(page.analyses||[]).map((analysis,keywordIndex)=>`<button type="button" class="report-nav-button nav-keyword" data-target="${keywordId(pageIndex,keywordIndex)}"><span class="report-nav-label">${esc(analysis.query||analysis.keyword?.keyword||`Keyword ${keywordIndex+1}`)}</span></button>`).join('')}`).join('');const buttons=[...navEl.querySelectorAll('.report-nav-button')];const sections=buttons.map(button=>document.getElementById(button.dataset.target||'')).filter(Boolean);buttons.forEach(button=>button.addEventListener('click',()=>{const target=document.getElementById(button.dataset.target||'');if(target)target.scrollIntoView({block:'start',behavior:'smooth'});}));function update(){let active=0;for(let i=0;i<sections.length;i++){if(sections[i].getBoundingClientRect().top<160)active=i;}buttons.forEach((button,i)=>{const selected=i===active;button.classList.toggle('is-active',selected);button.setAttribute('aria-current',selected?'page':'false');});}document.addEventListener('scroll',update,{passive:true});update();}

function bindScatterInteractions(){document.querySelectorAll('.scatter-wrap').forEach(wrap=>{const svg=wrap.querySelector('svg.scatter');const tooltip=wrap.querySelector('.scatter-tooltip');if(!svg||!tooltip)return;const base=(svg.getAttribute('data-base-viewbox')||'0 0 820 390').split(/\\s+/).map(Number);let vb={x:base[0],y:base[1],w:base[2],h:base[3]};const setVb=()=>svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);function zoomAt(factor,cx=base[2]/2,cy=base[3]/2){const nx=cx-(cx-vb.x)*factor;const ny=cy-(cy-vb.y)*factor;vb={x:nx,y:ny,w:vb.w*factor,h:vb.h*factor};setVb();}function pointFromEvent(event){const rect=svg.getBoundingClientRect();return{x:vb.x+(event.clientX-rect.left)/Math.max(rect.width,1)*vb.w,y:vb.y+(event.clientY-rect.top)/Math.max(rect.height,1)*vb.h};}function show(point){let detail=null;try{detail=JSON.parse(point.getAttribute('data-detail')||'{}');}catch(_){detail={type:'point',explanation:point.getAttribute('data-tooltip')||point.getAttribute('aria-label')||''};}tooltip.innerHTML=pointDetailHtml(detail);tooltip.classList.add('open');tooltip.querySelector('.tip-close')?.addEventListener('click',event=>{event.stopPropagation();tooltip.classList.remove('open');});}wrap.querySelectorAll('.scatter-point').forEach(point=>{point.addEventListener('click',event=>{event.stopPropagation();show(point);});point.addEventListener('focus',()=>show(point));});svg.addEventListener('wheel',event=>{event.preventDefault();const p=pointFromEvent(event);zoomAt(event.deltaY<0?0.82:1.22,p.x,p.y);},{passive:false});let drag=null;svg.addEventListener('mousedown',event=>{if(event.target.classList?.contains('scatter-point'))return;drag={x:event.clientX,y:event.clientY,vx:vb.x,vy:vb.y};svg.classList.add('is-panning');});window.addEventListener('mousemove',event=>{if(!drag)return;const rect=svg.getBoundingClientRect();vb.x=drag.vx-(event.clientX-drag.x)/Math.max(rect.width,1)*vb.w;vb.y=drag.vy-(event.clientY-drag.y)/Math.max(rect.height,1)*vb.h;setVb();});window.addEventListener('mouseup',()=>{drag=null;svg.classList.remove('is-panning');});svg.addEventListener('dblclick',()=>{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();});wrap.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',event=>{event.stopPropagation();const action=button.getAttribute('data-zoom');if(action==='in')zoomAt(0.78);else if(action==='out')zoomAt(1.28);else{vb={x:base[0],y:base[1],w:base[2],h:base[3]};setVb();}}));});document.addEventListener('click',event=>{const target=event.target;if(target?.closest?.('.scatter-tooltip')||target?.closest?.('.scatter-point'))return;document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.scatter-tooltip.open').forEach(t=>t.classList.remove('open'));});}
app.innerHTML = (data.pages||[]).map(pageSection).join('') || '<div class="empty">No analyzed pages in this report.</div>';
buildNav();
bindScatterInteractions();
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
