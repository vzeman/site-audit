"""Top-level orchestration: domain → cached pages → embeddings → reports.

Each domain becomes a "project" on disk, so cache + report stay
co-located:

::

    projects/<slug>/
      cache/
        http.sqlite
        embeddings_<model_slug>.npz
      report/
        index.html              ← self-contained viewer
        site_metrics.json
        section_report.json
        ... (CSV/JSON outputs)

Library users can also call ``run`` directly with a ``PipelineConfig``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import collections
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import numpy as np
from bs4 import BeautifulSoup

from .analyzer import PageInfo, analyze, deduplicate_pages_by_url, section_for_url
from .ahrefs import AhrefsConfig, build_analysis as build_ahrefs_analysis
from .ahrefs import fetch_snapshot as fetch_ahrefs_snapshot
from .ahrefs import write_semantic_cache as write_ahrefs_semantic_cache
from .anchor_relevance import build_anchor_relevance
from .answer_blocks import build_answer_blocks
from .dataforseo import DataForSEOConfig, build_analysis as build_dataforseo_analysis
from .dataforseo import fetch_snapshot as fetch_dataforseo_snapshot
from .answerability import score_all as score_answerability
from .answerability import to_payload as answerability_payload
from .best_pages import build_best_page_explainers
from .cache import EmbeddingCache, HttpCache, ParagraphEmbeddingCache, content_hash, domain_slug
from .cannibalization import build_cannibalization
from .canonical_consistency import analyze as analyze_canonical_consistency
from .duplicate_fragments import build_duplicate_fragments
from .cluster_labels import cluster_overlap_matrix, label_clusters
from .competitive_analysis import (
    CompetitiveAutoConfig,
    build_auto_competitive_targets,
    compare_serp_targets,
    load_competitive_targets,
)
from .content_quality import (
    improvement_payload,
    per_page_improvement,
    title_mismatch,
    title_mismatch_payload,
    wrong_home_paragraphs,
    wrong_home_payload,
)
from .contextual_links import build_contextual_link_impact
from .conversion import analyze as analyze_conversion
from .conversion import to_payload as conversion_payload
from .conversion_balance import build_conversion_balance
from .paragraph_clustering import (
    cluster_and_label as cluster_paragraphs,
    project_paragraphs,
    to_overlap_payload as paragraph_cluster_overlap_payload,
    to_scatter_payload as paragraph_scatter_payload,
    to_summary_payload as paragraph_clusters_payload,
)
from .crawler import Crawler, CrawlConfig, DEFAULT_EXCLUDE_PATTERNS
from .embedder import DEFAULT_EMBED_MAX_SEQ_LENGTH, DEFAULT_MODEL, EmbedInput, Embedder
from .entities import analyze as analyze_entities
from .entities import to_payload as entities_payload
from .entity_coverage import build_entity_coverage
from .external_links import analyze as analyze_external
from .external_links import to_payload as external_payload
from .extraction_cache import ExtractionCache
from .extractor import ExtractedPage, extract
from .freshness import analyze as analyze_freshness
from .freshness import to_payload as freshness_payload
from .freshness_impact import build_freshness_impact
from .gsc import GSCConfig, build_analysis as build_gsc_analysis
from .gsc import fetch_snapshot as fetch_gsc_snapshot
from .google_ads import GoogleAdsConfig, build_analysis as build_google_ads_analysis
from .google_ads import fetch_snapshot as fetch_google_ads_snapshot
from .header_analysis import analyse as analyse_headers
from .header_analysis import headers_for_scatter
from .heading_impact import build_heading_impact
from .history import build_history_snapshot, save_report_snapshot
from .internal_link_patterns import build_internal_link_patterns
from .linkbuilding import analyse as analyse_linkbuilding
from .html_report import write_html_report
from .indexability import analyze as analyze_indexability
from .indexability import to_payload as indexability_payload
from .information_gain import build_information_gain
from .keyword_coverage import (
    auto_mine_queries, load_queries_from_file, match_queries,
    match_queries_to_paragraphs, paragraph_match_payload,
    to_payload as queries_payload,
)
from .keyword_attribution import build_keyword_attribution
from .linkgraph import analyze as analyze_linkgraph
from .linkgraph import annotate_internal_link_rel_stats
from .linkgraph import annotate_link_target_statuses
from .linkgraph import high_demand_low_link_payload
from .linkgraph import hub_bottleneck_payload
from .linkgraph import link_removal_simulation_payload
from .linkgraph import link_flow_payload
from .linkgraph import to_payload as linkgraph_payload
from .linkgraph import traffic_weighted_pagerank_payload
from .media_accessibility import analyze as analyze_media_accessibility
from .media_accessibility import to_payload as media_accessibility_payload
from .metadata_quality import analyze as analyze_metadata_quality
from .metadata_quality import to_payload as metadata_quality_payload
from .page_types import analyze as analyze_page_types
from .page_types import to_payload as page_types_payload
from .paragraph_density import compute_rows as compute_paragraph_density_rows
from .paragraph_density import density_lookup as paragraph_density_lookup
from .paragraph_density import to_payload as paragraph_density_payload
from .paragraph_impact import build_paragraph_impact
from .paragraph_links import build_addition_simulation as build_link_addition_simulation
from .paragraph_links import recommend as recommend_paragraph_links
from .paragraph_links import to_payload as paragraph_links_payload
from .performance import analyze as analyze_performance
from .performance import to_payload as performance_payload
from .performance_explainer import build_performance_explainer
from .recommendations import synthesize as synthesize_recommendations
from .recommendations import to_payload as recommendations_payload
from .report import build_duplicate_rows, build_outlier_rows, write_all, write_technical_audit_bundle
from .resource_status import analyze as analyze_resource_status
from .resource_status import to_payload as resource_status_payload
from .scatter import project
from .semantic_ablation import build_semantic_ablation
from .search_fusion import build_combined_search_analysis
from .structured_data import analyze as analyze_structured_data
from .structured_data import to_payload as structured_data_payload
from .sitemap_coverage import analyze as analyze_sitemap_coverage
from .template_patterns import build_template_patterns
from .technical_seo import build_technical_seo
from .trust_signals import build_trust_signals
from .weak_paragraphs import build_weak_paragraphs
from .winning_paragraphs import build_winning_paragraphs

LOG = logging.getLogger(__name__)
DEFAULT_EMBED_BODY_CHARS = 12000


def build_embed_text(
    title: str,
    description: str,
    body: str,
    *,
    body_char_limit: int = DEFAULT_EMBED_BODY_CHARS,
) -> str:
    """Build bounded text for page-level embeddings.

    Sentence-transformer models truncate to their token window. Capping large
    page bodies before tokenization avoids spending minutes on text the model
    cannot use.
    """
    body_text = (body or "").strip()
    if body_char_limit > 0 and len(body_text) > body_char_limit:
        body_text = body_text[:body_char_limit]
    return ". ".join(
        part.strip()
        for part in [title or "", description or "", body_text]
        if part and part.strip()
    )


def _configured_embed_body_chars(default: int) -> int:
    raw = os.environ.get("SITE_AUDIT_EMBED_BODY_CHARS")
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning(
            "Ignoring invalid SITE_AUDIT_EMBED_BODY_CHARS=%r; using %d",
            raw,
            default,
        )
        return default


def _configured_embed_max_seq_length(default: int) -> int:
    raw = os.environ.get("SITE_AUDIT_EMBED_MAX_SEQ_LENGTH")
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning(
            "Ignoring invalid SITE_AUDIT_EMBED_MAX_SEQ_LENGTH=%r; using %d",
            raw,
            default,
        )
        return default


def _tag_has_any_class(tag, classes: set[str]) -> bool:
    values = tag.get("class") or []
    if isinstance(values, str):
        values = values.split()
    return any(value in classes for value in values)


def _prepare_extraction_body(
    body: str,
    *,
    strip_header_footer: bool,
    include_classes: list[str],
    exclude_classes: list[str],
) -> str:
    if not (strip_header_footer or include_classes or exclude_classes) or not body:
        return body
    try:
        soup = BeautifulSoup(body, "html.parser")
    except Exception:
        return body
    if strip_header_footer:
        for tag in soup.find_all(["header", "footer"]):
            tag.decompose()
    if exclude_classes:
        excluded = set(exclude_classes)
        for tag in soup.find_all(True):
            if _tag_has_any_class(tag, excluded):
                tag.decompose()
    if include_classes:
        included = set(include_classes)
        selected = []
        selected_ids = set()
        for tag in soup.find_all(True):
            if not _tag_has_any_class(tag, included):
                continue
            if any(id(parent) in selected_ids for parent in tag.parents):
                continue
            selected.append(tag)
            selected_ids.add(id(tag))
        scoped = BeautifulSoup("<html><body></body></html>", "html.parser")
        if soup.head:
            scoped.html.insert(0, soup.head.extract())
        target = scoped.body or scoped
        for tag in selected:
            target.append(tag.extract())
        soup = scoped
    return str(soup)


def _read_first_body(body_paths: list[str]) -> str:
    for path in body_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text:
            return text
    return ""


def _extract_page_process_worker(payload: dict) -> tuple[int, ExtractedPage | None, dict]:
    idx = int(payload["idx"])
    body = payload.get("body") or _read_first_body(payload.get("body_paths") or [])
    if not body:
        return idx, None, {"misses": 1}
    body = _prepare_extraction_body(
        body,
        strip_header_footer=bool(payload.get("strip_header_footer")),
        include_classes=list(payload.get("content_include_classes") or []),
        exclude_classes=list(payload.get("content_exclude_classes") or []),
    )
    if not body:
        return idx, None, {"misses": 1}
    url = payload["url"]
    max_chars = int(payload.get("max_chars") or 0)
    x_robots_tag = payload.get("x_robots_tag") or ""
    cache = ExtractionCache(Path(payload["extraction_cache_dir"]))
    ext = cache.get(url, body, max_chars=max_chars, x_robots_tag=x_robots_tag)
    if ext is not None:
        return idx, ext, {"hits": 1}
    ext = extract(url, body, max_chars=max_chars, x_robots_tag=x_robots_tag)
    if ext is not None:
        cache.put(url, body, ext, max_chars=max_chars, x_robots_tag=x_robots_tag)
        return idx, ext, {"misses": 1, "writes": 1}
    return idx, None, {"misses": 1}


class StageTimings:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    @contextmanager
    def track(self, name: str, **meta):
        started = time.perf_counter()
        LOG.info("  stage start: %s", name)
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            row = {"stage": name, "status": status, "seconds": round(elapsed, 3)}
            row.update({k: v for k, v in meta.items() if v is not None})
            self.rows.append(row)
            LOG.info("  stage done: %s in %.1fs (%s)", name, elapsed, status)

    def write(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "stage_timings.json").write_text(
            json.dumps({"stages": self.rows}, indent=2),
            encoding="utf-8",
        )


def _basic_linkgraph_payload(
    pages: list[PageInfo],
    outlinks_map: dict[str, list[tuple[str, str]]],
    indexability: dict,
) -> dict:
    urls = {p.url for p in pages}
    status_by_url: dict[str, int] = {}
    for row in (
        list((indexability or {}).get("per_page") or [])
        + list((indexability or {}).get("skipped") or [])
        + list((indexability or {}).get("noindex_pages") or [])
    ):
        url = row.get("url")
        if url:
            status_by_url[url] = int(row.get("http_status") or row.get("status_code") or 0)

    incoming: dict[str, set[str]] = {url: set() for url in urls}
    rows: list[dict] = []
    edge_count = 0
    for page in pages:
        links = [(link, anchor) for link, anchor in outlinks_map.get(page.url, []) if link in urls]
        edge_count += len(links)
        for link, _ in links:
            incoming.setdefault(link, set()).add(page.url)
        http_links = [link for link, _ in links if urlparse(link).scheme.lower() == "http"]
        https_links = [link for link, _ in links if urlparse(link).scheme.lower() == "https"]
        broken_links = [link for link, _ in links if status_by_url.get(link, 0) >= 400]
        redirect_links = [link for link, _ in links if 300 <= status_by_url.get(link, 0) < 400]
        rows.append({
            "url": page.url,
            "in_degree": 0,
            "out_degree": len(links),
            "click_depth": "",
            "raw_internal_link_count": len(links),
            "internal_http_link_count": len(http_links),
            "internal_http_links": http_links[:50],
            "internal_https_link_count": len(https_links),
            "internal_https_links": https_links[:50],
            "broken_internal_link_count": len(broken_links),
            "broken_internal_links": broken_links[:50],
            "redirect_internal_link_count": len(redirect_links),
            "redirect_internal_links": redirect_links[:50],
            "incoming_nofollow_internal_link_count": 0,
            "incoming_dofollow_internal_link_count": 0,
            "incoming_nofollow_internal_links": [],
            "incoming_dofollow_internal_links": [],
            "outgoing_nofollow_internal_link_count": 0,
            "outgoing_nofollow_internal_links": [],
        })

    row_by_url = {row["url"]: row for row in rows}
    for url, sources in incoming.items():
        row = row_by_url.get(url)
        if row is None:
            continue
        row["in_degree"] = len(sources)
        row["incoming_dofollow_internal_link_count"] = len(sources)
        row["incoming_dofollow_internal_links"] = sorted(sources)[:50]

    orphan_count = sum(1 for row in rows if int(row.get("in_degree") or 0) == 0)
    return {
        "edge_count": edge_count,
        "orphan_count": orphan_count,
        "max_click_depth": 0,
        "page_link_counts": rows,
        "recommendations": [],
    }


def _canonical_dedupe_key(url: str) -> str:
    url, _ = urldefrag(url or "")
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return (url or "").rstrip("/")
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), netloc, path, "", parsed.query, ""))


def _absolute_canonical(page_url: str, canonical_url: str) -> str:
    canonical_url = (canonical_url or "").strip()
    if not canonical_url:
        return ""
    return urljoin(page_url, canonical_url)


def filter_to_unique_canonical_pages(
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    embed_inputs: list[EmbedInput],
    extraction_rows: list[dict],
) -> tuple[list[PageInfo], list[ExtractedPage], list[EmbedInput], int]:
    """Keep one analyzed row per HTML canonical URL.

    Pages without a canonical tag stay in the corpus so missing-canonical pages
    remain diagnosable. Pages that explicitly canonicalize to a different URL
    are removed from embeddings/content analysis and marked in extraction rows.
    """
    if not pages:
        return pages, extracted_pages, embed_inputs, 0

    row_by_url = {
        row.get("url"): row
        for row in extraction_rows
        if row.get("status") == "analyzed" and row.get("url")
    }
    groups: dict[str, list[int]] = {}
    self_canonical_indices: set[int] = set()
    keep_missing_canonical: set[int] = set()

    for idx, (page, ext) in enumerate(zip(pages, extracted_pages)):
        canonical = _absolute_canonical(page.url, ext.canonical_url)
        page_key = _canonical_dedupe_key(page.url)
        if not canonical:
            key = f"url:{page_key}"
            keep_missing_canonical.add(idx)
        else:
            canonical_key = _canonical_dedupe_key(canonical)
            key = f"canonical:{canonical_key}"
            if page_key == canonical_key:
                self_canonical_indices.add(idx)
        groups.setdefault(key, []).append(idx)

    keep: set[int] = set(keep_missing_canonical)
    canonical_kept_by_group: dict[str, int] = {}
    for key, indices in groups.items():
        if key.startswith("url:"):
            keep.add(indices[0])
            canonical_kept_by_group[key] = indices[0]
            continue
        self_indices = [idx for idx in indices if idx in self_canonical_indices]
        if self_indices:
            keep_idx = self_indices[0]
            keep.add(keep_idx)
            canonical_kept_by_group[key] = keep_idx

    dropped = 0
    for key, indices in groups.items():
        kept_idx = canonical_kept_by_group.get(key)
        kept_url = pages[kept_idx].url if kept_idx is not None else ""
        for idx in indices:
            if idx in keep:
                continue
            dropped += 1
            row = row_by_url.get(pages[idx].url)
            if not row:
                continue
            row["status"] = "skipped"
            row["reason"] = "canonical_duplicate"
            row["canonical_kept_url"] = kept_url
            row["canonical_target_normalized"] = key.split(":", 1)[1]

    if not dropped:
        return pages, extracted_pages, embed_inputs, 0

    kept_indices = [idx for idx in range(len(pages)) if idx in keep]
    return (
        [pages[idx] for idx in kept_indices],
        [extracted_pages[idx] for idx in kept_indices],
        [embed_inputs[idx] for idx in kept_indices],
        dropped,
    )


@dataclass
class PipelineConfig:
    domain: str
    projects_root: Path = Path("projects")
    # Optional overrides — if set, take precedence over projects_root.
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    model: str = DEFAULT_MODEL
    max_pages: int = 10000
    max_workers: int = 8
    link_parse_processes: int = 0
    extraction_workers: int = 0
    analysis_workers: int = 0
    adaptive_concurrency: bool = True
    min_crawl_workers: int = 1
    adaptive_success_threshold: int = 50
    adaptive_slow_seconds: float = 3.0
    adaptive_max_rss_mb: int = 0
    resume: bool = False
    write_checkpoints: bool = True
    request_delay: float = 0.0
    duplicate_threshold: float = 0.92
    duplicate_knn: int = 10
    scatter_clusters: int = 30
    follow_subdomains: bool = False
    respect_robots: bool = True
    use_http_cache: bool = True
    use_embedding_cache: bool = True
    crawl_discovered_links: bool = True
    strip_header_footer: bool = False
    content_include_classes: list[str] = field(default_factory=list)
    content_exclude_classes: list[str] = field(default_factory=list)
    url_include_patterns: list[str] = field(default_factory=list)
    url_exclude_patterns: list[str] = field(default_factory=list)
    sitemap_urls: list[str] = field(default_factory=list)
    sitemap_include_patterns: list[str] = field(default_factory=list)
    sitemap_exclude_patterns: list[str] = field(default_factory=list)
    sitemap_lastmod_after: Optional[str] = None
    sitemap_lastmod_within_days: Optional[int] = None
    skip_scatterplot: bool = False
    max_chars: int = 4000
    embed_body_chars: int = DEFAULT_EMBED_BODY_CHARS
    embed_max_seq_length: int = DEFAULT_EMBED_MAX_SEQ_LENGTH
    audit_preset: str = "standard"
    technical_only: bool = False
    allow_large_embeddings: bool = False
    large_site_embedding_threshold: int = 20000

    # New analyses
    enable_cluster_labels: bool = True
    enable_keyword_coverage: bool = True
    enable_answerability: bool = True
    enable_answer_blocks: bool = True
    enable_freshness_impact: bool = True
    enable_cannibalization: bool = True
    enable_duplicate_fragments: bool = True
    enable_template_patterns: bool = True
    enable_trust_signals: bool = True
    enable_conversion_balance: bool = True
    enable_linkgraph: bool = True
    enable_external_links: bool = True
    enable_paragraph_links: bool = True
    enable_paragraph_clustering: bool = True
    enable_paragraph_impact: bool = True
    enable_semantic_ablation: bool = True
    enable_keyword_attribution: bool = True
    enable_weak_paragraphs: bool = True
    enable_heading_impact: bool = True
    enable_entity_coverage: bool = True
    enable_information_gain: bool = True
    enable_content_quality: bool = True
    enable_paragraph_fanout: bool = True
    check_external_links: bool = False     # opt-in HEAD requests
    paragraph_link_top_k: int = 200
    paragraph_link_per_page: int = 8
    paragraph_similarity_floor: float = 0.65
    paragraph_lift_floor: float = 0.05
    paragraph_scatter_sample: int = 5000
    paragraph_num_clusters: int = 60
    competitive_pairs_file: Optional[Path] = None
    competitive_auto: bool = False
    competitive_auto_clusters: int = 3
    competitive_auto_keywords_per_cluster: int = 1
    competitive_auto_results_per_keyword: int = 5
    competitive_auto_min_relevance: float = 0.35
    competitive_auto_min_position: int = 2
    competitive_auto_max_position: int = 20
    competitive_auto_product_seeds: list[str] = field(default_factory=list)
    competitive_auto_allow_nonlatin: bool = False
    competitive_auto_refresh_serp: bool = False
    queries_file: Optional[Path] = None
    auto_queries_max: int = 200
    coverage_threshold: float = 0.55
    cannibalization_threshold: float = 0.72
    link_similarity_threshold: float = 0.85
    link_recommendations_top_k: int = 75
    search_provider: str = "auto"
    save_snapshot: bool = True
    enable_gsc: bool = True
    enable_google_ads: bool = True
    use_google_ads_keywords: bool = False
    gsc_property_url: Optional[str] = None
    gsc_start_date: Optional[str] = None
    gsc_end_date: Optional[str] = None
    gsc_top_pages_limit: int = 1000
    gsc_keywords_limit: int = 1000
    gsc_refresh: bool = False
    google_ads_customer_id: Optional[str] = None
    google_ads_login_customer_id: Optional[str] = None
    google_ads_start_date: Optional[str] = None
    google_ads_end_date: Optional[str] = None
    google_ads_search_terms_limit: int = 1000
    google_ads_min_cost: float = 0.0
    google_ads_refresh: bool = False
    enable_dataforseo: bool = True
    enable_ahrefs: bool = True
    ahrefs_date: Optional[str] = None
    ahrefs_country: Optional[str] = None
    ahrefs_mode: str = "subdomains"
    ahrefs_top_pages_limit: int = 1000
    ahrefs_keywords_limit: int = 1000
    ahrefs_refresh: bool = False
    ahrefs_semantic_sample: int = 500
    dataforseo_location_code: Optional[int] = None
    dataforseo_location_name: Optional[str] = None
    dataforseo_language_code: Optional[str] = None
    dataforseo_language_name: Optional[str] = None
    dataforseo_top_pages_limit: int = 1000
    dataforseo_keywords_limit: int = 1000
    dataforseo_refresh: bool = False
    dataforseo_include_clickstream: bool = False


def _domain_only(domain: str) -> str:
    if "://" in domain:
        host = urlparse(domain).netloc
    else:
        host = domain
    return host.split("/")[0].lower()


def project_paths(config: PipelineConfig) -> tuple[Path, Path]:
    """Return (cache_dir, report_dir) for this run."""
    host = _domain_only(config.domain)
    slug = domain_slug(host)
    project_dir = Path(config.projects_root) / slug
    cache_dir = Path(config.cache_dir) if config.cache_dir else project_dir / "cache"
    report_dir = Path(config.output_dir) if config.output_dir else project_dir / "report"
    return cache_dir, report_dir


def _checkpoint_path(cache_dir: Path, name: str) -> Path:
    return Path(cache_dir) / "checkpoints" / f"{name}.json"


def _write_checkpoint(cache_dir: Path, name: str, payload: dict) -> None:
    path = _checkpoint_path(cache_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(path)


def _read_checkpoint(cache_dir: Path, name: str) -> dict | None:
    path = _checkpoint_path(cache_dir, name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extraction_checkpoint_payload(
    *,
    domain: str,
    max_chars: int,
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    embed_inputs: list[EmbedInput],
    outlinks_map: dict[str, list[tuple[str, str]]],
    external_map: dict[str, list[tuple[str, str]]],
    extraction_rows: list[dict],
    noindex_dropped: int,
    canonical_dropped: int,
) -> dict:
    return {
        "version": 1,
        "domain": domain,
        "max_chars": max_chars,
        "pages": [asdict(page) for page in pages],
        "extracted_pages": [asdict(page) for page in extracted_pages],
        "embed_inputs": [asdict(item) for item in embed_inputs],
        "outlinks_map": {url: [list(link) for link in links] for url, links in outlinks_map.items()},
        "external_map": {url: [list(link) for link in links] for url, links in external_map.items()},
        "extraction_rows": extraction_rows,
        "noindex_dropped": noindex_dropped,
        "canonical_dropped": canonical_dropped,
        "created_at": time.time(),
    }


def _load_extraction_checkpoint(cache_dir: Path, *, domain: str, max_chars: int) -> dict | None:
    payload = _read_checkpoint(cache_dir, "extraction")
    if not payload:
        return None
    if payload.get("version") != 1:
        return None
    if payload.get("domain") != domain or int(payload.get("max_chars") or 0) != int(max_chars):
        return None
    try:
        return {
            "pages": [PageInfo(**row) for row in payload.get("pages") or []],
            "extracted_pages": [ExtractedPage(**row) for row in payload.get("extracted_pages") or []],
            "embed_inputs": [EmbedInput(**row) for row in payload.get("embed_inputs") or []],
            "outlinks_map": {
                url: [tuple(link) for link in links]
                for url, links in (payload.get("outlinks_map") or {}).items()
            },
            "external_map": {
                url: [tuple(link) for link in links]
                for url, links in (payload.get("external_map") or {}).items()
            },
            "extraction_rows": list(payload.get("extraction_rows") or []),
            "noindex_dropped": int(payload.get("noindex_dropped") or 0),
            "canonical_dropped": int(payload.get("canonical_dropped") or 0),
        }
    except (TypeError, ValueError):
        return None


def _search_payload_usable(payload: dict) -> bool:
    if not payload:
        return False
    meta = payload.get("meta", {}) or {}
    if meta.get("status") != "ok":
        return False
    summary = payload.get("summary", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    return bool(
        summary.get("top_pages")
        or summary.get("organic_keywords")
        or metrics.get("org_traffic")
        or metrics.get("org_keywords")
    )


def run(config: PipelineConfig) -> dict:
    host = _domain_only(config.domain)
    cache_dir, report_dir = project_paths(config)
    stage_timings = StageTimings()
    embed_body_chars = _configured_embed_body_chars(config.embed_body_chars)
    embed_max_seq_length = _configured_embed_max_seq_length(config.embed_max_seq_length)

    LOG.info("=== site-audit run for %s ===", host)
    LOG.info("  cache:  %s", cache_dir)
    LOG.info("  report: %s", report_dir)
    LOG.info("  embed body chars: %d", embed_body_chars)
    LOG.info(
        "  embed max sequence length: %s",
        embed_max_seq_length if embed_max_seq_length > 0 else "model default",
    )

    http_cache = HttpCache(cache_dir / "http.sqlite")
    http_cache.clean_tracking_duplicates(min_candidates=100)

    # 1) Crawl
    crawl_config = CrawlConfig(
        domain=config.domain,
        max_pages=config.max_pages,
        max_workers=config.max_workers,
        link_parse_processes=config.link_parse_processes,
        adaptive_concurrency=config.adaptive_concurrency,
        min_crawl_workers=config.min_crawl_workers,
        adaptive_success_threshold=config.adaptive_success_threshold,
        adaptive_slow_seconds=config.adaptive_slow_seconds,
        adaptive_max_rss_mb=config.adaptive_max_rss_mb,
        request_delay=config.request_delay,
        follow_subdomains=config.follow_subdomains,
        respect_robots=config.respect_robots,
        use_cache=config.use_http_cache,
        crawl_discovered_links=config.crawl_discovered_links,
        strip_header_footer=config.strip_header_footer,
        content_include_classes=config.content_include_classes,
        content_exclude_classes=config.content_exclude_classes,
        include_patterns=config.url_include_patterns,
        exclude_patterns=list(DEFAULT_EXCLUDE_PATTERNS) + list(config.url_exclude_patterns),
        sitemap_urls=config.sitemap_urls,
        sitemap_include_patterns=config.sitemap_include_patterns,
        sitemap_exclude_patterns=config.sitemap_exclude_patterns,
        sitemap_lastmod_after=config.sitemap_lastmod_after,
        sitemap_lastmod_within_days=config.sitemap_lastmod_within_days,
    )
    crawler = Crawler(crawl_config, http_cache)
    with stage_timings.track("crawl", max_pages=config.max_pages, workers=config.max_workers):
        fetched = crawler.discover_and_crawl()
    LOG.info("  fetched %d pages (cache: %s)", len(fetched), http_cache.stats())

    # 2) Extract
    pages: list[PageInfo] = []
    extracted_pages = []  # list[ExtractedPage] in same order as `pages`
    embed_inputs: list[EmbedInput] = []
    outlinks_map: dict[str, list[tuple[str, str]]] = {}
    external_map: dict[str, list[tuple[str, str]]] = {}
    extraction_rows: list[dict] = []
    extraction_cache = ExtractionCache(cache_dir / "extracted_pages")

    noindex_dropped = 0
    canonical_dropped = 0
    fetched_total = len(fetched)
    loaded_extraction = _load_extraction_checkpoint(cache_dir, domain=host, max_chars=config.max_chars) if config.resume else None

    def _fetch_common_row(r) -> dict:
        return {
            "url": r.url,
            "http_status": getattr(r, "status", 0),
            "content_type": getattr(r, "content_type", ""),
            "x_robots_tag": getattr(r, "x_robots_tag", ""),
            "requested_url": getattr(r, "requested_url", ""),
            "redirect_target_url": getattr(r, "redirect_target_url", ""),
            "redirect_chain": list(getattr(r, "redirect_chain", []) or []),
            "redirect_hop_count": int(getattr(r, "redirect_hop_count", 0) or 0),
            "redirect_status_codes": list(getattr(r, "redirect_status_codes", []) or []),
        }

    def _skip_row(r, reason: str, ext: ExtractedPage | None = None) -> dict:
        row = _fetch_common_row(r)
        row.update({"status": "skipped", "reason": reason})
        if ext is not None:
            row.update({
                "title": ext.title,
                "canonical_url": ext.canonical_url,
                "robots_content": ext.robots_content,
                "noindex_source": ext.noindex_source,
                "nofollow": ext.nofollow,
                "nofollow_source": ext.nofollow_source,
                "meta_refresh_redirect": ext.meta_refresh_redirect,
                "meta_refresh_target_url": ext.meta_refresh_target_url,
                "title_tag_count": ext.title_tag_count,
                "meta_description_tag_count": ext.meta_description_tag_count,
                "language": ext.language or "",
                "word_count": ext.word_count,
            })
        return row

    def _body_candidate_urls(r) -> list[str]:
        return [
            getattr(r, "body_cache_url", "") or "",
            getattr(r, "requested_url", "") or "",
            getattr(r, "url", "") or "",
        ]

    def _body_for_fetch(r) -> str:
        body = getattr(r, "body", "") or ""
        if body:
            return body
        candidates = _body_candidate_urls(r)
        seen_candidates: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            cached = http_cache.get(candidate)
            if cached is None:
                continue
            text = cached.text
            if text:
                prepare_body = getattr(crawler, "_prepare_html_body", None)
                if callable(prepare_body):
                    return prepare_body(text)
                return text
        return ""

    def _extract_one(r) -> ExtractedPage | None:
        x_robots_tag = getattr(r, "x_robots_tag", "")
        body = _body_for_fetch(r)
        if not body:
            return None
        ext = extraction_cache.get(
            r.url,
            body,
            max_chars=config.max_chars,
            x_robots_tag=x_robots_tag,
        )
        if ext is not None:
            return ext
        ext = extract(r.url, body, max_chars=config.max_chars, x_robots_tag=x_robots_tag)
        if ext is not None:
            extraction_cache.put(
                r.url,
                body,
                ext,
                max_chars=config.max_chars,
                x_robots_tag=x_robots_tag,
            )
        return ext

    if loaded_extraction is not None:
        pages = loaded_extraction["pages"]
        extracted_pages = loaded_extraction["extracted_pages"]
        embed_inputs = loaded_extraction["embed_inputs"]
        outlinks_map = loaded_extraction["outlinks_map"]
        external_map = loaded_extraction["external_map"]
        extraction_rows = loaded_extraction["extraction_rows"]
        noindex_dropped = loaded_extraction["noindex_dropped"]
        canonical_dropped = loaded_extraction["canonical_dropped"]
        LOG.info("  resumed extraction checkpoint with %d usable pages", len(pages))
    else:
        with stage_timings.track("extraction", fetched_pages=fetched_total):
            extractable: list[tuple[int, object]] = []
            precomputed_rows: dict[int, dict] = {}
            for idx, r in enumerate(fetched, 1):
                http_status = int(getattr(r, "status", 0) or 0)
                fetch_error = getattr(r, "error", "") or ""
                if fetch_error:
                    row = _skip_row(r, fetch_error)
                    row["http_status"] = http_status
                    precomputed_rows[idx] = row
                    continue
                if http_status and not 200 <= http_status < 400:
                    row = _skip_row(r, "non_2xx_status")
                    row["http_status"] = http_status
                    precomputed_rows[idx] = row
                    continue
                extractable.append((idx, r))

            extraction_workers = max(1, int(config.extraction_workers or config.max_workers or 1))
            extraction_results: dict[int, ExtractedPage | None] = {}
            process_payloads = []
            file_backed_extractable = 0
            if extraction_workers > 1 and len(extractable) >= 1000:
                for idx, r in extractable:
                    seen_paths: set[str] = set()
                    body_paths: list[str] = []
                    for candidate in _body_candidate_urls(r):
                        if not candidate:
                            continue
                        path = http_cache.body_file_path(candidate)
                        path_str = str(path)
                        if path_str in seen_paths or not path.is_file():
                            continue
                        seen_paths.add(path_str)
                        body_paths.append(path_str)
                    if body_paths or getattr(r, "body", ""):
                        file_backed_extractable += 1
                    process_payloads.append({
                        "idx": idx,
                        "url": r.url,
                        "body": getattr(r, "body", "") or "",
                        "body_paths": body_paths,
                        "max_chars": config.max_chars,
                        "x_robots_tag": getattr(r, "x_robots_tag", "") or "",
                        "extraction_cache_dir": str(extraction_cache.cache_dir),
                        "strip_header_footer": config.strip_header_footer,
                        "content_include_classes": config.content_include_classes,
                        "content_exclude_classes": config.content_exclude_classes,
                    })
            if (
                extraction_workers > 1
                and len(extractable) >= 1000
                and file_backed_extractable >= max(1000, int(len(extractable) * 0.8))
            ):
                LOG.info("  extracting HTML with %d processes", extraction_workers)
                extraction_cache_counts: collections.Counter[str] = collections.Counter()
                with ProcessPoolExecutor(max_workers=extraction_workers) as pool:
                    for idx, ext, stats in pool.map(_extract_page_process_worker, process_payloads, chunksize=8):
                        extraction_results[idx] = ext
                        extraction_cache_counts.update(stats or {})
                extraction_cache.hits += extraction_cache_counts.get("hits", 0)
                extraction_cache.misses += extraction_cache_counts.get("misses", 0)
                extraction_cache.writes += extraction_cache_counts.get("writes", 0)
            elif extraction_workers > 1 and len(extractable) > 1:
                if len(extractable) >= 1000 and process_payloads:
                    LOG.info(
                        "  extracting HTML with %d workers (SQLite body fallback; %d/%d file-backed)",
                        extraction_workers,
                        file_backed_extractable,
                        len(extractable),
                    )
                else:
                    LOG.info("  extracting HTML with %d workers", extraction_workers)
                with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
                    extracted = pool.map(_extract_one, [r for _, r in extractable])
                    extraction_results = {
                        idx: ext for (idx, _), ext in zip(extractable, extracted)
                    }
            else:
                extraction_results = {idx: _extract_one(r) for idx, r in extractable}

            for idx, r in enumerate(fetched, 1):
                precomputed = precomputed_rows.get(idx)
                if precomputed is not None:
                    extraction_rows.append(precomputed)
                    continue
                ext = extraction_results.get(idx)
                if ext is None or not ext.title:
                    extraction_rows.append(_skip_row(r, "unusable"))
                    continue
                if ext.noindex:
                    # The page asked search engines not to index it — exclude from
                    # the analysis corpus. We still consumed its outlinks during
                    # the crawl (so internal links from a noindex landing page
                    # contribute to authority), but the page itself does not
                    # count toward focus / clusters / coverage / recommendations.
                    noindex_dropped += 1
                    row = _skip_row(r, "noindex", ext)
                    row["source"] = ext.noindex_source
                    extraction_rows.append(row)
                    continue
                section = section_for_url(r.url)
                embed_text = build_embed_text(
                    ext.title,
                    ext.description,
                    ext.body,
                    body_char_limit=embed_body_chars,
                )
                if not embed_text.strip():
                    extraction_rows.append(_skip_row(r, "empty_embedding_text", ext))
                    continue
                page = PageInfo(
                    url=r.url,
                    title=ext.title,
                    description=ext.description,
                    section=section,
                    word_count=ext.word_count,
                    language=ext.language,
                )
                pages.append(page)
                extracted_pages.append(ext)
                embed_inputs.append(EmbedInput(url=r.url, text=embed_text))
                outlinks_map[r.url] = r.outlinks or []
                external_map[r.url] = r.external_links or []
                row = _fetch_common_row(r)
                row.update({
                    "title": ext.title,
                    "status": "analyzed",
                    "reason": "",
                    "canonical_url": ext.canonical_url,
                    "robots_content": ext.robots_content,
                    "noindex_source": ext.noindex_source,
                    "nofollow": ext.nofollow,
                    "nofollow_source": ext.nofollow_source,
                    "meta_refresh_redirect": ext.meta_refresh_redirect,
                    "meta_refresh_target_url": ext.meta_refresh_target_url,
                    "title_tag_count": ext.title_tag_count,
                    "meta_description_tag_count": ext.meta_description_tag_count,
                    "language": ext.language or "",
                    "word_count": ext.word_count,
                })
                extraction_rows.append(row)
                if idx % 500 == 0 or idx == fetched_total:
                    LOG.info("  extracted %d / %d fetched pages (%d usable)", idx, fetched_total, len(pages))
    if noindex_dropped:
        LOG.info("  dropped %d noindex pages (meta robots / X-Robots-Tag)", noindex_dropped)
    extraction_cache_stats = extraction_cache.stats()
    LOG.info(
        "  extraction cache: %d hits · %d misses · %d writes",
        extraction_cache_stats.get("hits", 0),
        extraction_cache_stats.get("misses", 0),
        extraction_cache_stats.get("writes", 0),
    )

    if loaded_extraction is None:
        pages, extracted_pages, embed_inputs, canonical_dropped = filter_to_unique_canonical_pages(
            pages,
            extracted_pages,
            embed_inputs,
            extraction_rows,
        )
        if canonical_dropped:
            LOG.info(
                "  dropped %d non-canonical duplicate pages before embedding",
                canonical_dropped,
            )
        if config.write_checkpoints:
            _write_checkpoint(
                cache_dir,
                "extraction",
                _extraction_checkpoint_payload(
                    domain=host,
                    max_chars=config.max_chars,
                    pages=pages,
                    extracted_pages=extracted_pages,
                    embed_inputs=embed_inputs,
                    outlinks_map=outlinks_map,
                    external_map=external_map,
                    extraction_rows=extraction_rows,
                    noindex_dropped=noindex_dropped,
                    canonical_dropped=canonical_dropped,
                ),
            )
            LOG.info("  checkpoint: extraction state saved")

    if not pages:
        LOG.warning("No usable pages — aborting before embedding.")
        return {"pages": 0}

    LOG.info("  prepared %d pages for embedding", len(pages))

    with stage_timings.track("technical-foundation", pages=len(pages)):
        indexability_data = indexability_payload(
            analyze_indexability(fetched, extraction_rows, {p.url for p in pages})
        )
        sitemap_coverage_data = analyze_sitemap_coverage(
            getattr(crawler, "sitemap_entries", []),
            fetched,
            extraction_rows,
            indexability_data,
            sitemap_errors=getattr(crawler, "sitemap_errors", []),
        )
        canonical_consistency_data = analyze_canonical_consistency(
            extraction_rows,
            indexability_data,
        )
    ix_summary = indexability_data.get("summary", {}) or {}
    LOG.info(
        "  indexability: %.0f%% analyzed · %d noindex · %d skipped",
        (ix_summary.get("indexable_share", 0.0) or 0.0) * 100,
        ix_summary.get("noindex_pages", 0),
        ix_summary.get("skipped_pages", 0),
    )
    sc_summary = sitemap_coverage_data.get("summary", {}) or {}
    LOG.info(
        "  sitemap coverage: %d sitemap URLs · %d not fetched · %d non-indexable · %d crawled missing from sitemap",
        sc_summary.get("total_sitemap_urls", 0),
        sc_summary.get("sitemap_not_fetched", 0),
        sc_summary.get("sitemap_non_indexable", 0),
        sc_summary.get("crawled_not_in_sitemap", 0),
    )
    cc_summary = canonical_consistency_data.get("summary", {}) or {}
    LOG.info(
        "  canonical consistency: %d pages with issues · %d missing · %d external · %d non-self",
        cc_summary.get("pages_with_canonical_issues", 0),
        cc_summary.get("missing_canonical", 0),
        cc_summary.get("canonical_external_host", 0),
        cc_summary.get("canonical_non_self", 0),
    )

    def _external_links_task() -> dict:
        if not config.enable_external_links:
            return {}
        word_counts = {p.url: p.word_count for p in pages}
        pages_with_external = [(p.url, external_map.get(p.url, [])) for p in pages]
        ext_result = analyze_external(
            pages,
            word_counts,
            pages_with_external,
            check_links=config.check_external_links,
            http_cache=http_cache,
            max_workers=config.max_workers,
        )
        return external_payload(ext_result)

    def _analysis_task(name: str, fn):
        with stage_timings.track(f"analysis:{name}"):
            return fn()

    analysis_tasks = {
        "performance": lambda: performance_payload(analyze_performance(fetched)),
        "resource_status": lambda: resource_status_payload(
            analyze_resource_status(fetched, http_cache=getattr(crawler, "cache", None))
        ),
        "metadata_quality": lambda: metadata_quality_payload(analyze_metadata_quality(extracted_pages)),
        "media_accessibility": lambda: media_accessibility_payload(analyze_media_accessibility(extracted_pages)),
        "freshness": lambda: freshness_payload(analyze_freshness(extracted_pages)),
        "conversion": lambda: conversion_payload(analyze_conversion(extracted_pages)),
        "page_types": lambda: page_types_payload(analyze_page_types(extracted_pages)),
        "entities": lambda: entities_payload(analyze_entities(extracted_pages)),
        "structured_data": lambda: structured_data_payload(analyze_structured_data(extracted_pages)),
        "external_links": _external_links_task,
    }
    analysis_workers = max(1, int(config.analysis_workers or min(config.max_workers or 1, len(analysis_tasks))))
    if analysis_workers > 1:
        LOG.info("  running post-extraction analyses with %d workers", analysis_workers)
        with ThreadPoolExecutor(max_workers=analysis_workers) as pool:
            futures = {
                name: pool.submit(_analysis_task, name, fn)
                for name, fn in analysis_tasks.items()
            }
            analysis_results = {name: future.result() for name, future in futures.items()}
    else:
        analysis_results = {
            name: _analysis_task(name, fn)
            for name, fn in analysis_tasks.items()
        }

    performance_data = analysis_results["performance"]
    pf_summary = performance_data.get("summary", {}) or {}
    LOG.info(
        "  performance: median HTML %.0f KB · %.0f%% render-blocking · %d heavy pages",
        (pf_summary.get("median_html_weight_bytes", 0) or 0) / 1024,
        (pf_summary.get("render_blocking_share", 0.0) or 0.0) * 100,
        pf_summary.get("heavy_pages", 0),
    )
    resource_status_data = analysis_results["resource_status"]
    rs_summary = resource_status_data.get("summary", {}) or {}
    LOG.info(
        "  resource status: %d resources · %d broken JavaScript",
        rs_summary.get("total_resources", 0),
        rs_summary.get("broken_javascript", 0),
    )

    metadata_quality_data = analysis_results["metadata_quality"]
    mq_summary = metadata_quality_data.get("summary", {}) or {}
    LOG.info(
        "  metadata quality: %.0f%% pages with issues · %d missing descriptions · %d missing canonicals",
        (mq_summary.get("issue_share", 0.0) or 0.0) * 100,
        mq_summary.get("missing_description", 0),
        mq_summary.get("missing_canonical", 0),
    )

    media_accessibility_data = analysis_results["media_accessibility"]
    ma_summary = media_accessibility_data.get("summary", {}) or {}
    LOG.info(
        "  media accessibility: %.0f%% pages with issues · %d missing image alts · %d videos without captions",
        (ma_summary.get("issue_share", 0.0) or 0.0) * 100,
        ma_summary.get("images_missing_alt", 0),
        ma_summary.get("videos_missing_captions", 0),
    )

    freshness_data = analysis_results["freshness"]
    fr_summary = freshness_data.get("summary", {}) or {}
    LOG.info(
        "  freshness: %.0f%% date coverage · %d stale · %d missing dates",
        (fr_summary.get("date_coverage", 0.0) or 0.0) * 100,
        fr_summary.get("pages_stale", 0),
        fr_summary.get("missing_dates", 0),
    )

    conversion_data = analysis_results["conversion"]
    cv_summary = conversion_data.get("summary", {}) or {}
    LOG.info(
        "  conversion: %.0f%% CTA coverage · %.0f%% primary CTA coverage · %d forms · %d lead pages without capture",
        (cv_summary.get("cta_coverage", 0.0) or 0.0) * 100,
        (cv_summary.get("primary_cta_coverage", 0.0) or 0.0) * 100,
        cv_summary.get("total_forms", 0),
        cv_summary.get("lead_pages_without_capture", 0),
    )

    page_types_data = analysis_results["page_types"]
    pt_summary = page_types_data.get("summary", {}) or {}
    LOG.info(
        "  page types: %d types · %d template families · dominant %s / %s",
        pt_summary.get("page_type_count", 0),
        pt_summary.get("template_family_count", 0),
        pt_summary.get("dominant_page_type", "—"),
        pt_summary.get("dominant_template_family", "—"),
    )

    entities_data = analysis_results["entities"]
    ent_summary = entities_data.get("summary", {}) or {}
    LOG.info(
        "  entities: %d unique · %.0f%% coverage · authority %.1f",
        ent_summary.get("unique_entities", 0),
        (ent_summary.get("entity_coverage", 0.0) or 0.0) * 100,
        ent_summary.get("topical_authority_score", 0.0),
    )

    structured_data_data = analysis_results["structured_data"]
    external_payload_data: dict = analysis_results["external_links"]
    if external_payload_data:
        top_domains = external_payload_data.get("top_domains") or []
        LOG.info(
            "  external links: %d distinct domains, top %s ; %d broken (%s)",
            len(top_domains),
            top_domains[0]["domain"] if top_domains else "—",
            len(external_payload_data.get("broken_links") or []),
            "checked" if config.check_external_links else "not checked",
        )

    def _write_pre_embedding_technical(mode: str, status: str, message: str = "") -> dict:
        with stage_timings.track("technical-report", mode=mode):
            technical_link_payload = (
                _basic_linkgraph_payload(pages, outlinks_map, indexability_data)
                if config.enable_linkgraph
                else {}
            )
            technical_seo_data = build_technical_seo(
                pages,
                indexability=indexability_data,
                metadata_quality=metadata_quality_data,
                performance=performance_data,
                canonical_consistency=canonical_consistency_data,
                linkgraph=technical_link_payload,
                search_payload={},
                page_types=page_types_data,
                header_analysis={},
                structured_data=structured_data_data,
                media_accessibility=media_accessibility_data,
                resource_status=resource_status_data,
                sitemap_coverage=sitemap_coverage_data,
                external_links=external_payload_data,
                robots_txt=getattr(crawler, "robots_txt_info", {}),
                duplicate_rows=[],
            )
            tech_summary = technical_seo_data.get("summary", {}) or {}
            run_summary = {
                "status": status,
                "message": message,
                "preset": config.audit_preset,
                "pages": len(pages),
                "fetched_pages": len(fetched),
                "skipped_pages": len([r for r in extraction_rows if r.get("status") == "skipped"]),
                "noindex_dropped": noindex_dropped,
                "canonical_duplicates_dropped": canonical_dropped,
                "technical_issues": tech_summary.get("total_issues", 0),
                "high_technical_issues": tech_summary.get("high_issues", 0),
                "embedding_threshold": config.large_site_embedding_threshold,
            }
            write_technical_audit_bundle(
                report_dir,
                domain=host,
                mode=mode,
                summary=run_summary,
                timings=stage_timings.rows,
                technical_seo=technical_seo_data,
                indexability=indexability_data,
                sitemap_coverage=sitemap_coverage_data,
                canonical_consistency=canonical_consistency_data,
                performance=performance_data,
                resource_status=resource_status_data,
                metadata_quality=metadata_quality_data,
                media_accessibility=media_accessibility_data,
                page_types=page_types_data,
                entities=entities_data,
                freshness=freshness_data,
                conversion=conversion_data,
                structured_data=structured_data_data,
                external_links=external_payload_data,
                linkgraph=technical_link_payload,
            )
        stage_timings.write(report_dir)
        LOG.info("  technical audit bundle: %s", report_dir)
        return {
            "domain": host,
            "status": status,
            "message": message,
            "pages": len(pages),
            "site_focus_score": 0.0,
            "calibrated_focus": 0.0,
            "topic_dim": 0.0,
            "section_coherence": 0.0,
            "site_radius": 0.0,
            "outliers": 0,
            "duplicate_pairs": 0,
            "clusters": 0,
            "queries_evaluated": 0,
            "linkgraph_edges": technical_link_payload.get("edge_count", 0),
            "linkgraph_orphans": technical_link_payload.get("orphan_count", 0),
            "max_click_depth": technical_link_payload.get("max_click_depth", 0),
            "link_recommendations": 0,
            "external_domains": len((external_payload_data or {}).get("top_domains", [])),
            "broken_external": len((external_payload_data or {}).get("broken_links", [])),
            "technical_issues": tech_summary.get("total_issues", 0),
            "high_technical_issues": tech_summary.get("high_issues", 0),
            "report_dir": str(report_dir),
            "cache_dir": str(cache_dir),
            "html_report": None,
        }

    if config.technical_only:
        return _write_pre_embedding_technical(
            "technical",
            "technical_only",
            "Technical-only audit completed without semantic embeddings.",
        )

    if len(embed_inputs) > config.large_site_embedding_threshold and not config.allow_large_embeddings:
        message = (
            f"Stopped before embedding {len(embed_inputs)} pages. "
            "Use --allow-large-embeddings or --preset full-content to run semantic analysis."
        )
        LOG.warning("  %s", message)
        return _write_pre_embedding_technical(
            "large_site_embedding_safeguard",
            "stopped_before_large_embedding",
            message,
        )

    # 3) Embed pages (model loaded here, reused below for queries)
    with stage_timings.track("page-embeddings", pages=len(embed_inputs)):
        embed_cache = EmbeddingCache(
            cache_dir / f"embeddings_{config.model.replace('/', '_').replace('-', '_')}.npz"
        )
        embedder = Embedder(
            config.model,
            max_seq_length=embed_max_seq_length,
        )
        embeddings = embedder.encode_pages(
            embed_inputs,
            embed_cache,
            use_cache=config.use_embedding_cache,
        )
    LOG.info("  embeddings shape: %s", embeddings.shape)

    # Drop URL duplicates that snuck in via redirect collapses (different
    # request URLs that resolved to the same canonical URL).
    deduped_pages, deduped_embs, kept_idx = deduplicate_pages_by_url(pages, embeddings)
    if len(deduped_pages) != len(pages):
        LOG.info("  deduped %d → %d unique URLs", len(pages), len(deduped_pages))
        pages = deduped_pages
        embeddings = deduped_embs
        extracted_pages = [extracted_pages[i] for i in kept_idx]
        embed_inputs = [embed_inputs[i] for i in kept_idx]
        # outlinks / external maps are url-keyed, so they don't need filtering

    # 4) Core analysis
    result = analyze(
        pages,
        embeddings,
        duplicate_threshold=config.duplicate_threshold,
        duplicate_knn=config.duplicate_knn,
    )

    # 5) Scatter projection + clusters
    coords = labels = None
    if not config.skip_scatterplot:
        labels, coords = project(embeddings, num_clusters=config.scatter_clusters)
        LOG.info("  scatter projection done (%d clusters)", int(max(labels) + 1) if len(labels) else 0)

    # 6) Cluster labeling (c-TF-IDF)
    cluster_summaries = []
    if config.enable_cluster_labels and labels is not None:
        cluster_summaries = label_clusters(
            pages,
            embeddings,
            labels,
            site_centroid=result.site_centroid,
            cluster_texts=[ei.text for ei in embed_inputs],
        )
        LOG.info("  labelled %d clusters", len(cluster_summaries))
    cluster_overlap = cluster_overlap_matrix(cluster_summaries) if cluster_summaries else {}

    # 7) Keyword coverage
    coverage_payload: list[dict] = []
    if config.enable_keyword_coverage:
        queries: list[tuple[str, str]] = []
        if config.queries_file:
            for q in load_queries_from_file(Path(config.queries_file)):
                queries.append((q, "manual"))
        if not queries:
            queries = auto_mine_queries(extracted_pages, max_queries=config.auto_queries_max)

        if queries:
            LOG.info("  matching %d queries against %d pages", len(queries), len(pages))
            q_embs = embedder.encode([q for q, _ in queries], show_progress=False)
            matches = match_queries(
                queries, q_embs, pages, embeddings,
                coverage_threshold=config.coverage_threshold,
                cannibalization_threshold=config.cannibalization_threshold,
            )
            coverage_payload = queries_payload(matches)

    # 8) Answerability per page
    ans_payload: list[dict] = []
    if config.enable_answerability:
        ans_payload = answerability_payload(score_answerability(extracted_pages))
        LOG.info("  scored %d pages for answerability", len(ans_payload))

    # 9) Link graph + recommendations
    link_payload: dict = {}
    link_result = None
    if config.enable_linkgraph:
        pages_with_outlinks = [(p.url, outlinks_map.get(p.url, [])) for p in pages]
        # crawler.base_url already strips trailing slashes
        from .crawler import _starting_url  # local import to avoid cycles
        home_url = _starting_url(config.domain)
        link_result = analyze_linkgraph(
            pages, embeddings, pages_with_outlinks,
            home_url=home_url,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
            embedder=embedder,
            similarity_threshold=config.link_similarity_threshold,
            top_recommendations=config.link_recommendations_top_k,
        )
        link_payload = linkgraph_payload(link_result, pages)
        link_payload = annotate_link_target_statuses(link_payload, outlinks_map, indexability_data)
        link_payload = annotate_internal_link_rel_stats(link_payload, extracted_pages)
        LOG.info(
            "  link graph: %d edges, %d orphans, %d dead-ends, %d recs, max depth %d",
            link_result.edge_count, len(link_result.orphans),
            len(link_result.dead_ends), len(link_result.recommendations),
            max(link_result.click_depth.values()) if link_result.click_depth else 0,
        )

    # 11) Paragraph extraction + embedding (shared by every paragraph-level analysis below)
    paragraph_records: list = []
    if (
        config.enable_paragraph_links
        or config.enable_paragraph_clustering
        or config.enable_paragraph_impact
        or config.enable_semantic_ablation
        or config.enable_keyword_attribution
        or config.enable_weak_paragraphs
        or config.enable_heading_impact
        or config.enable_answer_blocks
        or config.enable_freshness_impact
        or config.enable_cannibalization
        or config.enable_duplicate_fragments
        or config.enable_information_gain
        or config.enable_content_quality
        or config.enable_paragraph_fanout
    ):
        paragraph_cache = ParagraphEmbeddingCache(
            cache_dir / f"paragraphs_{config.model.replace('/', '_').replace('-', '_')}.npz"
        )
        triples: list[tuple[int, int, str]] = []
        hashes: list[str] = []
        for i, ext in enumerate(extracted_pages):
            for j, para in enumerate(ext.paragraphs or []):
                triples.append((i, j, para))
                hashes.append(content_hash(f"{para}|{config.model}"))

        if triples:
            misses_idx: list[int] = []
            cached_embs: dict[int, np.ndarray] = {}
            for k, ((page_i, _, _), h) in enumerate(zip(triples, hashes)):
                src_url = pages[page_i].url
                cached = paragraph_cache.get(src_url, h) if config.use_embedding_cache else None
                if cached is not None:
                    cached_embs[k] = cached
                else:
                    misses_idx.append(k)

            LOG.info(
                "  paragraphs: %d total | %d cached | %d to embed",
                len(triples), len(triples) - len(misses_idx), len(misses_idx),
            )

            if misses_idx:
                miss_texts = [triples[k][2] for k in misses_idx]
                new_embs = embedder.encode(miss_texts, batch_size=64, show_progress=True)
                for slot, k in enumerate(misses_idx):
                    page_i, _, _ = triples[k]
                    paragraph_cache.put(pages[page_i].url, hashes[k], new_embs[slot])
                    cached_embs[k] = new_embs[slot]
                paragraph_cache.save()

            paragraph_records = [
                (triples[k][0], triples[k][1], triples[k][2], cached_embs[k])
                for k in range(len(triples))
            ]

    # 11.a-pre2) Header (H1-H6) audit. Embeds every header in one batched
    # call and compares against the host page's paragraph centroid so we
    # can flag headers that don't describe the content under them. Also
    # surfaces missing/duplicate H1s, level skips, and keyword frequency
    # at each header level.
    header_analysis_data: dict = {}
    header_scatter_data: dict = {}
    if paragraph_records:
        header_analysis_data = analyse_headers(
            pages, extracted_pages, embeddings, paragraph_records, embedder=embedder,
        )
        s = header_analysis_data.get("summary", {}) or {}
        LOG.info(
            "  headers: %d total · missing H1 %d · multi-H1 %d · drifty %d · structural issues %d",
            s.get("total_headers", 0), s.get("pages_missing_h1", 0),
            s.get("pages_multi_h1", 0), s.get("drifty_headers", 0),
            s.get("structural_issues", 0),
        )
        header_scatter_data = headers_for_scatter(
            pages, extracted_pages, paragraph_records, embedder=embedder,
        )

    # 11.a-pre) Paragraph link density — counts internal vs external <a href>
    # inside each paragraph at extraction time. Cheap (no embeddings) and
    # used both as a standalone editorial signal and as a saturation filter
    # for the link-recommendation step that follows.
    paragraph_density_rows: list = []
    paragraph_density_data: dict = {}
    paragraph_saturation: dict[tuple[int, int], float] = {}
    if paragraph_records:
        paragraph_density_rows = compute_paragraph_density_rows(pages, paragraph_records, extracted_pages)
        paragraph_density_data = paragraph_density_payload(paragraph_density_rows)
        paragraph_saturation = paragraph_density_lookup(paragraph_density_rows)
        s = paragraph_density_data.get("summary", {})
        LOG.info(
            "  paragraph density: median %.2f / p90 %.2f / p99 %.2f links per 100w · "
            "%d spammy · %.0f%% with zero links",
            s.get("median_density_per_100w", 0.0),
            s.get("p90_density_per_100w", 0.0),
            s.get("p99_density_per_100w", 0.0),
            s.get("spammy_count", 0),
            (s.get("zero_link_share", 0.0) or 0.0) * 100,
        )

    # 11a) Paragraph-level link recommendations
    paragraph_recs_payload: list[dict] = []
    if config.enable_paragraph_links and paragraph_records:
        ans_lookup = {a["url"]: a["score"] for a in ans_payload} if ans_payload else None
        recs = recommend_paragraph_links(
            pages,
            embeddings,
            paragraph_records,
            pages_with_outlinks=[(p.url, outlinks_map.get(p.url, [])) for p in pages],
            answerability_by_url=ans_lookup,
            embedder=embedder,
            paragraph_saturation=paragraph_saturation,
            similarity_floor=config.paragraph_similarity_floor,
            lift_floor=config.paragraph_lift_floor,
            top_k_per_page=config.paragraph_link_per_page,
            top_k_total=config.paragraph_link_top_k,
        )
        paragraph_recs_payload = paragraph_links_payload(recs)
        LOG.info("  paragraph link recs: %d", len(paragraph_recs_payload))

    # 11b) Paragraph topic clustering + scatter
    paragraph_clusters_data: list = []
    paragraph_cluster_overlap_data: dict = {}
    paragraph_scatter_data: dict = {}
    if config.enable_paragraph_clustering and paragraph_records and len(paragraph_records) >= 8:
        para_cluster_labels, para_cluster_summaries = cluster_paragraphs(
            paragraph_records, pages, result.site_centroid,
            num_clusters=config.paragraph_num_clusters,
        )
        paragraph_clusters_data = paragraph_clusters_payload(para_cluster_summaries)
        paragraph_cluster_overlap_data = paragraph_cluster_overlap_payload(para_cluster_summaries)
        cluster_label_lookup = {
            s.cluster_id: ", ".join(k["keyword"] for k in s.keywords[:3])
            for s in para_cluster_summaries
        }

        para_embs = np.stack([r[3] for r in paragraph_records]).astype(np.float32)
        chosen, coords_p = project_paragraphs(para_embs, sample_cap=config.paragraph_scatter_sample)
        paragraph_scatter_data = paragraph_scatter_payload(
            paragraph_records, pages, para_cluster_labels, chosen, coords_p, cluster_label_lookup,
        )
        LOG.info(
            "  paragraph clusters: %d (over %d paragraphs, %d shown in scatter)",
            len(paragraph_clusters_data), len(paragraph_records), paragraph_scatter_data.get("shown", 0),
        )

    # 11c) Paragraph-level query fanout
    paragraph_fanout_payload: list[dict] = []
    if config.enable_paragraph_fanout and paragraph_records and coverage_payload:
        # reuse the queries from coverage_payload so we share embeddings work
        # (the queries are auto-mined or from --queries-file)
        queries_for_fanout: list[tuple[str, str]] = [(q["query"], q["source"]) for q in coverage_payload]
        if queries_for_fanout:
            q_embs_fanout = embedder.encode([q for q, _ in queries_for_fanout], show_progress=False)
            from .keyword_coverage import match_queries_to_paragraphs as _match
            fanout = _match(
                queries_for_fanout, q_embs_fanout, pages, paragraph_records,
                paragraph_floor=config.paragraph_similarity_floor,
            )
            paragraph_fanout_payload = paragraph_match_payload(fanout)
            LOG.info(
                "  paragraph fanout: %d queries → %d gaps, %d scattered, %d focused",
                len(paragraph_fanout_payload),
                sum(1 for q in paragraph_fanout_payload if q["status"] == "gap"),
                sum(1 for q in paragraph_fanout_payload if q["status"] == "scattered"),
                sum(1 for q in paragraph_fanout_payload if q["status"] == "focused"),
            )

    # 11d) Title embeddings + content-quality diagnostics
    title_mismatch_data: list[dict] = []
    wrong_home_data: list[dict] = []
    page_improvement_data: list[dict] = []
    if config.enable_content_quality:
        title_texts = [p.title or p.url for p in pages]
        title_embeds = embedder.encode(title_texts, batch_size=64, show_progress=False)
        section_centroids = {s.name: s.centroid for s in result.sections.values()}
        title_results = title_mismatch(
            pages, embeddings, title_embeds, paragraph_records,
            section_centroids, labels, cluster_summaries,
        )
        title_mismatch_data = title_mismatch_payload(title_results, top_n=80)

        wrong_home_results = wrong_home_paragraphs(pages, embeddings, paragraph_records)
        wrong_home_data = wrong_home_payload(wrong_home_results)

        improvement_results = per_page_improvement(
            pages, title_results, ans_payload,
            result.dist_to_section, result.duplicate_pairs, wrong_home_results,
            (external_payload_data or {}).get("per_page", []),
            top_n=120,
        )
        page_improvement_data = improvement_payload(improvement_results)
        LOG.info(
            "  content quality: %d title mismatches, %d wrong-home paragraphs, %d pages flagged for editing",
            sum(1 for r in title_mismatch_data if r["title_to_content"] < 0.55),
            len(wrong_home_data),
            len(page_improvement_data),
        )

    # 11e) Linkbuilding overview — site-level link health, anchor
    # quality audit, and a UMAP scatter of the most-used anchor texts.
    linkbuilding_data: dict = {}
    if pages:
        linkbuilding_data = analyse_linkbuilding(
            pages, extracted_pages, link_payload,
            paragraph_density_payload=paragraph_density_data,
            embedder=embedder,
        )
        s = linkbuilding_data.get("summary", {}) or {}
        LOG.info(
            "  linkbuilding: %d total links (%d internal / %d external) · "
            "%.0f%% with descriptive anchor · %d empty links · %d image-only without alt",
            s.get("total_links", 0),
            s.get("internal_links", 0),
            s.get("external_links", 0),
            (s.get("descriptive_anchor_share", 0.0) or 0.0) * 100,
            s.get("empty_links", 0),
            s.get("image_links_no_alt", 0),
        )

    # 11.f) Organic search enrichment. Cache-first by default. GSC is the
    # preferred first-party provider in auto mode; Ahrefs/DataForSEO remain
    # fallback proxy providers when GSC is unavailable.
    ahrefs_data: dict = {}
    provider_choice = (config.search_provider or "auto").lower()
    collect_all_search = provider_choice in {"all", "combined"}
    search_provider_payloads: list[dict] = []
    if provider_choice not in {"none", "disabled", "off"}:
        if config.enable_gsc and provider_choice in {"auto", "gsc", "all", "combined"}:
            gsc_config = GSCConfig(
                enabled=True,
                property_url=config.gsc_property_url,
                start_date=config.gsc_start_date,
                end_date=config.gsc_end_date,
                top_pages_limit=config.gsc_top_pages_limit,
                keywords_limit=config.gsc_keywords_limit,
                refresh=config.gsc_refresh,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            snapshot = fetch_gsc_snapshot(host, cache_dir, gsc_config)
            gsc_analysis = build_gsc_analysis(
                snapshot,
                pages,
                embeddings,
                coords=coords,
                cluster_labels=labels,
                cluster_summaries=cluster_summaries,
                extracted_pages=extracted_pages,
                paragraph_records=paragraph_records,
                linkbuilding=linkbuilding_data,
                embedder=embedder,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            candidate = gsc_analysis.payload
            if _search_payload_usable(candidate):
                search_provider_payloads.append(candidate)
            if _search_payload_usable(candidate) or provider_choice == "gsc":
                ahrefs_data = candidate
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    gsc_analysis.semantic_rows,
                    gsc_analysis.semantic_embeddings,
                )

        needs_google_ads = provider_choice in {"google_ads", "all", "combined"} or (
            provider_choice == "auto"
            and config.use_google_ads_keywords
            and not _search_payload_usable(ahrefs_data)
        )
        if config.enable_google_ads and needs_google_ads:
            google_ads_config = GoogleAdsConfig(
                enabled=True,
                customer_id=config.google_ads_customer_id,
                login_customer_id=config.google_ads_login_customer_id,
                start_date=config.google_ads_start_date,
                end_date=config.google_ads_end_date,
                search_terms_limit=config.google_ads_search_terms_limit,
                min_cost=config.google_ads_min_cost,
                refresh=config.google_ads_refresh,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            snapshot = fetch_google_ads_snapshot(cache_dir, google_ads_config)
            google_ads_analysis = build_google_ads_analysis(
                snapshot,
                pages,
                embeddings,
                coords=coords,
                cluster_labels=labels,
                cluster_summaries=cluster_summaries,
                extracted_pages=extracted_pages,
                paragraph_records=paragraph_records,
                linkbuilding=linkbuilding_data,
                embedder=embedder,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            candidate = google_ads_analysis.payload
            if _search_payload_usable(candidate):
                search_provider_payloads.append(candidate)
            if _search_payload_usable(candidate) or provider_choice == "google_ads":
                ahrefs_data = candidate
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    google_ads_analysis.semantic_rows,
                    google_ads_analysis.semantic_embeddings,
                )

        ahrefs_config = AhrefsConfig(
            enabled=True,
            date=config.ahrefs_date,
            country=config.ahrefs_country,
            mode=config.ahrefs_mode,
            top_pages_limit=config.ahrefs_top_pages_limit,
            keywords_limit=config.ahrefs_keywords_limit,
            refresh=config.ahrefs_refresh,
            semantic_sample_cap=config.ahrefs_semantic_sample,
        )
        needs_ahrefs = provider_choice in {"ahrefs", "all", "combined"} or (
            provider_choice == "auto" and not _search_payload_usable(ahrefs_data)
        )
        if config.enable_ahrefs and needs_ahrefs:
            snapshot = fetch_ahrefs_snapshot(host, cache_dir, ahrefs_config)
            ahrefs_analysis = build_ahrefs_analysis(
                snapshot,
                pages,
                embeddings,
                coords=coords,
                cluster_labels=labels,
                cluster_summaries=cluster_summaries,
                extracted_pages=extracted_pages,
                paragraph_records=paragraph_records,
                linkbuilding=linkbuilding_data,
                embedder=embedder,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            candidate = ahrefs_analysis.payload
            if _search_payload_usable(candidate):
                search_provider_payloads.append(candidate)
            if _search_payload_usable(candidate) or provider_choice == "ahrefs":
                ahrefs_data = candidate
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    ahrefs_analysis.semantic_rows,
                    ahrefs_analysis.semantic_embeddings,
                )

        needs_dataforseo = provider_choice in {"dataforseo", "all", "combined"} or (
            provider_choice == "auto" and not _search_payload_usable(ahrefs_data)
        )
        if config.enable_dataforseo and needs_dataforseo:
            dataforseo_config = DataForSEOConfig(
                enabled=True,
                location_code=config.dataforseo_location_code,
                location_name=config.dataforseo_location_name,
                language_code=config.dataforseo_language_code,
                language_name=config.dataforseo_language_name,
                top_pages_limit=config.dataforseo_top_pages_limit,
                keywords_limit=config.dataforseo_keywords_limit,
                refresh=config.dataforseo_refresh,
                include_clickstream=config.dataforseo_include_clickstream,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            snapshot = fetch_dataforseo_snapshot(host, cache_dir, dataforseo_config)
            dataforseo_analysis = build_dataforseo_analysis(
                snapshot,
                pages,
                embeddings,
                coords=coords,
                cluster_labels=labels,
                cluster_summaries=cluster_summaries,
                extracted_pages=extracted_pages,
                paragraph_records=paragraph_records,
                linkbuilding=linkbuilding_data,
                embedder=embedder,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            candidate = dataforseo_analysis.payload
            if _search_payload_usable(candidate):
                search_provider_payloads.append(candidate)
            if _search_payload_usable(candidate) or not ahrefs_data:
                ahrefs_data = dataforseo_analysis.payload
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    dataforseo_analysis.semantic_rows,
                    dataforseo_analysis.semantic_embeddings,
                )

        if collect_all_search and search_provider_payloads:
            combined_analysis = build_combined_search_analysis(
                search_provider_payloads,
                pages,
                embeddings,
                extracted_pages=extracted_pages,
                paragraph_records=paragraph_records,
                linkbuilding=linkbuilding_data,
                embedder=embedder,
                semantic_sample_cap=config.ahrefs_semantic_sample,
            )
            if combined_analysis.payload:
                ahrefs_data = combined_analysis.payload
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    combined_analysis.semantic_rows,
                    combined_analysis.semantic_embeddings,
                )

        search_meta = ahrefs_data.get("meta", {}) or {}
        search_summary = ahrefs_data.get("summary", {}) or {}
        provider_label = search_meta.get("provider_label") or search_meta.get("provider") or "search"
        if search_summary:
            LOG.info(
                "  %s search data: %s · %d top pages · %d keywords · %d matched organic visits",
                provider_label,
                search_meta.get("cache_status", "unknown"),
                search_summary.get("top_pages", 0),
                search_summary.get("organic_keywords", 0),
                search_summary.get("matched_traffic", 0),
            )
        elif search_meta:
            LOG.info("  %s search data: %s", provider_label, search_meta.get("status", "unavailable"))

    competitive_data: dict = {}
    competitive_targets = []
    competitive_auto_meta: dict = {}
    if config.competitive_pairs_file and config.competitive_pairs_file.is_file():
        competitive_targets.extend(load_competitive_targets(config.competitive_pairs_file))
    if config.competitive_auto:
        auto_config = CompetitiveAutoConfig(
            enabled=True,
            max_clusters=config.competitive_auto_clusters,
            keywords_per_cluster=config.competitive_auto_keywords_per_cluster,
            results_per_keyword=config.competitive_auto_results_per_keyword,
            min_position=config.competitive_auto_min_position,
            max_position=config.competitive_auto_max_position,
            min_relevance=config.competitive_auto_min_relevance,
            product_seeds=config.competitive_auto_product_seeds,
            allow_nonlatin=config.competitive_auto_allow_nonlatin,
            refresh_serp=config.competitive_auto_refresh_serp,
            location_code=config.dataforseo_location_code,
            location_name=config.dataforseo_location_name,
            language_code=config.dataforseo_language_code,
            language_name=config.dataforseo_language_name,
        )
        auto_targets, competitive_auto_meta = build_auto_competitive_targets(
            host,
            ahrefs_data,
            pages,
            embedder,
            cache_dir,
            auto_config,
        )
        competitive_targets.extend(auto_targets)
        auto_summary = competitive_auto_meta.get("summary") or {}
        LOG.info(
            "  competitive auto: %s · %d clusters · %d SERP targets",
            competitive_auto_meta.get("status", "unknown"),
            auto_summary.get("clusters", 0),
            auto_summary.get("serp_targets", 0),
        )
    if competitive_targets:
        LOG.info("  competitive: comparing against %d competitor URLs", len(competitive_targets))
        competitive_data = compare_serp_targets(
            competitive_targets, pages, embeddings, paragraph_records,
            extracted_pages, embedder, http_cache,
            user_agent=crawl_config.user_agent,
        )
        if competitive_auto_meta:
            competitive_data["auto"] = competitive_auto_meta
    elif competitive_auto_meta:
        competitive_data = {
            "summary": {"queries": 0, "competitor_urls": 0, "serp_clusters": 0},
            "comparisons": [],
            "serp_clusters": [],
            "auto": competitive_auto_meta,
        }

    structured_data_data = structured_data_payload(
        analyze_structured_data(extracted_pages, search_payload=ahrefs_data)
    )
    sd_summary = structured_data_data.get("summary", {}) or {}
    LOG.info(
        "  structured data: %.0f%% coverage · %d invalid JSON-LD blocks · %d schema types · %d opportunities",
        (sd_summary.get("schema_coverage", 0.0) or 0.0) * 100,
        sd_summary.get("invalid_jsonld_blocks", 0),
        sd_summary.get("schema_type_count", 0),
        sd_summary.get("schema_opportunities", 0),
    )

    trust_signals_data: dict = {}
    if config.enable_trust_signals:
        trust_signals_data = build_trust_signals(
            pages,
            extracted_pages,
            search_payload=ahrefs_data,
        )
        ts_summary = trust_signals_data.get("summary", {}) or {}
        if ts_summary.get("status") == "ok":
            LOG.info(
                "  trust signals: avg %.1f · %d high-priority pages · %d missing evidence items",
                ts_summary.get("avg_trust_score", 0.0),
                ts_summary.get("high_priority_pages", 0),
                ts_summary.get("missing_evidence_items", 0),
            )

    conversion_balance_data: dict = {}
    if config.enable_conversion_balance:
        conversion_balance_data = build_conversion_balance(
            pages,
            extracted_pages,
            search_payload=ahrefs_data,
        )
        cb_summary = conversion_balance_data.get("summary", {}) or {}
        if cb_summary.get("status") == "ok":
            LOG.info(
                "  conversion balance: SEO %.1f · conversion %.1f · %d high-risk money pages",
                cb_summary.get("avg_seo_support", 0.0),
                cb_summary.get("avg_conversion_support", 0.0),
                cb_summary.get("high_risk_money_pages", 0),
            )

    entity_coverage_data: dict = {}
    if config.enable_entity_coverage:
        entity_coverage_data = build_entity_coverage(
            pages,
            extracted_pages,
            search_payload=ahrefs_data,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
        )
        ec_summary = entity_coverage_data.get("summary", {}) or {}
        if ec_summary.get("status") == "ok":
            LOG.info(
                "  entity coverage: %.0f%% avg · %d low-coverage pages · %d pages with core gaps",
                (ec_summary.get("avg_coverage", 0.0) or 0.0) * 100,
                ec_summary.get("low_coverage_pages", 0),
                ec_summary.get("pages_with_core_gaps", 0),
            )

    information_gain_data: dict = {}
    if config.enable_information_gain:
        information_gain_data = build_information_gain(
            pages,
            extracted_pages,
            paragraph_records,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
        )
        ig_summary = information_gain_data.get("summary", {}) or {}
        if ig_summary.get("status") == "ok":
            LOG.info(
                "  information gain: avg %.1f · %d low-score pages · %d high-score pages",
                ig_summary.get("avg_page_score", 0.0),
                ig_summary.get("low_score_pages", 0),
                ig_summary.get("high_score_pages", 0),
            )

    answer_blocks_data: dict = {}
    if config.enable_answer_blocks:
        answer_blocks_data = build_answer_blocks(
            pages,
            extracted_pages,
            paragraph_records,
            coverage=coverage_payload,
            paragraph_fanout=paragraph_fanout_payload,
            search_payload=ahrefs_data,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
        )
        ablocks_summary = answer_blocks_data.get("summary", {}) or {}
        if ablocks_summary.get("status") == "ok":
            LOG.info(
                "  answer blocks: %d blocks · %d query opportunities · %d/%d strong clusters",
                ablocks_summary.get("blocks", 0),
                ablocks_summary.get("opportunity_queries", 0),
                ablocks_summary.get("strong_query_clusters", 0),
                ablocks_summary.get("top_query_clusters", 0),
            )

    freshness_impact_data: dict = {}
    if config.enable_freshness_impact:
        freshness_impact_data = build_freshness_impact(
            pages,
            extracted_pages,
            freshness_data,
            search_payload=ahrefs_data,
            paragraph_records=paragraph_records,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
            coords=coords,
        )
        fi_summary = freshness_impact_data.get("summary", {}) or {}
        if fi_summary.get("status") == "ok":
            LOG.info(
                "  freshness impact: %d high-impact sections · %d traffic at risk",
                fi_summary.get("high_impact_sections", 0),
                fi_summary.get("traffic_at_risk", 0),
            )

    paragraph_impact_data: dict = {}
    if config.enable_paragraph_impact and paragraph_records and ahrefs_data:
        paragraph_impact_data = build_paragraph_impact(
            pages,
            extracted_pages,
            paragraph_records,
            ahrefs_data,
            embedder=embedder,
        )
        pi_summary = paragraph_impact_data.get("summary", {}) or {}
        if pi_summary.get("status") == "ok":
            LOG.info(
                "  paragraph impact: %d scored paragraphs · %.0f attributed organic visits",
                pi_summary.get("scored_paragraphs", 0),
                pi_summary.get("attributed_traffic", 0.0),
            )
        elif pi_summary:
            LOG.info("  paragraph impact: %s", pi_summary.get("status", "unavailable"))

    semantic_ablation_data: dict = {}
    if config.enable_semantic_ablation and paragraph_records:
        semantic_ablation_data = build_semantic_ablation(
            pages,
            embeddings,
            extracted_pages,
            paragraph_records,
            ahrefs_data,
            embedder=embedder,
        )
        ab_summary = semantic_ablation_data.get("summary", {}) or {}
        if ab_summary.get("status") == "ok":
            LOG.info(
                "  semantic ablation: %d topic carriers · %d noise candidates",
                ab_summary.get("topic_carriers", 0),
                ab_summary.get("noise_candidates", 0),
            )
        elif ab_summary:
            LOG.info("  semantic ablation: %s", ab_summary.get("status", "unavailable"))

    keyword_attribution_data: dict = {}
    if config.enable_keyword_attribution and paragraph_records and ahrefs_data:
        keyword_attribution_data = build_keyword_attribution(
            pages,
            extracted_pages,
            paragraph_records,
            ahrefs_data,
            embedder=embedder,
        )
        ka_summary = keyword_attribution_data.get("summary", {}) or {}
        if ka_summary.get("status") == "ok":
            LOG.info(
                "  keyword attribution: %d keywords · %d unmatched",
                ka_summary.get("keyword_rows", 0),
                ka_summary.get("unmatched_keywords", 0),
            )
        elif ka_summary:
            LOG.info("  keyword attribution: %s", ka_summary.get("status", "unavailable"))

    cannibalization_data: dict = {}
    if config.enable_cannibalization:
        cannibalization_data = build_cannibalization(
            pages,
            embeddings,
            extracted_pages,
            paragraph_records,
            keyword_attribution=keyword_attribution_data,
            search_payload=ahrefs_data,
            linkgraph=link_payload,
            indexability=indexability_data,
            outlinks_map=outlinks_map,
        )
        cannibal_summary = cannibalization_data.get("summary", {}) or {}
        if cannibal_summary.get("status") == "ok":
            LOG.info(
                "  cannibalization: %d page conflicts · %d paragraph overlaps · %d traffic at risk",
                cannibal_summary.get("page_conflicts", 0),
                cannibal_summary.get("paragraph_conflicts", 0),
                cannibal_summary.get("traffic_at_risk", 0),
            )
        elif cannibal_summary:
            LOG.info("  cannibalization: %s", cannibal_summary.get("status", "unavailable"))

    duplicate_fragments_data: dict = {}
    if config.enable_duplicate_fragments:
        duplicate_fragments_data = build_duplicate_fragments(
            pages,
            extracted_pages,
            paragraph_records,
            search_payload=ahrefs_data,
            keyword_attribution=keyword_attribution_data,
        )
        df_summary = duplicate_fragments_data.get("summary", {}) or {}
        if df_summary.get("status") == "ok":
            LOG.info(
                "  duplicate fragments: %d groups · %d strong patterns · %d harmful boilerplate",
                df_summary.get("groups", 0),
                df_summary.get("strong_patterns", 0),
                df_summary.get("harmful_boilerplate", 0),
            )
        elif df_summary:
            LOG.info("  duplicate fragments: %s", df_summary.get("status", "unavailable"))

    template_patterns_data: dict = {}
    if config.enable_template_patterns:
        template_patterns_data = build_template_patterns(
            pages,
            extracted_pages,
            page_types=page_types_data,
            search_payload=ahrefs_data,
            linkgraph=link_payload,
        )
        tp_summary = template_patterns_data.get("summary", {}) or {}
        if tp_summary.get("status") == "ok":
            LOG.info(
                "  template patterns: %d patterns · %d recommendations · source %s",
                tp_summary.get("patterns", 0),
                tp_summary.get("recommendations", 0),
                tp_summary.get("performance_source", ""),
            )
        elif tp_summary:
            LOG.info("  template patterns: %s", tp_summary.get("status", "unavailable"))

    winning_paragraphs_data: dict = {}
    if paragraph_impact_data:
        winning_paragraphs_data = build_winning_paragraphs(
            paragraph_impact_data,
            semantic_ablation_data,
            keyword_attribution_data,
        )
        wp_summary = winning_paragraphs_data.get("summary", {}) or {}
        if wp_summary.get("status") == "ok":
            LOG.info(
                "  winning paragraphs: %d rows · %d topic carriers",
                wp_summary.get("rows", 0),
                wp_summary.get("topic_carriers", 0),
            )

    weak_paragraphs_data: dict = {}
    if config.enable_weak_paragraphs and paragraph_records:
        weak_paragraphs_data = build_weak_paragraphs(
            pages,
            embeddings,
            extracted_pages,
            paragraph_records,
            search_payload=ahrefs_data,
            paragraph_impact=paragraph_impact_data,
            semantic_ablation=semantic_ablation_data,
            keyword_attribution=keyword_attribution_data,
            paragraph_density_rows=paragraph_density_rows,
            freshness=freshness_data,
            cluster_labels=labels,
        )
        wp_summary = weak_paragraphs_data.get("summary", {}) or {}
        if wp_summary.get("status") == "ok":
            LOG.info(
                "  weak paragraphs: %d flagged · %d main-content · %d template · %.0f traffic opportunity",
                wp_summary.get("flagged_rows", 0),
                wp_summary.get("main_content_rows", 0),
                wp_summary.get("template_rows", 0),
                wp_summary.get("total_traffic_opportunity", 0.0),
            )
        elif wp_summary:
            LOG.info("  weak paragraphs: %s", wp_summary.get("status", "unavailable"))

    heading_impact_data: dict = {}
    if config.enable_heading_impact and paragraph_records:
        heading_impact_data = build_heading_impact(
            pages,
            extracted_pages,
            paragraph_records,
            paragraph_impact=paragraph_impact_data,
            keyword_attribution=keyword_attribution_data,
            freshness=freshness_data,
            cluster_labels=labels,
        )
        hi_summary = heading_impact_data.get("summary", {}) or {}
        if hi_summary.get("status") == "ok":
            LOG.info(
                "  heading impact: %d headings · %d high-demand · %d rename opportunities",
                hi_summary.get("headings", 0),
                hi_summary.get("high_demand_headings", 0),
                hi_summary.get("rename_opportunities", 0),
            )
        elif hi_summary:
            LOG.info("  heading impact: %s", hi_summary.get("status", "unavailable"))

    if link_result is not None:
        link_payload["traffic_weighted_pagerank"] = traffic_weighted_pagerank_payload(
            link_result,
            pages,
            embeddings,
            search_payload=ahrefs_data,
            page_types=page_types_data,
            indexability=indexability_data,
        )
        twpr_summary = (link_payload.get("traffic_weighted_pagerank") or {}).get("summary", {}) or {}
        if twpr_summary.get("status") == "ok":
            LOG.info(
                "  traffic-weighted PageRank: %.0f%% alignment · %d underserved search pages · %d authority-heavy pages",
                (twpr_summary.get("authority_traffic_alignment", 0.0) or 0.0) * 100,
                twpr_summary.get("high_traffic_low_authority_pages", 0),
                twpr_summary.get("high_authority_low_value_pages", 0),
            )
        link_payload["hub_bottlenecks"] = hub_bottleneck_payload(
            link_result,
            pages,
            traffic_authority=link_payload.get("traffic_weighted_pagerank") or {},
            page_types=page_types_data,
        )
        hub_summary = (link_payload.get("hub_bottlenecks") or {}).get("summary", {}) or {}
        if hub_summary.get("status") == "ok":
            LOG.info(
                "  hub/bottleneck pages: %d bottlenecks · %d bridges · %.0f%% resilience",
                hub_summary.get("bottleneck_pages", 0),
                hub_summary.get("bridge_pages", 0),
                (hub_summary.get("architecture_resilience", 0.0) or 0.0) * 100,
            )
        link_payload["link_removal_simulation"] = link_removal_simulation_payload(
            link_result,
            pages,
            embeddings,
            paragraph_records,
            traffic_authority=link_payload.get("traffic_weighted_pagerank") or {},
        )
        removal_summary = (link_payload.get("link_removal_simulation") or {}).get("summary", {}) or {}
        if removal_summary.get("status") == "ok":
            LOG.info(
                "  link removal simulation: %d critical · %d weak/harmful · %d template/nav",
                removal_summary.get("critical_links", 0),
                removal_summary.get("irrelevant_links", 0) + removal_summary.get("potentially_harmful_links", 0),
                removal_summary.get("template_navigation_links", 0),
            )
        link_payload["link_addition_simulation"] = build_link_addition_simulation(
            paragraph_recs_payload,
            pages,
            link_payload,
            paragraph_saturation=paragraph_saturation,
        )
        addition_summary = (link_payload.get("link_addition_simulation") or {}).get("summary", {}) or {}
        if addition_summary.get("status") == "ok":
            paragraph_recs_payload = (link_payload["link_addition_simulation"].get("recommendations") or paragraph_recs_payload)
            LOG.info(
                "  link addition simulation: %d high-priority · %.1f avg expected benefit",
                addition_summary.get("high_priority", 0),
                addition_summary.get("avg_expected_benefit", 0.0),
            )
        link_payload["high_demand_low_link"] = high_demand_low_link_payload(
            link_result,
            pages,
            search_payload=ahrefs_data,
            traffic_authority=link_payload.get("traffic_weighted_pagerank") or {},
            link_addition=link_payload.get("link_addition_simulation") or {},
            page_types=page_types_data,
        )
        hdl_summary = (link_payload.get("high_demand_low_link") or {}).get("summary", {}) or {}
        if hdl_summary.get("status") == "ok":
            LOG.info(
                "  high-demand low-link pages: %d high-priority · %d opportunity traffic · %.0f%% demand/support alignment",
                hdl_summary.get("high_demand_low_support_pages", 0),
                hdl_summary.get("opportunity_traffic", 0),
                (hdl_summary.get("demand_support_alignment", 0.0) or 0.0) * 100,
            )
        link_payload["anchor_relevance"] = build_anchor_relevance(
            extracted_pages,
            embeddings,
            search_payload=ahrefs_data,
            entities_payload=entities_data,
            embedder=embedder,
        )
        anchor_summary = (link_payload.get("anchor_relevance") or {}).get("summary", {}) or {}
        if anchor_summary.get("status") == "ok":
            LOG.info(
                "  anchor relevance: %.0f%% descriptive · %d weak · %d empty/image-only",
                (anchor_summary.get("descriptive_rate", 0.0) or 0.0) * 100,
                anchor_summary.get("weak_links", 0),
                anchor_summary.get("empty_links", 0) + anchor_summary.get("image_only_links", 0),
            )
        link_payload["contextual_link_impact"] = build_contextual_link_impact(
            extracted_pages,
            paragraph_records,
            embeddings,
            linkgraph=link_payload,
            paragraph_impact=paragraph_impact_data,
        )
        context_summary = (link_payload.get("contextual_link_impact") or {}).get("summary", {}) or {}
        if context_summary.get("status") == "ok":
            LOG.info(
                "  contextual link impact: %.1f avg · %d high-impact main-content · %d template",
                context_summary.get("avg_contextual_impact", 0.0),
                context_summary.get("high_impact_contextual_links", 0),
                context_summary.get("template_links", 0),
            )
        link_payload["internal_link_patterns"] = build_internal_link_patterns(
            pages,
            extracted_pages,
            page_types=page_types_data,
            linkgraph=link_payload,
        )
        pattern_summary = (link_payload.get("internal_link_patterns") or {}).get("summary", {}) or {}
        if pattern_summary.get("status") == "ok":
            LOG.info(
                "  internal link patterns: %d patterns · %d weak-page recommendations · %.0f%% avg confidence",
                pattern_summary.get("patterns", 0),
                pattern_summary.get("recommendations", 0),
                (pattern_summary.get("avg_confidence", 0.0) or 0.0) * 100,
            )
        link_payload["link_flow"] = link_flow_payload(
            link_result,
            pages,
            (ahrefs_data.get("top_pages") or []) if ahrefs_data else [],
            page_types=page_types_data,
            traffic_authority=link_payload.get("traffic_weighted_pagerank") or {},
            link_removal=link_payload.get("link_removal_simulation") or {},
            contextual_links=link_payload.get("contextual_link_impact") or {},
        )

    # 12) Action plan: synthesise prioritised recommendations from the
    # already-built payloads. Cheap (no embeddings), pure aggregation.
    duplicate_rows = build_duplicate_rows(result)
    outlier_rows = build_outlier_rows(result)
    recommendations = synthesize_recommendations(
        duplicates_rows=duplicate_rows,
        outliers_rows=outlier_rows,
        coverage_payload=coverage_payload,
        answerability_payload=ans_payload,
        linkgraph_payload=link_payload,
        search_payload=ahrefs_data,
        paragraph_links=paragraph_recs_payload,
        wrong_home_payload=wrong_home_data,
        title_mismatch=title_mismatch_data,
        external_links_payload=external_payload_data,
    )
    recommendations_data = recommendations_payload(recommendations)
    LOG.info(
        "  recommendations: %d actions (high %d / med %d / low %d)",
        recommendations_data["total"],
        recommendations_data["by_priority"].get("high", 0),
        recommendations_data["by_priority"].get("medium", 0),
        recommendations_data["by_priority"].get("low", 0),
    )

    best_pages_data = build_best_page_explainers(
        pages,
        search_payload=ahrefs_data,
        linkgraph=link_payload,
        structured_data=structured_data_data,
        freshness=freshness_data,
        entity_coverage=entity_coverage_data,
        information_gain=information_gain_data,
        paragraph_impact=paragraph_impact_data,
        winning_paragraphs=winning_paragraphs_data,
        template_patterns=template_patterns_data,
        header_analysis=header_analysis_data,
        answerability=ans_payload,
        conversion_balance=conversion_balance_data,
        metadata_quality=metadata_quality_data,
        page_types=page_types_data,
    )
    bp_summary = best_pages_data.get("summary", {}) or {}
    if bp_summary.get("status") == "ok":
        LOG.info(
            "  best pages: %d explainers across %d clusters",
            bp_summary.get("pages", 0),
            bp_summary.get("clusters", 0),
        )

    performance_explainer_data = build_performance_explainer(
        pages,
        extracted_pages,
        search_payload=ahrefs_data,
        linkgraph=link_payload,
        structured_data=structured_data_data,
        freshness=freshness_data,
        entities=entities_data,
        entity_coverage=entity_coverage_data,
        information_gain=information_gain_data,
        answerability=ans_payload,
        conversion=conversion_data,
        conversion_balance=conversion_balance_data,
        performance=performance_data,
        metadata_quality=metadata_quality_data,
        media_accessibility=media_accessibility_data,
        paragraph_density=paragraph_density_data,
    )
    pe_summary = performance_explainer_data.get("summary", {}) or {}
    LOG.info(
        "  performance explainer: %s · %d pages · cv R2 %.3f",
        pe_summary.get("status", "unknown"),
        pe_summary.get("sample_size", 0),
        pe_summary.get("validation_r2", 0.0) or 0.0,
    )
    technical_seo_data = build_technical_seo(
        pages,
        indexability=indexability_data,
        metadata_quality=metadata_quality_data,
        performance=performance_data,
        canonical_consistency=canonical_consistency_data,
        linkgraph=link_payload,
        search_payload=ahrefs_data,
        page_types=page_types_data,
        header_analysis=header_analysis_data,
        structured_data=structured_data_data,
        media_accessibility=media_accessibility_data,
        resource_status=resource_status_data,
        sitemap_coverage=sitemap_coverage_data,
        external_links=external_payload_data,
        robots_txt=getattr(crawler, "robots_txt_info", {}),
        duplicate_rows=duplicate_rows,
    )
    tech_summary = technical_seo_data.get("summary", {}) or {}
    LOG.info(
        "  technical SEO model: %d pages · %d issues · %d high",
        tech_summary.get("total_pages", 0),
        tech_summary.get("total_issues", 0),
        tech_summary.get("high_issues", 0),
    )
    history_snapshot_data = build_history_snapshot(
        host,
        pages,
        extracted_pages,
        outlinks_map=outlinks_map,
        structured_data=structured_data_data,
        freshness=freshness_data,
        metadata_quality=metadata_quality_data,
        indexability=indexability_data,
        search_payload=ahrefs_data,
    )

    # 13) Reports
    summary = write_all(
        report_dir,
        result,
        model_name=config.model,
        domain=host,
        coords=coords,
        cluster_labels=labels,
        cluster_summaries=cluster_summaries,
        coverage=coverage_payload,
        answerability=ans_payload,
        linkgraph=link_payload,
        external_links=external_payload_data,
        paragraph_link_recs=paragraph_recs_payload,
        cluster_overlap=cluster_overlap,
        paragraph_clusters=paragraph_clusters_data,
        paragraph_cluster_overlap=paragraph_cluster_overlap_data,
        paragraph_scatter=paragraph_scatter_data,
        paragraph_fanout=paragraph_fanout_payload,
        paragraph_impact=paragraph_impact_data,
        semantic_ablation=semantic_ablation_data,
        keyword_attribution=keyword_attribution_data,
        answer_blocks=answer_blocks_data,
        freshness_impact=freshness_impact_data,
        cannibalization=cannibalization_data,
        duplicate_fragments=duplicate_fragments_data,
        template_patterns=template_patterns_data,
        winning_paragraphs=winning_paragraphs_data,
        weak_paragraphs=weak_paragraphs_data,
        heading_impact=heading_impact_data,
        entity_coverage=entity_coverage_data,
        information_gain=information_gain_data,
        title_mismatch=title_mismatch_data,
        wrong_home=wrong_home_data,
        page_improvement=page_improvement_data,
        competitive=competitive_data,
        recommendations=recommendations_data,
        paragraph_density=paragraph_density_data,
        header_analysis=header_analysis_data,
        header_scatter=header_scatter_data,
        linkbuilding=linkbuilding_data,
        structured_data=structured_data_data,
        trust_signals=trust_signals_data,
        conversion_balance=conversion_balance_data,
        metadata_quality=metadata_quality_data,
        media_accessibility=media_accessibility_data,
        resource_status=resource_status_data,
        page_types=page_types_data,
        entities=entities_data,
        freshness=freshness_data,
        conversion=conversion_data,
        indexability=indexability_data,
        sitemap_coverage=sitemap_coverage_data,
        canonical_consistency=canonical_consistency_data,
        performance=performance_data,
        ahrefs=ahrefs_data,
        best_pages=best_pages_data,
        performance_explainer=performance_explainer_data,
        history_snapshot=history_snapshot_data,
        technical_seo=technical_seo_data,
    )

    template_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    html_path = None
    if template_path.is_file():
        html_path = write_html_report(
            report_dir,
            template_path,
            result,
            model_name=config.model,
            domain=host,
            coords=coords,
            cluster_labels=labels,
            cluster_summaries=cluster_summaries,
            coverage=coverage_payload,
            answerability=ans_payload,
            linkgraph=link_payload,
            external_links=external_payload_data,
            paragraph_link_recs=paragraph_recs_payload,
            cluster_overlap=cluster_overlap,
            paragraph_clusters=paragraph_clusters_data,
            paragraph_cluster_overlap=paragraph_cluster_overlap_data,
            paragraph_scatter=paragraph_scatter_data,
            paragraph_fanout=paragraph_fanout_payload,
            paragraph_impact=paragraph_impact_data,
            semantic_ablation=semantic_ablation_data,
            keyword_attribution=keyword_attribution_data,
            answer_blocks=answer_blocks_data,
            freshness_impact=freshness_impact_data,
            cannibalization=cannibalization_data,
            duplicate_fragments=duplicate_fragments_data,
            template_patterns=template_patterns_data,
            winning_paragraphs=winning_paragraphs_data,
            weak_paragraphs=weak_paragraphs_data,
            heading_impact=heading_impact_data,
            entity_coverage=entity_coverage_data,
            information_gain=information_gain_data,
            title_mismatch=title_mismatch_data,
            wrong_home=wrong_home_data,
            page_improvement=page_improvement_data,
            competitive=competitive_data,
            recommendations=recommendations_data,
            paragraph_density=paragraph_density_data,
            header_analysis=header_analysis_data,
            header_scatter=header_scatter_data,
            linkbuilding=linkbuilding_data,
            structured_data=structured_data_data,
            trust_signals=trust_signals_data,
            conversion_balance=conversion_balance_data,
            metadata_quality=metadata_quality_data,
            media_accessibility=media_accessibility_data,
            page_types=page_types_data,
            entities=entities_data,
            freshness=freshness_data,
            conversion=conversion_data,
            indexability=indexability_data,
            performance=performance_data,
            ahrefs=ahrefs_data,
            best_pages=best_pages_data,
            performance_explainer=performance_explainer_data,
        )
        LOG.info("  HTML report: %s", html_path)

    if config.save_snapshot:
        snapshot_dir = save_report_snapshot(host, config.projects_root, report_dir)
        LOG.info("  history snapshot: %s", snapshot_dir)
    stage_timings.write(report_dir)

    LOG.info("=== summary ===")
    LOG.info("  pages: %d", len(pages))
    LOG.info("  siteFocusScore: %.4f", result.site_metrics["focus_score"])
    LOG.info("  siteRadius:     %.4f", result.site_metrics["radius"])
    LOG.info("  outliers: %d  near-duplicate pairs: %d", summary["outliers"], summary["duplicates"])
    LOG.info("  reports: %s", report_dir)

    return {
        "domain": host,
        "pages": len(pages),
        "site_focus_score": result.site_metrics["focus_score"],
        "calibrated_focus": result.calibrated_focus_score,
        "topic_dim": (result.topic_dim or {}).get("effective_dim", 0.0),
        "section_coherence": (result.coherence or {}).get("ratio", 0.0),
        "site_radius": result.site_metrics["radius"],
        "outliers": summary["outliers"],
        "duplicate_pairs": summary["duplicates"],
        "clusters": len(cluster_summaries),
        "queries_evaluated": len(coverage_payload),
        "linkgraph_edges": link_payload.get("edge_count", 0),
        "linkgraph_orphans": link_payload.get("orphan_count", 0),
        "max_click_depth": link_payload.get("max_click_depth", 0),
        "link_recommendations": len(link_payload.get("recommendations", [])),
        "external_domains": len((external_payload_data or {}).get("top_domains", [])),
        "broken_external": len((external_payload_data or {}).get("broken_links", [])),
        "search_provider": (ahrefs_data.get("meta", {}) or {}).get("provider", ""),
        "search_status": (ahrefs_data.get("meta", {}) or {}).get("status", ""),
        "search_top_pages_traffic": (ahrefs_data.get("summary", {}) or {}).get("top_pages_traffic", 0),
        "ahrefs_status": (ahrefs_data.get("meta", {}) or {}).get("status", ""),
        "ahrefs_top_pages_traffic": (ahrefs_data.get("summary", {}) or {}).get("top_pages_traffic", 0),
        "report_dir": str(report_dir),
        "cache_dir": str(cache_dir),
        "html_report": str(html_path) if html_path else None,
        "stage_timings": stage_timings.rows,
    }
