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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from .analyzer import PageInfo, analyze, deduplicate_pages_by_url, section_for_url
from .ahrefs import AhrefsConfig, build_analysis as build_ahrefs_analysis
from .ahrefs import fetch_snapshot as fetch_ahrefs_snapshot
from .ahrefs import write_semantic_cache as write_ahrefs_semantic_cache
from .answer_blocks import build_answer_blocks
from .dataforseo import DataForSEOConfig, build_analysis as build_dataforseo_analysis
from .dataforseo import fetch_snapshot as fetch_dataforseo_snapshot
from .answerability import score_all as score_answerability
from .answerability import to_payload as answerability_payload
from .cache import EmbeddingCache, HttpCache, ParagraphEmbeddingCache, content_hash, domain_slug
from .cluster_labels import cluster_overlap_matrix, label_clusters
from .competitive_analysis import (
    compare_one as compare_competitor,
    load_competitive_pairs,
    to_payload as competitive_payload,
)
from .content_quality import (
    improvement_payload,
    per_page_improvement,
    title_mismatch,
    title_mismatch_payload,
    wrong_home_paragraphs,
    wrong_home_payload,
)
from .conversion import analyze as analyze_conversion
from .conversion import to_payload as conversion_payload
from .paragraph_clustering import (
    cluster_and_label as cluster_paragraphs,
    project_paragraphs,
    to_scatter_payload as paragraph_scatter_payload,
    to_summary_payload as paragraph_clusters_payload,
)
from .crawler import Crawler, CrawlConfig, DEFAULT_EXCLUDE_PATTERNS
from .embedder import DEFAULT_MODEL, EmbedInput, Embedder
from .entities import analyze as analyze_entities
from .entities import to_payload as entities_payload
from .entity_coverage import build_entity_coverage
from .external_links import analyze as analyze_external
from .external_links import to_payload as external_payload
from .extractor import extract
from .freshness import analyze as analyze_freshness
from .freshness import to_payload as freshness_payload
from .header_analysis import analyse as analyse_headers
from .header_analysis import headers_for_scatter
from .heading_impact import build_heading_impact
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
from .linkgraph import link_flow_payload
from .linkgraph import to_payload as linkgraph_payload
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
from .paragraph_links import recommend as recommend_paragraph_links
from .paragraph_links import to_payload as paragraph_links_payload
from .performance import analyze as analyze_performance
from .performance import to_payload as performance_payload
from .recommendations import synthesize as synthesize_recommendations
from .recommendations import to_payload as recommendations_payload
from .report import build_duplicate_rows, build_outlier_rows, write_all
from .scatter import project
from .semantic_ablation import build_semantic_ablation
from .structured_data import analyze as analyze_structured_data
from .structured_data import to_payload as structured_data_payload
from .weak_paragraphs import build_weak_paragraphs
from .winning_paragraphs import build_winning_paragraphs

LOG = logging.getLogger(__name__)


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

    # New analyses
    enable_cluster_labels: bool = True
    enable_keyword_coverage: bool = True
    enable_answerability: bool = True
    enable_answer_blocks: bool = True
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
    queries_file: Optional[Path] = None
    auto_queries_max: int = 200
    coverage_threshold: float = 0.55
    cannibalization_threshold: float = 0.72
    link_similarity_threshold: float = 0.85
    link_recommendations_top_k: int = 75
    search_provider: str = "auto"
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

    LOG.info("=== site-audit run for %s ===", host)
    LOG.info("  cache:  %s", cache_dir)
    LOG.info("  report: %s", report_dir)

    http_cache = HttpCache(cache_dir / "http.sqlite")
    embed_cache = EmbeddingCache(
        cache_dir / f"embeddings_{config.model.replace('/', '_').replace('-', '_')}.npz"
    )

    # 1) Crawl
    crawl_config = CrawlConfig(
        domain=config.domain,
        max_pages=config.max_pages,
        max_workers=config.max_workers,
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
    fetched = crawler.discover_and_crawl()
    LOG.info("  fetched %d pages (cache: %s)", len(fetched), http_cache.stats())

    # 2) Extract
    pages: list[PageInfo] = []
    extracted_pages = []  # list[ExtractedPage] in same order as `pages`
    embed_inputs: list[EmbedInput] = []
    outlinks_map: dict[str, list[tuple[str, str]]] = {}
    external_map: dict[str, list[tuple[str, str]]] = {}
    extraction_rows: list[dict] = []

    noindex_dropped = 0
    fetched_total = len(fetched)
    for idx, r in enumerate(fetched, 1):
        ext = extract(r.url, r.body, max_chars=config.max_chars, x_robots_tag=getattr(r, "x_robots_tag", ""))
        if ext is None or not ext.title:
            extraction_rows.append({
                "url": r.url,
                "status": "skipped",
                "reason": "unusable",
                "http_status": getattr(r, "status", 0),
            })
            continue
        if ext.noindex:
            # The page asked search engines not to index it — exclude from
            # the analysis corpus. We still consumed its outlinks during
            # the crawl (so internal links from a noindex landing page
            # contribute to authority), but the page itself does not
            # count toward focus / clusters / coverage / recommendations.
            noindex_dropped += 1
            extraction_rows.append({
                "url": r.url,
                "title": ext.title,
                "status": "skipped",
                "reason": "noindex",
                "source": ext.noindex_source,
                "http_status": getattr(r, "status", 0),
            })
            continue
        section = section_for_url(r.url)
        embed_text = ". ".join(part for part in [ext.title, ext.description, ext.body] if part)
        if not embed_text.strip():
            extraction_rows.append({
                "url": r.url,
                "title": ext.title,
                "status": "skipped",
                "reason": "empty_embedding_text",
                "http_status": getattr(r, "status", 0),
            })
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
        extraction_rows.append({
            "url": r.url,
            "title": ext.title,
            "status": "analyzed",
            "reason": "",
            "http_status": getattr(r, "status", 0),
        })
        if idx % 500 == 0 or idx == fetched_total:
            LOG.info("  extracted %d / %d fetched pages (%d usable)", idx, fetched_total, len(pages))
    if noindex_dropped:
        LOG.info("  dropped %d noindex pages (meta robots / X-Robots-Tag)", noindex_dropped)

    if not pages:
        LOG.warning("No usable pages — aborting before embedding.")
        return {"pages": 0}

    LOG.info("  prepared %d pages for embedding", len(pages))

    indexability_data = indexability_payload(
        analyze_indexability(fetched, extraction_rows, {p.url for p in pages})
    )
    ix_summary = indexability_data.get("summary", {}) or {}
    LOG.info(
        "  indexability: %.0f%% analyzed · %d noindex · %d skipped",
        (ix_summary.get("indexable_share", 0.0) or 0.0) * 100,
        ix_summary.get("noindex_pages", 0),
        ix_summary.get("skipped_pages", 0),
    )

    performance_data = performance_payload(analyze_performance(fetched))
    pf_summary = performance_data.get("summary", {}) or {}
    LOG.info(
        "  performance: median HTML %.0f KB · %.0f%% render-blocking · %d heavy pages",
        (pf_summary.get("median_html_weight_bytes", 0) or 0) / 1024,
        (pf_summary.get("render_blocking_share", 0.0) or 0.0) * 100,
        pf_summary.get("heavy_pages", 0),
    )

    # 3) Embed pages (model loaded here, reused below for queries)
    embedder = Embedder(config.model)
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

    structured_data_data = structured_data_payload(analyze_structured_data(extracted_pages))
    sd_summary = structured_data_data.get("summary", {}) or {}
    LOG.info(
        "  structured data: %.0f%% coverage · %d invalid JSON-LD blocks · %d schema types",
        (sd_summary.get("schema_coverage", 0.0) or 0.0) * 100,
        sd_summary.get("invalid_jsonld_blocks", 0),
        sd_summary.get("schema_type_count", 0),
    )

    metadata_quality_data = metadata_quality_payload(analyze_metadata_quality(extracted_pages))
    mq_summary = metadata_quality_data.get("summary", {}) or {}
    LOG.info(
        "  metadata quality: %.0f%% pages with issues · %d missing descriptions · %d missing canonicals",
        (mq_summary.get("issue_share", 0.0) or 0.0) * 100,
        mq_summary.get("missing_description", 0),
        mq_summary.get("missing_canonical", 0),
    )

    media_accessibility_data = media_accessibility_payload(analyze_media_accessibility(extracted_pages))
    ma_summary = media_accessibility_data.get("summary", {}) or {}
    LOG.info(
        "  media accessibility: %.0f%% pages with issues · %d missing image alts · %d videos without captions",
        (ma_summary.get("issue_share", 0.0) or 0.0) * 100,
        ma_summary.get("images_missing_alt", 0),
        ma_summary.get("videos_missing_captions", 0),
    )

    freshness_data = freshness_payload(analyze_freshness(extracted_pages))
    fr_summary = freshness_data.get("summary", {}) or {}
    LOG.info(
        "  freshness: %.0f%% date coverage · %d stale · %d missing dates",
        (fr_summary.get("date_coverage", 0.0) or 0.0) * 100,
        fr_summary.get("pages_stale", 0),
        fr_summary.get("missing_dates", 0),
    )

    conversion_data = conversion_payload(analyze_conversion(extracted_pages))
    cv_summary = conversion_data.get("summary", {}) or {}
    LOG.info(
        "  conversion: %.0f%% CTA coverage · %.0f%% primary CTA coverage · %d forms · %d lead pages without capture",
        (cv_summary.get("cta_coverage", 0.0) or 0.0) * 100,
        (cv_summary.get("primary_cta_coverage", 0.0) or 0.0) * 100,
        cv_summary.get("total_forms", 0),
        cv_summary.get("lead_pages_without_capture", 0),
    )

    page_types_data = page_types_payload(analyze_page_types(extracted_pages))
    pt_summary = page_types_data.get("summary", {}) or {}
    LOG.info(
        "  page types: %d types · %d template families · dominant %s / %s",
        pt_summary.get("page_type_count", 0),
        pt_summary.get("template_family_count", 0),
        pt_summary.get("dominant_page_type", "—"),
        pt_summary.get("dominant_template_family", "—"),
    )

    entities_data = entities_payload(analyze_entities(extracted_pages))
    ent_summary = entities_data.get("summary", {}) or {}
    LOG.info(
        "  entities: %d unique · %.0f%% coverage · authority %.1f",
        ent_summary.get("unique_entities", 0),
        (ent_summary.get("entity_coverage", 0.0) or 0.0) * 100,
        ent_summary.get("topical_authority_score", 0.0),
    )

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
        LOG.info(
            "  link graph: %d edges, %d orphans, %d dead-ends, %d recs, max depth %d",
            link_result.edge_count, len(link_result.orphans),
            len(link_result.dead_ends), len(link_result.recommendations),
            max(link_result.click_depth.values()) if link_result.click_depth else 0,
        )

    # 10) External link analysis
    external_payload_data: dict = {}
    if config.enable_external_links:
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
        external_payload_data = external_payload(ext_result)
        LOG.info(
            "  external links: %d distinct domains, top %s ; %d broken (%s)",
            len(ext_result.top_domains),
            ext_result.top_domains[0]["domain"] if ext_result.top_domains else "—",
            len(ext_result.broken_links),
            "checked" if config.check_external_links else "not checked",
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
    paragraph_scatter_data: dict = {}
    if config.enable_paragraph_clustering and paragraph_records and len(paragraph_records) >= 8:
        para_cluster_labels, para_cluster_summaries = cluster_paragraphs(
            paragraph_records, pages, result.site_centroid,
            num_clusters=config.paragraph_num_clusters,
        )
        paragraph_clusters_data = paragraph_clusters_payload(para_cluster_summaries)
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

    # 11e) Competitive analysis (optional, takes a query/url file)
    competitive_data: list[dict] = []
    if config.competitive_pairs_file and config.competitive_pairs_file.is_file():
        pairs = load_competitive_pairs(config.competitive_pairs_file)
        if pairs:
            LOG.info("  competitive: comparing against %d competitor URLs", len(pairs))
            comparisons = []
            for q, url in pairs:
                cmp_result = compare_competitor(
                    q, url, pages, embeddings, paragraph_records,
                    extracted_pages, embedder, http_cache,
                    user_agent=crawl_config.user_agent,
                )
                comparisons.append(cmp_result)
            competitive_data = competitive_payload(comparisons)

    # 11.f) Linkbuilding overview — site-level link health, anchor
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

    # 11.g) Organic search enrichment. Cache-first by default. Ahrefs remains
    # the primary provider; DataForSEO is fetched only when selected or when
    # Ahrefs has no usable cache/API payload.
    ahrefs_data: dict = {}
    provider_choice = (config.search_provider or "auto").lower()
    if provider_choice not in {"none", "disabled", "off"}:
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
        if config.enable_ahrefs and provider_choice in {"auto", "ahrefs"}:
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
            if _search_payload_usable(candidate) or provider_choice == "ahrefs":
                ahrefs_data = candidate
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    ahrefs_analysis.semantic_rows,
                    ahrefs_analysis.semantic_embeddings,
                )

        needs_dataforseo = provider_choice == "dataforseo" or (
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
            if _search_payload_usable(dataforseo_analysis.payload) or not ahrefs_data:
                ahrefs_data = dataforseo_analysis.payload
                write_ahrefs_semantic_cache(
                    cache_dir,
                    config.model,
                    dataforseo_analysis.semantic_rows,
                    dataforseo_analysis.semantic_embeddings,
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
        link_payload["link_flow"] = link_flow_payload(
            link_result,
            pages,
            (ahrefs_data.get("top_pages") or []) if ahrefs_data else [],
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
        paragraph_scatter=paragraph_scatter_data,
        paragraph_fanout=paragraph_fanout_payload,
        paragraph_impact=paragraph_impact_data,
        semantic_ablation=semantic_ablation_data,
        keyword_attribution=keyword_attribution_data,
        answer_blocks=answer_blocks_data,
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
        metadata_quality=metadata_quality_data,
        media_accessibility=media_accessibility_data,
        page_types=page_types_data,
        entities=entities_data,
        freshness=freshness_data,
        conversion=conversion_data,
        indexability=indexability_data,
        performance=performance_data,
        ahrefs=ahrefs_data,
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
            paragraph_scatter=paragraph_scatter_data,
            paragraph_fanout=paragraph_fanout_payload,
            paragraph_impact=paragraph_impact_data,
            semantic_ablation=semantic_ablation_data,
            keyword_attribution=keyword_attribution_data,
            answer_blocks=answer_blocks_data,
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
            metadata_quality=metadata_quality_data,
            media_accessibility=media_accessibility_data,
            page_types=page_types_data,
            entities=entities_data,
            freshness=freshness_data,
            conversion=conversion_data,
            indexability=indexability_data,
            performance=performance_data,
            ahrefs=ahrefs_data,
        )
        LOG.info("  HTML report: %s", html_path)

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
    }
