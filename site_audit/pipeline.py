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

from .analyzer import PageInfo, analyze, deduplicate_pages_by_url, section_for_url
from .answerability import score_all as score_answerability
from .answerability import to_payload as answerability_payload
from .cache import EmbeddingCache, HttpCache, ParagraphEmbeddingCache, content_hash, domain_slug
from .cluster_labels import cluster_overlap_matrix, label_clusters
from .crawler import Crawler, CrawlConfig
from .embedder import DEFAULT_MODEL, EmbedInput, Embedder
from .external_links import analyze as analyze_external
from .external_links import to_payload as external_payload
from .extractor import extract
from .html_report import write_html_report
from .keyword_coverage import (
    auto_mine_queries, load_queries_from_file, match_queries, to_payload as queries_payload,
)
from .linkgraph import analyze as analyze_linkgraph
from .linkgraph import to_payload as linkgraph_payload
from .paragraph_links import recommend as recommend_paragraph_links
from .paragraph_links import to_payload as paragraph_links_payload
from .report import write_all
from .scatter import project

LOG = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    domain: str
    projects_root: Path = Path("projects")
    # Optional overrides — if set, take precedence over projects_root.
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    model: str = DEFAULT_MODEL
    max_pages: int = 2000
    max_workers: int = 8
    request_delay: float = 0.0
    duplicate_threshold: float = 0.92
    duplicate_knn: int = 10
    scatter_clusters: int = 30
    follow_subdomains: bool = False
    respect_robots: bool = True
    use_http_cache: bool = True
    use_embedding_cache: bool = True
    skip_scatterplot: bool = False
    max_chars: int = 4000

    # New analyses
    enable_cluster_labels: bool = True
    enable_keyword_coverage: bool = True
    enable_answerability: bool = True
    enable_linkgraph: bool = True
    enable_external_links: bool = True
    enable_paragraph_links: bool = True
    check_external_links: bool = False     # opt-in HEAD requests
    paragraph_link_top_k: int = 200
    paragraph_link_per_page: int = 8
    paragraph_similarity_floor: float = 0.65
    paragraph_lift_floor: float = 0.05
    queries_file: Optional[Path] = None
    auto_queries_max: int = 200
    coverage_threshold: float = 0.55
    cannibalization_threshold: float = 0.72
    link_similarity_threshold: float = 0.85
    link_recommendations_top_k: int = 75


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

    for r in fetched:
        ext = extract(r.url, r.body, max_chars=config.max_chars)
        if ext is None or not ext.title:
            continue
        section = section_for_url(r.url)
        embed_text = ". ".join(part for part in [ext.title, ext.description, ext.body] if part)
        if not embed_text.strip():
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

    if not pages:
        LOG.warning("No usable pages — aborting before embedding.")
        return {"pages": 0}

    LOG.info("  prepared %d pages for embedding", len(pages))

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

    # 9) Link graph + recommendations
    link_payload: dict = {}
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

    # 11) Paragraph-level link recommendations
    paragraph_recs_payload: list[dict] = []
    paragraph_records = None  # populated on demand for visualizations later
    if config.enable_paragraph_links:
        paragraph_cache = ParagraphEmbeddingCache(
            cache_dir / f"paragraphs_{config.model.replace('/', '_').replace('-', '_')}.npz"
        )
        # Collect (page_index, para_index, text) and compute hashes
        triples: list[tuple[int, int, str]] = []
        hashes: list[str] = []
        for i, ext in enumerate(extracted_pages):
            for j, para in enumerate(ext.paragraphs or []):
                triples.append((i, j, para))
                hashes.append(content_hash(f"{para}|{config.model}"))

        if triples:
            cached_idx: list[int] = []
            misses_idx: list[int] = []
            cached_embs: dict[int, np.ndarray] = {}
            for k, ((page_i, _, _), h) in enumerate(zip(triples, hashes)):
                src_url = pages[page_i].url
                cached = paragraph_cache.get(src_url, h) if config.use_embedding_cache else None
                if cached is not None:
                    cached_embs[k] = cached
                    cached_idx.append(k)
                else:
                    misses_idx.append(k)

            LOG.info(
                "  paragraphs: %d total | %d cached | %d to embed",
                len(triples), len(cached_idx), len(misses_idx),
            )

            if misses_idx:
                miss_texts = [triples[k][2] for k in misses_idx]
                new_embs = embedder.encode(miss_texts, batch_size=64, show_progress=True)
                for slot, k in enumerate(misses_idx):
                    page_i, _, _ = triples[k]
                    src_url = pages[page_i].url
                    paragraph_cache.put(src_url, hashes[k], new_embs[slot])
                    cached_embs[k] = new_embs[slot]
                paragraph_cache.save()

            paragraph_records = [
                (triples[k][0], triples[k][1], triples[k][2], cached_embs[k])
                for k in range(len(triples))
            ]

            ans_lookup = {a["url"]: a["score"] for a in ans_payload} if ans_payload else None
            recs = recommend_paragraph_links(
                pages,
                embeddings,
                paragraph_records,
                pages_with_outlinks=[(p.url, outlinks_map.get(p.url, [])) for p in pages],
                answerability_by_url=ans_lookup,
                embedder=embedder,
                similarity_floor=config.paragraph_similarity_floor,
                lift_floor=config.paragraph_lift_floor,
                top_k_per_page=config.paragraph_link_per_page,
                top_k_total=config.paragraph_link_top_k,
            )
            paragraph_recs_payload = paragraph_links_payload(recs)
            LOG.info("  paragraph link recs: %d", len(paragraph_recs_payload))

    # 12) Reports
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
        "report_dir": str(report_dir),
        "cache_dir": str(cache_dir),
        "html_report": str(html_path) if html_path else None,
    }
