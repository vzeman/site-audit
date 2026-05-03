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
from .paragraph_clustering import (
    cluster_and_label as cluster_paragraphs,
    project_paragraphs,
    to_scatter_payload as paragraph_scatter_payload,
    to_summary_payload as paragraph_clusters_payload,
)
from .crawler import Crawler, CrawlConfig
from .embedder import DEFAULT_MODEL, EmbedInput, Embedder
from .external_links import analyze as analyze_external
from .external_links import to_payload as external_payload
from .extractor import extract
from .header_analysis import analyse as analyse_headers
from .header_analysis import headers_for_scatter
from .linkbuilding import analyse as analyse_linkbuilding
from .html_report import write_html_report
from .keyword_coverage import (
    auto_mine_queries, load_queries_from_file, match_queries,
    match_queries_to_paragraphs, paragraph_match_payload,
    to_payload as queries_payload,
)
from .linkgraph import analyze as analyze_linkgraph
from .linkgraph import to_payload as linkgraph_payload
from .paragraph_density import compute_rows as compute_paragraph_density_rows
from .paragraph_density import density_lookup as paragraph_density_lookup
from .paragraph_density import to_payload as paragraph_density_payload
from .paragraph_links import recommend as recommend_paragraph_links
from .paragraph_links import to_payload as paragraph_links_payload
from .recommendations import synthesize as synthesize_recommendations
from .recommendations import to_payload as recommendations_payload
from .report import build_duplicate_rows, build_outlier_rows, write_all
from .scatter import project
from .structured_data import analyze as analyze_structured_data
from .structured_data import to_payload as structured_data_payload

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
    skip_scatterplot: bool = False
    max_chars: int = 4000

    # New analyses
    enable_cluster_labels: bool = True
    enable_keyword_coverage: bool = True
    enable_answerability: bool = True
    enable_linkgraph: bool = True
    enable_external_links: bool = True
    enable_paragraph_links: bool = True
    enable_paragraph_clustering: bool = True
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

    noindex_dropped = 0
    for r in fetched:
        ext = extract(r.url, r.body, max_chars=config.max_chars, x_robots_tag=getattr(r, "x_robots_tag", ""))
        if ext is None or not ext.title:
            continue
        if ext.noindex:
            # The page asked search engines not to index it — exclude from
            # the analysis corpus. We still consumed its outlinks during
            # the crawl (so internal links from a noindex landing page
            # contribute to authority), but the page itself does not
            # count toward focus / clusters / coverage / recommendations.
            noindex_dropped += 1
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
    if noindex_dropped:
        LOG.info("  dropped %d noindex pages (meta robots / X-Robots-Tag)", noindex_dropped)

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

    structured_data_data = structured_data_payload(analyze_structured_data(extracted_pages))
    sd_summary = structured_data_data.get("summary", {}) or {}
    LOG.info(
        "  structured data: %.0f%% coverage · %d invalid JSON-LD blocks · %d schema types",
        (sd_summary.get("schema_coverage", 0.0) or 0.0) * 100,
        sd_summary.get("invalid_jsonld_blocks", 0),
        sd_summary.get("schema_type_count", 0),
    )

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

    # 11) Paragraph extraction + embedding (shared by every paragraph-level analysis below)
    paragraph_records: list = []
    if config.enable_paragraph_links or config.enable_paragraph_clustering or config.enable_content_quality or config.enable_paragraph_fanout:
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
