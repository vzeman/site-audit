"""Command-line interface: ``site-audit run <domain>`` / ``serve <domain>``.

Two subcommands:

* ``run`` crawls + analyzes + writes reports to ``projects/<domain>/report/``.
* ``serve`` starts a local viewer that loads that report directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from . import compare as _compare
from .embedder import DEFAULT_MODEL
from .pipeline import PipelineConfig, project_paths, run
from .server import serve


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_command(args: argparse.Namespace) -> int:
    # `--clean` is a single-flag shortcut for "wipe the project's cache and
    # re-run from scratch". It deletes the cache directory (HTTP + embedding
    # + paragraph npz) before the pipeline starts; the `--no-*-cache` flags
    # only *bypass* the caches, they don't reset them. Use --clean when the
    # crawler / extractor / embedder logic itself has changed and you want
    # the new logic applied to every page.
    if args.clean:
        from .pipeline import PipelineConfig as _PC
        probe = _PC(
            domain=args.domain,
            projects_root=Path(args.projects_root),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        cache_dir, _ = project_paths(probe)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"  cleaned cache: {cache_dir}")

    config = PipelineConfig(
        domain=args.domain,
        projects_root=Path(args.projects_root),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model=args.model,
        max_pages=args.max_pages,
        max_workers=args.workers,
        request_delay=args.request_delay,
        duplicate_threshold=args.duplicate_threshold,
        duplicate_knn=args.duplicate_knn,
        scatter_clusters=args.scatter_clusters,
        follow_subdomains=args.follow_subdomains,
        respect_robots=not args.ignore_robots,
        use_http_cache=not args.no_http_cache,
        use_embedding_cache=not args.no_embedding_cache,
        crawl_discovered_links=not args.sitemap_only,
        strip_header_footer=args.strip_header_footer,
        content_include_classes=args.content_include_class,
        content_exclude_classes=args.content_exclude_class,
        skip_scatterplot=args.no_scatterplot,
        max_chars=args.max_chars,
        enable_cluster_labels=not args.no_cluster_labels,
        enable_keyword_coverage=not args.no_keyword_coverage,
        enable_answerability=not args.no_answerability,
        enable_linkgraph=not args.no_linkgraph,
        enable_external_links=not args.no_external_links,
        enable_paragraph_links=not args.no_paragraph_links,
        enable_paragraph_clustering=not args.no_paragraph_clustering,
        enable_weak_paragraphs=not args.no_weak_paragraphs,
        enable_content_quality=not args.no_content_quality,
        enable_paragraph_fanout=not args.no_paragraph_fanout,
        check_external_links=args.check_external,
        competitive_pairs_file=Path(args.competitive) if args.competitive else None,
        queries_file=Path(args.queries_file) if args.queries_file else None,
        auto_queries_max=args.auto_queries_max,
        coverage_threshold=args.coverage_threshold,
        cannibalization_threshold=args.cannibalization_threshold,
        link_similarity_threshold=args.link_similarity_threshold,
        link_recommendations_top_k=args.link_recommendations,
        url_include_patterns=args.url_include,
        url_exclude_patterns=args.url_exclude,
        sitemap_urls=args.sitemap_url,
        sitemap_include_patterns=args.sitemap_include,
        sitemap_exclude_patterns=args.sitemap_exclude,
        sitemap_lastmod_after=args.sitemap_lastmod_after,
        sitemap_lastmod_within_days=args.sitemap_lastmod_within_days,
        search_provider="none" if args.no_search_data else args.search_provider,
        enable_dataforseo=not args.no_dataforseo,
        enable_ahrefs=not args.no_ahrefs,
        ahrefs_date=args.ahrefs_date,
        ahrefs_country=args.ahrefs_country,
        ahrefs_mode=args.ahrefs_mode,
        ahrefs_top_pages_limit=args.ahrefs_top_pages_limit,
        ahrefs_keywords_limit=args.ahrefs_keywords_limit,
        ahrefs_refresh=args.ahrefs_refresh,
        ahrefs_semantic_sample=args.ahrefs_semantic_sample,
        dataforseo_location_code=args.dataforseo_location_code,
        dataforseo_location_name=args.dataforseo_location_name,
        dataforseo_language_code=args.dataforseo_language_code,
        dataforseo_language_name=args.dataforseo_language_name,
        dataforseo_top_pages_limit=args.dataforseo_top_pages_limit,
        dataforseo_keywords_limit=args.dataforseo_keywords_limit,
        dataforseo_refresh=args.dataforseo_refresh,
        dataforseo_include_clickstream=args.dataforseo_include_clickstream,
    )
    summary = run(config)
    if summary.get("pages", 0) == 0:
        print("No pages were processed — check the domain and try again.")
        return 1
    print("\nDone.")
    print(f"  pages:               {summary['pages']}")
    print(f"  raw focus:           {summary['site_focus_score']:.4f}")
    print(f"  calibrated focus:    {summary.get('calibrated_focus', 0):.4f}")
    print(f"  topic dim:           {summary.get('topic_dim', 0):.1f}")
    print(f"  section coherence:   {summary.get('section_coherence', 0):.2f}")
    print(f"  radius:              {summary['site_radius']:.4f}")
    print(f"  outliers:            {summary['outliers']}")
    print(f"  near-dup pairs:      {summary['duplicate_pairs']}")
    print(f"  topic clusters:      {summary['clusters']}")
    print(f"  queries evaluated:   {summary['queries_evaluated']}")
    print(f"  link edges:          {summary['linkgraph_edges']}")
    print(f"  orphans:             {summary['linkgraph_orphans']}")
    print(f"  max click depth:     {summary.get('max_click_depth', 0)}")
    print(f"  link recs:           {summary['link_recommendations']}")
    print(f"  cited domains:       {summary.get('external_domains', 0)}")
    print(f"  broken outbound:     {summary.get('broken_external', 0)}")
    if summary.get("search_status") == "ok" or summary.get("ahrefs_status") == "ok":
        provider = summary.get("search_provider") or "search"
        print(f"  {provider} traffic:  {summary.get('search_top_pages_traffic', summary.get('ahrefs_top_pages_traffic', 0)):,}")
    print(f"  report dir:          {summary['report_dir']}")
    if summary.get("html_report"):
        print(f"  HTML report:         {summary['html_report']}")
    print(f"\nLaunch the live viewer with:")
    print(f"  site-audit serve {summary['domain']}")
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    if args.all:
        domains = sorted(p.name for p in projects_root.iterdir()
                         if p.is_dir() and not p.name.startswith("_")
                         and (p / "report" / "site_metrics.json").exists())
    else:
        domains = list(args.domains)
    if len(domains) < 2:
        print("compare needs at least two domains with completed reports.")
        return 1

    print(f"Comparing {len(domains)} domains: {', '.join(domains)}")
    payload = _compare.build_payload(domains, projects_root)
    if not payload.get("domains"):
        print("No domains had a usable report — nothing to compare.")
        return 1

    out_dir = projects_root / "_compare" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        __import__("json").dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )

    template = Path(__file__).resolve().parent.parent / "ui" / "compare.html"
    if template.is_file():
        out_html = out_dir / "index.html"
        _compare.write_html(template, payload, out_html)
        print(f"Wrote {out_html}")
    else:
        print(f"compare.html template missing at {template}; only JSON written.")

    package_path = _compare.package_comparison(out_dir, projects_root, payload.get("domains", []))
    print(f"Wrote {package_path}")

    return 0


def _serve_command(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve().parent.parent
    ui_dir = Path(args.ui_dir) if args.ui_dir else here / "ui"
    cfg = PipelineConfig(
        domain=args.domain,
        projects_root=Path(args.projects_root),
        cache_dir=None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    _, report_dir = project_paths(cfg)
    serve(
        report_dir=report_dir,
        ui_dir=ui_dir,
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="site-audit", description="Crawl any website, embed its pages, surface duplicates, outliers, topic clusters, GEO scoring, and internal-link recommendations.")
    p.add_argument("--version", action="version", version=f"site-audit {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Crawl + analyze a domain")
    run_p.add_argument("domain", help="Domain or URL, e.g. example.com or https://example.com")
    run_p.add_argument("--projects-root", default="projects", help="Where projects live (default: projects/)")
    run_p.add_argument("--cache-dir", default=None, help="Override cache directory")
    run_p.add_argument("--output-dir", default=None, help="Override report directory")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    run_p.add_argument("--max-pages", type=int, default=10000)
    run_p.add_argument("--workers", type=int, default=8)
    run_p.add_argument("--request-delay", type=float, default=0.0, help="Seconds to sleep before each request (slow down for rate-limited sites)")
    run_p.add_argument("--duplicate-threshold", type=float, default=0.92)
    run_p.add_argument("--duplicate-knn", type=int, default=10)
    run_p.add_argument("--scatter-clusters", type=int, default=30)
    run_p.add_argument("--max-chars", type=int, default=4000)
    run_p.add_argument("--follow-subdomains", action="store_true")
    run_p.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (use sparingly)")
    run_p.add_argument("--sitemap-url", action="append", default=[],
                       help="Only discover URLs from this sitemap URL; repeat for multiple sitemaps")
    run_p.add_argument("--sitemap-only", action="store_true",
                       help="Fetch only sitemap-discovered URLs; do not enqueue internal page links")
    run_p.add_argument("--strip-header-footer", action="store_true",
                       help="Remove <header> and <footer> HTML before text extraction and link analysis")
    run_p.add_argument("--content-include-class", "--include-class", action="append", default=[],
                       help="Only analyze HTML elements with this class; repeat for multiple classes")
    run_p.add_argument("--content-exclude-class", "--exclude-class", action="append", default=[],
                       help="Remove HTML elements with this class before analysis; repeat for multiple classes")
    run_p.add_argument("--sitemap-include", action="append", default=[],
                       help="Regex whitelist for sitemap URLs when following sitemap indexes")
    run_p.add_argument("--sitemap-exclude", action="append", default=[],
                       help="Regex blacklist for sitemap URLs when following sitemap indexes")
    run_p.add_argument("--sitemap-lastmod-after", default=None,
                       help="Only keep sitemap URLs whose <lastmod> is on/after YYYY-MM-DD")
    run_p.add_argument("--sitemap-lastmod-within-days", type=int, default=None,
                       help="Only keep sitemap URLs whose <lastmod> is within the last N days")
    run_p.add_argument("--url-include", "--include-url", action="append", default=[],
                       help="Regex whitelist for discovered/crawled page URLs; repeat for OR matching")
    run_p.add_argument("--url-exclude", "--exclude-url", action="append", default=[],
                       help="Regex blacklist for discovered/crawled page URLs; repeat to add rules")
    run_p.add_argument("--no-http-cache", action="store_true")
    run_p.add_argument("--no-embedding-cache", action="store_true")
    run_p.add_argument("--clean", action="store_true",
                       help="Delete the project's cache directory before running. "
                       "Use after a crawler/extractor/embedder change so every page is re-processed.")
    run_p.add_argument("--no-scatterplot", action="store_true")
    run_p.add_argument("--no-cluster-labels", action="store_true")
    run_p.add_argument("--no-keyword-coverage", action="store_true")
    run_p.add_argument("--no-answerability", action="store_true")
    run_p.add_argument("--no-linkgraph", action="store_true")
    run_p.add_argument("--no-external-links", action="store_true")
    run_p.add_argument("--no-paragraph-links", action="store_true",
                       help="Skip paragraph-level internal link recommendation embeddings")
    run_p.add_argument("--no-paragraph-clustering", action="store_true")
    run_p.add_argument("--no-weak-paragraphs", action="store_true")
    run_p.add_argument("--no-content-quality", action="store_true")
    run_p.add_argument("--no-paragraph-fanout", action="store_true")
    run_p.add_argument("--check-external", action="store_true", help="HEAD-check every outbound URL (slow, results cached)")
    run_p.add_argument("--search-provider", default="auto",
                       choices=["auto", "ahrefs", "dataforseo", "none"],
                       help="Search-demand data source. auto uses Ahrefs first, then DataForSEO fallback")
    run_p.add_argument("--no-search-data", action="store_true",
                       help="Skip all paid search-data provider enrichment")
    run_p.add_argument("--no-ahrefs", action="store_true",
                       help="Skip Ahrefs API enrichment even when AHREFS_API_KEY is set")
    run_p.add_argument("--no-dataforseo", action="store_true",
                       help="Skip DataForSEO fallback enrichment even when DATAFORSEO credentials are set")
    run_p.add_argument("--ahrefs-refresh", action="store_true",
                       help="Ignore cached Ahrefs snapshots and fetch fresh API data")
    run_p.add_argument("--ahrefs-date", default=None,
                       help="Ahrefs report date in YYYY-MM-DD. Default: reuse latest cache, otherwise today")
    run_p.add_argument("--ahrefs-country", default=None,
                       help="Optional Ahrefs country code, e.g. US, GB, SK")
    run_p.add_argument("--ahrefs-mode", default="subdomains",
                       choices=["exact", "prefix", "domain", "subdomains"],
                       help="Ahrefs target mode (default: subdomains)")
    run_p.add_argument("--ahrefs-top-pages-limit", type=int, default=1000,
                       help="Rows to request from Ahrefs top-pages (default: 1000)")
    run_p.add_argument("--ahrefs-keywords-limit", type=int, default=1000,
                       help="Rows to request from Ahrefs organic-keywords (default: 1000)")
    run_p.add_argument("--ahrefs-semantic-sample", type=int, default=500,
                       help="Max entities per type in the search-demand semantic map (default: 500)")
    run_p.add_argument("--dataforseo-refresh", action="store_true",
                       help="Ignore cached DataForSEO snapshots and fetch fresh API data")
    run_p.add_argument("--dataforseo-location-code", type=int, default=None,
                       help="Optional DataForSEO location code, e.g. 2840 for United States")
    run_p.add_argument("--dataforseo-location-name", default=None,
                       help="Optional DataForSEO location name, e.g. United States")
    run_p.add_argument("--dataforseo-language-code", default=None,
                       help="Optional DataForSEO language code, e.g. en")
    run_p.add_argument("--dataforseo-language-name", default=None,
                       help="Optional DataForSEO language name, e.g. English")
    run_p.add_argument("--dataforseo-top-pages-limit", type=int, default=1000,
                       help="Rows to request from DataForSEO relevant-pages (default: 1000)")
    run_p.add_argument("--dataforseo-keywords-limit", type=int, default=1000,
                       help="Rows to request from DataForSEO ranked-keywords (default: 1000)")
    run_p.add_argument("--dataforseo-include-clickstream", action="store_true",
                       help="Request DataForSEO clickstream data where supported")
    run_p.add_argument("--competitive", default=None, help="TSV file with `query<TAB>competitor_url` per line")
    run_p.add_argument("--queries-file", default=None, help="Optional file: one target query per line")
    run_p.add_argument("--auto-queries-max", type=int, default=200)
    run_p.add_argument("--coverage-threshold", type=float, default=0.55, help="Min similarity for query→page to count as 'covered'")
    run_p.add_argument("--cannibalization-threshold", type=float, default=0.72, help="Pages above this similarity competing for the same query")
    run_p.add_argument("--link-similarity-threshold", type=float, default=0.85, help="Min similarity for an internal-link recommendation")
    run_p.add_argument("--link-recommendations", type=int, default=75)
    run_p.set_defaults(func=_run_command)

    cmp_p = sub.add_parser("compare", help="Build a side-by-side comparison HTML across multiple already-crawled domains")
    cmp_p.add_argument("domains", nargs="*", help="Domains to compare (use --all to compare every project)")
    cmp_p.add_argument("--all", action="store_true", help="Compare every domain with a completed report under projects/")
    cmp_p.add_argument("--projects-root", default="projects")
    cmp_p.add_argument("--name", default="latest", help="Subdir under projects/_compare/ to write into (default: latest)")
    cmp_p.set_defaults(func=_compare_command)

    serve_p = sub.add_parser("serve", help="Serve the local viewer for a previously-generated report")
    serve_p.add_argument("domain", help="Domain whose report to serve")
    serve_p.add_argument("--projects-root", default="projects")
    serve_p.add_argument("--output-dir", default=None)
    serve_p.add_argument("--ui-dir", default=None)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.set_defaults(func=_serve_command)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
