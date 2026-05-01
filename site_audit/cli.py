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
        skip_scatterplot=args.no_scatterplot,
        max_chars=args.max_chars,
        enable_cluster_labels=not args.no_cluster_labels,
        enable_keyword_coverage=not args.no_keyword_coverage,
        enable_answerability=not args.no_answerability,
        enable_linkgraph=not args.no_linkgraph,
        enable_external_links=not args.no_external_links,
        enable_paragraph_clustering=not args.no_paragraph_clustering,
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
    run_p.add_argument("--no-paragraph-clustering", action="store_true")
    run_p.add_argument("--no-content-quality", action="store_true")
    run_p.add_argument("--no-paragraph-fanout", action="store_true")
    run_p.add_argument("--check-external", action="store_true", help="HEAD-check every outbound URL (slow, results cached)")
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
