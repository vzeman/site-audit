"""Command-line interface: ``site-audit run <domain>`` / ``serve <domain>``.

Two subcommands:

* ``run`` crawls + analyzes + writes reports to ``output/<domain>/``.
* ``serve`` starts a local viewer at http://127.0.0.1:8765/ that loads
  the report files and renders the same scatterplot as the Hugo site.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .embedder import DEFAULT_MODEL
from .pipeline import PipelineConfig, run
from .server import serve


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_command(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        domain=args.domain,
        output_root=Path(args.output_root),
        cache_root=Path(args.cache_root),
        model=args.model,
        max_pages=args.max_pages,
        max_workers=args.workers,
        duplicate_threshold=args.duplicate_threshold,
        duplicate_knn=args.duplicate_knn,
        scatter_clusters=args.scatter_clusters,
        follow_subdomains=args.follow_subdomains,
        respect_robots=not args.ignore_robots,
        use_http_cache=not args.no_http_cache,
        use_embedding_cache=not args.no_embedding_cache,
        skip_scatterplot=args.no_scatterplot,
        max_chars=args.max_chars,
    )
    summary = run(config)
    if summary.get("pages", 0) == 0:
        print("No pages were processed — check the domain and try again.")
        return 1
    print("\nDone.")
    print(f"  pages:        {summary['pages']}")
    print(f"  focus score:  {summary['site_focus_score']:.4f}")
    print(f"  radius:       {summary['site_radius']:.4f}")
    print(f"  outliers:     {summary['outliers']}")
    print(f"  dup. pairs:   {summary['duplicate_pairs']}")
    print(f"  report dir:   {summary['output_dir']}")
    print("\nLaunch the viewer with:")
    print(f"  site-audit serve {summary['domain']}")
    return 0


def _serve_command(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve().parent.parent
    ui_dir = Path(args.ui_dir) if args.ui_dir else here / "ui"
    serve(
        domain=args.domain,
        output_root=Path(args.output_root),
        ui_dir=ui_dir,
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="site-audit", description="Crawl any website, embed its pages, surface duplicates and outliers.")
    p.add_argument("--version", action="version", version=f"site-audit {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Crawl + analyze a domain")
    run_p.add_argument("domain", help="Domain or URL, e.g. example.com or https://example.com")
    run_p.add_argument("--output-root", default="output", help="Where reports are written (default: output/)")
    run_p.add_argument("--cache-root", default="cache", help="Where caches live (default: cache/)")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    run_p.add_argument("--max-pages", type=int, default=2000)
    run_p.add_argument("--workers", type=int, default=8)
    run_p.add_argument("--duplicate-threshold", type=float, default=0.92)
    run_p.add_argument("--duplicate-knn", type=int, default=10)
    run_p.add_argument("--scatter-clusters", type=int, default=30)
    run_p.add_argument("--max-chars", type=int, default=4000, help="Max body chars passed to the embedder")
    run_p.add_argument("--follow-subdomains", action="store_true")
    run_p.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (use sparingly)")
    run_p.add_argument("--no-http-cache", action="store_true")
    run_p.add_argument("--no-embedding-cache", action="store_true")
    run_p.add_argument("--no-scatterplot", action="store_true", help="Skip UMAP projection")
    run_p.set_defaults(func=_run_command)

    serve_p = sub.add_parser("serve", help="Serve the local viewer for a previously-generated report")
    serve_p.add_argument("domain", help="Domain whose report to serve")
    serve_p.add_argument("--output-root", default="output")
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
