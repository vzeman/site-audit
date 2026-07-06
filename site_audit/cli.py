"""Command-line interface: ``site-audit run <domain>`` / ``serve <domain>``.

Two subcommands:

* ``run`` crawls + analyzes + writes reports to ``projects/<domain>/report/``.
* ``serve`` starts a local viewer that loads that report directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import sys
import textwrap
import warnings
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .ai_agent import DEFAULT_OPENROUTER_MODEL, openrouter_model
from . import compare as _compare
from .benchmark import benchmark_callable, fingerprint_files, write_benchmark
from .cache import HttpCache, domain_slug
from .config_env import apply_env_defaults
from .embedder import DEFAULT_MODEL
from .history import compare_snapshots, list_snapshots, save_report_snapshot, write_history_html
from .pipeline import PipelineConfig, project_paths, run
from .server import serve
from .serp_gap import SerpGapConfig, run as run_serp_gap
from .settings_ui import serve_settings_ui


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        return
    for logger_name in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "faiss",
        "faiss.loader",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"n_jobs value 1 overridden to 1 by setting random_state.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*get_extended_attention_mask.*deprecated.*",
        category=UserWarning,
    )
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as transformers_logging  # type: ignore
        transformers_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_logging  # type: ignore
        hub_logging.set_verbosity_error()
    except Exception:
        pass


def _run_command(args: argparse.Namespace) -> int:
    # `--clean` is a single-flag shortcut for "wipe the project's cache and
    # re-run from scratch". It deletes the cache directory (HTTP + embedding
    # + paragraph npz) before the pipeline starts; the `--no-*-cache` flags
    # only *bypass* the caches, they don't reset them. Use --clean when the
    # crawler / extractor / embedder logic itself has changed and you want
    # the new logic applied to every page.
    if not args.domain:
        print("run needs a domain. Pass it as an argument or set SITE_AUDIT_RUN_DOMAIN in .env.")
        return 1

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

    preset = args.preset
    technical_only = args.technical_only or preset == "technical"
    allow_large_embeddings = args.allow_large_embeddings or preset == "full-content"

    config = PipelineConfig(
        domain=args.domain,
        projects_root=Path(args.projects_root),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model=args.model,
        max_pages=args.max_pages,
        max_workers=args.workers,
        link_parse_processes=args.link_parse_processes,
        extraction_workers=args.extraction_workers,
        analysis_workers=args.analysis_workers,
        adaptive_concurrency=not args.no_adaptive_concurrency,
        min_crawl_workers=args.min_crawl_workers,
        adaptive_success_threshold=args.adaptive_success_threshold,
        adaptive_slow_seconds=args.adaptive_slow_seconds,
        adaptive_max_rss_mb=args.adaptive_max_rss_mb,
        resume=args.resume,
        write_checkpoints=not args.no_checkpoints,
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
        embed_body_chars=args.embed_body_chars,
        embed_max_seq_length=args.embed_max_seq_length,
        embedding_batch_size=args.embedding_batch_size,
        audit_preset=preset,
        technical_only=technical_only,
        allow_large_embeddings=allow_large_embeddings,
        large_site_embedding_threshold=args.large_site_embedding_threshold,
        enable_cluster_labels=not args.no_cluster_labels,
        enable_keyword_coverage=not args.no_keyword_coverage,
        enable_answerability=not args.no_answerability,
        enable_answer_blocks=not args.no_answer_blocks,
        enable_chunk_retrievability=not args.no_chunk_retrievability,
        enable_freshness_impact=not args.no_freshness_impact,
        enable_cannibalization=not args.no_cannibalization,
        enable_duplicate_fragments=not args.no_duplicate_fragments,
        enable_template_patterns=not args.no_template_patterns,
        enable_trust_signals=not args.no_trust_signals,
        enable_conversion_balance=not args.no_conversion_balance,
        enable_linkgraph=not args.no_linkgraph,
        enable_external_links=not args.no_external_links,
        enable_paragraph_links=not args.no_paragraph_links,
        enable_paragraph_clustering=not args.no_paragraph_clustering,
        enable_weak_paragraphs=not args.no_weak_paragraphs,
        enable_heading_impact=not args.no_heading_impact,
        enable_entity_coverage=not args.no_entity_coverage,
        enable_information_gain=not args.no_information_gain,
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
        save_snapshot=not args.no_snapshot,
        enable_gsc=not args.no_gsc,
        enable_google_ads=not args.no_google_ads,
        use_google_ads_keywords=args.use_google_ads_keywords,
        gsc_property_url=args.gsc_property_url,
        gsc_start_date=args.gsc_start_date,
        gsc_end_date=args.gsc_end_date,
        gsc_top_pages_limit=args.gsc_top_pages_limit,
        gsc_keywords_limit=args.gsc_keywords_limit,
        gsc_refresh=args.gsc_refresh,
        google_ads_customer_id=args.google_ads_customer_id,
        google_ads_login_customer_id=args.google_ads_login_customer_id,
        google_ads_start_date=args.google_ads_start_date,
        google_ads_end_date=args.google_ads_end_date,
        google_ads_search_terms_limit=args.google_ads_search_terms_limit,
        google_ads_min_cost=args.google_ads_min_cost,
        google_ads_refresh=args.google_ads_refresh,
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
        competitive_auto=args.competitive_auto,
        competitive_auto_clusters=args.competitive_auto_clusters,
        competitive_auto_keywords_per_cluster=args.competitive_auto_keywords_per_cluster,
        competitive_auto_results_per_keyword=args.competitive_auto_results_per_keyword,
        competitive_auto_min_relevance=args.competitive_auto_min_relevance,
        competitive_auto_min_position=args.competitive_auto_min_position,
        competitive_auto_max_position=args.competitive_auto_max_position,
        competitive_auto_product_seeds=args.competitive_auto_product_seed,
        competitive_auto_allow_nonlatin=args.competitive_auto_allow_nonlatin,
        competitive_auto_refresh_serp=args.competitive_auto_refresh_serp,
    )
    summary = run(config)
    if summary.get("pages", 0) == 0:
        print("No pages were processed — check the domain and try again.")
        return 1
    print("\nDone.")
    if summary.get("status") in {"technical_only", "stopped_before_large_embedding"}:
        print(f"  status:              {summary['status']}")
        if summary.get("message"):
            print(f"  note:                {summary['message']}")
        print(f"  pages:               {summary['pages']}")
        print(f"  technical issues:    {summary.get('technical_issues', 0)}")
        print(f"  high issues:         {summary.get('high_technical_issues', 0)}")
        print(f"  link edges:          {summary.get('linkgraph_edges', 0)}")
        print(f"  orphans:             {summary.get('linkgraph_orphans', 0)}")
        print(f"  cited domains:       {summary.get('external_domains', 0)}")
        print(f"  broken outbound:     {summary.get('broken_external', 0)}")
        print(f"  report dir:          {summary['report_dir']}")
        return 0
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


def _cache_migrate_command(args: argparse.Namespace) -> int:
    if not args.domain and not args.cache_dir:
        print("cache-migrate needs a domain or --cache-dir.")
        return 1
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cfg = PipelineConfig(domain=args.domain, projects_root=Path(args.projects_root))
        cache_dir, _ = project_paths(cfg)
    cache_path = cache_dir / "http.sqlite"
    if not cache_path.exists():
        print(f"HTTP cache not found: {cache_path}")
        return 1

    cache = HttpCache(cache_path)
    before = cache.stats()
    print(f"Migrating HTTP cache: {cache_path}")
    print(f"  rows:         {before.get('entries', 0)}")
    print(f"  sqlite size:  {before.get('sqlite_size_bytes', 0):,} bytes")
    print(f"  body size:    {before.get('body_size_bytes', 0):,} bytes")

    last_reported = 0

    def progress(row: dict) -> None:
        nonlocal last_reported
        processed = int(row.get("processed") or 0)
        if processed - last_reported < args.progress_interval and processed != int(row.get("total") or 0):
            return
        last_reported = processed
        print(
            f"  processed {processed:,}/{int(row.get('total') or 0):,} "
            f"(moved {int(row.get('moved') or 0):,})"
        )

    result = cache.migrate_bodies_to_files(
        batch_size=args.batch_size,
        keep_backup=not args.delete_original,
        progress_callback=progress,
    )
    after = cache.stats()
    print("Done.")
    print(f"  rows:         {result['rows']:,}")
    print(f"  moved:        {result['moved']:,}")
    print(f"  preserved:    {result['preserved']:,}")
    print(f"  body bytes:   {result['body_bytes_moved']:,}")
    print(f"  sqlite size:  {after.get('sqlite_size_bytes', 0):,} bytes")
    print(f"  body size:    {after.get('body_size_bytes', 0):,} bytes")
    if result.get("backup_path"):
        print(f"  backup:       {result['backup_path']}")
    return 0


def _serp_gap_command(args: argparse.Namespace) -> int:
    if getattr(args, "menu", False):
        if not _run_serp_gap_menu(args):
            return 0
    if not args.domain:
        print("serp-gap needs a domain.")
        return 1
    _maybe_prompt_openrouter_key(args)
    config = SerpGapConfig(
        domain=args.domain,
        projects_root=Path(args.projects_root),
        model=args.model,
        urls=args.url,
        url_include_patterns=args.url_include,
        url_exclude_patterns=args.url_exclude,
        keyword_source=args.keyword_source,
        keywords=args.keyword,
        keywords_file=Path(args.keywords_file) if args.keywords_file else None,
        keywords_per_page=args.keywords_per_page,
        results_per_keyword=args.results_per_keyword,
        max_pages=args.max_pages,
        max_competitor_pages=args.max_competitor_pages,
        max_paragraphs_per_page=args.max_paragraphs_per_page,
        provider=args.provider,
        country=args.country,
        language=args.language,
        min_ranking_position=args.min_ranking_position,
        max_ranking_position=args.max_ranking_position,
        min_impressions=args.min_impressions,
        min_traffic=args.min_traffic,
        use_h1_keyword=args.use_h1_keyword,
        include_serp_keyword_suggestions=args.include_serp_keyword_suggestions,
        max_serp_keyword_suggestions=args.max_serp_keyword_suggestions,
        use_ahrefs_metrics=args.use_ahrefs_metrics,
        ahrefs_refresh=args.ahrefs_refresh,
        ahrefs_date=args.ahrefs_date,
        ahrefs_country=args.ahrefs_country,
        ahrefs_mode=args.ahrefs_mode,
        ahrefs_top_pages_limit=args.ahrefs_top_pages_limit,
        ahrefs_keywords_limit=args.ahrefs_keywords_limit,
        refresh_serp=args.refresh_serp,
        refresh_competitors=args.refresh_competitors,
        budget_usd=args.budget_usd,
        dry_run=args.dry_run,
        ai_agent=args.ai_agent,
        ai_agent_provider=args.ai_agent_provider,
        ai_agent_model=_resolved_ai_agent_model(args.ai_agent_model),
        ai_agent_refresh=args.ai_agent_refresh,
    )
    payload = run_serp_gap(config)
    status = payload.get("status", "unknown")
    if status == "missing_base_report":
        print(payload.get("message", "No existing audit report found."))
        print(f"Run: site-audit run {args.domain} --search-provider all")
        return 1
    if status in {"missing_serper_api_key", "no_pages"}:
        print(payload.get("message", status))
        return 1

    summary = payload.get("summary") or {}
    out_dir = Path(summary.get("report_dir") or (Path(args.projects_root) / domain_slug(args.domain) / "serp_gap" / "report"))
    print("\nSERP gap complete." if status == "ok" else f"\nSERP gap status: {status}")
    print(f"  pages selected:      {summary.get('pages_selected', 0)}")
    print(f"  pages analyzed:      {summary.get('pages_analyzed', 0)}")
    print(f"  keywords selected:   {summary.get('keywords_selected', 0)}")
    print(f"  SERP API calls:      {summary.get('serp_api_calls', 0)} total / {summary.get('serp_api_calls_after_cache', 0)} uncached")
    print(f"  URLs downloaded:     {summary.get('urls_downloaded', 0)}")
    print(f"  competitor URLs est: {summary.get('competitor_urls_estimated', 0)}")
    print(f"  missing topics:      {summary.get('missing_topics', 0)}")
    language_info = payload.get("language_detection") or {}
    if language_info:
        language = language_info.get("language") or "provider default"
        source = language_info.get("source") or language_info.get("status") or ""
        print(f"  SERP language:       {language}" + (f" · {source}" if source else ""))
    agent = payload.get("ai_agent") or {}
    if agent.get("enabled"):
        print(
            f"  AI agent:            {agent.get('status', '')} · "
            f"{agent.get('provider', '')} · {agent.get('editor_briefs', 0)} editor brief(s)"
        )
    print(f"  report JSON:         {out_dir / 'serp_gap.json'}")
    print(f"  TODO markdown:       {out_dir / 'serp_gap_todo.md'}")
    print(f"  HTML report:         {out_dir / 'index.html'}")
    return 0 if status in {"ok", "dry_run"} else 1


def _serve_command(args: argparse.Namespace) -> int:
    if not args.domain:
        print("serve needs a domain. Pass it as an argument or set SITE_AUDIT_SERVE_DOMAIN in .env.")
        return 1
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


def _maybe_prompt_openrouter_key(args: argparse.Namespace) -> None:
    if not getattr(args, "ai_agent", False) or getattr(args, "dry_run", False):
        return
    if not getattr(args, "ai_agent_interactive_setup", True):
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    import getpass
    import os

    from .config_env import load_dotenv, update_env_file

    load_dotenv()
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return
    print("OPENROUTER_API_KEY is required for AI-authored serp-gap keyword selection and editor briefs.")
    key = getpass.getpass("Paste OpenRouter API key (hidden, leave empty to skip AI calls): ").strip()
    if not key:
        return
    env_file = Path(".env")
    updates = {"OPENROUTER_API_KEY": key}
    model = _resolved_ai_agent_model(getattr(args, "ai_agent_model", DEFAULT_OPENROUTER_MODEL))
    if model:
        updates["OPENROUTER_MODEL"] = model
    update_env_file(env_file, updates)
    os.environ["OPENROUTER_API_KEY"] = key
    if updates.get("OPENROUTER_MODEL"):
        os.environ["OPENROUTER_MODEL"] = updates["OPENROUTER_MODEL"]
    print(f"Saved OpenRouter settings to {env_file}.")


def _resolved_ai_agent_model(value: str) -> str:
    if value == DEFAULT_OPENROUTER_MODEL:
        return openrouter_model(DEFAULT_OPENROUTER_MODEL)
    return value


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_SELECTED = "\033[1;36m"


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb":
        return text
    return f"{code}{text}{_ANSI_RESET}"


def _menu_title(title: str, description: str = "") -> None:
    print()
    for line in _box_lines(title, description):
        print(line)


def _terminal_width() -> int:
    return max(64, shutil.get_terminal_size((100, 24)).columns)


def _menu_box_width() -> int:
    return min(92, max(64, _terminal_width() - 4))


def _frame_line(text: str, width: int) -> str:
    inner_width = width - 4
    clipped = text[:inner_width]
    return f"| {clipped.ljust(inner_width)} |"


def _box_lines(title: str, description: str = "", *, footer: str = "") -> list[str]:
    width = _menu_box_width()
    border = "+" + "-" * (width - 2) + "+"
    lines = [_color(border, _ANSI_DIM), _color(_frame_line(title, width), _ANSI_BOLD + _ANSI_CYAN), _color(border, _ANSI_DIM)]
    if description:
        for wrapped in textwrap.wrap(description, width=width - 4):
            lines.append(_frame_line(wrapped, width))
    if footer:
        lines.append(_color(border, _ANSI_DIM))
        for wrapped in textwrap.wrap(footer, width=width - 4):
            lines.append(_frame_line(wrapped, width))
    lines.append(_color(border, _ANSI_DIM))
    return lines


def _choice_lines(
    label: str,
    description: str,
    choices: list[tuple[str, str, str]],
    index: int,
) -> list[str]:
    width = _menu_box_width()
    border = "+" + "-" * (width - 2) + "+"
    lines = [
        _color(border, _ANSI_DIM),
        _color(_frame_line(label, width), _ANSI_BOLD + _ANSI_CYAN),
        _color(border, _ANSI_DIM),
    ]
    for wrapped in textwrap.wrap(description, width=width - 4):
        lines.append(_frame_line(wrapped, width))
    lines.append(_frame_line("", width))
    for pos, (_, name, help_text) in enumerate(choices):
        marker = ">" if pos == index else " "
        row = _frame_line(f"{marker} {pos + 1}. {name}", width)
        lines.append(_color(row, _ANSI_SELECTED) if pos == index else row)
        if pos == index and help_text:
            for help_line in textwrap.wrap(help_text, width=width - 8):
                lines.append(_color(_frame_line(f"    {help_line}", width), _ANSI_DIM))
    lines.append(_frame_line("", width))
    lines.append(_frame_line("Keys: Up/Down or j/k move | Enter select | number jumps | q/Esc cancel", width))
    lines.append(_color(border, _ANSI_DIM))
    return lines


def _can_use_keyboard_menu() -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if os.getenv("SITE_AUDIT_SIMPLE_MENU"):
        return False
    try:
        import termios
        termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        return False
    return True


def _read_menu_key() -> str:
    import select

    ch = sys.stdin.read(1)
    if ch in {"\r", "\n"}:
        return "enter"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in {"\x04", "q", "Q"}:
        return "cancel"
    if ch in {"j", "J"}:
        return "down"
    if ch in {"k", "K"}:
        return "up"
    if ch == " ":
        return "space"
    if ch.isdigit():
        return ch
    if ch == "\x1b":
        if not select.select([sys.stdin], [], [], 0.04)[0]:
            return "cancel"
        nxt = sys.stdin.read(1)
        if nxt != "[":
            return "cancel"
        code = sys.stdin.read(1)
        return {
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
        }.get(code, "cancel")
    return ch


def _clear_rendered_lines(count: int) -> None:
    if count <= 0:
        return
    sys.stdout.write(f"\033[{count}F")
    for _ in range(count):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{count}F")


def _render_keyboard_choice(
    label: str,
    description: str,
    choices: list[tuple[str, str, str]],
    default_index: int,
    *,
    cancel_index: int | None = None,
) -> int | None:
    if not _can_use_keyboard_menu():
        return None

    import termios
    import tty

    index = max(0, min(default_index, len(choices) - 1))
    result_index = index
    rendered_lines = 0

    def lines_for() -> list[str]:
        return ["", *_choice_lines(label, description, choices, index)]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    sys.stdout.write("\033[?25l")
    try:
        tty.setcbreak(fd)
        while True:
            if rendered_lines:
                _clear_rendered_lines(rendered_lines)
            current_lines = lines_for()
            sys.stdout.write("\n".join(current_lines) + "\n")
            sys.stdout.flush()
            rendered_lines = len(current_lines)
            key = _read_menu_key()
            if key in {"up", "left"}:
                index = (index - 1) % len(choices)
            elif key in {"down", "right"}:
                index = (index + 1) % len(choices)
            elif key == "enter":
                result_index = index
                return index
            elif key == "cancel":
                result_index = cancel_index if cancel_index is not None else default_index
                return result_index
            elif key.isdigit():
                number = int(key)
                if 1 <= number <= len(choices):
                    index = number - 1
                    result_index = index
                    return index
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[?25h")
        if rendered_lines:
            _clear_rendered_lines(rendered_lines)
        selected = choices[result_index if 0 <= result_index < len(choices) else default_index][1]
        sys.stdout.write(f"{_color(label, _ANSI_BOLD + _ANSI_CYAN)}\n")
        sys.stdout.write(f"  {_color('Selected:', _ANSI_GREEN)} {selected}\n")
        sys.stdout.flush()


def _url_host(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.netloc or parsed.path.split("/", 1)[0]).lower().strip()


def _report_exists_for_domain(domain: str, projects_root: Path) -> bool:
    if not domain:
        return False
    return (projects_root / domain_slug(domain) / "report" / "pages.json").is_file()


def _domain_from_target_url(url: str, projects_root: Path, current_domain: str | None = None) -> str:
    host = _url_host(url)
    candidates: list[str] = []
    if current_domain:
        candidates.append(current_domain)
    if host:
        candidates.append(host)
        if host.startswith("www."):
            candidates.append(host[4:])
        else:
            candidates.append(f"www.{host}")
    seen: set[str] = set()
    unique = [item for item in candidates if item and not (item in seen or seen.add(item))]
    for candidate in unique:
        if _report_exists_for_domain(candidate, projects_root):
            return candidate
    return host or (current_domain or "")


def _run_serp_gap_menu(args: argparse.Namespace) -> bool:
    print()
    for line in _box_lines(
        "SERP Gap guided setup",
        "Paste a URL, choose keyword and SERP settings, then run the page-level competitor gap analysis.",
        footer="Menus support arrows, j/k, number shortcuts, Enter, and q/Esc.",
    ):
        print(line)

    scope = _menu_choice(
        "Target scope",
        "Choose whether to analyze one exact page URL or a group of URLs from an existing audit.",
        [
            ("url", "Exact page URL", "Best for page-level recommendations. The audited domain is derived from the URL host."),
            ("pattern", "URL pattern", "Use a path glob/regex such as /features/* to analyze multiple audited pages."),
        ],
        default="url" if not args.url_include else "pattern",
    )
    if scope == "url":
        url_default = args.url[0] if args.url else ""
        url = _menu_text(
            "Target page URL",
            "Full URL to analyze. The audited project domain is derived from the URL host.",
            url_default,
            required=True,
        )
        args.url = [url] if url else []
        args.url_include = []
        args.url_exclude = []
        args.domain = _domain_from_target_url(url, Path(args.projects_root), args.domain)
        if not args.domain:
            args.domain = _menu_text(
                "Audited domain",
                "Could not derive a domain from the URL. Enter the domain used by `site-audit run`.",
                args.domain or "",
                required=True,
            )
        print(f"  {_color('Project domain:', _ANSI_GREEN)} {args.domain}")
    else:
        args.domain = _menu_text(
            "Audited domain",
            "Domain with an existing projects/<domain>/report/pages.json. Run `site-audit run` first.",
            args.domain or "",
            required=True,
        )
        if not args.domain:
            print("No domain selected.")
            return False
        include_default = args.url_include[0] if args.url_include else ""
        exclude_default = args.url_exclude[0] if args.url_exclude else ""
        include = _menu_text("URL include pattern", "Path glob or regex, for example /features/*.", include_default, required=True)
        exclude = _menu_text("URL exclude pattern", "Optional path glob/regex to skip unwanted URLs.", exclude_default)
        args.url = []
        args.url_include = [include] if include else []
        args.url_exclude = [exclude] if exclude else []

    keyword_mode = _menu_choice(
        "Keyword selection",
        "Controls which search terms drive SERP fetching and competitor comparison.",
        [
            ("ai", "AI auto keywords", "Recommended for URL-only runs. Uses page content/search rows, then OpenRouter agent if available."),
            ("manual", "Manual keywords", "Use when you already know the exact terms the page should rank for."),
            ("search", "Existing search data", "Use GSC/Ahrefs/DataForSEO/Google Ads rows from the base audit."),
            ("h1", "Title/H1 only", "Cheap fallback using the page title or H1 as a synthetic keyword."),
        ],
        default="ai" if args.ai_agent else "search",
    )
    if keyword_mode == "ai":
        args.keyword_source = "auto"
        args.keyword = []
        args.ai_agent = True
    elif keyword_mode == "manual":
        args.keyword_source = "file"
        args.keyword = _menu_list(
            "Manual keywords",
            "Enter comma-separated keywords. Keep this focused; each keyword triggers SERP and competitor work.",
            args.keyword,
            required=True,
        )
    elif keyword_mode == "search":
        args.keyword_source = _menu_choice(
            "Search data source",
            "Pick a specific source or keep auto to use the best available rows.",
            [
                ("auto", "Auto", "Use any available search rows."),
                ("gsc", "Google Search Console", "Uses impressions/clicks/position from GSC exports."),
                ("ahrefs", "Ahrefs", "Uses organic keyword rows when Ahrefs is configured."),
                ("dataforseo", "DataForSEO", "Uses DataForSEO keyword rows from the base audit."),
                ("google_ads", "Google Ads", "Uses paid query rows from the base audit."),
            ],
            default=args.keyword_source if args.keyword_source in {"auto", "gsc", "ahrefs", "dataforseo", "google_ads"} else "auto",
        )
        args.keyword = []
    else:
        args.keyword_source = "h1"
        args.keyword = []
        args.use_h1_keyword = True

    args.provider = _menu_choice(
        "SERP provider",
        "Provider used to fetch live Google results for selected keywords.",
        [
            ("auto", "Auto", "Use Serper if configured, otherwise DataForSEO."),
            ("dataforseo", "DataForSEO", "Best when you need country/location codes and reliable SERP payloads."),
            ("serper", "Serper", "Simple Google SERP API when SERPER_API_KEY is configured."),
        ],
        default=args.provider,
    )
    args.country = _menu_text(
        "Country / location",
        "Optional. For DataForSEO use a location code such as 2840 for United States.",
        args.country or ("2840" if args.provider == "dataforseo" else ""),
    ) or None
    args.language = _menu_text(
        "Language",
        "Optional SERP language code, for example en. Leave empty to let the AI agent detect it from the page.",
        args.language or "",
    ) or None

    args.dry_run = _menu_bool(
        "Dry run",
        "Writes selected pages/keywords and cost plan without calling SERP providers or fetching competitors.",
        args.dry_run,
    )

    advanced = _menu_bool(
        "Show advanced options",
        "Enable this to tune limits, keyword expansion, Ahrefs metrics, cache refresh, and AI-provider settings.",
        False,
    )
    if advanced:
        args.keywords_per_page = _menu_int(
            "Keywords per page",
            "Caps analysis cost and keeps each page focused on the strongest target terms.",
            args.keywords_per_page,
        )
        args.results_per_keyword = _menu_int(
            "Results per keyword",
            "How many top organic competitors to fetch and compare for each keyword.",
            args.results_per_keyword,
        )
        args.max_pages = _menu_int(
            "Max selected pages",
            "Safety limit for URL pattern runs.",
            args.max_pages,
        )
        args.max_competitor_pages = _menu_int(
            "Max competitor pages",
            "Global cap on external competitor pages downloaded in one run.",
            args.max_competitor_pages,
        )
        args.include_serp_keyword_suggestions = _menu_bool(
            "Add SERP keyword suggestions",
            "Also analyze People Also Ask/Search suggestions. Useful for topic discovery, but increases cost.",
            args.include_serp_keyword_suggestions,
        )
        args.use_ahrefs_metrics = _menu_bool(
            "Use Ahrefs metrics",
            "Attaches Ahrefs position, traffic, and volume when AHREFS_API_KEY is configured.",
            args.use_ahrefs_metrics,
        )
        if args.use_ahrefs_metrics:
            args.ahrefs_country = _menu_text("Ahrefs country", "Optional country database such as US, GB, or SK.", args.ahrefs_country or "") or None
            args.ahrefs_refresh = _menu_bool("Refresh Ahrefs cache", "Fetch a fresh Ahrefs snapshot instead of reusing compatible cache.", args.ahrefs_refresh)
        args.ai_agent = _menu_bool(
            "Enable AI agent",
            "Infers URL keywords and writes paragraph-level markdown TODO briefs with draft copy.",
            args.ai_agent,
        )
        if args.ai_agent:
            args.ai_agent_provider = _menu_choice(
                "AI agent provider",
                "Harnext runs the coding-agent CLI; OpenRouter direct uses chat completions only. Both require OPENROUTER_API_KEY.",
                [
                    ("harnext", "Harnext", "Default coding-agent workflow. Requires the harnext Python SDK and CLI."),
                    ("openrouter", "OpenRouter direct", "Call OpenRouter chat completions directly."),
                ],
                default=args.ai_agent_provider,
            )
            args.ai_agent_model = _menu_text(
                "AI agent model",
                "OpenRouter model that writes keyword and paragraph-level recommendations.",
                args.ai_agent_model or DEFAULT_OPENROUTER_MODEL,
            )
            args.ai_agent_refresh = _menu_bool(
                "Refresh AI completions",
                "Ignore cached agent completions and request new output from the provider.",
                args.ai_agent_refresh,
            )
        args.refresh_serp = _menu_bool("Refresh SERP cache", "Call the SERP provider even when cached SERP payloads exist.", args.refresh_serp)
        args.refresh_competitors = _menu_bool("Refresh competitor pages", "Refetch competitor HTML instead of reusing cached copies.", args.refresh_competitors)
        args.budget_usd = _menu_float("Budget cap USD", "Optional estimated SERP API budget cap. Leave empty for no cap.", args.budget_usd)

    print()
    for line in _box_lines("Equivalent CLI command", _serp_gap_command_preview(args)):
        print(line)
    return _menu_bool("Run now", "Executes the command with the settings above.", True)


def _menu_text(label: str, description: str, default: str = "", *, required: bool = False) -> str:
    _menu_title(label, description)
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"  {_color('Value', _ANSI_YELLOW)}{suffix}: ").strip() or default
        if value or not required:
            return value
        print(_color("  This value is required.", _ANSI_YELLOW))


def _menu_list(label: str, description: str, default: list[str] | None = None, *, required: bool = False) -> list[str]:
    current = ", ".join(default or [])
    raw = _menu_text(label, description, current, required=required)
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def _menu_choice(
    label: str,
    description: str,
    choices: list[tuple[str, str, str]],
    *,
    default: str,
    cancel_value: str | None = None,
) -> str:
    default_index = next((i for i, item in enumerate(choices, start=1) if item[0] == default), 1)
    cancel_index = None
    if cancel_value is not None:
        cancel_index = next((i for i, item in enumerate(choices) if item[0] == cancel_value), default_index - 1)
    keyboard_index = _render_keyboard_choice(
        label,
        description,
        choices,
        default_index - 1,
        cancel_index=cancel_index,
    )
    if keyboard_index is not None:
        return choices[keyboard_index][0]

    _menu_title(label, description)
    for index, (_, name, help_text) in enumerate(choices, start=1):
        marker = "*" if index == default_index else " "
        print(f"  {index}. [{marker}] {_color(name, _ANSI_BOLD)} - {help_text}")
    while True:
        raw = input(f"  {_color('Choose', _ANSI_YELLOW)} 1-{len(choices)} [{default_index}]: ").strip()
        if not raw:
            return choices[default_index - 1][0]
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if 1 <= index <= len(choices):
            return choices[index - 1][0]
        print(_color("  Enter a valid number.", _ANSI_YELLOW))


def _menu_bool(label: str, description: str, default: bool) -> bool:
    if _can_use_keyboard_menu():
        value = _menu_choice(
            label,
            description,
            [
                ("true", "[x] Yes", "Enable this option."),
                ("false", "[ ] No", "Keep this option disabled."),
            ],
            default="true" if default else "false",
        )
        return value == "true"

    state = "[x]" if default else "[ ]"
    _menu_title(f"{state} {label}", description)
    raw = input(f"  Enable? [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"1", "y", "yes", "true", "on"}


def _menu_int(label: str, description: str, default: int) -> int:
    while True:
        raw = _menu_text(label, description, str(default))
        try:
            return int(raw)
        except ValueError:
            print(_color("  Enter an integer.", _ANSI_YELLOW))


def _menu_float(label: str, description: str, default: float | None) -> float | None:
    raw = _menu_text(label, description, "" if default is None else str(default))
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("  Invalid number; leaving budget cap empty.")
        return None


def _serp_gap_command_preview(args: argparse.Namespace) -> str:
    parts = ["site-audit", "serp-gap", args.domain or ""]
    for url in args.url or []:
        parts.extend(["--url", url])
    for pattern in args.url_include or []:
        parts.extend(["--url-include", pattern])
    for pattern in args.url_exclude or []:
        parts.extend(["--url-exclude", pattern])
    parts.extend(["--keyword-source", args.keyword_source])
    for keyword in args.keyword or []:
        parts.extend(["--keyword", keyword])
    parts.extend(["--provider", args.provider])
    if args.country:
        parts.extend(["--country", args.country])
    if args.language:
        parts.extend(["--language", args.language])
    if args.keywords_per_page != 3:
        parts.extend(["--keywords-per-page", str(args.keywords_per_page)])
    if args.results_per_keyword != 5:
        parts.extend(["--results-per-keyword", str(args.results_per_keyword)])
    if args.max_pages != 20:
        parts.extend(["--max-pages", str(args.max_pages)])
    if args.max_competitor_pages != 100:
        parts.extend(["--max-competitor-pages", str(args.max_competitor_pages)])
    if args.use_h1_keyword:
        parts.append("--use-h1-keyword")
    if args.use_ahrefs_metrics:
        parts.append("--use-ahrefs-metrics")
    if args.ahrefs_refresh:
        parts.append("--ahrefs-refresh")
    if args.ahrefs_country:
        parts.extend(["--ahrefs-country", args.ahrefs_country])
    if args.include_serp_keyword_suggestions:
        parts.append("--include-serp-keyword-suggestions")
        if args.max_serp_keyword_suggestions != 8:
            parts.extend(["--max-serp-keyword-suggestions", str(args.max_serp_keyword_suggestions)])
    if not args.ai_agent:
        parts.append("--no-ai-agent")
    elif args.ai_agent_provider != "harnext":
        parts.extend(["--ai-agent-provider", args.ai_agent_provider])
    if args.ai_agent and args.ai_agent_model != DEFAULT_OPENROUTER_MODEL:
        parts.extend(["--ai-agent-model", args.ai_agent_model])
    if args.dry_run:
        parts.append("--dry-run")
    if args.budget_usd is not None:
        parts.extend(["--budget-usd", str(args.budget_usd)])
    if args.refresh_serp:
        parts.append("--refresh-serp")
    if args.refresh_competitors:
        parts.append("--refresh-competitors")
    if args.ai_agent_refresh:
        parts.append("--ai-agent-refresh")
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def _run_main_menu(parser: argparse.ArgumentParser) -> int:
    choice = _menu_choice(
        "Site Audit",
        "Choose a workflow. Each guided path explains the main options before it runs.",
        [
            ("serp-gap", "SERP gap analysis", "Analyze one URL against SERP competitors and generate AI-agent TODO tasks."),
            ("run", "Run domain audit", "Crawl a domain and create the base report required by SERP gap."),
            ("serve", "Open report viewer", "Serve an existing report in the local browser UI."),
            ("settings", "Settings", "Open the local .env editor for API keys and defaults."),
            ("exit", "Exit", "Do nothing."),
        ],
        default="serp-gap",
        cancel_value="exit",
    )
    if choice == "exit":
        return 0
    args = parser.parse_args([choice, "--menu"] if choice == "serp-gap" else [choice])
    apply_env_defaults(args, parser, [choice, "--menu"] if choice == "serp-gap" else [choice])
    if choice == "run":
        if not _configure_run_menu(args):
            return 0
    elif choice == "serve":
        if not _configure_serve_menu(args):
            return 0
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


def _configure_run_menu(args: argparse.Namespace) -> bool:
    args.domain = _menu_text(
        "Domain to audit",
        "Domain or full URL to crawl and analyze, for example www.example.com.",
        args.domain or "",
        required=True,
    )
    args.max_pages = _menu_int(
        "Max pages",
        "Safety limit for crawl size. Use a smaller number for a quick first audit.",
        args.max_pages,
    )
    args.search_provider = _menu_choice(
        "Search data provider",
        "Adds keyword/search-demand overlays to the base report when credentials are configured.",
        [
            ("auto", "Auto", "Use GSC first, then other configured providers as fallback."),
            ("all", "All", "Combine all configured search providers."),
            ("gsc", "Google Search Console", "Use GSC clicks, impressions, and average position."),
            ("ahrefs", "Ahrefs", "Use Ahrefs traffic and volume data."),
            ("dataforseo", "DataForSEO", "Use DataForSEO keyword data."),
            ("none", "None", "Skip search-demand enrichment."),
        ],
        default=args.search_provider,
    )
    args.no_search_data = args.search_provider == "none"
    return _menu_bool("Run audit now", "Starts crawling and report generation with these settings.", True)


def _configure_serve_menu(args: argparse.Namespace) -> bool:
    args.domain = _menu_text(
        "Domain report to serve",
        "Domain slug under projects/<domain>/report. Use the same domain passed to `site-audit run`.",
        args.domain or "",
        required=True,
    )
    return _menu_bool("Start local viewer now", "Starts the report server and keeps running until interrupted.", True)


def _settings_command(args: argparse.Namespace) -> int:
    parser = build_parser()
    serve_settings_ui(
        parser,
        env_file=Path(args.env_file),
        host=args.host,
        port=args.port,
        projects_root=Path(args.projects_root),
        ui_dir=Path(args.ui_dir) if args.ui_dir else None,
    )
    return 0


def _history_snapshot_command(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    cfg = PipelineConfig(domain=args.domain, projects_root=projects_root)
    _, report_dir = project_paths(cfg)
    if not report_dir.exists():
        print(f"No report found at {report_dir}. Run `site-audit run {args.domain}` first.")
        return 1
    target = save_report_snapshot(
        args.domain,
        projects_root,
        report_dir,
        snapshot_id=args.id,
        overwrite=args.overwrite,
    )
    print(f"Wrote snapshot {target.name}: {target}")
    return 0


def _history_list_command(args: argparse.Namespace) -> int:
    rows = list_snapshots(args.domain, Path(args.projects_root))
    if not rows:
        print(f"No snapshots for {args.domain}.")
        return 0
    for row in rows:
        print(
            f"{row['snapshot_id']}\t{row.get('created_at', '')}\t"
            f"{row.get('pages', 0)} pages\t{row.get('traffic', 0)} traffic"
        )
    return 0


def _history_compare_command(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    payload = compare_snapshots(
        args.domain,
        args.before,
        args.after,
        projects_root,
        window_days=args.window_days,
    )
    out_dir = projects_root / domain_slug(args.domain) / "history" / (args.name or f"{args.before}_vs_{args.after}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    template = Path(__file__).resolve().parent.parent / "ui" / "history.html"
    if template.is_file():
        out_html = out_dir / "index.html"
        write_history_html(template, payload, out_html)
        print(f"Wrote {out_html}")
    print(f"Wrote {out_dir / 'history.json'}")
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    domain = args.domain
    report_dir = Path(args.report_dir) if args.report_dir else Path(args.projects_root) / domain_slug(domain) / "report"
    if not report_dir.is_dir():
        print(f"Report directory does not exist: {report_dir}")
        return 1
    patterns = args.include or ["*.json", "*.csv", "index.html"]
    files = []
    for pattern in patterns:
        files.extend(path for path in report_dir.glob(pattern) if path.is_file())
    if not files:
        print(f"No benchmarkable files found in {report_dir}")
        return 1

    result = benchmark_callable(
        f"cached-report:{domain}",
        lambda: {"files": len(files), "fingerprint": fingerprint_files(files)},
    )
    out_path = Path(args.output) if args.output else report_dir / "benchmark.json"
    write_benchmark(out_path, result)
    print(f"Wrote benchmark: {out_path}")
    print(f"  files: {len(files)}")
    print(f"  wall seconds: {result.wall_seconds:.3f}")
    print(f"  max RSS MB: {result.max_rss_mb:.1f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="site-audit", description="Crawl any website, embed its pages, surface duplicates, outliers, topic clusters, GEO scoring, and internal-link recommendations.")
    p.add_argument("--version", action="version", version=f"site-audit {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Crawl + analyze a domain")
    run_p.add_argument("domain", nargs="?", help="Domain or URL, e.g. example.com or https://example.com")
    run_p.add_argument("--projects-root", default="projects", help="Where projects live (default: projects/)")
    run_p.add_argument("--cache-dir", default=None, help="Override cache directory")
    run_p.add_argument("--output-dir", default=None, help="Override report directory")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    run_p.add_argument("--max-pages", type=int, default=10000)
    run_p.add_argument("--preset", choices=["technical", "standard", "full-content"], default="standard",
                       help="Audit preset: technical skips embeddings, standard safeguards large crawls, full-content allows large embeddings")
    run_p.add_argument("--technical-only", action="store_true",
                       help="Write crawl/indexability/technical SEO exports and skip semantic embeddings")
    run_p.add_argument("--allow-large-embeddings", action="store_true",
                       help="Allow semantic embedding stages even when the crawl exceeds the large-site threshold")
    run_p.add_argument("--large-site-embedding-threshold", type=int, default=20000,
                       help="Page count above which standard runs stop after technical exports unless large embeddings are allowed")
    run_p.add_argument("--embed-max-seq-length", type=int, default=512,
                       help="Max transformer tokens per embedded text; 0 uses the model default (default: 512)")
    run_p.add_argument("--embedding-batch-size", type=int, default=32,
                       help="Page embedding batch size (default: 32)")
    run_p.add_argument(
        "--workers",
        "--max-workers",
        dest="workers",
        type=int,
        default=0,
        help=(
            "Maximum worker cap for adaptive crawl/extraction/analysis stages; "
            "0 auto-selects from CPU count"
        ),
    )
    run_p.add_argument("--link-parse-processes", type=int, default=0,
                       help="Processes for cached crawl link parsing; 0 auto, 1 disables the process pool")
    run_p.add_argument("--no-adaptive-concurrency", action="store_true",
                       help="Disable crawl worker auto-throttling on timeouts, 429s, and server errors")
    run_p.add_argument("--min-crawl-workers", type=int, default=1,
                       help="Minimum worker count when adaptive crawl concurrency backs off")
    run_p.add_argument("--adaptive-success-threshold", type=int, default=50,
                       help="Successful responses needed before adaptive crawl concurrency increases by one")
    run_p.add_argument("--adaptive-slow-seconds", type=float, default=3.0,
                       help="Back off crawl workers when a live response takes longer than this many seconds")
    run_p.add_argument("--adaptive-max-rss-mb", type=int, default=0,
                       help="Back off crawl workers when process RSS exceeds this MB; default auto-selects a machine-based limit")
    run_p.add_argument("--extraction-workers", type=int, default=0,
                       help="Exact worker override for HTML extraction; 0 lets the adaptive controller choose")
    run_p.add_argument("--analysis-workers", type=int, default=0,
                       help="Exact worker override for independent post-extraction analyses; 0 lets the adaptive controller choose")
    run_p.add_argument("--request-delay", type=float, default=0.0, help="Seconds to sleep before each request (slow down for rate-limited sites)")
    run_p.add_argument("--duplicate-threshold", type=float, default=0.92)
    run_p.add_argument("--duplicate-knn", type=int, default=10)
    run_p.add_argument("--scatter-clusters", type=int, default=30)
    run_p.add_argument("--max-chars", type=int, default=4000)
    run_p.add_argument(
        "--embed-body-chars",
        type=int,
        default=12000,
        help=(
            "Maximum extracted body characters used for page-level embeddings; "
            "0 disables the cap. Can also be overridden with SITE_AUDIT_EMBED_BODY_CHARS."
        ),
    )
    run_p.add_argument("--follow-subdomains", action="store_true")
    run_p.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (use sparingly)")
    run_p.add_argument("--sitemap-url", action="append", default=[],
                       help="Only discover URLs from this sitemap URL; repeat for multiple sitemaps")
    run_p.add_argument("--sitemap-only", action="store_true",
                       help="Fetch only sitemap-discovered URLs; do not enqueue internal page links")
    run_p.add_argument("--strip-header-footer", action=argparse.BooleanOptionalAction, default=True,
                       help="Remove <header> and <footer> HTML before text extraction and link analysis (default)")
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
    run_p.add_argument("--resume", action="store_true",
                       help="Resume from compatible stage checkpoints when available")
    run_p.add_argument("--no-checkpoints", action="store_true",
                       help="Do not write resumable stage checkpoints")
    run_p.add_argument("--no-scatterplot", action="store_true")
    run_p.add_argument("--no-cluster-labels", action="store_true")
    run_p.add_argument("--no-keyword-coverage", action="store_true")
    run_p.add_argument("--no-answerability", action="store_true")
    run_p.add_argument("--no-answer-blocks", action="store_true")
    run_p.add_argument("--no-chunk-retrievability", action="store_true")
    run_p.add_argument("--no-freshness-impact", action="store_true")
    run_p.add_argument("--no-cannibalization", action="store_true")
    run_p.add_argument("--no-duplicate-fragments", action="store_true")
    run_p.add_argument("--no-template-patterns", action="store_true")
    run_p.add_argument("--no-trust-signals", action="store_true")
    run_p.add_argument("--no-conversion-balance", action="store_true")
    run_p.add_argument("--no-linkgraph", action="store_true")
    run_p.add_argument("--no-external-links", action="store_true")
    run_p.add_argument("--no-paragraph-links", action="store_true",
                       help="Skip paragraph-level internal link recommendation embeddings")
    run_p.add_argument("--no-paragraph-clustering", action="store_true")
    run_p.add_argument("--no-weak-paragraphs", action="store_true")
    run_p.add_argument("--no-heading-impact", action="store_true")
    run_p.add_argument("--no-entity-coverage", action="store_true")
    run_p.add_argument("--no-information-gain", action="store_true")
    run_p.add_argument("--no-content-quality", action="store_true")
    run_p.add_argument("--no-paragraph-fanout", action="store_true")
    run_p.add_argument("--check-external", action="store_true", help="HEAD-check every outbound URL (slow, results cached)")
    run_p.add_argument("--search-provider", default="auto",
                       choices=["auto", "all", "combined", "gsc", "google_ads", "ahrefs", "dataforseo", "none"],
                       help="Search-demand data source. auto uses GSC first, then optional Google Ads, then Ahrefs/DataForSEO fallback; all combines every enabled provider")
    run_p.add_argument("--no-search-data", action="store_true",
                       help="Skip all paid search-data provider enrichment")
    run_p.add_argument("--no-gsc", action="store_true",
                       help="Skip Google Search Console enrichment even when GSC credentials are set")
    run_p.add_argument("--no-google-ads", action="store_true",
                       help="Skip Google Ads enrichment even when Google Ads credentials are set")
    run_p.add_argument("--use-google-ads-keywords", action="store_true",
                       help="Allow --search-provider auto to use Google Ads search terms after GSC and before Ahrefs/DataForSEO")
    run_p.add_argument("--no-ahrefs", action="store_true",
                       help="Skip Ahrefs API enrichment even when AHREFS_API_KEY is set")
    run_p.add_argument("--no-dataforseo", action="store_true",
                       help="Skip DataForSEO fallback enrichment even when DATAFORSEO credentials are set")
    run_p.add_argument("--gsc-refresh", action="store_true",
                       help="Ignore cached Google Search Console snapshots and fetch fresh API data")
    run_p.add_argument("--gsc-property-url", default=None,
                       help="GSC property URL, e.g. sc-domain:example.com or https://www.example.com/")
    run_p.add_argument("--gsc-start-date", default=None,
                       help="GSC start date in YYYY-MM-DD. Default: 28-day window ending 3 days ago")
    run_p.add_argument("--gsc-end-date", default=None,
                       help="GSC end date in YYYY-MM-DD. Default: 3 days ago")
    run_p.add_argument("--gsc-top-pages-limit", type=int, default=1000,
                       help="Rows to request from GSC page report (default: 1000)")
    run_p.add_argument("--gsc-keywords-limit", type=int, default=1000,
                       help="Rows to request from GSC query/page report (default: 1000)")
    run_p.add_argument("--google-ads-refresh", action="store_true",
                       help="Ignore cached Google Ads search-term snapshots and fetch fresh API data")
    run_p.add_argument("--google-ads-customer-id", default=None,
                       help="Google Ads customer ID to query, with or without dashes")
    run_p.add_argument("--google-ads-login-customer-id", default=None,
                       help="Optional manager account ID for the login-customer-id header")
    run_p.add_argument("--google-ads-start-date", default=None,
                       help="Google Ads start date in YYYY-MM-DD. Default: 90-day window ending yesterday")
    run_p.add_argument("--google-ads-end-date", default=None,
                       help="Google Ads end date in YYYY-MM-DD. Default: yesterday")
    run_p.add_argument("--google-ads-search-terms-limit", type=int, default=1000,
                       help="Rows to request from Google Ads search_term_view, sorted by spend (default: 1000)")
    run_p.add_argument("--google-ads-min-cost", type=float, default=0.0,
                       help="Minimum spend in account currency for a Google Ads search term to be imported")
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
    run_p.add_argument("--no-snapshot", action="store_true",
                       help="Do not copy this finished report into projects/<domain>/snapshots/")
    run_p.add_argument("--competitive", default=None, help="TSV file with `query<TAB>competitor_url` per line")
    run_p.add_argument("--competitive-auto", action="store_true",
                       help="Auto-select relevant search clusters and fetch top SERP URLs from DataForSEO for paragraph-gap analysis")
    run_p.add_argument("--competitive-auto-clusters", type=int, default=3,
                       help="Max relevant keyword clusters to analyze in --competitive-auto mode (default: 3)")
    run_p.add_argument("--competitive-auto-keywords-per-cluster", type=int, default=1,
                       help="Max keywords per selected cluster to fetch SERPs for (default: 1)")
    run_p.add_argument("--competitive-auto-results-per-keyword", type=int, default=5,
                       help="Top organic SERP URLs per keyword to analyze (default: 5)")
    run_p.add_argument("--competitive-auto-min-relevance", type=float, default=0.35,
                       help="Minimum keyword business-relevance score for auto competitive analysis (default: 0.35)")
    run_p.add_argument("--competitive-auto-min-position", type=int, default=2,
                       help="Only auto-analyze keywords ranking at this position or worse (default: 2)")
    run_p.add_argument("--competitive-auto-max-position", type=int, default=20,
                       help="Only auto-analyze keywords ranking at this position or better (default: 20)")
    run_p.add_argument("--competitive-auto-product-seed", action="append", default=[],
                       help="Product/service seed phrase for relevance filtering; repeat for multiple seeds")
    run_p.add_argument("--competitive-auto-allow-nonlatin", action="store_true",
                       help="Allow non-Latin keywords in auto competitive selection")
    run_p.add_argument("--competitive-auto-refresh-serp", action="store_true",
                       help="Ignore cached DataForSEO SERP snapshots for auto competitive targets")
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

    cache_p = sub.add_parser("cache-migrate", help="Move legacy HTTP bodies from SQLite into cache/http_bodies and rebuild the DB")
    cache_p.add_argument("domain", nargs="?", help="Domain whose projects/<domain>/cache/http.sqlite should be migrated")
    cache_p.add_argument("--projects-root", default="projects")
    cache_p.add_argument("--cache-dir", default=None, help="Direct cache directory containing http.sqlite")
    cache_p.add_argument("--batch-size", type=int, default=500)
    cache_p.add_argument("--progress-interval", type=int, default=5000)
    cache_p.add_argument("--delete-original", action="store_true",
                         help="Delete the legacy SQLite DB after the rebuilt compact DB is in place instead of keeping a .bak")
    cache_p.set_defaults(func=_cache_migrate_command)

    serp_p = sub.add_parser("serp-gap", help="Analyze selected audited pages against live SERP competitors")
    serp_p.add_argument("domain", nargs="?", help="Domain with an existing site-audit report")
    serp_p.add_argument("--projects-root", default="projects")
    serp_p.add_argument("--menu", action="store_true",
                        help="Open an interactive terminal menu that explains and fills common SERP gap options")
    serp_p.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    serp_p.add_argument("--url", action="append", default=[],
                        help="Exact page URL to analyze even if it was not in pages.json; repeat for multiple URLs")
    serp_p.add_argument("--url-include", "--include-url", action="append", default=[],
                        help="URL/path glob or regex to include; repeat for OR matching")
    serp_p.add_argument("--url-exclude", "--exclude-url", action="append", default=[],
                        help="URL/path glob or regex to exclude; repeat for OR matching")
    serp_p.add_argument("--keyword-source", default="auto",
                        choices=["auto", "gsc", "ahrefs", "dataforseo", "google_ads", "h1", "file"],
                        help="Ranking keyword source (default: auto)")
    serp_p.add_argument("--keyword", action="append", default=[],
                        help="Suggested keyword to analyze for every selected page; repeat for multiple keywords")
    serp_p.add_argument("--keywords-file", default=None,
                        help="Optional TSV with url<TAB>keyword rows, or one keyword per line")
    serp_p.add_argument("--keywords-per-page", type=int, default=3)
    serp_p.add_argument("--results-per-keyword", type=int, default=5)
    serp_p.add_argument("--max-pages", type=int, default=20)
    serp_p.add_argument("--max-competitor-pages", type=int, default=100)
    serp_p.add_argument("--max-paragraphs-per-page", type=int, default=80)
    serp_p.add_argument("--provider", default="auto", choices=["auto", "serper", "dataforseo"])
    serp_p.add_argument("--country", default=None, help="SERP country code/name. For DataForSEO, numeric location code is also accepted.")
    serp_p.add_argument("--language", default=None, help="SERP language code, e.g. en. Omit to auto-detect with the AI agent/page language.")
    serp_p.add_argument("--min-ranking-position", type=int, default=1)
    serp_p.add_argument("--max-ranking-position", type=int, default=30)
    serp_p.add_argument("--min-impressions", type=int, default=0)
    serp_p.add_argument("--min-traffic", type=float, default=0.0)
    serp_p.add_argument("--use-h1-keyword", action="store_true",
                        help="Also use the page title/H1 as a synthetic keyword candidate")
    serp_p.add_argument("--include-serp-keyword-suggestions", action="store_true",
                        help="Also analyze People Also Ask and People Also Search keyword suggestions from SERP payloads")
    serp_p.add_argument("--max-serp-keyword-suggestions", type=int, default=8,
                        help="Max SERP keyword suggestions to add per page when --include-serp-keyword-suggestions is enabled")
    serp_p.add_argument("--use-ahrefs-metrics", action="store_true",
                        help="Fetch/reuse Ahrefs metrics and attach matching keyword traffic/volume to the SERP gap report")
    serp_p.add_argument("--ahrefs-refresh", action="store_true",
                        help="Ignore cached Ahrefs snapshots when --use-ahrefs-metrics is enabled")
    serp_p.add_argument("--ahrefs-date", default=None,
                        help="Ahrefs report date in YYYY-MM-DD. Default: reuse latest cache, otherwise today")
    serp_p.add_argument("--ahrefs-country", default=None,
                        help="Optional Ahrefs country code, e.g. US, GB, SK")
    serp_p.add_argument("--ahrefs-mode", default="subdomains",
                        choices=["exact", "prefix", "domain", "subdomains"],
                        help="Ahrefs target mode (default: subdomains)")
    serp_p.add_argument("--ahrefs-top-pages-limit", type=int, default=1000,
                        help="Rows to request from Ahrefs top-pages (default: 1000)")
    serp_p.add_argument("--ahrefs-keywords-limit", type=int, default=1000,
                        help="Rows to request from Ahrefs organic-keywords (default: 1000)")
    serp_p.add_argument("--refresh-serp", action="store_true")
    serp_p.add_argument("--refresh-competitors", action="store_true")
    serp_p.add_argument("--budget-usd", type=float, default=None)
    serp_p.add_argument("--dry-run", action="store_true",
                        help="Write a plan without calling SERP providers or fetching competitors")
    serp_p.add_argument("--ai-agent", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the Harnext/OpenRouter AI agent for URL-only keyword inference, editor TODO briefs, and final article drafts")
    serp_p.add_argument("--ai-agent-provider", default="harnext", choices=["harnext", "openrouter"],
                        help="AI agent provider (default: harnext; use openrouter to bypass the Harnext coding-agent CLI)")
    serp_p.add_argument("--ai-agent-model", default=DEFAULT_OPENROUTER_MODEL,
                        help=f"OpenRouter model for AI-agent tasks (default: {DEFAULT_OPENROUTER_MODEL})")
    serp_p.add_argument("--ai-agent-refresh", action="store_true",
                        help="Ignore cached AI-agent prompts/completions and call the provider again")
    serp_p.add_argument("--no-ai-agent-interactive-setup", dest="ai_agent_interactive_setup", action="store_false",
                        help="Do not prompt for OPENROUTER_API_KEY in interactive runs")
    serp_p.set_defaults(ai_agent_interactive_setup=True)
    serp_p.set_defaults(func=_serp_gap_command)

    serve_p = sub.add_parser("serve", help="Serve the local viewer for a previously-generated report")
    serve_p.add_argument("domain", nargs="?", help="Domain whose report to serve")
    serve_p.add_argument("--projects-root", default="projects")
    serve_p.add_argument("--output-dir", default=None)
    serve_p.add_argument("--ui-dir", default=None)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.set_defaults(func=_serve_command)

    bench_p = sub.add_parser("benchmark", help="Benchmark cached report artifact reads without re-crawling")
    bench_p.add_argument("domain", help="Domain/project slug to benchmark")
    bench_p.add_argument("--projects-root", default="projects")
    bench_p.add_argument("--report-dir", default=None, help="Override report directory")
    bench_p.add_argument("--include", action="append", default=[],
                         help="Glob of report files to include; repeat for multiple patterns")
    bench_p.add_argument("--output", default=None, help="Benchmark JSON output path")
    bench_p.set_defaults(func=_benchmark_command)

    settings_p = sub.add_parser("settings", help="Open a local UI for editing .env-backed defaults")
    settings_p.add_argument("--env-file", default=".env", help="Local env file to edit (default: .env)")
    settings_p.add_argument("--projects-root", default="projects", help="Projects directory for report/comparison links")
    settings_p.add_argument("--ui-dir", default=None, help="UI assets directory for report rendering")
    settings_p.add_argument("--host", default="127.0.0.1")
    settings_p.add_argument("--port", type=int, default=8780)
    settings_p.set_defaults(func=_settings_command)

    hist_p = sub.add_parser("history", help="Store and compare historical snapshots for one domain")
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True)
    hist_snap = hist_sub.add_parser("snapshot", help="Copy the current report into a named historical snapshot")
    hist_snap.add_argument("domain")
    hist_snap.add_argument("--projects-root", default="projects")
    hist_snap.add_argument("--id", default=None, help="Snapshot id. Default: UTC timestamp")
    hist_snap.add_argument("--overwrite", action="store_true")
    hist_snap.set_defaults(func=_history_snapshot_command)

    hist_list = hist_sub.add_parser("list", help="List snapshots for a domain")
    hist_list.add_argument("domain")
    hist_list.add_argument("--projects-root", default="projects")
    hist_list.set_defaults(func=_history_list_command)

    hist_cmp = hist_sub.add_parser("compare", help="Compare two snapshots for one domain")
    hist_cmp.add_argument("domain")
    hist_cmp.add_argument("before")
    hist_cmp.add_argument("after")
    hist_cmp.add_argument("--projects-root", default="projects")
    hist_cmp.add_argument("--name", default=None, help="Output subdir under projects/<domain>/history/")
    hist_cmp.add_argument("--window-days", type=int, default=None,
                          help="Observation window in days between tracked changes and metric movement")
    hist_cmp.set_defaults(func=_history_compare_command)

    return p


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not raw_argv:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return _run_main_menu(parser)
        parser.print_help()
        return 0
    args = parser.parse_args(raw_argv)
    apply_env_defaults(args, parser, raw_argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
