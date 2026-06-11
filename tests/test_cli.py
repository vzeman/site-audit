from site_audit.cli import _run_serp_gap_menu, build_parser


def test_run_parser_accepts_crawl_filter_flags() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "example.com",
            "--sitemap-url",
            "https://example.com/en-sitemap.xml",
            "--sitemap-only",
            "--strip-header-footer",
            "--content-include-class",
            "article-detail-content",
            "--content-exclude-class",
            "sidebar",
            "--url-include",
            "/en/",
            "--url-exclude",
            "/private/",
            "--sitemap-include",
            "en-sitemap",
            "--sitemap-exclude",
            "image-sitemap",
            "--sitemap-lastmod-after",
            "2025-05-04",
            "--sitemap-lastmod-within-days",
            "365",
            "--no-paragraph-links",
            "--search-provider",
            "gsc",
            "--gsc-property-url",
            "sc-domain:example.com",
            "--gsc-start-date",
            "2026-04-01",
            "--gsc-end-date",
            "2026-04-30",
            "--gsc-top-pages-limit",
            "300",
            "--gsc-keywords-limit",
            "600",
            "--gsc-refresh",
            "--no-gsc",
            "--ahrefs-country",
            "US",
            "--ahrefs-date",
            "2026-05-08",
            "--ahrefs-top-pages-limit",
            "250",
            "--ahrefs-keywords-limit",
            "500",
            "--ahrefs-refresh",
            "--no-answer-blocks",
            "--no-freshness-impact",
            "--no-cannibalization",
            "--no-duplicate-fragments",
            "--no-template-patterns",
            "--no-trust-signals",
            "--no-conversion-balance",
            "--no-snapshot",
            "--competitive-auto",
            "--competitive-auto-clusters",
            "5",
            "--competitive-auto-keywords-per-cluster",
            "2",
            "--competitive-auto-results-per-keyword",
            "4",
            "--competitive-auto-min-relevance",
            "0.42",
            "--competitive-auto-min-position",
            "3",
            "--competitive-auto-max-position",
            "15",
            "--competitive-auto-product-seed",
            "AI workflow automation",
            "--competitive-auto-product-seed",
            "AI agents",
            "--competitive-auto-allow-nonlatin",
            "--competitive-auto-refresh-serp",
        ]
    )

    assert args.sitemap_url == ["https://example.com/en-sitemap.xml"]
    assert args.sitemap_only is True
    assert args.strip_header_footer is True
    assert args.content_include_class == ["article-detail-content"]
    assert args.content_exclude_class == ["sidebar"]
    assert args.url_include == ["/en/"]
    assert args.url_exclude == ["/private/"]
    assert args.sitemap_include == ["en-sitemap"]
    assert args.sitemap_exclude == ["image-sitemap"]
    assert args.sitemap_lastmod_after == "2025-05-04"
    assert args.sitemap_lastmod_within_days == 365
    assert args.no_paragraph_links is True
    assert args.search_provider == "gsc"
    assert args.gsc_property_url == "sc-domain:example.com"
    assert args.gsc_start_date == "2026-04-01"
    assert args.gsc_end_date == "2026-04-30"
    assert args.gsc_top_pages_limit == 300
    assert args.gsc_keywords_limit == 600
    assert args.gsc_refresh is True
    assert args.no_gsc is True
    assert args.ahrefs_country == "US"
    assert args.ahrefs_date == "2026-05-08"
    assert args.ahrefs_top_pages_limit == 250
    assert args.ahrefs_keywords_limit == 500
    assert args.ahrefs_refresh is True
    assert args.no_answer_blocks is True
    assert args.no_freshness_impact is True
    assert args.no_cannibalization is True
    assert args.no_duplicate_fragments is True
    assert args.no_template_patterns is True
    assert args.no_trust_signals is True
    assert args.no_conversion_balance is True
    assert args.no_snapshot is True
    assert args.competitive_auto is True
    assert args.competitive_auto_clusters == 5
    assert args.competitive_auto_keywords_per_cluster == 2
    assert args.competitive_auto_results_per_keyword == 4
    assert args.competitive_auto_min_relevance == 0.42
    assert args.competitive_auto_min_position == 3
    assert args.competitive_auto_max_position == 15
    assert args.competitive_auto_product_seed == ["AI workflow automation", "AI agents"]
    assert args.competitive_auto_allow_nonlatin is True
    assert args.competitive_auto_refresh_serp is True


def test_run_parser_strips_header_footer_by_default() -> None:
    args = build_parser().parse_args(["run", "example.com"])

    assert args.strip_header_footer is True


def test_run_parser_accepts_disabling_header_footer_stripping() -> None:
    args = build_parser().parse_args(["run", "example.com", "--no-strip-header-footer"])

    assert args.strip_header_footer is False


def test_serp_gap_parser_accepts_budget_and_keyword_options() -> None:
    args = build_parser().parse_args(
        [
            "serp-gap",
            "example.com",
            "--url",
            "https://www.example.com/",
            "--url-include",
            "/features/*",
            "--keyword",
            "live chat software",
            "--keyword",
            "helpdesk software",
            "--keywords-per-page",
            "2",
            "--results-per-keyword",
            "10",
            "--provider",
            "dataforseo",
            "--budget-usd",
            "5",
            "--include-serp-keyword-suggestions",
            "--max-serp-keyword-suggestions",
            "4",
            "--use-ahrefs-metrics",
            "--ahrefs-refresh",
            "--ahrefs-date",
            "2026-06-01",
            "--ahrefs-country",
            "US",
            "--ahrefs-mode",
            "domain",
            "--ahrefs-top-pages-limit",
            "25",
            "--ahrefs-keywords-limit",
            "50",
            "--ai-agent-provider",
            "openrouter",
            "--ai-agent-model",
            "deepseek/deepseek-v4-pro",
            "--ai-agent-refresh",
            "--no-ai-agent-interactive-setup",
            "--dry-run",
        ]
    )

    assert args.domain == "example.com"
    assert args.url == ["https://www.example.com/"]
    assert args.url_include == ["/features/*"]
    assert args.keyword == ["live chat software", "helpdesk software"]
    assert args.keywords_per_page == 2
    assert args.results_per_keyword == 10
    assert args.provider == "dataforseo"
    assert args.budget_usd == 5
    assert args.include_serp_keyword_suggestions is True
    assert args.max_serp_keyword_suggestions == 4
    assert args.use_ahrefs_metrics is True
    assert args.ahrefs_refresh is True
    assert args.ahrefs_date == "2026-06-01"
    assert args.ahrefs_country == "US"
    assert args.ahrefs_mode == "domain"
    assert args.ahrefs_top_pages_limit == 25
    assert args.ahrefs_keywords_limit == 50
    assert args.ai_agent is True
    assert args.ai_agent_provider == "openrouter"
    assert args.ai_agent_model == "deepseek/deepseek-v4-pro"
    assert args.ai_agent_refresh is True
    assert args.ai_agent_interactive_setup is False
    assert args.dry_run is True


def test_serp_gap_parser_can_disable_ai_agent() -> None:
    args = build_parser().parse_args(["serp-gap", "example.com", "--no-ai-agent"])

    assert args.ai_agent is False


def test_serp_gap_parser_accepts_interactive_menu_without_domain() -> None:
    args = build_parser().parse_args(["serp-gap", "--menu"])

    assert args.menu is True
    assert args.domain is None


def test_serp_gap_menu_fills_common_options(monkeypatch) -> None:
    args = build_parser().parse_args(["serp-gap", "--menu"])
    answers = iter([
        "example.com",
        "1",
        "https://www.example.com/blog/ai-support-paradox/",
        "1",
        "2",
        "2840",
        "",
        "y",
        "n",
        "y",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert _run_serp_gap_menu(args) is True
    assert args.domain == "example.com"
    assert args.url == ["https://www.example.com/blog/ai-support-paradox/"]
    assert args.keyword_source == "auto"
    assert args.ai_agent is True
    assert args.provider == "dataforseo"
    assert args.country == "2840"
    assert args.language == "en"
    assert args.dry_run is True


def test_history_parser_accepts_snapshot_and_compare_commands() -> None:
    snapshot_args = build_parser().parse_args(
        ["history", "snapshot", "example.com", "--projects-root", "/tmp/projects", "--id", "baseline", "--overwrite"]
    )
    compare_args = build_parser().parse_args(
        [
            "history", "compare", "example.com", "baseline", "after",
            "--projects-root", "/tmp/projects", "--name", "baseline-vs-after", "--window-days", "28",
        ]
    )

    assert snapshot_args.history_command == "snapshot"
    assert snapshot_args.domain == "example.com"
    assert snapshot_args.projects_root == "/tmp/projects"
    assert snapshot_args.id == "baseline"
    assert snapshot_args.overwrite is True
    assert compare_args.history_command == "compare"
    assert compare_args.before == "baseline"
    assert compare_args.after == "after"
    assert compare_args.name == "baseline-vs-after"
    assert compare_args.window_days == 28


def test_run_parser_accepts_combined_search_provider() -> None:
    args = build_parser().parse_args(["run", "example.com", "--search-provider", "all"])

    assert args.search_provider == "all"
