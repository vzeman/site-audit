from site_audit.cli import build_parser


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
    assert args.ahrefs_country == "US"
    assert args.ahrefs_date == "2026-05-08"
    assert args.ahrefs_top_pages_limit == 250
    assert args.ahrefs_keywords_limit == 500
    assert args.ahrefs_refresh is True
    assert args.no_answer_blocks is True
