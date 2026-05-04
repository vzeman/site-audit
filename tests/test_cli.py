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
        ]
    )

    assert args.sitemap_url == ["https://example.com/en-sitemap.xml"]
    assert args.sitemap_only is True
    assert args.strip_header_footer is True
    assert args.url_include == ["/en/"]
    assert args.url_exclude == ["/private/"]
    assert args.sitemap_include == ["en-sitemap"]
    assert args.sitemap_exclude == ["image-sitemap"]
    assert args.sitemap_lastmod_after == "2025-05-04"
    assert args.sitemap_lastmod_within_days == 365
    assert args.no_paragraph_links is True
