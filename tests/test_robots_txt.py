from site_audit.robots_txt import analyze


def test_robots_txt_analyzer_flags_syntax_error() -> None:
    payload = analyze(
        "https://example.com/robots.txt",
        200,
        "User-agent: *\nDisallow /private\nSitemap: https://example.com/sitemap.xml\n",
    )

    assert payload["issues"] == ["robots_txt_has_syntax_error"]
    assert payload["syntax_errors"][0]["line"] == 2


def test_robots_txt_analyzer_allows_valid_and_extension_directives() -> None:
    payload = analyze(
        "https://example.com/robots.txt",
        200,
        "User-agent: *\nDisallow: /private\nClean-param: sid\n",
    )

    assert payload["issues"] == []
    assert payload["syntax_errors"] == []
