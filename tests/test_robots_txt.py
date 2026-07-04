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


def test_robots_txt_analyzer_flags_redirect_loop_error() -> None:
    payload = analyze("https://example.com/robots.txt", 0, "", error="redirect_loop")

    assert payload["issues"] == [
        "robots_txt_has_too_many_redirects_or_redirect_loop",
        "robots_txt_is_not_accessible",
    ]


def test_robots_txt_analyzer_flags_too_many_redirects() -> None:
    payload = analyze(
        "https://example.com/robots.txt",
        200,
        "User-agent: *\nDisallow:\n",
        redirect_status_codes=[301, 301, 302, 301, 302, 301],
    )

    assert payload["issues"] == ["robots_txt_has_too_many_redirects_or_redirect_loop"]
    assert payload["redirect_status_codes"] == [301, 301, 302, 301, 302, 301]


def test_robots_txt_analyzer_flags_not_accessible() -> None:
    payload = analyze("https://example.com/robots.txt", 404, "")

    assert payload["issues"] == ["robots_txt_is_not_accessible"]
    assert payload["status"] == 404
