from site_audit.robots_txt import analyze, evaluate_path, has_blanket_disallow


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


def test_robots_txt_analyzer_flags_changed_content() -> None:
    payload = analyze(
        "https://example.com/robots.txt",
        200,
        "User-agent: *\nDisallow: /new\n",
        previous_body="User-agent: *\nDisallow: /old\n",
    )

    assert payload["issues"] == ["robots_txt_changed"]
    assert payload["content_hash"] != payload["previous_content_hash"]


def test_robots_txt_analyzer_does_not_flag_unchanged_content() -> None:
    body = "User-agent: *\nDisallow: /same\n"
    payload = analyze("https://example.com/robots.txt", 200, body, previous_body=body)

    assert payload["issues"] == []


def test_evaluate_path_matches_user_agent_token_case_insensitively() -> None:
    body = "User-agent: gptbot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"

    decision = evaluate_path(body, "GPTBot", "/")
    assert decision["allowed"] is False
    assert decision["explicitly_named"] is True
    assert evaluate_path(body, "ClaudeBot", "/")["allowed"] is True


def test_evaluate_path_treats_empty_disallow_as_allow_all() -> None:
    body = "User-agent: GPTBot\nDisallow:\n\nUser-agent: *\nDisallow: /\n"

    assert evaluate_path(body, "GPTBot", "/")["allowed"] is True
    assert evaluate_path(body, "CCBot", "/")["allowed"] is False


def test_evaluate_path_ignores_rules_before_user_agent_and_strips_comments() -> None:
    body = (
        "Disallow: /\n"          # rule before any group — ignored
        "# global comment\n"
        "  User-agent: *   # everyone\n"
        "\tDisallow: /private  # trailing comment\n"
    )

    assert evaluate_path(body, "GPTBot", "/")["allowed"] is True
    assert evaluate_path(body, "GPTBot", "/private/x")["allowed"] is False


def test_evaluate_path_supports_wildcard_and_anchor_patterns() -> None:
    body = "User-agent: *\nDisallow: /*.pdf$\nDisallow: /tmp*/x\nAllow: /\n"

    assert evaluate_path(body, "GPTBot", "/")["allowed"] is True
    assert evaluate_path(body, "GPTBot", "/doc.pdf")["allowed"] is False
    assert evaluate_path(body, "GPTBot", "/doc.pdf.html")["allowed"] is True
    assert evaluate_path(body, "GPTBot", "/tmp123/x")["allowed"] is False


def test_evaluate_path_merges_duplicate_groups_for_same_token() -> None:
    body = (
        "User-agent: GPTBot\n"
        "Allow: /public\n\n"
        "User-agent: GPTBot\n"
        "Disallow: /\n\n"
        "User-agent: *\n"
        "Allow: /\n"
    )

    # RFC 9309 section 2.2.1: rules of groups sharing a token are combined.
    assert evaluate_path(body, "GPTBot", "/")["allowed"] is False
    assert evaluate_path(body, "GPTBot", "/public/page")["allowed"] is True


def test_has_blanket_disallow_defeated_by_equal_allow() -> None:
    assert has_blanket_disallow("User-agent: *\nDisallow: /\n") is True
    assert has_blanket_disallow("User-agent: *\nDisallow: /\nAllow: /\n") is False


def test_has_blanket_disallow_detects_wildcard_path_variants() -> None:
    assert has_blanket_disallow("User-agent: *\nDisallow: /*\n") is True
    assert has_blanket_disallow("User-agent: *\nDisallow: *\n") is True
    assert has_blanket_disallow("User-agent: *\nDisallow: /private\n") is False
