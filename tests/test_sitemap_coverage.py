from dataclasses import dataclass

from site_audit.report import write_sitemap_coverage_exports
from site_audit.sitemap_coverage import analyze


@dataclass
class _Fetched:
    url: str
    status: int = 200
    requested_url: str = ""
    redirect_target_url: str = ""
    redirect_status_codes: list[int] | None = None
    redirect_hop_count: int = 0
    error: str = ""


def test_sitemap_coverage_classifies_matrix_and_actions() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
            {
                "url": "https://example.com/noindex",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-02",
            },
            {
                "url": "https://example.com/not-fetched",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-03",
            },
        ],
        [
            _Fetched("https://example.com/"),
            _Fetched("https://example.com/noindex"),
            _Fetched("https://example.com/crawled-only"),
        ],
        [
            {"url": "https://example.com/", "title": "Home", "status": "analyzed", "http_status": 200},
            {
                "url": "https://example.com/noindex",
                "title": "Noindex",
                "status": "skipped",
                "reason": "noindex",
                "http_status": 200,
            },
            {
                "url": "https://example.com/crawled-only",
                "title": "Crawled Only",
                "status": "analyzed",
                "http_status": 200,
            },
        ],
        {
            "per_page": [
                {"url": "https://example.com/", "indexability_status": "indexable", "issues": []},
                {"url": "https://example.com/noindex", "indexability_status": "noindex", "issues": ["noindex"]},
                {"url": "https://example.com/crawled-only", "indexability_status": "indexable", "issues": []},
            ]
        },
    )

    assert payload["summary"]["total_sitemap_urls"] == 3
    assert payload["summary"]["fetched_sitemap_urls"] == 2
    assert payload["summary"]["sitemap_not_fetched"] == 1
    assert payload["summary"]["sitemap_non_indexable"] == 1
    assert payload["summary"]["noindex_page_in_sitemap"] == 1
    assert payload["summary"]["sitemap_indexable"] == 1
    assert payload["summary"]["indexable_page_not_in_sitemap"] == 1
    assert payload["summary"]["crawled_not_in_sitemap"] == 1
    assert payload["summary"]["sitemap_fetch_coverage_share"] == 2 / 3
    assert payload["summary"]["sitemap_indexable_share"] == 1 / 3
    by_url = {row["url"]: row for row in payload["rows"]}
    assert by_url["https://example.com/"]["coverage_status"] == "sitemap_indexable"
    assert by_url["https://example.com/noindex"]["coverage_status"] == "sitemap_non_indexable"
    assert by_url["https://example.com/noindex"]["indexability_issues"] == ["noindex"]
    assert by_url["https://example.com/noindex"]["sitemap_issue_types"] == ["noindex_page_in_sitemap"]
    assert by_url["https://example.com/not-fetched"]["coverage_status"] == "sitemap_not_fetched"
    assert by_url["https://example.com/crawled-only"]["coverage_status"] == "crawled_not_in_sitemap"
    assert by_url["https://example.com/crawled-only"]["sitemap_issue_types"] == ["indexable_page_not_in_sitemap"]
    assert "Remove non-indexable" in by_url["https://example.com/noindex"]["recommended_action"]
    assert {issue["issue"] for issue in payload["issues"]} == {
        "noindex_page_in_sitemap",
        "indexable_page_not_in_sitemap",
        "sitemap_non_indexable",
        "sitemap_not_fetched",
        "crawled_not_in_sitemap",
    }


def test_sitemap_coverage_flags_3xx_redirect_in_sitemap() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/old",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
        ],
        [
            _Fetched(
                "https://example.com/final",
                requested_url="https://example.com/old",
                redirect_target_url="https://example.com/final",
                redirect_status_codes=[301],
                redirect_hop_count=1,
            ),
        ],
        [],
        {"per_page": [{"url": "https://example.com/old", "indexability_status": "indexable", "issues": []}]},
    )

    assert payload["summary"]["3xx_redirect_in_sitemap"] == 1
    row = next(row for row in payload["rows"] if row["url"] == "https://example.com/old")
    assert row["url"] == "https://example.com/old"
    assert row["crawled"] is True
    assert row["redirect_target_url"] == "https://example.com/final"
    assert row["redirect_status_codes"] == [301]
    assert row["sitemap_issue_types"] == ["3xx_redirect_in_sitemap"]
    issues = [row for row in payload["issues"] if row["issue"] == "3xx_redirect_in_sitemap"]
    assert len(issues) == 1
    assert issues[0]["redirect_target_url"] == "https://example.com/final"


def test_sitemap_coverage_flags_4xx_page_in_sitemap() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/missing",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
        ],
        [_Fetched("https://example.com/missing", status=404)],
        [{"url": "https://example.com/missing", "status": "skipped", "reason": "non_2xx_status", "http_status": 404}],
        {"per_page": [{"url": "https://example.com/missing", "indexability_status": "not_indexable", "issues": []}]},
    )

    assert payload["summary"]["4xx_page_in_sitemap"] == 1
    row = payload["rows"][0]
    assert row["url"] == "https://example.com/missing"
    assert row["http_status"] == 404
    assert row["sitemap_issue_types"] == ["4xx_page_in_sitemap"]
    issues = [row for row in payload["issues"] if row["issue"] == "4xx_page_in_sitemap"]
    assert len(issues) == 1
    assert "4XX page" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_5xx_page_in_sitemap() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/error",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
        ],
        [_Fetched("https://example.com/error", status=503)],
        [{"url": "https://example.com/error", "status": "skipped", "reason": "non_2xx_status", "http_status": 503}],
        {"per_page": [{"url": "https://example.com/error", "indexability_status": "not_indexable", "issues": []}]},
    )

    assert payload["summary"]["5xx_page_in_sitemap"] == 1
    row = payload["rows"][0]
    assert row["url"] == "https://example.com/error"
    assert row["http_status"] == 503
    assert row["sitemap_issue_types"] == ["5xx_page_in_sitemap"]
    issues = [row for row in payload["issues"] if row["issue"] == "5xx_page_in_sitemap"]
    assert len(issues) == 1
    assert "server error" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_non_canonical_page_in_sitemap() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/duplicate",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
        ],
        [_Fetched("https://example.com/duplicate")],
        [
            {
                "url": "https://example.com/duplicate",
                "title": "Duplicate",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/canonical",
            }
        ],
        {"per_page": [{"url": "https://example.com/duplicate", "indexability_status": "indexable", "issues": []}]},
    )

    assert payload["summary"]["non_canonical_page_in_sitemap"] == 1
    row = payload["rows"][0]
    assert row["canonical_url"] == "https://example.com/canonical"
    assert row["sitemap_issue_types"] == ["non_canonical_page_in_sitemap"]
    issues = [row for row in payload["issues"] if row["issue"] == "non_canonical_page_in_sitemap"]
    assert len(issues) == 1
    assert "canonical URL" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_page_from_sitemap_timed_out() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/slow",
                "source_sitemaps": ["https://example.com/sitemap.xml"],
                "lastmod": "2026-05-01",
            },
        ],
        [_Fetched("https://example.com/slow", status=0, error="timed_out")],
        [{"url": "https://example.com/slow", "status": "skipped", "reason": "timed_out", "http_status": 0}],
        {"per_page": [{"url": "https://example.com/slow", "indexability_status": "timed_out", "issues": []}]},
    )

    assert payload["summary"]["page_from_sitemap_timed_out"] == 1
    row = payload["rows"][0]
    assert row["extraction_reason"] == "timed_out"
    assert row["sitemap_issue_types"] == ["page_from_sitemap_timed_out"]
    issues = [row for row in payload["issues"] if row["issue"] == "page_from_sitemap_timed_out"]
    assert len(issues) == 1
    assert "timeout" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_syntax_error() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/broken-sitemap.xml",
                "issue": "sitemap_has_syntax_error",
                "http_status": "",
                "size_bytes": "",
                "message": "not well-formed",
            }
        ],
    )

    assert payload["summary"]["sitemap_has_syntax_error"] == 1
    assert payload["sitemap_errors"] == [
        {
            "sitemap_url": "https://example.com/broken-sitemap.xml",
            "issue": "sitemap_has_syntax_error",
            "http_status": "",
            "size_bytes": "",
            "url_count": "",
            "message": "not well-formed",
        }
    ]
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_has_syntax_error"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/broken-sitemap.xml"
    assert "XML syntax" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_is_not_accessible() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/missing-sitemap.xml",
                "issue": "sitemap_is_not_accessible",
                "http_status": 404,
                "message": "HTTP 404",
            }
        ],
    )

    assert payload["summary"]["sitemap_is_not_accessible"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_is_not_accessible"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing-sitemap.xml"
    assert issues[0]["http_status"] == 404
    assert "Restore access" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_larger_than_50mb() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/huge-sitemap.xml",
                "issue": "sitemap_larger_than_50mb",
                "http_status": 200,
                "size_bytes": 60 * 1024 * 1024,
                "message": "62914560 bytes",
            }
        ],
    )

    assert payload["summary"]["sitemap_larger_than_50mb"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_larger_than_50mb"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/huge-sitemap.xml"
    assert issues[0]["size_bytes"] == 60 * 1024 * 1024
    assert "50MB" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_with_over_50k_urls() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/large-count-sitemap.xml",
                "issue": "sitemap_with_over_50k_urls",
                "url_count": 50_001,
                "message": "50001 URLs",
            }
        ],
    )

    assert payload["summary"]["sitemap_with_over_50k_urls"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_with_over_50k_urls"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/large-count-sitemap.xml"
    assert issues[0]["url_count"] == 50_001
    assert "50,000 URLs" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_in_wrong_format() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/feed.xml",
                "issue": "sitemap_in_the_wrong_format",
                "http_status": 200,
                "message": "root element rss",
            }
        ],
    )

    assert payload["summary"]["sitemap_in_the_wrong_format"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_in_the_wrong_format"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/feed.xml"
    assert "urlset or sitemapindex" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_includes_urls_out_of_scope() -> None:
    payload = analyze(
        [],
        [],
        [],
        sitemap_errors=[
            {
                "sitemap_url": "https://example.com/sitemap.xml",
                "issue": "sitemap_includes_urls_out_of_its_scope",
                "url_count": 2,
                "message": "2 out-of-scope URLs",
            }
        ],
    )

    assert payload["summary"]["sitemap_includes_urls_out_of_its_scope"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "sitemap_includes_urls_out_of_its_scope"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/sitemap.xml"
    assert issues[0]["url_count"] == 2
    assert "outside the audited site scope" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_sitemap_url_count_decreased() -> None:
    payload = analyze(
        [
            {"url": "https://example.com/a", "source_sitemaps": ["https://example.com/sitemap.xml"]},
            {"url": "https://example.com/b", "source_sitemaps": ["https://example.com/sitemap.xml"]},
        ],
        [],
        [],
        previous_total_sitemap_urls=5,
    )

    assert payload["summary"]["no_of_urls_in_sitemap_decreased"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "no_of_urls_in_sitemap_decreased"]
    assert len(issues) == 1
    assert issues[0]["url"] == "sitemaps://total-urls"
    assert issues[0]["url_count"] == 2
    assert "decrease" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_page_in_multiple_sitemaps() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/page",
                "source_sitemaps": ["https://example.com/a.xml", "https://example.com/b.xml"],
            },
        ],
        [_Fetched("https://example.com/page")],
        [{"url": "https://example.com/page", "status": "analyzed", "http_status": 200}],
        {"per_page": [{"url": "https://example.com/page", "indexability_status": "indexable", "issues": []}]},
    )

    assert payload["summary"]["page_in_multiple_sitemaps"] == 1
    row = payload["rows"][0]
    assert row["source_sitemaps"] == ["https://example.com/a.xml", "https://example.com/b.xml"]
    assert row["sitemap_issue_types"] == ["page_in_multiple_sitemaps"]
    issues = [row for row in payload["issues"] if row["issue"] == "page_in_multiple_sitemaps"]
    assert len(issues) == 1
    assert "one canonical sitemap" in issues[0]["recommended_action"]


def test_sitemap_coverage_flags_pages_added_to_sitemaps() -> None:
    payload = analyze(
        [
            {"url": "https://example.com/existing", "source_sitemaps": ["https://example.com/sitemap.xml"]},
            {"url": "https://example.com/new", "source_sitemaps": ["https://example.com/sitemap.xml"]},
        ],
        [_Fetched("https://example.com/existing"), _Fetched("https://example.com/new")],
        [
            {"url": "https://example.com/existing", "status": "analyzed", "http_status": 200},
            {"url": "https://example.com/new", "status": "analyzed", "http_status": 200},
        ],
        {
            "per_page": [
                {"url": "https://example.com/existing", "indexability_status": "indexable", "issues": []},
                {"url": "https://example.com/new", "indexability_status": "indexable", "issues": []},
            ]
        },
        previous_sitemap_urls=["https://example.com/existing"],
    )

    assert payload["summary"]["pages_added_to_sitemaps"] == 1
    row = next(row for row in payload["rows"] if row["url"] == "https://example.com/new")
    assert row["sitemap_issue_types"] == ["pages_added_to_sitemaps"]
    issues = [row for row in payload["issues"] if row["issue"] == "pages_added_to_sitemaps"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/new"


def test_sitemap_coverage_flags_pages_removed_from_sitemaps() -> None:
    payload = analyze(
        [
            {"url": "https://example.com/existing", "source_sitemaps": ["https://example.com/sitemap.xml"]},
        ],
        [_Fetched("https://example.com/existing")],
        [{"url": "https://example.com/existing", "status": "analyzed", "http_status": 200}],
        {"per_page": [{"url": "https://example.com/existing", "indexability_status": "indexable", "issues": []}]},
        previous_sitemap_urls=["https://example.com/existing", "https://example.com/removed"],
    )

    assert payload["summary"]["pages_removed_from_sitemaps"] == 1
    issues = [row for row in payload["issues"] if row["issue"] == "pages_removed_from_sitemaps"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/removed"
    assert "intentionally dropped" in issues[0]["recommended_action"]


def test_sitemap_coverage_exports_json_and_csv(tmp_path) -> None:
    payload = {
        "summary": {"total_sitemap_urls": 1},
        "rows": [{"url": "https://example.com/", "source_sitemaps": ["https://example.com/sitemap.xml"]}],
        "issues": [{"url": "https://example.com/noindex", "issue": "sitemap_non_indexable"}],
    }

    write_sitemap_coverage_exports(tmp_path, payload)

    assert (tmp_path / "sitemap_coverage.json").exists()
    assert "https://example.com/" in (tmp_path / "sitemap_coverage.csv").read_text(encoding="utf-8")
    assert "sitemap_non_indexable" in (tmp_path / "sitemap_coverage_issues.csv").read_text(encoding="utf-8")
