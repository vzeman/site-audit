from dataclasses import dataclass

from site_audit.report import write_sitemap_coverage_exports
from site_audit.sitemap_coverage import analyze


@dataclass
class _Fetched:
    url: str
    status: int = 200


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
    assert payload["summary"]["sitemap_indexable"] == 1
    assert payload["summary"]["crawled_not_in_sitemap"] == 1
    assert payload["summary"]["sitemap_fetch_coverage_share"] == 2 / 3
    assert payload["summary"]["sitemap_indexable_share"] == 1 / 3
    by_url = {row["url"]: row for row in payload["rows"]}
    assert by_url["https://example.com/"]["coverage_status"] == "sitemap_indexable"
    assert by_url["https://example.com/noindex"]["coverage_status"] == "sitemap_non_indexable"
    assert by_url["https://example.com/noindex"]["indexability_issues"] == ["noindex"]
    assert by_url["https://example.com/not-fetched"]["coverage_status"] == "sitemap_not_fetched"
    assert by_url["https://example.com/crawled-only"]["coverage_status"] == "crawled_not_in_sitemap"
    assert "Remove non-indexable" in by_url["https://example.com/noindex"]["recommended_action"]
    assert {issue["issue"] for issue in payload["issues"]} == {
        "sitemap_non_indexable",
        "sitemap_not_fetched",
        "crawled_not_in_sitemap",
    }


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
