from dataclasses import dataclass
from pathlib import Path

from site_audit.compare import build_payload
from site_audit.indexability import analyze, to_payload
from site_audit.report import write_indexability_issues_csv


@dataclass
class _Fetched:
    url: str
    status: int = 200
    content_type: str = "text/html"
    x_robots_tag: str = ""


def test_indexability_payload_counts_funnel_and_reasons() -> None:
    fetched = [
        _Fetched("https://example.com/"),
        _Fetched("https://example.com/noindex", x_robots_tag="noindex"),
        _Fetched("https://example.com/bad", status=500),
    ]
    rows = [
        {
            "url": "https://example.com/",
            "title": "Home",
            "status": "analyzed",
            "reason": "",
            "http_status": 200,
            "canonical_url": "https://example.com/",
            "robots_content": "index,follow",
            "word_count": 400,
        },
        {
            "url": "https://example.com/noindex",
            "title": "Noindex",
            "status": "skipped",
            "reason": "noindex",
            "source": "meta",
            "http_status": 200,
            "canonical_url": "https://example.com/noindex",
            "robots_content": "noindex",
            "noindex_source": "meta",
            "word_count": 120,
        },
        {
            "url": "https://example.com/bad",
            "title": "Bad",
            "status": "skipped",
            "reason": "unusable",
            "http_status": 500,
        },
    ]

    payload = to_payload(analyze(fetched, rows, {"https://example.com/"}))

    assert payload["summary"]["fetched_pages"] == 3
    assert payload["summary"]["analyzed_pages"] == 1
    assert payload["summary"]["skipped_pages"] == 2
    assert payload["summary"]["noindex_pages"] == 1
    assert payload["summary"]["indexable_share"] == 1 / 3
    assert payload["summary"]["noindex_share"] == 1 / 3
    assert payload["summary"]["unusable_pages"] == 1
    assert payload["summary"]["indexable_pages"] == 1
    assert payload["summary"]["non_indexable_pages"] == 2
    assert payload["summary"]["pages_with_indexability_issues"] == 2
    assert payload["summary"]["issue_count"] == 3
    assert payload["status_counts"] == {"200": 2, "500": 1}
    assert payload["noindex_pages"][0]["url"] == "https://example.com/noindex"
    assert payload["issue_counts"] == {"noindex": 1, "unusable": 1, "non_2xx_status": 1}
    by_url = {row["url"]: row for row in payload["per_page"]}
    assert by_url["https://example.com/"]["indexability_status"] == "indexable"
    assert by_url["https://example.com/"]["canonical_url"] == "https://example.com/"
    assert by_url["https://example.com/noindex"]["indexability_status"] == "noindex"
    assert by_url["https://example.com/noindex"]["x_robots_tag"] == "noindex"
    assert by_url["https://example.com/noindex"]["issues"] == ["noindex"]
    assert by_url["https://example.com/bad"]["issues"] == ["unusable", "non_2xx_status"]
    assert "remove the noindex" in by_url["https://example.com/noindex"]["recommended_action"]
    assert payload["interpretation"]["how_to_use"]


def test_indexability_issue_csv_export(tmp_path: Path) -> None:
    payload = {
        "issues": [
            {
                "url": "https://example.com/noindex",
                "title": "Noindex",
                "issue": "noindex",
                "http_status": 200,
                "issues": ["noindex"],
                "recommended_action": "Review.",
            }
        ]
    }

    write_indexability_issues_csv(tmp_path, payload)

    csv_text = (tmp_path / "indexability_issues.csv").read_text(encoding="utf-8")
    assert "url,title,issue,http_status,issues,recommended_action" in csv_text
    assert "https://example.com/noindex" in csv_text
    assert "noindex" in csv_text


def test_compare_leaderboard_includes_indexability_metrics(tmp_path: Path) -> None:
    for domain, indexable_share, skipped_pages in [
        ("a.example", 0.8, 2),
        ("b.example", 0.4, 6),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":10}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "indexability.json").write_text(
            (
                '{"summary":{"indexable_share":%s,'
                '"noindex_share":0.1,'
                '"skipped_pages":%d}}'
            ) % (indexable_share, skipped_pages),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["indexable_share"] == 0.8
    assert rows["a.example"]["noindex_share"] == 0.1
    assert rows["a.example"]["skipped_pages"] == 2
    assert rows["b.example"]["indexable_share"] == 0.4
    assert rows["b.example"]["skipped_pages"] == 6
