from types import SimpleNamespace

from site_audit.report import write_technical_seo_exports
from site_audit.technical_seo import build_technical_seo


def test_technical_seo_model_merges_existing_page_signals() -> None:
    pages = [
        SimpleNamespace(
            url="https://example.com/a",
            title="A",
            section="blog",
            word_count=500,
            language="en",
        )
    ]
    indexability = {"skipped": [], "noindex_pages": []}
    metadata = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "canonical_url": "",
                "robots_content": "",
                "issues": ["missing_canonical", "missing_description"],
            }
        ]
    }
    performance = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "status": 200,
                "weight_bucket": "very_heavy",
                "html_weight_bytes": 700000,
                "estimated_weight_bytes": 3500000,
                "resource_tag_count": 90,
                "render_blocking_count": 4,
            }
        ]
    }
    search = {
        "top_pages": [
            {"matched_url": "https://example.com/a", "traffic": 1200, "keywords": 9, "top_keyword": "example keyword"}
        ]
    }
    page_types = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "page_type": "article",
                "template_family": "content_template",
                "template_signature": "sig",
            }
        ]
    }

    payload = build_technical_seo(
        pages,
        indexability=indexability,
        metadata_quality=metadata,
        performance=performance,
        search_payload=search,
        page_types=page_types,
    )

    assert payload["summary"]["total_pages"] == 1
    assert payload["summary"]["high_issues"] >= 1
    page = payload["pages"][0]
    assert page["url"] == "https://example.com/a"
    assert page["traffic"] == 1200
    assert page["page_type"] == "article"
    assert page["fix_scope"] == "template"
    assert page["technical_issue_count"] >= 4
    issue_types = {row["issue_type"] for row in payload["issues"]}
    assert "missing_canonical" in issue_types
    assert "very_heavy_page" in issue_types
    assert "render_blocking_resources" in issue_types


def test_technical_seo_model_includes_skipped_pages() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                }
            ],
            "noindex_pages": [],
        },
    )

    assert payload["summary"]["total_pages"] == 1
    assert payload["pages"][0]["indexability_status"] == "noindex"
    assert payload["issues"][0]["category"] == "indexability"


def test_technical_seo_model_includes_full_issue_catalog() -> None:
    payload = build_technical_seo([])

    assert payload["summary"]["catalog_issue_types"] >= 150
    names = {row["name"] for row in payload["issue_catalog"]}
    assert "Canonical points to 4XX" in names
    assert "Duplicate pages without canonical" in names
    assert "Structured data has schema.org validation error" in names


def test_write_technical_seo_exports_writes_json_and_csv(tmp_path) -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/a", title="A", section="", word_count=100, language="en")],
        metadata_quality={"per_page": [{"url": "https://example.com/a", "issues": ["missing_title"]}]},
    )

    write_technical_seo_exports(tmp_path, payload)

    assert (tmp_path / "technical_pages.json").is_file()
    assert (tmp_path / "technical_issues.json").is_file()
    assert (tmp_path / "technical_issue_catalog.json").is_file()
    assert (tmp_path / "technical_pages.csv").is_file()
    assert (tmp_path / "technical_issues.csv").is_file()
    assert (tmp_path / "technical_issue_catalog.csv").is_file()
    csv_text = (tmp_path / "technical_pages.csv").read_text(encoding="utf-8")
    assert "technical_severity_score" in csv_text
    assert "missing_title" in csv_text
    issue_json = (tmp_path / "technical_issues.json").read_text(encoding="utf-8")
    assert "issue_catalog" in issue_json
