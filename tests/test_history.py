import json
from pathlib import Path

from site_audit.analyzer import PageInfo
from site_audit.extractor import ExtractedPage
from site_audit.history import build_history_snapshot, compare_snapshots, list_snapshots, save_report_snapshot


def _snapshot_page(title: str, paragraph: str, traffic: int, links: list[str]) -> tuple[list[PageInfo], list[ExtractedPage], dict]:
    page = PageInfo(
        url="https://example.com/a",
        title=title,
        description="Support automation guide",
        section="blog",
        word_count=120,
        language="en",
    )
    extracted = ExtractedPage(
        url=page.url,
        title=title,
        description=page.description,
        body=paragraph,
        word_count=120,
        language="en",
        headers_rich=[{"level": 1, "text": title}, {"level": 2, "text": "Implementation"}],
        paragraphs=[paragraph],
        schema_types=["Article"],
    )
    search = {
        "top_pages": [{
            "matched_url": page.url,
            "traffic": traffic,
            "clicks": traffic // 2,
            "impressions": traffic * 10,
            "keywords": 3,
            "top_keyword": "support automation",
            "top_keyword_position": 5,
        }]
    }
    return [page], [extracted], search


def test_history_snapshot_captures_page_paragraph_link_schema_and_metrics() -> None:
    pages, extracted, search = _snapshot_page("Support automation", "Original paragraph.", 100, ["/b"])

    payload = build_history_snapshot(
        "example.com",
        pages,
        extracted,
        outlinks_map={"https://example.com/a": [("https://example.com/b", "B")]},
        structured_data={"per_page": [{"url": "https://example.com/a", "types": ["Article"], "valid_blocks": 1}]},
        freshness={"per_page": [{"url": "https://example.com/a", "bucket": "fresh", "date": "2026-01-01"}]},
        metadata_quality={"per_page": [{"url": "https://example.com/a", "canonical_url": "https://example.com/a", "issues": []}]},
        search_payload=search,
    )

    row = payload["pages"][0]
    assert row["paragraphs"][0]["hash"]
    assert row["links"] == ["https://example.com/b"]
    assert row["canonical_url"] == "https://example.com/a"
    assert row["canonical_hash"]
    assert row["schema_types"] == ["Article"]
    assert row["metrics"]["traffic"] == 100
    assert row["metrics"]["clicks"] == 50
    assert row["metrics"]["impressions"] == 1000
    assert payload["summary"]["total_traffic"] == 100
    assert payload["summary"]["total_clicks"] == 50
    assert payload["summary"]["total_impressions"] == 1000
    assert payload["summary"]["avg_position"] == 5


def test_compare_snapshots_reports_content_link_schema_metadata_and_metric_deltas(tmp_path: Path) -> None:
    before_pages, before_extracted, before_search = _snapshot_page("Support automation", "Original paragraph.", 100, ["/b"])
    after_pages, after_extracted, after_search = _snapshot_page("Support automation updated", "Updated paragraph with proof.", 155, ["/b", "/c"])
    before_payload = build_history_snapshot(
        "example.com",
        before_pages,
        before_extracted,
        outlinks_map={"https://example.com/a": [("https://example.com/b", "B")]},
        structured_data={"per_page": [{"url": "https://example.com/a", "types": ["Article"], "valid_blocks": 1}]},
        freshness={"per_page": [{"url": "https://example.com/a", "bucket": "stale", "date": "2024-01-01"}]},
        metadata_quality={"per_page": [{"url": "https://example.com/a", "issues": ["missing_description"]}]},
        search_payload=before_search,
        snapshot_id="before",
    )
    after_payload = build_history_snapshot(
        "example.com",
        after_pages,
        after_extracted,
        outlinks_map={"https://example.com/a": [("https://example.com/b", "B"), ("https://example.com/c", "C")]},
        structured_data={"per_page": [{"url": "https://example.com/a", "types": ["Article", "FAQPage"], "valid_blocks": 2}]},
        freshness={"per_page": [{"url": "https://example.com/a", "bucket": "fresh", "date": "2026-02-01"}]},
        metadata_quality={"per_page": [{"url": "https://example.com/a", "issues": []}]},
        search_payload=after_search,
        snapshot_id="after",
    )
    for snapshot_id, payload in [("before", before_payload), ("after", after_payload)]:
        report = tmp_path / "example.com" / "snapshots" / snapshot_id / "report"
        report.mkdir(parents=True)
        (report / "history_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")

    diff = compare_snapshots("example.com", "before", "after", tmp_path)
    row = diff["changes"][0]

    assert diff["summary"]["changed_pages"] == 1
    assert diff["summary"]["clicks_delta"] == 27
    assert diff["summary"]["impressions_delta"] == 550
    assert diff["summary"]["avg_position_delta"] == 0
    assert diff["summary"]["content_changes"] > 0
    assert row["traffic_delta"] == 55
    assert row["clicks_delta"] == 27
    assert row["impressions_delta"] == 550
    assert "paragraphs" in row["changed_fields"]
    assert "links" in row["changed_fields"]
    assert "schema" in row["changed_fields"]
    assert "metadata" in row["changed_fields"]
    assert row["links_added"] == 1
    assert row["schema_added"] == ["FAQPage"]
    assert row["confidence"] in {"medium", "low-medium"}
    assert diff["summary"]["caveats"]


def test_compare_snapshots_reports_canonical_url_changes(tmp_path: Path) -> None:
    before_pages, before_extracted, before_search = _snapshot_page("Support automation", "Original paragraph.", 100, [])
    after_pages, after_extracted, after_search = _snapshot_page("Support automation", "Original paragraph.", 100, [])
    before_payload = build_history_snapshot(
        "example.com",
        before_pages,
        before_extracted,
        metadata_quality={"per_page": [{"url": "https://example.com/a", "canonical_url": "https://example.com/a", "issues": []}]},
        search_payload=before_search,
        snapshot_id="before",
    )
    after_payload = build_history_snapshot(
        "example.com",
        after_pages,
        after_extracted,
        metadata_quality={"per_page": [{"url": "https://example.com/a", "canonical_url": "https://example.com/canonical", "issues": []}]},
        search_payload=after_search,
        snapshot_id="after",
    )
    for snapshot_id, payload in [("before", before_payload), ("after", after_payload)]:
        report = tmp_path / "example.com" / "snapshots" / snapshot_id / "report"
        report.mkdir(parents=True)
        (report / "history_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")

    diff = compare_snapshots("example.com", "before", "after", tmp_path)
    row = diff["changes"][0]

    assert "canonical" in row["changed_fields"]
    assert row["canonical_before"] == "https://example.com/a"
    assert row["canonical_after"] == "https://example.com/canonical"


def test_save_report_snapshot_copies_current_report_and_lists_it(tmp_path: Path) -> None:
    report = tmp_path / "example.com" / "report"
    report.mkdir(parents=True)
    (report / "site_metrics.json").write_text("{}", encoding="utf-8")
    (report / "history_snapshot.json").write_text(json.dumps({"summary": {"pages": 1, "total_traffic": 10}}), encoding="utf-8")

    target = save_report_snapshot("example.com", tmp_path, report, snapshot_id="manual")
    rows = list_snapshots("example.com", tmp_path)

    assert (target / "report" / "site_metrics.json").exists()
    assert rows[0]["snapshot_id"] == "manual"
    assert rows[0]["pages"] == 1
