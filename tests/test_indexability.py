from dataclasses import dataclass
from pathlib import Path

from site_audit.compare import build_payload
from site_audit.indexability import analyze, to_payload


@dataclass
class _Fetched:
    url: str
    status: int = 200


def test_indexability_payload_counts_funnel_and_reasons() -> None:
    fetched = [
        _Fetched("https://example.com/"),
        _Fetched("https://example.com/noindex"),
        _Fetched("https://example.com/bad"),
    ]
    rows = [
        {"url": "https://example.com/", "status": "analyzed", "reason": "", "http_status": 200},
        {
            "url": "https://example.com/noindex",
            "status": "skipped",
            "reason": "noindex",
            "source": "meta",
            "http_status": 200,
        },
        {
            "url": "https://example.com/bad",
            "status": "skipped",
            "reason": "unusable",
            "http_status": 200,
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
    assert payload["status_counts"] == {"200": 3}
    assert payload["noindex_pages"][0]["url"] == "https://example.com/noindex"


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
