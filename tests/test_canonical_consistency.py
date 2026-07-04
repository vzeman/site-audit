from site_audit.canonical_consistency import analyze
from site_audit.report import write_canonical_consistency_exports


def test_canonical_consistency_flags_missing_external_non_self_and_bad_targets() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/",
                "title": "Home",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/",
            },
            {
                "url": "https://example.com/missing",
                "title": "Missing",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "",
            },
            {
                "url": "https://example.com/external",
                "title": "External",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://other.example/external",
            },
            {
                "url": "https://example.com/variant",
                "title": "Variant",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/",
            },
            {
                "url": "https://example.com/to-noindex",
                "title": "Target Noindex",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/noindex",
            },
            {
                "url": "https://example.com/noindex",
                "title": "Noindex",
                "status": "skipped",
                "reason": "noindex",
                "http_status": 200,
                "canonical_url": "https://example.com/noindex",
            },
        ],
        {
            "per_page": [
                {"url": "https://example.com/", "indexability_status": "indexable"},
                {"url": "https://example.com/noindex", "indexability_status": "noindex"},
            ]
        },
    )

    assert payload["summary"]["total_pages"] == 6
    assert payload["summary"]["pages_with_canonical_issues"] == 4
    assert payload["summary"]["missing_canonical"] == 1
    assert payload["summary"]["canonical_external_host"] == 1
    assert payload["summary"]["canonical_non_self"] == 3
    assert payload["summary"]["canonical_target_not_crawled"] == 1
    assert payload["summary"]["canonical_target_non_indexable"] == 1
    assert payload["summary"]["canonical_target_shared"] == 2
    by_url = {row["url"]: row for row in payload["rows"]}
    assert by_url["https://example.com/"]["canonical_status"] == "ok"
    assert by_url["https://example.com/missing"]["issues"] == ["missing_canonical"]
    assert "canonical_external_host" in by_url["https://example.com/external"]["issues"]
    assert "canonical_target_not_crawled" in by_url["https://example.com/external"]["issues"]
    assert "canonical_target_shared" in by_url["https://example.com/variant"]["issues"]
    assert "canonical_target_non_indexable" in by_url["https://example.com/to-noindex"]["issues"]
    assert payload["interpretation"]["how_to_use"]


def test_canonical_consistency_flags_canonical_points_to_4xx() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/source",
                "title": "Source",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/broken",
            },
            {
                "url": "https://example.com/broken",
                "title": "",
                "status": "skipped",
                "reason": "non_2xx_status",
                "http_status": 404,
                "canonical_url": "",
            },
        ],
        {
            "per_page": [
                {"url": "https://example.com/source", "indexability_status": "indexable", "http_status": 200},
                {"url": "https://example.com/broken", "indexability_status": "not_analyzed", "http_status": 404},
            ]
        },
    )

    by_url = {row["url"]: row for row in payload["rows"]}
    assert "canonical_points_to_4xx" in by_url["https://example.com/source"]["issues"]
    assert by_url["https://example.com/source"]["canonical_target_http_status"] == 404
    assert payload["summary"]["canonical_points_to_4xx"] == 1
    issue = next(row for row in payload["issues"] if row["issue"] == "canonical_points_to_4xx")
    assert issue["canonical_target_http_status"] == 404


def test_canonical_consistency_flags_canonical_points_to_5xx() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/source",
                "title": "Source",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/error",
            },
            {
                "url": "https://example.com/error",
                "title": "",
                "status": "skipped",
                "reason": "non_2xx_status",
                "http_status": 503,
                "canonical_url": "",
            },
        ],
        {
            "per_page": [
                {"url": "https://example.com/source", "indexability_status": "indexable", "http_status": 200},
                {"url": "https://example.com/error", "indexability_status": "not_analyzed", "http_status": 503},
            ]
        },
    )

    by_url = {row["url"]: row for row in payload["rows"]}
    assert "canonical_points_to_5xx" in by_url["https://example.com/source"]["issues"]
    assert by_url["https://example.com/source"]["canonical_target_http_status"] == 503
    assert payload["summary"]["canonical_points_to_5xx"] == 1


def test_canonical_consistency_flags_canonical_points_to_redirect() -> None:
    payload = analyze(
        [
            {
                "url": "https://example.com/source",
                "title": "Source",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/redirecting",
            },
            {
                "url": "https://example.com/final",
                "title": "Final",
                "status": "analyzed",
                "http_status": 200,
                "canonical_url": "https://example.com/final",
                "requested_url": "https://example.com/redirecting",
                "redirect_target_url": "https://example.com/final",
            },
        ],
        {
            "per_page": [
                {"url": "https://example.com/source", "indexability_status": "indexable", "http_status": 200},
                {"url": "https://example.com/final", "indexability_status": "indexable", "http_status": 200},
            ]
        },
    )

    by_url = {row["url"]: row for row in payload["rows"]}
    assert "canonical_points_to_redirect" in by_url["https://example.com/source"]["issues"]
    assert by_url["https://example.com/source"]["canonical_redirect_target_url"] == "https://example.com/final"
    assert payload["summary"]["canonical_points_to_redirect"] == 1


def test_canonical_consistency_exports_json_and_csv(tmp_path) -> None:
    payload = {
        "summary": {"total_pages": 1},
        "rows": [{"url": "https://example.com/", "issues": []}],
        "issues": [{"url": "https://example.com/missing", "issue": "missing_canonical"}],
    }

    write_canonical_consistency_exports(tmp_path, payload)

    assert (tmp_path / "canonical_consistency.json").exists()
    assert "https://example.com/" in (tmp_path / "canonical_consistency.csv").read_text(encoding="utf-8")
    assert "missing_canonical" in (tmp_path / "canonical_consistency_issues.csv").read_text(encoding="utf-8")
