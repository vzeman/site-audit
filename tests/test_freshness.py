from datetime import date
from pathlib import Path

from site_audit.compare import build_payload
from site_audit.extractor import ExtractedPage, extract
from site_audit.freshness import analyze, to_payload


def _page(
    url: str,
    title: str = "Freshness test page",
    date_candidates=None,
) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="",
        body="Useful content for testing freshness.",
        word_count=120,
        language="en",
        date_candidates=date_candidates or [],
        has_dates=bool(date_candidates),
    )


def test_extract_date_candidates_from_meta_time_and_jsonld() -> None:
    html = """
    <html><head>
      <title>Freshness extraction test</title>
      <meta property="article:published_time" content="2024-03-15T09:30:00+00:00">
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"Article","dateModified":"2024-04-20"}
      </script>
    </head><body>
      <h1>Freshness extraction test</h1>
      <time datetime="2024-04-01">April 1, 2024</time>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It discusses content freshness, publication dates, update timestamps,
      JSON LD article metadata, and visible time elements in enough detail to
      pass the minimum body threshold during extraction tests.</p>
    </body></html>
    """

    page = extract("https://example.com/freshness", html, max_chars=2000)

    assert page is not None
    assert page.has_dates is True
    assert page.date_published == "2024-03-15"
    assert page.date_modified == "2024-04-20"
    assert {"date": "2024-04-01", "source": "time", "kind": "visible"} in page.date_candidates
    assert {"date": "2024-04-20", "source": "jsonld:Article.dateModified", "kind": "modified"} in page.date_candidates


def test_extract_schema_date_candidates_from_graph_and_value_objects() -> None:
    html = """
    <html><head>
      <title>Graph freshness extraction test</title>
      <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@graph": [
            {"@type": "WebPage", "name": "Container"},
            {
              "@type": "NewsArticle",
              "headline": "Schema date article",
              "datePublished": {"@value": "2025-09-14T13:33:04+02:00"},
              "dateModified": {"@value": "2025-09-14T14:01:40+02:00"}
            }
          ]
        }
      </script>
    </head><body>
      <h1>Graph freshness extraction test</h1>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It checks schema.org NewsArticle dateModified and datePublished values
      when they are nested in an @graph and represented as value objects.</p>
    </body></html>
    """

    page = extract("https://example.com/graph-freshness", html, max_chars=2000)

    assert page is not None
    assert {"date": "2025-09-14", "source": "jsonld:NewsArticle.datePublished", "kind": "published"} in page.date_candidates
    assert {"date": "2025-09-14", "source": "jsonld:NewsArticle.dateModified", "kind": "modified"} in page.date_candidates


def test_freshness_payload_buckets_stale_missing_and_future_dates() -> None:
    report = to_payload(analyze([
        _page("https://example.com/fresh", date_candidates=[
            {"date": "2025-12-01", "source": "meta:article:modified_time", "kind": "modified"},
        ]),
        _page("https://example.com/stale", date_candidates=[
            {"date": "2024-06-01", "source": "jsonld:datePublished", "kind": "published"},
        ]),
        _page("https://example.com/old", date_candidates=[
            {"date": "2022-01-01", "source": "time", "kind": "visible"},
        ]),
        _page("https://example.com/missing"),
        _page("https://example.com/future", date_candidates=[
            {"date": "2026-06-01", "source": "body", "kind": "visible"},
        ]),
    ], today=date(2026, 1, 1)))

    summary = report["summary"]
    assert summary["total_pages"] == 5
    assert summary["pages_with_date"] == 4
    assert summary["date_coverage"] == 0.8
    assert summary["missing_dates"] == 1
    assert summary["pages_stale"] == 2
    assert summary["pages_very_stale"] == 1
    assert summary["future_dates"] == 1
    assert report["buckets"]["fresh"] == 1
    assert report["buckets"]["stale"] == 1
    assert report["buckets"]["very_stale"] == 1
    assert report["buckets"]["unknown"] == 1
    assert report["buckets"]["future"] == 1

    rows = {row["url"]: row for row in report["per_page"]}
    assert rows["https://example.com/stale"]["issues"] == ["stale"]
    assert rows["https://example.com/old"]["issues"] == ["very_stale"]
    assert rows["https://example.com/missing"]["issues"] == ["missing_date"]
    assert rows["https://example.com/future"]["issues"] == ["future_date"]


def test_freshness_prefers_latest_modified_date_over_published() -> None:
    report = to_payload(analyze([
        _page("https://example.com/post", date_candidates=[
            {"date": "2020-01-01", "source": "jsonld:datePublished", "kind": "published"},
            {"date": "2025-10-15", "source": "jsonld:dateModified", "kind": "modified"},
        ]),
    ], today=date(2026, 1, 1)))

    row = report["per_page"][0]
    assert row["date"] == "2025-10-15"
    assert row["date_kind"] == "modified"
    assert row["bucket"] == "fresh"


def test_compare_leaderboard_includes_freshness_metrics(tmp_path: Path) -> None:
    for domain, date_coverage, stale_share in [
        ("a.example", 0.8, 0.1),
        ("b.example", 0.4, 0.5),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":5}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "freshness.json").write_text(
            (
                '{"summary":{"date_coverage":%s,'
                '"stale_share":%s,'
                '"total_pages":5,'
                '"missing_dates":2,'
                '"pages_very_stale":1,'
                '"median_age_days":120},'
                '"buckets":{"fresh":2,"aging":1,"stale":1,"unknown":1}}'
            ) % (date_coverage, stale_share),
            encoding="utf-8",
        )
        (report_dir / "freshness_impact.json").write_text(
            '{"summary":{"traffic_at_risk":25,"high_impact_sections":2,"avg_freshness_risk":55},'
            '"clusters":[{"label":"support","avg_freshness_risk":55,"max_priority_score":140,'
            '"traffic_at_risk":25,"sections":3,"stale_sections":2}]}',
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["freshness_date_coverage"] == 0.8
    assert rows["a.example"]["freshness_stale_share"] == 0.1
    assert rows["a.example"]["freshness_missing_dates"] == 2
    assert rows["b.example"]["freshness_date_coverage"] == 0.4
    assert rows["b.example"]["freshness_stale_share"] == 0.5
    assert rows["a.example"]["freshness_impact_traffic_at_risk"] == 25
    assert rows["a.example"]["freshness_high_impact_sections"] == 2
    dist = {row["domain"]: row for row in payload["distributions"]}
    assert dist["a.example"]["freshness_buckets"]["fresh"] == 2
    assert dist["a.example"]["freshness_summary"]["total_pages"] == 5
    assert payload["freshness_impact"]["clusters"][0]["cluster"] == "support"
