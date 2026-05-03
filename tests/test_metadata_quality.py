from pathlib import Path

from site_audit.compare import build_payload
from site_audit.extractor import ExtractedPage, extract
from site_audit.metadata_quality import analyze, to_payload


def _page(
    url: str,
    title: str = "Useful Search Title",
    description: str = "This description is long enough to be useful in search results.",
    canonical_url: str = "https://example.com/page",
    og_title: str = "OG title",
    og_description: str = "OG description",
    twitter_card: str = "summary",
) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description=description,
        body="Useful content for testing metadata quality.",
        word_count=120,
        language="en",
        canonical_url=canonical_url,
        og_title=og_title,
        og_description=og_description,
        twitter_card=twitter_card,
    )


def test_extract_serp_metadata_fields() -> None:
    html = """
    <html><head>
      <title>Complete metadata example</title>
      <meta name="description" content="A complete description for search result snippets and previews.">
      <link rel="canonical" href="https://example.com/canonical">
      <meta name="robots" content="index,follow">
      <meta property="og:title" content="Open graph title">
      <meta property="og:description" content="Open graph description">
      <meta property="og:image" content="https://example.com/image.png">
      <meta name="twitter:card" content="summary_large_image">
      <meta name="twitter:title" content="Twitter title">
      <meta name="twitter:description" content="Twitter description">
    </head><body>
      <h1>Complete metadata example</h1>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It discusses titles, descriptions, canonical URLs, robots directives,
      Open Graph data, and Twitter cards in enough detail to pass the minimum
      body threshold during extraction tests.</p>
    </body></html>
    """

    page = extract("https://example.com/page", html, max_chars=2000)

    assert page is not None
    assert page.description == "A complete description for search result snippets and previews."
    assert page.canonical_url == "https://example.com/canonical"
    assert page.robots_content == "index,follow"
    assert page.og_title == "Open graph title"
    assert page.og_description == "Open graph description"
    assert page.og_image == "https://example.com/image.png"
    assert page.twitter_card == "summary_large_image"
    assert page.twitter_title == "Twitter title"
    assert page.twitter_description == "Twitter description"


def test_metadata_quality_payload_flags_duplicates_and_missing_fields() -> None:
    report = to_payload(analyze([
        _page("https://example.com/a", title="Same title", description=""),
        _page("https://example.com/b", title="Same title", canonical_url="", og_description=""),
        _page("https://example.com/c", canonical_url="https://other.example/c", twitter_card=""),
    ]))

    assert report["summary"]["total_pages"] == 3
    assert report["summary"]["pages_with_issues"] == 3
    assert report["summary"]["missing_description"] == 1
    assert report["summary"]["duplicate_title_pages"] == 2
    assert report["summary"]["missing_canonical"] == 1
    assert report["summary"]["canonical_external_host"] == 1
    assert report["summary"]["incomplete_open_graph"] == 1
    assert report["summary"]["missing_twitter_card"] == 1
    assert report["issues_by_type"]["duplicate_title"] == 2
    assert any("missing_description" in row["issues"] for row in report["per_page"])


def test_compare_leaderboard_includes_metadata_quality_metrics(tmp_path: Path) -> None:
    for domain, issue_share, missing_description in [
        ("a.example", 0.25, 1),
        ("b.example", 0.75, 3),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":4}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "metadata_quality.json").write_text(
            (
                '{"summary":{"issue_share":%s,'
                '"missing_description":%d,'
                '"duplicate_title_pages":2,'
                '"missing_canonical":1,'
                '"incomplete_open_graph":4}}'
            ) % (issue_share, missing_description),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["metadata_issue_share"] == 0.25
    assert rows["a.example"]["missing_description"] == 1
    assert rows["a.example"]["duplicate_title_pages"] == 2
    assert rows["b.example"]["metadata_issue_share"] == 0.75
    assert rows["b.example"]["missing_description"] == 3
