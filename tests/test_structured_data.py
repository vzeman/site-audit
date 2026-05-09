from pathlib import Path

from site_audit.compare import build_payload
from site_audit.extractor import ExtractedPage, extract
from site_audit.structured_data import analyze, to_payload


def _page(url: str, schema_types=None, schema_blocks=None) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=url.rsplit("/", 1)[-1] or "Home",
        description="",
        body="Useful content for testing structured data.",
        word_count=120,
        language="en",
        schema_types=schema_types or [],
        schema_blocks=schema_blocks or [],
    )


def _rich_page(url: str, title: str, headings: list[str], body: str, **kwargs) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="Helpful page description.",
        body=body,
        word_count=len(body.split()),
        language="en",
        h1=title,
        h1_count=1,
        headings=headings,
        headers_rich=[
            {"level": 1, "text": title, "order": 0},
            *[{"level": 2, "text": heading, "order": i + 1} for i, heading in enumerate(headings)],
        ],
        paragraphs=["Answer paragraph one.", "Answer paragraph two.", "Answer paragraph three."],
        **kwargs,
    )


def test_extract_jsonld_blocks_with_graph_and_invalid_json() -> None:
    html = """
    <html><head>
      <title>Schema test</title>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@graph":[
          {"@type":"Organization","name":"Acme","url":"https://example.com"},
          {"@type":["FAQPage","WebPage"],"mainEntity":[]}
        ]}
      </script>
      <script type="application/ld+json">{"@type":"Article",</script>
    </head><body>
      <h1>Schema test</h1>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It discusses structured data, JSON LD, organizations, questions, and
      search result eligibility in enough detail to pass the minimum body
      threshold used by the audit extractor during tests.</p>
    </body></html>
    """

    page = extract("https://example.com/schema", html, max_chars=2000)

    assert page is not None
    assert page.schema_types == ["FAQPage", "Organization", "WebPage"]
    assert len(page.schema_blocks) == 2
    assert page.schema_blocks[0]["valid"] is True
    assert page.schema_blocks[0]["types"] == ["FAQPage", "Organization", "WebPage"]
    assert "mainEntity" in page.schema_blocks[0]["keys"]
    assert page.schema_blocks[1]["valid"] is False
    assert page.schema_blocks[1]["error"]


def test_structured_data_payload_summarizes_coverage_and_missing_properties() -> None:
    report = to_payload(analyze([
        _page(
            "https://example.com/blog/post",
            schema_types=["Article"],
            schema_blocks=[{
                "format": "json-ld",
                "valid": True,
                "types": ["Article"],
                "keys": ["@type", "headline", "datePublished"],
                "error": "",
            }],
        ),
        _page(
            "https://example.com/bad",
            schema_blocks=[{
                "format": "json-ld",
                "valid": False,
                "types": [],
                "keys": [],
                "error": "Expecting property name",
            }],
        ),
        _page("https://example.com/no-schema"),
    ]))

    assert report["summary"]["total_pages"] == 3
    assert report["summary"]["pages_with_schema"] == 1
    assert report["summary"]["schema_coverage"] == 1 / 3
    assert report["summary"]["invalid_jsonld_blocks"] == 1
    assert report["summary"]["pages_with_invalid_jsonld"] == 1
    assert report["top_types"] == [{"type": "Article", "pages": 1}]
    assert report["invalid_blocks"][0]["url"] == "https://example.com/bad"
    assert report["missing_recommended"][0]["missing"] == [
        {"type": "Article", "missing": ["author"]}
    ]
    assert report["summary"]["schema_opportunities"] >= 1


def test_structured_data_opportunities_include_schema_evidence_and_google_reference() -> None:
    url = "https://example.com/support/faq"
    report = to_payload(analyze([
        _rich_page(
            url,
            "Support FAQ",
            ["What is live chat?", "How does routing work?", "Can I add automation?"],
            "This FAQ answers common support automation questions with visible answers for every question.",
        )
    ], search_payload={
        "organic_keywords": [
            {"matched_url": url, "keyword": "support faq", "traffic": 25, "position": 2, "intents": ["informational"], "cluster_label": "support"}
        ]
    }))

    faq = next(row for row in report["opportunities"] if row["schema_type"] == "FAQPage")
    assert faq["target_url"] == url
    assert faq["required_evidence"] == ["visible questions", "visible answers for each question"]
    assert "visible questions" in faq["present_evidence"]
    assert faq["guideline_url"].startswith("https://developers.google.com/search/docs/appearance/structured-data/")
    assert faq["keyword_intents"] == ["informational"]
    assert report["clusters"][0]["cluster"] == "support"


def test_compare_leaderboard_includes_structured_data_metrics(tmp_path: Path) -> None:
    for domain, coverage, invalid in [
        ("a.example", 0.75, 1),
        ("b.example", 0.25, 3),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":4}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "structured_data.json").write_text(
            (
                '{"summary":{"schema_coverage":%s,'
                '"invalid_jsonld_blocks":%d,'
                '"schema_type_count":2,'
                '"pages_missing_schema":1}}'
            ) % (coverage, invalid),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["schema_coverage"] == 0.75
    assert rows["a.example"]["invalid_jsonld_blocks"] == 1
    assert rows["a.example"]["schema_type_count"] == 2
    assert rows["b.example"]["schema_coverage"] == 0.25
    assert rows["b.example"]["invalid_jsonld_blocks"] == 3
