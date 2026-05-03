import json
from pathlib import Path

from site_audit.compare import build_payload
from site_audit.entities import analyze, extract_entities_from_text, to_payload
from site_audit.extractor import ExtractedPage, extract


def _page(
    url: str,
    title: str,
    body: str,
    headings: list[str] | None = None,
    schema_blocks: list[dict] | None = None,
) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="entity analysis for lower-case copy.",
        body=body,
        word_count=len(body.split()),
        language="en",
        headings=headings or [],
        h1=headings[0] if headings else title,
        headers_rich=[{"level": 1, "text": headings[0], "order": 0}] if headings else [],
        schema_types=["Organization"] if schema_blocks else [],
        schema_blocks=schema_blocks or [],
    )


def test_extracts_capitalized_entity_phrases() -> None:
    entities = extract_entities_from_text(
        "Acme Analytics works with Google Cloud, OpenAI, and Market Research teams."
    )

    assert "Acme Analytics" in entities
    assert "Google Cloud" in entities
    assert "OpenAI" in entities
    assert "Market Research" in entities


def test_extract_adds_schema_organization_names() -> None:
    html = """
    <html><head>
      <title>Acme Analytics Strategy</title>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"Organization","name":"Acme Analytics LLC"}
      </script>
    </head><body>
      <h1>Acme Analytics Entity Strategy</h1>
      <p>Acme Analytics helps Google Cloud teams improve Digital Strategy.</p>
    </body></html>
    """

    page = extract("https://example.com/entities", html, max_chars=2000)

    assert page is not None
    assert page.schema_blocks[0]["names"] == ["Acme Analytics LLC"]
    payload = to_payload(analyze([page]))
    assert payload["organizations"][0]["organization"] == "Acme Analytics LLC"


def test_entities_payload_summarizes_coverage_depth_and_per_page_counts() -> None:
    payload = to_payload(analyze([
        _page(
            "https://example.com/strategy",
            "Digital Strategy for Acme Analytics",
            "Acme Analytics explains Digital Strategy, Market Research, Google Cloud, OpenAI, "
            "Search Analytics, and Customer Data Platforms for B2B Growth.",
            headings=["Digital Strategy with Acme Analytics"],
            schema_blocks=[{"valid": True, "types": ["Organization"], "names": ["Acme Analytics LLC"]}],
        ),
        _page(
            "https://example.com/research",
            "Market Research Playbook",
            "Market Research connects Customer Data Platforms, Google Cloud, and Search Analytics.",
            headings=["Market Research and Search Analytics"],
        ),
        _page("https://example.com/plain", "plain page", "short lower-case copy only."),
    ]))

    summary = payload["summary"]
    assert summary["total_pages"] == 3
    assert summary["pages_with_entities"] == 2
    assert summary["entity_coverage"] == 2 / 3
    assert summary["unique_entities"] >= 6
    assert summary["organization_count"] == 1
    assert summary["topical_authority_score"] > 0
    assert payload["per_page"][0]["url"] == "https://example.com/plain"
    assert any(row["entity"] == "Google Cloud" for row in payload["top_entities"])


def test_compare_leaderboard_includes_entity_metrics(tmp_path: Path) -> None:
    for domain, score in (("a.example", 72.5), ("b.example", 20.0)):
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(json.dumps({"page_count": 2, "model": "test-model"}))
        (report_dir / "pages.json").write_text(json.dumps([]))
        (report_dir / "entities.json").write_text(json.dumps({
            "summary": {
                "entity_coverage": 0.75,
                "unique_entities": 12,
                "avg_entities_per_page": 6.0,
                "entity_reuse_share": 0.5,
                "organization_count": 2,
                "organization_coverage": 0.5,
                "topical_depth_share": 0.5,
                "topical_authority_score": score,
            }
        }))

    payload = build_payload(["a.example", "b.example"], tmp_path)
    rows = {row["domain"]: row for row in payload["leaderboard"]}

    assert rows["a.example"]["topical_authority_score"] == 72.5
    assert rows["a.example"]["unique_entities"] == 12
    assert rows["b.example"]["entity_coverage"] == 0.75
