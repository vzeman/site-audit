from pathlib import Path

from site_audit.compare import build_payload
from site_audit.extractor import ExtractedPage
from site_audit.page_types import analyze, classify_page, to_payload


def _page(
    url: str,
    title: str = "Test page",
    schema_types=None,
    word_count: int = 500,
    headings=None,
    h1: str = "Test page",
    list_count: int = 0,
    table_count: int = 0,
    paragraphs=None,
    has_dates: bool = False,
) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="Useful test page description.",
        body="Useful content for page type classification.",
        word_count=word_count,
        language="en",
        h1=h1,
        h1_count=1 if h1 else 0,
        headings=headings or [],
        headers_rich=[
            {"level": 1, "text": h1, "order": 0},
            *[
                {"level": 2, "text": heading, "order": i + 1}
                for i, heading in enumerate(headings or [])
            ],
        ] if h1 else [],
        list_count=list_count,
        table_count=table_count,
        schema_types=schema_types or [],
        paragraphs=paragraphs or ["paragraph one", "paragraph two", "paragraph three", "paragraph four"],
        has_dates=has_dates,
    )


def test_classify_schema_and_url_page_types() -> None:
    assert classify_page(_page(
        "https://example.com/blog/how-to-audit-pages",
        schema_types=["BlogPosting"],
        has_dates=True,
    )).page_type == "blog_post"
    assert classify_page(_page(
        "https://example.com/products/site-audit",
        schema_types=["Product"],
    )).page_type == "product"
    assert classify_page(_page(
        "https://example.com/contact",
        title="Contact Acme",
    )).page_type == "contact"


def test_listing_and_faq_use_structural_signals() -> None:
    listing = classify_page(_page(
        "https://example.com/category/tools",
        word_count=300,
        list_count=4,
        paragraphs=["short intro"],
    ))
    faq = classify_page(_page(
        "https://example.com/support/questions",
        headings=["What is auditing?", "How does it work?", "Can I export data?"],
    ))

    assert listing.page_type == "listing"
    assert "many_lists_low_body" in listing.signals
    assert faq.page_type == "faq"
    assert "question_headings" in faq.signals


def test_payload_summarizes_types_and_template_families() -> None:
    report = to_payload(analyze([
        _page("https://example.com/"),
        _page("https://example.com/blog/a", schema_types=["Article"], has_dates=True),
        _page("https://example.com/blog/b", schema_types=["Article"], has_dates=True),
        _page("https://example.com/products/widget", schema_types=["Product"]),
    ]))

    assert report["summary"]["total_pages"] == 4
    assert report["summary"]["page_type_count"] == 3
    assert report["summary"]["dominant_page_type"] == "blog_post"
    assert report["summary"]["template_signature_count"] >= 3
    counts = {row["page_type"]: row["pages"] for row in report["type_counts"]}
    assert counts["blog_post"] == 2
    assert counts["home"] == 1
    assert counts["product"] == 1


def test_compare_leaderboard_includes_page_type_metrics(tmp_path: Path) -> None:
    for domain, dominant_type, type_count in [
        ("a.example", "article", 4),
        ("b.example", "product", 2),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":4}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "page_types.json").write_text(
            (
                '{"summary":{"page_type_count":%d,'
                '"template_family_count":3,'
                '"template_signature_count":5,'
                '"dominant_page_type":"%s",'
                '"dominant_template_family":"content_template"}}'
            ) % (type_count, dominant_type),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["page_type_count"] == 4
    assert rows["a.example"]["template_family_count"] == 3
    assert rows["a.example"]["dominant_page_type"] == "article"
    assert rows["b.example"]["page_type_count"] == 2
    assert rows["b.example"]["dominant_page_type"] == "product"
