from pathlib import Path

from site_audit.compare import build_payload
from site_audit.conversion import analyze, to_payload
from site_audit.extractor import ExtractedPage, extract


def _page(
    url: str,
    title: str = "Useful page",
    conversion_signals: dict | None = None,
) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="",
        body="Useful content for testing conversion signals.",
        word_count=120,
        language="en",
        conversion_signals=conversion_signals or {},
    )


def test_extract_conversion_signals_from_ctas_forms_and_contact_links() -> None:
    html = """
    <html><head><title>Book a consultation</title></head><body>
      <h1>Book a consultation</h1>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It explains a professional service, who it helps, how consultations work,
      and why visitors should submit a request for a tailored quote today.</p>
      <a class="btn primary" href="/contact">Request a quote</a>
      <button>Book demo</button>
      <a href="tel:+421900123456">Call us</a>
      <form action="/lead" method="post">
        <input type="text" name="name">
        <input type="email" name="email">
        <textarea name="message"></textarea>
        <button type="submit">Send request</button>
      </form>
    </body></html>
    """

    page = extract("https://example.com/services", html, max_chars=2000)

    assert page is not None
    signals = page.conversion_signals
    assert signals["cta_count"] == 4
    assert signals["primary_cta_count"] == 3
    assert signals["form_count"] == 1
    assert signals["form_field_count"] == 3
    assert signals["forms"][0]["has_submit"] is True
    assert signals["contact_link_count"] == 1
    assert [cta["text"] for cta in signals["ctas"]] == ["Request a quote", "Book demo", "Call us", "Send request"]


def test_conversion_payload_flags_missing_and_weak_capture_paths() -> None:
    report = to_payload(analyze([
        _page(
            "https://example.com/contact",
            title="Contact sales",
            conversion_signals={
                "cta_count": 1,
                "primary_cta_count": 1,
                "ctas": [{"text": "Contact sales", "primary": True}],
                "form_count": 1,
                "form_field_count": 2,
                "forms": [{"field_count": 2, "has_submit": True}],
                "contact_link_count": 0,
            },
        ),
        _page("https://example.com/pricing", title="Pricing"),
        _page(
            "https://example.com/blog/post",
            conversion_signals={
                "cta_count": 1,
                "primary_cta_count": 0,
                "ctas": [{"text": "Learn more", "primary": False}],
                "form_count": 0,
                "contact_link_count": 0,
            },
        ),
        _page(
            "https://example.com/services",
            conversion_signals={
                "cta_count": 9,
                "primary_cta_count": 9,
                "ctas": [{"text": f"Book demo {i}", "primary": True} for i in range(9)],
                "form_count": 1,
                "forms": [{"field_count": 1, "has_submit": False}],
                "contact_link_count": 0,
            },
        ),
    ]))

    assert report["summary"]["total_pages"] == 4
    assert report["summary"]["pages_with_cta"] == 3
    assert report["summary"]["cta_coverage"] == 0.75
    assert report["summary"]["primary_cta_coverage"] == 0.5
    assert report["summary"]["form_coverage"] == 0.5
    assert report["summary"]["lead_pages_without_capture"] == 1
    assert report["summary"]["cta_overload_pages"] == 1
    assert report["summary"]["forms_without_submit_pages"] == 1
    assert report["issues_by_type"]["no_cta"] == 1
    assert report["issues_by_type"]["only_generic_ctas"] == 1
    assert report["top_ctas"][0] == {"text": "Contact sales", "count": 1}
    assert report["per_page"][0]["url"] == "https://example.com/pricing"


def test_compare_leaderboard_includes_conversion_metrics(tmp_path: Path) -> None:
    for domain, cta_coverage, primary_coverage in [
        ("a.example", 0.8, 0.6),
        ("b.example", 0.3, 0.2),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":5}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "conversion.json").write_text(
            (
                '{"summary":{"cta_coverage":%s,'
                '"primary_cta_coverage":%s,'
                '"form_coverage":0.4,'
                '"avg_ctas_per_page":2.5,'
                '"lead_pages_without_capture":1,'
                '"cta_overload_pages":2}}'
            ) % (cta_coverage, primary_coverage),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["cta_coverage"] == 0.8
    assert rows["a.example"]["primary_cta_coverage"] == 0.6
    assert rows["a.example"]["form_coverage"] == 0.4
    assert rows["a.example"]["avg_ctas_per_page"] == 2.5
    assert rows["b.example"]["cta_coverage"] == 0.3
    assert rows["b.example"]["lead_pages_without_capture"] == 1
