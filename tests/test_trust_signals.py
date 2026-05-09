from site_audit.analyzer import PageInfo
from site_audit.extractor import ExtractedPage
from site_audit.trust_signals import build_trust_signals


def _page(url: str, title: str, body: str, **kwargs) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="Helpful description",
        body=body,
        word_count=len(body.split()),
        language="en",
        h1=title,
        h1_count=1,
        headings=kwargs.pop("headings", []),
        headers_rich=[{"level": 1, "text": title, "order": 0}],
        paragraphs=kwargs.pop("paragraphs", [body]),
        **kwargs,
    )


def test_trust_signals_score_components_and_missing_evidence_against_leaders():
    pages = [
        PageInfo("https://example.com/blog/strong", "Strong guide", "", "blog", 900),
        PageInfo("https://example.com/blog/weak", "Weak guide", "", "blog", 700),
    ]
    strong = _page(
        pages[0].url,
        pages[0].title,
        "By Jane Expert. Updated 2026-01-01. We tested 120 support tickets according to a benchmark report. "
        "Customer story and testimonial show product experience with screenshots.",
        external_link_count=3,
        stat_count=3,
        has_dates=True,
        schema_types=["BlogPosting", "Organization"],
        schema_blocks=[{"valid": True, "types": ["BlogPosting"], "keys": ["@type", "headline", "datePublished", "author"]}],
        media_items=[{"type": "image", "alt": "dashboard screenshot", "src": "/screenshot.png"}],
    )
    weak = _page(
        pages[1].url,
        pages[1].title,
        "This guide explains support automation and makes claims about better outcomes without sources.",
        external_link_count=0,
        stat_count=0,
        schema_types=[],
        schema_blocks=[],
    )
    search_payload = {
        "top_pages": [
            {"matched_url": pages[0].url, "traffic": 100, "keywords": 8, "top_keyword": "support automation", "cluster_label": "support automation"},
            {"matched_url": pages[1].url, "traffic": 40, "keywords": 3, "top_keyword": "support automation guide", "cluster_label": "support automation"},
        ]
    }

    payload = build_trust_signals(pages, [strong, weak], search_payload=search_payload)

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["high_priority_pages"] >= 1
    strong_row = next(row for row in payload["pages"] if row["url"] == pages[0].url)
    weak_row = next(row for row in payload["pages"] if row["url"] == pages[1].url)
    assert weak_row["components"]["evidence"] < strong_row["components"]["evidence"]
    assert any("citations" in item.lower() or "structured data" in item.lower() for item in weak_row["missing_signals"])
    assert payload["missing_evidence"][0]["stronger_examples"]
    assert payload["clusters"][0]["benchmark_components"]


def test_trust_signals_handles_empty_input():
    payload = build_trust_signals([], [])

    assert payload["summary"]["status"] == "no_pages"
    assert payload["pages"] == []
