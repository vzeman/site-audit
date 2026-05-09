from site_audit.analyzer import PageInfo
from site_audit.conversion_balance import build_conversion_balance
from site_audit.extractor import ExtractedPage


def _page(url: str, title: str, conversion_signals=None) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="Description",
        body="Support automation pricing demo trial product platform.",
        word_count=120,
        language="en",
        h1=title,
        h1_count=1,
        headings=["Pricing", "Demo"],
        headers_rich=[{"level": 1, "text": title, "order": 0}, {"level": 2, "text": "Pricing", "order": 1}],
        conversion_signals=conversion_signals or {},
    )


def test_conversion_balance_labels_money_pages_and_actions():
    pages = [
        PageInfo("https://example.com/pricing", "Pricing", "", "pricing", 120),
        PageInfo("https://example.com/blog/support-automation", "Support automation", "", "blog", 500),
    ]
    extracted = [
        _page(pages[0].url, pages[0].title),
        _page(
            pages[1].url,
            pages[1].title,
            conversion_signals={
                "cta_count": 2,
                "primary_cta_count": 1,
                "ctas": [{"text": "Book a demo", "href": "/demo", "primary": True}],
                "form_count": 1,
            },
        ),
    ]
    search_payload = {
        "top_pages": [
            {"matched_url": pages[0].url, "traffic": 100, "keywords": 8, "top_keyword": "support automation pricing", "top_keyword_position": 3},
            {"matched_url": pages[1].url, "traffic": 60, "keywords": 5, "top_keyword": "support automation", "top_keyword_position": 5},
        ],
        "organic_keywords": [
            {"matched_url": pages[0].url, "keyword": "support automation pricing", "traffic": 80, "position": 3, "intents": ["commercial"]},
        ],
    }

    payload = build_conversion_balance(pages, extracted, search_payload=search_payload)

    pricing = next(row for row in payload["pages"] if row["url"] == pages[0].url)
    assert pricing["money_page"] is True
    assert pricing["balance_label"] == "high_risk_money_page"
    assert "primary CTA" in pricing["recommended_action"] or "lead capture" in pricing["recommended_action"]
    assert payload["summary"]["high_risk_money_pages"] == 1
    assert payload["high_traffic_weak_conversion"]
    assert payload["cta_warnings"]


def test_conversion_balance_handles_empty_input():
    payload = build_conversion_balance([], [])

    assert payload["summary"]["status"] == "no_pages"
    assert payload["pages"] == []
