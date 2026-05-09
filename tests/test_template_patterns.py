import json

from site_audit.analyzer import PageInfo
from site_audit.extractor import ExtractedPage
from site_audit.template_patterns import build_template_patterns


def _extracted(
    url: str,
    title: str,
    *,
    headings: list[str],
    traffic_shape: str,
) -> ExtractedPage:
    strong = traffic_shape == "strong"
    paragraphs = [
        "This concise intro explains the topic for buyers and searchers.",
        "Detailed context with examples and practical criteria.",
        "More context with specific use cases and proof.",
        "A section with implementation advice.",
        "A section with comparison details.",
        "A closing section with next steps.",
    ] if strong else [
        "Generic short intro.",
        "A thin body paragraph.",
    ]
    schema_types = ["BlogPosting", "FAQPage"] if strong else ["BlogPosting"]
    conversion = {"cta_count": 2, "primary_cta_count": 1, "form_count": 0, "contact_link_count": 0} if strong else {}
    return ExtractedPage(
        url=url,
        title=title,
        description="Description",
        body=" ".join(paragraphs),
        word_count=1300 if strong else 420,
        language="en",
        h1=title,
        h1_count=1,
        headings=headings,
        headers_rich=[
            {"level": 1, "text": title, "order": 0},
            *[{"level": 2, "text": heading, "order": i + 1} for i, heading in enumerate(headings)],
        ],
        list_count=3 if strong else 0,
        table_count=1 if strong else 0,
        schema_types=schema_types,
        external_link_count=2 if strong else 0,
        stat_count=3 if strong else 0,
        paragraphs=paragraphs,
        conversion_signals=conversion,
    )


def test_template_patterns_mine_high_performer_structure_and_recommend_weak_pages():
    pages = [
        PageInfo(f"https://example.com/blog/page-{i}", f"Page {i}", "", "blog", 1200 if i < 3 else 400)
        for i in range(6)
    ]
    extracted = [
        _extracted(
            page.url,
            page.title,
            headings=[
                "What is workflow automation?",
                "Use cases",
                "Comparison table",
                "Examples",
                "How to choose",
            ] if i < 3 else ["Overview", "Benefits"],
            traffic_shape="strong" if i < 3 else "weak",
        )
        for i, page in enumerate(pages)
    ]
    page_types = {
        "per_page": [
            {
                "url": page.url,
                "page_type": "blog_post",
                "template_family": "content_template",
                "template_signature": "sig-strong" if i < 3 else "sig-weak",
            }
            for i, page in enumerate(pages)
        ]
    }
    search_payload = {
        "top_pages": [
            {"matched_url": pages[0].url, "traffic": 120, "keywords": 10, "top_keyword": "workflow automation"},
            {"matched_url": pages[1].url, "traffic": 90, "keywords": 8, "top_keyword": "automation examples"},
            {"matched_url": pages[2].url, "traffic": 70, "keywords": 6, "top_keyword": "automation use cases"},
            {"matched_url": pages[3].url, "traffic": 8, "keywords": 1},
            {"matched_url": pages[4].url, "traffic": 4, "keywords": 1},
            {"matched_url": pages[5].url, "traffic": 2, "keywords": 1},
        ]
    }

    payload = build_template_patterns(pages, extracted, page_types=page_types, search_payload=search_payload)

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["performance_source"] == "search"
    assert payload["patterns"]
    keys = {row["feature_key"] for row in payload["patterns"]}
    assert {"comparison_table", "question_headings", "primary_cta"} & keys
    first = payload["patterns"][0]
    assert first["sample_size"] == 6
    assert first["confidence"] >= 0.45
    assert first["sample_urls"]
    assert first["affected_weak_pages"]
    assert payload["recommendations"]
    assert payload["comparisons"][0]["common_top_features"]


def test_template_patterns_require_performance_signal():
    page = PageInfo("https://example.com/a", "A", "", "blog", 100)
    extracted = [_extracted(page.url, page.title, headings=["Overview"], traffic_shape="weak")]

    payload = build_template_patterns([page], extracted)

    assert payload["summary"]["status"] == "insufficient_performance_data"
    assert payload["patterns"] == []


def test_template_patterns_payload_is_json_serializable():
    page = PageInfo("https://example.com/a", "A", "", "blog", 100)
    extracted = [_extracted(page.url, page.title, headings=["Overview"], traffic_shape="weak")]
    payload = build_template_patterns([page], extracted)

    json.dumps(payload)
