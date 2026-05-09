from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.answer_blocks import build_answer_blocks


def test_answer_blocks_reports_strong_blocks_and_query_format_gaps():
    pages = [
        PageInfo("https://example.com/helpdesk", "What is help desk automation?", "", "blog", 160),
        PageInfo("https://example.com/setup", "Help desk setup", "", "blog", 140),
    ]
    strong_text = (
        "Help desk automation is a workflow system that routes support tickets, assigns owners, "
        "and sends SLA alerts so Acme Support can resolve customer requests 31% faster."
    )
    weak_text = "Help desk setup helps teams work better. Learn more about choosing the right tools for your business."
    extracted = [
        SimpleNamespace(
            paragraphs=[strong_text],
            headers_rich=[{"level": 2, "order": 1, "text": "What is help desk automation?"}],
            h1="What is help desk automation?",
            schema_types=["FAQPage", "Article"],
            list_count=1,
            table_count=0,
        ),
        SimpleNamespace(
            paragraphs=[weak_text],
            headers_rich=[{"level": 2, "order": 1, "text": "Setup overview"}],
            h1="Help desk setup",
            schema_types=[],
            list_count=0,
            table_count=0,
        ),
    ]
    paragraph_records = [
        (0, 0, strong_text, np.array([1.0, 0.0], dtype=np.float32)),
        (1, 0, weak_text, np.array([0.0, 1.0], dtype=np.float32)),
    ]
    search_payload = {
        "meta": {"provider_label": "Search"},
        "organic_keywords": [
            {
                "keyword": "what is help desk automation",
                "matched_url": "https://example.com/helpdesk",
                "traffic": 50,
                "volume": 1000,
                "position": 2,
                "intents": ["informational"],
            },
            {
                "keyword": "how to setup help desk automation",
                "matched_url": "https://example.com/setup",
                "traffic": 40,
                "volume": 800,
                "position": 7,
                "intents": ["informational"],
            },
        ],
    }

    payload = build_answer_blocks(
        pages,
        extracted,
        paragraph_records,
        search_payload=search_payload,
        cluster_labels=[1, 1],
    )

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["strong_blocks"] >= 1
    strong = next(row for row in payload["blocks"] if row["url"].endswith("/helpdesk"))
    assert strong["score"] >= 70
    assert strong["answer_type"] in {"definition", "faq", "statistic"}
    assert strong["evidence"]
    opportunity = next(row for row in payload["opportunities"] if row["url"].endswith("/setup"))
    assert opportunity["recommended_format"] == "steps"
    assert opportunity["schema_recommendation"] == "HowTo"
    assert "heading" in opportunity["reason"] or "schema" in opportunity["reason"]
    cluster = payload["clusters"][0]
    assert cluster["queries"] == 2
    assert cluster["opportunity_queries"] == 1


def test_answer_blocks_falls_back_to_page_titles_without_search_data():
    pages = [PageInfo("https://example.com/faq", "Can live chat reduce wait time?", "", "faq", 120)]
    text = "Live chat can reduce wait time by routing visitors to available agents and showing queue context before the first reply."
    extracted = [
        SimpleNamespace(
            paragraphs=[text],
            headers_rich=[{"level": 2, "order": 1, "text": "Can live chat reduce wait time?"}],
            h1="Can live chat reduce wait time?",
            schema_types=["FAQPage"],
            list_count=0,
            table_count=0,
        )
    ]
    paragraph_records = [(0, 0, text, np.array([1.0], dtype=np.float32))]

    payload = build_answer_blocks(pages, extracted, paragraph_records)

    assert payload["summary"]["queries"] == 1
    assert payload["blocks"][0]["query"] == "Can live chat reduce wait time?"
    assert payload["blocks"][0]["score"] >= 70
