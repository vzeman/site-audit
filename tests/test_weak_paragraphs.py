from datetime import date
from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.weak_paragraphs import build_weak_paragraphs


def _page(url: str, title: str = "Page") -> PageInfo:
    return PageInfo(url=url, title=title, description="", section="/blog/", word_count=120)


def test_weak_paragraphs_merges_decay_ablation_and_keyword_signals():
    pages = [_page("https://example.com/blog/a", "A")]
    extracted = [
        SimpleNamespace(
            paragraphs=["Our customer support software has no evidence and no links in this generic paragraph"],
            headers_rich=[{"text": "Support software", "level": 2, "order": 1}],
            h1="Support software",
        )
    ]
    paragraph_records = [(0, 0, extracted[0].paragraphs[0], np.array([1.0, 0.0], dtype=np.float32))]
    page_embeddings = np.array([[0.0, 1.0]], dtype=np.float32)
    semantic_ablation = {
        "rows": [
            {
                "url": "https://example.com/blog/a",
                "paragraph_index": 0,
                "classification": "noise_candidate",
                "alignment_delta": -0.03,
            }
        ]
    }
    keyword_attribution = {
        "keywords": [
            {
                "url": "https://example.com/blog/a",
                "best_paragraph_index": 0,
                "keyword": "support software",
                "traffic": 20,
                "position": 8,
                "status": "weak_paragraph",
            }
        ]
    }

    payload = build_weak_paragraphs(
        pages,
        page_embeddings,
        extracted,
        paragraph_records,
        semantic_ablation=semantic_ablation,
        keyword_attribution=keyword_attribution,
        freshness={"per_page": [{"url": "https://example.com/blog/a", "bucket": "unknown"}]},
        today=date(2026, 5, 9),
    )

    row = payload["rows"][0]
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["main_content_rows"] == 1
    assert "off_topic" in row["issue_types"]
    assert "intent_mismatch" in row["issue_types"]
    assert row["recommended_action"] == "move"
    assert row["traffic_opportunity"] == 20
    assert row["estimated_recoverable_traffic"] > 0


def test_weak_paragraphs_separates_repeated_boilerplate_from_main_content():
    pages = [_page(f"https://example.com/p/{i}", f"P{i}") for i in range(4)]
    text = "Copyright 2022 Example Inc. All rights reserved."
    extracted = [
        SimpleNamespace(paragraphs=[text], headers_rich=[], h1="Template")
        for _ in pages
    ]
    paragraph_records = [
        (i, 0, text, np.array([1.0, 0.0], dtype=np.float32))
        for i in range(4)
    ]
    page_embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))

    payload = build_weak_paragraphs(
        pages,
        page_embeddings,
        extracted,
        paragraph_records,
        today=date(2026, 5, 9),
    )

    row = payload["rows"][0]
    assert payload["summary"]["template_rows"] == 4
    assert payload["summary"]["main_content_rows"] == 0
    assert row["content_kind"] == "template"
    assert row["editorial_recommendation"] is False
    assert "duplicate" in row["issue_types"]
    assert row["recommended_action"] == "remove"
