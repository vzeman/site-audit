from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.duplicate_fragments import build_duplicate_fragments


def test_duplicate_fragments_distinguishes_reusable_patterns_from_boilerplate():
    pages = [
        PageInfo("https://example.com/a", "A", "", "blog", 120),
        PageInfo("https://example.com/b", "B", "", "blog", 120),
        PageInfo("https://example.com/c", "C", "", "legal", 50),
    ]
    strong = "Acme SLA automation routes urgent tickets to senior agents and reduced first response time by 31%."
    legal = "Copyright Example Inc. All rights reserved. Privacy policy and terms apply."
    extracted = [
        SimpleNamespace(paragraphs=[strong, legal], headers_rich=[{"level": 2, "order": 1, "text": "SLA automation"}], h1="A"),
        SimpleNamespace(paragraphs=[strong, legal], headers_rich=[{"level": 2, "order": 1, "text": "SLA automation"}], h1="B"),
        SimpleNamespace(paragraphs=[legal], headers_rich=[{"level": 2, "order": 1, "text": "Legal"}], h1="C"),
    ]
    paragraph_records = [
        (0, 0, strong, np.array([1.0, 0.0], dtype=np.float32)),
        (0, 1, legal, np.array([0.0, 1.0], dtype=np.float32)),
        (1, 0, strong, np.array([1.0, 0.0], dtype=np.float32)),
        (1, 1, legal, np.array([0.0, 1.0], dtype=np.float32)),
        (2, 0, legal, np.array([0.0, 1.0], dtype=np.float32)),
    ]
    keyword_attribution = {
        "keywords": [
            {"url": pages[0].url, "best_paragraph_index": 0, "keyword": "sla automation", "traffic": 40, "position": 3},
            {"url": pages[1].url, "best_paragraph_index": 0, "keyword": "ticket routing automation", "traffic": 25, "position": 8},
        ]
    }

    payload = build_duplicate_fragments(
        pages,
        extracted,
        paragraph_records,
        keyword_attribution=keyword_attribution,
    )

    classifications = {row["classification"] for row in payload["groups"]}
    assert payload["summary"]["status"] == "ok"
    assert "strong_reusable_pattern" in classifications
    assert "harmful_boilerplate" in classifications
    pattern = payload["pattern_library"][0]
    assert pattern["affected_urls"] == 2
    assert pattern["attributed_traffic"] == 65
    assert all(ex["url"] for ex in pattern["examples"])


def test_duplicate_fragments_returns_no_paragraphs_status():
    payload = build_duplicate_fragments([], [], None)

    assert payload["summary"]["status"] == "no_paragraphs"
    assert payload["groups"] == []
