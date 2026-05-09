from __future__ import annotations

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.paragraph_impact import build_paragraph_impact


class _Extracted:
    def __init__(self):
        self.paragraphs = [
            "Call center software with live chat routing and help desk automation for support teams.",
            "Company history and generic mission statement with very little product detail.",
            "Pricing plans include automation, call routing, and reporting for customer service.",
        ]
        self.paragraph_link_counts = [(1, 0), (0, 0), (0, 1)]
        self.headers_rich = [
            {"level": 1, "text": "Call center software", "order": 1},
            {"level": 2, "text": "Pricing plans", "order": 2},
        ]
        self.h1 = "Call center software"
        self.has_dates = True


class _Embedder:
    def encode(self, texts, batch_size=64, show_progress=False):
        vectors = {
            "call center software": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "customer service pricing": np.array([0.8, 0.2, 0.0], dtype=np.float32),
        }
        return np.stack([vectors.get(t, np.array([0.0, 0.0, 1.0], dtype=np.float32)) for t in texts])


def test_paragraph_impact_scores_keyword_aligned_paragraphs_higher():
    pages = [
        PageInfo(
            url="https://example.com/call-center",
            title="Call center software",
            description="",
            section="",
            word_count=200,
            language="en",
        )
    ]
    ext = _Extracted()
    paragraph_records = [
        (0, 0, ext.paragraphs[0], np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        (0, 1, ext.paragraphs[1], np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (0, 2, ext.paragraphs[2], np.array([0.8, 0.2, 0.0], dtype=np.float32)),
    ]
    search_payload = {
        "meta": {"provider_label": "TestSearch"},
        "top_pages": [
            {
                "matched_url": "https://example.com/call-center",
                "traffic": 100,
                "keywords": 8,
                "top_keyword": "call center software",
                "top_keyword_position": 2,
                "top_keyword_volume": 1000,
            }
        ],
        "organic_keywords": [
            {
                "matched_url": "https://example.com/call-center",
                "keyword": "customer service pricing",
                "traffic": 20,
                "volume": 500,
                "position": 8,
            }
        ],
    }

    payload = build_paragraph_impact(
        pages,
        [ext],
        paragraph_records,
        search_payload,
        embedder=_Embedder(),
        top_n=10,
    )

    rows = payload["top_paragraphs"]
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["scored_paragraphs"] == 3
    assert rows[0]["paragraph_index"] == 0
    assert rows[0]["impact_score"] > rows[-1]["impact_score"]
    assert rows[0]["best_keyword"] == "call center software"
    assert round(sum(r["attributed_traffic"] for r in rows), 1) == 100.0


def test_paragraph_impact_returns_no_search_data_without_provider_rows():
    pages = [
        PageInfo("https://example.com/", "Home", "", "", 10, "en"),
    ]
    payload = build_paragraph_impact(
        pages,
        [_Extracted()],
        [(0, 0, "A useful paragraph", np.array([1.0], dtype=np.float32))],
        {},
        embedder=_Embedder(),
    )

    assert payload["summary"]["status"] == "no_search_data"
    assert payload["top_paragraphs"] == []
