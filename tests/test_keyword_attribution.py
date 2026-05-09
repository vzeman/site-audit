from __future__ import annotations

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.keyword_attribution import build_keyword_attribution


class _Extracted:
    def __init__(self):
        self.paragraphs = [
            "Call center software routes support calls and escalations.",
            "The office has a long company history.",
        ]
        self.headers_rich = [
            {"level": 1, "text": "Call center software", "order": 1},
            {"level": 2, "text": "Company history", "order": 2},
        ]
        self.h1 = "Call center software"


class _Embedder:
    def encode(self, texts, batch_size=64, show_progress=False):
        mapping = {
            "call center software": np.array([1.0, 0.0], dtype=np.float32),
            "unrelated keyword": np.array([0.0, 1.0], dtype=np.float32),
            "Call center software": np.array([1.0, 0.0], dtype=np.float32),
            "Company history": np.array([0.0, 1.0], dtype=np.float32),
        }
        return np.stack([mapping.get(t, np.array([0.2, 0.2], dtype=np.float32)) for t in texts])


def _norm(v):
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_keyword_attribution_maps_keyword_to_heading_and_paragraph():
    pages = [PageInfo("https://example.com/call-center", "Call center software", "", "", 100, "en")]
    ext = _Extracted()
    paragraph_records = [
        (0, 0, ext.paragraphs[0], _norm([1.0, 0.0])),
        (0, 1, ext.paragraphs[1], _norm([0.0, 1.0])),
    ]
    search_payload = {
        "meta": {"provider_label": "TestSearch"},
        "organic_keywords": [
            {"matched_url": "https://example.com/call-center", "keyword": "call center software", "traffic": 30, "volume": 300, "position": 3},
            {"matched_url": "https://example.com/call-center", "keyword": "unrelated keyword", "traffic": 5, "volume": 100, "position": 20},
        ],
        "top_pages": [],
    }

    payload = build_keyword_attribution(
        pages,
        [ext],
        paragraph_records,
        search_payload,
        embedder=_Embedder(),
    )

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["keyword_rows"] == 2
    first = payload["keywords"][0]
    assert first["keyword"] == "call center software"
    assert first["best_heading"] == "Call center software"
    assert first["best_paragraph_index"] == 0
    assert payload["headings"][0]["traffic"] == 30
    assert payload["paragraphs"][0]["paragraph_index"] == 0
