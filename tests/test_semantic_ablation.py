from __future__ import annotations

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.semantic_ablation import build_semantic_ablation


class _Extracted:
    has_dates = False


class _Embedder:
    def encode(self, texts, batch_size=64, show_progress=False):
        vectors = {
            "call center software": np.array([1.0, 0.0], dtype=np.float32),
        }
        return np.stack([vectors[t] for t in texts])


def _norm(v):
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_semantic_ablation_marks_topic_carrier_and_noise_candidate():
    pages = [PageInfo("https://example.com/call-center", "Call center software", "", "", 100, "en")]
    page_embeddings = np.stack([_norm([1.0, 0.0])])
    paragraph_records = [
        (0, 0, "Call center software routes support calls.", _norm([1.0, 0.0])),
        (0, 1, "Unrelated company culture text.", _norm([0.0, 1.0])),
    ]
    search_payload = {
        "meta": {"provider_label": "TestSearch"},
        "top_pages": [
            {"matched_url": "https://example.com/call-center", "traffic": 50, "top_keyword": "call center software"}
        ],
        "organic_keywords": [],
    }

    payload = build_semantic_ablation(
        pages,
        page_embeddings,
        [_Extracted()],
        paragraph_records,
        search_payload,
        embedder=_Embedder(),
    )

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["topic_carriers"] == 1
    assert payload["summary"]["noise_candidates"] == 1
    carrier = payload["topic_carriers"][0]
    noise = payload["noise_candidates"][0]
    assert carrier["paragraph_index"] == 0
    assert carrier["alignment_delta"] > 0
    assert noise["paragraph_index"] == 1
    assert noise["alignment_delta"] < 0
