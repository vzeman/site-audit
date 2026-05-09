from types import SimpleNamespace

import numpy as np

from site_audit.paragraph_links import build_addition_simulation, recommend, to_payload


class CountingEmbedder:
    def __init__(self) -> None:
        self.encoded_count = 0

    def encode(self, texts, batch_size=256, show_progress=True):
        self.encoded_count += len(texts)
        return np.ones((len(texts), 3), dtype=np.float32)


def test_recommend_trims_candidates_before_anchor_embedding() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source"),
        SimpleNamespace(url="https://example.com/target-a", title="Target A"),
        SimpleNamespace(url="https://example.com/target-b", title="Target B"),
    ]
    page_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paragraph_records = [
        (
            0,
            index,
            f"Relevant anchor phrase number {index} about target content and internal linking.",
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )
        for index in range(20)
    ]
    embedder = CountingEmbedder()

    recs = recommend(
        pages,
        page_embeddings,
        paragraph_records,
        pages_with_outlinks=[],
        embedder=embedder,
        similarity_floor=0.1,
        lift_floor=0.0,
        top_k_per_page=2,
        top_k_total=1,
    )

    assert len(recs) == 1
    assert embedder.encoded_count <= 120


def test_addition_simulation_scores_components_and_density_cap() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="blog"),
        SimpleNamespace(url="https://example.com/target", title="Target", section="docs"),
    ]
    recs = to_payload([
        SimpleNamespace(
            source_url=pages[0].url,
            paragraph_index=2,
            paragraph_excerpt="Contextual paragraph about target topics.",
            target_url=pages[1].url,
            target_title=pages[1].title,
            fit=0.9,
            lift=0.2,
            suggested_anchor="target topics",
            anchor_confidence=0.8,
        )
    ])
    linkgraph = {
        "page_link_counts": [
            {"url": pages[0].url, "pagerank": 0.6, "out_degree": 4},
            {"url": pages[1].url, "pagerank": 0.1, "in_degree": 1},
        ],
        "traffic_weighted_pagerank": {
            "pages": [
                {"url": pages[0].url, "weighted_pagerank_percentile": 0.9, "traffic_weighted_pagerank": 0.05, "out_degree": 4},
                {"url": pages[1].url, "weighted_pagerank_percentile": 0.2, "traffic": 500, "keywords": 10, "authority_traffic_gap": 0.7, "in_degree": 1},
            ]
        },
    }

    payload = build_addition_simulation(recs, pages, linkgraph, paragraph_saturation={(0, 2): 1.5})

    row = payload["recommendations"][0]
    assert row["expected_benefit_score"] > 70
    assert row["priority"] == "high"
    assert row["current_target_in_degree"] == 1
    assert row["after_target_in_degree"] == 2
    assert row["score_components"]["authority_flow"] > 0
    assert row["respects_density_cap"] is True
