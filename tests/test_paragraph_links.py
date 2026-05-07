from types import SimpleNamespace

import numpy as np

from site_audit.paragraph_links import recommend


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
