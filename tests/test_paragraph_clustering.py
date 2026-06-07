import numpy as np

from site_audit.paragraph_clustering import ParagraphClusterSummary, to_overlap_payload


def test_paragraph_cluster_overlap_payload_uses_centroid_similarity() -> None:
    rows = [
        ParagraphClusterSummary(
            cluster_id=2,
            paragraph_count=5,
            keywords=[{"keyword": "alpha"}],
            cohesion=0.9,
            site_alignment=0.8,
            distinct_pages=2,
            top_paragraphs=[],
            _centroid=np.array([1.0, 0.0], dtype=np.float32),
        ),
        ParagraphClusterSummary(
            cluster_id=7,
            paragraph_count=3,
            keywords=[{"keyword": "beta"}],
            cohesion=0.7,
            site_alignment=0.6,
            distinct_pages=1,
            top_paragraphs=[],
            _centroid=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]

    payload = to_overlap_payload(rows)

    assert [c["label"] for c in payload["clusters"]] == ["alpha", "beta"]
    assert payload["clusters"][0]["paragraph_count"] == 5
    assert payload["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
