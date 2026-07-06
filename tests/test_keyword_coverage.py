from types import SimpleNamespace

import numpy as np

from site_audit.keyword_coverage import match_queries_to_paragraphs


def test_match_queries_to_paragraphs_streams_and_counts_distinct_pages() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/a", title="A"),
        SimpleNamespace(url="https://example.com/b", title="B"),
        SimpleNamespace(url="https://example.com/c", title="C"),
    ]
    paragraph_records = [
        (0, 0, "alpha paragraph one", np.array([1.0, 0.0], dtype=np.float32)),
        (1, 0, "alpha paragraph two", np.array([0.9, 0.1], dtype=np.float32)),
        (2, 0, "alpha paragraph three", np.array([0.8, 0.2], dtype=np.float32)),
        (2, 1, "weak paragraph", np.array([0.0, 1.0], dtype=np.float32)),
    ]

    matches = match_queries_to_paragraphs(
        [("alpha", "manual")],
        np.array([[1.0, 0.0]], dtype=np.float32),
        pages,
        paragraph_records,
        paragraph_floor=0.75,
        scattered_threshold_pages=3,
        top_k_paragraphs=2,
    )

    assert matches[0].best_url == "https://example.com/a"
    assert matches[0].distinct_pages_above_floor == 3
    assert matches[0].status == "scattered"
    assert len(matches[0].top_paragraphs) == 2
