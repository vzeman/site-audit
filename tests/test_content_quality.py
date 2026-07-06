from types import SimpleNamespace

import numpy as np

from site_audit.content_quality import wrong_home_paragraphs


def test_wrong_home_paragraphs_keeps_top_findings() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/a", title="A"),
        SimpleNamespace(url="https://example.com/b", title="B"),
    ]
    page_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paragraph_records = [
        (0, 0, "belongs on page b", np.array([0.0, 1.0], dtype=np.float32)),
        (0, 1, "belongs on page a", np.array([1.0, 0.0], dtype=np.float32)),
    ]

    rows = wrong_home_paragraphs(pages, page_embeddings, paragraph_records, top_n=1)

    assert len(rows) == 1
    assert rows[0].source_url == "https://example.com/a"
    assert rows[0].suggested_home_url == "https://example.com/b"
