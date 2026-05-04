from types import SimpleNamespace

import numpy as np

from site_audit.header_analysis import analyse, headers_for_scatter


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=256, show_progress=False):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text)
            vectors.append([
                float(seed % 7 + 1),
                float(seed % 11 + 1),
                float(seed % 13 + 1),
            ])
        return np.array(vectors, dtype=np.float32)


def _page(index: int = 0):
    return SimpleNamespace(
        url=f"https://example.com/{index}",
        title=f"Page {index}",
        word_count=200,
    )


def _extracted(headers):
    return SimpleNamespace(
        headers_rich=headers,
        h1_count=sum(1 for header in headers if header["level"] == 1),
        h1=next((header["text"] for header in headers if header["level"] == 1), ""),
    )


def test_header_analysis_embeds_duplicate_header_text_once() -> None:
    embedder = _Embedder()
    pages = [_page()]
    extracted_pages = [_extracted([
        {"level": 1, "order": 0, "text": "Repeated"},
        {"level": 2, "order": 1, "text": "Repeated"},
        {"level": 2, "order": 2, "text": "Unique"},
    ])]
    paragraph_records = [(0, 0, "Body paragraph", np.array([1.0, 0.0, 0.0], dtype=np.float32))]

    result = analyse(
        pages,
        extracted_pages,
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        paragraph_records,
        embedder=embedder,
    )

    assert embedder.calls == [["Repeated", "Unique"]]
    assert result["summary"]["total_headers"] == 3


def test_header_scatter_embeds_duplicate_header_text_once() -> None:
    embedder = _Embedder()
    pages = [_page()]
    extracted_pages = [_extracted([
        {"level": 1, "order": 0, "text": "Repeated"},
        {"level": 2, "order": 1, "text": "Repeated"},
        {"level": 3, "order": 2, "text": "Unique"},
    ])]

    result = headers_for_scatter(
        pages,
        extracted_pages,
        paragraph_records=[],
        embedder=embedder,
    )

    assert embedder.calls == [["Repeated", "Unique"]]
    assert result["total_headers"] == 3
    assert result["shown"] == 3
