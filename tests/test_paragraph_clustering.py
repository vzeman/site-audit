import builtins
from types import SimpleNamespace

import numpy as np

from site_audit.paragraph_clustering import (
    ParagraphClusterSummary,
    cluster_and_label,
    project_paragraphs,
    to_overlap_payload,
)


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


def test_large_paragraph_clustering_avoids_native_engines(monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_PARAGRAPH_CLUSTER_NATIVE_MAX", "4")
    monkeypatch.setenv("SITE_AUDIT_PARAGRAPH_CLUSTER_MAX_POINTS", "0")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"faiss", "umap"}:
            raise AssertionError(f"{name} should not be imported for large paragraph sets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pages = [SimpleNamespace(url=f"https://example.com/{idx}", title=f"Page {idx}") for idx in range(10)]
    paragraph_records = [
        (idx, 0, f"Paragraph text about topic {idx}", np.eye(10, 4, dtype=np.float32)[idx])
        for idx in range(10)
    ]
    site_centroid = np.ones(4, dtype=np.float32)
    site_centroid = site_centroid / np.linalg.norm(site_centroid)

    labels, summaries = cluster_and_label(paragraph_records, pages, site_centroid, num_clusters=4)
    chosen, coords = project_paragraphs(
        np.stack([row[3] for row in paragraph_records]).astype(np.float32),
        sample_cap=8,
    )

    assert labels.shape == (10,)
    assert summaries
    assert len(chosen) == 8
    assert coords.shape == (8, 2)
    assert np.isfinite(coords).all()


def test_very_large_paragraph_clustering_can_skip(monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_PARAGRAPH_CLUSTER_MAX_POINTS", "4")
    pages = [SimpleNamespace(url=f"https://example.com/{idx}", title=f"Page {idx}") for idx in range(10)]
    paragraph_records = [
        (idx, 0, f"Paragraph text about topic {idx}", np.eye(10, 4, dtype=np.float32)[idx])
        for idx in range(10)
    ]

    labels, summaries = cluster_and_label(
        paragraph_records,
        pages,
        np.ones(4, dtype=np.float32) / 2.0,
        num_clusters=4,
    )

    assert labels.shape == (10,)
    assert labels.sum() == 0
    assert summaries == []
