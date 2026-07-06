import builtins
import sys
from types import SimpleNamespace

import numpy as np

from site_audit.scatter import project


class _FakeIndex:
    def search(self, matrix, k):
        labels = (np.arange(len(matrix)) % 3).astype(np.int64).reshape(-1, 1)
        distances = np.zeros_like(labels, dtype=np.float32)
        return distances, labels


class _FakeKmeans:
    def __init__(self, **kwargs):
        self.index = _FakeIndex()

    def train(self, matrix):
        return None


def test_large_scatter_projection_skips_umap(monkeypatch) -> None:
    fake_faiss = SimpleNamespace(Kmeans=_FakeKmeans)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setenv("SITE_AUDIT_UMAP_MAX_POINTS", "4")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise AssertionError("UMAP should not be imported for large projections")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    matrix = np.eye(6, 4, dtype=np.float32)

    labels, coords = project(matrix, num_clusters=3)

    assert labels.tolist() == [0, 1, 2, 0, 1, 2]
    assert coords.shape == (6, 2)
    assert np.isfinite(coords).all()
