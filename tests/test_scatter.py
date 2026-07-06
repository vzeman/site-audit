import builtins
import numpy as np

from site_audit.scatter import project


def test_large_scatter_projection_skips_umap(monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_UMAP_MAX_POINTS", "4")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"faiss", "umap"}:
            raise AssertionError(f"{name} should not be imported for large projections")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    matrix = np.eye(6, 4, dtype=np.float32)

    labels, coords = project(matrix, num_clusters=3)

    assert labels.shape == (6,)
    assert coords.shape == (6, 2)
    assert np.isfinite(coords).all()
