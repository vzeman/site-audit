"""UMAP 2D projection + FAISS k-means clustering for the scatterplot.

Same recipe as the Hugo project: cluster colors come from k-means over
the normalized embeddings; coordinates come from UMAP with cosine
distance. Both libraries are heavy, so we import them lazily.
"""

from __future__ import annotations

import logging
import os

import numpy as np

LOG = logging.getLogger(__name__)

DEFAULT_UMAP_MAX_POINTS = 20000


def _stub_coords(n: int) -> np.ndarray:
    coords = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        coords[i, 0] = float(i)
    return coords


def _configured_umap_max_points(default: int = DEFAULT_UMAP_MAX_POINTS) -> int:
    raw = os.environ.get("SITE_AUDIT_UMAP_MAX_POINTS")
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Ignoring invalid SITE_AUDIT_UMAP_MAX_POINTS=%r; using %d", raw, default)
        return default


def _pca_coords(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1.0, norms)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(arr, full_matrices=False)
        coords = (u[:, :2] * s[:2]).astype(np.float32)
    except np.linalg.LinAlgError:
        return _stub_coords(n)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return coords.astype(np.float32)


def _fallback_labels(n: int, k: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=int)
    return (np.arange(n, dtype=int) % max(1, k)).astype(int)


def _quantile_labels(coords: np.ndarray, k: int) -> np.ndarray:
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=int)
    if k <= 1:
        return np.zeros(n, dtype=int)
    values = np.asarray(coords[:, 0], dtype=np.float32)
    if not np.isfinite(values).all() or float(values.max() - values.min()) == 0.0:
        return _fallback_labels(n, k)
    try:
        cuts = np.quantile(values, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(cuts, values, side="right").astype(int)
    except Exception as exc:  # pragma: no cover - numeric edge
        LOG.warning("Quantile cluster labels failed (%s); using deterministic fallback labels", exc)
        return _fallback_labels(n, k)


def project(embeddings: np.ndarray, num_clusters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster_labels, coords[n, 2])."""
    n, d = embeddings.shape
    # UMAP's spectral init uses scipy.sparse.linalg.eigsh which requires
    # the matrix dimension to exceed the eigenvalue count. With
    # n_components=2 it asks for 3 eigenvalues, so n=3,4 hit
    # "k >= N" errors. Stub coords for any site that small.
    if n < 5:
        labels = np.zeros(n, dtype=int)
        return labels, _stub_coords(n)

    k = max(2, min(num_clusters, max(2, n // 2)))
    emb_f32 = embeddings.astype(np.float32)

    umap_max_points = _configured_umap_max_points()
    if umap_max_points and n > umap_max_points:
        LOG.info(
            "Skipping native scatter engines for %d points above SITE_AUDIT_UMAP_MAX_POINTS=%d; using PCA/quantile labels",
            n,
            umap_max_points,
        )
        coords = _pca_coords(emb_f32)
        return _quantile_labels(coords, k), coords

    import faiss  # type: ignore

    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=50,
        verbose=False,
        seed=42,
        min_points_per_centroid=1,
    )
    kmeans.train(emb_f32)
    _, labels = kmeans.index.search(emb_f32, 1)
    cluster_labels = labels.flatten().astype(int)

    import umap  # type: ignore

    n_neighbors = max(2, min(15, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    try:
        coords = reducer.fit_transform(emb_f32)
    except Exception as exc:  # rare scipy/eigensolver corner cases on tiny corpora
        LOG.warning("UMAP failed on %d-page site (%s); falling back to PCA coords", n, exc)
        coords = _pca_coords(emb_f32)
    return cluster_labels, coords.astype(np.float32)
