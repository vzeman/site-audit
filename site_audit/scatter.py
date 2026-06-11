"""UMAP 2D projection + FAISS k-means clustering for the scatterplot.

Same recipe as the Hugo project: cluster colors come from k-means over
the normalized embeddings; coordinates come from UMAP with cosine
distance. Both libraries are heavy, so we import them lazily.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def _stub_coords(n: int) -> np.ndarray:
    coords = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        coords[i, 0] = float(i)
    return coords


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

    import faiss  # type: ignore

    k = max(2, min(num_clusters, max(2, n // 2)))
    emb_f32 = embeddings.astype(np.float32)
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
        LOG.warning("UMAP failed on %d-page site (%s); falling back to stub coords", n, exc)
        coords = _stub_coords(n)
    return cluster_labels, coords.astype(np.float32)
