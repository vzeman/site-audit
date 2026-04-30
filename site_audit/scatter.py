"""UMAP 2D projection + FAISS k-means clustering for the scatterplot.

Same recipe as the Hugo project: cluster colors come from k-means over
the normalized embeddings; coordinates come from UMAP with cosine
distance. Both libraries are heavy, so we import them lazily.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def project(embeddings: np.ndarray, num_clusters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster_labels, coords[n, 2])."""
    n, d = embeddings.shape
    if n < 3:
        # Degenerate sites: stub coordinates so the UI still renders.
        labels = np.zeros(n, dtype=int)
        coords = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            coords[i, 0] = float(i)
        return labels, coords

    import faiss  # type: ignore

    k = max(2, min(num_clusters, max(2, n // 2)))
    emb_f32 = embeddings.astype(np.float32)
    kmeans = faiss.Kmeans(d=d, k=k, niter=50, verbose=False, seed=42)
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
    coords = reducer.fit_transform(emb_f32)
    return cluster_labels, coords.astype(np.float32)
