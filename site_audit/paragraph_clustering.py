"""Topic clustering at the paragraph level + 2D projection.

The page-level clusters tell us "what does each *page* talk about."
The paragraph-level clusters tell us "what micro-topics exist on the
site, regardless of which page they live on" — which is closer to how
AI answer engines actually retrieve content (paragraph chunks). A
sub-topic that exists as a paragraph on 6 different pages but has no
dedicated page is a content opportunity.

We mirror the page-level recipe:

* k-means over normalized paragraph embeddings (FAISS)
* c-TF-IDF labels per cluster (BERTopic-style)
* UMAP 2D projection for the scatterplot

For very large sites (more than ~5 000 paragraphs) the SVG scatter would
lag, so we sample-down to ``scatter_sample`` paragraphs (default 5 000)
keeping the cluster information intact.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cluster_labels import _compute_ctfidf

LOG = logging.getLogger(__name__)


def _native_max_points() -> int:
    raw = os.getenv("SITE_AUDIT_PARAGRAPH_CLUSTER_NATIVE_MAX", "50000")
    try:
        return max(0, int(raw))
    except ValueError:
        return 50000


def _pca_scores(embs: np.ndarray, dims: int = 2) -> np.ndarray:
    if len(embs) == 0:
        return np.zeros((0, dims), dtype=np.float32)
    x = embs.astype(np.float32, copy=False)
    centered = x - x.mean(axis=0, keepdims=True)
    if len(x) < 2:
        return np.zeros((len(x), dims), dtype=np.float32)
    cov = (centered.T @ centered) / max(1, len(x) - 1)
    _, vecs = np.linalg.eigh(cov)
    comps = vecs[:, -dims:][:, ::-1].astype(np.float32, copy=False)
    scores = centered @ comps
    if scores.shape[1] < dims:
        scores = np.pad(scores, ((0, 0), (0, dims - scores.shape[1])), mode="constant")
    return scores.astype(np.float32, copy=False)


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1 or len(values) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    edges = np.unique(edges)
    if len(edges) == 0:
        return np.zeros(len(values), dtype=int)
    return np.searchsorted(edges, values, side="right").astype(int)


def _large_site_labels(embs: np.ndarray, k: int) -> np.ndarray:
    if len(embs) == 0:
        return np.zeros(0, dtype=int)
    scores = _pca_scores(embs, dims=2)
    x_bins = max(1, int(np.ceil(np.sqrt(k))))
    y_bins = max(1, int(np.ceil(k / x_bins)))
    labels = _quantile_bins(scores[:, 0], x_bins) * y_bins + _quantile_bins(scores[:, 1], y_bins)
    unique = sorted(int(v) for v in np.unique(labels))
    remap = {old: idx for idx, old in enumerate(unique[:k])}
    overflow = max(0, len(remap) - 1)
    return np.array([remap.get(int(v), overflow) for v in labels], dtype=int)


@dataclass
class ParagraphClusterSummary:
    cluster_id: int
    paragraph_count: int
    keywords: list[dict]
    cohesion: float
    site_alignment: float
    distinct_pages: int                # how many pages contribute paragraphs
    top_paragraphs: list[dict]         # [{url, page_title, excerpt, similarity}]
    _centroid: np.ndarray | None = None


def cluster_and_label(
    paragraph_records: list,            # [(page_index, para_index, text, embedding)]
    pages,                              # list of PageInfo
    site_centroid: np.ndarray,
    num_clusters: int = 60,
    top_k_keywords: int = 10,
    top_paragraphs: int = 4,
) -> tuple[np.ndarray, list[ParagraphClusterSummary]]:
    """Return ``(cluster_labels, summaries)`` over the given paragraphs."""
    if not paragraph_records:
        return np.zeros(0, dtype=int), []

    embs = np.stack([r[3] for r in paragraph_records]).astype(np.float32)
    n = len(embs)
    k = max(2, min(num_clusters, max(2, n // 8)))

    if n > _native_max_points():
        LOG.info(
            "  paragraph clusters: using large-site PCA/quantile fallback for %d paragraphs",
            n,
        )
        cluster_labels = _large_site_labels(embs, k)
        k = int(max(cluster_labels) + 1) if len(cluster_labels) else 0
    else:
        import faiss  # type: ignore

        kmeans = faiss.Kmeans(
            d=embs.shape[1],
            k=k,
            niter=50,
            verbose=False,
            seed=42,
            min_points_per_centroid=1,
        )
        kmeans.train(embs)
        _, labels = kmeans.index.search(embs, 1)
        cluster_labels = labels.flatten().astype(int)

    # c-TF-IDF over per-cluster aggregated paragraph text
    doc_parts: list[list[str]] = [[] for _ in range(k)]
    for i, c in enumerate(cluster_labels):
        doc_parts[int(c)].append(paragraph_records[i][2])
    docs = [" ".join(parts) for parts in doc_parts]
    try:
        ctfidf, words = _compute_ctfidf(docs, ngram_range=(1, 2), min_df=2)
    except Exception as exc:
        LOG.warning("paragraph c-TF-IDF failed (%s)", exc)
        ctfidf, words = np.zeros((k, 0), dtype=np.float32), []

    summaries: list[ParagraphClusterSummary] = []
    for cid in range(k):
        idxs = [i for i, c in enumerate(cluster_labels) if c == cid]
        if not idxs:
            continue
        sub = embs[idxs]
        norm = np.linalg.norm(sub.mean(axis=0))
        centroid = sub.mean(axis=0) / norm if norm > 0 else sub.mean(axis=0)
        sims = np.clip(sub @ centroid, -1.0, 1.0)
        cohesion = float(np.mean(sims))
        site_alignment = float(np.clip(centroid @ site_centroid, -1.0, 1.0))

        order = np.argsort(-sims)
        top_paragraphs_payload = []
        seen_urls: set[str] = set()
        for rank in order:
            pi, para_i, text, _ = paragraph_records[idxs[int(rank)]]
            url = pages[pi].url
            if url in seen_urls:
                continue
            seen_urls.add(url)
            top_paragraphs_payload.append({
                "url": url,
                "page_title": pages[pi].title,
                "excerpt": text[:240],
                "similarity": float(round(sims[int(rank)], 4)),
            })
            if len(top_paragraphs_payload) >= top_paragraphs:
                break

        keywords_payload: list[dict] = []
        if words:
            scores = ctfidf[cid]
            top_idx = np.argsort(-scores)[: top_k_keywords * 3]
            seen_substr: set[str] = set()
            for j in top_idx:
                if scores[j] <= 0:
                    continue
                kw = words[int(j)]
                if any(kw in s or s in kw for s in seen_substr if abs(len(kw) - len(s)) < 4 and kw != s):
                    continue
                seen_substr.add(kw)
                keywords_payload.append({"keyword": kw, "score": float(round(scores[int(j)], 4))})
                if len(keywords_payload) >= top_k_keywords:
                    break

        distinct_pages = len({pages[paragraph_records[i][0]].url for i in idxs})

        summaries.append(ParagraphClusterSummary(
            cluster_id=cid,
            paragraph_count=len(idxs),
            keywords=keywords_payload,
            cohesion=cohesion,
            site_alignment=site_alignment,
            distinct_pages=distinct_pages,
            top_paragraphs=top_paragraphs_payload,
            _centroid=centroid,
        ))

    summaries.sort(key=lambda s: (s.paragraph_count * s.cohesion, s.distinct_pages), reverse=True)
    return cluster_labels, summaries


def project_paragraphs(embs: np.ndarray, sample_cap: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (chosen_indices, coords[len(chosen), 2]).

    Sampling is uniform random when n exceeds ``sample_cap``; this keeps
    the scatter SVG-renderable for large sites.
    """
    n = len(embs)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 2), dtype=np.float32)

    if n > sample_cap:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(n, sample_cap, replace=False))
    else:
        chosen = np.arange(n)

    sub = embs[chosen]
    if len(sub) < 4:
        coords = np.zeros((len(sub), 2), dtype=np.float32)
        for i in range(len(sub)):
            coords[i, 0] = float(i)
        return chosen, coords

    if n > _native_max_points():
        coords = _pca_scores(sub.astype(np.float32), dims=2)
        mins = coords.min(axis=0, keepdims=True)
        spans = coords.max(axis=0, keepdims=True) - mins
        spans[spans == 0] = 1.0
        coords = (coords - mins) / spans
        return chosen, coords.astype(np.float32)

    import umap  # type: ignore

    n_neighbors = max(2, min(15, len(sub) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(sub.astype(np.float32))
    return chosen, coords.astype(np.float32)


def to_summary_payload(summaries: list[ParagraphClusterSummary]) -> list[dict]:
    return [
        {
            "cluster_id": s.cluster_id,
            "paragraph_count": s.paragraph_count,
            "distinct_pages": s.distinct_pages,
            "cohesion": round(s.cohesion, 4),
            "site_alignment": round(s.site_alignment, 4),
            "label": ", ".join(k["keyword"] for k in s.keywords[:4]) or f"cluster {s.cluster_id}",
            "keywords": s.keywords,
            "top_paragraphs": s.top_paragraphs,
        }
        for s in summaries
    ]


def to_overlap_payload(summaries: list[ParagraphClusterSummary], limit: int = 80) -> dict:
    """Return a centroid-similarity matrix for paragraph topic clusters."""
    usable = [s for s in summaries if getattr(s, "_centroid", None) is not None]
    usable.sort(key=lambda s: (s.paragraph_count, s.cohesion), reverse=True)
    usable = usable[: max(0, limit)]
    if not usable:
        return {"clusters": [], "matrix": []}

    centroids = np.stack([s._centroid for s in usable if s._centroid is not None]).astype(np.float32)
    sim = np.clip(centroids @ centroids.T, -1.0, 1.0)
    return {
        "clusters": [
            {
                "cluster_id": s.cluster_id,
                "label": ", ".join(k["keyword"] for k in s.keywords[:3]) or f"cluster {s.cluster_id}",
                "paragraph_count": s.paragraph_count,
                "distinct_pages": s.distinct_pages,
            }
            for s in usable
        ],
        "matrix": [[float(round(v, 4)) for v in row] for row in sim],
    }


def to_scatter_payload(
    paragraph_records: list,
    pages,
    cluster_labels: np.ndarray,
    chosen: np.ndarray,
    coords: np.ndarray,
    cluster_label_lookup: dict[int, str] | None = None,
) -> dict:
    rows: list[dict] = []
    for i, ci in enumerate(chosen):
        pi, para_i, text, _ = paragraph_records[int(ci)]
        rows.append({
            "url": pages[pi].url,
            "title": pages[pi].title,
            "paragraph_index": int(para_i),
            "excerpt": text[:200],
            "cluster": int(cluster_labels[int(ci)]),
            "cluster_label": (cluster_label_lookup or {}).get(int(cluster_labels[int(ci)]), ""),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        })
    return {
        "total_paragraphs": int(len(paragraph_records)),
        "shown": int(len(rows)),
        "num_clusters": int(max(cluster_labels) + 1) if len(cluster_labels) else 0,
        "paragraphs": rows,
    }
