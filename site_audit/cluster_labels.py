"""Auto-label k-means clusters with class-discriminative keywords (c-TF-IDF).

This is the BERTopic recipe in ~30 lines: treat each cluster as one big
"document" (the concatenation of its pages' text), then for each
candidate term compute

    tf_class_x  = freq_x_in_class / total_terms_in_class
    idf_x       = log(avg_terms_per_class / freq_x_across_all_classes)
    score_x     = tf_class_x * idf_x

The top-scoring terms per class are by definition the ones that
characterize *this* cluster relative to the rest of the site, which is
exactly what we want for a scatter-plot legend.

We also report a couple of cluster-level metrics that the report uses
to rank clusters by importance:

* ``page_count``      — size of the cluster
* ``cohesion``        — mean cosine similarity between cluster pages and
                        their centroid; high = tight topic
* ``site_alignment``  — cosine of cluster centroid to site centroid;
                        high = core to the site, low = peripheral
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOG = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 2


@dataclass
class ClusterSummary:
    cluster_id: int
    page_count: int
    keywords: list[dict]            # [{keyword, score}]
    top_pages: list[dict]           # [{title, url, similarity}]
    cohesion: float
    site_alignment: float


def _aggregate_texts(texts: Sequence[str], cluster_labels: np.ndarray, n_clusters: int) -> list[str]:
    docs = [""] * n_clusters
    for text, c in zip(texts, cluster_labels):
        docs[int(c)] = (docs[int(c)] + " " + (text or "")).strip()
    return docs


def _compute_ctfidf(
    docs: list[str],
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_features: int = 20_000,
) -> tuple[np.ndarray, list[str]]:
    """Return (ctfidf[n_clusters, n_features], feature_names)."""
    from sklearn.feature_extraction.text import CountVectorizer

    if not any(docs):
        return np.zeros((len(docs), 0), dtype=np.float32), []

    vec = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]+\b",
    )
    counts = vec.fit_transform(docs).toarray().astype(np.float32)
    words = list(vec.get_feature_names_out())

    total_per_class = counts.sum(axis=1, keepdims=True)
    total_per_class[total_per_class == 0] = 1.0
    tf = counts / total_per_class

    avg_terms = float(total_per_class.mean())
    word_freq = counts.sum(axis=0)
    word_freq = np.where(word_freq == 0, 1.0, word_freq)
    idf = np.log(avg_terms / word_freq)

    return tf * idf, words


def label_clusters(
    pages,                              # list of PageInfo
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    site_centroid: np.ndarray,
    cluster_texts: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    top_pages_per_cluster: int = 5,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
) -> list[ClusterSummary]:
    """Compute keywords + cohesion + site-alignment per cluster.

    ``cluster_texts`` should be the same text used for embedding
    (title + description + body). If omitted we fall back to the page
    title — which still produces something but with much weaker labels.
    """
    if len(cluster_labels) == 0:
        return []

    n_clusters = int(max(cluster_labels) + 1)
    if n_clusters <= 0:
        return []

    if cluster_texts is None:
        cluster_texts = [p.title for p in pages]

    by_cluster: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for i, c in enumerate(cluster_labels):
        by_cluster[int(c)].append(i)

    docs = _aggregate_texts(cluster_texts, cluster_labels, n_clusters)
    try:
        ctfidf, words = _compute_ctfidf(docs, ngram_range=ngram_range, min_df=min_df)
    except Exception as exc:  # pragma: no cover (sklearn absent / edge case)
        LOG.warning("c-TF-IDF failed (%s) — clusters will have no keywords", exc)
        ctfidf, words = np.zeros((n_clusters, 0), dtype=np.float32), []

    summaries: list[ClusterSummary] = []
    for cid in range(n_clusters):
        idxs = by_cluster[cid]
        if not idxs:
            continue

        # cohesion = mean cosine of each page in cluster to cluster centroid
        sub = embeddings[idxs]
        norm = np.linalg.norm(sub.mean(axis=0))
        if norm == 0:
            centroid = sub.mean(axis=0)
        else:
            centroid = sub.mean(axis=0) / norm
        sims_to_centroid = np.clip(sub @ centroid, -1.0, 1.0)
        cohesion = float(np.mean(sims_to_centroid))

        site_alignment = float(np.clip(centroid @ site_centroid, -1.0, 1.0))

        # top pages = pages closest to centroid
        order = np.argsort(-sims_to_centroid)
        top_pages_payload = []
        for k in order[:top_pages_per_cluster]:
            page_i = idxs[int(k)]
            top_pages_payload.append({
                "title": pages[page_i].title,
                "url": pages[page_i].url,
                "similarity": float(round(sims_to_centroid[int(k)], 4)),
            })

        keywords_payload: list[dict] = []
        if words:
            scores = ctfidf[cid]
            top_idx = np.argsort(-scores)[:top_k * 3]  # over-pull, then de-dupe substrings
            seen_substr: set[str] = set()
            for j in top_idx:
                if scores[j] <= 0:
                    continue
                kw = words[int(j)]
                if any(kw in s or s in kw for s in seen_substr if abs(len(kw) - len(s)) < 4 and kw != s):
                    continue
                seen_substr.add(kw)
                keywords_payload.append({"keyword": kw, "score": float(round(scores[int(j)], 4))})
                if len(keywords_payload) >= top_k:
                    break

        summaries.append(ClusterSummary(
            cluster_id=cid,
            page_count=len(idxs),
            keywords=keywords_payload,
            top_pages=top_pages_payload,
            cohesion=cohesion,
            site_alignment=site_alignment,
        ))

    # Rank: bigger + more cohesive clusters first; site-aligned ones break ties.
    summaries.sort(key=lambda s: (s.page_count * s.cohesion, s.site_alignment), reverse=True)
    return summaries
