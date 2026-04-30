"""Compute the audit metrics over a matrix of normalized page embeddings.

The math mirrors ``generate_site_audit.py`` from the Hugo project so the
two reports stay directly comparable:

* siteFocusScore — mean cosine similarity to the global centroid
* siteRadius     — std-dev of cosine distance to the global centroid
* per-section focus / radius / p95 distance
* near-duplicate pairs via FAISS kNN
* drift to site centroid + drift to section centroid per page
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)


# --- helpers ---------------------------------------------------------------


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def focus_metrics(embeddings: np.ndarray, centroid: np.ndarray) -> dict:
    sims = np.clip(embeddings @ centroid, -1.0, 1.0)
    distances = 1.0 - sims
    return {
        "focus_score": float(np.mean(sims)),
        "radius": float(np.std(distances)),
        "mean_distance": float(np.mean(distances)),
        "p95_distance": float(np.percentile(distances, 95)),
        "max_distance": float(np.max(distances)),
        "count": int(len(embeddings)),
    }


def pairwise_stats(embeddings: np.ndarray, max_sample: int = 3000, seed: int = 0) -> dict:
    """Pairwise cosine-similarity distribution.

    The mean is the most useful single number here because — unlike
    centroid alignment — it doesn't care about geometry, just about how
    similar the average pair of pages looks. The p10 percentile is the
    model's anisotropy floor (gte-multilingual-base lands around 0.5 even
    on completely unrelated text), so we expose it as the "model floor"
    that ``calibrated_focus`` subtracts off.
    """
    n = len(embeddings)
    if n < 2:
        return {"sample_size": n, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    if n > max_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_sample, replace=False)
        sub = embeddings[idx]
    else:
        sub = embeddings

    sims = sub @ sub.T
    np.fill_diagonal(sims, np.nan)
    flat = sims[~np.isnan(sims)]
    return {
        "sample_size": int(len(sub)),
        "mean": float(np.mean(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
    }


def section_coherence_ratio(
    embeddings: np.ndarray,
    pages: list,
    max_sample: int = 3000,
    seed: int = 0,
) -> dict:
    """Mean intra-section similarity divided by mean inter-section similarity.

    A site with strong topical structure scores >1.5; a site where URL
    sections don't reflect content structure scores ~1.0. This is the
    single most diagnostic GEO signal — it answers "do my URL paths
    match my content".
    """
    n = len(pages)
    if n < 4:
        return {"sample_size": n, "intra": 0.0, "inter": 0.0, "ratio": 0.0}

    if n > max_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_sample, replace=False)
        sub = embeddings[idx]
        sub_pages = [pages[i] for i in idx]
    else:
        sub = embeddings
        sub_pages = pages

    sections = np.array([p.section for p in sub_pages])
    sims = sub @ sub.T
    n_sub = len(sub)

    same = sections[:, None] == sections[None, :]
    diag = np.eye(n_sub, dtype=bool)

    intra_mask = same & ~diag
    inter_mask = ~same

    intra_vals = sims[intra_mask]
    inter_vals = sims[inter_mask]
    if len(intra_vals) == 0 or len(inter_vals) == 0:
        return {"sample_size": int(n_sub), "intra": 0.0, "inter": 0.0, "ratio": 0.0}

    intra = float(np.mean(intra_vals))
    inter = float(np.mean(inter_vals))
    ratio = intra / inter if inter > 0 else 0.0

    return {
        "sample_size": int(n_sub),
        "intra": intra,
        "inter": inter,
        "ratio": ratio,
    }


def effective_topic_dimension(embeddings: np.ndarray, max_sample: int = 3000, seed: int = 0) -> dict:
    """Effective topic dimension via spectral entropy of the covariance.

    PCA the (mean-centered) embeddings, normalize the eigenvalues to a
    probability distribution, then return ``exp(H)`` where ``H`` is the
    Shannon entropy of that distribution. Intuitively this is the
    "effective number of independent topics" the site spans:

    * 2–4 → laser focused on one or two themes (a single-product site)
    * 5–10 → focused with a few sub-topics (a vertical SaaS)
    * 15–30 → broad publisher / multi-product site
    * 50+ → unfocused, no dominant axis

    This metric is *not* model-anchored, unlike focus_score, so the
    absolute number is comparable across sites and corpora.
    """
    n = len(embeddings)
    if n < 4:
        return {"effective_dim": float(n), "top_eigenvalue_share": 1.0, "sample_size": n}
    if n > max_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_sample, replace=False)
        sub = embeddings[idx]
    else:
        sub = embeddings
    centered = sub - sub.mean(axis=0, keepdims=True)
    # SVD beats eigendecomposition of cov for skinny matrices (n << d=768).
    # Singular values^2 are eigenvalues of the (sample) covariance matrix.
    try:
        s = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"effective_dim": float(n), "top_eigenvalue_share": 1.0, "sample_size": int(len(sub))}
    eig = (s ** 2).astype(np.float64)
    total = float(eig.sum())
    if total <= 0:
        return {"effective_dim": float(n), "top_eigenvalue_share": 1.0, "sample_size": int(len(sub))}
    p = eig / total
    p_safe = p[p > 1e-12]
    H = float(-np.sum(p_safe * np.log(p_safe)))
    return {
        "effective_dim": float(np.exp(H)),
        "top_eigenvalue_share": float(p[0]) if len(p) else 0.0,
        "top5_eigenvalue_share": float(p[:5].sum()) if len(p) >= 1 else 0.0,
        "sample_size": int(len(sub)),
    }


def calibrated_focus(focus_score: float, p10_pairwise: float) -> float:
    """Strip out the embedding model's anisotropy floor.

    Modern multilingual models like ``gte-multilingual-base`` produce
    cosine similarities in roughly [0.5, 0.9] for any English text — the
    raw focus_score never goes near 0 even for fragmented sites. Using
    the site's own p10 pairwise similarity as the floor recovers a [0, 1]
    interpretation: 0 means "no more focused than 10% of random page
    pairs", 1 means "every page lies on the centroid".
    """
    if p10_pairwise >= 1.0:
        return 0.0
    val = (focus_score - p10_pairwise) / (1.0 - p10_pairwise)
    if val < 0:
        return 0.0
    if val > 1:
        return 1.0
    return float(val)


def centroid_distribution(embeddings: np.ndarray, centroid: np.ndarray, bins: int = 20) -> dict:
    """Histogram of per-page cosine similarity to the global centroid.

    Reporting just the mean is misleading because two sites with very
    different shapes can land on the same mean. The histogram tells the
    user whether the distribution is unimodal-tight or bimodal-spread.
    """
    sims = np.clip(embeddings @ centroid, -1.0, 1.0)
    if len(sims) == 0:
        return {"bins": [], "counts": []}
    lo = float(min(0.0, sims.min()))
    hi = float(max(1.0, sims.max()))
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(sims, bins=edges)
    return {
        "lo": lo,
        "hi": hi,
        "edges": [float(e) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "min": float(sims.min()),
        "max": float(sims.max()),
        "median": float(np.median(sims)),
    }


# --- structures ------------------------------------------------------------


@dataclass
class PageInfo:
    url: str
    title: str
    description: str
    section: str
    word_count: int
    language: str | None = None


@dataclass
class SectionStats:
    name: str
    indices: list[int]
    centroid: np.ndarray
    metrics: dict


@dataclass
class AuditResult:
    pages: list[PageInfo]
    embeddings: np.ndarray
    site_centroid: np.ndarray
    site_metrics: dict
    sections: dict[str, SectionStats]
    dist_to_site: np.ndarray
    dist_to_section: np.ndarray
    duplicate_pairs: list[tuple[int, int, float]]
    duplicate_partners: dict[int, list[dict]] = field(default_factory=dict)
    pairwise: dict = field(default_factory=dict)
    coherence: dict = field(default_factory=dict)
    topic_dim: dict = field(default_factory=dict)
    centroid_hist: dict = field(default_factory=dict)
    calibrated_focus_score: float = 0.0


# --- core ------------------------------------------------------------------


def section_for_url(url: str) -> str:
    """Use the first non-empty path segment as the section name."""
    from urllib.parse import urlparse

    path = urlparse(url).path or "/"
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "root"
    if len(parts) == 1:
        # heuristic: treat single-segment URLs as their own page in 'root'
        # so we don't fragment the chart with one-page sections.
        return "root"
    return parts[0]


def build_section_stats(pages: list[PageInfo], embeddings: np.ndarray) -> dict[str, SectionStats]:
    bucket: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pages):
        bucket[p.section].append(i)

    out: dict[str, SectionStats] = {}
    for section, idxs in bucket.items():
        sub = embeddings[idxs]
        centroid = l2_normalize(sub.mean(axis=0))
        out[section] = SectionStats(
            name=section,
            indices=idxs,
            centroid=centroid,
            metrics=focus_metrics(sub, centroid),
        )
    return out


def find_near_duplicates(
    embeddings: np.ndarray,
    threshold: float = 0.92,
    knn: int = 10,
) -> list[tuple[int, int, float]]:
    if len(embeddings) == 0:
        return []
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError("faiss is required for duplicate detection") from exc

    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    k = min(knn, n)
    sims, idxs = index.search(embeddings.astype(np.float32), k)

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for rank in range(k):
            j = int(idxs[i, rank])
            sim = float(sims[i, rank])
            if j < 0 or j == i or sim < threshold:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key[0], key[1], sim))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def recommend_action(
    page: PageInfo,
    dist_site: float,
    dist_section: float,
    section_p95: float,
    section_size: int,
    has_duplicate: bool,
) -> str:
    reasons = []
    if has_duplicate:
        reasons.append("near-duplicate: merge or canonicalize")
    if section_size < 3:
        reasons.append("orphan section: expand or consolidate")
    if dist_section > section_p95:
        reasons.append("off-topic for its section: refocus or move")
    if dist_site > 0.65:
        reasons.append("off-brand for the whole site: consider removing")
    if page.word_count < 200 and dist_section > section_p95 * 0.8:
        reasons.append("thin + off-topic: strong removal candidate")
    return "; ".join(reasons)


def analyze(
    pages: list[PageInfo],
    embeddings: np.ndarray,
    duplicate_threshold: float = 0.92,
    duplicate_knn: int = 10,
) -> AuditResult:
    site_centroid = l2_normalize(embeddings.mean(axis=0))
    site_metrics = focus_metrics(embeddings, site_centroid)

    sections = build_section_stats(pages, embeddings)

    dist_to_site = 1.0 - np.clip(embeddings @ site_centroid, -1.0, 1.0)

    dist_to_section = np.zeros(len(pages), dtype=np.float32)
    for stats in sections.values():
        for idx in stats.indices:
            sim = float(np.clip(embeddings[idx] @ stats.centroid, -1.0, 1.0))
            dist_to_section[idx] = 1.0 - sim

    pairs = find_near_duplicates(embeddings, duplicate_threshold, duplicate_knn)

    partners: dict[int, list[dict]] = defaultdict(list)
    for i, j, sim in pairs:
        partners[i].append({"url": pages[j].url, "title": pages[j].title, "similarity": round(sim, 4)})
        partners[j].append({"url": pages[i].url, "title": pages[i].title, "similarity": round(sim, 4)})

    pw = pairwise_stats(embeddings)
    coherence = section_coherence_ratio(embeddings, pages)
    topic_dim = effective_topic_dimension(embeddings)
    hist = centroid_distribution(embeddings, site_centroid)
    cal_focus = calibrated_focus(site_metrics["focus_score"], pw.get("p10", 0.0))

    return AuditResult(
        pages=pages,
        embeddings=embeddings,
        site_centroid=site_centroid,
        site_metrics=site_metrics,
        sections=sections,
        dist_to_site=dist_to_site,
        dist_to_section=dist_to_section,
        duplicate_pairs=pairs,
        duplicate_partners=dict(partners),
        pairwise=pw,
        coherence=coherence,
        topic_dim=topic_dim,
        centroid_hist=hist,
        calibrated_focus_score=cal_focus,
    )
