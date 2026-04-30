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
    )
