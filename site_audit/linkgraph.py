"""Internal-link graph analyses: PageRank, orphans, dead-ends, link recs.

The crawler already records same-site outlinks per page, so we just
build a directed graph and run a few cheap analyses on top.

PageRank is implemented inline (no networkx dep) — for sites under
~50k pages this is fast enough and keeps the dependency list small.

Link recommendations rest on a simple observation: if two pages have
high cosine similarity (≥ ~0.85) but neither links to the other,
shipping a link between them is high-leverage internal SEO. AI engines
also follow internal anchors when reasoning about topical authority,
so this doubles as GEO advice.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class LinkGraphResult:
    in_degree: dict[str, int]
    out_degree: dict[str, int]
    pagerank: dict[str, float]
    orphans: list[str]              # in-degree == 0
    dead_ends: list[str]            # out-degree == 0
    edge_count: int
    recommendations: list[dict]     # [{from, to, similarity, both_in_section?}]


def build_graph(pages_with_outlinks: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    valid = {url for url, _ in pages_with_outlinks}
    graph: dict[str, list[str]] = {}
    for url, outs in pages_with_outlinks:
        seen: set[str] = set()
        clean = []
        for target in outs:
            if target == url or target not in valid or target in seen:
                continue
            seen.add(target)
            clean.append(target)
        graph[url] = clean
    return graph


def pagerank(graph: dict[str, list[str]], damping: float = 0.85, iterations: int = 50) -> dict[str, float]:
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    pr = {u: 1.0 / n for u in nodes}
    base = (1.0 - damping) / n
    for _ in range(iterations):
        new = {u: base for u in nodes}
        leak = 0.0
        for url, outs in graph.items():
            if not outs:
                leak += pr[url]
                continue
            share = damping * pr[url] / len(outs)
            for v in outs:
                new[v] += share
        # distribute "leaked" PR (from dead-ends) uniformly
        if leak:
            spread = damping * leak / n
            for u in nodes:
                new[u] += spread
        pr = new

    # normalize so values sum to 1.0 (they should already, modulo float drift)
    total = sum(pr.values()) or 1.0
    return {u: v / total for u, v in pr.items()}


def link_recommendations(
    pages,                                  # list of PageInfo
    embeddings: np.ndarray,
    graph: dict[str, list[str]],
    similarity_threshold: float = 0.85,
    knn: int = 20,
    top_k: int = 75,
) -> list[dict]:
    if len(pages) < 2:
        return []

    try:
        import faiss  # type: ignore
    except Exception:
        LOG.warning("faiss unavailable — skipping link recommendations")
        return []

    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    k = min(knn, n)
    sims, idxs = index.search(embeddings.astype(np.float32), k)

    edge_set: set[tuple[str, str]] = set()
    for src, outs in graph.items():
        for tgt in outs:
            edge_set.add((src, tgt))

    seen_pairs: set[tuple[int, int]] = set()
    recs: list[dict] = []
    for i in range(n):
        for rank in range(k):
            j = int(idxs[i, rank])
            sim = float(sims[i, rank])
            if j == i or sim < similarity_threshold:
                continue
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            url_a = pages[i].url
            url_b = pages[j].url
            if (url_a, url_b) in edge_set or (url_b, url_a) in edge_set:
                continue
            recs.append({
                "url_a": url_a,
                "title_a": pages[i].title,
                "section_a": pages[i].section,
                "url_b": url_b,
                "title_b": pages[j].title,
                "section_b": pages[j].section,
                "similarity": round(sim, 4),
                "same_section": pages[i].section == pages[j].section,
            })

    recs.sort(key=lambda r: r["similarity"], reverse=True)
    return recs[:top_k]


def analyze(
    pages,                              # list of PageInfo
    embeddings: np.ndarray,
    pages_with_outlinks: list[tuple[str, list[str]]],
    similarity_threshold: float = 0.85,
    top_recommendations: int = 75,
) -> LinkGraphResult:
    graph = build_graph(pages_with_outlinks)

    in_deg: dict[str, int] = defaultdict(int)
    for outs in graph.values():
        for tgt in outs:
            in_deg[tgt] += 1
    out_deg = {u: len(o) for u, o in graph.items()}
    edge_count = sum(out_deg.values())

    orphans = sorted([u for u in graph if in_deg.get(u, 0) == 0])
    dead_ends = sorted([u for u, n in out_deg.items() if n == 0])

    pr = pagerank(graph)

    recs = link_recommendations(
        pages,
        embeddings,
        graph,
        similarity_threshold=similarity_threshold,
        top_k=top_recommendations,
    )

    return LinkGraphResult(
        in_degree=dict(in_deg),
        out_degree=out_deg,
        pagerank=pr,
        orphans=orphans,
        dead_ends=dead_ends,
        edge_count=edge_count,
        recommendations=recs,
    )


def to_payload(result: LinkGraphResult, pages, top_n: int = 25) -> dict:
    by_url = {p.url: p for p in pages}
    pr_sorted = sorted(result.pagerank.items(), key=lambda kv: kv[1], reverse=True)
    top_pages = []
    for url, pr in pr_sorted[:top_n]:
        p = by_url.get(url)
        top_pages.append({
            "url": url,
            "title": p.title if p else url,
            "section": p.section if p else "",
            "pagerank": round(float(pr), 6),
            "in_degree": int(result.in_degree.get(url, 0)),
            "out_degree": int(result.out_degree.get(url, 0)),
        })

    orphans_payload = []
    for url in result.orphans[:top_n]:
        p = by_url.get(url)
        orphans_payload.append({
            "url": url,
            "title": p.title if p else url,
            "section": p.section if p else "",
            "out_degree": int(result.out_degree.get(url, 0)),
        })

    return {
        "edge_count": result.edge_count,
        "node_count": len(result.in_degree) + len(set(result.out_degree.keys()) - set(result.in_degree.keys())),
        "orphan_count": len(result.orphans),
        "dead_end_count": len(result.dead_ends),
        "top_authority_pages": top_pages,
        "orphans": orphans_payload,
        "recommendations": result.recommendations,
    }
