"""Internal-link graph analyses.

Built on top of the same-site outlinks the crawler records per page.
We compute:

* **PageRank** — damped iterative, no networkx dep.
* **HITS** — hubs vs authorities. Hubs link to many authoritative
  pages; authorities are linked from many hubs. PageRank treats both
  the same; HITS separates them, which is more diagnostic for
  link-building because hub status comes from *outbound* structure.
* **Click depth** — BFS shortest-path from the homepage. Pages > 4
  clicks deep are effectively buried for both crawlers and users.
* **Orphans / dead-ends** — in-degree == 0 / out-degree == 0.
* **Topic-cluster authorities** — per cluster, the page with the
  highest PageRank. These are the canonical entry pages per topic.
* **Anchor-text analysis** per target — top anchors, generic-anchor
  share, anchor↔target topic mismatch (via embeddings of anchor text).
* **Internal-link recommendations** — high-similarity page pairs that
  aren't currently linked.
"""

from __future__ import annotations

import collections
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)


_GENERIC_ANCHORS = {
    "click here", "click", "here", "this", "this page", "more", "read more",
    "learn more", "see more", "find out more", "more info", "go", "details",
    "link", "this link", "website", "site", "page", "view", "view more",
}


@dataclass
class LinkGraphResult:
    in_degree: dict[str, int]
    out_degree: dict[str, int]
    pagerank: dict[str, float]
    hub_score: dict[str, float]
    authority_score: dict[str, float]
    click_depth: dict[str, int]
    orphans: list[str]
    dead_ends: list[str]
    edge_count: int
    recommendations: list[dict]
    cluster_authorities: list[dict] = field(default_factory=list)
    anchor_analysis: list[dict] = field(default_factory=list)


# --- Graph construction ---------------------------------------------------


def build_graph(
    pages_with_outlinks: list[tuple[str, list[tuple[str, str]]]],
) -> tuple[dict[str, list[str]], dict[tuple[str, str], list[str]]]:
    """Return (graph, anchors_by_edge)."""
    valid = {url for url, _ in pages_with_outlinks}
    graph: dict[str, list[str]] = {}
    anchors: dict[tuple[str, str], list[str]] = defaultdict(list)
    for url, outs in pages_with_outlinks:
        seen: set[str] = set()
        clean: list[str] = []
        for target, anchor in outs:
            if target == url or target not in valid:
                continue
            if anchor:
                anchors[(url, target)].append(anchor)
            if target in seen:
                continue
            seen.add(target)
            clean.append(target)
        graph[url] = clean
    return graph, dict(anchors)


# --- PageRank -------------------------------------------------------------


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
        if leak:
            spread = damping * leak / n
            for u in nodes:
                new[u] += spread
        pr = new
    total = sum(pr.values()) or 1.0
    return {u: v / total for u, v in pr.items()}


# --- HITS hubs / authorities ---------------------------------------------


def hits(graph: dict[str, list[str]], iterations: int = 30) -> tuple[dict[str, float], dict[str, float]]:
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}, {}
    in_edges: dict[str, list[str]] = defaultdict(list)
    for src, outs in graph.items():
        for tgt in outs:
            in_edges[tgt].append(src)

    hub = {u: 1.0 for u in nodes}
    auth = {u: 1.0 for u in nodes}

    for _ in range(iterations):
        # authority = sum of hub scores of pages linking in
        new_auth = {u: 0.0 for u in nodes}
        for u in nodes:
            for src in in_edges.get(u, []):
                new_auth[u] += hub.get(src, 0.0)
        # normalize
        norm = (sum(v * v for v in new_auth.values())) ** 0.5 or 1.0
        auth = {u: v / norm for u, v in new_auth.items()}

        # hub = sum of authority scores of pages it links to
        new_hub = {u: 0.0 for u in nodes}
        for u in nodes:
            for tgt in graph.get(u, []):
                new_hub[u] += auth.get(tgt, 0.0)
        norm = (sum(v * v for v in new_hub.values())) ** 0.5 or 1.0
        hub = {u: v / norm for u, v in new_hub.items()}

    return hub, auth


# --- Click depth ----------------------------------------------------------


def click_depth(graph: dict[str, list[str]], start_url: str) -> dict[str, int]:
    if not graph:
        return {}
    if start_url not in graph:
        # Fallback: find the shortest URL whose host root matches.
        from urllib.parse import urlparse
        try:
            host = urlparse(start_url).netloc.lower()
            host_root = host[4:] if host.startswith("www.") else host
        except Exception:
            host_root = ""
        def _matches_host(url: str) -> bool:
            try:
                u_host = urlparse(url).netloc.lower()
            except Exception:
                return False
            u_root = u_host[4:] if u_host.startswith("www.") else u_host
            return u_root == host_root if host_root else True

        candidates = [u for u in graph if _matches_host(u)]
        if not candidates:
            candidates = list(graph.keys())
        # prefer shortest path (closest to "/" homepage), break ties by total length
        start_url = min(candidates, key=lambda u: (urlparse(u).path.count("/"), len(u)))

    depths: dict[str, int] = {start_url: 0}
    queue = collections.deque([start_url])
    while queue:
        u = queue.popleft()
        d = depths[u]
        for v in graph.get(u, []):
            if v not in depths:
                depths[v] = d + 1
                queue.append(v)
    return depths


# --- Cluster authorities --------------------------------------------------


def cluster_authorities(
    pages,                               # list of PageInfo
    cluster_labels: np.ndarray | None,
    pagerank_scores: dict[str, float],
    cluster_summaries=None,
) -> list[dict]:
    if cluster_labels is None or len(cluster_labels) == 0:
        return []
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_labels):
        by_cluster[int(c)].append(i)

    label_lookup: dict[int, str] = {}
    if cluster_summaries:
        for s in cluster_summaries:
            label_lookup[s.cluster_id] = ", ".join(k["keyword"] for k in s.keywords[:3]) or f"cluster {s.cluster_id}"

    out: list[dict] = []
    for cid, idxs in by_cluster.items():
        # find best page in cluster by pagerank
        best_i = max(idxs, key=lambda i: pagerank_scores.get(pages[i].url, 0.0))
        runners = sorted(idxs, key=lambda i: pagerank_scores.get(pages[i].url, 0.0), reverse=True)[1:4]
        out.append({
            "cluster_id": cid,
            "label": label_lookup.get(cid, f"cluster {cid}"),
            "page_count": len(idxs),
            "authority": {
                "url": pages[best_i].url,
                "title": pages[best_i].title,
                "pagerank": round(float(pagerank_scores.get(pages[best_i].url, 0.0)), 6),
            },
            "runners_up": [
                {"url": pages[i].url, "title": pages[i].title, "pagerank": round(float(pagerank_scores.get(pages[i].url, 0.0)), 6)}
                for i in runners
            ],
        })
    out.sort(key=lambda r: r["page_count"], reverse=True)
    return out


# --- Anchor-text analysis -------------------------------------------------


def _is_generic_anchor(text: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned in _GENERIC_ANCHORS or len(cleaned) <= 2


def anchor_analysis(
    pages,                                    # list of PageInfo
    embeddings: np.ndarray,
    anchors_by_edge: dict[tuple[str, str], list[str]],
    embedder=None,                            # Embedder, optional, for anchor-target similarity
    top_n: int = 25,
) -> list[dict]:
    """Per-target anchor profile + topic-mismatch via embeddings.

    If ``embedder`` is provided we also compute the cosine between the
    anchor text and the target page's embedding, which surfaces
    "anchor doesn't describe target" issues.
    """
    if not anchors_by_edge:
        return []

    by_target: dict[str, list[tuple[str, str]]] = defaultdict(list)  # target -> [(src, anchor), ...]
    for (src, tgt), anchors in anchors_by_edge.items():
        for a in anchors:
            by_target[tgt].append((src, a))

    page_idx = {p.url: i for i, p in enumerate(pages)}
    rows: list[dict] = []

    # Precompute anchor embeddings in one batch when an embedder is supplied.
    anchor_text_to_emb: dict[str, np.ndarray] = {}
    if embedder is not None:
        unique_anchors = sorted({a for entries in by_target.values() for _, a in entries if a and not _is_generic_anchor(a)})
        unique_anchors = unique_anchors[:2000]  # cap embedding cost
        if unique_anchors:
            embs = embedder.encode(unique_anchors, batch_size=64, show_progress=False)
            for txt, vec in zip(unique_anchors, embs):
                anchor_text_to_emb[txt] = vec

    for tgt, entries in by_target.items():
        i = page_idx.get(tgt)
        if i is None:
            continue
        anchors = [a for _, a in entries if a]
        if not anchors:
            continue
        from collections import Counter
        counter = Counter(a.strip().lower() for a in anchors)
        top_anchors = counter.most_common(5)
        generic = sum(c for a, c in counter.items() if _is_generic_anchor(a))
        descriptive = len(anchors) - generic

        # Anchor↔target topic mismatch
        mismatch_score: float | None = None
        worst_anchor: str | None = None
        if anchor_text_to_emb:
            best_emb = embeddings[i]
            sims = []
            for a in {x.strip() for x in anchors}:
                vec = anchor_text_to_emb.get(a)
                if vec is None:
                    vec = anchor_text_to_emb.get(a.lower())
                if vec is None:
                    continue
                sim = float(np.clip(vec @ best_emb, -1.0, 1.0))
                sims.append((a, sim))
            if sims:
                sims.sort(key=lambda x: x[1])
                worst_anchor, mismatch_score = sims[0]
                # report 1 - sim so higher number = bigger problem
                mismatch_score = round(1.0 - float(mismatch_score), 4)

        rows.append({
            "target_url": tgt,
            "target_title": pages[i].title,
            "inbound_link_count": len(anchors),
            "unique_anchors": len(counter),
            "generic_anchor_share": round(generic / max(1, len(anchors)), 3),
            "descriptive_anchor_share": round(descriptive / max(1, len(anchors)), 3),
            "top_anchors": [{"anchor": a, "count": c} for a, c in top_anchors],
            "worst_anchor_mismatch": worst_anchor,
            "worst_anchor_distance": mismatch_score,
        })

    rows.sort(key=lambda r: (r["generic_anchor_share"], r["worst_anchor_distance"] or 0), reverse=True)
    return rows[:top_n]


# --- Internal-link recommendations ---------------------------------------


def link_recommendations(
    pages,
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


# --- Top-level entry point ------------------------------------------------


def analyze(
    pages,
    embeddings: np.ndarray,
    pages_with_outlinks: list[tuple[str, list[tuple[str, str]]]],
    home_url: str,
    cluster_labels: np.ndarray | None = None,
    cluster_summaries=None,
    embedder=None,
    similarity_threshold: float = 0.85,
    top_recommendations: int = 75,
) -> LinkGraphResult:
    graph, anchors = build_graph(pages_with_outlinks)

    in_deg: dict[str, int] = defaultdict(int)
    for outs in graph.values():
        for tgt in outs:
            in_deg[tgt] += 1
    out_deg = {u: len(o) for u, o in graph.items()}
    edge_count = sum(out_deg.values())

    pr = pagerank(graph)
    hub, auth = hits(graph)
    depth = click_depth(graph, home_url)

    orphans = sorted([u for u in graph if in_deg.get(u, 0) == 0])
    dead_ends = sorted([u for u, n in out_deg.items() if n == 0])

    cluster_auth = cluster_authorities(pages, cluster_labels, pr, cluster_summaries)
    anchors_payload = anchor_analysis(pages, embeddings, anchors, embedder=embedder)

    recs = link_recommendations(
        pages, embeddings, graph,
        similarity_threshold=similarity_threshold,
        top_k=top_recommendations,
    )

    return LinkGraphResult(
        in_degree=dict(in_deg),
        out_degree=out_deg,
        pagerank=pr,
        hub_score=hub,
        authority_score=auth,
        click_depth=depth,
        orphans=orphans,
        dead_ends=dead_ends,
        edge_count=edge_count,
        recommendations=recs,
        cluster_authorities=cluster_auth,
        anchor_analysis=anchors_payload,
    )


# --- Payload --------------------------------------------------------------


def to_payload(result: LinkGraphResult, pages, top_n: int = 25) -> dict:
    by_url = {p.url: p for p in pages}

    def _enrich(url: str) -> dict:
        p = by_url.get(url)
        return {
            "url": url,
            "title": p.title if p else url,
            "section": p.section if p else "",
            "pagerank": round(float(result.pagerank.get(url, 0.0)), 6),
            "hub_score": round(float(result.hub_score.get(url, 0.0)), 6),
            "authority_score": round(float(result.authority_score.get(url, 0.0)), 6),
            "click_depth": int(result.click_depth.get(url, -1)) if url in result.click_depth else None,
            "in_degree": int(result.in_degree.get(url, 0)),
            "out_degree": int(result.out_degree.get(url, 0)),
        }

    pr_sorted = sorted(result.pagerank.items(), key=lambda kv: kv[1], reverse=True)
    auth_sorted = sorted(result.authority_score.items(), key=lambda kv: kv[1], reverse=True)
    hub_sorted = sorted(result.hub_score.items(), key=lambda kv: kv[1], reverse=True)

    deep_pages = sorted(
        [(u, d) for u, d in result.click_depth.items() if d >= 4],
        key=lambda kv: kv[1], reverse=True,
    )

    # Per-page in/out degree for the full crawled set — drives the
    # "internal links per page" chart on the UI side. We keep just url +
    # title + degrees + click_depth so the JSON stays small even on big sites.
    page_link_counts: list[dict] = []
    for p in pages:
        page_link_counts.append({
            "url": p.url,
            "title": p.title,
            "section": p.section,
            "in_degree": int(result.in_degree.get(p.url, 0)),
            "out_degree": int(result.out_degree.get(p.url, 0)),
            "click_depth": int(result.click_depth.get(p.url, -1)) if p.url in result.click_depth else None,
        })

    return {
        "edge_count": result.edge_count,
        "node_count": len(set(list(result.in_degree.keys()) + list(result.out_degree.keys()))),
        "orphan_count": len(result.orphans),
        "dead_end_count": len(result.dead_ends),
        "deep_page_count": len(deep_pages),
        "max_click_depth": max(result.click_depth.values()) if result.click_depth else 0,
        "top_authority_pages": [_enrich(u) for u, _ in pr_sorted[:top_n]],
        "top_hits_authorities": [_enrich(u) for u, _ in auth_sorted[:top_n]],
        "top_hits_hubs": [_enrich(u) for u, _ in hub_sorted[:top_n]],
        "deep_pages": [{"url": u, "click_depth": d, "title": (by_url.get(u).title if by_url.get(u) else u)} for u, d in deep_pages[:top_n]],
        "orphans": [_enrich(u) for u in result.orphans[:top_n]],
        "dead_ends": [_enrich(u) for u in result.dead_ends[:top_n]],
        "cluster_authorities": result.cluster_authorities,
        "anchor_analysis": result.anchor_analysis,
        "recommendations": result.recommendations,
        "page_link_counts": page_link_counts,
    }
