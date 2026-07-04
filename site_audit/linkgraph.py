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
import bisect
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

import numpy as np

LOG = logging.getLogger(__name__)


_GENERIC_ANCHORS = {
    "click here", "click", "here", "this", "this page", "more", "read more",
    "learn more", "see more", "find out more", "more info", "go", "details",
    "link", "this link", "website", "site", "page", "view", "view more",
}


@dataclass
class LinkGraphResult:
    graph: dict[str, list[str]]
    edge_anchor_texts: dict[tuple[str, str], list[str]]
    edge_anchor_count: dict[tuple[str, str], int]
    in_degree: dict[str, int]
    out_degree: dict[str, int]
    pagerank: dict[str, float]
    hub_score: dict[str, float]
    authority_score: dict[str, float]
    click_depth: dict[str, int]
    edge_anchor_quality: dict[tuple[str, str], float]
    edge_contextual_relevance: dict[tuple[str, str], float]
    orphans: list[str]
    dead_ends: list[str]
    edge_count: int
    recommendations: list[dict]
    internal_http_links: dict[str, list[str]] = field(default_factory=dict)
    internal_https_links: dict[str, list[str]] = field(default_factory=dict)
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


def _outgoing_links_by_scheme(
    pages_with_outlinks: list[tuple[str, list[tuple[str, str]]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    http_links: dict[str, list[str]] = {}
    https_links: dict[str, list[str]] = {}
    for url, outs in pages_with_outlinks:
        seen_http: set[str] = set()
        seen_https: set[str] = set()
        for target, _anchor in outs:
            scheme = urlparse(target or "").scheme.lower()
            if scheme == "http" and target not in seen_http:
                seen_http.add(target)
                http_links.setdefault(url, []).append(target)
            elif scheme == "https" and target not in seen_https:
                seen_https.add(target)
                https_links.setdefault(url, []).append(target)
    return http_links, https_links


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


def weighted_pagerank(
    graph: dict[str, list[str]],
    edge_weights: dict[tuple[str, str], float] | None = None,
    *,
    personalization: dict[str, float] | None = None,
    damping: float = 0.85,
    iterations: int = 50,
) -> dict[str, float]:
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    if personalization:
        raw = {u: max(0.0, float(personalization.get(u, 0.0))) for u in nodes}
        total_raw = sum(raw.values())
        base_dist = {u: (raw[u] / total_raw if total_raw else 1.0 / n) for u in nodes}
    else:
        base_dist = {u: 1.0 / n for u in nodes}

    weights = edge_weights or {}
    out_weight_sum: dict[str, float] = {}
    for src, outs in graph.items():
        out_weight_sum[src] = sum(max(0.0, float(weights.get((src, tgt), 1.0))) for tgt in outs)

    pr = dict(base_dist)
    for _ in range(iterations):
        new = {u: (1.0 - damping) * base_dist[u] for u in nodes}
        leak = 0.0
        for src, outs in graph.items():
            if not outs or out_weight_sum.get(src, 0.0) <= 0:
                leak += pr[src]
                continue
            for tgt in outs:
                weight = max(0.0, float(weights.get((src, tgt), 1.0)))
                if weight <= 0:
                    continue
                new[tgt] += damping * pr[src] * (weight / out_weight_sum[src])
        if leak:
            for u in nodes:
                new[u] += damping * leak * base_dist[u]
        pr = new

    total = sum(pr.values()) or 1.0
    return {u: v / total for u, v in pr.items()}


def betweenness_centrality(graph: dict[str, list[str]], *, max_sources: int = 450) -> dict[str, float]:
    nodes = list(graph.keys())
    if not nodes:
        return {}
    degree = {u: len(graph.get(u, [])) for u in nodes}
    for outs in graph.values():
        for v in outs:
            degree[v] = degree.get(v, 0) + 1
    sources = sorted(nodes, key=lambda u: degree.get(u, 0), reverse=True)[:max_sources]
    cb = {v: 0.0 for v in nodes}
    for s in sources:
        stack: list[str] = []
        pred: dict[str, list[str]] = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        dist = dict.fromkeys(nodes, -1)
        sigma[s] = 1.0
        dist[s] = 0
        queue = collections.deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.get(v, []):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    scale = 1.0 / max(1, len(sources))
    return {u: v * scale for u, v in cb.items()}


def edge_anchor_quality(anchors_by_edge: dict[tuple[str, str], list[str]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for edge, anchors in anchors_by_edge.items():
        if not anchors:
            out[edge] = 0.45
            continue
        scores = []
        for anchor in anchors:
            cleaned = re.sub(r"\s+", " ", str(anchor or "").strip())
            if not cleaned:
                scores.append(0.2)
            elif _is_generic_anchor(cleaned):
                scores.append(0.35)
            else:
                words = len(re.findall(r"[a-z0-9]+", cleaned.lower()))
                scores.append(min(1.0, 0.58 + min(5, words) * 0.075))
        repeat_boost = min(0.1, math.log1p(len(anchors)) * 0.035)
        out[edge] = round(max(0.05, min(1.0, sum(scores) / len(scores) + repeat_boost)), 4)
    return out


def edge_contextual_relevance(
    pages,
    embeddings: np.ndarray | None,
    graph: dict[str, list[str]],
) -> dict[tuple[str, str], float]:
    if embeddings is None or len(pages) == 0 or getattr(embeddings, "shape", (0,))[0] < len(pages):
        return {}
    page_idx = {p.url: i for i, p in enumerate(pages)}
    out: dict[tuple[str, str], float] = {}
    for src, targets in graph.items():
        i = page_idx.get(src)
        if i is None:
            continue
        for tgt in targets:
            j = page_idx.get(tgt)
            if j is None:
                continue
            sim = float(np.clip(embeddings[i] @ embeddings[j], -1.0, 1.0))
            out[(src, tgt)] = round(max(0.05, min(1.0, (sim + 1.0) / 2.0)), 4)
    return out


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
    canonical_edge_set: set[tuple[str, str]] = set()
    for src, outs in graph.items():
        for tgt in outs:
            edge_set.add((src, tgt))
            canonical_edge_set.add((_canonical_url(src), _canonical_url(tgt)))

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
            canonical_a = _canonical_url(url_a)
            canonical_b = _canonical_url(url_b)
            if canonical_a == canonical_b:
                continue
            if (url_a, url_b) in edge_set or (url_b, url_a) in edge_set:
                continue
            if (canonical_a, canonical_b) in canonical_edge_set or (canonical_b, canonical_a) in canonical_edge_set:
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
    internal_http_links, internal_https_links = _outgoing_links_by_scheme(pages_with_outlinks)

    in_deg: dict[str, int] = defaultdict(int)
    for outs in graph.values():
        for tgt in outs:
            in_deg[tgt] += 1
    out_deg = {u: len(o) for u, o in graph.items()}
    edge_count = sum(out_deg.values())

    pr = pagerank(graph)
    hub, auth = hits(graph)
    depth = click_depth(graph, home_url)
    anchor_quality = edge_anchor_quality(anchors)
    contextual_relevance = edge_contextual_relevance(pages, embeddings, graph)

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
        graph=graph,
        edge_anchor_texts=anchors,
        edge_anchor_count={edge: len(labels) for edge, labels in anchors.items()},
        in_degree=dict(in_deg),
        out_degree=out_deg,
        pagerank=pr,
        hub_score=hub,
        authority_score=auth,
        click_depth=depth,
        edge_anchor_quality=anchor_quality,
        edge_contextual_relevance=contextual_relevance,
        orphans=orphans,
        dead_ends=dead_ends,
        edge_count=edge_count,
        recommendations=recs,
        internal_http_links=internal_http_links,
        internal_https_links=internal_https_links,
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
        http_links = result.internal_http_links.get(p.url, [])
        https_links = result.internal_https_links.get(p.url, [])
        page_link_counts.append({
            "url": p.url,
            "title": p.title,
            "section": p.section,
            "pagerank": round(float(result.pagerank.get(p.url, 0.0)), 8),
            "authority_score": round(float(result.authority_score.get(p.url, 0.0)), 8),
            "hub_score": round(float(result.hub_score.get(p.url, 0.0)), 8),
            "in_degree": int(result.in_degree.get(p.url, 0)),
            "out_degree": int(result.out_degree.get(p.url, 0)),
            "internal_http_link_count": len(http_links),
            "internal_http_links": http_links[:25],
            "internal_https_link_count": len(https_links),
            "internal_https_links": https_links[:25],
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


def _canonical_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return str(url or "").rstrip("/")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{parsed.scheme.lower() or 'https'}://{netloc}{path}".rstrip("/") or str(url or "").rstrip("/")


def _directory(url: str) -> str:
    try:
        parts = [p for p in (urlparse(url).path or "/").split("/") if p]
    except Exception:
        parts = []
    return f"/{parts[0]}/" if parts else "/"


def _safe_int(value) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile_lookup(values: dict[str, float]) -> dict[str, float]:
    positive = sorted(float(v) for v in values.values() if float(v) > 0)
    if not positive:
        return {k: 0.0 for k in values}
    n = len(positive)
    out = {}
    for key, value in values.items():
        value = float(value)
        if value <= 0:
            out[key] = 0.0
        else:
            out[key] = round(bisect.bisect_right(positive, value) / n, 4)
    return out


def _page_type_lookup(page_types: dict | None) -> dict[str, str]:
    lookup = {}
    for row in (page_types or {}).get("per_page") or []:
        url = row.get("url") or ""
        if url:
            lookup[_canonical_url(url)] = row.get("page_type") or ""
    return lookup


def _search_lookup(pages, search_payload: dict | None) -> dict[str, dict]:
    by_url = {_canonical_url(p.url): p.url for p in pages}
    out: dict[str, dict] = defaultdict(lambda: {
        "traffic": 0,
        "keyword_traffic": 0,
        "keywords": 0,
        "top_keyword": "",
        "top_keyword_position": None,
        "cluster": "",
        "_kw_max_traffic": -1,
    })

    def match(row: dict) -> str | None:
        for key in ("matched_url", "url"):
            raw = row.get(key) or ""
            canonical = _canonical_url(raw)
            if canonical in by_url:
                return by_url[canonical]
        return None

    for row in (search_payload or {}).get("top_pages") or []:
        url = match(row)
        if not url:
            continue
        ctx = out[url]
        ctx["traffic"] = max(_safe_int(ctx.get("traffic")), _safe_int(row.get("traffic")))
        ctx["keywords"] = max(_safe_int(ctx.get("keywords")), _safe_int(row.get("keywords")))
        if row.get("top_keyword"):
            ctx["top_keyword"] = row.get("top_keyword")
        if row.get("top_keyword_position") is not None or row.get("position") is not None:
            ctx["top_keyword_position"] = _safe_int(row.get("top_keyword_position") or row.get("position"))
        if row.get("cluster_label") or row.get("cluster"):
            ctx["cluster"] = row.get("cluster_label") or row.get("cluster")

    for row in (search_payload or {}).get("organic_keywords") or []:
        url = match(row)
        if not url:
            continue
        ctx = out[url]
        kw_traffic = _safe_int(row.get("traffic"))
        ctx["keyword_traffic"] += kw_traffic
        ctx["keywords"] = _safe_int(ctx.get("keywords")) + 1
        if kw_traffic > _safe_int(ctx.get("_kw_max_traffic")):
            ctx["top_keyword"] = row.get("keyword") or ctx.get("top_keyword") or ""
            ctx["top_keyword_position"] = _safe_int(row.get("position"))
            ctx["_kw_max_traffic"] = kw_traffic
        if row.get("cluster_label") or row.get("cluster"):
            ctx["cluster"] = row.get("cluster_label") or row.get("cluster")

    for ctx in out.values():
        if not _safe_int(ctx.get("traffic")):
            ctx["traffic"] = _safe_int(ctx.get("keyword_traffic"))
        ctx.pop("_kw_max_traffic", None)
        ctx.pop("keyword_traffic", None)
    return out


def _raw_search_lookup(search_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"traffic": 0, "keywords": 0, "top_keyword": "", "top_keyword_position": None})
    for row in (search_payload or {}).get("top_pages") or []:
        raw = row.get("matched_url") or row.get("url") or ""
        if not raw:
            continue
        ctx = out[_canonical_url(raw)]
        ctx["traffic"] = max(_safe_int(ctx.get("traffic")), _safe_int(row.get("traffic")))
        ctx["keywords"] = max(_safe_int(ctx.get("keywords")), _safe_int(row.get("keywords")))
        if row.get("top_keyword"):
            ctx["top_keyword"] = row.get("top_keyword")
        if row.get("top_keyword_position") is not None or row.get("position") is not None:
            ctx["top_keyword_position"] = _safe_int(row.get("top_keyword_position") or row.get("position"))
    for row in (search_payload or {}).get("organic_keywords") or []:
        raw = row.get("matched_url") or row.get("url") or ""
        if not raw:
            continue
        ctx = out[_canonical_url(raw)]
        ctx["traffic"] += _safe_int(row.get("traffic"))
        ctx["keywords"] += 1
        if not ctx.get("top_keyword"):
            ctx["top_keyword"] = row.get("keyword") or ""
            ctx["top_keyword_position"] = _safe_int(row.get("position"))
    return out


def _edge_weights(result: LinkGraphResult) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for src, targets in result.graph.items():
        for tgt in targets:
            edge = (src, tgt)
            anchor = float(result.edge_anchor_quality.get(edge, 0.5))
            relevance = float(result.edge_contextual_relevance.get(edge, 0.5))
            repeated = math.log1p(max(1, result.edge_anchor_count.get(edge, 1))) * 0.04
            weights[edge] = round(max(0.05, 0.48 + anchor * 0.24 + relevance * 0.28 + repeated), 6)
    return weights


def _label_authority_gap(row: dict) -> tuple[str, str]:
    traffic_pct = _safe_float(row.get("traffic_percentile"))
    authority_pct = _safe_float(row.get("weighted_pagerank_percentile"))
    traffic = _safe_int(row.get("traffic"))
    in_degree = _safe_int(row.get("in_degree"))
    depth = row.get("click_depth")
    if traffic > 0 and in_degree == 0:
        return "ranked_orphan", "Add contextual links from the homepage, section hub, or a semantically close authority page."
    if traffic_pct >= 0.7 and authority_pct <= 0.45:
        return "high_traffic_low_authority", "Promote this search-demand page from relevant hubs with descriptive anchors."
    if traffic_pct >= 0.6 and depth is not None and _safe_int(depth) >= 4:
        return "buried_demand", "Move the page closer to the homepage or a section hub; keep the existing ranking intent intact."
    if authority_pct >= 0.75 and traffic_pct <= 0.25:
        return "high_authority_low_value", "Reuse this authority as a source link hub, refresh its search target, or consolidate if it has no strategic role."
    if traffic_pct >= 0.55 and authority_pct >= 0.55:
        return "aligned", "Maintain the internal-link pattern and monitor ranking movement."
    return "neutral", "No urgent PageRank/search-demand mismatch."


def traffic_weighted_pagerank_payload(
    result: LinkGraphResult,
    pages,
    embeddings: np.ndarray | None = None,
    *,
    search_payload: dict | None = None,
    page_types: dict | None = None,
    indexability: dict | None = None,
) -> dict:
    if not result.graph:
        return {"summary": {"status": "no_graph", "total_pages": 0}, "pages": [], "mismatches": {}, "clusters": [], "directories": [], "page_types": []}

    search = _search_lookup(pages, search_payload)
    page_types_by_url = _page_type_lookup(page_types)
    weights = _edge_weights(result)
    weighted_pr = weighted_pagerank(result.graph, weights)
    personalization = {
        url: 1.0 + math.log1p(_safe_int((search.get(url) or {}).get("traffic"))) * 3.0 + math.sqrt(_safe_int((search.get(url) or {}).get("keywords")))
        for url in result.graph
    }
    traffic_weighted_pr = weighted_pagerank(result.graph, weights, personalization=personalization)

    out_weight_sum: dict[str, float] = {}
    for src, targets in result.graph.items():
        out_weight_sum[src] = sum(max(0.05, weights.get((src, tgt), 1.0)) for tgt in targets)
    inbound_flow: dict[str, float] = defaultdict(float)
    edge_rows: list[dict] = []
    for src, targets in result.graph.items():
        src_ctx = search.get(src) or {}
        src_traffic = _safe_int(src_ctx.get("traffic"))
        src_strength = float(traffic_weighted_pr.get(src, 0.0)) * (1.0 + math.log1p(src_traffic)) * (1.0 + float(result.hub_score.get(src, 0.0)))
        for tgt in targets:
            edge = (src, tgt)
            if out_weight_sum.get(src, 0.0) <= 0:
                continue
            share = max(0.05, weights.get(edge, 1.0)) / out_weight_sum[src]
            flow = src_strength * share
            inbound_flow[tgt] += flow
            tgt_ctx = search.get(tgt) or {}
            edge_rows.append({
                "source": src,
                "target": tgt,
                "weight": round(float(weights.get(edge, 1.0)), 6),
                "anchor_quality": round(float(result.edge_anchor_quality.get(edge, 0.5)), 4),
                "contextual_relevance": round(float(result.edge_contextual_relevance.get(edge, 0.5)), 4),
                "source_traffic": src_traffic,
                "target_traffic": _safe_int(tgt_ctx.get("traffic")),
                "source_pagerank": round(float(result.pagerank.get(src, 0.0)), 8),
                "source_traffic_weighted_pagerank": round(float(traffic_weighted_pr.get(src, 0.0)), 8),
                "flow_score": round(float(flow), 10),
            })

    traffic_values = {p.url: float(_safe_int((search.get(p.url) or {}).get("traffic"))) for p in pages}
    pr_pct = _percentile_lookup({p.url: float(result.pagerank.get(p.url, 0.0)) for p in pages})
    weighted_pct = _percentile_lookup({p.url: float(weighted_pr.get(p.url, 0.0)) for p in pages})
    traffic_weighted_pct = _percentile_lookup({p.url: float(traffic_weighted_pr.get(p.url, 0.0)) for p in pages})
    traffic_pct = _percentile_lookup(traffic_values)
    flow_pct = _percentile_lookup({p.url: float(inbound_flow.get(p.url, 0.0)) for p in pages})

    rows: list[dict] = []
    for page in pages:
        ctx = search.get(page.url) or {}
        cluster = ctx.get("cluster") or page.section or _directory(page.url)
        row = {
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "directory": _directory(page.url),
            "cluster": cluster,
            "page_type": page_types_by_url.get(_canonical_url(page.url), ""),
            "indexability": "analyzed",
            "traffic": _safe_int(ctx.get("traffic")),
            "keywords": _safe_int(ctx.get("keywords")),
            "top_keyword": ctx.get("top_keyword") or "",
            "top_keyword_position": ctx.get("top_keyword_position"),
            "pagerank": round(float(result.pagerank.get(page.url, 0.0)), 8),
            "weighted_pagerank": round(float(weighted_pr.get(page.url, 0.0)), 8),
            "traffic_weighted_pagerank": round(float(traffic_weighted_pr.get(page.url, 0.0)), 8),
            "link_flow_score": round(float(inbound_flow.get(page.url, 0.0)), 10),
            "in_degree": int(result.in_degree.get(page.url, 0)),
            "out_degree": int(result.out_degree.get(page.url, 0)),
            "click_depth": int(result.click_depth.get(page.url, -1)) if page.url in result.click_depth else None,
            "traffic_percentile": traffic_pct.get(page.url, 0.0),
            "pagerank_percentile": pr_pct.get(page.url, 0.0),
            "weighted_pagerank_percentile": weighted_pct.get(page.url, 0.0),
            "traffic_weighted_pagerank_percentile": traffic_weighted_pct.get(page.url, 0.0),
            "link_flow_percentile": flow_pct.get(page.url, 0.0),
        }
        row["authority_traffic_gap"] = round(float(row["traffic_percentile"]) - float(row["weighted_pagerank_percentile"]), 4)
        row["keyword_opportunity"] = round(float(row["traffic"]) * max(0.0, float(row["authority_traffic_gap"])), 2)
        label, action = _label_authority_gap(row)
        row["mismatch_label"] = label
        row["recommended_action"] = action
        rows.append(row)

    rows.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("authority_traffic_gap"))), reverse=True)
    edge_rows.sort(key=lambda r: _safe_float(r.get("flow_score")), reverse=True)

    def aggregate(key: str) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key) or "unknown")].append(row)
        out = []
        for name, bucket in buckets.items():
            traffic = sum(_safe_int(r.get("traffic")) for r in bucket)
            demand = [r for r in bucket if _safe_int(r.get("traffic")) > 0]
            out.append({
                key: name,
                "label": name,
                "pages": len(bucket),
                "traffic": traffic,
                "avg_pagerank_percentile": round(sum(_safe_float(r.get("pagerank_percentile")) for r in bucket) / max(1, len(bucket)), 4),
                "avg_weighted_pagerank_percentile": round(sum(_safe_float(r.get("weighted_pagerank_percentile")) for r in bucket) / max(1, len(bucket)), 4),
                "avg_authority_traffic_gap": round(sum(_safe_float(r.get("authority_traffic_gap")) for r in demand) / max(1, len(demand)), 4),
                "underserved_pages": sum(1 for r in bucket if r.get("mismatch_label") in {"high_traffic_low_authority", "ranked_orphan", "buried_demand"}),
                "authority_without_demand": sum(1 for r in bucket if r.get("mismatch_label") == "high_authority_low_value"),
            })
        out.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("underserved_pages"))), reverse=True)
        return out

    raw_search = _raw_search_lookup(search_payload)
    skipped: list[dict] = []
    seen_skipped: set[str] = set()
    for skipped_row in ((indexability or {}).get("skipped") or []) + ((indexability or {}).get("noindex_pages") or []):
        url = skipped_row.get("url") or ""
        if not url:
            continue
        canonical = _canonical_url(url)
        if canonical in seen_skipped:
            continue
        seen_skipped.add(canonical)
        ctx = raw_search.get(canonical) or {}
        if _safe_int(ctx.get("traffic")) <= 0 and skipped_row.get("reason") != "noindex":
            continue
        skipped.append({
            "url": url,
            "title": skipped_row.get("title") or url,
            "status": skipped_row.get("status") or "skipped",
            "reason": skipped_row.get("reason") or "",
            "traffic": _safe_int(ctx.get("traffic")),
            "keywords": _safe_int(ctx.get("keywords")),
            "top_keyword": ctx.get("top_keyword") or "",
            "top_keyword_position": ctx.get("top_keyword_position"),
            "recommended_action": "Resolve indexability before sending more internal authority to this URL.",
        })
    skipped.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)

    demand_rows = [r for r in rows if _safe_int(r.get("traffic")) > 0]
    total_traffic = sum(_safe_int(r.get("traffic")) for r in rows)
    weighted_abs_gap = sum(abs(_safe_float(r.get("authority_traffic_gap"))) * max(1, _safe_int(r.get("traffic"))) for r in demand_rows)
    weight_denom = sum(max(1, _safe_int(r.get("traffic"))) for r in demand_rows) or 1
    alignment = max(0.0, min(1.0, 1.0 - weighted_abs_gap / weight_denom))
    orphan_traffic = sum(_safe_int(r.get("traffic")) for r in rows if _safe_int(r.get("traffic")) > 0 and _safe_int(r.get("in_degree")) == 0)

    high_traffic_low_authority = [r for r in rows if r.get("mismatch_label") in {"high_traffic_low_authority", "ranked_orphan", "buried_demand"}]
    high_authority_low_value = [r for r in rows if r.get("mismatch_label") == "high_authority_low_value"]

    return {
        "summary": {
            "status": "ok",
            "model": "traffic_weighted_pagerank_v1",
            "total_pages": len(rows),
            "traffic_pages": len(demand_rows),
            "total_traffic": total_traffic,
            "authority_traffic_alignment": round(alignment, 4),
            "high_traffic_low_authority_pages": len(high_traffic_low_authority),
            "high_authority_low_value_pages": len(high_authority_low_value),
            "orphan_traffic": orphan_traffic,
            "orphan_traffic_share": round(orphan_traffic / max(1, total_traffic), 4),
            "non_indexable_search_pages": len(skipped),
        },
        "pages": rows,
        "mismatches": {
            "high_traffic_low_authority": high_traffic_low_authority[:200],
            "high_authority_low_value": high_authority_low_value[:200],
            "non_indexable_search_pages": skipped[:200],
        },
        "clusters": aggregate("cluster")[:120],
        "directories": aggregate("directory")[:120],
        "page_types": aggregate("page_type")[:120],
        "edges": edge_rows[:800],
        "interpretation": {
            "authority_traffic_gap": "Positive values mean traffic demand is stronger than internal authority support; negative values mean authority is concentrated on lower-demand pages.",
            "weighted_pagerank": "Weighted PageRank favors descriptive anchors and semantically related source-target links. Traffic-weighted PageRank also personalizes the random walk toward pages with organic demand.",
        },
    }


_NAV_LINK_RE = re.compile(r"\b(home|menu|login|sign in|account|cart|privacy|terms|contact|pricing|demo|about)\b", re.I)
_NAV_TARGET_RE = re.compile(r"^/(login|sign-in|signin|account|cart|checkout|privacy|terms|legal|contact|about|pricing|demo)/?$", re.I)


def hub_bottleneck_payload(
    result: LinkGraphResult,
    pages,
    *,
    traffic_authority: dict | None = None,
    page_types: dict | None = None,
) -> dict:
    if not result.graph:
        return {"summary": {"status": "no_graph", "total_pages": 0}, "pages": [], "cluster_edges": []}

    page_by_url = {p.url: p for p in pages}
    authority_by_url = {
        row.get("url"): row
        for row in (traffic_authority or {}).get("pages", [])
        if row.get("url")
    }
    type_by_url = _page_type_lookup(page_types)
    reverse: dict[str, list[str]] = defaultdict(list)
    for src, targets in result.graph.items():
        for tgt in targets:
            reverse[tgt].append(src)
    between = betweenness_centrality(result.graph)
    between_pct = _percentile_lookup(between)
    pagerank_pct = _percentile_lookup({url: float(result.pagerank.get(url, 0.0)) for url in result.graph})
    traffic_pct = _percentile_lookup({url: float((authority_by_url.get(url) or {}).get("traffic", 0) or 0) for url in result.graph})
    bridge_values: dict[str, int] = {}
    cluster_edges: Counter[tuple[str, str]] = Counter()

    def cluster_for(url: str) -> str:
        page = page_by_url.get(url)
        return str((authority_by_url.get(url) or {}).get("cluster") or (page.section if page else "") or _directory(url))

    rows = []
    for url in result.graph:
        incoming = reverse.get(url, [])
        outgoing = result.graph.get(url, [])
        source_clusters = sorted({cluster_for(src) for src in incoming if cluster_for(src)})
        target_clusters = sorted({cluster_for(tgt) for tgt in outgoing if cluster_for(tgt)})
        pairs = {(src_c, tgt_c) for src_c in source_clusters for tgt_c in target_clusters if src_c != tgt_c}
        for pair in pairs:
            cluster_edges[pair] += 1
        bridge_values[url] = len(pairs)
    bridge_pct = _percentile_lookup({url: float(v) for url, v in bridge_values.items()})

    for url in result.graph:
        page = page_by_url.get(url)
        incoming = reverse.get(url, [])
        outgoing = result.graph.get(url, [])
        authority = authority_by_url.get(url) or {}
        source_clusters = sorted({cluster_for(src) for src in incoming if cluster_for(src)})
        target_clusters = sorted({cluster_for(tgt) for tgt in outgoing if cluster_for(tgt)})
        affected_clusters = sorted(set(source_clusters + target_clusters))
        risk = (
            between_pct.get(url, 0.0) * 34.0
            + bridge_pct.get(url, 0.0) * 26.0
            + pagerank_pct.get(url, 0.0) * 18.0
            + traffic_pct.get(url, 0.0) * 12.0
            + (10.0 if (len(incoming) <= 1 or len(outgoing) <= 1) and (incoming or outgoing) else 0.0)
        )
        if len(incoming) == 0:
            role = "orphan_risk"
        elif len(outgoing) == 0:
            role = "dead_end_risk"
        elif between_pct.get(url, 0.0) >= 0.8 and bridge_values.get(url, 0) > 0:
            role = "bottleneck"
        elif bridge_values.get(url, 0) >= 2:
            role = "cluster_bridge"
        elif pagerank_pct.get(url, 0.0) >= 0.8 and len(outgoing) >= 2:
            role = "authority_hub"
        else:
            role = "support_page"
        row = {
            "url": url,
            "title": page.title if page else url,
            "section": page.section if page else "",
            "directory": _directory(url),
            "page_type": type_by_url.get(_canonical_url(url), ""),
            "cluster": cluster_for(url),
            "role": role,
            "traffic": _safe_int(authority.get("traffic")),
            "pagerank": round(float(result.pagerank.get(url, 0.0)), 8),
            "betweenness": round(float(between.get(url, 0.0)), 8),
            "betweenness_percentile": between_pct.get(url, 0.0),
            "pagerank_percentile": pagerank_pct.get(url, 0.0),
            "traffic_percentile": traffic_pct.get(url, 0.0),
            "cluster_bridge_count": bridge_values.get(url, 0),
            "source_clusters": source_clusters[:20],
            "target_clusters": target_clusters[:20],
            "affected_clusters": affected_clusters[:30],
            "incoming_pages": [{"url": src, "title": getattr(page_by_url.get(src), "title", src), "cluster": cluster_for(src)} for src in incoming[:20]],
            "outgoing_pages": [{"url": tgt, "title": getattr(page_by_url.get(tgt), "title", tgt), "cluster": cluster_for(tgt)} for tgt in outgoing[:20]],
            "in_degree": len(incoming),
            "out_degree": len(outgoing),
            "resilience_risk": round(max(0.0, min(100.0, risk)), 2),
        }
        row["recommended_action"] = (
            "Add redundant paths between affected clusters and protect this page during navigation/template edits."
            if role == "bottleneck"
            else "Create alternate cross-links so this page is not the only bridge between clusters."
            if role == "cluster_bridge"
            else "Use this hub to distribute links to under-supported demand pages."
            if role == "authority_hub"
            else "Add relevant outbound links so visitors and crawlers can continue to related pages."
            if role == "dead_end_risk"
            else "Link to this page from a relevant hub or remove it from the indexable set."
            if role == "orphan_risk"
            else "No urgent architecture change."
        )
        rows.append(row)

    rows.sort(key=lambda r: (_safe_float(r.get("resilience_risk")), _safe_float(r.get("betweenness"))), reverse=True)
    role_counts = collections.Counter(row["role"] for row in rows)
    bottlenecks = [r for r in rows if r["role"] == "bottleneck"]
    bridges = [r for r in rows if r["role"] == "cluster_bridge"]
    cluster_edge_rows = [
        {"source_cluster": src, "target_cluster": tgt, "bridge_pages": count}
        for (src, tgt), count in cluster_edges.most_common(300)
    ]
    resilience = max(0.0, 1.0 - (len(bottlenecks) + len(bridges) * 0.5 + role_counts.get("dead_end_risk", 0) * 0.25) / max(1, len(rows)))
    return {
        "summary": {
            "status": "ok",
            "model": "hub_bottleneck_v1",
            "total_pages": len(rows),
            "architecture_resilience": round(resilience, 4),
            "bottleneck_pages": len(bottlenecks),
            "bridge_pages": len(bridges),
            "authority_hubs": role_counts.get("authority_hub", 0),
            "dead_end_risks": role_counts.get("dead_end_risk", 0),
            "orphan_risks": role_counts.get("orphan_risk", 0),
        },
        "pages": rows,
        "bottlenecks": bottlenecks[:200],
        "bridges": bridges[:200],
        "authority_hubs": [r for r in rows if r["role"] == "authority_hub"][:200],
        "risks": [r for r in rows if r["role"] in {"bottleneck", "cluster_bridge", "dead_end_risk", "orphan_risk"}][:300],
        "cluster_edges": cluster_edge_rows,
        "interpretation": {
            "betweenness": "How often a page sits on shortest internal-link paths in an approximate directed Brandes pass.",
            "cluster_bridge_count": "Number of source-cluster to target-cluster relationships this page connects.",
        },
    }


def _ranking_opportunity(position: int, volume: int) -> float:
    if position <= 0:
        return 0.0
    weight = math.log1p(max(0, volume))
    if position <= 3:
        return weight * 0.25
    if position <= 10:
        return weight * 1.0
    if position <= 20:
        return weight * 0.65
    if position <= 50:
        return weight * 0.25
    return weight * 0.08


def _high_demand_search_lookup(pages, search_payload: dict | None) -> dict[str, dict]:
    by_url = {_canonical_url(p.url): p.url for p in pages}
    out: dict[str, dict] = defaultdict(lambda: {
        "traffic": 0,
        "keyword_traffic": 0,
        "keywords": 0,
        "volume": 0,
        "ranking_opportunity": 0.0,
        "position_weight": 0.0,
        "weighted_position_sum": 0.0,
        "top_keyword": "",
        "top_keyword_position": None,
        "top_keyword_volume": 0,
        "cluster": "",
        "top_keywords": [],
    })

    def match(row: dict) -> str | None:
        for key in ("matched_url", "url"):
            canonical = _canonical_url(row.get(key) or "")
            if canonical in by_url:
                return by_url[canonical]
        return None

    def add_keyword(ctx: dict, keyword: str, traffic: int, volume: int, position: int) -> None:
        if keyword and len(ctx["top_keywords"]) < 18:
            ctx["top_keywords"].append({
                "keyword": keyword,
                "traffic": traffic,
                "volume": volume,
                "position": position or None,
            })
        ctx["volume"] += volume
        ctx["ranking_opportunity"] += _ranking_opportunity(position, volume)
        weight = max(1.0, float(traffic), math.sqrt(max(0, volume)))
        if position > 0:
            ctx["position_weight"] += weight
            ctx["weighted_position_sum"] += position * weight
        if traffic > _safe_int(ctx.get("_top_keyword_traffic", -1)):
            ctx["top_keyword"] = keyword or ctx.get("top_keyword") or ""
            ctx["top_keyword_position"] = position or ctx.get("top_keyword_position")
            ctx["top_keyword_volume"] = volume or _safe_int(ctx.get("top_keyword_volume"))
            ctx["_top_keyword_traffic"] = traffic

    for row in (search_payload or {}).get("top_pages") or []:
        url = match(row)
        if not url:
            continue
        ctx = out[url]
        traffic = _safe_int(row.get("traffic"))
        keywords = _safe_int(row.get("keywords"))
        volume = _safe_int(row.get("top_keyword_volume") or row.get("volume"))
        position = _safe_int(row.get("top_keyword_position") or row.get("position"))
        ctx["traffic"] = max(_safe_int(ctx.get("traffic")), traffic)
        ctx["keywords"] = max(_safe_int(ctx.get("keywords")), keywords)
        if row.get("cluster_label") or row.get("cluster"):
            ctx["cluster"] = row.get("cluster_label") or row.get("cluster")
        if row.get("top_keyword"):
            add_keyword(ctx, str(row.get("top_keyword") or ""), traffic, volume, position)

    for row in (search_payload or {}).get("organic_keywords") or []:
        url = match(row)
        if not url:
            continue
        ctx = out[url]
        traffic = _safe_int(row.get("traffic"))
        volume = _safe_int(row.get("volume") or row.get("search_volume"))
        position = _safe_int(row.get("position"))
        ctx["keyword_traffic"] += traffic
        ctx["keywords"] = _safe_int(ctx.get("keywords")) + 1
        if row.get("cluster_label") or row.get("cluster"):
            ctx["cluster"] = row.get("cluster_label") or row.get("cluster")
        add_keyword(ctx, str(row.get("keyword") or ""), traffic, volume, position)

    for ctx in out.values():
        if not _safe_int(ctx.get("traffic")):
            ctx["traffic"] = _safe_int(ctx.get("keyword_traffic"))
        weight = _safe_float(ctx.get("position_weight"))
        ctx["avg_position"] = round(_safe_float(ctx.get("weighted_position_sum")) / weight, 2) if weight else None
        ctx["top_keywords"] = sorted(
            ctx.get("top_keywords") or [],
            key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("volume"))),
            reverse=True,
        )[:12]
        ctx.pop("_top_keyword_traffic", None)
        ctx.pop("keyword_traffic", None)
        ctx.pop("position_weight", None)
        ctx.pop("weighted_position_sum", None)
    return out


def _score_scale(values: dict[str, float]) -> dict[str, float]:
    vmax = max((float(v) for v in values.values()), default=0.0)
    if vmax <= 0:
        return {k: 0.0 for k in values}
    return {k: max(0.0, float(v)) / vmax for k, v in values.items()}


def _demand_support_label(demand_score: float, support_score: float) -> str:
    if demand_score >= 65 and support_score < 45:
        return "high_demand_low_support"
    if demand_score >= 55 and support_score < 60:
        return "demand_support_gap"
    if demand_score >= 55:
        return "supported_demand"
    if support_score >= 70 and demand_score < 35:
        return "over_supported_low_demand"
    if demand_score < 35 and support_score < 35:
        return "low_demand_low_support"
    return "balanced"


def high_demand_low_link_payload(
    result: LinkGraphResult,
    pages,
    *,
    search_payload: dict | None = None,
    traffic_authority: dict | None = None,
    link_addition: dict | None = None,
    page_types: dict | None = None,
) -> dict:
    if not result.graph:
        return {"summary": {"status": "no_graph", "total_pages": 0}, "pages": [], "opportunities": []}

    page_by_url = {p.url: p for p in pages}
    type_by_url = _page_type_lookup(page_types)
    search = _high_demand_search_lookup(pages, search_payload)
    authority_by_url = {
        row.get("url"): row
        for row in (traffic_authority or {}).get("pages", [])
        if row.get("url")
    }
    reverse: dict[str, list[str]] = defaultdict(list)
    for src, targets in result.graph.items():
        for tgt in targets:
            reverse[tgt].append(src)

    def cluster_for(url: str) -> str:
        page = page_by_url.get(url)
        auth = authority_by_url.get(url) or {}
        ctx = search.get(url) or {}
        return str(auth.get("cluster") or ctx.get("cluster") or (page.section if page else "") or _directory(url))

    inbound_anchor_quality: dict[str, list[float]] = defaultdict(list)
    inbound_context_quality: dict[str, list[float]] = defaultdict(list)
    contextual_counts: Counter[str] = Counter()
    for (src, tgt), anchor_quality in result.edge_anchor_quality.items():
        inbound_anchor_quality[tgt].append(_safe_float(anchor_quality))
        relevance = _safe_float(result.edge_contextual_relevance.get((src, tgt), 0.5))
        inbound_context_quality[tgt].append(relevance)
        if anchor_quality >= 0.58 and relevance >= 0.62:
            contextual_counts[tgt] += 1

    urls = list(result.graph.keys())
    traffic_n = _score_scale({url: math.log1p(_safe_int((search.get(url) or {}).get("traffic"))) for url in urls})
    keyword_n = _score_scale({url: math.sqrt(_safe_int((search.get(url) or {}).get("keywords"))) for url in urls})
    volume_n = _score_scale({url: math.log1p(_safe_int((search.get(url) or {}).get("volume"))) for url in urls})
    ranking_n = _score_scale({url: _safe_float((search.get(url) or {}).get("ranking_opportunity")) for url in urls})

    base_rows: list[dict] = []
    for url in urls:
        page = page_by_url.get(url)
        ctx = search.get(url) or {}
        auth = authority_by_url.get(url) or {}
        in_degree = _safe_int(result.in_degree.get(url))
        avg_anchor = sum(inbound_anchor_quality.get(url, [])) / max(1, len(inbound_anchor_quality.get(url, [])))
        avg_context = sum(inbound_context_quality.get(url, [])) / max(1, len(inbound_context_quality.get(url, [])))
        contextual_support = avg_context * min(1.0, contextual_counts.get(url, 0) / 4.0)
        inlink_support = min(1.0, math.log1p(in_degree) / math.log1p(12))
        demand_score = (
            traffic_n.get(url, 0.0) * 42.0
            + keyword_n.get(url, 0.0) * 18.0
            + volume_n.get(url, 0.0) * 22.0
            + ranking_n.get(url, 0.0) * 18.0
        )
        support_score = (
            inlink_support * 26.0
            + _safe_float(auth.get("weighted_pagerank_percentile")) * 28.0
            + _safe_float(auth.get("link_flow_percentile")) * 16.0
            + avg_anchor * min(1.0, in_degree / 4.0) * 15.0
            + contextual_support * 15.0
        )
        gap = max(0.0, demand_score - support_score)
        label = _demand_support_label(demand_score, support_score)
        row = {
            "url": url,
            "title": page.title if page else url,
            "section": page.section if page else "",
            "directory": _directory(url),
            "cluster": cluster_for(url),
            "page_type": type_by_url.get(_canonical_url(url), auth.get("page_type", "")),
            "traffic": _safe_int(ctx.get("traffic") if ctx else auth.get("traffic")),
            "keywords": _safe_int(ctx.get("keywords") if ctx else auth.get("keywords")),
            "volume": _safe_int(ctx.get("volume")),
            "top_keyword": ctx.get("top_keyword") or auth.get("top_keyword") or "",
            "top_keyword_position": ctx.get("top_keyword_position") or auth.get("top_keyword_position"),
            "top_keyword_volume": _safe_int(ctx.get("top_keyword_volume")),
            "avg_position": ctx.get("avg_position"),
            "ranking_opportunity": round(_safe_float(ctx.get("ranking_opportunity")), 4),
            "top_keywords": ctx.get("top_keywords") or [],
            "demand_score": round(demand_score, 2),
            "support_score": round(max(0.0, min(100.0, support_score)), 2),
            "demand_support_gap": round(gap, 2),
            "classification": label,
            "priority": "high" if label == "high_demand_low_support" else "medium" if label == "demand_support_gap" else "low",
            "in_degree": in_degree,
            "out_degree": _safe_int(result.out_degree.get(url)),
            "click_depth": int(result.click_depth.get(url, -1)) if url in result.click_depth else None,
            "pagerank": round(float(result.pagerank.get(url, 0.0)), 8),
            "weighted_pagerank_percentile": _safe_float(auth.get("weighted_pagerank_percentile")),
            "traffic_percentile": _safe_float(auth.get("traffic_percentile")),
            "link_flow_percentile": _safe_float(auth.get("link_flow_percentile")),
            "avg_inbound_anchor_quality": round(avg_anchor, 4),
            "avg_inbound_contextual_relevance": round(avg_context, 4),
            "contextual_inlinks": contextual_counts.get(url, 0),
            "incoming_clusters": sorted({cluster_for(src) for src in reverse.get(url, []) if cluster_for(src)}),
        }
        row["opportunity_score"] = round(gap * (1.0 + math.log1p(row["traffic"]) / 6.0), 2)
        base_rows.append(row)

    row_by_url = {row["url"]: row for row in base_rows}
    source_candidates_by_target: dict[str, list[dict]] = defaultdict(list)
    for rec in (link_addition or {}).get("recommendations") or []:
        src = rec.get("source_url") or ""
        tgt = rec.get("target_url") or ""
        if src not in row_by_url or tgt not in row_by_url:
            continue
        src_row = row_by_url[src]
        source_candidates_by_target[tgt].append({
            "source_url": src,
            "source_title": rec.get("source_title") or src_row.get("title") or src,
            "source_cluster": src_row.get("cluster") or cluster_for(src),
            "source_directory": src_row.get("directory") or _directory(src),
            "source_page_type": src_row.get("page_type") or "",
            "source_support_score": src_row.get("support_score", 0.0),
            "paragraph_index": rec.get("paragraph_index"),
            "paragraph_excerpt": rec.get("paragraph_excerpt") or "",
            "suggested_anchor": rec.get("suggested_anchor") or row_by_url[tgt].get("top_keyword") or row_by_url[tgt].get("title") or "",
            "expected_benefit_score": _safe_float(rec.get("expected_benefit_score")),
            "priority": rec.get("priority") or "",
        })

    support_sources = sorted(base_rows, key=lambda r: (_safe_float(r.get("support_score")), _safe_float(r.get("pagerank"))), reverse=True)
    for row in base_rows:
        incoming_clusters = set(row.get("incoming_clusters") or [])
        candidates = sorted(
            source_candidates_by_target.get(row["url"], []),
            key=lambda r: _safe_float(r.get("expected_benefit_score")),
            reverse=True,
        )
        if not candidates and row["classification"] in {"high_demand_low_support", "demand_support_gap"}:
            existing_sources = set(reverse.get(row["url"], []))
            for source in support_sources:
                src = source["url"]
                if src == row["url"] or src in existing_sources or row["url"] in result.graph.get(src, []):
                    continue
                if source.get("cluster") != row.get("cluster") and len(candidates) >= 2:
                    continue
                candidates.append({
                    "source_url": src,
                    "source_title": source.get("title") or src,
                    "source_cluster": source.get("cluster") or cluster_for(src),
                    "source_directory": source.get("directory") or _directory(src),
                    "source_page_type": source.get("page_type") or "",
                    "source_support_score": source.get("support_score", 0.0),
                    "paragraph_index": None,
                    "paragraph_excerpt": "",
                    "suggested_anchor": row.get("top_keyword") or row.get("title") or "",
                    "expected_benefit_score": round((row.get("demand_support_gap", 0.0) or 0.0) * 0.75 + _safe_float(source.get("support_score")) * 0.25, 2),
                    "priority": row.get("priority") or "",
                })
                if len(candidates) >= 5:
                    break
        candidates = candidates[:6]
        missing_clusters = []
        seen_clusters = set()
        for candidate in candidates:
            cluster = candidate.get("source_cluster") or ""
            if not cluster or cluster in incoming_clusters or cluster in seen_clusters:
                continue
            seen_clusters.add(cluster)
            missing_clusters.append({
                "cluster": cluster,
                "candidate_sources": sum(1 for c in candidates if c.get("source_cluster") == cluster),
            })
        anchors = []
        for candidate in candidates:
            anchor = candidate.get("suggested_anchor") or ""
            if anchor and anchor not in anchors:
                anchors.append(anchor)
        if row.get("top_keyword") and row["top_keyword"] not in anchors:
            anchors.append(row["top_keyword"])
        row["source_candidates"] = candidates
        row["missing_source_clusters"] = missing_clusters[:8]
        row["suggested_anchors"] = anchors[:8]
        row["recommended_action"] = (
            "Add contextual links from the suggested source pages and cover the missing source clusters first."
            if row["classification"] == "high_demand_low_support"
            else "Add one or two relevant contextual links before expanding more content for this intent."
            if row["classification"] == "demand_support_gap"
            else "Maintain the current support pattern while monitoring ranking movement."
            if row["classification"] == "supported_demand"
            else "Reuse this page as a source hub or consolidate if it has no strategic demand."
            if row["classification"] == "over_supported_low_demand"
            else "No urgent internal-link action."
        )

    base_rows.sort(key=lambda r: (_safe_float(r.get("opportunity_score")), _safe_int(r.get("traffic"))), reverse=True)

    def aggregate(key: str) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in base_rows:
            buckets[str(row.get(key) or "unknown")].append(row)
        out = []
        for name, bucket in buckets.items():
            demand_rows = [r for r in bucket if _safe_int(r.get("traffic")) > 0 or _safe_int(r.get("keywords")) > 0]
            traffic = sum(_safe_int(r.get("traffic")) for r in bucket)
            opp = [r for r in bucket if r.get("classification") in {"high_demand_low_support", "demand_support_gap"}]
            out.append({
                key: name,
                "label": name,
                "pages": len(bucket),
                "classified_top_pages": len(demand_rows),
                "traffic": traffic,
                "opportunities": len(opp),
                "opportunity_traffic": sum(_safe_int(r.get("traffic")) for r in opp),
                "avg_demand_score": round(sum(_safe_float(r.get("demand_score")) for r in demand_rows) / max(1, len(demand_rows)), 2),
                "avg_support_score": round(sum(_safe_float(r.get("support_score")) for r in demand_rows) / max(1, len(demand_rows)), 2),
                "avg_demand_support_gap": round(sum(_safe_float(r.get("demand_support_gap")) for r in demand_rows) / max(1, len(demand_rows)), 2),
            })
        out.sort(key=lambda r: (_safe_int(r.get("opportunity_traffic")), _safe_int(r.get("opportunities")), _safe_float(r.get("avg_demand_support_gap"))), reverse=True)
        return out

    top_rows = [r for r in base_rows if _safe_int(r.get("traffic")) > 0 or _safe_int(r.get("keywords")) > 0 or _safe_int(r.get("volume")) > 0]
    opportunities = [r for r in base_rows if r.get("classification") in {"high_demand_low_support", "demand_support_gap"}]
    high_opps = [r for r in base_rows if r.get("classification") == "high_demand_low_support"]
    traffic_weight = sum(max(1, _safe_int(r.get("traffic"))) for r in top_rows) or 1
    weighted_gap = sum(_safe_float(r.get("demand_support_gap")) * max(1, _safe_int(r.get("traffic"))) for r in top_rows)
    alignment = max(0.0, min(1.0, 1.0 - weighted_gap / (100.0 * traffic_weight)))
    return {
        "summary": {
            "status": "ok",
            "model": "high_demand_low_link_v1",
            "total_pages": len(base_rows),
            "classified_top_pages": len(top_rows),
            "demand_support_alignment": round(alignment, 4),
            "high_demand_low_support_pages": len(high_opps),
            "opportunity_pages": len(opportunities),
            "high_demand_low_support_traffic": sum(_safe_int(r.get("traffic")) for r in high_opps),
            "opportunity_traffic": sum(_safe_int(r.get("traffic")) for r in opportunities),
            "source_candidates": sum(len(r.get("source_candidates") or []) for r in opportunities),
        },
        "pages": base_rows,
        "opportunities": opportunities[:250],
        "high_priority": high_opps[:200],
        "directories": aggregate("directory")[:120],
        "clusters": aggregate("cluster")[:120],
        "page_types": aggregate("page_type")[:120],
        "interpretation": {
            "demand_score": "Search demand score from organic traffic, keyword count, keyword volume, and near-top ranking opportunity.",
            "support_score": "Internal support score from inbound links, weighted PageRank percentile, inbound link flow, anchor quality, and contextual relevance.",
        },
    }


def _edge_placement(src: str, tgt: str, anchors: list[str], anchor_quality: float, relevance: float, target_in_degree: int) -> str:
    try:
        target_path = urlparse(tgt).path or "/"
    except Exception:
        target_path = "/"
    anchor_blob = " ".join(anchors or [])
    generic_or_nav = any(_is_generic_anchor(a) for a in anchors or []) or bool(_NAV_LINK_RE.search(anchor_blob))
    if bool(_NAV_TARGET_RE.match(target_path)) or (generic_or_nav and target_in_degree >= 5 and relevance < 0.72):
        return "template_navigation"
    if anchor_quality >= 0.58 and relevance >= 0.62:
        return "contextual"
    if relevance < 0.45 or anchor_quality < 0.42:
        return "weak_context"
    return "mixed"


def _classify_removal_link(score: float, placement: str, relevance: float, anchor_quality: float, target_traffic: int, target_in_degree: int) -> str:
    if relevance < 0.38 and anchor_quality < 0.45:
        return "potentially_harmful"
    if score >= 70 and target_traffic > 0:
        return "critical"
    if score >= 35:
        return "useful"
    if placement == "template_navigation" or target_in_degree >= 8:
        return "redundant"
    if relevance < 0.5:
        return "irrelevant"
    return "useful" if score >= 18 else "redundant"


def _best_paragraph_context(
    source_idx: int,
    target_idx: int,
    paragraph_records: list[tuple[int, int, str, np.ndarray]] | None,
    embeddings: np.ndarray | None,
) -> dict:
    if not paragraph_records or embeddings is None or target_idx < 0 or target_idx >= len(embeddings):
        return {}
    target_vec = embeddings[target_idx]
    best: tuple[float, int, str] | None = None
    for page_i, para_i, text, vec in paragraph_records:
        if int(page_i) != source_idx:
            continue
        try:
            score = float(np.clip(vec @ target_vec, -1.0, 1.0))
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, int(para_i), str(text or ""))
    if best is None:
        return {}
    excerpt = re.sub(r"\s+", " ", best[2]).strip()
    return {
        "paragraph_index": best[1],
        "paragraph_fit": round(best[0], 4),
        "paragraph_excerpt": excerpt[:260],
    }


def link_removal_simulation_payload(
    result: LinkGraphResult,
    pages,
    embeddings: np.ndarray | None = None,
    paragraph_records: list[tuple[int, int, str, np.ndarray]] | None = None,
    *,
    traffic_authority: dict | None = None,
    max_candidates: int = 2500,
    max_context_rows: int = 250,
) -> dict:
    if not result.graph:
        return {"summary": {"status": "no_graph", "total_edges": 0}, "links": [], "critical_links": [], "edit_warnings": []}

    page_by_url = {p.url: p for p in pages}
    page_idx = {p.url: i for i, p in enumerate(pages)}
    authority_by_url = {
        row.get("url"): row
        for row in (traffic_authority or {}).get("pages", [])
        if row.get("url")
    }
    weights = _edge_weights(result)
    out_weight_sum: dict[str, float] = {}
    for src, targets in result.graph.items():
        out_weight_sum[src] = sum(max(0.05, weights.get((src, tgt), 1.0)) for tgt in targets)

    raw_rows: list[dict] = []
    for src, targets in result.graph.items():
        src_row = authority_by_url.get(src) or {}
        src_pr = _safe_float(src_row.get("traffic_weighted_pagerank")) or _safe_float(result.pagerank.get(src))
        src_standard_pr = _safe_float(result.pagerank.get(src))
        src_traffic = _safe_int(src_row.get("traffic"))
        for tgt in targets:
            tgt_row = authority_by_url.get(tgt) or {}
            edge = (src, tgt)
            edge_weight = max(0.05, _safe_float(weights.get(edge, 1.0)))
            share = edge_weight / max(0.05, out_weight_sum.get(src, 1.0))
            anchor_quality = _safe_float(result.edge_anchor_quality.get(edge, 0.5))
            relevance = _safe_float(result.edge_contextual_relevance.get(edge, 0.5))
            target_traffic = _safe_int(tgt_row.get("traffic"))
            target_keywords = _safe_int(tgt_row.get("keywords"))
            target_gap = max(0.0, _safe_float(tgt_row.get("authority_traffic_gap")))
            target_in_degree = _safe_int(result.in_degree.get(tgt))
            direct_pr_loss = 0.85 * src_standard_pr * share
            weighted_pr_loss = 0.85 * src_pr * share
            demand_boost = 1.0 + math.log1p(target_traffic) + math.sqrt(target_keywords) * 0.2 + target_gap
            quality_boost = 0.35 + anchor_quality * 0.35 + relevance * 0.3
            source_boost = 1.0 + math.log1p(src_traffic) * 0.25
            raw_loss = weighted_pr_loss * demand_boost * quality_boost * source_boost
            anchors = result.edge_anchor_texts.get(edge, [])
            placement = _edge_placement(src, tgt, anchors, anchor_quality, relevance, target_in_degree)
            raw_rows.append({
                "source_url": src,
                "source_title": getattr(page_by_url.get(src), "title", src),
                "target_url": tgt,
                "target_title": getattr(page_by_url.get(tgt), "title", tgt),
                "source_section": getattr(page_by_url.get(src), "section", ""),
                "target_section": getattr(page_by_url.get(tgt), "section", ""),
                "anchor_samples": anchors[:5],
                "anchor_count": int(result.edge_anchor_count.get(edge, 0)),
                "placement": placement,
                "edge_weight": round(edge_weight, 6),
                "edge_share": round(share, 6),
                "pagerank_loss": round(direct_pr_loss, 10),
                "weighted_pagerank_loss": round(weighted_pr_loss, 10),
                "raw_loss": raw_loss,
                "anchor_quality": round(anchor_quality, 4),
                "contextual_relevance": round(relevance, 4),
                "source_traffic": src_traffic,
                "target_traffic": target_traffic,
                "target_keywords": target_keywords,
                "target_authority_gap": round(target_gap, 4),
                "target_in_degree": target_in_degree,
                "target_click_depth": tgt_row.get("click_depth"),
            })

    raw_rows.sort(key=lambda r: _safe_float(r.get("raw_loss")), reverse=True)
    sampled = raw_rows[:max_candidates]
    max_loss = max((_safe_float(r.get("raw_loss")) for r in sampled), default=0.0) or 1.0
    for rank, row in enumerate(sampled, 1):
        score = min(100.0, _safe_float(row.get("raw_loss")) / max_loss * 100.0)
        row["rank"] = rank
        row["removal_loss_score"] = round(score, 2)
        row["classification"] = _classify_removal_link(
            score,
            str(row.get("placement") or ""),
            _safe_float(row.get("contextual_relevance")),
            _safe_float(row.get("anchor_quality")),
            _safe_int(row.get("target_traffic")),
            _safe_int(row.get("target_in_degree")),
        )
        row["recommended_action"] = (
            "Protect this link during edits; changing or removing it likely weakens authority flow to a demand page."
            if row["classification"] == "critical"
            else "Keep if editorially natural; it contributes useful discovery or semantic reinforcement."
            if row["classification"] == "useful"
            else "Review before expanding similar links; this edge looks low-impact or template-driven."
            if row["classification"] == "redundant"
            else "Consider replacing with a more relevant contextual link and clearer anchor text."
        )

    context_limit = min(max_context_rows, len(sampled))
    for row in sampled[:context_limit]:
        src_i = page_idx.get(row["source_url"], -1)
        tgt_i = page_idx.get(row["target_url"], -1)
        row.update(_best_paragraph_context(src_i, tgt_i, paragraph_records, embeddings))

    counts = collections.Counter(row.get("classification") for row in sampled)
    placement_counts = collections.Counter(row.get("placement") for row in sampled)
    critical = [r for r in sampled if r.get("classification") == "critical"]
    edit_warnings: dict[str, dict] = {}
    for row in critical:
        src = row["source_url"]
        warning = edit_warnings.setdefault(src, {
            "source_url": src,
            "source_title": row.get("source_title") or src,
            "critical_links": 0,
            "protected_targets": [],
            "max_loss_score": 0.0,
            "warning": "This page contains critical internal links. Review protected targets before deleting sections, changing anchors, or simplifying navigation.",
        })
        warning["critical_links"] += 1
        warning["max_loss_score"] = max(_safe_float(warning.get("max_loss_score")), _safe_float(row.get("removal_loss_score")))
        if len(warning["protected_targets"]) < 8:
            warning["protected_targets"].append({
                "target_url": row["target_url"],
                "target_title": row.get("target_title") or row["target_url"],
                "loss_score": row.get("removal_loss_score"),
                "anchor_samples": row.get("anchor_samples") or [],
            })
    warnings = sorted(edit_warnings.values(), key=lambda r: (_safe_float(r.get("max_loss_score")), _safe_int(r.get("critical_links"))), reverse=True)

    return {
        "summary": {
            "status": "ok",
            "model": "internal_link_removal_simulation_v1",
            "total_edges": sum(len(v) for v in result.graph.values()),
            "simulated_edges": len(sampled),
            "critical_links": counts.get("critical", 0),
            "useful_links": counts.get("useful", 0),
            "redundant_links": counts.get("redundant", 0),
            "irrelevant_links": counts.get("irrelevant", 0),
            "potentially_harmful_links": counts.get("potentially_harmful", 0),
            "contextual_links": placement_counts.get("contextual", 0),
            "template_navigation_links": placement_counts.get("template_navigation", 0),
            "weak_context_links": placement_counts.get("weak_context", 0),
        },
        "links": sampled[:700],
        "critical_links": critical[:200],
        "template_links": [r for r in sampled if r.get("placement") == "template_navigation"][:200],
        "weak_or_harmful_links": [r for r in sampled if r.get("classification") in {"irrelevant", "potentially_harmful"}][:200],
        "edit_warnings": warnings[:200],
        "interpretation": {
            "removal_loss_score": "Approximate first-order loss from removing the edge, weighted by source authority, target traffic/keyword demand, anchor quality, and semantic relevance.",
            "placement": "Contextual links have descriptive anchors and source-target semantic fit. Template/navigation links are separated so editors do not confuse repeated global links with body-copy recommendations.",
        },
    }


def link_flow_payload(
    result: LinkGraphResult,
    pages,
    top_pages: list[dict] | None = None,
    *,
    page_types: dict | None = None,
    traffic_authority: dict | None = None,
    link_removal: dict | None = None,
    contextual_links: dict | None = None,
    max_nodes: int = 360,
    max_edges: int = 1600,
) -> dict:
    """Small edge-bundling payload for the internal link-equity flow view.

    The full graph can be too dense for radial bundling. This samples pages by
    link authority and organic traffic, then keeps the strongest links among
    those pages. Traffic is optional; when present, high-demand pages are forced
    into the node set so the chart can show whether link equity reaches them.
    """
    by_url = {p.url: p for p in pages}
    page_type_by_url = _page_type_lookup(page_types)
    authority_by_url = {
        row.get("url"): row
        for row in (traffic_authority or {}).get("pages", [])
        if row.get("url")
    }
    removal_by_edge = {
        (row.get("source_url"), row.get("target_url")): row
        for row in (link_removal or {}).get("links", [])
        if row.get("source_url") and row.get("target_url")
    }
    contextual_by_edge: dict[tuple[str, str], dict] = {}
    for row in (contextual_links or {}).get("links", []):
        key = (row.get("source_url"), row.get("target_url"))
        if not key[0] or not key[1]:
            continue
        current = contextual_by_edge.get(key)
        if current is None or float(row.get("contextual_link_impact", 0.0) or 0.0) > float(current.get("contextual_link_impact", 0.0) or 0.0):
            contextual_by_edge[key] = row
    traffic_by_url: dict[str, dict] = {}
    for row in top_pages or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        traffic_by_url[url] = row

    nodes = set(result.graph.keys())
    if not nodes:
        return {"nodes": [], "edges": [], "total_edges": 0, "shown_edges": 0, "shown_nodes": 0}

    def _norm_lookup(values: dict[str, float]) -> dict[str, float]:
        vmax = max(values.values()) if values else 0.0
        if vmax <= 0:
            return {k: 0.0 for k in values}
        return {k: float(v) / vmax for k, v in values.items()}

    traffic = {u: float((traffic_by_url.get(u) or {}).get("traffic", 0) or 0) for u in nodes}
    pr_n = _norm_lookup({u: float(result.pagerank.get(u, 0.0)) for u in nodes})
    auth_n = _norm_lookup({u: float(result.authority_score.get(u, 0.0)) for u in nodes})
    hub_n = _norm_lookup({u: float(result.hub_score.get(u, 0.0)) for u in nodes})
    in_n = _norm_lookup({u: float(result.in_degree.get(u, 0)) for u in nodes})
    out_n = _norm_lookup({u: float(result.out_degree.get(u, 0)) for u in nodes})
    traffic_n = _norm_lookup({u: np.log1p(v) for u, v in traffic.items()})

    score = {
        u: (
            0.34 * pr_n.get(u, 0.0)
            + 0.20 * auth_n.get(u, 0.0)
            + 0.15 * hub_n.get(u, 0.0)
            + 0.21 * traffic_n.get(u, 0.0)
            + 0.07 * in_n.get(u, 0.0)
            + 0.03 * out_n.get(u, 0.0)
        )
        for u in nodes
    }

    selected: set[str] = set()

    def _add_top(values: dict[str, float], n: int) -> None:
        for url, _ in sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:n]:
            if url in nodes:
                selected.add(url)

    _add_top(score, max_nodes)
    _add_top(traffic, min(120, max_nodes // 3))
    _add_top({u: float(result.in_degree.get(u, 0)) for u in nodes}, min(80, max_nodes // 4))
    _add_top({u: float(result.hub_score.get(u, 0.0)) for u in nodes}, min(60, max_nodes // 5))
    _add_top({u: float(result.pagerank.get(u, 0.0)) for u in nodes}, min(80, max_nodes // 4))

    if len(selected) > max_nodes:
        selected = set(sorted(selected, key=lambda u: score.get(u, 0.0), reverse=True)[:max_nodes])

    page_rows = []
    for url in sorted(selected, key=lambda u: score.get(u, 0.0), reverse=True):
        page = by_url.get(url)
        traffic_row = traffic_by_url.get(url) or {}
        authority_row = authority_by_url.get(url) or {}
        page_rows.append({
            "url": url,
            "title": page.title if page else url,
            "section": page.section if page else "",
            "directory": authority_row.get("directory") or _directory(url),
            "cluster": authority_row.get("cluster") or traffic_row.get("cluster_label") or (page.section if page else ""),
            "page_type": authority_row.get("page_type") or page_type_by_url.get(_canonical_url(url), ""),
            "traffic": int(traffic_row.get("traffic", 0) or 0),
            "keywords": int(traffic_row.get("keywords", 0) or 0),
            "top_keyword": traffic_row.get("top_keyword", ""),
            "pagerank": round(float(result.pagerank.get(url, 0.0)), 8),
            "weighted_pagerank": round(float(authority_row.get("weighted_pagerank", 0.0) or 0.0), 8),
            "traffic_weighted_pagerank": round(float(authority_row.get("traffic_weighted_pagerank", 0.0) or 0.0), 8),
            "authority_traffic_gap": round(float(authority_row.get("authority_traffic_gap", 0.0) or 0.0), 4),
            "mismatch_label": authority_row.get("mismatch_label", ""),
            "hub_score": round(float(result.hub_score.get(url, 0.0)), 8),
            "authority_score": round(float(result.authority_score.get(url, 0.0)), 8),
            "in_degree": int(result.in_degree.get(url, 0)),
            "out_degree": int(result.out_degree.get(url, 0)),
            "click_depth": int(result.click_depth.get(url, -1)) if url in result.click_depth else None,
            "flow_score": round(float(score.get(url, 0.0)), 6),
        })

    page_meta = {row["url"]: row for row in page_rows}
    edge_rows = []
    for src in selected:
        for tgt in result.graph.get(src, []):
            if tgt not in selected:
                continue
            weight = max(1, int(result.edge_anchor_count.get((src, tgt), 0) or 1))
            source_pr = float(result.pagerank.get(src, 0.0))
            target_traffic = float(traffic.get(tgt, 0.0))
            removal_row = removal_by_edge.get((src, tgt)) or {}
            contextual_row = contextual_by_edge.get((src, tgt)) or {}
            source_node = page_meta.get(src, {})
            target_node = page_meta.get(tgt, {})
            anchor_samples = [
                anchor
                for anchor, _ in Counter(result.edge_anchor_texts.get((src, tgt), [])).most_common(5)
                if anchor
            ]
            flow_score = (
                source_pr * 2.5
                + float(result.hub_score.get(src, 0.0))
                + float(result.authority_score.get(tgt, 0.0))
                + traffic_n.get(tgt, 0.0)
                + np.log1p(weight) * 0.05
            )
            edge_rows.append({
                "source": src,
                "target": tgt,
                "source_title": source_node.get("title") or src,
                "target_title": target_node.get("title") or tgt,
                "source_directory": source_node.get("directory") or _directory(src),
                "target_directory": target_node.get("directory") or _directory(tgt),
                "source_cluster": source_node.get("cluster") or source_node.get("section") or "",
                "target_cluster": target_node.get("cluster") or target_node.get("section") or "",
                "source_page_type": source_node.get("page_type") or "",
                "target_page_type": target_node.get("page_type") or "",
                "weight": weight,
                "source_pagerank": round(source_pr, 8),
                "target_pagerank": round(float(result.pagerank.get(tgt, 0.0)), 8),
                "target_traffic": int(target_traffic),
                "target_keywords": int(target_node.get("keywords", 0) or 0),
                "target_mismatch_label": target_node.get("mismatch_label", ""),
                "target_authority_traffic_gap": round(float(target_node.get("authority_traffic_gap", 0.0) or 0.0), 4),
                "anchor_samples": anchor_samples,
                "score": round(float(flow_score), 8),
                "removal_loss_score": round(float(removal_row.get("removal_loss_score", 0.0) or 0.0), 2),
                "removal_classification": removal_row.get("classification", ""),
                "placement": removal_row.get("placement", ""),
                "contextual_link_impact": round(float(contextual_row.get("contextual_link_impact", 0.0) or 0.0), 2),
                "context_type": contextual_row.get("context_type", ""),
            })
    edge_rows.sort(key=lambda r: r["score"], reverse=True)
    edge_rows = edge_rows[:max_edges]

    return {
        "nodes": page_rows,
        "edges": edge_rows,
        "shown_nodes": len(page_rows),
        "shown_edges": len(edge_rows),
        "total_edges": int(result.edge_count),
        "node_limit": max_nodes,
        "edge_limit": max_edges,
    }
