"""In-paragraph internal-link recommendations.

Page-level recommendations ("page A and page B are 0.91 similar, link
them") are useful but vague: the editor still has to decide *where* on
page A to put the link. Paragraph-level recommendations close that gap
by handing the editor:

  - the *exact paragraph* on the source page
  - the *exact target* page
  - a suggested *anchor phrase* drawn from words actually present in
    the paragraph

The "lift" metric is what separates this from a page-level rec: we
require the paragraph to be more similar to the target than the *page
as a whole* is. That filters out cases where a page is generally on the
target's topic — those are already surfaced as page-level recs.

Algorithm sketch::

    for each paragraph P on source page X:
        sims = embeddings[T] @ P  for all T != X
        top = take top-K candidates above SIM_FLOOR
        for each candidate T:
            lift = sim(P, T) - sim(X_centroid, T)
            if lift < LIFT_FLOOR: skip
            if X already has an internal link to T: skip
            anchor = best n-gram in P by cosine to T's embedding
            yield rec(P, T, fit, lift, anchor)
        keep top-N per (page, target) pair after diversity filter
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import numpy as np

LOG = logging.getLogger(__name__)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "our", "your",
    "their", "you", "we", "they", "i", "us", "them", "if", "so", "than",
    "then", "do", "does", "did", "have", "has", "had", "will", "would",
    "could", "should", "can", "may", "might", "shall", "into", "out", "up",
    "down", "very", "more", "most", "some", "any", "no", "not", "only",
    "also", "other", "such", "about", "over", "under", "all",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_NAV_PATH_RE = re.compile(r"^/(cart|checkout|login|sign[-_]?in|sign[-_]?up|account|search|contact|terms|privacy|legal|admin|wp-admin|cdn-cgi)/?", re.I)


@dataclass
class ParagraphLinkRec:
    source_url: str
    paragraph_index: int
    paragraph_excerpt: str
    target_url: str
    target_title: str
    fit: float
    lift: float
    suggested_anchor: str
    anchor_confidence: float


# --- helpers --------------------------------------------------------------


def _is_nav_url(url: str) -> bool:
    try:
        path = urlparse(url).path or "/"
    except Exception:
        return False
    return bool(_NAV_PATH_RE.match(path))


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


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _candidate_ngrams(text: str, max_n: int = 5, min_n: int = 2) -> list[str]:
    tokens = _tokenize(text)
    out: set[str] = set()
    for n in range(min_n, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            window = tokens[i : i + n]
            # discard windows that are *only* stopwords
            if all(t.lower() in _STOPWORDS for t in window):
                continue
            phrase = " ".join(window)
            if 4 <= len(phrase) <= 60:
                out.add(phrase)
    return list(out)[:120]  # cap candidates per paragraph


def _existing_link_targets_for_page(
    pages_with_outlinks: list[tuple[str, list[tuple[str, str]]]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for src, outs in pages_with_outlinks:
        for tgt, _ in outs:
            out[src].add(tgt)
    return dict(out)


# --- algorithm ------------------------------------------------------------


def recommend(
    pages,                                  # list of PageInfo
    page_embeddings: np.ndarray,
    paragraph_records: list[tuple[int, int, str, np.ndarray]],
    # paragraph_records[i] = (page_index, para_index, para_text, para_embedding)
    pages_with_outlinks: list[tuple[str, list[tuple[str, str]]]],
    answerability_by_url: dict[str, float] | None = None,
    embedder=None,                          # used for anchor scoring
    paragraph_saturation: dict[tuple[int, int], float] | None = None,
    saturation_floor_per_100w: float = 5.0,
    similarity_floor: float = 0.65,
    lift_floor: float = 0.05,
    top_k_per_page: int = 8,
    top_k_total: int = 200,
) -> list[ParagraphLinkRec]:
    if not paragraph_records or len(pages) == 0:
        return []

    n_pages = len(pages)
    para_embs = np.stack([r[3] for r in paragraph_records]).astype(np.float32)
    n_paras = len(para_embs)

    LOG.info("  paragraph link recs: scoring %d paragraphs vs %d pages", n_paras, n_pages)

    # full sim matrix, paragraphs x pages — n_paras * n_pages * 4 bytes
    # 30 000 * 5 000 = 600 MB which is too much. Block-process if so.
    block_size = max(1, 5_000_000 // n_pages)
    sims_blocks: list[np.ndarray] = []
    for start in range(0, n_paras, block_size):
        sub = para_embs[start : start + block_size]
        sims_blocks.append(np.clip(sub @ page_embeddings.T, -1.0, 1.0))
    sims = np.vstack(sims_blocks)

    # average per-page similarity to its own pages' paragraphs is what
    # we'll use as the "fit baseline" for lift. Equivalent to:
    # sim(X_centroid, T) approximated by mean(sim(P_in_X, T))
    by_page: dict[int, list[int]] = defaultdict(list)
    for pi, (page_i, _, _, _) in enumerate(paragraph_records):
        by_page[page_i].append(pi)
    page_avg_sims = np.zeros((n_pages, n_pages), dtype=np.float32)
    for page_i, para_idxs in by_page.items():
        page_avg_sims[page_i] = sims[para_idxs].mean(axis=0)

    existing_links = _existing_link_targets_for_page(pages_with_outlinks)
    canonical_existing_links = {
        _canonical_url(src): {_canonical_url(tgt) for tgt in targets}
        for src, targets in existing_links.items()
    }

    # answerability filter: low-quality target pages get skipped
    drop_target_idx: set[int] = set()
    for i, p in enumerate(pages):
        if _is_nav_url(p.url):
            drop_target_idx.add(i)
        if answerability_by_url is not None and answerability_by_url.get(p.url, 10.0) < 1.0:
            drop_target_idx.add(i)

    # anchor scoring: we batch-encode candidate n-grams per paragraph
    can_score_anchors = embedder is not None

    # Pass 1: pick accepted (paragraph, target) pairs without anchor scoring.
    accepted: list[tuple[int, int, float, float]] = []  # (pi, tgt_i, fit, lift)
    page_target_seen: set[tuple[int, int]] = set()  # one rec per (source_page, target_page)
    saturated_skipped = 0

    for pi, (page_i, para_i, _, _) in enumerate(paragraph_records):
        # Already link-saturated paragraphs don't get new recommendations.
        if paragraph_saturation is not None:
            density = paragraph_saturation.get((page_i, para_i), 0.0)
            if density >= saturation_floor_per_100w:
                saturated_skipped += 1
                continue
        row = sims[pi]
        page_avg = page_avg_sims[page_i]
        order = np.argsort(-row)
        para_recs_count = 0
        for tgt_i in order[:30]:
            tgt_i = int(tgt_i)
            if tgt_i == page_i or tgt_i in drop_target_idx:
                continue
            fit = float(row[tgt_i])
            if fit < similarity_floor:
                break  # row is sorted, nothing else will pass
            lift = fit - float(page_avg[tgt_i])
            if lift < lift_floor:
                continue
            src_url = pages[page_i].url
            tgt_url = pages[tgt_i].url
            canonical_src = _canonical_url(src_url)
            canonical_tgt = _canonical_url(tgt_url)
            if canonical_src == canonical_tgt:
                continue
            if tgt_url in existing_links.get(src_url, set()):
                continue
            if canonical_tgt in canonical_existing_links.get(canonical_src, set()):
                continue
            if (page_i, tgt_i) in page_target_seen:
                continue
            page_target_seen.add((page_i, tgt_i))
            accepted.append((pi, tgt_i, fit, lift))
            para_recs_count += 1
            if para_recs_count >= top_k_per_page:
                break

    if saturated_skipped:
        LOG.info(
            "  paragraph link recs: %d paragraphs skipped as link-saturated (>= %.1f links per 100w)",
            saturated_skipped, saturation_floor_per_100w,
        )

    accepted.sort(key=lambda item: (item[3], item[2]), reverse=True)
    if top_k_total <= 0:
        return []
    if len(accepted) > top_k_total:
        LOG.info(
            "  paragraph link recs: trimming %d accepted candidates to top %d before anchor scoring",
            len(accepted), top_k_total,
        )
        accepted = accepted[:top_k_total]

    # Pass 2: encode every needed ngram in one batched call.
    para_ngrams: dict[int, list[str]] = {}
    para_ng_offsets: dict[int, tuple[int, int]] = {}
    all_ng_embs: np.ndarray = np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
    if can_score_anchors and accepted:
        unique_pis = sorted({pi for pi, _, _, _ in accepted})
        for pi in unique_pis:
            _, _, para_text, _ = paragraph_records[pi]
            para_ngrams[pi] = _candidate_ngrams(para_text)
        all_ngrams: list[str] = []
        for pi in unique_pis:
            ngrams = para_ngrams[pi]
            start = len(all_ngrams)
            all_ngrams.extend(ngrams)
            para_ng_offsets[pi] = (start, start + len(ngrams))
        if all_ngrams:
            LOG.info(
                "  paragraph link recs: encoding %d ngrams from %d paragraphs",
                len(all_ngrams), len(unique_pis),
            )
            all_ng_embs = embedder.encode(all_ngrams, batch_size=256, show_progress=True)

    # Pass 3: build records, anchor selection is now a numpy dot product.
    raw_recs: list[ParagraphLinkRec] = []
    for pi, tgt_i, fit, lift in accepted:
        page_i, para_i, para_text, _ = paragraph_records[pi]
        src_url = pages[page_i].url
        tgt_url = pages[tgt_i].url
        suggested_anchor = pages[tgt_i].title[:60]
        anchor_conf = 0.0
        if can_score_anchors and pi in para_ng_offsets:
            start, end = para_ng_offsets[pi]
            if end > start:
                ng_sims = all_ng_embs[start:end] @ page_embeddings[tgt_i]
                best_idx = int(np.argmax(ng_sims))
                suggested_anchor = para_ngrams[pi][best_idx]
                anchor_conf = float(ng_sims[best_idx])
        raw_recs.append(ParagraphLinkRec(
            source_url=src_url,
            paragraph_index=para_i,
            paragraph_excerpt=para_text[:280],
            target_url=tgt_url,
            target_title=pages[tgt_i].title,
            fit=round(fit, 4),
            lift=round(lift, 4),
            suggested_anchor=suggested_anchor,
            anchor_confidence=round(anchor_conf, 4),
        ))

    raw_recs.sort(key=lambda r: (r.lift, r.fit), reverse=True)
    return _dedupe_same_paragraph_anchor(raw_recs)


def _normalize_anchor(anchor: str) -> str:
    return re.sub(r"\s+", " ", str(anchor or "").strip().lower())


def _dedupe_same_paragraph_anchor(recs: list[ParagraphLinkRec]) -> list[ParagraphLinkRec]:
    out: list[ParagraphLinkRec] = []
    seen: set[tuple[str, int, str]] = set()
    for rec in recs:
        anchor = _normalize_anchor(rec.suggested_anchor)
        if not anchor:
            continue
        key = (rec.source_url, int(rec.paragraph_index), anchor)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def to_payload(recs: Iterable[ParagraphLinkRec]) -> list[dict]:
    return [
        {
            "source_url": r.source_url,
            "paragraph_index": r.paragraph_index,
            "paragraph_excerpt": r.paragraph_excerpt,
            "target_url": r.target_url,
            "target_title": r.target_title,
            "fit": r.fit,
            "lift": r.lift,
            "suggested_anchor": r.suggested_anchor,
            "anchor_confidence": r.anchor_confidence,
        }
        for r in recs
    ]


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


def _scale(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(100.0, value / max_value * 100.0))


def build_addition_simulation(
    recommendations: list[dict],
    pages,
    linkgraph: dict | None,
    *,
    paragraph_saturation: dict[tuple[int, int], float] | None = None,
    saturation_floor_per_100w: float = 5.0,
) -> dict:
    if not recommendations:
        return {"summary": {"status": "no_recommendations", "total_recommendations": 0}, "recommendations": []}

    linkgraph = linkgraph or {}
    authority_payload = linkgraph.get("traffic_weighted_pagerank") or {}
    authority_by_url = {
        row.get("url"): row
        for row in authority_payload.get("pages", [])
        if row.get("url")
    }
    counts_by_url = {
        row.get("url"): row
        for row in linkgraph.get("page_link_counts", [])
        if row.get("url")
    }
    page_by_url = {p.url: p for p in pages}
    page_idx = {p.url: i for i, p in enumerate(pages)}

    target_traffic_values = [
        math.log1p(_safe_int((authority_by_url.get(r.get("target_url")) or {}).get("traffic")))
        for r in recommendations
    ]
    target_keyword_values = [
        math.sqrt(_safe_int((authority_by_url.get(r.get("target_url")) or {}).get("keywords")))
        for r in recommendations
    ]
    max_traffic = max(target_traffic_values, default=0.0)
    max_keywords = max(target_keyword_values, default=0.0)

    rows: list[dict] = []
    for rec in recommendations:
        source_url = rec.get("source_url") or ""
        target_url = rec.get("target_url") or ""
        source_auth = authority_by_url.get(source_url) or counts_by_url.get(source_url) or {}
        target_auth = authority_by_url.get(target_url) or counts_by_url.get(target_url) or {}
        source_page = page_by_url.get(source_url)
        target_page = page_by_url.get(target_url)
        source_pr_pct = _safe_float(source_auth.get("weighted_pagerank_percentile") or source_auth.get("pagerank_percentile"))
        source_pr = _safe_float(source_auth.get("traffic_weighted_pagerank") or source_auth.get("pagerank"))
        source_out = _safe_int(source_auth.get("out_degree"))
        target_in = _safe_int(target_auth.get("in_degree"))
        target_traffic = _safe_int(target_auth.get("traffic"))
        target_keywords = _safe_int(target_auth.get("keywords"))
        target_gap = max(0.0, _safe_float(target_auth.get("authority_traffic_gap")))
        fit = max(0.0, _safe_float(rec.get("fit")))
        lift = max(0.0, _safe_float(rec.get("lift")))
        anchor_conf = max(0.0, _safe_float(rec.get("anchor_confidence")))
        paragraph_index = _safe_int(rec.get("paragraph_index"))
        source_i = page_idx.get(source_url, -1)
        density = paragraph_saturation.get((source_i, paragraph_index), 0.0) if paragraph_saturation is not None else 0.0

        authority_component = max(0.0, min(100.0, source_pr_pct * 100.0))
        relevance_component = max(0.0, min(100.0, fit * 65.0 + lift * 220.0))
        opportunity_component = max(
            0.0,
            min(
                100.0,
                _scale(math.log1p(target_traffic), max_traffic) * 0.62
                + _scale(math.sqrt(target_keywords), max_keywords) * 0.18
                + target_gap * 20.0,
            ),
        )
        deficit_component = max(0.0, min(100.0, (1.0 - _safe_float(target_auth.get("weighted_pagerank_percentile"))) * 65.0 + max(0, 6 - target_in) * 6.0))
        anchor_component = max(35.0, min(100.0, anchor_conf * 100.0)) if anchor_conf else 45.0
        density_component = 100.0 if density < saturation_floor_per_100w else 0.0
        expected = (
            authority_component * 0.22
            + relevance_component * 0.24
            + opportunity_component * 0.22
            + deficit_component * 0.16
            + anchor_component * 0.10
            + density_component * 0.06
        )
        estimated_gain = source_pr * (1.0 / max(1, source_out + 1)) * (0.5 + fit / 2.0) * (0.7 + anchor_component / 300.0)
        row = {
            **rec,
            "source_title": getattr(source_page, "title", source_url),
            "source_section": getattr(source_page, "section", ""),
            "target_section": getattr(target_page, "section", ""),
            "expected_benefit_score": round(expected, 2),
            "priority": "high" if expected >= 72 else "medium" if expected >= 48 else "low",
            "estimated_pagerank_gain": round(float(estimated_gain), 10),
            "current_target_in_degree": target_in,
            "after_target_in_degree": target_in + 1,
            "source_out_degree": source_out,
            "after_source_out_degree": source_out + 1,
            "target_traffic": target_traffic,
            "target_keywords": target_keywords,
            "target_authority_gap": round(target_gap, 4),
            "paragraph_links_per_100w": round(float(density), 4),
            "respects_density_cap": density < saturation_floor_per_100w,
            "score_components": {
                "authority_flow": round(authority_component, 2),
                "relevance": round(relevance_component, 2),
                "opportunity": round(opportunity_component, 2),
                "internal_link_deficit": round(deficit_component, 2),
                "anchor_relevance": round(anchor_component, 2),
                "density_cap": round(density_component, 2),
            },
            "recommended_action": "Add this contextual link in the suggested paragraph with the proposed anchor, then keep paragraph link density below the cap.",
        }
        rows.append(row)

    rows.sort(key=lambda r: (_safe_float(r.get("expected_benefit_score")), _safe_float(r.get("lift"))), reverse=True)
    priorities = Counter(row["priority"] for row in rows)
    pattern_counts = Counter(
        f"{row.get('source_section') or 'unknown'} -> {row.get('target_section') or 'unknown'}"
        for row in rows
    )
    patterns = [
        {"pattern": pattern, "count": count}
        for pattern, count in pattern_counts.most_common(40)
    ]
    return {
        "summary": {
            "status": "ok",
            "model": "internal_link_addition_simulation_v1",
            "total_recommendations": len(rows),
            "high_priority": priorities.get("high", 0),
            "medium_priority": priorities.get("medium", 0),
            "low_priority": priorities.get("low", 0),
            "avg_expected_benefit": round(sum(_safe_float(r.get("expected_benefit_score")) for r in rows) / max(1, len(rows)), 2),
            "density_safe_recommendations": sum(1 for r in rows if r.get("respects_density_cap")),
        },
        "recommendations": rows,
        "graph_preview": rows[:30],
        "patterns": patterns,
        "interpretation": {
            "expected_benefit_score": "Weighted estimate of authority flow, paragraph-target relevance, target traffic/keyword opportunity, internal-link deficit, anchor relevance, and paragraph density safety.",
            "before_after": "The simulation treats each recommendation as one new edge: target in-degree increases by one and source out-degree increases by one.",
        },
    }
