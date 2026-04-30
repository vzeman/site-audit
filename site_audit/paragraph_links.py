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
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

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
    from urllib.parse import urlparse
    try:
        path = urlparse(url).path or "/"
    except Exception:
        return False
    return bool(_NAV_PATH_RE.match(path))


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

    # answerability filter: low-quality target pages get skipped
    drop_target_idx: set[int] = set()
    for i, p in enumerate(pages):
        if _is_nav_url(p.url):
            drop_target_idx.add(i)
        if answerability_by_url is not None and answerability_by_url.get(p.url, 10.0) < 1.0:
            drop_target_idx.add(i)

    # anchor scoring: we batch-encode candidate n-grams per paragraph
    can_score_anchors = embedder is not None

    raw_recs: list[ParagraphLinkRec] = []
    page_target_seen: set[tuple[int, int]] = set()  # one rec per (source_page, target_page)

    for pi, (page_i, para_i, para_text, _) in enumerate(paragraph_records):
        row = sims[pi]
        page_avg = page_avg_sims[page_i]

        order = np.argsort(-row)
        page_recs_count = 0
        for tgt_i in order[:30]:  # explore top 30 candidates
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
            if tgt_url in existing_links.get(src_url, set()):
                continue
            if (page_i, tgt_i) in page_target_seen:
                continue
            page_target_seen.add((page_i, tgt_i))

            # anchor selection
            if can_score_anchors:
                ngrams = _candidate_ngrams(para_text)
                if ngrams:
                    ng_embs = embedder.encode(ngrams, batch_size=128, show_progress=False)
                    ng_sims = ng_embs @ page_embeddings[tgt_i]
                    best_idx = int(np.argmax(ng_sims))
                    suggested_anchor = ngrams[best_idx]
                    anchor_conf = float(ng_sims[best_idx])
                else:
                    suggested_anchor = pages[tgt_i].title[:60]
                    anchor_conf = 0.0
            else:
                suggested_anchor = pages[tgt_i].title[:60]
                anchor_conf = 0.0

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
            page_recs_count += 1
            if page_recs_count >= top_k_per_page:
                break

    # sort by lift then fit; keep top_k_total
    raw_recs.sort(key=lambda r: (r.lift, r.fit), reverse=True)
    return raw_recs[:top_k_total]


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
