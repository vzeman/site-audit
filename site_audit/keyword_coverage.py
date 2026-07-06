"""Map a list of target queries onto pages in the same vector space.

Three things drop out of one cosine-similarity pass:

* **Best page per query** (the canonical answer for that query)
* **Coverage gaps**     — queries with no page above a usefulness floor
* **Cannibalization**   — queries with N+ pages above a "too similar"
                          ceiling, meaning the site is competing with
                          itself for that intent

Queries can be:
  - explicit (``--queries-file`` with one query per line, optionally
    grouped by section: ``# section name`` then queries underneath)
  - auto-mined from the crawl: page titles + question-form H2/H3
"""

from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)


_QUESTION_PREFIXES = (
    "how ", "what ", "why ", "when ", "where ", "which ", "who ", "can ",
    "is ", "are ", "should ", "do ", "does ", "will ",
)


@dataclass
class QueryMatch:
    query: str
    source: str              # 'manual', 'title', 'h2'
    best_url: str
    best_title: str
    best_similarity: float
    runner_ups: list[dict]   # [{url, title, similarity}]
    status: str              # 'covered', 'gap', 'cannibalized'
    candidates_above_threshold: int


def load_queries_from_file(path: Path) -> list[str]:
    queries: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    # de-dupe while preserving order
    seen: set[str] = set()
    out = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    return any(t.startswith(prefix) for prefix in _QUESTION_PREFIXES)


def auto_mine_queries(pages, max_queries: int = 200) -> list[tuple[str, str]]:
    """Return list of (query, source). Source is 'title' or 'h2'."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    # 1) question-form headings first (highest signal)
    for p in pages:
        for h in getattr(p, "headings", []) or []:
            if not _looks_like_question(h):
                continue
            if len(h) < 8 or len(h) > 140:
                continue
            key = re.sub(r"\s+", " ", h.lower()).strip("? ")
            if key in seen:
                continue
            seen.add(key)
            out.append((h, "h2"))

    # 2) titles (cap each title to a reasonable phrase length)
    for p in pages:
        title = (p.title or "").strip()
        if not title or len(title) < 6 or len(title) > 140:
            continue
        # strip the typical " | Brand" suffix
        title_clean = re.split(r"[|·–—-]\s+\w[\w\s&]+$", title)[0].strip()
        key = title_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((title_clean, "title"))

    return out[:max_queries]


def match_queries(
    queries_with_source: list[tuple[str, str]],
    query_embeddings: np.ndarray,
    pages,
    page_embeddings: np.ndarray,
    coverage_threshold: float = 0.55,
    cannibalization_threshold: float = 0.72,
    max_runner_ups: int = 4,
) -> list[QueryMatch]:
    if len(queries_with_source) == 0 or len(pages) == 0:
        return []

    sims = query_embeddings @ page_embeddings.T  # [Q, P]
    sims = np.clip(sims, -1.0, 1.0)

    out: list[QueryMatch] = []
    for qi, (query, source) in enumerate(queries_with_source):
        row = sims[qi]
        order = np.argsort(-row)
        best = int(order[0])
        best_sim = float(row[best])

        runner_ups = []
        for pi in order[1:max_runner_ups + 1]:
            runner_ups.append({
                "url": pages[int(pi)].url,
                "title": pages[int(pi)].title,
                "similarity": float(round(row[int(pi)], 4)),
            })

        above = int(np.sum(row >= cannibalization_threshold))
        if best_sim < coverage_threshold:
            status = "gap"
        elif above >= 3:
            status = "cannibalized"
        else:
            status = "covered"

        out.append(QueryMatch(
            query=query,
            source=source,
            best_url=pages[best].url,
            best_title=pages[best].title,
            best_similarity=float(round(best_sim, 4)),
            runner_ups=runner_ups,
            status=status,
            candidates_above_threshold=above,
        ))

    return out


def to_payload(matches: Iterable[QueryMatch]) -> list[dict]:
    return [
        {
            "query": m.query,
            "source": m.source,
            "status": m.status,
            "best_similarity": m.best_similarity,
            "best_url": m.best_url,
            "best_title": m.best_title,
            "candidates_above_threshold": m.candidates_above_threshold,
            "runner_ups": m.runner_ups,
        }
        for m in matches
    ]


@dataclass
class ParagraphQueryMatch:
    query: str
    source: str
    best_similarity: float
    best_url: str
    best_paragraph: str
    distinct_pages_above_floor: int    # # of distinct pages that have ≥1 paragraph above the floor
    status: str                        # 'gap', 'scattered', 'focused', 'covered'
    top_paragraphs: list[dict]         # [{url, page_title, excerpt, similarity}]


def match_queries_to_paragraphs(
    queries_with_source: list[tuple[str, str]],
    query_embeddings: np.ndarray,
    pages,
    paragraph_records: list,           # [(page_index, para_index, text, embedding)]
    paragraph_floor: float = 0.65,
    scattered_threshold_pages: int = 4,
    top_k_paragraphs: int = 5,
) -> list[ParagraphQueryMatch]:
    """For each query, surface the top paragraphs across the whole site.

    This models how AI answer engines retrieve content: they pick
    paragraph-sized chunks, not whole pages. Statuses:

    * **gap** — best paragraph similarity below the floor (no chunk is
      a plausible answer)
    * **scattered** — paragraphs above the floor are spread across
      ``scattered_threshold_pages`` or more pages (no clear citation target)
    * **focused** — the top 3 paragraphs are on the same page
    * **covered** — between scattered and focused
    """
    if not queries_with_source or not paragraph_records:
        return []

    para_embs = np.stack([r[3] for r in paragraph_records]).astype(np.float32)
    block_size = 20000
    candidate_limit = max(1, top_k_paragraphs * 4)

    out: list[ParagraphQueryMatch] = []
    for qi, (query, source) in enumerate(queries_with_source):
        q = query_embeddings[qi].astype(np.float32, copy=False)
        heap: list[tuple[float, int]] = []
        pages_above_floor: set[str] = set()
        for start in range(0, len(para_embs), block_size):
            row = np.clip(para_embs[start : start + block_size] @ q, -1.0, 1.0)
            above = np.flatnonzero(row >= paragraph_floor)
            for local_idx in above:
                page_i = paragraph_records[start + int(local_idx)][0]
                pages_above_floor.add(pages[page_i].url)
            if len(row) <= candidate_limit:
                top_local = np.arange(len(row))
            else:
                top_local = np.argpartition(row, -candidate_limit)[-candidate_limit:]
            for local_idx in top_local:
                sim = float(row[int(local_idx)])
                global_idx = start + int(local_idx)
                if len(heap) < candidate_limit:
                    heapq.heappush(heap, (sim, global_idx))
                elif sim > heap[0][0]:
                    heapq.heapreplace(heap, (sim, global_idx))
        order = [idx for _, idx in sorted(heap, reverse=True)]

        top: list[dict] = []
        seen_urls: list[str] = []
        for pi in order:
            page_i, para_i, text, _ = paragraph_records[int(pi)]
            url = pages[page_i].url
            sim = float(np.clip(para_embs[int(pi)] @ q, -1.0, 1.0))
            if sim < paragraph_floor and len(top) >= 1:
                break
            top.append({
                "url": url,
                "page_title": pages[page_i].title,
                "excerpt": text[:240],
                "similarity": float(round(sim, 4)),
            })
            seen_urls.append(url)
            if len(top) >= top_k_paragraphs:
                break

        best_sim = top[0]["similarity"] if top else 0.0
        best_url = top[0]["url"] if top else ""
        best_para = top[0]["excerpt"] if top else ""
        # # of distinct pages that have ≥1 paragraph above floor
        distinct_pages = len(pages_above_floor)

        if best_sim < paragraph_floor:
            status = "gap"
        else:
            top3_pages = {t["url"] for t in top[:3]}
            if distinct_pages >= scattered_threshold_pages:
                status = "scattered"
            elif len(top3_pages) == 1:
                status = "focused"
            else:
                status = "covered"

        out.append(ParagraphQueryMatch(
            query=query,
            source=source,
            best_similarity=best_sim,
            best_url=best_url,
            best_paragraph=best_para,
            distinct_pages_above_floor=distinct_pages,
            status=status,
            top_paragraphs=top,
        ))

    return out


def paragraph_match_payload(matches: list[ParagraphQueryMatch]) -> list[dict]:
    return [
        {
            "query": m.query,
            "source": m.source,
            "status": m.status,
            "best_similarity": m.best_similarity,
            "best_url": m.best_url,
            "best_paragraph": m.best_paragraph,
            "distinct_pages_above_floor": m.distinct_pages_above_floor,
            "top_paragraphs": m.top_paragraphs,
        }
        for m in matches
    ]
