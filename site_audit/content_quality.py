"""Content-quality diagnostics + per-page "needs editing" composite score.

The audit so far surfaces structural issues page-by-page (focus, drift,
duplicates, answerability, link orphans). What it doesn't yet do is
*synthesize* those signals into a ranked "fix this page first" list
with a bullet-list of *why*.

This module produces three outputs:

1. **Title-content mismatch** — for each page, cosine between title and
   the *paragraph centroid* of the same page. Pages with title cosine
   below ~0.55 have a title that doesn't reflect the content. AI
   answer engines decide candidate pages from titles + structured data,
   so misleading titles silently hurt GEO performance. The dominant
   paragraph cluster's keywords give an immediate rewrite hint.

2. **Wrong-home paragraphs** — for each paragraph P on page X, the
   most-similar non-host page T. If P fits T meaningfully better than
   X (sim_T - sim_X_centroid > 0.10 AND sim_T > 0.70), we suggest
   moving the paragraph or splitting the page.

3. **Per-page improvement score** — weighted aggregation of all
   diagnostic signals. Output includes a ``reasons`` list so the editor
   sees *why* the page is flagged, not just a number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)


_TITLE_FAIL_THRESHOLD = 0.55          # title-content cosine below this = mismatch
_WRONG_HOME_LIFT_FLOOR = 0.10
_WRONG_HOME_SIM_FLOOR = 0.70


@dataclass
class TitleMismatch:
    url: str
    title: str
    title_to_content: float
    title_to_section_centroid: float
    suggested_keywords: list[str]


@dataclass
class WrongHomeParagraph:
    source_url: str
    paragraph_index: int
    paragraph_excerpt: str
    suggested_home_url: str
    suggested_home_title: str
    sim_to_suggested: float
    sim_to_host_centroid: float
    lift: float


@dataclass
class PageImprovement:
    url: str
    title: str
    score: float
    reasons: list[str]


# --- title ↔ content -----------------------------------------------------


def title_mismatch(
    pages,
    page_embeddings: np.ndarray,
    title_embeddings: np.ndarray,
    paragraph_records: list | None,
    section_centroids: dict[str, np.ndarray],
    cluster_labels: np.ndarray | None,
    cluster_summaries=None,
) -> list[TitleMismatch]:
    """Cosine of each page title to its paragraph centroid (or page emb if no paragraphs)."""
    out: list[TitleMismatch] = []
    if len(pages) == 0:
        return out

    # Build per-page paragraph centroid; fall back to page emb when the
    # page has no usable paragraphs.
    by_page: dict[int, list[np.ndarray]] = {}
    if paragraph_records:
        for pi, _, _, vec in paragraph_records:
            by_page.setdefault(pi, []).append(vec)

    cluster_label_lookup: dict[int, str] = {}
    if cluster_summaries:
        for s in cluster_summaries:
            cluster_label_lookup[s.cluster_id] = ", ".join(k["keyword"] for k in s.keywords[:4])

    for i, p in enumerate(pages):
        title_vec = title_embeddings[i]
        if not np.any(title_vec):
            continue
        if i in by_page:
            content_vec = np.mean(np.stack(by_page[i]), axis=0)
            n = np.linalg.norm(content_vec)
            content_vec = content_vec / n if n > 0 else content_vec
        else:
            content_vec = page_embeddings[i]
        title_to_content = float(np.clip(title_vec @ content_vec, -1.0, 1.0))

        sec_centroid = section_centroids.get(p.section)
        title_to_section = (
            float(np.clip(title_vec @ sec_centroid, -1.0, 1.0)) if sec_centroid is not None else 0.0
        )

        suggested_keywords: list[str] = []
        if cluster_labels is not None and len(cluster_labels) > i:
            cid = int(cluster_labels[i])
            label = cluster_label_lookup.get(cid, "")
            if label:
                suggested_keywords = [k.strip() for k in label.split(",") if k.strip()]

        out.append(TitleMismatch(
            url=p.url,
            title=p.title,
            title_to_content=round(title_to_content, 4),
            title_to_section_centroid=round(title_to_section, 4),
            suggested_keywords=suggested_keywords,
        ))

    out.sort(key=lambda r: r.title_to_content)
    return out


# --- wrong-home paragraphs ----------------------------------------------


def wrong_home_paragraphs(
    pages,
    page_embeddings: np.ndarray,
    paragraph_records: list,
    top_n: int = 60,
) -> list[WrongHomeParagraph]:
    if not paragraph_records or len(pages) < 2:
        return []

    n_pages = len(pages)
    by_page: dict[int, list[int]] = {}
    for k, (pi, _, _, _) in enumerate(paragraph_records):
        by_page.setdefault(pi, []).append(k)

    # paragraph centroid per page (used for sim_to_host_centroid)
    page_para_centroid = np.zeros_like(page_embeddings)
    for pi, idxs in by_page.items():
        embs = np.stack([paragraph_records[k][3] for k in idxs])
        m = embs.mean(axis=0)
        norm = np.linalg.norm(m)
        page_para_centroid[pi] = m / norm if norm > 0 else m

    out: list[WrongHomeParagraph] = []
    for k, (pi, para_i, text, vec) in enumerate(paragraph_records):
        sims_to_pages = np.clip(page_embeddings @ vec, -1.0, 1.0)
        sims_to_pages[pi] = -1.0  # exclude host
        target_i = int(np.argmax(sims_to_pages))
        sim_to_target = float(sims_to_pages[target_i])
        sim_to_host = float(np.clip(page_para_centroid[pi] @ vec, -1.0, 1.0))
        lift = sim_to_target - sim_to_host
        if sim_to_target < _WRONG_HOME_SIM_FLOOR:
            continue
        if lift < _WRONG_HOME_LIFT_FLOOR:
            continue
        out.append(WrongHomeParagraph(
            source_url=pages[pi].url,
            paragraph_index=para_i,
            paragraph_excerpt=text[:240],
            suggested_home_url=pages[target_i].url,
            suggested_home_title=pages[target_i].title,
            sim_to_suggested=round(sim_to_target, 4),
            sim_to_host_centroid=round(sim_to_host, 4),
            lift=round(lift, 4),
        ))

    out.sort(key=lambda r: r.lift, reverse=True)
    return out[:top_n]


# --- per-page improvement score -----------------------------------------


def per_page_improvement(
    pages,
    title_mismatches: list[TitleMismatch],
    answerability: list[dict] | None,
    distance_to_section: np.ndarray,
    duplicate_pairs: Iterable[tuple[int, int, float]],
    wrong_homes: list[WrongHomeParagraph],
    external_per_page: list[dict] | None,
    top_n: int = 100,
) -> list[PageImprovement]:
    by_url_title_match = {tm.url: tm for tm in title_mismatches}
    by_url_answer = {a["url"]: a for a in (answerability or [])}
    by_url_external = {e["url"]: e for e in (external_per_page or [])}
    duplicate_urls: dict[str, set[str]] = {}
    pages_by_idx = {i: p for i, p in enumerate(pages)}
    for i, j, _ in duplicate_pairs:
        a, b = pages_by_idx[i].url, pages_by_idx[j].url
        duplicate_urls.setdefault(a, set()).add(b)
        duplicate_urls.setdefault(b, set()).add(a)

    wrong_home_count: dict[str, int] = {}
    for wh in wrong_homes:
        wrong_home_count[wh.source_url] = wrong_home_count.get(wh.source_url, 0) + 1

    avg_answer = (
        sum(a["score"] for a in answerability) / len(answerability)
        if answerability else 5.0
    )

    rows: list[PageImprovement] = []
    for i, p in enumerate(pages):
        score = 0.0
        reasons: list[str] = []

        tm = by_url_title_match.get(p.url)
        if tm and tm.title_to_content < _TITLE_FAIL_THRESHOLD:
            penalty = (_TITLE_FAIL_THRESHOLD - tm.title_to_content) * 10
            score += 2.0 * penalty
            hint = ""
            if tm.suggested_keywords:
                hint = f" — try keywords: {', '.join(tm.suggested_keywords[:3])}"
            reasons.append(f"Title doesn't match content (cosine {tm.title_to_content}){hint}")

        a = by_url_answer.get(p.url)
        if a:
            gap = avg_answer - a["score"]
            if gap > 1.0:
                score += 2.0 * gap
                missing = []
                breakdown_keys = (a.get("breakdown") or {}).keys()
                if "faq_schema" not in breakdown_keys and "structured_schema" not in breakdown_keys:
                    missing.append("schema")
                if "q_headings" not in breakdown_keys:
                    missing.append("question-form headings")
                if "statistics" not in breakdown_keys:
                    missing.append("statistics")
                if "external_citations" not in breakdown_keys:
                    missing.append("external citations")
                reasons.append(
                    f"Low answer-ability ({a['score']}/10, site avg {avg_answer:.1f})"
                    + (f" — missing: {', '.join(missing[:3])}" if missing else "")
                )

        ds = float(distance_to_section[i]) if i < len(distance_to_section) else 0.0
        if ds > 0.30:
            score += 1.0 * (ds - 0.20) * 10
            reasons.append(f"Drift from section centroid {ds:.3f} — page may belong elsewhere")

        if p.url in duplicate_urls:
            n = len(duplicate_urls[p.url])
            score += 1.5 * min(3, n)
            reasons.append(f"Near-duplicate of {n} other page(s) — merge or canonicalize")

        if p.url in wrong_home_count:
            n = wrong_home_count[p.url]
            score += 1.0 * min(5, n)
            reasons.append(f"{n} paragraph(s) fit a different page better — move them or split this page")

        ext = by_url_external.get(p.url)
        if ext and ext.get("citation_density", 0) == 0 and p.word_count > 200:
            score += 0.5
            reasons.append("No outbound citations — looks unsourced to AI engines")

        if not reasons:
            continue
        rows.append(PageImprovement(
            url=p.url,
            title=p.title,
            score=round(score, 2),
            reasons=reasons,
        ))

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows[:top_n]


# --- payloads -----------------------------------------------------------


def title_mismatch_payload(rows: list[TitleMismatch], top_n: int = 50) -> list[dict]:
    return [
        {
            "url": r.url,
            "title": r.title,
            "title_to_content": r.title_to_content,
            "title_to_section_centroid": r.title_to_section_centroid,
            "suggested_keywords": r.suggested_keywords,
        }
        for r in rows[:top_n]
    ]


def wrong_home_payload(rows: list[WrongHomeParagraph]) -> list[dict]:
    return [
        {
            "source_url": r.source_url,
            "paragraph_index": r.paragraph_index,
            "paragraph_excerpt": r.paragraph_excerpt,
            "suggested_home_url": r.suggested_home_url,
            "suggested_home_title": r.suggested_home_title,
            "sim_to_suggested": r.sim_to_suggested,
            "sim_to_host_centroid": r.sim_to_host_centroid,
            "lift": r.lift,
        }
        for r in rows
    ]


def improvement_payload(rows: list[PageImprovement]) -> list[dict]:
    return [
        {"url": r.url, "title": r.title, "score": r.score, "reasons": r.reasons}
        for r in rows
    ]
