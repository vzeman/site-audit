"""Compare a competitor URL against our best-matching page for a query.

Answers the question: "Why is competitor X ranking for this query and we
aren't?" The diff is at two levels:

1. **Structural**: do they have FAQ/Article schema, more question-form
   headings, more statistics, more outbound citations? AI answer
   engines reward exactly these signals.
2. **Topical**: their paragraphs cluster into N sub-topics. For each
   sub-topic, do we have a paragraph above the similarity floor on
   our best-matching page? If not, that sub-topic is a content gap
   we should fill.

Input: a TSV file with ``query<TAB>competitor_url`` per line, optionally
prefixed by ``# section`` headers that group queries.

We re-use the project's ``HttpCache`` so the same competitor URL isn't
fetched twice across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .answerability import score_page
from .crawler import Crawler, CrawlConfig
from .extractor import extract

LOG = logging.getLogger(__name__)


@dataclass
class CompetitorComparison:
    query: str
    competitor_url: str
    competitor_title: str
    our_best_url: str
    our_best_title: str
    our_best_similarity: float
    answerability_ours: float
    answerability_theirs: float
    structural_gaps: list[dict]              # [{signal, ours, theirs, advice}]
    missing_topics: list[dict]               # [{label, sample_competitor_paragraph}]
    paragraph_count_ours: int
    paragraph_count_theirs: int
    error: Optional[str] = None


def load_competitive_pairs(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Accept tab, two-spaces, or pipe as separator.
        if "\t" in line:
            q, u = line.split("\t", 1)
        elif " | " in line:
            q, u = line.split(" | ", 1)
        elif "  " in line:
            q, u = line.split("  ", 1)
        else:
            # treat the whole line as URL with empty query (rare)
            continue
        q = q.strip()
        u = u.strip()
        if q and u:
            out.append((q, u))
    return out


def _fetch_one(url: str, http_cache, user_agent: str) -> Optional[str]:
    """Fetch a single URL via the same retry+TLS-impersonation stack."""
    cfg = CrawlConfig(
        domain=url,                # crawler will use this as the home origin
        max_pages=1,
        max_workers=1,
        respect_robots=True,
        use_cache=True,
        user_agent=user_agent,
    )
    crawler = Crawler(cfg, http_cache)
    crawler._warm_session()
    res = crawler._fetch(url)
    if res is None or "html" not in res.content_type:
        return None
    return res.body


def _structural_diff(ours_ext, theirs_ext) -> list[dict]:
    diffs: list[dict] = []
    ours_score = score_page(ours_ext)
    theirs_score = score_page(theirs_ext)
    # FAQ / structured schema
    has_faq = lambda ext: any(t in {"FAQPage", "QAPage", "Question"} for t in (ext.schema_types or []))
    has_struct = lambda ext: any(t in {"HowTo", "Article", "NewsArticle", "BlogPosting", "Product", "Recipe"} for t in (ext.schema_types or []))
    if has_faq(theirs_ext) and not has_faq(ours_ext):
        diffs.append({
            "signal": "FAQ/QA schema",
            "ours": False, "theirs": True,
            "advice": "Add FAQPage JSON-LD with the page's Q-form headings.",
        })
    elif has_struct(theirs_ext) and not has_struct(ours_ext):
        diffs.append({
            "signal": "Article/HowTo/Product schema",
            "ours": False, "theirs": True,
            "advice": "Add structured-data JSON-LD describing the page type.",
        })

    # question-form heading count
    def q_count(headings):
        n = 0
        for h in headings or []:
            t = h.lower()
            if t.endswith("?") or t.split(" ", 1)[0] in {"how", "what", "why", "when", "where", "which", "who"}:
                n += 1
        return n
    q_us = q_count(ours_ext.headings)
    q_th = q_count(theirs_ext.headings)
    if q_th >= q_us + 2:
        diffs.append({
            "signal": "Question-form headings (H2/H3 ending '?' or starting How/What/Why)",
            "ours": q_us, "theirs": q_th,
            "advice": "Reframe section headings as questions. AI answer engines retrieve question-shaped chunks.",
        })

    # statistics
    if theirs_ext.stat_count >= ours_ext.stat_count + 3:
        diffs.append({
            "signal": "Statistics / numbers with units",
            "ours": ours_ext.stat_count, "theirs": theirs_ext.stat_count,
            "advice": "Add concrete statistics with units (%, mg, hours, …). LLMs preferentially cite atomic facts.",
        })

    # external citations
    if theirs_ext.external_link_count >= ours_ext.external_link_count + 3:
        diffs.append({
            "signal": "External / authoritative citations",
            "ours": ours_ext.external_link_count, "theirs": theirs_ext.external_link_count,
            "advice": "Cite authoritative external sources (academic, .gov, .edu) so the page reads as researched.",
        })

    # tables
    if theirs_ext.table_count > ours_ext.table_count:
        diffs.append({
            "signal": "Comparison / data tables",
            "ours": ours_ext.table_count, "theirs": theirs_ext.table_count,
            "advice": "Tables are over-represented in AI Overviews. Add a comparison or specs table.",
        })

    # word count
    if theirs_ext.word_count >= 1.5 * max(ours_ext.word_count, 1) and theirs_ext.word_count >= 800:
        diffs.append({
            "signal": "Page depth (word count)",
            "ours": ours_ext.word_count, "theirs": theirs_ext.word_count,
            "advice": "Competitor goes deeper. Expand with sub-topics — see 'missing topics' below.",
        })

    return diffs


def _missing_topics(
    theirs_para_embs: np.ndarray,
    theirs_paragraphs: list[str],
    our_page_emb: np.ndarray,
    our_paragraphs_embs: np.ndarray,
    our_paragraphs: list[str],
    threshold: float = 0.70,
    n_clusters: int = 8,
    top_examples: int = 1,
) -> list[dict]:
    """Cluster the competitor's paragraphs, then find the ones we don't cover."""
    if len(theirs_para_embs) < 4:
        return []

    # decide cluster count: at most n_clusters, at most n/3, at least 2
    k = max(2, min(n_clusters, len(theirs_para_embs) // 3))
    try:
        import faiss  # type: ignore
        kmeans = faiss.Kmeans(d=theirs_para_embs.shape[1], k=k, niter=30, verbose=False, seed=42)
        kmeans.train(theirs_para_embs.astype(np.float32))
        _, labels = kmeans.index.search(theirs_para_embs.astype(np.float32), 1)
        labels = labels.flatten().astype(int)
    except Exception:
        # if k-means is unavailable, give up (still emit empty list)
        return []

    # c-TF-IDF labels
    from .cluster_labels import _compute_ctfidf
    docs = [""] * k
    for i, c in enumerate(labels):
        docs[int(c)] = (docs[int(c)] + " " + (theirs_paragraphs[i] or "")).strip()
    try:
        ctfidf, words = _compute_ctfidf(docs, ngram_range=(1, 2), min_df=1)
    except Exception:
        ctfidf, words = np.zeros((k, 0), dtype=np.float32), []

    out: list[dict] = []
    for cid in range(k):
        idxs = [i for i, c in enumerate(labels) if c == cid]
        if not idxs:
            continue
        # cluster centroid
        sub = theirs_para_embs[idxs]
        m = sub.mean(axis=0)
        norm = np.linalg.norm(m)
        centroid = m / norm if norm > 0 else m
        # do we cover this cluster?
        covered = False
        if len(our_paragraphs_embs) > 0:
            best_sim = float(np.max(our_paragraphs_embs @ centroid))
        else:
            best_sim = float(our_page_emb @ centroid)
        covered = best_sim >= threshold

        # cluster label
        label = ""
        if words:
            scores = ctfidf[cid]
            top_idx = np.argsort(-scores)[:8]
            kw = []
            seen: set[str] = set()
            for j in top_idx:
                if scores[j] <= 0:
                    continue
                w = words[int(j)]
                if any(w in s or s in w for s in seen if abs(len(w) - len(s)) < 4 and w != s):
                    continue
                seen.add(w)
                kw.append(w)
                if len(kw) >= 4:
                    break
            label = ", ".join(kw)
        if not label:
            label = f"sub-topic {cid}"

        if covered:
            continue

        # representative paragraph
        sims = sub @ centroid
        best_local = int(np.argmax(sims))
        sample_excerpt = theirs_paragraphs[idxs[best_local]][:280]

        out.append({
            "label": label,
            "competitor_paragraph_count": len(idxs),
            "best_similarity_we_have": round(best_sim, 4),
            "sample_competitor_paragraph": sample_excerpt,
        })

    out.sort(key=lambda r: r["best_similarity_we_have"])
    return out


def compare_one(
    query: str,
    competitor_url: str,
    pages,
    page_embeddings: np.ndarray,
    paragraph_records: list,
    extracted_pages: list,
    embedder,
    http_cache,
    user_agent: str,
) -> CompetitorComparison:
    # 1) find our best page for this query
    q_emb = embedder.encode([query])[0]
    sims = np.clip(page_embeddings @ q_emb, -1.0, 1.0)
    our_idx = int(np.argmax(sims))
    our_best_url = pages[our_idx].url
    our_best_title = pages[our_idx].title
    our_best_sim = float(sims[our_idx])

    # 2) fetch competitor page
    body = _fetch_one(competitor_url, http_cache, user_agent)
    if body is None:
        return CompetitorComparison(
            query=query, competitor_url=competitor_url, competitor_title="",
            our_best_url=our_best_url, our_best_title=our_best_title,
            our_best_similarity=round(our_best_sim, 4),
            answerability_ours=0, answerability_theirs=0,
            structural_gaps=[], missing_topics=[],
            paragraph_count_ours=0, paragraph_count_theirs=0,
            error="could not fetch competitor URL",
        )
    theirs_ext = extract(competitor_url, body, max_chars=8000)
    if theirs_ext is None:
        return CompetitorComparison(
            query=query, competitor_url=competitor_url, competitor_title="",
            our_best_url=our_best_url, our_best_title=our_best_title,
            our_best_similarity=round(our_best_sim, 4),
            answerability_ours=0, answerability_theirs=0,
            structural_gaps=[], missing_topics=[],
            paragraph_count_ours=0, paragraph_count_theirs=0,
            error="competitor page had no usable content",
        )

    # 3) structural diff
    ours_ext = extracted_pages[our_idx]
    structural = _structural_diff(ours_ext, theirs_ext)

    ours_ans = score_page(ours_ext).score
    theirs_ans = score_page(theirs_ext).score

    # 4) embed competitor paragraphs in OUR vector space
    theirs_paragraphs = theirs_ext.paragraphs or []
    if theirs_paragraphs:
        theirs_para_embs = embedder.encode(theirs_paragraphs, batch_size=64).astype(np.float32)
    else:
        theirs_para_embs = np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)

    # 5) gather our page's paragraphs from the existing records
    our_para_embs_list = [r[3] for r in paragraph_records if r[0] == our_idx]
    our_paras = [r[2] for r in paragraph_records if r[0] == our_idx]
    our_paras_embs = (
        np.stack(our_para_embs_list).astype(np.float32) if our_para_embs_list
        else np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
    )

    missing = _missing_topics(
        theirs_para_embs, theirs_paragraphs,
        page_embeddings[our_idx], our_paras_embs, our_paras,
    )

    return CompetitorComparison(
        query=query,
        competitor_url=competitor_url,
        competitor_title=theirs_ext.title or competitor_url,
        our_best_url=our_best_url,
        our_best_title=our_best_title,
        our_best_similarity=round(our_best_sim, 4),
        answerability_ours=round(ours_ans, 2),
        answerability_theirs=round(theirs_ans, 2),
        structural_gaps=structural,
        missing_topics=missing,
        paragraph_count_ours=len(our_paras),
        paragraph_count_theirs=len(theirs_paragraphs),
    )


def to_payload(rows: Iterable[CompetitorComparison]) -> list[dict]:
    return [
        {
            "query": r.query,
            "competitor_url": r.competitor_url,
            "competitor_title": r.competitor_title,
            "our_best_url": r.our_best_url,
            "our_best_title": r.our_best_title,
            "our_best_similarity": r.our_best_similarity,
            "answerability_ours": r.answerability_ours,
            "answerability_theirs": r.answerability_theirs,
            "structural_gaps": r.structural_gaps,
            "missing_topics": r.missing_topics,
            "paragraph_count_ours": r.paragraph_count_ours,
            "paragraph_count_theirs": r.paragraph_count_theirs,
            "error": r.error,
        }
        for r in rows
    ]
