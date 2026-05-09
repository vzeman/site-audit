"""Header (H1–H6) audit + header-content alignment.

Three things in one module:

* **Structural validation** — missing H1, multiple H1s, skipped levels
  (e.g. H1 → H3), deep pages with no H2/H3, empty headers. SEO-101
  problems that templates often introduce silently.
* **Header-content alignment** via embeddings — embed every header,
  compare to its parent page's paragraph centroid. Headers far from
  their page's centre describe content the page doesn't actually have
  (or worse, click-bait). Title↔H1 cosine catches the same problem
  on the most visible header.
* **Header keyword frequency** — what terms dominate the H1 / H2 / H3
  layer across the whole site. Tells the editor what the site
  *visually* claims to be about (the words a skim-reader sees), and
  surfaces over-used boilerplate ("Contact us", "Read more").

Embeddings for headers are computed in one batched call and aligned
with the existing page-level paragraph centroids — no extra crawling
or per-page loops in Python.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

LOG = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']{1,}")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "our", "your",
    "their", "you", "we", "they", "i", "us", "them", "if", "so", "than",
    "then", "do", "does", "did", "have", "has", "had", "will", "would",
    "could", "should", "can", "may", "might", "shall", "into", "out", "up",
    "down", "very", "more", "most", "some", "any", "no", "not", "only",
    "also", "other", "such", "about", "over", "under", "all", "more",
    "what", "when", "why", "how", "who", "which",
}


# --- public payload types -----------------------------------------------


@dataclass
class _HeaderRow:
    page_index: int
    page_url: str
    page_title: str
    level: int
    order: int
    text: str
    cosine_to_page: float          # cosine(header_emb, page_paragraph_centroid)


# --- helpers -------------------------------------------------------------


def _normalised_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _meaningful_tokens(text: str) -> list[str]:
    return [t for t in _normalised_tokens(text) if t not in _STOPWORDS and len(t) > 1]


def _level_skips(levels: list[int]) -> int:
    """Count adjacent transitions that skip a level (e.g. H1 → H3)."""
    skips = 0
    last = None
    for lv in levels:
        if last is not None and lv > last + 1:
            skips += 1
        last = lv
    return skips


# --- embeddings ----------------------------------------------------------


def _gather_headers(extracted_pages, pages) -> list[tuple[int, dict]]:
    """Flatten every header on every kept page. Returns [(page_index, header_dict)]."""
    out: list[tuple[int, dict]] = []
    for i, ext in enumerate(extracted_pages):
        for h in (getattr(ext, "headers_rich", []) or []):
            text = (h.get("text") or "").strip()
            if not text:
                continue
            out.append((i, {**h, "text": text}))
    return out


def _page_paragraph_centroids(
    n_pages: int,
    paragraph_records: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-page mean of paragraph embeddings + a boolean mask of "has paragraphs"."""
    if not paragraph_records:
        return np.zeros((n_pages, 0), dtype=np.float32), np.zeros(n_pages, dtype=bool)
    dim = paragraph_records[0][3].shape[0]
    sums = np.zeros((n_pages, dim), dtype=np.float32)
    counts = np.zeros(n_pages, dtype=np.int64)
    for page_i, _, _, emb in paragraph_records:
        sums[page_i] += emb.astype(np.float32)
        counts[page_i] += 1
    mask = counts > 0
    centroids = np.zeros_like(sums)
    centroids[mask] = sums[mask] / counts[mask, None]
    # normalise so cosine == dot
    norms = np.linalg.norm(centroids[mask], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids[mask] = centroids[mask] / norms
    return centroids, mask


def _encode_texts_dedup(
    texts: list[str],
    embedder,
    *,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """Embed unique strings once, then expand embeddings back to input order."""
    unique_texts: list[str] = []
    unique_index: dict[str, int] = {}
    positions: list[int] = []
    for text in texts:
        index = unique_index.get(text)
        if index is None:
            index = len(unique_texts)
            unique_index[text] = index
            unique_texts.append(text)
        positions.append(index)
    unique_embeddings = embedder.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress=show_progress,
    ).astype(np.float32)
    return unique_embeddings[np.array(positions, dtype=np.int64)]


# --- main entrypoint -----------------------------------------------------


def analyse(
    pages,
    extracted_pages,
    page_embeddings: np.ndarray,
    paragraph_records: list,
    embedder=None,
    *,
    title_h1_misalign_threshold: float = 0.6,
    header_drift_threshold: float = 0.65,  # 1 - cosine threshold; below this cosine ⇒ flag
) -> dict:
    """Embed every header, compare to its page's paragraph centroid, summarise."""
    n_pages = len(pages)

    # 1) Per-page structural problems (cheap, no embeddings needed).
    structural: list[dict] = []
    for i, ext in enumerate(extracted_pages):
        rich = getattr(ext, "headers_rich", []) or []
        levels = [h["level"] for h in rich]
        h1_count = int(getattr(ext, "h1_count", 0))
        problems: list[str] = []
        if h1_count == 0:
            problems.append("missing H1")
        elif h1_count > 1:
            problems.append(f"{h1_count} H1 tags (should be exactly 1)")
        skips = _level_skips(levels)
        if skips:
            problems.append(f"{skips} skipped header level{'s' if skips > 1 else ''}")
        if pages[i].word_count >= 500 and not any(lv in (2, 3) for lv in levels):
            problems.append("long page with no H2/H3 sub-sections")
        if problems:
            structural.append({
                "url": pages[i].url,
                "title": pages[i].title,
                "h1_count": h1_count,
                "h1": getattr(ext, "h1", ""),
                "header_count": len(rich),
                "levels": levels,
                "problems": problems,
            })

    # 2) Embed every header in one batched call. Skip if no embedder.
    flat = _gather_headers(extracted_pages, pages)
    header_embeddings: Optional[np.ndarray] = None
    if flat and embedder is not None:
        texts = [h["text"] for _, h in flat]
        unique_count = len(set(texts))
        LOG.info(
            "  header analysis: embedding %d headers (%d unique) across %d pages",
            len(texts), unique_count, n_pages,
        )
        header_embeddings = _encode_texts_dedup(
            texts,
            embedder,
            batch_size=256,
            show_progress=False,
        )
        # normalise so dot == cosine
        norms = np.linalg.norm(header_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        header_embeddings = header_embeddings / norms

    # 3) Build per-page paragraph centroids — header drift is measured
    #    against the actual content the page has, not the title alone.
    page_centroids, has_paras = _page_paragraph_centroids(n_pages, paragraph_records)

    # 4) Per-header rows + drift to host-page paragraph centroid.
    header_rows: list[_HeaderRow] = []
    if header_embeddings is not None:
        for k, (page_i, h) in enumerate(flat):
            cosine = 0.0
            if has_paras[page_i] and page_centroids.shape[1] > 0:
                cosine = float(header_embeddings[k] @ page_centroids[page_i])
            header_rows.append(_HeaderRow(
                page_index=page_i,
                page_url=pages[page_i].url,
                page_title=pages[page_i].title,
                level=int(h["level"]),
                order=int(h["order"]),
                text=h["text"],
                cosine_to_page=round(cosine, 4),
            ))

    # 5) Title↔H1 cosine. Re-uses page embedding for the title proxy: page
    #    embeddings include title + description + body, so we compare H1
    #    against that. Tighter implementations could embed the title
    #    standalone — that doubles the embedding cost for marginal gain.
    title_h1_issues: list[dict] = []
    if header_embeddings is not None and len(page_embeddings) == n_pages:
        # normalise the page embeddings
        pe_norms = np.linalg.norm(page_embeddings, axis=1, keepdims=True)
        pe_norms[pe_norms == 0] = 1.0
        page_embs_norm = page_embeddings / pe_norms
        # We need the position of each page's *first* H1 in `flat`
        first_h1_for_page: dict[int, int] = {}
        for k, (page_i, h) in enumerate(flat):
            if h["level"] == 1 and page_i not in first_h1_for_page:
                first_h1_for_page[page_i] = k
        for page_i, k in first_h1_for_page.items():
            cosine = float(header_embeddings[k] @ page_embs_norm[page_i])
            if cosine < title_h1_misalign_threshold:
                title_h1_issues.append({
                    "url": pages[page_i].url,
                    "title": pages[page_i].title,
                    "h1": flat[k][1]["text"],
                    "title_to_h1_cosine": round(cosine, 4),
                })
        title_h1_issues.sort(key=lambda r: r["title_to_h1_cosine"])

    # 6) Drifty headers: cosine(header, page_centroid) below threshold.
    drifty_headers = sorted(
        [r for r in header_rows if r.cosine_to_page < header_drift_threshold and has_paras[r.page_index]],
        key=lambda r: r.cosine_to_page,
    )

    # 7) Header keyword frequency by level.
    by_level_counter: dict[int, Counter] = defaultdict(Counter)
    by_level_doc_count: dict[int, Counter] = defaultdict(Counter)
    for page_i, h in flat:
        toks = _meaningful_tokens(h["text"])
        by_level_counter[h["level"]].update(toks)
        for t in set(toks):
            by_level_doc_count[h["level"]][t] += 1

    keyword_freq: dict[str, list[dict]] = {}
    for lv in sorted(by_level_counter.keys()):
        c = by_level_counter[lv]
        d = by_level_doc_count[lv]
        # sort by total count, but break ties by document spread
        items = sorted(c.items(), key=lambda kv: (kv[1], d[kv[0]]), reverse=True)
        keyword_freq[f"h{lv}"] = [
            {"keyword": k, "count": int(v), "documents": int(d[k])}
            for k, v in items[:25]
        ]

    # 8) Summary stats
    pages_with_h1 = sum(1 for ext in extracted_pages if getattr(ext, "h1_count", 0) >= 1)
    pages_missing_h1 = n_pages - pages_with_h1
    pages_multi_h1 = sum(1 for ext in extracted_pages if getattr(ext, "h1_count", 0) > 1)
    by_level_total = Counter(h["level"] for _, h in flat)

    drift_values = [r.cosine_to_page for r in header_rows
                    if has_paras[r.page_index] and page_centroids.shape[1] > 0]

    summary = {
        "total_pages": n_pages,
        "total_headers": len(flat),
        "pages_with_h1": pages_with_h1,
        "pages_missing_h1": pages_missing_h1,
        "pages_multi_h1": pages_multi_h1,
        "by_level": {f"h{lv}": int(by_level_total.get(lv, 0)) for lv in (1, 2, 3, 4, 5, 6)},
        "structural_issues": len(structural),
        "title_h1_misaligned": len(title_h1_issues),
        "drifty_headers": len(drifty_headers),
        "median_header_cosine": float(np.median(drift_values)) if drift_values else 0.0,
        "p10_header_cosine": float(np.percentile(drift_values, 10)) if drift_values else 0.0,
    }

    # Per-page summary used by the recommendations module + UI table
    per_page: list[dict] = []
    headers_by_page: dict[int, list[_HeaderRow]] = defaultdict(list)
    for r in header_rows:
        headers_by_page[r.page_index].append(r)
    for i, ext in enumerate(extracted_pages):
        rich = getattr(ext, "headers_rich", []) or []
        if not rich:
            continue
        rs = headers_by_page.get(i, [])
        levels = [h["level"] for h in rich]
        per_page.append({
            "url": pages[i].url,
            "title": pages[i].title,
            "h1_count": int(getattr(ext, "h1_count", 0)),
            "h1": getattr(ext, "h1", ""),
            "header_count": len(rich),
            "by_level": {f"h{lv}": int(sum(1 for l in levels if l == lv)) for lv in (1, 2, 3, 4, 5, 6)},
            "level_skips": _level_skips(levels),
            "median_header_cosine": float(np.median([r.cosine_to_page for r in rs])) if rs else None,
            "min_header_cosine": float(min((r.cosine_to_page for r in rs), default=0.0)) if rs else None,
        })

    return {
        "summary": summary,
        "structural_issues": structural[:200],
        "title_h1_misaligned": title_h1_issues[:80],
        "drifty_headers": [
            {
                "url": r.page_url,
                "title": r.page_title,
                "level": r.level,
                "order": r.order,
                "text": r.text,
                "cosine_to_page": r.cosine_to_page,
            }
            for r in drifty_headers[:120]
        ],
        "keyword_frequency": keyword_freq,
        "per_page": per_page,
    }


def headers_for_scatter(
    pages,
    extracted_pages,
    paragraph_records: list,
    embedder=None,
    sample_cap: int = 4000,
    seed: int = 42,
) -> dict:
    """Project headers via UMAP using the paragraph centroid space.

    Uses the existing paragraph embeddings as anchors so headers land in
    the same coordinate system as the paragraph scatter — drift is
    visually obvious. Headers are sub-sampled when there are too many.
    """
    flat = _gather_headers(extracted_pages, pages)
    if not flat or embedder is None:
        return {"total_headers": 0, "shown": 0, "headers": []}

    rng = np.random.default_rng(seed)
    n = len(flat)
    if n > sample_cap:
        idx = np.sort(rng.choice(n, sample_cap, replace=False))
        flat = [flat[i] for i in idx]
    texts = [h["text"] for _, h in flat]
    embs = _encode_texts_dedup(texts, embedder, batch_size=256, show_progress=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    # Combine with a sample of paragraph embeddings so the UMAP shape is
    # influenced by the content space (otherwise headers cluster purely
    # by their own corpus). Sample at most sample_cap paragraphs.
    if paragraph_records:
        m = len(paragraph_records)
        para_idx = np.sort(rng.choice(m, min(m, sample_cap), replace=False))
        para_embs = np.stack([paragraph_records[i][3] for i in para_idx]).astype(np.float32)
        pn = np.linalg.norm(para_embs, axis=1, keepdims=True)
        pn[pn == 0] = 1.0
        para_embs = para_embs / pn
        joint = np.vstack([embs, para_embs])
    else:
        joint = embs

    if len(joint) < 5:
        # Stub coordinates if too few points to UMAP safely.
        coords = np.zeros((len(flat), 2), dtype=np.float32)
        for i in range(len(flat)):
            coords[i, 0] = float(i)
        return {
            "total_headers": n,
            "shown": len(flat),
            "headers": [
                {
                    "url": pages[pi].url,
                    "title": pages[pi].title,
                    "level": int(h["level"]),
                    "order": int(h["order"]),
                    "text": h["text"],
                    "x": float(coords[k, 0]),
                    "y": float(coords[k, 1]),
                }
                for k, (pi, h) in enumerate(flat)
            ],
        }

    import umap  # type: ignore
    n_neighbors = max(2, min(15, len(joint) - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
        metric="cosine", random_state=seed,
    )
    coords_all = reducer.fit_transform(joint).astype(np.float32)
    coords = coords_all[: len(embs)]

    return {
        "total_headers": n,
        "shown": len(flat),
        "headers": [
            {
                "url": pages[pi].url,
                "title": pages[pi].title,
                "level": int(h["level"]),
                "text": h["text"],
                "order": int(h["order"]),
                "x": float(coords[k, 0]),
                "y": float(coords[k, 1]),
            }
            for k, (pi, h) in enumerate(flat)
        ],
    }
