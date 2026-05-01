"""Linkbuilding overview — site-level link health + anchor quality + anchor scatter.

Pulls together every link signal we already collect (link graph, per-page
in/out degrees, paragraph density, link-quality counters from the
extractor) into one coherent "how is this site doing on internal/external
linking" view, plus two new things:

* **Anchor quality audit** — what share of links have descriptive anchor
  text, what share rely only on a ``title``/``aria-label`` attribute,
  what share are image-only with no ``alt``, what share are *empty*
  (no anchor signal at all). Empty/image-no-alt links are accessibility
  failures and waste the SEO juice their target would otherwise get.

* **Anchor-text scatter** — embed every distinct anchor phrase used on
  the site, project to 2D. Anchor clusters reveal what the site
  *visually claims to link to about*; an anchor sitting far from the
  paragraph cloud is a generic ("click here", numeric pagination) or
  off-topic anchor.

No new crawling, no extra HTTP. The cost is one batched embedder call
for the unique anchor texts (capped to ``anchor_sample_cap``).
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

LOG = logging.getLogger(__name__)


_GENERIC_ANCHORS = {
    "click here", "click", "here", "more", "read more", "learn more",
    "details", "see more", "view more", "view all", "full story",
    "continue reading", "go", "link", "this", "this page", "this article",
    "next", "previous", "next page", "prev", "back", "open", "open link",
    "buy", "buy now", "shop", "shop now", "go to", "view", "see",
    "download", "watch", "play",
}
_NUMERIC_RE = re.compile(r"^\s*\d+\s*$")


def _is_generic(anchor: str) -> bool:
    a = (anchor or "").strip().lower()
    if not a:
        return False
    if a in _GENERIC_ANCHORS:
        return True
    if _NUMERIC_RE.match(a):
        return True
    if len(a) <= 2:
        return True
    return False


def _is_descriptive(anchor: str) -> bool:
    a = (anchor or "").strip()
    return bool(a) and len(a) >= 12 and not _is_generic(a)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[i])


def analyse(
    pages,
    extracted_pages,
    linkgraph_payload: dict | None,
    paragraph_density_payload: dict | None = None,
    embedder=None,
    anchor_sample_cap: int = 600,
    seed: int = 42,
) -> dict:
    """Build the linkbuilding payload (summary + audit + scatter)."""
    n_pages = len(pages)

    # --- 1. Aggregate per-page link-quality counters --------------------
    totals: Counter = Counter()
    pages_with_zero_internal_out = 0
    pages_with_zero_external_out = 0
    pages_with_only_image_links = 0
    for ext in extracted_pages:
        lq = getattr(ext, "link_quality", {}) or {}
        if not lq:
            continue
        for k, v in lq.items():
            totals[k] += int(v)
        if int(lq.get("internal", 0)) == 0:
            pages_with_zero_internal_out += 1
        if int(lq.get("external", 0)) == 0:
            pages_with_zero_external_out += 1
        if int(lq.get("total", 0)) > 0 and int(lq.get("has_text", 0)) == 0:
            pages_with_only_image_links += 1

    total_links = int(totals.get("total", 0))

    # --- 2. Site-level anchor classification -----------------------------
    anchor_counts: Counter = Counter()
    anchor_internal: dict[str, int] = defaultdict(int)
    anchor_external: dict[str, int] = defaultdict(int)
    generic_count = 0
    descriptive_count = 0
    for ext in extracted_pages:
        for r in (getattr(ext, "link_audit_rows", []) or []):
            a = (r.get("anchor") or "").strip()
            if not a:
                continue
            anchor_counts[a] += 1
            if r.get("is_internal"):
                anchor_internal[a] += 1
            else:
                anchor_external[a] += 1
            if _is_generic(a):
                generic_count += 1
            elif _is_descriptive(a):
                descriptive_count += 1

    distinct_anchors = len(anchor_counts)
    total_anchored = sum(anchor_counts.values())
    generic_share = (generic_count / total_anchored) if total_anchored else 0.0
    descriptive_share = (descriptive_count / total_anchored) if total_anchored else 0.0

    # --- 3. Pull in link graph for in/out-degree distribution ----------
    page_link_counts = (linkgraph_payload or {}).get("page_link_counts") or []
    in_degrees = [int(r.get("in_degree", 0)) for r in page_link_counts]
    out_degrees = [int(r.get("out_degree", 0)) for r in page_link_counts]

    summary = {
        "pages": n_pages,
        "total_links": total_links,
        "internal_links": int(totals.get("internal", 0)),
        "external_links": int(totals.get("external", 0)),
        "internal_external_ratio": round(
            int(totals.get("internal", 0)) / max(1, int(totals.get("external", 1))), 2
        ),
        "links_with_text": int(totals.get("has_text", 0)),
        "links_text_share": round(int(totals.get("has_text", 0)) / max(1, total_links), 4),
        "image_only_links": int(totals.get("image_only", 0)),
        "image_only_share": round(int(totals.get("image_only", 0)) / max(1, total_links), 4),
        "image_links_no_alt": int(totals.get("image_no_alt", 0)),
        "image_no_alt_share_of_image_links": round(
            int(totals.get("image_no_alt", 0)) / max(1, int(totals.get("image_only", 1))), 4
        ),
        "empty_links": int(totals.get("empty_link", 0)),
        "empty_share": round(int(totals.get("empty_link", 0)) / max(1, total_links), 4),
        "links_with_title_attr": int(totals.get("has_title", 0)),
        "title_attr_share": round(int(totals.get("has_title", 0)) / max(1, total_links), 4),
        "distinct_anchors": distinct_anchors,
        "generic_anchor_share": round(generic_share, 4),
        "descriptive_anchor_share": round(descriptive_share, 4),
        "pages_with_zero_internal_out": pages_with_zero_internal_out,
        "pages_with_zero_external_out": pages_with_zero_external_out,
        "pages_only_image_links": pages_with_only_image_links,
        "median_in_degree": _percentile([float(v) for v in in_degrees], 0.5),
        "p90_in_degree": _percentile([float(v) for v in in_degrees], 0.9),
        "median_out_degree": _percentile([float(v) for v in out_degrees], 0.5),
        "p90_out_degree": _percentile([float(v) for v in out_degrees], 0.9),
        "mean_in_degree": round(float(np.mean(in_degrees)), 2) if in_degrees else 0.0,
        "mean_out_degree": round(float(np.mean(out_degrees)), 2) if out_degrees else 0.0,
        "orphan_pages": int((linkgraph_payload or {}).get("orphan_count", 0)),
        "dead_end_pages": int((linkgraph_payload or {}).get("dead_end_count", 0)),
    }
    if paragraph_density_payload and isinstance(paragraph_density_payload.get("summary"), dict):
        ps = paragraph_density_payload["summary"]
        summary["median_paragraph_link_density_per_100w"] = float(ps.get("median_density_per_100w", 0.0))
        summary["spammy_paragraph_count"] = int(ps.get("spammy_count", 0))

    # --- 4. Top anchors by frequency ------------------------------------
    top_internal = sorted(
        ({"anchor": a, "count": c, "is_generic": _is_generic(a)}
         for a, c in anchor_internal.items() if c > 0),
        key=lambda x: x["count"], reverse=True,
    )[:60]
    top_external = sorted(
        ({"anchor": a, "count": c, "is_generic": _is_generic(a)}
         for a, c in anchor_external.items() if c > 0),
        key=lambda x: x["count"], reverse=True,
    )[:60]
    top_generic = sorted(
        ({"anchor": a, "count": anchor_counts[a]}
         for a in anchor_counts if _is_generic(a)),
        key=lambda x: x["count"], reverse=True,
    )[:30]

    # --- 5. Anchor-text scatter (embed top-N anchors, UMAP) -------------
    scatter_payload: dict = {"total_distinct_anchors": distinct_anchors, "shown": 0, "anchors": []}
    if embedder is not None and anchor_counts:
        # take the top-N most-frequent anchors so the scatter is dominated
        # by the editorial voice, not long-tail noise
        top_for_scatter = [a for a, _ in anchor_counts.most_common(anchor_sample_cap)]
        if top_for_scatter:
            embs = embedder.encode(top_for_scatter, batch_size=256, show_progress=False).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            if len(embs) >= 5:
                import umap  # type: ignore
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=max(2, min(15, len(embs) - 1)),
                    min_dist=0.1,
                    metric="cosine",
                    random_state=seed,
                )
                coords = reducer.fit_transform(embs).astype(np.float32)
            else:
                coords = np.zeros((len(embs), 2), dtype=np.float32)
                for i in range(len(embs)):
                    coords[i, 0] = float(i)
            scatter_payload = {
                "total_distinct_anchors": distinct_anchors,
                "shown": len(top_for_scatter),
                "anchors": [
                    {
                        "anchor": a,
                        "count": int(anchor_counts[a]),
                        "internal": int(anchor_internal.get(a, 0)),
                        "external": int(anchor_external.get(a, 0)),
                        "is_generic": _is_generic(a),
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                    }
                    for i, a in enumerate(top_for_scatter)
                ],
            }

    return {
        "summary": summary,
        "top_internal_anchors": top_internal,
        "top_external_anchors": top_external,
        "top_generic_anchors": top_generic,
        "scatter": scatter_payload,
    }
