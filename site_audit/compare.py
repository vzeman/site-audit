"""Cross-domain comparison.

The per-domain reports show absolute facts about one site. This module
joins multiple already-crawled projects into a single comparison view:

* **Leaderboard** — every metric we track, side by side, sortable. The
  "who is best at X" view.
* **Combined Semantic Scatter** — pages from every domain projected via
  *one* shared UMAP. The only way the spatial picture is meaningful;
  per-domain UMAPs live in unrelated coordinate spaces.
* **Overlapped distributions** — link-density / GEO-score / in-degree
  histograms for all domains in the same chart.

Cheap: reads the existing project payloads + the cached embeddings npz.
No re-crawl, no re-embedding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger(__name__)


# --- per-domain loading ---------------------------------------------------


@dataclass
class _Project:
    domain: str
    project_dir: Path
    metrics: dict
    pages: list[dict]
    answerability: list[dict]
    page_link_counts: list[dict]
    paragraph_density: dict
    recommendations: dict
    external_links: dict
    linkbuilding: dict
    header_analysis: dict
    structured_data: dict
    embeddings: Optional[np.ndarray]    # aligned with pages
    embedded_pages: list[dict]          # subset of pages that have an embedding


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except Exception as exc:
        LOG.warning("  %s: %s", path, exc)
        return default


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def _load_embeddings(project_dir: Path, model: str) -> dict[str, np.ndarray]:
    """url → embedding, from cache. Empty if the npz is missing."""
    path = project_dir / "cache" / f"embeddings_{_model_slug(model)}.npz"
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        urls = data["urls"]
        embs = data["embeddings"]
        return {str(urls[i]): embs[i] for i in range(len(urls))}
    except Exception as exc:
        LOG.warning("  embeddings cache for %s unreadable: %s", project_dir, exc)
        return {}


def _load_project(domain: str, projects_root: Path) -> Optional[_Project]:
    project_dir = projects_root / domain
    report = project_dir / "report"
    if not (report / "site_metrics.json").exists():
        LOG.warning("  skip %s: no report (run `site-audit run %s` first)", domain, domain)
        return None

    metrics = _load_json(report / "site_metrics.json", {})
    pages = _load_json(report / "pages.json", [])
    answerability = _load_json(report / "answerability.json", [])
    linkgraph = _load_json(report / "linkgraph.json", {})
    paragraph_density = _load_json(report / "paragraph_density.json", {})
    recommendations = _load_json(report / "recommendations.json", {})
    external_links = _load_json(report / "external_links.json", {})
    linkbuilding = _load_json(report / "linkbuilding.json", {})
    header_analysis = _load_json(report / "header_analysis.json", {})
    structured_data = _load_json(report / "structured_data.json", {})

    page_link_counts = (linkgraph.get("page_link_counts") or []) if isinstance(linkgraph, dict) else []
    model = metrics.get("model", "Alibaba-NLP/gte-multilingual-base")

    embed_lookup = _load_embeddings(project_dir, model)
    embedded_pages: list[dict] = []
    embeddings: list[np.ndarray] = []
    for p in pages:
        emb = embed_lookup.get(p.get("url"))
        if emb is None:
            continue
        embedded_pages.append(p)
        embeddings.append(emb)
    embed_array = np.stack(embeddings).astype(np.float32) if embeddings else None

    return _Project(
        domain=domain,
        project_dir=project_dir,
        metrics=metrics,
        pages=pages,
        answerability=answerability,
        page_link_counts=page_link_counts,
        paragraph_density=paragraph_density,
        recommendations=recommendations,
        external_links=external_links,
        linkbuilding=linkbuilding,
        header_analysis=header_analysis,
        structured_data=structured_data,
        embeddings=embed_array,
        embedded_pages=embedded_pages,
    )


# --- combined scatter -----------------------------------------------------


def _combined_umap(
    projects: list[_Project],
    sample_per_domain: int = 1500,
    seed: int = 42,
) -> tuple[list[dict], int]:
    """Project every domain's page embeddings via ONE UMAP. Returns the rows
    for the scatter payload + the total number of embeddings projected.
    """
    rng = np.random.default_rng(seed)
    chunks: list[tuple[_Project, np.ndarray, list[dict]]] = []
    total_dim: Optional[int] = None
    for proj in projects:
        if proj.embeddings is None or not len(proj.embeddings):
            continue
        n = len(proj.embeddings)
        if n > sample_per_domain:
            idx = np.sort(rng.choice(n, sample_per_domain, replace=False))
            sub_embs = proj.embeddings[idx]
            sub_pages = [proj.embedded_pages[i] for i in idx]
        else:
            sub_embs = proj.embeddings
            sub_pages = proj.embedded_pages
        if total_dim is None:
            total_dim = sub_embs.shape[1]
        elif sub_embs.shape[1] != total_dim:
            LOG.warning(
                "  %s embeddings have dim %d; expected %d — skipping",
                proj.domain, sub_embs.shape[1], total_dim,
            )
            continue
        chunks.append((proj, sub_embs, sub_pages))

    if not chunks:
        return [], 0

    big = np.vstack([c[1] for c in chunks])
    n_total = len(big)

    if n_total < 4:
        coords = np.zeros((n_total, 2), dtype=np.float32)
        for i in range(n_total):
            coords[i, 0] = float(i)
    else:
        import umap  # type: ignore
        n_neighbors = max(2, min(15, n_total - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        coords = reducer.fit_transform(big.astype(np.float32)).astype(np.float32)

    rows: list[dict] = []
    cursor = 0
    for proj, sub_embs, sub_pages in chunks:
        n = len(sub_embs)
        for k in range(n):
            p = sub_pages[k]
            rows.append({
                "domain": proj.domain,
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "section": p.get("section", ""),
                "x": float(coords[cursor + k, 0]),
                "y": float(coords[cursor + k, 1]),
            })
        cursor += n
    return rows, n_total


# --- leaderboard ----------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[i])


def _leaderboard_row(proj: _Project) -> dict:
    m = proj.metrics or {}
    answer_scores = [float(a.get("score", 0.0)) for a in (proj.answerability or [])]
    in_degrees = [int(r.get("in_degree", 0)) for r in (proj.page_link_counts or [])]
    out_degrees = [int(r.get("out_degree", 0)) for r in (proj.page_link_counts or [])]
    pd_summary = (proj.paragraph_density or {}).get("summary", {}) or {}
    pd_pages = (proj.paragraph_density or {}).get("per_page") or []
    pd_page_distribution = _distribution([float(r.get("links_per_100w", 0.0)) for r in pd_pages])
    rec_pri = (proj.recommendations or {}).get("by_priority", {}) or {}
    rec_cat = (proj.recommendations or {}).get("by_category", {}) or {}
    ext_summary = (proj.external_links or {}).get("citation_density_summary", {}) or {}
    lb = (proj.linkbuilding or {}).get("summary", {}) or {}
    ha = (proj.header_analysis or {}).get("summary", {}) or {}
    sd = (proj.structured_data or {}).get("summary", {}) or {}

    n_pages = len(proj.page_link_counts) or m.get("page_count") or len(proj.pages) or 0
    orphan_share = (sum(1 for r in proj.page_link_counts if r.get("in_degree") == 0) / n_pages) if n_pages else 0.0

    return {
        "domain": proj.domain,
        "pages": int(n_pages),
        # Focus / topic shape
        "site_focus_score": float(m.get("site_focus_score", 0.0)),
        "calibrated_focus": float(m.get("site_focus_score_calibrated", 0.0)),
        "site_radius": float(m.get("site_radius", 0.0)),
        "topic_dimension": float((m.get("topic_dimension") or {}).get("effective_dim") or 0.0),
        "section_coherence": float((m.get("section_coherence") or {}).get("ratio") or 0.0),
        # GEO answer-ability
        "answerability_median": _percentile(answer_scores, 0.5),
        "answerability_p90": _percentile(answer_scores, 0.9),
        "answerability_below_4_share": (
            sum(1 for v in answer_scores if v < 4.0) / len(answer_scores) if answer_scores else 0.0
        ),
        # Internal linking
        "median_in_degree": _percentile([float(v) for v in in_degrees], 0.5),
        "p90_in_degree": _percentile([float(v) for v in in_degrees], 0.9),
        "median_out_degree": _percentile([float(v) for v in out_degrees], 0.5),
        "orphan_share": orphan_share,
        # Paragraph link density
        "paragraph_density_median": float(pd_summary.get("median_page_density_per_100w", pd_page_distribution["median"])),
        "paragraph_density_p90": float(pd_summary.get("p90_page_density_per_100w", pd_page_distribution["p90"])),
        "spammy_paragraph_count": int(pd_summary.get("spammy_count", 0)),
        "zero_link_paragraph_share": float(pd_summary.get("zero_link_share", 0.0)),
        # External citation
        "citation_density_median": float(ext_summary.get("median", 0.0)),
        "schema_coverage": float(sd.get("schema_coverage", 0.0)),
        "invalid_jsonld_blocks": int(sd.get("invalid_jsonld_blocks", 0)),
        "schema_type_count": int(sd.get("schema_type_count", 0)),
        "pages_missing_schema": int(sd.get("pages_missing_schema", 0)),
        # Action plan
        "recommendations_total": int((proj.recommendations or {}).get("total", 0)),
        "rec_high": int(rec_pri.get("high", 0)),
        "rec_medium": int(rec_pri.get("medium", 0)),
        "rec_content_debt": int(rec_cat.get("content_debt", 0)),
        "rec_coverage": int(rec_cat.get("coverage", 0)),
        "rec_geo": int(rec_cat.get("geo", 0)),
        "rec_linking": int(rec_cat.get("linking", 0)),
        "rec_onpage": int(rec_cat.get("onpage", 0)),
        # Linkbuilding (raw + ratios)
        "total_links": int(lb.get("total_links", 0)),
        "internal_links": int(lb.get("internal_links", 0)),
        "external_links": int(lb.get("external_links", 0)),
        "internal_external_ratio": float(lb.get("internal_external_ratio", 0.0)),
        "descriptive_anchor_share": float(lb.get("descriptive_anchor_share", 0.0)),
        "generic_anchor_share": float(lb.get("generic_anchor_share", 0.0)),
        "empty_links": int(lb.get("empty_links", 0)),
        "empty_link_share": float(lb.get("empty_share", 0.0)),
        "image_links_no_alt": int(lb.get("image_links_no_alt", 0)),
        "image_no_alt_share": float(lb.get("image_no_alt_share_of_image_links", 0.0)),
        "distinct_anchors": int(lb.get("distinct_anchors", 0)),
        # Header health
        "headers_total": int(ha.get("total_headers", 0)),
        "pages_missing_h1": int(ha.get("pages_missing_h1", 0)),
        "pages_missing_h1_share": (
            ha.get("pages_missing_h1", 0) / ha.get("total_pages", 1)
            if ha.get("total_pages") else 0.0
        ),
        "pages_multi_h1": int(ha.get("pages_multi_h1", 0)),
        "drifty_headers": int(ha.get("drifty_headers", 0)),
        "title_h1_misaligned": int(ha.get("title_h1_misaligned", 0)),
    }


# --- distributions (overlay charts) --------------------------------------


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"values": [], "min": 0.0, "max": 0.0, "median": 0.0, "p90": 0.0}
    s = sorted(values)
    return {
        "values": [float(v) for v in values],
        "min": float(s[0]),
        "max": float(s[-1]),
        "median": _percentile(s, 0.5),
        "p90": _percentile(s, 0.9),
    }


def _distributions_for_overlay(proj: _Project) -> dict:
    return {
        "domain": proj.domain,
        "in_degree": _distribution([float(r.get("in_degree", 0)) for r in proj.page_link_counts]),
        "out_degree": _distribution([float(r.get("out_degree", 0)) for r in proj.page_link_counts]),
        "answerability": _distribution([float(a.get("score", 0.0)) for a in (proj.answerability or [])]),
        "paragraph_density_per_page": _distribution(
            [float(r.get("links_per_100w", 0.0))
             for r in ((proj.paragraph_density or {}).get("per_page") or [])]
        ),
    }


# --- public entrypoint ----------------------------------------------------


def build_payload(domains: list[str], projects_root: Path) -> dict:
    projects: list[_Project] = []
    for d in domains:
        proj = _load_project(d, projects_root)
        if proj is not None:
            projects.append(proj)

    if not projects:
        return {"domains": [], "leaderboard": [], "scatter": [], "distributions": []}

    leaderboard = [_leaderboard_row(p) for p in projects]
    distributions = [_distributions_for_overlay(p) for p in projects]

    LOG.info("  building combined UMAP across %d domains", len(projects))
    scatter_rows, n_total = _combined_umap(projects)
    LOG.info("  combined UMAP: %d points projected", n_total)

    return {
        "domains": [p.domain for p in projects],
        "leaderboard": leaderboard,
        "scatter": scatter_rows,
        "scatter_total": n_total,
        "distributions": distributions,
    }


def write_html(template_path: Path, payload: dict, out_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("__COMPARE_JSON__", json.dumps(payload, separators=(",", ":")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
