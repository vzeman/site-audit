"""Render the audit data into a self-contained HTML file.

We take ``ui/index.html`` (which is also what the server uses), inject
JSON payloads into placeholder ``<script type="application/json">`` tags,
and write the result to ``output/<domain>/index.html``. The resulting
file works offline — no fetches needed — so users can double-click it
or attach it to email.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .analyzer import AuditResult, recommend_action

_PLACEHOLDERS = {
    "__SCATTERPLOT_JSON__": "scatter",
    "__METRICS_JSON__": "metrics",
    "__SECTIONS_JSON__": "sections",
    "__OUTLIERS_JSON__": "outliers",
    "__DUPLICATES_JSON__": "duplicates",
    "__CLUSTERS_JSON__": "clusters",
    "__COVERAGE_JSON__": "coverage",
    "__ANSWERABILITY_JSON__": "answerability",
    "__LINKGRAPH_JSON__": "linkgraph",
    "__EXTERNAL_JSON__": "external",
    "__PARAGRAPH_LINKS_JSON__": "paragraph_links",
    "__CLUSTER_OVERLAP_JSON__": "cluster_overlap",
    "__PARAGRAPH_CLUSTERS_JSON__": "paragraph_clusters",
    "__PARAGRAPH_SCATTER_JSON__": "paragraph_scatter",
    "__PARAGRAPH_FANOUT_JSON__": "paragraph_fanout",
    "__TITLE_MISMATCH_JSON__": "title_mismatch",
    "__WRONG_HOME_JSON__": "wrong_home",
    "__PAGE_IMPROVEMENT_JSON__": "page_improvement",
    "__COMPETITIVE_JSON__": "competitive",
    "__RECOMMENDATIONS_JSON__": "recommendations",
    "__PARAGRAPH_DENSITY_JSON__": "paragraph_density",
}


def _scatterplot_payload(result: AuditResult, coords: Optional[np.ndarray], cluster_labels: Optional[np.ndarray]) -> Optional[dict]:
    if coords is None or cluster_labels is None:
        return None
    sm = result.site_metrics
    max_drift = float(max(
        sm["max_distance"],
        float(np.max(result.dist_to_site)) if len(result.dist_to_site) else 0,
        float(np.max(result.dist_to_section)) if len(result.dist_to_section) else 0,
        1e-9,
    ))
    pages_payload = []
    for i, p in enumerate(result.pages):
        pages_payload.append({
            "title": p.title,
            "url": p.url,
            "section": p.section,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": int(cluster_labels[i]),
            "drift_site": round(float(result.dist_to_site[i]), 4),
            "drift_section": round(float(result.dist_to_section[i]), 4),
            "drift_norm": round(float(result.dist_to_section[i]) / max_drift, 4),
            "word_count": p.word_count,
            "duplicate_of": result.duplicate_partners.get(i, []),
        })
    num_clusters = int(max(cluster_labels) + 1) if len(cluster_labels) else 0
    return {
        "total_pages": len(result.pages),
        "num_clusters": num_clusters,
        "max_drift": max_drift,
        "site_focus_score": sm["focus_score"],
        "site_radius": sm["radius"],
        "pages": pages_payload,
    }


def _metrics_payload(result: AuditResult, model_name: str, domain: str) -> dict:
    sm = result.site_metrics
    return {
        "domain": domain,
        "model": model_name,
        "page_count": sm["count"],
        "site_focus_score": sm["focus_score"],
        "site_focus_score_calibrated": result.calibrated_focus_score,
        "site_radius": sm["radius"],
        "mean_distance_to_centroid": sm["mean_distance"],
        "p95_distance_to_centroid": sm["p95_distance"],
        "max_distance_to_centroid": sm["max_distance"],
        "pairwise": result.pairwise,
        "section_coherence": result.coherence,
        "topic_dimension": result.topic_dim,
        "centroid_histogram": result.centroid_hist,
    }


def _sections_payload(result: AuditResult) -> list:
    out = []
    for s in result.sections.values():
        out.append({
            "section": s.name,
            "page_count": s.metrics["count"],
            "focus_score": s.metrics["focus_score"],
            "radius": s.metrics["radius"],
            "p95_distance_to_section_centroid": s.metrics["p95_distance"],
        })
    out.sort(key=lambda x: x["focus_score"])
    return out


def _outliers_payload(result: AuditResult) -> list:
    duplicate_set = {i for pair in result.duplicate_pairs for i in pair[:2]}
    section_size = {s.name: s.metrics["count"] for s in result.sections.values()}
    section_p95 = {s.name: s.metrics["p95_distance"] for s in result.sections.values()}

    rows = []
    for i, p in enumerate(result.pages):
        ds = float(result.dist_to_section[i])
        d_all = float(result.dist_to_site[i])
        sec_p95 = section_p95.get(p.section, 1.0)
        is_outlier = ds > sec_p95 or d_all > 0.65
        if not is_outlier and i not in duplicate_set:
            continue
        action = recommend_action(
            p,
            dist_site=d_all,
            dist_section=ds,
            section_p95=sec_p95,
            section_size=section_size.get(p.section, 0),
            has_duplicate=i in duplicate_set,
        )
        if not action:
            continue
        rows.append({
            "url": p.url,
            "section": p.section,
            "title": p.title,
            "word_count": p.word_count,
            "distance_to_site_centroid": round(d_all, 4),
            "distance_to_section_centroid": round(ds, 4),
            "section_p95_distance": round(sec_p95, 4),
            "recommendation": action,
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    return rows


def _duplicates_payload(result: AuditResult) -> list:
    rows = []
    for i, j, sim in result.duplicate_pairs:
        a = result.pages[i]
        b = result.pages[j]
        if sim >= 0.97:
            action = "merge (duplicate)"
        elif sim >= 0.94:
            action = "consolidate or canonicalize"
        else:
            action = "review — strong overlap"
        rows.append({
            "similarity": round(sim, 4),
            "same_section": a.section == b.section,
            "section_a": a.section,
            "url_a": a.url,
            "title_a": a.title,
            "section_b": b.section,
            "url_b": b.url,
            "title_b": b.title,
            "recommendation": action,
        })
    return rows


def _safe_json(payload) -> str:
    """Inline JSON safely inside a <script> tag.

    We have to escape ``</`` so the string can't terminate the surrounding
    ``</script>``. ``ensure_ascii=False`` keeps unicode legible while still
    being valid because we set the document charset to utf-8.
    """
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _clusters_payload(cluster_summaries) -> list:
    if not cluster_summaries:
        return []
    out = []
    for s in cluster_summaries:
        out.append({
            "cluster_id": s.cluster_id,
            "page_count": s.page_count,
            "cohesion": round(s.cohesion, 4),
            "site_alignment": round(s.site_alignment, 4),
            "label": ", ".join(k["keyword"] for k in s.keywords[:4]) or f"cluster {s.cluster_id}",
            "keywords": s.keywords,
            "top_pages": s.top_pages,
        })
    return out


def write_html_report(
    output_dir: Path,
    template_path: Path,
    result: AuditResult,
    model_name: str,
    domain: str,
    coords: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_summaries=None,
    coverage: Optional[list] = None,
    answerability: Optional[list] = None,
    linkgraph: Optional[dict] = None,
    external_links: Optional[dict] = None,
    paragraph_link_recs: Optional[list] = None,
    cluster_overlap: Optional[dict] = None,
    paragraph_clusters: Optional[list] = None,
    paragraph_scatter: Optional[dict] = None,
    paragraph_fanout: Optional[list] = None,
    title_mismatch: Optional[list] = None,
    wrong_home: Optional[list] = None,
    page_improvement: Optional[list] = None,
    competitive: Optional[list] = None,
    recommendations: Optional[dict] = None,
    paragraph_density: Optional[dict] = None,
) -> Path:
    template = template_path.read_text(encoding="utf-8")

    payloads = {
        "scatter": _scatterplot_payload(result, coords, cluster_labels) or {},
        "metrics": _metrics_payload(result, model_name, domain),
        "sections": _sections_payload(result),
        "outliers": _outliers_payload(result),
        "duplicates": _duplicates_payload(result),
        "clusters": _clusters_payload(cluster_summaries),
        "coverage": coverage or [],
        "answerability": answerability or [],
        "linkgraph": linkgraph or {},
        "external": external_links or {},
        "paragraph_links": paragraph_link_recs or [],
        "cluster_overlap": cluster_overlap or {},
        "paragraph_clusters": paragraph_clusters or [],
        "paragraph_scatter": paragraph_scatter or {},
        "paragraph_fanout": paragraph_fanout or [],
        "title_mismatch": title_mismatch or [],
        "wrong_home": wrong_home or [],
        "page_improvement": page_improvement or [],
        "competitive": competitive or [],
        "recommendations": recommendations or {},
        "paragraph_density": paragraph_density or {},
    }

    rendered = template
    for placeholder, key in _PLACEHOLDERS.items():
        rendered = rendered.replace(placeholder, _safe_json(payloads[key]))

    out_path = Path(output_dir) / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path
