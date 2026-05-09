"""Serialize an ``AuditResult`` (+ optional UMAP projection) to disk.

Output paths mirror the Hugo audit pipeline so the existing D3
viewer template can render against either source unchanged:

::

    out/
      <domain>/
        site_metrics.json
        section_report.json
        page_drift.csv
        outliers.csv
        duplicates.csv
        scatterplot.json
        pages.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .analyzer import AuditResult, recommend_action


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_site_metrics(path: Path, result: AuditResult, model_name: str, domain: str) -> None:
    sm = result.site_metrics
    payload = {
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
        "sections": sorted(
            [
                {
                    "section": s.name,
                    "page_count": s.metrics["count"],
                    "focus_score": s.metrics["focus_score"],
                    "radius": s.metrics["radius"],
                    "p95_distance": s.metrics["p95_distance"],
                }
                for s in result.sections.values()
            ],
            key=lambda x: x["focus_score"],
            reverse=True,
        ),
        "interpretation": {
            "site_focus_score": "Raw mean cosine of pages to site centroid. Anchored to the embedding model's anisotropy; for gte-multilingual-base lives in roughly 0.5–0.9.",
            "site_focus_score_calibrated": "(focus - p10_pairwise) / (1 - p10_pairwise) — strips the model floor. 0 = no more focused than 10% of random pairs, 1 = perfectly aligned.",
            "site_radius": "Std-dev of per-page cosine distance to the site centroid. Lower is tighter.",
            "section_coherence_ratio": "Mean intra-section similarity / mean inter-section similarity. >1.5 = URL structure matches content; ~1.0 = sections are arbitrary.",
            "topic_dimension.effective_dim": "Effective number of independent topics (PCA spectral entropy). 2-4 = laser-focused, 15-30 = broad publisher.",
        },
    }
    _write_json(path, payload)


def write_section_report(path: Path, result: AuditResult) -> None:
    out = []
    for s in result.sections.values():
        section_pages = [result.pages[i] for i in s.indices]
        out.append({
            "section": s.name,
            "page_count": s.metrics["count"],
            "focus_score": s.metrics["focus_score"],
            "radius": s.metrics["radius"],
            "p95_distance_to_section_centroid": s.metrics["p95_distance"],
            "example_titles": [p.title for p in section_pages[:5]],
        })
    out.sort(key=lambda x: x["focus_score"])
    _write_json(path, out)


def write_page_drift(path: Path, result: AuditResult) -> None:
    rows = []
    for i, p in enumerate(result.pages):
        rows.append({
            "url": p.url,
            "section": p.section,
            "title": p.title,
            "word_count": p.word_count,
            "distance_to_site_centroid": round(float(result.dist_to_site[i]), 4),
            "distance_to_section_centroid": round(float(result.dist_to_section[i]), 4),
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    _write_csv(path, rows)


def build_outlier_rows(result: AuditResult) -> list[dict]:
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


def write_outliers(path: Path, result: AuditResult) -> list[dict]:
    rows = build_outlier_rows(result)
    _write_csv(path, rows)
    return rows


def build_duplicate_rows(result: AuditResult) -> list[dict]:
    rows = []
    for i, j, sim in result.duplicate_pairs:
        a = result.pages[i]
        b = result.pages[j]
        same_section = a.section == b.section
        if sim >= 0.97:
            action = "merge (duplicate)"
        elif sim >= 0.94:
            action = "consolidate or canonicalize"
        else:
            action = "review — strong overlap"
        rows.append({
            "similarity": round(sim, 4),
            "same_section": same_section,
            "section_a": a.section,
            "url_a": a.url,
            "title_a": a.title,
            "section_b": b.section,
            "url_b": b.url,
            "title_b": b.title,
            "recommendation": action,
        })
    return rows


def write_duplicates(path: Path, result: AuditResult) -> list[dict]:
    rows = build_duplicate_rows(result)
    _write_csv(path, rows)
    return rows


def write_scatterplot(
    path: Path,
    result: AuditResult,
    coords: Optional[np.ndarray],
    cluster_labels: Optional[np.ndarray],
) -> None:
    if coords is None or cluster_labels is None:
        return
    pages_payload = []
    sm = result.site_metrics
    max_drift = float(max(
        sm["max_distance"],
        float(np.max(result.dist_to_site)) if len(result.dist_to_site) else 0,
        float(np.max(result.dist_to_section)) if len(result.dist_to_section) else 0,
        1e-9,
    ))
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
    payload = {
        "total_pages": len(result.pages),
        "num_clusters": num_clusters,
        "max_drift": max_drift,
        "site_focus_score": sm["focus_score"],
        "site_radius": sm["radius"],
        "pages": pages_payload,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))


def write_pages(path: Path, result: AuditResult) -> None:
    payload = [
        {
            "url": p.url,
            "title": p.title,
            "description": p.description,
            "section": p.section,
            "word_count": p.word_count,
            "language": p.language,
        }
        for p in result.pages
    ]
    _write_json(path, payload)


def write_clusters(path: Path, summaries) -> None:
    if summaries is None:
        return
    payload = []
    for s in summaries:
        payload.append({
            "cluster_id": s.cluster_id,
            "page_count": s.page_count,
            "cohesion": round(s.cohesion, 4),
            "site_alignment": round(s.site_alignment, 4),
            "label": ", ".join(k["keyword"] for k in s.keywords[:4]),
            "keywords": s.keywords,
            "top_pages": s.top_pages,
        })
    _write_json(path, payload)


def write_all(
    output_dir: Path,
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
    paragraph_impact: Optional[dict] = None,
    semantic_ablation: Optional[dict] = None,
    keyword_attribution: Optional[dict] = None,
    answer_blocks: Optional[dict] = None,
    freshness_impact: Optional[dict] = None,
    winning_paragraphs: Optional[dict] = None,
    weak_paragraphs: Optional[dict] = None,
    heading_impact: Optional[dict] = None,
    entity_coverage: Optional[dict] = None,
    information_gain: Optional[dict] = None,
    title_mismatch: Optional[list] = None,
    wrong_home: Optional[list] = None,
    page_improvement: Optional[list] = None,
    competitive: Optional[list] = None,
    recommendations: Optional[dict] = None,
    paragraph_density: Optional[dict] = None,
    header_analysis: Optional[dict] = None,
    header_scatter: Optional[dict] = None,
    linkbuilding: Optional[dict] = None,
    structured_data: Optional[dict] = None,
    metadata_quality: Optional[dict] = None,
    media_accessibility: Optional[dict] = None,
    page_types: Optional[dict] = None,
    entities: Optional[dict] = None,
    freshness: Optional[dict] = None,
    conversion: Optional[dict] = None,
    indexability: Optional[dict] = None,
    performance: Optional[dict] = None,
    ahrefs: Optional[dict] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_site_metrics(output_dir / "site_metrics.json", result, model_name, domain)
    write_section_report(output_dir / "section_report.json", result)
    write_page_drift(output_dir / "page_drift.csv", result)
    outliers = write_outliers(output_dir / "outliers.csv", result)
    duplicates = write_duplicates(output_dir / "duplicates.csv", result)
    write_pages(output_dir / "pages.json", result)
    write_scatterplot(output_dir / "scatterplot.json", result, coords, cluster_labels)
    if cluster_summaries:
        write_clusters(output_dir / "clusters.json", cluster_summaries)
    if coverage is not None:
        _write_json(output_dir / "keyword_coverage.json", coverage)
    if answerability is not None:
        _write_json(output_dir / "answerability.json", answerability)
    if linkgraph is not None:
        _write_json(output_dir / "linkgraph.json", linkgraph)
    if external_links is not None:
        _write_json(output_dir / "external_links.json", external_links)
    if paragraph_link_recs is not None:
        _write_json(output_dir / "paragraph_link_recommendations.json", paragraph_link_recs)
    if cluster_overlap is not None:
        _write_json(output_dir / "cluster_overlap.json", cluster_overlap)
    if paragraph_clusters is not None:
        _write_json(output_dir / "paragraph_clusters.json", paragraph_clusters)
    if paragraph_scatter is not None:
        _write_json(output_dir / "paragraph_scatter.json", paragraph_scatter)
    if paragraph_fanout is not None:
        _write_json(output_dir / "paragraph_fanout.json", paragraph_fanout)
    if paragraph_impact is not None:
        _write_json(output_dir / "paragraph_impact.json", paragraph_impact)
    if semantic_ablation is not None:
        _write_json(output_dir / "semantic_ablation.json", semantic_ablation)
    if keyword_attribution is not None:
        _write_json(output_dir / "keyword_attribution.json", keyword_attribution)
    if answer_blocks is not None:
        _write_json(output_dir / "answer_blocks.json", answer_blocks)
    if freshness_impact is not None:
        _write_json(output_dir / "freshness_impact.json", freshness_impact)
    if winning_paragraphs is not None:
        _write_json(output_dir / "winning_paragraphs.json", winning_paragraphs)
    if weak_paragraphs is not None:
        _write_json(output_dir / "weak_paragraphs.json", weak_paragraphs)
    if heading_impact is not None:
        _write_json(output_dir / "heading_impact.json", heading_impact)
    if entity_coverage is not None:
        _write_json(output_dir / "entity_coverage.json", entity_coverage)
    if information_gain is not None:
        _write_json(output_dir / "information_gain.json", information_gain)
    if title_mismatch is not None:
        _write_json(output_dir / "title_mismatch.json", title_mismatch)
    if wrong_home is not None:
        _write_json(output_dir / "wrong_home_paragraphs.json", wrong_home)
    if page_improvement is not None:
        _write_json(output_dir / "page_improvement.json", page_improvement)
    if competitive is not None:
        _write_json(output_dir / "competitive_analysis.json", competitive)
    if recommendations is not None:
        _write_json(output_dir / "recommendations.json", recommendations)
    if paragraph_density is not None:
        _write_json(output_dir / "paragraph_density.json", paragraph_density)
    if header_analysis is not None:
        _write_json(output_dir / "header_analysis.json", header_analysis)
    if header_scatter is not None:
        _write_json(output_dir / "header_scatter.json", header_scatter)
    if linkbuilding is not None:
        _write_json(output_dir / "linkbuilding.json", linkbuilding)
    if structured_data is not None:
        _write_json(output_dir / "structured_data.json", structured_data)
    if metadata_quality is not None:
        _write_json(output_dir / "metadata_quality.json", metadata_quality)
    if media_accessibility is not None:
        _write_json(output_dir / "media_accessibility.json", media_accessibility)
    if page_types is not None:
        _write_json(output_dir / "page_types.json", page_types)
    if entities is not None:
        _write_json(output_dir / "entities.json", entities)
    if freshness is not None:
        _write_json(output_dir / "freshness.json", freshness)
    if conversion is not None:
        _write_json(output_dir / "conversion.json", conversion)
    if indexability is not None:
        _write_json(output_dir / "indexability.json", indexability)
    if performance is not None:
        _write_json(output_dir / "performance.json", performance)
    if ahrefs is not None:
        _write_json(output_dir / "search.json", ahrefs)
        _write_json(output_dir / "ahrefs.json", ahrefs)
    return {"outliers": len(outliers), "duplicates": len(result.duplicate_pairs)}
