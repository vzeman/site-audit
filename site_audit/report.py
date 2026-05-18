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
import re
import shutil
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


def _slim_linkgraph_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if isinstance(out.get("anchor_relevance"), dict):
        anchor = dict(out["anchor_relevance"])
        anchor["links"] = []
        anchor["weak_links"] = (anchor.get("weak_links") or [])[:500]
        out["anchor_relevance"] = anchor
    if isinstance(out.get("contextual_link_impact"), dict):
        contextual = dict(out["contextual_link_impact"])
        contextual["links"] = []
        contextual["top_contextual_links"] = (contextual.get("top_contextual_links") or [])[:500]
        contextual["weak_context_links"] = (contextual.get("weak_context_links") or [])[:500]
        contextual["source_pages"] = [
            {**row, "strongest_outbound_links": (row.get("strongest_outbound_links") or [])[:5]}
            for row in (contextual.get("source_pages") or [])[:250]
        ]
        out["contextual_link_impact"] = contextual
    if isinstance(out.get("link_flow"), dict):
        flow = dict(out["link_flow"])
        flow["edges"] = (flow.get("edges") or [])[:2500]
        flow["nodes"] = (flow.get("nodes") or [])[:2500]
        out["link_flow"] = flow
    if isinstance(out.get("hub_bottlenecks"), dict):
        hubs = dict(out["hub_bottlenecks"])
        hubs["pages"] = []
        hubs["risks"] = (hubs.get("risks") or [])[:250]
        hubs["bridges"] = (hubs.get("bridges") or [])[:150]
        hubs["bottlenecks"] = (hubs.get("bottlenecks") or [])[:150]
        hubs["authority_hubs"] = (hubs.get("authority_hubs") or [])[:150]
        out["hub_bottlenecks"] = hubs
    if isinstance(out.get("high_demand_low_link"), dict):
        demand = dict(out["high_demand_low_link"])
        demand["pages"] = (demand.get("pages") or [])[:2000]
        out["high_demand_low_link"] = demand
    if isinstance(out.get("traffic_weighted_pagerank"), dict):
        pagerank = dict(out["traffic_weighted_pagerank"])
        pagerank["pages"] = (pagerank.get("pages") or [])[:2500]
        out["traffic_weighted_pagerank"] = pagerank
    if isinstance(out.get("page_link_counts"), list):
        out["page_link_counts"] = out["page_link_counts"][:3000]
    return out


def write_internal_linkbuilding_csv(
    path: Path,
    result: AuditResult,
    recommendations: list[dict],
    search_payload: Optional[dict] = None,
) -> list[dict]:
    page_by_url = {page.url: page for page in result.pages}
    paid_keywords = _paid_keywords(search_payload)
    rows: list[dict] = []
    seen_paragraph_anchors: set[tuple[str, str, str]] = set()
    for rec in recommendations or []:
        source_url = rec.get("source_url") or ""
        target_url = rec.get("target_url") or ""
        suggested_anchor = rec.get("suggested_anchor") or rec.get("anchor") or ""
        paid_candidate = _best_paid_anchor_candidate(
            paid_keywords,
            rec.get("paragraph_excerpt") or "",
            target_url,
            rec.get("target_title") or "",
            page_by_url.get(target_url),
        )
        anchor = paid_candidate.get("keyword") or suggested_anchor
        if not source_url or not target_url or not anchor:
            continue
        paragraph_index = str(rec.get("paragraph_index", ""))
        anchor_key = (source_url, paragraph_index, _normalize_anchor(anchor))
        if anchor_key in seen_paragraph_anchors:
            continue
        seen_paragraph_anchors.add(anchor_key)
        target_page = page_by_url.get(target_url)
        source_page = page_by_url.get(source_url)
        destination_title = rec.get("target_title") or getattr(target_page, "title", "") or target_url
        destination_description = getattr(target_page, "description", "") if target_page else ""
        rows.append({
            "url_where_to_place_link": source_url,
            "source_page_title": rec.get("source_title") or getattr(source_page, "title", "") or source_url,
            "paragraph_index": paragraph_index,
            "paragraph_excerpt": rec.get("paragraph_excerpt", ""),
            "exact_keywords_to_link": anchor,
            "original_suggested_anchor": suggested_anchor,
            "anchor_source": "paid_converting_keyword" if paid_candidate else "semantic_paragraph_match",
            "paid_keyword_candidate": paid_candidate.get("keyword", ""),
            "paid_conversions": paid_candidate.get("paid_conversions", ""),
            "paid_conversion_value": paid_candidate.get("paid_conversion_value", ""),
            "paid_cost": paid_candidate.get("paid_cost", ""),
            "destination_url": target_url,
            "link_title": destination_description or destination_title,
            "destination_title": destination_title,
            "destination_meta_description": destination_description,
            "priority": rec.get("priority", ""),
            "expected_benefit_score": rec.get("expected_benefit_score", ""),
            "fit": rec.get("fit", ""),
            "lift": rec.get("lift", ""),
            "anchor_confidence": rec.get("anchor_confidence", ""),
        })
    _write_csv(path, rows)
    return rows


def write_technical_seo_exports(output_dir: Path, technical_seo: dict) -> None:
    pages = list((technical_seo or {}).get("pages") or [])
    issues = list((technical_seo or {}).get("issues") or [])
    page_payload = {
        "summary": (technical_seo or {}).get("summary", {}),
        "pages": pages,
        "interpretation": (technical_seo or {}).get("interpretation", {}),
    }
    issue_payload = {
        "summary": (technical_seo or {}).get("summary", {}),
        "issue_counts": (technical_seo or {}).get("issue_counts", {}),
        "category_counts": (technical_seo or {}).get("category_counts", {}),
        "severity_counts": (technical_seo or {}).get("severity_counts", {}),
        "issues": issues,
        "interpretation": (technical_seo or {}).get("interpretation", {}),
    }
    _write_json(output_dir / "technical_pages.json", page_payload)
    _write_json(output_dir / "technical_issues.json", issue_payload)
    _write_csv(output_dir / "technical_pages.csv", [_csv_safe_row(row) for row in pages])
    _write_csv(output_dir / "technical_issues.csv", [_csv_safe_row(row) for row in issues])


def write_indexability_issues_csv(output_dir: Path, indexability: dict) -> None:
    issues = list((indexability or {}).get("issues") or [])
    _write_csv(output_dir / "indexability_issues.csv", [_csv_safe_row(row) for row in issues])


def write_sitemap_coverage_exports(output_dir: Path, sitemap_coverage: dict) -> None:
    _write_json(output_dir / "sitemap_coverage.json", sitemap_coverage)
    _write_csv(output_dir / "sitemap_coverage.csv", [_csv_safe_row(row) for row in (sitemap_coverage or {}).get("rows", [])])
    _write_csv(output_dir / "sitemap_coverage_issues.csv", [_csv_safe_row(row) for row in (sitemap_coverage or {}).get("issues", [])])


def write_canonical_consistency_exports(output_dir: Path, canonical_consistency: dict) -> None:
    _write_json(output_dir / "canonical_consistency.json", canonical_consistency)
    _write_csv(output_dir / "canonical_consistency.csv", [_csv_safe_row(row) for row in (canonical_consistency or {}).get("rows", [])])
    _write_csv(output_dir / "canonical_consistency_issues.csv", [_csv_safe_row(row) for row in (canonical_consistency or {}).get("issues", [])])


def _csv_safe_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, list):
            out[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = value
    return out


def _paid_keywords(search_payload: Optional[dict]) -> list[dict]:
    rows = []
    for row in (search_payload or {}).get("organic_keywords") or []:
        if row.get("provider") == "google_ads" or row.get("paid_conversions") or row.get("paid_cost"):
            if row.get("keyword"):
                rows.append(row)
    rows.sort(
        key=lambda r: (
            _safe_float(r.get("paid_conversions")),
            _safe_float(r.get("paid_conversion_value")),
            _safe_float(r.get("paid_cost")),
            _safe_float(r.get("clicks")),
        ),
        reverse=True,
    )
    return rows


def _normalize_anchor(anchor: str) -> str:
    return re.sub(r"\s+", " ", str(anchor or "").strip().lower())


def _best_paid_anchor_candidate(
    paid_keywords: list[dict],
    paragraph_excerpt: str,
    target_url: str,
    target_title: str,
    target_page,
) -> dict:
    if not paid_keywords or not paragraph_excerpt:
        return {}
    paragraph = paragraph_excerpt.lower()
    target_text = " ".join([
        target_url,
        target_title,
        getattr(target_page, "title", "") if target_page else "",
        getattr(target_page, "description", "") if target_page else "",
    ]).lower()
    for row in paid_keywords:
        keyword = str(row.get("keyword") or "").strip()
        if len(keyword) < 3:
            continue
        lower = keyword.lower()
        if lower in paragraph and _keyword_plausible_for_target(lower, target_text):
            return row
    return {}


def _keyword_plausible_for_target(keyword: str, target_text: str) -> bool:
    tokens = [t for t in keyword.replace("-", " ").split() if len(t) > 2]
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in target_text)
    return matches >= max(1, min(2, len(tokens)))


def _safe_float(value) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _copy_report_docs(output_dir: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    docs = {
        root / "docs" / "serp-paragraph-gap-analysis.md": output_dir / "serp-paragraph-gap-analysis.md",
        root / "docs" / "report-sections.md": output_dir / "report-sections.md",
    }
    for source, target in docs.items():
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


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
    cannibalization: Optional[dict] = None,
    duplicate_fragments: Optional[dict] = None,
    template_patterns: Optional[dict] = None,
    winning_paragraphs: Optional[dict] = None,
    weak_paragraphs: Optional[dict] = None,
    heading_impact: Optional[dict] = None,
    entity_coverage: Optional[dict] = None,
    information_gain: Optional[dict] = None,
    title_mismatch: Optional[list] = None,
    wrong_home: Optional[list] = None,
    page_improvement: Optional[list] = None,
    competitive: Optional[dict | list] = None,
    recommendations: Optional[dict] = None,
    paragraph_density: Optional[dict] = None,
    header_analysis: Optional[dict] = None,
    header_scatter: Optional[dict] = None,
    linkbuilding: Optional[dict] = None,
    structured_data: Optional[dict] = None,
    trust_signals: Optional[dict] = None,
    conversion_balance: Optional[dict] = None,
    metadata_quality: Optional[dict] = None,
    media_accessibility: Optional[dict] = None,
    page_types: Optional[dict] = None,
    entities: Optional[dict] = None,
    freshness: Optional[dict] = None,
    conversion: Optional[dict] = None,
    indexability: Optional[dict] = None,
    sitemap_coverage: Optional[dict] = None,
    canonical_consistency: Optional[dict] = None,
    performance: Optional[dict] = None,
    ahrefs: Optional[dict] = None,
    best_pages: Optional[dict] = None,
    performance_explainer: Optional[dict] = None,
    history_snapshot: Optional[dict] = None,
    technical_seo: Optional[dict] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_report_docs(output_dir)
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
        _write_json(output_dir / "linkgraph.json", _slim_linkgraph_payload(linkgraph))
    if external_links is not None:
        _write_json(output_dir / "external_links.json", external_links)
    if paragraph_link_recs is not None:
        _write_json(output_dir / "paragraph_link_recommendations.json", paragraph_link_recs)
        write_internal_linkbuilding_csv(output_dir / "internal_linkbuilding_recommendations.csv", result, paragraph_link_recs, ahrefs)
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
    if cannibalization is not None:
        _write_json(output_dir / "cannibalization.json", cannibalization)
    if duplicate_fragments is not None:
        _write_json(output_dir / "duplicate_fragments.json", duplicate_fragments)
    if template_patterns is not None:
        _write_json(output_dir / "template_patterns.json", template_patterns)
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
    if trust_signals is not None:
        _write_json(output_dir / "trust_signals.json", trust_signals)
    if conversion_balance is not None:
        _write_json(output_dir / "conversion_balance.json", conversion_balance)
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
        write_indexability_issues_csv(output_dir, indexability)
    if sitemap_coverage is not None:
        write_sitemap_coverage_exports(output_dir, sitemap_coverage)
    if canonical_consistency is not None:
        write_canonical_consistency_exports(output_dir, canonical_consistency)
    if performance is not None:
        _write_json(output_dir / "performance.json", performance)
    if ahrefs is not None:
        _write_json(output_dir / "search.json", ahrefs)
        _write_json(output_dir / "ahrefs.json", ahrefs)
        provider = str((ahrefs.get("meta", {}) or {}).get("provider", "")).lower()
        if provider in {"gsc", "dataforseo"}:
            _write_json(output_dir / f"{provider}.json", ahrefs)
    if best_pages is not None:
        _write_json(output_dir / "best_pages.json", best_pages)
    if performance_explainer is not None:
        _write_json(output_dir / "performance_explainer.json", performance_explainer)
    if history_snapshot is not None:
        _write_json(output_dir / "history_snapshot.json", history_snapshot)
    if technical_seo is not None:
        write_technical_seo_exports(output_dir, technical_seo)
    return {"outliers": len(outliers), "duplicates": len(result.duplicate_pairs)}
