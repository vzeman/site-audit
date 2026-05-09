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
import hashlib
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .ahrefs import load_semantic_cache

LOG = logging.getLogger(__name__)

COMPARISON_PACKAGE_NAME = "comparison-package.zip"


# --- per-domain loading ---------------------------------------------------


@dataclass
class _Project:
    domain: str
    project_dir: Path
    metrics: dict
    pages: list[dict]
    answerability: list[dict]
    answer_blocks: dict
    cannibalization: dict
    duplicate_fragments: dict
    template_patterns: dict
    page_link_counts: list[dict]
    linkgraph: dict
    link_flow: dict
    paragraph_density: dict
    recommendations: dict
    external_links: dict
    linkbuilding: dict
    header_analysis: dict
    structured_data: dict
    trust_signals: dict
    conversion_balance: dict
    metadata_quality: dict
    media_accessibility: dict
    page_types: dict
    entities: dict
    entity_coverage: dict
    information_gain: dict
    freshness: dict
    freshness_impact: dict
    conversion: dict
    indexability: dict
    performance: dict
    ahrefs: dict
    ahrefs_semantic_rows: list[dict]
    ahrefs_semantic_embeddings: Optional[np.ndarray]
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
    answer_blocks = _load_json(report / "answer_blocks.json", {})
    cannibalization = _load_json(report / "cannibalization.json", {})
    duplicate_fragments = _load_json(report / "duplicate_fragments.json", {})
    template_patterns = _load_json(report / "template_patterns.json", {})
    linkgraph = _load_json(report / "linkgraph.json", {})
    paragraph_density = _load_json(report / "paragraph_density.json", {})
    recommendations = _load_json(report / "recommendations.json", {})
    external_links = _load_json(report / "external_links.json", {})
    linkbuilding = _load_json(report / "linkbuilding.json", {})
    header_analysis = _load_json(report / "header_analysis.json", {})
    structured_data = _load_json(report / "structured_data.json", {})
    trust_signals = _load_json(report / "trust_signals.json", {})
    conversion_balance = _load_json(report / "conversion_balance.json", {})
    metadata_quality = _load_json(report / "metadata_quality.json", {})
    media_accessibility = _load_json(report / "media_accessibility.json", {})
    page_types = _load_json(report / "page_types.json", {})
    entities = _load_json(report / "entities.json", {})
    entity_coverage = _load_json(report / "entity_coverage.json", {})
    information_gain = _load_json(report / "information_gain.json", {})
    freshness = _load_json(report / "freshness.json", {})
    freshness_impact = _load_json(report / "freshness_impact.json", {})
    conversion = _load_json(report / "conversion.json", {})
    indexability = _load_json(report / "indexability.json", {})
    performance = _load_json(report / "performance.json", {})
    ahrefs = _load_json(report / "ahrefs.json", {})

    page_link_counts = (linkgraph.get("page_link_counts") or []) if isinstance(linkgraph, dict) else []
    link_flow = (linkgraph.get("link_flow") or {}) if isinstance(linkgraph, dict) else {}
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
    ahrefs_semantic_rows, ahrefs_semantic_embeddings = load_semantic_cache(project_dir, model)

    return _Project(
        domain=domain,
        project_dir=project_dir,
        metrics=metrics,
        pages=pages,
        answerability=answerability,
        answer_blocks=answer_blocks,
        cannibalization=cannibalization,
        duplicate_fragments=duplicate_fragments,
        template_patterns=template_patterns,
        page_link_counts=page_link_counts,
        linkgraph=linkgraph if isinstance(linkgraph, dict) else {},
        link_flow=link_flow,
        paragraph_density=paragraph_density,
        recommendations=recommendations,
        external_links=external_links,
        linkbuilding=linkbuilding,
        header_analysis=header_analysis,
        structured_data=structured_data,
        trust_signals=trust_signals,
        conversion_balance=conversion_balance,
        metadata_quality=metadata_quality,
        media_accessibility=media_accessibility,
        page_types=page_types,
        entities=entities,
        entity_coverage=entity_coverage,
        information_gain=information_gain,
        freshness=freshness,
        freshness_impact=freshness_impact,
        conversion=conversion,
        indexability=indexability,
        performance=performance,
        ahrefs=ahrefs,
        ahrefs_semantic_rows=ahrefs_semantic_rows,
        ahrefs_semantic_embeddings=ahrefs_semantic_embeddings,
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
        traffic_by_url = _ahrefs_page_traffic_lookup(proj)
        freshness_by_url = _freshness_lookup(proj)
        n = len(sub_embs)
        for k in range(n):
            p = sub_pages[k]
            ah = traffic_by_url.get(p.get("url", ""), {})
            freshness = freshness_by_url.get(p.get("url", ""), {})
            rows.append({
                "domain": proj.domain,
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "section": p.get("section", ""),
                "x": float(coords[cursor + k, 0]),
                "y": float(coords[cursor + k, 1]),
                "traffic": int(ah.get("traffic", 0)),
                "keywords": int(ah.get("keywords", 0)),
                "top_keyword": ah.get("top_keyword", ""),
                "freshness_bucket": freshness.get("bucket", "unknown"),
                "freshness_age_days": freshness.get("age_days"),
                "freshness_date": freshness.get("date", ""),
                "freshness_issues": freshness.get("issues", []),
            })
        cursor += n
    return rows, n_total


def _combined_ahrefs_semantic_umap(
    projects: list[_Project],
    sample_per_domain: int = 1800,
    seed: int = 42,
    include_types: Optional[set[str]] = None,
) -> tuple[list[dict], int]:
    rng = np.random.default_rng(seed)
    chunks: list[tuple[_Project, np.ndarray, list[dict]]] = []
    dim: Optional[int] = None
    for proj in projects:
        rows = proj.ahrefs_semantic_rows or []
        embs = proj.ahrefs_semantic_embeddings
        if embs is None or not rows or len(rows) != len(embs):
            continue
        if include_types is not None:
            idx_keep = [i for i, row in enumerate(rows) if row.get("type") in include_types]
            if not idx_keep:
                continue
            rows = [rows[i] for i in idx_keep]
            embs = embs[np.array(idx_keep, dtype=np.int64)]
        n = len(rows)
        if n > sample_per_domain:
            idx = np.sort(rng.choice(n, sample_per_domain, replace=False))
            sub_embs = embs[idx]
            sub_rows = [rows[i] for i in idx]
        else:
            sub_embs = embs
            sub_rows = rows
        if dim is None:
            dim = sub_embs.shape[1]
        elif sub_embs.shape[1] != dim:
            LOG.warning("  %s Ahrefs semantic vectors have incompatible dimensions", proj.domain)
            continue
        chunks.append((proj, sub_embs, sub_rows))

    if not chunks:
        return [], 0

    big = np.vstack([chunk[1] for chunk in chunks]).astype(np.float32)
    if len(big) < 5:
        coords = np.zeros((len(big), 2), dtype=np.float32)
        for i in range(len(big)):
            coords[i, 0] = float(i)
    else:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, len(big) - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        coords = reducer.fit_transform(big).astype(np.float32)

    out: list[dict] = []
    cursor = 0
    for proj, sub_embs, sub_rows in chunks:
        for i, row in enumerate(sub_rows):
            out.append({
                **row,
                "domain": proj.domain,
                "x": float(coords[cursor + i, 0]),
                "y": float(coords[cursor + i, 1]),
            })
        cursor += len(sub_embs)
    return out, len(big)


def _semantic_entity_maps(projects: list[_Project]) -> dict:
    specs = {
        "link_titles": {
            "types": {"link_title"},
            "label": "Link titles",
            "description": "Anchor/link-title text projected across domains.",
        },
        "headers": {
            "types": {"header"},
            "label": "Header names",
            "description": "H1-H6 headings projected across domains.",
        },
        "page_titles": {
            "types": {"page_title"},
            "label": "Page titles",
            "description": "HTML page titles projected across domains.",
        },
    }
    maps: dict[str, dict] = {}
    for i, (key, spec) in enumerate(specs.items()):
        rows, total = _combined_ahrefs_semantic_umap(
            projects,
            sample_per_domain=1000,
            seed=73 + i,
            include_types=set(spec["types"]),
        )
        maps[key] = {
            "label": spec["label"],
            "description": spec["description"],
            "entity_types": sorted(spec["types"]),
            "points": rows,
            "total": total,
        }
    return maps


def _ahrefs_page_traffic_lookup(proj: _Project) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in ((proj.ahrefs or {}).get("top_pages") or []):
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        payload = {
            "traffic": int(row.get("traffic", 0) or 0),
            "keywords": int(row.get("keywords", 0) or 0),
            "top_keyword": row.get("top_keyword", ""),
        }
        _store_url_lookup(out, url, payload, score_key="traffic")
    return out


def _freshness_lookup(proj: _Project) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in ((proj.freshness or {}).get("per_page") or []):
        url = row.get("url") or ""
        if url:
            _store_url_lookup(out, url, row)
    return out


# --- leaderboard ----------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[i])


COMPARISON_METRIC_GROUPS = [
    {
        "key": "search",
        "label": "Search Demand",
        "metrics": [
            {"key": "ahrefs_org_traffic", "label": "Organic traffic", "better": "high", "fmt": "int"},
            {"key": "ahrefs_org_keywords", "label": "Organic keywords", "better": "high", "fmt": "int"},
            {"key": "ahrefs_matched_traffic", "label": "Matched page traffic", "better": "high", "fmt": "int"},
            {"key": "ahrefs_top_pages_value_usd", "label": "Traffic value $", "better": "high", "fmt": "int"},
            {"key": "ahrefs_top3_keywords", "label": "Top 3 keywords", "better": "high", "fmt": "int"},
        ],
    },
    {
        "key": "topic",
        "label": "Topic Shape",
        "metrics": [
            {"key": "calibrated_focus", "label": "Calibrated focus", "better": "high", "fmt": "0.000"},
            {"key": "site_radius", "label": "Semantic radius", "better": "low", "fmt": "0.000"},
            {"key": "section_coherence", "label": "Section coherence", "better": "high", "fmt": "0.00"},
            {"key": "topic_dimension", "label": "Topic dimension", "better": "na", "fmt": "0.0"},
        ],
    },
    {
        "key": "geo",
        "label": "GEO & Schema",
        "metrics": [
            {"key": "answerability_median", "label": "GEO median", "better": "high", "fmt": "0.0"},
            {"key": "answerability_p90", "label": "GEO p90", "better": "high", "fmt": "0.0"},
            {"key": "answerability_below_4_share", "label": "GEO < 4 share", "better": "low", "fmt": "pct"},
            {"key": "answer_block_strong_cluster_share", "label": "Strong answer clusters", "better": "high", "fmt": "pct"},
            {"key": "answer_block_opportunities", "label": "Answer gaps", "better": "low", "fmt": "int"},
            {"key": "schema_coverage", "label": "Schema coverage", "better": "high", "fmt": "pct"},
            {"key": "invalid_jsonld_blocks", "label": "Invalid JSON-LD", "better": "low", "fmt": "int"},
            {"key": "schema_type_count", "label": "Schema types", "better": "high", "fmt": "int"},
            {"key": "schema_opportunities", "label": "Schema opportunities", "better": "low", "fmt": "int"},
            {"key": "trust_avg_score", "label": "Trust score", "better": "high", "fmt": "0.0"},
            {"key": "trust_high_priority_pages", "label": "Trust gaps", "better": "low", "fmt": "int"},
        ],
    },
    {
        "key": "linking",
        "label": "Internal Linking",
        "metrics": [
            {"key": "median_in_degree", "label": "Median inbound links", "better": "high", "fmt": "0"},
            {"key": "p90_in_degree", "label": "P90 inbound links", "better": "high", "fmt": "0"},
            {"key": "median_out_degree", "label": "Median outbound links", "better": "high", "fmt": "0"},
            {"key": "orphan_share", "label": "Orphan share", "better": "low", "fmt": "pct"},
            {"key": "authority_traffic_alignment", "label": "Authority-demand alignment", "better": "high", "fmt": "pct"},
            {"key": "high_traffic_low_authority_pages", "label": "Underserved search pages", "better": "low", "fmt": "int"},
            {"key": "authority_without_demand_pages", "label": "Authority without demand", "better": "low", "fmt": "int"},
            {"key": "demand_support_alignment", "label": "Demand/support alignment", "better": "high", "fmt": "pct"},
            {"key": "high_demand_low_support_pages", "label": "High-demand low-support", "better": "low", "fmt": "int"},
            {"key": "high_demand_low_support_traffic", "label": "Low-support traffic", "better": "low", "fmt": "int"},
            {"key": "critical_internal_links", "label": "Critical links", "better": "na", "fmt": "int"},
            {"key": "weak_internal_links", "label": "Weak/harmful links", "better": "low", "fmt": "int"},
            {"key": "high_priority_link_additions", "label": "High-priority additions", "better": "na", "fmt": "int"},
            {"key": "avg_link_addition_benefit", "label": "Addition benefit", "better": "high", "fmt": "0.0"},
            {"key": "anchor_relevance_descriptive_rate", "label": "Anchor relevance", "better": "high", "fmt": "pct"},
            {"key": "anchor_relevance_weak_links", "label": "Weak anchors", "better": "low", "fmt": "int"},
            {"key": "contextual_link_avg_impact", "label": "Context link impact", "better": "high", "fmt": "0.0"},
            {"key": "contextual_template_links", "label": "Template links", "better": "low", "fmt": "int"},
            {"key": "internal_link_patterns", "label": "Link patterns", "better": "high", "fmt": "int"},
            {"key": "internal_link_pattern_recommendations", "label": "Pattern gaps", "better": "low", "fmt": "int"},
            {"key": "internal_link_pattern_confidence", "label": "Pattern confidence", "better": "high", "fmt": "pct"},
            {"key": "architecture_resilience", "label": "Architecture resilience", "better": "high", "fmt": "pct"},
            {"key": "bottleneck_pages", "label": "Bottlenecks", "better": "low", "fmt": "int"},
            {"key": "descriptive_anchor_share", "label": "Descriptive anchors", "better": "high", "fmt": "pct"},
            {"key": "generic_anchor_share", "label": "Generic anchors", "better": "low", "fmt": "pct"},
        ],
    },
    {
        "key": "content",
        "label": "Content Quality",
        "metrics": [
            {"key": "metadata_issue_share", "label": "Metadata issue share", "better": "low", "fmt": "pct"},
            {"key": "freshness_date_coverage", "label": "Date coverage", "better": "high", "fmt": "pct"},
            {"key": "freshness_stale_share", "label": "Stale share", "better": "low", "fmt": "pct"},
            {"key": "freshness_impact_traffic_at_risk", "label": "Fresh traffic at risk", "better": "low", "fmt": "int"},
            {"key": "freshness_high_impact_sections", "label": "High-impact stale sections", "better": "low", "fmt": "int"},
            {"key": "entity_coverage", "label": "Entity coverage", "better": "high", "fmt": "pct"},
            {"key": "topical_authority_score", "label": "Authority score", "better": "high", "fmt": "0.0"},
            {"key": "cannibalization_page_conflicts", "label": "Intent conflicts", "better": "low", "fmt": "int"},
            {"key": "cannibalization_paragraph_conflicts", "label": "Paragraph overlaps", "better": "low", "fmt": "int"},
            {"key": "duplicate_fragment_groups", "label": "Duplicate fragments", "better": "low", "fmt": "int"},
            {"key": "duplicate_strong_patterns", "label": "Reusable patterns", "better": "high", "fmt": "int"},
            {"key": "template_success_patterns", "label": "Template patterns", "better": "high", "fmt": "int"},
            {"key": "template_pattern_recommendations", "label": "Template fixes", "better": "low", "fmt": "int"},
            {"key": "zero_link_paragraph_share", "label": "Zero-link paragraphs", "better": "low", "fmt": "pct"},
            {"key": "spammy_paragraph_count", "label": "Spammy paragraphs", "better": "low", "fmt": "int"},
        ],
    },
    {
        "key": "technical",
        "label": "Technical & UX",
        "metrics": [
            {"key": "indexable_share", "label": "Analyzed/indexable share", "better": "high", "fmt": "pct"},
            {"key": "media_accessibility_issue_share", "label": "Media issue share", "better": "low", "fmt": "pct"},
            {"key": "median_html_weight_bytes", "label": "Median HTML bytes", "better": "low", "fmt": "int"},
            {"key": "heavy_page_share", "label": "Heavy page share", "better": "low", "fmt": "pct"},
            {"key": "render_blocking_share", "label": "Render-blocking share", "better": "low", "fmt": "pct"},
            {"key": "pages_missing_h1_share", "label": "Missing H1 share", "better": "low", "fmt": "pct"},
            {"key": "pages_multi_h1", "label": "Multi-H1 pages", "better": "low", "fmt": "int"},
        ],
    },
    {
        "key": "conversion",
        "label": "Conversion",
        "metrics": [
            {"key": "cta_coverage", "label": "CTA coverage", "better": "high", "fmt": "pct"},
            {"key": "primary_cta_coverage", "label": "Primary CTA coverage", "better": "high", "fmt": "pct"},
            {"key": "form_coverage", "label": "Form coverage", "better": "high", "fmt": "pct"},
            {"key": "conversion_balance_efficiency", "label": "Traffic conversion support", "better": "high", "fmt": "pct"},
            {"key": "conversion_balance_high_risk", "label": "High-risk money pages", "better": "low", "fmt": "int"},
            {"key": "lead_pages_without_capture", "label": "Lead leaks", "better": "low", "fmt": "int"},
            {"key": "cta_overload_pages", "label": "CTA overload", "better": "low", "fmt": "int"},
        ],
    },
]


def _comparison_metric_payload(leaderboard: list[dict]) -> dict:
    domains = [row.get("domain", "") for row in leaderboard]
    scorecards = {
        domain: {"domain": domain, "overall_score": 0.0, "scores": {}, "wins": 0}
        for domain in domains
    }
    biggest_gaps: list[dict] = []
    groups_payload: list[dict] = []

    for group in COMPARISON_METRIC_GROUPS:
        metric_payloads: list[dict] = []
        group_score_values = {domain: [] for domain in domains}

        for metric in group["metrics"]:
            values: dict[str, float] = {}
            for row in leaderboard:
                domain = row.get("domain", "")
                try:
                    values[domain] = float(row.get(metric["key"], 0.0) or 0.0)
                except (TypeError, ValueError):
                    values[domain] = 0.0

            numeric_values = list(values.values())
            vmin = min(numeric_values) if numeric_values else 0.0
            vmax = max(numeric_values) if numeric_values else 0.0
            better = metric.get("better", "na")

            winner = ""
            worst = ""
            if domains and better == "high":
                winner = max(values, key=values.get)
                worst = min(values, key=values.get)
            elif domains and better == "low":
                winner = min(values, key=values.get)
                worst = max(values, key=values.get)

            if winner:
                scorecards[winner]["wins"] += 1

            if better in {"high", "low"} and vmax != vmin:
                for domain, value in values.items():
                    normalized = (value - vmin) / (vmax - vmin)
                    score = normalized if better == "high" else 1.0 - normalized
                    group_score_values[domain].append(score)

                best_value = values[winner] if winner else 0.0
                worst_value = values[worst] if worst else 0.0
                denom = max(abs(best_value), abs(worst_value), 1.0)
                relative_gap = abs(best_value - worst_value) / denom
                biggest_gaps.append({
                    "group": group["label"],
                    "metric": metric["label"],
                    "key": metric["key"],
                    "fmt": metric["fmt"],
                    "better": better,
                    "winner": winner,
                    "winner_value": best_value,
                    "worst": worst,
                    "worst_value": worst_value,
                    "relative_gap": relative_gap,
                    "values": values,
                })

            metric_payloads.append({
                **metric,
                "values": values,
                "winner": winner,
                "worst": worst,
                "min": vmin,
                "max": vmax,
            })

        for domain, scores in group_score_values.items():
            scorecards[domain]["scores"][group["key"]] = (
                sum(scores) / len(scores) if scores else 0.0
            )

        groups_payload.append({
            "key": group["key"],
            "label": group["label"],
            "metrics": metric_payloads,
        })

    for card in scorecards.values():
        scores = list(card["scores"].values())
        card["overall_score"] = sum(scores) / len(scores) if scores else 0.0

    return {
        "metric_groups": groups_payload,
        "scorecards": sorted(scorecards.values(), key=lambda c: c["overall_score"], reverse=True),
        "biggest_gaps": sorted(biggest_gaps, key=lambda g: g["relative_gap"], reverse=True)[:25],
    }


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
    ts = (proj.trust_signals or {}).get("summary", {}) or {}
    cb = (proj.conversion_balance or {}).get("summary", {}) or {}
    mq = (proj.metadata_quality or {}).get("summary", {}) or {}
    ma = (proj.media_accessibility or {}).get("summary", {}) or {}
    pt = (proj.page_types or {}).get("summary", {}) or {}
    ent = (proj.entities or {}).get("summary", {}) or {}
    fr = (proj.freshness or {}).get("summary", {}) or {}
    fi = (proj.freshness_impact or {}).get("summary", {}) or {}
    cv = (proj.conversion or {}).get("summary", {}) or {}
    ix = (proj.indexability or {}).get("summary", {}) or {}
    pf = (proj.performance or {}).get("summary", {}) or {}
    ah = proj.ahrefs or {}
    ah_summary = ah.get("summary", {}) or {}
    ah_metrics = ah.get("metrics", {}) or {}
    twpr = ((proj.linkgraph or {}).get("traffic_weighted_pagerank") or {}).get("summary", {}) or {}
    removal = ((proj.linkgraph or {}).get("link_removal_simulation") or {}).get("summary", {}) or {}
    addition = ((proj.linkgraph or {}).get("link_addition_simulation") or {}).get("summary", {}) or {}
    anchor_rel = ((proj.linkgraph or {}).get("anchor_relevance") or {}).get("summary", {}) or {}
    context_links = ((proj.linkgraph or {}).get("contextual_link_impact") or {}).get("summary", {}) or {}
    link_patterns = ((proj.linkgraph or {}).get("internal_link_patterns") or {}).get("summary", {}) or {}
    hubs = ((proj.linkgraph or {}).get("hub_bottlenecks") or {}).get("summary", {}) or {}
    hdl = ((proj.linkgraph or {}).get("high_demand_low_link") or {}).get("summary", {}) or {}
    ablocks = (proj.answer_blocks or {}).get("summary", {}) or {}
    ablock_clusters = int(ablocks.get("top_query_clusters", 0) or 0)
    cannibal = (proj.cannibalization or {}).get("summary", {}) or {}
    dupfrag = (proj.duplicate_fragments or {}).get("summary", {}) or {}
    templates = (proj.template_patterns or {}).get("summary", {}) or {}

    n_pages = len(proj.page_link_counts) or m.get("page_count") or len(proj.pages) or 0
    orphan_share = (sum(1 for r in proj.page_link_counts if r.get("in_degree") == 0) / n_pages) if n_pages else 0.0

    return {
        "domain": proj.domain,
        "pages": int(n_pages),
        "ahrefs_status": str((ah.get("meta", {}) or {}).get("status", "")),
        "ahrefs_org_traffic": int(ah_metrics.get("org_traffic", 0) or ah_summary.get("top_pages_traffic", 0) or 0),
        "ahrefs_org_keywords": int(ah_metrics.get("org_keywords", 0) or ah_summary.get("organic_keywords", 0) or 0),
        "ahrefs_top3_keywords": int(ah_metrics.get("org_keywords_1_3", 0) or 0),
        "ahrefs_top_pages": int(ah_summary.get("top_pages", 0) or 0),
        "ahrefs_matched_traffic": int(ah_summary.get("matched_traffic", 0) or 0),
        "ahrefs_matched_traffic_share": float(ah_summary.get("matched_traffic_share", 0.0) or 0.0),
        "ahrefs_top_pages_value_usd": float(ah_summary.get("top_pages_value_usd", 0.0) or 0.0),
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
        "answer_block_strong_cluster_share": (
            int(ablocks.get("strong_query_clusters", 0) or 0) / ablock_clusters if ablock_clusters else 0.0
        ),
        "answer_block_opportunities": int(ablocks.get("opportunity_queries", 0) or 0),
        "answer_block_strong_blocks": int(ablocks.get("strong_blocks", 0) or 0),
        # Internal linking
        "median_in_degree": _percentile([float(v) for v in in_degrees], 0.5),
        "p90_in_degree": _percentile([float(v) for v in in_degrees], 0.9),
        "median_out_degree": _percentile([float(v) for v in out_degrees], 0.5),
        "orphan_share": orphan_share,
        "authority_traffic_alignment": float(twpr.get("authority_traffic_alignment", 0.0)),
        "high_traffic_low_authority_pages": int(twpr.get("high_traffic_low_authority_pages", 0)),
        "authority_without_demand_pages": int(twpr.get("high_authority_low_value_pages", 0)),
        "orphan_traffic_share": float(twpr.get("orphan_traffic_share", 0.0)),
        "demand_support_alignment": float(hdl.get("demand_support_alignment", 0.0)),
        "high_demand_low_support_pages": int(hdl.get("high_demand_low_support_pages", 0)),
        "high_demand_low_support_traffic": int(hdl.get("high_demand_low_support_traffic", 0)),
        "critical_internal_links": int(removal.get("critical_links", 0)),
        "weak_internal_links": int(removal.get("irrelevant_links", 0)) + int(removal.get("potentially_harmful_links", 0)),
        "template_navigation_links": int(removal.get("template_navigation_links", 0)),
        "high_priority_link_additions": int(addition.get("high_priority", 0)),
        "avg_link_addition_benefit": float(addition.get("avg_expected_benefit", 0.0)),
        "anchor_relevance_descriptive_rate": float(anchor_rel.get("descriptive_rate", 0.0)),
        "anchor_relevance_weak_links": int(anchor_rel.get("weak_links", 0)),
        "contextual_link_avg_impact": float(context_links.get("avg_contextual_impact", 0.0)),
        "contextual_template_links": int(context_links.get("template_links", 0)),
        "contextual_high_impact_links": int(context_links.get("high_impact_contextual_links", 0)),
        "internal_link_patterns": int(link_patterns.get("patterns", 0)),
        "internal_link_pattern_recommendations": int(link_patterns.get("recommendations", 0)),
        "internal_link_pattern_confidence": float(link_patterns.get("avg_confidence", 0.0)),
        "architecture_resilience": float(hubs.get("architecture_resilience", 0.0)),
        "bottleneck_pages": int(hubs.get("bottleneck_pages", 0)),
        "bridge_pages": int(hubs.get("bridge_pages", 0)),
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
        "schema_opportunities": int(sd.get("schema_opportunities", 0)),
        "high_priority_schema_opportunities": int(sd.get("high_priority_schema_opportunities", 0)),
        "schema_opportunity_clusters": int(sd.get("schema_opportunity_clusters", 0)),
        "trust_avg_score": float(ts.get("avg_trust_score", 0.0)),
        "trust_high_priority_pages": int(ts.get("high_priority_pages", 0)),
        "trust_missing_evidence_items": int(ts.get("missing_evidence_items", 0)),
        "metadata_issue_share": float(mq.get("issue_share", 0.0)),
        "missing_description": int(mq.get("missing_description", 0)),
        "duplicate_title_pages": int(mq.get("duplicate_title_pages", 0)),
        "missing_canonical": int(mq.get("missing_canonical", 0)),
        "incomplete_open_graph": int(mq.get("incomplete_open_graph", 0)),
        "media_accessibility_issue_share": float(ma.get("issue_share", 0.0)),
        "images_missing_alt": int(ma.get("images_missing_alt", 0)),
        "linked_images_empty_alt": int(ma.get("linked_images_empty_alt", 0)),
        "videos_missing_captions": int(ma.get("videos_missing_captions", 0)),
        "iframes_missing_title": int(ma.get("iframes_missing_title", 0)),
        "page_type_count": int(pt.get("page_type_count", 0)),
        "template_family_count": int(pt.get("template_family_count", 0)),
        "template_signature_count": int(pt.get("template_signature_count", 0)),
        "dominant_page_type": str(pt.get("dominant_page_type", "")),
        "dominant_template_family": str(pt.get("dominant_template_family", "")),
        "freshness_date_coverage": float(fr.get("date_coverage", 0.0)),
        "freshness_stale_share": float(fr.get("stale_share", 0.0)),
        "freshness_missing_dates": int(fr.get("missing_dates", 0)),
        "freshness_very_stale_pages": int(fr.get("pages_very_stale", 0)),
        "freshness_median_age_days": int(fr.get("median_age_days") or 0),
        "freshness_impact_traffic_at_risk": int(fi.get("traffic_at_risk", 0)),
        "freshness_high_impact_sections": int(fi.get("high_impact_sections", 0)),
        "freshness_avg_impact_risk": float(fi.get("avg_freshness_risk", 0.0)),
        "entity_coverage": float(ent.get("entity_coverage", 0.0)),
        "unique_entities": int(ent.get("unique_entities", 0)),
        "avg_entities_per_page": float(ent.get("avg_entities_per_page", 0.0)),
        "entity_reuse_share": float(ent.get("entity_reuse_share", 0.0)),
        "organization_count": int(ent.get("organization_count", 0)),
        "organization_coverage": float(ent.get("organization_coverage", 0.0)),
        "topical_depth_share": float(ent.get("topical_depth_share", 0.0)),
        "topical_authority_score": float(ent.get("topical_authority_score", 0.0)),
        "cannibalization_page_conflicts": int(cannibal.get("page_conflicts", 0)),
        "cannibalization_paragraph_conflicts": int(cannibal.get("paragraph_conflicts", 0)),
        "cannibalization_traffic_at_risk": int(cannibal.get("traffic_at_risk", 0)),
        "duplicate_fragment_groups": int(dupfrag.get("groups", 0)),
        "duplicate_strong_patterns": int(dupfrag.get("strong_patterns", 0)),
        "duplicate_harmful_boilerplate": int(dupfrag.get("harmful_boilerplate", 0)),
        "template_success_patterns": int(templates.get("patterns", 0)),
        "template_pattern_recommendations": int(templates.get("recommendations", 0)),
        "template_pattern_segments": int(templates.get("segments_compared", 0)),
        "template_pattern_median_confidence": float(templates.get("median_confidence", 0.0)),
        "cta_coverage": float(cv.get("cta_coverage", 0.0)),
        "primary_cta_coverage": float(cv.get("primary_cta_coverage", 0.0)),
        "form_coverage": float(cv.get("form_coverage", 0.0)),
        "conversion_balance_efficiency": float(cb.get("conversion_efficiency", 0.0)),
        "conversion_balance_high_risk": int(cb.get("high_risk_money_pages", 0)),
        "conversion_balance_avg_seo": float(cb.get("avg_seo_support", 0.0)),
        "conversion_balance_avg_conversion": float(cb.get("avg_conversion_support", 0.0)),
        "avg_ctas_per_page": float(cv.get("avg_ctas_per_page", 0.0)),
        "lead_pages_without_capture": int(cv.get("lead_pages_without_capture", 0)),
        "cta_overload_pages": int(cv.get("cta_overload_pages", 0)),
        "indexable_share": float(ix.get("indexable_share", 0.0)),
        "noindex_share": float(ix.get("noindex_share", 0.0)),
        "skipped_pages": int(ix.get("skipped_pages", 0)),
        "median_html_weight_bytes": int(pf.get("median_html_weight_bytes", 0)),
        "p90_estimated_weight_bytes": int(pf.get("p90_estimated_weight_bytes", 0)),
        "avg_resource_tags_per_page": float(pf.get("avg_resource_tags_per_page", 0.0)),
        "render_blocking_share": float(pf.get("render_blocking_share", 0.0)),
        "heavy_page_share": float(pf.get("heavy_page_share", 0.0)),
        "total_images": int(pf.get("total_images", 0)),
        "total_scripts": int(pf.get("total_scripts", 0)),
        "total_stylesheets": int(pf.get("total_stylesheets", 0)),
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
    top_pages = (proj.ahrefs or {}).get("top_pages") or []
    organic_keywords = (proj.ahrefs or {}).get("organic_keywords") or []
    return {
        "domain": proj.domain,
        "in_degree": _distribution([float(r.get("in_degree", 0)) for r in proj.page_link_counts]),
        "out_degree": _distribution([float(r.get("out_degree", 0)) for r in proj.page_link_counts]),
        "answerability": _distribution([float(a.get("score", 0.0)) for a in (proj.answerability or [])]),
        "paragraph_density_per_page": _distribution(
            [float(r.get("links_per_100w", 0.0))
             for r in ((proj.paragraph_density or {}).get("per_page") or [])]
        ),
        "ahrefs_page_traffic": _distribution([float(r.get("traffic", 0) or 0) for r in top_pages]),
        "ahrefs_keyword_traffic": _distribution([float(r.get("traffic", 0) or 0) for r in organic_keywords]),
        "freshness_buckets": dict((proj.freshness or {}).get("buckets") or {}),
        "freshness_summary": dict((proj.freshness or {}).get("summary") or {}),
    }


def _search_payload(projects: list[_Project]) -> dict:
    summaries = []
    directories = []
    clusters = []
    for proj in projects:
        ah = proj.ahrefs or {}
        summaries.append({
            "domain": proj.domain,
            "summary": ah.get("summary", {}) or {},
            "metrics": ah.get("metrics", {}) or {},
            "meta": ah.get("meta", {}) or {},
        })
        for row in ah.get("directories", []) or []:
            directories.append({"domain": proj.domain, **row})
        for row in ah.get("clusters", []) or []:
            clusters.append({"domain": proj.domain, **row})
    return {
        "summaries": summaries,
        "directories": directories,
        "clusters": clusters,
    }


def _entity_alignment_comparison(projects: list[_Project]) -> dict:
    entity_types: dict[str, dict] = {}
    summaries: list[dict] = []
    recommendations: list[dict] = []
    for proj in projects:
        alignment = (proj.ahrefs or {}).get("entity_alignment") or {}
        summary = alignment.get("summary", {}) or {}
        if summary:
            summaries.append({"domain": proj.domain, **summary})
        for row in alignment.get("entity_types") or []:
            typ = row.get("type") or ""
            if not typ:
                continue
            group = entity_types.setdefault(typ, {
                "type": typ,
                "label": row.get("label") or typ,
                "domains": [],
            })
            group["domains"].append({
                "domain": proj.domain,
                "average_score": _safe_float(row.get("average_score")),
                "below_threshold": _safe_int(row.get("below_threshold")),
                "missing": _safe_int(row.get("missing")),
                "rows": _safe_int(row.get("rows")),
            })
        for rec in (alignment.get("recommendations") or [])[:80]:
            recommendations.append({"domain": proj.domain, **rec})
    recommendations.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
    return {
        "summaries": summaries,
        "entity_types": list(entity_types.values()),
        "recommendations": recommendations[:300],
    }


def _entity_coverage_comparison(projects: list[_Project]) -> dict:
    entity_domains: dict[str, dict[str, dict]] = defaultdict(dict)
    cluster_rows: list[dict] = []
    page_rows: list[dict] = []
    for proj in projects:
        coverage = proj.entity_coverage or {}
        for cluster in coverage.get("clusters") or []:
            cluster_rows.append({"domain": proj.domain, **cluster})
            for entity in (cluster.get("expected_entities") or [])[:40]:
                name = entity.get("entity") or ""
                if not name:
                    continue
                entity_domains[name][proj.domain] = {
                    "domain": proj.domain,
                    "entity": name,
                    "weight": _safe_float(entity.get("weight")),
                    "class": entity.get("class") or "",
                    "cluster": cluster.get("cluster"),
                    "cluster_label": cluster.get("label") or "",
                    "pages": _safe_int(entity.get("pages")),
                }
        for page in (coverage.get("pages") or [])[:120]:
            page_rows.append({"domain": proj.domain, **page})
    matrix = []
    for entity, domains in entity_domains.items():
        total_weight = sum(_safe_float(row.get("weight")) for row in domains.values())
        matrix.append({
            "entity": entity,
            "domain_count": len(domains),
            "total_weight": round(total_weight, 4),
            "class": next((row.get("class") for row in domains.values() if row.get("class")), ""),
            "domains": [domains.get(domain, {"domain": domain, "entity": entity, "weight": 0.0}) for domain in [p.domain for p in projects]],
        })
    matrix.sort(key=lambda r: (_safe_int(r.get("domain_count")), _safe_float(r.get("total_weight"))), reverse=True)
    page_rows.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_float(r.get("coverage"))), reverse=True)
    return {
        "matrix": matrix[:120],
        "clusters": cluster_rows[:300],
        "low_coverage_pages": [r for r in page_rows if _safe_float(r.get("coverage")) < 0.5][:200],
    }


def _information_gain_comparison(projects: list[_Project]) -> dict:
    domains = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    low_pages: list[dict] = []
    for proj in projects:
        payload = proj.information_gain or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for cluster in payload.get("clusters") or []:
            key = str(cluster.get("label") or cluster.get("cluster") or "cluster")
            clusters[key][proj.domain] = {
                "domain": proj.domain,
                "cluster": key,
                "avg_score": _safe_float(cluster.get("avg_score")),
                "pages": _safe_int(cluster.get("pages")),
                "low_score_pages": _safe_int(cluster.get("low_score_pages")),
            }
        for page in (payload.get("pages") or [])[:80]:
            if _safe_float(page.get("information_gain_score")) < 55:
                low_pages.append({"domain": proj.domain, **page})
    matrix = []
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [values.get(domain, {"domain": domain, "cluster": cluster, "avg_score": 0.0, "pages": 0}) for domain in [p.domain for p in projects]],
        })
    matrix.sort(key=lambda r: sum(_safe_float(d.get("avg_score")) for d in r["domains"]) / max(len(r["domains"]), 1))
    low_pages.sort(key=lambda r: _safe_float(r.get("information_gain_score")))
    return {"domains": domains, "clusters": matrix[:80], "low_pages": low_pages[:200]}


def _structured_data_opportunity_comparison(projects: list[_Project]) -> dict:
    domains = []
    type_domains: dict[str, dict[str, dict]] = defaultdict(dict)
    cluster_domains: dict[str, dict[str, dict]] = defaultdict(dict)
    recommendations: list[dict] = []
    invalid: list[dict] = []
    for proj in projects:
        payload = proj.structured_data or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("top_types") or []:
            schema_type = row.get("type") or ""
            if not schema_type:
                continue
            type_domains[schema_type][proj.domain] = {
                "domain": proj.domain,
                "schema_type": schema_type,
                "pages": _safe_int(row.get("pages")),
                "opportunities": 0,
                "high_priority": 0,
            }
        for row in payload.get("opportunities") or []:
            schema_type = row.get("schema_type") or "unknown"
            stats = type_domains[schema_type].setdefault(proj.domain, {
                "domain": proj.domain,
                "schema_type": schema_type,
                "pages": 0,
                "opportunities": 0,
                "high_priority": 0,
            })
            stats["opportunities"] += 1
            if row.get("priority") == "high":
                stats["high_priority"] += 1
            recommendations.append({"domain": proj.domain, **row})
        for row in payload.get("clusters") or []:
            cluster = str(row.get("cluster") or "unclustered")
            cluster_domains[cluster][proj.domain] = {
                "domain": proj.domain,
                "cluster": cluster,
                "schema_coverage": _safe_float(row.get("schema_coverage")),
                "pages": _safe_int(row.get("pages")),
                "traffic": _safe_int(row.get("traffic")),
                "opportunities": _safe_int(row.get("opportunities")),
                "invalid_blocks": _safe_int(row.get("invalid_blocks")),
                "top_schema_types": row.get("top_schema_types") or [],
            }
        for row in payload.get("invalid_blocks") or []:
            invalid.append({"domain": proj.domain, **row})

    project_domains = [p.domain for p in projects]
    type_matrix = []
    for schema_type, values in type_domains.items():
        type_matrix.append({
            "schema_type": schema_type,
            "domains": [
                values.get(domain, {"domain": domain, "schema_type": schema_type, "pages": 0, "opportunities": 0, "high_priority": 0})
                for domain in project_domains
            ],
        })
    type_matrix.sort(key=lambda r: sum(_safe_int(d.get("pages")) + _safe_int(d.get("opportunities")) for d in r["domains"]), reverse=True)

    cluster_matrix = []
    for cluster, values in cluster_domains.items():
        cluster_matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {"domain": domain, "cluster": cluster, "schema_coverage": 0.0, "pages": 0, "traffic": 0, "opportunities": 0, "invalid_blocks": 0})
                for domain in project_domains
            ],
        })
    cluster_matrix.sort(key=lambda r: sum(_safe_int(d.get("traffic")) + _safe_int(d.get("opportunities")) * 10 for d in r["domains"]), reverse=True)
    recommendations.sort(key=lambda r: (1 if r.get("priority") == "high" else 0, _safe_int(r.get("traffic"))), reverse=True)
    invalid.sort(key=lambda r: (r.get("domain", ""), r.get("url", "")))
    return {
        "domains": domains,
        "types": type_matrix[:80],
        "clusters": cluster_matrix[:100],
        "recommendations": recommendations[:300],
        "invalid": invalid[:200],
    }


def _trust_signal_comparison(projects: list[_Project]) -> dict:
    domains = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    missing: list[dict] = []
    pages: list[dict] = []
    for proj in projects:
        payload = proj.trust_signals or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("clusters") or []:
            cluster = str(row.get("cluster") or "unclustered")
            clusters[cluster][proj.domain] = {
                "domain": proj.domain,
                "cluster": cluster,
                "avg_trust_score": _safe_float(row.get("avg_trust_score")),
                "leader_score": _safe_float(row.get("leader_score")),
                "pages": _safe_int(row.get("pages")),
                "traffic": _safe_int(row.get("traffic")),
                "leader_examples": row.get("leader_examples") or [],
            }
        for row in payload.get("missing_evidence") or []:
            missing.append({"domain": proj.domain, **row})
        for row in (payload.get("pages") or [])[:80]:
            pages.append({"domain": proj.domain, **row})
    project_domains = [p.domain for p in projects]
    matrix = []
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {"domain": domain, "cluster": cluster, "avg_trust_score": 0.0, "leader_score": 0.0, "pages": 0, "traffic": 0})
                for domain in project_domains
            ],
        })
    matrix.sort(key=lambda r: sum(_safe_int(d.get("traffic")) for d in r["domains"]), reverse=True)
    missing.sort(key=lambda r: (1 if r.get("priority") == "high" else 0, _safe_int(r.get("traffic"))), reverse=True)
    pages.sort(key=lambda r: ({"high": 2, "medium": 1, "low": 0}.get(r.get("priority"), 0), _safe_int(r.get("traffic"))), reverse=True)
    return {"domains": domains, "clusters": matrix[:100], "missing_evidence": missing[:300], "pages": pages[:240]}


def _conversion_balance_comparison(projects: list[_Project]) -> dict:
    domains = []
    pages: list[dict] = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    high_risk: list[dict] = []
    for proj in projects:
        payload = proj.conversion_balance or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        by_cluster: dict[str, list[dict]] = defaultdict(list)
        for row in payload.get("pages") or []:
            item = {"domain": proj.domain, **row}
            pages.append(item)
            by_cluster[str(row.get("cluster") or "unclustered")].append(row)
            if row.get("balance_label") == "high_risk_money_page" or (row.get("money_page") and _safe_float(row.get("conversion_support")) < 45):
                high_risk.append(item)
        for cluster, rows in by_cluster.items():
            clusters[cluster][proj.domain] = {
                "domain": proj.domain,
                "cluster": cluster,
                "pages": len(rows),
                "traffic": sum(_safe_int(r.get("traffic")) for r in rows),
                "avg_seo_support": sum(_safe_float(r.get("seo_support")) for r in rows) / max(1, len(rows)),
                "avg_conversion_support": sum(_safe_float(r.get("conversion_support")) for r in rows) / max(1, len(rows)),
                "high_risk": sum(1 for r in rows if r.get("balance_label") == "high_risk_money_page"),
            }
    project_domains = [p.domain for p in projects]
    matrix = []
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {"domain": domain, "cluster": cluster, "pages": 0, "traffic": 0, "avg_seo_support": 0.0, "avg_conversion_support": 0.0, "high_risk": 0})
                for domain in project_domains
            ],
        })
    matrix.sort(key=lambda r: sum(_safe_int(d.get("traffic")) for d in r["domains"]), reverse=True)
    pages.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("seo_support")) - _safe_float(r.get("conversion_support"))), reverse=True)
    high_risk.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
    return {"domains": domains, "clusters": matrix[:100], "pages": pages[:500], "high_risk": high_risk[:200]}


def _answer_blocks_comparison(projects: list[_Project]) -> dict:
    domains = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    opportunities: list[dict] = []
    for proj in projects:
        payload = proj.answer_blocks or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for cluster in payload.get("clusters") or []:
            key = str(cluster.get("label") or cluster.get("cluster") or "cluster")
            clusters[key][proj.domain] = {
                "domain": proj.domain,
                "cluster": key,
                "strong_query_share": _safe_float(cluster.get("strong_query_share")),
                "avg_best_score": _safe_float(cluster.get("avg_best_score")),
                "queries": _safe_int(cluster.get("queries")),
                "traffic": _safe_int(cluster.get("traffic")),
                "opportunity_queries": _safe_int(cluster.get("opportunity_queries")),
                "recommended_format": cluster.get("recommended_format") or "",
                "status": cluster.get("status") or "",
            }
        for row in (payload.get("opportunities") or [])[:80]:
            opportunities.append({"domain": proj.domain, **row})
    matrix = []
    project_domains = [p.domain for p in projects]
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "cluster": cluster,
                    "strong_query_share": 0.0,
                    "avg_best_score": 0.0,
                    "queries": 0,
                    "traffic": 0,
                    "opportunity_queries": 0,
                    "recommended_format": "",
                })
                for domain in project_domains
            ],
        })
    matrix.sort(
        key=lambda r: (
            sum(_safe_float(d.get("strong_query_share")) for d in r["domains"]) / max(len(r["domains"]), 1),
            -sum(_safe_int(d.get("traffic")) for d in r["domains"]),
        )
    )
    opportunities.sort(key=lambda r: (_safe_int(r.get("keyword_traffic")), -_safe_float(r.get("best_score"))), reverse=True)
    return {"domains": domains, "clusters": matrix[:80], "opportunities": opportunities[:200]}


def _freshness_impact_comparison(projects: list[_Project]) -> dict:
    domains = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    sections: list[dict] = []
    for proj in projects:
        payload = proj.freshness_impact or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for cluster in payload.get("clusters") or []:
            key = str(cluster.get("label") or cluster.get("cluster") or "cluster")
            clusters[key][proj.domain] = {
                "domain": proj.domain,
                "cluster": key,
                "avg_freshness_risk": _safe_float(cluster.get("avg_freshness_risk")),
                "max_priority_score": _safe_float(cluster.get("max_priority_score")),
                "traffic_at_risk": _safe_int(cluster.get("traffic_at_risk")),
                "sections": _safe_int(cluster.get("sections")),
                "stale_sections": _safe_int(cluster.get("stale_sections")),
            }
        for row in (payload.get("sections") or [])[:80]:
            if _safe_float(row.get("freshness_risk")) >= 50 or _safe_int(row.get("traffic")) > 0:
                sections.append({"domain": proj.domain, **row})
    project_domains = [p.domain for p in projects]
    matrix = []
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "cluster": cluster,
                    "avg_freshness_risk": 0.0,
                    "max_priority_score": 0.0,
                    "traffic_at_risk": 0,
                    "sections": 0,
                    "stale_sections": 0,
                })
                for domain in project_domains
            ],
        })
    matrix.sort(
        key=lambda r: (
            sum(_safe_float(d.get("avg_freshness_risk")) for d in r["domains"]) / max(len(r["domains"]), 1),
            sum(_safe_int(d.get("traffic_at_risk")) for d in r["domains"]),
        ),
        reverse=True,
    )
    sections.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_int(r.get("traffic"))), reverse=True)
    return {"domains": domains, "clusters": matrix[:80], "sections": sections[:200]}


def _cannibalization_comparison(projects: list[_Project]) -> dict:
    domains = []
    classes: dict[str, dict[str, dict]] = defaultdict(dict)
    conflicts: list[dict] = []
    for proj in projects:
        payload = proj.cannibalization or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        class_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "traffic_at_risk": 0})
        for row in payload.get("page_conflicts") or []:
            cls = row.get("classification") or "unknown"
            class_counts[cls]["count"] += 1
            class_counts[cls]["traffic_at_risk"] += _safe_int(row.get("traffic_at_risk"))
            conflicts.append({"domain": proj.domain, **row})
        for cls, values in class_counts.items():
            classes[cls][proj.domain] = {
                "domain": proj.domain,
                "classification": cls,
                "count": values["count"],
                "traffic_at_risk": values["traffic_at_risk"],
            }
    project_domains = [p.domain for p in projects]
    matrix = []
    for cls, values in classes.items():
        matrix.append({
            "classification": cls,
            "domains": [
                values.get(domain, {"domain": domain, "classification": cls, "count": 0, "traffic_at_risk": 0})
                for domain in project_domains
            ],
        })
    matrix.sort(key=lambda r: sum(_safe_int(d.get("traffic_at_risk")) + _safe_int(d.get("count")) for d in r["domains"]), reverse=True)
    conflicts.sort(key=lambda r: (_safe_int(r.get("traffic_at_risk")), _safe_int(r.get("traffic"))), reverse=True)
    return {"domains": domains, "classes": matrix, "conflicts": conflicts[:200]}


def _duplicate_fragments_comparison(projects: list[_Project]) -> dict:
    domains = []
    classes: dict[str, dict[str, dict]] = defaultdict(dict)
    examples: list[dict] = []
    for proj in projects:
        payload = proj.duplicate_fragments or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "traffic": 0})
        for row in payload.get("groups") or []:
            cls = row.get("classification") or "unknown"
            counts[cls]["count"] += 1
            counts[cls]["traffic"] += _safe_int(row.get("attributed_traffic")) + _safe_int(row.get("page_traffic_sum"))
            if len(examples) < 200:
                examples.append({"domain": proj.domain, **row})
        for cls, values in counts.items():
            classes[cls][proj.domain] = {
                "domain": proj.domain,
                "classification": cls,
                "count": values["count"],
                "traffic": values["traffic"],
            }
    project_domains = [p.domain for p in projects]
    matrix = []
    for cls, values in classes.items():
        matrix.append({
            "classification": cls,
            "domains": [
                values.get(domain, {"domain": domain, "classification": cls, "count": 0, "traffic": 0})
                for domain in project_domains
            ],
        })
    matrix.sort(key=lambda r: sum(_safe_int(d.get("count")) + _safe_int(d.get("traffic")) for d in r["domains"]), reverse=True)
    examples.sort(key=lambda r: (_safe_int(r.get("attributed_traffic")) + _safe_int(r.get("page_traffic_sum")), _safe_int(r.get("count"))), reverse=True)
    return {"domains": domains, "classes": matrix, "examples": examples[:200]}


def _template_patterns_comparison(projects: list[_Project]) -> dict:
    domains = []
    features: dict[str, dict[str, dict]] = defaultdict(dict)
    examples: list[dict] = []
    recommendations: list[dict] = []
    for proj in projects:
        payload = proj.template_patterns or {}
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        buckets: dict[str, dict] = defaultdict(lambda: {
            "label": "",
            "category": "",
            "count": 0,
            "lift_sum": 0.0,
            "confidence_sum": 0.0,
            "recommendations": 0,
            "sample_size": 0,
        })
        for row in payload.get("patterns") or []:
            key = row.get("feature_key") or row.get("label") or "unknown"
            bucket = buckets[key]
            bucket["label"] = row.get("label") or key
            bucket["category"] = row.get("category") or ""
            bucket["count"] += 1
            bucket["lift_sum"] += _safe_float(row.get("observed_lift"))
            bucket["confidence_sum"] += _safe_float(row.get("confidence"))
            bucket["recommendations"] += len(row.get("affected_weak_pages") or [])
            bucket["sample_size"] += _safe_int(row.get("sample_size"))
            examples.append({"domain": proj.domain, **row})
        for row in payload.get("recommendations") or []:
            recommendations.append({"domain": proj.domain, **row})
        for key, values in buckets.items():
            count = max(1, values["count"])
            features[key][proj.domain] = {
                "domain": proj.domain,
                "feature_key": key,
                "label": values["label"],
                "category": values["category"],
                "count": values["count"],
                "avg_lift": values["lift_sum"] / count,
                "avg_confidence": values["confidence_sum"] / count,
                "recommendations": values["recommendations"],
                "sample_size": values["sample_size"],
            }
    project_domains = [p.domain for p in projects]
    matrix = []
    for feature_key, values in features.items():
        label = next((v.get("label") for v in values.values() if v.get("label")), feature_key)
        category = next((v.get("category") for v in values.values() if v.get("category")), "")
        matrix.append({
            "feature_key": feature_key,
            "label": label,
            "category": category,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "feature_key": feature_key,
                    "label": label,
                    "category": category,
                    "count": 0,
                    "avg_lift": 0.0,
                    "avg_confidence": 0.0,
                    "recommendations": 0,
                    "sample_size": 0,
                })
                for domain in project_domains
            ],
        })
    matrix.sort(
        key=lambda r: (
            sum(_safe_int(d.get("count")) for d in r["domains"]),
            sum(_safe_float(d.get("avg_lift")) for d in r["domains"]),
            sum(_safe_int(d.get("recommendations")) for d in r["domains"]),
        ),
        reverse=True,
    )
    examples.sort(key=lambda r: (_safe_float(r.get("confidence")), _safe_float(r.get("observed_lift"))), reverse=True)
    recommendations.sort(key=lambda r: (_safe_float(r.get("confidence")), _safe_float(r.get("observed_lift"))), reverse=True)
    return {
        "domains": domains,
        "features": matrix[:120],
        "examples": examples[:200],
        "recommendations": recommendations[:300],
    }


# --- competitive search/content opportunities ----------------------------


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_keyword(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _string_list(value: object, fallback_key: str = "") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,|]", value) if v.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                candidate = item.get(fallback_key) or item.get("feature") or item.get("intent") or item.get("type")
                if candidate:
                    out.append(str(candidate).strip())
        return [v for v in out if v]
    return []


def _url_keys(url: object) -> set[str]:
    raw = str(url or "").strip()
    if not raw:
        return set()

    keys = {raw, raw.rstrip("/")}
    try:
        parts = urlsplit(raw)
    except ValueError:
        return {k for k in keys if k}

    if not parts.netloc:
        return {k for k in keys if k}

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    path_trimmed = path.rstrip("/") or "/"
    netlocs = {netloc}
    if netloc.startswith("www."):
        netlocs.add(netloc[4:])
    else:
        netlocs.add(f"www.{netloc}")
    schemes = {scheme}
    if scheme in {"http", "https"}:
        schemes.add("https" if scheme == "http" else "http")

    for candidate_scheme in schemes:
        for candidate_netloc in netlocs:
            for candidate_path in {path, path_trimmed}:
                normalized = urlunsplit((candidate_scheme, candidate_netloc, candidate_path, "", ""))
                keys.add(normalized)
                keys.add(normalized.rstrip("/"))
    return {k for k in keys if k}


def _store_url_lookup(
    out: dict[str, dict],
    url: object,
    row: dict,
    score_key: str = "",
) -> None:
    for key in _url_keys(url):
        current = out.get(key)
        if current is None:
            out[key] = row
        elif score_key and _safe_float(row.get(score_key)) > _safe_float(current.get(score_key)):
            out[key] = row


def _url_lookup_from_rows(
    rows: list[dict],
    url_fields: tuple[str, ...] = ("url",),
    score_key: str = "",
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for field in url_fields:
            if row.get(field):
                _store_url_lookup(out, row.get(field), row, score_key=score_key)
    return out


def _lookup_url(lookup: dict[str, dict], *urls: object) -> dict:
    for url in urls:
        for key in _url_keys(url):
            row = lookup.get(key)
            if row is not None:
                return row
    return {}


def _keyword_gap_payload(projects: list[_Project]) -> dict:
    domains = [p.domain for p in projects]
    by_keyword: dict[str, dict[str, dict]] = {}

    for proj in projects:
        domain_best: dict[str, dict] = {}
        for row in ((proj.ahrefs or {}).get("organic_keywords") or []):
            keyword_key = _normalize_keyword(row.get("keyword"))
            if not keyword_key:
                continue
            payload = {
                "domain": proj.domain,
                "keyword": str(row.get("keyword") or keyword_key),
                "traffic": _safe_int(row.get("traffic")),
                "volume": _safe_int(row.get("volume")),
                "position": _safe_float(row.get("position"), 999.0),
                "url": row.get("matched_url") or row.get("url") or "",
                "source_url": row.get("url") or "",
                "page_title": row.get("page_title") or "",
                "section": row.get("section") or "",
                "cluster_label": row.get("cluster_label") or "",
                "country": row.get("country") or "",
                "intent": ", ".join(_string_list(row.get("intents"), "intent")[:3]),
                "intents": _string_list(row.get("intents"), "intent"),
                "serp_features": _string_list(row.get("serp_features"), "feature"),
            }
            current = domain_best.get(keyword_key)
            if current is None or (
                payload["traffic"],
                payload["volume"],
                -payload["position"],
            ) > (
                _safe_int(current.get("traffic")),
                _safe_int(current.get("volume")),
                -_safe_float(current.get("position"), 999.0),
            ):
                domain_best[keyword_key] = payload

        for keyword_key, row in domain_best.items():
            by_keyword.setdefault(keyword_key, {})[proj.domain] = row

    rows: list[dict] = []
    for keyword_key, domain_rows in by_keyword.items():
        present = list(domain_rows.values())
        if not present:
            continue
        leader = max(
            present,
            key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("volume")), -_safe_float(r.get("position"), 999.0)),
        )
        domain_entries = []
        for domain in domains:
            row = domain_rows.get(domain)
            if row:
                domain_entries.append(row)
            else:
                domain_entries.append({
                    "domain": domain,
                    "keyword": leader.get("keyword", keyword_key),
                    "traffic": 0,
                    "volume": _safe_int(leader.get("volume")),
                    "position": None,
                    "url": "",
                    "page_title": "",
                    "section": "",
                    "cluster_label": "",
                    "country": "",
                    "intent": leader.get("intent", ""),
                    "intents": leader.get("intents", []),
                    "serp_features": leader.get("serp_features", []),
                    "missing": True,
                })
        total_traffic = sum(_safe_int(row.get("traffic")) for row in present)
        rows.append({
            "keyword": leader.get("keyword", keyword_key),
            "keyword_key": keyword_key,
            "domain_count": len(present),
            "leader_domain": leader.get("domain", ""),
            "leader_traffic": _safe_int(leader.get("traffic")),
            "leader_position": leader.get("position"),
            "leader_url": leader.get("url", ""),
            "leader_title": leader.get("page_title", ""),
            "volume": max(_safe_int(row.get("volume")) for row in present),
            "total_traffic": total_traffic,
            "intent": leader.get("intent", ""),
            "serp_features": leader.get("serp_features", []),
            "missing_domains": [d for d in domains if d not in domain_rows],
            "domains": domain_entries,
        })

    rows.sort(key=lambda r: (_safe_int(r.get("total_traffic")), _safe_int(r.get("volume"))), reverse=True)

    opportunities: list[dict] = []
    for row in rows:
        leader_domain = row.get("leader_domain", "")
        leader_traffic = _safe_int(row.get("leader_traffic"))
        if not leader_domain or leader_traffic <= 0:
            continue
        for domain_row in row.get("domains", []):
            domain = domain_row.get("domain", "")
            if domain == leader_domain:
                continue
            own_traffic = _safe_int(domain_row.get("traffic"))
            own_position = domain_row.get("position")
            missing = bool(domain_row.get("missing"))
            underperforming = (
                not missing
                and leader_traffic >= max(10, own_traffic * 2)
                and _safe_float(row.get("leader_position"), 999.0) < _safe_float(own_position, 999.0)
            )
            if not missing and not underperforming:
                continue
            opportunities.append({
                "domain": domain,
                "keyword": row.get("keyword", ""),
                "gap_type": "missing" if missing else "underperforming",
                "competitor": leader_domain,
                "competitor_traffic": leader_traffic,
                "competitor_position": row.get("leader_position"),
                "competitor_url": row.get("leader_url", ""),
                "competitor_title": row.get("leader_title", ""),
                "own_traffic": own_traffic,
                "own_position": own_position,
                "volume": row.get("volume", 0),
                "intent": row.get("intent", ""),
                "serp_features": row.get("serp_features", []),
            })

    opportunities.sort(
        key=lambda r: (_safe_int(r.get("competitor_traffic")), _safe_int(r.get("volume"))),
        reverse=True,
    )

    return {
        "keywords": rows[:500],
        "opportunities": opportunities[:300],
        "exclusive": [r for r in rows if _safe_int(r.get("domain_count")) == 1][:200],
        "shared": [r for r in rows if _safe_int(r.get("domain_count")) > 1][:200],
    }


def _cluster_key(value: object) -> str:
    text = _normalize_keyword(value)
    return text or "unknown"


def _cluster_lookup_rows(rows: list[dict], label_fields: tuple[str, ...] = ("label", "cluster")) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        for field in label_fields:
            key = _cluster_key(row.get(field))
            if key and key != "unknown":
                out.setdefault(key, row)
    return out


def _semantic_cluster_samples(proj: _Project, cluster_id: object, top_url: str, entity_type: str, limit: int) -> list[str]:
    samples: list[str] = []
    top_keys = _url_keys(top_url)
    cluster_text = str(cluster_id if cluster_id is not None else "")
    for row in proj.ahrefs_semantic_rows or []:
        if row.get("type") != entity_type:
            continue
        same_cluster = cluster_text and str(row.get("cluster")) == cluster_text
        same_url = bool(top_keys and (_url_keys(row.get("url")) & top_keys))
        if not same_cluster and not same_url:
            continue
        label = str(row.get("label") or "").strip()
        if label and label not in samples:
            samples.append(label)
        if len(samples) >= limit:
            break
    return samples


def _entry_has_element(entry: dict, kind: str, label: str) -> bool:
    label_key = _normalize_keyword(label)
    if not label_key:
        return False
    if kind == "schema":
        return label in set(entry.get("schema_types") or [])
    if kind == "entity":
        return any(_normalize_keyword(v) == label_key for v in (entry.get("entities") or []))
    if kind == "heading":
        return any(_normalize_keyword(v) == label_key for v in (entry.get("headings") or []))
    if kind == "answer_block":
        return _safe_float(entry.get("answer_block_share")) >= 0.6
    if kind == "freshness":
        return entry.get("freshness_bucket") in {"fresh", "aging"}
    if kind == "link_support":
        return _safe_int(entry.get("in_degree")) >= 3
    if kind == "paragraph_archetype":
        return bool(entry.get("paragraph_examples"))
    return False


def _cluster_reasons(entry: dict) -> list[str]:
    reasons = []
    if _safe_int(entry.get("traffic")):
        reasons.append(f"{_safe_int(entry.get('traffic'))} estimated traffic")
    if _safe_int(entry.get("top3_keywords")):
        reasons.append(f"{_safe_int(entry.get('top3_keywords'))} top-3 keywords")
    if _safe_float(entry.get("answer_block_share")) >= 0.6:
        reasons.append("strong answer-block coverage")
    if entry.get("schema_types"):
        reasons.append("schema coverage")
    if _safe_int(entry.get("in_degree")) >= 3:
        reasons.append("internal link support")
    if entry.get("freshness_bucket") in {"fresh", "aging"}:
        reasons.append(f"{entry.get('freshness_bucket')} content")
    return reasons[:6]


def _project_cluster_entries(proj: _Project) -> dict[str, dict]:
    link_lookup = _page_link_lookup(proj)
    structured_lookup = _url_lookup_from_rows(((proj.structured_data or {}).get("per_page") or []), ("url",))
    freshness_lookup = _url_lookup_from_rows(((proj.freshness or {}).get("per_page") or []), ("url",))
    page_type_lookup = _url_lookup_from_rows(((proj.page_types or {}).get("per_page") or []), ("url",))
    answer_lookup = _cluster_lookup_rows(((proj.answer_blocks or {}).get("clusters") or []), ("label", "cluster"))
    info_lookup = _cluster_lookup_rows(((proj.information_gain or {}).get("clusters") or []), ("label", "cluster"))
    entity_lookup = _cluster_lookup_rows(((proj.entity_coverage or {}).get("clusters") or []), ("label", "cluster"))

    cluster_id_to_key: dict[str, str] = {}
    groups: dict[str, dict] = {}
    for row in ((proj.ahrefs or {}).get("clusters") or []):
        label = row.get("label") or row.get("key") or row.get("cluster")
        key = _cluster_key(label)
        if key == "unknown":
            continue
        cluster_id = row.get("cluster") if row.get("cluster") is not None else row.get("key")
        if cluster_id is not None:
            cluster_id_to_key[str(cluster_id)] = key
        top_pages = list(row.get("top_pages") or [])
        top_page = top_pages[0] if top_pages else {}
        groups[key] = {
            "domain": proj.domain,
            "cluster": key,
            "cluster_label": str(label or key),
            "cluster_id": cluster_id,
            "traffic": _safe_int(row.get("traffic")),
            "keyword_traffic": _safe_int(row.get("keyword_traffic")),
            "keywords_total": _safe_int(row.get("keywords_total") or row.get("keyword_rows")),
            "keyword_rows": _safe_int(row.get("keyword_rows")),
            "top3_keywords": _safe_int(row.get("top3_keywords")),
            "top10_keywords": _safe_int(row.get("top10_keywords")),
            "top20_keywords": _safe_int(row.get("top20_keywords")),
            "top_keywords": list(row.get("top_keywords") or [])[:10],
            "serp_features": _string_list(row.get("serp_features"), "feature")[:10],
            "intents": _string_list(row.get("intents"), "intent")[:8],
            "top_url": top_page.get("matched_url") or top_page.get("url") or "",
            "top_title": top_page.get("title") or top_page.get("page_title") or "",
            "owned_urls": [
                page.get("matched_url") or page.get("url") or ""
                for page in top_pages[:8]
                if page.get("matched_url") or page.get("url")
            ],
            "keyword_samples": [kw.get("keyword") for kw in (row.get("top_keywords") or [])[:8] if kw.get("keyword")],
        }

    keyword_rows_by_key: dict[str, list[dict]] = defaultdict(list)
    for row in ((proj.ahrefs or {}).get("organic_keywords") or []):
        cluster_ref = row.get("cluster")
        key = cluster_id_to_key.get(str(cluster_ref)) if cluster_ref is not None else ""
        key = key or _cluster_key(row.get("cluster_label") or row.get("keyword"))
        if key == "unknown":
            continue
        keyword_rows_by_key[key].append(row)
        group = groups.setdefault(key, {
            "domain": proj.domain,
            "cluster": key,
            "cluster_label": str(row.get("cluster_label") or key),
            "cluster_id": cluster_ref,
            "traffic": 0,
            "keyword_traffic": 0,
            "keywords_total": 0,
            "keyword_rows": 0,
            "top3_keywords": 0,
            "top10_keywords": 0,
            "top20_keywords": 0,
            "top_keywords": [],
            "serp_features": [],
            "intents": [],
            "top_url": "",
            "top_title": "",
            "owned_urls": [],
            "keyword_samples": [],
        })
        traffic = _safe_int(row.get("traffic"))
        if not group.get("top_url") or traffic > _safe_int(group.get("_top_url_traffic")):
            group["top_url"] = row.get("matched_url") or row.get("url") or ""
            group["top_title"] = row.get("page_title") or row.get("title") or ""
            group["_top_url_traffic"] = traffic
        if not group.get("traffic"):
            group["traffic"] += traffic
        group["keyword_traffic"] += traffic
        group["keyword_rows"] += 1
        group["keywords_total"] = max(_safe_int(group.get("keywords_total")), group["keyword_rows"])
        position = _safe_float(row.get("position"), 999.0)
        if position <= 3:
            group["top3_keywords"] += 1
        if position <= 10:
            group["top10_keywords"] += 1
        if position <= 20:
            group["top20_keywords"] += 1
        keyword = row.get("keyword")
        if keyword and keyword not in group["keyword_samples"] and len(group["keyword_samples"]) < 10:
            group["keyword_samples"].append(keyword)
        for feature in _string_list(row.get("serp_features"), "feature"):
            if feature not in group["serp_features"]:
                group["serp_features"].append(feature)
        for intent in _string_list(row.get("intents"), "intent"):
            if intent not in group["intents"]:
                group["intents"].append(intent)
        url = row.get("matched_url") or row.get("url") or ""
        if url and url not in group["owned_urls"] and len(group["owned_urls"]) < 8:
            group["owned_urls"].append(url)

    for key, group in groups.items():
        top_url = group.get("top_url") or (group.get("owned_urls") or [""])[0]
        structured = _lookup_url(structured_lookup, top_url)
        freshness = _lookup_url(freshness_lookup, top_url)
        page_type = _lookup_url(page_type_lookup, top_url)
        link = _lookup_url(link_lookup, top_url)
        answer = answer_lookup.get(key, {})
        info = info_lookup.get(key, {})
        entity = entity_lookup.get(key, {})
        cluster_id = group.get("cluster_id")
        group["top_url"] = top_url
        group["schema_types"] = _string_list(structured.get("types"), "type")
        group["page_type"] = page_type.get("page_type") or ""
        group["template_family"] = page_type.get("template_family") or ""
        group["freshness_bucket"] = freshness.get("bucket") or ""
        group["freshness_age_days"] = _safe_int(freshness.get("age_days"))
        group["in_degree"] = _safe_int(link.get("in_degree"))
        group["out_degree"] = _safe_int(link.get("out_degree"))
        group["click_depth"] = link.get("click_depth")
        group["answer_block_share"] = _safe_float(answer.get("strong_query_share"))
        group["answer_block_score"] = _safe_float(answer.get("avg_best_score"))
        group["answer_recommended_format"] = answer.get("recommended_format") or ""
        group["information_gain_score"] = _safe_float(info.get("avg_score"))
        group["entities"] = [
            e.get("entity")
            for e in (entity.get("expected_entities") or [])[:20]
            if e.get("entity")
        ]
        group["headings"] = _semantic_cluster_samples(proj, cluster_id, top_url, "header", 12)
        group["page_titles"] = _semantic_cluster_samples(proj, cluster_id, top_url, "page_title", 5)
        group["paragraph_examples"] = _semantic_cluster_samples(proj, cluster_id, top_url, "paragraph", 5)
        score = (
            _safe_int(group.get("traffic"))
            + _safe_int(group.get("keyword_rows")) * 5
            + _safe_int(group.get("top3_keywords")) * 20
            + _safe_float(group.get("answer_block_share")) * 90
            + len(group.get("schema_types") or []) * 8
            + min(12, _safe_int(group.get("in_degree"))) * 4
            + min(100.0, _safe_float(group.get("information_gain_score"))) * 0.35
        )
        group["coverage_score"] = round(float(score), 2)
        group["coverage_reasons"] = _cluster_reasons(group)
        group.pop("_top_url_traffic", None)
    return groups


def _keyword_cluster_gap_payload(projects: list[_Project]) -> dict:
    domains = [p.domain for p in projects]
    per_project = {proj.domain: _project_cluster_entries(proj) for proj in projects}
    cluster_keys = sorted({key for entries in per_project.values() for key in entries})
    clusters: list[dict] = []
    recommendations: list[dict] = []
    section_diffs: list[dict] = []
    cache_samples: list[dict] = []
    cache_entry_count = 0

    for proj in projects:
        meta = (proj.ahrefs or {}).get("meta", {}) or {}
        provider = meta.get("provider") or "search"
        cache_status = meta.get("cache_status") or ""
        for row in ((proj.ahrefs or {}).get("organic_keywords") or [])[:2000]:
            keyword = row.get("keyword")
            if not keyword:
                continue
            cache_entry_count += 1
            if len(cache_samples) < 25:
                raw_key = f"{proj.domain}|{provider}|{keyword}"
                cache_samples.append({
                    "domain": proj.domain,
                    "provider": provider,
                    "keyword": keyword,
                    "cache_status": cache_status,
                    "cache_key": hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16],
                })

    for key in cluster_keys:
        entries = []
        for domain in domains:
            entry = per_project.get(domain, {}).get(key)
            if entry:
                entries.append(entry)
            else:
                entries.append({
                    "domain": domain,
                    "cluster": key,
                    "cluster_label": key,
                    "coverage_status": "missing",
                    "traffic": 0,
                    "keyword_rows": 0,
                    "keywords_total": 0,
                    "top3_keywords": 0,
                    "top10_keywords": 0,
                    "coverage_score": 0.0,
                    "top_url": "",
                    "top_title": "",
                    "section": "new page",
                    "schema_types": [],
                    "entities": [],
                    "headings": [],
                    "paragraph_examples": [],
                    "serp_features": [],
                    "intents": [],
                    "coverage_reasons": [],
                })
        leader = max(entries, key=lambda r: (_safe_float(r.get("coverage_score")), _safe_int(r.get("traffic"))))
        if _safe_float(leader.get("coverage_score")) <= 0:
            continue

        leader_headings = list(dict.fromkeys(leader.get("headings") or []))[:8]
        leader_entities = list(dict.fromkeys(leader.get("entities") or []))[:10]
        leader_schema = list(dict.fromkeys(leader.get("schema_types") or []))[:8]
        leader_paragraphs = list(dict.fromkeys(leader.get("paragraph_examples") or []))[:4]

        missing_elements: list[dict] = []

        def register_gap(kind: str, label: str, domain_entry: dict, action: str, section: str) -> None:
            if not label:
                return
            competitor_domains = [e["domain"] for e in entries if _entry_has_element(e, kind, label)]
            missing_domains = [e["domain"] for e in entries if not _entry_has_element(e, kind, label)]
            prevalence = len(competitor_domains) / max(len(entries), 1)
            opportunity = max(0, _safe_int(leader.get("traffic")) - _safe_int(domain_entry.get("traffic")))
            missing_elements.append({
                "type": kind,
                "label": label,
                "competitor_domains": competitor_domains,
                "missing_domains": missing_domains,
                "prevalence": round(prevalence, 3),
                "traffic_opportunity": opportunity,
            })
            recommendations.append({
                "domain": domain_entry.get("domain"),
                "cluster": key,
                "cluster_label": leader.get("cluster_label") or key,
                "target_url": domain_entry.get("top_url") or "",
                "target_title": domain_entry.get("top_title") or f"Create page for {leader.get('cluster_label') or key}",
                "section": section,
                "missing_type": kind,
                "missing_element": label,
                "action": action,
                "competitor_domain": leader.get("domain"),
                "competitor_url": leader.get("top_url") or "",
                "competitor_title": leader.get("top_title") or "",
                "competitor_prevalence": round(prevalence, 3),
                "traffic_opportunity": opportunity,
                "leader_traffic": _safe_int(leader.get("traffic")),
                "own_traffic": _safe_int(domain_entry.get("traffic")),
            })

        for entry in entries:
            if entry.get("domain") == leader.get("domain"):
                continue
            missing_headings = [h for h in leader_headings if not _entry_has_element(entry, "heading", h)][:5]
            missing_entities = [e for e in leader_entities if not _entry_has_element(entry, "entity", e)][:5]
            missing_schema = [s for s in leader_schema if not _entry_has_element(entry, "schema", s)][:4]
            answer_gap = _safe_float(leader.get("answer_block_share")) >= 0.6 and _safe_float(entry.get("answer_block_share")) < 0.4
            freshness_gap = _entry_has_element(leader, "freshness", "fresh") and not _entry_has_element(entry, "freshness", "fresh")
            link_gap = _entry_has_element(leader, "link_support", "internal links") and not _entry_has_element(entry, "link_support", "internal links")
            paragraph_gap = bool(leader_paragraphs) and not entry.get("paragraph_examples")

            for heading in missing_headings[:3]:
                register_gap("heading", heading, entry, "Add or rename an H2/H3 section matching the competitor outline.", "Headings")
            for entity in missing_entities[:3]:
                register_gap("entity", entity, entry, "Add a paragraph or subsection that naturally covers this missing entity.", "Entities")
            for schema in missing_schema:
                register_gap("schema", schema, entry, "Add the schema type when the page content supports it.", "Schema")
            if answer_gap:
                register_gap("answer_block", "snippet-ready answer block", entry, "Add a concise answer block for the cluster's highest-demand questions.", "Answer block")
            if freshness_gap:
                register_gap("freshness", "fresh or recently updated evidence", entry, "Refresh the page and expose a visible updated/published date where appropriate.", "Freshness")
            if link_gap:
                register_gap("link_support", "strong internal inlinks", entry, "Promote the page from relevant hubs using descriptive anchors.", "Internal links")
            if paragraph_gap:
                register_gap("paragraph_archetype", leader_paragraphs[0][:160], entry, "Add a section with the same explanatory role, rewritten for this domain.", "Paragraphs")

            if missing_headings or missing_entities or missing_schema or answer_gap or freshness_gap or link_gap or paragraph_gap:
                section_diffs.append({
                    "domain": entry.get("domain"),
                    "cluster": key,
                    "cluster_label": leader.get("cluster_label") or key,
                    "target_url": entry.get("top_url") or "",
                    "target_title": entry.get("top_title") or f"Create page for {leader.get('cluster_label') or key}",
                    "competitor_domain": leader.get("domain"),
                    "competitor_url": leader.get("top_url") or "",
                    "competitor_title": leader.get("top_title") or "",
                    "missing_headings": missing_headings,
                    "missing_entities": missing_entities,
                    "missing_schema": missing_schema,
                    "answer_block_gap": answer_gap,
                    "freshness_gap": freshness_gap,
                    "link_gap": link_gap,
                    "paragraph_examples": leader_paragraphs,
                    "traffic_opportunity": max(0, _safe_int(leader.get("traffic")) - _safe_int(entry.get("traffic"))),
                })

        traffic_opportunity = sum(max(0, _safe_int(leader.get("traffic")) - _safe_int(e.get("traffic"))) for e in entries if e.get("domain") != leader.get("domain"))
        clusters.append({
            "cluster": key,
            "cluster_label": leader.get("cluster_label") or key,
            "leader_domain": leader.get("domain"),
            "leader_url": leader.get("top_url") or "",
            "leader_title": leader.get("top_title") or "",
            "leader_score": _safe_float(leader.get("coverage_score")),
            "leader_reasons": leader.get("coverage_reasons") or [],
            "traffic_opportunity": traffic_opportunity,
            "missing_elements": sorted(missing_elements, key=lambda r: (_safe_int(r.get("traffic_opportunity")), _safe_float(r.get("prevalence"))), reverse=True)[:30],
            "domains": entries,
        })

    clusters.sort(key=lambda r: (_safe_int(r.get("traffic_opportunity")), _safe_float(r.get("leader_score"))), reverse=True)
    recommendations.sort(key=lambda r: (_safe_int(r.get("traffic_opportunity")), _safe_float(r.get("competitor_prevalence"))), reverse=True)
    section_diffs.sort(key=lambda r: _safe_int(r.get("traffic_opportunity")), reverse=True)
    return {
        "summary": {
            "clusters": len(clusters),
            "recommendations": len(recommendations),
            "section_diffs": len(section_diffs),
            "cache_entries": cache_entry_count,
            "cache_status": "derived_from_cached_provider_snapshots",
        },
        "clusters": clusters[:160],
        "recommendations": recommendations[:400],
        "section_diffs": section_diffs[:250],
        "cache": {
            "status": "derived_from_cached_provider_snapshots",
            "description": "No per-keyword provider calls are made here; rows are derived from cached Ahrefs/DataForSEO domain snapshots and keyed by domain, keyword, and provider.",
            "entries": cache_entry_count,
            "samples": cache_samples,
        },
    }


def _serp_feature_payload(projects: list[_Project]) -> dict:
    by_feature: dict[str, dict[str, dict]] = {}
    for proj in projects:
        for row in ((proj.ahrefs or {}).get("organic_keywords") or []):
            keyword = str(row.get("keyword") or "").strip()
            traffic = _safe_int(row.get("traffic"))
            for feature in _string_list(row.get("serp_features"), "feature"):
                feature_rows = by_feature.setdefault(feature, {})
                stats = feature_rows.setdefault(proj.domain, {
                    "domain": proj.domain,
                    "feature": feature,
                    "count": 0,
                    "traffic": 0,
                    "sample_keywords": [],
                })
                stats["count"] += 1
                stats["traffic"] += traffic
                if keyword and keyword not in stats["sample_keywords"] and len(stats["sample_keywords"]) < 6:
                    stats["sample_keywords"].append(keyword)

    flat: list[dict] = []
    matrix: list[dict] = []
    domains = [p.domain for p in projects]
    for feature, domain_rows in by_feature.items():
        domain_values = []
        total_count = 0
        total_traffic = 0
        for domain in domains:
            stats = domain_rows.get(domain, {
                "domain": domain,
                "feature": feature,
                "count": 0,
                "traffic": 0,
                "sample_keywords": [],
            })
            total_count += _safe_int(stats.get("count"))
            total_traffic += _safe_int(stats.get("traffic"))
            domain_values.append(stats)
            flat.append(stats)
        matrix.append({
            "feature": feature,
            "total_count": total_count,
            "total_traffic": total_traffic,
            "domains": domain_values,
        })

    matrix.sort(key=lambda r: (_safe_int(r.get("total_traffic")), _safe_int(r.get("total_count"))), reverse=True)
    flat.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("count"))), reverse=True)
    return {"matrix": matrix[:60], "features": flat[:300]}


def _efficiency_payload(projects: list[_Project]) -> dict:
    def rows_for(source_key: str, entity_type: str) -> list[dict]:
        rows: list[dict] = []
        for proj in projects:
            for row in ((proj.ahrefs or {}).get(source_key) or []):
                pages = max(1, _safe_int(row.get("pages") or row.get("page_count") or row.get("matched_pages") or 0))
                matched_pages = max(0, _safe_int(row.get("matched_pages")))
                traffic = _safe_int(row.get("traffic"))
                keyword_rows = _safe_int(row.get("keyword_rows"))
                keywords_total = _safe_int(row.get("keywords_total") or keyword_rows)
                top3 = _safe_int(row.get("top3_keywords"))
                label = str(row.get("label") or row.get("key") or "root")
                rows.append({
                    "domain": proj.domain,
                    "type": entity_type,
                    "key": str(row.get("key") or label),
                    "label": label,
                    "traffic": traffic,
                    "keyword_traffic": _safe_int(row.get("keyword_traffic")),
                    "value_usd": _safe_float(row.get("value_usd")),
                    "pages": pages,
                    "matched_pages": matched_pages,
                    "keyword_rows": keyword_rows,
                    "keywords_total": keywords_total,
                    "top3_keywords": top3,
                    "top10_keywords": _safe_int(row.get("top10_keywords")),
                    "top20_keywords": _safe_int(row.get("top20_keywords")),
                    "traffic_per_page": traffic / pages,
                    "traffic_per_matched_page": traffic / max(1, matched_pages or pages),
                    "keywords_per_page": keywords_total / pages,
                    "top3_per_page": top3 / pages,
                    "top_keywords": (row.get("top_keywords") or [])[:8],
                    "serp_features": (row.get("serp_features") or [])[:8],
                    "intents": (row.get("intents") or [])[:8],
                })
        rows.sort(key=lambda r: (_safe_float(r.get("traffic_per_page")), _safe_int(r.get("traffic"))), reverse=True)
        return rows[:250]

    return {
        "clusters": rows_for("clusters", "cluster"),
        "directories": rows_for("directories", "directory"),
    }


def _page_link_lookup(proj: _Project) -> dict[str, dict]:
    return _url_lookup_from_rows(proj.page_link_counts or [], ("url",))


def _authority_lookup(proj: _Project) -> dict[str, dict]:
    rows = (((proj.linkgraph or {}).get("traffic_weighted_pagerank") or {}).get("pages") or [])
    if not rows:
        rows = (proj.linkgraph or {}).get("top_authority_pages") or []
    return _url_lookup_from_rows(rows, ("url",), score_key="pagerank")


def _authority_demand_payload(projects: list[_Project]) -> dict:
    pages: list[dict] = []
    ranked_orphans: list[dict] = []
    buried_demand: list[dict] = []
    thin_internal_support: list[dict] = []
    unmatched_pages: list[dict] = []
    authority_without_demand: list[dict] = []

    for proj in projects:
        link_lookup = _page_link_lookup(proj)
        authority_lookup = _authority_lookup(proj)
        top_pages = (proj.ahrefs or {}).get("top_pages") or []
        top_page_lookup = _url_lookup_from_rows(top_pages, ("matched_url", "url"), score_key="traffic")
        in_degree_values = [
            float(_safe_int(row.get("in_degree")))
            for row in (proj.page_link_counts or [])
            if row.get("in_degree") is not None
        ]
        low_link_threshold = max(1, int(_percentile(in_degree_values, 0.25))) if in_degree_values else 1
        traffic_values = [_safe_int(row.get("traffic")) for row in top_pages if _safe_int(row.get("traffic")) > 0]
        traffic_median = _percentile([float(v) for v in traffic_values], 0.5) if traffic_values else 0.0

        project_rows: list[dict] = []
        for row in top_pages:
            url = row.get("matched_url") or row.get("url") or ""
            if not url:
                continue
            link = _lookup_url(link_lookup, row.get("matched_url"), row.get("url"))
            authority = _lookup_url(authority_lookup, row.get("matched_url"), row.get("url"))
            linkgraph_found = bool(link)
            click_depth = link.get("click_depth") if linkgraph_found else None
            out = {
                "domain": proj.domain,
                "url": url,
                "source_url": row.get("url") or "",
                "title": row.get("title") or row.get("top_keyword_title") or "",
                "section": row.get("section") or "",
                "cluster_label": row.get("cluster_label") or "",
                "traffic": _safe_int(row.get("traffic")),
                "keywords": _safe_int(row.get("keywords")),
                "top_keyword": row.get("top_keyword") or "",
                "top_keyword_position": row.get("top_keyword_position"),
                "referring_domains": _safe_int(row.get("referring_domains")),
                "url_rating": _safe_float(row.get("url_rating")),
                "in_degree": _safe_int(link.get("in_degree")) if linkgraph_found else 0,
                "out_degree": _safe_int(link.get("out_degree")) if linkgraph_found else 0,
                "click_depth": click_depth,
                "linkgraph_found": linkgraph_found,
                "pagerank": _safe_float(authority.get("pagerank")),
                "weighted_pagerank": _safe_float(authority.get("weighted_pagerank")),
                "traffic_weighted_pagerank": _safe_float(authority.get("traffic_weighted_pagerank")),
                "traffic_percentile": _safe_float(authority.get("traffic_percentile")),
                "weighted_pagerank_percentile": _safe_float(authority.get("weighted_pagerank_percentile")),
                "authority_traffic_gap": _safe_float(authority.get("authority_traffic_gap")),
                "mismatch_label": authority.get("mismatch_label") or "",
                "authority_score": _safe_float(authority.get("authority_score")),
                "hub_score": _safe_float(authority.get("hub_score")),
            }
            pages.append(out)
            project_rows.append(out)

        for row in project_rows:
            if _safe_int(row.get("traffic")) <= 0:
                continue
            if not row.get("linkgraph_found"):
                unmatched_pages.append(row)
            elif _safe_int(row.get("in_degree")) <= 0:
                ranked_orphans.append(row)
            elif row.get("click_depth") is not None and _safe_int(row.get("click_depth")) >= 4:
                buried_demand.append(row)
            if (
                row.get("linkgraph_found")
                and _safe_int(row.get("traffic")) >= traffic_median
                and _safe_int(row.get("in_degree")) <= low_link_threshold
            ):
                thin_internal_support.append(row)

        positive_search = [r for r in project_rows if _safe_int(r.get("traffic")) > 0]
        low_search_threshold = _percentile(
            [float(_safe_int(r.get("traffic"))) for r in positive_search],
            0.25,
        ) if positive_search else 0.0
        authority_rows = (((proj.linkgraph or {}).get("traffic_weighted_pagerank") or {}).get("pages") or [])
        if not authority_rows:
            authority_rows = (proj.linkgraph or {}).get("top_authority_pages") or []
        for row in authority_rows:
            search_row = _lookup_url(top_page_lookup, row.get("url"))
            traffic = _safe_int(row.get("traffic") if row.get("traffic") is not None else search_row.get("traffic"))
            if traffic > low_search_threshold:
                continue
            authority_without_demand.append({
                "domain": proj.domain,
                "url": row.get("url") or "",
                "title": row.get("title") or search_row.get("title") or "",
                "section": row.get("section") or search_row.get("section") or "",
                "traffic": traffic,
                "keywords": _safe_int(row.get("keywords") if row.get("keywords") is not None else search_row.get("keywords")),
                "top_keyword": row.get("top_keyword") or search_row.get("top_keyword") or "",
                "in_degree": _safe_int(row.get("in_degree")),
                "out_degree": _safe_int(row.get("out_degree")),
                "click_depth": row.get("click_depth"),
                "pagerank": _safe_float(row.get("pagerank")),
                "weighted_pagerank": _safe_float(row.get("weighted_pagerank")),
                "traffic_weighted_pagerank": _safe_float(row.get("traffic_weighted_pagerank")),
                "authority_traffic_gap": _safe_float(row.get("authority_traffic_gap")),
                "mismatch_label": row.get("mismatch_label") or "",
                "authority_score": _safe_float(row.get("authority_score")),
                "hub_score": _safe_float(row.get("hub_score")),
            })

    pages.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("pagerank"))), reverse=True)
    ranked_orphans.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
    buried_demand.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("click_depth"))), reverse=True)
    thin_internal_support.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
    unmatched_pages.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
    authority_without_demand.sort(key=lambda r: _safe_float(r.get("pagerank")), reverse=True)

    return {
        "pages": pages[:700],
        "ranked_orphans": ranked_orphans[:150],
        "buried_demand": buried_demand[:150],
        "thin_internal_support": thin_internal_support[:150],
        "unmatched_pages": unmatched_pages[:150],
        "authority_without_demand": authority_without_demand[:150],
    }


def _traffic_weighted_pagerank_comparison(projects: list[_Project]) -> dict:
    domains = []
    pages: list[dict] = []
    underserved: list[dict] = []
    authority_waste: list[dict] = []
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("traffic_weighted_pagerank") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("pages") or []:
            item = {"domain": proj.domain, **row}
            pages.append(item)
            if row.get("mismatch_label") in {"high_traffic_low_authority", "ranked_orphan", "buried_demand"}:
                underserved.append(item)
            if row.get("mismatch_label") == "high_authority_low_value":
                authority_waste.append(item)
        for row in payload.get("clusters") or []:
            cluster = str(row.get("cluster") or row.get("label") or "unknown")
            clusters[cluster][proj.domain] = {"domain": proj.domain, **row}

    project_domains = [p.domain for p in projects]
    matrix = []
    for cluster, values in clusters.items():
        matrix.append({
            "cluster": cluster,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "cluster": cluster,
                    "label": cluster,
                    "pages": 0,
                    "traffic": 0,
                    "avg_authority_traffic_gap": 0.0,
                    "underserved_pages": 0,
                    "authority_without_demand": 0,
                })
                for domain in project_domains
            ],
        })
    pages.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("authority_traffic_gap"))), reverse=True)
    underserved.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("authority_traffic_gap"))), reverse=True)
    authority_waste.sort(key=lambda r: (_safe_float(r.get("weighted_pagerank_percentile")), -_safe_int(r.get("traffic"))), reverse=True)
    matrix.sort(key=lambda r: sum(_safe_int(d.get("traffic")) for d in r["domains"]), reverse=True)
    return {
        "domains": domains,
        "pages": pages[:900],
        "underserved": underserved[:250],
        "authority_without_demand": authority_waste[:250],
        "clusters": matrix[:120],
    }


def _link_removal_comparison(projects: list[_Project]) -> dict:
    domains = []
    critical: list[dict] = []
    weak: list[dict] = []
    warnings: list[dict] = []
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("link_removal_simulation") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("critical_links") or []:
            critical.append({"domain": proj.domain, **row})
        for row in payload.get("weak_or_harmful_links") or []:
            weak.append({"domain": proj.domain, **row})
        for row in payload.get("edit_warnings") or []:
            warnings.append({"domain": proj.domain, **row})
    critical.sort(key=lambda r: _safe_float(r.get("removal_loss_score")), reverse=True)
    weak.sort(key=lambda r: _safe_float(r.get("removal_loss_score")), reverse=True)
    warnings.sort(key=lambda r: (_safe_float(r.get("max_loss_score")), _safe_int(r.get("critical_links"))), reverse=True)
    return {
        "domains": domains,
        "critical_links": critical[:300],
        "weak_or_harmful_links": weak[:300],
        "edit_warnings": warnings[:200],
    }


def _link_addition_comparison(projects: list[_Project]) -> dict:
    domains = []
    recommendations: list[dict] = []
    patterns: dict[str, dict[str, dict]] = defaultdict(dict)
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("link_addition_simulation") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("recommendations") or []:
            recommendations.append({"domain": proj.domain, **row})
        for row in payload.get("patterns") or []:
            pattern = row.get("pattern") or ""
            if pattern:
                patterns[pattern][proj.domain] = {"domain": proj.domain, **row}
    project_domains = [p.domain for p in projects]
    matrix = []
    for pattern, values in patterns.items():
        matrix.append({
            "pattern": pattern,
            "domains": [values.get(domain, {"domain": domain, "pattern": pattern, "count": 0}) for domain in project_domains],
        })
    recommendations.sort(key=lambda r: _safe_float(r.get("expected_benefit_score")), reverse=True)
    matrix.sort(key=lambda r: sum(_safe_int(d.get("count")) for d in r["domains"]), reverse=True)
    return {
        "domains": domains,
        "recommendations": recommendations[:400],
        "patterns": matrix[:100],
    }


def _anchor_relevance_comparison(projects: list[_Project]) -> dict:
    domains = []
    weak_links: list[dict] = []
    directories: dict[str, dict[str, dict]] = defaultdict(dict)
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("anchor_relevance") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("weak_links") or []:
            weak_links.append({"domain": proj.domain, **row})
        for row in payload.get("by_target_directory") or []:
            directory = row.get("target_directory") or row.get("label") or "unknown"
            directories[directory][proj.domain] = {"domain": proj.domain, **row}
    project_domains = [p.domain for p in projects]
    matrix = []
    for directory, values in directories.items():
        matrix.append({
            "directory": directory,
            "domains": [values.get(domain, {"domain": domain, "target_directory": directory, "links": 0, "avg_score": 0.0, "descriptive_rate": 0.0, "weak_links": 0}) for domain in project_domains],
        })
    weak_links.sort(key=lambda r: _safe_float(r.get("score")))
    matrix.sort(key=lambda r: sum(_safe_int(d.get("weak_links")) for d in r["domains"]), reverse=True)
    return {"domains": domains, "weak_links": weak_links[:400], "directories": matrix[:120]}


def _contextual_link_comparison(projects: list[_Project]) -> dict:
    domains = []
    top_links: list[dict] = []
    source_pages: list[dict] = []
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("contextual_link_impact") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("top_contextual_links") or []:
            top_links.append({"domain": proj.domain, **row})
        for row in payload.get("source_pages") or []:
            source_pages.append({"domain": proj.domain, **row})
    top_links.sort(key=lambda r: _safe_float(r.get("contextual_link_impact")), reverse=True)
    source_pages.sort(key=lambda r: _safe_float(r.get("avg_contextual_impact")), reverse=True)
    return {"domains": domains, "top_contextual_links": top_links[:400], "source_pages": source_pages[:250]}


def _hub_bottleneck_comparison(projects: list[_Project]) -> dict:
    domains = []
    pages: list[dict] = []
    cluster_edges: list[dict] = []
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("hub_bottlenecks") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("risks") or []:
            pages.append({"domain": proj.domain, **row})
        for row in payload.get("cluster_edges") or []:
            cluster_edges.append({"domain": proj.domain, **row})
    pages.sort(key=lambda r: _safe_float(r.get("resilience_risk")), reverse=True)
    cluster_edges.sort(key=lambda r: _safe_int(r.get("bridge_pages")), reverse=True)
    return {"domains": domains, "risks": pages[:400], "cluster_edges": cluster_edges[:300]}


def _high_demand_low_link_comparison(projects: list[_Project]) -> dict:
    domains = []
    pages: list[dict] = []
    opportunities: list[dict] = []
    directories: dict[str, dict[str, dict]] = defaultdict(dict)
    clusters: dict[str, dict[str, dict]] = defaultdict(dict)
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("high_demand_low_link") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        for row in payload.get("pages") or []:
            item = {"domain": proj.domain, **row}
            pages.append(item)
            if row.get("classification") in {"high_demand_low_support", "demand_support_gap"}:
                opportunities.append(item)
        for row in payload.get("directories") or []:
            directory = row.get("directory") or row.get("label") or "unknown"
            directories[directory][proj.domain] = {"domain": proj.domain, **row}
        for row in payload.get("clusters") or []:
            cluster = row.get("cluster") or row.get("label") or "unknown"
            clusters[cluster][proj.domain] = {"domain": proj.domain, **row}

    project_domains = [p.domain for p in projects]

    def matrix_rows(groups: dict[str, dict[str, dict]], key: str) -> list[dict]:
        rows = []
        for name, values in groups.items():
            rows.append({
                key: name,
                "label": name,
                "domains": [
                    values.get(domain, {
                        "domain": domain,
                        key: name,
                        "label": name,
                        "pages": 0,
                        "classified_top_pages": 0,
                        "traffic": 0,
                        "opportunities": 0,
                        "opportunity_traffic": 0,
                        "avg_demand_score": 0.0,
                        "avg_support_score": 0.0,
                        "avg_demand_support_gap": 0.0,
                    })
                    for domain in project_domains
                ],
            })
        rows.sort(key=lambda r: sum(_safe_int(d.get("opportunity_traffic")) for d in r["domains"]), reverse=True)
        return rows

    pages.sort(key=lambda r: (_safe_float(r.get("opportunity_score")), _safe_int(r.get("traffic"))), reverse=True)
    opportunities.sort(key=lambda r: (_safe_float(r.get("opportunity_score")), _safe_int(r.get("traffic"))), reverse=True)
    return {
        "domains": domains,
        "pages": pages[:900],
        "opportunities": opportunities[:400],
        "directories": matrix_rows(directories, "directory")[:120],
        "clusters": matrix_rows(clusters, "cluster")[:120],
    }


def _internal_link_patterns_comparison(projects: list[_Project]) -> dict:
    domains = []
    rules: dict[str, dict[str, dict]] = defaultdict(dict)
    examples: list[dict] = []
    recommendations: list[dict] = []
    for proj in projects:
        payload = ((proj.linkgraph or {}).get("internal_link_patterns") or {})
        summary = payload.get("summary", {}) or {}
        if summary:
            domains.append({"domain": proj.domain, **summary})
        buckets: dict[str, dict] = defaultdict(lambda: {
            "label": "",
            "count": 0,
            "confidence_sum": 0.0,
            "support_sum": 0,
            "recommendations": 0,
        })
        rec_count_by_pattern = Counter(
            row.get("pattern_id") for row in payload.get("recommendations") or [] if row.get("pattern_id")
        )
        for row in payload.get("patterns") or []:
            key = row.get("rule_key") or row.get("inferred_rule") or "unknown"
            bucket = buckets[key]
            bucket["label"] = row.get("inferred_rule") or key
            bucket["count"] += 1
            bucket["confidence_sum"] += _safe_float(row.get("confidence"))
            bucket["support_sum"] += _safe_int(row.get("support_count"))
            bucket["recommendations"] += rec_count_by_pattern.get(row.get("pattern_id"), 0)
            examples.append({"domain": proj.domain, **row})
        for row in payload.get("recommendations") or []:
            recommendations.append({"domain": proj.domain, **row})
        for key, values in buckets.items():
            count = max(1, values["count"])
            rules[key][proj.domain] = {
                "domain": proj.domain,
                "rule_key": key,
                "label": values["label"],
                "count": values["count"],
                "avg_confidence": values["confidence_sum"] / count,
                "support_count": values["support_sum"],
                "recommendations": values["recommendations"],
            }

    project_domains = [p.domain for p in projects]
    matrix = []
    for rule_key, values in rules.items():
        label = next((v.get("label") for v in values.values() if v.get("label")), rule_key)
        matrix.append({
            "rule_key": rule_key,
            "label": label,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "rule_key": rule_key,
                    "label": label,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "support_count": 0,
                    "recommendations": 0,
                })
                for domain in project_domains
            ],
        })
    matrix.sort(
        key=lambda r: (
            sum(_safe_int(d.get("count")) for d in r["domains"]),
            sum(_safe_int(d.get("support_count")) for d in r["domains"]),
            sum(_safe_float(d.get("avg_confidence")) for d in r["domains"]),
        ),
        reverse=True,
    )
    examples.sort(key=lambda r: (_safe_float(r.get("confidence")), _safe_int(r.get("support_count"))), reverse=True)
    recommendations.sort(key=lambda r: (_safe_float(r.get("confidence")), _safe_float(r.get("lift_score_difference"))), reverse=True)
    return {
        "domains": domains,
        "rules": matrix[:140],
        "examples": examples[:250],
        "recommendations": recommendations[:300],
    }


def _readiness_payload(projects: list[_Project]) -> dict:
    rows: list[dict] = []
    weak_high_traffic: list[dict] = []

    for proj in projects:
        answer_lookup = _url_lookup_from_rows(proj.answerability or [], ("url",))
        structured_lookup = _url_lookup_from_rows(((proj.structured_data or {}).get("per_page") or []), ("url",))
        metadata_lookup = _url_lookup_from_rows(((proj.metadata_quality or {}).get("per_page") or []), ("url",))
        freshness_lookup = _freshness_lookup(proj)
        conversion_lookup = _url_lookup_from_rows(((proj.conversion or {}).get("per_page") or []), ("url",))
        link_lookup = _page_link_lookup(proj)

        for top in ((proj.ahrefs or {}).get("top_pages") or []):
            url = top.get("matched_url") or top.get("url") or ""
            if not url:
                continue
            source_url = top.get("url") or ""
            answer = _lookup_url(answer_lookup, top.get("matched_url"), source_url)
            structured = _lookup_url(structured_lookup, top.get("matched_url"), source_url)
            metadata = _lookup_url(metadata_lookup, top.get("matched_url"), source_url)
            freshness = _lookup_url(freshness_lookup, top.get("matched_url"), source_url)
            conversion = _lookup_url(conversion_lookup, top.get("matched_url"), source_url)
            link = _lookup_url(link_lookup, top.get("matched_url"), source_url)

            issues: list[str] = []
            geo_score = _safe_float(answer.get("score"))
            geo_points = min(25.0, max(0.0, geo_score) / 10.0 * 25.0)
            if geo_score < 4:
                issues.append("low_geo_score")

            schema_types = _string_list(structured.get("types"))
            valid_schema = _safe_int(structured.get("valid_blocks"))
            invalid_schema = _safe_int(structured.get("invalid_blocks"))
            schema_points = 15.0 if valid_schema else (8.0 if schema_types else 0.0)
            schema_points = max(0.0, schema_points - min(8.0, invalid_schema * 4.0))
            if not valid_schema and not schema_types:
                issues.append("missing_schema")
            if invalid_schema:
                issues.append("invalid_schema")

            metadata_issues = _string_list(metadata.get("issues"))
            metadata_points = max(0.0, 15.0 - len(metadata_issues) * 3.0)
            issues.extend(f"metadata:{issue}" for issue in metadata_issues[:4])

            bucket = freshness.get("bucket") or "unknown"
            freshness_points = {
                "fresh": 15.0,
                "aging": 12.0,
                "stale": 7.0,
                "very_stale": 2.0,
                "future": 8.0,
                "unknown": 0.0,
            }.get(str(bucket), 0.0)
            freshness_issues = _string_list(freshness.get("issues"))
            issues.extend(f"freshness:{issue}" for issue in freshness_issues[:3])

            primary_ctas = _safe_int(conversion.get("primary_cta_count"))
            ctas = _safe_int(conversion.get("cta_count"))
            forms = _safe_int(conversion.get("form_count"))
            contact_links = _safe_int(conversion.get("contact_link_count"))
            conversion_points = 15.0 if (primary_ctas or forms or contact_links) else (9.0 if ctas else 0.0)
            conversion_issues = _string_list(conversion.get("issues"))
            conversion_points = max(0.0, conversion_points - min(6.0, len(conversion_issues) * 2.0))
            issues.extend(f"conversion:{issue}" for issue in conversion_issues[:3])
            if _safe_int(conversion.get("lead_page")) and not (forms or contact_links):
                issues.append("lead_without_capture")
            if not (primary_ctas or forms or contact_links or ctas):
                issues.append("no_clear_cta")

            link_found = bool(link)
            in_degree = _safe_int(link.get("in_degree")) if link_found else 0
            click_depth = link.get("click_depth") if link_found else None
            if not link_found:
                internal_points = 0.0
                issues.append("not_matched_in_crawl")
            elif in_degree >= 3 and (click_depth is None or _safe_int(click_depth) <= 3):
                internal_points = 15.0
            elif in_degree > 0:
                internal_points = 8.0
            else:
                internal_points = 3.0
                issues.append("no_internal_inlinks")
            if click_depth is not None and _safe_int(click_depth) >= 4:
                issues.append("deep_click_depth")

            readiness_score = round(
                geo_points
                + schema_points
                + metadata_points
                + freshness_points
                + conversion_points
                + internal_points,
                1,
            )
            row = {
                "domain": proj.domain,
                "url": url,
                "source_url": source_url,
                "title": top.get("title") or metadata.get("title") or "",
                "section": top.get("section") or "",
                "cluster_label": top.get("cluster_label") or "",
                "traffic": _safe_int(top.get("traffic")),
                "keywords": _safe_int(top.get("keywords")),
                "top_keyword": top.get("top_keyword") or "",
                "geo_score": geo_score,
                "schema_types": schema_types,
                "valid_schema_blocks": valid_schema,
                "invalid_schema_blocks": invalid_schema,
                "metadata_issues": metadata_issues,
                "freshness_bucket": bucket,
                "freshness_age_days": freshness.get("age_days"),
                "cta_count": ctas,
                "primary_cta_count": primary_ctas,
                "form_count": forms,
                "contact_link_count": contact_links,
                "lead_page": bool(conversion.get("lead_page")),
                "in_degree": in_degree,
                "click_depth": click_depth,
                "readiness_score": readiness_score,
                "issues": issues[:12],
            }
            rows.append(row)
            if _safe_int(row.get("traffic")) > 0 and readiness_score < 70:
                weak_high_traffic.append(row)

    rows.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_float(r.get("readiness_score"))), reverse=True)
    weak_high_traffic.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_float(r.get("readiness_score"))), reverse=True)
    return {
        "top_pages": rows[:500],
        "weak_high_traffic": weak_high_traffic[:200],
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
    comparison_metrics = _comparison_metric_payload(leaderboard)

    LOG.info("  building combined UMAP across %d domains", len(projects))
    scatter_rows, n_total = _combined_umap(projects)
    LOG.info("  combined UMAP: %d points projected", n_total)
    ahrefs_semantic_rows, ahrefs_semantic_total = _combined_ahrefs_semantic_umap(projects)
    if ahrefs_semantic_total:
        LOG.info("  Ahrefs semantic UMAP: %d points projected", ahrefs_semantic_total)
    semantic_entity_maps = _semantic_entity_maps(projects)
    for key, entity_map in semantic_entity_maps.items():
        if entity_map.get("total"):
            LOG.info("  semantic %s UMAP: %d points projected", key, entity_map["total"])

    return {
        "domains": [p.domain for p in projects],
        "leaderboard": leaderboard,
        "scatter": scatter_rows,
        "scatter_total": n_total,
        "ahrefs_semantic_scatter": ahrefs_semantic_rows,
        "ahrefs_semantic_total": ahrefs_semantic_total,
        "semantic_entity_maps": semantic_entity_maps,
        "search": _search_payload(projects),
        "entity_alignment": _entity_alignment_comparison(projects),
        "entity_coverage": _entity_coverage_comparison(projects),
        "information_gain": _information_gain_comparison(projects),
        "structured_data_opportunities": _structured_data_opportunity_comparison(projects),
        "trust_signals": _trust_signal_comparison(projects),
        "conversion_balance": _conversion_balance_comparison(projects),
        "answer_blocks": _answer_blocks_comparison(projects),
        "freshness_impact": _freshness_impact_comparison(projects),
        "cannibalization": _cannibalization_comparison(projects),
        "duplicate_fragments": _duplicate_fragments_comparison(projects),
        "template_patterns": _template_patterns_comparison(projects),
        "keyword_gaps": _keyword_gap_payload(projects),
        "keyword_cluster_gaps": _keyword_cluster_gap_payload(projects),
        "serp_features": _serp_feature_payload(projects),
        "content_efficiency": _efficiency_payload(projects),
        "authority_demand": _authority_demand_payload(projects),
        "traffic_weighted_pagerank": _traffic_weighted_pagerank_comparison(projects),
        "link_removal_simulation": _link_removal_comparison(projects),
        "link_addition_simulation": _link_addition_comparison(projects),
        "anchor_relevance": _anchor_relevance_comparison(projects),
        "contextual_link_impact": _contextual_link_comparison(projects),
        "high_demand_low_link": _high_demand_low_link_comparison(projects),
        "internal_link_patterns": _internal_link_patterns_comparison(projects),
        "hub_bottlenecks": _hub_bottleneck_comparison(projects),
        "traffic_readiness": _readiness_payload(projects),
        "link_flows": [
            {"domain": p.domain, **(p.link_flow or {})}
            for p in projects
            if (p.link_flow or {}).get("nodes") and (p.link_flow or {}).get("edges")
        ],
        "distributions": distributions,
        **comparison_metrics,
    }


def write_html(template_path: Path, payload: dict, out_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("__COMPARE_JSON__", json.dumps(payload, separators=(",", ":")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")


def package_comparison(
    out_dir: Path,
    projects_root: Path,
    domains: list[str],
    package_name: str = COMPARISON_PACKAGE_NAME,
) -> Path:
    """Create a customer-shareable ZIP for a comparison.

    The package contains the comparison presentation files plus the generated
    report files for each compared domain. It deliberately excludes caches,
    embedding files, and other intermediate project data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / package_name
    if zip_path.exists():
        zip_path.unlink()

    manifest = {
        "comparison": out_dir.name,
        "domains": domains,
        "open": "index.html",
        "domain_reports": [f"domains/{domain}/index.html" for domain in domains],
    }
    readme = (
        "Site Audit comparison package\n\n"
        "Open index.html to view the cross-domain comparison.\n"
        "Individual domain reports are under domains/<domain>/index.html.\n"
        "Only generated presentation/report files are included; crawl caches and "
        "embedding caches are intentionally excluded.\n"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

        for file_path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            if file_path == zip_path:
                continue
            zf.write(file_path, file_path.relative_to(out_dir).as_posix())

        for domain in domains:
            report_dir = projects_root / domain / "report"
            if not report_dir.is_dir():
                LOG.warning("  skip %s in comparison package: no report dir", domain)
                continue
            for file_path in sorted(p for p in report_dir.rglob("*") if p.is_file()):
                arcname = Path("domains") / domain / file_path.relative_to(report_dir)
                zf.write(file_path, arcname.as_posix())

    return zip_path
