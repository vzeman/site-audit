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
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .ahrefs import load_semantic_cache
from .best_pages import build_best_page_comparison

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
    weak_paragraphs: dict
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
    best_pages: dict
    performance_explainer: dict
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
    weak_paragraphs = _load_json(report / "weak_paragraphs.json", {})
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
    best_pages = _load_json(report / "best_pages.json", {})
    performance_explainer = _load_json(report / "performance_explainer.json", {})

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
        weak_paragraphs=weak_paragraphs,
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
        best_pages=best_pages,
        performance_explainer=performance_explainer,
        ahrefs_semantic_rows=ahrefs_semantic_rows,
        ahrefs_semantic_embeddings=ahrefs_semantic_embeddings,
        embeddings=embed_array,
        embedded_pages=embedded_pages,
    )


# --- combined scatter -----------------------------------------------------


def _project_compare_embeddings(matrix: np.ndarray, *, seed: int, min_umap_points: int) -> np.ndarray:
    n = len(matrix)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n < min_umap_points:
        coords = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            coords[i, 0] = float(i)
        return coords
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, n - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(matrix.astype(np.float32)).astype(np.float32)
    except ModuleNotFoundError:
        LOG.warning("  umap-learn is not installed; using deterministic PCA projection")
    except Exception as exc:
        LOG.warning("  UMAP projection failed (%s); using deterministic PCA projection", exc)
    return _pca_projection_2d(matrix)


def _pca_projection_2d(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1.0, norms)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(arr, full_matrices=False)
        coords = (u[:, :2] * s[:2]).astype(np.float32)
    except np.linalg.LinAlgError:
        coords = np.zeros((n, 2), dtype=np.float32)
        coords[:, 0] = np.arange(n, dtype=np.float32)
        return coords
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return coords.astype(np.float32)


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

    coords = _project_compare_embeddings(big.astype(np.float32), seed=seed, min_umap_points=4)

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
    coords = _project_compare_embeddings(big, seed=seed, min_umap_points=5)

    out: list[dict] = []
    cursor = 0
    for proj, sub_embs, sub_rows in chunks:
        authority_lookup = _authority_lookup(proj)
        link_lookup = _page_link_lookup(proj)
        freshness_lookup = _freshness_lookup(proj)
        page_type_lookup = _url_lookup_from_rows(((proj.page_types or {}).get("per_page") or []), ("url",))
        for i, row in enumerate(sub_rows):
            authority = _lookup_url(authority_lookup, row.get("url"))
            link = _lookup_url(link_lookup, row.get("url"))
            freshness = _lookup_url(freshness_lookup, row.get("url"))
            page_type = _lookup_url(page_type_lookup, row.get("url"))
            out.append({
                **row,
                "domain": proj.domain,
                "x": float(coords[cursor + i, 0]),
                "y": float(coords[cursor + i, 1]),
                "pagerank": _safe_float(authority.get("pagerank")),
                "weighted_pagerank_percentile": _safe_float(authority.get("weighted_pagerank_percentile")),
                "authority_traffic_gap": _safe_float(authority.get("authority_traffic_gap")),
                "in_degree": _safe_int(link.get("in_degree")),
                "out_degree": _safe_int(link.get("out_degree")),
                "click_depth": link.get("click_depth"),
                "freshness_bucket": freshness.get("bucket") or "unknown",
                "freshness_age_days": freshness.get("age_days"),
                "freshness_date": freshness.get("date") or "",
                "page_type": page_type.get("page_type") or "",
            })
        cursor += len(sub_embs)
    return out, len(big)


def _semantic_entity_maps(projects: list[_Project]) -> dict:
    specs = {
        "pages": {
            "types": {"page"},
            "label": "Pages",
            "description": "Ranking pages projected by full-page embeddings with traffic and authority context.",
        },
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
        "paragraphs": {
            "types": {"paragraph"},
            "label": "Paragraphs",
            "description": "Paragraph snippets from ranking pages projected across domains.",
        },
        "keywords": {
            "types": {"keyword"},
            "label": "Keywords",
            "description": "Ranking keywords projected against visible content entities.",
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


def _gini(values: list[float]) -> float:
    positive = sorted(float(v) for v in values if float(v) > 0)
    if not positive:
        return 0.0
    n = len(positive)
    total = sum(positive)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(positive))
    return max(0.0, min(1.0, (2.0 * weighted) / (n * total) - (n + 1.0) / n))


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
            {"key": "spammy_paragraph_count", "label": "Link-stuffed paragraphs", "better": "low", "fmt": "int"},
            {"key": "weak_paragraph_count", "label": "Weak paragraphs", "better": "low", "fmt": "int"},
            {"key": "weak_paragraph_high_severity", "label": "High-severity weak paragraphs", "better": "low", "fmt": "int"},
            {"key": "weak_paragraph_template_rows", "label": "Boilerplate weak paragraphs", "better": "low", "fmt": "int"},
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


METRIC_REPORT_SECTIONS: dict[str, tuple[str, str]] = {
    "ahrefs_org_traffic": ("ahrefs-block", "Search demand"),
    "ahrefs_org_keywords": ("ahrefs-block", "Search demand"),
    "ahrefs_matched_traffic": ("ahrefs-block", "Search demand"),
    "ahrefs_top_pages_value_usd": ("ahrefs-block", "Search demand"),
    "ahrefs_top3_keywords": ("ahrefs-block", "Search demand"),
    "calibrated_focus": ("audit-scatter-container", "Topic map"),
    "site_radius": ("audit-scatter-container", "Topic map"),
    "section_coherence": ("audit-sections", "Sections"),
    "topic_dimension": ("audit-scatter-container", "Topic map"),
    "answerability_median": ("answerability-block", "GEO score"),
    "answerability_p90": ("answerability-block", "GEO score"),
    "answerability_below_4_share": ("answerability-block", "GEO score"),
    "answer_block_strong_cluster_share": ("answer-blocks-block", "Answer blocks"),
    "answer_block_opportunities": ("answer-blocks-block", "Answer blocks"),
    "schema_coverage": ("structured-data-block", "Structured data"),
    "invalid_jsonld_blocks": ("structured-data-block", "Structured data"),
    "schema_type_count": ("structured-data-block", "Structured data"),
    "schema_opportunities": ("structured-data-block", "Structured data"),
    "trust_avg_score": ("trust-signals-block", "Trust signals"),
    "trust_high_priority_pages": ("trust-signals-block", "Trust signals"),
    "median_in_degree": ("link-counts-block", "Internal links"),
    "p90_in_degree": ("link-counts-block", "Internal links"),
    "median_out_degree": ("link-counts-block", "Internal links"),
    "orphan_share": ("linkgraph-block", "Orphan pages"),
    "authority_traffic_alignment": ("traffic-pagerank-block", "Traffic PageRank"),
    "high_traffic_low_authority_pages": ("traffic-pagerank-block", "Traffic PageRank"),
    "authority_without_demand_pages": ("traffic-pagerank-block", "Traffic PageRank"),
    "demand_support_alignment": ("high-demand-link-block", "Demand/link support"),
    "high_demand_low_support_pages": ("high-demand-link-block", "Demand/link support"),
    "high_demand_low_support_traffic": ("high-demand-link-block", "Demand/link support"),
    "critical_internal_links": ("link-removal-block", "Link risk"),
    "weak_internal_links": ("link-removal-block", "Link risk"),
    "high_priority_link_additions": ("link-addition-block", "Link additions"),
    "avg_link_addition_benefit": ("link-addition-block", "Link additions"),
    "anchor_relevance_descriptive_rate": ("anchor-block", "Anchor relevance"),
    "anchor_relevance_weak_links": ("anchor-block", "Anchor relevance"),
    "contextual_link_avg_impact": ("contextual-link-block", "Contextual links"),
    "contextual_template_links": ("contextual-link-block", "Contextual links"),
    "internal_link_patterns": ("internal-link-patterns-block", "Link patterns"),
    "internal_link_pattern_recommendations": ("internal-link-patterns-block", "Link patterns"),
    "internal_link_pattern_confidence": ("internal-link-patterns-block", "Link patterns"),
    "architecture_resilience": ("hub-bottleneck-block", "Architecture"),
    "bottleneck_pages": ("hub-bottleneck-block", "Architecture"),
    "descriptive_anchor_share": ("linkbuilding-block", "Linkbuilding"),
    "generic_anchor_share": ("linkbuilding-block", "Linkbuilding"),
    "metadata_issue_share": ("metadata-quality-block", "Metadata"),
    "freshness_date_coverage": ("freshness-block", "Freshness"),
    "freshness_stale_share": ("freshness-block", "Freshness"),
    "freshness_impact_traffic_at_risk": ("freshness-impact-block", "Freshness impact"),
    "freshness_high_impact_sections": ("freshness-impact-block", "Freshness impact"),
    "entity_coverage": ("entity-coverage-block", "Entity coverage"),
    "topical_authority_score": ("entities-block", "Entities"),
    "cannibalization_page_conflicts": ("cannibalization-block", "Cannibalization"),
    "cannibalization_paragraph_conflicts": ("cannibalization-block", "Cannibalization"),
    "duplicate_fragment_groups": ("duplicate-fragments-block", "Duplicate fragments"),
    "duplicate_strong_patterns": ("duplicate-fragments-block", "Duplicate fragments"),
    "template_success_patterns": ("template-patterns-block", "Template patterns"),
    "template_pattern_recommendations": ("template-patterns-block", "Template patterns"),
    "zero_link_paragraph_share": ("para-density-block", "Paragraph links"),
    "spammy_paragraph_count": ("para-density-block", "Paragraph links"),
    "weak_paragraph_count": ("weak-paragraphs-block", "Weak paragraphs"),
    "weak_paragraph_high_severity": ("weak-paragraphs-block", "Weak paragraphs"),
    "weak_paragraph_template_rows": ("weak-paragraphs-block", "Weak paragraphs"),
    "indexable_share": ("indexability-block", "Indexability"),
    "media_accessibility_issue_share": ("media-accessibility-block", "Media accessibility"),
    "median_html_weight_bytes": ("performance-block", "Performance"),
    "heavy_page_share": ("performance-block", "Performance"),
    "render_blocking_share": ("performance-block", "Performance"),
    "pages_missing_h1_share": ("headers-block", "Headers"),
    "pages_multi_h1": ("headers-block", "Headers"),
    "cta_coverage": ("conversion-block", "Conversion"),
    "primary_cta_coverage": ("conversion-block", "Conversion"),
    "form_coverage": ("conversion-block", "Conversion"),
    "conversion_balance_efficiency": ("conversion-balance-block", "Conversion balance"),
    "conversion_balance_high_risk": ("conversion-balance-block", "Conversion balance"),
    "lead_pages_without_capture": ("conversion-block", "Conversion"),
    "cta_overload_pages": ("conversion-block", "Conversion"),
}


def _metric_report_section(metric_key: str) -> dict:
    section_id, label = METRIC_REPORT_SECTIONS.get(metric_key, ("", "Domain report"))
    return {"id": section_id, "label": label}


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
                    "report_section": _metric_report_section(metric["key"]),
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
                "report_section": _metric_report_section(metric["key"]),
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
    weak_summary = (proj.weak_paragraphs or {}).get("summary", {}) or {}
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
        "weak_paragraph_count": int(weak_summary.get("flagged_rows", 0)),
        "weak_paragraph_high_severity": int(weak_summary.get("high_severity_rows", 0)),
        "weak_paragraph_template_rows": int(weak_summary.get("template_rows", 0)),
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


def _action_board_payload(projects: list[_Project]) -> dict:
    items: list[dict] = []
    by_domain: dict[str, dict] = {}
    score_model = {}
    for proj in projects:
        payload = proj.recommendations or {}
        score_model = score_model or payload.get("score_model") or {}
        domain_stats = by_domain.setdefault(proj.domain, {
            "domain": proj.domain,
            "actions": 0,
            "high": 0,
            "traffic_opportunity": 0.0,
            "avg_priority_score": 0.0,
        })
        for row in payload.get("items") or []:
            if not isinstance(row, dict):
                continue
            item = {"domain": proj.domain, **row}
            items.append(item)
            domain_stats["actions"] += 1
            if row.get("priority") == "high":
                domain_stats["high"] += 1
            domain_stats["traffic_opportunity"] += _safe_float(row.get("traffic_opportunity"))
            domain_stats["avg_priority_score"] += _safe_float(row.get("priority_score"))

    for stat in by_domain.values():
        if stat["actions"]:
            stat["avg_priority_score"] = round(stat["avg_priority_score"] / stat["actions"], 1)
        stat["traffic_opportunity"] = round(stat["traffic_opportunity"], 1)

    items.sort(key=lambda r: (
        _safe_float(r.get("priority_score")),
        _safe_float(r.get("impact")),
        _safe_float(r.get("traffic_opportunity")),
    ), reverse=True)
    filters = {
        "domains": [p.domain for p in projects],
        "priorities": ["high", "medium", "low"],
        "categories": sorted({str(r.get("category") or "") for r in items if r.get("category")}),
        "types": sorted({str(r.get("type") or "") for r in items if r.get("type")}),
        "owners": sorted({str(r.get("owner") or "") for r in items if r.get("owner")}),
        "clusters": sorted({str(r.get("cluster") or "") for r in items if r.get("cluster")})[:250],
    }
    return {
        "summary": {
            "status": "ok" if items else "no_recommendations",
            "model": "comparison_action_board_v1",
            "actions": len(items),
            "domains": len(projects),
            "high": sum(1 for r in items if r.get("priority") == "high"),
            "traffic_opportunity": round(sum(_safe_float(r.get("traffic_opportunity")) for r in items), 1),
        },
        "score_model": score_model,
        "filters": filters,
        "domains": sorted(by_domain.values(), key=lambda r: (_safe_int(r.get("high")), _safe_float(r.get("avg_priority_score"))), reverse=True),
        "items": items[:1000],
    }


def _playbook_slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:90] or "playbook"


def _playbook_blueprint(category: str, source_type: str, pattern: str) -> dict:
    text = f"{category} {source_type} {pattern}".lower()
    if "link" in text or "anchor" in text or "pagerank" in text:
        return {
            "owner": "SEO",
            "trigger_conditions": [
                "Target page has search demand or conversion value but weaker internal support than comparable winning pages.",
                "A source page or hub can add a descriptive contextual link without exceeding paragraph link-density limits.",
            ],
            "implementation_steps": [
                "Confirm the target page and the source-page paragraph are topically aligned.",
                "Add the link in main content using the suggested or closest natural anchor.",
                "Avoid repeated template/sidebar-only links when a contextual placement is available.",
                "Re-crawl and confirm in-degree, click depth, and authority-demand alignment improved.",
            ],
            "acceptance_criteria": [
                "Every listed target has at least one relevant new contextual internal link.",
                "Anchor text describes the target topic and does not repeat the same exact phrase site-wide.",
                "The target appears in the next link graph with improved internal support.",
            ],
            "validation_metric": "Target in-degree, click depth, traffic-weighted PageRank, authority-demand alignment, and target-page organic traffic.",
        }
    if "schema" in text or "structured" in text or "faqpage" in text or "article" in text:
        return {
            "owner": "SEO",
            "trigger_conditions": [
                "Target page contains visible content that supports a schema type used by stronger competing pages.",
                "The current page has missing or invalid structured data for that content type.",
            ],
            "implementation_steps": [
                "Map the visible page content to the required and recommended schema properties.",
                "Add JSON-LD only for content that is actually visible on the page.",
                "Validate the markup and fix warnings that affect eligibility.",
                "Re-run the audit and verify schema coverage and invalid-block counts.",
            ],
            "acceptance_criteria": [
                "All listed pages contain valid JSON-LD for the recommended schema type.",
                "Required properties are populated from visible page content.",
                "No new invalid JSON-LD blocks are introduced.",
            ],
            "validation_metric": "Valid schema blocks, schema coverage by cluster, invalid JSON-LD count, and rich-result eligibility checks.",
        }
    if "fresh" in text or "date" in text or "updated" in text:
        return {
            "owner": "Content",
            "trigger_conditions": [
                "Winning pages expose fresher evidence or updated dates while target pages are stale or missing date signals.",
                "The page carries traffic or ranking keywords where freshness can affect user trust.",
            ],
            "implementation_steps": [
                "Review outdated claims, screenshots, statistics, integrations, pricing references, and examples.",
                "Update the affected sections with current evidence and remove stale claims.",
                "Expose a visible published or updated date when appropriate for the content type.",
                "Re-run freshness and readiness checks after publishing.",
            ],
            "acceptance_criteria": [
                "All listed pages move out of stale or unknown freshness buckets where a date is appropriate.",
                "Updated claims include current evidence or citations.",
                "Freshness risk and traffic-at-risk metrics decline on the next report.",
            ],
            "validation_metric": "Freshness bucket, visible date coverage, freshness-impact risk, and organic traffic on refreshed pages.",
        }
    if "conversion" in text or "cta" in text or "form" in text or "lead" in text:
        return {
            "owner": "CRO",
            "trigger_conditions": [
                "A page has search demand or commercial intent but weaker conversion support than winning pages.",
                "Comparable successful templates expose a primary CTA, form, demo path, or lead capture element.",
            ],
            "implementation_steps": [
                "Match the CTA to the page intent and avoid adding competing primary actions.",
                "Place the primary CTA near the high-intent section or after the main answer.",
                "Add supporting proof or objection-handling copy near the conversion path.",
                "Validate conversion support and CTA overload metrics after release.",
            ],
            "acceptance_criteria": [
                "Every target page has a clear primary conversion path appropriate for its intent.",
                "Lead pages include a form, contact path, demo path, or equivalent capture mechanism.",
                "Conversion balance improves without creating CTA overload.",
            ],
            "validation_metric": "Primary CTA coverage, form/contact coverage, conversion balance score, lead-page capture rate, and conversions if available.",
        }
    if "title" in text or "metadata" in text or "h1" in text:
        return {
            "owner": "SEO",
            "trigger_conditions": [
                "Target pages have metadata, title, or heading alignment issues on pages with search demand.",
                "Search keywords, page title, and visible H1/H2 language are semantically misaligned.",
            ],
            "implementation_steps": [
                "Map the primary ranking keyword and intent to the title, H1, and opening section.",
                "Rewrite metadata and headings using natural, specific language rather than keyword stuffing.",
                "Keep one clear H1 and preserve page intent during the rewrite.",
                "Re-run metadata and semantic alignment checks.",
            ],
            "acceptance_criteria": [
                "Target pages have unique, intent-matched titles and descriptions.",
                "The H1 and main headings align with the target page topic.",
                "Metadata quality issues are cleared for the listed URLs.",
            ],
            "validation_metric": "Metadata issue share, title-to-content alignment, title/H1 alignment, rankings, and CTR where available.",
        }
    return {
        "owner": "Content",
        "trigger_conditions": [
            "A target cluster or page lacks a content pattern repeatedly present in stronger pages.",
            "The missing pattern matches the page intent, template, or keyword cluster instead of being copied blindly.",
        ],
        "implementation_steps": [
            "Review the source examples and identify the role the pattern plays in the page.",
            "Add or rewrite the missing section using domain-specific examples, entities, and proof.",
            "Align headings, paragraphs, and internal links to the target keyword cluster.",
            "Re-run comparison and domain reports to confirm support-score and priority-score movement.",
        ],
        "acceptance_criteria": [
            "Every listed target page contains the recommended pattern in a page-specific form.",
            "The new section is supported by visible headings, paragraphs, examples, and internal links.",
            "Keyword-content support and priority scores improve in the next comparison.",
        ],
        "validation_metric": "Keyword-content support score, paragraph archetype coverage, entity coverage, priority score, rankings, and organic traffic.",
    }


def _seo_playbooks_payload(
    projects: list[_Project],
    *,
    winning_patterns: dict,
    paragraph_archetypes: dict,
    template_patterns: dict,
    structured_data_opportunities: dict,
    internal_link_patterns: dict,
    action_board: dict,
) -> dict:
    domains = [p.domain for p in projects]
    playbooks: dict[str, dict] = {}

    def ensure(
        key: str,
        *,
        pattern: str,
        category: str,
        source_type: str,
        cluster: str = "",
        cluster_label: str = "",
        source_domain: str = "",
    ) -> dict:
        blueprint = _playbook_blueprint(category, source_type, pattern)
        row = playbooks.setdefault(key, {
            "playbook_id": key,
            "pattern": pattern,
            "category": category or "content",
            "source_type": source_type,
            "source_types": [],
            "owner": blueprint["owner"],
            "cluster": cluster,
            "cluster_label": cluster_label or cluster,
            "priority_score": 0.0,
            "confidence": 0.0,
            "expected_benefit_score": 0.0,
            "traffic_opportunity": 0.0,
            "trigger_conditions": list(blueprint["trigger_conditions"]),
            "implementation_steps": list(blueprint["implementation_steps"]),
            "acceptance_criteria": list(blueprint["acceptance_criteria"]),
            "validation_metric": blueprint["validation_metric"],
            "evidence": [],
            "source_examples": [],
            "targets": [],
            "_target_keys": set(),
            "_evidence_keys": set(),
            "_example_keys": set(),
            "_domains": Counter(),
            "_source_domains": Counter(),
            "_confidence_sum": 0.0,
            "_confidence_count": 0,
        })
        if source_type not in row["source_types"]:
            row["source_types"].append(source_type)
        if source_domain:
            row["_source_domains"][source_domain] += 1
        return row

    def add_evidence(pb: dict, evidence: dict) -> None:
        compact = {k: v for k, v in evidence.items() if v not in (None, "", [], {})}
        if not compact:
            return
        key = json.dumps(compact, sort_keys=True, default=str)[:500]
        if key in pb["_evidence_keys"]:
            return
        pb["_evidence_keys"].add(key)
        pb["evidence"].append(compact)

    def add_example(pb: dict, example: dict) -> None:
        compact = {k: v for k, v in example.items() if v not in (None, "", [], {})}
        if not compact:
            return
        key = "|".join(str(compact.get(k, "")) for k in ("domain", "url", "title", "excerpt", "label"))
        if key in pb["_example_keys"]:
            return
        pb["_example_keys"].add(key)
        pb["source_examples"].append(compact)

    def target_rows_from_action(row: dict) -> list[dict]:
        targets = row.get("targets") or []
        if not isinstance(targets, list) or not targets:
            return [row]
        out = []
        for url in targets:
            out.append({**row, "target_url": url, "target_title": row.get("title") or url})
        return out

    def add_target(pb: dict, row: dict, *, action_override: str = "") -> None:
        domain = row.get("target_domain") or row.get("domain") or ""
        url = row.get("target_url") or row.get("url") or row.get("source_url") or ""
        title = row.get("target_title") or row.get("title") or row.get("source_title") or url
        action = (
            action_override
            or row.get("concrete_change")
            or row.get("action")
            or row.get("instruction")
            or row.get("recommended_action")
            or row.get("reason")
            or pb.get("pattern")
            or ""
        )
        key = f"{domain}|{url}|{action[:140]}"
        if key in pb["_target_keys"]:
            return
        priority = max(
            _safe_float(row.get("priority_score")),
            _safe_float(row.get("expected_benefit_score")),
            _safe_float(row.get("impact")),
        )
        confidence = _safe_float(row.get("confidence"))
        traffic = max(
            _safe_float(row.get("traffic_opportunity")),
            _safe_float(row.get("target_traffic")),
            _safe_float(row.get("traffic")),
            _safe_float(row.get("opportunity")),
        )
        target = {
            "domain": domain,
            "url": url,
            "title": title,
            "cluster": row.get("cluster") or row.get("cluster_label") or pb.get("cluster") or "",
            "page_type": row.get("target_page_type") or row.get("page_type") or "",
            "owner": row.get("owner") or pb.get("owner") or "",
            "type": row.get("type") or row.get("category") or pb.get("category") or "",
            "priority_score": round(priority, 2),
            "confidence": round(confidence, 3),
            "traffic_opportunity": round(traffic, 1),
            "action": action,
            "missing_element": row.get("missing_element") or row.get("missing_pattern") or row.get("schema_type") or "",
            "suggested_anchor": row.get("suggested_anchor") or "",
            "suggested_target_url": row.get("suggested_target_url") or "",
            "source_recommendation_id": row.get("id") or row.get("pattern_key") or row.get("pattern_id") or "",
        }
        pb["_target_keys"].add(key)
        pb["targets"].append(target)
        if domain:
            pb["_domains"][domain] += 1
        pb["priority_score"] = max(_safe_float(pb.get("priority_score")), priority)
        pb["traffic_opportunity"] = _safe_float(pb.get("traffic_opportunity")) + traffic
        if confidence:
            pb["_confidence_sum"] += confidence
            pb["_confidence_count"] += 1

    # Winning-pattern transfer already merges competitor cluster gaps,
    # template transplants, and structural metric gaps.
    for pattern in (winning_patterns or {}).get("patterns") or []:
        targets = pattern.get("target_recommendations") or []
        if not targets:
            continue
        key = f"winning::{_playbook_slug(pattern.get('pattern_key') or pattern.get('title'))}"
        pb = ensure(
            key,
            pattern=pattern.get("title") or "Transfer winning SEO pattern",
            category=pattern.get("category") or "content",
            source_type="winning_pattern_transfer",
            cluster=pattern.get("cluster") or "",
            cluster_label=pattern.get("cluster_label") or "",
            source_domain=pattern.get("source_domain") or "",
        )
        source_evidence = pattern.get("source_evidence") or {}
        add_evidence(pb, {"type": "winning_pattern", "source_domain": pattern.get("source_domain") or "", **source_evidence})
        if source_evidence.get("source_url") or source_evidence.get("source_title"):
            add_example(pb, {
                "domain": pattern.get("source_domain") or "",
                "url": source_evidence.get("source_url") or "",
                "title": source_evidence.get("source_title") or "",
                "detail": source_evidence.get("evidence_type") or pattern.get("category") or "",
            })
        for rec in targets:
            add_target(pb, rec)

    # Template success patterns: reusable page-structure and conversion blocks.
    template_examples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in (template_patterns or {}).get("examples") or []:
        feature = str(row.get("feature_key") or row.get("label") or "template")
        page_type = str(row.get("page_type") or "")
        template_examples[(feature, page_type)].append(row)
    for rec in (template_patterns or {}).get("recommendations") or []:
        feature = str(rec.get("feature_key") or rec.get("missing_pattern") or "template")
        page_type = str(rec.get("page_type") or "")
        pattern_label = rec.get("missing_pattern") or feature
        key = f"template::{_playbook_slug(feature)}::{_playbook_slug(page_type)}"
        pb = ensure(
            key,
            pattern=f"Add {pattern_label} to {page_type or 'matching'} pages",
            category=rec.get("category") or "content",
            source_type="template_success_pattern",
        )
        add_evidence(pb, {
            "type": "template_success_pattern",
            "feature_key": feature,
            "page_type": page_type,
            "confidence": rec.get("confidence"),
            "observed_lift": rec.get("observed_lift"),
        })
        for example in template_examples.get((feature, page_type), [])[:4]:
            add_example(pb, {
                "domain": example.get("domain") or "",
                "label": example.get("label") or feature,
                "detail": example.get("recommendation") or "",
                "confidence": example.get("confidence"),
                "observed_lift": example.get("observed_lift"),
            })
            for sample in (example.get("sample_urls") or [])[:3]:
                add_example(pb, {"domain": example.get("domain") or "", **sample})
        add_target(pb, rec)

    # Paragraph archetypes: repeatable content roles seen on stronger pages.
    for rec in (paragraph_archetypes or {}).get("recommendations") or []:
        archetype = rec.get("label") or rec.get("archetype") or "paragraph archetype"
        key = f"paragraph::{_playbook_slug(rec.get('cluster'))}::{_playbook_slug(rec.get('archetype') or archetype)}"
        pb = ensure(
            key,
            pattern=f"Add {str(archetype).lower()} sections to {rec.get('cluster_label') or rec.get('cluster') or 'matching'} pages",
            category="content",
            source_type="paragraph_archetype",
            cluster=rec.get("cluster") or "",
            cluster_label=rec.get("cluster_label") or "",
            source_domain=rec.get("source_domain") or "",
        )
        add_evidence(pb, {
            "type": "paragraph_archetype",
            "source_domain": rec.get("source_domain"),
            "archetype": rec.get("archetype"),
            "traffic_opportunity": rec.get("traffic_opportunity"),
        })
        for example in (rec.get("source_examples") or [])[:6]:
            add_example(pb, {"domain": rec.get("source_domain") or "", **example})
        add_target(pb, rec)

    # Structured-data opportunities aggregate schema tasks by type and cluster.
    for rec in (structured_data_opportunities or {}).get("recommendations") or []:
        schema_type = rec.get("schema_type") or "schema"
        cluster = rec.get("cluster") or rec.get("cluster_label") or ""
        key = f"schema::{_playbook_slug(schema_type)}::{_playbook_slug(cluster)}"
        pb = ensure(
            key,
            pattern=f"Add valid {schema_type} structured data",
            category="schema",
            source_type="schema_opportunity",
            cluster=cluster,
            cluster_label=cluster,
        )
        add_evidence(pb, {
            "type": "schema_opportunity",
            "schema_type": schema_type,
            "reason": rec.get("reason"),
            "missing_evidence": rec.get("missing_evidence"),
            "missing_recommended_properties": rec.get("missing_recommended_properties"),
            "guideline_url": rec.get("guideline_url"),
        })
        add_target(pb, rec, action_override=rec.get("reason") or f"Add {schema_type} schema if visible content supports it.")

    # Internal-link pattern gaps are reusable architecture rules.
    link_examples: dict[str, list[dict]] = defaultdict(list)
    for row in (internal_link_patterns or {}).get("examples") or []:
        for key in (row.get("pattern_id"), row.get("rule_key"), row.get("inferred_rule")):
            if key:
                link_examples[str(key)].append(row)
    for rec in (internal_link_patterns or {}).get("recommendations") or []:
        pattern_key = str(rec.get("pattern_id") or rec.get("rule_key") or rec.get("missing_pattern") or "internal-link")
        label = rec.get("missing_pattern") or rec.get("recommended_action") or pattern_key
        key = f"internal::{_playbook_slug(pattern_key)}"
        pb = ensure(
            key,
            pattern=f"Apply internal-link pattern: {label}",
            category="link",
            source_type="internal_link_pattern",
        )
        add_evidence(pb, {
            "type": "internal_link_pattern",
            "pattern_id": rec.get("pattern_id"),
            "missing_pattern": rec.get("missing_pattern"),
            "confidence": rec.get("confidence"),
            "lift_score_difference": rec.get("lift_score_difference"),
        })
        for example in link_examples.get(pattern_key, [])[:5]:
            add_example(pb, {
                "domain": example.get("domain") or "",
                "label": example.get("inferred_rule") or pattern_key,
                "support_count": example.get("support_count"),
                "confidence": example.get("confidence"),
            })
            for sample in (example.get("sample_links") or [])[:4]:
                add_example(pb, {"domain": example.get("domain") or "", **sample})
        add_target(pb, rec)

    # Priority action board: captures recurring fixes that were not generated
    # from a cross-domain transfer section above.
    action_groups: dict[str, list[dict]] = defaultdict(list)
    for row in (action_board or {}).get("items") or []:
        key = "::".join([
            "action",
            _playbook_slug(row.get("category") or "action"),
            _playbook_slug(row.get("type") or row.get("title") or "fix"),
            _playbook_slug(row.get("cluster") or "global"),
        ])
        action_groups[key].append(row)
    for key, rows in action_groups.items():
        max_priority = max((_safe_float(row.get("priority_score")) for row in rows), default=0.0)
        if len(rows) < 2 and max_priority < 60.0:
            continue
        sample = rows[0]
        label = str(sample.get("type") or sample.get("title") or "SEO fix").replace("_", " ")
        cluster = sample.get("cluster") or ""
        pb = ensure(
            key,
            pattern=f"Repeatable fix: {label}{' in ' + str(cluster) if cluster else ''}",
            category=sample.get("category") or "content",
            source_type="fix_priority_score",
            cluster=cluster,
            cluster_label=cluster,
        )
        add_evidence(pb, {
            "type": "fix_priority_score",
            "recommendation_type": sample.get("type"),
            "category": sample.get("category"),
            "priority_model": ((action_board or {}).get("score_model") or {}).get("model"),
            "grouped_recommendations": len(rows),
        })
        for row in rows:
            for target in target_rows_from_action(row):
                add_target(pb, target)

    rows: list[dict] = []
    for pb in playbooks.values():
        if not pb["targets"]:
            continue
        pb["targets"].sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("traffic_opportunity"))), reverse=True)
        domain_rows = []
        for domain, count in pb["_domains"].most_common():
            domain_targets = [t for t in pb["targets"] if t.get("domain") == domain]
            domain_rows.append({
                "domain": domain,
                "target_count": count,
                "traffic_opportunity": round(sum(_safe_float(t.get("traffic_opportunity")) for t in domain_targets), 1),
                "avg_priority_score": round(sum(_safe_float(t.get("priority_score")) for t in domain_targets) / max(1, len(domain_targets)), 2),
            })
        confidence_count = _safe_int(pb.get("_confidence_count"))
        pb["confidence"] = round(pb["_confidence_sum"] / confidence_count, 3) if confidence_count else round(_safe_float(pb.get("confidence")), 3)
        pb["traffic_opportunity"] = round(_safe_float(pb.get("traffic_opportunity")), 1)
        pb["target_count"] = len(pb["targets"])
        pb["affected_domains"] = domain_rows
        pb["source_domains"] = [domain for domain, _ in pb["_source_domains"].most_common()]
        pb["expected_benefit_score"] = round(min(100.0, _safe_float(pb.get("priority_score")) * 0.72 + min(24.0, math.log1p(pb["traffic_opportunity"]) * 3.8) + min(12.0, pb["target_count"] * 1.6)), 2)
        pb["expected_benefit"] = (
            f"{int(round(pb['traffic_opportunity'])):,} traffic opportunity across {pb['target_count']} target page"
            f"{'' if pb['target_count'] == 1 else 's'}"
        )
        pb["reusable_scope"] = "cross-domain" if len(domain_rows) > 1 or len(pb["source_domains"]) > 1 else "domain-pattern"
        first_trigger = (pb.get("trigger_conditions") or ["matching pages meet the trigger conditions"])[0]
        pb["portable_template"] = f"Reuse when {first_trigger[:1].lower() + first_trigger[1:]}"
        pb["checklist"] = [
            {
                "task": f"{target.get('domain')}: {target.get('action') or pb.get('pattern')}",
                "target_url": target.get("url") or "",
                "target_title": target.get("title") or "",
                "owner": target.get("owner") or pb.get("owner") or "",
            }
            for target in pb["targets"][:80]
        ]
        for key in list(pb.keys()):
            if key.startswith("_"):
                del pb[key]
        pb["evidence"] = pb["evidence"][:10]
        pb["source_examples"] = pb["source_examples"][:10]
        pb["targets"] = pb["targets"][:140]
        rows.append(pb)

    rows.sort(key=lambda r: (_safe_float(r.get("expected_benefit_score")), _safe_float(r.get("priority_score")), _safe_int(r.get("target_count"))), reverse=True)
    return {
        "summary": {
            "status": "ok" if rows else "no_playbooks",
            "model": "cross_domain_seo_playbooks_v1",
            "playbooks": len(rows),
            "targets": sum(_safe_int(row.get("target_count")) for row in rows),
            "domains": len(domains),
            "cross_domain": sum(1 for row in rows if row.get("reusable_scope") == "cross-domain"),
            "traffic_opportunity": round(sum(_safe_float(row.get("traffic_opportunity")) for row in rows), 1),
        },
        "filters": {
            "domains": domains,
            "categories": sorted({str(row.get("category") or "") for row in rows if row.get("category")}),
            "source_types": sorted({source for row in rows for source in (row.get("source_types") or [])}),
            "clusters": sorted({str(row.get("cluster_label") or row.get("cluster") or "") for row in rows if row.get("cluster_label") or row.get("cluster")})[:250],
            "owners": sorted({str(row.get("owner") or "") for row in rows if row.get("owner")}),
        },
        "playbooks": rows[:220],
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
            "description": "No per-keyword provider calls are made here; rows are derived from cached GSC/Ahrefs/DataForSEO domain snapshots and keyed by domain, keyword, and provider.",
            "entries": cache_entry_count,
            "samples": cache_samples,
        },
    }


def _strongest_cluster_payload(
    projects: list[_Project],
    page_scatter: list[dict],
    semantic_scatter: list[dict],
) -> dict:
    domains = [p.domain for p in projects]
    per_project = {proj.domain: _project_cluster_entries(proj) for proj in projects}
    cluster_maps: dict[str, dict] = {}
    entries_by_domain: dict[str, dict[str, dict]] = {}

    def cluster_maps_for(proj: _Project) -> dict:
        id_to_key: dict[str, str] = {}
        label_by_key: dict[str, str] = {}
        raw_by_key: dict[str, dict] = {}
        url_lookup: dict[str, dict] = {}

        def put_url(url: object, key: str) -> None:
            if url and key and key != "unknown":
                _store_url_lookup(url_lookup, url, {"cluster": key})

        for row in ((proj.ahrefs or {}).get("clusters") or []):
            label = row.get("label") or row.get("key") or row.get("cluster")
            key = _cluster_key(label)
            if key == "unknown":
                continue
            label_by_key.setdefault(key, str(label or key))
            raw_by_key[key] = row
            for ident in (row.get("cluster"), row.get("key")):
                if ident is not None:
                    id_to_key[str(ident)] = key
            for page in row.get("top_pages") or []:
                put_url(page.get("matched_url") or page.get("url"), key)

        for row in ((proj.ahrefs or {}).get("top_pages") or []):
            key = id_to_key.get(str(row.get("cluster"))) if row.get("cluster") is not None else ""
            key = key or _cluster_key(row.get("cluster_label") or row.get("top_keyword") or "")
            if key != "unknown":
                put_url(row.get("matched_url") or row.get("url"), key)

        for row in ((proj.ahrefs or {}).get("organic_keywords") or []):
            key = id_to_key.get(str(row.get("cluster"))) if row.get("cluster") is not None else ""
            key = key or _cluster_key(row.get("cluster_label") or "")
            if key != "unknown":
                put_url(row.get("matched_url") or row.get("url"), key)

        return {
            "id_to_key": id_to_key,
            "label_by_key": label_by_key,
            "raw_by_key": raw_by_key,
            "url_lookup": url_lookup,
        }

    def key_for_row(domain: str, row: dict) -> str:
        maps = cluster_maps.get(domain) or {}
        id_to_key = maps.get("id_to_key") or {}
        cluster_ref = row.get("cluster")
        if cluster_ref is not None and str(cluster_ref) in id_to_key:
            return id_to_key[str(cluster_ref)]
        url = row.get("matched_url") or row.get("url")
        if url:
            found = _lookup_url(maps.get("url_lookup") or {}, url)
            if found.get("cluster"):
                return str(found["cluster"])
        key = _cluster_key(row.get("cluster_label") or row.get("label") or "")
        return key if key != "unknown" else ""

    def key_for_url(domain: str, url: object) -> str:
        found = _lookup_url((cluster_maps.get(domain) or {}).get("url_lookup") or {}, url)
        return str(found.get("cluster") or "")

    for proj in projects:
        cluster_maps[proj.domain] = cluster_maps_for(proj)

    coord_samples: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for row in semantic_scatter or []:
        domain = row.get("domain") or ""
        key = key_for_row(domain, row)
        if not key or row.get("x") is None or row.get("y") is None:
            continue
        weight = 1.0 + math.log1p(max(0.0, _safe_float(row.get("size") or row.get("traffic") or row.get("volume"))))
        coord_samples[(domain, key)].append((float(row["x"]), float(row["y"]), weight))
    for row in page_scatter or []:
        domain = row.get("domain") or ""
        key = key_for_url(domain, row.get("url"))
        if not key or row.get("x") is None or row.get("y") is None:
            continue
        weight = 1.0 + math.log1p(max(0.0, _safe_float(row.get("traffic"))))
        coord_samples[(domain, key)].append((float(row["x"]), float(row["y"]), weight))

    def coords_for(domain: str, key: str) -> tuple[float | None, float | None]:
        samples = coord_samples.get((domain, key)) or []
        if not samples:
            return None, None
        total = sum(sample[2] for sample in samples) or 1.0
        return (
            sum(sample[0] * sample[2] for sample in samples) / total,
            sum(sample[1] * sample[2] for sample in samples) / total,
        )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def freshness_value(bucket: str) -> float:
        return {
            "fresh": 1.0,
            "aging": 0.72,
            "stale": 0.38,
            "very_stale": 0.12,
            "future": 0.45,
            "unknown": 0.0,
            "": 0.0,
        }.get(str(bucket or "").lower(), 0.0)

    def page_rows_for(proj: _Project, key: str, entry: dict) -> list[dict]:
        page_lookup = _url_lookup_from_rows(proj.pages or [], ("url",))
        top_page_lookup = _url_lookup_from_rows(((proj.ahrefs or {}).get("top_pages") or []), ("matched_url", "url"), score_key="traffic")
        link_lookup = _page_link_lookup(proj)
        authority_lookup = _authority_lookup(proj)
        freshness_lookup = _freshness_lookup(proj)
        conversion_lookup = _url_lookup_from_rows(((proj.conversion or {}).get("per_page") or []), ("url",))
        urls = list(dict.fromkeys((entry.get("owned_urls") or []) + ([entry.get("top_url")] if entry.get("top_url") else [])))
        rows = []
        for url in urls[:16]:
            page = _lookup_url(page_lookup, url)
            search = _lookup_url(top_page_lookup, url)
            link = _lookup_url(link_lookup, url)
            authority = _lookup_url(authority_lookup, url)
            freshness = _lookup_url(freshness_lookup, url)
            conversion = _lookup_url(conversion_lookup, url)
            rows.append({
                "url": url,
                "title": search.get("title") or page.get("title") or url,
                "section": search.get("section") or page.get("section") or "",
                "traffic": _safe_int(search.get("traffic")),
                "keywords": _safe_int(search.get("keywords")),
                "top_keyword": search.get("top_keyword") or "",
                "word_count": _safe_int(page.get("word_count")),
                "in_degree": _safe_int(link.get("in_degree")),
                "out_degree": _safe_int(link.get("out_degree")),
                "pagerank": _safe_float(authority.get("pagerank") if authority else link.get("pagerank")),
                "freshness_bucket": freshness.get("bucket") or "",
                "primary_cta_count": _safe_int(conversion.get("primary_cta_count")),
                "cta_count": _safe_int(conversion.get("cta_count")),
            })
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("pagerank"))), reverse=True)
        return rows[:10]

    def keyword_rows_for(proj: _Project, key: str) -> list[dict]:
        rows = []
        for row in ((proj.ahrefs or {}).get("organic_keywords") or []):
            if key_for_row(proj.domain, row) != key:
                continue
            rows.append({
                "keyword": row.get("keyword") or "",
                "url": row.get("matched_url") or row.get("url") or "",
                "page_title": row.get("page_title") or row.get("title") or "",
                "position": _safe_float(row.get("position")),
                "traffic": _safe_int(row.get("traffic")),
                "volume": _safe_int(row.get("volume")),
                "intents": _string_list(row.get("intents"), "intent")[:5],
                "serp_features": _string_list(row.get("serp_features"), "feature")[:6],
            })
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_float(r.get("position"), 999.0)), reverse=True)
        return rows[:18]

    def paragraph_rows_for(proj: _Project, key: str) -> list[dict]:
        rows = []
        for row in proj.ahrefs_semantic_rows or []:
            if row.get("type") != "paragraph" or key_for_row(proj.domain, row) != key:
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            rows.append({
                "url": row.get("url") or "",
                "paragraph_index": row.get("paragraph_index"),
                "excerpt": label[:260],
                "traffic": _safe_int(row.get("traffic")),
            })
        rows.sort(key=lambda r: _safe_int(r.get("traffic")), reverse=True)
        return rows[:8]

    def link_rows_for(proj: _Project, key: str) -> list[dict]:
        rows = []
        removal = ((proj.linkgraph or {}).get("link_removal_simulation") or {})
        for row in removal.get("links") or removal.get("critical_links") or []:
            src_key = key_for_url(proj.domain, row.get("source_url"))
            tgt_key = key_for_url(proj.domain, row.get("target_url"))
            if key not in {src_key, tgt_key}:
                continue
            rows.append({
                "type": "existing",
                "source_url": row.get("source_url") or "",
                "source_title": row.get("source_title") or row.get("source_url") or "",
                "target_url": row.get("target_url") or "",
                "target_title": row.get("target_title") or row.get("target_url") or "",
                "anchor_samples": (row.get("anchor_samples") or [])[:4],
                "score": _safe_float(row.get("removal_loss_score") or row.get("contextual_link_impact")),
                "classification": row.get("classification") or row.get("placement") or "",
                "direction": "within_cluster" if src_key == tgt_key == key else ("incoming" if tgt_key == key else "outgoing"),
            })
        addition = ((proj.linkgraph or {}).get("link_addition_simulation") or {})
        for row in addition.get("recommendations") or []:
            src_key = key_for_url(proj.domain, row.get("source_url"))
            tgt_key = key_for_url(proj.domain, row.get("target_url"))
            if key not in {src_key, tgt_key}:
                continue
            rows.append({
                "type": "recommended",
                "source_url": row.get("source_url") or "",
                "source_title": row.get("source_title") or row.get("source_url") or "",
                "target_url": row.get("target_url") or "",
                "target_title": row.get("target_title") or row.get("target_url") or "",
                "anchor_samples": [row.get("suggested_anchor") or ""],
                "score": _safe_float(row.get("expected_benefit_score")),
                "classification": row.get("priority") or "",
                "direction": "within_cluster" if src_key == tgt_key == key else ("incoming" if tgt_key == key else "outgoing"),
            })
        rows.sort(key=lambda r: _safe_float(r.get("score")), reverse=True)
        return rows[:10]

    for proj in projects:
        entries = {}
        raw_by_key = cluster_maps[proj.domain]["raw_by_key"]
        total_domain_pages = max(_safe_int(proj.metrics.get("page_count")), len(proj.pages), 1)
        entity_cluster_lookup = _cluster_lookup_rows(((proj.entity_coverage or {}).get("clusters") or []), ("label", "cluster"))
        for key, base_entry in per_project.get(proj.domain, {}).items():
            raw = raw_by_key.get(key, {})
            entry = dict(base_entry)
            pages = page_rows_for(proj, key, entry)
            keywords = keyword_rows_for(proj, key)
            paragraphs = paragraph_rows_for(proj, key)
            links = link_rows_for(proj, key)
            x, y = coords_for(proj.domain, key)
            positions = [_safe_float(row.get("position")) for row in keywords if _safe_float(row.get("position")) > 0]
            page_count = max(
                _safe_int(raw.get("pages")),
                _safe_int(raw.get("matched_pages")),
                _safe_int(raw.get("page_count")),
                len(entry.get("owned_urls") or []),
                len(pages),
            )
            entity_cluster = entity_cluster_lookup.get(key, {})
            entity_coverage = _safe_float(entity_cluster.get("avg_coverage"), default=-1.0)
            if entity_coverage < 0:
                entity_coverage = min(1.0, len(entry.get("entities") or []) / 10.0)
            avg_pagerank = avg([_safe_float(page.get("pagerank")) for page in pages if page.get("pagerank") is not None])
            avg_in_degree = avg([float(_safe_int(page.get("in_degree"))) for page in pages])
            freshness_score = avg([freshness_value(page.get("freshness_bucket") or entry.get("freshness_bucket")) for page in pages]) or freshness_value(entry.get("freshness_bucket"))
            conversion_score = avg([
                1.0 if _safe_int(page.get("primary_cta_count")) else (0.55 if _safe_int(page.get("cta_count")) else 0.0)
                for page in pages
            ])
            word_counts = [_safe_int(page.get("word_count")) for page in pages if _safe_int(page.get("word_count"))]
            traffic = _safe_int(entry.get("traffic"))
            keyword_rows = max(_safe_int(entry.get("keyword_rows")), len(keywords))
            keywords_total = max(_safe_int(entry.get("keywords_total")), keyword_rows)
            top3_keywords = _safe_int(entry.get("top3_keywords"))
            avg_position = avg(positions)
            entry.update({
                "cluster": key,
                "cluster_label": entry.get("cluster_label") or (cluster_maps[proj.domain]["label_by_key"].get(key) or key),
                "traffic": traffic,
                "keywords_total": keywords_total,
                "keyword_rows": keyword_rows,
                "avg_position": round(avg_position, 2) if avg_position else 0.0,
                "pages": page_count,
                "matched_pages": max(_safe_int(raw.get("matched_pages")), len(pages)),
                "paragraph_count": len(paragraphs),
                "entity_coverage": round(entity_coverage, 4),
                "avg_pagerank": round(avg_pagerank, 8),
                "avg_in_degree": round(avg_in_degree, 2),
                "freshness_score": round(freshness_score, 4),
                "conversion_score": round(conversion_score, 4),
                "avg_word_count": round(avg(word_counts), 1) if word_counts else 0.0,
                "domain_page_share": round(page_count / total_domain_pages, 4),
                "x": x,
                "y": y,
                "top_pages": pages,
                "keywords": keywords,
                "paragraphs": paragraphs,
                "links": links,
            })
            entries[key] = entry
        entries_by_domain[proj.domain] = entries

    domain_totals = {}
    for domain, entries in entries_by_domain.items():
        domain_totals[domain] = {
            "traffic": sum(_safe_int(entry.get("traffic")) for entry in entries.values()),
            "keywords": sum(_safe_int(entry.get("keyword_rows")) for entry in entries.values()),
            "pages": max(_safe_int(next((p.metrics.get("page_count") for p in projects if p.domain == domain), 0)), len(next((p.pages for p in projects if p.domain == domain), [])), 1),
        }

    cluster_keys = sorted({key for entries in entries_by_domain.values() for key in entries})
    clusters: list[dict] = []
    leaderboard: list[dict] = []
    semantic_points: list[dict] = []
    metric_specs = [
        ("traffic", "Traffic"),
        ("keyword_rows", "Keywords"),
        ("pages", "Pages"),
        ("entity_coverage", "Entity coverage"),
        ("freshness_score", "Freshness"),
        ("avg_in_degree", "Link support"),
        ("conversion_score", "Conversion"),
    ]

    for key in cluster_keys:
        domain_rows = []
        labels = [entries_by_domain.get(domain, {}).get(key, {}).get("cluster_label") for domain in domains if entries_by_domain.get(domain, {}).get(key)]
        label = str(next((v for v in labels if v), key))
        for domain in domains:
            entry = dict(entries_by_domain.get(domain, {}).get(key) or {
                "domain": domain,
                "cluster": key,
                "cluster_label": label,
                "traffic": 0,
                "keyword_rows": 0,
                "keywords_total": 0,
                "avg_position": 0.0,
                "pages": 0,
                "paragraph_count": 0,
                "entity_coverage": 0.0,
                "avg_pagerank": 0.0,
                "avg_in_degree": 0.0,
                "freshness_score": 0.0,
                "conversion_score": 0.0,
                "avg_word_count": 0.0,
                "domain_page_share": 0.0,
                "top_pages": [],
                "keywords": [],
                "paragraphs": [],
                "links": [],
                "x": None,
                "y": None,
            })
            totals = domain_totals.get(domain, {})
            traffic_share = _safe_int(entry.get("traffic")) / max(1, _safe_int(totals.get("traffic")))
            keyword_share = _safe_int(entry.get("keyword_rows")) / max(1, _safe_int(totals.get("keywords")))
            position_score = max(0.0, (21.0 - _safe_float(entry.get("avg_position"), 21.0)) / 20.0) if _safe_float(entry.get("avg_position")) else 0.0
            authority_score = min(1.0, _safe_float(entry.get("avg_in_degree")) / 8.0)
            strength = (
                traffic_share * 32.0
                + keyword_share * 18.0
                + position_score * 15.0
                + _safe_float(entry.get("entity_coverage")) * 10.0
                + authority_score * 10.0
                + _safe_float(entry.get("freshness_score")) * 8.0
                + _safe_float(entry.get("conversion_score")) * 7.0
            )
            raw_strength = (
                math.log1p(_safe_int(entry.get("traffic"))) * 18.0
                + math.sqrt(max(0, _safe_int(entry.get("keyword_rows")))) * 8.0
                + _safe_int(entry.get("top3_keywords")) * 4.0
                + authority_score * 10.0
            )
            entry.update({
                "traffic_share": round(traffic_share, 4),
                "keyword_share": round(keyword_share, 4),
                "position_score": round(position_score, 4),
                "authority_score": round(authority_score, 4),
                "strength_score": round(strength, 2),
                "raw_strength_score": round(raw_strength, 2),
            })
            domain_rows.append(entry)
            if entry.get("x") is not None and entry.get("y") is not None:
                for metric_key, metric_label in metric_specs:
                    semantic_points.append({
                        "domain": domain,
                        "cluster": key,
                        "cluster_label": label,
                        "metric_type": metric_key,
                        "metric_label": metric_label,
                        "metric_value": _safe_float(entry.get(metric_key)),
                        "traffic": _safe_int(entry.get("traffic")),
                        "keywords": _safe_int(entry.get("keyword_rows")),
                        "pages": _safe_int(entry.get("pages")),
                        "strength_score": _safe_float(entry.get("strength_score")),
                        "x": float(entry["x"]),
                        "y": float(entry["y"]),
                    })
        leader = max(domain_rows, key=lambda row: (_safe_float(row.get("strength_score")), _safe_int(row.get("traffic"))))
        runner = max([r for r in domain_rows if r.get("domain") != leader.get("domain")] or [{}], key=lambda row: (_safe_float(row.get("strength_score")), _safe_int(row.get("traffic"))))
        total_traffic = sum(_safe_int(row.get("traffic")) for row in domain_rows)
        total_keywords = sum(_safe_int(row.get("keyword_rows")) for row in domain_rows)
        traffic_gap = sum(max(0, _safe_int(leader.get("traffic")) - _safe_int(row.get("traffic"))) for row in domain_rows if row.get("domain") != leader.get("domain"))
        matrix_domains = [
            {
                "domain": row.get("domain"),
                "traffic": _safe_int(row.get("traffic")),
                "keyword_rows": _safe_int(row.get("keyword_rows")),
                "pages": _safe_int(row.get("pages")),
                "paragraph_count": _safe_int(row.get("paragraph_count")),
                "avg_position": _safe_float(row.get("avg_position")),
                "entity_coverage": _safe_float(row.get("entity_coverage")),
                "avg_in_degree": _safe_float(row.get("avg_in_degree")),
                "freshness_score": _safe_float(row.get("freshness_score")),
                "conversion_score": _safe_float(row.get("conversion_score")),
                "strength_score": _safe_float(row.get("strength_score")),
                "traffic_share": _safe_float(row.get("traffic_share")),
                "keyword_share": _safe_float(row.get("keyword_share")),
            }
            for row in domain_rows
        ]
        cluster_row = {
            "cluster": key,
            "cluster_label": label,
            "winner_domain": leader.get("domain") or "",
            "winner_score": _safe_float(leader.get("strength_score")),
            "runner_domain": runner.get("domain") or "",
            "runner_score": _safe_float(runner.get("strength_score")),
            "score_gap": round(_safe_float(leader.get("strength_score")) - _safe_float(runner.get("strength_score")), 2),
            "total_traffic": total_traffic,
            "total_keywords": total_keywords,
            "traffic_gap": traffic_gap,
            "domains": domain_rows,
        }
        clusters.append(cluster_row)
        leaderboard.append({
            "cluster": key,
            "cluster_label": label,
            "winner_domain": leader.get("domain") or "",
            "winner_score": _safe_float(leader.get("strength_score")),
            "winner_traffic": _safe_int(leader.get("traffic")),
            "runner_domain": runner.get("domain") or "",
            "runner_score": _safe_float(runner.get("strength_score")),
            "score_gap": cluster_row["score_gap"],
            "traffic_gap": traffic_gap,
            "total_traffic": total_traffic,
            "total_keywords": total_keywords,
            "top_page": (leader.get("top_pages") or [{}])[0],
            "top_keywords": (leader.get("keywords") or [])[:5],
        })

    clusters.sort(key=lambda r: (_safe_int(r.get("total_traffic")), _safe_float(r.get("score_gap"))), reverse=True)
    leaderboard.sort(key=lambda r: (_safe_int(r.get("total_traffic")), _safe_float(r.get("score_gap"))), reverse=True)
    matrix = [
        {
            "cluster": row["cluster"],
            "cluster_label": row["cluster_label"],
            "winner_domain": row["winner_domain"],
            "domains": [
                {
                    "domain": domain_row.get("domain"),
                    "strength_score": domain_row.get("strength_score"),
                    "traffic": domain_row.get("traffic"),
                    "keywords": domain_row.get("keyword_rows"),
                    "pages": domain_row.get("pages"),
                }
                for domain_row in row["domains"]
            ],
        }
        for row in clusters[:120]
    ]
    return {
        "summary": {
            "clusters": len(clusters),
            "semantic_points": len(semantic_points),
            "metric_facets": len(metric_specs),
            "domains": len(domains),
        },
        "metric_facets": [{"key": key, "label": label} for key, label in metric_specs],
        "leaderboard": leaderboard[:160],
        "matrix": matrix,
        "clusters": clusters[:160],
        "semantic_points": semantic_points[:1600],
        "export_note": "All cluster metrics and drilldowns are embedded in comparison.json under strongest_clusters.",
    }


def _winning_pattern_transfer_payload(
    projects: list[_Project],
    strongest_clusters: dict,
    keyword_cluster_gaps: dict,
    pattern_transplants: dict,
) -> dict:
    domains = [p.domain for p in projects]
    patterns: dict[str, dict] = {}
    coverage: dict[str, dict[str, dict]] = defaultdict(dict)
    recommendations: list[dict] = []
    clusters = strongest_clusters.get("clusters") or []
    cluster_lookup = {row.get("cluster"): row for row in clusters}
    url_cluster: dict[str, dict[str, dict]] = defaultdict(dict)

    for cluster in clusters:
        key = cluster.get("cluster") or ""
        for domain_row in cluster.get("domains") or []:
            domain = domain_row.get("domain") or ""
            for page in domain_row.get("top_pages") or []:
                if page.get("url"):
                    _store_url_lookup(url_cluster[domain], page.get("url"), {"cluster": key, "cluster_label": cluster.get("cluster_label") or key})

    def lookup_url_cluster(domain: str, url: object) -> dict:
        if not url:
            return {}
        return _lookup_url(url_cluster.get(domain, {}), url)

    def normalize(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:70] or "pattern"

    def category_for(kind: str, label: str = "") -> str:
        text = f"{kind} {label}".lower()
        if any(token in text for token in ("link", "anchor", "inlink", "pagerank")):
            return "link"
        if "schema" in text or "structured" in text:
            return "schema"
        if "fresh" in text or "date" in text or "updated" in text:
            return "freshness"
        if any(token in text for token in ("cta", "form", "conversion", "lead", "demo", "trial", "contact")):
            return "conversion"
        return "content"

    def effort_for(category: str, kind: str) -> str:
        if category == "freshness":
            return "low"
        if category in {"schema", "link", "conversion"}:
            return "medium"
        if kind in {"paragraph_archetype", "content_depth"}:
            return "high"
        return "medium"

    def confidence_score(intent_match: float, opportunity: int, prevalence: float, source_strength: float, effort: str) -> float:
        effort_penalty = {"low": 0.0, "medium": 0.06, "high": 0.12}.get(effort, 0.08)
        raw = 0.34 + intent_match * 0.22 + min(0.18, math.log1p(max(0, opportunity)) / 40.0) + prevalence * 0.16 + min(0.16, source_strength / 100.0 * 0.16) - effort_penalty
        return round(max(0.25, min(0.95, raw)), 3)

    def priority_score(opportunity: int, confidence: float, source_strength: float, effort: str) -> float:
        effort_boost = {"low": 8.0, "medium": 4.0, "high": 0.0}.get(effort, 2.0)
        return round(min(100.0, math.log1p(max(0, opportunity)) * 9.0 + confidence * 36.0 + min(22.0, source_strength / 100.0 * 22.0) + effort_boost), 2)

    def ensure_pattern(
        pattern_key: str,
        *,
        category: str,
        cluster_key: str,
        cluster_label: str,
        title: str,
        source_domain: str,
        source_evidence: dict,
    ) -> dict:
        pattern = patterns.setdefault(pattern_key, {
            "pattern_key": pattern_key,
            "category": category,
            "cluster": cluster_key,
            "cluster_label": cluster_label,
            "title": title,
            "source_domain": source_domain,
            "source_evidence": source_evidence,
            "target_recommendations": [],
            "priority_score": 0.0,
            "confidence": 0.0,
        })
        coverage[pattern_key][source_domain] = {
            "domain": source_domain,
            "status": "source",
            "covered": True,
            "recommendations": 0,
        }
        return pattern

    def add_recommendation(pattern: dict, rec: dict) -> None:
        pattern["target_recommendations"].append(rec)
        pattern["priority_score"] = max(_safe_float(pattern.get("priority_score")), _safe_float(rec.get("priority_score")))
        pattern["confidence"] = max(_safe_float(pattern.get("confidence")), _safe_float(rec.get("confidence")))
        recommendations.append({"pattern_key": pattern["pattern_key"], **rec})
        target_domain = rec.get("target_domain") or rec.get("domain") or ""
        if target_domain:
            coverage[pattern["pattern_key"]][target_domain] = {
                "domain": target_domain,
                "status": "gap",
                "covered": False,
                "recommendations": coverage[pattern["pattern_key"]].get(target_domain, {}).get("recommendations", 0) + 1,
            }

    # Section-level competitor gaps already encode content, schema, freshness,
    # and internal-link features that the winning cluster has and the target lacks.
    for rec in keyword_cluster_gaps.get("recommendations") or []:
        source_domain = rec.get("competitor_domain") or ""
        target_domain = rec.get("domain") or ""
        cluster_key = rec.get("cluster") or ""
        if not source_domain or not target_domain or source_domain == target_domain:
            continue
        kind = rec.get("missing_type") or "content"
        label = rec.get("missing_element") or kind
        category = category_for(kind, label)
        effort = effort_for(category, kind)
        cluster = cluster_lookup.get(cluster_key) or {}
        leader = next((d for d in (cluster.get("domains") or []) if d.get("domain") == source_domain), {})
        target = next((d for d in (cluster.get("domains") or []) if d.get("domain") == target_domain), {})
        opportunity = _safe_int(rec.get("traffic_opportunity"))
        prevalence = _safe_float(rec.get("competitor_prevalence"), 0.5)
        source_strength = _safe_float(leader.get("strength_score"), _safe_float(rec.get("leader_traffic")))
        confidence = confidence_score(1.0, opportunity, prevalence, source_strength, effort)
        priority = priority_score(opportunity, confidence, source_strength, effort)
        pattern_key = f"winning::{cluster_key}::{category}::{normalize(kind)}::{normalize(label)}"
        pattern = ensure_pattern(
            pattern_key,
            category=category,
            cluster_key=cluster_key,
            cluster_label=rec.get("cluster_label") or cluster_key,
            title=f"{category.title()} pattern: {label}",
            source_domain=source_domain,
            source_evidence={
                "source_url": rec.get("competitor_url") or "",
                "source_title": rec.get("competitor_title") or "",
                "leader_traffic": _safe_int(rec.get("leader_traffic")),
                "source_strength": round(source_strength, 2),
                "competitor_prevalence": round(prevalence, 3),
                "evidence_type": kind,
            },
        )
        add_recommendation(pattern, {
            "category": category,
            "cluster": cluster_key,
            "cluster_label": rec.get("cluster_label") or cluster_key,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "target_url": rec.get("target_url") or target.get("top_url") or "",
            "target_title": rec.get("target_title") or target.get("top_title") or "",
            "concrete_change": rec.get("action") or "Copy the winning pattern in a domain-specific way.",
            "missing_element": label,
            "implementation_effort": effort,
            "opportunity": opportunity,
            "priority_score": priority,
            "confidence": confidence,
            "intent_match": 1.0,
            "page_type_match": 0.75,
        })

    # Metric-led patterns from the strongest-cluster payload catch cases where
    # the gap is structural rather than a single missing heading/entity/schema row.
    metric_specs = [
        {
            "category": "content",
            "kind": "content_depth",
            "label": "paragraph depth and topical examples",
            "metric": "paragraph_count",
            "condition": lambda leader, target: _safe_int(leader.get("paragraph_count")) >= 2 and _safe_int(target.get("paragraph_count")) < max(1, _safe_int(leader.get("paragraph_count")) * 0.6),
            "action": "Add explanatory paragraphs that cover the same user questions and examples as the winning cluster, rewritten for this domain.",
        },
        {
            "category": "link",
            "kind": "internal_link_support",
            "label": "strong internal inlink support",
            "metric": "avg_in_degree",
            "condition": lambda leader, target: _safe_float(leader.get("avg_in_degree")) >= 3 and _safe_float(target.get("avg_in_degree")) < max(2.0, _safe_float(leader.get("avg_in_degree")) * 0.6),
            "action": "Add descriptive internal links from relevant hubs and cluster-neighbor pages to the target page.",
        },
        {
            "category": "freshness",
            "kind": "freshness",
            "label": "visible freshness maintenance",
            "metric": "freshness_score",
            "condition": lambda leader, target: _safe_float(leader.get("freshness_score")) >= 0.7 and _safe_float(target.get("freshness_score")) < 0.45,
            "action": "Refresh the target page and expose a visible published or updated date where the content supports it.",
        },
        {
            "category": "conversion",
            "kind": "conversion_surface",
            "label": "clear conversion surface",
            "metric": "conversion_score",
            "condition": lambda leader, target: _safe_float(leader.get("conversion_score")) >= 0.65 and _safe_float(target.get("conversion_score")) < 0.4,
            "action": "Add a clear primary CTA or lead path comparable to the winning page type.",
        },
    ]

    for cluster in clusters:
        leader = next((d for d in cluster.get("domains") or [] if d.get("domain") == cluster.get("winner_domain")), {})
        if not leader:
            continue
        source_domain = leader.get("domain") or ""
        cluster_key = cluster.get("cluster") or ""
        cluster_label = cluster.get("cluster_label") or cluster_key
        for target in cluster.get("domains") or []:
            target_domain = target.get("domain") or ""
            if not target_domain or target_domain == source_domain:
                continue
            opportunity = max(0, _safe_int(leader.get("traffic")) - _safe_int(target.get("traffic")))
            if opportunity <= 0 and _safe_float(cluster.get("score_gap")) < 8:
                continue
            for spec in metric_specs:
                if not spec["condition"](leader, target):
                    continue
                category = spec["category"]
                effort = effort_for(category, spec["kind"])
                metric = spec["metric"]
                source_strength = _safe_float(leader.get("strength_score"))
                confidence = confidence_score(1.0, opportunity, 1.0, source_strength, effort)
                priority = priority_score(opportunity, confidence, source_strength, effort)
                pattern_key = f"winning::{cluster_key}::{category}::{normalize(spec['kind'])}"
                pattern = ensure_pattern(
                    pattern_key,
                    category=category,
                    cluster_key=cluster_key,
                    cluster_label=cluster_label,
                    title=f"{category.title()} pattern: {spec['label']}",
                    source_domain=source_domain,
                    source_evidence={
                        "source_url": (leader.get("top_pages") or [{}])[0].get("url") or "",
                        "source_title": (leader.get("top_pages") or [{}])[0].get("title") or "",
                        "source_strength": round(source_strength, 2),
                        "source_metric": metric,
                        "source_metric_value": leader.get(metric),
                        "target_metric_value": target.get(metric),
                        "evidence_type": spec["kind"],
                    },
                )
                add_recommendation(pattern, {
                    "category": category,
                    "cluster": cluster_key,
                    "cluster_label": cluster_label,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "target_url": (target.get("top_pages") or [{}])[0].get("url") or target.get("top_url") or "",
                    "target_title": (target.get("top_pages") or [{}])[0].get("title") or target.get("top_title") or "",
                    "concrete_change": spec["action"],
                    "missing_element": spec["label"],
                    "implementation_effort": effort,
                    "opportunity": opportunity,
                    "priority_score": priority,
                    "confidence": confidence,
                    "intent_match": 1.0,
                    "page_type_match": 0.7,
                })

    # Reuse high-confidence template/internal-link transplants when their source
    # domain is the cluster winner or when the target URL can be mapped into a
    # winning-cluster gap.
    for rec in pattern_transplants.get("recommendations") or []:
        source_domain = rec.get("source_domain") or ""
        target_domain = rec.get("target_domain") or ""
        if not source_domain or not target_domain or source_domain == target_domain:
            continue
        cluster_info = lookup_url_cluster(target_domain, rec.get("target_url")) or lookup_url_cluster(source_domain, rec.get("suggested_target_url"))
        cluster_key = cluster_info.get("cluster") or ""
        cluster_label = cluster_info.get("cluster_label") or cluster_key or "cross-domain"
        category = category_for(rec.get("pattern_type") or "", f"{rec.get('source_pattern') or ''} {rec.get('suggested_heading') or ''} {rec.get('suggested_anchor') or ''}")
        effort = effort_for(category, rec.get("pattern_type") or "")
        opportunity = _safe_int(rec.get("target_traffic"))
        source_strength = _safe_float(rec.get("priority_score"))
        confidence = max(_safe_float(rec.get("confidence")), confidence_score(0.75 if cluster_key else 0.55, opportunity, 0.75, source_strength, effort))
        priority = max(_safe_float(rec.get("priority_score")), priority_score(opportunity, confidence, source_strength, effort))
        pattern_key = f"winning::{cluster_key or 'cross-domain'}::{category}::{normalize(rec.get('source_pattern') or rec.get('pattern_key'))}"
        pattern = ensure_pattern(
            pattern_key,
            category=category,
            cluster_key=cluster_key,
            cluster_label=cluster_label,
            title=f"{category.title()} pattern: {rec.get('source_pattern') or rec.get('pattern_key')}",
            source_domain=source_domain,
            source_evidence={
                **(rec.get("source_evidence") or {}),
                "source_pattern": rec.get("source_pattern") or "",
                "evidence_type": rec.get("pattern_type") or "",
            },
        )
        add_recommendation(pattern, {
            "category": category,
            "cluster": cluster_key,
            "cluster_label": cluster_label,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "target_url": rec.get("target_url") or "",
            "target_title": rec.get("target_title") or "",
            "concrete_change": rec.get("concrete_change") or "Copy the winning pattern in a domain-specific way.",
            "missing_element": rec.get("source_pattern") or rec.get("suggested_heading") or rec.get("suggested_anchor") or "",
            "implementation_effort": effort,
            "opportunity": opportunity,
            "priority_score": round(priority, 2),
            "confidence": round(confidence, 3),
            "intent_match": 0.75 if cluster_key else 0.55,
            "page_type_match": 0.75,
        })

    pattern_rows = list(patterns.values())
    for pattern in pattern_rows:
        pattern_key = pattern["pattern_key"]
        domain_rows = []
        for domain in domains:
            domain_rows.append(coverage.get(pattern_key, {}).get(domain, {
                "domain": domain,
                "status": "unknown",
                "covered": False,
                "recommendations": 0,
            }))
        pattern["domains"] = domain_rows
        pattern["target_recommendations"].sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("confidence"))), reverse=True)
        pattern["recommendation_count"] = len(pattern["target_recommendations"])

    pattern_rows.sort(key=lambda r: (_safe_float(r.get("priority_score")), len(r.get("target_recommendations") or [])), reverse=True)
    recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("confidence"))), reverse=True)
    type_counts = Counter(row.get("category") or "content" for row in pattern_rows)
    coverage_rows = [
        {
            "pattern_key": row["pattern_key"],
            "title": row.get("title") or "",
            "category": row.get("category") or "",
            "cluster": row.get("cluster") or "",
            "cluster_label": row.get("cluster_label") or "",
            "source_domain": row.get("source_domain") or "",
            "domains": row.get("domains") or [],
            "recommendations": row.get("recommendation_count", 0),
        }
        for row in pattern_rows[:160]
    ]
    return {
        "summary": {
            "patterns": len(pattern_rows),
            "recommendations": len(recommendations),
            "categories": dict(type_counts),
        },
        "patterns": pattern_rows[:220],
        "recommendations": recommendations[:500],
        "coverage": coverage_rows,
    }


def _keyword_content_matrix_payload(
    projects: list[_Project],
    strongest_clusters: dict,
    keyword_cluster_gaps: dict,
    winning_patterns: dict,
) -> dict:
    domains = [p.domain for p in projects]
    gap_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rec_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    winning_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in keyword_cluster_gaps.get("section_diffs") or []:
        gap_lookup[(row.get("cluster") or "", row.get("domain") or "")].append(row)
    for row in keyword_cluster_gaps.get("recommendations") or []:
        rec_lookup[(row.get("cluster") or "", row.get("domain") or "")].append(row)
    for row in winning_patterns.get("recommendations") or []:
        winning_lookup[(row.get("cluster") or "", row.get("target_domain") or "")].append(row)

    def component_scores(row: dict, gaps: list[dict]) -> tuple[dict, list[dict]]:
        missing: list[dict] = []
        top_pages = row.get("top_pages") or []
        keywords = row.get("keywords") or []
        headings = row.get("headings") or []
        paragraphs = row.get("paragraphs") or row.get("paragraph_examples") or []
        entities = row.get("entities") or []
        schema_types = row.get("schema_types") or []
        gap_headings = list(dict.fromkeys(h for gap in gaps for h in (gap.get("missing_headings") or [])))[:8]
        gap_entities = list(dict.fromkeys(e for gap in gaps for e in (gap.get("missing_entities") or [])))[:8]
        gap_schema = list(dict.fromkeys(s for gap in gaps for s in (gap.get("missing_schema") or [])))[:6]
        answer_gap = any(bool(gap.get("answer_block_gap")) for gap in gaps)
        freshness_gap = any(bool(gap.get("freshness_gap")) for gap in gaps)
        link_gap = any(bool(gap.get("link_gap")) for gap in gaps)

        title_score = 1.0 if top_pages or row.get("top_url") else 0.0
        if not title_score:
            missing.append({"type": "title", "label": "target page title", "action": "Create or map a target page for this keyword cluster."})
        heading_score = 1.0 if headings and not gap_headings else (0.55 if headings else 0.0)
        for heading in gap_headings[:4]:
            missing.append({"type": "heading", "label": heading, "action": "Add or rename an H2/H3 section matching the winning outline."})
        paragraph_score = min(1.0, _safe_int(row.get("paragraph_count")) / 3.0) if paragraphs else 0.0
        if paragraph_score < 0.5:
            missing.append({"type": "paragraph", "label": "supporting paragraph examples", "action": "Add explanatory paragraphs that directly support ranking keywords."})
        entity_score = max(_safe_float(row.get("entity_coverage")), min(1.0, len(entities) / 10.0))
        for entity in gap_entities[:4]:
            missing.append({"type": "entity", "label": entity, "action": "Cover this missing entity in body copy or headings."})
        schema_score = 1.0 if schema_types and not gap_schema else (0.55 if schema_types else 0.0)
        for schema in gap_schema[:3]:
            missing.append({"type": "schema", "label": schema, "action": "Add schema when the page content supports it."})
        answer_score = _safe_float(row.get("answer_block_share"))
        if answer_gap or answer_score < 0.4:
            missing.append({"type": "answer_block", "label": "snippet-ready answer block", "action": "Add a concise answer block for the cluster's primary questions."})
        link_score = min(1.0, _safe_float(row.get("avg_in_degree")) / 4.0)
        if link_gap or link_score < 0.5:
            missing.append({"type": "link", "label": "descriptive internal links", "action": "Add internal links from relevant hubs and neighboring pages."})
        freshness_score = _safe_float(row.get("freshness_score"))
        if freshness_gap or freshness_score < 0.45:
            missing.append({"type": "freshness", "label": "fresh or updated evidence", "action": "Refresh the page and expose date evidence where appropriate."})
        conversion_score = _safe_float(row.get("conversion_score"))

        components = {
            "title": round(title_score, 4),
            "headings": round(heading_score, 4),
            "paragraphs": round(paragraph_score, 4),
            "entities": round(entity_score, 4),
            "schema": round(schema_score, 4),
            "answer_blocks": round(answer_score, 4),
            "links": round(link_score, 4),
            "freshness": round(freshness_score, 4),
            "conversion": round(conversion_score, 4),
        }
        return components, missing

    weights = {
        "title": 12.0,
        "headings": 13.0,
        "paragraphs": 13.0,
        "entities": 14.0,
        "schema": 10.0,
        "answer_blocks": 10.0,
        "links": 12.0,
        "freshness": 8.0,
        "conversion": 8.0,
    }
    matrix: list[dict] = []
    flat_cells: list[dict] = []
    intents = Counter()
    directories = Counter()

    for cluster in strongest_clusters.get("clusters") or []:
        cluster_key = cluster.get("cluster") or ""
        cluster_label = cluster.get("cluster_label") or cluster_key
        domain_cells = []
        winner = next((d for d in cluster.get("domains") or [] if d.get("domain") == cluster.get("winner_domain")), {})
        winner_traffic = _safe_int(winner.get("traffic"))
        row_intents = Counter()
        row_directories = Counter()
        for domain_row in cluster.get("domains") or []:
            domain = domain_row.get("domain") or ""
            gaps = gap_lookup.get((cluster_key, domain), [])
            components, missing = component_scores(domain_row, gaps)
            support_score = round(sum(components[key] * weight for key, weight in weights.items()), 2)
            top_pages = domain_row.get("top_pages") or []
            top_keywords = domain_row.get("keywords") or []
            for intent in domain_row.get("intents") or []:
                intents[str(intent)] += 1
                row_intents[str(intent)] += 1
            for page in top_pages:
                directory = page.get("section") or domain_row.get("section") or ""
                if directory:
                    directories[directory] += 1
                    row_directories[directory] += 1
            recommendations = []
            for rec in rec_lookup.get((cluster_key, domain), [])[:8]:
                recommendations.append({
                    "source": "cluster_gap",
                    "type": rec.get("missing_type") or "",
                    "label": rec.get("missing_element") or "",
                    "action": rec.get("action") or "",
                    "priority_score": _safe_float(rec.get("traffic_opportunity")),
                    "confidence": _safe_float(rec.get("competitor_prevalence")),
                    "target_url": rec.get("target_url") or "",
                    "target_title": rec.get("target_title") or "",
                })
            for rec in winning_lookup.get((cluster_key, domain), [])[:8]:
                recommendations.append({
                    "source": "winning_pattern",
                    "type": rec.get("category") or "",
                    "label": rec.get("missing_element") or "",
                    "action": rec.get("concrete_change") or "",
                    "priority_score": _safe_float(rec.get("priority_score")),
                    "confidence": _safe_float(rec.get("confidence")),
                    "target_url": rec.get("target_url") or "",
                    "target_title": rec.get("target_title") or "",
                })
            recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("confidence"))), reverse=True)
            cell = {
                "domain": domain,
                "cluster": cluster_key,
                "cluster_label": cluster_label,
                "support_score": support_score,
                "components": components,
                "missing": missing[:18],
                "recommendations": recommendations[:12],
                "target_url": (top_pages[0] or {}).get("url") if top_pages else domain_row.get("top_url") or "",
                "target_title": (top_pages[0] or {}).get("title") if top_pages else domain_row.get("top_title") or "",
                "traffic": _safe_int(domain_row.get("traffic")),
                "traffic_potential": max(0, winner_traffic - _safe_int(domain_row.get("traffic"))),
                "keywords": _safe_int(domain_row.get("keyword_rows")),
                "avg_position": _safe_float(domain_row.get("avg_position")),
                "pages": _safe_int(domain_row.get("pages")),
                "paragraphs": _safe_int(domain_row.get("paragraph_count")),
                "top_pages": top_pages[:6],
                "top_keywords": top_keywords[:8],
                "intents": list(domain_row.get("intents") or [])[:6],
                "directory": (top_pages[0] or {}).get("section") if top_pages else domain_row.get("section") or "",
            }
            domain_cells.append(cell)
            flat_cells.append(cell)
        matrix.append({
            "cluster": cluster_key,
            "cluster_label": cluster_label,
            "winner_domain": cluster.get("winner_domain") or "",
            "total_traffic": _safe_int(cluster.get("total_traffic")),
            "traffic_potential": sum(_safe_int(cell.get("traffic_potential")) for cell in domain_cells),
            "intents": [intent for intent, _ in row_intents.most_common(6)],
            "directories": [directory for directory, _ in row_directories.most_common(6)],
            "domains": domain_cells,
        })

    matrix.sort(key=lambda row: (_safe_int(row.get("traffic_potential")), _safe_int(row.get("total_traffic"))), reverse=True)
    flat_cells.sort(key=lambda row: (_safe_int(row.get("traffic_potential")), -_safe_float(row.get("support_score"))), reverse=True)
    return {
        "summary": {
            "clusters": len(matrix),
            "cells": len(flat_cells),
            "domains": len(domains),
        },
        "domains": domains,
        "components": [{"key": key, "label": key.replace("_", " ").title(), "weight": weight} for key, weight in weights.items()],
        "filters": {
            "intents": [key for key, _ in intents.most_common(30)],
            "directories": [key for key, _ in directories.most_common(40)],
        },
        "matrix": matrix[:180],
        "cells": flat_cells[:900],
    }


def _paragraph_archetype_comparison(projects: list[_Project], strongest_clusters: dict) -> dict:
    domains = [p.domain for p in projects]
    archetype_defs = {
        "intro": "Intro",
        "definition": "Definition",
        "features": "Feature block",
        "use_case": "Use case",
        "faq": "FAQ",
        "comparison": "Comparison",
        "pricing": "Pricing",
        "proof": "Proof",
        "process": "Process",
        "integration": "Integration",
        "security": "Security",
        "conversion": "Conversion",
        "explanation": "Explanation",
    }
    cluster_rows = strongest_clusters.get("clusters") or []
    cluster_lookup = {row.get("cluster"): row for row in cluster_rows}
    cluster_order = {row.get("cluster"): i for i, row in enumerate(cluster_rows)}
    page_cluster_lookup: dict[str, dict[str, dict]] = defaultdict(dict)
    id_maps: dict[str, dict[str, str]] = {}

    for proj in projects:
        id_to_key: dict[str, str] = {}
        for row in ((proj.ahrefs or {}).get("clusters") or []):
            key = _cluster_key(row.get("label") or row.get("key") or row.get("cluster"))
            if key == "unknown":
                continue
            for ident in (row.get("cluster"), row.get("key")):
                if ident is not None:
                    id_to_key[str(ident)] = key
        id_maps[proj.domain] = id_to_key

    for cluster in cluster_rows:
        key = cluster.get("cluster") or ""
        for domain_row in cluster.get("domains") or []:
            domain = domain_row.get("domain") or ""
            for page in domain_row.get("top_pages") or []:
                if page.get("url"):
                    _store_url_lookup(page_cluster_lookup[domain], page.get("url"), {"cluster": key, "cluster_label": cluster.get("cluster_label") or key})

    def paragraph_cluster(domain: str, row: dict) -> str:
        ref = row.get("cluster")
        if ref is not None and str(ref) in id_maps.get(domain, {}):
            return id_maps[domain][str(ref)]
        found = _lookup_url(page_cluster_lookup.get(domain, {}), row.get("url"))
        if found.get("cluster"):
            return str(found["cluster"])
        key = _cluster_key(row.get("cluster_label") or "")
        return key if key != "unknown" else ""

    def classify_archetype(text: str, paragraph_index: int | None = None) -> str:
        lower = re.sub(r"\s+", " ", str(text or "").lower())
        if "?" in lower or re.search(r"\b(faq|frequently asked|common questions)\b", lower):
            return "faq"
        if re.search(r"\b(pricing|price|cost|plan|subscription|per month|\$|€)\b", lower):
            return "pricing"
        if re.search(r"\b(vs\.?|versus|alternative|compare|comparison|difference between)\b", lower):
            return "comparison"
        if re.search(r"\b(case study|customer|review|testimonial|trusted by|g2|capterra|certified|award)\b", lower):
            return "proof"
        if re.search(r"\b(use case|for teams|for businesses|when you need|scenario|example)\b", lower):
            return "use_case"
        if re.search(r"\b(feature|capability|includes|offers|built-in|dashboard|workflow|automation)\b", lower):
            return "features"
        if re.search(r"\b(how to|step|steps|process|guide|workflow|setup|configure)\b", lower):
            return "process"
        if re.search(r"\b(integration|integrates|api|webhook|zapier|slack|salesforce|hubspot|shopify|teams|zendesk)\b", lower):
            return "integration"
        if re.search(r"\b(security|gdpr|soc 2|hipaa|compliance|encryption|privacy|sla)\b", lower):
            return "security"
        if re.search(r"\b(book a demo|start trial|get started|contact sales|sign up|request demo)\b", lower):
            return "conversion"
        if re.search(r"\b(is a|is an|refers to|means|defined as|what is)\b", lower):
            return "definition"
        if paragraph_index == 0:
            return "intro"
        return "explanation"

    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    timelines: list[dict] = []
    page_timeline: dict[tuple[str, str], dict] = {}
    page_type_filter = Counter()
    intent_filter = Counter()

    for proj in projects:
        page_type_lookup = _url_lookup_from_rows(((proj.page_types or {}).get("per_page") or []), ("url",))
        page_lookup = _url_lookup_from_rows(proj.pages or [], ("url",))
        for row in proj.ahrefs_semantic_rows or []:
            if row.get("type") != "paragraph":
                continue
            cluster_key = paragraph_cluster(proj.domain, row)
            if not cluster_key:
                continue
            cluster = cluster_lookup.get(cluster_key) or {}
            domain_cluster = next((d for d in (cluster.get("domains") or []) if d.get("domain") == proj.domain), {})
            page = _lookup_url(page_lookup, row.get("url"))
            page_type_row = _lookup_url(page_type_lookup, row.get("url"))
            page_type = page_type_row.get("page_type") or domain_cluster.get("page_type") or page.get("section") or ""
            intents = list(domain_cluster.get("intents") or cluster.get("intents") or [])
            for intent in intents:
                intent_filter[str(intent)] += 1
            if page_type:
                page_type_filter[page_type] += 1
            archetype = classify_archetype(row.get("label") or "", _safe_int(row.get("paragraph_index"), -1))
            key = (cluster_key, archetype)
            stats = by_key[key].setdefault(proj.domain, {
                "domain": proj.domain,
                "cluster": cluster_key,
                "cluster_label": cluster.get("cluster_label") or cluster_key,
                "archetype": archetype,
                "label": archetype_defs.get(archetype, archetype),
                "count": 0,
                "traffic": 0,
                "page_types": Counter(),
                "intents": Counter(),
                "examples": [],
            })
            stats["count"] += 1
            stats["traffic"] += _safe_int(row.get("traffic"))
            if page_type:
                stats["page_types"][page_type] += 1
            for intent in intents:
                stats["intents"][str(intent)] += 1
            if len(stats["examples"]) < 6:
                stats["examples"].append({
                    "url": row.get("url") or "",
                    "title": page.get("title") or row.get("url") or "",
                    "paragraph_index": row.get("paragraph_index"),
                    "excerpt": str(row.get("label") or "")[:260],
                    "traffic": _safe_int(row.get("traffic")),
                    "page_type": page_type,
                })
            timeline_key = (proj.domain, row.get("url") or "")
            timeline = page_timeline.setdefault(timeline_key, {
                "domain": proj.domain,
                "cluster": cluster_key,
                "cluster_label": cluster.get("cluster_label") or cluster_key,
                "url": row.get("url") or "",
                "title": page.get("title") or row.get("url") or "",
                "page_type": page_type,
                "intents": intents[:6],
                "traffic": _safe_int(row.get("traffic")),
                "segments": [],
            })
            timeline["traffic"] = max(_safe_int(timeline.get("traffic")), _safe_int(row.get("traffic")))
            if len(timeline["segments"]) < 30:
                timeline["segments"].append({
                    "paragraph_index": row.get("paragraph_index"),
                    "archetype": archetype,
                    "label": archetype_defs.get(archetype, archetype),
                    "excerpt": str(row.get("label") or "")[:220],
                })

    for timeline in page_timeline.values():
        timeline["segments"].sort(key=lambda r: _safe_int(r.get("paragraph_index")))
        timelines.append(timeline)
    timelines.sort(key=lambda r: (_safe_int(r.get("traffic")), len(r.get("segments") or [])), reverse=True)

    matrix = []
    recommendations: list[dict] = []
    for (cluster_key, archetype), values in by_key.items():
        cluster = cluster_lookup.get(cluster_key) or {}
        leader_domain = cluster.get("winner_domain") or max(values.values(), key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("count")))).get("domain")
        source = values.get(leader_domain) or max(values.values(), key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("count"))))
        row_domains = []
        for domain in domains:
            current = values.get(domain)
            if current:
                current = dict(current)
                current["page_types"] = [key for key, _ in current["page_types"].most_common(6)]
                current["intents"] = [key for key, _ in current["intents"].most_common(6)]
                current["covered"] = True
                row_domains.append(current)
                continue
            target_cluster = next((d for d in (cluster.get("domains") or []) if d.get("domain") == domain), {})
            traffic_opp = max(0, _safe_int(source.get("traffic")) - _safe_int(target_cluster.get("traffic")))
            missing = {
                "domain": domain,
                "cluster": cluster_key,
                "cluster_label": cluster.get("cluster_label") or cluster_key,
                "archetype": archetype,
                "label": archetype_defs.get(archetype, archetype),
                "count": 0,
                "traffic": 0,
                "page_types": [],
                "intents": list(target_cluster.get("intents") or [])[:6],
                "examples": [],
                "covered": False,
                "traffic_opportunity": traffic_opp,
            }
            row_domains.append(missing)
            if traffic_opp > 0 or _safe_int(target_cluster.get("traffic")) > 0:
                confidence = round(min(0.92, 0.42 + min(0.18, math.log1p(traffic_opp) / 45.0) + min(0.18, _safe_int(source.get("count")) * 0.05) + 0.14), 3)
                priority = round(min(100.0, math.log1p(traffic_opp) * 9.0 + confidence * 34.0 + min(20.0, _safe_int(source.get("count")) * 5.0)), 2)
                recommendations.append({
                    "cluster": cluster_key,
                    "cluster_label": cluster.get("cluster_label") or cluster_key,
                    "archetype": archetype,
                    "label": archetype_defs.get(archetype, archetype),
                    "source_domain": source.get("domain"),
                    "source_examples": (source.get("examples") or [])[:4],
                    "target_domain": domain,
                    "target_url": (target_cluster.get("top_pages") or [{}])[0].get("url") or "",
                    "target_title": (target_cluster.get("top_pages") or [{}])[0].get("title") or "",
                    "traffic_opportunity": traffic_opp,
                    "priority_score": priority,
                    "confidence": confidence,
                    "action": f"Add a {archetype_defs.get(archetype, archetype).lower()} section that matches the intent of the stronger page without copying wording.",
                })
        total_count = sum(_safe_int(d.get("count")) for d in row_domains)
        total_traffic = sum(_safe_int(d.get("traffic")) for d in row_domains)
        matrix.append({
            "cluster": cluster_key,
            "cluster_label": cluster.get("cluster_label") or cluster_key,
            "archetype": archetype,
            "label": archetype_defs.get(archetype, archetype),
            "winner_domain": leader_domain,
            "total_count": total_count,
            "total_traffic": total_traffic,
            "page_types": sorted({pt for d in row_domains for pt in (d.get("page_types") or [])}),
            "intents": sorted({intent for d in row_domains for intent in (d.get("intents") or [])}),
            "domains": row_domains,
        })

    matrix.sort(key=lambda r: (cluster_order.get(r.get("cluster"), 9999), -_safe_int(r.get("total_traffic")), r.get("archetype") or ""))
    recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("confidence"))), reverse=True)
    return {
        "summary": {
            "matrix_rows": len(matrix),
            "recommendations": len(recommendations),
            "timelines": len(timelines),
        },
        "archetypes": [{"key": key, "label": label} for key, label in archetype_defs.items()],
        "filters": {
            "page_types": [key for key, _ in page_type_filter.most_common(30)],
            "intents": [key for key, _ in intent_filter.most_common(30)],
        },
        "matrix": matrix[:220],
        "recommendations": recommendations[:400],
        "timelines": timelines[:220],
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


def _internal_link_architecture_comparison(projects: list[_Project]) -> dict:
    domains = [p.domain for p in projects]
    scorecards: list[dict] = []
    cluster_domains: dict[str, dict[str, dict]] = defaultdict(dict)
    cluster_edges: list[dict] = []
    recommendations: list[dict] = []

    def directory_from_url(url: object) -> str:
        try:
            parts = [part for part in (urlsplit(str(url or "")).path or "/").split("/") if part]
        except ValueError:
            return "/"
        if not parts:
            return "/"
        return "/" + parts[0] + "/"

    def cluster_name(value: object, fallback: object = "") -> str:
        raw = str(value or fallback or "").strip()
        if not raw:
            return "root"
        key = _cluster_key(raw)
        return key if key != "unknown" else raw[:80]

    def page_cluster(row: dict) -> str:
        return cluster_name(
            row.get("cluster")
            or row.get("cluster_label")
            or row.get("section")
            or row.get("directory")
            or row.get("page_type")
            or directory_from_url(row.get("url"))
        )

    def money_page_type(value: object) -> bool:
        return str(value or "").lower() in {"product", "service", "home", "contact", "pricing", "demo", "signup"}

    def add_recommendation(row: dict) -> None:
        if not row.get("target_url") or not row.get("source_url"):
            return
        row.setdefault("priority_score", 0.0)
        row["priority_score"] = round(_safe_float(row.get("priority_score")), 2)
        row["confidence"] = round(_safe_float(row.get("confidence"), 0.62), 3)
        recommendations.append(row)

    for proj in projects:
        lg = proj.linkgraph or {}
        page_rows = list(proj.page_link_counts or [])
        row_count = len(page_rows)
        node_count = _safe_int(lg.get("node_count")) or row_count or len(proj.pages)
        total_edges = _safe_int(lg.get("edge_count")) or sum(_safe_int(r.get("out_degree")) for r in page_rows)
        orphan_count = _safe_int(lg.get("orphan_count"), -1)
        if orphan_count < 0:
            orphan_count = sum(1 for r in page_rows if _safe_int(r.get("in_degree")) == 0)
        dead_end_count = _safe_int(lg.get("dead_end_count"), -1)
        if dead_end_count < 0:
            dead_end_count = sum(1 for r in page_rows if _safe_int(r.get("out_degree")) == 0)
        deep_count = _safe_int(lg.get("deep_page_count"), -1)
        if deep_count < 0:
            deep_count = sum(1 for r in page_rows if r.get("click_depth") is not None and _safe_int(r.get("click_depth")) >= 4)

        pagerank_values = [_safe_float(r.get("pagerank")) for r in page_rows if _safe_float(r.get("pagerank")) > 0]
        if not pagerank_values:
            pagerank_values = [_safe_float(r.get("pagerank")) for r in (lg.get("top_authority_pages") or []) if _safe_float(r.get("pagerank")) > 0]
        pr_total = sum(pagerank_values) or 0.0
        pr_sorted = sorted(pagerank_values, reverse=True)
        pagerank_top1_share = (sum(pr_sorted[:1]) / pr_total) if pr_total else 0.0
        pagerank_top5_share = (sum(pr_sorted[:5]) / pr_total) if pr_total else 0.0
        pagerank_hhi = sum((value / pr_total) ** 2 for value in pagerank_values) if pr_total else 0.0

        context_summary = ((lg.get("contextual_link_impact") or {}).get("summary") or {})
        total_context_links = _safe_int(context_summary.get("total_links")) or total_edges
        contextual_links = _safe_int(context_summary.get("main_content_links"))
        template_links = _safe_int(context_summary.get("template_links"))
        if not contextual_links and total_context_links and template_links:
            contextual_links = max(0, total_context_links - template_links)
        contextual_share = contextual_links / max(1, total_context_links) if total_context_links else 0.0
        template_share = template_links / max(1, total_context_links) if total_context_links else 0.0

        twpr = ((lg.get("traffic_weighted_pagerank") or {}).get("summary") or {})
        hdl = ((lg.get("high_demand_low_link") or {}).get("summary") or {})
        hubs = ((lg.get("hub_bottlenecks") or {}).get("summary") or {})
        top_pages_total = sum(_safe_int(r.get("traffic")) for r in ((proj.ahrefs or {}).get("top_pages") or []))
        high_demand_pages = _safe_int(hdl.get("high_demand_low_support_pages") or hdl.get("opportunity_pages"))
        high_demand_traffic = _safe_int(hdl.get("high_demand_low_support_traffic") or hdl.get("opportunity_traffic"))
        classified_pages = _safe_int(hdl.get("classified_top_pages")) or len(((lg.get("high_demand_low_link") or {}).get("pages") or []))
        high_demand_rate = high_demand_pages / max(1, classified_pages) if classified_pages else 0.0
        high_demand_traffic_share = high_demand_traffic / max(1, top_pages_total) if top_pages_total else 0.0
        authority_alignment = _safe_float(twpr.get("authority_traffic_alignment"))
        demand_support_alignment = _safe_float(hdl.get("demand_support_alignment"))
        resilience = _safe_float(hubs.get("architecture_resilience"))

        link_lookup = _page_link_lookup(proj)
        authority_lookup = _authority_lookup(proj)
        money_rows: dict[str, dict] = {}
        for row in ((proj.conversion_balance or {}).get("pages") or []):
            if row.get("money_page") or money_page_type(row.get("page_type")):
                url = row.get("url") or ""
                if url:
                    money_rows[url] = row
        for row in ((lg.get("high_demand_low_link") or {}).get("pages") or []):
            if money_page_type(row.get("page_type")):
                url = row.get("url") or ""
                if url:
                    money_rows.setdefault(url, row)

        money_scores: list[float] = []
        weak_money_pages = 0
        weak_money_traffic = 0
        money_inlinks: list[float] = []
        for url, row in money_rows.items():
            link = _lookup_url(link_lookup, url)
            auth = _lookup_url(authority_lookup, url)
            in_degree = _safe_int(link.get("in_degree") if link else row.get("in_degree"))
            click_depth = link.get("click_depth") if link else row.get("click_depth")
            weighted_pct = _safe_float(auth.get("weighted_pagerank_percentile") or row.get("weighted_pagerank_percentile"))
            depth_component = 0.5
            if click_depth is not None:
                depth_component = max(0.0, min(1.0, 1.0 - (_safe_int(click_depth) - 1) / 5.0))
            support = min(1.0, in_degree / 5.0) * 0.45 + weighted_pct * 0.35 + depth_component * 0.20
            money_scores.append(support)
            money_inlinks.append(float(in_degree))
            if support < 0.45 or row.get("balance_label") == "high_risk_money_page":
                weak_money_pages += 1
                weak_money_traffic += _safe_int(row.get("traffic"))
        money_page_support_score = round(sum(money_scores) / len(money_scores) * 100.0, 2) if money_scores else 0.0
        avg_money_inlinks = sum(money_inlinks) / len(money_inlinks) if money_inlinks else 0.0

        orphan_rate = orphan_count / max(1, node_count)
        dead_end_rate = dead_end_count / max(1, node_count)
        deep_page_rate = deep_count / max(1, node_count)
        balanced_authority = 1.0 - min(1.0, max(0.0, pagerank_top5_share - 0.35) / 0.55)
        money_component = money_page_support_score / 100.0 if money_scores else 0.65
        architecture_score = round(100.0 * (
            (1.0 - orphan_rate) * 0.14
            + (1.0 - dead_end_rate) * 0.10
            + (1.0 - deep_page_rate) * 0.09
            + balanced_authority * 0.12
            + contextual_share * 0.14
            + authority_alignment * 0.14
            + demand_support_alignment * 0.13
            + resilience * 0.08
            + money_component * 0.06
        ), 2)

        scorecards.append({
            "domain": proj.domain,
            "architecture_score": architecture_score,
            "pages": node_count,
            "internal_edges": total_edges,
            "avg_in_degree": round(sum(_safe_int(r.get("in_degree")) for r in page_rows) / max(1, row_count), 2) if row_count else 0.0,
            "avg_out_degree": round(sum(_safe_int(r.get("out_degree")) for r in page_rows) / max(1, row_count), 2) if row_count else 0.0,
            "orphan_count": orphan_count,
            "orphan_rate": round(orphan_rate, 4),
            "dead_end_count": dead_end_count,
            "dead_end_rate": round(dead_end_rate, 4),
            "deep_page_count": deep_count,
            "deep_page_rate": round(deep_page_rate, 4),
            "pagerank_top1_share": round(pagerank_top1_share, 4),
            "pagerank_top5_share": round(pagerank_top5_share, 4),
            "pagerank_gini": round(_gini(pagerank_values), 4),
            "pagerank_hhi": round(pagerank_hhi, 4),
            "contextual_link_share": round(contextual_share, 4),
            "template_link_share": round(template_share, 4),
            "authority_traffic_alignment": round(authority_alignment, 4),
            "demand_support_alignment": round(demand_support_alignment, 4),
            "high_demand_low_link_pages": high_demand_pages,
            "high_demand_low_link_rate": round(high_demand_rate, 4),
            "high_demand_low_link_traffic": high_demand_traffic,
            "high_demand_low_link_traffic_share": round(high_demand_traffic_share, 4),
            "architecture_resilience": round(resilience, 4),
            "bottleneck_pages": _safe_int(hubs.get("bottleneck_pages")),
            "bridge_pages": _safe_int(hubs.get("bridge_pages")),
            "money_pages": len(money_scores),
            "money_page_support_score": money_page_support_score,
            "avg_money_page_inlinks": round(avg_money_inlinks, 2),
            "weak_money_pages": weak_money_pages,
            "weak_money_traffic": weak_money_traffic,
        })

        cluster_bucket: dict[str, dict] = defaultdict(lambda: {
            "domain": proj.domain,
            "cluster": "",
            "label": "",
            "pages": 0,
            "traffic": 0,
            "keywords": 0,
            "inbound_links": 0,
            "outbound_links": 0,
            "orphan_pages": 0,
            "dead_end_pages": 0,
            "deep_pages": 0,
            "pagerank": 0.0,
            "weighted_pagerank": 0.0,
            "underserved_pages": 0,
            "opportunities": 0,
            "opportunity_traffic": 0,
            "avg_support_score": 0.0,
            "support_rows": 0,
            "cross_inbound_links": 0,
            "cross_outbound_links": 0,
            "source_clusters": Counter(),
            "target_clusters": Counter(),
        })

        authority_pages = ((lg.get("traffic_weighted_pagerank") or {}).get("pages") or [])
        source_rows = authority_pages if authority_pages else page_rows
        page_cluster_lookup: dict[str, str] = {}
        for row in source_rows:
            url = row.get("url") or ""
            cluster = page_cluster(row)
            if url:
                page_cluster_lookup[url] = cluster
            bucket = cluster_bucket[cluster]
            bucket["cluster"] = cluster
            bucket["label"] = row.get("cluster") or row.get("cluster_label") or row.get("section") or cluster
            bucket["pages"] += 1
            bucket["traffic"] += _safe_int(row.get("traffic"))
            bucket["keywords"] += _safe_int(row.get("keywords"))
            bucket["inbound_links"] += _safe_int(row.get("in_degree"))
            bucket["outbound_links"] += _safe_int(row.get("out_degree"))
            bucket["pagerank"] += _safe_float(row.get("pagerank"))
            bucket["weighted_pagerank"] += _safe_float(row.get("weighted_pagerank"))
            if _safe_int(row.get("in_degree")) <= 0:
                bucket["orphan_pages"] += 1
            if _safe_int(row.get("out_degree")) <= 0:
                bucket["dead_end_pages"] += 1
            if row.get("click_depth") is not None and _safe_int(row.get("click_depth")) >= 4:
                bucket["deep_pages"] += 1
            if row.get("mismatch_label") in {"high_traffic_low_authority", "ranked_orphan", "buried_demand"}:
                bucket["underserved_pages"] += 1

        for row in ((lg.get("high_demand_low_link") or {}).get("clusters") or []):
            cluster = cluster_name(row.get("cluster") or row.get("label"))
            bucket = cluster_bucket[cluster]
            bucket["cluster"] = cluster
            bucket["label"] = row.get("label") or row.get("cluster") or cluster
            bucket["opportunities"] += _safe_int(row.get("opportunities"))
            bucket["opportunity_traffic"] += _safe_int(row.get("opportunity_traffic"))
            bucket["avg_support_score"] += _safe_float(row.get("avg_support_score"))
            bucket["support_rows"] += 1
            if not bucket["traffic"]:
                bucket["traffic"] += _safe_int(row.get("traffic"))
            if not bucket["pages"]:
                bucket["pages"] += _safe_int(row.get("pages"))

        flow = lg.get("link_flow") or {}
        flow_nodes = {row.get("url"): row for row in (flow.get("nodes") or []) if row.get("url")}
        for url, row in flow_nodes.items():
            page_cluster_lookup.setdefault(url, page_cluster(row))
        for edge in flow.get("edges") or []:
            source_url = edge.get("source")
            target_url = edge.get("target")
            source_cluster = page_cluster_lookup.get(source_url) or page_cluster(flow_nodes.get(source_url) or {"url": source_url})
            target_cluster = page_cluster_lookup.get(target_url) or page_cluster(flow_nodes.get(target_url) or {"url": target_url})
            weight = max(1, _safe_int(edge.get("weight"), 1))
            target_bucket = cluster_bucket[target_cluster]
            source_bucket = cluster_bucket[source_cluster]
            if target_cluster != source_cluster:
                target_bucket["cross_inbound_links"] += weight
                target_bucket["source_clusters"][source_cluster] += weight
                source_bucket["cross_outbound_links"] += weight
                source_bucket["target_clusters"][target_cluster] += weight
            cluster_edges.append({
                "domain": proj.domain,
                "source_cluster": source_cluster,
                "target_cluster": target_cluster,
                "links": weight,
                "target_traffic": _safe_int(edge.get("target_traffic")),
                "source_pagerank": _safe_float(edge.get("source_pagerank")),
                "contextual_link_impact": _safe_float(edge.get("contextual_link_impact")),
            })

        if not flow.get("edges"):
            for edge in ((lg.get("hub_bottlenecks") or {}).get("cluster_edges") or []):
                source_cluster = cluster_name(edge.get("source_cluster"))
                target_cluster = cluster_name(edge.get("target_cluster"))
                links = max(1, _safe_int(edge.get("bridge_pages"), 1))
                cluster_bucket[source_cluster]["target_clusters"][target_cluster] += links
                cluster_bucket[source_cluster]["cross_outbound_links"] += links
                cluster_bucket[target_cluster]["source_clusters"][source_cluster] += links
                cluster_bucket[target_cluster]["cross_inbound_links"] += links
                cluster_edges.append({
                    "domain": proj.domain,
                    "source_cluster": source_cluster,
                    "target_cluster": target_cluster,
                    "links": links,
                    "target_traffic": 0,
                    "source_pagerank": 0.0,
                    "contextual_link_impact": 0.0,
                })

        for cluster, bucket in cluster_bucket.items():
            support_score = bucket["avg_support_score"] / max(1, bucket["support_rows"]) if bucket["support_rows"] else 0.0
            link_score = min(30.0, math.log1p(bucket["inbound_links"]) * 8.0)
            cross_score = min(20.0, len(bucket["source_clusters"]) * 5.0 + math.log1p(bucket["cross_inbound_links"]) * 3.0)
            demand_penalty = min(28.0, math.log1p(bucket["opportunity_traffic"]) * 3.0 + bucket["opportunities"] * 3.0)
            orphan_penalty = min(18.0, bucket["orphan_pages"] * 5.0)
            connectivity_score = max(0.0, min(100.0, support_score * 0.42 + link_score + cross_score - demand_penalty - orphan_penalty + 22.0))
            cluster_domains[cluster][proj.domain] = {
                "domain": proj.domain,
                "cluster": cluster,
                "label": bucket["label"] or cluster,
                "pages": _safe_int(bucket["pages"]),
                "traffic": _safe_int(bucket["traffic"]),
                "keywords": _safe_int(bucket["keywords"]),
                "inbound_links": _safe_int(bucket["inbound_links"]),
                "outbound_links": _safe_int(bucket["outbound_links"]),
                "cross_inbound_links": _safe_int(bucket["cross_inbound_links"]),
                "cross_outbound_links": _safe_int(bucket["cross_outbound_links"]),
                "source_clusters": [key for key, _ in bucket["source_clusters"].most_common(6)],
                "target_clusters": [key for key, _ in bucket["target_clusters"].most_common(6)],
                "orphan_pages": _safe_int(bucket["orphan_pages"]),
                "dead_end_pages": _safe_int(bucket["dead_end_pages"]),
                "deep_pages": _safe_int(bucket["deep_pages"]),
                "underserved_pages": _safe_int(bucket["underserved_pages"]),
                "opportunities": _safe_int(bucket["opportunities"]),
                "opportunity_traffic": _safe_int(bucket["opportunity_traffic"]),
                "avg_support_score": round(support_score, 2),
                "connectivity_score": round(connectivity_score, 2),
                "pagerank": round(_safe_float(bucket["pagerank"]), 8),
                "weighted_pagerank": round(_safe_float(bucket["weighted_pagerank"]), 8),
            }

        for row in ((lg.get("link_addition_simulation") or {}).get("recommendations") or [])[:120]:
            add_recommendation({
                "domain": proj.domain,
                "type": "link_addition",
                "reason": "High expected-benefit internal link placement",
                "source_url": row.get("source_url") or "",
                "source_title": row.get("source_title") or row.get("source_url") or "",
                "target_url": row.get("target_url") or "",
                "target_title": row.get("target_title") or row.get("target_url") or "",
                "target_cluster": cluster_name(row.get("target_cluster") or row.get("cluster") or row.get("target_directory")),
                "suggested_anchor": row.get("suggested_anchor") or "",
                "paragraph_index": row.get("paragraph_index"),
                "paragraph_excerpt": row.get("paragraph_excerpt") or "",
                "priority_score": _safe_float(row.get("expected_benefit_score")),
                "confidence": 0.72 if row.get("priority") == "high" else 0.62,
            })

        for row in ((lg.get("high_demand_low_link") or {}).get("opportunities") or [])[:120]:
            candidate = (row.get("source_candidates") or [{}])[0]
            add_recommendation({
                "domain": proj.domain,
                "type": "high_demand_low_support",
                "reason": "High-demand page has weaker internal support than its search demand",
                "source_url": candidate.get("source_url") or "",
                "source_title": candidate.get("source_title") or candidate.get("source_url") or "",
                "target_url": row.get("url") or "",
                "target_title": row.get("title") or row.get("url") or "",
                "target_cluster": cluster_name(row.get("cluster") or row.get("section")),
                "suggested_anchor": candidate.get("suggested_anchor") or (row.get("suggested_anchors") or [""])[0],
                "paragraph_index": candidate.get("paragraph_index"),
                "paragraph_excerpt": candidate.get("paragraph_excerpt") or "",
                "traffic": _safe_int(row.get("traffic")),
                "priority_score": _safe_float(row.get("opportunity_score")) or (_safe_float(row.get("demand_support_gap")) + math.log1p(_safe_int(row.get("traffic"))) * 6.0),
                "confidence": 0.7,
            })

        for row in ((lg.get("internal_link_patterns") or {}).get("recommendations") or [])[:80]:
            target = (row.get("suggested_targets") or [{}])[0]
            add_recommendation({
                "domain": proj.domain,
                "type": "missing_link_pattern",
                "reason": row.get("recommended_action") or row.get("missing_pattern") or "Missing internal link pattern",
                "source_url": row.get("source_url") or "",
                "source_title": row.get("source_title") or row.get("source_url") or "",
                "target_url": target.get("url") or row.get("target_url") or "",
                "target_title": target.get("title") or row.get("target_title") or target.get("url") or "",
                "target_cluster": cluster_name(target.get("cluster") or row.get("target_cluster") or row.get("source_page_type")),
                "suggested_anchor": row.get("suggested_anchor") or "",
                "traffic": _safe_int(row.get("traffic")),
                "priority_score": _safe_float(row.get("lift_score_difference")) + _safe_float(row.get("confidence")) * 55.0,
                "confidence": _safe_float(row.get("confidence"), 0.55),
            })

    scorecards.sort(key=lambda r: _safe_float(r.get("architecture_score")), reverse=True)
    leader = scorecards[0] if scorecards else {}
    for rec in recommendations:
        rec["benchmark_domain"] = leader.get("domain") or ""
    recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_int(r.get("traffic"))), reverse=True)

    cluster_rows = []
    for cluster, values in cluster_domains.items():
        domain_rows = []
        for domain in domains:
            domain_rows.append(values.get(domain, {
                "domain": domain,
                "cluster": cluster,
                "label": cluster,
                "pages": 0,
                "traffic": 0,
                "keywords": 0,
                "inbound_links": 0,
                "outbound_links": 0,
                "cross_inbound_links": 0,
                "cross_outbound_links": 0,
                "source_clusters": [],
                "target_clusters": [],
                "orphan_pages": 0,
                "dead_end_pages": 0,
                "deep_pages": 0,
                "underserved_pages": 0,
                "opportunities": 0,
                "opportunity_traffic": 0,
                "avg_support_score": 0.0,
                "connectivity_score": 0.0,
                "pagerank": 0.0,
                "weighted_pagerank": 0.0,
            }))
        cluster_rows.append({
            "cluster": cluster,
            "label": next((row.get("label") for row in domain_rows if row.get("label") and row.get("label") != cluster), cluster),
            "total_traffic": sum(_safe_int(row.get("traffic")) for row in domain_rows),
            "total_opportunity_traffic": sum(_safe_int(row.get("opportunity_traffic")) for row in domain_rows),
            "domains": domain_rows,
        })
    cluster_rows.sort(key=lambda r: (_safe_int(r.get("total_opportunity_traffic")), _safe_int(r.get("total_traffic"))), reverse=True)
    cluster_edges.sort(key=lambda r: (_safe_int(r.get("links")), _safe_int(r.get("target_traffic"))), reverse=True)

    return {
        "summary": {
            "domains": len(domains),
            "clusters": len(cluster_rows),
            "recommendations": len(recommendations),
            "leader_domain": leader.get("domain") or "",
        },
        "scorecards": scorecards,
        "clusters": cluster_rows[:160],
        "cluster_edges": cluster_edges[:600],
        "recommendations": recommendations[:400],
    }


def _pattern_transplant_payload(projects: list[_Project]) -> dict:
    domains = [p.domain for p in projects]
    pattern_domains: dict[str, dict[str, dict]] = defaultdict(dict)
    pattern_meta: dict[str, dict] = {}
    recommendations: list[dict] = []

    template_recs: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    template_coverage: dict[str, dict[str, dict]] = defaultdict(dict)
    internal_recs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    internal_coverage: dict[str, dict[str, dict]] = defaultdict(dict)

    def pattern_token(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    def template_key(feature: object, label: object, page_type: object) -> str:
        token = pattern_token(feature) or pattern_token(label)
        if not token:
            return ""
        return f"template::{token}::{page_type or ''}"

    def template_tokens(*values: object) -> list[str]:
        tokens: list[str] = []
        for value in values:
            token = pattern_token(value)
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    for proj in projects:
        for rec in (proj.template_patterns or {}).get("recommendations") or []:
            page_type = rec.get("page_type") or ""
            for token in template_tokens(rec.get("feature_key"), rec.get("missing_pattern"), rec.get("label")):
                template_recs[proj.domain][(token, page_type)].append(rec)
        for pattern in (proj.template_patterns or {}).get("patterns") or []:
            key = template_key(pattern.get("feature_key"), pattern.get("label"), pattern.get("page_type"))
            if not key:
                continue
            template_coverage[key][proj.domain] = pattern
            pattern_meta.setdefault(key, {
                "pattern_type": "template",
                "label": pattern.get("label") or pattern.get("feature_key") or "Template pattern",
                "page_type": pattern.get("page_type") or "",
            })
        for rec in (((proj.linkgraph or {}).get("internal_link_patterns") or {}).get("recommendations") or []):
            key = rec.get("rule_key") or rec.get("pattern_rule_key") or ""
            if not key:
                # Older generated reports may not repeat the rule key on recs;
                # keep pattern_id recommendations available for same-domain UI,
                # but avoid cross-domain matching without the full rule.
                continue
            internal_recs[proj.domain][key].append(rec)
        for pattern in (((proj.linkgraph or {}).get("internal_link_patterns") or {}).get("patterns") or []):
            key = f"internal::{pattern.get('rule_key') or pattern.get('inferred_rule') or ''}"
            if not key.strip(":"):
                continue
            internal_coverage[key][proj.domain] = pattern
            pattern_meta.setdefault(key, {
                "pattern_type": "internal_link",
                "label": pattern.get("inferred_rule") or "Internal link pattern",
                "source_page_type": pattern.get("source_page_type") or "",
                "target_page_type": pattern.get("target_page_type") or "",
            })

    def add_coverage(pattern_key: str, domain: str, row: dict, pattern_type: str, label: str) -> None:
        pattern_domains[pattern_key][domain] = {
            "domain": domain,
            "pattern_key": pattern_key,
            "pattern_type": pattern_type,
            "label": label,
            "covered": True,
            "confidence": _safe_float(row.get("confidence")),
            "support_count": _safe_int(row.get("support_count") or row.get("sample_size")),
            "recommendations": 0,
        }

    for key, by_domain in template_coverage.items():
        meta = pattern_meta.get(key, {})
        for domain, row in by_domain.items():
            add_coverage(key, domain, row, "template", meta.get("label") or key)
    for key, by_domain in internal_coverage.items():
        meta = pattern_meta.get(key, {})
        for domain, row in by_domain.items():
            add_coverage(key, domain, row, "internal_link", meta.get("label") or key)

    for source in projects:
        source_template_patterns = (source.template_patterns or {}).get("patterns") or []
        for pattern in source_template_patterns:
            feature = pattern.get("feature_key") or pattern.get("label") or ""
            label = pattern.get("label") or feature
            tokens = template_tokens(pattern.get("feature_key"), label)
            if not tokens:
                continue
            page_type = pattern.get("page_type") or ""
            confidence = _safe_float(pattern.get("confidence"))
            sample_size = _safe_int(pattern.get("sample_size"))
            if confidence < 0.55 or sample_size < 2:
                continue
            pattern_key = f"template::{tokens[0]}::{page_type}"
            for target in projects:
                if target.domain == source.domain:
                    continue
                target_rows = []
                seen_rows: set[tuple[str, str]] = set()
                for token in tokens:
                    candidate_rows = list(template_recs[target.domain].get((token, page_type), []))
                    if page_type:
                        candidate_rows.extend(template_recs[target.domain].get((token, ""), []))
                    for candidate in candidate_rows:
                        row_key = (str(candidate.get("url") or ""), str(candidate.get("missing_pattern") or candidate.get("feature_key") or ""))
                        if row_key in seen_rows:
                            continue
                        seen_rows.add(row_key)
                        target_rows.append(candidate)
                for rec in target_rows[:8]:
                    rec_page_type = rec.get("page_type") or page_type
                    if page_type and rec_page_type and rec_page_type != page_type:
                        continue
                    target_traffic = _safe_int(rec.get("traffic"))
                    lift = _safe_float(pattern.get("observed_lift"))
                    domain_gap = 0 if target.domain in template_coverage.get(pattern_key, {}) else 1
                    priority = min(100.0, confidence * 38.0 + min(24.0, math.log1p(target_traffic) * 4.0) + min(24.0, max(0.0, lift) * 12.0) + domain_gap * 14.0)
                    recommendations.append({
                        "pattern_type": "template",
                        "pattern_key": pattern_key,
                        "source_domain": source.domain,
                        "target_domain": target.domain,
                        "source_pattern": label,
                        "source_evidence": {
                            "confidence": confidence,
                            "observed_lift": lift,
                            "sample_size": sample_size,
                            "sample_urls": (pattern.get("sample_urls") or [])[:5],
                        },
                        "target_url": rec.get("url") or "",
                        "target_title": rec.get("title") or rec.get("url") or "",
                        "target_page_type": rec_page_type,
                        "target_traffic": target_traffic,
                        "target_keywords": _safe_int(rec.get("keywords")),
                        "concrete_change": rec.get("recommendation") or pattern.get("recommendation") or f"Add the {label} pattern to this page.",
                        "suggested_heading": rec.get("missing_pattern") or label,
                        "suggested_anchor": "",
                        "expected_benefit_components": {
                            "source_confidence": round(confidence, 3),
                            "source_lift": round(lift, 3),
                            "target_traffic": target_traffic,
                            "domain_pattern_gap": domain_gap,
                        },
                        "priority_score": round(priority, 2),
                        "confidence": round(confidence, 3),
                    })

        source_internal_patterns = (((source.linkgraph or {}).get("internal_link_patterns") or {}).get("patterns") or [])
        for pattern in source_internal_patterns:
            rule_key = pattern.get("rule_key") or ""
            if not rule_key:
                continue
            confidence = _safe_float(pattern.get("confidence"))
            support = _safe_int(pattern.get("support_count"))
            if confidence < 0.55 or support < 2:
                continue
            pattern_key = f"internal::{rule_key}"
            for target in projects:
                if target.domain == source.domain:
                    continue
                for rec in internal_recs[target.domain].get(rule_key, [])[:8]:
                    if pattern.get("source_page_type") and rec.get("source_page_type") and pattern.get("source_page_type") != rec.get("source_page_type"):
                        continue
                    target_traffic = _safe_int(rec.get("traffic"))
                    domain_gap = 0 if target.domain in internal_coverage.get(pattern_key, {}) else 1
                    lift = _safe_float(pattern.get("lift_score_difference"))
                    priority = min(100.0, confidence * 40.0 + min(22.0, support * 3.0) + min(20.0, math.log1p(target_traffic) * 4.0) + min(10.0, max(0.0, lift) / 4.0) + domain_gap * 8.0)
                    first_target = (rec.get("suggested_targets") or [{}])[0]
                    recommendations.append({
                        "pattern_type": "internal_link",
                        "pattern_key": pattern_key,
                        "source_domain": source.domain,
                        "target_domain": target.domain,
                        "source_pattern": pattern.get("inferred_rule") or rule_key,
                        "source_evidence": {
                            "confidence": confidence,
                            "support_count": support,
                            "top_rate": _safe_float(pattern.get("top_rate")),
                            "weak_rate": _safe_float(pattern.get("weak_rate")),
                            "sample_links": (pattern.get("sample_links") or [])[:5],
                        },
                        "target_url": rec.get("source_url") or "",
                        "target_title": rec.get("source_title") or rec.get("source_url") or "",
                        "target_page_type": rec.get("source_page_type") or pattern.get("source_page_type") or "",
                        "target_traffic": target_traffic,
                        "target_keywords": _safe_int(rec.get("keywords")),
                        "concrete_change": rec.get("recommended_action") or "Add a contextual internal link that follows the source pattern.",
                        "suggested_heading": "",
                        "suggested_anchor": rec.get("suggested_anchor") or "",
                        "suggested_target_url": first_target.get("url") or "",
                        "suggested_target_title": first_target.get("title") or "",
                        "expected_benefit_components": {
                            "source_confidence": round(confidence, 3),
                            "support_count": support,
                            "target_traffic": target_traffic,
                            "domain_pattern_gap": domain_gap,
                        },
                        "priority_score": round(priority, 2),
                        "confidence": round(confidence, 3),
                    })

    rec_counts = Counter(r["pattern_key"] for r in recommendations)
    rec_counts_by_domain = Counter((r["pattern_key"], r.get("target_domain") or "") for r in recommendations)
    coverage_rows = []
    for pattern_key, meta in pattern_meta.items():
        label = meta.get("label") or pattern_key
        ptype = meta.get("pattern_type") or ("internal_link" if pattern_key.startswith("internal::") else "template")
        coverage_rows.append({
            "pattern_key": pattern_key,
            "label": label,
            "pattern_type": ptype,
            "recommendations": rec_counts.get(pattern_key, 0),
            "domains": [
                {
                    **pattern_domains.get(pattern_key, {}).get(domain, {
                        "domain": domain,
                        "pattern_key": pattern_key,
                        "pattern_type": ptype,
                        "label": label,
                        "covered": False,
                        "confidence": 0.0,
                        "support_count": 0,
                    }),
                    "recommendations": rec_counts_by_domain.get((pattern_key, domain), 0),
                }
                for domain in domains
            ],
        })
    coverage_rows.sort(key=lambda r: (sum(1 for d in r["domains"] if d.get("covered")), rec_counts.get(r["pattern_key"], 0)), reverse=True)
    recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_float(r.get("confidence"))), reverse=True)
    covered_by_domain = {
        domain: sum(1 for row in coverage_rows if any(d.get("domain") == domain and d.get("covered") for d in row["domains"]))
        for domain in domains
    }
    total_patterns = len(coverage_rows)
    return {
        "summary": {
            "patterns": total_patterns,
            "recommendations": len(recommendations),
            "domains": len(domains),
        },
        "domains": [
            {
                "domain": domain,
                "covered_patterns": covered_by_domain.get(domain, 0),
                "coverage_rate": round(covered_by_domain.get(domain, 0) / max(1, total_patterns), 4),
                "recommendations": sum(1 for r in recommendations if r.get("target_domain") == domain),
            }
            for domain in domains
        ],
        "coverage": coverage_rows[:180],
        "recommendations": recommendations[:400],
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


def _performance_explainer_comparison(projects: list[_Project]) -> dict:
    domains = [p.domain for p in projects]
    summaries: list[dict] = []
    feature_groups: dict[str, dict[str, dict]] = defaultdict(dict)
    page_rows: list[dict] = []
    cluster_groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for proj in projects:
        payload = proj.performance_explainer or {}
        summary = payload.get("summary") or {}
        if summary:
            summaries.append({"domain": proj.domain, **summary})
        for feature in payload.get("features") or []:
            key = feature.get("feature") or feature.get("label") or ""
            if not key:
                continue
            feature_groups[key][proj.domain] = {
                "domain": proj.domain,
                "feature": key,
                "label": feature.get("label") or key,
                "group": feature.get("group") or "",
                "coefficient": _safe_float(feature.get("coefficient")),
                "direction": feature.get("direction") or "",
                "permutation_importance": _safe_float(feature.get("permutation_importance")),
                "abs_coefficient": _safe_float(feature.get("abs_coefficient")),
            }
        for page in (payload.get("pages") or [])[:180]:
            item = {"domain": proj.domain, **page}
            page_rows.append(item)
            section = page.get("section") or "unknown"
            for factor in (page.get("top_positive") or [])[:4] + (page.get("top_negative") or [])[:4]:
                feature_key = factor.get("feature") or ""
                if not feature_key:
                    continue
                bucket = cluster_groups[(section, feature_key)].setdefault(proj.domain, {
                    "domain": proj.domain,
                    "section": section,
                    "feature": feature_key,
                    "label": factor.get("label") or feature_key,
                    "group": factor.get("group") or "",
                    "importance_sum": 0.0,
                    "positive_sum": 0.0,
                    "negative_sum": 0.0,
                    "pages": 0,
                    "traffic": 0.0,
                })
                contribution = _safe_float(factor.get("contribution"))
                bucket["importance_sum"] += abs(contribution)
                if contribution > 0:
                    bucket["positive_sum"] += contribution
                elif contribution < 0:
                    bucket["negative_sum"] += contribution
                bucket["pages"] += 1
                bucket["traffic"] += _safe_float(page.get("traffic"))

    feature_matrix = []
    for feature_key, values in feature_groups.items():
        label = next((v.get("label") for v in values.values() if v.get("label")), feature_key)
        group = next((v.get("group") for v in values.values() if v.get("group")), "")
        feature_matrix.append({
            "feature": feature_key,
            "label": label,
            "group": group,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "feature": feature_key,
                    "label": label,
                    "group": group,
                    "coefficient": 0.0,
                    "direction": "neutral",
                    "permutation_importance": 0.0,
                    "abs_coefficient": 0.0,
                })
                for domain in domains
            ],
        })
    feature_matrix.sort(
        key=lambda row: (
            sum(_safe_float(d.get("permutation_importance")) for d in row["domains"]),
            sum(_safe_float(d.get("abs_coefficient")) for d in row["domains"]),
        ),
        reverse=True,
    )

    cluster_matrix = []
    for (section, feature_key), values in cluster_groups.items():
        label = next((v.get("label") for v in values.values() if v.get("label")), feature_key)
        cluster_matrix.append({
            "section": section,
            "feature": feature_key,
            "label": label,
            "domains": [
                values.get(domain, {
                    "domain": domain,
                    "section": section,
                    "feature": feature_key,
                    "label": label,
                    "importance_sum": 0.0,
                    "positive_sum": 0.0,
                    "negative_sum": 0.0,
                    "pages": 0,
                    "traffic": 0.0,
                })
                for domain in domains
            ],
        })
    cluster_matrix.sort(
        key=lambda row: sum(_safe_float(d.get("importance_sum")) for d in row["domains"]),
        reverse=True,
    )
    page_rows.sort(key=lambda r: (_safe_float(r.get("traffic")), abs(_safe_float(r.get("residual_log")))), reverse=True)

    return {
        "summary": {
            "status": "ok" if feature_matrix or page_rows else "no_models",
            "model": "comparison_performance_explainer_v1",
            "domains": len(domains),
            "features": len(feature_matrix),
            "pages": len(page_rows),
            "clusters": len(cluster_matrix),
        },
        "domains": summaries,
        "features": feature_matrix[:120],
        "clusters": cluster_matrix[:180],
        "pages": page_rows[:500],
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
    keyword_cluster_gaps = _keyword_cluster_gap_payload(projects)
    strongest_clusters = _strongest_cluster_payload(projects, scatter_rows, ahrefs_semantic_rows)
    pattern_transplants = _pattern_transplant_payload(projects)
    winning_patterns = _winning_pattern_transfer_payload(
        projects,
        strongest_clusters,
        keyword_cluster_gaps,
        pattern_transplants,
    )
    keyword_content_matrix = _keyword_content_matrix_payload(
        projects,
        strongest_clusters,
        keyword_cluster_gaps,
        winning_patterns,
    )
    paragraph_archetypes = _paragraph_archetype_comparison(projects, strongest_clusters)
    structured_data_opportunities = _structured_data_opportunity_comparison(projects)
    template_patterns = _template_patterns_comparison(projects)
    internal_link_patterns = _internal_link_patterns_comparison(projects)
    action_board = _action_board_payload(projects)
    seo_playbooks = _seo_playbooks_payload(
        projects,
        winning_patterns=winning_patterns,
        paragraph_archetypes=paragraph_archetypes,
        template_patterns=template_patterns,
        structured_data_opportunities=structured_data_opportunities,
        internal_link_patterns=internal_link_patterns,
        action_board=action_board,
    )
    best_page_explainers = build_best_page_comparison([
        {"domain": p.domain, "best_pages": p.best_pages or {}}
        for p in projects
    ])

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
        "structured_data_opportunities": structured_data_opportunities,
        "trust_signals": _trust_signal_comparison(projects),
        "conversion_balance": _conversion_balance_comparison(projects),
        "answer_blocks": _answer_blocks_comparison(projects),
        "freshness_impact": _freshness_impact_comparison(projects),
        "cannibalization": _cannibalization_comparison(projects),
        "duplicate_fragments": _duplicate_fragments_comparison(projects),
        "template_patterns": template_patterns,
        "keyword_gaps": _keyword_gap_payload(projects),
        "keyword_cluster_gaps": keyword_cluster_gaps,
        "strongest_clusters": strongest_clusters,
        "winning_patterns": winning_patterns,
        "keyword_content_matrix": keyword_content_matrix,
        "paragraph_archetypes": paragraph_archetypes,
        "best_page_explainers": best_page_explainers,
        "action_board": action_board,
        "seo_playbooks": seo_playbooks,
        "serp_features": _serp_feature_payload(projects),
        "content_efficiency": _efficiency_payload(projects),
        "authority_demand": _authority_demand_payload(projects),
        "traffic_weighted_pagerank": _traffic_weighted_pagerank_comparison(projects),
        "link_removal_simulation": _link_removal_comparison(projects),
        "link_addition_simulation": _link_addition_comparison(projects),
        "anchor_relevance": _anchor_relevance_comparison(projects),
        "contextual_link_impact": _contextual_link_comparison(projects),
        "high_demand_low_link": _high_demand_low_link_comparison(projects),
        "internal_link_patterns": internal_link_patterns,
        "internal_link_architecture": _internal_link_architecture_comparison(projects),
        "pattern_transplants": pattern_transplants,
        "hub_bottlenecks": _hub_bottleneck_comparison(projects),
        "traffic_readiness": _readiness_payload(projects),
        "performance_explainer": _performance_explainer_comparison(projects),
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
