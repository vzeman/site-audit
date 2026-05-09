"""Entity coverage scoring by page and topic cluster."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .analyzer import PageInfo
from .entities import extract_entities_from_text
from .paragraph_impact import _normalize_url, _to_int

_GENERIC_ENTITIES = {
    "Home", "Contact", "Privacy", "Terms", "Cookie", "Search", "Login",
    "Menu", "Read More", "Learn More", "Facebook", "Linkedin", "Twitter",
}
_INTEGRATION_RE = re.compile(r"\b(api|integration|zapier|slack|salesforce|hubspot|shopify|teams|whatsapp|zendesk|jira)\b", re.I)
_PROOF_RE = re.compile(r"\b(iso|gdpr|hipaa|soc\s*2|pci|g2|capterra|gartner|forrester|sla|compliance)\b", re.I)
_PRODUCT_RE = re.compile(r"\b(software|platform|suite|tool|solution|app|automation|crm|helpdesk|ticketing)\b", re.I)
_LOCATION_RE = re.compile(r"\b(united states|usa|uk|europe|germany|france|spain|italy|canada|australia|india|slovakia|czech|london|new york|berlin|paris)\b", re.I)


def _page_traffic(search_payload: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        out[_normalize_url(url)] = max(out.get(_normalize_url(url), 0), _to_int(row.get("traffic")))
    return out


def _cluster_labels(cluster_summaries) -> dict[int, str]:
    labels: dict[int, str] = {}
    for summary in cluster_summaries or []:
        cid = getattr(summary, "cluster_id", None)
        keywords = getattr(summary, "keywords", []) or []
        if cid is None:
            continue
        labels[int(cid)] = ", ".join(k.get("keyword", "") for k in keywords[:3] if k.get("keyword")) or f"cluster {cid}"
    return labels


def _source_texts(page: PageInfo, ext) -> dict[str, str]:
    headers = " ".join(str(h.get("text") or "") for h in (getattr(ext, "headers_rich", []) or []))
    return {
        "title": page.title or "",
        "description": page.description or getattr(ext, "description", "") or "",
        "heading": " ".join(part for part in (getattr(ext, "h1", ""), headers) if part),
        "body": getattr(ext, "body", "") or " ".join(getattr(ext, "paragraphs", []) or []),
    }


def _page_entities(page: PageInfo, ext) -> tuple[Counter[str], dict[str, set[str]], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    heading_targets: dict[str, list[str]] = defaultdict(list)
    for source, text in _source_texts(page, ext).items():
        for entity in extract_entities_from_text(text):
            if entity in _GENERIC_ENTITIES:
                continue
            counts[entity] += 1
            sources[entity].add(source)
            if source == "heading":
                heading_targets[entity].append(entity)
    return counts, sources, heading_targets


def _classify(entity: str, rank: int, core_cutoff: int) -> str:
    if _INTEGRATION_RE.search(entity):
        return "integration"
    if _PROOF_RE.search(entity):
        return "proof"
    if _LOCATION_RE.search(entity):
        return "location"
    if _PRODUCT_RE.search(entity):
        return "product"
    if re.search(r"\b(?:Inc|LLC|Ltd|GmbH|Corp|Company|Group)\b", entity):
        return "competitor"
    return "core" if rank <= core_cutoff else "supporting"


def build_entity_coverage(
    pages: list[PageInfo],
    extracted_pages: list,
    *,
    search_payload: dict | None = None,
    cluster_labels: Iterable[int] | None = None,
    cluster_summaries=None,
    expected_per_cluster: int = 30,
    top_n: int = 700,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "pages": [], "clusters": []}

    clusters = list(cluster_labels) if cluster_labels is not None else [0] * len(pages)
    label_lookup = _cluster_labels(cluster_summaries)
    traffic_by_url = _page_traffic(search_payload)
    page_entities: list[dict] = []
    cluster_entity_weights: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_entity_pages: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    entity_cluster_presence: dict[str, set[int]] = defaultdict(set)
    entity_heading_examples: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    for i, page in enumerate(pages):
        ext = extracted_pages[i] if i < len(extracted_pages) else None
        cid = int(clusters[i]) if i < len(clusters) else 0
        counts, sources, heading_targets = _page_entities(page, ext)
        traffic = traffic_by_url.get(_normalize_url(page.url), 0)
        page_entities.append({
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "cluster": cid,
            "traffic": traffic,
            "counts": counts,
            "sources": sources,
        })
        traffic_weight = 1.0 + math.log1p(max(traffic, 0))
        for entity, count in counts.items():
            source_bonus = 1.0
            if "heading" in sources.get(entity, set()):
                source_bonus += 0.45
            if "title" in sources.get(entity, set()):
                source_bonus += 0.25
            weight = (count ** 0.72) * traffic_weight * source_bonus
            cluster_entity_weights[cid][entity] += weight
            cluster_entity_pages[cid][entity].add(page.url)
            entity_cluster_presence[entity].add(cid)
            for heading in heading_targets.get(entity, []):
                entity_heading_examples[cid][entity][heading] += 1

    expected_by_cluster: dict[int, list[dict]] = {}
    for cid, counter in cluster_entity_weights.items():
        rows: list[dict] = []
        for rank, (entity, weight) in enumerate(counter.most_common(expected_per_cluster), 1):
            clusters_with_entity = len(entity_cluster_presence.get(entity, set()))
            uniqueness = 1.0 / max(1, clusters_with_entity)
            adjusted = float(weight) * (0.75 + uniqueness * 0.25)
            rows.append({
                "entity": entity,
                "weight": round(adjusted, 4),
                "raw_weight": round(float(weight), 4),
                "pages": len(cluster_entity_pages[cid][entity]),
                "class": _classify(entity, rank, core_cutoff=10),
                "suggested_section": (entity_heading_examples[cid][entity].most_common(1)[0][0] if entity_heading_examples[cid][entity] else ""),
            })
        expected_by_cluster[cid] = rows

    scored_pages: list[dict] = []
    for page_row in page_entities:
        cid = int(page_row["cluster"])
        expected = expected_by_cluster.get(cid, [])
        expected_weight = sum(float(e.get("weight", 0.0)) for e in expected) or 1.0
        present = set(page_row["counts"])
        covered_weight = sum(float(e.get("weight", 0.0)) for e in expected if e["entity"] in present)
        missing = [e for e in expected if e["entity"] not in present]
        core_missing = [e for e in missing if e.get("class") == "core"][:10]
        supporting_missing = [e for e in missing if e.get("class") != "core"][:12]
        expected_entities = {e["entity"] for e in expected}
        noisy = [
            {
                "entity": entity,
                "mentions": int(count),
                "class": _classify(entity, expected_per_cluster + 1, 10),
            }
            for entity, count in page_row["counts"].most_common(30)
            if entity not in expected_entities and count >= 2
        ][:10]
        coverage = covered_weight / expected_weight
        scored_pages.append({
            "url": page_row["url"],
            "title": page_row["title"],
            "section": page_row["section"],
            "cluster": cid,
            "cluster_label": label_lookup.get(cid, f"cluster {cid}"),
            "traffic": int(page_row["traffic"]),
            "coverage": round(coverage, 4),
            "coverage_pct": round(coverage * 100, 1),
            "expected_entities": len(expected),
            "present_expected_entities": sum(1 for e in expected if e["entity"] in present),
            "entity_count": len(present),
            "missing_core_entities": core_missing,
            "missing_supporting_entities": supporting_missing,
            "overrepresented_entities": noisy,
            "recommendations": [
                {
                    "entity": e["entity"],
                    "class": e.get("class", "core"),
                    "target": e.get("suggested_section") or label_lookup.get(cid, f"cluster {cid}"),
                    "action": f"Add {e['entity']} to the section covering {e.get('suggested_section') or label_lookup.get(cid, f'cluster {cid}')}.",
                }
                for e in (core_missing + supporting_missing)[:6]
            ],
        })

    scored_pages.sort(key=lambda r: (int(r.get("traffic", 0)), -float(r.get("coverage", 0.0))), reverse=True)
    clusters_payload = []
    for cid, expected in expected_by_cluster.items():
        cluster_pages = [p for p in scored_pages if int(p.get("cluster", -1)) == cid]
        clusters_payload.append({
            "cluster": cid,
            "label": label_lookup.get(cid, f"cluster {cid}"),
            "pages": len(cluster_pages),
            "avg_coverage": round(sum(float(p.get("coverage", 0.0)) for p in cluster_pages) / max(len(cluster_pages), 1), 4),
            "traffic": sum(int(p.get("traffic", 0)) for p in cluster_pages),
            "expected_entities": expected,
        })
    clusters_payload.sort(key=lambda r: (int(r.get("traffic", 0)), int(r.get("pages", 0))), reverse=True)

    summary = {
        "status": "ok" if scored_pages else "no_entities",
        "model": "entity_coverage_v1",
        "pages": len(scored_pages),
        "clusters": len(clusters_payload),
        "avg_coverage": round(sum(float(p.get("coverage", 0.0)) for p in scored_pages) / max(len(scored_pages), 1), 4),
        "low_coverage_pages": sum(1 for p in scored_pages if float(p.get("coverage", 0.0)) < 0.5),
        "pages_with_core_gaps": sum(1 for p in scored_pages if p.get("missing_core_entities")),
        "traffic_at_risk": sum(int(p.get("traffic", 0)) for p in scored_pages if float(p.get("coverage", 0.0)) < 0.5),
    }
    return {
        "summary": summary,
        "pages": scored_pages[:top_n],
        "clusters": clusters_payload[:200],
        "interpretation": {
            "coverage": "Weighted share of expected cluster entities present on the page.",
            "expected_entities": "Expected entities are mined from same-cluster pages, weighted by mentions, search traffic, title/heading proximity, and cluster uniqueness.",
            "recommendations": "Targets are suggested from headings where the entity already appears in strong same-cluster pages, falling back to the cluster label.",
        },
    }
