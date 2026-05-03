"""Entity and topical-authority analysis.

This module stays deliberately offline and heuristic: it extracts capitalized
entity-like phrases from metadata, headings, body copy, and lightweight schema
diagnostics, then rolls them into page/site coverage signals.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable

from .extractor import ExtractedPage


@dataclass
class EntityReport:
    summary: dict
    top_entities: list[dict]
    organizations: list[dict]
    per_page: list[dict]


_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\wÀ-ÖØ-Þà-öø-ÿ'’-]+|[A-Z]{2,})"
    r"(?:\s+(?:&|and|of|for|the|[A-Z][\wÀ-ÖØ-Þà-öø-ÿ'’-]+|[A-Z]{2,})){0,5}"
)
_ORG_SUFFIX_RE = re.compile(
    r"\b(?:Inc\.?|LLC|Ltd\.?|Limited|Corp\.?|Corporation|Company|Co\.?|Group|Agency|"
    r"Studio|Labs?|University|Institute|Foundation|Association|Partners|GmbH|AG|S\.r\.o\.|a\.s\.)\b",
    re.I,
)
_STOPWORDS = {
    "About", "All", "And", "Article", "Blog", "Browse", "Careers", "Contact",
    "Copyright", "Cookie", "Email", "Facebook", "Faq", "Home", "Instagram",
    "Learn", "Linkedin", "Login", "Menu", "News", "Next", "Our", "Page",
    "Previous", "Privacy", "Read", "Resources", "Search", "Services", "Share",
    "Terms", "The", "This", "Twitter", "Welcome", "You", "Your",
}


def _clean_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" -–—:;,.()[]{}\"'")
    value = re.sub(r"^(?:The|A|An)\s+", "", value)
    value = re.sub(r"\s+(?:and|of|for|the)$", "", value, flags=re.I)
    return value.strip()


def _is_entity(value: str) -> bool:
    if not value or value in _STOPWORDS:
        return False
    if len(value) < 3 or len(value) > 90:
        return False
    words = value.split()
    if len(words) == 1 and value not in value.upper() and value in _STOPWORDS:
        return False
    if value.lower() in {word.lower() for word in _STOPWORDS}:
        return False
    if re.fullmatch(r"\d+", value):
        return False
    return any(ch.isupper() for ch in value)


def extract_entities_from_text(text: str) -> list[str]:
    """Return normalized capitalized entity-like phrases from text."""
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text or ""):
        entity = _clean_entity(match.group(0))
        if not _is_entity(entity):
            continue
        key = entity.casefold()
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)
    return entities


def _schema_entity_names(page: ExtractedPage) -> list[str]:
    names: list[str] = []
    for block in page.schema_blocks or []:
        for name in block.get("names") or []:
            clean = _clean_entity(str(name))
            if _is_entity(clean):
                names.append(clean)
    return names


def _text_sources(page: ExtractedPage) -> dict[str, str]:
    heading_text = " ".join(
        str(h.get("text", "")) for h in (page.headers_rich or []) if h.get("text")
    ) or " ".join(page.headings or [])
    metadata = " ".join(
        part for part in (
            page.title, page.description, page.og_title, page.og_description,
            page.twitter_title, page.twitter_description,
        ) if part
    )
    return {
        "schema": " ".join(_schema_entity_names(page)),
        "metadata": metadata,
        "heading": " ".join(part for part in (page.h1, heading_text) if part),
        "body": page.body or "",
    }


def _is_organization(entity: str, schema_orgs: set[str]) -> bool:
    if entity in schema_orgs:
        return True
    return bool(_ORG_SUFFIX_RE.search(entity))


def analyze(pages: Iterable[ExtractedPage]) -> EntityReport:
    page_list = list(pages)
    entity_mentions: Counter[str] = Counter()
    entity_pages: dict[str, set[str]] = defaultdict(set)
    entity_sources: dict[str, set[str]] = defaultdict(set)
    organization_mentions: Counter[str] = Counter()
    organization_pages: dict[str, set[str]] = defaultdict(set)
    per_page: list[dict] = []

    for page in page_list:
        source_entities: dict[str, list[str]] = {
            source: extract_entities_from_text(text)
            for source, text in _text_sources(page).items()
            if text
        }
        schema_orgs = set(source_entities.get("schema", []))
        page_counter: Counter[str] = Counter()
        source_map: dict[str, set[str]] = defaultdict(set)
        for source, entities in source_entities.items():
            for entity in entities:
                page_counter[entity] += 1
                source_map[entity].add(source)

        organizations = sorted(
            entity for entity in page_counter
            if _is_organization(entity, schema_orgs)
        )
        for entity, mentions in page_counter.items():
            entity_mentions[entity] += mentions
            entity_pages[entity].add(page.url)
            entity_sources[entity].update(source_map[entity])
        for organization in organizations:
            organization_mentions[organization] += page_counter[organization]
            organization_pages[organization].add(page.url)

        distinct = len(page_counter)
        mentions = sum(page_counter.values())
        depth_score = min(1.0, (distinct / 8.0) * 0.6 + (mentions / 16.0) * 0.4)
        signals: list[str] = []
        if source_entities.get("schema"):
            signals.append("schema_entities")
        if source_entities.get("heading"):
            signals.append("heading_entities")
        if distinct >= 8:
            signals.append("entity_rich")
        if organizations:
            signals.append("organization_mentions")
        per_page.append({
            "url": page.url,
            "title": page.title,
            "entity_count": distinct,
            "entity_mentions": mentions,
            "organization_count": len(organizations),
            "top_entities": [
                {"entity": entity, "mentions": count}
                for entity, count in page_counter.most_common(8)
            ],
            "organizations": organizations[:8],
            "topical_depth_score": round(depth_score, 3),
            "signals": signals,
        })

    total_pages = len(page_list)
    entity_counts = [row["entity_count"] for row in per_page]
    mention_counts = [row["entity_mentions"] for row in per_page]
    pages_with_entities = sum(1 for count in entity_counts if count > 0)
    depth_pages = sum(1 for row in per_page if row["entity_count"] >= 5 and row["entity_mentions"] >= 8)
    thin_pages = sum(1 for row in per_page if row["entity_count"] < 3)
    reused_entities = sum(1 for entity, urls in entity_pages.items() if len(urls) >= 2)
    unique_entities = len(entity_mentions)
    unique_orgs = len(organization_mentions)
    entity_coverage = pages_with_entities / total_pages if total_pages else 0.0
    depth_share = depth_pages / total_pages if total_pages else 0.0
    reuse_share = reused_entities / unique_entities if unique_entities else 0.0
    org_coverage = (sum(1 for row in per_page if row["organization_count"] > 0) / total_pages) if total_pages else 0.0
    topical_authority_score = round(
        100.0 * (entity_coverage * 0.3 + depth_share * 0.3 + reuse_share * 0.25 + org_coverage * 0.15),
        1,
    )

    top_entities = [
        {
            "entity": entity,
            "mentions": mentions,
            "pages": len(entity_pages[entity]),
            "coverage": len(entity_pages[entity]) / total_pages if total_pages else 0.0,
            "sources": sorted(entity_sources[entity]),
        }
        for entity, mentions in entity_mentions.most_common(100)
    ]
    organizations = [
        {
            "organization": organization,
            "mentions": mentions,
            "pages": len(organization_pages[organization]),
            "coverage": len(organization_pages[organization]) / total_pages if total_pages else 0.0,
        }
        for organization, mentions in organization_mentions.most_common(50)
    ]
    per_page.sort(key=lambda row: (row["entity_count"], row["entity_mentions"], row["url"]))

    summary = {
        "total_pages": total_pages,
        "pages_with_entities": pages_with_entities,
        "entity_coverage": entity_coverage,
        "unique_entities": unique_entities,
        "total_entity_mentions": sum(entity_mentions.values()),
        "avg_entities_per_page": (sum(entity_counts) / total_pages) if total_pages else 0.0,
        "median_entities_per_page": float(median(entity_counts)) if entity_counts else 0.0,
        "avg_entity_mentions_per_page": (sum(mention_counts) / total_pages) if total_pages else 0.0,
        "entities_reused_across_pages": reused_entities,
        "entity_reuse_share": reuse_share,
        "organization_count": unique_orgs,
        "organization_coverage": org_coverage,
        "topical_depth_pages": depth_pages,
        "topical_depth_share": depth_share,
        "thin_entity_pages": thin_pages,
        "topical_authority_score": topical_authority_score,
    }
    return EntityReport(
        summary=summary,
        top_entities=top_entities,
        organizations=organizations,
        per_page=per_page,
    )


def to_payload(report: EntityReport) -> dict:
    return {
        "summary": report.summary,
        "top_entities": report.top_entities,
        "organizations": report.organizations,
        "per_page": report.per_page,
    }
