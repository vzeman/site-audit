"""Compare a competitor URL against our best-matching page for a query.

Answers the question: "Why is competitor X ranking for this query and we
aren't?" The diff is at two levels:

1. **Structural**: do they have FAQ/Article schema, more question-form
   headings, more statistics, more outbound citations? AI answer
   engines reward exactly these signals.
2. **Topical**: their paragraphs cluster into N sub-topics. For each
   sub-topic, do we have a paragraph above the similarity floor on
   our best-matching page? If not, that sub-topic is a content gap
   we should fill.

Input: a TSV file with ``query<TAB>competitor_url`` per line, optionally
prefixed by ``# section`` headers that group queries.

We re-use the project's ``HttpCache`` so the same competitor URL isn't
fetched twice across runs.
"""

from __future__ import annotations

import logging
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, NamedTuple, Optional
from urllib.parse import urlparse

import numpy as np
import requests

from .answerability import score_page
from .crawler import Crawler, CrawlConfig
from .extractor import extract
from .dataforseo import dataforseo_credentials

LOG = logging.getLogger(__name__)
DATAFORSEO_SERP_BASE_URL = "https://api.dataforseo.com/v3/serp"
_LOCALE_PREFIXES = {
    "ar", "cs", "da", "de", "es", "fi", "fr", "hu", "id", "it", "ja", "ko",
    "nl", "no", "pl", "pt", "ro", "sk", "sv", "th", "tr", "uk", "vi", "zh",
}
_LOW_INTENT_PATHS = {"author", "tag", "category", "glossary", "woordenlijst", "glosario", "glossario"}
_PRODUCT_INTENT_PATHS = {"services", "ai-tools", "ai-flow-templates", "integrations", "blog", "faq"}
_COMMERCIAL_WORDS = re.compile(
    r"\b(best|top|software|platform|tool|tools|app|apps|automation|agent|agents|chatbot|"
    r"workflow|business|compare|comparison|alternative|pricing|solution|service|services)\b",
    re.I,
)
_NOISE_WORDS = re.compile(
    r"\b(free\s+online|definition|meaning|wiki|lyrics|game|movie|jobs?|salary|course|"
    r"template|generator|checker|translator|image|photo)\b",
    re.I,
)


@dataclass
class CompetitorComparison:
    query: str
    competitor_url: str
    competitor_title: str
    our_best_url: str
    our_best_title: str
    our_best_similarity: float
    answerability_ours: float
    answerability_theirs: float
    structural_gaps: list[dict]              # [{signal, ours, theirs, advice}]
    missing_topics: list[dict]               # [{label, sample_competitor_paragraph}]
    paragraph_count_ours: int
    paragraph_count_theirs: int
    error: Optional[str] = None


class CompetitiveTarget(NamedTuple):
    query: str
    competitor_url: str
    cluster: str = ""
    rank: int = 0


@dataclass
class CompetitorPage:
    target: CompetitiveTarget
    title: str
    paragraphs: list[str]
    paragraph_embeddings: np.ndarray
    structural_gaps: list[dict]
    answerability: float
    paragraph_count: int
    error: Optional[str] = None
    h1: str = ""
    headers_rich: list[dict] = field(default_factory=list)
    content_sequence: list[dict] = field(default_factory=list)


@dataclass
class CompetitiveAutoConfig:
    enabled: bool = False
    max_clusters: int = 3
    keywords_per_cluster: int = 1
    results_per_keyword: int = 5
    min_position: int = 2
    max_position: int = 20
    min_relevance: float = 0.35
    product_seeds: list[str] | None = None
    allow_nonlatin: bool = False
    refresh_serp: bool = False
    location_code: Optional[int] = None
    location_name: Optional[str] = None
    language_code: Optional[str] = None
    language_name: Optional[str] = None


def _split_competitive_line(line: str) -> list[str]:
    if "\t" in line:
        return [p.strip() for p in line.split("\t") if p.strip()]
    if " | " in line:
        return [p.strip() for p in line.split(" | ") if p.strip()]
    if "  " in line:
        return [p.strip() for p in line.split("  ") if p.strip()]
    return []


def _parse_target(parts: list[str], current_cluster: str = "") -> CompetitiveTarget | None:
    if len(parts) < 2:
        return None
    url_i = next((i for i, p in enumerate(parts) if p.startswith(("http://", "https://"))), -1)
    if url_i < 0:
        return None
    url = parts[url_i]
    rank = 0
    cluster = current_cluster
    query = ""
    before = parts[:url_i]
    if len(before) == 1:
        query = before[0]
    elif len(before) >= 2:
        cluster = before[0] or cluster
        query = before[1]
        for value in before[2:]:
            try:
                rank = int(value)
                break
            except ValueError:
                continue
    for value in parts[url_i + 1:]:
        if not rank:
            try:
                rank = int(value)
                continue
            except ValueError:
                pass
        if not cluster:
            cluster = value
    if not query:
        return None
    return CompetitiveTarget(query=query, competitor_url=url, cluster=cluster, rank=rank)


def _safe_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _target_domain(domain: str) -> str:
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = (parsed.netloc or parsed.path).split("/")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _is_own_url(url: str, domain: str) -> bool:
    host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
    target = _target_domain(domain)
    return host == target or host.endswith(f".{target}")


def _ascii_ratio(text: str) -> float:
    letters = [ch for ch in text or "" if ch.isalpha()]
    if not letters:
        return 1.0
    ascii_letters = [ch for ch in letters if ord(ch) < 128]
    return len(ascii_letters) / max(len(letters), 1)


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_competitive_pairs(path: Path) -> list[tuple[str, str]]:
    return [(t.query, t.competitor_url) for t in load_competitive_targets(path)]


def load_competitive_targets(path: Path) -> list[CompetitiveTarget]:
    targets: list[CompetitiveTarget] = []
    current_cluster = ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_cluster = line.lstrip("#").strip()
            continue
        target = _parse_target(_split_competitive_line(line), current_cluster=current_cluster)
        if target is not None:
            targets.append(target)
    return targets


def _fetch_one(url: str, http_cache, user_agent: str) -> Optional[str]:
    """Fetch a single URL via the same retry+TLS-impersonation stack."""
    cfg = CrawlConfig(
        domain=url,                # crawler will use this as the home origin
        max_pages=1,
        max_workers=1,
        respect_robots=True,
        use_cache=True,
        user_agent=user_agent,
    )
    crawler = Crawler(cfg, http_cache)
    crawler._warm_session()
    res = crawler._fetch(url)
    if res is None or "html" not in res.content_type:
        return None
    return res.body


def _page_multiplier(page) -> float:
    url = getattr(page, "url", "") or ""
    parsed = urlparse(url)
    parts = [p for p in (parsed.path or "").split("/") if p]
    multiplier = 1.0
    if parts and parts[0].lower() in _LOCALE_PREFIXES:
        multiplier *= 0.72
        parts = parts[1:]
    first = parts[0].lower() if parts else ""
    if first in _LOW_INTENT_PATHS:
        multiplier *= 0.45
    if first in _PRODUCT_INTENT_PATHS or not first:
        multiplier *= 1.06
    language = (getattr(page, "language", "") or "").lower()
    if language and language not in {"en", "eng"}:
        multiplier *= 0.82
    title = (getattr(page, "title", "") or "").lower()
    if first == "author" or title.startswith(("author", "autor", "autor:")):
        multiplier *= 0.5
    return multiplier


def _url_intent_multiplier(url: str) -> float:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    parts = [p for p in (parsed.path or "").split("/") if p]
    multiplier = 1.0
    if parts and parts[0].lower() in _LOCALE_PREFIXES:
        multiplier *= 0.72
        parts = parts[1:]
    first = parts[0].lower() if parts else ""
    if first in _LOW_INTENT_PATHS:
        multiplier *= 0.45
    if first in _PRODUCT_INTENT_PATHS or not first:
        multiplier *= 1.06
    return multiplier


def _best_our_page(query: str, pages, page_embeddings: np.ndarray, embedder) -> tuple[int, float]:
    q_emb = embedder.encode([query])[0]
    sims = np.clip(page_embeddings @ q_emb, -1.0, 1.0)
    adjusted = np.asarray([
        float(sim) * _page_multiplier(page)
        for sim, page in zip(sims, pages)
    ], dtype=np.float32)
    idx = int(np.argmax(adjusted))
    return idx, float(sims[idx])


def _page_by_url(pages) -> dict[str, object]:
    return {getattr(page, "url", ""): page for page in pages if getattr(page, "url", "")}


def _keyword_cluster(row: dict) -> str:
    return (
        str(row.get("cluster_label") or "").strip()
        or str(row.get("section") or "").strip()
        or str(row.get("matched_url") or row.get("url") or "").strip()
        or "unclustered"
    )


def _keyword_relevance(
    row: dict,
    *,
    page_lookup: dict[str, object],
    seed_embeddings: np.ndarray,
    embedder,
    config: CompetitiveAutoConfig,
) -> tuple[float, list[str]]:
    keyword = str(row.get("keyword") or "").strip()
    reasons: list[str] = []
    if not keyword:
        return 0.0, ["missing_keyword"]
    if not config.allow_nonlatin and _ascii_ratio(keyword) < 0.85:
        return 0.0, ["nonlatin_keyword"]
    position = _safe_int(row.get("position"))
    if position and (position < config.min_position or position > config.max_position):
        return 0.0, ["position_out_of_range"]
    url = row.get("matched_url") or row.get("url") or ""
    page = page_lookup.get(url)
    url_multiplier = _page_multiplier(page) if page is not None else _url_intent_multiplier(url)
    if url_multiplier < 0.8:
        return 0.0, ["low_intent_page"]
    if _NOISE_WORDS.search(keyword):
        reasons.append("noise_term")
    if _COMMERCIAL_WORDS.search(keyword):
        reasons.append("commercial_modifier")

    score = 0.0
    if "commercial_modifier" in reasons:
        score += 0.35
    intents = {str(x).lower() for x in (row.get("intents") or [])}
    if intents & {"commercial", "transactional"}:
        score += 0.25
        reasons.append("commercial_intent")
    if url:
        score += min(0.25, max(0.0, (url_multiplier - 0.8) * 0.6))
        reasons.append("eligible_target_page")
    traffic = max(_safe_float(row.get("traffic")), _safe_float(row.get("paid_cost")))
    volume = _safe_float(row.get("volume"))
    if traffic > 0 or volume > 0:
        score += 0.1
        reasons.append("has_demand")

    if len(seed_embeddings):
        q_emb = embedder.encode([keyword], batch_size=16, show_progress=False)[0]
        seed_sim = float(np.max(seed_embeddings @ q_emb))
        score = 0.65 * score + 0.35 * max(0.0, seed_sim)
        if seed_sim >= config.min_relevance:
            reasons.append("seed_match")
        else:
            reasons.append("weak_seed_match")

    if "noise_term" in reasons:
        score *= 0.45
    return score, reasons


def select_competitive_auto_keywords(
    search_payload: dict,
    pages,
    embedder,
    config: CompetitiveAutoConfig,
) -> dict:
    rows = list((search_payload or {}).get("organic_keywords") or [])
    if not rows:
        return {"status": "no_search_keywords", "selected_keywords": [], "clusters": [], "rejected": []}

    page_lookup = _page_by_url(pages)
    seeds = [s.strip() for s in (config.product_seeds or []) if s and s.strip()]
    seed_embeddings = (
        embedder.encode(seeds, batch_size=16, show_progress=False).astype(np.float32)
        if seeds else np.zeros((0, 0), dtype=np.float32)
    )
    buckets: dict[str, dict] = {}
    rejected: list[dict] = []
    for row in rows:
        relevance, reasons = _keyword_relevance(
            row,
            page_lookup=page_lookup,
            seed_embeddings=seed_embeddings,
            embedder=embedder,
            config=config,
        )
        if relevance < config.min_relevance:
            if len(rejected) < 80:
                rejected.append({
                    "keyword": row.get("keyword") or "",
                    "position": _safe_int(row.get("position")),
                    "traffic": _safe_int(row.get("traffic")),
                    "matched_url": row.get("matched_url") or row.get("url") or "",
                    "relevance": round(relevance, 4),
                    "reasons": reasons,
                })
            continue
        cluster = _keyword_cluster(row)
        bucket = buckets.setdefault(cluster, {
            "cluster": cluster,
            "traffic": 0,
            "volume": 0,
            "keywords": [],
            "matched_urls": {},
        })
        traffic = max(_safe_int(row.get("traffic")), int(round(_safe_float(row.get("paid_cost")))))
        volume = _safe_int(row.get("volume"))
        position = _safe_int(row.get("position"))
        opportunity = (traffic or max(1, volume // 100)) * max(1, min(12, position or 10))
        item = {
            "keyword": row.get("keyword") or "",
            "position": position,
            "traffic": traffic,
            "paid_cost": round(_safe_float(row.get("paid_cost")), 2),
            "volume": volume,
            "matched_url": row.get("matched_url") or row.get("url") or "",
            "cluster": cluster,
            "relevance": round(relevance, 4),
            "opportunity_score": round(opportunity * relevance, 2),
            "reasons": reasons,
        }
        bucket["keywords"].append(item)
        bucket["traffic"] += traffic
        bucket["volume"] += volume
        if item["matched_url"]:
            bucket["matched_urls"][item["matched_url"]] = bucket["matched_urls"].get(item["matched_url"], 0) + traffic

    clusters = []
    for bucket in buckets.values():
        bucket["keywords"].sort(key=lambda r: (r["opportunity_score"], r["traffic"], r["volume"]), reverse=True)
        top_keywords = bucket["keywords"][: max(1, config.keywords_per_cluster)]
        clusters.append({
            "cluster": bucket["cluster"],
            "traffic": bucket["traffic"],
            "volume": bucket["volume"],
            "keywords": top_keywords,
            "keyword_count": len(bucket["keywords"]),
            "matched_urls": sorted(
                [{"url": url, "traffic": traffic} for url, traffic in bucket["matched_urls"].items()],
                key=lambda r: r["traffic"],
                reverse=True,
            )[:5],
            "score": round(sum(float(k["opportunity_score"]) for k in top_keywords), 2),
        })
    clusters.sort(key=lambda r: (r["score"], r["traffic"], r["volume"]), reverse=True)
    clusters = clusters[: max(0, config.max_clusters)]
    selected = [kw for cluster in clusters for kw in cluster["keywords"]]
    return {
        "status": "ok" if selected else "no_relevant_keywords",
        "selected_keywords": selected,
        "clusters": clusters,
        "rejected": rejected,
        "summary": {
            "source_keywords": len(rows),
            "eligible_clusters": len(buckets),
            "selected_clusters": len(clusters),
            "selected_keywords": len(selected),
            "max_clusters": config.max_clusters,
            "keywords_per_cluster": config.keywords_per_cluster,
            "results_per_keyword": config.results_per_keyword,
            "min_relevance": config.min_relevance,
            "product_seeds": seeds,
        },
    }


def _serp_cache_path(cache_dir: Path, task: dict) -> Path:
    root = Path(cache_dir) / "dataforseo_serp"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"serp_{date.today().isoformat()}_{_cache_key(task)}.json"


def _load_serp_cache(cache_dir: Path, task: dict) -> dict | None:
    root = Path(cache_dir) / "dataforseo_serp"
    if not root.exists():
        return None
    key = _cache_key(task)
    candidates = sorted(root.glob(f"serp_*_{key}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _fetch_dataforseo_serp(keyword: str, cache_dir: Path, config: CompetitiveAutoConfig) -> dict:
    task: dict = {
        "keyword": keyword,
        "depth": max(10, min(100, config.results_per_keyword * 2)),
    }
    task["location_code"] = int(config.location_code or 2840)
    if config.language_code:
        task["language_code"] = config.language_code
    else:
        task["language_code"] = "en"
    if not config.refresh_serp:
        cached = _load_serp_cache(cache_dir, task)
        if cached is not None:
            cached.setdefault("meta", {})["cache_status"] = "hit"
            return cached
    login, password = dataforseo_credentials()
    if not login or not password:
        return {
            "meta": {
                "status": "missing_api_key",
                "cache_status": "miss",
                "message": "Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD to fetch live SERP URLs.",
            },
            "raw": {},
        }
    endpoint = f"{DATAFORSEO_SERP_BASE_URL}/google/organic/live/advanced"
    resp = requests.post(endpoint, json=[task], auth=(login, password), timeout=120)
    if resp.status_code >= 400:
        return {
            "meta": {"status": "error", "cache_status": "miss", "message": f"HTTP {resp.status_code}: {resp.text[:500]}"},
            "raw": {},
        }
    raw = resp.json()
    status_code = _safe_int(raw.get("status_code"))
    if status_code not in (20000, 20100):
        return {
            "meta": {"status": "error", "cache_status": "miss", "message": raw.get("status_message") or "DataForSEO SERP request failed"},
            "raw": raw,
        }
    for task_row in raw.get("tasks") or []:
        task_status = _safe_int(task_row.get("status_code"))
        if task_status not in (20000, 20100):
            return {
                "meta": {
                    "status": "error",
                    "cache_status": "miss",
                    "message": task_row.get("status_message") or "DataForSEO SERP task failed",
                    "task": task,
                },
                "raw": raw,
            }
    payload = {"meta": {"status": "ok", "cache_status": "miss", "fetched_at": date.today().isoformat(), "task": task}, "raw": raw}
    _serp_cache_path(cache_dir, task).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _serp_items(payload: dict) -> list[dict]:
    out: list[dict] = []
    raw = payload.get("raw") or {}
    for task in raw.get("tasks") or []:
        for result in task.get("result") or []:
            for item in result.get("items") or []:
                if isinstance(item, dict):
                    out.append(item)
    return out


def build_auto_competitive_targets(
    domain: str,
    search_payload: dict,
    pages,
    embedder,
    cache_dir: Path,
    config: CompetitiveAutoConfig,
) -> tuple[list[CompetitiveTarget], dict]:
    selection = select_competitive_auto_keywords(search_payload, pages, embedder, config)
    targets: list[CompetitiveTarget] = []
    serp_rows: list[dict] = []
    if selection.get("status") != "ok":
        return [], {"status": selection.get("status"), "selection": selection, "serp": []}

    for selected in selection.get("selected_keywords") or []:
        keyword = selected.get("keyword") or ""
        cluster = selected.get("cluster") or keyword
        serp_payload = _fetch_dataforseo_serp(keyword, cache_dir, config)
        meta = serp_payload.get("meta") or {}
        if meta.get("status") != "ok":
            serp_rows.append({"keyword": keyword, "cluster": cluster, "status": meta.get("status"), "message": meta.get("message", "")})
            continue
        seen: set[str] = set()
        keyword_targets = []
        for item in _serp_items(serp_payload):
            if item.get("type") != "organic":
                continue
            url = item.get("url") or ""
            if not url or _is_own_url(url, domain) or url in seen:
                continue
            seen.add(url)
            rank = _safe_int(item.get("rank_group") or item.get("rank_absolute"))
            target = CompetitiveTarget(query=keyword, competitor_url=url, cluster=cluster, rank=rank)
            keyword_targets.append(target)
            if len(keyword_targets) >= config.results_per_keyword:
                break
        targets.extend(keyword_targets)
        serp_rows.append({
            "keyword": keyword,
            "cluster": cluster,
            "status": "ok",
            "cache_status": meta.get("cache_status", ""),
            "targets": [
                {"url": target.competitor_url, "rank": target.rank}
                for target in keyword_targets
            ],
        })
    return targets, {
        "status": "ok" if targets else "no_serp_targets",
        "selection": selection,
        "serp": serp_rows,
        "summary": {
            "selected_keywords": len(selection.get("selected_keywords") or []),
            "serp_targets": len(targets),
            "clusters": len(selection.get("clusters") or []),
        },
    }


def _structural_diff(ours_ext, theirs_ext) -> list[dict]:
    diffs: list[dict] = []
    ours_score = score_page(ours_ext)
    theirs_score = score_page(theirs_ext)
    # FAQ / structured schema
    has_faq = lambda ext: any(t in {"FAQPage", "QAPage", "Question"} for t in (ext.schema_types or []))
    has_struct = lambda ext: any(t in {"HowTo", "Article", "NewsArticle", "BlogPosting", "Product", "Recipe"} for t in (ext.schema_types or []))
    if has_faq(theirs_ext) and not has_faq(ours_ext):
        diffs.append({
            "signal": "FAQ/QA schema",
            "ours": False, "theirs": True,
            "advice": "Add FAQPage JSON-LD with the page's Q-form headings.",
        })
    elif has_struct(theirs_ext) and not has_struct(ours_ext):
        diffs.append({
            "signal": "Article/HowTo/Product schema",
            "ours": False, "theirs": True,
            "advice": "Add structured-data JSON-LD describing the page type.",
        })

    # question-form heading count
    def q_count(headings):
        n = 0
        for h in headings or []:
            t = h.lower()
            if t.endswith("?") or t.split(" ", 1)[0] in {"how", "what", "why", "when", "where", "which", "who"}:
                n += 1
        return n
    q_us = q_count(ours_ext.headings)
    q_th = q_count(theirs_ext.headings)
    if q_th >= q_us + 2:
        diffs.append({
            "signal": "Question-form headings (H2/H3 ending '?' or starting How/What/Why)",
            "ours": q_us, "theirs": q_th,
            "advice": "Reframe section headings as questions. AI answer engines retrieve question-shaped chunks.",
        })

    # statistics
    if theirs_ext.stat_count >= ours_ext.stat_count + 3:
        diffs.append({
            "signal": "Statistics / numbers with units",
            "ours": ours_ext.stat_count, "theirs": theirs_ext.stat_count,
            "advice": "Add concrete statistics with units (%, mg, hours, …). LLMs preferentially cite atomic facts.",
        })

    # external citations
    if theirs_ext.external_link_count >= ours_ext.external_link_count + 3:
        diffs.append({
            "signal": "External / authoritative citations",
            "ours": ours_ext.external_link_count, "theirs": theirs_ext.external_link_count,
            "advice": "Cite authoritative external sources (academic, .gov, .edu) so the page reads as researched.",
        })

    # tables
    if theirs_ext.table_count > ours_ext.table_count:
        diffs.append({
            "signal": "Comparison / data tables",
            "ours": ours_ext.table_count, "theirs": theirs_ext.table_count,
            "advice": "Tables are over-represented in AI Overviews. Add a comparison or specs table.",
        })

    # word count
    if theirs_ext.word_count >= 1.5 * max(ours_ext.word_count, 1) and theirs_ext.word_count >= 800:
        diffs.append({
            "signal": "Page depth (word count)",
            "ours": ours_ext.word_count, "theirs": theirs_ext.word_count,
            "advice": "Competitor goes deeper. Expand with sub-topics — see 'missing topics' below.",
        })

    return diffs


def structural_diff(ours_ext, theirs_ext) -> list[dict]:
    """Public wrapper used by serp_gap to compute per-competitor structural gaps."""
    return _structural_diff(ours_ext, theirs_ext)


def _missing_topics(
    theirs_para_embs: np.ndarray,
    theirs_paragraphs: list[str],
    our_page_emb: np.ndarray,
    our_paragraphs_embs: np.ndarray,
    our_paragraphs: list[str],
    threshold: float = 0.70,
    n_clusters: int = 8,
    top_examples: int = 1,
) -> list[dict]:
    """Cluster the competitor's paragraphs, then find the ones we don't cover."""
    if len(theirs_para_embs) < 4:
        return []

    # decide cluster count: at most n_clusters, at most n/3, at least 2
    k = max(2, min(n_clusters, len(theirs_para_embs) // 3))
    try:
        import faiss  # type: ignore
        kmeans = faiss.Kmeans(
            d=theirs_para_embs.shape[1],
            k=k,
            niter=30,
            verbose=False,
            seed=42,
            min_points_per_centroid=1,
        )
        kmeans.train(theirs_para_embs.astype(np.float32))
        _, labels = kmeans.index.search(theirs_para_embs.astype(np.float32), 1)
        labels = labels.flatten().astype(int)
    except Exception:
        # if k-means is unavailable, give up (still emit empty list)
        return []

    # c-TF-IDF labels
    from .cluster_labels import _compute_ctfidf
    docs = [""] * k
    for i, c in enumerate(labels):
        docs[int(c)] = (docs[int(c)] + " " + (theirs_paragraphs[i] or "")).strip()
    try:
        ctfidf, words = _compute_ctfidf(docs, ngram_range=(1, 2), min_df=1)
    except Exception:
        ctfidf, words = np.zeros((k, 0), dtype=np.float32), []

    out: list[dict] = []
    for cid in range(k):
        idxs = [i for i, c in enumerate(labels) if c == cid]
        if not idxs:
            continue
        # cluster centroid
        sub = theirs_para_embs[idxs]
        m = sub.mean(axis=0)
        norm = np.linalg.norm(m)
        centroid = m / norm if norm > 0 else m
        # do we cover this cluster?
        covered = False
        if len(our_paragraphs_embs) > 0:
            best_sim = float(np.max(our_paragraphs_embs @ centroid))
        else:
            best_sim = float(our_page_emb @ centroid)
        covered = best_sim >= threshold

        # cluster label
        label = ""
        if words:
            scores = ctfidf[cid]
            top_idx = np.argsort(-scores)[:8]
            kw = []
            seen: set[str] = set()
            for j in top_idx:
                if scores[j] <= 0:
                    continue
                w = words[int(j)]
                if any(w in s or s in w for s in seen if abs(len(w) - len(s)) < 4 and w != s):
                    continue
                seen.add(w)
                kw.append(w)
                if len(kw) >= 4:
                    break
            label = ", ".join(kw)
        if not label:
            label = f"sub-topic {cid}"

        if covered:
            continue

        # representative paragraph
        sims = sub @ centroid
        best_local = int(np.argmax(sims))
        sample_excerpt = theirs_paragraphs[idxs[best_local]][:280]

        out.append({
            "label": label,
            "competitor_paragraph_count": len(idxs),
            "best_similarity_we_have": round(best_sim, 4),
            "sample_competitor_paragraph": sample_excerpt,
        })

    out.sort(key=lambda r: r["best_similarity_we_have"])
    return out


def _topic_label(docs: list[str], labels: np.ndarray, cid: int) -> str:
    try:
        from .cluster_labels import _compute_ctfidf
        k = int(labels.max()) + 1 if len(labels) else 0
        cluster_docs = [""] * k
        for i, c in enumerate(labels):
            cluster_docs[int(c)] = (cluster_docs[int(c)] + " " + (docs[i] or "")).strip()
        ctfidf, words = _compute_ctfidf(cluster_docs, ngram_range=(1, 2), min_df=1)
    except Exception:
        return f"topic {cid + 1}"
    if not words or cid >= ctfidf.shape[0]:
        return f"topic {cid + 1}"
    scores = ctfidf[cid]
    top_idx = np.argsort(-scores)[:10]
    terms: list[str] = []
    seen: set[str] = set()
    for j in top_idx:
        if scores[j] <= 0:
            continue
        term = words[int(j)]
        if any(term in old or old in term for old in seen if term != old):
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 4:
            break
    return ", ".join(terms) or f"topic {cid + 1}"


def _cluster_paragraphs(embeddings: np.ndarray, max_clusters: int = 12) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(embeddings) < 6:
        labels = np.full((len(embeddings),), -1, dtype=np.int32)
        next_label = 0
        for i, emb in enumerate(embeddings):
            if labels[i] >= 0:
                continue
            labels[i] = next_label
            sims = embeddings @ emb
            for j in range(i + 1, len(embeddings)):
                if labels[j] < 0 and sims[j] >= 0.78:
                    labels[j] = next_label
            next_label += 1
        return labels
    k = max(2, min(max_clusters, len(embeddings) // 3))
    try:
        import faiss  # type: ignore
        kmeans = faiss.Kmeans(
            d=embeddings.shape[1],
            k=k,
            niter=35,
            verbose=False,
            seed=42,
            min_points_per_centroid=1,
        )
        kmeans.train(embeddings.astype(np.float32))
        _, labels = kmeans.index.search(embeddings.astype(np.float32), 1)
        return labels.flatten().astype(np.int32)
    except Exception:
        # Deterministic fallback: group each paragraph with the earliest
        # paragraph above a high similarity floor. It is rough, but keeps the
        # report useful when faiss is unavailable.
        labels = np.full((len(embeddings),), -1, dtype=np.int32)
        next_label = 0
        for i, emb in enumerate(embeddings):
            if labels[i] >= 0:
                continue
            labels[i] = next_label
            sims = embeddings @ emb
            for j in range(i + 1, len(embeddings)):
                if labels[j] < 0 and sims[j] >= 0.78:
                    labels[j] = next_label
            next_label += 1
        return labels


def build_serp_paragraph_gap(
    *,
    query: str,
    cluster: str,
    our_url: str,
    our_title: str,
    our_paragraphs: list[str],
    our_paragraph_embeddings: np.ndarray,
    competitor_pages: list[CompetitorPage],
    missing_threshold: float = 0.62,
    covered_threshold: float = 0.78,
) -> dict:
    usable = [p for p in competitor_pages if not p.error and len(p.paragraph_embeddings) > 0]
    if not usable:
        return {
            "query": query,
            "cluster": cluster or query,
            "our_url": our_url,
            "our_title": our_title,
            "competitors": len(competitor_pages),
            "status": "no_competitor_paragraphs",
            "topics": [],
            "missing_topics": [],
            "weak_topics": [],
            "off_intent_paragraphs": [],
        }

    texts: list[str] = []
    owners: list[CompetitorPage] = []
    for page in usable:
        for text in page.paragraphs:
            texts.append(text)
            owners.append(page)
    embeddings = np.vstack([p.paragraph_embeddings for p in usable]).astype(np.float32)
    labels = _cluster_paragraphs(embeddings)
    competitor_count = len({p.target.competitor_url for p in usable})
    topics: list[dict] = []
    centroids: list[np.ndarray] = []

    for cid in sorted(set(int(x) for x in labels)):
        idxs = [i for i, label in enumerate(labels) if int(label) == cid]
        if not idxs:
            continue
        sub = embeddings[idxs]
        centroid = sub.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid.astype(np.float32))
        owner_urls = []
        owner_titles = {}
        best_rank = 999
        for i in idxs:
            page = owners[i]
            owner_urls.append(page.target.competitor_url)
            owner_titles[page.target.competitor_url] = page.title
            if page.target.rank:
                best_rank = min(best_rank, page.target.rank)
        unique_urls = sorted(set(owner_urls))
        prevalence = len(unique_urls) / max(competitor_count, 1)
        if len(our_paragraph_embeddings):
            sims = our_paragraph_embeddings @ centroid
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])
            best_para = our_paragraphs[best_i] if best_i < len(our_paragraphs) else ""
        else:
            best_i = None
            best_sim = 0.0
            best_para = ""
        if best_sim >= covered_threshold:
            coverage = "covered"
        elif best_sim >= missing_threshold:
            coverage = "partial"
        else:
            coverage = "missing"
        if coverage == "missing" and prevalence >= 0.8:
            priority = "critical"
        elif coverage in {"missing", "partial"} and prevalence >= 0.6:
            priority = "high"
        elif coverage == "missing":
            priority = "medium"
        else:
            priority = "covered"
        representative_scores = sub @ centroid
        representative_order = np.argsort(-representative_scores)[:3]
        examples = []
        seen_example_urls: set[str] = set()
        for local_i in representative_order:
            global_i = idxs[int(local_i)]
            page = owners[global_i]
            if page.target.competitor_url in seen_example_urls:
                continue
            seen_example_urls.add(page.target.competitor_url)
            examples.append({
                "url": page.target.competitor_url,
                "title": page.title,
                "rank": page.target.rank,
                "paragraph": texts[global_i][:360],
            })
        topics.append({
            "label": _topic_label(texts, labels, cid),
            "centroid": [round(float(x), 5) for x in centroid.tolist()],
            "coverage": coverage,
            "priority": priority,
            "competitor_paragraphs": len(idxs),
            "competitor_urls": unique_urls,
            "competitor_coverage": len(unique_urls),
            "competitor_prevalence": round(prevalence, 3),
            "best_competitor_rank": None if best_rank == 999 else best_rank,
            "our_best_similarity": round(best_sim, 4),
            "our_best_paragraph_index": best_i,
            "our_best_paragraph": best_para[:280],
            "examples": examples,
        })

    priority_order = {"critical": 0, "high": 1, "medium": 2, "covered": 3}
    topics.sort(key=lambda r: (
        priority_order.get(r["priority"], 9),
        -float(r.get("competitor_prevalence", 0.0)),
        int(r.get("best_competitor_rank") or 99),
        -int(r.get("competitor_paragraphs", 0)),
    ))

    off_intent: list[dict] = []
    review_candidates: list[dict] = []
    if len(our_paragraph_embeddings) and centroids:
        topic_matrix = np.vstack(centroids).astype(np.float32)
        for i, (text, emb) in enumerate(zip(our_paragraphs, our_paragraph_embeddings)):
            words = len((text or "").split())
            if words < 35:
                continue
            best_topic_sim = float(np.max(topic_matrix @ emb))
            review_candidates.append({
                "paragraph_index": i,
                "similarity_to_serp_topics": round(best_topic_sim, 4),
                "paragraph": text[:300],
                "review_reason": "lowest similarity to SERP topic centroids",
            })
            if best_topic_sim < 0.52:
                off_intent.append({
                    "paragraph_index": i,
                    "similarity_to_serp_topics": round(best_topic_sim, 4),
                    "paragraph": text[:300],
                    "review_reason": "below off-intent threshold",
                })
        off_intent.sort(key=lambda r: r["similarity_to_serp_topics"])
        review_candidates.sort(key=lambda r: r["similarity_to_serp_topics"])

    def _gap_num(value) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    structural_counts: dict[str, dict] = {}
    for page in usable:
        for gap in page.structural_gaps:
            signal = gap.get("signal") or ""
            if not signal:
                continue
            row = structural_counts.setdefault(signal, {
                "signal": signal,
                "competitors": 0,
                "advice": gap.get("advice") or "",
                "ours": gap.get("ours"),
                "max_theirs": gap.get("theirs"),
            })
            row["competitors"] += 1
            if _gap_num(gap.get("theirs")) > _gap_num(row.get("max_theirs")):
                row["max_theirs"] = gap.get("theirs")

    return {
        "query": query,
        "cluster": cluster or query,
        "our_url": our_url,
        "our_title": our_title,
        "competitors": competitor_count,
        "status": "ok",
        "topics": topics,
        "missing_topics": [t for t in topics if t["coverage"] == "missing"],
        "weak_topics": [t for t in topics if t["coverage"] == "partial"],
        "covered_topics": [t for t in topics if t["coverage"] == "covered"],
        "off_intent_paragraphs": off_intent[:10],
        "own_paragraphs_to_review": review_candidates[:10],
        "structural_patterns": sorted(
            structural_counts.values(),
            key=lambda r: int(r.get("competitors", 0)),
            reverse=True,
        ),
        "summary": {
            "topics": len(topics),
            "missing": sum(1 for t in topics if t["coverage"] == "missing"),
            "partial": sum(1 for t in topics if t["coverage"] == "partial"),
            "covered": sum(1 for t in topics if t["coverage"] == "covered"),
            "critical": sum(1 for t in topics if t["priority"] == "critical"),
            "high": sum(1 for t in topics if t["priority"] == "high"),
            "off_intent_paragraphs": len(off_intent),
        },
    }


def compare_one(
    query: str,
    competitor_url: str,
    pages,
    page_embeddings: np.ndarray,
    paragraph_records: list,
    extracted_pages: list,
    embedder,
    http_cache,
    user_agent: str,
) -> CompetitorComparison:
    # 1) find our best page for this query
    our_idx, our_best_sim = _best_our_page(query, pages, page_embeddings, embedder)
    our_best_url = pages[our_idx].url
    our_best_title = pages[our_idx].title

    # 2) fetch competitor page
    body = _fetch_one(competitor_url, http_cache, user_agent)
    if body is None:
        return CompetitorComparison(
            query=query, competitor_url=competitor_url, competitor_title="",
            our_best_url=our_best_url, our_best_title=our_best_title,
            our_best_similarity=round(our_best_sim, 4),
            answerability_ours=0, answerability_theirs=0,
            structural_gaps=[], missing_topics=[],
            paragraph_count_ours=0, paragraph_count_theirs=0,
            error="could not fetch competitor URL",
        )
    theirs_ext = extract(competitor_url, body, max_chars=8000)
    if theirs_ext is None:
        return CompetitorComparison(
            query=query, competitor_url=competitor_url, competitor_title="",
            our_best_url=our_best_url, our_best_title=our_best_title,
            our_best_similarity=round(our_best_sim, 4),
            answerability_ours=0, answerability_theirs=0,
            structural_gaps=[], missing_topics=[],
            paragraph_count_ours=0, paragraph_count_theirs=0,
            error="competitor page had no usable content",
        )

    # 3) structural diff
    ours_ext = extracted_pages[our_idx]
    structural = _structural_diff(ours_ext, theirs_ext)

    ours_ans = score_page(ours_ext).score
    theirs_ans = score_page(theirs_ext).score

    # 4) embed competitor paragraphs in OUR vector space
    theirs_paragraphs = theirs_ext.paragraphs or []
    if theirs_paragraphs:
        theirs_para_embs = embedder.encode(theirs_paragraphs, batch_size=64).astype(np.float32)
    else:
        theirs_para_embs = np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)

    # 5) gather our page's paragraphs from the existing records
    our_para_embs_list = [r[3] for r in paragraph_records if r[0] == our_idx]
    our_paras = [r[2] for r in paragraph_records if r[0] == our_idx]
    our_paras_embs = (
        np.stack(our_para_embs_list).astype(np.float32) if our_para_embs_list
        else np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
    )

    missing = _missing_topics(
        theirs_para_embs, theirs_paragraphs,
        page_embeddings[our_idx], our_paras_embs, our_paras,
    )

    return CompetitorComparison(
        query=query,
        competitor_url=competitor_url,
        competitor_title=theirs_ext.title or competitor_url,
        our_best_url=our_best_url,
        our_best_title=our_best_title,
        our_best_similarity=round(our_best_sim, 4),
        answerability_ours=round(ours_ans, 2),
        answerability_theirs=round(theirs_ans, 2),
        structural_gaps=structural,
        missing_topics=missing,
        paragraph_count_ours=len(our_paras),
        paragraph_count_theirs=len(theirs_paragraphs),
    )


def compare_serp_targets(
    targets: list[CompetitiveTarget],
    pages,
    page_embeddings: np.ndarray,
    paragraph_records: list,
    extracted_pages: list,
    embedder,
    http_cache,
    user_agent: str,
) -> dict:
    by_query: dict[tuple[str, str], list[CompetitorPage]] = {}
    legacy_rows: list[CompetitorComparison] = []
    best_page_cache: dict[str, tuple[int, float]] = {}

    for target in targets:
        if target.query not in best_page_cache:
            best_page_cache[target.query] = _best_our_page(target.query, pages, page_embeddings, embedder)
        our_idx, our_sim = best_page_cache[target.query]
        ours_ext = extracted_pages[our_idx]
        our_paras = [r[2] for r in paragraph_records if r[0] == our_idx]

        body = _fetch_one(target.competitor_url, http_cache, user_agent)
        if body is None:
            error = "could not fetch competitor URL"
            page = CompetitorPage(target, "", [], np.zeros((0, page_embeddings.shape[1]), dtype=np.float32), [], 0.0, 0, error)
            by_query.setdefault((target.cluster or target.query, target.query), []).append(page)
            legacy_rows.append(CompetitorComparison(
                query=target.query,
                competitor_url=target.competitor_url,
                competitor_title="",
                our_best_url=pages[our_idx].url,
                our_best_title=pages[our_idx].title,
                our_best_similarity=round(our_sim, 4),
                answerability_ours=0,
                answerability_theirs=0,
                structural_gaps=[],
                missing_topics=[],
                paragraph_count_ours=len(our_paras),
                paragraph_count_theirs=0,
                error=error,
            ))
            continue

        theirs_ext = extract(target.competitor_url, body, max_chars=8000)
        if theirs_ext is None:
            error = "competitor page had no usable content"
            page = CompetitorPage(target, "", [], np.zeros((0, page_embeddings.shape[1]), dtype=np.float32), [], 0.0, 0, error)
            by_query.setdefault((target.cluster or target.query, target.query), []).append(page)
            legacy_rows.append(CompetitorComparison(
                query=target.query,
                competitor_url=target.competitor_url,
                competitor_title="",
                our_best_url=pages[our_idx].url,
                our_best_title=pages[our_idx].title,
                our_best_similarity=round(our_sim, 4),
                answerability_ours=0,
                answerability_theirs=0,
                structural_gaps=[],
                missing_topics=[],
                paragraph_count_ours=len(our_paras),
                paragraph_count_theirs=0,
                error=error,
            ))
            continue

        structural = _structural_diff(ours_ext, theirs_ext)
        theirs_paragraphs = theirs_ext.paragraphs or []
        if theirs_paragraphs:
            theirs_para_embs = embedder.encode(theirs_paragraphs, batch_size=64).astype(np.float32)
        else:
            theirs_para_embs = np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
        our_para_embs_list = [r[3] for r in paragraph_records if r[0] == our_idx]
        our_paras_embs = (
            np.stack(our_para_embs_list).astype(np.float32) if our_para_embs_list
            else np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
        )
        missing = _missing_topics(
            theirs_para_embs,
            theirs_paragraphs,
            page_embeddings[our_idx],
            our_paras_embs,
            our_paras,
        )
        answerability_ours = score_page(ours_ext).score
        answerability_theirs = score_page(theirs_ext).score
        legacy_rows.append(CompetitorComparison(
            query=target.query,
            competitor_url=target.competitor_url,
            competitor_title=theirs_ext.title or target.competitor_url,
            our_best_url=pages[our_idx].url,
            our_best_title=pages[our_idx].title,
            our_best_similarity=round(our_sim, 4),
            answerability_ours=round(answerability_ours, 2),
            answerability_theirs=round(answerability_theirs, 2),
            structural_gaps=structural,
            missing_topics=missing,
            paragraph_count_ours=len(our_paras),
            paragraph_count_theirs=len(theirs_paragraphs),
        ))
        by_query.setdefault((target.cluster or target.query, target.query), []).append(CompetitorPage(
            target=target,
            title=theirs_ext.title or target.competitor_url,
            paragraphs=theirs_paragraphs,
            paragraph_embeddings=theirs_para_embs,
            structural_gaps=structural,
            answerability=round(answerability_theirs, 2),
            paragraph_count=len(theirs_paragraphs),
            h1=theirs_ext.h1,
            headers_rich=theirs_ext.headers_rich,
        ))

    serp_clusters: list[dict] = []
    for (cluster, query), competitor_pages in by_query.items():
        our_idx, _ = best_page_cache[query]
        our_para_embs_list = [r[3] for r in paragraph_records if r[0] == our_idx]
        our_paras = [r[2] for r in paragraph_records if r[0] == our_idx]
        our_paras_embs = (
            np.stack(our_para_embs_list).astype(np.float32) if our_para_embs_list
            else np.zeros((0, page_embeddings.shape[1]), dtype=np.float32)
        )
        serp_clusters.append(build_serp_paragraph_gap(
            query=query,
            cluster=cluster,
            our_url=pages[our_idx].url,
            our_title=pages[our_idx].title,
            our_paragraphs=our_paras,
            our_paragraph_embeddings=our_paras_embs,
            competitor_pages=competitor_pages,
        ))

    serp_clusters.sort(key=lambda r: (
        int((r.get("summary") or {}).get("critical", 0)),
        int((r.get("summary") or {}).get("high", 0)),
        int((r.get("summary") or {}).get("missing", 0)),
    ), reverse=True)

    return {
        "summary": {
            "queries": len(by_query),
            "competitor_urls": len(targets),
            "serp_clusters": len(serp_clusters),
            "critical_missing_topics": sum(int((r.get("summary") or {}).get("critical", 0)) for r in serp_clusters),
            "high_priority_topics": sum(int((r.get("summary") or {}).get("high", 0)) for r in serp_clusters),
        },
        "comparisons": to_payload(legacy_rows),
        "serp_clusters": serp_clusters,
    }


def to_payload(rows: Iterable[CompetitorComparison]) -> list[dict]:
    return [
        {
            "query": r.query,
            "competitor_url": r.competitor_url,
            "competitor_title": r.competitor_title,
            "our_best_url": r.our_best_url,
            "our_best_title": r.our_best_title,
            "our_best_similarity": r.our_best_similarity,
            "answerability_ours": r.answerability_ours,
            "answerability_theirs": r.answerability_theirs,
            "structural_gaps": r.structural_gaps,
            "missing_topics": r.missing_topics,
            "paragraph_count_ours": r.paragraph_count_ours,
            "paragraph_count_theirs": r.paragraph_count_theirs,
            "error": r.error,
        }
        for r in rows
    ]
