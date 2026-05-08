"""Optional DataForSEO Labs enrichment for organic search demand.

DataForSEO is used as a cache-first fallback when Ahrefs is unavailable, or
when the caller explicitly selects it. The normalized payload intentionally
matches the Ahrefs enrichment schema so report visualizations can reuse the
same charts regardless of provider.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import numpy as np
import requests

from .ahrefs import (
    AhrefsAnalysis,
    _aggregate_clusters,
    _aggregate_directories,
    _cluster_lookup,
    _load_json,
    _match_page,
    _page_cluster,
    _page_lookup,
    _params_without_date,
    _position_bucket,
    _semantic_map,
    _slug,
    _to_float,
    _to_int,
    _top_position_counts,
    _write_json,
    load_dotenv,
)
from .analyzer import PageInfo, section_for_url

LOG = logging.getLogger(__name__)

DATAFORSEO_BASE_URL = "https://api.dataforseo.com/v3/dataforseo_labs"


@dataclass
class DataForSEOConfig:
    enabled: bool = True
    login: Optional[str] = None
    password: Optional[str] = None
    search_engine: str = "google"
    location_code: Optional[int] = None
    location_name: Optional[str] = None
    language_code: Optional[str] = None
    language_name: Optional[str] = None
    top_pages_limit: int = 1000
    keywords_limit: int = 1000
    ranked_item_types: list[str] = field(
        default_factory=lambda: ["organic", "featured_snippet", "local_pack", "ai_overview_reference"]
    )
    include_clickstream: bool = False
    refresh: bool = False
    reuse_latest: bool = True
    semantic_sample_cap: int = 500


def dataforseo_credentials() -> tuple[str, str]:
    load_dotenv()
    login = (
        os.environ.get("DATAFORSEO_LOGIN")
        or os.environ.get("DATAFORSEO_EMAIL")
        or os.environ.get("DATAFORSEO_API_LOGIN")
        or ""
    )
    password = (
        os.environ.get("DATAFORSEO_PASSWORD")
        or os.environ.get("DATAFORSEO_API_KEY")
        or os.environ.get("DATAFORSEO_API_PASSWORD")
        or ""
    )
    return login, password


def _target_domain(domain: str) -> str:
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = (parsed.netloc or parsed.path).split("/")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _json_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_params(domain: str, config: DataForSEOConfig) -> dict:
    return {
        "target": _target_domain(domain),
        "search_engine": config.search_engine,
        "location_code": config.location_code or "",
        "location_name": config.location_name or "",
        "language_code": config.language_code or "",
        "language_name": config.language_name or "",
        "top_pages_limit": int(config.top_pages_limit),
        "keywords_limit": int(config.keywords_limit),
        "ranked_item_types": list(config.ranked_item_types or []),
        "include_clickstream": bool(config.include_clickstream),
    }


class DataForSEOClient:
    def __init__(
        self,
        login: str,
        password: str,
        cache_dir: Path,
        requester: Optional[Callable[[str, dict], dict]] = None,
    ):
        self.login = login
        self.password = password
        self.cache_dir = Path(cache_dir)
        self.requester = requester

    def load_or_fetch(self, domain: str, config: DataForSEOConfig) -> dict:
        cache_root = self.cache_dir / "dataforseo" / _slug(_target_domain(domain))
        params = _cache_params(domain, config)

        if not config.refresh:
            cached = self._load_cached_snapshot(cache_root, params, config.reuse_latest)
            if cached is not None:
                meta = cached.setdefault("meta", {})
                meta["cache_status"] = "hit"
                return cached

        if not self.login or not self.password:
            return {
                "meta": {
                    "status": "missing_api_key",
                    "cache_status": "miss",
                    "provider": "dataforseo",
                    "provider_label": "DataForSEO",
                    "message": (
                        "Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in .env or the "
                        "environment to fetch DataForSEO data."
                    ),
                    "params": params,
                },
                "raw": {},
            }

        try:
            raw = self._fetch_all(params)
        except Exception as exc:
            LOG.warning("DataForSEO API fetch failed for %s: %s", domain, exc)
            return {
                "meta": {
                    "status": "error",
                    "cache_status": "miss",
                    "provider": "dataforseo",
                    "provider_label": "DataForSEO",
                    "message": str(exc),
                    "params": params,
                },
                "raw": {},
            }

        snapshot = {
            "meta": {
                "status": "ok",
                "cache_status": "miss",
                "provider": "dataforseo",
                "provider_label": "DataForSEO",
                "fetched_at": date.today().isoformat(),
                "params": params,
                "params_no_date": params,
                "api_cost_usd": _api_cost(raw),
            },
            "raw": raw,
        }
        _write_json(self._cache_path(cache_root, params), snapshot)
        return snapshot

    def _load_cached_snapshot(self, cache_root: Path, params: dict, reuse_latest: bool) -> dict | None:
        if not reuse_latest:
            return None
        candidates: list[tuple[float, dict]] = []
        for path in cache_root.glob("snapshot_*.json"):
            data = _load_json(path)
            if not data:
                continue
            meta = data.get("meta", {}) or {}
            have = meta.get("params_no_date") or _params_without_date(meta.get("params", {}) or {})
            if have == params:
                candidates.append((path.stat().st_mtime, data))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _cache_path(self, cache_root: Path, params: dict) -> Path:
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"snapshot_{date.today().isoformat()}_{_json_key(params)}.json"

    def _fetch_all(self, params: dict) -> dict:
        engine = params["search_engine"]
        return {
            "domain_rank_overview": self._request(f"{engine}/domain_rank_overview/live", self._base_task(params)),
            "relevant_pages": self._request_with_fallback(
                f"{engine}/relevant_pages/live",
                {
                    **self._base_task(params),
                    "limit": params["top_pages_limit"],
                    "order_by": ["metrics.organic.etv,desc", "metrics.organic.count,desc"],
                },
            ),
            "ranked_keywords": self._request_with_fallback(
                f"{engine}/ranked_keywords/live",
                {
                    **self._base_task(params),
                    "limit": params["keywords_limit"],
                    "item_types": params["ranked_item_types"],
                    "load_rank_absolute": True,
                    "order_by": ["ranked_serp_element.serp_item.etv,desc"],
                    "include_clickstream_data": bool(params["include_clickstream"]),
                },
            ),
        }

    def _base_task(self, params: dict) -> dict:
        task = {"target": params["target"]}
        if params.get("location_code"):
            task["location_code"] = int(params["location_code"])
        elif params.get("location_name"):
            task["location_name"] = params["location_name"]
        if params.get("language_code"):
            task["language_code"] = params["language_code"]
        elif params.get("language_name"):
            task["language_name"] = params["language_name"]
        return task

    def _request(self, endpoint: str, task: dict) -> dict:
        if self.requester is not None:
            return self.requester(endpoint, task)
        url = f"{DATAFORSEO_BASE_URL}/{endpoint}"
        resp = requests.post(url, json=[task], auth=(self.login, self.password), timeout=120)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(f"{endpoint} returned HTTP {resp.status_code}: {detail}")
        data = resp.json()
        if _to_int(data.get("status_code")) not in (20000, 20100):
            raise RuntimeError(f"{endpoint}: {data.get('status_message') or data}")
        for task_row in data.get("tasks") or []:
            if _to_int(task_row.get("status_code")) not in (20000, 20100):
                raise RuntimeError(f"{endpoint}: {task_row.get('status_message') or task_row}")
        return data

    def _request_with_fallback(self, endpoint: str, task: dict) -> dict:
        try:
            return self._request(endpoint, task)
        except RuntimeError as exc:
            if "order_by" not in task:
                raise
            fallback = dict(task)
            fallback.pop("order_by", None)
            LOG.warning("DataForSEO %s rejected ordered request; retrying without order_by: %s", endpoint, exc)
            return self._request(endpoint, fallback)


def fetch_snapshot(domain: str, cache_dir: Path, config: DataForSEOConfig) -> dict:
    if not config.enabled:
        return {
            "meta": {
                "status": "disabled",
                "cache_status": "disabled",
                "provider": "dataforseo",
                "provider_label": "DataForSEO",
                "params": _cache_params(domain, config),
            },
            "raw": {},
        }
    login, password = dataforseo_credentials()
    return DataForSEOClient(
        login=config.login or login,
        password=config.password or password,
        cache_dir=cache_dir,
    ).load_or_fetch(domain, config)


def build_analysis(
    snapshot: dict,
    pages: list[PageInfo],
    embeddings: np.ndarray,
    *,
    coords: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_summaries=None,
    extracted_pages: Optional[list] = None,
    paragraph_records: Optional[list] = None,
    linkbuilding: Optional[dict] = None,
    embedder=None,
    semantic_sample_cap: int = 500,
) -> AhrefsAnalysis:
    raw = snapshot.get("raw", {}) if isinstance(snapshot, dict) else {}
    meta = dict(snapshot.get("meta", {}) or {})
    meta.setdefault("provider", "dataforseo")
    meta.setdefault("provider_label", "DataForSEO")
    if not raw:
        return AhrefsAnalysis(
            payload={
                "meta": meta,
                "summary": {},
                "metrics": {},
                "pages_by_traffic": {},
                "top_pages": [],
                "organic_keywords": [],
                "directories": [],
                "clusters": [],
                "semantic_map": {"points": [], "shown": 0},
            },
            semantic_rows=[],
            semantic_embeddings=None,
        )

    page_lookup = _page_lookup(pages)
    cluster_lookup = _cluster_lookup(cluster_summaries)
    keyword_rows = _normalize_keywords(
        _result_items(raw.get("ranked_keywords", {})),
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
    )
    keyword_by_url = _top_keyword_by_url(keyword_rows)
    page_traffic_by_index: dict[int, int] = {}
    top_pages = _normalize_pages(
        _result_items(raw.get("relevant_pages", {})),
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
        coords,
        page_traffic_by_index,
        keyword_by_url,
    )
    directories = _aggregate_directories(top_pages, keyword_rows)
    clusters = _aggregate_clusters(top_pages, keyword_rows, cluster_lookup)
    metrics = _normalize_metrics(raw)
    pages_by_traffic = _traffic_buckets(top_pages)
    semantic_points, semantic_rows, semantic_embeddings = _semantic_map(
        pages,
        embeddings,
        top_pages,
        keyword_rows,
        extracted_pages or [],
        paragraph_records or [],
        linkbuilding or {},
        embedder=embedder,
        sample_cap=semantic_sample_cap,
    )

    total_top_traffic = sum(int(r.get("traffic", 0)) for r in top_pages)
    matched_traffic = sum(int(r.get("traffic", 0)) for r in top_pages if r.get("matched"))
    total_value_cents = sum(int(r.get("value_cents", 0)) for r in top_pages)
    summary = {
        "provider": "dataforseo",
        "provider_label": "DataForSEO",
        "top_pages": len(top_pages),
        "organic_keywords": len(keyword_rows),
        "top_pages_traffic": total_top_traffic,
        "matched_top_pages": sum(1 for r in top_pages if r.get("matched")),
        "matched_traffic": matched_traffic,
        "matched_traffic_share": round(matched_traffic / total_top_traffic, 4) if total_top_traffic else 0.0,
        "top_pages_value_usd": round(total_value_cents / 100, 2),
        "paid_traffic": sum(int(r.get("paid_traffic", 0)) for r in top_pages),
        "featured_snippet_traffic": sum(int(r.get("featured_snippet_traffic", 0)) for r in top_pages),
        "local_pack_traffic": sum(int(r.get("local_pack_traffic", 0)) for r in top_pages),
        "ai_overview_traffic": sum(int(r.get("ai_overview_traffic", 0)) for r in top_pages),
        "top3_keywords": metrics.get("org_keywords_1_3", 0),
        "top10_keywords": metrics.get("org_keywords_1_10", 0),
        "directories": len(directories),
        "traffic_clusters": len(clusters),
        "api_cost_usd": meta.get("api_cost_usd", 0.0),
    }

    payload = {
        "meta": meta,
        "summary": summary,
        "metrics": metrics,
        "pages_by_traffic": pages_by_traffic,
        "top_pages": top_pages,
        "organic_keywords": keyword_rows,
        "directories": directories,
        "clusters": clusters,
        "position_buckets": _keyword_position_buckets(keyword_rows),
        "serp_features": _feature_counts(keyword_rows),
        "serp_types": _type_counts(keyword_rows),
        "intents": _intent_counts(keyword_rows),
        "semantic_map": {
            "points": semantic_points,
            "shown": len(semantic_points),
            "entity_types": ["page", "page_title", "keyword", "header", "paragraph", "link_title"],
        },
    }
    return AhrefsAnalysis(payload=payload, semantic_rows=semantic_rows, semantic_embeddings=semantic_embeddings)


def _api_cost(raw: dict) -> float:
    total = 0.0
    for response in raw.values():
        if isinstance(response, dict):
            total += _to_float(response.get("cost"))
            for task in response.get("tasks") or []:
                total += _to_float(task.get("cost"))
    return round(total, 6)


def _result_items(response: dict) -> list[dict]:
    for task in response.get("tasks") or []:
        for result in task.get("result") or []:
            items = result.get("items") or []
            if isinstance(items, list):
                return items
    return []


def _result_metrics(response: dict) -> dict:
    for task in response.get("tasks") or []:
        for result in task.get("result") or []:
            if isinstance(result.get("metrics"), dict):
                return result["metrics"]
            items = result.get("items") or []
            if items and isinstance(items[0].get("metrics"), dict):
                return items[0]["metrics"]
    return {}


def _normalize_metrics(raw: dict) -> dict:
    metrics = (
        _result_metrics(raw.get("domain_rank_overview", {}))
        or _result_metrics(raw.get("ranked_keywords", {}))
        or {}
    )
    organic = metrics.get("organic") or {}
    paid = metrics.get("paid") or {}
    featured = metrics.get("featured_snippet") or {}
    local = metrics.get("local_pack") or {}
    ai = metrics.get("ai_overview_reference") or {}
    return {
        "org_traffic": _metric_int(organic, "etv"),
        "org_keywords": _metric_int(organic, "count"),
        "org_keywords_1_3": _metric_int(organic, "pos_1")
        + _metric_int(organic, "pos_2_3"),
        "org_keywords_1_10": _metric_int(organic, "pos_1")
        + _metric_int(organic, "pos_2_3")
        + _metric_int(organic, "pos_4_10"),
        "org_keywords_11_20": _metric_int(organic, "pos_11_20"),
        "org_keywords_21_50": _metric_int(organic, "pos_21_30")
        + _metric_int(organic, "pos_31_40")
        + _metric_int(organic, "pos_41_50"),
        "org_cost": round(_metric_float(organic, "estimated_paid_traffic_cost"), 2),
        "paid_traffic": _metric_int(paid, "etv"),
        "paid_keywords": _metric_int(paid, "count"),
        "paid_cost": round(_metric_float(paid, "estimated_paid_traffic_cost"), 2),
        "featured_snippet_traffic": _metric_int(featured, "etv"),
        "featured_snippet_keywords": _metric_int(featured, "count"),
        "local_pack_traffic": _metric_int(local, "etv"),
        "local_pack_keywords": _metric_int(local, "count"),
        "ai_overview_traffic": _metric_int(ai, "etv"),
        "ai_overview_keywords": _metric_int(ai, "count"),
    }


def _metric_int(metrics: dict, key: str) -> int:
    return int(round(_metric_float(metrics, key)))


def _metric_float(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, list):
        return sum(_to_float(v) for v in value)
    return _to_float(value)


def _normalize_pages(
    raw_pages: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
    coords: Optional[np.ndarray],
    page_traffic_by_index: dict[int, int],
    keyword_by_url: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for raw in raw_pages:
        url = raw.get("page_address") or raw.get("url") or ""
        page_index = _match_page(url, page_lookup)
        page = pages[page_index] if page_index is not None else None
        metrics = raw.get("metrics") or {}
        organic = metrics.get("organic") or {}
        paid = metrics.get("paid") or {}
        featured = metrics.get("featured_snippet") or {}
        local = metrics.get("local_pack") or {}
        ai = metrics.get("ai_overview_reference") or {}
        traffic = _metric_int(organic, "etv")
        if page_index is not None:
            page_traffic_by_index[page_index] = max(page_traffic_by_index.get(page_index, 0), traffic)
        cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
        top_keyword = keyword_by_url.get(url) or keyword_by_url.get(_match_key(url)) or {}
        value_usd = _metric_float(organic, "estimated_paid_traffic_cost")
        row = {
            "provider": "dataforseo",
            "url": url,
            "matched_url": page.url if page else "",
            "matched": page is not None,
            "title": page.title if page else "",
            "section": page.section if page else section_for_url(url),
            "cluster": cluster_id,
            "cluster_label": cluster_label,
            "traffic": traffic,
            "keywords": _metric_int(organic, "count"),
            "value_cents": int(round(value_usd * 100)),
            "value_usd": round(value_usd, 2),
            "paid_traffic": _metric_int(paid, "etv"),
            "paid_keywords": _metric_int(paid, "count"),
            "featured_snippet_traffic": _metric_int(featured, "etv"),
            "featured_snippet_keywords": _metric_int(featured, "count"),
            "local_pack_traffic": _metric_int(local, "etv"),
            "local_pack_keywords": _metric_int(local, "count"),
            "ai_overview_traffic": _metric_int(ai, "etv"),
            "ai_overview_keywords": _metric_int(ai, "count"),
            "top3_keywords": _metric_int(organic, "pos_1") + _metric_int(organic, "pos_2_3"),
            "top10_keywords": _metric_int(organic, "pos_1")
            + _metric_int(organic, "pos_2_3")
            + _metric_int(organic, "pos_4_10"),
            "top20_keywords": _metric_int(organic, "pos_1")
            + _metric_int(organic, "pos_2_3")
            + _metric_int(organic, "pos_4_10")
            + _metric_int(organic, "pos_11_20"),
            "position_buckets": _page_position_buckets(organic),
            "top_keyword": top_keyword.get("keyword", ""),
            "top_keyword_position": _to_int(top_keyword.get("position")),
            "top_keyword_title": top_keyword.get("title", ""),
            "top_keyword_country": top_keyword.get("country", ""),
            "top_keyword_volume": _to_int(top_keyword.get("volume")),
            "referring_domains": _to_int(raw.get("referring_domains")),
            "url_rating": 0,
            "page_type": "",
        }
        if coords is not None and page_index is not None and page_index < len(coords):
            row["x"] = float(coords[page_index, 0])
            row["y"] = float(coords[page_index, 1])
        rows.append(row)
    rows.sort(key=lambda r: r["traffic"], reverse=True)
    return rows


def _normalize_keywords(
    raw_keywords: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
) -> list[dict]:
    rows: list[dict] = []
    for raw in raw_keywords:
        keyword_data = raw.get("keyword_data") or {}
        keyword_info = keyword_data.get("keyword_info") or {}
        serp = raw.get("ranked_serp_element") or {}
        serp_item = serp.get("serp_item") or {}
        keyword = keyword_data.get("keyword") or raw.get("keyword") or ""
        if not keyword:
            continue
        url = serp_item.get("url") or serp_item.get("domain") or ""
        page_index = _match_page(url, page_lookup)
        page = pages[page_index] if page_index is not None else None
        cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
        intents = _keyword_intents(keyword_data)
        traffic = _to_int(serp_item.get("etv") or raw.get("etv"))
        cpc_usd = _to_float(keyword_info.get("cpc"))
        position = _to_int(serp.get("rank_group") or serp_item.get("rank_group") or serp.get("rank_absolute"))
        rows.append({
            "provider": "dataforseo",
            "keyword": keyword,
            "url": url,
            "matched_url": page.url if page else "",
            "matched": page is not None,
            "page_title": page.title if page else serp_item.get("title", ""),
            "section": page.section if page else section_for_url(url),
            "cluster": cluster_id,
            "cluster_label": cluster_label,
            "position": position,
            "rank_absolute": _to_int(serp.get("rank_absolute") or serp_item.get("rank_absolute")),
            "traffic": traffic,
            "volume": _to_int(keyword_info.get("search_volume")),
            "cpc_cents": int(round(cpc_usd * 100)),
            "cpc_usd": round(cpc_usd, 2),
            "traffic_value_usd": round(_to_float(serp_item.get("estimated_paid_traffic_cost")), 2),
            "country": str(raw.get("location_code") or ""),
            "serp_type": serp_item.get("type") or "",
            "intents": intents,
            "serp_features": _serp_features(keyword_data, serp),
            "keyword_difficulty": _to_int((keyword_data.get("keyword_properties") or {}).get("keyword_difficulty")),
            "last_update": serp_item.get("last_updated_time") or raw.get("last_updated_time") or "",
            "title": serp_item.get("title", ""),
            "description": serp_item.get("description", ""),
            "is_new": bool(serp.get("is_new")),
            "rank_change": _to_int(serp.get("rank_changes_absolute")),
            "position_bucket": _position_bucket(position),
        })
    rows.sort(key=lambda r: (int(r.get("traffic", 0)), int(r.get("volume", 0))), reverse=True)
    return rows


def _top_keyword_by_url(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        current = out.get(url) or out.get(_match_key(url))
        if current and int(current.get("traffic", 0) or 0) >= int(row.get("traffic", 0) or 0):
            continue
        value = {
            "keyword": row.get("keyword", ""),
            "position": row.get("position", 0),
            "title": row.get("page_title", ""),
            "country": row.get("country", ""),
            "volume": row.get("volume", 0),
            "traffic": row.get("traffic", 0),
        }
        out[url] = value
        out[_match_key(url)] = value
    return out


def _match_key(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{host}{path}"


def _keyword_intents(keyword_data: dict) -> list[str]:
    info = keyword_data.get("search_intent_info") or {}
    labels = []
    main = info.get("main_intent")
    if main:
        labels.append(str(main))
    for intent in info.get("foreign_intent") or []:
        if intent and intent not in labels:
            labels.append(str(intent))
    return labels


def _serp_features(keyword_data: dict, serp: dict) -> list[str]:
    features = []
    for source in [
        serp.get("serp_item_types"),
        (keyword_data.get("serp_info") or {}).get("serp_item_types"),
    ]:
        for item in source or []:
            if item and item not in features:
                features.append(str(item))
    serp_item_type = ((serp.get("serp_item") or {}).get("type") or "").strip()
    if serp_item_type and serp_item_type not in features:
        features.append(serp_item_type)
    return features


def _page_position_buckets(metrics: dict) -> dict[str, int]:
    return {
        "pos_1": _metric_int(metrics, "pos_1"),
        "pos_2_3": _metric_int(metrics, "pos_2_3"),
        "pos_4_10": _metric_int(metrics, "pos_4_10"),
        "pos_11_20": _metric_int(metrics, "pos_11_20"),
        "pos_21_50": _metric_int(metrics, "pos_21_30")
        + _metric_int(metrics, "pos_31_40")
        + _metric_int(metrics, "pos_41_50"),
        "pos_51_plus": _metric_int(metrics, "pos_51_60")
        + _metric_int(metrics, "pos_61_70")
        + _metric_int(metrics, "pos_71_80")
        + _metric_int(metrics, "pos_81_90")
        + _metric_int(metrics, "pos_91_100"),
    }


def _traffic_buckets(top_pages: list[dict]) -> dict:
    buckets = {
        "range0_pages": 0,
        "range0_traffic": 0,
        "range1_10_pages": 0,
        "range1_10_traffic": 0,
        "range10_100_pages": 0,
        "range10_100_traffic": 0,
        "range100_pages": 0,
        "range100_traffic": 0,
    }
    for row in top_pages:
        traffic = int(row.get("traffic", 0) or 0)
        if traffic >= 100:
            key = "range100"
        elif traffic >= 10:
            key = "range10_100"
        elif traffic >= 1:
            key = "range1_10"
        else:
            key = "range0"
        buckets[f"{key}_pages"] += 1
        buckets[f"{key}_traffic"] += traffic
    return buckets


def _keyword_position_buckets(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        bucket = _position_bucket(row.get("position"))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _feature_counts(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        for feature in row.get("serp_features") or []:
            counts[str(feature)] = counts.get(str(feature), 0) + 1
    return [
        {"feature": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    ]


def _type_counts(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("serp_type"):
            counts[str(row["serp_type"])] = counts.get(str(row["serp_type"]), 0) + 1
    return [
        {"type": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]
    ]


def _intent_counts(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        for intent in row.get("intents") or []:
            counts[str(intent)] = counts.get(str(intent), 0) + 1
    return [
        {"intent": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]
    ]
