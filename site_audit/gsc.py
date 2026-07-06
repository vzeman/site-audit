"""Optional Google Search Console enrichment for first-party Google metrics.

The normalized payload mirrors the Ahrefs/DataForSEO schema used by the rest
of the pipeline: ``traffic`` means clicks for GSC rows, while impressions, CTR,
and average position remain available for GSC-specific reporting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote, urlparse

import numpy as np
import requests

from .ahrefs import (
    AhrefsAnalysis,
    _aggregate_clusters,
    _aggregate_directories,
    _cluster_lookup,
    _entity_alignment,
    _load_json,
    _match_page,
    _page_cluster,
    _page_lookup,
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

GSC_BASE_URL = "https://searchconsole.googleapis.com/webmasters/v3/sites"
GSC_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
MAX_QUERY_PAGES_ROWS = 5000


@dataclass
class GSCConfig:
    enabled: bool = True
    property_url: Optional[str] = None
    access_token: Optional[str] = None
    service_account_file: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    search_type: str = "web"
    top_pages_limit: int = 1000
    keywords_limit: int = 1000
    row_limit: int = 25000
    refresh: bool = False
    reuse_latest: bool = True
    semantic_sample_cap: int = 500


def _default_dates() -> tuple[str, str]:
    end = date.today() - timedelta(days=3)
    start = end - timedelta(days=27)
    return start.isoformat(), end.isoformat()


def _target_domain(domain: str) -> str:
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = (parsed.netloc or parsed.path).split("/")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _env_property_url() -> str:
    load_dotenv()
    return (
        os.environ.get("GSC_PROPERTY_URL")
        or os.environ.get("GSC_SITE_URL")
        or os.environ.get("GOOGLE_SEARCH_CONSOLE_SITE_URL")
        or ""
    )


def _property_url(domain: str, config: GSCConfig) -> str:
    return config.property_url or _env_property_url() or f"sc-domain:{_target_domain(domain)}"


def _json_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_params(domain: str, config: GSCConfig) -> dict:
    default_start, default_end = _default_dates()
    start_date = config.start_date or os.environ.get("GSC_START_DATE") or default_start
    end_date = config.end_date or os.environ.get("GSC_END_DATE") or default_end
    return {
        "property_url": _property_url(domain, config),
        "start_date": start_date,
        "end_date": end_date,
        "search_type": config.search_type or "web",
        "top_pages_limit": int(config.top_pages_limit),
        "keywords_limit": int(config.keywords_limit),
        "row_limit": int(config.row_limit),
    }


def _service_account_token(path: str) -> tuple[str, str, str]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except Exception:
        return (
            "",
            "missing_dependency",
            "Install google-auth or set GSC_ACCESS_TOKEN to fetch Google Search Console data.",
        )
    try:
        credentials = service_account.Credentials.from_service_account_file(path, scopes=[GSC_SCOPE])
        credentials.refresh(Request())
        return credentials.token or "", "ok", ""
    except Exception as exc:
        return "", "credential_error", str(exc)


def resolve_access_token(config: GSCConfig) -> tuple[str, str, str]:
    load_dotenv()
    token = (
        config.access_token
        or os.environ.get("GSC_ACCESS_TOKEN")
        or os.environ.get("GOOGLE_SEARCH_CONSOLE_ACCESS_TOKEN")
        or ""
    )
    if token:
        return token, "ok", ""
    service_file = (
        config.service_account_file
        or os.environ.get("GSC_SERVICE_ACCOUNT_FILE")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or ""
    )
    if service_file:
        return _service_account_token(service_file)
    return (
        "",
        "missing_credentials",
        "Set GSC_ACCESS_TOKEN or GSC_SERVICE_ACCOUNT_FILE in .env or the environment to fetch Google Search Console data.",
    )


class GSCClient:
    def __init__(
        self,
        access_token: str,
        cache_dir: Path,
        *,
        credential_status: str = "ok",
        credential_message: str = "",
        requester: Optional[Callable[[str, dict, str], dict]] = None,
    ):
        self.access_token = access_token
        self.cache_dir = Path(cache_dir)
        self.credential_status = credential_status
        self.credential_message = credential_message
        self.requester = requester

    def load_or_fetch(self, domain: str, config: GSCConfig) -> dict:
        params = _cache_params(domain, config)
        cache_root = self.cache_dir / "gsc" / _slug(params["property_url"])

        if not config.refresh:
            cached = self._load_cached_snapshot(cache_root, params, config.reuse_latest)
            if cached is not None:
                meta = cached.setdefault("meta", {})
                meta["cache_status"] = "hit"
                return cached

        if not self.access_token:
            return {
                "meta": {
                    "status": self.credential_status or "missing_credentials",
                    "cache_status": "miss",
                    "provider": "gsc",
                    "provider_label": "Google Search Console",
                    "message": self.credential_message,
                    "params": params,
                },
                "raw": {},
            }

        try:
            raw = self._fetch_all(params)
        except Exception as exc:
            LOG.warning("Google Search Console fetch failed for %s: %s", domain, exc)
            return {
                "meta": {
                    "status": "error",
                    "cache_status": "miss",
                    "provider": "gsc",
                    "provider_label": "Google Search Console",
                    "message": str(exc),
                    "params": params,
                },
                "raw": {},
            }

        snapshot = {
            "meta": {
                "status": "ok",
                "cache_status": "miss",
                "provider": "gsc",
                "provider_label": "Google Search Console",
                "fetched_at": date.today().isoformat(),
                "params": params,
                "params_no_date": params,
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
            if (meta.get("params_no_date") or meta.get("params") or {}) == params:
                candidates.append((path.stat().st_mtime, data))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _cache_path(self, cache_root: Path, params: dict) -> Path:
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"snapshot_{params['start_date']}_{params['end_date']}_{_json_key(params)}.json"

    def _fetch_all(self, params: dict) -> dict:
        site_url = params["property_url"]
        return {
            "totals": self._request(site_url, self._body(params, [], 1)),
            "daily": self._request(site_url, self._body(params, ["date"], min(params["row_limit"], 1000))),
            "pages": self._request(site_url, self._body(params, ["page"], params["top_pages_limit"])),
            "queries": self._request(site_url, self._body(params, ["query"], params["keywords_limit"])),
            "query_page": self._request(site_url, self._body(params, ["query", "page"], params["keywords_limit"])),
            "countries": self._request(site_url, self._body(params, ["country"], 1000)),
            "devices": self._request(site_url, self._body(params, ["device"], 1000)),
            "search_appearances": self._request(site_url, self._body(params, ["searchAppearance"], 1000)),
        }

    def _body(self, params: dict, dimensions: list[str], row_limit: int) -> dict:
        body = {
            "startDate": params["start_date"],
            "endDate": params["end_date"],
            "dimensions": dimensions,
            "rowLimit": int(row_limit),
            "startRow": 0,
        }
        if params["search_type"] and params["search_type"] != "web":
            body["type"] = params["search_type"]
        return body

    def _request(self, site_url: str, body: dict) -> dict:
        if self.requester is not None:
            return self.requester(site_url, body, self.access_token)
        url = f"{GSC_BASE_URL}/{quote(site_url, safe='')}/searchAnalytics/query"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} {response.text[:500]}")
        return response.json()


def fetch_snapshot(domain: str, cache_dir: Path, config: GSCConfig) -> dict:
    if not config.enabled:
        return {
            "meta": {
                "status": "disabled",
                "cache_status": "disabled",
                "provider": "gsc",
                "provider_label": "Google Search Console",
                "params": _cache_params(domain, config),
            },
            "raw": {},
        }
    token, status, message = resolve_access_token(config)
    return GSCClient(
        access_token=token,
        cache_dir=cache_dir,
        credential_status=status,
        credential_message=message,
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
    meta.setdefault("provider", "gsc")
    meta.setdefault("provider_label", "Google Search Console")
    if not raw:
        return AhrefsAnalysis(
            payload={
                "meta": meta,
                "summary": {},
                "metrics": {},
                "pages_by_traffic": {},
                "top_pages": [],
                "organic_keywords": [],
                "query_pages": [],
                "directories": [],
                "clusters": [],
                "daily": [],
                "countries": [],
                "devices": [],
                "search_appearances": [],
                "semantic_map": {"points": [], "shown": 0},
            },
            semantic_rows=[],
            semantic_embeddings=None,
        )

    page_lookup = _page_lookup(pages)
    cluster_lookup = _cluster_lookup(cluster_summaries)
    query_page_rows = _normalize_query_page(
        _rows(raw.get("query_page", {})),
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
    )
    query_rows = _normalize_query_rows(_rows(raw.get("queries", {})), query_page_rows)
    top_query_by_page = _top_query_by_page(query_page_rows)
    page_keyword_counts = _keyword_counts_by_page(query_page_rows)
    page_traffic_by_index: dict[int, int] = {}
    top_pages = _normalize_pages(
        _rows(raw.get("pages", {})),
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
        coords,
        page_traffic_by_index,
        top_query_by_page,
        page_keyword_counts,
    )
    if not top_pages and query_page_rows:
        top_pages = _pages_from_keywords(
            query_page_rows,
            pages,
            page_lookup,
            cluster_labels,
            cluster_lookup,
            coords,
            page_traffic_by_index,
        )

    organic_keywords = query_page_rows or query_rows
    query_pages = _query_pages_payload(query_page_rows)
    directories = _aggregate_directories(top_pages, organic_keywords)
    clusters = _aggregate_clusters(top_pages, organic_keywords, cluster_lookup)
    metrics = _metrics(raw, top_pages, organic_keywords)
    semantic_points, semantic_rows, semantic_embeddings = _semantic_map(
        pages,
        embeddings,
        top_pages,
        organic_keywords,
        extracted_pages or [],
        paragraph_records or [],
        linkbuilding or {},
        embedder=embedder,
        sample_cap=semantic_sample_cap,
    )

    total_top_clicks = sum(int(r.get("traffic", 0)) for r in top_pages)
    matched_clicks = sum(int(r.get("traffic", 0)) for r in top_pages if r.get("matched"))
    summary = {
        "provider": "gsc",
        "provider_label": "Google Search Console",
        "top_pages": len(top_pages),
        "organic_keywords": len(organic_keywords),
        "top_pages_traffic": total_top_clicks,
        "matched_top_pages": sum(1 for r in top_pages if r.get("matched")),
        "matched_traffic": matched_clicks,
        "matched_traffic_share": round(matched_clicks / total_top_clicks, 4) if total_top_clicks else 0.0,
        "total_clicks": int(metrics.get("org_traffic", total_top_clicks)),
        "total_impressions": int(metrics.get("gsc_impressions", 0)),
        "avg_ctr": round(_to_float(metrics.get("gsc_ctr")), 4),
        "avg_position": round(_to_float(metrics.get("gsc_avg_position")), 2),
        "top3_keywords": metrics.get("org_keywords_1_3", 0),
        "top10_keywords": metrics.get("org_keywords_1_10", 0),
        "directories": len(directories),
        "traffic_clusters": len(clusters),
        "start_date": (meta.get("params") or {}).get("start_date", ""),
        "end_date": (meta.get("params") or {}).get("end_date", ""),
    }
    entity_alignment = _entity_alignment(semantic_rows, semantic_embeddings)

    payload = {
        "meta": meta,
        "summary": summary,
        "metrics": metrics,
        "pages_by_traffic": _traffic_buckets(top_pages),
        "top_pages": top_pages,
        "organic_keywords": organic_keywords,
        "query_pages": query_pages,
        "directories": directories,
        "clusters": clusters,
        "position_buckets": _position_buckets(organic_keywords),
        "countries": _dimension_rows(_rows(raw.get("countries", {})), "country"),
        "devices": _dimension_rows(_rows(raw.get("devices", {})), "device"),
        "search_appearances": _dimension_rows(_rows(raw.get("search_appearances", {})), "search_appearance"),
        "daily": _daily_rows(_rows(raw.get("daily", {}))),
        "semantic_map": {
            "points": semantic_points,
            "shown": len(semantic_points),
            "entity_types": ["page", "page_title", "keyword", "header", "paragraph", "link_title"],
        },
        "entity_alignment": entity_alignment,
    }
    return AhrefsAnalysis(payload=payload, semantic_rows=semantic_rows, semantic_embeddings=semantic_embeddings)


def _rows(response: dict) -> list[dict]:
    rows = response.get("rows") or []
    return rows if isinstance(rows, list) else []


def _keys(row: dict) -> list[str]:
    keys = row.get("keys") or []
    return [str(k) for k in keys] if isinstance(keys, list) else []


def _ctr(clicks: float, impressions: float) -> float:
    return round(clicks / impressions, 4) if impressions else 0.0


def _row_ctr(row: dict) -> float:
    if row.get("ctr") not in (None, ""):
        return round(_to_float(row.get("ctr")), 4)
    return _ctr(_to_float(row.get("clicks")), _to_float(row.get("impressions")))


def _normalize_query_page(
    raw_rows: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
) -> list[dict]:
    rows: list[dict] = []
    for raw in raw_rows:
        keys = _keys(raw)
        if len(keys) < 2:
            continue
        keyword, url = keys[0], keys[1]
        page_index = _match_page(url, page_lookup)
        page = pages[page_index] if page_index is not None else None
        cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
        clicks = _to_float(raw.get("clicks"))
        impressions = _to_float(raw.get("impressions"))
        position = _to_float(raw.get("position"))
        rows.append({
            "provider": "gsc",
            "keyword": keyword,
            "url": url,
            "matched_url": page.url if page else "",
            "matched": page is not None,
            "page_title": page.title if page else "",
            "section": page.section if page else section_for_url(url),
            "cluster": cluster_id,
            "cluster_label": cluster_label,
            "position": round(position, 2),
            "traffic": int(round(clicks)),
            "clicks": round(clicks, 2),
            "impressions": round(impressions, 2),
            "ctr": _row_ctr(raw),
            "volume": int(round(impressions)),
            "cpc_cents": 0,
            "cpc_usd": 0.0,
            "country": "",
            "serp_type": "organic",
            "intents": [],
            "serp_features": [],
            "last_update": "",
            "position_bucket": _position_bucket(position),
        })
    rows.sort(key=lambda r: (r["clicks"], r["impressions"]), reverse=True)
    return rows


def _normalize_query_rows(raw_rows: list[dict], query_page_rows: list[dict]) -> list[dict]:
    page_by_keyword: dict[str, dict] = {}
    for row in query_page_rows:
        current = page_by_keyword.get(row["keyword"])
        if current is None or _to_float(row.get("clicks")) > _to_float(current.get("clicks")):
            page_by_keyword[row["keyword"]] = row
    rows: list[dict] = []
    for raw in raw_rows:
        keys = _keys(raw)
        if not keys:
            continue
        keyword = keys[0]
        best_page = page_by_keyword.get(keyword, {})
        clicks = _to_float(raw.get("clicks"))
        impressions = _to_float(raw.get("impressions"))
        position = _to_float(raw.get("position"))
        rows.append({
            "provider": "gsc",
            "keyword": keyword,
            "url": best_page.get("url", ""),
            "matched_url": best_page.get("matched_url", ""),
            "matched": bool(best_page.get("matched")),
            "page_title": best_page.get("page_title", ""),
            "section": best_page.get("section", ""),
            "cluster": best_page.get("cluster"),
            "cluster_label": best_page.get("cluster_label", ""),
            "position": round(position, 2),
            "traffic": int(round(clicks)),
            "clicks": round(clicks, 2),
            "impressions": round(impressions, 2),
            "ctr": _row_ctr(raw),
            "volume": int(round(impressions)),
            "cpc_cents": 0,
            "cpc_usd": 0.0,
            "country": "",
            "serp_type": "organic",
            "intents": [],
            "serp_features": [],
            "last_update": "",
            "position_bucket": _position_bucket(position),
        })
    rows.sort(key=lambda r: (r["clicks"], r["impressions"]), reverse=True)
    return rows


def _query_pages_payload(query_page_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in query_page_rows:
        rows.append({
            "query": row.get("keyword") or "",
            "url": row.get("matched_url") or row.get("url") or "",
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "ctr": row.get("ctr"),
            "position": row.get("position", 0),
            "source": "gsc",
            "provider": "gsc",
            "provider_label": "Google Search Console",
            "matched_url": row.get("matched_url", ""),
            "page_title": row.get("page_title", ""),
            "cluster": row.get("cluster"),
            "cluster_label": row.get("cluster_label", ""),
            "intents": [],
        })
    rows.sort(key=lambda r: _to_float(r.get("impressions")), reverse=True)
    return rows[:MAX_QUERY_PAGES_ROWS]


def _top_query_by_page(query_rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in query_rows:
        key = row.get("url") or row.get("matched_url") or ""
        if not key:
            continue
        current = out.get(key)
        if current is None or _to_float(row.get("clicks")) > _to_float(current.get("clicks")):
            out[key] = row
    return out


def _keyword_counts_by_page(query_rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = defaultdict(lambda: {"keywords": set(), "top3": 0, "top10": 0, "top20": 0})
    for row in query_rows:
        key = row.get("url") or row.get("matched_url") or ""
        if not key:
            continue
        grouped[key]["keywords"].add(row.get("keyword", ""))
        top3, top10, top20 = _top_position_counts(row.get("position"))
        grouped[key]["top3"] += top3
        grouped[key]["top10"] += top10
        grouped[key]["top20"] += top20
    return {
        key: {
            "keywords": len(value["keywords"]),
            "top3": int(value["top3"]),
            "top10": int(value["top10"]),
            "top20": int(value["top20"]),
        }
        for key, value in grouped.items()
    }


def _normalize_pages(
    raw_rows: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
    coords: Optional[np.ndarray],
    page_traffic_by_index: dict[int, int],
    top_query_by_page: dict[str, dict],
    page_keyword_counts: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for raw in raw_rows:
        keys = _keys(raw)
        if not keys:
            continue
        url = keys[0]
        row = _page_row(
            url,
            raw,
            pages,
            page_lookup,
            cluster_labels,
            cluster_lookup,
            coords,
            page_traffic_by_index,
            top_query_by_page,
            page_keyword_counts,
        )
        rows.append(row)
    rows.sort(key=lambda r: (r["clicks"], r["impressions"]), reverse=True)
    return rows


def _page_row(
    url: str,
    raw: dict,
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
    coords: Optional[np.ndarray],
    page_traffic_by_index: dict[int, int],
    top_query_by_page: dict[str, dict],
    page_keyword_counts: dict[str, dict],
) -> dict:
    page_index = _match_page(url, page_lookup)
    page = pages[page_index] if page_index is not None else None
    clicks = _to_float(raw.get("clicks"))
    impressions = _to_float(raw.get("impressions"))
    position = _to_float(raw.get("position"))
    if page_index is not None:
        page_traffic_by_index[page_index] = max(page_traffic_by_index.get(page_index, 0), int(round(clicks)))
    cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
    top_query = top_query_by_page.get(url) or top_query_by_page.get(page.url if page else "") or {}
    counts = page_keyword_counts.get(url) or page_keyword_counts.get(page.url if page else "") or {}
    row = {
        "provider": "gsc",
        "url": url,
        "matched_url": page.url if page else "",
        "matched": page is not None,
        "title": page.title if page else "",
        "section": page.section if page else section_for_url(url),
        "cluster": cluster_id,
        "cluster_label": cluster_label,
        "traffic": int(round(clicks)),
        "clicks": round(clicks, 2),
        "impressions": round(impressions, 2),
        "ctr": _row_ctr(raw),
        "position": round(position, 2),
        "keywords": int(counts.get("keywords", 0)),
        "value_cents": 0,
        "value_usd": 0.0,
        "top_keyword": top_query.get("keyword", ""),
        "top_keyword_position": top_query.get("position", 0),
        "top_keyword_title": page.title if page else "",
        "top_keyword_country": "",
        "top_keyword_volume": int(round(_to_float(top_query.get("impressions")))),
        "top_keyword_clicks": round(_to_float(top_query.get("clicks")), 2),
        "top_keyword_impressions": round(_to_float(top_query.get("impressions")), 2),
        "top_keyword_ctr": _to_float(top_query.get("ctr")),
        "referring_domains": 0,
        "url_rating": 0,
        "page_type": "",
        "top3_keywords": int(counts.get("top3", 0)),
        "top10_keywords": int(counts.get("top10", 0)),
        "top20_keywords": int(counts.get("top20", 0)),
        "position_buckets": {_position_bucket(top_query.get("position")): 1} if top_query.get("position") else {},
    }
    if coords is not None and page_index is not None and page_index < len(coords):
        row["x"] = float(coords[page_index, 0])
        row["y"] = float(coords[page_index, 1])
    return row


def _pages_from_keywords(
    keyword_rows: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
    coords: Optional[np.ndarray],
    page_traffic_by_index: dict[int, int],
) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in keyword_rows:
        url = row.get("url") or row.get("matched_url") or ""
        if not url:
            continue
        group = grouped.setdefault(url, {"keys": [url], "clicks": 0.0, "impressions": 0.0, "weighted_position": 0.0})
        clicks = _to_float(row.get("clicks"))
        impressions = _to_float(row.get("impressions"))
        group["clicks"] += clicks
        group["impressions"] += impressions
        group["weighted_position"] += _to_float(row.get("position")) * max(impressions, 1.0)
    top_query = _top_query_by_page(keyword_rows)
    counts = _keyword_counts_by_page(keyword_rows)
    rows = []
    for url, raw in grouped.items():
        impressions = _to_float(raw.get("impressions"))
        raw["ctr"] = _ctr(_to_float(raw.get("clicks")), impressions)
        raw["position"] = raw["weighted_position"] / max(impressions, 1.0)
        rows.append(_page_row(url, raw, pages, page_lookup, cluster_labels, cluster_lookup, coords, page_traffic_by_index, top_query, counts))
    rows.sort(key=lambda r: (r["clicks"], r["impressions"]), reverse=True)
    return rows


def _metrics(raw: dict, top_pages: list[dict], keywords: list[dict]) -> dict:
    totals = _rows(raw.get("totals", {}))
    total = totals[0] if totals else {}
    clicks = _to_float(total.get("clicks")) or sum(_to_float(r.get("clicks")) for r in top_pages)
    impressions = _to_float(total.get("impressions")) or sum(_to_float(r.get("impressions")) for r in top_pages)
    avg_position = _to_float(total.get("position")) or _weighted_position(keywords)
    buckets = _position_buckets(keywords)
    return {
        "org_traffic": int(round(clicks)),
        "org_keywords": len({r.get("keyword") for r in keywords if r.get("keyword")}),
        "org_keywords_1_3": int(buckets.get("pos_1", 0) + buckets.get("pos_2_3", 0)),
        "org_keywords_1_10": int(buckets.get("pos_1", 0) + buckets.get("pos_2_3", 0) + buckets.get("pos_4_10", 0)),
        "org_keywords_11_20": int(buckets.get("pos_11_20", 0)),
        "org_keywords_21_50": int(buckets.get("pos_21_50", 0)),
        "org_cost": 0.0,
        "gsc_clicks": round(clicks, 2),
        "gsc_impressions": round(impressions, 2),
        "gsc_ctr": _row_ctr(total) if total else _ctr(clicks, impressions),
        "gsc_avg_position": round(avg_position, 2),
    }


def _weighted_position(rows: list[dict]) -> float:
    total_weight = 0.0
    weighted = 0.0
    for row in rows:
        weight = max(_to_float(row.get("impressions")), 1.0)
        pos = _to_float(row.get("position"))
        if pos <= 0:
            continue
        weighted += pos * weight
        total_weight += weight
    return weighted / total_weight if total_weight else 0.0


def _position_buckets(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        counts[_position_bucket(row.get("position"))] += 1
    counts.pop("unknown", None)
    return dict(counts)


def _traffic_buckets(top_pages: list[dict]) -> dict:
    buckets = {"1_10": 0, "11_100": 0, "101_1000": 0, "1001_plus": 0}
    for row in top_pages:
        clicks = _to_int(row.get("clicks"))
        if clicks <= 10:
            buckets["1_10"] += 1
        elif clicks <= 100:
            buckets["11_100"] += 1
        elif clicks <= 1000:
            buckets["101_1000"] += 1
        else:
            buckets["1001_plus"] += 1
    return buckets


def _dimension_rows(raw_rows: list[dict], key_name: str) -> list[dict]:
    rows = []
    for raw in raw_rows:
        keys = _keys(raw)
        if not keys:
            continue
        clicks = _to_float(raw.get("clicks"))
        impressions = _to_float(raw.get("impressions"))
        rows.append({
            key_name: keys[0],
            "clicks": round(clicks, 2),
            "traffic": int(round(clicks)),
            "impressions": round(impressions, 2),
            "ctr": _row_ctr(raw),
            "position": round(_to_float(raw.get("position")), 2),
        })
    rows.sort(key=lambda r: (r["clicks"], r["impressions"]), reverse=True)
    return rows


def _daily_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for raw in raw_rows:
        keys = _keys(raw)
        if not keys:
            continue
        clicks = _to_float(raw.get("clicks"))
        impressions = _to_float(raw.get("impressions"))
        rows.append({
            "date": keys[0],
            "clicks": round(clicks, 2),
            "traffic": int(round(clicks)),
            "impressions": round(impressions, 2),
            "ctr": _row_ctr(raw),
            "position": round(_to_float(raw.get("position")), 2),
        })
    rows.sort(key=lambda r: r["date"])
    return rows
