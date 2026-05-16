"""Optional Google Ads search-term enrichment.

This provider is intentionally opt-in. It uses paid search terms as a strong
business-relevance signal for competitive paragraph-gap analysis: if the
business is already spending money on a query, that query is usually closer to
product/service intent than incidental organic traffic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests

from .ahrefs import (
    AhrefsAnalysis,
    _aggregate_clusters,
    _aggregate_directories,
    _cluster_lookup,
    _entity_alignment,
    _load_json,
    _position_bucket,
    _semantic_map,
    _slug,
    _to_float,
    _to_int,
    _write_json,
    load_dotenv,
)
from .analyzer import PageInfo

LOG = logging.getLogger(__name__)

GOOGLE_ADS_API_VERSION = "v22"
GOOGLE_ADS_SCOPE = "https://www.googleapis.com/auth/adwords"
GOOGLE_ADS_TOKEN_URL = "https://oauth2.googleapis.com/token"


@dataclass
class GoogleAdsConfig:
    enabled: bool = True
    developer_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    customer_id: Optional[str] = None
    login_customer_id: Optional[str] = None
    api_version: str = GOOGLE_ADS_API_VERSION
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    search_terms_limit: int = 1000
    min_cost: float = 0.0
    refresh: bool = False
    reuse_latest: bool = True
    semantic_sample_cap: int = 500


def _default_dates() -> tuple[str, str]:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=89)
    return start.isoformat(), end.isoformat()


def _clean_customer_id(value: str | None) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _json_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback).strip()


def _cache_params(config: GoogleAdsConfig) -> dict:
    load_dotenv()
    default_start, default_end = _default_dates()
    return {
        "customer_id": _clean_customer_id(config.customer_id or _env("GOOGLE_ADS_CUSTOMER_ID")),
        "login_customer_id": _clean_customer_id(config.login_customer_id or _env("GOOGLE_ADS_LOGIN_CUSTOMER_ID")),
        "api_version": (config.api_version or _env("GOOGLE_ADS_API_VERSION", GOOGLE_ADS_API_VERSION)).strip(" /"),
        "start_date": config.start_date or _env("GOOGLE_ADS_START_DATE") or default_start,
        "end_date": config.end_date or _env("GOOGLE_ADS_END_DATE") or default_end,
        "search_terms_limit": int(config.search_terms_limit),
        "min_cost": float(config.min_cost or 0.0),
    }


def _credentials(config: GoogleAdsConfig) -> dict:
    load_dotenv()
    return {
        "developer_token": config.developer_token or _env("GOOGLE_ADS_DEVELOPER_TOKEN"),
        "client_id": config.client_id or _env("GOOGLE_ADS_CLIENT_ID"),
        "client_secret": config.client_secret or _env("GOOGLE_ADS_CLIENT_SECRET"),
        "refresh_token": config.refresh_token or _env("GOOGLE_ADS_REFRESH_TOKEN"),
    }


def _missing_credentials(params: dict, creds: dict) -> list[str]:
    missing = [key for key, value in creds.items() if not value]
    if not params.get("customer_id"):
        missing.append("customer_id")
    return missing


def _access_token(creds: dict) -> str:
    response = requests.post(
        GOOGLE_ADS_TOKEN_URL,
        data={
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OAuth refresh failed: {response.status_code} {response.text[:300]}")
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth refresh did not return an access_token")
    return str(token)


class GoogleAdsClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        requester: Optional[Callable[[str, dict, dict], dict]] = None,
        token_getter: Optional[Callable[[dict], str]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.requester = requester
        self.token_getter = token_getter or _access_token

    def load_or_fetch(self, config: GoogleAdsConfig) -> dict:
        params = _cache_params(config)
        creds = _credentials(config)
        cache_root = self.cache_dir / "google_ads" / _slug(params.get("customer_id") or "customer")

        if not config.refresh:
            cached = self._load_cached_snapshot(cache_root, params, config.reuse_latest)
            if cached is not None:
                meta = cached.setdefault("meta", {})
                meta["cache_status"] = "hit"
                return cached

        missing = _missing_credentials(params, creds)
        if missing:
            return {
                "meta": {
                    "status": "missing_credentials",
                    "cache_status": "miss",
                    "provider": "google_ads",
                    "provider_label": "Google Ads",
                    "message": "Missing Google Ads credentials: " + ", ".join(missing),
                    "params": params,
                },
                "raw": {},
            }

        try:
            token = self.token_getter(creds)
            raw = self._fetch(params, creds, token)
        except Exception as exc:
            LOG.warning("Google Ads fetch failed: %s", exc)
            return {
                "meta": {
                    "status": "error",
                    "cache_status": "miss",
                    "provider": "google_ads",
                    "provider_label": "Google Ads",
                    "message": str(exc),
                    "params": params,
                },
                "raw": {},
            }

        snapshot = {
            "meta": {
                "status": "ok",
                "cache_status": "miss",
                "provider": "google_ads",
                "provider_label": "Google Ads",
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

    def _fetch(self, params: dict, creds: dict, access_token: str) -> dict:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "developer-token": creds["developer_token"],
            "Content-Type": "application/json",
        }
        if params.get("login_customer_id"):
            headers["login-customer-id"] = params["login_customer_id"]
        url = (
            f"https://googleads.googleapis.com/{params['api_version']}/customers/"
            f"{params['customer_id']}/googleAds:search"
        )
        body = {
            "query": _search_terms_query(params),
            "pageSize": min(max(1, int(params["search_terms_limit"])), 10000),
        }
        results: list[dict] = []
        next_page_token = ""
        while True:
            if next_page_token:
                body["pageToken"] = next_page_token
            payload = self.requester(url, body, headers) if self.requester else self._request(url, body, headers)
            results.extend(payload.get("results") or [])
            next_page_token = str(payload.get("nextPageToken") or "")
            if not next_page_token or len(results) >= int(params["search_terms_limit"]):
                break
        return {"search_terms": results[: int(params["search_terms_limit"])]}

    def _request(self, url: str, body: dict, headers: dict) -> dict:
        response = requests.post(url, headers=headers, json=body, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"Google Ads API error {response.status_code}: {response.text[:500]}")
        return response.json()


def _search_terms_query(params: dict) -> str:
    min_cost_micros = int(float(params.get("min_cost") or 0.0) * 1_000_000)
    where = [
        f"segments.date BETWEEN '{params['start_date']}' AND '{params['end_date']}'",
        "metrics.cost_micros > 0",
    ]
    if min_cost_micros > 0:
        where.append(f"metrics.cost_micros >= {min_cost_micros}")
    return f"""
        SELECT
          search_term_view.search_term,
          segments.keyword.info.text,
          segments.search_term_match_type,
          campaign.id,
          campaign.name,
          ad_group.id,
          ad_group.name,
          customer.currency_code,
          metrics.cost_micros,
          metrics.clicks,
          metrics.impressions,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc
        FROM search_term_view
        WHERE {' AND '.join(where)}
        ORDER BY metrics.cost_micros DESC
        LIMIT {int(params['search_terms_limit'])}
    """


def fetch_snapshot(cache_dir: Path, config: GoogleAdsConfig) -> dict:
    return GoogleAdsClient(cache_dir=cache_dir).load_or_fetch(config)


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
    meta.setdefault("provider", "google_ads")
    meta.setdefault("provider_label", "Google Ads")
    raw_rows = raw.get("search_terms") or []
    if not raw_rows:
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
                "position_buckets": {},
                "semantic_map": {"points": [], "shown": 0},
            },
            semantic_rows=[],
            semantic_embeddings=None,
        )

    cluster_lookup = _cluster_lookup(cluster_summaries)
    keywords = _normalize_search_terms(raw_rows)
    top_pages: list[dict] = []
    directories = _aggregate_directories(top_pages, keywords)
    clusters = _aggregate_clusters(top_pages, keywords, cluster_lookup)
    semantic_points, semantic_rows, semantic_embeddings = _semantic_map(
        pages,
        embeddings,
        top_pages,
        keywords,
        extracted_pages or [],
        paragraph_records or [],
        linkbuilding or {},
        embedder=embedder,
        sample_cap=semantic_sample_cap,
    )
    total_cost = sum(_to_float(r.get("paid_cost")) for r in keywords)
    total_clicks = sum(_to_int(r.get("clicks")) for r in keywords)
    total_impressions = sum(_to_int(r.get("impressions")) for r in keywords)
    total_conversions = sum(_to_float(r.get("paid_conversions")) for r in keywords)
    total_conversion_value = sum(_to_float(r.get("paid_conversion_value")) for r in keywords)
    summary = {
        "provider": "google_ads",
        "provider_label": "Google Ads",
        "top_pages": 0,
        "organic_keywords": len(keywords),
        "paid_search_terms": len(keywords),
        "top_pages_traffic": int(round(total_cost)),
        "matched_top_pages": 0,
        "matched_traffic": 0,
        "matched_traffic_share": 0.0,
        "paid_cost": round(total_cost, 2),
        "paid_clicks": total_clicks,
        "paid_impressions": total_impressions,
        "paid_conversions": round(total_conversions, 2),
        "paid_conversion_value": round(total_conversion_value, 2),
        "avg_cpc": round(total_cost / total_clicks, 2) if total_clicks else 0.0,
        "cost_per_conversion": round(total_cost / total_conversions, 2) if total_conversions else 0.0,
        "roas": round(total_conversion_value / total_cost, 2) if total_cost else 0.0,
        "start_date": (meta.get("params") or {}).get("start_date", ""),
        "end_date": (meta.get("params") or {}).get("end_date", ""),
    }
    payload = {
        "meta": meta,
        "summary": summary,
        "metrics": {
            "org_traffic": int(round(total_cost)),
            "org_keywords": len(keywords),
            "paid_cost": round(total_cost, 2),
            "paid_clicks": total_clicks,
            "paid_impressions": total_impressions,
            "paid_conversions": round(total_conversions, 2),
            "paid_conversion_value": round(total_conversion_value, 2),
        },
        "pages_by_traffic": {},
        "top_pages": top_pages,
        "organic_keywords": keywords,
        "directories": directories,
        "clusters": clusters,
        "position_buckets": {},
        "semantic_map": {
            "points": semantic_points,
            "shown": len(semantic_points),
            "entity_types": ["keyword", "page", "page_title", "header", "paragraph", "link_title"],
        },
        "entity_alignment": _entity_alignment(semantic_rows, semantic_embeddings),
    }
    return AhrefsAnalysis(payload=payload, semantic_rows=semantic_rows, semantic_embeddings=semantic_embeddings)


def _normalize_search_terms(raw_rows: list[dict]) -> list[dict]:
    by_term: dict[str, dict] = {}
    for raw in raw_rows:
        term = _nested(raw, "searchTermView", "searchTerm")
        if not term:
            continue
        metrics = raw.get("metrics") or {}
        campaign = raw.get("campaign") or {}
        ad_group = raw.get("adGroup") or {}
        customer = raw.get("customer") or {}
        keyword_text = _nested(raw, "segments", "keyword", "info", "text")
        cost = _to_float(metrics.get("costMicros")) / 1_000_000
        clicks = _to_int(metrics.get("clicks"))
        impressions = _to_int(metrics.get("impressions"))
        conversions = _to_float(metrics.get("conversions"))
        current = by_term.setdefault(str(term), {
            "provider": "google_ads",
            "keyword": str(term),
            "url": "",
            "matched_url": "",
            "matched": False,
            "page_title": "",
            "section": "paid-search",
            "cluster": None,
            "cluster_label": "Paid search terms",
            "position": 0,
            "traffic": 0,
            "clicks": 0,
            "impressions": 0,
            "volume": 0,
            "cpc_cents": 0,
            "cpc_usd": 0.0,
            "paid_cost": 0.0,
            "paid_conversions": 0.0,
            "paid_conversion_value": 0.0,
            "currency": str(customer.get("currencyCode") or ""),
            "paid_keyword": str(keyword_text or ""),
            "campaigns": set(),
            "ad_groups": set(),
            "country": "",
            "serp_type": "paid_search_term",
            "intents": ["commercial", "transactional"],
            "serp_features": [],
            "last_update": "",
            "position_bucket": _position_bucket(0),
        })
        current["traffic"] += int(round(cost))
        current["clicks"] += clicks
        current["impressions"] += impressions
        current["volume"] += impressions
        current["paid_cost"] += cost
        current["paid_conversions"] += conversions
        current["paid_conversion_value"] += _to_float(metrics.get("conversionsValue"))
        if clicks:
            current["cpc_usd"] = current["paid_cost"] / current["clicks"]
            current["cpc_cents"] = int(round(current["cpc_usd"] * 100))
        if campaign.get("name"):
            current["campaigns"].add(str(campaign.get("name")))
        if ad_group.get("name"):
            current["ad_groups"].add(str(ad_group.get("name")))
    rows = []
    for row in by_term.values():
        row["paid_cost"] = round(float(row["paid_cost"]), 2)
        row["paid_conversions"] = round(float(row["paid_conversions"]), 2)
        row["paid_conversion_value"] = round(float(row["paid_conversion_value"]), 2)
        row["cpc_usd"] = round(float(row["cpc_usd"]), 2)
        row["campaigns"] = sorted(row["campaigns"])[:5]
        row["ad_groups"] = sorted(row["ad_groups"])[:5]
        rows.append(row)
    rows.sort(key=lambda r: (_to_float(r.get("paid_cost")), _to_int(r.get("clicks")), _to_int(r.get("impressions"))), reverse=True)
    return rows


def _nested(obj: dict, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur
