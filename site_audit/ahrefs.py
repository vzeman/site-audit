"""Optional Ahrefs API enrichment for organic traffic and keyword context.

The module is deliberately cache-first. Ahrefs API requests spend credits, so
the pipeline first reuses the latest compatible snapshot for a domain unless
the caller explicitly asks for a refresh.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse, urlunparse

import numpy as np
import requests

from .analyzer import PageInfo, section_for_url

LOG = logging.getLogger(__name__)

AHREFS_BASE_URL = "https://api.ahrefs.com/v3/site-explorer"

TOP_PAGES_SELECT = ",".join(
    [
        "url",
        "keywords",
        "sum_traffic",
        "value",
        "top_keyword",
        "top_keyword_best_position",
        "top_keyword_best_position_title",
        "top_keyword_country",
        "top_keyword_volume",
        "referring_domains",
        "ur",
        "page_type",
    ]
)

ORGANIC_KEYWORDS_SELECT = ",".join(
    [
        "keyword",
        "best_position",
        "best_position_url",
        "sum_traffic",
        "volume",
        "cpc",
        "keyword_country",
        "is_branded",
        "is_commercial",
        "is_informational",
        "is_navigational",
        "is_transactional",
        "serp_features",
        "last_update",
    ]
)

METRICS_ENDPOINT = "metrics"
PAGES_BY_TRAFFIC_ENDPOINT = "pages-by-traffic"
TOP_PAGES_ENDPOINT = "top-pages"
ORGANIC_KEYWORDS_ENDPOINT = "organic-keywords"


@dataclass
class AhrefsConfig:
    enabled: bool = True
    api_key: Optional[str] = None
    date: Optional[str] = None
    country: Optional[str] = None
    mode: str = "subdomains"
    protocol: str = "both"
    volume_mode: str = "monthly"
    top_pages_limit: int = 1000
    keywords_limit: int = 1000
    refresh: bool = False
    reuse_latest: bool = True
    semantic_sample_cap: int = 500


@dataclass
class AhrefsAnalysis:
    payload: dict
    semantic_rows: list[dict]
    semantic_embeddings: Optional[np.ndarray]


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE lines from .env without adding a dependency."""
    env_path = path or _find_dotenv()
    if env_path is None or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def ahrefs_api_key() -> str:
    load_dotenv()
    return os.environ.get("AHREFS_API_KEY") or os.environ.get("AHREFS_TOKEN") or ""


def _find_dotenv(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("://", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", value).strip("_") or "domain"


def _json_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _position_bucket(position: int | float | str | None) -> str:
    pos = _to_int(position)
    if pos <= 0:
        return "unknown"
    if pos == 1:
        return "pos_1"
    if pos <= 3:
        return "pos_2_3"
    if pos <= 10:
        return "pos_4_10"
    if pos <= 20:
        return "pos_11_20"
    if pos <= 50:
        return "pos_21_50"
    return "pos_51_plus"


def _top_position_counts(position: int | float | str | None) -> tuple[int, int, int]:
    pos = _to_int(position)
    if pos <= 0:
        return 0, 0, 0
    return (1 if pos <= 3 else 0, 1 if pos <= 10 else 0, 1 if pos <= 20 else 0)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower()
    path = unquote(parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower() or "https", host, path, "", "", ""))


def _url_keys(url: str) -> list[str]:
    norm = _normalize_url(url)
    if not norm:
        return []
    parsed = urlparse(norm)
    keys = [norm]
    host = parsed.netloc
    path = parsed.path or "/"
    if host.startswith("www."):
        keys.append(urlunparse((parsed.scheme, host[4:], path, "", "", "")))
    else:
        keys.append(urlunparse((parsed.scheme, f"www.{host}", path, "", "", "")))
    keys.append(path.rstrip("/") or "/")
    return list(dict.fromkeys(keys))


def _cache_params(domain: str, config: AhrefsConfig, request_date: str | None = None) -> dict:
    params = {
        "target": domain,
        "mode": config.mode,
        "protocol": config.protocol,
        "volume_mode": config.volume_mode,
        "country": (config.country or "").upper(),
        "top_pages_limit": int(config.top_pages_limit),
        "keywords_limit": int(config.keywords_limit),
        "top_pages_select": TOP_PAGES_SELECT,
        "organic_keywords_select": ORGANIC_KEYWORDS_SELECT,
    }
    if request_date:
        params["date"] = request_date
    return params


def _params_without_date(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "date"}


class AhrefsClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        requester: Optional[Callable[[str, dict], dict]] = None,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.requester = requester

    def load_or_fetch(self, domain: str, config: AhrefsConfig) -> dict:
        """Return a raw Ahrefs snapshot, loading cache before HTTP."""
        cache_root = self.cache_dir / "ahrefs" / _slug(domain)
        requested_date = config.date
        params_for_match = _cache_params(domain, config, requested_date)

        if not config.refresh:
            cached = self._load_cached_snapshot(cache_root, params_for_match, requested_date, config.reuse_latest)
            if cached is not None:
                meta = cached.setdefault("meta", {})
                meta["cache_status"] = "hit"
                return cached

        if not self.api_key:
            return {
                "meta": {
                    "status": "missing_api_key",
                    "cache_status": "miss",
                    "message": "Set AHREFS_API_KEY in .env or the environment to fetch Ahrefs data.",
                    "params": params_for_match,
                },
                "raw": {},
            }

        request_date = requested_date or date.today().isoformat()
        params = _cache_params(domain, config, request_date)
        try:
            raw = self._fetch_all(params)
        except Exception as exc:
            LOG.warning("Ahrefs API fetch failed for %s: %s", domain, exc)
            return {
                "meta": {
                    "status": "error",
                    "cache_status": "miss",
                    "message": str(exc),
                    "params": params,
                },
                "raw": {},
            }

        snapshot = {
            "meta": {
                "status": "ok",
                "cache_status": "miss",
                "fetched_at": date.today().isoformat(),
                "params": params,
                "params_no_date": _params_without_date(params),
            },
            "raw": raw,
        }
        path = self._cache_path(cache_root, request_date, params)
        _write_json(path, snapshot)
        return snapshot

    def _load_cached_snapshot(
        self,
        cache_root: Path,
        params: dict,
        requested_date: str | None,
        reuse_latest: bool,
    ) -> dict | None:
        if requested_date:
            exact = self._cache_path(cache_root, requested_date, params)
            data = _load_json(exact)
            if data:
                return data
            return None
        if not reuse_latest:
            return None

        wanted = _params_without_date(params)
        candidates: list[tuple[float, dict]] = []
        for path in cache_root.glob("snapshot_*.json"):
            data = _load_json(path)
            if not data:
                continue
            meta = data.get("meta", {}) or {}
            have = meta.get("params_no_date") or _params_without_date(meta.get("params", {}) or {})
            if have == wanted:
                candidates.append((path.stat().st_mtime, data))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _cache_path(self, cache_root: Path, request_date: str, params: dict) -> Path:
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"snapshot_{request_date}_{_json_key(params)}.json"

    def _fetch_all(self, params: dict) -> dict:
        common = {
            "target": params["target"],
            "mode": params["mode"],
            "protocol": params["protocol"],
            "volume_mode": params["volume_mode"],
            "output": "json",
        }
        if params.get("country"):
            common["country"] = params["country"]

        dated = {**common, "date": params["date"]}
        return {
            "metrics": self._request(METRICS_ENDPOINT, dated),
            "pages_by_traffic": self._request(PAGES_BY_TRAFFIC_ENDPOINT, common),
            "top_pages": self._request(
                TOP_PAGES_ENDPOINT,
                {
                    **dated,
                    "limit": params["top_pages_limit"],
                    "select": TOP_PAGES_SELECT,
                    "order_by": "sum_traffic:desc",
                },
            ),
            "organic_keywords": self._request(
                ORGANIC_KEYWORDS_ENDPOINT,
                {
                    **dated,
                    "limit": params["keywords_limit"],
                    "select": ORGANIC_KEYWORDS_SELECT,
                    "order_by": "sum_traffic:desc",
                },
            ),
        }

    def _request(self, endpoint: str, params: dict) -> dict:
        if self.requester is not None:
            return self.requester(endpoint, params)
        url = f"{AHREFS_BASE_URL}/{endpoint}"
        resp = requests.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(f"{endpoint} returned HTTP {resp.status_code}: {detail}")
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"{endpoint}: {data['error']}")
        return data


def fetch_snapshot(domain: str, cache_dir: Path, config: AhrefsConfig) -> dict:
    if not config.enabled:
        return {
            "meta": {"status": "disabled", "cache_status": "disabled", "params": _cache_params(domain, config)},
            "raw": {},
        }
    api_key = config.api_key or ahrefs_api_key()
    return AhrefsClient(api_key=api_key, cache_dir=cache_dir).load_or_fetch(domain, config)


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
    page_traffic_by_index: dict[int, int] = {}

    top_pages = _normalize_top_pages(
        raw.get("top_pages", {}).get("pages") or [],
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
        coords,
        page_traffic_by_index,
    )
    organic_keywords = _normalize_keywords(
        raw.get("organic_keywords", {}).get("keywords") or [],
        pages,
        page_lookup,
        cluster_labels,
        cluster_lookup,
    )
    directories = _aggregate_directories(top_pages, organic_keywords)
    clusters = _aggregate_clusters(top_pages, organic_keywords, cluster_lookup)
    metrics = raw.get("metrics", {}).get("metrics") or {}
    traffic_buckets = raw.get("pages_by_traffic", {}).get("pages") or {}

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

    total_top_traffic = sum(int(r.get("traffic", 0)) for r in top_pages)
    matched_traffic = sum(int(r.get("traffic", 0)) for r in top_pages if r.get("matched"))
    top3_keywords = _to_int(metrics.get("org_keywords_1_3")) or sum(
        1 for r in organic_keywords if 0 < int(r.get("position", 0) or 0) <= 3
    )
    top10_keywords = (
        _to_int(metrics.get("org_keywords_1_10"))
        or top3_keywords
        + sum(1 for r in organic_keywords if 4 <= int(r.get("position", 0) or 0) <= 10)
    )
    summary = {
        "provider": "ahrefs",
        "provider_label": "Ahrefs",
        "top_pages": len(top_pages),
        "organic_keywords": len(organic_keywords),
        "top_pages_traffic": total_top_traffic,
        "matched_top_pages": sum(1 for r in top_pages if r.get("matched")),
        "matched_traffic": matched_traffic,
        "matched_traffic_share": round(matched_traffic / total_top_traffic, 4) if total_top_traffic else 0.0,
        "top_pages_value_usd": round(sum(int(r.get("value_cents", 0)) for r in top_pages) / 100, 2),
        "top3_keywords": top3_keywords,
        "top10_keywords": top10_keywords,
        "directories": len(directories),
        "traffic_clusters": len(clusters),
    }
    meta.setdefault("provider", "ahrefs")
    meta.setdefault("provider_label", "Ahrefs")

    payload = {
        "meta": meta,
        "summary": summary,
        "metrics": metrics,
        "pages_by_traffic": traffic_buckets,
        "top_pages": top_pages,
        "organic_keywords": organic_keywords,
        "directories": directories,
        "clusters": clusters,
        "semantic_map": {
            "points": semantic_points,
            "shown": len(semantic_points),
            "entity_types": ["page", "page_title", "keyword", "header", "paragraph", "link_title"],
        },
    }
    return AhrefsAnalysis(payload=payload, semantic_rows=semantic_rows, semantic_embeddings=semantic_embeddings)


def _page_lookup(pages: list[PageInfo]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, page in enumerate(pages):
        for key in _url_keys(page.url):
            lookup.setdefault(key, i)
    return lookup


def _match_page(url: str, lookup: dict[str, int]) -> int | None:
    for key in _url_keys(url):
        if key in lookup:
            return lookup[key]
    return None


def _cluster_lookup(cluster_summaries) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for summary in cluster_summaries or []:
        cid = int(getattr(summary, "cluster_id", -1))
        keywords = getattr(summary, "keywords", []) or []
        label = ", ".join(k.get("keyword", "") for k in keywords[:4] if k.get("keyword")) or f"cluster {cid}"
        out[cid] = {
            "cluster": cid,
            "label": label,
            "page_count": int(getattr(summary, "page_count", 0)),
            "cohesion": round(float(getattr(summary, "cohesion", 0.0)), 4),
            "site_alignment": round(float(getattr(summary, "site_alignment", 0.0)), 4),
        }
    return out


def _page_cluster(
    index: int | None,
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
) -> tuple[int | None, str]:
    if index is None or cluster_labels is None or index >= len(cluster_labels):
        return None, ""
    cid = int(cluster_labels[index])
    return cid, (cluster_lookup.get(cid) or {}).get("label", f"cluster {cid}")


def _normalize_top_pages(
    raw_pages: list[dict],
    pages: list[PageInfo],
    page_lookup: dict[str, int],
    cluster_labels: Optional[np.ndarray],
    cluster_lookup: dict[int, dict],
    coords: Optional[np.ndarray],
    page_traffic_by_index: dict[int, int],
) -> list[dict]:
    rows: list[dict] = []
    for raw in raw_pages:
        url = raw.get("url") or raw.get("raw_url") or ""
        page_index = _match_page(url, page_lookup)
        page = pages[page_index] if page_index is not None else None
        traffic = _to_int(raw.get("sum_traffic", raw.get("traffic", raw.get("sum_traffic_merged"))))
        if page_index is not None:
            page_traffic_by_index[page_index] = max(page_traffic_by_index.get(page_index, 0), traffic)
        cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
        row = {
            "provider": "ahrefs",
            "url": url,
            "matched_url": page.url if page else "",
            "matched": page is not None,
            "title": page.title if page else "",
            "section": page.section if page else section_for_url(url),
            "cluster": cluster_id,
            "cluster_label": cluster_label,
            "traffic": traffic,
            "keywords": _to_int(raw.get("keywords", raw.get("keywords_merged"))),
            "value_cents": _to_int(raw.get("value", raw.get("value_merged"))),
            "value_usd": round(_to_int(raw.get("value", raw.get("value_merged"))) / 100, 2),
            "top_keyword": raw.get("top_keyword") or "",
            "top_keyword_position": _to_int(raw.get("top_keyword_best_position")),
            "top_keyword_title": raw.get("top_keyword_best_position_title") or "",
            "top_keyword_country": raw.get("top_keyword_country") or "",
            "top_keyword_volume": _to_int(raw.get("top_keyword_volume")),
            "referring_domains": _to_int(raw.get("referring_domains")),
            "url_rating": round(_to_float(raw.get("ur")), 2),
            "page_type": raw.get("page_type") or "",
        }
        top3, top10, top20 = _top_position_counts(row["top_keyword_position"])
        row["top3_keywords"] = top3
        row["top10_keywords"] = top10
        row["top20_keywords"] = top20
        row["position_buckets"] = (
            {_position_bucket(row["top_keyword_position"]): 1}
            if row["top_keyword_position"]
            else {}
        )
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
        keyword = raw.get("keyword") or raw.get("keyword_merged") or ""
        if not keyword:
            continue
        url = raw.get("best_position_url") or raw.get("best_position_url_raw") or ""
        page_index = _match_page(url, page_lookup)
        page = pages[page_index] if page_index is not None else None
        cluster_id, cluster_label = _page_cluster(page_index, cluster_labels, cluster_lookup)
        intents = [
            name
            for name, field in [
                ("branded", "is_branded"),
                ("commercial", "is_commercial"),
                ("informational", "is_informational"),
                ("navigational", "is_navigational"),
                ("transactional", "is_transactional"),
            ]
            if raw.get(field)
        ]
        rows.append({
            "provider": "ahrefs",
            "keyword": keyword,
            "url": url,
            "matched_url": page.url if page else "",
            "matched": page is not None,
            "page_title": page.title if page else "",
            "section": page.section if page else section_for_url(url),
            "cluster": cluster_id,
            "cluster_label": cluster_label,
            "position": _to_int(raw.get("best_position")),
            "traffic": _to_int(raw.get("sum_traffic", raw.get("sum_traffic_merged"))),
            "volume": _to_int(raw.get("volume", raw.get("volume_merged"))),
            "cpc_cents": _to_int(raw.get("cpc", raw.get("cpc_merged"))),
            "cpc_usd": round(_to_int(raw.get("cpc", raw.get("cpc_merged"))) / 100, 2),
            "country": raw.get("keyword_country") or "",
            "serp_type": "organic",
            "intents": intents,
            "serp_features": raw.get("serp_features") or [],
            "last_update": raw.get("last_update") or "",
        })
    rows.sort(key=lambda r: r["traffic"], reverse=True)
    return rows


def _empty_group(key: str, label: str = "") -> dict:
    return {
        "key": key,
        "label": label or key,
        "traffic": 0,
        "keyword_traffic": 0,
        "paid_traffic": 0,
        "featured_snippet_traffic": 0,
        "local_pack_traffic": 0,
        "ai_overview_traffic": 0,
        "value_cents": 0,
        "value_usd": 0.0,
        "pages": set(),
        "matched_pages": 0,
        "keyword_rows": 0,
        "keywords_total": 0,
        "top3_keywords": 0,
        "top10_keywords": 0,
        "top20_keywords": 0,
        "position_buckets": Counter(),
        "serp_types": Counter(),
        "serp_features": Counter(),
        "intents": Counter(),
        "top_keywords": Counter(),
        "top_pages": [],
    }


def _add_page_to_group(group: dict, row: dict) -> None:
    group["traffic"] += int(row.get("traffic", 0))
    group["paid_traffic"] += int(row.get("paid_traffic", 0) or 0)
    group["featured_snippet_traffic"] += int(row.get("featured_snippet_traffic", 0) or 0)
    group["local_pack_traffic"] += int(row.get("local_pack_traffic", 0) or 0)
    group["ai_overview_traffic"] += int(row.get("ai_overview_traffic", 0) or 0)
    group["value_cents"] += int(row.get("value_cents", 0))
    if row.get("matched_url") or row.get("url"):
        group["pages"].add(row.get("matched_url") or row.get("url"))
    if row.get("matched"):
        group["matched_pages"] += 1
    group["keywords_total"] += int(row.get("keywords", 0))
    group["top3_keywords"] += int(row.get("top3_keywords", 0) or 0)
    group["top10_keywords"] += int(row.get("top10_keywords", 0) or 0)
    group["top20_keywords"] += int(row.get("top20_keywords", 0) or 0)
    for bucket, count in (row.get("position_buckets") or {}).items():
        group["position_buckets"][bucket] += int(count or 0)
    if row.get("top_keyword"):
        group["top_keywords"][row["top_keyword"]] += max(1, int(row.get("traffic", 0)))
    group["top_pages"].append({
        "url": row.get("matched_url") or row.get("url"),
        "title": row.get("title") or row.get("top_keyword_title") or row.get("url"),
        "traffic": int(row.get("traffic", 0)),
        "paid_traffic": int(row.get("paid_traffic", 0) or 0),
        "keywords": int(row.get("keywords", 0) or 0),
        "top_keyword": row.get("top_keyword", ""),
    })


def _add_keyword_to_group(group: dict, row: dict) -> None:
    group["keyword_rows"] += 1
    group["keyword_traffic"] += int(row.get("traffic", 0) or 0)
    top3, top10, top20 = _top_position_counts(row.get("position"))
    group["top3_keywords"] += top3
    group["top10_keywords"] += top10
    group["top20_keywords"] += top20
    group["position_buckets"][_position_bucket(row.get("position"))] += 1
    if row.get("serp_type"):
        group["serp_types"][str(row["serp_type"])] += 1
    for feature in row.get("serp_features") or []:
        if feature:
            group["serp_features"][str(feature)] += 1
    for intent in row.get("intents") or []:
        if intent:
            group["intents"][str(intent)] += 1
    group["top_keywords"][row["keyword"]] += max(1, int(row.get("traffic", 0)))


def _finalize_groups(groups: dict[Any, dict]) -> list[dict]:
    rows: list[dict] = []
    for group in groups.values():
        top_pages = sorted(group["top_pages"], key=lambda r: r["traffic"], reverse=True)[:10]
        rows.append({
            "key": group["key"],
            "label": group["label"],
            "traffic": int(group["traffic"]),
            "keyword_traffic": int(group["keyword_traffic"]),
            "paid_traffic": int(group["paid_traffic"]),
            "featured_snippet_traffic": int(group["featured_snippet_traffic"]),
            "local_pack_traffic": int(group["local_pack_traffic"]),
            "ai_overview_traffic": int(group["ai_overview_traffic"]),
            "value_cents": int(group["value_cents"]),
            "value_usd": round(int(group["value_cents"]) / 100, 2),
            "pages": len(group["pages"]),
            "matched_pages": int(group["matched_pages"]),
            "keywords_total": int(group["keywords_total"]),
            "keyword_rows": int(group["keyword_rows"]),
            "top3_keywords": int(group["top3_keywords"]),
            "top10_keywords": int(group["top10_keywords"]),
            "top20_keywords": int(group["top20_keywords"]),
            "position_buckets": dict(group["position_buckets"]),
            "serp_types": [
                {"type": k, "count": int(v)}
                for k, v in group["serp_types"].most_common(10)
            ],
            "serp_features": [
                {"feature": k, "count": int(v)}
                for k, v in group["serp_features"].most_common(12)
            ],
            "intents": [
                {"intent": k, "count": int(v)}
                for k, v in group["intents"].most_common(8)
            ],
            "top_keywords": [
                {"keyword": k, "traffic": int(v)}
                for k, v in group["top_keywords"].most_common(12)
            ],
            "top_pages": top_pages,
        })
    rows.sort(key=lambda r: r["traffic"], reverse=True)
    return rows


def _aggregate_directories(top_pages: list[dict], keywords: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in top_pages:
        key = row.get("section") or "root"
        groups.setdefault(key, _empty_group(key, key))
        _add_page_to_group(groups[key], row)
    for row in keywords:
        key = row.get("section") or "root"
        groups.setdefault(key, _empty_group(key, key))
        _add_keyword_to_group(groups[key], row)
    return _finalize_groups(groups)


def _aggregate_clusters(
    top_pages: list[dict],
    keywords: list[dict],
    cluster_lookup: dict[int, dict],
) -> list[dict]:
    groups: dict[int, dict] = {}
    for row in top_pages:
        cid = row.get("cluster")
        if cid is None:
            continue
        info = cluster_lookup.get(int(cid), {})
        groups.setdefault(int(cid), _empty_group(str(cid), info.get("label", f"cluster {cid}")))
        groups[int(cid)].update({
            "cluster": int(cid),
            "page_count": info.get("page_count", 0),
            "cohesion": info.get("cohesion", 0.0),
            "site_alignment": info.get("site_alignment", 0.0),
        })
        _add_page_to_group(groups[int(cid)], row)
    for row in keywords:
        cid = row.get("cluster")
        if cid is None:
            continue
        info = cluster_lookup.get(int(cid), {})
        groups.setdefault(int(cid), _empty_group(str(cid), info.get("label", f"cluster {cid}")))
        groups[int(cid)].update({
            "cluster": int(cid),
            "page_count": info.get("page_count", 0),
            "cohesion": info.get("cohesion", 0.0),
            "site_alignment": info.get("site_alignment", 0.0),
        })
        _add_keyword_to_group(groups[int(cid)], row)
    rows = _finalize_groups(groups)
    for row in rows:
        row["cluster"] = int(row["key"])
        source = groups.get(row["cluster"], {})
        row["page_count"] = int(source.get("page_count", 0))
        row["cohesion"] = float(source.get("cohesion", 0.0))
        row["site_alignment"] = float(source.get("site_alignment", 0.0))
    return rows


def _semantic_map(
    pages: list[PageInfo],
    embeddings: np.ndarray,
    top_pages: list[dict],
    organic_keywords: list[dict],
    extracted_pages: list,
    paragraph_records: list,
    linkbuilding: dict,
    *,
    embedder=None,
    sample_cap: int = 500,
) -> tuple[list[dict], list[dict], Optional[np.ndarray]]:
    if embedder is None or embeddings.size == 0:
        return [], [], None

    page_index_by_url = {_normalize_url(p.url): i for i, p in enumerate(pages)}
    traffic_by_index: dict[int, int] = {}
    top_page_indices: list[int] = []
    for row in top_pages:
        url = row.get("matched_url") or row.get("url") or ""
        idx = page_index_by_url.get(_normalize_url(url))
        if idx is None:
            continue
        if idx not in traffic_by_index:
            top_page_indices.append(idx)
        traffic_by_index[idx] = max(traffic_by_index.get(idx, 0), int(row.get("traffic", 0)))

    top_page_indices = sorted(top_page_indices, key=lambda i: traffic_by_index.get(i, 0), reverse=True)
    rows: list[dict] = []
    vectors: list[np.ndarray] = []

    def add_vector(row: dict, vector: np.ndarray) -> None:
        if vector.size == 0:
            return
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        rows.append(row)
        vectors.append(vec)

    for idx in top_page_indices[:sample_cap]:
        page = pages[idx]
        add_vector(
            {
                "type": "page",
                "label": page.title or page.url,
                "url": page.url,
                "section": page.section,
                "cluster": _cluster_for_top_page(top_pages, page.url),
                "traffic": traffic_by_index.get(idx, 0),
                "size": traffic_by_index.get(idx, 0),
            },
            embeddings[idx],
        )

    title_texts: list[str] = []
    title_rows: list[dict] = []
    for idx in top_page_indices[:sample_cap]:
        page = pages[idx]
        if not page.title:
            continue
        title_texts.append(page.title)
        title_rows.append({
            "type": "page_title",
            "label": page.title,
            "url": page.url,
            "section": page.section,
            "cluster": _cluster_for_top_page(top_pages, page.url),
            "traffic": traffic_by_index.get(idx, 0),
            "size": traffic_by_index.get(idx, 0),
        })
    _encode_and_add(title_texts, title_rows, embedder, add_vector)

    keyword_rows = []
    keyword_texts = []
    for row in organic_keywords[:sample_cap]:
        keyword_texts.append(row["keyword"])
        keyword_rows.append({
            "type": "keyword",
            "label": row["keyword"],
            "url": row.get("matched_url") or row.get("url") or "",
            "section": row.get("section", ""),
            "cluster": row.get("cluster"),
            "traffic": int(row.get("traffic", 0)),
            "volume": int(row.get("volume", 0)),
            "position": int(row.get("position", 0)),
            "size": max(int(row.get("traffic", 0)), int(row.get("volume", 0))),
        })
    _encode_and_add(keyword_texts, keyword_rows, embedder, add_vector)

    preferred_pages = set(top_page_indices[:sample_cap])
    header_texts: list[str] = []
    header_rows: list[dict] = []
    for idx in top_page_indices:
        if len(header_texts) >= sample_cap or idx >= len(extracted_pages):
            break
        page = pages[idx]
        for header in getattr(extracted_pages[idx], "headers_rich", []) or []:
            text = (header.get("text") or "").strip()
            if not text:
                continue
            header_texts.append(text)
            header_rows.append({
                "type": "header",
                "label": text,
                "url": page.url,
                "section": page.section,
                "level": int(header.get("level", 0) or 0),
                "cluster": _cluster_for_top_page(top_pages, page.url),
                "traffic": traffic_by_index.get(idx, 0),
                "size": traffic_by_index.get(idx, 0),
            })
            if len(header_texts) >= sample_cap:
                break
    _encode_and_add(header_texts, header_rows, embedder, add_vector)

    para_added = 0
    for page_i, para_i, text, emb in paragraph_records:
        if para_added >= sample_cap:
            break
        if preferred_pages and page_i not in preferred_pages:
            continue
        page = pages[page_i]
        add_vector(
            {
                "type": "paragraph",
                "label": text[:180],
                "url": page.url,
                "section": page.section,
                "paragraph_index": int(para_i),
                "cluster": _cluster_for_top_page(top_pages, page.url),
                "traffic": traffic_by_index.get(page_i, 0),
                "size": traffic_by_index.get(page_i, 0),
            },
            emb,
        )
        para_added += 1

    anchor_counts: Counter = Counter()
    for bucket in ("top_internal_anchors", "top_external_anchors", "top_generic_anchors"):
        for row in (linkbuilding or {}).get(bucket, []) or []:
            anchor = (row.get("anchor") or "").strip()
            if anchor:
                anchor_counts[anchor] += int(row.get("count", 0) or 0)
    anchor_texts = []
    anchor_rows = []
    for anchor, count in anchor_counts.most_common(max(50, sample_cap // 2)):
        anchor_texts.append(anchor)
        anchor_rows.append({
            "type": "link_title",
            "label": anchor,
            "url": "",
            "section": "",
            "traffic": 0,
            "count": int(count),
            "size": int(count),
        })
    _encode_and_add(anchor_texts, anchor_rows, embedder, add_vector)

    if not vectors:
        return [], [], None
    matrix = np.stack(vectors).astype(np.float32)
    coords = _project_vectors(matrix)
    points = []
    for row, xy in zip(rows, coords):
        points.append({**row, "x": float(xy[0]), "y": float(xy[1])})
    return points, rows, matrix


def _encode_and_add(texts: list[str], rows: list[dict], embedder, add_vector: Callable[[dict, np.ndarray], None]) -> None:
    if not texts:
        return
    embs = embedder.encode(texts, batch_size=128, show_progress=False).astype(np.float32)
    for row, emb in zip(rows, embs):
        add_vector(row, emb)


def _project_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 5:
        coords = np.zeros((len(vectors), 2), dtype=np.float32)
        for i in range(len(vectors)):
            coords[i, 0] = float(i)
        return coords
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(15, len(vectors) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors.astype(np.float32)).astype(np.float32)


def _cluster_for_top_page(top_pages: list[dict], url: str) -> int | None:
    norm = _normalize_url(url)
    for row in top_pages:
        if _normalize_url(row.get("matched_url") or row.get("url") or "") == norm:
            return row.get("cluster")
    return None


def semantic_cache_paths(cache_dir: Path, model_name: str) -> tuple[Path, Path]:
    slug = model_name.replace("/", "_").replace("-", "_")
    root = Path(cache_dir) / "ahrefs"
    return root / f"semantic_entities_{slug}.npz", root / f"semantic_entities_{slug}.meta.json"


def write_semantic_cache(cache_dir: Path, model_name: str, rows: list[dict], embeddings: Optional[np.ndarray]) -> None:
    if embeddings is None or not len(rows):
        return
    npz_path, meta_path = semantic_cache_paths(cache_dir, model_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, embeddings=embeddings.astype(np.float32))
    _write_json(meta_path, rows)


def load_semantic_cache(project_dir: Path, model_name: str) -> tuple[list[dict], Optional[np.ndarray]]:
    npz_path, meta_path = semantic_cache_paths(project_dir / "cache", model_name)
    rows = _load_json(meta_path) or []
    if not isinstance(rows, list) or not npz_path.is_file():
        return [], None
    try:
        data = np.load(npz_path, allow_pickle=False)
        embs = data["embeddings"].astype(np.float32)
    except Exception as exc:
        LOG.warning("  Ahrefs semantic cache unreadable for %s: %s", project_dir, exc)
        return [], None
    if len(rows) != len(embs):
        LOG.warning("  Ahrefs semantic cache row/vector mismatch for %s", project_dir)
        return [], None
    return rows, embs
