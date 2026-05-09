"""Local correlation model explaining page search performance.

This is intentionally conservative: it trains a small ridge regression on
observable page features and organic traffic labels. The output is framed as
correlation/estimate evidence, not as direct ranking-factor proof.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np


MIN_MODEL_PAGES = 8
MIN_POSITIVE_LABELS = 3


@dataclass(frozen=True)
class FeatureDef:
    key: str
    label: str
    group: str
    definition: str


FEATURES: tuple[FeatureDef, ...] = (
    FeatureDef("content_word_count_log", "Word count", "content", "log1p(body word count)"),
    FeatureDef("content_title_chars", "Title length", "content", "HTML title character count"),
    FeatureDef("content_description_chars", "Description length", "content", "Meta description character count"),
    FeatureDef("content_heading_count", "Heading count", "content", "Extracted H1-H6 heading count"),
    FeatureDef("content_paragraph_count", "Paragraph count", "content", "Extracted paragraph block count"),
    FeatureDef("content_avg_paragraph_words", "Average paragraph length", "content", "Average words per extracted paragraph"),
    FeatureDef("content_table_count", "Table count", "content", "HTML table count"),
    FeatureDef("content_list_count", "List count", "content", "Ordered/unordered list count"),
    FeatureDef("content_stat_count", "Proof/stat count", "content", "Numbers with units, percentages, counts, or money"),
    FeatureDef("content_external_links", "External links", "content", "External links found in the extracted page"),
    FeatureDef("schema_type_count", "Schema type count", "schema", "Distinct JSON-LD schema types"),
    FeatureDef("schema_valid_blocks", "Valid schema blocks", "schema", "Valid structured-data blocks for the page"),
    FeatureDef("schema_invalid_blocks", "Invalid schema blocks", "schema", "Invalid structured-data blocks for the page"),
    FeatureDef("freshness_score", "Freshness score", "freshness", "Bucket score: fresh=1, aging=.75, stale=.35, very stale=.1, missing=0"),
    FeatureDef("freshness_age_log", "Content age", "freshness", "log1p(age in days) when detected"),
    FeatureDef("links_in_degree_log", "Inbound internal links", "links", "log1p(internal in-degree)"),
    FeatureDef("links_out_degree_log", "Outbound internal links", "links", "log1p(internal out-degree)"),
    FeatureDef("links_click_depth_inverse", "Click-depth closeness", "links", "1/(1+click depth), higher means closer to the root"),
    FeatureDef("links_pagerank", "PageRank", "links", "Internal PageRank from the crawl link graph"),
    FeatureDef("links_authority_gap", "Authority-demand gap", "links", "Traffic-weighted PageRank mismatch score"),
    FeatureDef("entities_count_log", "Entity count", "entities", "log1p(distinct extracted entities)"),
    FeatureDef("entities_depth_score", "Entity depth score", "entities", "Per-page topical-depth entity score"),
    FeatureDef("entities_coverage", "Entity coverage", "entities", "Expected cluster-entity coverage for the page"),
    FeatureDef("information_gain_score", "Information gain", "content", "Originality/information-gain score"),
    FeatureDef("answerability_score", "Answerability", "geo", "GEO answerability score"),
    FeatureDef("conversion_cta_count", "CTA count", "conversion", "Detected CTA count"),
    FeatureDef("conversion_primary_cta_count", "Primary CTA count", "conversion", "Detected primary CTA count"),
    FeatureDef("conversion_form_count", "Form count", "conversion", "Detected form count"),
    FeatureDef("conversion_support", "Conversion support", "conversion", "Conversion support score when available"),
    FeatureDef("performance_html_weight_log", "HTML weight", "performance", "log1p(crawled HTML bytes)"),
    FeatureDef("performance_estimated_weight_log", "Estimated page weight", "performance", "log1p(HTML plus heuristic resource weights)"),
    FeatureDef("performance_resource_tags_log", "Resource tags", "performance", "log1p(images, scripts, stylesheets, fonts, preloads)"),
    FeatureDef("performance_render_blocking", "Render-blocking resources", "performance", "Render-blocking CSS/script count"),
    FeatureDef("performance_image_count_log", "Image count", "performance", "log1p(image count)"),
    FeatureDef("metadata_issue_count", "Metadata issues", "onpage", "Count of metadata-quality issues"),
    FeatureDef("media_issue_count", "Media accessibility issues", "technical", "Count of media accessibility issues"),
    FeatureDef("paragraph_link_density", "Paragraph link density", "links", "Average links per 100 words across page paragraphs"),
    FeatureDef("search_keywords_log", "Ranking keywords", "search", "log1p(search-provider keyword count for the URL)"),
    FeatureDef("search_position_score", "Best position score", "search", "Scaled top keyword position, higher is better"),
    FeatureDef("search_refdomains_log", "Referring domains", "search", "log1p(search-provider referring domains when available)"),
    FeatureDef("search_url_rating", "URL rating", "search", "Search-provider URL rating when available"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


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
    path = parts.path or "/"
    path_trimmed = path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    netlocs = {netloc, netloc[4:] if netloc.startswith("www.") else f"www.{netloc}"}
    schemes = {(parts.scheme or "https").lower(), "https", "http"}
    for scheme in schemes:
        for host in netlocs:
            for candidate_path in {path, path_trimmed}:
                normalized = urlunsplit((scheme, host, candidate_path, "", ""))
                keys.add(normalized)
                keys.add(normalized.rstrip("/"))
    return {key for key in keys if key}


def _store_url(out: dict[str, dict], url: object, row: dict, score_key: str = "") -> None:
    if not url:
        return
    for key in _url_keys(url):
        current = out.get(key)
        if current is None or not score_key or _safe_float(row.get(score_key)) >= _safe_float(current.get(score_key)):
            out[key] = row


def _lookup_url(lookup: dict[str, dict], *urls: object) -> dict:
    best: dict = {}
    for url in urls:
        for key in _url_keys(url):
            row = lookup.get(key)
            if row and len(row) >= len(best):
                best = row
    return best


def _row_lookup(rows: Iterable[dict], keys: tuple[str, ...] = ("url",), score_key: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        for key in keys:
            if row.get(key):
                _store_url(out, row.get(key), row, score_key=score_key)
    return out


def _search_lookup(search_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url")
        if not url:
            continue
        payload = {
            "traffic": _safe_float(row.get("traffic")),
            "keywords": _safe_int(row.get("keywords") or row.get("keywords_total")),
            "top_keyword": row.get("top_keyword") or "",
            "position": _safe_float(row.get("top_keyword_position") or row.get("position")),
            "referring_domains": _safe_int(row.get("referring_domains")),
            "url_rating": _safe_float(row.get("url_rating")),
            "traffic_value": _safe_float(row.get("value") or row.get("traffic_value")),
        }
        _store_url(out, url, payload, score_key="traffic")

    keyword_aggs: dict[str, dict] = defaultdict(lambda: {
        "keyword_rows": 0,
        "keyword_traffic": 0.0,
        "best_position": 999.0,
        "top3_keywords": 0,
        "top_keyword": "",
    })
    for row in (search_payload or {}).get("organic_keywords") or []:
        url = row.get("matched_url") or row.get("url")
        if not url:
            continue
        for key in _url_keys(url):
            agg = keyword_aggs[key]
            agg["keyword_rows"] += 1
            traffic = _safe_float(row.get("traffic"))
            agg["keyword_traffic"] += traffic
            position = _safe_float(row.get("position"), 999.0)
            if 0 < position < agg["best_position"]:
                agg["best_position"] = position
                agg["top_keyword"] = row.get("keyword") or agg["top_keyword"]
            if 0 < position <= 3:
                agg["top3_keywords"] += 1
    for key, agg in keyword_aggs.items():
        current = out.get(key, {})
        merged = {
            **current,
            "traffic": max(_safe_float(current.get("traffic")), _safe_float(agg.get("keyword_traffic"))),
            "keywords": max(_safe_int(current.get("keywords")), _safe_int(agg.get("keyword_rows"))),
            "position": min(
                _safe_float(current.get("position"), 999.0) or 999.0,
                _safe_float(agg.get("best_position"), 999.0),
            ),
            "top3_keywords": _safe_int(agg.get("top3_keywords")),
            "top_keyword": current.get("top_keyword") or agg.get("top_keyword") or "",
        }
        out[key] = merged
    return out


def _link_lookup(linkgraph: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(linkgraph, dict):
        return out
    for source in ("page_link_counts", "top_authority_pages", "top_hits_authorities", "top_hits_hubs"):
        for row in linkgraph.get(source) or []:
            _store_url(out, row.get("url"), row, score_key="pagerank")
    for row in ((linkgraph.get("traffic_weighted_pagerank") or {}).get("pages") or []):
        current = _lookup_url(out, row.get("url"))
        merged = {**current, **row}
        _store_url(out, row.get("url"), merged, score_key="traffic")
    return out


def _feature_value(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _safe_float(value)


def _position_score(position: float) -> float:
    if position <= 0 or position >= 999:
        return 0.0
    return max(0.0, min(1.0, (21.0 - position) / 20.0))


def _freshness_score(bucket: object) -> float:
    return {
        "fresh": 1.0,
        "aging": 0.75,
        "stale": 0.35,
        "very_stale": 0.10,
        "future": 0.55,
        "unknown": 0.0,
    }.get(str(bucket or "unknown"), 0.0)


def _metadata_issue_count(row: dict) -> int:
    issues = row.get("issues") or []
    if isinstance(issues, list):
        return len(issues)
    if isinstance(issues, str):
        return len([p for p in issues.split("|") if p.strip()])
    return 0


def _media_issue_count(row: dict) -> int:
    issues = row.get("issues") or []
    if isinstance(issues, list):
        return len(issues)
    return sum(_safe_int(row.get(key)) for key in ("images_missing_alt", "linked_images_empty_alt", "videos_missing_captions", "iframes_missing_title"))


def _build_rows(
    pages: list,
    extracted_pages: list | None,
    *,
    search_payload: dict | None,
    linkgraph: dict | None,
    structured_data: dict | None,
    freshness: dict | None,
    entities: dict | None,
    entity_coverage: dict | None,
    information_gain: dict | None,
    answerability: list | dict | None,
    conversion: dict | None,
    conversion_balance: dict | None,
    performance: dict | None,
    metadata_quality: dict | None,
    media_accessibility: dict | None,
    paragraph_density: dict | None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    search = _search_lookup(search_payload)
    links = _link_lookup(linkgraph)
    structured = _row_lookup((structured_data or {}).get("per_page") or [], ("url",))
    fresh = _row_lookup((freshness or {}).get("per_page") or [], ("url",))
    entity_rows = _row_lookup((entities or {}).get("per_page") or [], ("url",))
    entity_cov = _row_lookup((entity_coverage or {}).get("pages") or [], ("url",))
    info_gain = _row_lookup((information_gain or {}).get("pages") or [], ("url",))
    answer = _row_lookup(answerability if isinstance(answerability, list) else [], ("url",))
    conv = _row_lookup((conversion or {}).get("per_page") or [], ("url",))
    conv_balance = _row_lookup((conversion_balance or {}).get("pages") or [], ("url",))
    perf = _row_lookup((performance or {}).get("per_page") or [], ("url",))
    meta = _row_lookup((metadata_quality or {}).get("per_page") or [], ("url",))
    media = _row_lookup((media_accessibility or {}).get("per_page") or [], ("url",))
    density = _row_lookup((paragraph_density or {}).get("per_page") or [], ("url",))

    rows: list[dict] = []
    x_rows: list[list[float]] = []
    labels: list[float] = []
    extracted_pages = extracted_pages or []

    for i, page in enumerate(pages):
        url = getattr(page, "url", "") or ""
        ext = extracted_pages[i] if i < len(extracted_pages) else None
        search_row = _lookup_url(search, url)
        link = _lookup_url(links, url)
        structured_row = _lookup_url(structured, url)
        freshness_row = _lookup_url(fresh, url)
        entity_row = _lookup_url(entity_rows, url)
        entity_cov_row = _lookup_url(entity_cov, url)
        info_row = _lookup_url(info_gain, url)
        answer_row = _lookup_url(answer, url)
        conv_row = _lookup_url(conv, url)
        conv_balance_row = _lookup_url(conv_balance, url)
        perf_row = _lookup_url(perf, url)
        meta_row = _lookup_url(meta, url)
        media_row = _lookup_url(media, url)
        density_row = _lookup_url(density, url)

        paragraphs = list(getattr(ext, "paragraphs", []) or [])
        paragraph_words = [len(str(p).split()) for p in paragraphs if str(p).strip()]
        headers = list(getattr(ext, "headers_rich", []) or [])
        if not headers:
            headers = [{"text": h} for h in (getattr(ext, "headings", []) or [])]
        schema_types = structured_row.get("types") or getattr(ext, "schema_types", []) or []
        if isinstance(schema_types, str):
            schema_types = [schema_types]
        traffic = _safe_float(search_row.get("traffic"))
        position = _safe_float(search_row.get("position"), 999.0)
        click_depth = link.get("click_depth")
        if click_depth is None:
            click_depth = 8
        bucket = freshness_row.get("bucket") or ("fresh" if getattr(ext, "has_dates", False) else "unknown")

        feature_row = {
            "url": url,
            "title": getattr(page, "title", "") or "",
            "section": getattr(page, "section", "") or "",
            "traffic": traffic,
            "label_log_traffic": math.log1p(max(0.0, traffic)),
            "top_keyword": search_row.get("top_keyword") or "",
            "top_keyword_position": position if position < 999 else None,
            "content_word_count_log": math.log1p(max(0, _safe_int(getattr(page, "word_count", 0)))),
            "content_title_chars": len(getattr(page, "title", "") or ""),
            "content_description_chars": len(getattr(page, "description", "") or getattr(ext, "description", "") or ""),
            "content_heading_count": len(headers),
            "content_paragraph_count": len(paragraphs),
            "content_avg_paragraph_words": (sum(paragraph_words) / len(paragraph_words)) if paragraph_words else 0.0,
            "content_table_count": _safe_int(getattr(ext, "table_count", 0)),
            "content_list_count": _safe_int(getattr(ext, "list_count", 0)),
            "content_stat_count": _safe_int(getattr(ext, "stat_count", 0)),
            "content_external_links": _safe_int(getattr(ext, "external_link_count", 0)),
            "schema_type_count": len(schema_types),
            "schema_valid_blocks": _safe_int(structured_row.get("valid_blocks")),
            "schema_invalid_blocks": _safe_int(structured_row.get("invalid_blocks")),
            "freshness_score": _freshness_score(bucket),
            "freshness_age_log": math.log1p(max(0, _safe_int(freshness_row.get("age_days")))),
            "links_in_degree_log": math.log1p(max(0, _safe_int(link.get("in_degree")))),
            "links_out_degree_log": math.log1p(max(0, _safe_int(link.get("out_degree")))),
            "links_click_depth_inverse": 1.0 / (1.0 + max(0, _safe_float(click_depth))),
            "links_pagerank": _safe_float(link.get("pagerank")) * 1000.0,
            "links_authority_gap": _safe_float(link.get("authority_traffic_gap")),
            "entities_count_log": math.log1p(max(0, _safe_int(entity_row.get("entity_count")))),
            "entities_depth_score": _safe_float(entity_row.get("topical_depth_score")),
            "entities_coverage": _safe_float(entity_cov_row.get("coverage")),
            "information_gain_score": _safe_float(info_row.get("information_gain_score")),
            "answerability_score": _safe_float(answer_row.get("score")),
            "conversion_cta_count": _safe_int(conv_row.get("cta_count")),
            "conversion_primary_cta_count": _safe_int(conv_row.get("primary_cta_count")),
            "conversion_form_count": _safe_int(conv_row.get("form_count")),
            "conversion_support": _safe_float(conv_balance_row.get("conversion_support")),
            "performance_html_weight_log": math.log1p(max(0, _safe_int(perf_row.get("html_weight_bytes")))),
            "performance_estimated_weight_log": math.log1p(max(0, _safe_int(perf_row.get("estimated_weight_bytes")))),
            "performance_resource_tags_log": math.log1p(max(0, _safe_int(perf_row.get("resource_tag_count")))),
            "performance_render_blocking": _safe_int(perf_row.get("render_blocking_count")),
            "performance_image_count_log": math.log1p(max(0, _safe_int(perf_row.get("image_count")))),
            "metadata_issue_count": _metadata_issue_count(meta_row),
            "media_issue_count": _media_issue_count(media_row),
            "paragraph_link_density": _safe_float(density_row.get("links_per_100w")),
            "search_keywords_log": math.log1p(max(0, _safe_int(search_row.get("keywords")))),
            "search_position_score": _position_score(position),
            "search_refdomains_log": math.log1p(max(0, _safe_int(search_row.get("referring_domains")))),
            "search_url_rating": _safe_float(search_row.get("url_rating")),
        }
        rows.append(feature_row)
        x_rows.append([_feature_value(feature_row, feature.key) for feature in FEATURES])
        labels.append(feature_row["label_log_traffic"])

    return rows, np.asarray(x_rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_train = x[train_idx]
    y_train = y[train_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    xz = (x_train - mean) / std
    yc = y_train - y_train.mean()
    reg = np.eye(xz.shape[1]) * alpha
    try:
        coef = np.linalg.solve(xz.T @ xz + reg, xz.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xz.T @ xz + reg) @ xz.T @ yc
    return coef, mean, std, float(y_train.mean())


def _predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray, intercept: float) -> np.ndarray:
    return intercept + ((x - mean) / std) @ coef


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _cross_validate(x: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    n = len(y)
    if n < MIN_MODEL_PAGES:
        return {"folds": 0, "r2": 0.0, "mae_log": 0.0}
    folds = min(5, max(3, n // 6))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    predictions = np.zeros(n, dtype=np.float64)
    tested = np.zeros(n, dtype=bool)
    for fold in np.array_split(indices, folds):
        if not len(fold):
            continue
        train = np.array([i for i in indices if i not in set(fold)], dtype=np.int64)
        coef, mean, std, intercept = _fit_ridge(x, y, train)
        predictions[fold] = _predict(x[fold], coef, mean, std, intercept)
        tested[fold] = True
    y_test = y[tested]
    pred_test = predictions[tested]
    return {
        "folds": folds,
        "r2": round(_r2(y_test, pred_test), 4) if len(y_test) else 0.0,
        "mae_log": round(float(np.mean(np.abs(y_test - pred_test))), 4) if len(y_test) else 0.0,
    }


def _permutation_importance(x: np.ndarray, y: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray, intercept: float, seed: int = 42) -> list[float]:
    pred = _predict(x, coef, mean, std, intercept)
    baseline = _r2(y, pred)
    rng = np.random.default_rng(seed)
    importances: list[float] = []
    for j in range(x.shape[1]):
        scores = []
        for _ in range(3):
            xp = x.copy()
            xp[:, j] = xp[rng.permutation(len(xp)), j]
            scores.append(max(0.0, baseline - _r2(y, _predict(xp, coef, mean, std, intercept))))
        importances.append(float(np.mean(scores)))
    return importances


def _feature_payload(rows: list[dict], x: np.ndarray, coef: np.ndarray, importances: list[float]) -> list[dict]:
    out = []
    for j, feature in enumerate(FEATURES):
        values = x[:, j] if len(x) else np.asarray([], dtype=np.float64)
        coefficient = float(coef[j]) if len(coef) else 0.0
        out.append({
            "feature": feature.key,
            "label": feature.label,
            "group": feature.group,
            "definition": feature.definition,
            "coefficient": round(coefficient, 5),
            "direction": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "neutral",
            "permutation_importance": round(float(importances[j]) if j < len(importances) else 0.0, 5),
            "abs_coefficient": round(abs(coefficient), 5),
            "median": round(float(np.median(values)), 4) if len(values) else 0.0,
            "p90": round(float(np.percentile(values, 90)), 4) if len(values) else 0.0,
        })
    out.sort(key=lambda r: (r["permutation_importance"], r["abs_coefficient"]), reverse=True)
    return out


def _page_explanations(rows: list[dict], x: np.ndarray, y: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray, intercept: float) -> list[dict]:
    if not len(rows):
        return []
    xz = (x - mean) / std
    contribs = xz * coef
    pred_log = _predict(x, coef, mean, std, intercept)
    out = []
    feature_meta = {feature.key: feature for feature in FEATURES}
    for i, row in enumerate(rows):
        pairs = []
        for j, feature in enumerate(FEATURES):
            value = _feature_value(row, feature.key)
            contribution = float(contribs[i, j])
            pairs.append({
                "feature": feature.key,
                "label": feature.label,
                "group": feature.group,
                "value": round(value, 4),
                "contribution": round(contribution, 5),
                "direction": "helping" if contribution > 0 else "hurting" if contribution < 0 else "neutral",
            })
        positive = sorted([p for p in pairs if p["contribution"] > 0], key=lambda p: p["contribution"], reverse=True)[:6]
        negative = sorted([p for p in pairs if p["contribution"] < 0], key=lambda p: p["contribution"])[:6]
        out.append({
            "url": row.get("url") or "",
            "title": row.get("title") or row.get("url") or "",
            "section": row.get("section") or "",
            "traffic": round(_safe_float(row.get("traffic")), 1),
            "top_keyword": row.get("top_keyword") or "",
            "top_keyword_position": row.get("top_keyword_position"),
            "actual_log_traffic": round(float(y[i]), 4),
            "predicted_log_traffic": round(float(pred_log[i]), 4),
            "predicted_traffic": round(max(0.0, math.expm1(float(pred_log[i]))), 1),
            "residual_log": round(float(y[i] - pred_log[i]), 4),
            "top_positive": positive,
            "top_negative": negative,
            "feature_snapshot": {
                feature.key: round(_feature_value(row, feature.key), 4)
                for feature in FEATURES
                if abs(_feature_value(row, feature.key)) > 0
            },
        })
    out.sort(key=lambda r: (_safe_float(r.get("traffic")), abs(_safe_float(r.get("residual_log")))), reverse=True)
    return out


def build_performance_explainer(
    pages: list,
    extracted_pages: list | None = None,
    *,
    search_payload: dict | None = None,
    linkgraph: dict | None = None,
    structured_data: dict | None = None,
    freshness: dict | None = None,
    entities: dict | None = None,
    entity_coverage: dict | None = None,
    information_gain: dict | None = None,
    answerability: list | dict | None = None,
    conversion: dict | None = None,
    conversion_balance: dict | None = None,
    performance: dict | None = None,
    metadata_quality: dict | None = None,
    media_accessibility: dict | None = None,
    paragraph_density: dict | None = None,
    seed: int = 42,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "model": "ridge_performance_explainer_v1", "sample_size": 0}, "features": [], "pages": []}

    rows, x, y = _build_rows(
        pages,
        extracted_pages,
        search_payload=search_payload,
        linkgraph=linkgraph,
        structured_data=structured_data,
        freshness=freshness,
        entities=entities,
        entity_coverage=entity_coverage,
        information_gain=information_gain,
        answerability=answerability,
        conversion=conversion,
        conversion_balance=conversion_balance,
        performance=performance,
        metadata_quality=metadata_quality,
        media_accessibility=media_accessibility,
        paragraph_density=paragraph_density,
    )
    sample_size = len(rows)
    positive_labels = sum(1 for row in rows if _safe_float(row.get("traffic")) > 0)
    warnings: list[str] = [
        "Correlation model only: outputs are estimates from observed page features, not direct Google ranking factors.",
        "Search-provider features such as keyword count and position can leak outcome information; interpret them as context, not causal levers.",
    ]
    if sample_size < 30:
        warnings.append("Small sample size; validation and feature importance may be unstable.")
    if positive_labels < MIN_POSITIVE_LABELS:
        warnings.append("Too few pages with positive search traffic labels for a reliable model.")
    if sample_size and x.shape[1] / max(1, sample_size) > 1.5:
        warnings.append("Feature count is high relative to sample size; coefficients are regularized and should be read directionally.")

    if sample_size < MIN_MODEL_PAGES or positive_labels < MIN_POSITIVE_LABELS or float(np.var(y)) <= 1e-9:
        return {
            "summary": {
                "status": "insufficient_labels",
                "model": "ridge_performance_explainer_v1",
                "method": "Feature matrix built, model skipped due to weak labels.",
                "sample_size": sample_size,
                "feature_count": len(FEATURES),
                "positive_label_pages": positive_labels,
                "label": "log1p organic traffic from search provider payload",
                "validation_metric": "not_available",
                "warnings": warnings,
            },
            "feature_definitions": [feature.__dict__ for feature in FEATURES],
            "features": [],
            "pages": rows[:500],
        }

    train_idx = np.arange(sample_size, dtype=np.int64)
    coef, mean, std, intercept = _fit_ridge(x, y, train_idx)
    validation = _cross_validate(x, y, seed=seed)
    if validation.get("r2", 0.0) <= 0:
        warnings.append("Cross-validation R2 is weak or negative; use feature direction as exploratory evidence only.")
    importances = _permutation_importance(x, y, coef, mean, std, intercept, seed=seed)
    features = _feature_payload(rows, x, coef, importances)
    page_rows = _page_explanations(rows, x, y, coef, mean, std, intercept)
    group_importance: dict[str, float] = defaultdict(float)
    for feature in features:
        group_importance[feature["group"]] += _safe_float(feature.get("permutation_importance")) + _safe_float(feature.get("abs_coefficient")) * 0.1
    groups = [
        {"group": group, "importance": round(score, 5)}
        for group, score in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "summary": {
            "status": "ok",
            "model": "ridge_performance_explainer_v1",
            "method": "Standardized ridge regression with deterministic cross-validation and permutation importance.",
            "sample_size": sample_size,
            "feature_count": len(FEATURES),
            "positive_label_pages": positive_labels,
            "label": "log1p organic traffic from search provider payload",
            "validation_metric": "cross_validated_log_traffic_r2",
            "validation_r2": validation["r2"],
            "validation_mae_log": validation["mae_log"],
            "folds": validation["folds"],
            "warnings": warnings,
        },
        "feature_definitions": [feature.__dict__ for feature in FEATURES],
        "features": features,
        "feature_groups": groups,
        "pages": page_rows[:700],
    }
