from pathlib import Path

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.dataforseo import DataForSEOClient, DataForSEOConfig, build_analysis


def _response(items: list[dict], *, metrics: dict | None = None) -> dict:
    result = {"items": items}
    if metrics is not None:
        result["metrics"] = metrics
    return {
        "status_code": 20000,
        "cost": 0.01,
        "tasks": [{"status_code": 20000, "cost": 0.01, "result": [result]}],
    }


def test_dataforseo_client_reuses_cached_snapshot_without_credentials(tmp_path: Path) -> None:
    calls = []

    def requester(endpoint: str, task: dict) -> dict:
        calls.append((endpoint, task))
        if endpoint.endswith("domain_rank_overview/live"):
            return _response([], metrics={"organic": {"count": 1, "etv": 10}})
        if endpoint.endswith("relevant_pages/live"):
            return _response([{"page_address": "https://example.com/a", "metrics": {"organic": {"count": 1, "etv": 10}}}])
        if endpoint.endswith("ranked_keywords/live"):
            return _response([])
        return _response([])

    cfg = DataForSEOConfig(top_pages_limit=10, keywords_limit=10)
    first = DataForSEOClient("login", "password", tmp_path, requester=requester).load_or_fetch("example.com", cfg)
    assert first["meta"]["status"] == "ok"
    assert len(calls) == 3

    second = DataForSEOClient("", "", tmp_path, requester=requester).load_or_fetch("example.com", cfg)
    assert second["meta"]["status"] == "ok"
    assert second["meta"]["cache_status"] == "hit"
    assert len(calls) == 3


def test_dataforseo_analysis_matches_pages_and_aggregates_search_features() -> None:
    pages = [
        PageInfo(
            url="https://www.example.com/blog/a/",
            title="AI Agent Guide",
            description="",
            section="blog",
            word_count=500,
            language="en",
        )
    ]
    snapshot = {
        "meta": {"status": "ok", "cache_status": "hit", "provider": "dataforseo"},
        "raw": {
            "domain_rank_overview": _response([], metrics={"organic": {"count": 2, "etv": 120, "pos_1": 1, "pos_2_3": 1}}),
            "relevant_pages": _response([
                {
                    "page_address": "https://example.com/blog/a",
                    "metrics": {
                        "organic": {
                            "count": 2,
                            "etv": 120,
                            "estimated_paid_traffic_cost": 3.5,
                            "pos_1": 1,
                            "pos_2_3": 1,
                        },
                        "featured_snippet": {"count": 1, "etv": 20},
                    },
                }
            ]),
            "ranked_keywords": _response([
                {
                    "keyword_data": {
                        "keyword": "ai agent",
                        "keyword_info": {"search_volume": 1000, "cpc": 2.3},
                        "search_intent_info": {"main_intent": "informational"},
                        "serp_info": {"serp_item_types": ["organic", "featured_snippet", "ai_overview_reference"]},
                    },
                    "ranked_serp_element": {
                        "rank_group": 2,
                        "rank_absolute": 2,
                        "serp_item": {
                            "type": "organic",
                            "url": "https://www.example.com/blog/a/",
                            "title": "AI Agent Guide",
                            "etv": 100,
                        },
                    },
                },
                {
                    "keyword_data": {
                        "keyword": "ai agent",
                        "keyword_info": {"search_volume": 1000},
                        "keyword_properties": {"keyword_difficulty": 42},
                    },
                    "ranked_serp_element": {
                        "rank_absolute": 1,
                        "serp_item": {
                            "type": "ai_overview_reference",
                            "url": "https://example.com/blog/a",
                            "title": "AI Agent Guide",
                        },
                    },
                }
            ]),
        },
    }

    payload = build_analysis(
        snapshot,
        pages,
        np.eye(1, dtype=np.float32),
        cluster_labels=np.array([0]),
        cluster_summaries=[],
        embedder=None,
    ).payload

    assert payload["summary"]["provider"] == "dataforseo"
    assert payload["summary"]["top_pages_traffic"] == 120
    assert payload["summary"]["top3_keywords"] == 2
    assert payload["top_pages"][0]["top_keyword"] == "ai agent"
    assert payload["top_pages"][0]["featured_snippet_traffic"] == 20
    assert payload["organic_keywords"][0]["serp_type"] == "organic"
    assert payload["organic_keywords"][0]["has_ai_overview"] is True
    assert payload["serp_features"][0]["feature"] == "organic"
    assert payload["intents"][0]["intent"] == "informational"
    assert payload["ai_overview_citations"] == [
        {
            "keyword": "ai agent",
            "url": "https://example.com/blog/a",
            "search_volume": 1000,
            "keyword_difficulty": 42,
            "rank_absolute": 1,
            "serp_title": "AI Agent Guide",
        }
    ]
    assert payload["summary"]["organic_keywords"] == 1
    query_page = payload["query_pages"][0]
    assert query_page["query"] == "ai agent"
    assert query_page["ctr"] is None
    assert query_page["source"] == "dataforseo"
    assert query_page["provider"] == "dataforseo"
    assert query_page["provider_label"] == "DataForSEO"
    assert query_page["impressions"] == 1000
    assert query_page["has_ai_overview"] is True


def test_dataforseo_query_pages_feed_striking_distance() -> None:
    from site_audit.striking_distance import build_striking_distance

    pages = [
        PageInfo(
            url="https://example.com/blog/a/",
            title="AI Agent Guide",
            description="",
            section="blog",
            word_count=500,
            language="en",
        )
    ]
    snapshot = {
        "meta": {"status": "ok", "cache_status": "hit", "provider": "dataforseo"},
        "raw": {
            "domain_rank_overview": _response([]),
            "relevant_pages": _response([]),
            "ranked_keywords": _response([
                {
                    "keyword_data": {
                        "keyword": "ai agent workflows",
                        "keyword_info": {"search_volume": 1000},
                    },
                    "ranked_serp_element": {
                        "rank_group": 8,
                        "rank_absolute": 8,
                        "serp_item": {
                            "type": "organic",
                            "url": "https://example.com/blog/a/",
                            "title": "AI Agent Guide",
                            "etv": 30,
                        },
                    },
                }
            ]),
        },
    }

    payload = build_analysis(snapshot, pages, np.eye(1, dtype=np.float32), embedder=None).payload
    striking = build_striking_distance(payload, pages)

    assert striking["available"] is True
    assert striking["summary"]["status"] == "ok"
    assert striking["rows"][0]["query"] == "ai agent workflows"
    assert striking["rows"][0]["position"] == 8
