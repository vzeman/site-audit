from pathlib import Path

import numpy as np

from site_audit.ahrefs import AhrefsClient, AhrefsConfig, _entity_alignment, build_analysis
from site_audit.analyzer import PageInfo


def test_ahrefs_client_reuses_latest_cached_snapshot_without_api_key(tmp_path: Path) -> None:
    calls = []

    def requester(endpoint: str, params: dict) -> dict:
        calls.append((endpoint, params))
        if endpoint == "top-pages":
            return {"pages": [{"url": "https://example.com/blog/a", "sum_traffic": 10}]}
        if endpoint == "organic-keywords":
            return {"keywords": [{"keyword": "ai agent", "best_position_url": "https://example.com/blog/a"}]}
        if endpoint == "metrics":
            return {"metrics": {"org_traffic": 10, "org_keywords": 1}}
        if endpoint == "pages-by-traffic":
            return {"pages": {"range100_pages": 1, "range100_traffic": 10}}
        return {}

    cfg = AhrefsConfig(api_key="secret", date="2026-05-08", country="US", top_pages_limit=10, keywords_limit=10)
    first = AhrefsClient("secret", tmp_path, requester=requester).load_or_fetch("example.com", cfg)
    assert first["meta"]["status"] == "ok"
    assert len(calls) == 4
    assert all(params.get("country") == "us" for _, params in calls)

    latest_cfg = AhrefsConfig(api_key=None, date=None, country="US", top_pages_limit=10, keywords_limit=10)
    second = AhrefsClient("", tmp_path, requester=requester).load_or_fetch("example.com", latest_cfg)
    assert second["meta"]["status"] == "ok"
    assert second["meta"]["cache_status"] == "hit"
    assert len(calls) == 4


def test_ahrefs_analysis_aggregates_pages_keywords_directories_and_clusters() -> None:
    pages = [
        PageInfo(
            url="https://www.example.com/blog/a/",
            title="AI Agent Guide",
            description="",
            section="blog",
            word_count=500,
            language="en",
        ),
        PageInfo(
            url="https://www.example.com/features/b/",
            title="Workflow Automation",
            description="",
            section="features",
            word_count=400,
            language="en",
        ),
    ]
    embeddings = np.eye(2, dtype=np.float32)
    snapshot = {
        "meta": {"status": "ok", "cache_status": "hit", "params": {"date": "2026-05-08"}},
        "raw": {
            "metrics": {"metrics": {"org_traffic": 150, "org_keywords": 8}},
            "pages_by_traffic": {"pages": {"range100_pages": 2, "range100_traffic": 150}},
            "top_pages": {
                "pages": [
                    {
                        "url": "https://example.com/blog/a",
                        "sum_traffic": 100,
                        "keywords": 5,
                        "value": 1234,
                        "top_keyword": "ai agent",
                    },
                    {
                        "url": "https://www.example.com/features/b/",
                        "sum_traffic": 50,
                        "keywords": 3,
                        "top_keyword": "workflow automation",
                    },
                ]
            },
            "organic_keywords": {
                "keywords": [
                    {
                        "keyword": "ai agent",
                        "best_position": 2,
                        "best_position_url": "https://www.example.com/blog/a/",
                        "sum_traffic": 90,
                        "volume": 1000,
                    },
                    {
                        "keyword": "workflow automation",
                        "best_position": 5,
                        "best_position_url": "https://www.example.com/features/b/",
                        "sum_traffic": 40,
                        "volume": 500,
                    },
                ]
            },
        },
    }

    analysis = build_analysis(
        snapshot,
        pages,
        embeddings,
        cluster_labels=np.array([0, 1]),
        cluster_summaries=[],
        embedder=None,
    )
    payload = analysis.payload

    assert payload["summary"]["top_pages_traffic"] == 150
    assert payload["summary"]["matched_top_pages"] == 2
    assert payload["summary"]["matched_traffic_share"] == 1.0
    assert payload["directories"][0]["label"] == "blog"
    assert payload["directories"][0]["traffic"] == 100
    assert payload["clusters"][0]["traffic"] == 100
    assert payload["organic_keywords"][0]["keyword"] == "ai agent"


def test_entity_alignment_scores_keyword_against_visible_entities() -> None:
    rows = [
        {"type": "keyword", "label": "support ticket", "url": "https://example.com/a", "traffic": 20, "volume": 100, "position": 4, "cluster": 1},
        {"type": "page_title", "label": "Pricing", "url": "https://example.com/a"},
        {"type": "header", "label": "Overview", "url": "https://example.com/a"},
        {"type": "paragraph", "label": "Support ticket software helps route cases.", "url": "https://example.com/a"},
        {"type": "page", "label": "Support ticket software", "url": "https://example.com/a"},
        {"type": "link_title", "label": "support ticket software"},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    payload = _entity_alignment(rows, embeddings)

    assert payload["summary"]["status"] == "ok"
    assert payload["by_url"][0]["type_scores"]["paragraph"] == 1.0
    assert payload["by_url"][0]["type_scores"]["page_title"] == 0.0
    issues = {row["issue"] for row in payload["recommendations"]}
    assert {"title_mismatch", "heading_mismatch"}.issubset(issues)
