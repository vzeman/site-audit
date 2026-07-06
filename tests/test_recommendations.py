import pytest

from site_audit.ctr_curve import estimate_clicks_gain, expected_ctr
from site_audit.recommendations import synthesize, to_payload


def test_recommendations_include_deterministic_priority_components() -> None:
    recs = synthesize(
        duplicates_rows=[
            {
                "similarity": 0.98,
                "url_a": "https://example.com/a",
                "url_b": "https://example.com/a-copy",
            }
        ],
        linkgraph_payload={
            "top_authority_pages": [{"url": "https://example.com/a", "pagerank": 0.02}],
            "traffic_weighted_pagerank": {
                "pages": [
                    {
                        "url": "https://example.com/a",
                        "traffic": 1200,
                        "keywords": 18,
                        "cluster": "support",
                        "pagerank": 0.02,
                    }
                ]
            },
        },
        search_payload={
            "top_pages": [
                {
                    "matched_url": "https://example.com/a",
                    "traffic": 1200,
                    "keywords": 18,
                    "cluster_label": "support",
                }
            ]
        },
    )
    payload = to_payload(recs)
    item = payload["items"][0]

    assert payload["score_model"]["model"] == "fix_priority_score_v1"
    assert item["priority_score"] > 0
    assert item["impact"] > 0
    assert item["confidence"] > 0
    assert item["effort_score"] > 0
    assert item["risk"] > 0
    assert item["owner"] == "Content"
    assert item["type"] == "merge_duplicate"
    assert item["cluster"] == "support"
    assert item["evidence"]["score_components"]["priority_score"] == item["priority_score"]


def test_linkgraph_url_a_url_b_recommendations_render_page_labels() -> None:
    recs = synthesize(
        linkgraph_payload={
            "recommendations": [
                {
                    "url_a": "https://example.com/source",
                    "title_a": "Source guide",
                    "url_b": "https://example.com/target",
                    "title_b": "Target guide",
                    "similarity": 0.95,
                }
            ]
        }
    )

    payload = to_payload(recs)
    item = next(row for row in payload["items"] if row["type"] == "internal_link")

    assert "None" not in item["instruction"]
    assert "Source guide" in item["instruction"]
    assert "Target guide" in item["instruction"]
    assert item["targets"] == ["https://example.com/source", "https://example.com/target"]


def test_answerability_recommendations_fallback_to_url_when_title_missing() -> None:
    recs = synthesize(
        answerability_payload=[
            {
                "url": "https://example.com/weak",
                "score": 2.0,
                "flags": ["few/no statistics"],
            }
        ]
    )

    payload = to_payload(recs)
    item = next(row for row in payload["items"] if row["type"] == "answerability")

    assert "on  page" not in item["title"]
    assert "https://example.com/weak" in item["title"]


def test_ctr_anomaly_recommendations_land_in_payload() -> None:
    recs = synthesize(
        search_payload={
            "query_pages": [
                {
                    "query": "support automation",
                    "matched_url": "https://example.com/support",
                    "position": 8,
                    "impressions": 100000,
                }
            ]
        },
        ctr_anomalies_payload={
            "recommendations": [
                {
                    "title": 'Title underperforms position #2 for "support automation"',
                    "action": (
                        'Rewrite title/meta of https://example.com/support: CTR is 5.0% vs 15.0% expected '
                        'at position 2 — ~100 missed clicks/30 days. Probable cause: unclear. '
                        'Include "support automation" phrasing in the title (<=65 chars).'
                    ),
                    "url": "https://example.com/support",
                    "query": "support automation",
                    "position": 2,
                    "actual_ctr": 0.05,
                    "expected_ctr": 0.15,
                    "missed_clicks": 100,
                    "probable_cause": "unclear",
                    "period": "30 days",
                    "current_title": "Support automation guide",
                }
            ]
        }
    )

    payload = to_payload(recs)
    item = next(row for row in payload["items"] if row["id"].startswith("ctr-"))

    assert item["category"] == "onpage"
    assert item["type"] == "title_rewrite"
    assert item["title"] == 'Title underperforms position #2 for "support automation"'
    assert item["targets"] == ["https://example.com/support"]
    assert item["evidence"]["estimated_clicks_gain"] == 100
    assert item["estimated_clicks_gain"] == 100
    assert item["traffic_opportunity"] == 100


def test_recommendation_gain_uses_query_pages_before_keyword_volume() -> None:
    recs = synthesize(
        linkgraph_payload={
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/target",
                    "target_title": "Target",
                    "similarity": 0.91,
                }
            ]
        },
        search_payload={
            "top_pages": [
                {
                    "matched_url": "https://example.com/target",
                    "traffic": 10,
                    "top_keyword": "fallback query",
                    "top_keyword_position": 9,
                    "top_keyword_volume": 500,
                }
            ],
            "query_pages": [
                {
                    "query": "higher impression query",
                    "matched_url": "https://example.com/target",
                    "position": 8,
                    "impressions": 1000,
                },
                {
                    "query": "lower impression query",
                    "matched_url": "https://example.com/target",
                    "position": 6,
                    "impressions": 200,
                },
            ],
        },
    )

    item = to_payload(recs)["items"][0]

    assert item["estimate_basis"] == "gsc_impressions"
    assert item["estimated_clicks_gain"] == pytest.approx(estimate_clicks_gain(1000, 8, 4), abs=0.01)
    assert item["gain_label"] == f"≈ +{item['estimated_clicks_gain']:.0f} clicks/period"


def test_recommendation_gain_uses_top_keyword_volume_and_mentions_query() -> None:
    recs = synthesize(
        linkgraph_payload={
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/target",
                    "target_title": "Target",
                    "similarity": 0.91,
                }
            ]
        },
        search_payload={
            "top_pages": [
                {
                    "matched_url": "https://example.com/target",
                    "traffic": 10,
                    "top_keyword": "support automation",
                    "top_keyword_position": 10,
                    "top_keyword_volume": 2000,
                }
            ],
        },
    )

    item = to_payload(recs)["items"][0]

    assert item["estimate_basis"] == "keyword_volume"
    assert item["estimated_clicks_gain"] == pytest.approx(estimate_clicks_gain(2000, 10, 5), abs=0.01)
    assert 'Top query: "support automation" (position 10, 2000/mo).' in item["instruction"]


def test_coverage_gap_gain_uses_volume_at_position_five() -> None:
    recs = synthesize(
        coverage_payload=[
            {
                "status": "gap",
                "query": "support automation tools",
                "source": "manual",
                "best_similarity": 0.2,
                "volume": 1000,
            }
        ]
    )

    item = to_payload(recs)["items"][0]

    assert item["type"] == "coverage_gap"
    assert item["estimate_basis"] == "keyword_volume"
    assert item["estimated_clicks_gain"] == pytest.approx(1000 * expected_ctr(5), abs=0.01)


def test_percentile_business_norm_orders_identical_recs_by_traffic() -> None:
    recs = synthesize(
        linkgraph_payload={
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/low",
                    "target_title": "Low",
                    "similarity": 0.9,
                },
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/high",
                    "target_title": "High",
                    "similarity": 0.9,
                },
            ]
        },
        search_payload={
            "top_pages": [
                {"matched_url": "https://example.com/low", "traffic": 1000},
                {"matched_url": "https://example.com/high", "traffic": 100000},
            ]
        },
    )

    items = to_payload(recs)["items"]

    assert items[0]["targets"][-1] == "https://example.com/high"
    assert items[0]["priority_score"] > items[1]["priority_score"]


def test_coverage_gap_does_not_borrow_neighbor_page_keyword_data() -> None:
    recs = synthesize(
        coverage_payload=[
            {
                "status": "gap",
                "query": "query without volume",
                "source": "manual",
                "best_similarity": 0.2,
                "best_url": "https://example.com/neighbor",
            }
        ],
        search_payload={
            "top_pages": [
                {
                    "matched_url": "https://example.com/neighbor",
                    "traffic": 5000,
                    "top_keyword": "neighbor keyword",
                    "top_keyword_position": 2,
                    "top_keyword_volume": 9000,
                }
            ],
        },
    )

    gap = next(item for item in to_payload(recs)["items"] if item["id"].startswith("gap-"))

    # The gap query has no volume of its own; the neighbor page's keyword
    # volume must not be attributed to it, and its top query must not be
    # appended to a "write a new page" instruction.
    assert gap["estimated_clicks_gain"] is None
    assert "Top query:" not in gap["instruction"]


def test_no_search_provider_reproduces_legacy_ordering_and_buckets() -> None:
    # Cross-category fixture with linkgraph-only data (no search provider).
    # Expected ids/impacts/priorities were captured by running this exact
    # fixture against origin/main before the click-model change; the
    # degraded path must reproduce them so audits without search data keep
    # their historical action-plan ordering and buckets.
    recs = synthesize(
        duplicates_rows=[
            {"similarity": 0.98, "url_a": "https://example.com/a", "url_b": "https://example.com/a-copy"},
        ],
        outliers_rows=[
            {"url": "https://example.com/outlier", "similarity": 0.12, "title": "Outlier"},
        ],
        answerability_payload=[
            {"url": "https://example.com/geo", "score": 1.0, "flags": ["no schema markup"], "title": "Geo"},
        ],
        title_mismatch=[
            {"url": "https://example.com/title", "title_to_content": 0.2, "title": "Bad title", "suggested_keywords": ["support"]},
        ],
        linkgraph_payload={
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/target",
                    "target_title": "Target",
                    "similarity": 0.91,
                }
            ],
            "orphans": [{"url": "https://example.com/orphan", "title": "Orphan"}],
            "traffic_weighted_pagerank": {
                "pages": [
                    {"url": "https://example.com/target", "pagerank": 0.004, "traffic": 0},
                    {"url": "https://example.com/geo", "pagerank": 0.0001, "traffic": 0},
                ]
            },
        },
    )

    items = to_payload(recs)["items"]

    assert [item["id"] for item in items] == ["link-0", "dup-0", "title-0", "geo-0", "orphan-0", "out-0"]
    assert [item["impact"] for item in items] == [72.0, 62.0, 49.0, 54.2, 10.0, 0.0]
    assert [item["priority"] for item in items] == ["medium", "low", "low", "low", "low", "low"]
    assert all(item["estimated_clicks_gain"] is None for item in items)


def test_recommendations_are_deterministic_with_modeled_business_data() -> None:
    kwargs = {
        "linkgraph_payload": {
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "source_title": "Source",
                    "target_url": "https://example.com/target",
                    "target_title": "Target",
                    "similarity": 0.91,
                }
            ]
        },
        "search_payload": {
            "top_pages": [{"matched_url": "https://example.com/target", "traffic": 10}],
            "query_pages": [
                {
                    "query": "support automation",
                    "matched_url": "https://example.com/target",
                    "position": 8,
                    "impressions": 1000,
                }
            ],
        },
    }

    first = to_payload(synthesize(**kwargs))
    second = to_payload(synthesize(**kwargs))

    assert first["items"] == second["items"]
