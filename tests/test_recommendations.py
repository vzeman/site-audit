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
