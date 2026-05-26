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
