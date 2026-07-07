from site_audit.recommendations import Recommendation, synthesize, to_payload


def _finalized(*recs: Recommendation) -> dict:
    return to_payload(recs)


def test_duplicate_drop_target_suppresses_improve_recommendation() -> None:
    payload = synthesize(
        duplicates_rows=[
            {
                "similarity": 0.98,
                "url_a": "https://example.com/canonical",
                "url_b": "https://example.com/drop",
            }
        ],
        answerability_payload=[
            {"url": "https://example.com/drop", "score": 2.0, "flags": ["missing FAQ"]},
            {"url": "https://example.com/canonical", "score": 2.0, "flags": ["missing FAQ"]},
        ],
        linkgraph_payload={
            "top_authority_pages": [
                {"url": "https://example.com/canonical", "pagerank": 0.02},
                {"url": "https://example.com/drop", "pagerank": 0.01},
            ]
        },
    )
    report = to_payload(payload)

    ids = {item["id"] for item in report["items"]}
    suppressed = report["suppressed"]
    dup_id = next(item["id"] for item in report["items"] if item["id"].startswith("dup-"))

    assert dup_id in ids
    assert any(item["targets"] == ["https://example.com/canonical"] for item in report["items"])
    assert not any(item["targets"] == ["https://example.com/drop"] for item in report["items"])
    assert len(suppressed) == 1
    assert suppressed[0]["targets"] == ["https://example.com/drop"]
    assert suppressed[0]["suppressed_by"] == dup_id
    assert dup_id in suppressed[0]["suppressed_reason"]
    assert report["summary"]["suppressed"] == 1


def test_cannibalization_runner_up_suppresses_improve_but_best_url_remains() -> None:
    report = to_payload(synthesize(
        coverage_payload=[
            {
                "status": "cannibalized",
                "query": "support automation",
                "best_url": "https://example.com/best",
                "best_similarity": 0.9,
                "candidates_above_threshold": 3,
                "runner_ups": [
                    {"url": "https://example.com/runner"},
                    {"url": "https://example.com/runner-two"},
                ],
            }
        ],
        title_mismatch=[
            {"url": "https://example.com/best", "title_to_content": 0.2, "title": "Best"},
            {"url": "https://example.com/runner", "title_to_content": 0.2, "title": "Runner"},
        ],
    ))

    assert any(item["targets"] == ["https://example.com/best"] for item in report["items"])
    assert not any(item["targets"] == ["https://example.com/runner"] for item in report["items"])
    assert report["suppressed"][0]["targets"] == ["https://example.com/runner"]
    assert report["suppressed"][0]["suppressed_by"].startswith("cann-")


def test_cards_group_url_recommendations_and_sum_modeled_gain() -> None:
    report = _finalized(
        Recommendation(
            id="geo-page",
            category="geo",
            priority="medium",
            title="Answerability",
            instruction="Add FAQ.",
            targets=["https://example.com/page"],
            evidence={"current_title": "Page title"},
            priority_score=40.0,
            effort_score=55.0,
            estimated_clicks_gain=10.0,
        ),
        Recommendation(
            id="title-page",
            category="onpage",
            priority="high",
            title="Rewrite title",
            instruction="Rewrite.",
            targets=["https://example.com/page"],
            priority_score=70.0,
            effort_score=25.0,
            estimated_clicks_gain=5.0,
        ),
        Recommendation(
            id="orphan-page",
            category="linking",
            priority="low",
            title="Add links",
            instruction="Add links.",
            targets=["https://example.com/page"],
            priority_score=20.0,
            effort_score=25.0,
        ),
    )

    card = report["cards"][0]

    assert report["summary"]["cards"] == 1
    assert card["url"] == "https://example.com/page"
    assert card["title"] == "Page title"
    assert card["total_estimated_clicks_gain"] == 15.0
    assert card["top_priority"] == "high"
    assert card["top_priority_score"] == 70.0
    assert card["categories"] == ["geo", "linking", "onpage"]
    assert card["recommendation_ids"] == ["geo-page", "title-page", "orphan-page"]
    # Label reflects the heaviest member effort, not the summed score.
    assert card["effort_total"] == "medium"


def test_multi_target_recommendation_lands_on_canonical_card() -> None:
    report = _finalized(
        Recommendation(
            id="dup-pair",
            category="content_debt",
            priority="high",
            title="Merge duplicate",
            instruction="Redirect duplicate.",
            targets=["https://example.com/canonical", "https://example.com/drop"],
            priority_score=80.0,
            effort_score=25.0,
        )
    )

    card = report["cards"][0]

    assert card["url"] == "https://example.com/canonical"
    assert card["related_urls"] == ["https://example.com/drop"]
    assert card["recommendation_ids"] == ["dup-pair"]


def test_coverage_gap_uses_new_content_pseudo_card() -> None:
    report = to_payload(synthesize(
        coverage_payload=[
            {
                "status": "gap",
                "query": "support automation tools",
                "source": "manual",
                "best_similarity": 0.2,
                "best_url": "https://example.com/neighbor",
            }
        ]
    ))

    card = report["cards"][0]

    assert card["url"] == ""
    assert card["query"] == "support automation tools"
    assert card["related_urls"] == ["https://example.com/neighbor"]
    assert len(card["recommendation_ids"]) == 1
    assert card["recommendation_ids"][0].startswith("gap-")


def test_recommendation_payload_is_deterministic_with_cards_and_suppression() -> None:
    kwargs = {
        "duplicates_rows": [
            {
                "similarity": 0.98,
                "url_a": "https://example.com/a",
                "url_b": "https://example.com/b",
            }
        ],
        "answerability_payload": [
            {"url": "https://example.com/b", "score": 2.0, "flags": ["missing FAQ"]},
            {"url": "https://example.com/a", "score": 2.0, "flags": ["missing FAQ"]},
        ],
        "linkgraph_payload": {
            "top_authority_pages": [{"url": "https://example.com/a", "pagerank": 0.02}]
        },
    }

    assert to_payload(synthesize(**kwargs)) == to_payload(synthesize(**kwargs))


def test_items_shape_and_summary_counts_remain_kept_only() -> None:
    report = to_payload(synthesize(
        duplicates_rows=[
            {
                "similarity": 0.98,
                "url_a": "https://example.com/a",
                "url_b": "https://example.com/b",
            }
        ],
        answerability_payload=[
            {"url": "https://example.com/b", "score": 2.0, "flags": ["missing FAQ"]},
        ],
        linkgraph_payload={
            "top_authority_pages": [{"url": "https://example.com/a", "pagerank": 0.02}]
        },
    ))

    item = report["items"][0]

    assert report["total"] == 1
    assert report["by_category"] == {"content_debt": 1}
    assert "suppressed" not in item
    assert "suppressed_by" not in item
    assert "suppressed_reason" not in item
    assert set(item) == {
        "id",
        "category",
        "type",
        "priority",
        "priority_score",
        "impact",
        "confidence",
        "effort_score",
        "risk",
        "owner",
        "cluster",
        "traffic_opportunity",
        "estimated_clicks_gain",
        "estimate_basis",
        "gain_label",
        "title",
        "instruction",
        "targets",
        "evidence",
        "effort",
        "score",
    }


def test_max_total_truncation_never_resurrects_suppressed_actions() -> None:
    # More recommendations than max_total: improve-recs for redirect-slated
    # pages must never reappear in items just because their suppressor was
    # truncated out of the report.
    duplicates = [
        {"similarity": 0.99, "url_a": f"https://example.com/k{i}", "url_b": f"https://example.com/d{i}"}
        for i in range(8)
    ]
    answerability = [
        {"url": f"https://example.com/d{i}", "score": 1.0, "flags": ["no schema markup"], "title": f"Doomed {i}"}
        for i in range(8)
    ]

    recs = synthesize(duplicates_rows=duplicates, answerability_payload=answerability, max_total=3)
    report = to_payload(recs)

    assert len(report["items"]) <= 3
    removal_targets = {
        item["targets"][1]
        for item in report["items"]
        if item["id"].startswith("dup-") and len(item["targets"]) > 1
    }
    improve_targets = {
        item["targets"][0]
        for item in report["items"]
        if not item["id"].startswith("dup-") and item["targets"]
    }
    assert not removal_targets & improve_targets
    item_ids = {item["id"] for item in report["items"]}
    for row in report["suppressed"]:
        assert row["suppressed_by"] in item_ids


def test_link_recommendation_to_removal_target_is_suppressed() -> None:
    report = to_payload(synthesize(
        duplicates_rows=[
            {"similarity": 0.99, "url_a": "https://example.com/keep", "url_b": "https://example.com/drop"},
        ],
        linkgraph_payload={
            "recommendations": [
                {
                    "source_url": "https://example.com/other",
                    "source_title": "Other",
                    "target_url": "https://example.com/drop",
                    "target_title": "Drop",
                    "similarity": 0.9,
                }
            ]
        },
    ))

    link_rows = [row for row in report["suppressed"] if row["id"].startswith("link-")]
    assert len(link_rows) == 1
    assert "link destination is slated for redirect/merge" in link_rows[0]["suppressed_reason"]
    assert all(not item["id"].startswith("link-") for item in report["items"])
