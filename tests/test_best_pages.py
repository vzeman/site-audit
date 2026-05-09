from site_audit.best_pages import build_best_page_comparison, build_best_page_explainers


def test_best_page_explainer_combines_evidence_and_causation_note() -> None:
    url = "https://example.com/blog/support-automation"
    payload = build_best_page_explainers(
        [{"url": url, "title": "Support automation", "section": "blog", "word_count": 900}],
        search_payload={
            "top_pages": [
                {
                    "url": url,
                    "matched_url": url,
                    "title": "Support automation",
                    "traffic": 1200,
                    "keywords": 24,
                    "top_keyword": "support automation",
                    "top_keyword_position": 2,
                    "cluster_label": "support automation",
                }
            ],
            "organic_keywords": [
                {
                    "keyword": "support automation",
                    "matched_url": url,
                    "position": 2,
                    "traffic": 800,
                    "volume": 2000,
                    "intents": ["informational"],
                    "serp_features": ["question"],
                }
            ],
        },
        linkgraph={
            "page_link_counts": [{"url": url, "in_degree": 6, "out_degree": 4, "click_depth": 2}],
            "traffic_weighted_pagerank": {
                "pages": [{"url": url, "pagerank": 0.3, "weighted_pagerank_percentile": 0.9}]
            },
        },
        structured_data={"per_page": [{"url": url, "types": ["Article"], "valid_blocks": 1, "invalid_blocks": 0}]},
        freshness={"per_page": [{"url": url, "bucket": "fresh", "age_days": 20}]},
        entity_coverage={"pages": [{"url": url, "coverage": 0.82, "coverage_pct": 82.0, "cluster_label": "support automation"}]},
        information_gain={"pages": [{"url": url, "information_gain_score": 81, "positive_evidence": ["1 data/statistic signals"]}]},
        winning_paragraphs={
            "rows": [
                {
                    "url": url,
                    "paragraph_index": 2,
                    "excerpt": "Support automation routes repetitive requests.",
                    "impact_score": 22.4,
                    "attributed_traffic": 300,
                    "impact_tier": "high",
                }
            ]
        },
        template_patterns={
            "patterns": [
                {
                    "label": "Proof statistics",
                    "feature_key": "proof_stats",
                    "recommendation": "Add current proof statistics.",
                    "observed_lift": 1.4,
                    "confidence": 0.71,
                    "sample_urls": [{"url": url, "title": "Support automation"}],
                }
            ]
        },
    )

    assert payload["summary"]["status"] == "ok"
    page = payload["pages"][0]
    assert page["url"] == url
    assert page["traffic"] == 1200
    assert page["performance_score"] > 60
    assert page["causation_note"]
    assert any(signal["category"] == "Demand" for signal in page["strengths"])
    assert any(pattern["type"] == "template" for pattern in page["transferable_patterns"])
    assert page["evidence_links"][0]["url"] == url


def test_best_page_comparison_recommends_copying_cluster_winner() -> None:
    leader = {
        "domain": "leader.example",
        "best_pages": {
            "pages": [
                {
                    "url": "https://leader.example/a",
                    "title": "A",
                    "cluster_label": "support",
                    "traffic": 1000,
                    "keywords": 30,
                    "performance_score": 86,
                    "copy_recommendations": ["Add proof statistics."],
                    "strengths": [{"label": "Proof statistics", "evidence": "strong"}],
                }
            ]
        },
    }
    laggard = {
        "domain": "laggard.example",
        "best_pages": {
            "pages": [
                {
                    "url": "https://laggard.example/a",
                    "title": "A",
                    "cluster_label": "support",
                    "traffic": 80,
                    "keywords": 4,
                    "performance_score": 44,
                    "weak_spots": [{"label": "No schema", "evidence": "missing"}],
                }
            ]
        },
    }

    payload = build_best_page_comparison([leader, laggard])

    assert payload["summary"]["status"] == "ok"
    assert payload["clusters"][0]["leader_domain"] == "leader.example"
    assert payload["recommendations"][0]["target_domain"] == "laggard.example"
    assert payload["recommendations"][0]["copy"] == ["Add proof statistics."]
    assert payload["side_by_side"][0]["leader"]["domain"] == "leader.example"
