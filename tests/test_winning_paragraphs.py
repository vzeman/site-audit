from site_audit.winning_paragraphs import build_winning_paragraphs


def test_winning_paragraphs_merges_impact_ablation_and_keywords():
    impact = {
        "top_paragraphs": [
            {
                "url": "https://example.com/a",
                "title": "Example A",
                "section": "Support",
                "heading": "Call center software",
                "paragraph_index": 0,
                "paragraph_excerpt": "Call center software helps teams answer customers faster.",
                "impact_tier": "high",
                "impact_score": 100.0,
                "attributed_traffic": 24.0,
                "relevance_score": 82.0,
                "best_keyword": "call center software",
                "components": {
                    "semantic": 0.8,
                    "keyword_overlap": 0.7,
                    "heading_match": 0.8,
                    "link_context": 0.8,
                    "freshness": 1.0,
                },
            }
        ]
    }
    ablation = {
        "rows": [
            {
                "url": "https://example.com/a",
                "paragraph_index": 0,
                "classification": "topic_carrier",
                "classification_label": "Topic carrier",
                "alignment_delta": 0.04,
                "self_alignment": 0.9,
            }
        ]
    }
    attribution = {
        "keywords": [
            {
                "url": "https://example.com/a",
                "best_paragraph_index": 0,
                "keyword": "call center software",
                "traffic": 12,
                "position": 3,
                "status": "matched",
            }
        ]
    }

    payload = build_winning_paragraphs(impact, ablation, attribution)

    row = payload["rows"][0]
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["topic_carriers"] == 1
    assert row["recommended_action"] == "protect_and_expand"
    assert row["recommended_action_label"] == "Protect and expand"
    assert row["attributed_keyword_count"] == 1
    assert row["attributed_keyword_traffic"] == 12
    assert row["alignment_delta"] == 0.04


def test_winning_paragraphs_recommends_contextual_links_for_weak_link_support():
    impact = {
        "top_paragraphs": [
            {
                "url": "https://example.com/b",
                "title": "Example B",
                "paragraph_index": 2,
                "paragraph_excerpt": "Ticketing software keeps support queues organized.",
                "impact_tier": "medium",
                "impact_score": 72.0,
                "attributed_traffic": 9.0,
                "components": {
                    "heading_match": 0.9,
                    "link_context": 0.1,
                    "freshness": 1.0,
                },
            }
        ]
    }

    payload = build_winning_paragraphs(impact, {}, {})

    assert payload["rows"][0]["recommended_action"] == "add_contextual_links"
    assert payload["summary"]["actions"]["add_contextual_links"] == 1
