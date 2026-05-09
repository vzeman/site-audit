from pathlib import Path
import json
import zipfile

import numpy as np

from site_audit.compare import build_payload, package_comparison


def _write_report(
    root: Path,
    domain: str,
    metrics: dict,
    extras: dict[str, str],
    pages: list[dict] | None = None,
) -> None:
    report_dir = root / domain / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "site_metrics.json").write_text(
        __import__("json").dumps({
            "domain": domain,
            "model": "test-model",
            "page_count": metrics.get("page_count", 10),
            **metrics,
        }),
        encoding="utf-8",
    )
    (report_dir / "pages.json").write_text(json.dumps(pages or []), encoding="utf-8")
    for name, payload in extras.items():
        (report_dir / name).write_text(payload, encoding="utf-8")


def _write_semantic_cache(root: Path, domain: str) -> None:
    cache_dir = root / domain / "cache" / "ahrefs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paragraph = (
        "What is support automation? It routes tickets, answers common questions, and gives teams a repeatable workflow."
        if domain.startswith("strong")
        else "Support automation software helps teams organize requests and improve customer service workflows."
    )
    rows = [
        {"type": "page", "label": f"{domain} page", "url": f"https://{domain}/a", "cluster": "support", "traffic": 20, "size": 20},
        {"type": "link_title", "label": f"{domain} anchor", "count": 4, "size": 4},
        {"type": "header", "label": f"{domain} H1", "url": f"https://{domain}/a", "cluster": "support", "level": 1, "traffic": 20, "size": 10},
        {"type": "page_title", "label": f"{domain} title", "url": f"https://{domain}/a", "cluster": "support", "traffic": 20, "size": 20},
        {"type": "paragraph", "label": paragraph, "url": f"https://{domain}/a", "cluster": "support", "paragraph_index": 0, "traffic": 20, "size": 20},
        {"type": "keyword", "label": f"{domain} support automation", "url": f"https://{domain}/a", "cluster": "support", "traffic": 12, "volume": 200, "position": 4, "size": 200},
    ]
    (cache_dir / "semantic_entities_test_model.meta.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )
    np.savez_compressed(
        cache_dir / "semantic_entities_test_model.npz",
        embeddings=np.array([[0.9, 0.1], [1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.3, 0.9], [0.8, 0.2]], dtype=np.float32),
    )


def _write_embedding_cache(root: Path, domain: str, urls: list[str]) -> None:
    cache_dir = root / domain / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    embs = np.eye(len(urls), 4, dtype=np.float32)
    np.savez_compressed(cache_dir / "embeddings_test_model.npz", urls=np.array(urls), embeddings=embs)


def test_compare_payload_includes_scorecards_metric_groups_and_gaps(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        "strong.example",
        {
            "page_count": 10,
            "site_focus_score": 0.8,
            "site_focus_score_calibrated": 0.7,
            "site_radius": 0.05,
            "section_coherence": {"ratio": 1.2},
            "topic_dimension": {"effective_dim": 80},
        },
        {
            "answerability.json": '[{"score":7},{"score":8}]',
            "structured_data.json": '{"summary":{"schema_coverage":1,"invalid_jsonld_blocks":0,"schema_type_count":5,"schema_opportunities":1,"high_priority_schema_opportunities":0},"top_types":[{"type":"Article","pages":5}],"opportunities":[{"url":"https://strong.example/a","title":"A","schema_type":"FAQPage","priority":"medium","reason":"FAQ content","missing_evidence":["visible answers"],"missing_recommended_properties":["mainEntity"],"guideline_url":"https://developers.google.com/search/docs/appearance/structured-data/faqpage"}],"clusters":[{"cluster":"support","pages":3,"traffic":100,"schema_coverage":1,"opportunities":1,"invalid_blocks":0,"top_schema_types":[{"type":"Article","pages":3}]}]}',
            "trust_signals.json": '{"summary":{"avg_trust_score":82,"high_priority_pages":0,"missing_evidence_items":1},"clusters":[{"cluster":"support","pages":3,"traffic":100,"avg_trust_score":82,"leader_score":90}],"missing_evidence":[{"url":"https://strong.example/a","title":"A","cluster":"support","priority":"medium","missing_signal":"Add citations","trust_score":70,"traffic":100}],"pages":[{"url":"https://strong.example/a","title":"A","priority":"low","trust_score":82,"traffic":100}]}',
            "linkgraph.json": '{"page_link_counts":[{"in_degree":4,"out_degree":6},{"in_degree":6,"out_degree":8}]}',
            "linkbuilding.json": '{"summary":{"descriptive_anchor_share":0.8,"generic_anchor_share":0.1}}',
            "metadata_quality.json": '{"summary":{"issue_share":0.1}}',
            "freshness.json": '{"summary":{"date_coverage":1,"stale_share":0.1}}',
            "entities.json": '{"summary":{"entity_coverage":1,"topical_authority_score":80}}',
            "paragraph_density.json": '{"summary":{"zero_link_share":0.2,"spammy_count":1}}',
            "indexability.json": '{"summary":{"indexable_share":1}}',
            "media_accessibility.json": '{"summary":{"issue_share":0.05}}',
            "performance.json": '{"summary":{"median_html_weight_bytes":1000,"heavy_page_share":0.1,"render_blocking_share":0.2}}',
            "header_analysis.json": '{"summary":{"total_pages":10,"pages_missing_h1":0,"pages_multi_h1":0}}',
            "conversion.json": '{"summary":{"cta_coverage":1,"primary_cta_coverage":0.9,"form_coverage":0.8}}',
            "conversion_balance.json": '{"summary":{"conversion_efficiency":0.9,"high_risk_money_pages":0,"avg_seo_support":70,"avg_conversion_support":80},"clusters":[{"cluster":"support","pages":2,"traffic":100,"avg_seo_support":70,"avg_conversion_support":80,"high_risk":0}],"pages":[{"url":"https://strong.example/a","title":"A","cluster":"support","traffic":100,"money_page":true,"seo_support":70,"conversion_support":80,"balance_label":"balanced"}],"high_traffic_weak_conversion":[]}',
            "answer_blocks.json": '{"summary":{"top_query_clusters":2,"strong_query_clusters":2,"opportunity_queries":0,"strong_blocks":6},"clusters":[{"label":"support","queries":4,"traffic":100,"strong_query_share":1,"avg_best_score":82,"opportunity_queries":0,"recommended_format":"","status":"strong"}]}',
            "cannibalization.json": '{"summary":{"page_conflicts":1,"paragraph_conflicts":2,"traffic_at_risk":15},"page_conflicts":[{"classification":"duplicate_competing_page","traffic_at_risk":15,"traffic":100,"label":"support","preferred_winner_url":"https://strong.example/a"}]}',
            "duplicate_fragments.json": '{"summary":{"groups":2,"strong_patterns":1,"harmful_boilerplate":1},"groups":[{"classification":"strong_reusable_pattern","count":1,"attributed_traffic":50,"page_traffic_sum":100}]}',
            "template_patterns.json": '{"summary":{"patterns":2,"recommendations":1,"segments_compared":1,"median_confidence":0.7},"patterns":[{"feature_key":"primary_cta","label":"Primary CTA","category":"conversion","page_type":"article","observed_lift":1.2,"confidence":0.7,"sample_size":8,"recommendation":"Add a primary CTA block after the main answer.","sample_urls":[{"url":"https://strong.example/a","title":"A"}],"affected_weak_pages":[{"url":"https://strong.example/weak","title":"Weak"}]}],"recommendations":[{"url":"https://strong.example/weak","title":"Weak","feature_key":"primary_cta","page_type":"article","missing_pattern":"Primary CTA","confidence":0.7,"observed_lift":1.2}]}',
            "ahrefs.json": '{"summary":{"top_pages":2,"organic_keywords":10,"matched_traffic":1000,"matched_traffic_share":1,"top_pages_value_usd":250},"metrics":{"org_traffic":1200,"org_keywords":20,"org_keywords_1_3":5},"organic_keywords":[{"keyword":"support automation","cluster_label":"support","matched_url":"https://strong.example/a","page_title":"Support automation","position":1,"traffic":80,"volume":1000,"serp_features":["question"]}],"clusters":[{"key":"support","label":"support","traffic":100,"keyword_rows":1,"keywords_total":1,"top3_keywords":1,"top_pages":[{"matched_url":"https://strong.example/a","title":"Support automation"}],"top_keywords":[{"keyword":"support automation","traffic":80}]}]}',
        },
    )
    _write_semantic_cache(tmp_path, "strong.example")
    _write_report(
        tmp_path,
        "weak.example",
        {
            "page_count": 10,
            "site_focus_score": 0.5,
            "site_focus_score_calibrated": 0.3,
            "site_radius": 0.15,
            "section_coherence": {"ratio": 0.8},
            "topic_dimension": {"effective_dim": 40},
        },
        {
            "answerability.json": '[{"score":2},{"score":3}]',
            "structured_data.json": '{"summary":{"schema_coverage":0.2,"invalid_jsonld_blocks":3,"schema_type_count":1,"schema_opportunities":4,"high_priority_schema_opportunities":2},"top_types":[{"type":"Article","pages":1}],"opportunities":[{"url":"https://weak.example/a","title":"A","schema_type":"Article","priority":"high","reason":"Article page missing primary schema","missing_evidence":["published or modified date"],"missing_recommended_properties":["author"],"guideline_url":"https://developers.google.com/search/docs/appearance/structured-data/article"}],"clusters":[{"cluster":"support","pages":3,"traffic":8,"schema_coverage":0.2,"opportunities":4,"invalid_blocks":3,"top_schema_types":[{"type":"Article","pages":1}]}],"invalid_blocks":[{"url":"https://weak.example/bad","title":"Bad","error":"Expecting property name"}]}',
            "trust_signals.json": '{"summary":{"avg_trust_score":38,"high_priority_pages":2,"missing_evidence_items":5},"clusters":[{"cluster":"support","pages":3,"traffic":8,"avg_trust_score":38,"leader_score":45}],"missing_evidence":[{"url":"https://weak.example/a","title":"A","cluster":"support","priority":"high","missing_signal":"Add author","trust_score":30,"traffic":8}],"pages":[{"url":"https://weak.example/a","title":"A","priority":"high","trust_score":30,"traffic":8}]}',
            "linkgraph.json": '{"page_link_counts":[{"in_degree":0,"out_degree":1},{"in_degree":1,"out_degree":1}]}',
            "linkbuilding.json": '{"summary":{"descriptive_anchor_share":0.2,"generic_anchor_share":0.6}}',
            "metadata_quality.json": '{"summary":{"issue_share":0.8}}',
            "freshness.json": '{"summary":{"date_coverage":0.2,"stale_share":0.8}}',
            "entities.json": '{"summary":{"entity_coverage":0.3,"topical_authority_score":20}}',
            "paragraph_density.json": '{"summary":{"zero_link_share":0.9,"spammy_count":10}}',
            "indexability.json": '{"summary":{"indexable_share":0.6}}',
            "media_accessibility.json": '{"summary":{"issue_share":0.5}}',
            "performance.json": '{"summary":{"median_html_weight_bytes":5000,"heavy_page_share":0.8,"render_blocking_share":1}}',
            "header_analysis.json": '{"summary":{"total_pages":10,"pages_missing_h1":5,"pages_multi_h1":3}}',
            "conversion.json": '{"summary":{"cta_coverage":0.2,"primary_cta_coverage":0.1,"form_coverage":0}}',
            "conversion_balance.json": '{"summary":{"conversion_efficiency":0.1,"high_risk_money_pages":2,"avg_seo_support":65,"avg_conversion_support":20},"clusters":[{"cluster":"support","pages":2,"traffic":80,"avg_seo_support":65,"avg_conversion_support":20,"high_risk":2}],"pages":[{"url":"https://weak.example/a","title":"A","cluster":"support","traffic":80,"money_page":true,"seo_support":65,"conversion_support":20,"balance_label":"high_risk_money_page"}],"high_traffic_weak_conversion":[{"url":"https://weak.example/a","title":"A","cluster":"support","traffic":80,"money_page":true,"seo_support":65,"conversion_support":20,"balance_label":"high_risk_money_page"}]}',
            "answer_blocks.json": '{"summary":{"top_query_clusters":2,"strong_query_clusters":0,"opportunity_queries":5,"strong_blocks":1},"clusters":[{"label":"support","queries":3,"traffic":80,"strong_query_share":0,"avg_best_score":42,"opportunity_queries":3,"recommended_format":"faq","status":"gap"}]}',
            "cannibalization.json": '{"summary":{"page_conflicts":3,"paragraph_conflicts":4,"traffic_at_risk":90},"page_conflicts":[{"classification":"consolidation_candidate","traffic_at_risk":90,"traffic":120,"label":"support","preferred_winner_url":"https://weak.example/a"}]}',
            "duplicate_fragments.json": '{"summary":{"groups":5,"strong_patterns":0,"harmful_boilerplate":4},"groups":[{"classification":"harmful_boilerplate","count":4,"attributed_traffic":0,"page_traffic_sum":0}]}',
            "template_patterns.json": '{"summary":{"patterns":0,"recommendations":4,"segments_compared":1,"median_confidence":0},"patterns":[],"recommendations":[{"url":"https://weak.example/a","title":"A","feature_key":"primary_cta","page_type":"article","missing_pattern":"Primary CTA","recommendation":"Add a primary CTA block after the main answer.","traffic":80,"keywords":3,"confidence":0.6,"observed_lift":1.0}]}',
            "ahrefs.json": '{"summary":{"top_pages":2,"organic_keywords":3,"matched_traffic":100,"matched_traffic_share":0.5,"top_pages_value_usd":20},"metrics":{"org_traffic":150,"org_keywords":4,"org_keywords_1_3":0},"organic_keywords":[{"keyword":"support automation","cluster_label":"support","matched_url":"https://weak.example/a","page_title":"Support automation","position":8,"traffic":8,"volume":1000,"serp_features":["question"]}],"clusters":[{"key":"support","label":"support","traffic":8,"keyword_rows":1,"keywords_total":1,"top3_keywords":0,"top_pages":[{"matched_url":"https://weak.example/a","title":"Support automation"}],"top_keywords":[{"keyword":"support automation","traffic":8}]}]}',
        },
    )
    _write_semantic_cache(tmp_path, "weak.example")

    payload = build_payload(["strong.example", "weak.example"], tmp_path)

    assert payload["metric_groups"]
    assert payload["scorecards"][0]["domain"] == "strong.example"
    assert payload["scorecards"][0]["overall_score"] > payload["scorecards"][1]["overall_score"]
    assert payload["biggest_gaps"]
    assert any(gap["winner"] == "strong.example" for gap in payload["biggest_gaps"])
    assert any(group["key"] == "search" for group in payload["metric_groups"])
    assert payload["leaderboard"][0]["ahrefs_org_traffic"] == 1200
    assert payload["semantic_entity_maps"]["link_titles"]["total"] == 2
    assert payload["semantic_entity_maps"]["headers"]["total"] == 2
    assert payload["semantic_entity_maps"]["page_titles"]["total"] == 2
    assert payload["semantic_entity_maps"]["pages"]["total"] == 2
    assert payload["semantic_entity_maps"]["paragraphs"]["total"] == 2
    assert payload["semantic_entity_maps"]["keywords"]["total"] == 2
    assert "pagerank" in payload["ahrefs_semantic_scatter"][0]
    assert "freshness_bucket" in payload["ahrefs_semantic_scatter"][0]
    assert "keyword_gaps" in payload
    assert "serp_features" in payload
    assert "content_efficiency" in payload
    assert "authority_demand" in payload
    assert "traffic_readiness" in payload
    assert payload["answer_blocks"]["clusters"][0]["cluster"] == "support"
    assert payload["answer_blocks"]["clusters"][0]["domains"][0]["strong_query_share"] == 1
    assert payload["cannibalization"]["classes"]
    assert payload["leaderboard"][1]["cannibalization_page_conflicts"] == 3
    assert payload["duplicate_fragments"]["classes"]
    assert payload["leaderboard"][1]["duplicate_fragment_groups"] == 5
    assert payload["template_patterns"]["features"]
    assert payload["leaderboard"][0]["template_success_patterns"] == 2
    assert payload["leaderboard"][1]["template_pattern_recommendations"] == 4
    assert payload["pattern_transplants"]["recommendations"]
    assert payload["pattern_transplants"]["recommendations"][0]["source_domain"] == "strong.example"
    assert payload["pattern_transplants"]["recommendations"][0]["target_domain"] == "weak.example"
    assert payload["pattern_transplants"]["recommendations"][0]["pattern_type"] == "template"
    assert payload["pattern_transplants"]["coverage"]
    assert payload["pattern_transplants"]["domains"][1]["recommendations"] >= 1
    assert payload["structured_data_opportunities"]["types"]
    assert payload["structured_data_opportunities"]["clusters"]
    assert payload["leaderboard"][1]["schema_opportunities"] == 4
    assert payload["trust_signals"]["clusters"]
    assert payload["leaderboard"][0]["trust_avg_score"] == 82
    assert payload["leaderboard"][1]["trust_high_priority_pages"] == 2
    assert payload["conversion_balance"]["clusters"]
    assert payload["leaderboard"][0]["conversion_balance_efficiency"] == 0.9
    assert payload["leaderboard"][1]["conversion_balance_high_risk"] == 2
    assert payload["keyword_cluster_gaps"]["clusters"]
    assert payload["keyword_cluster_gaps"]["clusters"][0]["leader_domain"] == "strong.example"
    assert payload["keyword_cluster_gaps"]["recommendations"]
    assert payload["keyword_cluster_gaps"]["cache"]["entries"] >= 2
    assert payload["strongest_clusters"]["leaderboard"]
    assert payload["strongest_clusters"]["leaderboard"][0]["winner_domain"] == "strong.example"
    assert payload["strongest_clusters"]["matrix"]
    assert payload["strongest_clusters"]["semantic_points"]
    assert payload["strongest_clusters"]["clusters"][0]["domains"][0]["keywords"]
    assert payload["winning_patterns"]["patterns"]
    first_pattern = payload["winning_patterns"]["patterns"][0]
    assert first_pattern["source_evidence"]
    assert first_pattern["target_recommendations"][0]["priority_score"] > 0
    assert first_pattern["target_recommendations"][0]["confidence"] > 0
    assert payload["winning_patterns"]["coverage"]
    assert payload["keyword_content_matrix"]["matrix"]
    matrix_cell = payload["keyword_content_matrix"]["matrix"][0]["domains"][1]
    assert matrix_cell["components"]["headings"] >= 0
    assert matrix_cell["missing"]
    assert any(cell["recommendations"] for cell in payload["keyword_content_matrix"]["cells"])
    assert payload["paragraph_archetypes"]["matrix"]
    assert payload["paragraph_archetypes"]["timelines"]
    assert payload["paragraph_archetypes"]["recommendations"][0]["source_examples"]
    assert payload["seo_playbooks"]["playbooks"]
    assert payload["seo_playbooks"]["playbooks"][0]["implementation_steps"]
    assert payload["seo_playbooks"]["playbooks"][0]["acceptance_criteria"]
    assert payload["seo_playbooks"]["playbooks"][0]["validation_metric"]


def test_compare_payload_includes_competitive_opportunity_sections(tmp_path: Path) -> None:
    alpha_url = "https://alpha.example/blog/ai-workflow"
    beta_url = "https://beta.example/blog/automation-tool"
    _write_report(
        tmp_path,
        "alpha.example",
        {"page_count": 2},
        {
            "answerability.json": json.dumps([{"url": alpha_url, "score": 2.0}]),
            "structured_data.json": json.dumps({"per_page": [{"url": alpha_url, "types": [], "valid_blocks": 0, "invalid_blocks": 0}]}),
            "metadata_quality.json": json.dumps({"per_page": [{"url": alpha_url, "title": "AI workflow", "issues": ["missing_description"]}]}),
            "freshness.json": json.dumps({"per_page": [{"url": alpha_url, "bucket": "very_stale", "age_days": 800, "issues": ["very_stale"]}]}),
            "conversion.json": json.dumps({"per_page": [{"url": alpha_url, "cta_count": 0, "primary_cta_count": 0, "form_count": 0, "lead_page": True}]}),
            "linkgraph.json": json.dumps({
                "page_link_counts": [{"url": alpha_url, "title": "AI workflow", "in_degree": 0, "out_degree": 3, "click_depth": 4}],
                "top_authority_pages": [{"url": "https://alpha.example/", "title": "Alpha", "pagerank": 0.9, "authority_score": 0.8, "in_degree": 8, "out_degree": 10}],
                "traffic_weighted_pagerank": {
                    "summary": {"authority_traffic_alignment": 0.35, "high_traffic_low_authority_pages": 1, "high_authority_low_value_pages": 0, "orphan_traffic_share": 1.0},
                    "pages": [{"url": alpha_url, "title": "AI workflow", "section": "blog", "cluster": "workflow", "directory": "/blog/", "page_type": "article", "traffic": 100, "keywords": 8, "top_keyword": "ai workflow", "in_degree": 0, "out_degree": 3, "click_depth": 4, "pagerank": 0.01, "weighted_pagerank": 0.01, "traffic_weighted_pagerank": 0.02, "traffic_percentile": 1.0, "weighted_pagerank_percentile": 0.2, "authority_traffic_gap": 0.8, "mismatch_label": "ranked_orphan"}],
                    "clusters": [{"cluster": "workflow", "label": "workflow", "pages": 1, "traffic": 100, "avg_authority_traffic_gap": 0.8, "underserved_pages": 1, "authority_without_demand": 0}],
                },
                "link_removal_simulation": {
                    "summary": {"critical_links": 1, "useful_links": 0, "redundant_links": 0, "irrelevant_links": 0, "potentially_harmful_links": 0, "template_navigation_links": 0, "simulated_edges": 1},
                    "critical_links": [{"source_url": "https://alpha.example/", "source_title": "Alpha", "target_url": alpha_url, "target_title": "AI workflow", "anchor_samples": ["AI workflow"], "removal_loss_score": 91.0, "classification": "critical", "placement": "contextual", "target_traffic": 100}],
                    "weak_or_harmful_links": [],
                    "edit_warnings": [{"source_url": "https://alpha.example/", "source_title": "Alpha", "critical_links": 1, "max_loss_score": 91.0}],
                },
                "link_addition_simulation": {
                    "summary": {"total_recommendations": 1, "high_priority": 1, "medium_priority": 0, "avg_expected_benefit": 82.0, "density_safe_recommendations": 1},
                    "recommendations": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "target_url": alpha_url, "target_title": "AI workflow", "paragraph_index": 1, "paragraph_excerpt": "Useful paragraph", "suggested_anchor": "AI workflow", "expected_benefit_score": 82.0, "priority": "high", "current_target_in_degree": 0, "after_target_in_degree": 1, "score_components": {"authority_flow": 90, "relevance": 80, "opportunity": 70, "internal_link_deficit": 90}}],
                    "patterns": [{"pattern": "guide -> blog", "count": 1}],
                },
                "anchor_relevance": {
                    "summary": {"total_internal_links": 2, "avg_score": 82, "descriptive_rate": 1.0, "weak_links": 0},
                    "weak_links": [],
                    "by_target_directory": [{"target_directory": "/blog/", "links": 2, "avg_score": 82, "descriptive_rate": 1.0, "weak_links": 0}],
                },
                "contextual_link_impact": {
                    "summary": {"total_links": 2, "avg_contextual_impact": 78, "main_content_links": 2, "high_impact_contextual_links": 1, "template_links": 0},
                    "top_contextual_links": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "target_url": alpha_url, "target_title": "AI workflow", "paragraph_index": 1, "paragraph_excerpt": "Useful paragraph", "contextual_link_impact": 88, "contextual_similarity": 0.8, "structural_authority_score": 70}],
                    "source_pages": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "avg_contextual_impact": 78, "main_content_links": 2, "template_links": 0}],
                },
                "internal_link_patterns": {
                    "summary": {"patterns": 1, "recommendations": 1, "avg_confidence": 0.74, "total_links": 4, "page_types_with_patterns": 1},
                    "patterns": [{"pattern_id": "link_pattern_1", "rule_key": "blog_post|product|keyword_phrase|main_content|cross_cluster|cross_directory|deeper|close", "inferred_rule": "blog posts link to product pages", "source_page_type": "blog_post", "target_page_type": "product", "support_count": 3, "confidence": 0.74, "sample_links": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "target_url": alpha_url, "target_title": "AI workflow", "anchor": "AI workflow"}]}],
                    "recommendations": [{"pattern_id": "link_pattern_1", "source_url": alpha_url, "source_title": "AI workflow", "missing_pattern": "blog posts link to product pages", "suggested_anchor": "AI workflow", "confidence": 0.74, "lift_score_difference": 20}],
                },
                "high_demand_low_link": {
                    "summary": {"demand_support_alignment": 0.42, "classified_top_pages": 1, "high_demand_low_support_pages": 1, "opportunity_pages": 1, "high_demand_low_support_traffic": 100, "opportunity_traffic": 100, "source_candidates": 1},
                    "pages": [{"url": alpha_url, "title": "AI workflow", "section": "blog", "cluster": "workflow", "directory": "/blog/", "page_type": "article", "traffic": 100, "keywords": 8, "volume": 1300, "top_keyword": "ai workflow", "demand_score": 92, "support_score": 18, "demand_support_gap": 74, "opportunity_score": 130, "classification": "high_demand_low_support", "source_candidates": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "source_cluster": "guide", "suggested_anchor": "AI workflow", "expected_benefit_score": 82}], "suggested_anchors": ["AI workflow"], "missing_source_clusters": [{"cluster": "guide", "candidate_sources": 1}]}],
                    "opportunities": [{"url": alpha_url, "title": "AI workflow", "section": "blog", "cluster": "workflow", "directory": "/blog/", "page_type": "article", "traffic": 100, "keywords": 8, "volume": 1300, "top_keyword": "ai workflow", "demand_score": 92, "support_score": 18, "demand_support_gap": 74, "opportunity_score": 130, "classification": "high_demand_low_support", "source_candidates": [{"source_url": "https://alpha.example/guide", "source_title": "Guide", "source_cluster": "guide", "suggested_anchor": "AI workflow", "expected_benefit_score": 82}], "suggested_anchors": ["AI workflow"], "missing_source_clusters": [{"cluster": "guide", "candidate_sources": 1}]}],
                    "directories": [{"directory": "/blog/", "label": "/blog/", "pages": 1, "classified_top_pages": 1, "traffic": 100, "opportunities": 1, "opportunity_traffic": 100, "avg_demand_score": 92, "avg_support_score": 18, "avg_demand_support_gap": 74}],
                    "clusters": [{"cluster": "workflow", "label": "workflow", "pages": 1, "classified_top_pages": 1, "traffic": 100, "opportunities": 1, "opportunity_traffic": 100, "avg_demand_score": 92, "avg_support_score": 18, "avg_demand_support_gap": 74}],
                },
                "hub_bottlenecks": {
                    "summary": {"architecture_resilience": 0.9, "bottleneck_pages": 0, "bridge_pages": 1, "authority_hubs": 1, "dead_end_risks": 0, "orphan_risks": 0},
                    "risks": [{"url": "https://alpha.example/hub", "title": "Hub", "role": "cluster_bridge", "resilience_risk": 55, "cluster_bridge_count": 2, "affected_clusters": ["guide", "blog"], "in_degree": 3, "out_degree": 4}],
                    "cluster_edges": [{"source_cluster": "guide", "target_cluster": "blog", "bridge_pages": 1}],
                },
            }),
            "ahrefs.json": json.dumps({
                "top_pages": [
                    {"url": alpha_url, "matched_url": alpha_url, "title": "AI workflow", "traffic": 100, "keywords": 8, "top_keyword": "ai workflow", "section": "blog"}
                ],
                "organic_keywords": [
                    {"keyword": "ai workflow", "url": alpha_url, "matched_url": alpha_url, "page_title": "AI workflow", "position": 2, "traffic": 70, "volume": 1000, "intents": ["informational"], "serp_features": ["ai_overview", "question"]},
                    {"keyword": "alpha only", "url": alpha_url, "matched_url": alpha_url, "page_title": "AI workflow", "position": 4, "traffic": 20, "volume": 300, "intents": ["commercial"], "serp_features": ["video_th"]},
                ],
                "clusters": [{"key": "c1", "label": "ai workflow", "traffic": 100, "pages": 1, "matched_pages": 1, "keywords_total": 8, "keyword_rows": 2, "top3_keywords": 1, "top_keywords": [{"keyword": "ai workflow", "traffic": 70}]}],
                "directories": [{"key": "blog", "label": "blog", "traffic": 100, "pages": 1, "matched_pages": 1, "keywords_total": 8, "keyword_rows": 2, "top3_keywords": 1}],
            }),
            "best_pages.json": json.dumps({
                "summary": {"status": "ok", "pages": 1, "clusters": 1, "causation_note": "Observed correlations, not confirmed causation."},
                "pages": [{
                    "url": alpha_url,
                    "title": "AI workflow",
                    "cluster_label": "ai workflow",
                    "traffic": 100,
                    "keywords": 8,
                    "performance_score": 82,
                    "copy_recommendations": ["Promote the page from related workflow hubs."],
                    "strengths": [{"category": "Internal Links", "label": "Critical links", "evidence": "Home page passes authority."}],
                    "weak_spots": [],
                }],
            }),
        },
        pages=[{"url": alpha_url, "title": "AI workflow", "section": "blog"}],
    )
    _write_report(
        tmp_path,
        "beta.example",
        {"page_count": 1},
        {
            "answerability.json": json.dumps([{"url": beta_url, "score": 8.0}]),
            "structured_data.json": json.dumps({"per_page": [{"url": beta_url, "types": ["Article"], "valid_blocks": 1, "invalid_blocks": 0}]}),
            "metadata_quality.json": json.dumps({"per_page": [{"url": beta_url, "title": "Automation tool", "issues": []}]}),
            "freshness.json": json.dumps({"per_page": [{"url": beta_url, "bucket": "fresh", "age_days": 20, "issues": []}]}),
            "conversion.json": json.dumps({"per_page": [{"url": beta_url, "cta_count": 2, "primary_cta_count": 1, "form_count": 0, "lead_page": True}]}),
            "linkgraph.json": json.dumps({
                "page_link_counts": [{"url": beta_url, "title": "Automation tool", "in_degree": 5, "out_degree": 2, "click_depth": 2}],
                "top_authority_pages": [{"url": beta_url, "title": "Automation tool", "pagerank": 0.5, "authority_score": 0.6, "in_degree": 5, "out_degree": 2}],
                "traffic_weighted_pagerank": {
                    "summary": {"authority_traffic_alignment": 0.9, "high_traffic_low_authority_pages": 0, "high_authority_low_value_pages": 0, "orphan_traffic_share": 0.0},
                    "pages": [{"url": beta_url, "title": "Automation tool", "section": "blog", "cluster": "automation", "directory": "/blog/", "page_type": "article", "traffic": 15, "keywords": 2, "top_keyword": "automation tool", "in_degree": 5, "out_degree": 2, "click_depth": 2, "pagerank": 0.5, "weighted_pagerank": 0.5, "traffic_weighted_pagerank": 0.6, "traffic_percentile": 1.0, "weighted_pagerank_percentile": 1.0, "authority_traffic_gap": 0.0, "mismatch_label": "aligned"}],
                    "clusters": [{"cluster": "automation", "label": "automation", "pages": 1, "traffic": 15, "avg_authority_traffic_gap": 0.0, "underserved_pages": 0, "authority_without_demand": 0}],
                },
                "link_removal_simulation": {
                    "summary": {"critical_links": 0, "useful_links": 1, "redundant_links": 0, "irrelevant_links": 0, "potentially_harmful_links": 0, "template_navigation_links": 0, "simulated_edges": 1},
                    "critical_links": [],
                    "weak_or_harmful_links": [],
                    "edit_warnings": [],
                },
                "link_addition_simulation": {
                    "summary": {"total_recommendations": 1, "high_priority": 0, "medium_priority": 1, "avg_expected_benefit": 55.0, "density_safe_recommendations": 1},
                    "recommendations": [{"source_url": "https://beta.example/guide", "source_title": "Guide", "target_url": beta_url, "target_title": "Automation tool", "paragraph_index": 1, "paragraph_excerpt": "Useful paragraph", "suggested_anchor": "Automation tool", "expected_benefit_score": 55.0, "priority": "medium", "current_target_in_degree": 5, "after_target_in_degree": 6, "score_components": {"authority_flow": 50, "relevance": 60, "opportunity": 40, "internal_link_deficit": 20}}],
                    "patterns": [{"pattern": "guide -> blog", "count": 1}],
                },
                "anchor_relevance": {
                    "summary": {"total_internal_links": 2, "avg_score": 35, "descriptive_rate": 0.5, "weak_links": 1},
                    "weak_links": [{"source_url": "https://beta.example/guide", "source_title": "Guide", "target_url": beta_url, "target_title": "Automation tool", "anchor": "click here", "suggested_anchor": "Automation tool", "score": 22, "label": "vague"}],
                    "by_target_directory": [{"target_directory": "/blog/", "links": 2, "avg_score": 35, "descriptive_rate": 0.5, "weak_links": 1}],
                },
                "contextual_link_impact": {
                    "summary": {"total_links": 2, "avg_contextual_impact": 40, "main_content_links": 1, "high_impact_contextual_links": 0, "template_links": 1},
                    "top_contextual_links": [{"source_url": "https://beta.example/guide", "source_title": "Guide", "target_url": beta_url, "target_title": "Automation tool", "paragraph_index": 1, "paragraph_excerpt": "Useful paragraph", "contextual_link_impact": 45, "contextual_similarity": 0.5, "structural_authority_score": 35}],
                    "source_pages": [{"source_url": "https://beta.example/guide", "source_title": "Guide", "avg_contextual_impact": 40, "main_content_links": 1, "template_links": 1}],
                },
                "internal_link_patterns": {
                    "summary": {"patterns": 0, "recommendations": 1, "avg_confidence": 0.0, "total_links": 2, "page_types_with_patterns": 0},
                    "patterns": [],
                    "recommendations": [{"pattern_id": "missing", "rule_key": "blog_post|product|keyword_phrase|main_content|cross_cluster|cross_directory|deeper|close", "source_url": beta_url, "source_title": "Automation tool", "source_page_type": "blog_post", "missing_pattern": "blog posts link to product pages", "recommended_action": "Add a contextual link from this blog post to the product page.", "suggested_anchor": "Automation tool", "suggested_targets": [{"url": "https://beta.example/product", "title": "Product", "traffic": 10, "keywords": 1}], "confidence": 0.5, "lift_score_difference": 12, "traffic": 15, "keywords": 2}],
                },
                "high_demand_low_link": {
                    "summary": {"demand_support_alignment": 0.88, "classified_top_pages": 1, "high_demand_low_support_pages": 0, "opportunity_pages": 0, "high_demand_low_support_traffic": 0, "opportunity_traffic": 0, "source_candidates": 0},
                    "pages": [{"url": beta_url, "title": "Automation tool", "section": "blog", "cluster": "automation", "directory": "/blog/", "page_type": "article", "traffic": 15, "keywords": 2, "volume": 500, "top_keyword": "automation tool", "demand_score": 58, "support_score": 72, "demand_support_gap": 0, "opportunity_score": 0, "classification": "supported_demand", "source_candidates": [], "suggested_anchors": ["Automation tool"], "missing_source_clusters": []}],
                    "opportunities": [],
                    "directories": [{"directory": "/blog/", "label": "/blog/", "pages": 1, "classified_top_pages": 1, "traffic": 15, "opportunities": 0, "opportunity_traffic": 0, "avg_demand_score": 58, "avg_support_score": 72, "avg_demand_support_gap": 0}],
                    "clusters": [{"cluster": "automation", "label": "automation", "pages": 1, "classified_top_pages": 1, "traffic": 15, "opportunities": 0, "opportunity_traffic": 0, "avg_demand_score": 58, "avg_support_score": 72, "avg_demand_support_gap": 0}],
                },
                "hub_bottlenecks": {
                    "summary": {"architecture_resilience": 0.4, "bottleneck_pages": 2, "bridge_pages": 0, "authority_hubs": 0, "dead_end_risks": 1, "orphan_risks": 1},
                    "risks": [{"url": "https://beta.example/hub", "title": "Hub", "role": "bottleneck", "resilience_risk": 88, "cluster_bridge_count": 1, "affected_clusters": ["guide", "blog"], "in_degree": 1, "out_degree": 1}],
                    "cluster_edges": [{"source_cluster": "guide", "target_cluster": "blog", "bridge_pages": 1}],
                },
            }),
            "ahrefs.json": json.dumps({
                "top_pages": [
                    {"url": beta_url, "matched_url": beta_url, "title": "Automation tool", "traffic": 15, "keywords": 2, "top_keyword": "automation tool", "section": "blog"}
                ],
                "organic_keywords": [
                    {"keyword": "automation tool", "url": beta_url, "matched_url": beta_url, "page_title": "Automation tool", "position": 6, "traffic": 15, "volume": 500, "intents": ["commercial"], "serp_features": ["question"]}
                ],
                "clusters": [{"key": "c2", "label": "automation", "traffic": 15, "pages": 1, "matched_pages": 1, "keywords_total": 2, "keyword_rows": 1, "top3_keywords": 0}],
                "directories": [{"key": "blog", "label": "blog", "traffic": 15, "pages": 1, "matched_pages": 1, "keywords_total": 2, "keyword_rows": 1, "top3_keywords": 0}],
            }),
            "best_pages.json": json.dumps({
                "summary": {"status": "ok", "pages": 1, "clusters": 1, "causation_note": "Observed correlations, not confirmed causation."},
                "pages": [{
                    "url": beta_url,
                    "title": "Automation tool",
                    "cluster_label": "ai workflow",
                    "traffic": 15,
                    "keywords": 2,
                    "performance_score": 48,
                    "copy_recommendations": [],
                    "strengths": [],
                    "weak_spots": [{"category": "Internal Links", "label": "Weak anchors", "evidence": "Anchors are vague."}],
                }],
            }),
        },
        pages=[{"url": beta_url, "title": "Automation tool", "section": "blog"}],
    )

    payload = build_payload(["alpha.example", "beta.example"], tmp_path)

    assert any(
        row["domain"] == "beta.example" and row["keyword"] == "ai workflow" and row["gap_type"] == "missing"
        for row in payload["keyword_gaps"]["opportunities"]
    )
    assert payload["serp_features"]["matrix"][0]["feature"] in {"ai_overview", "question"}
    assert payload["content_efficiency"]["clusters"]
    assert payload["authority_demand"]["ranked_orphans"][0]["url"] == alpha_url
    assert payload["traffic_weighted_pagerank"]["underserved"][0]["url"] == alpha_url
    assert payload["link_removal_simulation"]["critical_links"][0]["target_url"] == alpha_url
    assert payload["link_addition_simulation"]["recommendations"][0]["target_url"] == alpha_url
    assert payload["leaderboard"][0]["critical_internal_links"] == 1
    assert payload["leaderboard"][0]["high_priority_link_additions"] == 1
    assert payload["leaderboard"][0]["anchor_relevance_descriptive_rate"] == 1.0
    assert payload["anchor_relevance"]["weak_links"][0]["domain"] == "beta.example"
    assert payload["leaderboard"][0]["contextual_link_avg_impact"] == 78
    assert payload["contextual_link_impact"]["top_contextual_links"][0]["domain"] == "alpha.example"
    assert payload["leaderboard"][0]["internal_link_patterns"] == 1
    assert payload["internal_link_patterns"]["rules"][0]["domains"][0]["support_count"] == 3
    assert payload["internal_link_patterns"]["recommendations"][0]["domain"] == "alpha.example"
    assert any(r["pattern_type"] == "internal_link" for r in payload["pattern_transplants"]["recommendations"])
    assert payload["leaderboard"][0]["demand_support_alignment"] == 0.42
    assert payload["leaderboard"][0]["high_demand_low_support_pages"] == 1
    assert payload["high_demand_low_link"]["opportunities"][0]["url"] == alpha_url
    assert payload["high_demand_low_link"]["directories"][0]["domains"][0]["opportunity_traffic"] == 100
    assert payload["leaderboard"][0]["architecture_resilience"] == 0.9
    assert payload["hub_bottlenecks"]["risks"][0]["domain"] == "beta.example"
    architecture = payload["internal_link_architecture"]
    assert architecture["scorecards"]
    assert any(row["domain"] == "alpha.example" and row["orphan_rate"] > 0 for row in architecture["scorecards"])
    assert architecture["clusters"]
    assert architecture["recommendations"][0]["source_url"]
    assert architecture["recommendations"][0]["target_url"]
    assert payload["leaderboard"][0]["authority_traffic_alignment"] == 0.35
    assert payload["leaderboard"][1]["authority_traffic_alignment"] == 0.9
    assert payload["traffic_readiness"]["weak_high_traffic"][0]["url"] == alpha_url
    assert any(
        domain_row.get("links")
        for cluster in payload["strongest_clusters"]["clusters"]
        for domain_row in cluster["domains"]
        if domain_row["domain"] == "alpha.example"
    )
    assert any(pattern["category"] == "link" for pattern in payload["winning_patterns"]["patterns"])
    assert payload["best_page_explainers"]["recommendations"][0]["source_domain"] == "alpha.example"


def test_compare_scatter_includes_freshness_for_pages(tmp_path: Path) -> None:
    page_url = "https://fresh.example/blog/a"
    _write_report(
        tmp_path,
        "fresh.example",
        {"page_count": 1},
        {
            "freshness.json": json.dumps({
                "summary": {"total_pages": 1},
                "buckets": {"stale": 1},
                "per_page": [
                    {
                        "url": page_url,
                        "title": "Old page",
                        "date": "2024-01-01",
                        "age_days": 858,
                        "bucket": "stale",
                        "issues": ["stale"],
                    }
                ],
            }),
            "ahrefs.json": json.dumps({
                "top_pages": [
                    {
                        "matched_url": page_url,
                        "traffic": 321,
                        "keywords": 8,
                        "top_keyword": "old topic",
                    }
                ]
            }),
        },
        pages=[{"url": page_url, "title": "Old page", "section": "blog"}],
    )
    _write_embedding_cache(tmp_path, "fresh.example", [page_url])

    payload = build_payload(["fresh.example"], tmp_path)

    assert payload["scatter"]
    row = payload["scatter"][0]
    assert row["freshness_bucket"] == "stale"
    assert row["freshness_age_days"] == 858
    assert row["freshness_date"] == "2024-01-01"
    assert row["traffic"] == 321


def test_compare_payload_includes_unified_action_board(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        "alpha.example",
        {"page_count": 1},
        {
            "recommendations.json": json.dumps({
                "total": 1,
                "by_category": {"linking": 1},
                "by_priority": {"high": 1, "medium": 0, "low": 0},
                "score_model": {"model": "fix_priority_score_v1"},
                "items": [
                    {
                        "id": "plink-1",
                        "category": "linking",
                        "type": "paragraph_link",
                        "priority": "high",
                        "priority_score": 71.5,
                        "impact": 84,
                        "confidence": 78,
                        "effort": "quick",
                        "effort_score": 25,
                        "risk": 14,
                        "owner": "SEO",
                        "cluster": "support",
                        "traffic_opportunity": 500,
                        "title": "Add in-paragraph link",
                        "instruction": "Add link.",
                        "targets": ["https://alpha.example/a", "https://alpha.example/b"],
                        "evidence": {"score_components": {"priority_score": 71.5}},
                    }
                ],
            })
        },
        pages=[{"url": "https://alpha.example/a", "title": "A", "section": "blog"}],
    )

    payload = build_payload(["alpha.example"], tmp_path)

    assert payload["action_board"]["summary"]["actions"] == 1
    assert payload["action_board"]["items"][0]["domain"] == "alpha.example"
    assert payload["action_board"]["items"][0]["priority_score"] == 71.5
    assert payload["action_board"]["filters"]["owners"] == ["SEO"]
    playbooks = payload["seo_playbooks"]["playbooks"]
    assert payload["seo_playbooks"]["summary"]["playbooks"] == 1
    assert playbooks[0]["source_type"] == "fix_priority_score"
    assert playbooks[0]["target_count"] == 2
    assert {target["url"] for target in playbooks[0]["targets"]} == {
        "https://alpha.example/a",
        "https://alpha.example/b",
    }
    assert playbooks[0]["evidence"][0]["priority_model"] == "fix_priority_score_v1"
    assert playbooks[0]["checklist"][0]["target_url"] == "https://alpha.example/a"


def test_package_comparison_includes_reports_and_excludes_caches(tmp_path: Path) -> None:
    out_dir = tmp_path / "_compare" / "customer"
    out_dir.mkdir(parents=True)
    (out_dir / "index.html").write_text("<html>comparison</html>", encoding="utf-8")
    (out_dir / "comparison.json").write_text("{}", encoding="utf-8")

    report_dir = tmp_path / "example.com" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html>domain</html>", encoding="utf-8")
    (report_dir / "pages.json").write_text("[]", encoding="utf-8")
    cache_dir = tmp_path / "example.com" / "cache"
    cache_dir.mkdir()
    (cache_dir / "embeddings_test.npz").write_bytes(b"cache")

    zip_path = package_comparison(out_dir, tmp_path, ["example.com"])

    assert zip_path == out_dir / "comparison-package.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    assert "index.html" in names
    assert "comparison.json" in names
    assert "README.txt" in names
    assert "manifest.json" in names
    assert "domains/example.com/index.html" in names
    assert "domains/example.com/pages.json" in names
    assert not any("/cache/" in name or name.startswith("cache/") for name in names)
    assert "comparison-package.zip" not in names
