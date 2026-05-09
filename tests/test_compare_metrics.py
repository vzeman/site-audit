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
    rows = [
        {"type": "link_title", "label": f"{domain} anchor", "count": 4, "size": 4},
        {"type": "header", "label": f"{domain} H1", "level": 1, "size": 10},
        {"type": "page_title", "label": f"{domain} title", "traffic": 20, "size": 20},
    ]
    (cache_dir / "semantic_entities_test_model.meta.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )
    np.savez_compressed(
        cache_dir / "semantic_entities_test_model.npz",
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
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
            "answer_blocks.json": '{"summary":{"top_query_clusters":2,"strong_query_clusters":2,"opportunity_queries":0,"strong_blocks":6},"clusters":[{"label":"support","queries":4,"traffic":100,"strong_query_share":1,"avg_best_score":82,"opportunity_queries":0,"recommended_format":"","status":"strong"}]}',
            "cannibalization.json": '{"summary":{"page_conflicts":1,"paragraph_conflicts":2,"traffic_at_risk":15},"page_conflicts":[{"classification":"duplicate_competing_page","traffic_at_risk":15,"traffic":100,"label":"support","preferred_winner_url":"https://strong.example/a"}]}',
            "duplicate_fragments.json": '{"summary":{"groups":2,"strong_patterns":1,"harmful_boilerplate":1},"groups":[{"classification":"strong_reusable_pattern","count":1,"attributed_traffic":50,"page_traffic_sum":100}]}',
            "template_patterns.json": '{"summary":{"patterns":2,"recommendations":1,"segments_compared":1,"median_confidence":0.7},"patterns":[{"feature_key":"primary_cta","label":"Primary CTA","category":"conversion","observed_lift":1.2,"confidence":0.7,"sample_size":8,"affected_weak_pages":[{"url":"https://strong.example/weak","title":"Weak"}]}],"recommendations":[{"url":"https://strong.example/weak","missing_pattern":"Primary CTA","confidence":0.7,"observed_lift":1.2}]}',
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
            "answer_blocks.json": '{"summary":{"top_query_clusters":2,"strong_query_clusters":0,"opportunity_queries":5,"strong_blocks":1},"clusters":[{"label":"support","queries":3,"traffic":80,"strong_query_share":0,"avg_best_score":42,"opportunity_queries":3,"recommended_format":"faq","status":"gap"}]}',
            "cannibalization.json": '{"summary":{"page_conflicts":3,"paragraph_conflicts":4,"traffic_at_risk":90},"page_conflicts":[{"classification":"consolidation_candidate","traffic_at_risk":90,"traffic":120,"label":"support","preferred_winner_url":"https://weak.example/a"}]}',
            "duplicate_fragments.json": '{"summary":{"groups":5,"strong_patterns":0,"harmful_boilerplate":4},"groups":[{"classification":"harmful_boilerplate","count":4,"attributed_traffic":0,"page_traffic_sum":0}]}',
            "template_patterns.json": '{"summary":{"patterns":0,"recommendations":4,"segments_compared":1,"median_confidence":0},"patterns":[],"recommendations":[{"url":"https://weak.example/a","missing_pattern":"Primary CTA","confidence":0.6,"observed_lift":1.0}]}',
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
    assert payload["structured_data_opportunities"]["types"]
    assert payload["structured_data_opportunities"]["clusters"]
    assert payload["leaderboard"][1]["schema_opportunities"] == 4
    assert payload["trust_signals"]["clusters"]
    assert payload["leaderboard"][0]["trust_avg_score"] == 82
    assert payload["leaderboard"][1]["trust_high_priority_pages"] == 2
    assert payload["keyword_cluster_gaps"]["clusters"]
    assert payload["keyword_cluster_gaps"]["clusters"][0]["leader_domain"] == "strong.example"
    assert payload["keyword_cluster_gaps"]["recommendations"]
    assert payload["keyword_cluster_gaps"]["cache"]["entries"] >= 2


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
    assert payload["traffic_readiness"]["weak_high_traffic"][0]["url"] == alpha_url


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
