from pathlib import Path
import json
import zipfile

import numpy as np

from site_audit.compare import build_payload, package_comparison


def _write_report(root: Path, domain: str, metrics: dict, extras: dict[str, str]) -> None:
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
    (report_dir / "pages.json").write_text("[]", encoding="utf-8")
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
            "structured_data.json": '{"summary":{"schema_coverage":1,"invalid_jsonld_blocks":0,"schema_type_count":5}}',
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
            "ahrefs.json": '{"summary":{"top_pages":2,"organic_keywords":10,"matched_traffic":1000,"matched_traffic_share":1,"top_pages_value_usd":250},"metrics":{"org_traffic":1200,"org_keywords":20,"org_keywords_1_3":5}}',
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
            "structured_data.json": '{"summary":{"schema_coverage":0.2,"invalid_jsonld_blocks":3,"schema_type_count":1}}',
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
            "ahrefs.json": '{"summary":{"top_pages":2,"organic_keywords":3,"matched_traffic":100,"matched_traffic_share":0.5,"top_pages_value_usd":20},"metrics":{"org_traffic":150,"org_keywords":4,"org_keywords_1_3":0}}',
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
