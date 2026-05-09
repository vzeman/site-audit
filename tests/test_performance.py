from dataclasses import dataclass
import json
from pathlib import Path

from site_audit.compare import build_payload
from site_audit.performance import analyze, to_payload


@dataclass
class _Fetched:
    url: str
    body: str
    status: int = 200
    content_type: str = "text/html"
    content_length_bytes: int = 0


def test_performance_payload_counts_resources_and_blocking_heuristics() -> None:
    html = """
    <html><head>
      <link rel="stylesheet" href="/app.css">
      <link rel="stylesheet" href="/print.css" media="print">
      <link rel="preload" as="font" href="/font.woff2" type="font/woff2">
      <style>.hero{color:red}</style>
      <script src="/blocking.js"></script>
      <script src="/deferred.js" defer></script>
      <script type="module" src="/module.js"></script>
      <script>window.inline = true;</script>
    </head><body>
      <img src="/one.jpg">
      <img src="/two.jpg">
      <div style="display:none"></div>
    </body></html>
    """

    payload = to_payload(analyze([_Fetched("https://example.com/", html, content_length_bytes=1234)]))
    row = payload["per_page"][0]

    assert payload["summary"]["total_pages"] == 1
    assert payload["summary"]["status_counts"] == {"200": 1}
    assert row["content_size_bytes"] == 1234
    assert row["html_weight_bytes"] == 1234
    assert row["image_count"] == 2
    assert row["script_count"] == 4
    assert row["external_script_count"] == 3
    assert row["inline_script_count"] == 1
    assert row["stylesheet_count"] == 2
    assert row["inline_style_count"] == 1
    assert row["style_attr_count"] == 1
    assert row["font_count"] == 1
    assert row["preload_count"] == 1
    assert row["render_blocking_css_count"] == 1
    assert row["render_blocking_script_count"] == 1
    assert payload["summary"]["pages_with_render_blocking"] == 1


def test_performance_payload_buckets_and_heavy_pages() -> None:
    payload = to_payload(analyze([
        _Fetched("https://example.com/light", "<html></html>", content_length_bytes=100_000),
        _Fetched("https://example.com/heavy", "<html>" + "<img>" * 30 + "</html>", content_length_bytes=100_000),
    ]))

    assert payload["buckets"]["light"] == 1
    assert payload["buckets"]["very_heavy"] == 1
    assert payload["summary"]["heavy_pages"] == 1
    assert payload["summary"]["heavy_page_share"] == 0.5
    assert payload["top_heavy_pages"][0]["url"] == "https://example.com/heavy"


def test_compare_leaderboard_includes_performance_metrics(tmp_path: Path) -> None:
    for domain, median_html, blocking_share in [
        ("a.example", 20000, 0.25),
        ("b.example", 80000, 0.75),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":5}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "performance.json").write_text(
            (
                '{"summary":{"median_html_weight_bytes":%d,'
                '"p90_estimated_weight_bytes":150000,'
                '"avg_resource_tags_per_page":7.5,'
                '"render_blocking_share":%s,'
                '"heavy_page_share":0.2,'
                '"total_images":10,'
                '"total_scripts":8,'
                '"total_stylesheets":4}}'
            ) % (median_html, blocking_share),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["median_html_weight_bytes"] == 20000
    assert rows["a.example"]["render_blocking_share"] == 0.25
    assert rows["a.example"]["avg_resource_tags_per_page"] == 7.5
    assert rows["b.example"]["median_html_weight_bytes"] == 80000
    assert rows["b.example"]["heavy_page_share"] == 0.2


def test_compare_payload_includes_performance_explainer(tmp_path: Path) -> None:
    for domain, coef in [("a.example", 0.42), ("b.example", -0.25)]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            json.dumps({"domain": domain, "model": "test-model", "page_count": 3}),
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "performance_explainer.json").write_text(
            json.dumps({
                "summary": {"status": "ok", "sample_size": 12, "validation_r2": 0.31, "warnings": ["Correlation model only."]},
                "features": [
                    {
                        "feature": "links_in_degree_log",
                        "label": "Inbound internal links",
                        "group": "links",
                        "coefficient": coef,
                        "direction": "positive" if coef > 0 else "negative",
                        "permutation_importance": abs(coef) / 10,
                        "abs_coefficient": abs(coef),
                    }
                ],
                "pages": [
                    {
                        "url": f"https://{domain}/a",
                        "title": "A",
                        "section": "blog",
                        "traffic": 100,
                        "predicted_traffic": 80,
                        "residual_log": 0.2,
                        "top_positive": [{"feature": "links_in_degree_log", "label": "Inbound internal links", "group": "links", "contribution": coef}],
                        "top_negative": [],
                    }
                ],
            }),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    explainer = payload["performance_explainer"]
    assert explainer["summary"]["features"] == 1
    assert explainer["features"][0]["feature"] == "links_in_degree_log"
    assert explainer["features"][0]["domains"][0]["domain"] == "a.example"
    assert explainer["pages"][0]["domain"] in {"a.example", "b.example"}
