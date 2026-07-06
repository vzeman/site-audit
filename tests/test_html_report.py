from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import AuditResult
from site_audit.html_report import write_html_report


def test_large_html_report_uses_external_json_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_HTML_EXTERNAL_JSON_MAX_PAGES", "1")
    template = tmp_path / "template.html"
    template.write_text(
        '<script id="data-metrics" type="application/json">__METRICS_JSON__</script>',
        encoding="utf-8",
    )
    pages = [
        SimpleNamespace(url=f"https://example.com/{idx}", title=f"Page {idx}", section="root", word_count=10)
        for idx in range(2)
    ]
    result = AuditResult(
        pages=pages,
        embeddings=np.zeros((2, 2), dtype=np.float32),
        site_centroid=np.zeros(2, dtype=np.float32),
        site_metrics={"count": 2, "focus_score": 0.0, "radius": 0.0, "mean_distance": 0.0, "p95_distance": 0.0, "max_distance": 0.0},
        sections={},
        dist_to_site=np.zeros(2, dtype=np.float32),
        dist_to_section=np.zeros(2, dtype=np.float32),
        duplicate_pairs=[],
    )

    out = write_html_report(tmp_path, template, result, model_name="test", domain="example.com")

    assert "__METRICS_JSON__" in out.read_text(encoding="utf-8")
