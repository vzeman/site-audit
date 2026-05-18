import json

from site_audit.server import _render_comparisons_page, _render_home, _render_reports_page, _render_scans_page


def test_server_home_renders_main_menu_reports_and_comparisons(tmp_path) -> None:
    report_dir = tmp_path / "flowhunt.io" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "site_metrics.json").write_text("{}", encoding="utf-8")
    compare_dir = tmp_path / "_compare" / "flowhunt-vs-lindy"
    compare_dir.mkdir(parents=True)
    (compare_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (compare_dir / "comparison.json").write_text(
        json.dumps({"domains": ["flowhunt.io", "lindy.ai"]}),
        encoding="utf-8",
    )

    html = _render_home(tmp_path)

    assert 'href="/"' in html
    assert 'href="/reports"' in html
    assert 'href="/comparisons"' in html
    assert 'href="/scans"' in html
    assert 'href="/reports/flowhunt.io/"' in html
    assert 'href="/comparisons/flowhunt-vs-lindy/"' in html
    assert "google.com/s2/favicons?domain=flowhunt.io" in html
    assert "google.com/s2/favicons?domain=lindy.ai" in html


def test_server_subpages_keep_main_menu(tmp_path) -> None:
    html_pages = [
        _render_reports_page(tmp_path),
        _render_comparisons_page(tmp_path),
        _render_scans_page(tmp_path, []),
    ]

    for html in html_pages:
        assert 'href="/"' in html
        assert 'href="/reports"' in html
        assert 'href="/comparisons"' in html
        assert 'href="/scans"' in html
