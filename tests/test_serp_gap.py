import json
from pathlib import Path

from site_audit.serp_gap import SerpGapConfig, run


def _write_base_report(root: Path) -> None:
    report = root / "example.com" / "report"
    report.mkdir(parents=True)
    homepage = "https://www.example.com/"
    feature = "https://www.example.com/features/live-chat/"
    (report / "pages.json").write_text(
        json.dumps(
            [
                {"url": homepage, "title": "Example Home", "description": "", "section": "/", "word_count": 500},
                {"url": feature, "title": "Live Chat", "description": "", "section": "features", "word_count": 800},
            ]
        ),
        encoding="utf-8",
    )
    (report / "search.json").write_text(
        json.dumps(
            {
                "meta": {"provider": "gsc"},
                "organic_keywords": [
                    {
                        "keyword": "live chat software",
                        "matched_url": feature,
                        "provider": "gsc",
                        "position": 8,
                        "impressions": 1000,
                        "clicks": 20,
                    },
                    {
                        "keyword": "ignored",
                        "matched_url": homepage,
                        "provider": "gsc",
                        "position": 50,
                        "impressions": 1000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_serp_gap_dry_run_selects_pattern_pages_and_keywords(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["summary"]["pages_selected"] == 1
    assert payload["summary"]["keywords_selected"] == 1
    assert payload["selected_keywords"][0]["keyword"] == "live chat software"
    assert (tmp_path / "example.com" / "serp_gap" / "report" / "serp_gap.json").is_file()
    assert (tmp_path / "example.com" / "cache" / "serp_gap").is_dir()
    assert not (tmp_path / "example.com" / "serp_gap" / "cache").exists()


def test_serp_gap_manual_keywords_can_target_homepage(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/"],
            keyword_source="file",
            keywords=["helpdesk software", "call center software"],
            keywords_per_page=5,
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert [row["keyword"] for row in payload["selected_keywords"]] == [
        "helpdesk software",
        "call center software",
    ]
    assert all(row["source"] == "manual" for row in payload["selected_keywords"])


def test_serp_gap_budget_cap_stops_before_paid_work(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            budget_usd=0.0,
        )
    )

    assert payload["status"] == "budget_exceeded"
    assert payload["summary"]["budget_status"] == "over_budget"


def test_serp_gap_html_includes_scatter_and_cluster_sections(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            dry_run=True,
        )
    )

    html = (tmp_path / "example.com" / "serp_gap" / "report" / "index.html").read_text(encoding="utf-8")

    assert "Semantic Scatterplot" in html
    assert "Semantic Clusters" in html
    assert "Topic Relations" in html
    assert "Keyword and URL Semantic Map" in html
    assert "All Keywords and URLs" in html
    assert "overview_scatter" in html
    assert "overview-section" in html
    assert "nearest_keyword" in html
    assert "Wheel to zoom, drag to pan, double-click to reset" in html
    assert "pointTooltip" in html
    assert "scatter-tooltip" in html
    assert "bindScatterInteractions" in html
    assert "click a dot for details" in html
    assert "data-zoom=\"in\"" in html
    assert "Wheel to zoom, drag to pan, double-click to reset" in html
    assert "domainColor" in html
    assert "function markerSvg" in html
    assert "function pointSize" in html
    assert "data-entity-filter" in html
    assert "bindScatterFilters" in html
    assert "topicChart" in html
    assert "topic-chart" in html
    assert "data-detail" in html
    assert "tip-badge" in html
    assert "keyword_distance" in html
    assert "keyword_similarity" in html
    assert "function pointLabel" in html
    assert ".scatter-tooltip.open" in html
    assert "page-section" in html
    assert "report-sidebar" in html
    assert 'id="report-nav"' in html
    assert "report-nav-button" in html
    assert "report-nav-label" in html
    assert "--audit-accent" in html
    assert "buildNav" in html
