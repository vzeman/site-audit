from pathlib import Path

from site_audit.cli import build_parser
from site_audit.config_env import apply_env_defaults, env_names, update_env_file
from site_audit.settings_ui import _render_home, _render_reports_page, _schema


def test_env_defaults_apply_to_run_options(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["run"])
    monkeypatch.setenv("SITE_AUDIT_RUN_DOMAIN", "example.com")
    monkeypatch.setenv("SITE_AUDIT_RUN_MAX_PAGES", "321")
    monkeypatch.setenv("SITE_AUDIT_RUN_NO_GSC", "true")
    monkeypatch.setenv("SITE_AUDIT_RUN_COMPETITIVE_AUTO_PRODUCT_SEED", "help desk, live chat")

    apply_env_defaults(args, parser, ["run"])

    assert args.domain == "example.com"
    assert args.max_pages == 321
    assert args.no_gsc is True
    assert args.competitive_auto_product_seed == ["help desk", "live chat"]


def test_env_defaults_apply_to_boolean_optional_run_options(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "example.com"])
    monkeypatch.setenv("SITE_AUDIT_RUN_STRIP_HEADER_FOOTER", "false")

    apply_env_defaults(args, parser, ["run", "example.com"])

    assert args.strip_header_footer is False


def test_env_names_normalize_hyphenated_commands() -> None:
    assert env_names("serp-gap", "ai_agent") == [
        "SITE_AUDIT_SERP_GAP_AI_AGENT",
        "SITE_AUDIT_AI_AGENT",
    ]


def test_cli_options_override_env_defaults(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "cli.com", "--max-pages", "50"])
    monkeypatch.setenv("SITE_AUDIT_RUN_DOMAIN", "env.com")
    monkeypatch.setenv("SITE_AUDIT_RUN_MAX_PAGES", "321")

    apply_env_defaults(args, parser, ["run", "cli.com", "--max-pages", "50"])

    assert args.domain == "cli.com"
    assert args.max_pages == 50


def test_blank_env_values_do_not_override_defaults(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["run"])
    monkeypatch.setenv("SITE_AUDIT_RUN_SITEMAP_LASTMOD_WITHIN_DAYS", "")

    apply_env_defaults(args, parser, ["run"])

    assert args.sitemap_lastmod_within_days is None


def test_update_env_file_preserves_unrelated_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nEXISTING=\"keep\"\nSITE_AUDIT_RUN_MAX_PAGES=\"10\"\n", encoding="utf-8")

    update_env_file(env_file, {"SITE_AUDIT_RUN_MAX_PAGES": "20", "SITE_AUDIT_RUN_WORKERS": "4"})

    text = env_file.read_text(encoding="utf-8")
    assert "# comment" in text
    assert 'EXISTING="keep"' in text
    assert 'SITE_AUDIT_RUN_MAX_PAGES="20"' in text
    assert 'SITE_AUDIT_RUN_WORKERS="4"' in text


def test_settings_schema_includes_cli_and_provider_credentials() -> None:
    rows = _schema(build_parser())
    keys = {row["env_key"] for row in rows}

    assert "SITE_AUDIT_RUN_MAX_PAGES" in keys
    assert "SITE_AUDIT_RUN_GOOGLE_ADS_CUSTOMER_ID" in keys
    assert "SITE_AUDIT_SERP_GAP_AI_AGENT" in keys
    assert "GOOGLE_ADS_REFRESH_TOKEN" in keys
    assert "AHREFS_API_KEY" in keys
    assert "OPENROUTER_API_KEY" in keys
    assert "OPENROUTER_MODEL" in keys


def test_settings_schema_includes_field_explanations() -> None:
    rows = _schema(build_parser())
    by_key = {row["env_key"]: row for row in rows}

    domain = by_key["SITE_AUDIT_RUN_DOMAIN"]["details"]
    seed = by_key["SITE_AUDIT_RUN_COMPETITIVE_AUTO_PRODUCT_SEED"]["details"]
    ads = by_key["GOOGLE_ADS_REFRESH_TOKEN"]["details"]
    openrouter = by_key["OPENROUTER_API_KEY"]["details"]
    fallback = by_key["SITE_AUDIT_RUN_DUPLICATE_THRESHOLD"]["details"]

    assert "site to crawl" in domain["what"]
    assert "Comma-separated" in seed["format"]
    assert "without logging in" in ads["why"]
    assert "AI-agent" in openrouter["why"]
    assert fallback["what"]
    assert fallback["example"]


def test_settings_home_links_to_reports_comparisons_and_settings(tmp_path: Path) -> None:
    (tmp_path / "example.com" / "report").mkdir(parents=True)
    (tmp_path / "example.com" / "report" / "site_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "_compare" / "alpha-vs-beta").mkdir(parents=True)
    (tmp_path / "_compare" / "alpha-vs-beta" / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "_compare" / "alpha-vs-beta" / "comparison.json").write_text(
        '{"domains":["alpha.example","beta.example"]}',
        encoding="utf-8",
    )

    html = _render_home(tmp_path)
    reports = _render_reports_page(tmp_path)

    assert 'href="/reports"' in html
    assert 'href="/comparisons"' in html
    assert 'href="/settings"' in html
    assert html.index("Comparisons") < html.index("Domain Reports")
    assert 'href="/comparisons/alpha-vs-beta/"' in html
    assert 'href="/reports/example.com/"' in html
    assert "domain=alpha.example" in html
    assert "domain=beta.example" in html
    assert "domain=example.com" in html
    assert 'href="/reports/example.com/"' in reports
