"""Small local app for reports, comparisons, and .env-backed settings."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

from .config_env import collect_actions, env_names, read_env_file, update_env_file


def _domain_report_index(projects_root: Path, domain: str, ui_dir: Path) -> Path:
    report_index = projects_root / domain / "report" / "index.html"
    if report_index.is_file():
        return report_index
    return ui_dir / "index.html"


FIELD_DETAILS = {
    "SITE_AUDIT_RUN_DOMAIN": {
        "what": "The site to crawl and analyze.",
        "why": "Allows `site-audit run` without retyping the domain.",
        "format": "Domain or full URL.",
        "example": "example.com or https://www.example.com",
    },
    "SITE_AUDIT_RUN_MAX_PAGES": {
        "what": "Maximum number of pages to crawl.",
        "why": "Controls crawl cost, runtime, and report size.",
        "format": "Integer.",
        "example": "500 for a fast sample, 10000 for a broad crawl.",
    },
    "SITE_AUDIT_RUN_WORKERS": {
        "what": "Concurrent page fetches.",
        "why": "Higher values finish faster but can stress rate-limited sites.",
        "format": "Integer.",
        "example": "4 for polite crawls, 8 default, 16 for robust sites.",
    },
    "SITE_AUDIT_RUN_SEARCH_PROVIDER": {
        "what": "Search-demand provider used for traffic and keyword overlays.",
        "why": "Determines where keyword opportunities come from.",
        "format": "One of the dropdown values.",
        "example": "auto, all, gsc, google_ads, ahrefs, dataforseo, none.",
    },
    "SITE_AUDIT_RUN_COMPETITIVE_AUTO": {
        "what": "Automatically select business-relevant keywords and analyze ranking competitor pages.",
        "why": "Avoids manually building a competitor TSV when you want SERP paragraph gaps.",
        "format": "Boolean.",
        "example": "true",
    },
    "SITE_AUDIT_RUN_COMPETITIVE_AUTO_PRODUCT_SEED": {
        "what": "Product/service phrases used to filter relevant keywords.",
        "why": "Prevents wasting DataForSEO and crawl work on keywords that do not match the business.",
        "format": "Comma-separated or one phrase per line.",
        "example": "help desk software, live chat software",
    },
    "SITE_AUDIT_RUN_GOOGLE_ADS_CUSTOMER_ID": {
        "what": "Google Ads customer account to query.",
        "why": "Required when Google Ads search terms are used as the keyword source.",
        "format": "Digits with or without dashes.",
        "example": "123-456-7890",
    },
    "SITE_AUDIT_RUN_GOOGLE_ADS_LOGIN_CUSTOMER_ID": {
        "what": "Manager account used for Google Ads API access.",
        "why": "Needed when the audited customer account is under an MCC/manager account.",
        "format": "Digits with or without dashes.",
        "example": "999-888-7777",
    },
    "SITE_AUDIT_RUN_GOOGLE_ADS_START_DATE": {
        "what": "Start date for Google Ads search-term spend.",
        "why": "Limits paid keyword selection to the business period you care about.",
        "format": "YYYY-MM-DD.",
        "example": "2026-02-01",
    },
    "SITE_AUDIT_RUN_GOOGLE_ADS_END_DATE": {
        "what": "End date for Google Ads search-term spend.",
        "why": "Pairs with the start date to define the Ads reporting window.",
        "format": "YYYY-MM-DD.",
        "example": "2026-04-30",
    },
    "SITE_AUDIT_RUN_GOOGLE_ADS_MIN_COST": {
        "what": "Minimum spend required for a paid search term.",
        "why": "Filters out noise and one-off low-spend query variants.",
        "format": "Number in the account currency.",
        "example": "50",
    },
    "SITE_AUDIT_RUN_DATAFORSEO_LOCATION_CODE": {
        "what": "DataForSEO geographic location code.",
        "why": "Controls which country/location SERP and keyword data represents.",
        "format": "Integer DataForSEO location code.",
        "example": "2840 for United States.",
    },
    "SITE_AUDIT_RUN_DATAFORSEO_LANGUAGE_CODE": {
        "what": "DataForSEO language code.",
        "why": "Keeps SERP and keyword data aligned with the audience language.",
        "format": "ISO-like language code.",
        "example": "en",
    },
    "SITE_AUDIT_RUN_AHREFS_COUNTRY": {
        "what": "Country database for Ahrefs keyword data.",
        "why": "Keeps rankings, volume, and traffic estimates country-specific.",
        "format": "Ahrefs country code.",
        "example": "US, GB, SK.",
    },
    "SITE_AUDIT_SERVE_DOMAIN": {
        "what": "Domain report to open in the local viewer.",
        "why": "Allows `site-audit serve` without retyping the domain.",
        "format": "Same domain slug used during the run.",
        "example": "example.com",
    },
    "GSC_PROPERTY_URL": {
        "what": "Google Search Console property.",
        "why": "Tells GSC which verified site to query.",
        "format": "GSC property URL.",
        "example": "sc-domain:example.com or https://www.example.com/",
    },
    "GSC_ACCESS_TOKEN": {
        "what": "Short-lived OAuth token for Search Console.",
        "why": "Allows fetching GSC data without a service account.",
        "format": "OAuth access token.",
        "example": "ya29....",
    },
    "GSC_SERVICE_ACCOUNT_FILE": {
        "what": "Path to a Search Console service-account JSON file.",
        "why": "Allows repeatable GSC access without manual token refreshes.",
        "format": "Absolute file path.",
        "example": "/Users/me/keys/gsc-service-account.json",
    },
    "AHREFS_API_KEY": {
        "what": "Ahrefs API token.",
        "why": "Enables organic keyword, top page, and traffic enrichment.",
        "format": "Secret token.",
        "example": "ahrefs_xxx",
    },
    "OPENROUTER_API_KEY": {
        "what": "OpenRouter API key.",
        "why": "Enables AI-agent keyword inference and paragraph-level SERP gap TODO briefs.",
        "format": "Secret token stored in local .env, which is ignored by git.",
        "example": "sk-or-v1-...",
    },
    "OPENROUTER_MODEL": {
        "what": "Default OpenRouter model for AI-agent work.",
        "why": "Controls which model writes keyword recommendations and editor briefs.",
        "format": "OpenRouter model id.",
        "example": "deepseek/deepseek-v4-pro",
    },
    "HARNEXT_API_KEY": {
        "what": "Optional Harnext API key.",
        "why": "Reserved for Harnext SDK workflows when a Harnext account requires its own key.",
        "format": "Secret token, if required by the SDK/account.",
        "example": "harnext_xxx",
    },
    "DATAFORSEO_LOGIN": {
        "what": "DataForSEO API login.",
        "why": "Required for DataForSEO keyword and SERP requests.",
        "format": "Account email/login.",
        "example": "you@example.com",
    },
    "DATAFORSEO_PASSWORD": {
        "what": "DataForSEO API password.",
        "why": "Required with the login for DataForSEO API authentication.",
        "format": "Secret password.",
        "example": "your API password",
    },
    "GOOGLE_ADS_DEVELOPER_TOKEN": {
        "what": "Developer token from the Google Ads API Center.",
        "why": "Google Ads API requires it on every request.",
        "format": "Secret token.",
        "example": "abc123...",
    },
    "GOOGLE_ADS_CLIENT_ID": {
        "what": "OAuth client ID for the Google Ads integration.",
        "why": "Used to refresh access tokens for the authorized Ads user.",
        "format": "OAuth client ID.",
        "example": "123.apps.googleusercontent.com",
    },
    "GOOGLE_ADS_CLIENT_SECRET": {
        "what": "OAuth client secret for the Google Ads integration.",
        "why": "Used with the refresh token to get access tokens.",
        "format": "Secret string.",
        "example": "GOCSPX-...",
    },
    "GOOGLE_ADS_REFRESH_TOKEN": {
        "what": "OAuth refresh token with the Google Ads scope.",
        "why": "Lets the audit access Ads data repeatedly without logging in each run.",
        "format": "Secret refresh token.",
        "example": "1//0g...",
    },
    "GOOGLE_ADS_CUSTOMER_ID": {
        "what": "Default Google Ads customer account.",
        "why": "Used when no `--google-ads-customer-id` is provided.",
        "format": "Digits with or without dashes.",
        "example": "123-456-7890",
    },
    "GOOGLE_ADS_LOGIN_CUSTOMER_ID": {
        "what": "Default Google Ads manager account.",
        "why": "Required for accounts accessed through an MCC/manager hierarchy.",
        "format": "Digits with or without dashes.",
        "example": "999-888-7777",
    },
}

EXTRA_SETTINGS = [
    {"command": "credentials", "env_key": "GSC_PROPERTY_URL", "flag": "GSC property URL", "default": "", "help": "Search Console property, e.g. sc-domain:example.com.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "GSC_ACCESS_TOKEN", "flag": "GSC access token", "default": "", "help": "Optional short-lived Search Console access token.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "GSC_SERVICE_ACCOUNT_FILE", "flag": "GSC service account file", "default": "", "help": "Absolute path to a Search Console service-account JSON file.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "AHREFS_API_KEY", "flag": "Ahrefs API key", "default": "", "help": "Ahrefs API key for organic search-demand enrichment.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "DATAFORSEO_LOGIN", "flag": "DataForSEO login", "default": "", "help": "DataForSEO API login/email.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "DATAFORSEO_PASSWORD", "flag": "DataForSEO password", "default": "", "help": "DataForSEO API password.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_DEVELOPER_TOKEN", "flag": "Google Ads developer token", "default": "", "help": "Developer token from the Google Ads API Center.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_CLIENT_ID", "flag": "Google Ads OAuth client ID", "default": "", "help": "OAuth desktop/web client ID.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_CLIENT_SECRET", "flag": "Google Ads OAuth client secret", "default": "", "help": "OAuth client secret.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_REFRESH_TOKEN", "flag": "Google Ads refresh token", "default": "", "help": "Refresh token with the adwords OAuth scope.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_CUSTOMER_ID", "flag": "Google Ads customer ID", "default": "", "help": "Client account ID to query.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "GOOGLE_ADS_LOGIN_CUSTOMER_ID", "flag": "Google Ads manager ID", "default": "", "help": "Optional manager account ID for login-customer-id.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "OPENROUTER_API_KEY", "flag": "OpenRouter API key", "default": "", "help": "Required for AI-agent keyword inference and editor TODO briefs.", "choices": [], "kind": "secret"},
    {"command": "credentials", "env_key": "OPENROUTER_MODEL", "flag": "OpenRouter model", "default": "deepseek/deepseek-v4-pro", "help": "Default model for AI-agent tasks.", "choices": [], "kind": "text"},
    {"command": "credentials", "env_key": "HARNEXT_API_KEY", "flag": "Harnext API key", "default": "", "help": "Optional Harnext SDK account key if your setup requires one.", "choices": [], "kind": "secret"},
]


def serve_settings_ui(
    parser: argparse.ArgumentParser,
    *,
    env_file: Path,
    host: str = "127.0.0.1",
    port: int = 8780,
    projects_root: Path | str = "projects",
    ui_dir: Path | str | None = None,
) -> None:
    env_file = Path(env_file)
    projects_root = Path(projects_root)
    ui_dir = Path(ui_dir) if ui_dir else Path(__file__).resolve().parent.parent / "ui"
    schema = _schema(parser)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):  # noqa: N802
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                self._send_html(_render_home(projects_root))
                return
            if path == "/settings":
                self._send_html(_render_settings_page(schema, env_file, saved=False))
                return
            if path == "/reports":
                self._send_html(_render_reports_page(projects_root))
                return
            if path == "/comparisons":
                self._send_html(_render_comparisons_page(projects_root))
                return
            if path.startswith("/reports/") and path.rstrip("/").count("/") == 2:
                domain = path.strip("/").split("/", 1)[1]
                self._serve_file(_domain_report_index(projects_root, domain, ui_dir))
                return
            if path.startswith("/reports-data/"):
                parts = path[len("/reports-data/"):].split("/", 1)
                if len(parts) != 2:
                    self.send_error(404, "Not found")
                    return
                domain, rel = parts
                self._serve_file(projects_root / domain / "report" / rel)
                return
            if path.startswith("/comparisons/") and path.rstrip("/").count("/") == 2:
                name = path.strip("/").split("/", 1)[1]
                self._serve_file(projects_root / "_compare" / name / "index.html")
                return
            if path.startswith("/comparison-data/"):
                rel = path[len("/comparison-data/"):]
                self._serve_file(projects_root / "_compare" / rel)
                return
            self.send_error(404, "Not found")

        def do_POST(self):  # noqa: N802
            path = unquote(self.path.split("?", 1)[0])
            if path != "/settings":
                self.send_error(404, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(body, keep_blank_values=True)
            allowed = {row["env_key"] for row in schema}
            updates = {
                key: values[-1]
                for key, values in form.items()
                if key in allowed and values
            }
            update_env_file(env_file, updates)
            self._send_html(_render_settings_page(schema, env_file, saved=True))

        def _serve_file(self, file_path: Path) -> None:
            try:
                file_path = file_path.resolve()
            except Exception:
                self.send_error(404, "Not found")
                return
            if not file_path.is_file():
                self.send_error(404, "Not found")
                return
            ctype, _ = mimetypes.guess_type(str(file_path))
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  ➜  Site-audit local app: http://{host}:{port}/")
    print(f"     editing: {env_file}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
        httpd.server_close()


def _schema(parser: argparse.ArgumentParser) -> list[dict]:
    rows: list[dict] = []
    for action in collect_actions(parser):
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.dest in {"help", "version"}:
            continue
        command = _command_for_action(parser, action)
        # Prefer command-specific keys for repeated option names such as
        # projects_root, output_dir, host, and port.
        env_key = env_names(command, action.dest)[0]
        rows.append({
            "command": command or "global",
            "env_key": env_key,
            "dest": action.dest,
            "flag": ", ".join(action.option_strings) if action.option_strings else f"<{action.dest}>",
            "default": _default_value(action),
            "help": action.help or "",
            "choices": list(action.choices or []),
            "kind": _kind(action),
        })
    rows.extend(EXTRA_SETTINGS)
    for row in rows:
        row["details"] = _field_details(row)
    rows.sort(key=lambda r: (r["command"], r["env_key"]))
    return rows


def _field_details(row: dict) -> dict[str, str]:
    if row["env_key"] in FIELD_DETAILS:
        return FIELD_DETAILS[row["env_key"]]
    dest = str(row.get("dest") or row.get("env_key") or "").replace("_", " ")
    return {
        "what": _fallback_what(row, dest),
        "why": _fallback_why(row),
        "format": _fallback_format(row),
        "example": _fallback_example(row),
    }


def _fallback_what(row: dict, dest: str) -> str:
    help_text = str(row.get("help") or "").strip()
    if help_text and help_text != argparse.SUPPRESS:
        return help_text.rstrip(".") + "."
    return f"Configures {dest}."


def _fallback_why(row: dict) -> str:
    command = row.get("command") or "site-audit"
    if row.get("kind") == "bool":
        return f"Use this to turn the {command} behavior on or off without passing the flag every time."
    return f"Use this when the same {command} value is needed across repeated runs."


def _fallback_format(row: dict) -> str:
    if row.get("kind") == "bool":
        return "Boolean: true or false."
    if row.get("kind") == "number":
        return "Number."
    if row.get("kind") == "choice":
        return "One of: " + ", ".join(str(c) for c in row.get("choices") or []) + "."
    if row.get("kind") == "list":
        return "Comma-separated values or one value per line."
    if row.get("kind") == "secret":
        return "Secret value. Do not commit it."
    return "Text."


def _fallback_example(row: dict) -> str:
    if row.get("choices"):
        return str((row.get("choices") or [""])[0])
    default = str(row.get("default") or "")
    if default:
        return default
    if row.get("kind") == "bool":
        return "true"
    if row.get("kind") == "number":
        return "100"
    if row.get("kind") == "list":
        return "value one, value two"
    if row.get("kind") == "secret":
        return "paste secret here"
    return "example value"


def _command_for_action(parser: argparse.ArgumentParser, target: argparse.Action) -> str:
    found = ""

    def visit(p: argparse.ArgumentParser, command: str) -> None:
        nonlocal found
        if found:
            return
        if target in p._actions:
            found = command
            return
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, subparser in action.choices.items():
                    visit(subparser, name)

    visit(parser, "")
    return found


def _default_value(action: argparse.Action) -> str:
    value = action.default
    if value in (None, argparse.SUPPRESS):
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return "bool"
    if isinstance(action, argparse._AppendAction) or action.nargs in {"+", "*"}:
        return "list"
    if action.choices:
        return "choice"
    if action.type in {int, float}:
        return "number"
    if any(token in action.dest for token in ("token", "password", "secret")):
        return "secret"
    return "text"


def _layout(title: str, active: str, body: str, *, subtitle: str = "") -> str:
    nav_items = [
        ("home", "/", "Overview"),
        ("reports", "/reports", "Reports"),
        ("comparisons", "/comparisons", "Comparisons"),
        ("settings", "/settings", "Settings"),
    ]
    nav = "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in nav_items
    )
    subtitle_html = f'<div class="hint">{escape(subtitle)}</div>' if subtitle else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background:#f6f2ea; color:#231f1a; }}
    header {{ position:sticky; top:0; z-index:2; display:grid; grid-template-columns: 1fr auto; gap:20px; align-items:center; padding:18px 28px; background:#fffdfa; border-bottom:1px solid #eadfce; }}
    nav {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    nav a {{ color:#4f463c; text-decoration:none; font-weight:800; font-size:13px; padding:9px 11px; border-radius:8px; }}
    nav a.active, nav a:hover {{ background:#231f1a; color:#fffdfa; }}
    main {{ max-width:1120px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 4px; font-size:28px; }}
    h2 {{ margin:28px 0 12px; font-size:18px; text-transform:capitalize; }}
    h3 {{ margin:0 0 6px; font-size:16px; }}
    .hint, .muted {{ color:#756d65; font-size:13px; line-height:1.45; }}
    .notice {{ margin-top:12px; display:inline-block; padding:8px 12px; border-radius:8px; background:#e8f7ef; color:#17613d; }}
    .card {{ background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
    .panel {{ background:#fffdfa; border:1px solid #eadfce; border-radius:10px; padding:18px; }}
    .cards {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; margin-top:12px; }}
    .item-card {{ display:flex; flex-direction:column; min-height:118px; color:inherit; text-decoration:none; }}
    .item-card:hover {{ border-color:#d77d24; box-shadow:0 2px 10px rgba(80,54,24,.08); }}
    .card-head {{ display:flex; align-items:center; gap:10px; }}
    .favicon-stack {{ display:flex; flex:0 0 auto; min-width:32px; }}
    .favicon-stack img {{ width:28px; height:28px; border-radius:7px; border:1px solid #eadfce; background:#fff; object-fit:contain; }}
    .favicon-stack img + img {{ margin-left:-9px; }}
    .card-title {{ font-weight:900; line-height:1.25; word-break:break-word; }}
    .card-meta {{ margin-top:8px; }}
    .card-action {{ margin-top:auto; padding-top:14px; color:#a04400; font-weight:900; font-size:13px; }}
    .empty {{ margin-top:12px; }}
    .panel a, td a, li a {{ color:#a04400; font-weight:800; text-decoration:none; }}
    .metric {{ margin-top:12px; color:#9a5200; font-size:26px; font-weight:900; }}
    .row {{ display:grid; grid-template-columns: 260px 1fr; gap:18px; padding:14px 16px; border-top:1px solid #f0e7da; }}
    .row:first-child {{ border-top:0; }}
    label {{ display:block; font-weight:700; font-size:13px; }}
    code {{ display:block; margin-top:4px; color:#9a5200; font-size:12px; word-break:break-all; }}
    input, select, textarea {{ width:100%; box-sizing:border-box; border:1px solid #d8cbb9; border-radius:8px; padding:9px 10px; font:inherit; background:white; }}
    textarea {{ min-height:74px; }}
    .help {{ margin-top:6px; color:#756d65; font-size:12px; line-height:1.35; }}
    .details {{ margin-top:9px; display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:7px; }}
    .detail {{ padding:8px 9px; border:1px solid #f0e7da; border-radius:8px; background:#fff8ef; color:#4e463e; font-size:12px; line-height:1.35; }}
    .detail strong {{ display:block; margin-bottom:2px; color:#9a5200; font-size:10px; text-transform:uppercase; letter-spacing:0.06em; }}
    .default {{ margin-top:5px; color:#9b938b; font-size:12px; }}
    .actions {{ position:sticky; bottom:0; padding:16px 0; background:linear-gradient(transparent, #f6f2ea 30%); }}
    button {{ border:0; border-radius:8px; padding:11px 16px; background:#ff8a1f; color:white; font-weight:800; cursor:pointer; }}
    table {{ width:100%; border-collapse:collapse; background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
    td, th {{ padding:11px 12px; border-top:1px solid #f0e7da; text-align:left; font-size:13px; vertical-align:top; }}
    th {{ border-top:0; color:#756d65; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    ul.list {{ margin:0; padding:0; list-style:none; background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
    ul.list li {{ display:flex; justify-content:space-between; gap:14px; padding:12px 14px; border-top:1px solid #f0e7da; }}
    ul.list li:first-child {{ border-top:0; }}
    @media (max-width: 820px) {{
      header {{ grid-template-columns:1fr; }}
      nav {{ justify-content:flex-start; }}
      .cards, .details {{ grid-template-columns:1fr; }}
      .row {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{escape(title)}</h1>
      {subtitle_html}
    </div>
    <nav>{nav}</nav>
  </header>
  <main>{body}</main>
</body>
</html>"""


def _render_home(projects_root: Path) -> str:
    reports = _report_rows(projects_root)
    comparisons = _comparison_rows(projects_root)
    comparison_cards = "".join(
        _item_card(
            row["name"],
            row["url"],
            f'updated {_format_ts(row["updated_at"])}',
            "Open comparison",
            row.get("domains") or [],
        )
        for row in comparisons
    ) or '<div class="panel muted empty">No generated comparisons found under projects/_compare.</div>'
    report_cards = "".join(
        _item_card(row["domain"], row["url"], f'updated {_format_ts(row["updated_at"])}', "Open report", [row["domain"]])
        for row in reports
    ) or '<div class="panel muted empty">No generated reports found under projects/&lt;domain&gt;/report.</div>'
    body = f"""
      <section>
        <h2>Comparisons</h2>
        <div class="hint">Generated cross-domain dashboards from <code style="display:inline">projects/_compare</code>.</div>
        <div class="cards">{comparison_cards}</div>
      </section>
      <section>
        <h2>Domain Reports</h2>
        <div class="hint">Completed audits from <code style="display:inline">projects/&lt;domain&gt;/report</code>.</div>
        <div class="cards">{report_cards}</div>
      </section>
    """
    return _layout("Site Audit", "home", body, subtitle="Local app for reports, comparisons, and configuration.")


def _item_card(title: str, url: str, meta: str, action: str, domains: list[str] | None = None) -> str:
    icons = _favicon_stack(domains or [])
    return f"""
      <a class="panel item-card" href="{escape(url)}">
        <div class="card-head">{icons}<div class="card-title">{escape(title)}</div></div>
        <div class="hint card-meta">{escape(meta)}</div>
        <div class="card-action">{escape(action)}</div>
      </a>"""


def _favicon_stack(domains: list[str]) -> str:
    clean_domains = [_clean_domain(domain) for domain in domains if _clean_domain(domain)]
    if not clean_domains:
        return '<span class="favicon-stack"></span>'
    imgs = "".join(
        f'<img src="{escape(_favicon_url(domain))}" alt="{escape(domain)} favicon" loading="lazy">'
        for domain in clean_domains[:3]
    )
    return f'<span class="favicon-stack">{imgs}</span>'


def _favicon_url(domain: str) -> str:
    return f"https://www.google.com/s2/favicons?domain={escape(domain)}&sz=64"


def _clean_domain(value: str) -> str:
    domain = str(value or "").strip().replace("https://", "").replace("http://", "")
    return domain.split("/", 1)[0].strip()


def _render_reports_page(projects_root: Path) -> str:
    rows = _report_rows(projects_root)
    if rows:
        body = "<table><thead><tr><th>Domain</th><th>Updated</th><th>Open</th></tr></thead><tbody>" + "".join(
            f'<tr><td>{escape(row["domain"])}</td><td>{escape(_format_ts(row["updated_at"]))}</td><td><a href="{escape(row["url"])}">Open report</a></td></tr>'
            for row in rows
        ) + "</tbody></table>"
    else:
        body = '<div class="panel muted">No generated reports found under projects/&lt;domain&gt;/report.</div>'
    return _layout("Reports", "reports", body, subtitle=f"Reading generated reports from {projects_root}.")


def _render_comparisons_page(projects_root: Path) -> str:
    rows = _comparison_rows(projects_root)
    if rows:
        body = '<ul class="list">' + "".join(
            f'<li><a href="{escape(row["url"])}">{escape(row["name"])}</a><span class="muted">updated {escape(_format_ts(row["updated_at"]))}</span></li>'
            for row in rows
        ) + "</ul>"
    else:
        body = '<div class="panel muted">No generated comparisons found under projects/_compare.</div>'
    return _layout("Comparisons", "comparisons", body, subtitle="Open outputs from site-audit compare.")


def _render_settings_page(schema: list[dict], env_file: Path, *, saved: bool) -> str:
    values = read_env_file(env_file)
    groups: dict[str, list[dict]] = {}
    for row in schema:
        groups.setdefault(row["command"], []).append(row)
    sections = "\n".join(_render_group(name, rows, values) for name, rows in groups.items())
    status = '<div class="notice">Saved to .env.</div>' if saved else ""
    body = f"""
    {status}
    <form method="post">
      {sections}
      <div class="actions"><button type="submit">Save settings</button></div>
    </form>"""
    return _layout(
        "Settings",
        "settings",
        body,
        subtitle=f"Editing {env_file}. Values here become defaults; command-line flags still override them.",
    )


def _report_rows(projects_root: Path) -> list[dict]:
    rows = []
    for report_dir in sorted(projects_root.glob("*/report")):
        metrics = report_dir / "site_metrics.json"
        if metrics.is_file():
            rows.append({
                "domain": report_dir.parent.name,
                "url": f"/reports/{report_dir.parent.name}/",
                "updated_at": int(metrics.stat().st_mtime),
            })
    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    return rows


def _comparison_rows(projects_root: Path) -> list[dict]:
    rows = []
    for index in sorted((projects_root / "_compare").glob("*/index.html")):
        domains = _comparison_domains(index.parent)
        rows.append({
            "name": index.parent.name,
            "url": f"/comparisons/{index.parent.name}/",
            "updated_at": int(index.stat().st_mtime),
            "domains": domains,
        })
    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    return rows


def _comparison_domains(compare_dir: Path) -> list[str]:
    comparison_json = compare_dir / "comparison.json"
    if comparison_json.is_file():
        try:
            payload = json.loads(comparison_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        domains = payload.get("domains")
        if isinstance(domains, list):
            return [_clean_domain(str(domain)) for domain in domains if _clean_domain(str(domain))]
    parts = [
        part.strip()
        for part in compare_dir.name.replace("_", "-").split("-vs-")
        if part.strip()
    ]
    return [_clean_domain(part) for part in parts]


def _format_ts(value: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


def _render_group(name: str, rows: list[dict], values: dict[str, str]) -> str:
    body = "\n".join(_render_row(row, values.get(row["env_key"], "")) for row in rows)
    return f'<section><h2>{escape(name)}</h2><div class="card">{body}</div></section>'


def _render_row(row: dict, value: str) -> str:
    env_key = row["env_key"]
    current = value if value != "" else row["default"]
    control = _control(row, current)
    return f"""
      <div class="row">
        <div>
          <label for="{escape(env_key)}">{escape(row["flag"])}</label>
          <code>{escape(env_key)}</code>
          <div class="default">Default: {escape(row["default"] or "empty")}</div>
        </div>
        <div>
          {control}
          <div class="help">{escape(row["help"])}</div>
          {_render_details(row)}
        </div>
      </div>"""


def _render_details(row: dict) -> str:
    details = row.get("details") or {}
    items = [
        ("What", details.get("what", "")),
        ("Why", details.get("why", "")),
        ("Format", details.get("format", "")),
        ("Example", details.get("example", "")),
    ]
    return '<div class="details">' + "".join(
        f'<div class="detail"><strong>{escape(label)}</strong>{escape(text or "n/a")}</div>'
        for label, text in items
    ) + "</div>"


def _control(row: dict, current: str) -> str:
    key = escape(row["env_key"])
    val = escape(str(current))
    if row["kind"] == "bool":
        checked = " checked" if str(current).lower() in {"1", "true", "yes", "on"} else ""
        return f'<input type="hidden" name="{key}" value="false"><label><input style="width:auto" type="checkbox" name="{key}" value="true"{checked}> Enabled</label>'
    if row["kind"] == "list":
        return f'<textarea id="{key}" name="{key}" placeholder="one value per line or comma-separated">{val}</textarea>'
    if row["kind"] == "choice":
        options = "".join(
            f'<option value="{escape(str(choice))}"{" selected" if str(choice) == str(current) else ""}>{escape(str(choice))}</option>'
            for choice in row["choices"]
        )
        return f'<select id="{key}" name="{key}">{options}</select>'
    input_type = "password" if row["kind"] == "secret" else ("number" if row["kind"] == "number" else "text")
    step = ' step="any"' if input_type == "number" else ""
    return f'<input id="{key}" name="{key}" type="{input_type}" value="{val}"{step}>'
