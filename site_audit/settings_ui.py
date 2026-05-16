"""Small local settings editor for .env-backed site-audit defaults."""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .config_env import collect_actions, env_names, read_env_file, update_env_file

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
]


def serve_settings_ui(
    parser: argparse.ArgumentParser,
    *,
    env_file: Path,
    host: str = "127.0.0.1",
    port: int = 8780,
) -> None:
    env_file = Path(env_file)
    schema = _schema(parser)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):  # noqa: N802
            self._send_html(_render_page(schema, env_file, saved=False))

        def do_POST(self):  # noqa: N802
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
            self._send_html(_render_page(schema, env_file, saved=True))

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  ➜  Site-audit settings editor: http://{host}:{port}/")
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
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
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


def _render_page(schema: list[dict], env_file: Path, *, saved: bool) -> str:
    values = read_env_file(env_file)
    groups: dict[str, list[dict]] = {}
    for row in schema:
        groups.setdefault(row["command"], []).append(row)
    sections = "\n".join(_render_group(name, rows, values) for name, rows in groups.items())
    status = '<div class="notice">Saved to .env.</div>' if saved else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Site Audit Settings</title>
  <style>
    body {{ margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background:#f6f2ea; color:#231f1a; }}
    header {{ position:sticky; top:0; z-index:2; padding:20px 28px; background:#fffdfa; border-bottom:1px solid #eadfce; }}
    main {{ max-width:1120px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 4px; font-size:28px; }}
    h2 {{ margin:28px 0 12px; font-size:18px; text-transform:capitalize; }}
    .hint {{ color:#756d65; font-size:13px; }}
    .notice {{ margin-top:12px; display:inline-block; padding:8px 12px; border-radius:8px; background:#e8f7ef; color:#17613d; }}
    .card {{ background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
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
  </style>
</head>
<body>
  <header>
    <h1>Site Audit Settings</h1>
    <div class="hint">Editing <code style="display:inline">{escape(str(env_file))}</code>. Values here become defaults; command-line flags still override them.</div>
    {status}
  </header>
  <main>
    <form method="post">
      {sections}
      <div class="actions"><button type="submit">Save settings</button></div>
    </form>
  </main>
</body>
</html>"""


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
