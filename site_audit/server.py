"""Local report server with a lightweight scan dashboard."""

from __future__ import annotations

import json
import logging
import mimetypes
import subprocess
import sys
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

LOG = logging.getLogger(__name__)


def _build_handler(ui_dir: Path, data_dir: Path, projects_root: Path):
    scans: dict[str, dict] = {}
    scans_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            LOG.debug("%s - %s", self.address_string(), format % args)

        def do_GET(self):  # noqa: N802
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                self._send_html(_render_home(projects_root))
                return
            if path == "/scans":
                self._send_html(_render_scans_page(projects_root, _scan_rows()))
                return
            if path == "/reports":
                self._send_html(_render_reports_page(projects_root))
                return
            if path == "/comparisons":
                self._send_html(_render_comparisons_page(projects_root))
                return
            if path == "/api/reports":
                self._send_json({"reports": _report_rows(projects_root)})
                return
            if path == "/api/scans":
                self._send_json({"scans": _scan_rows()})
                return
            if path.startswith("/reports/") and path.rstrip("/").count("/") == 2:
                self._serve_file(ui_dir / "index.html")
                return
            if path.startswith("/comparisons/") and path.rstrip("/").count("/") == 2:
                name = path.strip("/").split("/", 1)[1]
                self._serve_file(projects_root / "_compare" / name / "index.html")
                return
            if path.startswith("/reports-data/"):
                parts = path[len("/reports-data/"):].split("/", 1)
                if len(parts) != 2:
                    self.send_error(404, "Not found")
                    return
                domain, rel = parts
                self._serve_file(projects_root / domain / "report" / rel)
                return
            if path.startswith("/data/"):
                rel = path[len("/data/"):]
                self._serve_file(data_dir / rel)
                return
            if path.startswith("/comparison-data/"):
                rel = path[len("/comparison-data/"):]
                self._serve_file(projects_root / "_compare" / rel)
                return
            target = ui_dir / path.lstrip("/")
            if target.is_file():
                self._serve_file(target)
                return
            self.send_error(404, "Not found")

        def do_POST(self):  # noqa: N802
            path = unquote(self.path.split("?", 1)[0])
            if path != "/api/scans":
                self.send_error(404, "Not found")
                return
            form = self._read_form()
            domain = (form.get("domain") or [""])[-1].strip()
            if not domain:
                self._send_json({"ok": False, "error": "Domain is required."}, status=400)
                return
            scan = _start_scan(domain, form)
            if "application/json" in self.headers.get("Content-Type", ""):
                self._send_json({"ok": True, "scan": scan})
                return
            self.send_response(303)
            self.send_header("Location", "/scans")
            self.end_headers()

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

        def _send_json(self, payload: dict, *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str, *, status: int = 200) -> None:
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _read_form(self) -> dict[str, list[str]]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8")
            if "application/json" in self.headers.get("Content-Type", ""):
                try:
                    payload = json.loads(body or "{}")
                except json.JSONDecodeError:
                    return {}
                return {str(key): [str(value)] for key, value in payload.items()}
            return parse_qs(body, keep_blank_values=True)

    def _scan_rows() -> list[dict]:
        with scans_lock:
            rows = [dict(row) for row in scans.values()]
        rows.sort(key=lambda row: row.get("started_at", 0), reverse=True)
        return rows

    def _start_scan(domain: str, form: dict[str, list[str]]) -> dict:
        now = int(time.time())
        slug = _safe_slug(domain)
        scan_id = f"{slug}-{now}"
        log_dir = projects_root / slug / "cache"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"scan-{now}.log"
        cmd = [sys.executable, "-m", "site_audit.cli", "run", domain, "--projects-root", str(projects_root)]
        _append_optional_arg(cmd, form, "max_pages", "--max-pages")
        _append_optional_arg(cmd, form, "search_provider", "--search-provider")
        if _form_bool(form, "competitive_auto"):
            cmd.append("--competitive-auto")
        if _form_bool(form, "clean"):
            cmd.append("--clean")
        _append_repeated_arg(cmd, form, "competitive_auto_product_seed", "--competitive-auto-product-seed")
        row = {
            "id": scan_id,
            "domain": domain,
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "command": " ".join(cmd),
            "log_path": str(log_path),
            "report_url": f"/reports/{slug}/",
        }
        with scans_lock:
            scans[scan_id] = row
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write("$ " + " ".join(cmd) + "\n\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), stdout=log_fh, stderr=subprocess.STDOUT, text=True)

        def wait_for_scan() -> None:
            code = proc.wait()
            log_fh.close()
            with scans_lock:
                current = scans.get(scan_id, row)
                current["status"] = "finished" if code == 0 else "failed"
                current["returncode"] = code
                current["finished_at"] = int(time.time())
                scans[scan_id] = current

        threading.Thread(target=wait_for_scan, daemon=True).start()
        return row

    return Handler


def serve(report_dir: Path, ui_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    report_dir = Path(report_dir)
    ui_dir = Path(ui_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"No report at {report_dir}. Run `site-audit run <domain>` first.")
    if not (ui_dir / "index.html").is_file():
        raise FileNotFoundError(f"UI assets missing: {ui_dir}/index.html")

    projects_root = report_dir.parent.parent if report_dir.name == "report" else Path("projects")
    handler = _build_handler(ui_dir, report_dir, projects_root)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"\n  ➜  Site-audit viewer ready: http://{host}:{port}/")
    print(f"     scan dashboard: http://{host}:{port}/scans")
    print(f"     serving report: {report_dir}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
        httpd.server_close()


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
        rows.append({
            "name": index.parent.name,
            "url": f"/comparisons/{index.parent.name}/",
            "updated_at": int(index.stat().st_mtime),
            "domains": _comparison_domains(index.parent),
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
    return _layout("Site Audit", "home", body, subtitle="Local app for reports, comparisons, and scans.")


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


def _layout(title: str, active: str, body: str, *, subtitle: str = "") -> str:
    nav_items = [
        ("home", "/", "Overview"),
        ("reports", "/reports", "Reports"),
        ("comparisons", "/comparisons", "Comparisons"),
        ("scans", "/scans", "Scans"),
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
    header {{ position:sticky; top:0; z-index:2; display:grid; grid-template-columns:1fr auto; gap:20px; align-items:center; padding:18px 28px; background:#fffdfa; border-bottom:1px solid #eadfce; }}
    nav {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    nav a {{ color:#4f463c; text-decoration:none; font-weight:800; font-size:13px; padding:9px 11px; border-radius:8px; }}
    nav a.active, nav a:hover {{ background:#231f1a; color:#fffdfa; }}
    main {{ max-width:1120px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 4px; font-size:28px; }}
    h2 {{ margin:28px 0 12px; font-size:18px; text-transform:capitalize; }}
    .hint, .muted {{ color:#756d65; font-size:13px; line-height:1.45; }}
    .panel {{ background:#fffdfa; border:1px solid #eadfce; border-radius:10px; padding:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin-top:12px; }}
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
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; }}
    label {{ display:block; font-weight:700; font-size:13px; margin-bottom:5px; }}
    input, select, textarea {{ width:100%; box-sizing:border-box; border:1px solid #d8cbb9; border-radius:8px; padding:9px 10px; font:inherit; background:white; }}
    textarea {{ min-height:80px; }}
    button {{ border:0; border-radius:8px; padding:11px 16px; background:#ff8a1f; color:white; font-weight:800; cursor:pointer; }}
    .status {{ display:inline-block; padding:3px 7px; border-radius:999px; background:#eee2d1; }}
    .status.running {{ background:#fff3c4; color:#805800; }}
    .status.finished {{ background:#dff4e8; color:#17613d; }}
    .status.failed {{ background:#f8d7da; color:#8a1f2d; }}
    table {{ width:100%; border-collapse:collapse; background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
    td, th {{ padding:11px 12px; border-top:1px solid #f0e7da; text-align:left; font-size:13px; vertical-align:top; }}
    th {{ border-top:0; color:#756d65; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    ul.list {{ margin:0; padding:0; list-style:none; background:#fffdfa; border:1px solid #eadfce; border-radius:10px; overflow:hidden; }}
    ul.list li {{ display:flex; justify-content:space-between; gap:14px; padding:12px 14px; border-top:1px solid #f0e7da; }}
    ul.list li:first-child {{ border-top:0; }}
    code {{ color:#9a5200; font-size:12px; word-break:break-all; }}
    @media (max-width:820px) {{
      header {{ grid-template-columns:1fr; }}
      nav {{ justify-content:flex-start; }}
      .cards, .grid {{ grid-template-columns:1fr; }}
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


def _format_ts(value: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


def _append_optional_arg(cmd: list[str], form: dict[str, list[str]], key: str, flag: str) -> None:
    value = (form.get(key) or [""])[-1].strip()
    if value:
        cmd.extend([flag, value])


def _append_repeated_arg(cmd: list[str], form: dict[str, list[str]], key: str, flag: str) -> None:
    raw = (form.get(key) or [""])[-1]
    for part in raw.replace("\n", ",").split(","):
        value = part.strip()
        if value:
            cmd.extend([flag, value])


def _form_bool(form: dict[str, list[str]], key: str) -> bool:
    return (form.get(key) or [""])[-1].lower() in {"1", "true", "yes", "on"}


def _safe_slug(value: str) -> str:
    return value.strip().replace("https://", "").replace("http://", "").strip("/").replace("/", "_")


def _render_scans_page(projects_root: Path, scans: list[dict]) -> str:
    reports = _report_rows(projects_root)
    scan_rows = "\n".join(
        f"""<tr>
          <td>{_esc(row.get("domain", ""))}</td>
          <td><span class="status {row.get("status", "")}">{_esc(row.get("status", ""))}</span></td>
          <td><a href="{_esc(row.get("report_url", "#"))}">Open report</a></td>
          <td><code>{_esc(row.get("log_path", ""))}</code></td>
        </tr>"""
        for row in scans
    ) or '<tr><td colspan="4" class="muted">No scans started from this server yet.</td></tr>'
    report_rows = "\n".join(
        f'<li><a href="{_esc(row["url"])}">{_esc(row["domain"])}</a> '
        f'<span class="muted">updated {time.strftime("%Y-%m-%d %H:%M", time.localtime(row["updated_at"]))}</span></li>'
        for row in reports
    ) or '<li class="muted">No generated reports found.</li>'
    body = f"""
    <section class="panel">
      <h2>Start Scan</h2>
      <form method="post" action="/api/scans" enctype="application/x-www-form-urlencoded">
        <div class="grid">
          <div><label for="domain">Domain</label><input id="domain" name="domain" placeholder="example.com" required></div>
          <div><label for="max_pages">Max pages</label><input id="max_pages" name="max_pages" type="number" min="1" placeholder="500"></div>
          <div>
            <label for="search_provider">Search provider</label>
            <select id="search_provider" name="search_provider">
              <option value="">default from .env / CLI</option>
              <option value="auto">auto</option>
              <option value="all">all</option>
              <option value="gsc">gsc</option>
              <option value="google_ads">google_ads</option>
              <option value="ahrefs">ahrefs</option>
              <option value="dataforseo">dataforseo</option>
              <option value="none">none</option>
            </select>
          </div>
          <div>
            <label><input style="width:auto" type="checkbox" name="competitive_auto" value="true"> Competitive auto analysis</label>
            <label><input style="width:auto" type="checkbox" name="clean" value="true"> Clean cache first</label>
          </div>
          <div style="grid-column:1/-1"><label for="competitive_auto_product_seed">Product/service seeds</label><textarea id="competitive_auto_product_seed" name="competitive_auto_product_seed" placeholder="help desk software, live chat software"></textarea></div>
        </div>
        <p class="hint">Long scans keep running in the background while this server stays open. Logs are written under the project's cache directory.</p>
        <button type="submit">Start scan</button>
      </form>
    </section>
    <section class="panel">
      <h2>Running And Recent Scans</h2>
      <table><thead><tr><th>Domain</th><th>Status</th><th>Result</th><th>Log</th></tr></thead><tbody>{scan_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Available Reports</h2>
      <ul>{report_rows}</ul>
    </section>"""
    return _layout(
        "Site Audit Scans",
        "scans",
        body,
        subtitle="Start a scan from the browser, then open generated reports without using the CLI.",
    )


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
