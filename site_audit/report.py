"""Serialize an ``AuditResult`` (+ optional UMAP projection) to disk.

Output paths mirror the Hugo audit pipeline so the existing D3
viewer template can render against either source unchanged:

::

    out/
      <domain>/
        site_metrics.json
        section_report.json
        page_drift.csv
        outliers.csv
        duplicates.csv
        scatterplot.json
        pages.json
"""

from __future__ import annotations

import csv
import html
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import numpy as np

from .analyzer import AuditResult, recommend_action

LOG = logging.getLogger(__name__)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_safe_iter(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = rows if isinstance(rows, list) else list(rows)
    if not rows_list:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows_list:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(_csv_safe_row(row))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _write_json_object_with_array(path: Path, payload: dict, array_key: str, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        first = True
        for key, value in payload.items():
            if key == array_key:
                continue
            if not first:
                f.write(",\n")
            f.write(f"  {json.dumps(str(key), ensure_ascii=False)}: ")
            json.dump(value, f, ensure_ascii=False, indent=2)
            first = False
        if not first:
            f.write(",\n")
        f.write(f"  {json.dumps(array_key, ensure_ascii=False)}: [")
        wrote_row = False
        for row in rows:
            if wrote_row:
                f.write(",")
            f.write("\n    ")
            json.dump(row, f, ensure_ascii=False)
            wrote_row = True
        if wrote_row:
            f.write("\n  ]\n")
        else:
            f.write("]\n")
        f.write("}\n")
    tmp_path.replace(path)


def _write_report_json(path: Path, payload, label: str) -> None:
    LOG.info("  report export: %s", label)
    _write_json(path, payload)


def _html_escape(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _format_number(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _html_escape(value)


def _technical_report_domain(domain: Optional[str], technical_seo: dict) -> str:
    if domain:
        return str(domain)
    for page in (technical_seo or {}).get("pages") or []:
        parsed = urlparse(str(page.get("url") or ""))
        if parsed.netloc:
            return parsed.netloc
    return "Technical SEO audit"


def _technical_report_rows(title: str, rows: list[tuple[str, object]]) -> str:
    rendered = "\n".join(
        f"<tr><th>{_html_escape(label)}</th><td>{_format_number(value)}</td></tr>"
        for label, value in rows
        if value not in (None, "")
    )
    if not rendered:
        return ""
    return f"""
      <section>
        <h2>{_html_escape(title)}</h2>
        <table>{rendered}</table>
      </section>
    """


def _technical_report_count_rows(counts: dict, limit: int = 25) -> str:
    if not isinstance(counts, dict) or not counts:
        return "<p>No issues recorded.</p>"
    rows = sorted(counts.items(), key=lambda item: item[1] or 0, reverse=True)[:limit]
    return "<table>" + "\n".join(
        f"<tr><th>{_html_escape(label)}</th><td>{_format_number(value)}</td></tr>"
        for label, value in rows
    ) + "</table>"


def _first_present(mapping: dict, keys: list[str]):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _inline_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _technical_report_links(output_dir: Path) -> str:
    files = [
        ("Run summary", "run_summary.json"),
        ("Technical pages JSON", "technical_pages.json"),
        ("Technical pages CSV", "technical_pages.csv"),
        ("Technical issues JSON", "technical_issues.json"),
        ("Technical issues CSV", "technical_issues.csv"),
        ("Issue catalog JSON", "technical_issue_catalog.json"),
        ("Issue catalog CSV", "technical_issue_catalog.csv"),
        ("Indexability JSON", "indexability.json"),
        ("Sitemap coverage JSON", "sitemap_coverage.json"),
        ("Canonical consistency JSON", "canonical_consistency.json"),
        ("Performance JSON", "performance.json"),
        ("Resource status JSON", "resource_status.json"),
    ]
    links = []
    for label, filename in files:
        if (output_dir / filename).is_file():
            links.append(f'<a href="{_html_escape(filename)}">{_html_escape(label)}</a>')
    if not links:
        return ""
    return "<div class=\"links\">" + "\n".join(links) + "</div>"


def write_technical_index_html(output_dir: Path, technical_seo: dict, domain: Optional[str] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_name = _technical_report_domain(domain, technical_seo)
    summary = (technical_seo or {}).get("summary") or {}
    issue_counts = (technical_seo or {}).get("issue_counts") or {}
    category_counts = (technical_seo or {}).get("category_counts") or {}
    severity_counts = (technical_seo or {}).get("severity_counts") or {}
    interpretation = (technical_seo or {}).get("interpretation") or {}
    summary_rows = [
        ("Pages analyzed", _first_present(summary, ["total_pages", "pages"])),
        ("Pages fetched", summary.get("fetched_pages")),
        ("Pages skipped", summary.get("skipped_pages")),
        ("Noindex pages dropped", summary.get("noindex_dropped")),
        ("Canonical duplicates dropped", summary.get("canonical_duplicates_dropped")),
        ("Pages with issues", summary.get("pages_with_issues")),
        ("Technical issues", _first_present(summary, ["total_issues", "technical_issues"])),
        ("High priority issues", _first_present(summary, ["high_issues", "high_technical_issues"])),
        ("Catalog issue types", summary.get("catalog_issue_types")),
    ]
    bootstrap = {
        "domain": domain_name,
        "summaryRows": [(label, value) for label, value in summary_rows if value not in (None, "")],
        "summary": summary,
        "severityCounts": severity_counts,
        "categoryCounts": category_counts,
        "issueCounts": issue_counts,
        "issueCatalog": (technical_seo or {}).get("issue_catalog") or [],
    }
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html_escape(domain_name)} technical SEO audit</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5f6b7a;
      --border: #d8e0ea;
      --accent: #0b6bcb;
      --danger: #b42318;
      --warn: #b54708;
      --info: #175cd3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      margin-bottom: 24px;
    }}
    .header-row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      font-weight: 650;
    }}
    p {{
      margin: 0 0 16px;
      color: var(--muted);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .card strong {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
      line-height: 1.1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 9px 0;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    tr:last-child th, tr:last-child td {{
      border-bottom: 0;
    }}
    th {{
      width: 70%;
      color: var(--muted);
      font-weight: 500;
      padding-right: 16px;
    }}
    td {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(140px, 2fr) 86px;
      gap: 12px;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }}
    .bar-row:last-child {{
      border-bottom: 0;
    }}
    .bar-label {{
      overflow-wrap: anywhere;
    }}
    .bar-track {{
      height: 10px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      min-width: 2px;
      background: var(--accent);
    }}
    .bar-fill.high {{ background: var(--danger); }}
    .bar-fill.medium {{ background: var(--warn); }}
    .bar-fill.low {{ background: var(--info); }}
    .bar-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 160px 160px;
      gap: 10px;
      margin: 12px 0 16px;
    }}
    input, select, button {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      min-height: 38px;
      padding: 7px 10px;
    }}
    button {{
      cursor: pointer;
    }}
    .issue-table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .issue-table th, .issue-table td {{
      min-width: 130px;
      padding: 10px 12px;
      text-align: left;
    }}
    .issue-table th:first-child, .issue-table td:first-child {{
      min-width: 260px;
    }}
    .issue-table th:last-child, .issue-table td:last-child {{
      min-width: 360px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 650;
      background: #eef2f6;
      color: var(--muted);
      text-transform: capitalize;
    }}
    .pill.high {{ background: #fee4e2; color: var(--danger); }}
    .pill.medium {{ background: #fef0c7; color: var(--warn); }}
    .pill.low {{ background: #d1e9ff; color: var(--info); }}
    .pager {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
      flex-wrap: wrap;
      color: var(--muted);
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid #b8d6f3;
      background: #eef6ff;
      border-radius: 6px;
      padding: 7px 10px;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 720px) {{
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .bar-row {{
        grid-template-columns: 1fr 86px;
      }}
      .bar-track {{
        grid-column: 1 / -1;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="header-row">
        <div>
          <h1>{_html_escape(domain_name)} technical SEO audit</h1>
          <p>Interactive technical SEO dashboard generated from the audit exports.</p>
        </div>
      </div>
    </header>
    <div id="cards" class="cards"></div>
    <div class="grid">
      <section>
        <h2>Issue severity</h2>
        <div id="severity-bars"></div>
      </section>
      <section>
        <h2>Issue categories</h2>
        <div id="category-bars"></div>
      </section>
      <section class="wide">
        <h2>Top issue types</h2>
        <div id="issue-bars"></div>
      </section>
      <section class="wide">
        <h2>All issues</h2>
        <p id="issue-status">Loading technical_issues.json...</p>
        <div class="toolbar">
          <input id="issue-search" type="search" placeholder="Filter by URL, issue type, category, severity, or message">
          <select id="severity-filter"><option value="">All severities</option></select>
          <select id="category-filter"><option value="">All categories</option></select>
        </div>
        <div class="issue-table-wrap">
          <table class="issue-table">
            <thead>
              <tr>
                <th>URL</th>
                <th>Severity</th>
                <th>Category</th>
                <th>Issue</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody id="issue-rows"></tbody>
          </table>
        </div>
        <div class="pager">
          <span id="pager-label"></span>
          <div>
            <button id="prev-page" type="button">Previous</button>
            <button id="next-page" type="button">Next</button>
          </div>
        </div>
      </section>
      <section class="wide">
        <h2>Exports</h2>
        <p>{_html_escape(interpretation.get("exports") or "Open the JSON or CSV files for raw row-level audit data.")}</p>
        {_technical_report_links(output_dir)}
      </section>
    </div>
  </main>
  <script id="technical-bootstrap" type="application/json">{_inline_json(bootstrap)}</script>
  <script>
    const bootstrap = JSON.parse(document.getElementById('technical-bootstrap').textContent);
    const fmt = new Intl.NumberFormat();
    const pageSize = 250;
    let allIssues = [];
    let filteredIssues = [];
    let page = 0;

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function normalize(value) {{
      return String(value ?? '').toLowerCase();
    }}

    function renderCards() {{
      const cards = [
        ['Pages analyzed', bootstrap.summary.total_pages ?? bootstrap.summary.pages],
        ['Pages with issues', bootstrap.summary.pages_with_issues],
        ['Technical issues', bootstrap.summary.total_issues ?? bootstrap.summary.technical_issues],
        ['High priority issues', bootstrap.summary.high_issues ?? bootstrap.summary.high_technical_issues],
        ['Catalog issue types', bootstrap.summary.catalog_issue_types],
      ].filter(([, value]) => value !== undefined && value !== null && value !== '');
      document.getElementById('cards').innerHTML = cards.map(([label, value]) =>
        `<div class="card"><span>${{escapeHtml(label)}}</span><strong>${{fmt.format(Number(value) || 0)}}</strong></div>`
      ).join('');
    }}

    function renderBars(id, counts, limit = 12) {{
      const entries = Object.entries(counts || {{}})
        .sort((a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0))
        .slice(0, limit);
      const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
      document.getElementById(id).innerHTML = entries.map(([label, value]) => {{
        const cls = ['high', 'medium', 'low'].includes(label) ? label : '';
        const pct = Math.max(1, Math.round(((Number(value) || 0) / max) * 100));
        return `<div class="bar-row">
          <div class="bar-label">${{escapeHtml(label.replaceAll('_', ' '))}}</div>
          <div class="bar-track"><div class="bar-fill ${{cls}}" style="width:${{pct}}%"></div></div>
          <div class="bar-value">${{fmt.format(Number(value) || 0)}}</div>
        </div>`;
      }}).join('') || '<p>No issues recorded.</p>';
    }}

    function issueField(issue, names) {{
      for (const name of names) {{
        const value = issue?.[name];
        if (value !== undefined && value !== null && value !== '') return value;
      }}
      return '';
    }}

    function fillFilters() {{
      const severities = [...new Set(allIssues.map(row => issueField(row, ['severity', 'importance'])).filter(Boolean))].sort();
      const categories = [...new Set(allIssues.map(row => issueField(row, ['category'])).filter(Boolean))].sort();
      document.getElementById('severity-filter').innerHTML = '<option value="">All severities</option>' + severities.map(v => `<option value="${{escapeHtml(v)}}">${{escapeHtml(v)}}</option>`).join('');
      document.getElementById('category-filter').innerHTML = '<option value="">All categories</option>' + categories.map(v => `<option value="${{escapeHtml(v)}}">${{escapeHtml(v)}}</option>`).join('');
    }}

    function applyFilters() {{
      const q = normalize(document.getElementById('issue-search').value);
      const severity = document.getElementById('severity-filter').value;
      const category = document.getElementById('category-filter').value;
      filteredIssues = allIssues.filter(issue => {{
        if (severity && issueField(issue, ['severity', 'importance']) !== severity) return false;
        if (category && issueField(issue, ['category']) !== category) return false;
        if (!q) return true;
        return [
          issueField(issue, ['url', 'page_url']),
          issueField(issue, ['severity', 'importance']),
          issueField(issue, ['category']),
          issueField(issue, ['issue_type', 'name', 'issue']),
          issueField(issue, ['message', 'details', 'description', 'recommendation']),
        ].some(value => normalize(value).includes(q));
      }});
      page = 0;
      renderIssuePage();
    }}

    function renderIssuePage() {{
      const start = page * pageSize;
      const rows = filteredIssues.slice(start, start + pageSize);
      document.getElementById('issue-rows').innerHTML = rows.map(issue => {{
        const severity = issueField(issue, ['severity', 'importance']);
        const issueName = issueField(issue, ['issue_type', 'name', 'issue']).replaceAll('_', ' ');
        const details = issueField(issue, ['message', 'details', 'description', 'recommendation']);
        return `<tr>
          <td>${{escapeHtml(issueField(issue, ['url', 'page_url']))}}</td>
          <td><span class="pill ${{escapeHtml(normalize(severity))}}">${{escapeHtml(severity)}}</span></td>
          <td>${{escapeHtml(issueField(issue, ['category']))}}</td>
          <td>${{escapeHtml(issueName)}}</td>
          <td>${{escapeHtml(details)}}</td>
        </tr>`;
      }}).join('');
      const end = Math.min(start + rows.length, filteredIssues.length);
      document.getElementById('pager-label').textContent = filteredIssues.length
        ? `${{fmt.format(start + 1)}}-${{fmt.format(end)}} of ${{fmt.format(filteredIssues.length)}} issues`
        : 'No matching issues';
      document.getElementById('prev-page').disabled = page === 0;
      document.getElementById('next-page').disabled = end >= filteredIssues.length;
    }}

    async function loadIssues() {{
      try {{
        const response = await fetch('technical_issues.json');
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const payload = await response.json();
        allIssues = Array.isArray(payload.issues) ? payload.issues : [];
        filteredIssues = allIssues;
        document.getElementById('issue-status').textContent = `${{fmt.format(allIssues.length)}} issues loaded from technical_issues.json.`;
        fillFilters();
        renderIssuePage();
      }} catch (err) {{
        document.getElementById('issue-status').innerHTML =
          `Unable to load all issue rows in this browser context. Use <a href="technical_issues.csv">technical_issues.csv</a> or serve the report with <code>site-audit serve</code>.`;
      }}
    }}

    document.getElementById('issue-search').addEventListener('input', applyFilters);
    document.getElementById('severity-filter').addEventListener('change', applyFilters);
    document.getElementById('category-filter').addEventListener('change', applyFilters);
    document.getElementById('prev-page').addEventListener('click', () => {{ page = Math.max(0, page - 1); renderIssuePage(); }});
    document.getElementById('next-page').addEventListener('click', () => {{ page += 1; renderIssuePage(); }});

    renderCards();
    renderBars('severity-bars', bootstrap.severityCounts, 6);
    renderBars('category-bars', bootstrap.categoryCounts, 12);
    renderBars('issue-bars', bootstrap.issueCounts, 25);
    loadIssues();
  </script>
</body>
</html>
"""
    out_path = output_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def _slim_linkgraph_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if isinstance(out.get("anchor_relevance"), dict):
        anchor = dict(out["anchor_relevance"])
        anchor["links"] = []
        anchor["weak_links"] = (anchor.get("weak_links") or [])[:500]
        out["anchor_relevance"] = anchor
    if isinstance(out.get("contextual_link_impact"), dict):
        contextual = dict(out["contextual_link_impact"])
        contextual["links"] = []
        contextual["top_contextual_links"] = (contextual.get("top_contextual_links") or [])[:500]
        contextual["weak_context_links"] = (contextual.get("weak_context_links") or [])[:500]
        contextual["source_pages"] = [
            {**row, "strongest_outbound_links": (row.get("strongest_outbound_links") or [])[:5]}
            for row in (contextual.get("source_pages") or [])[:250]
        ]
        out["contextual_link_impact"] = contextual
    if isinstance(out.get("link_flow"), dict):
        flow = dict(out["link_flow"])
        flow["edges"] = (flow.get("edges") or [])[:2500]
        flow["nodes"] = (flow.get("nodes") or [])[:2500]
        out["link_flow"] = flow
    if isinstance(out.get("hub_bottlenecks"), dict):
        hubs = dict(out["hub_bottlenecks"])
        hubs["pages"] = []
        hubs["risks"] = (hubs.get("risks") or [])[:250]
        hubs["bridges"] = (hubs.get("bridges") or [])[:150]
        hubs["bottlenecks"] = (hubs.get("bottlenecks") or [])[:150]
        hubs["authority_hubs"] = (hubs.get("authority_hubs") or [])[:150]
        out["hub_bottlenecks"] = hubs
    if isinstance(out.get("high_demand_low_link"), dict):
        demand = dict(out["high_demand_low_link"])
        demand["pages"] = (demand.get("pages") or [])[:2000]
        out["high_demand_low_link"] = demand
    if isinstance(out.get("traffic_weighted_pagerank"), dict):
        pagerank = dict(out["traffic_weighted_pagerank"])
        pagerank["pages"] = (pagerank.get("pages") or [])[:2500]
        out["traffic_weighted_pagerank"] = pagerank
    if isinstance(out.get("page_link_counts"), list):
        out["page_link_counts"] = out["page_link_counts"][:3000]
    return out


def write_internal_linkbuilding_csv(
    path: Path,
    result: AuditResult,
    recommendations: list[dict],
    search_payload: Optional[dict] = None,
) -> list[dict]:
    page_by_url = {page.url: page for page in result.pages}
    paid_keywords = _paid_keywords(search_payload)
    rows: list[dict] = []
    seen_paragraph_anchors: set[tuple[str, str, str]] = set()
    for rec in recommendations or []:
        source_url = rec.get("source_url") or ""
        target_url = rec.get("target_url") or ""
        suggested_anchor = rec.get("suggested_anchor") or rec.get("anchor") or ""
        if _canonical_url(source_url) == _canonical_url(target_url):
            continue
        paid_candidate = _best_paid_anchor_candidate(
            paid_keywords,
            rec.get("paragraph_excerpt") or "",
            target_url,
            rec.get("target_title") or "",
            page_by_url.get(target_url),
        )
        anchor = paid_candidate.get("keyword") or suggested_anchor
        if not source_url or not target_url or not anchor:
            continue
        paragraph_index = str(rec.get("paragraph_index", ""))
        anchor_key = (source_url, paragraph_index, _normalize_anchor(anchor))
        if anchor_key in seen_paragraph_anchors:
            continue
        seen_paragraph_anchors.add(anchor_key)
        target_page = page_by_url.get(target_url)
        source_page = page_by_url.get(source_url)
        destination_title = rec.get("target_title") or getattr(target_page, "title", "") or target_url
        destination_description = getattr(target_page, "description", "") if target_page else ""
        rows.append({
            "url_where_to_place_link": source_url,
            "source_page_title": rec.get("source_title") or getattr(source_page, "title", "") or source_url,
            "paragraph_index": paragraph_index,
            "paragraph_excerpt": rec.get("paragraph_excerpt", ""),
            "exact_keywords_to_link": anchor,
            "original_suggested_anchor": suggested_anchor,
            "anchor_source": "paid_converting_keyword" if paid_candidate else "semantic_paragraph_match",
            "paid_keyword_candidate": paid_candidate.get("keyword", ""),
            "paid_conversions": paid_candidate.get("paid_conversions", ""),
            "paid_conversion_value": paid_candidate.get("paid_conversion_value", ""),
            "paid_cost": paid_candidate.get("paid_cost", ""),
            "destination_url": target_url,
            "link_title": destination_description or destination_title,
            "destination_title": destination_title,
            "destination_meta_description": destination_description,
            "priority": rec.get("priority", ""),
            "expected_benefit_score": rec.get("expected_benefit_score", ""),
            "fit": rec.get("fit", ""),
            "lift": rec.get("lift", ""),
            "anchor_confidence": rec.get("anchor_confidence", ""),
        })
    _write_csv(path, rows)
    return rows


def write_technical_seo_exports(output_dir: Path, technical_seo: dict, domain: Optional[str] = None) -> None:
    pages = (technical_seo or {}).get("pages") or []
    issues = (technical_seo or {}).get("issues") or []
    catalog = (technical_seo or {}).get("issue_catalog") or []
    page_payload = {
        "summary": (technical_seo or {}).get("summary", {}),
        "interpretation": (technical_seo or {}).get("interpretation", {}),
    }
    issue_payload = {
        "summary": (technical_seo or {}).get("summary", {}),
        "issue_counts": (technical_seo or {}).get("issue_counts", {}),
        "category_counts": (technical_seo or {}).get("category_counts", {}),
        "severity_counts": (technical_seo or {}).get("severity_counts", {}),
        "issue_catalog": catalog,
        "interpretation": (technical_seo or {}).get("interpretation", {}),
    }
    catalog_payload = {
        "summary": (technical_seo or {}).get("summary", {}),
        "issue_catalog": catalog,
    }
    _write_json_object_with_array(output_dir / "technical_pages.json", page_payload, "pages", pages)
    _write_json_object_with_array(output_dir / "technical_issues.json", issue_payload, "issues", issues)
    _write_json(output_dir / "technical_issue_catalog.json", catalog_payload)
    _write_csv_safe_iter(output_dir / "technical_pages.csv", pages)
    _write_csv_safe_iter(output_dir / "technical_issues.csv", issues)
    _write_csv_safe_iter(output_dir / "technical_issue_catalog.csv", catalog)
    write_technical_index_html(output_dir, technical_seo, domain=domain)


def write_technical_audit_bundle(
    output_dir: Path,
    *,
    domain: str,
    mode: str,
    summary: dict,
    timings: list[dict],
    technical_seo: dict,
    indexability: dict | None = None,
    sitemap_coverage: dict | None = None,
    canonical_consistency: dict | None = None,
    performance: dict | None = None,
    resource_status: dict | None = None,
    metadata_quality: dict | None = None,
    media_accessibility: dict | None = None,
    page_types: dict | None = None,
    entities: dict | None = None,
    freshness: dict | None = None,
    conversion: dict | None = None,
    structured_data: dict | None = None,
    external_links: dict | None = None,
    linkgraph: dict | None = None,
) -> None:
    """Write the non-embedding technical audit bundle.

    This is intentionally JSON/CSV-first. The existing rich HTML report assumes
    an embedding-backed ``AuditResult``; technical-only runs should not have to
    load a model just to serialize crawl/indexability/resource findings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_report_docs(output_dir)
    _write_json(
        output_dir / "run_summary.json",
        {
            "domain": domain,
            "mode": mode,
            "summary": summary,
            "stage_timings": timings,
        },
    )
    if indexability is not None:
        _write_json(output_dir / "indexability.json", indexability)
        write_indexability_issues_csv(output_dir, indexability)
    if sitemap_coverage is not None:
        write_sitemap_coverage_exports(output_dir, sitemap_coverage)
    if canonical_consistency is not None:
        write_canonical_consistency_exports(output_dir, canonical_consistency)
    if performance is not None:
        _write_json(output_dir / "performance.json", performance)
    if resource_status is not None:
        _write_json(output_dir / "resource_status.json", resource_status)
    if metadata_quality is not None:
        _write_json(output_dir / "metadata_quality.json", metadata_quality)
    if media_accessibility is not None:
        _write_json(output_dir / "media_accessibility.json", media_accessibility)
    if page_types is not None:
        _write_json(output_dir / "page_types.json", page_types)
    if entities is not None:
        _write_json(output_dir / "entities.json", entities)
    if freshness is not None:
        _write_json(output_dir / "freshness.json", freshness)
    if conversion is not None:
        _write_json(output_dir / "conversion.json", conversion)
    if structured_data is not None:
        _write_json(output_dir / "structured_data.json", structured_data)
    if external_links is not None:
        _write_json(output_dir / "external_links.json", external_links)
    if linkgraph is not None:
        _write_json(output_dir / "linkgraph.json", _slim_linkgraph_payload(linkgraph))
    write_technical_seo_exports(output_dir, technical_seo, domain=domain)


def write_indexability_issues_csv(output_dir: Path, indexability: dict) -> None:
    issues = list((indexability or {}).get("issues") or [])
    _write_csv(output_dir / "indexability_issues.csv", [_csv_safe_row(row) for row in issues])


def write_sitemap_coverage_exports(output_dir: Path, sitemap_coverage: dict) -> None:
    _write_json(output_dir / "sitemap_coverage.json", sitemap_coverage)
    _write_csv(output_dir / "sitemap_coverage.csv", [_csv_safe_row(row) for row in (sitemap_coverage or {}).get("rows", [])])
    _write_csv(output_dir / "sitemap_coverage_issues.csv", [_csv_safe_row(row) for row in (sitemap_coverage or {}).get("issues", [])])


def write_canonical_consistency_exports(output_dir: Path, canonical_consistency: dict) -> None:
    _write_json(output_dir / "canonical_consistency.json", canonical_consistency)
    _write_csv(output_dir / "canonical_consistency.csv", [_csv_safe_row(row) for row in (canonical_consistency or {}).get("rows", [])])
    _write_csv(output_dir / "canonical_consistency_issues.csv", [_csv_safe_row(row) for row in (canonical_consistency or {}).get("issues", [])])


def _csv_safe_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, list):
            out[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = value
    return out


def _paid_keywords(search_payload: Optional[dict]) -> list[dict]:
    rows = []
    for row in (search_payload or {}).get("organic_keywords") or []:
        if row.get("provider") == "google_ads" or row.get("paid_conversions") or row.get("paid_cost"):
            if row.get("keyword"):
                rows.append(row)
    rows.sort(
        key=lambda r: (
            _safe_float(r.get("paid_conversions")),
            _safe_float(r.get("paid_conversion_value")),
            _safe_float(r.get("paid_cost")),
            _safe_float(r.get("clicks")),
        ),
        reverse=True,
    )
    return rows


def _normalize_anchor(anchor: str) -> str:
    return re.sub(r"\s+", " ", str(anchor or "").strip().lower())


def _canonical_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return str(url or "").rstrip("/")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{parsed.scheme.lower() or 'https'}://{netloc}{path}".rstrip("/") or str(url or "").rstrip("/")


def _best_paid_anchor_candidate(
    paid_keywords: list[dict],
    paragraph_excerpt: str,
    target_url: str,
    target_title: str,
    target_page,
) -> dict:
    if not paid_keywords or not paragraph_excerpt:
        return {}
    paragraph = paragraph_excerpt.lower()
    target_text = " ".join([
        target_url,
        target_title,
        getattr(target_page, "title", "") if target_page else "",
        getattr(target_page, "description", "") if target_page else "",
    ]).lower()
    for row in paid_keywords:
        keyword = str(row.get("keyword") or "").strip()
        if len(keyword) < 3:
            continue
        lower = keyword.lower()
        if lower in paragraph and _keyword_plausible_for_target(lower, target_text):
            return row
    return {}


def _keyword_plausible_for_target(keyword: str, target_text: str) -> bool:
    tokens = [t for t in keyword.replace("-", " ").split() if len(t) > 2]
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in target_text)
    return matches >= max(1, min(2, len(tokens)))


def _safe_float(value) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _copy_report_docs(output_dir: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    docs = {
        root / "docs" / "serp-paragraph-gap-analysis.md": output_dir / "serp-paragraph-gap-analysis.md",
        root / "docs" / "report-sections.md": output_dir / "report-sections.md",
    }
    for source, target in docs.items():
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def write_site_metrics(path: Path, result: AuditResult, model_name: str, domain: str) -> None:
    sm = result.site_metrics
    payload = {
        "domain": domain,
        "model": model_name,
        "page_count": sm["count"],
        "site_focus_score": sm["focus_score"],
        "site_focus_score_calibrated": result.calibrated_focus_score,
        "site_radius": sm["radius"],
        "mean_distance_to_centroid": sm["mean_distance"],
        "p95_distance_to_centroid": sm["p95_distance"],
        "max_distance_to_centroid": sm["max_distance"],
        "pairwise": result.pairwise,
        "section_coherence": result.coherence,
        "topic_dimension": result.topic_dim,
        "centroid_histogram": result.centroid_hist,
        "sections": sorted(
            [
                {
                    "section": s.name,
                    "page_count": s.metrics["count"],
                    "focus_score": s.metrics["focus_score"],
                    "radius": s.metrics["radius"],
                    "p95_distance": s.metrics["p95_distance"],
                }
                for s in result.sections.values()
            ],
            key=lambda x: x["focus_score"],
            reverse=True,
        ),
        "interpretation": {
            "site_focus_score": "Raw mean cosine of pages to site centroid. Anchored to the embedding model's anisotropy; for gte-multilingual-base lives in roughly 0.5–0.9.",
            "site_focus_score_calibrated": "(focus - p10_pairwise) / (1 - p10_pairwise) — strips the model floor. 0 = no more focused than 10% of random pairs, 1 = perfectly aligned.",
            "site_radius": "Std-dev of per-page cosine distance to the site centroid. Lower is tighter.",
            "section_coherence_ratio": "Mean intra-section similarity / mean inter-section similarity. >1.5 = URL structure matches content; ~1.0 = sections are arbitrary.",
            "topic_dimension.effective_dim": "Effective number of independent topics (PCA spectral entropy). 2-4 = laser-focused, 15-30 = broad publisher.",
        },
    }
    _write_json(path, payload)


def write_section_report(path: Path, result: AuditResult) -> None:
    out = []
    for s in result.sections.values():
        section_pages = [result.pages[i] for i in s.indices]
        out.append({
            "section": s.name,
            "page_count": s.metrics["count"],
            "focus_score": s.metrics["focus_score"],
            "radius": s.metrics["radius"],
            "p95_distance_to_section_centroid": s.metrics["p95_distance"],
            "example_titles": [p.title for p in section_pages[:5]],
        })
    out.sort(key=lambda x: x["focus_score"])
    _write_json(path, out)


def write_page_drift(path: Path, result: AuditResult) -> None:
    rows = []
    for i, p in enumerate(result.pages):
        rows.append({
            "url": p.url,
            "section": p.section,
            "title": p.title,
            "word_count": p.word_count,
            "distance_to_site_centroid": round(float(result.dist_to_site[i]), 4),
            "distance_to_section_centroid": round(float(result.dist_to_section[i]), 4),
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    _write_csv(path, rows)


def build_outlier_rows(result: AuditResult) -> list[dict]:
    duplicate_set = {i for pair in result.duplicate_pairs for i in pair[:2]}
    section_size = {s.name: s.metrics["count"] for s in result.sections.values()}
    section_p95 = {s.name: s.metrics["p95_distance"] for s in result.sections.values()}

    rows = []
    for i, p in enumerate(result.pages):
        ds = float(result.dist_to_section[i])
        d_all = float(result.dist_to_site[i])
        sec_p95 = section_p95.get(p.section, 1.0)
        is_outlier = ds > sec_p95 or d_all > 0.65
        if not is_outlier and i not in duplicate_set:
            continue
        action = recommend_action(
            p,
            dist_site=d_all,
            dist_section=ds,
            section_p95=sec_p95,
            section_size=section_size.get(p.section, 0),
            has_duplicate=i in duplicate_set,
        )
        if not action:
            continue
        rows.append({
            "url": p.url,
            "section": p.section,
            "title": p.title,
            "word_count": p.word_count,
            "distance_to_site_centroid": round(d_all, 4),
            "distance_to_section_centroid": round(ds, 4),
            "section_p95_distance": round(sec_p95, 4),
            "recommendation": action,
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    return rows


def write_outliers(path: Path, result: AuditResult) -> list[dict]:
    rows = build_outlier_rows(result)
    _write_csv(path, rows)
    return rows


def build_duplicate_rows(result: AuditResult) -> list[dict]:
    rows = []
    for i, j, sim in result.duplicate_pairs:
        a = result.pages[i]
        b = result.pages[j]
        same_section = a.section == b.section
        if sim >= 0.97:
            action = "merge (duplicate)"
        elif sim >= 0.94:
            action = "consolidate or canonicalize"
        else:
            action = "review — strong overlap"
        rows.append({
            "similarity": round(sim, 4),
            "same_section": same_section,
            "section_a": a.section,
            "url_a": a.url,
            "title_a": a.title,
            "section_b": b.section,
            "url_b": b.url,
            "title_b": b.title,
            "recommendation": action,
        })
    return rows


def write_duplicates(path: Path, result: AuditResult) -> list[dict]:
    rows = build_duplicate_rows(result)
    _write_csv(path, rows)
    return rows


def write_scatterplot(
    path: Path,
    result: AuditResult,
    coords: Optional[np.ndarray],
    cluster_labels: Optional[np.ndarray],
) -> None:
    if coords is None or cluster_labels is None:
        return
    pages_payload = []
    sm = result.site_metrics
    max_drift = float(max(
        sm["max_distance"],
        float(np.max(result.dist_to_site)) if len(result.dist_to_site) else 0,
        float(np.max(result.dist_to_section)) if len(result.dist_to_section) else 0,
        1e-9,
    ))
    for i, p in enumerate(result.pages):
        pages_payload.append({
            "title": p.title,
            "url": p.url,
            "section": p.section,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": int(cluster_labels[i]),
            "drift_site": round(float(result.dist_to_site[i]), 4),
            "drift_section": round(float(result.dist_to_section[i]), 4),
            "drift_norm": round(float(result.dist_to_section[i]) / max_drift, 4),
            "word_count": p.word_count,
            "duplicate_of": result.duplicate_partners.get(i, []),
        })
    num_clusters = int(max(cluster_labels) + 1) if len(cluster_labels) else 0
    payload = {
        "total_pages": len(result.pages),
        "num_clusters": num_clusters,
        "max_drift": max_drift,
        "site_focus_score": sm["focus_score"],
        "site_radius": sm["radius"],
        "pages": pages_payload,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))


def write_pages(path: Path, result: AuditResult) -> None:
    payload = [
        {
            "url": p.url,
            "title": p.title,
            "description": p.description,
            "section": p.section,
            "word_count": p.word_count,
            "language": p.language,
        }
        for p in result.pages
    ]
    _write_json(path, payload)


def write_clusters(path: Path, summaries) -> None:
    if summaries is None:
        return
    payload = []
    for s in summaries:
        payload.append({
            "cluster_id": s.cluster_id,
            "page_count": s.page_count,
            "cohesion": round(s.cohesion, 4),
            "site_alignment": round(s.site_alignment, 4),
            "label": ", ".join(k["keyword"] for k in s.keywords[:4]),
            "keywords": s.keywords,
            "top_pages": s.top_pages,
        })
    _write_json(path, payload)


def write_all(
    output_dir: Path,
    result: AuditResult,
    model_name: str,
    domain: str,
    coords: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_summaries=None,
    coverage: Optional[list] = None,
    answerability: Optional[list] = None,
    linkgraph: Optional[dict] = None,
    external_links: Optional[dict] = None,
    paragraph_link_recs: Optional[list] = None,
    cluster_overlap: Optional[dict] = None,
    paragraph_clusters: Optional[list] = None,
    paragraph_cluster_overlap: Optional[dict] = None,
    paragraph_scatter: Optional[dict] = None,
    paragraph_fanout: Optional[list] = None,
    paragraph_impact: Optional[dict] = None,
    semantic_ablation: Optional[dict] = None,
    keyword_attribution: Optional[dict] = None,
    answer_blocks: Optional[dict] = None,
    freshness_impact: Optional[dict] = None,
    striking_distance: Optional[dict] = None,
    ctr_anomalies: Optional[dict] = None,
    ai_access: Optional[dict] = None,
    ai_citations: Optional[dict] = None,
    crux: Optional[dict] = None,
    chunk_retrievability: Optional[dict] = None,
    cannibalization: Optional[dict] = None,
    duplicate_fragments: Optional[dict] = None,
    template_patterns: Optional[dict] = None,
    winning_paragraphs: Optional[dict] = None,
    weak_paragraphs: Optional[dict] = None,
    heading_impact: Optional[dict] = None,
    entity_coverage: Optional[dict] = None,
    information_gain: Optional[dict] = None,
    title_mismatch: Optional[list] = None,
    wrong_home: Optional[list] = None,
    page_improvement: Optional[list] = None,
    competitive: Optional[dict | list] = None,
    recommendations: Optional[dict] = None,
    paragraph_density: Optional[dict] = None,
    header_analysis: Optional[dict] = None,
    header_scatter: Optional[dict] = None,
    linkbuilding: Optional[dict] = None,
    structured_data: Optional[dict] = None,
    trust_signals: Optional[dict] = None,
    conversion_balance: Optional[dict] = None,
    metadata_quality: Optional[dict] = None,
    media_accessibility: Optional[dict] = None,
    resource_status: Optional[dict] = None,
    page_types: Optional[dict] = None,
    entities: Optional[dict] = None,
    freshness: Optional[dict] = None,
    conversion: Optional[dict] = None,
    indexability: Optional[dict] = None,
    sitemap_coverage: Optional[dict] = None,
    canonical_consistency: Optional[dict] = None,
    performance: Optional[dict] = None,
    ahrefs: Optional[dict] = None,
    best_pages: Optional[dict] = None,
    performance_explainer: Optional[dict] = None,
    history_snapshot: Optional[dict] = None,
    recommendation_outcomes: Optional[dict] = None,
    technical_seo: Optional[dict] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("  report export: core metrics and CSVs")
    _copy_report_docs(output_dir)
    write_site_metrics(output_dir / "site_metrics.json", result, model_name, domain)
    write_section_report(output_dir / "section_report.json", result)
    write_page_drift(output_dir / "page_drift.csv", result)
    outliers = write_outliers(output_dir / "outliers.csv", result)
    duplicates = write_duplicates(output_dir / "duplicates.csv", result)
    write_pages(output_dir / "pages.json", result)
    write_scatterplot(output_dir / "scatterplot.json", result, coords, cluster_labels)
    LOG.info("  report export: semantic and link payloads")
    if cluster_summaries:
        write_clusters(output_dir / "clusters.json", cluster_summaries)
    if coverage is not None:
        _write_json(output_dir / "keyword_coverage.json", coverage)
    if answerability is not None:
        _write_json(output_dir / "answerability.json", answerability)
    if linkgraph is not None:
        _write_json(output_dir / "linkgraph.json", _slim_linkgraph_payload(linkgraph))
    if external_links is not None:
        _write_json(output_dir / "external_links.json", external_links)
    if paragraph_link_recs is not None:
        _write_json(output_dir / "paragraph_link_recommendations.json", paragraph_link_recs)
        write_internal_linkbuilding_csv(output_dir / "internal_linkbuilding_recommendations.csv", result, paragraph_link_recs, ahrefs)
    if cluster_overlap is not None:
        _write_json(output_dir / "cluster_overlap.json", cluster_overlap)
    if paragraph_clusters is not None:
        _write_json(output_dir / "paragraph_clusters.json", paragraph_clusters)
    if paragraph_cluster_overlap is not None:
        _write_json(output_dir / "paragraph_cluster_overlap.json", paragraph_cluster_overlap)
    if paragraph_scatter is not None:
        _write_json(output_dir / "paragraph_scatter.json", paragraph_scatter)
    if paragraph_fanout is not None:
        _write_json(output_dir / "paragraph_fanout.json", paragraph_fanout)
    if paragraph_impact is not None:
        _write_json(output_dir / "paragraph_impact.json", paragraph_impact)
    if semantic_ablation is not None:
        _write_json(output_dir / "semantic_ablation.json", semantic_ablation)
    if keyword_attribution is not None:
        _write_json(output_dir / "keyword_attribution.json", keyword_attribution)
    if answer_blocks is not None:
        _write_json(output_dir / "answer_blocks.json", answer_blocks)
    if freshness_impact is not None:
        _write_json(output_dir / "freshness_impact.json", freshness_impact)
    if striking_distance is not None:
        _write_json(output_dir / "striking_distance.json", striking_distance)
    if ctr_anomalies is not None:
        _write_json(output_dir / "ctr_anomalies.json", ctr_anomalies)
    if ai_access is not None:
        _write_json(output_dir / "ai_access.json", ai_access)
    if ai_citations is not None:
        _write_json(output_dir / "ai_citations.json", ai_citations)
    if crux is not None:
        _write_json(output_dir / "crux.json", crux)
    if chunk_retrievability is not None:
        _write_json(output_dir / "chunk_retrievability.json", chunk_retrievability)
    if cannibalization is not None:
        _write_json(output_dir / "cannibalization.json", cannibalization)
    if duplicate_fragments is not None:
        _write_json(output_dir / "duplicate_fragments.json", duplicate_fragments)
    if template_patterns is not None:
        _write_json(output_dir / "template_patterns.json", template_patterns)
    if winning_paragraphs is not None:
        _write_json(output_dir / "winning_paragraphs.json", winning_paragraphs)
    if weak_paragraphs is not None:
        _write_json(output_dir / "weak_paragraphs.json", weak_paragraphs)
    if heading_impact is not None:
        _write_json(output_dir / "heading_impact.json", heading_impact)
    if entity_coverage is not None:
        _write_json(output_dir / "entity_coverage.json", entity_coverage)
    if information_gain is not None:
        _write_json(output_dir / "information_gain.json", information_gain)
    if title_mismatch is not None:
        _write_json(output_dir / "title_mismatch.json", title_mismatch)
    if wrong_home is not None:
        _write_json(output_dir / "wrong_home_paragraphs.json", wrong_home)
    if page_improvement is not None:
        _write_json(output_dir / "page_improvement.json", page_improvement)
    if competitive is not None:
        _write_json(output_dir / "competitive_analysis.json", competitive)
    if recommendations is not None:
        _write_json(output_dir / "recommendations.json", recommendations)
    if paragraph_density is not None:
        _write_json(output_dir / "paragraph_density.json", paragraph_density)
    if header_analysis is not None:
        _write_json(output_dir / "header_analysis.json", header_analysis)
    if header_scatter is not None:
        _write_json(output_dir / "header_scatter.json", header_scatter)
    if linkbuilding is not None:
        _write_json(output_dir / "linkbuilding.json", linkbuilding)
    LOG.info("  report export: technical/content payloads")
    if structured_data is not None:
        _write_json(output_dir / "structured_data.json", structured_data)
    if trust_signals is not None:
        _write_json(output_dir / "trust_signals.json", trust_signals)
    if conversion_balance is not None:
        _write_json(output_dir / "conversion_balance.json", conversion_balance)
    if metadata_quality is not None:
        _write_json(output_dir / "metadata_quality.json", metadata_quality)
    if media_accessibility is not None:
        _write_json(output_dir / "media_accessibility.json", media_accessibility)
    if resource_status is not None:
        _write_json(output_dir / "resource_status.json", resource_status)
    if page_types is not None:
        _write_json(output_dir / "page_types.json", page_types)
    if entities is not None:
        _write_json(output_dir / "entities.json", entities)
    if freshness is not None:
        _write_json(output_dir / "freshness.json", freshness)
    if conversion is not None:
        _write_json(output_dir / "conversion.json", conversion)
    if indexability is not None:
        _write_json(output_dir / "indexability.json", indexability)
        write_indexability_issues_csv(output_dir, indexability)
    if sitemap_coverage is not None:
        write_sitemap_coverage_exports(output_dir, sitemap_coverage)
    if canonical_consistency is not None:
        write_canonical_consistency_exports(output_dir, canonical_consistency)
    if performance is not None:
        _write_json(output_dir / "performance.json", performance)
    if ahrefs is not None:
        _write_json(output_dir / "search.json", ahrefs)
        _write_json(output_dir / "ahrefs.json", ahrefs)
        provider = str((ahrefs.get("meta", {}) or {}).get("provider", "")).lower()
        if provider in {"gsc", "dataforseo"}:
            _write_json(output_dir / f"{provider}.json", ahrefs)
    if best_pages is not None:
        _write_json(output_dir / "best_pages.json", best_pages)
    if performance_explainer is not None:
        _write_json(output_dir / "performance_explainer.json", performance_explainer)
    if history_snapshot is not None:
        _write_json(output_dir / "history_snapshot.json", history_snapshot)
    if recommendation_outcomes is not None:
        _write_json(output_dir / "recommendation_outcomes.json", recommendation_outcomes)
    if technical_seo is not None:
        LOG.info("  report export: technical SEO exports")
        write_technical_seo_exports(output_dir, technical_seo, domain=domain)
    LOG.info("  report export: done")
    return {"outliers": len(outliers), "duplicates": len(result.duplicate_pairs)}
