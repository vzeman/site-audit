"""Tiny static server that serves the UI plus the report directory.

Run ``site-audit serve <domain>`` and we expose:

  /              → ``ui/index.html``
  /data/...      → files under ``output/<domain_slug>/``

The server is ``http.server`` only — no Flask, no Django, just enough
glue so the D3 viewer can fetch JSON/CSV from the same origin.
"""

from __future__ import annotations

import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .cache import domain_slug

LOG = logging.getLogger(__name__)


def _build_handler(ui_dir: Path, data_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        # quieter logs
        def log_message(self, format, *args):
            LOG.debug("%s - %s", self.address_string(), format % args)

        def do_GET(self):  # noqa: N802 (stdlib API)
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                self._serve_file(ui_dir / "index.html")
                return
            if path.startswith("/data/"):
                rel = path[len("/data/"):]
                self._serve_file(data_dir / rel)
                return
            target = ui_dir / path.lstrip("/")
            if target.is_file():
                self._serve_file(target)
                return
            self.send_error(404, "Not found")

        def _serve_file(self, file_path: Path):
            try:
                file_path = file_path.resolve()
            except Exception:
                self.send_error(404, "Not found")
                return
            if not file_path.is_file():
                self.send_error(404, "Not found")
                return
            ctype, _ = mimetypes.guess_type(str(file_path))
            ctype = ctype or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(domain: str, output_root: Path, ui_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    slug = domain_slug(domain.replace("https://", "").replace("http://", "").split("/")[0])
    data_dir = Path(output_root) / slug
    if not data_dir.exists():
        raise FileNotFoundError(f"No report at {data_dir}. Run `site-audit run {domain}` first.")
    ui_dir = Path(ui_dir)
    if not (ui_dir / "index.html").is_file():
        raise FileNotFoundError(f"UI assets missing: {ui_dir}/index.html")

    handler = _build_handler(ui_dir, data_dir)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"\n  ➜  Site-audit viewer ready: http://{host}:{port}/")
    print(f"     serving report: {data_dir}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
        httpd.server_close()
