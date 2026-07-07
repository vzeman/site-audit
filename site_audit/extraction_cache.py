"""Persistent cache for extracted page artifacts.

HTML extraction is CPU-heavy on large crawls. This cache stores successful
``ExtractedPage`` payloads under a key derived from the response body, source
URL, and extraction options that can affect the output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .extractor import ExtractedPage

EXTRACTION_CACHE_VERSION = "v1"


def _body_bytes(body: bytes | str) -> bytes:
    if isinstance(body, bytes):
        return body
    return (body or "").encode("utf-8")


def _cache_key(url: str, body: bytes | str, *, max_chars: int, x_robots_tag: str) -> str:
    body_hash = hashlib.sha256(_body_bytes(body)).hexdigest()
    payload = "\0".join([
        EXTRACTION_CACHE_VERSION,
        url or "",
        body_hash,
        str(max_chars),
        x_robots_tag or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExtractionCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, url: str, body: bytes | str, *, max_chars: int, x_robots_tag: str = "") -> ExtractedPage | None:
        key = _cache_key(url, body, max_chars=max_chars, x_robots_tag=x_robots_tag)
        path = self._path_for_key(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            page = ExtractedPage(**(payload.get("page") or {}))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return page

    def put(
        self,
        url: str,
        body: bytes | str,
        page: ExtractedPage,
        *,
        max_chars: int,
        x_robots_tag: str = "",
    ) -> None:
        key = _cache_key(url, body, max_chars=max_chars, x_robots_tag=x_robots_tag)
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": EXTRACTION_CACHE_VERSION,
                    "key": key,
                    "page": asdict(page),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        self.writes += 1

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}

