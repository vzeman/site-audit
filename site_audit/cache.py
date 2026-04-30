"""Two persistent caches keyed by domain.

* ``HttpCache`` — SQLite-backed store of raw HTTP responses. Avoids
  re-downloading pages on subsequent runs. We key by URL and keep an
  ETag / Last-Modified pair so that a re-run can issue a conditional
  request and skip the body when the server returns 304.
* ``EmbeddingCache`` — NPZ archive mirroring the layout of the Hugo
  ``embedding_cache.py``. Keyed by ``(url, content_hash, model_name)``
  so a re-embed is a no-op as long as the page text hasn't changed.

The two caches live side-by-side under ``cache_dir/<domain>/`` so that
wiping one site's cache is a single ``rm -rf``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


def domain_slug(domain: str) -> str:
    return domain.replace("://", "_").replace("/", "_").replace(":", "_").strip("_")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- HTTP cache --------------------------------------------------------------

_HTTP_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    url TEXT PRIMARY KEY,
    status INTEGER NOT NULL,
    headers TEXT NOT NULL,
    body BLOB NOT NULL,
    fetched_at REAL NOT NULL,
    etag TEXT,
    last_modified TEXT,
    content_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_responses_fetched ON responses(fetched_at);
"""


@dataclass
class CachedResponse:
    url: str
    status: int
    headers: dict
    body: bytes
    fetched_at: float
    etag: Optional[str]
    last_modified: Optional[str]
    content_type: Optional[str]

    @property
    def text(self) -> str:
        # naive but works for HTML — most sites declare charset in headers
        encoding = "utf-8"
        ctype = (self.content_type or "").lower()
        if "charset=" in ctype:
            encoding = ctype.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        try:
            return self.body.decode(encoding, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


class HttpCache:
    """SQLite-backed HTTP response cache. One DB per domain."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_HTTP_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, url: str) -> Optional[CachedResponse]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT url, status, headers, body, fetched_at, etag, last_modified, content_type "
                "FROM responses WHERE url = ?",
                (url,),
            ).fetchone()
        if not row:
            return None
        return CachedResponse(
            url=row[0],
            status=row[1],
            headers=json.loads(row[2]),
            body=row[3],
            fetched_at=row[4],
            etag=row[5],
            last_modified=row[6],
            content_type=row[7],
        )

    def put(
        self,
        url: str,
        status: int,
        headers: dict,
        body: bytes,
    ) -> None:
        etag = headers.get("ETag") or headers.get("etag")
        last_modified = headers.get("Last-Modified") or headers.get("last-modified")
        content_type = headers.get("Content-Type") or headers.get("content-type")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(url, status, headers, body, fetched_at, etag, last_modified, content_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    int(status),
                    json.dumps(dict(headers)),
                    body,
                    time.time(),
                    etag,
                    last_modified,
                    content_type,
                ),
            )

    def known_urls(self) -> Iterable[str]:
        with self._connect() as conn:
            for (url,) in conn.execute("SELECT url FROM responses"):
                yield url

    def stats(self) -> dict:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"entries": count, "size_bytes": size}


# --- Embedding cache ---------------------------------------------------------


class EmbeddingCache:
    """NPZ-backed embedding cache. One file per (domain, model)."""

    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[str, np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        if not self.npz_path.exists():
            return
        try:
            data = np.load(self.npz_path, allow_pickle=False)
            urls = data["urls"]
            hashes = data["hashes"]
            embs = data["embeddings"]
            self._cache = {
                str(urls[i]): (str(hashes[i]), embs[i]) for i in range(len(urls))
            }
        except Exception as exc:  # corrupted archive, start fresh
            print(f"  embedding cache read failed ({exc}); starting clean")
            self._cache = {}

    def get(self, url: str, hash_: str) -> Optional[np.ndarray]:
        entry = self._cache.get(url)
        if entry and entry[0] == hash_:
            return entry[1]
        return None

    def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
        self._cache[url] = (hash_, embedding.astype(np.float32))

    def save(self) -> None:
        if not self._cache:
            return
        urls = np.array(list(self._cache.keys()))
        hashes = np.array([v[0] for v in self._cache.values()])
        embs = np.stack([v[1] for v in self._cache.values()]).astype(np.float32)
        np.savez_compressed(self.npz_path, urls=urls, hashes=hashes, embeddings=embs)

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "size_bytes": self.npz_path.stat().st_size if self.npz_path.exists() else 0,
        }
