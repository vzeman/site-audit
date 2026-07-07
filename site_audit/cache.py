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
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import numpy as np
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)


def domain_slug(domain: str) -> str:
    return domain.replace("://", "_").replace("/", "_").replace(":", "_").strip("_")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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
    content_type TEXT,
    canonical_url TEXT,
    body_path TEXT,
    body_sha256 TEXT,
    body_size_bytes INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_responses_fetched ON responses(fetched_at);
"""

# Migration: add canonical_url to databases created before this column existed.
_HTTP_MIGRATIONS = (
    "ALTER TABLE responses ADD COLUMN canonical_url TEXT",
    "ALTER TABLE responses ADD COLUMN body_path TEXT",
    "ALTER TABLE responses ADD COLUMN body_sha256 TEXT",
    "ALTER TABLE responses ADD COLUMN body_size_bytes INTEGER DEFAULT 0",
)

_TRACKING_PARAM_RE = re.compile(r"(?:^|[?&])(?:utm_[^=&]+|source|fbclid)=", re.IGNORECASE)
_TRACKING_SQL_WHERE = (
    "url GLOB '*[?&]utm_*' "
    "OR url GLOB '*[?&]source=*' "
    "OR url GLOB '*[?&]fbclid=*'"
)

_TRACKING_PARAM_RE = re.compile(r"(?:^|[?&])(?:utm_[^=&]+|source|fbclid)=", re.IGNORECASE)
_TRACKING_SQL_WHERE = (
    "url GLOB '*[?&]utm_*' "
    "OR url GLOB '*[?&]source=*' "
    "OR url GLOB '*[?&]fbclid=*'"
)


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
    canonical_url: Optional[str] = None

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
        self.body_dir = self.db_path.parent / "http_bodies"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.body_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._init_schema(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(_HTTP_SCHEMA)
        for migration in _HTTP_MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass

    def get(self, url: str) -> Optional[CachedResponse]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT url, status, headers, body, fetched_at, etag, last_modified, "
                "content_type, canonical_url, body_path "
                "FROM responses WHERE url = ?",
                (url,),
            ).fetchone()
        if not row:
            return None
        return CachedResponse(
            url=row[0],
            status=row[1],
            headers=json.loads(row[2]),
            body=self._read_body(row[3], row[9]),
            fetched_at=row[4],
            etag=row[5],
            last_modified=row[6],
            content_type=row[7],
            canonical_url=row[8],
        )

    def get_metadata(self, url: str) -> Optional[dict]:
        """Return cached response metadata without loading the response body."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, headers, fetched_at, etag, last_modified, content_type, "
                "canonical_url, body_size_bytes "
                "FROM responses WHERE url = ?",
                (url,),
            ).fetchone()
        if not row:
            return None
        headers = json.loads(row[1])
        size = _safe_int(row[7])
        if not size:
            size = _safe_int(headers.get("Content-Length") or headers.get("content-length"))
        return {
            "status": int(row[0]),
            "headers": headers,
            "fetched_at": row[2],
            "etag": row[3],
            "last_modified": row[4],
            "content_type": row[5],
            "canonical_url": row[6],
            "body_size_bytes": size,
        }

    def put(
        self,
        url: str,
        status: int,
        headers: dict,
        body: bytes,
        canonical_url: Optional[str] = None,
    ) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        etag = headers.get("ETag") or headers.get("etag")
        last_modified = headers.get("Last-Modified") or headers.get("last-modified")
        content_type = headers.get("Content-Type") or headers.get("content-type")
        body_path = self._write_body(url, body)
        body_hash = hashlib.sha256(body).hexdigest()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(url, status, headers, body, fetched_at, etag, last_modified, "
                "content_type, canonical_url, body_path, body_sha256, body_size_bytes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    int(status),
                    json.dumps(dict(headers)),
                    b"",
                    time.time(),
                    etag,
                    last_modified,
                    content_type,
                    canonical_url,
                    body_path,
                    body_hash,
                    len(body),
                ),
            )

    def _body_relative_path(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return f"{digest[:2]}/{digest}.body"

    def body_file_path(self, url: str) -> Path:
        return self.body_dir / self._body_relative_path(url)

    def _write_body(self, url: str, body: bytes) -> str:
        rel = self._body_relative_path(url)
        path = self.body_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(body)
        tmp_path.replace(path)
        return rel

    def _read_body(self, inline_body: bytes | None, body_path: str | None) -> bytes:
        if body_path:
            path = self.body_dir / body_path
            try:
                return path.read_bytes()
            except OSError:
                LOG.warning("  cache body missing for %s", path)
        return inline_body or b""

    def _delete_body(self, url: str) -> None:
        path = self.body_dir / self._body_relative_path(url)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOG.debug("could not delete cached body %s: %s", path, exc)

    def known_urls(self) -> Iterable[str]:
        with self._connect() as conn:
            for (url,) in conn.execute("SELECT url FROM responses"):
                yield url

    def stats(self) -> dict:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            body_size = conn.execute(
                "SELECT COALESCE(SUM(body_size_bytes), 0) FROM responses"
            ).fetchone()[0]
            sqlite_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "entries": count,
            "size_bytes": sqlite_size + int(body_size or 0),
            "sqlite_size_bytes": sqlite_size,
            "body_size_bytes": int(body_size or 0),
        }

    def migrate_bodies_to_files(
        self,
        *,
        batch_size: int = 500,
        keep_backup: bool = True,
        progress_callback=None,
    ) -> dict:
        """Move inline SQLite bodies to files and rebuild a compact metadata DB.

        Existing file-backed rows are preserved. Legacy inline-body rows are
        streamed out to ``http_bodies`` and inserted into a fresh SQLite file
        with an empty ``body`` column. After a successful rebuild the fresh DB
        atomically replaces the original one; optionally keep the old DB as a
        ``.bak`` file for recovery.
        """
        tmp_db = self.db_path.with_suffix(self.db_path.suffix + ".rebuild.tmp")
        backup_db = self.db_path.with_suffix(self.db_path.suffix + ".legacy-bodies.bak")
        tmp_db.unlink(missing_ok=True)
        backup_db.unlink(missing_ok=True)

        moved = 0
        preserved = 0
        rows_total = 0
        body_bytes = 0
        started = time.time()
        with sqlite3.connect(str(self.db_path)) as src, sqlite3.connect(str(tmp_db)) as dst:
            self._init_schema(dst)
            rows_total = int(src.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
            rows = src.execute(
                "SELECT url, status, headers, body, fetched_at, etag, last_modified, "
                "content_type, canonical_url, body_path, body_sha256, body_size_bytes "
                "FROM responses"
            )
            pending = 0
            for row in rows:
                (
                    url,
                    status,
                    headers,
                    body,
                    fetched_at,
                    etag,
                    last_modified,
                    content_type,
                    canonical_url,
                    body_path,
                    body_sha256,
                    body_size_bytes,
                ) = row
                inline_body = body or b""
                if inline_body and not body_path:
                    body_path = self._write_body(url, inline_body)
                    body_sha256 = hashlib.sha256(inline_body).hexdigest()
                    body_size_bytes = len(inline_body)
                    moved += 1
                    body_bytes += len(inline_body)
                else:
                    preserved += 1
                    body_size_bytes = _safe_int(body_size_bytes)
                    if not body_size_bytes and body_path:
                        path = self.body_dir / body_path
                        try:
                            body_size_bytes = path.stat().st_size
                        except OSError:
                            body_size_bytes = 0
                    if not body_sha256 and body_path:
                        path = self.body_dir / body_path
                        try:
                            body_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                        except OSError:
                            body_sha256 = ""

                dst.execute(
                    "INSERT OR REPLACE INTO responses "
                    "(url, status, headers, body, fetched_at, etag, last_modified, "
                    "content_type, canonical_url, body_path, body_sha256, body_size_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        url,
                        int(status),
                        headers,
                        b"",
                        fetched_at,
                        etag,
                        last_modified,
                        content_type,
                        canonical_url,
                        body_path,
                        body_sha256,
                        _safe_int(body_size_bytes),
                    ),
                )
                pending += 1
                if pending >= batch_size:
                    dst.commit()
                    pending = 0
                    if progress_callback:
                        progress_callback({"processed": moved + preserved, "total": rows_total, "moved": moved})
            dst.commit()

        if keep_backup:
            self.db_path.replace(backup_db)
        else:
            self.db_path.unlink(missing_ok=True)
        tmp_db.replace(self.db_path)

        return {
            "rows": rows_total,
            "moved": moved,
            "preserved": preserved,
            "body_bytes_moved": body_bytes,
            "sqlite_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "backup_path": str(backup_db) if keep_backup else "",
            "seconds": round(time.time() - started, 3),
        }

    def clean_tracking_duplicates(self, *, min_candidates: int = 100) -> dict:
        """Delete cached tracking-param URL variants that point at another canonical.

        The crawl cache is keyed by request URL, so large sites can accumulate many
        copies of the same page at URLs like ``?utm_*`` or ``?source=...``. We only
        delete rows whose request URL contains a known tracking parameter and whose
        stored redirect target or HTML canonical points somewhere else.
        """
        with self._connect() as conn:
            candidate_count = conn.execute(
                f"SELECT COUNT(*) FROM responses WHERE {_TRACKING_SQL_WHERE}"
            ).fetchone()[0]
            if candidate_count < min_candidates:
                return {"candidates": candidate_count, "deleted": 0}

            delete_urls: list[str] = []
            rows = conn.execute(
                "SELECT url, canonical_url, body, content_type, body_path FROM responses "
                f"WHERE {_TRACKING_SQL_WHERE}"
            )
            for url, stored_canonical, body, content_type, body_path in rows:
                if not _has_tracking_param(url):
                    continue
                canonical = _usable_canonical(url, stored_canonical)
                if not canonical and "html" in (content_type or "").lower():
                    canonical = _extract_html_canonical(url, self._read_body(body, body_path))
                if (
                    canonical
                    and _normalize_for_cache_dedupe(canonical)
                    != _normalize_for_cache_dedupe(url)
                ):
                    delete_urls.append(url)

            if delete_urls:
                conn.executemany("DELETE FROM responses WHERE url = ?", [(url,) for url in delete_urls])
        for url in delete_urls:
            self._delete_body(url)

        if delete_urls:
            LOG.info(
                "  cache cleanup: removed %d tracking URL variants from %d candidates",
                len(delete_urls),
                candidate_count,
            )
        return {"candidates": candidate_count, "deleted": len(delete_urls)}


def _has_tracking_param(url: str) -> bool:
    parsed = urlparse(url)
    return bool(_TRACKING_PARAM_RE.search("?" + (parsed.query or "")))


def _usable_canonical(base_url: str, canonical_url: Optional[str]) -> str:
    if not canonical_url:
        return ""
    return _absolute_url(base_url, canonical_url)


def _extract_html_canonical(base_url: str, body: bytes | str | None) -> str:
    if not body:
        return ""
    if isinstance(body, bytes):
        html = body.decode("utf-8", errors="replace")
    else:
        html = body
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    tag = soup.find(
        "link",
        rel=lambda value: value and "canonical" in [
            str(v).lower() for v in (value if isinstance(value, list) else [value])
        ],
    )
    if not tag:
        return ""
    return _absolute_url(base_url, tag.get("href", ""))


def _absolute_url(base_url: str, maybe_url: str) -> str:
    maybe_url = (maybe_url or "").strip()
    if not maybe_url:
        return ""
    return urljoin(base_url, maybe_url)


def _normalize_for_cache_dedupe(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))

    def clean_tracking_duplicates(self, *, min_candidates: int = 100) -> dict:
        """Delete cached tracking-param URL variants that point at another canonical.

        The crawl cache is keyed by request URL, so large sites can accumulate many
        copies of the same page at URLs like ``?utm_*`` or ``?source=...``. We only
        delete rows whose request URL contains a known tracking parameter and whose
        stored redirect target or HTML canonical points somewhere else.
        """
        with self._connect() as conn:
            candidate_count = conn.execute(
                f"SELECT COUNT(*) FROM responses WHERE {_TRACKING_SQL_WHERE}"
            ).fetchone()[0]
            if candidate_count < min_candidates:
                return {"candidates": candidate_count, "deleted": 0}

            delete_urls: list[str] = []
            rows = conn.execute(
                "SELECT url, canonical_url, body, content_type FROM responses "
                f"WHERE {_TRACKING_SQL_WHERE}"
            )
            for url, stored_canonical, body, content_type in rows:
                if not _has_tracking_param(url):
                    continue
                canonical = _usable_canonical(url, stored_canonical)
                if not canonical and "html" in (content_type or "").lower():
                    canonical = _extract_html_canonical(url, body)
                if (
                    canonical
                    and _normalize_for_cache_dedupe(canonical)
                    != _normalize_for_cache_dedupe(url)
                ):
                    delete_urls.append(url)

            if delete_urls:
                conn.executemany("DELETE FROM responses WHERE url = ?", [(url,) for url in delete_urls])

        if delete_urls:
            LOG.info(
                "  cache cleanup: removed %d tracking URL variants from %d candidates",
                len(delete_urls),
                candidate_count,
            )
        return {"candidates": candidate_count, "deleted": len(delete_urls)}


def _has_tracking_param(url: str) -> bool:
    parsed = urlparse(url)
    return bool(_TRACKING_PARAM_RE.search("?" + (parsed.query or "")))


def _usable_canonical(base_url: str, canonical_url: Optional[str]) -> str:
    if not canonical_url:
        return ""
    return _absolute_url(base_url, canonical_url)


def _extract_html_canonical(base_url: str, body: bytes | str | None) -> str:
    if not body:
        return ""
    if isinstance(body, bytes):
        html = body.decode("utf-8", errors="replace")
    else:
        html = body
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    tag = soup.find(
        "link",
        rel=lambda value: value and "canonical" in [
            str(v).lower() for v in (value if isinstance(value, list) else [value])
        ],
    )
    if not tag:
        return ""
    return _absolute_url(base_url, tag.get("href", ""))


def _absolute_url(base_url: str, maybe_url: str) -> str:
    maybe_url = (maybe_url or "").strip()
    if not maybe_url:
        return ""
    return urljoin(base_url, maybe_url)


def _normalize_for_cache_dedupe(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


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
            emb = entry[1]
            if np.isfinite(emb).all():
                return emb
            LOG.warning("Ignoring non-finite embedding cache entry for %s", url)
        return None

    def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
        emb = embedding.astype(np.float32)
        if not np.isfinite(emb).all():
            raise ValueError(f"Refusing to cache non-finite embedding for {url}")
        self._cache[url] = (hash_, emb)

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


class ParagraphEmbeddingCache:
    """NPZ cache of paragraph embeddings keyed by ``(url, paragraph_hash)``.

    A page typically has ~15-50 paragraphs, so for a 1 000-page site we're
    storing ~30 000 vectors. Lookups are by ``(url, hash(text|model))`` so
    re-runs only re-embed paragraphs whose text changed.
    """

    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str], np.ndarray] = {}
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
                (str(urls[i]), str(hashes[i])): embs[i] for i in range(len(urls))
            }
        except Exception as exc:  # pragma: no cover
            print(f"  paragraph cache read failed ({exc}); starting clean")
            self._cache = {}

    def get(self, url: str, hash_: str) -> Optional[np.ndarray]:
        return self._cache.get((url, hash_))

    def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
        self._cache[(url, hash_)] = embedding.astype(np.float32)

    def save(self) -> None:
        if not self._cache:
            return
        items = list(self._cache.items())
        urls = np.array([k[0] for k, _ in items])
        hashes = np.array([k[1] for k, _ in items])
        embs = np.stack([v for _, v in items]).astype(np.float32)
        np.savez_compressed(self.npz_path, urls=urls, hashes=hashes, embeddings=embs)

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "size_bytes": self.npz_path.stat().st_size if self.npz_path.exists() else 0,
        }
