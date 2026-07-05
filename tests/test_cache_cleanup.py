from site_audit.cache import HttpCache
import sqlite3


def _html(canonical: str) -> bytes:
    return f"""
        <html>
          <head><link rel="canonical" href="{canonical}"></head>
          <body>Page</body>
        </html>
    """.encode("utf-8")


def test_clean_tracking_duplicates_removes_html_canonical_variants(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    canonical = "https://example.com/page/"
    variant = "https://example.com/page/?utm_source=newsletter"
    cache.put(canonical, 200, {"Content-Type": "text/html"}, _html(canonical))
    cache.put(variant, 200, {"Content-Type": "text/html"}, _html(canonical))
    variant_body_path = tmp_path / "http_bodies" / cache._body_relative_path(variant)
    assert variant_body_path.is_file()

    result = cache.clean_tracking_duplicates(min_candidates=1)

    assert result == {"candidates": 1, "deleted": 1}
    assert cache.get(canonical) is not None
    assert cache.get(variant) is None
    assert not variant_body_path.exists()


def test_clean_tracking_duplicates_keeps_self_canonical_tracking_url(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    url = "https://example.com/page/?source=feed"
    cache.put(url, 200, {"Content-Type": "text/html"}, _html(url))

    result = cache.clean_tracking_duplicates(min_candidates=1)

    assert result == {"candidates": 1, "deleted": 0}
    assert cache.get(url) is not None


def test_clean_tracking_duplicates_respects_min_candidate_threshold(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    url = "https://example.com/page/?fbclid=abc"
    cache.put(url, 200, {"Content-Type": "text/html"}, _html("https://example.com/page/"))

    result = cache.clean_tracking_duplicates(min_candidates=2)

    assert result == {"candidates": 1, "deleted": 0}
    assert cache.get(url) is not None


def test_http_cache_stores_new_bodies_outside_sqlite(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    url = "https://example.com/page/"
    body = _html(url)

    cache.put(url, 200, {"Content-Type": "text/html"}, body)

    cached = cache.get(url)
    body_path = tmp_path / "http_bodies" / cache._body_relative_path(url)
    with sqlite3.connect(tmp_path / "http.sqlite") as conn:
        row = conn.execute(
            "SELECT body, body_path, body_sha256, body_size_bytes FROM responses WHERE url = ?",
            (url,),
        ).fetchone()

    assert cached is not None
    assert cached.body == body
    assert row[0] == b""
    assert row[1] == cache._body_relative_path(url)
    assert row[2]
    assert row[3] == len(body)
    assert body_path.read_bytes() == body
    assert cache.stats()["body_size_bytes"] == len(body)


def test_http_cache_reads_legacy_inline_bodies(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    url = "https://example.com/legacy/"
    body = _html(url)
    with sqlite3.connect(tmp_path / "http.sqlite") as conn:
        conn.execute(
            "INSERT INTO responses "
            "(url, status, headers, body, fetched_at, content_type, canonical_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, 200, '{"Content-Type": "text/html"}', body, 1.0, "text/html", url),
        )

    cached = cache.get(url)

    assert cached is not None
    assert cached.body == body


def test_http_cache_get_metadata_does_not_load_body(tmp_path) -> None:
    cache = HttpCache(tmp_path / "http.sqlite")
    url = "https://example.com/asset.js"
    body = b"console.log('large asset');"
    cache.put(url, 200, {"Content-Type": "application/javascript"}, body)

    meta = cache.get_metadata(url)

    assert meta is not None
    assert meta["status"] == 200
    assert meta["body_size_bytes"] == len(body)
    assert "body" not in meta
