from site_audit.cache import HttpCache


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

    result = cache.clean_tracking_duplicates(min_candidates=1)

    assert result == {"candidates": 1, "deleted": 1}
    assert cache.get(canonical) is not None
    assert cache.get(variant) is None


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
