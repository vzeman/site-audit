from site_audit.extraction_cache import ExtractionCache
from site_audit.extractor import extract


HTML = """
<html>
  <head>
    <title>Cached page</title>
    <meta name="description" content="Description">
    <link rel="canonical" href="https://example.com/page/">
  </head>
  <body><h1>Cached page</h1><p>Useful content for the cached page.</p></body>
</html>
"""


def test_extraction_cache_round_trips_extracted_page(tmp_path) -> None:
    cache = ExtractionCache(tmp_path / "extracted_pages")
    page = extract("https://example.com/page/", HTML)
    assert page is not None

    cache.put("https://example.com/page/", HTML, page, max_chars=4000)
    cached = cache.get("https://example.com/page/", HTML, max_chars=4000)

    assert cached is not None
    assert cached.url == page.url
    assert cached.title == "Cached page"
    assert cached.canonical_url == "https://example.com/page/"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["writes"] == 1


def test_extraction_cache_key_changes_with_body_and_options(tmp_path) -> None:
    cache = ExtractionCache(tmp_path / "extracted_pages")
    page = extract("https://example.com/page/", HTML)
    assert page is not None
    cache.put("https://example.com/page/", HTML, page, max_chars=4000)

    assert cache.get("https://example.com/page/", HTML.replace("Cached", "Changed"), max_chars=4000) is None
    assert cache.get("https://example.com/page/", HTML, max_chars=2000) is None
    assert cache.stats()["misses"] == 2

