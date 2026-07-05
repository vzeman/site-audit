import requests

from site_audit.crawler import AdaptiveConcurrency, CrawlConfig, Crawler, FetchResult, normalize_url


class _Cache:
    def get(self, url):
        return None


class _Response:
    def __init__(
        self,
        body: str,
        url: str = "https://example.com/sitemap.xml",
        status_code: int = 200,
        content_type: str = "application/xml",
        history: list | None = None,
    ):
        self.status_code = status_code
        self.content = body.encode("utf-8")
        self.text = body
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.reason = ""
        self.encoding = "utf-8"
        self.history = history or []


def _crawler(config: CrawlConfig, responses: dict[str, str]) -> Crawler:
    crawler = Crawler(config, _Cache())

    def fake_request(url: str, method: str = "GET"):
        body = responses.get(url)
        if body is None:
            return None
        return _Response(body, url)

    crawler._request_with_retry = fake_request
    return crawler


def test_explicit_sitemap_url_limits_discovery_to_that_sitemap() -> None:
    responses = {
        "https://example.com/selected.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/en/page</loc></url>
            </urlset>
        """,
        "https://example.com/sitemap.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/sk/page</loc></url>
            </urlset>
        """,
    }
    crawler = _crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            sitemap_urls=["https://example.com/selected.xml"],
        ),
        responses,
    )

    assert crawler._discover_via_sitemaps() == ["https://example.com/en/page"]


def test_url_include_and_exclude_patterns_filter_sitemap_pages() -> None:
    responses = {
        "https://example.com/sitemap.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/en/keep</loc></url>
              <url><loc>https://example.com/en/private</loc></url>
              <url><loc>https://example.com/sk/drop</loc></url>
            </urlset>
        """,
    }
    crawler = _crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            include_patterns=[r"/en/"],
            exclude_patterns=[r"/private"],
        ),
        responses,
    )

    assert crawler._discover_via_sitemaps() == ["https://example.com/en/keep"]


def test_sitemap_include_and_exclude_patterns_filter_sitemap_index_children() -> None:
    responses = {
        "https://example.com/sitemap.xml": """
            <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <sitemap><loc>https://example.com/en.xml</loc></sitemap>
              <sitemap><loc>https://example.com/blog.xml</loc></sitemap>
              <sitemap><loc>https://example.com/sk.xml</loc></sitemap>
            </sitemapindex>
        """,
        "https://example.com/en.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/en/page</loc></url>
            </urlset>
        """,
        "https://example.com/blog.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/blog/page</loc></url>
            </urlset>
        """,
    }
    crawler = _crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            sitemap_include_patterns=[r"(sitemap|en|blog)\.xml$"],
            sitemap_exclude_patterns=[r"blog\.xml$"],
        ),
        responses,
    )

    assert crawler._discover_via_sitemaps() == ["https://example.com/en/page"]


def test_sitemap_lastmod_after_filters_urlset_entries() -> None:
    responses = {
        "https://example.com/sitemap.xml": """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url>
                <loc>https://example.com/recent</loc>
                <lastmod>2026-05-04T10:00:00+00:00</lastmod>
              </url>
              <url>
                <loc>https://example.com/old</loc>
                <lastmod>2024-12-31</lastmod>
              </url>
              <url>
                <loc>https://example.com/unknown</loc>
              </url>
            </urlset>
        """,
    }
    crawler = _crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            sitemap_lastmod_after="2025-05-04",
        ),
        responses,
    )

    assert crawler._discover_via_sitemaps() == ["https://example.com/recent"]
    assert crawler.sitemap_entries == [
        {
            "url": "https://example.com/recent",
            "source_sitemaps": ["https://example.com/sitemap.xml"],
            "lastmod": "2026-05-04T10:00:00+00:00",
        }
    ]


def test_sitemap_only_keeps_outlinks_but_does_not_enqueue_them() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            max_pages=10,
            max_workers=1,
            respect_robots=False,
            crawl_discovered_links=False,
        ),
        _Cache(),
    )
    crawler._fetch = lambda url: FetchResult(
        url=url,
        status=200,
        body='<a href="https://example.com/linked">Linked</a>',
        content_type="text/html",
        from_cache=True,
    )

    results = crawler._bfs(["https://example.com/start"])

    assert [result.url for result in results] == ["https://example.com/start"]
    assert results[0].outlinks == [("https://example.com/linked", "Linked")]


def test_bfs_drains_active_futures_when_frontier_is_empty() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            max_pages=10,
            max_workers=2,
            respect_robots=False,
        ),
        _Cache(),
    )
    crawler._fetch = lambda url: FetchResult(
        url=url,
        status=200,
        body="",
        content_type="text/html",
        from_cache=False,
    )

    results = crawler._bfs(["https://example.com/a", "https://example.com/b"])

    assert {result.url for result in results} == {
        "https://example.com/a",
        "https://example.com/b",
    }


def test_adaptive_concurrency_backs_off_and_recovers() -> None:
    adaptive = AdaptiveConcurrency(max_workers=8, min_workers=2, success_threshold=2)

    previous = adaptive.record(FetchResult(
        url="https://example.com/slow",
        status=429,
        body="",
        content_type="text/html",
        from_cache=False,
    ))

    assert previous == 8
    assert adaptive.target_workers == 4

    adaptive.record(FetchResult(
        url="https://example.com/ok-1",
        status=200,
        body="",
        content_type="text/html",
        from_cache=False,
    ))
    adaptive.record(FetchResult(
        url="https://example.com/ok-2",
        status=200,
        body="",
        content_type="text/html",
        from_cache=False,
    ))

    assert adaptive.target_workers == 5


def test_adaptive_concurrency_backs_off_on_slow_live_response() -> None:
    adaptive = AdaptiveConcurrency(max_workers=6, min_workers=1, slow_seconds=1.0, max_rss_mb=999_999)

    previous = adaptive.record(FetchResult(
        url="https://example.com/slow",
        status=200,
        body="",
        content_type="text/html",
        from_cache=False,
        elapsed_seconds=2.5,
    ))

    assert previous == 6
    assert adaptive.target_workers == 3


def test_adaptive_concurrency_backs_off_on_local_memory_pressure() -> None:
    adaptive = AdaptiveConcurrency(max_workers=6, min_workers=1, max_rss_mb=1)

    adaptive.record(FetchResult(
        url="https://example.com/cache",
        status=200,
        body="",
        content_type="text/html",
        from_cache=True,
    ))

    assert adaptive.target_workers == 3


def test_normalize_url_strips_tracking_params_but_keeps_business_query_params() -> None:
    assert normalize_url(
        "https://example.com/page?utm_source=newsletter&source=feed&page=2&sort=price#reviews"
    ) == "https://example.com/page?page=2&sort=price"
    assert normalize_url(
        "https://example.com/page?gclid=abc&fbclid=def"
    ) == "https://example.com/page"


def test_bfs_dedupes_tracking_variants_before_enqueue() -> None:
    fetched: list[str] = []
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            max_pages=10,
            max_workers=1,
            respect_robots=False,
        ),
        _Cache(),
    )

    def fake_fetch(url: str) -> FetchResult:
        fetched.append(url)
        body = ""
        if url == "https://example.com/start":
            body = """
                <a href="/target?utm_source=newsletter">One</a>
                <a href="/target?source=feed">Two</a>
                <a href="/target">Three</a>
                <a href="/target?page=2&utm_campaign=spring">Paged</a>
            """
        return FetchResult(
            url=url,
            status=200,
            body=body,
            content_type="text/html",
            from_cache=False,
        )

    crawler._fetch = fake_fetch
    results = crawler._bfs(["https://example.com/start"])

    assert fetched == [
        "https://example.com/start",
        "https://example.com/target",
        "https://example.com/target?page=2",
    ]
    assert [result.url for result in results] == fetched


def test_fetch_preserves_requested_url_and_redirect_target() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, use_cache=False),
        _Cache(),
    )
    crawler._request_with_retry = lambda url: _Response(
        "<html><title>Final</title></html>",
        url="https://example.com/final",
        status_code=200,
        content_type="text/html",
    )

    result = crawler._fetch("https://example.com/redirecting")

    assert result is not None
    assert result.url == "https://example.com/final"
    assert result.requested_url == "https://example.com/redirecting"
    assert result.redirect_target_url == "https://example.com/final"


def test_fetch_preserves_redirect_chain_from_response_history() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, use_cache=False),
        _Cache(),
    )
    crawler._request_with_retry = lambda url: _Response(
        "<html><title>Final</title></html>",
        url="https://example.com/final",
        status_code=200,
        content_type="text/html",
        history=[
            _Response("", url="https://example.com/start", status_code=301, content_type="text/html"),
            _Response("", url="https://example.com/middle", status_code=302, content_type="text/html"),
        ],
    )

    result = crawler._fetch("https://example.com/start")

    assert result is not None
    assert result.redirect_chain == [
        "https://example.com/start",
        "https://example.com/middle",
        "https://example.com/final",
    ]
    assert result.redirect_hop_count == 2
    assert result.redirect_status_codes == [301, 302]


def test_request_with_retry_classifies_too_many_redirects_as_redirect_loop() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, use_cache=False),
        _Cache(),
    )

    def raise_loop(*args, **kwargs):
        raise requests.TooManyRedirects("Exceeded 30 redirects")

    crawler._session.get = raise_loop

    response = crawler._request_with_retry("https://example.com/loop")

    assert response is not None
    assert response.status_code == 0
    assert response.reason == "redirect_loop"


def test_strip_header_footer_removes_chrome_before_link_extraction() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, strip_header_footer=True),
        _Cache(),
    )
    body = crawler._prepare_html_body("""
        <html><body>
          <header><a href="/nav">Nav</a></header>
          <main><a href="/article">Article</a></main>
          <footer><a href="/legal">Legal</a></footer>
        </body></html>
    """)

    links = crawler._extract_links("https://example.com/", body)

    assert links == [("https://example.com/article", "Article")]


def test_content_include_classes_scope_link_extraction() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            content_include_classes=["article-detail-content"],
        ),
        _Cache(),
    )
    body = crawler._prepare_html_body("""
        <html><body>
          <aside><a href="/sidebar">Sidebar</a></aside>
          <article class="article-detail-content"><a href="/article">Article</a></article>
          <section class="article-detail-content"><a href="/more">More</a></section>
        </body></html>
    """)

    links = crawler._extract_links("https://example.com/", body)

    assert links == [
        ("https://example.com/article", "Article"),
        ("https://example.com/more", "More"),
    ]


def test_content_include_classes_preserve_head_metadata() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            content_include_classes=["article-detail-content"],
        ),
        _Cache(),
    )
    body = crawler._prepare_html_body("""
        <html>
          <head>
            <title>Article title</title>
            <script type="application/ld+json">
              {"@context":"https://schema.org","@type":"NewsArticle","dateModified":"2026-03-01"}
            </script>
          </head>
          <body>
            <aside><a href="/sidebar">Sidebar</a></aside>
            <article class="article-detail-content"><p>Article body.</p></article>
          </body>
        </html>
    """)

    assert "<title>Article title</title>" in body
    assert '"dateModified":"2026-03-01"' in body
    links = crawler._extract_links("https://example.com/", body)
    assert links == []


def test_content_exclude_classes_remove_repeated_blocks() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            content_exclude_classes=["sidebar", "related-posts"],
        ),
        _Cache(),
    )
    body = crawler._prepare_html_body("""
        <html><body>
          <main><a href="/article">Article</a></main>
          <aside class="sidebar"><a href="/sidebar">Sidebar</a></aside>
          <div class="related-posts featured"><a href="/related">Related</a></div>
        </body></html>
    """)

    links = crawler._extract_links("https://example.com/", body)

    assert links == [("https://example.com/article", "Article")]


def test_content_exclude_classes_apply_inside_included_scope() -> None:
    crawler = Crawler(
        CrawlConfig(
            "example.com",
            respect_robots=False,
            content_include_classes=["article-detail-content"],
            content_exclude_classes=["promo"],
        ),
        _Cache(),
    )
    body = crawler._prepare_html_body("""
        <html><body>
          <article class="article-detail-content">
            <a href="/article">Article</a>
            <div class="promo"><a href="/promo">Promo</a></div>
          </article>
          <footer><a href="/footer">Footer</a></footer>
        </body></html>
    """)

    links = crawler._extract_links("https://example.com/", body)

    assert links == [("https://example.com/article", "Article")]


def test_fetch_keeps_internal_error_page_for_technical_audit() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, use_cache=False),
        _Cache(),
    )
    crawler._request_with_retry = lambda url: _Response(
        "Not found",
        url="https://example.com/missing",
        status_code=404,
    )

    result = crawler._fetch("https://example.com/missing")

    assert result is not None
    assert result.url == "https://example.com/missing"
    assert result.status == 404
    assert result.body == ""


def test_fetch_keeps_timeout_for_technical_audit() -> None:
    crawler = Crawler(
        CrawlConfig("example.com", respect_robots=False, use_cache=False),
        _Cache(),
    )
    response = _Response("", url="https://example.com/slow", status_code=0)
    response.reason = "timed_out"
    crawler._request_with_retry = lambda url: response

    result = crawler._fetch("https://example.com/slow")

    assert result is not None
    assert result.url == "https://example.com/slow"
    assert result.status == 0
    assert result.error == "timed_out"
