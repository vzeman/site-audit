from site_audit.crawler import CrawlConfig, Crawler


class _Cache:
    def get(self, url):
        return None


class _Response:
    def __init__(self, body: str, url: str = "https://example.com/sitemap.xml"):
        self.status_code = 200
        self.content = body.encode("utf-8")
        self.text = body
        self.url = url
        self.headers = {"Content-Type": "application/xml"}


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
