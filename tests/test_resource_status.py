from types import SimpleNamespace

from site_audit.resource_status import analyze, to_payload


class _Cache:
    def __init__(self, statuses: dict[str, int]) -> None:
        self.statuses = statuses

    def get(self, url: str):
        status = self.statuses.get(url)
        if status is None:
            return None
        return SimpleNamespace(status=status)


def test_resource_status_payload_flags_broken_javascript_from_cache() -> None:
    fetch = SimpleNamespace(
        url="https://example.com/page",
        body="""
        <html><head>
          <script src="/broken.js"></script>
          <script src="https://cdn.example.com/ok.js"></script>
          <script>window.inline = true;</script>
        </head><body></body></html>
        """,
    )
    cache = _Cache({
        "https://example.com/broken.js": 404,
        "https://cdn.example.com/ok.js": 200,
    })

    report = to_payload(analyze([fetch], http_cache=cache))

    assert report["summary"]["total_javascript"] == 2
    assert report["summary"]["broken_javascript"] == 1
    assert report["issues_by_type"]["javascript_broken"] == 1
    assert report["per_page"][0]["issues"]["javascript_broken"] == 1
    assert report["resources_with_issues"] == [
        {
            "url": "https://example.com/page",
            "title": "",
            "index": 0,
            "type": "javascript",
            "src": "https://example.com/broken.js",
            "http_status": 404,
            "redirect_target_url": "",
            "size_bytes": 0,
            "issues": ["javascript_broken"],
        }
    ]


def test_resource_status_payload_flags_broken_css_from_cache() -> None:
    fetch = SimpleNamespace(
        url="https://example.com/page",
        body="""
        <html><head>
          <link rel="stylesheet" href="/broken.css">
          <link rel="stylesheet" href="https://cdn.example.com/ok.css">
        </head><body></body></html>
        """,
    )
    cache = _Cache({
        "https://example.com/broken.css": 404,
        "https://cdn.example.com/ok.css": 200,
    })

    report = to_payload(analyze([fetch], http_cache=cache))

    assert report["summary"]["total_css"] == 2
    assert report["summary"]["broken_css"] == 1
    assert report["issues_by_type"]["css_broken"] == 1
    assert report["per_page"][0]["issues"]["css_broken"] == 1
    assert report["per_page"][0]["css_count"] == 2
    assert report["resources_with_issues"][0]["src"] == "https://example.com/broken.css"


def test_resource_status_payload_uses_explicit_resource_items() -> None:
    fetch = SimpleNamespace(
        url="https://example.com/page",
        resource_items=[
            {"type": "javascript", "src": "https://cdn.example.com/broken.js", "http_status": 500},
            {"type": "javascript", "src": "https://cdn.example.com/ok.js", "http_status": 200},
        ],
    )

    report = to_payload(analyze([fetch]))

    assert report["summary"]["total_javascript"] == 2
    assert report["summary"]["broken_javascript"] == 1
    assert report["resources_with_issues"][0]["src"] == "https://cdn.example.com/broken.js"


def test_resource_status_payload_flags_large_css() -> None:
    report = to_payload(analyze([
        SimpleNamespace(
            url="https://example.com/page",
            resource_items=[
                {"type": "css", "src": "https://cdn.example.com/large.css", "size_bytes": 220_000},
                {"type": "css", "src": "https://cdn.example.com/ok.css", "size_bytes": 80_000},
            ],
        ),
    ]))

    assert report["summary"]["large_css"] == 1
    assert report["issues_by_type"]["css_file_size_too_large"] == 1
    large_rows = [
        row for row in report["resources_with_issues"]
        if "css_file_size_too_large" in row["issues"]
    ]
    assert len(large_rows) == 1
    assert large_rows[0]["src"] == "https://cdn.example.com/large.css"
    assert large_rows[0]["size_bytes"] == 220_000


def test_resource_status_payload_flags_https_pages_linking_to_http_javascript() -> None:
    report = to_payload(analyze([
        SimpleNamespace(
            url="https://example.com/secure",
            resource_items=[
                {"type": "javascript", "src": "http://cdn.example.com/insecure.js", "http_status": 200},
                {"type": "javascript", "src": "https://cdn.example.com/secure.js", "http_status": 200},
            ],
        ),
        SimpleNamespace(
            url="http://example.com/plain",
            resource_items=[
                {"type": "javascript", "src": "http://cdn.example.com/http-page.js", "http_status": 200},
            ],
        ),
    ]))

    assert report["summary"]["https_pages_linking_to_http_javascript"] == 1
    assert report["issues_by_type"]["https_page_links_to_http_javascript"] == 1
    insecure_rows = [
        row for row in report["resources_with_issues"]
        if "https_page_links_to_http_javascript" in row["issues"]
    ]
    assert len(insecure_rows) == 1
    assert insecure_rows[0]["url"] == "https://example.com/secure"
    assert insecure_rows[0]["src"] == "http://cdn.example.com/insecure.js"


def test_resource_status_payload_flags_redirected_javascript() -> None:
    report = to_payload(analyze([
        SimpleNamespace(
            url="https://example.com/page",
            resource_items=[
                {
                    "type": "javascript",
                    "src": "https://cdn.example.com/redirect.js",
                    "http_status": 301,
                    "redirect_target_url": "https://cdn.example.com/final.js",
                },
                {"type": "javascript", "src": "https://cdn.example.com/flagged.js", "redirected": True},
                {"type": "javascript", "src": "https://cdn.example.com/ok.js", "http_status": 200},
            ],
        ),
    ]))

    assert report["summary"]["redirected_javascript"] == 2
    assert report["issues_by_type"]["javascript_redirects"] == 2
    redirected_rows = [
        row for row in report["resources_with_issues"]
        if "javascript_redirects" in row["issues"]
    ]
    assert [row["src"] for row in redirected_rows] == [
        "https://cdn.example.com/redirect.js",
        "https://cdn.example.com/flagged.js",
    ]
    assert redirected_rows[0]["redirect_target_url"] == "https://cdn.example.com/final.js"


def test_resource_status_payload_flags_redirected_css() -> None:
    report = to_payload(analyze([
        SimpleNamespace(
            url="https://example.com/page",
            resource_items=[
                {
                    "type": "css",
                    "src": "https://cdn.example.com/redirect.css",
                    "http_status": 302,
                    "redirect_target_url": "https://cdn.example.com/final.css",
                },
                {"type": "css", "src": "https://cdn.example.com/flagged.css", "redirected": True},
                {"type": "css", "src": "https://cdn.example.com/ok.css", "http_status": 200},
            ],
        ),
    ]))

    assert report["summary"]["redirected_css"] == 2
    assert report["issues_by_type"]["css_redirects"] == 2
    redirected_rows = [
        row for row in report["resources_with_issues"]
        if "css_redirects" in row["issues"]
    ]
    assert [row["src"] for row in redirected_rows] == [
        "https://cdn.example.com/redirect.css",
        "https://cdn.example.com/flagged.css",
    ]
    assert redirected_rows[0]["redirect_target_url"] == "https://cdn.example.com/final.css"
