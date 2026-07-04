from types import SimpleNamespace

from site_audit import external_links


def test_external_links_payload_rolls_up_external_redirects_per_page(monkeypatch) -> None:
    def fake_check_links(targets, http_cache=None, max_workers=8, timeout=12.0):
        return [
            {
                "url": "https://external.example/old",
                "anchor": "Old",
                "status": 200,
                "final_url": "https://external.example/new",
                "redirect_status_codes": [301],
                "issues": ["external_3xx_redirect"],
                "from_cache": False,
            }
        ]

    monkeypatch.setattr(external_links, "_check_links", fake_check_links)

    result = external_links.to_payload(
        external_links.analyze(
            [SimpleNamespace(url="https://example.com/page")],
            {"https://example.com/page": 500},
            [
                (
                    "https://example.com/page",
                    [
                        ("https://external.example/old", "Old"),
                        ("https://external.example/clean", "Clean"),
                    ],
                )
            ],
            check_links=True,
        )
    )

    assert result["link_issues"][0]["issues"] == ["external_3xx_redirect"]
    assert result["broken_links"] == []
    assert result["per_page_issues"] == [
        {
            "url": "https://example.com/page",
            "issues": {"external_3xx_redirect": 1},
            "issue_count": 1,
            "external_links_with_issues": [
                {
                    "url": "https://external.example/old",
                    "anchor": "Old",
                    "status": 200,
                    "final_url": "https://external.example/new",
                    "redirect_status_codes": [301],
                    "issues": ["external_3xx_redirect"],
                }
            ],
        }
    ]


def test_external_links_payload_has_no_per_page_issues_when_checks_are_clean(monkeypatch) -> None:
    monkeypatch.setattr(external_links, "_check_links", lambda *args, **kwargs: [])

    result = external_links.to_payload(
        external_links.analyze(
            [SimpleNamespace(url="https://example.com/page")],
            {"https://example.com/page": 500},
            [("https://example.com/page", [("https://external.example/clean", "Clean")])],
            check_links=True,
        )
    )

    assert result["link_issues"] == []
    assert result["per_page_issues"] == []


def test_external_links_payload_keeps_external_4xx_as_broken_link(monkeypatch) -> None:
    def fake_check_links(targets, http_cache=None, max_workers=8, timeout=12.0):
        return [
            {
                "url": "https://external.example/missing",
                "anchor": "Missing",
                "status": 404,
                "issues": ["external_4xx"],
                "from_cache": False,
            }
        ]

    monkeypatch.setattr(external_links, "_check_links", fake_check_links)

    result = external_links.to_payload(
        external_links.analyze(
            [SimpleNamespace(url="https://example.com/page")],
            {"https://example.com/page": 500},
            [("https://example.com/page", [("https://external.example/missing", "Missing")])],
            check_links=True,
        )
    )

    assert result["broken_links"] == result["link_issues"]
    assert result["per_page_issues"][0]["issues"] == {"external_4xx": 1}


def test_external_links_payload_keeps_external_5xx_as_broken_link(monkeypatch) -> None:
    def fake_check_links(targets, http_cache=None, max_workers=8, timeout=12.0):
        return [
            {
                "url": "https://external.example/error",
                "anchor": "Error",
                "status": 503,
                "issues": ["external_5xx"],
                "from_cache": False,
            }
        ]

    monkeypatch.setattr(external_links, "_check_links", fake_check_links)

    result = external_links.to_payload(
        external_links.analyze(
            [SimpleNamespace(url="https://example.com/page")],
            {"https://example.com/page": 500},
            [("https://example.com/page", [("https://external.example/error", "Error")])],
            check_links=True,
        )
    )

    assert result["broken_links"] == result["link_issues"]
    assert result["per_page_issues"][0]["issues"] == {"external_5xx": 1}


def test_external_links_payload_keeps_external_time_out_as_broken_link(monkeypatch) -> None:
    def fake_check_links(targets, http_cache=None, max_workers=8, timeout=12.0):
        return [
            {
                "url": "https://external.example/slow",
                "anchor": "Slow",
                "status": None,
                "error": "timed out",
                "issues": ["external_time_out"],
                "from_cache": False,
            }
        ]

    monkeypatch.setattr(external_links, "_check_links", fake_check_links)

    result = external_links.to_payload(
        external_links.analyze(
            [SimpleNamespace(url="https://example.com/page")],
            {"https://example.com/page": 500},
            [("https://example.com/page", [("https://external.example/slow", "Slow")])],
            check_links=True,
        )
    )

    assert result["broken_links"] == result["link_issues"]
    assert result["per_page_issues"][0]["issues"] == {"external_time_out": 1}
