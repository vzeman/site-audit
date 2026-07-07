from site_audit.ai_access import AI_USER_AGENTS, build_ai_access
from site_audit.recommendations import synthesize, to_payload


def _agent(payload: dict, name: str) -> dict:
    return next(row for row in payload["agents"] if row["agent"] == name)


def test_ai_access_no_robots_allows_all_agents() -> None:
    payload = build_ai_access(None, None, None, "https://example.com")

    assert payload["available"] is False
    assert "no robots.txt found" in payload["notes"][0]
    assert payload["recommendations"] == []
    assert payload["summary"]["total_agents"] == len(AI_USER_AGENTS)
    assert all(row["allowed_root"] for row in payload["agents"])


def test_ai_access_blanket_block_creates_one_combined_recommendation() -> None:
    payload = build_ai_access("User-agent: *\nDisallow: /\n", None, None, "https://example.com")

    assert payload["summary"]["blanket_block"] is True
    assert all(not row["allowed_root"] for row in payload["agents"])
    assert len(payload["recommendations"]) == 1
    assert "blanket-blocks AI crawlers" in payload["recommendations"][0]


def test_ai_access_agent_specific_block_overrides_allowed_wildcard() -> None:
    robots = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    payload = build_ai_access(robots, None, None, "https://example.com")

    assert _agent(payload, "GPTBot")["allowed_root"] is False
    assert _agent(payload, "GPTBot")["explicitly_named"] is True
    assert _agent(payload, "OAI-SearchBot")["allowed_root"] is True
    assert payload["recommendations"] == []


def test_ai_access_multiple_user_agents_share_rules() -> None:
    robots = "User-agent: GPTBot\nUser-agent: ClaudeBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    payload = build_ai_access(robots, None, None, "https://example.com")

    assert _agent(payload, "GPTBot")["allowed_root"] is False
    assert _agent(payload, "ClaudeBot")["allowed_root"] is False
    assert _agent(payload, "PerplexityBot")["allowed_root"] is True


def test_ai_access_longest_user_agent_match_wins() -> None:
    robots = (
        "User-agent: Claude\n"
        "Disallow: /\n\n"
        "User-agent: Claude-SearchBot\n"
        "Allow: /\n\n"
        "User-agent: *\n"
        "Disallow: /\n"
    )
    payload = build_ai_access(robots, None, None, "https://example.com")

    assert _agent(payload, "Claude-SearchBot")["allowed_root"] is True
    assert _agent(payload, "Claude-SearchBot")["matched_group"] == "Claude-SearchBot"
    assert _agent(payload, "ClaudeBot")["allowed_root"] is False
    assert _agent(payload, "ClaudeBot")["matched_group"] == "Claude"


def test_google_extended_block_note_does_not_claim_search_or_ai_overview_impact() -> None:
    robots = "User-agent: Google-Extended\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    payload = build_ai_access(robots, None, None, "https://example.com")
    text = "\n".join(payload["recommendations"] + payload["notes"])

    assert _agent(payload, "Google-Extended")["allowed_root"] is False
    assert "AI Overview" not in text
    assert "cannot appear in Google Search" not in text
    assert payload["recommendations"] == []


def test_llms_payload_preserves_status_and_caps_first_lines() -> None:
    lines = [f"line {i}" for i in range(25)]
    payload = build_ai_access(
        None,
        {"present": True, "url": "https://example.com/llms.txt", "size_bytes": 100, "first_lines": lines},
        None,
        "https://example.com",
    )

    assert payload["llms_txt"]["present"] is True
    assert payload["llms_txt"]["first_lines"] == lines[:20]
    assert payload["llms_full_txt"]["present"] is False


def test_blocked_search_bot_creates_geo_recommendation_but_training_bot_does_not() -> None:
    payload = build_ai_access(
        "User-agent: PerplexityBot\nDisallow: /\n\nUser-agent: CCBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
        None,
        None,
        "https://example.com",
    )

    report = to_payload(synthesize(ai_access_payload=payload))
    items = report["items"]

    assert any(item["category"] == "geo" and item["evidence"]["agent"] == "PerplexityBot" for item in items)
    assert not any(item["evidence"].get("agent") == "CCBot" for item in items)
    assert any("Perplexity cannot crawl" in item["instruction"] for item in items)


def test_ai_access_unreachable_robots_reports_unknown_not_allowed() -> None:
    for status in (0, 500, 503):
        payload = build_ai_access(None, None, None, "https://example.com", robots_status=status)

        assert payload["available"] is False
        assert payload["evaluated"] is False
        assert "unreachable" in payload["reason"]
        assert "could not be evaluated" in payload["notes"][0]
        assert all(row["allowed_root"] is None for row in payload["agents"])
        assert payload["summary"]["search_blocked"] == 0
        assert payload["summary"]["user_fetch_blocked"] == 0
        assert payload["summary"]["training_blocked"] == 0
        assert payload["recommendations"] == []


def test_ai_access_missing_robots_status_404_still_means_all_allowed() -> None:
    payload = build_ai_access(None, None, None, "https://example.com", robots_status=404)

    assert payload["evaluated"] is True
    assert "no robots.txt found" in payload["reason"]
    assert all(row["allowed_root"] is True for row in payload["agents"])


def test_ai_access_recommendation_ids_are_stable_per_agent() -> None:
    both_blocked = build_ai_access(
        "User-agent: Claude-SearchBot\nDisallow: /\n\nUser-agent: PerplexityBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
        None,
        None,
        "https://example.com",
    )
    one_blocked = build_ai_access(
        "User-agent: PerplexityBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
        None,
        None,
        "https://example.com",
    )

    ids_both = {rec.id for rec in synthesize(ai_access_payload=both_blocked) if rec.id.startswith("geo-access")}
    ids_one = {rec.id for rec in synthesize(ai_access_payload=one_blocked) if rec.id.startswith("geo-access")}

    assert ids_both == {"geo-access-claude-searchbot", "geo-access-perplexitybot"}
    assert ids_one == {"geo-access-perplexitybot"}


def test_ai_access_recommendation_instructions_come_from_payload_strings() -> None:
    payload = build_ai_access(
        "User-agent: PerplexityBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
        None,
        None,
        "https://example.com",
    )

    recs = [rec for rec in synthesize(ai_access_payload=payload) if rec.id.startswith("geo-access")]
    assert [rec.instruction for rec in recs] == payload["recommendations"]


def _llms_crawler(responses: dict[str, object]):
    from site_audit.crawler import CrawlConfig, Crawler

    class _Cache:
        def get(self, url):
            return None

    crawler = Crawler(CrawlConfig(domain="example.com"), _Cache())

    def fake_request(url: str, method: str = "GET"):
        return responses.get(url)

    crawler._request_with_retry = fake_request
    return crawler


class _FakeResponse:
    def __init__(self, body: str, url: str, status_code: int = 200, content_type: str = "text/plain"):
        self.status_code = status_code
        self.text = body
        self.content = body.encode("utf-8")
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.reason = ""


def test_crawler_rejects_html_llms_txt_as_soft_404() -> None:
    url = "https://example.com/llms.txt"
    crawler = _llms_crawler({url: _FakeResponse("<!doctype html><html>SPA</html>", url, content_type="text/html; charset=utf-8")})

    info = crawler._fetch_ai_text_file("/llms.txt")

    assert info["present"] is False
    assert info["first_lines"] == []
    assert info["size_bytes"] == 0


def test_crawler_accepts_plain_text_llms_txt() -> None:
    url = "https://example.com/llms.txt"
    body = "\n".join(f"line {i}" for i in range(25))
    crawler = _llms_crawler({url: _FakeResponse(body, url)})

    info = crawler._fetch_ai_text_file("/llms.txt")

    assert info["present"] is True
    assert len(info["first_lines"]) == 20
    assert info["size_bytes"] == len(body.encode("utf-8"))


def test_crawler_fetches_robots_info_even_when_not_respecting_robots() -> None:
    robots_url = "https://example.com/robots.txt"
    body = "User-agent: GPTBot\nDisallow: /\n"
    crawler = _llms_crawler({robots_url: _FakeResponse(body, robots_url)})
    crawler.config.respect_robots = False

    assert crawler._load_robots() is None  # enforcement stays off
    assert crawler.robots_txt_info["status"] == 200
    assert crawler.robots_txt_info["body"] == body
