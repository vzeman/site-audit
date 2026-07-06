import json

from site_audit.ai_agent import AgentCompletion
from site_audit.fix_drafts import attach_fix_drafts, build_fix_drafts


PAGE_URL = "https://example.com/live-chat/"


class FakeClient:
    provider = "fake"

    def __init__(self, responses=None, exc: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.exc = exc
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, model: str, temperature: float = 0.2, timeout: int = 120):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        text = self.responses.pop(0)
        return AgentCompletion(text=text, provider=self.provider, model=model)


def _recommendations(items: list[dict]) -> dict:
    return {"items": items, "cards": [{"recommendations": [dict(item) for item in items]}]}


def _item(rec_id: str, evidence: dict | None = None, targets: list[str] | None = None) -> dict:
    return {
        "id": rec_id,
        "title": f"Recommendation {rec_id}",
        "instruction": "Rewrite this item.",
        "targets": targets if targets is not None else [PAGE_URL],
        "evidence": evidence or {},
    }


def _pages() -> list[dict]:
    return [
        {
            "url": PAGE_URL,
            "title": "Old Live Chat Title | Example",
            "description": "Live chat software for support teams. Connect with visitors quickly.",
            "word_count": 900,
            "headings": ["Live Chat Software", "Customer Support Workflows"],
            "paragraphs": ["Live chat software helps support teams answer visitors quickly."],
        }
    ]


def _search() -> dict:
    return {
        "query_pages": [
            {"matched_url": PAGE_URL, "keyword": "live chat software", "impressions": 500},
            {"matched_url": PAGE_URL, "keyword": "customer chat tool", "impressions": 100},
        ]
    }


def test_fix_drafts_attach_to_rec_ids_and_batches() -> None:
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
        _item("ctr-a", {"current_title": "Old CTR Title", "query": "support chat"}),
        _item("geo-a", {"flags": ["answer opening is indirect"]}),
        _item("gap-a", {"query": "help desk automation"}),
    ])
    client = FakeClient([
        json.dumps({
            "drafts": [
                {
                    "id": "title-a",
                    "proposed_title": "Live Chat Software for Support Teams",
                    "proposed_meta": "Live chat software for support teams that helps answer visitors quickly.",
                },
                {
                    "id": "ctr-a",
                    "proposed_title": "Support Chat Software for Fast Replies",
                    "proposed_meta": "Support chat software that helps teams answer visitors quickly.",
                },
            ]
        }),
        json.dumps({
            "drafts": [
                {
                    "id": "geo-a",
                    "questions": [
                        "What is live chat software?",
                        "How does live chat software help support teams?",
                        "Why use live chat software for visitors?",
                    ],
                },
                {
                    "id": "gap-a",
                    "proposed_title": "Help Desk Automation Guide",
                    "outline": [
                        "What help desk automation covers",
                        "Common support workflows",
                        "Automation setup checklist",
                        "Measurement and next steps",
                    ],
                },
            ]
        }),
    ])

    drafts = build_fix_drafts(payload, _pages(), _search(), client, model="fake-model", batch_size=2)
    attach_fix_drafts(payload, drafts)

    assert len(client.calls) == 2
    assert list(drafts["drafts"]) == ["title-a", "ctr-a", "geo-a", "gap-a"]
    assert payload["items"][0]["fix_draft"]["proposed_title"] == "Live Chat Software for Support Teams"
    assert payload["cards"][0]["recommendations"][2]["fix_draft"]["questions"][0].endswith("?")
    assert drafts["summary"]["llm"] == 4
    assert drafts["model_used"] == "fake-model"


def test_invalid_title_repairs_and_uses_repaired_value() -> None:
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
    ])
    client = FakeClient([
        json.dumps({
            "drafts": [{
                "id": "title-a",
                "proposed_title": "X" * 90,
                "proposed_meta": "Live chat software for support teams.",
            }]
        }),
        json.dumps({
            "drafts": [{
                "id": "title-a",
                "proposed_title": "Live Chat Software for Support Teams",
                "proposed_meta": "Live chat software for support teams.",
            }]
        }),
    ])

    drafts = build_fix_drafts(payload, _pages(), _search(), client, model="fake-model")

    assert len(client.calls) == 2
    assert drafts["drafts"]["title-a"]["proposed_title"] == "Live Chat Software for Support Teams"
    assert drafts["drafts"]["title-a"]["generated_by"] == "llm"
    assert drafts["summary"]["repaired"] == 1


def test_invented_digit_and_bad_question_fall_back() -> None:
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
        _item("geo-a", {"flags": ["answer opening is indirect"]}),
    ])
    bad = json.dumps({
        "drafts": [
            {
                "id": "title-a",
                "proposed_title": "Live Chat 73 Percent Faster",
                "proposed_meta": "Improve response time by 73% with live chat.",
            },
            {
                "id": "geo-a",
                "questions": [
                    "What is live chat software",
                    "How does live chat software help support teams?",
                    "Why use live chat software?",
                ],
            },
        ]
    })
    client = FakeClient([bad, bad])

    drafts = build_fix_drafts(payload, _pages(), _search(), client, model="fake-model")

    assert drafts["drafts"]["title-a"]["generated_by"] == "fallback"
    assert drafts["drafts"]["geo-a"]["generated_by"] == "fallback"
    assert drafts["summary"]["fallback"] == 2
    assert drafts["summary"]["failed"] == 2


def test_client_none_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr("site_audit.fix_drafts.openrouter_api_key", lambda: "")
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
        _item("geo-a", {"flags": ["answer opening is indirect"]}),
    ])

    first = build_fix_drafts(payload, _pages(), _search(), None)
    second = build_fix_drafts(payload, _pages(), _search(), None)

    assert first == second
    assert first["drafts"]["title-a"]["generated_by"] == "fallback"
    assert first["drafts"]["title-a"]["proposed_title"] == "live chat software — Example"
    assert first["drafts"]["geo-a"]["questions"] == [
        "What is live chat software?",
        "What is customer chat tool?",
        "How can this help?",
    ]
    assert first["summary"]["fallback"] == 2


def test_determinism_with_fake_client() -> None:
    responses = [json.dumps({
        "drafts": [
            {
                "id": "title-a",
                "proposed_title": "Live Chat Software for Support Teams",
                "proposed_meta": "Live chat software for support teams that helps answer visitors quickly.",
            },
            {
                "id": "geo-a",
                "questions": [
                    "What is live chat software?",
                    "How does live chat software help support teams?",
                    "Why use live chat software for visitors?",
                ],
            },
        ]
    })]

    def build() -> dict:
        payload = _recommendations([
            _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
            _item("geo-a", {"flags": ["answer opening is indirect"]}),
        ])
        return build_fix_drafts(payload, _pages(), _search(), FakeClient(responses), model="fake-model")

    first = build()
    second = build()

    assert first == second
    assert first["drafts"]["title-a"]["generated_by"] == "llm"
    assert first["model_used"] == "fake-model"


def test_malformed_json_repairs_then_falls_back() -> None:
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
    ])
    client = FakeClient(["not json", "still not json"])

    drafts = build_fix_drafts(payload, _pages(), _search(), client, model="fake-model")

    assert len(client.calls) == 2
    assert drafts["drafts"]["title-a"]["generated_by"] == "fallback"
    assert drafts["summary"]["fallback"] == 1
    assert drafts["summary"]["failed"] == 1


def test_client_raising_falls_back_and_records_error() -> None:
    payload = _recommendations([
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
    ])
    client = FakeClient(exc=RuntimeError("boom"))

    drafts = build_fix_drafts(payload, _pages(), _search(), client, model="fake-model")

    assert len(client.calls) == 1
    assert drafts["drafts"]["title-a"]["generated_by"] == "fallback"
    assert drafts["summary"]["fallback"] == 1
    assert drafts["summary"]["errors"] == [{"type": "RuntimeError", "items": 1}]


def test_selection_only_draftable_prefixes_and_caps(monkeypatch) -> None:
    monkeypatch.setattr("site_audit.fix_drafts.openrouter_api_key", lambda: "")
    payload = _recommendations([
        _item("link-a"),
        _item("title-a", {"current_title": "Old Live Chat Title | Example"}),
        _item("anchor-a"),
        _item("ctr-a", {"current_title": "Old CTR", "query": "support chat"}),
        _item("geo-a"),
    ])

    drafts = build_fix_drafts(payload, _pages(), _search(), None, max_items=2)

    assert drafts["summary"]["requested"] == 2
    assert list(drafts["drafts"]) == ["title-a", "ctr-a"]
