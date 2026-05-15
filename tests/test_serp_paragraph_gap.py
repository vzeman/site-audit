from pathlib import Path

import numpy as np

from site_audit.competitive_analysis import (
    CompetitiveAutoConfig,
    CompetitiveTarget,
    CompetitorPage,
    build_serp_paragraph_gap,
    load_competitive_pairs,
    load_competitive_targets,
    select_competitive_auto_keywords,
)
from site_audit.analyzer import PageInfo


def _norm(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom


class DummyEmbedder:
    def encode(self, texts, batch_size=32, show_progress=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "workflow" in lowered or "agent" in lowered or "automation" in lowered:
                vectors.append([1.0, 0.0])
            elif "lyrics" in lowered or "image" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7, 0.3])
        return _norm(vectors)


def test_load_competitive_targets_supports_cluster_headers_and_rank(tmp_path: Path) -> None:
    path = tmp_path / "competitive.tsv"
    path.write_text(
        "\n".join([
            "# AI agents",
            "ai workflow automation\thttps://lindy.ai/ai-workflow-automation\t1",
            "AI chatbots | ai chatbot platform | 2 | https://lindy.ai/ai-chatbot",
        ]),
        encoding="utf-8",
    )

    targets = load_competitive_targets(path)

    assert targets[0] == CompetitiveTarget(
        query="ai workflow automation",
        competitor_url="https://lindy.ai/ai-workflow-automation",
        cluster="AI agents",
        rank=1,
    )
    assert targets[1] == CompetitiveTarget(
        query="ai chatbot platform",
        competitor_url="https://lindy.ai/ai-chatbot",
        cluster="AI chatbots",
        rank=2,
    )
    assert load_competitive_pairs(path)[0] == ("ai workflow automation", "https://lindy.ai/ai-workflow-automation")


def test_build_serp_paragraph_gap_finds_missing_and_partial_topics() -> None:
    competitor_pages = [
        CompetitorPage(
            target=CompetitiveTarget("ai workflow automation", "https://lindy.ai/a", "AI agents", 1),
            title="Lindy A",
            paragraphs=[
                "AI workflow automation routes leads, summarizes calls, and updates CRM records.",
                "Implementation usually starts by mapping triggers, approvals, and fallback rules.",
            ],
            paragraph_embeddings=_norm([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            structural_gaps=[],
            answerability=80,
            paragraph_count=2,
        ),
        CompetitorPage(
            target=CompetitiveTarget("ai workflow automation", "https://lindy.ai/b", "AI agents", 2),
            title="Lindy B",
            paragraphs=[
                "Pricing depends on task volume, team seats, and the number of automations running.",
                "Implementation plans should define human review, escalation, and testing steps.",
            ],
            paragraph_embeddings=_norm([[0.0, 0.0, 1.0], [0.0, 0.95, 0.05]]),
            structural_gaps=[],
            answerability=78,
            paragraph_count=2,
        ),
    ]
    our_paragraphs = [
        "Our AI automation platform routes leads and updates CRM records.",
        "Teams can build automations visually without code.",
    ]
    our_embeddings = _norm([[0.96, 0.04, 0.0], [0.45, 0.55, 0.0]])

    payload = build_serp_paragraph_gap(
        query="ai workflow automation",
        cluster="AI agents",
        our_url="https://www.flowhunt.io/ai-workflow-automation/",
        our_title="AI Workflow Automation",
        our_paragraphs=our_paragraphs,
        our_paragraph_embeddings=our_embeddings,
        competitor_pages=competitor_pages,
        missing_threshold=0.62,
        covered_threshold=0.78,
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["missing"] >= 1
    assert payload["summary"]["covered"] >= 1
    missing = payload["missing_topics"]
    assert any(topic["our_best_similarity"] < 0.62 for topic in missing)
    assert payload["topics"][0]["priority"] in {"critical", "high", "medium"}


def test_select_competitive_auto_keywords_filters_noise_and_caps_clusters() -> None:
    pages = [
        PageInfo(
            url="https://flowhunt.io/services/ai-agents/",
            title="AI Agents",
            description="",
            section="services",
            word_count=800,
            language="en",
        ),
        PageInfo(
            url="https://flowhunt.io/glossary/chatgpt/",
            title="ChatGPT glossary",
            description="",
            section="glossary",
            word_count=300,
            language="en",
        ),
    ]
    payload = {
        "organic_keywords": [
            {
                "keyword": "best ai workflow automation platform",
                "position": 8,
                "traffic": 100,
                "volume": 1200,
                "matched_url": pages[0].url,
                "cluster_label": "AI workflow automation",
                "intents": ["commercial"],
            },
            {
                "keyword": "شات جي بي تي",
                "position": 9,
                "traffic": 500,
                "volume": 3000,
                "matched_url": pages[1].url,
                "cluster_label": "ChatGPT glossary",
                "intents": ["informational"],
            },
            {
                "keyword": "free image generator",
                "position": 6,
                "traffic": 80,
                "volume": 900,
                "matched_url": pages[0].url,
                "cluster_label": "AI tools",
                "intents": ["informational"],
            },
        ]
    }

    selection = select_competitive_auto_keywords(
        payload,
        pages,
        DummyEmbedder(),
        CompetitiveAutoConfig(
            enabled=True,
            max_clusters=1,
            keywords_per_cluster=1,
            product_seeds=["AI workflow automation"],
            min_relevance=0.35,
        ),
    )

    assert selection["status"] == "ok"
    assert selection["summary"]["selected_clusters"] == 1
    assert selection["selected_keywords"][0]["keyword"] == "best ai workflow automation platform"
    rejected_keywords = {row["keyword"] for row in selection["rejected"]}
    assert "شات جي بي تي" in rejected_keywords
    assert "free image generator" in rejected_keywords
