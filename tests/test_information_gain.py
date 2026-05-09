from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.information_gain import build_information_gain


def test_information_gain_rewards_original_evidence_and_flags_generic_copy():
    pages = [
        PageInfo("https://example.com/a", "Benchmark Study", "", "blog", 200),
        PageInfo("https://example.com/b", "Generic Guide", "", "blog", 120),
    ]
    extracted = [
        SimpleNamespace(
            h1="Benchmark Study",
            headers_rich=[],
            body="We tested 42 customer support workflows. For example, Acme Corp reduced response time by 31% after adding SLA Automation.",
            external_link_count=2,
            media_items=[{"type": "image"}],
            schema_types=["Article"],
        ),
        SimpleNamespace(
            h1="Generic Guide",
            headers_rich=[],
            body="In today's digital world it is important to learn more about customer support. This article will help businesses need better tools.",
            external_link_count=0,
            media_items=[],
            schema_types=[],
        ),
    ]
    paragraph_records = [
        (0, 0, extracted[0].body, np.array([1.0], dtype=np.float32)),
        (1, 0, extracted[1].body, np.array([1.0], dtype=np.float32)),
    ]

    payload = build_information_gain(pages, extracted, paragraph_records, cluster_labels=[0, 0])

    scores = {row["url"]: row["information_gain_score"] for row in payload["pages"]}
    assert payload["summary"]["status"] == "ok"
    assert scores["https://example.com/a"] > scores["https://example.com/b"]
    strong = next(row for row in payload["pages"] if row["url"].endswith("/a"))
    weak = next(row for row in payload["pages"] if row["url"].endswith("/b"))
    assert strong["positive_evidence"]
    assert strong["unique_facts"]
    assert weak["negative_reasons"]
    assert weak["recommendations"]


def test_information_gain_distinguishes_duplicated_boilerplate_paragraphs():
    pages = [PageInfo(f"https://example.com/{i}", f"Page {i}", "", "root", 80) for i in range(2)]
    text = "Copyright Example Company. Learn more about our services and contact us for more information."
    extracted = [
        SimpleNamespace(h1=f"Page {i}", headers_rich=[], body=text, external_link_count=0, media_items=[], schema_types=[])
        for i in range(2)
    ]
    paragraph_records = [(i, 0, text, np.array([1.0], dtype=np.float32)) for i in range(2)]

    payload = build_information_gain(pages, extracted, paragraph_records, cluster_labels=[0, 0])

    row = payload["paragraphs"][0]
    assert row["information_gain_score"] < 55
    assert any("duplicated" in reason for reason in row["negative_reasons"])
    assert any("boilerplate" in reason for reason in row["negative_reasons"])
