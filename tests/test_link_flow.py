import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.linkgraph import analyze, link_flow_payload


def test_link_flow_payload_keeps_traffic_nodes_and_weighted_edges() -> None:
    pages = [
        PageInfo(url="https://example.com/", title="Home", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/a", title="Guide A", description="", section="blog", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/b", title="Guide B", description="", section="blog", word_count=100, language="en"),
    ]
    outlinks = [
        (pages[0].url, [(pages[1].url, "guide"), (pages[1].url, "read guide"), (pages[2].url, "guide b")]),
        (pages[1].url, [(pages[2].url, "next")]),
        (pages[2].url, []),
    ]
    result = analyze(pages, np.eye(3, dtype=np.float32), outlinks, home_url=pages[0].url)
    payload = link_flow_payload(
        result,
        pages,
        [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 20, "top_keyword": "guide a"}],
        max_nodes=3,
        max_edges=10,
    )

    assert payload["shown_nodes"] == 3
    assert any(node["traffic"] == 1000 and node["top_keyword"] == "guide a" for node in payload["nodes"])
    assert any(edge["source"] == pages[0].url and edge["target"] == pages[1].url and edge["weight"] == 2 for edge in payload["edges"])
