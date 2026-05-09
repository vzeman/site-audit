import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.linkgraph import analyze, link_flow_payload, traffic_weighted_pagerank_payload


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
    authority = traffic_weighted_pagerank_payload(
        result,
        pages,
        np.eye(3, dtype=np.float32),
        search_payload={"top_pages": [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 20, "top_keyword": "guide a", "cluster_label": "guides"}]},
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
    )
    payload = link_flow_payload(
        result,
        pages,
        [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 20, "top_keyword": "guide a", "cluster_label": "guides"}],
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
        traffic_authority=authority,
        max_nodes=3,
        max_edges=10,
    )

    assert payload["shown_nodes"] == 3
    traffic_node = next(node for node in payload["nodes"] if node["traffic"] == 1000)
    assert traffic_node["top_keyword"] == "guide a"
    assert traffic_node["directory"] == "/blog/"
    assert traffic_node["cluster"] == "guides"
    assert traffic_node["page_type"] == "article"
    assert traffic_node["traffic_weighted_pagerank"] > 0
    assert any(edge["source"] == pages[0].url and edge["target"] == pages[1].url and edge["weight"] == 2 for edge in payload["edges"])


def test_traffic_weighted_pagerank_flags_mismatches_and_nonindexable_search_pages() -> None:
    pages = [
        PageInfo(url="https://example.com/", title="Home", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/demand", title="Demand guide", description="", section="blog", word_count=100, language="en"),
        PageInfo(url="https://example.com/old", title="Old hub", description="", section="old", word_count=100, language="en"),
    ]
    outlinks = [
        (pages[0].url, [(pages[2].url, "old hub")]),
        (pages[1].url, []),
        (pages[2].url, [(pages[0].url, "home")]),
    ]
    result = analyze(pages, np.eye(3, dtype=np.float32), outlinks, home_url=pages[0].url)
    payload = traffic_weighted_pagerank_payload(
        result,
        pages,
        np.eye(3, dtype=np.float32),
        search_payload={
            "top_pages": [
                {"matched_url": pages[1].url, "traffic": 500, "keywords": 12, "top_keyword": "demand guide", "cluster_label": "guides"},
                {"url": "https://example.com/noindex", "traffic": 80, "keywords": 3, "top_keyword": "blocked page"},
            ]
        },
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
        indexability={"skipped": [{"url": "https://example.com/noindex", "title": "Noindex", "status": "skipped", "reason": "noindex"}], "noindex_pages": []},
    )

    demand = next(row for row in payload["pages"] if row["url"] == pages[1].url)
    assert demand["traffic"] == 500
    assert demand["pagerank"] >= 0
    assert demand["weighted_pagerank_percentile"] >= 0
    assert demand["mismatch_label"] in {"ranked_orphan", "high_traffic_low_authority"}
    assert payload["summary"]["high_traffic_low_authority_pages"] >= 1
    assert payload["mismatches"]["non_indexable_search_pages"][0]["url"] == "https://example.com/noindex"
