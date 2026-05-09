from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.cannibalization import build_cannibalization


def test_cannibalization_flags_page_and_paragraph_conflicts_with_winner():
    pages = [
        PageInfo("https://example.com/help-desk-software", "Help desk software", "", "blog", 200),
        PageInfo("https://example.com/best-help-desk-tools", "Best help desk tools", "", "blog", 190),
    ]
    page_embeddings = np.array([[1.0, 0.0], [0.93, 0.1]], dtype=np.float32)
    page_embeddings = page_embeddings / np.linalg.norm(page_embeddings, axis=1, keepdims=True)
    para_a = "Help desk software centralizes support tickets, SLA automation, and reporting for support teams."
    para_b = "The best help desk tools centralize support tickets, SLA automation, and reporting for support teams."
    paragraph_records = [
        (0, 0, para_a, np.array([1.0, 0.0], dtype=np.float32)),
        (1, 0, para_b, np.array([0.98, 0.05], dtype=np.float32)),
    ]
    keyword_attribution = {
        "keywords": [
            {
                "keyword": "help desk software",
                "url": pages[0].url,
                "traffic": 100,
                "position": 2,
                "best_paragraph_index": 0,
                "best_paragraph_excerpt": para_a,
            },
            {
                "keyword": "best help desk software",
                "url": pages[1].url,
                "traffic": 35,
                "position": 8,
                "best_paragraph_index": 0,
                "best_paragraph_excerpt": para_b,
            },
        ]
    }
    search_payload = {
        "top_pages": [
            {"matched_url": pages[0].url, "traffic": 120, "keywords": 10, "top_keyword": "help desk software"},
            {"matched_url": pages[1].url, "traffic": 35, "keywords": 6, "top_keyword": "best help desk software"},
        ]
    }
    linkgraph = {
        "page_link_counts": [
            {"url": pages[0].url, "in_degree": 8, "click_depth": 1},
            {"url": pages[1].url, "in_degree": 2, "click_depth": 3},
        ]
    }

    payload = build_cannibalization(
        pages,
        page_embeddings,
        [SimpleNamespace(), SimpleNamespace()],
        paragraph_records,
        keyword_attribution=keyword_attribution,
        search_payload=search_payload,
        linkgraph=linkgraph,
    )

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["page_conflicts"] == 1
    conflict = payload["page_conflicts"][0]
    assert conflict["preferred_winner_url"] == pages[0].url
    assert conflict["competitors"][0]["recommended_action"] in {"merge_or_canonical", "retarget_or_merge", "differentiate_or_merge"}
    assert payload["paragraph_overlaps"]
    assert payload["paragraph_overlaps"][0]["preferred_winner_url"] == pages[0].url


def test_cannibalization_classifies_localized_variants_separately():
    pages = [
        PageInfo("https://example.com/en/live-chat", "Live chat software", "", "root", 100),
        PageInfo("https://example.com/de/live-chat", "Live chat software Germany", "", "root", 100),
    ]
    embs = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    keyword_attribution = {
        "keywords": [
            {"keyword": "live chat software", "url": pages[0].url, "traffic": 20, "position": 4},
            {"keyword": "live chat software germany", "url": pages[1].url, "traffic": 15, "position": 5},
        ]
    }

    payload = build_cannibalization(pages, embs, keyword_attribution=keyword_attribution)

    assert payload["page_conflicts"][0]["classification"] == "localized_variant"
    assert payload["summary"]["localized_or_variant_groups"] == 1
