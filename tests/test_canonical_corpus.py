from site_audit.analyzer import PageInfo
from site_audit.embedder import EmbedInput
from site_audit.extractor import ExtractedPage
from site_audit.pipeline import filter_to_unique_canonical_pages


def _page(url: str, canonical_url: str) -> tuple[PageInfo, ExtractedPage, EmbedInput, dict]:
    page = PageInfo(url=url, title=url, description="", section="root", word_count=100, language="en")
    extracted = ExtractedPage(
        url=url,
        title=url,
        description="",
        body="Body",
        word_count=100,
        language="en",
        canonical_url=canonical_url,
    )
    embed = EmbedInput(url=url, text=f"{url} Body")
    row = {
        "url": url,
        "title": url,
        "status": "analyzed",
        "reason": "",
        "canonical_url": canonical_url,
    }
    return page, extracted, embed, row


def test_filter_to_unique_canonical_pages_keeps_self_canonical_target() -> None:
    canonical = _page("https://example.com/page/", "https://example.com/page/")
    variant = _page("https://example.com/page/?utm_source=x", "https://example.com/page/")
    input_pages = [canonical[0], variant[0]]
    input_extracted = [canonical[1], variant[1]]
    input_embeds = [canonical[2], variant[2]]
    input_rows = [canonical[3], variant[3]]

    kept_pages, kept_extracted, kept_embeds, dropped = filter_to_unique_canonical_pages(
        input_pages,
        input_extracted,
        input_embeds,
        input_rows,
    )

    assert dropped == 1
    assert [page.url for page in kept_pages] == ["https://example.com/page/"]
    assert [page.url for page in kept_extracted] == ["https://example.com/page/"]
    assert [embed.url for embed in kept_embeds] == ["https://example.com/page/"]
    assert input_rows[1]["status"] == "skipped"
    assert input_rows[1]["reason"] == "canonical_duplicate"
    assert input_rows[1]["canonical_kept_url"] == "https://example.com/page/"


def test_filter_to_unique_canonical_pages_drops_non_crawled_canonical_target() -> None:
    variant = _page("https://example.com/page/?source=feed", "https://example.com/page/")

    kept_pages, _, _, dropped = filter_to_unique_canonical_pages(
        [variant[0]],
        [variant[1]],
        [variant[2]],
        [variant[3]],
    )

    assert dropped == 1
    assert kept_pages == []
    assert variant[3]["reason"] == "canonical_duplicate"
    assert variant[3]["canonical_kept_url"] == ""


def test_filter_to_unique_canonical_pages_keeps_missing_canonical_pages() -> None:
    missing = _page("https://example.com/missing", "")

    kept_pages, _, _, dropped = filter_to_unique_canonical_pages(
        [missing[0]],
        [missing[1]],
        [missing[2]],
        [missing[3]],
    )

    assert dropped == 0
    assert [page.url for page in kept_pages] == ["https://example.com/missing"]
    assert missing[3]["status"] == "analyzed"
