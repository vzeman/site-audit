from pathlib import Path

from site_audit.compare import build_payload
from site_audit.extractor import ExtractedPage, extract
from site_audit.media_accessibility import analyze, to_payload


def _page(url: str, media_items: list[dict]) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title="Media page",
        description="",
        body="Useful body content for a media accessibility test page.",
        word_count=120,
        language="en",
        media_items=media_items,
    )


def test_extract_media_accessibility_fields() -> None:
    html = """
    <html><head><title>Media accessibility example</title></head><body>
      <h1>Media accessibility example</h1>
      <p>This page has enough body copy for the extractor fallback to keep it.
      It discusses accessible images, videos, audio transcripts, embedded
      iframes, captions, alternative text, and screen reader semantics in
      enough detail to pass the minimum body threshold during extraction.</p>
      <img src="/hero.jpg">
      <img src="/decorative.png" alt="" role="presentation">
      <a href="/brand"><img src="/logo.png" alt=""></a>
      <video src="/demo.mp4"><track kind="captions" src="/demo.vtt"></video>
      <audio src="/podcast.mp3"></audio>
      <iframe src="/embed"></iframe>
    </body></html>
    """

    page = extract("https://example.com/media", html, max_chars=2000)

    assert page is not None
    assert len(page.media_items) == 6
    assert page.media_items[0]["type"] == "image"
    assert page.media_items[0]["alt_present"] is False
    assert page.media_items[1]["role"] == "presentation"
    assert page.media_items[2]["in_link"] is True
    assert page.media_items[3]["has_captions"] is True
    assert page.media_items[4]["type"] == "audio"
    assert page.media_items[5]["type"] == "iframe"


def test_media_accessibility_payload_flags_media_issues() -> None:
    report = to_payload(analyze([
        _page("https://example.com/a", [
            {"type": "image", "src": "/hero.jpg", "alt_present": False, "alt": ""},
            {"type": "image", "src": "/decorative.png", "alt_present": True, "alt": "", "role": "presentation"},
            {"type": "image", "src": "/brand-logo.png", "alt_present": True, "alt": "brand logo"},
            {"type": "image", "src": "/linked.png", "alt_present": True, "alt": "", "in_link": True},
        ]),
        _page("https://example.com/b", [
            {"type": "video", "src": "/demo.mp4", "has_captions": False},
            {"type": "audio", "src": "/podcast.mp3", "has_transcript_hint": False},
            {"type": "iframe", "src": "/embed", "title": ""},
        ]),
    ]))

    assert report["summary"]["total_pages"] == 2
    assert report["summary"]["pages_with_media"] == 2
    assert report["summary"]["pages_with_issues"] == 2
    assert report["summary"]["images_missing_alt"] == 1
    assert report["summary"]["broken_images"] == 0
    assert report["summary"]["decorative_images"] == 1
    assert report["summary"]["linked_images_empty_alt"] == 1
    assert report["summary"]["images_filename_alt"] == 1
    assert report["summary"]["videos_missing_captions"] == 1
    assert report["summary"]["audio_missing_transcript"] == 1
    assert report["summary"]["iframes_missing_title"] == 1
    assert report["issues_by_type"]["image_missing_alt"] == 1
    assert any(row["type"] == "video" for row in report["media_with_issues"])


def test_media_accessibility_payload_flags_broken_images() -> None:
    report = to_payload(analyze([
        _page("https://example.com/a", [
            {"type": "image", "src": "/broken.jpg", "alt_present": True, "alt": "Broken", "http_status": 404},
            {"type": "image", "src": "/flagged.jpg", "alt_present": True, "alt": "Flagged", "broken": True},
            {"type": "image", "src": "/ok.jpg", "alt_present": True, "alt": "OK", "http_status": 200},
        ]),
    ]))

    assert report["summary"]["broken_images"] == 2
    assert report["issues_by_type"]["image_broken"] == 2
    assert report["per_page"][0]["issues"]["image_broken"] == 2
    broken_rows = [row for row in report["media_with_issues"] if "image_broken" in row["issues"]]
    assert [row["src"] for row in broken_rows] == ["/broken.jpg", "/flagged.jpg"]
    assert broken_rows[0]["http_status"] == 404


def test_compare_leaderboard_includes_media_accessibility_metrics(tmp_path: Path) -> None:
    for domain, issue_share, missing_alt in [
        ("a.example", 0.25, 1),
        ("b.example", 0.75, 3),
    ]:
        report_dir = tmp_path / domain / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "site_metrics.json").write_text(
            '{"domain":"%s","model":"test-model","page_count":4}' % domain,
            encoding="utf-8",
        )
        (report_dir / "pages.json").write_text("[]", encoding="utf-8")
        (report_dir / "media_accessibility.json").write_text(
            (
                '{"summary":{"issue_share":%s,'
                '"images_missing_alt":%d,'
                '"linked_images_empty_alt":2,'
                '"videos_missing_captions":1,'
                '"iframes_missing_title":4}}'
            ) % (issue_share, missing_alt),
            encoding="utf-8",
        )

    payload = build_payload(["a.example", "b.example"], tmp_path)

    rows = {row["domain"]: row for row in payload["leaderboard"]}
    assert rows["a.example"]["media_accessibility_issue_share"] == 0.25
    assert rows["a.example"]["images_missing_alt"] == 1
    assert rows["a.example"]["linked_images_empty_alt"] == 2
    assert rows["b.example"]["media_accessibility_issue_share"] == 0.75
    assert rows["b.example"]["images_missing_alt"] == 3
