from pathlib import Path
import re

from site_audit.report import _copy_report_docs


def test_copy_report_docs_includes_serp_paragraph_gap_guide(tmp_path: Path) -> None:
    _copy_report_docs(tmp_path)

    guide = tmp_path / "serp-paragraph-gap-analysis.md"
    sections = tmp_path / "report-sections.md"
    assert guide.is_file()
    assert sections.is_file()
    text = guide.read_text(encoding="utf-8")
    assert "SERP Paragraph Gap Analysis" in text
    assert "Editorial Action Plan" in text
    section_text = sections.read_text(encoding="utf-8")
    assert "Report Section Guide" in section_text
    assert '<a id="competitive-block"></a>' in section_text


def test_report_template_links_serp_guide_to_github() -> None:
    template = Path("ui/index.html").read_text(encoding="utf-8")
    github_url = "https://github.com/vzeman/site-audit/blob/main/docs/serp-paragraph-gap-analysis.md"
    sections_url = "https://github.com/vzeman/site-audit/blob/main/docs/report-sections.md"

    assert github_url in template
    assert sections_url in template
    assert 'How to use this section' in template
    assert 'href="serp-paragraph-gap-analysis.md"' not in template


def test_report_template_has_section_navigation() -> None:
    template = Path("ui/index.html").read_text(encoding="utf-8")

    assert 'aria-label="Report sections"' in template
    assert 'id="report-nav"' in template
    assert "const REPORT_NAV_SECTIONS = [" in template
    assert "function showReportSection" in template
    assert "report-section-hidden" in template
    assert "initReportNavigation();" in template


def test_report_section_doc_links_have_matching_blocks_and_anchors() -> None:
    template = Path("ui/index.html").read_text(encoding="utf-8")
    guide = Path("docs/report-sections.md").read_text(encoding="utf-8")
    doc_blocks = template.split("const SECTION_DOC_BLOCKS = [", 1)[1].split("];", 1)[0]
    block_ids = re.findall(r"'([a-z0-9-]+-block)'", doc_blocks)
    html_ids = set(re.findall(r'id="([^"]+)"', template))
    guide_ids = set(re.findall(r'<a id="([^"]+)"></a>', guide))

    assert block_ids
    assert [block_id for block_id in block_ids if block_id not in html_ids] == []
    assert [block_id for block_id in block_ids if block_id not in guide_ids] == []
