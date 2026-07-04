"""Technical SEO issue taxonomy.

The catalog mirrors the Ahrefs-style project issue list so the audit report can
show every planned technical SEO detector, even while implementation happens
incrementally.
"""

from __future__ import annotations

import re


_RAW_ISSUES = """
Audit alerts||Automatically include new Error-level issues|Error|Always
Internal pages||404 page|Error|Always
Internal pages||4XX page|Error|Always
Internal pages||500 page|Error|Always
Internal pages||5XX page|Error|Always
Internal pages||Timed out|Error|Always
Internal pages||HTTPS/HTTP mixed content|Warning|Always
Indexability||Canonical points to 4XX|Error|Always
Indexability||Canonical points to 5XX|Error|Always
Indexability||Canonical points to redirect|Error|Always
Indexability||Page size exceeds Googlebot's 2 MB crawl limit|Error|Always
Indexability||Nofollow in HTML and HTTP header|Warning|Always
Indexability||Nofollow page|Warning|Always
Indexability||Noindex in HTML and HTTP header|Warning|Always
Indexability||Noindex page|Warning|Always
Indexability||Non-canonical page specified as canonical one|Warning|Always
Indexability||Canonical from HTTP to HTTPS|Notice|Always
Indexability||Canonical from HTTPS to HTTP|Notice|Always
Indexability||Canonical URL changed|Notice|Always
Indexability||Indexable page became non-indexable|Notice|Always
Indexability||Noindex and nofollow page|Notice|Always
Indexability||Noindex follow page|Notice|Always
Indexability||Noindex page became indexable|Notice|Always
Links|Indexable|Canonical URL has no incoming internal links|Error|Always
Links|Indexable|HTTPS page has internal links to HTTP|Error|Always
Links|Indexable|Orphan page (has no incoming internal links)|Error|Always
Links|Indexable|Page has links to broken page|Error|Always
Links|Indexable|Page has no outgoing links|Error|Always
Links|Indexable|Page has links to redirect|Warning|Always
Links|Indexable|Page has nofollow incoming internal links only|Warning|Always
Links|Indexable|Redirected page has no incoming internal links|Warning|Always
Links|Indexable|HTTP page has internal links to HTTPS|Notice|Always
Links|Indexable|Page has nofollow and dofollow incoming internal links|Notice|Always
Links|Indexable|Page has nofollow outgoing internal links|Notice|Always
Links|Indexable|Page has only one dofollow incoming internal link|Notice|Always
Links|Not indexable|HTTPS page has internal links to HTTP|Warning|Always
Links|Not indexable|Orphan page (has no incoming internal links)|Warning|Always
Links|Not indexable|Page has links to broken page|Warning|Always
Links|Not indexable|Page has no outgoing links|Warning|Always
Links|Not indexable|HTTP page has internal links to HTTPS|Notice|Always
Links|Not indexable|Page has links to redirect|Notice|Always
Links|Not indexable|Page has nofollow and dofollow incoming internal links|Notice|Always
Links|Not indexable|Page has nofollow incoming internal links only|Notice|Always
Links|Not indexable|Page has nofollow outgoing internal links|Notice|Always
Links|Not indexable|Page has only one dofollow incoming internal link|Notice|Always
Links|Not indexable|Redirected page has no incoming internal links|Notice|Always
Redirects||Broken redirect|Error|Always
Redirects||Redirect chain too long|Error|Always
Redirects||Redirect loop|Error|Always
Redirects||302 redirect|Warning|Always
Redirects||3XX redirect|Warning|Always
Redirects||HTTPS to HTTP redirect|Warning|Always
Redirects||HTTP to HTTPS redirect|Notice|Always
Redirects||Meta refresh redirect|Notice|Always
Redirects||Redirect chain|Notice|Always
Redirects||Redirect target changed|Notice|Always
Content|Indexable|Multiple meta description tags|Error|Always
Content|Indexable|Multiple title tags|Error|Always
Content|Indexable|Title tag missing or empty|Error|Always
Content|Indexable|H1 tag missing or empty|Warning|Always
Content|Indexable|Low word count|Warning|Always
Content|Indexable|Meta description tag missing or empty|Warning|Always
Content|Indexable|Meta description too long|Warning|Always
Content|Indexable|Meta description too short|Warning|Always
Content|Indexable|Title too long|Warning|Always
Content|Indexable|Title too short|Warning|Always
Content|Indexable|H1 tag changed|Notice|Always
Content|Indexable|Meta description changed|Notice|Always
Content|Indexable|Multiple H1 tags|Notice|Always
Content|Indexable|Page and SERP titles do not match|Notice|Always
Content|Indexable|Pages have high AI content levels|Notice|Always
Content|Indexable|SERP title changed|Notice|Always
Content|Indexable|Title tag changed|Notice|Always
Content|Indexable|Word count changed|Notice|Always
Content|Not indexable|Meta description tag missing or empty|Warning|Always
Content|Not indexable|Multiple meta description tags|Warning|Always
Content|Not indexable|Multiple title tags|Warning|Always
Content|Not indexable|Title tag missing or empty|Warning|Always
Content|Not indexable|H1 tag missing or empty|Notice|Always
Content|Not indexable|Low word count|Notice|Always
Content|Not indexable|Meta description too long|Notice|Always
Content|Not indexable|Meta description too short|Notice|Always
Content|Not indexable|Multiple H1 tags|Notice|Always
Content|Not indexable|Title too long|Notice|Always
Content|Not indexable|Title too short|Notice|Always
Social tags||Open Graph tags incomplete|Warning|Always
Social tags||Open Graph URL not matching canonical|Warning|Always
Social tags||X (Twitter) card incomplete|Warning|Always
Social tags||Open Graph tags missing|Notice|Always
Social tags||X (Twitter) card missing|Notice|Always
Duplicates||Duplicate pages without canonical|Error|Always
Localization||Hreflang and HTML lang mismatch|Error|Always
Localization||Hreflang annotation invalid|Error|Always
Localization||Hreflang to non-canonical|Error|Always
Localization||Hreflang to redirect or broken page|Error|Always
Localization||HTML lang attribute invalid|Error|Always
Localization||Missing reciprocal hreflang (no return-tag)|Error|Always
Localization||More than one page for same language in hreflang|Error|Always
Localization||Page referenced for more than one language in hreflang|Error|Always
Localization||Hreflang defined but HTML lang missing|Warning|Always
Localization||HTML lang attribute missing|Warning|Always
Localization||Self-reference hreflang annotation missing|Warning|Always
Localization||Not all pages from hreflang group were crawled|Notice|Always
Localization||X-default hreflang annotation missing|Notice|Always
Usability and performance||Content is not sized correctly|Warning|Always
Usability and performance||Document uses plugins|Warning|Always
Usability and performance||Font size too small|Warning|Always
Usability and performance||HTML file size too large|Warning|Always
Usability and performance||Not compressed|Warning|Always
Usability and performance||Page stopped passing CWV requirements|Warning|Always
Usability and performance||Pages with poor CLS|Warning|Always
Usability and performance||Pages with poor FID|Warning|Always
Usability and performance||Pages with poor INP|Warning|Always
Usability and performance||Pages with poor LCP|Warning|Always
Usability and performance||Slow page|Warning|Always
Usability and performance||Tap targets too small or too close together|Warning|Always
Usability and performance||Viewport not set|Warning|Always
Images||Image broken|Error|Always
Images||Image file size too large|Error|Always
Images||Page has broken image|Error|Always
Images||HTTPS page links to HTTP image|Warning|Always
Images||Image redirects|Warning|Always
Images||Missing alt text|Warning|Always
Images||Page has redirected image|Warning|Always
JavaScript||JavaScript broken|Error|Always
JavaScript||Page has broken JavaScript|Error|Always
JavaScript||HTTPS page links to HTTP JavaScript|Warning|Always
JavaScript||JavaScript redirects|Warning|Always
JavaScript||Page has redirected JavaScript|Warning|Always
CSS||CSS broken|Warning|Always
CSS||CSS file size too large|Warning|Always
CSS||CSS redirects|Warning|Always
CSS||HTTPS page links to HTTP CSS|Warning|Always
CSS||Page has broken CSS|Warning|Always
CSS||Page has redirected CSS|Warning|Always
Sitemaps||3XX redirect in sitemap|Error|Always
Sitemaps||4XX page in sitemap|Error|Always
Sitemaps||5XX page in sitemap|Error|Always
Sitemaps||Noindex page in sitemap|Error|Always
Sitemaps||Non-canonical page in sitemap|Error|Always
Sitemaps||Page from sitemap timed out|Error|Always
Sitemaps||Sitemap has syntax error|Error|Always
Sitemaps||Sitemap is not accessible|Error|Always
Sitemaps||Sitemap larger than 50MB|Error|Always
Sitemaps||Sitemap with over 50K URLs|Error|Always
Sitemaps||Sitemap in the wrong format|Warning|Always
Sitemaps||Sitemap includes URLs out of its scope|Warning|Always
Sitemaps||Indexable page not in sitemap|Notice|Always
Sitemaps||No. of URLs in sitemap decreased|Notice|Always
Sitemaps||Page in multiple sitemaps|Notice|Always
Sitemaps||Pages added to sitemaps|Notice|Always
Sitemaps||Pages removed from sitemaps|Notice|Always
External pages||External 3XX redirect|Notice|Always
External pages||External 4XX|Notice|Always
External pages||External 5XX|Notice|Always
External pages||External time out|Notice|Always
Other||3XX page receives organic traffic|Error|Always
Other||403 page receives organic traffic|Error|Always
Other||4XX page receives organic traffic|Error|Always
Other||Double slash in URL|Error|Always
Other||Noindex page receives organic traffic|Error|Always
Other||Robots.txt has syntax error|Error|Always
Other||Robots.txt has too many redirects or redirect loop|Error|Always
Other||Robots.txt is not accessible|Error|Always
Other||Robots.txt changed|Warning|Always
Other||More than three parameters in URL|Notice|Always
Other||No. of referring domains dropped|Notice|Always
Other||Non-canonical page receives organic traffic|Notice|Always
Other||Organic traffic dropped|Notice|Always
Other||Pages dropped from Top 10|Notice|Always
Other||Pages to submit to IndexNow|Notice|Always
Other||Robots.txt rules disallow to crawl|Notice|Always
Other||Structured data has Google rich results validation error|Notice|Always
Other||Structured data has schema.org validation error|Notice|Always
""".strip()

_IMPORTANCE_TO_SEVERITY = {
    "Error": "high",
    "Warning": "medium",
    "Notice": "low",
}


def issue_key(name: str, scope: str = "") -> str:
    base = f"{scope} {name}".strip().lower()
    base = base.replace("x (twitter)", "twitter").replace("https/http", "https http")
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base


def _build_catalog() -> list[dict]:
    rows: list[dict] = []
    for line in _RAW_ISSUES.splitlines():
        category, scope, name, importance, sensitivity = line.split("|")
        rows.append(
            {
                "key": issue_key(name, scope),
                "name": name,
                "category": category,
                "scope": scope,
                "importance": importance,
                "severity": _IMPORTANCE_TO_SEVERITY[importance],
                "sensitivity": sensitivity,
                "implemented": False,
            }
        )
    return rows


TECHNICAL_ISSUE_CATALOG = _build_catalog()
TECHNICAL_ISSUE_BY_KEY = {row["key"]: row for row in TECHNICAL_ISSUE_CATALOG}
