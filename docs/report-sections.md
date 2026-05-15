# Report Section Guide

This guide explains how to read each major section in a `site-audit` report and what action a user should take from it.

Use every section as a decision aid, not as proof of causality. SEO and GEO outcomes are influenced by content, internal links, backlinks, brand, freshness, search intent, technical health, and provider sampling. The useful workflow is:

1. Confirm the affected URL is the correct target page.
2. Check whether the issue is relevant to the business.
3. Prioritize high-impact pages and clusters.
4. Make one clear editorial, technical, or linking change.
5. Rerun the audit and monitor GSC/Ahrefs later.

<a id="hist-block"></a>
## Per-page similarity to site centroid

Shows how closely each page aligns with the overall site topic. Very low similarity pages may be off-topic, localized noise, thin utility pages, or unrelated content.

Action: review the low-similarity tail. Keep pages that serve a strategic purpose, move misplaced content to a better section, or deindex/remove pages that dilute topical focus.

<a id="action-plan-block"></a>
## Action plan

Aggregates findings from multiple analyses into a prioritized task list.

Action: start here when you need execution priorities. Validate the suggested URL and issue type, then assign each item to content, technical SEO, internal linking, or product/UX.

<a id="improvement-block"></a>
## GEO pages most in need of editing

Identifies pages with weak content quality, answerability, structural, or topical signals for generative search visibility.

Action: rewrite high-priority pages with clearer answers, better headings, stronger evidence, and more direct intent coverage.

<a id="title-mismatch-block"></a>
## Title does not match content

Finds pages where the title promises a topic that the body does not strongly support.

Action: either rewrite the title to match the page, or add sections that satisfy the title's promise.

<a id="headers-block"></a>
## Header structure

Audits H1-H6 structure, hierarchy, repeated headings, and heading drift.

Action: fix missing/multiple H1s, make H2/H3s descriptive, and ensure headings form a logical outline of the page.

<a id="heading-impact-block"></a>
## Heading impact map

Connects headings to search demand, paragraph support, and page context.

Action: rename headings that carry demand but are vague, and add paragraphs under headings that currently lack substance.

<a id="clusters-block"></a>
## Topic clusters

Groups pages by semantic similarity and auto-labels the dominant topics.

Action: use clusters to understand site architecture. Strong clusters can become hubs; weak or mixed clusters may need better grouping, internal links, or clearer page targeting.

<a id="treemap-block"></a>
## Topic mass treemap

Visualizes how much content exists in each topic cluster.

Action: compare content mass with business importance. Large low-value clusters may be bloat; small high-value clusters may need expansion.

<a id="ahrefs-block"></a>
## Organic search demand overlay

Shows Ahrefs, GSC, or DataForSEO search demand mapped onto audited pages and clusters.

Action: prioritize pages with meaningful demand, weak rankings, or high opportunity. Filter out irrelevant traffic before making content decisions.

<a id="best-pages-block"></a>
## Best page reverse engineering

Explains what successful pages on the site have in common.

Action: turn recurring strengths into a template checklist for new or refreshed pages.

<a id="paragraph-impact-block"></a>
## Winning paragraphs

Surfaces paragraphs that appear to carry important topic or keyword value.

Action: preserve these paragraphs during edits, improve their evidence, and add internal links around them.

<a id="weak-paragraphs-block"></a>
## Weak paragraphs and content decay

Flags duplicated, stale, generic, off-topic, unsupported, or poorly linked paragraphs.

Action: rewrite, merge, remove, cite, move, or link the flagged paragraphs depending on the recommended action.

<a id="semantic-ablation-block"></a>
## Semantic ablation

Estimates whether removing a paragraph would hurt or improve page-topic alignment.

Action: protect topic carriers. Consider removing or moving noise candidates if they do not serve users or conversions.

<a id="keyword-attribution-block"></a>
## Keyword traffic attribution

Maps ranking keywords to the page paragraph that best supports them.

Action: strengthen paragraphs that support valuable keywords, and create missing paragraphs where demand has no strong on-page support.

<a id="heatmap-block"></a>
## Cluster overlap heatmap

Shows which topic clusters are semantically close or overlapping.

Action: use overlap to find consolidation, hub, and internal-link opportunities. High overlap can also indicate cannibalization risk.

<a id="coverage-block"></a>
## Keyword coverage

Checks whether target queries have a strong matching page.

Action: create pages for gaps, differentiate cannibalized queries, and improve pages with weak coverage.

<a id="indexability-block"></a>
## Indexability funnel

Summarizes crawlability, noindex, canonical, robots, and skipped-page signals.

Action: fix accidental noindex/canonical/robots issues before content work. Content improvements do not matter if the page cannot be indexed.

<a id="structured-data-block"></a>
## Structured data health

Audits JSON-LD/schema coverage and opportunities.

Action: add schema only where visible page content supports it. Prioritize FAQ, Article, Product, HowTo, or Organization schema when relevant.

<a id="trust-signals-block"></a>
## E-E-A-T and trust signals

Looks for evidence, author, citations, dates, company, policy, and credibility signals.

Action: add author/reviewer details, dates, sources, examples, policies, and proof where users need confidence.

<a id="conversion-balance-block"></a>
## Conversion vs SEO balance

Compares informational usefulness with conversion pressure.

Action: reduce premature CTAs on informational pages, or add stronger conversion paths to commercial pages that already satisfy intent.

<a id="metadata-quality-block"></a>
## SERP metadata quality

Audits titles, descriptions, canonicals, and social metadata.

Action: rewrite metadata for clarity, intent match, and uniqueness. Fix missing or duplicated descriptions and bad canonicals.

<a id="media-accessibility-block"></a>
## Media accessibility

Checks images, videos, iframes, alt text, captions, and media metadata.

Action: add meaningful alt text, captions/transcripts, dimensions, and accessible labels where media supports the page.

<a id="page-types-block"></a>
## Page types and templates

Classifies pages into templates such as article, FAQ, product, tool, category, and contact.

Action: verify important keywords are mapped to the right page type. If the SERP wants comparison pages, a pure product page may not be enough.

<a id="template-patterns-block"></a>
## Template success patterns

Identifies reusable page/template traits associated with stronger outcomes.

Action: apply proven patterns to similar weak pages, but avoid blindly copying patterns across different intents.

<a id="entities-block"></a>
## Entities and topical authority

Extracts named entities and recurring concepts across the site.

Action: build topical authority around core entities, and remove or isolate entity clusters that are irrelevant to the business.

<a id="entity-coverage-block"></a>
## Entity coverage score

Compares each page's entity coverage against its topic context.

Action: add missing core entities when they help users understand the topic. Do not stuff entities that do not belong naturally.

<a id="information-gain-block"></a>
## Information gain and originality

Estimates whether pages add distinctive information or mostly repeat common content.

Action: add first-hand examples, workflows, data, screenshots, opinions, comparisons, or implementation details.

<a id="freshness-block"></a>
## Content freshness

Checks dates and stale content signals.

Action: update pages where freshness matters to the query. Add visible modified dates only when the content was truly reviewed.

<a id="freshness-impact-block"></a>
## Freshness impact

Connects freshness issues to sections or pages where search demand may be affected.

Action: prioritize refreshes where stale content overlaps with valuable traffic or rankings.

<a id="performance-block"></a>
## Lightweight performance signals

Reports simple page-weight and render-blocking indicators.

Action: use this as a triage signal. Run Lighthouse/WebPageTest before major performance engineering.

<a id="performance-explainer-block"></a>
## Performance explainer

Looks for relationships between performance features and observed outcomes.

Action: treat findings as correlation. Prioritize performance fixes that affect templates, critical pages, or user experience.

<a id="conversion-block"></a>
## Conversion and CTA signals

Audits CTAs, forms, contacts, and conversion paths.

Action: ensure commercial pages have clear next steps and informational pages have appropriate soft conversions.

<a id="answerability-block"></a>
## GEO answer-ability score

Scores whether pages are likely to answer questions directly and extractably.

Action: add concise answer blocks, question headings, lists, tables, definitions, and evidence.

<a id="answer-blocks-block"></a>
## Answer blocks and snippet candidates

Finds paragraphs that could serve as answer/snippet blocks.

Action: improve weak answer blocks with direct wording, specificity, and source-backed facts.

<a id="cannibalization-block"></a>
## Content cannibalization by intent

Finds multiple pages competing for the same or similar intent.

Action: merge, canonicalize, retarget, or differentiate pages. Avoid having several weak pages chase the same query.

<a id="duplicate-fragments-block"></a>
## Duplicate strong fragments

Finds repeated content fragments that may dilute pages.

Action: keep necessary template text, but remove or rewrite boilerplate that appears as main content across many pages.

<a id="linkgraph-block"></a>
## Top authority and orphan pages

Shows internal PageRank-style authority pages and pages with no inbound links.

Action: link important orphan pages from relevant hubs, and use authority pages to distribute internal equity.

<a id="linkflow-block"></a>
## Internal link equity flow

Visualizes how internal authority flows between pages or sections.

Action: strengthen flows into high-value pages and reduce wasted links to low-value destinations.

<a id="traffic-pagerank-block"></a>
## Traffic-weighted PageRank

Compares search demand with internal link support.

Action: prioritize high-demand pages with weak internal authority. Add contextual links from relevant strong pages.

<a id="high-demand-link-block"></a>
## High-demand low-link pages

Lists pages with search opportunity but poor internal-link support.

Action: add links from hubs, related articles, templates, and high-authority pages.

<a id="hub-bottleneck-block"></a>
## Hub and bottleneck pages

Identifies pages that connect or block important internal-link paths.

Action: strengthen useful hubs and reduce dependence on fragile bottlenecks.

<a id="link-removal-block"></a>
## Internal link removal simulation

Estimates risk from removing internal links.

Action: avoid removing links marked as important unless you replace them with better contextual links.

<a id="link-addition-block"></a>
## Suggested internal links

Recommends high-similarity source-target links that do not currently exist.

Action: add only links that are useful to readers and naturally fit the paragraph context.

<a id="linkbuilding-block"></a>
## Linkbuilding overview

Summarizes external backlink or outbound-link-oriented signals where available.

Action: use as a directional view of link health, anchor quality, and external authority gaps.

<a id="link-counts-block"></a>
## Internal links per page

Shows inbound and outbound internal-link distribution.

Action: fix pages with too few meaningful inbound links or excessive low-value outbound links.

<a id="para-density-block"></a>
## Paragraph link density

Measures how many links appear inside paragraph text.

Action: add contextual links to isolated paragraphs, and reduce spammy/link-stuffed paragraphs.

<a id="para-links-block"></a>
## In-paragraph link recommendations

Suggests paragraph-level internal links.

Action: add links where the source paragraph genuinely introduces the target topic.

<a id="wrong-home-block"></a>
## Paragraphs that belong on a different page

Finds paragraphs whose semantic home appears to be another page.

Action: move, rewrite, or link these paragraphs so each page stays focused.

<a id="para-clusters-block"></a>
## Paragraph topic clusters

Groups paragraphs across the site by topic.

Action: identify duplicated themes, missing hubs, scattered topics, and paragraphs that should be consolidated.

<a id="fanout-block"></a>
## Query to paragraph fanout

Shows whether a query is supported by one focused paragraph, scattered across many, or missing.

Action: consolidate scattered support into a clearer section, or create a paragraph for gaps.

<a id="competitive-block"></a>
## Competitor comparison

Compares your target page with competitor URLs at structural and paragraph-topic levels.

Action: confirm the target page, add relevant missing topics, improve partial topics, ignore irrelevant competitor topics, and rerun. If auto mode was used, first inspect the auto-selected keyword table and reject any keyword that does not match the product strategy. See also [SERP Paragraph Gap Analysis](serp-paragraph-gap-analysis.md).

<a id="hits-block"></a>
## HITS hubs and authorities

Shows pages that behave as hubs or authorities in the internal link graph.

Action: use strong hubs to link into strategic pages, and improve authority pages that deserve more support.

<a id="depth-block"></a>
## Buried pages

Lists pages far from the homepage by click depth.

Action: reduce depth for important pages using navigation, hubs, breadcrumbs, or contextual links.

<a id="cluster-auth-block"></a>
## Topic-cluster authorities

Shows authority pages inside each topic cluster.

Action: use cluster authorities as hubs and ensure they link to related conversion and support pages.

<a id="anchor-block"></a>
## Anchor-text analysis

Audits anchor text descriptiveness and relevance.

Action: replace vague anchors such as “read more” with descriptive anchors that match the target page.

<a id="contextual-link-block"></a>
## Contextual link impact

Scores the usefulness of links inside meaningful content areas.

Action: prioritize main-content links over template/nav links when supporting important pages.

<a id="internal-link-patterns-block"></a>
## Internal link pattern library

Finds reusable internal-linking patterns and weak pages that need them.

Action: apply successful patterns to similar pages, especially within the same page type or topic cluster.

<a id="external-block"></a>
## External links and citation density

Shows most-cited external domains and citation density.

Action: cite authoritative sources where claims need support. Remove low-quality or irrelevant external links.

<a id="broken-block"></a>
## Broken outbound links

Lists outbound links that failed checks when external checking is enabled.

Action: replace, remove, or update broken citations.
