# SERP Paragraph Gap Analysis

This report answers a narrow question:

> For a keyword or keyword cluster, what paragraph-level topics are common across ranking competitor pages, and which of those topics are missing or weak on our best matching page?

It does not prove why Google ranks a competitor higher. Rankings also depend on backlinks, internal links, brand, topical authority, freshness, technical health, SERP layout, personalization, and provider sampling. Treat the output as content evidence from the current SERP, not as a causal ranking diagnosis.

## When To Use It

Use this analysis for business-relevant keywords where you already know the page should compete, or where you are deciding whether to create a new page.

Do not use it for every keyword that brings traffic. Many keywords can be irrelevant to the product, too broad, localized, branded, informational-only, or attached to pages that should not become commercial targets.

Good candidates:

- product, service, use-case, and comparison queries
- keywords with impressions or traffic potential
- keywords where the site ranks below stronger competitors
- clusters with clear business value
- clusters where you have, or can create, a target page

Poor candidates:

- foreign-language traffic that is not part of the current strategy
- glossary, entertainment, or curiosity keywords with weak product fit
- competitor brand terms unless there is a deliberate comparison strategy
- keywords where the SERP intent is forums, videos, jobs, news, or local listings
- keywords whose best target page would not help the business

## Input Format

Run the audit with a competitive TSV:

```text
# AI automation platforms
AI automation platforms	best ai automation platforms	https://competitor.example/best-ai-platforms	1
AI automation platforms	best ai automation platforms	https://competitor.example/ai-automation-tools	2

# AI voice agents
AI voice agents	best ai voice agents	https://competitor.example/voice-agents	1
```

Supported row formats:

- `query<TAB>competitor_url`
- `cluster<TAB>query<TAB>competitor_url<TAB>rank`
- `cluster | query | rank | competitor_url`

Rows with the same cluster/query are grouped into one SERP paragraph model.

## How The Analysis Works

For each query group:

1. Find the best matching page on your site using query/page embeddings, with conservative demotion for low-intent pages such as author, tag, category, glossary, and localized pages.
2. Fetch each competitor URL.
3. Extract competitor paragraphs.
4. Embed competitor paragraphs in the same vector space as your site paragraphs.
5. Cluster competitor paragraphs into SERP topics.
6. Compare each SERP topic against paragraphs on your selected page.
7. Classify each topic as covered, partial, or missing.
8. Score priority based on competitor prevalence and coverage weakness.

## How To Read The Report

Start with the selected page.

If the report says the matching page is an author page, unrelated blog post, localized page, glossary page, or tool page that should not rank for the query, do not edit that page blindly. The correct action is to choose or create the real target page and rerun the analysis.

Then read topics in this order:

- `missing`: competitors cover this topic, but your selected page has no close paragraph.
- `partial`: your page has a related paragraph, but it is weaker or less direct than competitor coverage.
- `covered`: your page already has a close paragraph-level match.

Priority means:

- `critical`: missing topic seen across most competitors.
- `high`: missing or partial topic seen across many competitors.
- `medium`: topic may matter, but competitor evidence is weaker.
- `covered`: usually no content expansion needed.

## Editorial Action Plan

For each keyword cluster:

1. Confirm the target URL is the page you actually want to rank.
2. Review missing high-priority topics.
3. Decide which missing topics are relevant to your product and buying journey.
4. Add new sections only for relevant missing topics.
5. Improve existing paragraphs for partial topics.
6. Ignore irrelevant competitor topics instead of copying them.
7. Add concrete evidence: examples, workflows, pricing detail, comparisons, screenshots, tables, FAQs, citations, or implementation steps.
8. Add internal links from related pages to the target page.
9. Rerun the analysis.
10. Track ranking and impression changes later in GSC or Ahrefs.

## What To Do With Missing Topics

A missing topic should become a new section only when it is relevant.

Example:

```text
Topic: pricing, free plan, monthly cost
Coverage: missing
Seen on: 5/5 competitors
Priority: high
```

Possible action:

- Add a section titled `How much does an AI automation platform cost?`
- Cover free vs paid plans, seat limits, task volume, implementation cost, and when your product is more cost-effective.
- Add a small table if the query has buying/comparison intent.

## What To Do With Partial Topics

Partial means the page already touches the topic. Do not create a duplicate section by default.

Improve the closest paragraph:

- make the heading clearer
- add concrete examples
- add product-specific detail
- include a decision criterion
- add screenshots or tables
- link to a deeper supporting page

Example:

```text
Topic: visual workflow builder
Coverage: partial
Seen on: 5/5 competitors
```

Possible action:

- Expand the existing workflow-builder paragraph.
- Explain triggers, nodes, integrations, testing, human review, and deployment.
- Add an internal link to a workflow template or product documentation page.

## What To Ignore

Do not add every topic competitors mention.

Ignore a topic when:

- it does not match your product
- it attracts the wrong audience
- it belongs on another page
- it is only present on one weak competitor page
- it would dilute the page's intent
- it is a SERP artifact from a different page type

The right question is not:

> What are competitors saying that we can copy?

The right question is:

> Which search-intent requirements are common in winning pages, relevant to our product, and missing or weak on our target page?

## Keyword Selection With GSC, Ahrefs, Google Ads, And DataForSEO

Use Ahrefs, GSC, Google Ads, and DataForSEO for different jobs.

Ahrefs or GSC should decide which organic keywords deserve analysis:

- current position
- impressions or traffic potential
- matched URL
- business relevance
- cluster size
- ranking gap
- trend or lost traffic

When you want to compare sources instead of choosing one source, run with
`--search-provider all`. The report keeps the provider label on every
keyword and plots all sources in the same semantic space as titles, H1-H4
headings, paragraphs, and link titles. Use the filters to isolate GSC,
Google Ads, Ahrefs, DataForSEO, or site entities.

Google Ads should decide which paid search terms deserve analysis when product/service relevance matters more than raw organic traffic:

- highest spend
- clicks and impressions
- conversion volume/value where available
- campaign and ad group context
- business relevance confirmed by product seeds

See [Google Ads Keyword Source](google-ads-keyword-source.md) for exact setup instructions.

DataForSEO should fetch current SERP URLs after filtering:

- top organic URLs
- current titles/descriptions
- rank position
- SERP item type
- location/language-specific results

To avoid wasting API credits:

1. Start from Ahrefs/GSC keywords, or from Google Ads search terms when paid spend is the relevance signal.
2. Remove irrelevant languages, branded noise, glossary-only terms, and non-business topics.
3. Keep only product, service, use-case, and comparison clusters.
4. Require a relevant target URL or planned target URL.
5. Cap the run, for example: top 3 clusters, top 3 keywords per cluster, top 5 URLs per keyword.
6. Cache SERP results and competitor fetches.

## Auto Mode

Manual TSV input is safest when you already know the exact keywords and competitor URLs. Auto mode is useful when you want the audit to choose relevant opportunities from Ahrefs/GSC/Google Ads/DataForSEO search data and then fetch live SERP URLs from DataForSEO.

Run:

```bash
site-audit run flowhunt.io \
  --competitive-auto \
  --competitive-auto-product-seed "AI workflow automation" \
  --competitive-auto-product-seed "AI agents" \
  --competitive-auto-clusters 3 \
  --competitive-auto-keywords-per-cluster 1 \
  --competitive-auto-results-per-keyword 5
```

Auto mode does this:

1. Reads keywords from the selected search provider payload. With Google Ads, these are paid search terms sorted by spend.
2. Rejects non-Latin keywords by default.
3. Rejects low-intent pages such as author, tag, category, glossary, and localized pages.
4. Scores business relevance using commercial modifiers, intent labels, page type, demand, and optional product seed phrases.
5. Selects the highest-priority clusters and keywords within your caps.
6. Calls DataForSEO live SERP only for those selected keywords.
7. Runs paragraph gap analysis against the resulting competitor URLs.

Recommended cost controls:

- keep `--competitive-auto-clusters` low, for example `3` to `10`
- keep `--competitive-auto-keywords-per-cluster` at `1` or `2`
- keep `--competitive-auto-results-per-keyword` at `5`
- use product seeds for every run
- use `--competitive-auto-refresh-serp` only when you really need fresh SERPs

If your business intentionally targets non-Latin keywords, add:

```bash
--competitive-auto-allow-nonlatin
```

If the selected keywords are wrong, tighten product seeds or raise:

```bash
--competitive-auto-min-relevance
```

## Recommended User Workflow

For a content strategist:

```text
1. Pick a relevant keyword cluster.
2. Confirm or assign the target page.
3. Run SERP paragraph gap analysis.
4. Mark each missing/partial topic as add, improve, ignore, or move.
5. Produce a content brief.
6. Edit the page.
7. Add internal links.
8. Rerun the analysis.
9. Monitor rankings and impressions.
```

For a developer building automation:

```text
1. Build keyword relevance filters before DataForSEO calls.
2. Let users define allowed URL patterns and product seed terms.
3. Fetch SERP URLs only for selected keywords.
4. Group competitor pages by cluster.
5. Generate paragraph gap recommendations.
6. Save ignored topics so they do not reappear every run.
```
