# SERP Gap Command

`site-audit serp-gap` is a separate SERP research workflow for one already-audited domain. It compares selected URLs from your site against the current Google results for chosen or discovered keywords.

It is intentionally not part of `site-audit run` because it can spend SERP API credits, fetch external competitor pages, embed many paragraphs, and take materially longer than a normal domain audit.

## Basic Usage

Run a normal audit first:

```bash
site-audit run www.example.com --search-provider all
```

Then run SERP gap analysis for a page or URL pattern:

```bash
site-audit serp-gap www.example.com \
  --url https://www.example.com/live-chat-software/ \
  --keyword "live chat software" \
  --keyword "livechat software" \
  --keyword "livechat" \
  --keywords-per-page 3 \
  --results-per-keyword 5 \
  --provider dataforseo \
  --country 2840 \
  --language en
```

Output is written to:

```text
projects/<domain>/serp_gap/report/
  index.html
  serp_gap.json
  serp_gap.csv
```

Cache is kept with the rest of the domain cache:

```text
projects/<domain>/cache/serp_gap/
```

## URL Selection

Analyze exact URLs:

```bash
site-audit serp-gap example.com --url https://www.example.com/features/chat/
```

Or select audited URLs by path pattern:

```bash
site-audit serp-gap example.com \
  --url-include "/features/*" \
  --url-exclude "/features/legacy/*"
```

`--url` can analyze a URL even if it was not present in `pages.json`, but the domain must still have an existing base audit.

## Keyword Selection

Use explicit suggested keywords:

```bash
site-audit serp-gap example.com \
  --url https://www.example.com/help-desk-software/ \
  --keyword "helpdesk software" \
  --keyword "help desk ticketing system"
```

Use a TSV file:

```text
url<TAB>keyword
https://www.example.com/help-desk-software/<TAB>helpdesk software
https://www.example.com/live-chat-software/<TAB>live chat software
```

Then run:

```bash
site-audit serp-gap example.com --keywords-file serp-gap-keywords.tsv
```

When search-provider data exists in the base audit, automatic keyword selection can use:

```text
auto
gsc
ahrefs
dataforseo
google_ads
h1
file
```

## Providers

Supported SERP providers:

```text
dataforseo
serper
auto
```

`auto` uses Serper when `SERPER_API_KEY` or `SERPER_DEV_API_KEY` is set; otherwise it falls back to DataForSEO when credentials are available.

For DataForSEO, `--country` may be a numeric location code such as `2840` for United States. Use `--language en` for English SERPs.

## Budget Controls

Use dry-run mode before expensive runs:

```bash
site-audit serp-gap example.com \
  --url-include "/features/*" \
  --dry-run
```

Useful caps:

```text
--max-pages
--keywords-per-page
--results-per-keyword
--max-competitor-pages
--max-paragraphs-per-page
--budget-usd
```

If `--budget-usd` is provided and the estimated uncached SERP cost exceeds it, the command writes a budget-exceeded report and stops before paid work.

## Report Interpretation

Each analyzed page has its own section. Inside each page, every keyword has:

- SERP competitor list
- semantic scatterplot
- semantic cluster summary
- topic relation table
- paragraphs from your page to review

Topic coverage labels:

```text
covered  - your page has a close paragraph-level match
partial  - your page is related but weaker or less direct
missing  - competitors cover the topic and your page has no close match
```

Treat the output as editorial evidence from the current SERP, not as proof of ranking causality.

## Scatterplot Encoding

The scatterplot uses semantic vectors for keywords, page titles, H1s, headers, and paragraphs.

Color:

- each competitor domain gets a stable distinct color
- your page content is green
- keyword anchors use a dedicated keyword color

Shape:

- keyword: diamond
- title and H1: triangle
- headers: square
- paragraphs: circle

Size:

- keyword points use clicks when available
- competitor points use SERP rank, with higher-ranking results drawn larger
- other content points use readable defaults

Interactions:

- wheel to zoom
- drag to pan
- double-click to reset
- use `+`, `-`, and `Reset` chart buttons
- click a dot to open a persistent detail dialog
- press Escape to close open dialogs

The click dialog shows source/type badges, cluster, source, domain, SERP rank, URL, and a text excerpt.

## Common Recipes

Analyze one commercial page with suggested keywords:

```bash
site-audit serp-gap www.liveagent.com \
  --url https://www.liveagent.com/live-chat-software/ \
  --keyword "live chat software" \
  --keyword "livechat software" \
  --keyword "livechat" \
  --keywords-per-page 3 \
  --results-per-keyword 5 \
  --provider dataforseo \
  --country 2840 \
  --language en
```

Example output:

![SERP gap report for LiveAgent live chat software page](images/serp-gap-liveagent-example.png)

This example analyzes `https://www.liveagent.com/live-chat-software/`
against three keyword variants:

```text
live chat software
livechat software
livechat
```

The screenshot shows the page-level report section, keyword-level scatterplot,
domain-colored competitor points, keyword diamonds, title/H1 triangles,
header squares, paragraph circles, semantic cluster cards, and topic coverage
summary.

Analyze a feature directory with conservative caps:

```bash
site-audit serp-gap example.com \
  --url-include "/features/*" \
  --keywords-per-page 2 \
  --results-per-keyword 5 \
  --max-pages 10 \
  --budget-usd 5
```

Refresh current SERPs but reuse competitor cache:

```bash
site-audit serp-gap example.com \
  --url-include "/solutions/*" \
  --refresh-serp
```

Refresh competitor pages too:

```bash
site-audit serp-gap example.com \
  --url-include "/solutions/*" \
  --refresh-serp \
  --refresh-competitors
```
