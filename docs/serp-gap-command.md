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
  --keyword-source file \
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
projects/<domain>/cache/
  serp_gap/
  ahrefs/
```

The command should not create a separate cache tree outside the audited domain
project. For example, a LiveAgent run uses:

```text
projects/www.liveagent.com/cache/
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
  --keyword-source file \
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

Use `--keyword-source file` when you pass `--keyword` or
`--keywords-file` and want those explicit keywords to drive the run.

To expand explicit keywords with SERP-discovered related questions and
searches, enable:

```bash
site-audit serp-gap example.com \
  --url https://www.example.com/live-chat-software/ \
  --keyword-source file \
  --keyword "live chat software" \
  --include-serp-keyword-suggestions \
  --max-serp-keyword-suggestions 4
```

This adds People Also Ask and People Also Search suggestions from the SERP
payload. Those suggestions are useful for content coverage, but they often do
not have demand metrics unless GSC or Ahrefs has an exact matching keyword row.

## Providers

Supported SERP providers:

```text
dataforseo
serper
auto
```

`auto` uses Serper when `SERPER_API_KEY` or `SERPER_DEV_API_KEY` is set; otherwise it falls back to DataForSEO when credentials are available.

For DataForSEO, `--country` may be a numeric location code such as `2840` for United States. Use `--language en` for English SERPs.

## Ahrefs Metrics

Use Ahrefs when you want keyword traffic and volume context in the SERP gap
report:

```bash
site-audit serp-gap www.example.com \
  --url https://www.example.com/live-chat-software/ \
  --keyword-source file \
  --keyword "live chat software" \
  --provider dataforseo \
  --country 2840 \
  --language en \
  --use-ahrefs-metrics
```

Ahrefs enrichment is cache-first. It reuses compatible snapshots from:

```text
projects/<domain>/cache/ahrefs/
```

Useful Ahrefs flags:

```text
--use-ahrefs-metrics       attach Ahrefs position, traffic, and volume when keywords match
--ahrefs-refresh           ignore cached Ahrefs snapshots and fetch fresh data
--ahrefs-date YYYY-MM-DD   request a specific Ahrefs report date
--ahrefs-country us        optional Ahrefs country code, lowercase is safest
--ahrefs-mode subdomains   target mode: exact, prefix, domain, or subdomains
--ahrefs-top-pages-limit   top-pages rows to request
--ahrefs-keywords-limit    organic-keywords rows to request
```

Important: Ahrefs metrics are matched by keyword and URL when possible, then
by exact keyword as a fallback. The report does not invent traffic, clicks, or
impressions for manual keywords or SERP suggestions that do not appear in the
available API data. In that case the keyword metrics table shows
`No API metric match`, while the SERP columns still show observed ranking
positions from the live SERP fetch.

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

The top of the report is the aggregate section for all selected keywords and
all processed URLs. It includes:

- Top-10 URLs across selected keywords
- URL co-ranking graph
- table of every ranking URL, with one keyword row per ranking keyword
- keyword metrics table from available APIs
- all-keyword semantic scatterplot
- keyword frequency analysis and weighted word cloud
- topic traffic impact chart
- aggregate semantic clusters

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

## Keyword And URL Tables

The aggregate `Keyword Metrics From APIs` table shows one row per selected
keyword. Columns include:

```text
Keyword
Keyword source
API metrics source
Analyzed URL
Matched metrics URL
Source position
Impressions
Clicks
Traffic
Volume
SERP URLs
Best SERP
Best ranking URL
```

`Keyword source` tells you why the keyword was selected, for example `manual`,
`gsc`, `ahrefs`, `h1`, `serp_people_also_ask`, or
`serp_people_also_search`.

`API metrics source` tells you where demand metrics came from. If it says
`No API metric match`, the keyword still has SERP evidence but no matching
GSC/Ahrefs/DataForSEO/Google Ads demand row in the available cache/API
payload.

The `Top-10 URLs Across Selected Keywords` section shows each ranking URL with
a nested keyword table instead of tags. That table includes:

```text
Rank
Keyword
Impressions
Clicks
Traffic
Volume
Source position
Source
```

`Rank` is the URL position in the fetched SERP for that keyword. `Source
position` is the selected domain's source/API position for the keyword when
available, such as an Ahrefs or GSC average/best position.

## Aggregate Semantic Map

The aggregate `All Keywords, URLs, and Content` scatterplot places every
selected keyword, processed URL, title, H1-H6 heading, and paragraph into one
shared vector space.

It also adds a demand-weighted keyword centroid:

- the centroid is a synthetic keyword point
- it is weighted by impressions first, then Ahrefs volume, then traffic/clicks
- its tooltip shows summed demand metrics and keyword count
- content near the centroid is semantically aligned with the combined keyword
  demand center

Use the domain and entity checkboxes to hide or show domains and node types.

## Keyword Frequency Analysis

The aggregate keyword frequency panel counts repeated terms and short phrases
from processed content. It is split into:

```text
Title Keywords
H1 Keywords
H2-H6 Keywords
Paragraph Keywords
```

The weighted word cloud uses these weights:

```text
Title:     8
H1:        7
H2:        5
H3:        4
H4-H6:     3
Paragraph: 1
```

Bigger words are repeated more often after applying those weights. Use this
to see whether high-impact page elements emphasize the same terms as the
paragraph body and the SERP competitors.

## Topic Traffic Impact

The topic impact chart estimates which aggregate semantic clusters carry the
most keyword demand. Clusters are sorted by:

```text
traffic, then impressions, then volume, then clicks
```

Treat this as directional. It helps decide which topic clusters deserve
editorial attention first, but it is not a causal ranking model.

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

- keyword points use impressions when available, then Ahrefs volume
- the keyword centroid uses summed impressions or volume
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
  --keyword-source file \
  --keyword "live chat software" \
  --keyword "livechat software" \
  --keyword "livechat" \
  --keywords-per-page 3 \
  --results-per-keyword 5 \
  --provider dataforseo \
  --country 2840 \
  --language en \
  --max-pages 1 \
  --max-competitor-pages 15 \
  --max-paragraphs-per-page 50 \
  --include-serp-keyword-suggestions \
  --max-serp-keyword-suggestions 4 \
  --use-ahrefs-metrics
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

The screenshot shows the report layout with semantic scatterplots,
domain-colored competitor points, keyword diamonds, title/H1 triangles,
header squares, paragraph circles, semantic cluster cards, and topic coverage
summary. Current reports also include the aggregate keyword/API metrics table,
per-URL keyword tables, weighted keyword frequency clouds, a demand-weighted
keyword centroid, and topic traffic impact chart.

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
