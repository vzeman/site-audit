# site-audit

Crawl any website, embed every page, and surface near-duplicates,
outliers, topic clusters, GEO citability, and internal-link
recommendations on an interactive D3 scatterplot.

Drop in a domain, get back a self-contained HTML report.

```bash
site-audit run example.com
open projects/example.com/report/index.html
```

---

## Table of contents

- [Quickstart](#quickstart) — install + first run + open the report
- [Common recipes](#common-recipes) — what to type for typical jobs
- [What the report shows](#what-the-report-shows) — section-by-section guide
- [Comparing multiple domains](#comparing-multiple-domains) — `site-audit compare`
- [Project layout on disk](#project-layout-on-disk)
- [All the flags](#all-the-flags)
- [Re-runs and caching](#re-runs-and-caching)
- [Programmatic use](#programmatic-use)
- [Troubleshooting](#troubleshooting)
- [How the metrics work](#how-the-metrics-work)
- [License](#license)

---

## Quickstart

**Requirements**

- Python 3.10+
- ~ 1.5 GB free disk for the embedding model (downloaded once on first run)
- macOS / Linux (Apple Silicon supported; we set the libomp env vars
  needed for faiss + PyTorch coexistence automatically)

**Install**

```bash
git clone https://github.com/vzeman/site-audit.git
cd site-audit
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The `pip install` pulls the heavy ML deps
(`sentence-transformers`, `faiss-cpu`, `umap-learn`, `scikit-learn`).
The first audit run also downloads the default embedding model
(`Alibaba-NLP/gte-multilingual-base`, ~ 600 MB) into
`~/.cache/huggingface/` — subsequent runs use the cached weights.

**Run an audit**

```bash
site-audit run example.com
```

Limit discovery to specific sitemaps or URL patterns when a site exposes
language or section-specific indexes:

```bash
site-audit run flowhunt.io --sitemap-url https://www.flowhunt.io/sitemap.xml --sitemap-only
site-audit run example.com --url-include '/en/' --url-exclude '/private/'
site-audit run example.com --sitemap-include 'blog-sitemap' --sitemap-exclude 'images'
site-audit run postoj.sk --sitemap-lastmod-after 2025-05-04 --sitemap-only
site-audit run example.com --sitemap-lastmod-within-days 365
```

`--sitemap-lastmod-after` and `--sitemap-lastmod-within-days` keep only
URLs with a sitemap `<lastmod>` on or after the cutoff. URLs without
`<lastmod>` are excluded when a date filter is active.

For large sites, disable paragraph-heavy stages when you only need the
page-level report:

```bash
site-audit run flowhunt.io --sitemap-url https://www.flowhunt.io/sitemap.xml \
  --sitemap-only --no-paragraph-links --no-paragraph-clustering --no-paragraph-fanout
```

That:

1. Reads `robots.txt` + every `sitemap.xml` it can find
2. BFS-crawls the site (default cap: 10 000 pages, image / PDF / asset
   URLs filtered at the path level — works even when query strings hide
   the extension, e.g. `/img.jpg?v=42`)
3. **Honours `noindex`** — pages with `<meta name="robots" content="noindex">`
   or an `X-Robots-Tag: noindex` HTTP header are dropped from the
   analysis corpus before embedding (we still consume the page during
   the crawl so its outlinks contribute to discovery)
4. Extracts main-content text from each page, plus the count of
   internal vs external `<a href>` per paragraph and a full anchor
   quality audit (empty anchors, image-only links, missing alt text,
   missing title attributes)
5. Embeds every page with the multilingual sentence-transformer
6. Computes focus / radius / topic clusters / duplicate pairs / outliers
7. Builds the internal link graph (PageRank, HITS, click depth, orphans)
   and exposes per-page in/out-degree for every crawled page
8. Mines target queries from titles + question-form H2s, finds the best
   page for each, flags coverage gaps and cannibalization
9. Scores every page on GEO answer-ability (FAQ schema, question
   headings, lists/tables, dates, statistics, citations)
10. Profiles outbound links (top cited domains, citation density)
11. Embeds + clusters every paragraph (sub-topics that LLMs retrieve at)
    and projects them via UMAP for the scatter
12. **Embeds every header (H1–H6)**, compares to its host page's
    paragraph centroid (drifty headers don't describe their content),
    detects missing/duplicate H1s + level skips, mines header keyword
    frequency by level, projects headers via UMAP for the scatter
13. **Builds the linkbuilding overview** — site-level link health
    (total links, internal:external ratio, average in/out per page),
    anchor quality audit (descriptive vs generic share, empty links,
    image-without-alt rate), and embeds every distinct anchor for the
    anchor-text scatter
14. Detects title-vs-content mismatch and "wrong-home" paragraphs that
    fit a different page's centroid better than their host
15. Computes paragraph link density (links per 100 words) — used both
    as an editorial signal and as a saturation filter on the
    paragraph-link recommender
16. Recommends in-paragraph internal links: source paragraph, target
    page, suggested anchor phrase
17. **Synthesises every analysis above into a prioritised top-100
    "Action plan"** so the report tells you *what to do next*, not
    just what is true
18. Writes JSON/CSV outputs **and** a self-contained HTML report

**Open the report**

```bash
open projects/example.com/report/index.html      # macOS
xdg-open projects/example.com/report/index.html  # Linux
# …or just double-click the file in your file manager.
```

The HTML is fully self-contained — every JSON payload is inlined into
`<script type="application/json">` tags, so you can email the file or
drop it in a private S3 bucket without breaking it.

If you want a live server (e.g. for embedding the viewer in a browser
tab during development), run:

```bash
site-audit serve example.com
# → http://127.0.0.1:8765/
```

### What the report looks like

The single-file HTML report scrolls top-to-bottom through ~25 sections,
each answering a different "is this site healthy at X" question. Most
sections are self-explanatory; here are the visualisations you'll see:

- **Action plan** at the top — a prioritised list of concrete edits
  with `high / medium / low` priority badges, `quick / medium / deep`
  effort badges, and category chips. Filterable.
- **Headline metric cards** — calibrated focus, topic dimension,
  section coherence, page count, outlier / duplicate counts, raw
  focus score, site radius.
- **Semantic Scatterplot** — 2D UMAP of every page (or paragraph, or
  header — switchable). Coloured by section / cluster / drift /
  outliers + duplicates. Pan, zoom, click any dot to open the page.
- **Histogram of per-page similarity to the site centroid** — quickly
  shows whether a site is unimodal (focused) or bimodal (mixed).
- **Cluster treemap + cluster overlap heatmap** — topic mass and
  cannibalisation visualised side by side.
- **Internal links per page (in/out distribution)** — overlapping
  histogram and a scatter of `(in_degree, out_degree)`. Orphans and
  dead-ends are colour-coded.
- **Anchor-text scatter** — every distinct anchor phrase embedded and
  projected. Dot size = usage frequency, colour = internal / external
  / generic.
- **Paragraph link density histogram** with a dashed red line at the
  spammy threshold.
- Sortable / filterable tables for: weakest sections, top outliers,
  duplicate pairs, topic clusters, keyword coverage (gap/cannibalised/
  covered), GEO answer-ability per page, top authority pages, orphan
  pages, suggested internal links, in-paragraph link recommendations,
  wrong-home paragraphs, query → paragraph fanout, HITS authorities /
  hubs, buried pages, anchor analysis, external link profile, broken
  outbound, header structural issues, drifty headers, header keyword
  frequency.

Every section has its own "what is good / what is bad" interpretive
text built into the HTML.

The cross-domain comparison view (`site-audit compare …`) renders a
sortable tabbed leaderboard, a combined-corpus semantic scatter
coloured by domain, and overlay distribution charts (in-degree,
out-degree, GEO score, paragraph density) so you can see shape
differences across competitors at a glance.

---

## Common recipes

### Audit a small site quickly

```bash
site-audit run example.com --max-pages 500 --workers 16
```

### Audit a multi-language site (subdomains too)

```bash
site-audit run example.com --follow-subdomains --max-pages 10000
```

### Use a custom set of target queries instead of auto-mined ones

Create `my_queries.txt` with one query per line:

```
how to install <product>
<product> vs <competitor>
pricing for <product>
how does <product> handle GDPR
```

Then:

```bash
site-audit run example.com --queries-file my_queries.txt
```

The keyword-coverage section in the report will use *your* queries
instead of titles + H2s. This is by far the highest-leverage flag for
serious GEO work — the questions your customers actually ask
ChatGPT / Perplexity rarely match the H2s on your site.

### Verify outbound links resolve (slow, opt-in)

```bash
site-audit run example.com --check-external
```

This HEAD-requests every external URL the site links to. Results are
cached in the same SQLite cache so subsequent runs are free.

### Force a clean re-crawl

```bash
site-audit run example.com --clean
```

`--clean` deletes the project's cache directory (HTTP + embedding +
paragraph npz) before the run starts, so every page is re-fetched and
re-embedded with the current code. Use this after a crawler / extractor /
embedder change so the new logic actually applies to every page.

If you only want to *bypass* the caches without deleting them (next run
will re-populate them), use the per-cache flags:

```bash
site-audit run example.com --no-http-cache --no-embedding-cache
```

Or wipe the project entirely:

```bash
rm -rf projects/example.com
site-audit run example.com
```

### Disable individual analyses

Each analysis can be turned off:

```bash
site-audit run example.com \
  --no-cluster-labels \
  --no-keyword-coverage \
  --no-answerability \
  --no-linkgraph \
  --no-external-links \
  --no-scatterplot \
  --no-paragraph-clustering \
  --no-paragraph-fanout \
  --no-content-quality
```

(Useful for testing or for huge sites where you only want the duplicate
detection.)

### Compare your site against competitors

After you've run `site-audit run` on every domain you care about:

```bash
site-audit compare your-site.com competitor-1.com competitor-2.com
# or compare every project that has a finished report:
site-audit compare --all --name landscape-2026
open projects/_compare/landscape-2026/index.html
```

See [Comparing multiple domains](#comparing-multiple-domains) below for
what's in the comparison view.

### Audit several sites in parallel

The CLI is single-process, but each call uses an isolated project
folder so you can run multiple in parallel:

```bash
site-audit run site-a.com &
site-audit run site-b.com &
site-audit run site-c.com &
wait
```

---

## What the report shows

The HTML report is read top-to-bottom. Each section answers a different
question.

### 1. Headline metric cards

| Metric | What it tells you |
|---|---|
| **Calibrated focus** | 0–1, model-agnostic. Halfway = average focus. > 0.7 = tight. < 0.3 = scattered. |
| **Topic dimension** | Effective number of independent topics (PCA spectral entropy). 2–4 = laser focused, 15–30 = broad publisher. |
| **Section coherence** | Mean intra-section similarity / inter-section similarity. > 1.5 = your URLs match content. ≈ 1.0 = sections are arbitrary. |
| **Pages** | How many pages were actually analyzed (post-extraction). |
| Raw siteFocusScore | The classic mean-cosine-to-centroid number. Anchored to the embedding model's floor — for `gte-multilingual-base` it lives in roughly 0.5–0.9. Mostly useful as a sanity check. |
| siteRadius | Std-dev of cosine distance to the centroid. Lower = tighter spread. |
| Outlier pages | Pages flagged as off-topic for their section. |
| Near-duplicates | Page pairs with cosine ≥ 0.92. |

### 1b. Action plan (prioritised recommendations)

Right under the cards. This is the **"what should I do next"** section —
every other analysis below feeds into a single ranked list of concrete
edits. Each row has:

- a **priority** badge (high / medium / low),
- an **effort** badge (quick / medium / deep),
- a **category** (content debt / coverage / GEO / linking / on-page),
- a one-line title,
- a specific instruction (which page, which paragraph, what anchor, etc.),
- the evidence behind the call (similarity, lift, PageRank, GEO score,
  click depth …).

Filter by priority or category with the buttons at the top. Capped at
the top 100 actions site-wide; high-PageRank pages are weighted up so
fixing the load-bearing pages compounds. Generated by
`site_audit/recommendations.py` from every other JSON payload — no extra
embeddings, just synthesis.

### 1c. Header structure (H1–H6)

How well the document outline reflects the page's content. Five summary
cards across the top (total headers, pages missing H1, pages with
multiple H1s, drifty headers, title↔H1 mismatch) plus four sub-tables:

- **Structural issues** — pages missing H1, duplicate H1s, skipped
  levels (H1 → H3 = skip), long pages with no H2/H3 sub-sections.
- **Title ↔ H1 mismatch** — cosine of the page title to the H1 below
  0.6 (page promises one thing, the H1 promises another).
- **Drifty headers** — every header is embedded and compared to its
  host page's paragraph centroid; cosine < 0.65 ⇒ the header doesn't
  describe its content (click-bait, stray template fragment, or topic
  the page doesn't actually cover).
- **Header keyword frequency by level** — top keywords across all H1s,
  all H2s, all H3s. A few keywords dominating each level = focused
  brand voice; flat distribution = boilerplate or every page covers
  something different.

The full header set is also visible in the Semantic Scatterplot via the
**Headers** toggle — a header sitting far from the paragraph cloud is
visually obvious there.

### 2. Centroid-similarity histogram

Distribution of per-page cosine similarity to the site centroid.
A unimodal-tight distribution = focused site. A bimodal distribution
= you have two different audiences / two different products on one
domain.

### 3. Semantic scatterplot

Every dot is a page, paragraph, or header (toggle at the top), projected
to 2D via UMAP. Colour by section, cluster, drift heat-map, or "outliers
+ duplicates". Hover for details, click to open the page, scroll to zoom.

Three datasets via the **Show:** toggle:

- **Pages** (default) — one dot per crawled page. Color modes: Section,
  Cluster, Drift heatmap, Outliers + duplicates.
- **Paragraphs** — every extracted paragraph as a smaller dot, in the
  same projection. Sub-topic granularity, the level LLMs actually
  retrieve at. Drift / outliers / duplicates don't apply at paragraph
  granularity, so those modes fall back to cluster colouring.
- **Headers** — every H1–H6 as a dot, projected through the *paragraph
  embedding space* (so a header's distance from the paragraph cloud
  visually shows whether it describes its host content). Coloured by
  header level: H1 red → H2 orange → H3 blue → H4 green → … → H6 grey.

### 4. Weakest sections

Sections sorted by lowest cohesion. The bottom of this list is where
your URL structure is fighting your content structure.

### 5. Top outliers

Pages furthest from their section centroid. Each row has a
recommendation: refocus / consolidate / remove / merge.

### 6. Top duplicate pairs

Cosine ≥ 0.92, sorted by similarity. Each pair has a merge
recommendation.

### 7. Topic clusters (auto-labelled)

K-means clusters labelled with class-discriminative keywords
(BERTopic-style c-TF-IDF). Ranked by `page_count × cohesion` so the
biggest, tightest topics float to the top.

### 8. Keyword coverage

Per query: best matching page, runner-ups, status (`covered`, `gap`, or
`cannibalized`). Filter the list with the buttons.

- **Gap** — best page < `--coverage-threshold` (default 0.55). You
  have no good page for this query.
- **Cannibalized** — three or more pages above
  `--cannibalization-threshold` (default 0.72). Pick a canonical
  one, redirect the rest.

### 9. GEO answer-ability

0–10 per page. Sorted weakest first because that's where the wins are.
Each row shows the breakdown — exactly which signals are missing
(no FAQ schema, no question-form headings, no statistics, no outbound
citations …).

### 10. Internal link graph

- **Top authority pages (PageRank)** — your "load-bearing" pages.
- **Orphan pages** — zero inbound links. Linkbuilding work.
- **Suggested internal links** — high-similarity page pairs that
  aren't currently linked. Ship the top 10.
- **HITS authorities vs hubs** — authorities are linked from many
  hubs (topic destinations); hubs link to many authorities (topic
  indexes). They're often different pages from the PageRank list.
- **Buried pages (depth ≥ 4)** — pages that take 4+ clicks to reach
  from the homepage. Bring them up the tree.
- **Topic-cluster authorities** — canonical PR-best page per topic.
  These are the URLs you want LLM answer engines to land on.

### 10a. Linkbuilding overview

Site-level link health on one screen. Eight summary cards:

- **Total links** + breakdown of internal vs external.
- **Internal : external ratio** — content sites usually run 5:1 or higher;
  directory / aggregator sites run closer to 1:1 or even external-heavy.
- **Average / median / p90 inbound and outbound per page** — distribution
  shape for at-a-glance comparison with competitors.
- **Distinct anchor phrases** site-wide.
- **Descriptive anchor share** (≥ 60% is healthy).
- **Generic anchor share** ("click here", numeric, single-word — these
  waste link equity).
- **Empty links** — anchors with no text, no `title`, no `aria-label`,
  no image `alt`. Accessibility failure + dead SEO juice.
- **Image-only links without alt text** — the same problem in image
  form: useless to assistive tech and search engines.

Plus an **anchor-text scatter** — every distinct anchor phrase site-wide
is embedded and projected to 2D. Dot size = usage frequency. Colour:
blue for mostly-internal anchors, green for mostly-external, red for
generic ("click here"-class). Tight clusters = consistent editorial
voice; isolated red dots = generic anchors to clean up.

Below the scatter, three frequency lists: top internal anchors, top
external anchors, top generic anchors — so you can decide which patterns
to keep and which to rewrite.

### 10b. Internal links per page (in / out distribution)

A chart showing the distribution of inbound and outbound internal
links across every crawled page. Two modes:

- **Histogram** — overlapping bars: blue = in-degree, green = out-degree.
  Reveals shape problems at a glance: a flat in-degree distribution
  means most pages get exactly 0–1 internal links (linkbuilding debt);
  a tall narrow out-degree distribution means every page links to the
  same handful of destinations (no contextual linking).
- **Scatter (in vs out)** — each page is a dot at `(in_degree,
  out_degree)`. Red dots = orphans (in = 0), orange = dead-ends
  (out = 0). Click any dot to open the page.

Followed by an expandable table of the top-30 pages by in-degree.

### 11. In-paragraph link recommendations

Page-level recommendations are useful but vague — they tell you "link
A to B" without telling you *where* on A. This section closes that
gap: each row is a *specific paragraph* on a source page, a *specific
target page*, a *specific anchor phrase* (drawn from words actually
present in the paragraph), and a `lift` score = how much more relevant
that paragraph is to the target than the source page as a whole.
Already-saturated paragraphs (≥ 5 links per 100 words) are skipped
automatically — see *Paragraph link density* below.

### 11b. Paragraph link density

Counts of internal vs external `<a href>` per paragraph, divided by
word count × 100, gives "links per 100 words". Three things on screen:

- **Summary cards** — median density, p90 density, count of "spammy"
  paragraphs (≥ 5 links/100w). Plus the share of paragraphs with zero
  links — large blocks of running text that pass no authority.
- **Per-page distribution** — histogram of average density across
  pages, with a dashed line at the spammy threshold.
- **Two action lists** — paragraphs to clean up (link-stuffed) and
  long paragraphs that could host an inline link (zero links, ≥ 80
  words).

Same data also feeds two recommendations in the Action plan: clean up
link-stuffed paragraphs, add a link to long unlinked paragraphs on
high-PR pages.

### 11c. Wrong-home paragraphs

Paragraphs whose embedding fits a *different* page meaningfully better
than their host page (lift ≥ 0.10). These are content fragments that
ended up on the wrong URL — usually because a long page tried to do too
many things. Each row tells you which paragraph to move and where.

### 11d. Paragraph topic clusters + scatter

Sub-topic granularity, the level at which LLM answer engines actually
retrieve passages. The scatter is now part of the main Semantic
Scatterplot at the top (Paragraph mode); the cluster cards list the
discriminative keywords + a couple of example paragraphs per cluster.

### 11e. Query → paragraph fanout

For each auto-mined / supplied query, the top paragraphs site-wide
(across pages). Status:

- **gap** — no paragraph above the floor; you have no answer for this
  query at any granularity.
- **scattered** — answer fragmented across many pages; LLMs can't pick
  a single citation.
- **focused** — top 3 paragraphs are on the same page (best for AI
  answer engines).

### 12. Anchor-text analysis

Per target page: top inbound anchors, generic-anchor share
("click here", "more"), and worst-case anchor↔target topic mismatch
via embeddings. Pages dominated by generic anchors are missing
free internal-SEO juice.

### 12b. Title doesn't match content

Pages with cosine(title, paragraph_centroid) below 0.6. Each row
suggests keywords drawn from the page's actual content so the rewrite
is grounded.

### 13. External link profile

- **Most-cited domains** — where your authors look for sources.
  Authoritative-TLD domains (`.gov`, `.edu`, Wikipedia, NIH, …) are
  flagged.
- **Citation density** — external links per 1 000 words. Pages with
  zero outbound citations look unsourced to LLM answer engines.
- **Broken outbound links** — only populated when you pass
  `--check-external`.

### 14. Competitor comparison (optional)

Populated only when you pass `--competitive my_targets.tsv` (one
`query<TAB>competitor_url` per line). For each pair the report shows:

- the competitor page's GEO answer-ability score,
- which signals it has that your best-matching page is missing,
- the closest 3 paragraphs on your site for the same query,
- a recommended next step (write, expand, restructure …).

This is *per-target* competitive analysis — for cross-domain quality
comparison across whole sites, use `site-audit compare` (see below).

---

## Comparing multiple domains

`site-audit run` produces one report per domain. **`site-audit compare`
joins multiple already-crawled projects into a single comparison view**
— useful for benchmarking against competitors, auditing an agency
landscape, or just sanity-checking a site against itself over time.

```bash
# specific list (each must have a finished report)
site-audit compare your-site.com competitor-1.com competitor-2.com

# every domain that has a finished report under projects/
site-audit compare --all --name agencies-2026

open projects/_compare/agencies-2026/index.html
```

Output bundle (`projects/_compare/<name>/`):

- `index.html` — self-contained comparison viewer
- `comparison.json` — leaderboard rows + scatter coordinates + per-domain
  distributions, in case you want to build your own visualisations

The viewer has three sections:

1. **Leaderboard** — every metric we compute, side by side, in tabbed
   groups (Overview / Topic shape / GEO / Internal linking / Linkbuilding
   / Headers / Paragraph density / Action plan). Each column header
   sorts; cells are heat-mapped per-column (green = best for that
   metric's "good" direction, red = worst).
2. **Combined Semantic Scatter** — every domain's pages projected via
   *one shared UMAP* (the only way the spatial picture is meaningful
   across domains). Coloured by domain, with per-domain visibility
   toggles, hover for details, click to open the page.
3. **Distribution overlays** — in-degree, out-degree, GEO score, and
   paragraph link density distributions for all domains in the same
   chart, normalised to *share of pages* so different-sized sites can be
   compared honestly. Solid lines = density; faint bars = histogram.

Cheap to run: reads the per-domain `*.json` payloads and the cached
embeddings npz. No re-crawl, no re-embedding. Typical cost on 7 domains
× ~600 pages each ≈ 25 s for the combined UMAP.

Requirements: each domain in the comparison must already have been run
with `site-audit run` (the embeddings npz cache is what enables the
combined scatter), and they should all have used the same embedding
model (the comparison would still produce a leaderboard if models
differ, but skips the scatter for any domain whose embeddings have a
different dimensionality).

---

## Project layout on disk

Each domain becomes a project under `projects/`:

```
projects/
  example.com/
    cache/
      http.sqlite                         # raw HTTP responses, keyed by URL
      embeddings_<model_slug>.npz         # per-page embeddings, keyed by content hash
      paragraphs_<model_slug>.npz         # per-paragraph embeddings (incremental)
    report/
      index.html                          # ← self-contained viewer (open this)
      site_metrics.json                   # focus, radius, coherence, topic dim, sections
      section_report.json                 # cohesion ranking
      page_drift.csv                      # per-page drift to centroids
      outliers.csv                        # off-topic / orphan / thin pages
      duplicates.csv                      # near-duplicate pairs
      pages.json                          # page metadata
      scatterplot.json                    # input for the page scatter
      clusters.json                       # auto-labelled topic clusters
      cluster_overlap.json                # NxN cluster centroid similarity
      keyword_coverage.json               # query → best page mapping
      answerability.json                  # GEO 0-10 score per page
      linkgraph.json                      # PageRank, HITS, depth, orphans, recs,
                                          #   anchor analysis, page_link_counts
      external_links.json                 # outbound profile, broken links
      linkbuilding.json                   # site-level link health, anchor audit, anchor scatter
      indexability.json                   # crawl-to-analysis funnel + skipped/noindex pages
      structured_data.json                # schema.org JSON-LD coverage, validity, type mix
      metadata_quality.json               # SERP title/description/canonical/social metadata health
      media_accessibility.json            # image alt/caption/transcript/embed accessibility signals
      page_types.json                     # page type and template-family classifications
      entities.json                       # entity coverage, reuse, organizations, topical authority
      freshness.json                      # date coverage, stale pages, missing/future dates
      conversion.json                     # CTA, form, contact, and lead-capture signals
      performance.json                    # offline HTML/resource weight and render-blocking signals
      header_analysis.json                # H1-H6 structure, drifty headers, keyword freq
      header_scatter.json                 # header UMAP coordinates
      paragraph_clusters.json             # paragraph topic clusters
      paragraph_scatter.json              # paragraph UMAP coordinates
      paragraph_link_recommendations.json # in-paragraph link recs (with anchors)
      paragraph_density.json              # links/100w summary + spammy + unlinked
      paragraph_fanout.json               # query → top paragraphs site-wide
      title_mismatch.json                 # title↔content cosine + suggested kws
      wrong_home_paragraphs.json          # paragraphs that fit a different page
      page_improvement.json               # composite editing-priority score
      competitive_analysis.json           # only when --competitive supplied
      recommendations.json                # the action plan (top 100 prioritised)

  _compare/
    <name>/
      index.html                          # ← self-contained comparison viewer
      comparison.json                     # leaderboard + combined scatter + dists
```

Wipe the cache with `rm -rf projects/<domain>/cache/`. Wipe everything
with `rm -rf projects/<domain>/`. Wipe a comparison with `rm -rf
projects/_compare/<name>/`.

---

## All the flags

```bash
site-audit run <domain> [flags]
```

| Flag | Default | What it does |
|---|---|---|
| `--projects-root` | `projects/` | Where domain projects live. |
| `--cache-dir`, `--output-dir` | (auto) | Override individual paths if you don't want the standard layout. |
| `--max-pages` | 10000 | Hard cap on pages crawled. |
| `--workers` | 8 | Concurrent fetches. |
| `--follow-subdomains` | off | Allow `*.example.com`. |
| `--ignore-robots` | off | Skip `robots.txt` enforcement. Use sparingly. |
| `--no-http-cache` | off | Bypass the HTTP response cache (force re-fetch). |
| `--no-embedding-cache` | off | Bypass the embedding cache (force re-embed). |
| `--clean` | off | Delete the project's cache directory before the run. Forces every page to be re-crawled, re-extracted and re-embedded under the current code. |
| `--max-chars` | 4000 | Body characters fed to the embedder. Bigger ≠ always better. |
| `--model` | `Alibaba-NLP/gte-multilingual-base` | Any sentence-transformer model on Hugging Face. |
| `--scatter-clusters` | 30 | k-means clusters for scatter coloring. |
| `--no-scatterplot` | off | Skip UMAP projection (faster). |
| `--duplicate-threshold` | 0.92 | Cosine above which pages are flagged as near-duplicates. |
| `--duplicate-knn` | 10 | kNN depth for duplicate search. |
| `--queries-file` | — | File with one target query per line. |
| `--auto-queries-max` | 200 | Cap on auto-mined queries (titles + question-form H2s). |
| `--coverage-threshold` | 0.55 | Below this best-similarity, a query is a "gap". |
| `--cannibalization-threshold` | 0.72 | ≥ 3 pages above this similarity = "cannibalized". |
| `--no-keyword-coverage` | off | Skip query → page coverage analysis. |
| `--no-cluster-labels` | off | Skip c-TF-IDF cluster labelling. |
| `--no-answerability` | off | Skip GEO answer-ability scoring. |
| `--no-linkgraph` | off | Skip PageRank + link recommendations. |
| `--no-paragraph-clustering` | off | Skip paragraph topic clustering + scatter. |
| `--no-paragraph-fanout` | off | Skip query → paragraph fanout. |
| `--no-content-quality` | off | Skip title-mismatch + wrong-home-paragraph + composite improvement. |
| `--link-similarity-threshold` | 0.85 | Min similarity for an internal-link recommendation. |
| `--link-recommendations` | 75 | How many link recs to surface. |
| `--no-external-links` | off | Skip outbound-link profile. |
| `--check-external` | off | HEAD-check every outbound URL. Slow, results cached. |
| `--competitive` | — | TSV file with `query<TAB>competitor_url` per line, for the per-target competitive analysis. |
| `--request-delay` | 0 | Seconds to sleep before each fetch. Slow down for rate-limited sites. |
| `-v`, `--verbose` | off | DEBUG-level logging. |

```bash
site-audit serve <domain> [--projects-root projects/] [--port 8765]
```

Serves the report directory locally for the live viewer.

```bash
site-audit compare DOMAIN1 DOMAIN2 [...] [--projects-root projects/] [--name <subdir>]
site-audit compare --all                 [--projects-root projects/] [--name <subdir>]
```

Builds a side-by-side comparison HTML across multiple already-crawled
domains. Output goes to `projects/_compare/<name>/index.html`. See
[Comparing multiple domains](#comparing-multiple-domains) above.

---

## Re-runs and caching

The two caches under `projects/<domain>/cache/` make re-runs cheap:

- **HTTP cache** (`http.sqlite`) — keyed by URL. Even crawling a 6 000-
  page site is < 5 s on a re-run.
- **Embedding cache** (`embeddings_<model>.npz`) — keyed by
  `(url, sha256(text|model))`. Editing one page invalidates that one
  entry; the rest still hit.

Typical workflow:

```bash
# Initial run — slow because everything is cold
site-audit run example.com --max-pages 5000

# Edit a few pages on the live site, then re-run — ~ 30 s total
site-audit run example.com --max-pages 5000

# Switch to a different embedding model, full re-embed
site-audit run example.com --model intfloat/multilingual-e5-base
```

---

## Programmatic use

```python
from site_audit.pipeline import PipelineConfig, run

summary = run(PipelineConfig(
    domain="example.com",
    max_pages=500,
    queries_file="my_queries.txt",
    check_external_links=False,
))

print(summary)
# {
#   'domain': 'example.com',
#   'pages': 487,
#   'site_focus_score': 0.7712,
#   'calibrated_focus': 0.5491,
#   'topic_dim': 31.4,
#   'section_coherence': 1.18,
#   'outliers': 18,
#   'duplicate_pairs': 7,
#   'clusters': 30,
#   'queries_evaluated': 200,
#   'linkgraph_edges': 4527,
#   'linkgraph_orphans': 14,
#   'max_click_depth': 5,
#   'link_recommendations': 41,
#   'external_domains': 22,
#   'broken_external': 0,
#   'report_dir': 'projects/example.com/report',
#   'html_report': 'projects/example.com/report/index.html',
# }
```

Build a comparison across already-crawled domains:

```python
from pathlib import Path
from site_audit.compare import build_payload, write_html

payload = build_payload(
    ["your-site.com", "competitor-1.com", "competitor-2.com"],
    projects_root=Path("projects"),
)
# payload contains: domains, leaderboard, scatter, scatter_total, distributions
write_html(
    template_path=Path("ui/compare.html"),
    payload=payload,
    out_path=Path("projects/_compare/manual/index.html"),
)
```

Synthesise the action plan from already-saved JSON payloads (the same
function the pipeline calls, but you can pass arbitrary payloads):

```python
import json
from site_audit.recommendations import synthesize, to_payload

R = lambda p: json.load(open(p))
recs = synthesize(
    duplicates_rows=[],            # build from result.duplicate_pairs or duplicates.csv
    outliers_rows=[],              # build from result or outliers.csv
    coverage_payload=R("projects/example.com/report/keyword_coverage.json"),
    answerability_payload=R("projects/example.com/report/answerability.json"),
    linkgraph_payload=R("projects/example.com/report/linkgraph.json"),
    paragraph_links=R("projects/example.com/report/paragraph_link_recommendations.json"),
    wrong_home_payload=R("projects/example.com/report/wrong_home_paragraphs.json"),
    title_mismatch=R("projects/example.com/report/title_mismatch.json"),
    external_links_payload=R("projects/example.com/report/external_links.json"),
)
print(to_payload(recs)["by_priority"])
# {'high': 76, 'medium': 24, 'low': 0}
```

---

## Troubleshooting

**The first run takes forever.** First run downloads the
~ 600 MB embedding model. After that, both the model and the
per-domain caches make re-runs much faster.

**Every page got filtered out / "No usable pages".**
Check that the site renders content server-side. Single-page apps
(React/Vue/Angular SPAs that fetch content via JS) can't be audited
without a snapshot service. Try `curl -s https://example.com | head`
to see if the body has real text.

**Crawl ends with way fewer pages than the sitemap suggested.**
Most sitemap URLs are typically images / variants / 4xx redirects
that get filtered. The `Crawl finished: N pages` log line is
HTML-only successful fetches.

**`ValueError: The truth value of an array with more than one element
is ambiguous`.** Should be fixed in 0.2.0+. If you see it, capture the
log and open an issue.

**Apple Silicon: silent process exit at exactly the embedding step.**
Caused by faiss-cpu and PyTorch each loading their own libomp.
We set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` in
`site_audit/__init__.py` before any heavy imports — if you're calling
the modules from a different harness, set those env vars yourself
*before* importing `site_audit`.

**Permission errors on `robots.txt` parse.** The crawler logs a
warning and proceeds with an empty robots policy. If a site is
refusing your crawler entirely, try `--ignore-robots` (only with
permission) or set a different `User-Agent` via library use.

**Image / asset URLs showing up as "pages".** Fixed in 0.3+ — the URL
filter now operates on `urlparse(url).path` (so query strings can't
hide the extension, e.g. `/img.jpg?v=42`). If you have an old project
with image URLs in `pages.json`, just re-run; the cached entries won't
re-enter the frontier and will drop out naturally on next crawl.

**Noindex pages slipping into the analysis.** `<meta name="robots"
content="noindex">` and `X-Robots-Tag: noindex` are honoured at
extraction time and the page is dropped from the corpus before
embedding. The crawler still consumes the page during the BFS, so its
outlinks still contribute to discovery; only the noindex page itself
is excluded from focus / clusters / GEO / recommendations. Per-bot
directives (`googlebot`, `bingbot`, etc.) are honoured.

**Embedding step crashes with `index out of bounds` or
`AcceleratorError` on Apple Silicon.** The default model
`Alibaba-NLP/gte-multilingual-base` ships custom modeling code that is
incompatible with `transformers >= 5`. Pin it: `pip install
"transformers>=4.40,<5.0"`. If MPS itself is the problem (rare), force
CPU with `SITE_AUDIT_DEVICE=cpu site-audit run example.com`.

**Embedding step is slow on a big site.** Most cost is the per-page
forward pass. The paragraph-link recommender now batch-encodes every
candidate n-gram in a single call (~7× faster than the per-target
loop in earlier versions), but you can also disable individual
analyses (`--no-paragraph-clustering`, `--no-paragraph-fanout`,
`--no-content-quality`) to skip work you don't need.

**Choose the device.** `SITE_AUDIT_DEVICE=cuda|mps|cpu` overrides the
sentence-transformer's auto-detection. Useful for benchmarking or
when MPS is misbehaving on a particular torch build.

---

## How the metrics work

* **siteFocusScore** = mean cosine similarity of every page embedding to
  the global site centroid. `1.0` → laser focused. Anchored to the
  embedding model — for `gte-multilingual-base` lives in 0.5–0.9.
* **calibrated_focus_score** = `(focus - p10_pairwise) / (1 -
  p10_pairwise)`. The site's own p10 pairwise similarity is the
  per-corpus model floor. Output is a clean `[0, 1]`.
* **siteRadius** = std-dev of cosine distance to the centroid.
* **Effective topic dimension** = `exp(H)` where `H` is the Shannon
  entropy of the normalized eigenvalues of the (sample) covariance.
  Model-agnostic. 2–4 = laser focused; 15–30 = broad publisher.
* **Section coherence ratio** = mean intra-section similarity / mean
  inter-section similarity. > 1.5 = strong; ≈ 1.0 = arbitrary.
* **Drift to section centroid** = `1 - cosine(page, section_centroid)`.
  Pages above the section's 95th percentile become outliers.
* **Near-duplicate** = cosine ≥ `--duplicate-threshold` via FAISS k-NN.
* **Cluster cohesion** = mean cosine of pages in a cluster to its
  centroid.
* **Cluster site-alignment** = cosine of cluster centroid to the global
  site centroid. High = core to the site, low = peripheral.
* **Coverage status** — for each query: `gap` (best page < 0.55),
  `cannibalized` (≥3 pages above 0.72), or `covered`.
* **Answer-ability score** — 0–10 sum of weighted signals: FAQ schema
  (3), question-form headings (1–2), tables (1), lists (≤1),
  statistics (≤1.5), external citations (≤1), dates (0.5),
  question-form title (0.5), long-form (0.3).
* **PageRank** — classic damped iteration over the internal link graph.
* **HITS** — `hub` and `authority` scores; hubs point to many
  authorities, authorities are pointed to by many hubs.
* **Click depth** — BFS shortest-path from the homepage URL.
* **In-degree / out-degree** — count of distinct internal links *into*
  / *out of* a page. Surfaced per-page (full distribution chart) and
  used to flag orphans (in = 0) and dead-ends (out = 0).
* **Paragraph link density** = `(internal_links + external_links) /
  word_count × 100` per paragraph. Spammy threshold = 5 links per
  100 words; paragraphs at or above that score are excluded from
  receiving more in-paragraph link recommendations.
* **Wrong-home lift** = `cosine(paragraph, suggested_page_centroid) -
  cosine(paragraph, host_page_centroid)`. ≥ 0.10 = paragraph fits a
  different page meaningfully better than its current host.
* **Title-content cosine** = `cosine(title_embedding,
  paragraph_centroid_of_page)`. < 0.6 ⇒ misleading title.
* **Recommendation priority** — discrete `high` / `medium` / `low`.
  Priority is decided per category (very-similar duplicates are always
  high; orphan pages on the cluster-authority list are high; low GEO
  scores on high-PR pages are high; etc). Within a priority bucket the
  per-recommendation `score` is used to break ties — higher PageRank
  pages and higher-similarity / higher-lift signals float to the top.
* **Header drift** = `1 - cosine(header_embedding,
  host_page_paragraph_centroid)`. Every H1–H6 is embedded in one
  batched call; cosine < 0.65 means the header doesn't describe its
  page's actual content (drifty header).
* **Title↔H1 cosine** = `cosine(page_embedding, h1_embedding)`. < 0.6
  ⇒ the page title and the visible H1 disagree on the topic.
* **Anchor classification** — descriptive vs generic. Generic =
  `{"click here", "more", "read more", numeric, single-word, …}`. The
  share is computed across every `<a href>` site-wide for the
  Linkbuilding overview cards.
* **Empty-link share** — fraction of `<a href>` with no anchor text,
  no `title`, no `aria-label`, no image `alt`. Accessibility failure
  + dead SEO juice. Healthy ≤ 2%.
* **Combined-corpus UMAP** (used by `site-audit compare`) — every
  domain's page embeddings are concatenated and a *single* UMAP is fit
  across the whole stack. Only configuration that lets cross-domain
  spatial proximity actually mean something. Each domain is sub-
  sampled to ≤ 1 500 pages for the projection so the UMAP fit stays
  fast.

---

## License

MIT. See [LICENSE](LICENSE).
