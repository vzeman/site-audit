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

That:

1. Reads `robots.txt` + every `sitemap.xml` it can find
2. BFS-crawls the site (default cap: 2 000 pages)
3. Extracts main-content text from each page
4. Embeds every page with the multilingual sentence-transformer
5. Computes focus / radius / topic clusters / duplicate pairs / outliers
6. Builds the internal link graph (PageRank, HITS, click depth, orphans)
7. Mines target queries from titles + question-form H2s, finds the best
   page for each, flags coverage gaps and cannibalization
8. Scores every page on GEO answer-ability (FAQ schema, question
   headings, lists/tables, dates, statistics, citations)
9. Profiles outbound links (top cited domains, citation density)
10. Writes JSON/CSV outputs **and** a self-contained HTML report

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
site-audit run example.com --no-http-cache --no-embedding-cache
```

Or just delete the project folder:

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
  --no-scatterplot
```

(Useful for testing or for huge sites where you only want the duplicate
detection.)

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

### 2. Centroid-similarity histogram

Distribution of per-page cosine similarity to the site centroid.
A unimodal-tight distribution = focused site. A bimodal distribution
= you have two different audiences / two different products on one
domain.

### 3. Semantic scatterplot

Every dot is a page, projected to 2D via UMAP. Colour by section,
cluster, drift heat-map, or "outliers + duplicates". Hover for details,
click to open the page, scroll to zoom.

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

### 11. Anchor-text analysis

Per target page: top inbound anchors, generic-anchor share
("click here", "more"), and worst-case anchor↔target topic mismatch
via embeddings. Pages dominated by generic anchors are missing
free internal-SEO juice.

### 12. External link profile

- **Most-cited domains** — where your authors look for sources.
  Authoritative-TLD domains (`.gov`, `.edu`, Wikipedia, NIH, …) are
  flagged.
- **Citation density** — external links per 1 000 words. Pages with
  zero outbound citations look unsourced to LLM answer engines.
- **Broken outbound links** — only populated when you pass
  `--check-external`.

---

## Project layout on disk

Each domain becomes a project under `projects/`:

```
projects/
  example.com/
    cache/
      http.sqlite                         # raw HTTP responses, keyed by URL
      embeddings_<model_slug>.npz         # per-page embeddings, keyed by content hash
    report/
      index.html                          # ← self-contained viewer (open this)
      site_metrics.json                   # focus, radius, coherence, topic dim, sections
      section_report.json                 # cohesion ranking
      page_drift.csv                      # per-page drift to centroids
      outliers.csv                        # off-topic / orphan / thin pages
      duplicates.csv                      # near-duplicate pairs
      pages.json                          # page metadata
      scatterplot.json                    # input for the D3 viewer
      clusters.json                       # auto-labelled topic clusters
      keyword_coverage.json               # query → best page mapping
      answerability.json                  # GEO 0-10 score per page
      linkgraph.json                      # PageRank, HITS, depth, orphans, recs
      external_links.json                 # outbound profile, broken links
```

Wipe the cache with `rm -rf projects/<domain>/cache/`. Wipe everything
with `rm -rf projects/<domain>/`.

---

## All the flags

```bash
site-audit run <domain> [flags]
```

| Flag | Default | What it does |
|---|---|---|
| `--projects-root` | `projects/` | Where domain projects live. |
| `--cache-dir`, `--output-dir` | (auto) | Override individual paths if you don't want the standard layout. |
| `--max-pages` | 2000 | Hard cap on pages crawled. |
| `--workers` | 8 | Concurrent fetches. |
| `--follow-subdomains` | off | Allow `*.example.com`. |
| `--ignore-robots` | off | Skip `robots.txt` enforcement. Use sparingly. |
| `--no-http-cache` | off | Bypass the HTTP response cache (force re-fetch). |
| `--no-embedding-cache` | off | Bypass the embedding cache (force re-embed). |
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
| `--link-similarity-threshold` | 0.85 | Min similarity for an internal-link recommendation. |
| `--link-recommendations` | 75 | How many link recs to surface. |
| `--no-external-links` | off | Skip outbound-link profile. |
| `--check-external` | off | HEAD-check every outbound URL. Slow, results cached. |
| `-v`, `--verbose` | off | DEBUG-level logging. |

```bash
site-audit serve <domain> [--projects-root projects/] [--port 8765]
```

Serves the report directory locally for the live viewer.

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

---

## License

MIT. See [LICENSE](LICENSE).
