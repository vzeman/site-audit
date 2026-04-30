# site-audit

Crawl any website, embed every page, and surface near-duplicates,
outliers, topic clusters, GEO citability, and internal-link
recommendations on an interactive D3 scatterplot.

Same vector-space metrics (`siteFocusScore`, `siteRadius`, per-section
drift, near-duplicate kNN, UMAP scatter) as the Hugo site-audit
pipeline, *plus* a GEO-focused stack that's unique to this tool:

**Topic structure**
* **Auto-labelled topic clusters** (BERTopic-style c-TF-IDF)
* **Calibrated focus score** — strips out the embedding model's
  anisotropy floor so the number is interpretable as `[0, 1]`
* **Effective topic dimension** — model-agnostic count of independent
  themes via PCA spectral entropy
* **Section coherence ratio** — does your URL structure match content?

**Keyword & query coverage**
* **Keyword coverage map** — embed target queries in the same vector
  space, find the best page for each, flag gaps and cannibalization

**GEO citability**
* **Answer-ability score** — 0–10 per page based on FAQ schema,
  question-form headings, lists/tables, dates, statistics, citations

**Link analysis**
* **Internal link graph** — PageRank + HITS hubs/authorities,
  click depth from homepage, orphans, dead-ends
* **Topic-cluster authorities** — canonical entry page per cluster
* **Anchor-text analysis** — generic-anchor share + anchor↔target
  topic mismatch detection
* **Internal link recommendations** — high-similarity unlinked pairs
* **External link profile** — most-cited domains, citation density,
  authoritative-domain share, optional broken-link detection

## What it does

1. **Discover** — read `robots.txt` + `sitemap.xml` (and any sitemap
   indexes), then BFS-crawl anything else linked from those pages.
2. **Cache** — every HTTP response goes into a per-domain SQLite cache.
   Re-runs against the same domain skip the network entirely.
3. **Extract** — `trafilatura` pulls main-content text (with a
   BeautifulSoup fallback). Headings, lists, tables, JSON-LD schema and
   outbound citations are pulled too — they feed the GEO score.
4. **Embed** — multilingual sentence-transformer
   (`Alibaba-NLP/gte-multilingual-base` by default), L2-normalized.
   Embeddings are cached per `(url, content_hash, model)` in an NPZ
   archive — re-runs only embed pages whose content actually changed.
5. **Analyze** — global + per-section centroids, drift distances, FAISS
   k-NN duplicate search, c-TF-IDF cluster labels, query-coverage map,
   answer-ability scoring, internal-link graph.
6. **Project** — UMAP 2D + k-means coloring for the scatterplot.
7. **Report** — JSON / CSV reports plus a self-contained HTML viewer.

## Install

Python 3.10+.

```bash
git clone https://github.com/vzeman/site-audit.git
cd site-audit
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Heavy ML deps (`sentence-transformers`, `faiss-cpu`, `umap-learn`,
`scikit-learn`) install along with the package. First run downloads the
embedding model (~ 600 MB) into `~/.cache/huggingface/`.

## Usage

```bash
# 1) Crawl + analyze
site-audit run example.com

# 2) Open the interactive viewer (or just open the HTML directly)
site-audit serve example.com
# → http://127.0.0.1:8765/
```

Re-running the same command is cheap: the HTTP cache + embedding cache
both live under `projects/<domain>/cache/` and are reused.

## Project layout

Each domain becomes a project on disk:

```
projects/
  example.com/
    cache/
      http.sqlite                      # raw HTTP responses
      embeddings_<model_slug>.npz      # per-page embeddings
    report/
      index.html                       # ← self-contained viewer
      site_metrics.json                # focus score, radius, sections
      section_report.json              # cohesion ranking
      page_drift.csv                   # per-page drift to centroids
      outliers.csv                     # off-topic / orphan / thin pages
      duplicates.csv                   # near-duplicate pairs
      pages.json                       # page metadata
      scatterplot.json                 # input for the D3 viewer
      clusters.json                    # auto-labelled topic clusters
      keyword_coverage.json            # query → best page mapping
      answerability.json               # GEO 0-10 score per page
      linkgraph.json                   # PageRank, orphans, recs
```

## Useful flags

```bash
site-audit run example.com \
  --max-pages 5000 \
  --workers 16 \
  --queries-file my_target_keywords.txt \
  --duplicate-threshold 0.94 \
  --link-similarity-threshold 0.88 \
  --follow-subdomains
```

| Flag                            | Default | Notes                                                     |
| ------------------------------- | ------- | --------------------------------------------------------- |
| `--projects-root`               | `projects/` | Where domain projects live.                           |
| `--max-pages`                   | 2000    | Hard cap on pages crawled.                                |
| `--workers`                     | 8       | Concurrent fetches.                                       |
| `--duplicate-threshold`         | 0.92    | Cosine similarity above which pages are flagged.          |
| `--queries-file`                | —       | Optional list of target queries (one per line).           |
| `--auto-queries-max`            | 200     | Cap on titles + question-form H2s auto-mined as queries.  |
| `--coverage-threshold`          | 0.55    | Below this best-similarity, query counts as a "gap".      |
| `--cannibalization-threshold`   | 0.72    | Pages above this similarity competing for the same query. |
| `--link-similarity-threshold`   | 0.85    | Min similarity for an internal-link recommendation.       |
| `--link-recommendations`        | 75      | How many link recs to surface.                            |
| `--scatter-clusters`            | 30      | k-means clusters for scatter coloring.                    |
| `--max-chars`                   | 4000    | Body chars passed to the embedder.                        |
| `--follow-subdomains`           | off     | Allow `*.example.com`.                                    |
| `--ignore-robots`               | off     | Skip robots.txt enforcement.                              |
| `--no-http-cache`               | off     | Bypass the HTTP cache.                                    |
| `--no-embedding-cache`          | off     | Bypass the embedding cache.                               |
| `--no-scatterplot`              | off     | Skip UMAP projection.                                     |
| `--no-cluster-labels`           | off     | Skip c-TF-IDF cluster labelling.                          |
| `--no-keyword-coverage`         | off     | Skip query → page coverage analysis.                      |
| `--no-answerability`            | off     | Skip GEO answer-ability scoring.                          |
| `--no-linkgraph`                | off     | Skip PageRank + link recommendations.                     |
| `--model`                       | `Alibaba-NLP/gte-multilingual-base` | Any sentence-transformer model. |

## How the metrics work

* **siteFocusScore** = mean cosine similarity of every page embedding to
  the global site centroid. `1.0` → laser-focused; `<0.4` → all over the
  place.
* **siteRadius** = standard deviation of cosine distances to the
  centroid. Lower = tighter spread.
* **Section focus / radius** — the same metrics restricted to pages in
  one URL section (first path segment).
* **Drift to section centroid** — how far an individual page sits from
  the centroid of its own section. Pages above the section's 95th
  percentile become outliers.
* **Near-duplicate** — cosine similarity ≥ `--duplicate-threshold` via
  FAISS k-NN. Default 0.92.
* **Cluster cohesion** — mean cosine of pages in a cluster to its
  centroid. High = tight topic.
* **Cluster site-alignment** — cosine of cluster centroid to the global
  site centroid. High = core to the site, low = peripheral.
* **Coverage status** — for each query: `gap` (best page < 0.55),
  `cannibalized` (≥3 pages above 0.72), or `covered`.
* **Answer-ability score** — 0–10 sum of weighted signals: FAQ schema
  (3), question-form headings (1–2), tables (1), lists (≤1),
  statistics (≤1.5), external citations (≤1), dates (0.5),
  question-form title (0.5), long-form (0.3).
* **PageRank** — classic damped iteration over the internal link graph.
  Surfaces hub pages.

## Programmatic use

```python
from site_audit.pipeline import PipelineConfig, run

summary = run(PipelineConfig(domain="example.com", max_pages=500))
print(summary)
```

## Why a custom format

* The HTTP cache is keyed by URL. Wipe it with
  `rm projects/<domain>/cache/http.sqlite`.
* The embedding cache is keyed by `(url, sha256(text|model))`. Editing a
  page invalidates that one entry; the rest still hit.
* Reports are pure JSON / CSV with no proprietary wrapper, so they're
  easy to pipe into a notebook, BI tool, or LLM-driven follow-up.

## Limitations

* The crawler does not execute JavaScript — single-page apps need
  server-side rendering (or a snapshot service) to be auditable.
* Embedding ~5 000 pages on a CPU takes ~10 minutes the first time;
  subsequent runs are cache-bound.
* Auto-mined queries are only as good as the page titles and H2s. For
  serious GEO work, supply `--queries-file` with the actual queries
  your customers ask Perplexity / ChatGPT.
* On Apple Silicon we set `KMP_DUPLICATE_LIB_OK=TRUE` and
  `OMP_NUM_THREADS=1` automatically because PyTorch and faiss-cpu both
  ship libomp and refuse to coexist otherwise.

## License

MIT.
