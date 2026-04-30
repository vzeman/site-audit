# site-audit

Crawl any website, embed every page, and surface near-duplicates, outliers,
and topical structure on an interactive D3 scatterplot. Same metrics
(`siteFocusScore`, `siteRadius`, per-section drift, near-duplicate kNN,
UMAP scatter) as the Hugo site-audit pipeline, but pointed at any URL on
the public web instead of a local content directory.

![scatterplot](https://raw.githubusercontent.com/vzeman/site-audit/main/.github/screenshot.png)

## What it does

1. **Discover** — reads `robots.txt` + `sitemap.xml` (and any sitemap
   indices), then BFS-crawls anything else linked from the sitemap pages.
2. **Cache** — every HTTP response goes into a per-domain SQLite cache.
   Re-runs against the same domain skip the network entirely.
3. **Extract** — `trafilatura` pulls main-content text (with a
   BeautifulSoup fallback).
4. **Embed** — multilingual sentence-transformer (`gte-multilingual-base`
   by default), L2-normalized. Embeddings are cached per
   `(url, content_hash, model)` in an NPZ archive — re-runs only embed
   pages whose content actually changed.
5. **Analyze** — global + per-section centroids, drift distances, FAISS
   k-NN duplicate search.
6. **Project** — UMAP 2D + k-means coloring for the scatterplot.
7. **Report** — JSON / CSV reports plus a self-contained D3 viewer.

## Install

Python 3.10+.

```bash
git clone https://github.com/vzeman/site-audit.git
cd site-audit
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Heavy ML deps (`sentence-transformers`, `faiss-cpu`, `umap-learn`) install
along with the package. First run downloads the embedding model
(~ 600 MB) into `~/.cache/huggingface/`.

## Usage

```bash
# 1) Crawl + analyze
site-audit run example.com

# 2) Open the interactive viewer
site-audit serve example.com
# → http://127.0.0.1:8765/
```

Re-running the same command is cheap: the HTTP cache and the embedding
cache both live under `cache/<domain>/` and are reused.

### Useful flags

```bash
site-audit run example.com \
  --max-pages 5000 \
  --workers 16 \
  --duplicate-threshold 0.94 \
  --follow-subdomains
```

| Flag                       | Default | Notes                                                 |
| -------------------------- | ------- | ----------------------------------------------------- |
| `--max-pages`              | 2000    | Hard cap on pages crawled.                            |
| `--workers`                | 8       | Concurrent fetches.                                   |
| `--duplicate-threshold`    | 0.92    | Cosine similarity above which pages are flagged.      |
| `--duplicate-knn`          | 10      | kNN depth used to find duplicate candidates.          |
| `--scatter-clusters`       | 30      | Number of k-means clusters for scatter coloring.      |
| `--max-chars`              | 4000    | Body chars passed to the embedder.                    |
| `--follow-subdomains`      | off     | Allow `*.example.com` (uses registrable-domain match).|
| `--ignore-robots`          | off     | Skip robots.txt enforcement (use sparingly).          |
| `--no-http-cache`          | off     | Bypass the HTTP cache.                                |
| `--no-embedding-cache`     | off     | Bypass the embedding cache.                           |
| `--no-scatterplot`         | off     | Skip UMAP projection (faster on huge sites).          |
| `--model`                  | `Alibaba-NLP/gte-multilingual-base` | Any sentence-transformer model. |

## Output

```
output/<domain>/
├── site_metrics.json      # focus score, radius, per-section summary
├── section_report.json    # weakest sections sorted by cohesion
├── page_drift.csv         # one row per page, drift to site + section
├── outliers.csv           # off-topic / orphan / thin-and-off-topic pages
├── duplicates.csv         # near-duplicate pairs with merge recommendations
├── scatterplot.json       # input for the D3 viewer
└── pages.json             # raw page metadata
```

## How the metrics work

* **siteFocusScore** = mean cosine similarity of every page embedding to
  the global site centroid. `1.0` → laser-focused; `<0.4` → all over the
  place.
* **siteRadius** = standard deviation of cosine distances to the
  centroid. Lower = tighter spread.
* **Section focus / radius** = the same thing, restricted to pages in
  one URL section (first path segment).
* **Drift to section centroid** = how far an individual page sits from
  the centroid of its own section. Pages above the section's 95th
  percentile become outliers.
* **Near-duplicate** = cosine similarity ≥ `--duplicate-threshold` via
  FAISS k-NN. Default 0.92.

These are exactly the formulas used by `generate_site_audit.py` in
[FlowHunt-hugo](https://github.com/QualityUnit/FlowHunt-hugo) so the
reports are directly comparable.

## Caching layout

```
cache/<domain>/
├── http.sqlite                              # raw HTTP responses
└── embeddings_<model_slug>.npz              # per-page embeddings
```

* The HTTP cache is keyed by URL. Wipe it with `rm cache/<domain>/http.sqlite`.
* The embedding cache is keyed by `(url, sha256(text|model))`. Editing a
  page invalidates that one entry; the rest still hit.

## Programmatic use

```python
from site_audit.pipeline import PipelineConfig, run

summary = run(PipelineConfig(domain="example.com", max_pages=500))
print(summary)
```

## Limitations

* The crawler does not execute JavaScript — single-page apps need
  server-side rendering (or a snapshot service) to be auditable.
* Embedding ~5 000 pages on a CPU takes ~10 minutes the first time;
  subsequent runs are cache-bound.
* `faiss-cpu` wheels exist for x86_64 macOS / Linux. On Apple Silicon,
  `pip install faiss-cpu` will pull the universal2 wheel.

## License

MIT.
