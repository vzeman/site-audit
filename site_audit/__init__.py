"""site_audit — crawl any website, embed its pages, surface duplicates, outliers,
topic clusters, GEO scoring, and internal-link recommendations.

We set a few env vars BEFORE importing numpy / torch / faiss because on
macOS the Apple Silicon wheels of faiss-cpu and pytorch both ship their
own libomp, and OpenMP refuses to coexist with itself — the process
silently aborts when the second copy initializes. ``KMP_DUPLICATE_LIB_OK``
makes that a warning instead, ``OMP_NUM_THREADS=1`` keeps both runtimes
out of each other's way, and the others are just polite hygiene.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.2.0"
