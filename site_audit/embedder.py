"""Embed pages with sentence-transformers, hitting the persistent cache first.

This is functionally the same trick used in the Hugo site's
``embedding_cache.py``: hash the embed text + model name and only call
the model for misses. We keep the model load lazy so cache-only runs
don't pay the import cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .cache import EmbeddingCache, content_hash

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"


@dataclass
class EmbedInput:
    url: str
    text: str  # the canonical "embed_text" — title + description + body


def embed_pages(
    inputs: Iterable[EmbedInput],
    cache: EmbeddingCache,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Return a matrix of L2-normalized embeddings, in the input order."""
    inputs = list(inputs)
    if not inputs:
        return np.zeros((0, 0), dtype=np.float32)

    # Build the hash table for cache lookups
    hashes = [content_hash(f"{p.text}|{model_name}") for p in inputs]
    embeddings: List[np.ndarray | None] = [None] * len(inputs)

    misses_idx: list[int] = []
    if use_cache:
        for i, (page, h) in enumerate(zip(inputs, hashes)):
            cached = cache.get(page.url, h)
            if cached is not None:
                embeddings[i] = cached
            else:
                misses_idx.append(i)
    else:
        misses_idx = list(range(len(inputs)))

    hits = len(inputs) - len(misses_idx)
    LOG.info("Embedding cache: %d hits / %d misses (model=%s)", hits, len(misses_idx), model_name)

    if misses_idx:
        from sentence_transformers import SentenceTransformer  # lazy

        model = SentenceTransformer(model_name, trust_remote_code=True)
        miss_texts = [inputs[i].text for i in misses_idx]
        new_embs = model.encode(
            miss_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        new_embs = np.asarray(new_embs, dtype=np.float32)
        for k, i in enumerate(misses_idx):
            emb = new_embs[k]
            embeddings[i] = emb
            cache.put(inputs[i].url, hashes[i], emb)
        cache.save()

    return np.stack(embeddings).astype(np.float32)
