"""Embed text with sentence-transformers.

Two surfaces:

* :class:`Embedder` loads the model lazily, exposes ``encode(texts)``
  for ad-hoc batches (queries, headings, …) and ``encode_pages(pages, cache)``
  for the cached page-embedding path used by the pipeline.

* :func:`embed_pages` is kept as a thin wrapper for backwards compat.

Caching is per-page only (queries are usually < 1 000, not worth a cache).
"""

from __future__ import annotations

import logging
import os
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List

import numpy as np

from .cache import EmbeddingCache, content_hash

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"


def _quiet_model_load():
    if os.environ.get("SITE_AUDIT_MODEL_LOAD_LOGS"):
        return nullcontext()
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        return nullcontext()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(StringIO()))
    stack.enter_context(redirect_stderr(StringIO()))
    return stack


@dataclass
class EmbedInput:
    url: str
    text: str  # canonical "embed_text" — title + description + body


class Embedder:
    """Lazy wrapper around a sentence-transformer model.

    The model is loaded the first time ``encode`` is called and reused
    for every subsequent call in the same process. This matters because
    callers (page embedding, query embedding) want the same vector space
    without paying the load cost twice.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy

        device = os.environ.get("SITE_AUDIT_DEVICE") or None
        with _quiet_model_load():
            self._model = SentenceTransformer(
                self.model_name, trust_remote_code=True, device=device
            )
        # On macOS (Apple Silicon), the gte-multilingual-base model's
        # position_ids buffer (persistent=False) appears to contain garbage
        # memory after loading rather than the expected arange values.
        # Reinitializing it here works around the issue.
        import torch as _torch
        _am = self._model[0].auto_model
        _am.embeddings.register_buffer(
            "position_ids",
            _torch.arange(_am.config.max_position_embeddings, dtype=_torch.long),
            persistent=False,
        )
        # Cap the input sequence length. gte-multilingual-base allows up to
        # 8192 tokens, but a long page (× a 32-item batch) makes the attention
        # tensor large enough to exhaust the Apple-Silicon MPS allocator, which
        # then silently returns NaN embeddings instead of erroring. A page-level
        # topical embedding only needs the leading content, so we bound the
        # sequence length (override with SITE_AUDIT_MAX_SEQ_LENGTH). This keeps
        # MPS fast and NaN-free without measurably hurting embedding quality.
        try:
            max_seq = int(os.environ.get("SITE_AUDIT_MAX_SEQ_LENGTH", "1024"))
        except ValueError:
            max_seq = 1024
        if max_seq > 0 and (self._model.max_seq_length or 0) > max_seq:
            self._model.max_seq_length = max_seq

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        embs = np.asarray(embs, dtype=np.float32)
        # Guard against silent corruption: the MPS backend can return NaN
        # embeddings under memory pressure rather than raising. Caching those
        # poisons every downstream metric, so fail loudly instead.
        if embs.size and not np.isfinite(embs).all():
            n_bad = int((~np.isfinite(embs)).any(axis=1).sum())
            raise RuntimeError(
                f"Embedding model produced {n_bad}/{len(embs)} non-finite vectors "
                f"(device={getattr(self._model, 'device', '?')}, "
                f"max_seq_length={self._model.max_seq_length}). This usually means "
                "the accelerator ran out of memory; lower SITE_AUDIT_MAX_SEQ_LENGTH "
                "or set SITE_AUDIT_DEVICE=cpu."
            )
        return embs

    def encode_pages(
        self,
        inputs: Iterable[EmbedInput],
        cache: EmbeddingCache,
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        inputs = list(inputs)
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)

        hashes = [content_hash(f"{p.text}|{self.model_name}") for p in inputs]
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
        LOG.info(
            "Embedding cache: %d hits / %d misses (model=%s)",
            hits, len(misses_idx), self.model_name,
        )

        if misses_idx:
            miss_texts = [inputs[i].text for i in misses_idx]
            new_embs = self.encode(miss_texts, batch_size=batch_size, show_progress=True)
            for k, i in enumerate(misses_idx):
                emb = new_embs[k]
                embeddings[i] = emb
                cache.put(inputs[i].url, hashes[i], emb)
            cache.save()

        return np.stack(embeddings).astype(np.float32)


def embed_pages(
    inputs: Iterable[EmbedInput],
    cache: EmbeddingCache,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Backwards-compatible wrapper around ``Embedder.encode_pages``."""
    return Embedder(model_name).encode_pages(inputs, cache, use_cache, batch_size)
