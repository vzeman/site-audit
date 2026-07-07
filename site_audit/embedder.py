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
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List

import numpy as np

from .cache import EmbeddingCache, content_hash

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBED_CACHE_SAVE_EVERY = 2048
DEFAULT_EMBED_MAX_SEQ_LENGTH = 512


def _quiet_model_load():
    if os.environ.get("SITE_AUDIT_MODEL_LOAD_LOGS"):
        return nullcontext()
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        return nullcontext()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(StringIO()))
    stack.enter_context(redirect_stderr(StringIO()))
    return stack


def _non_finite_row_indices(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(~np.isfinite(arr).all(axis=1))


@contextmanager
def _cpu_thread_boost():
    """Temporarily raise torch's intra-op threads for a CPU fallback encode.

    ``site_audit.__init__`` pins ``OMP_NUM_THREADS=1`` so faiss and torch can
    share libomp, which makes CPU encoding of this model ~100x slower than the
    hardware allows. ``torch.set_num_threads`` overrides the pin for torch
    only, and the previous value is restored afterwards.
    """
    try:
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            torch.set_num_threads(previous)
        except Exception:
            pass


def _embed_cache_save_every(default: int = DEFAULT_EMBED_CACHE_SAVE_EVERY) -> int:
    raw = os.environ.get("SITE_AUDIT_EMBED_CACHE_SAVE_EVERY")
    if raw is None or raw == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        LOG.warning(
            "Ignoring invalid SITE_AUDIT_EMBED_CACHE_SAVE_EVERY=%r; using %d",
            raw,
            default,
        )
        return default


def _reset_position_ids(auto_model) -> None:
    embeddings = getattr(auto_model, "embeddings", None)
    if embeddings is None or not hasattr(auto_model, "config"):
        return
    current = getattr(embeddings, "position_ids", None)
    max_positions = getattr(auto_model.config, "max_position_embeddings", None)
    if not max_positions:
        return

    import torch as _torch

    replacement = _torch.arange(max_positions, dtype=_torch.long)
    if getattr(current, "ndim", 1) == 2:
        replacement = replacement.unsqueeze(0)
    embeddings.register_buffer(
        "position_ids",
        replacement,
        persistent=False,
    )


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

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_seq_length: int | None = DEFAULT_EMBED_MAX_SEQ_LENGTH,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length if max_seq_length and max_seq_length > 0 else None
        self._model = None
        self._device: str | None = None

    def _ensure(self, device: str | None = None) -> None:
        target_device = device if device is not None else (os.environ.get("SITE_AUDIT_DEVICE") or None)
        if self._model is not None and self._device == target_device:
            return
        from sentence_transformers import SentenceTransformer  # lazy

        with _quiet_model_load():
            self._model = SentenceTransformer(
                self.model_name, trust_remote_code=True, device=target_device
            )
        self._device = target_device
        if self.max_seq_length is not None:
            current_max = getattr(self._model, "max_seq_length", None)
            if current_max is None or current_max > self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
        # On macOS (Apple Silicon), the gte-multilingual-base model's
        # position_ids buffer (persistent=False) appears to contain garbage
        # memory after loading rather than the expected arange values.
        # Reinitializing it here works around the issue.
        _am = self._model[0].auto_model
        _reset_position_ids(_am)

    def _encode_current_model(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(embs, dtype=np.float32)

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        arr = self._encode_current_model(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        bad = _non_finite_row_indices(arr)
        if bad.size == 0:
            return arr
        if self._device != "cpu":
            # MPS returns NaN rows under allocator pressure instead of
            # raising; freeing its cache and re-encoding just the bad rows
            # usually recovers without leaving the accelerator.
            LOG.warning(
                "Embedding model returned %d/%d non-finite vectors on %s; "
                "clearing accelerator cache and retrying those rows",
                bad.size, len(texts), self._device or "auto",
            )
            self._clear_accelerator_cache()
            self._reencode_rows(arr, texts, bad, batch_size=batch_size)
            bad = _non_finite_row_indices(arr)
            if bad.size == 0:
                return arr
            # Only the still-bad rows go to CPU — re-encoding the whole
            # chunk there takes hours under the OMP_NUM_THREADS=1 pin.
            LOG.warning(
                "%d vectors still non-finite after accelerator retry; "
                "re-encoding those rows on CPU",
                bad.size,
            )
            self._model = None
            self._ensure("cpu")
            with _cpu_thread_boost():
                self._reencode_rows(arr, texts, bad, batch_size=batch_size)
            bad = _non_finite_row_indices(arr)
            if bad.size == 0:
                return arr
        raise RuntimeError("Embedding model returned non-finite vectors")

    def _reencode_rows(
        self,
        arr: np.ndarray,
        texts: list[str],
        indices: np.ndarray,
        *,
        batch_size: int,
    ) -> None:
        retry = self._encode_current_model(
            [texts[i] for i in indices],
            batch_size=batch_size,
            show_progress=False,
        )
        for k, i in enumerate(indices):
            arr[i] = retry[k]

    def _clear_accelerator_cache(self) -> None:
        try:
            import torch

            if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # best-effort; the CPU fallback still applies
            pass

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

        cache_fingerprint = f"{self.model_name}|seq={self.max_seq_length or 'model'}"
        hashes = [content_hash(f"{p.text}|{cache_fingerprint}") for p in inputs]
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
            save_every = _embed_cache_save_every()
            for offset in range(0, len(misses_idx), save_every):
                chunk_idx = misses_idx[offset : offset + save_every]
                miss_texts = [inputs[i].text for i in chunk_idx]
                new_embs = self.encode(
                    miss_texts,
                    batch_size=batch_size,
                    show_progress=True,
                )
                for k, i in enumerate(chunk_idx):
                    emb = new_embs[k]
                    embeddings[i] = emb
                    cache.put(inputs[i].url, hashes[i], emb)
                cache.save()
                LOG.info(
                    "Embedding cache persisted: %d / %d misses",
                    min(offset + len(chunk_idx), len(misses_idx)),
                    len(misses_idx),
                )

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
