import numpy as np

from site_audit.embedder import EmbedInput, Embedder


class FakeEmbeddingCache:
    def __init__(self) -> None:
        self.entries = {}
        self.save_count = 0

    def get(self, url: str, hash_: str):
        entry = self.entries.get(url)
        if entry and entry[0] == hash_:
            return entry[1]
        return None

    def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
        self.entries[url] = (hash_, embedding)

    def save(self) -> None:
        self.save_count += 1


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__("fake-model")

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        return np.asarray([[float(len(text)), float(i)] for i, text in enumerate(texts)], dtype=np.float32)


def test_encode_pages_persists_embedding_cache_in_chunks(monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_EMBED_CACHE_SAVE_EVERY", "2")
    cache = FakeEmbeddingCache()
    inputs = [
        EmbedInput(url=f"https://example.com/{i}", text="x" * (i + 1))
        for i in range(5)
    ]

    embeddings = FakeEmbedder().encode_pages(inputs, cache, use_cache=True, batch_size=2)

    assert embeddings.shape == (5, 2)
    assert cache.save_count == 3
    assert set(cache.entries) == {item.url for item in inputs}
