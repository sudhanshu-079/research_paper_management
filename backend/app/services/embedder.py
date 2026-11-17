"""Text -> vector. Two interchangeable backends behind one tiny interface.

All vectors are L2-normalised, so inner product == cosine similarity.
"""
import re
import threading
import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. The model is loaded once, on first use (lazy),
    because loading takes seconds and ~100 MB of RAM."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()
        self.dim = 384 if "MiniLM-L6" in model_name else 0  # filled in after load

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
                    self._model = SentenceTransformer(self.name, device="cpu")
                    self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype("float32")


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words feature hashing. NOT semantic: it only captures word overlap.
    Exists so the app and tests run without downloading a model."""

    def __init__(self, dim: int = 256):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
