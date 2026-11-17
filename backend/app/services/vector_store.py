"""Local FAISS vector store, one index file per (user, embedding model).

Per-user indexes mean a search can only ever see that user's vectors, so
access control doesn't depend on filtering after the fact.

Interface is small on purpose (add / remove / search) so another vector DB can replace it.
The index is a cache: it can always be rebuilt from `document_chunks`.
"""
import os
import re
import threading
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, directory: Path, model_name: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
        self._lock = threading.Lock()
        self._cache: dict[str, faiss.Index] = {}

    def _path(self, owner_id: str) -> Path:
        return self.dir / f"{owner_id}__{self.slug}.faiss"

    def _load(self, owner_id: str, dim: int | None = None):
        if owner_id in self._cache:
            return self._cache[owner_id]
        path = self._path(owner_id)
        if path.exists():
            index = faiss.read_index(str(path))
        elif dim is not None:
            # IDMap lets us attach our own integer ids and remove vectors later
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            return None
        self._cache[owner_id] = index
        return index

    def _save(self, owner_id: str):
        tmp = self._path(owner_id).with_suffix(".tmp")
        faiss.write_index(self._cache[owner_id], str(tmp))
        os.replace(tmp, self._path(owner_id))  # atomic swap, no half-written index on crash

    def add(self, owner_id: str, ids: list[int], vectors: np.ndarray) -> None:
        with self._lock:
            index = self._load(owner_id, dim=vectors.shape[1])
            if index.d != vectors.shape[1]:
                raise ValueError(f"Vector dim {vectors.shape[1]} != index dim {index.d}")
            index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
            self._save(owner_id)

    def remove(self, owner_id: str, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            index = self._load(owner_id)
            if index is None:
                return
            index.remove_ids(np.asarray(ids, dtype="int64"))
            self._save(owner_id)

    def search(self, owner_id: str, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            index = self._load(owner_id)
            if index is None or index.ntotal == 0:
                return []
            scores, ids = index.search(query.reshape(1, -1), min(k, index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def count(self, owner_id: str) -> int:
        with self._lock:
            index = self._load(owner_id)
            return 0 if index is None else index.ntotal
