"""Process-wide singletons for the embedder and vector store (so the model loads once)."""
from app.config import settings
from app.services.embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from app.services.vector_store import FaissStore

_embedder: Embedder | None = None
_store: FaissStore | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if settings.embedding_backend == "hashing":
            _embedder = HashingEmbedder()
        else:
            _embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return _embedder


def get_store() -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore(settings.vector_dir, get_embedder().name)
    return _store


def override(embedder=None, store=None) -> None:
    """Used by tests."""
    global _embedder, _store
    _embedder, _store = embedder, store
