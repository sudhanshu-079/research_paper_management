"""Embed a paper's chunks and put them in the user's vector index."""
from datetime import datetime, timezone

from app.services.registry import get_embedder, get_store

BATCH = 32


def _next_ids(db, n: int) -> list[int]:
    """Reserve n sequential integer ids (FAISS needs int64 ids, Mongo has ObjectIds)."""
    doc = db.counters.find_one_and_update(
        {"_id": "vec_id"}, {"$inc": {"seq": n}}, upsert=True, return_document=True)
    end = doc["seq"]
    return list(range(end - n + 1, end + 1))


def remove_paper_vectors(db, paper_id, owner_id) -> None:
    ids = [c["vec_id"] for c in db.document_chunks.find(
        {"paper_id": paper_id, "vec_id": {"$exists": True}}, {"vec_id": 1})]
    get_store().remove(str(owner_id), ids)


def index_paper(db, paper_id) -> None:
    """(Re)build vectors for one paper. Safe to call repeatedly."""
    paper = db.papers.find_one({"_id": paper_id})
    if not paper:
        return
    owner_id = paper["owner_id"]
    db.papers.update_one({"_id": paper_id}, {"$set": {"index_status": "indexing", "index_error": None}})
    try:
        remove_paper_vectors(db, paper_id, owner_id)  # drop old vectors on reindex
        chunks = list(db.document_chunks.find({"paper_id": paper_id}).sort("chunk_index", 1))
        if not chunks:
            raise ValueError("Paper has no text chunks to index")
        embedder, store = get_embedder(), get_store()
        ids = _next_ids(db, len(chunks))
        # Record the ids in Mongo BEFORE touching FAISS. If anything fails midway,
        # remove_paper_vectors() can still find and delete whatever was added.
        for c, vid in zip(chunks, ids):
            db.document_chunks.update_one({"_id": c["_id"]}, {"$set": {"vec_id": vid}})
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start:start + BATCH]
            vectors = embedder.encode([c["text"] for c in batch])
            store.add(str(owner_id), ids[start:start + BATCH], vectors)
        db.papers.update_one({"_id": paper_id}, {"$set": {
            "index_status": "indexed", "embedding_model": embedder.name,
            "indexed_at": datetime.now(timezone.utc)}})
    except Exception as exc:
        try:
            remove_paper_vectors(db, paper_id, owner_id)  # don't leave half-indexed vectors behind
        except Exception:
            pass
        db.papers.update_one({"_id": paper_id}, {"$set": {
            "index_status": "failed", "index_error": f"{type(exc).__name__}: {exc}"[:300]}})
