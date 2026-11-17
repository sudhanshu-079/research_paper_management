from pymongo import ASCENDING, DESCENDING


def create_indexes(db) -> None:
    db.users.create_index("email", unique=True)

    # Listing: a user's papers sorted by date
    db.papers.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
    # Duplicate detection. Partial indexes so papers without a DOI/file don't collide on null.
    db.papers.create_index(
        [("owner_id", ASCENDING), ("doi", ASCENDING)], unique=True,
        partialFilterExpression={"doi": {"$type": "string"}}, name="uniq_owner_doi")
    db.papers.create_index(
        [("owner_id", ASCENDING), ("file_hash", ASCENDING)], unique=True,
        partialFilterExpression={"file_hash": {"$type": "string"}}, name="uniq_owner_filehash")
    db.papers.create_index([("owner_id", ASCENDING), ("year", ASCENDING)])

    db.document_chunks.create_index([("paper_id", ASCENDING), ("chunk_index", ASCENDING)])
    db.document_chunks.create_index("owner_id")

    db.chat_sessions.create_index([("owner_id", ASCENDING), ("updated_at", DESCENDING)])
    db.chat_messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])

    db.research_comparisons.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
    db.research_gaps.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
