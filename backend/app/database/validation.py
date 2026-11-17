"""MongoDB JSON Schema validators (strict) for each collection."""

USERS_SCHEMA = {
    "bsonType": "object",
    "required": ["email", "password_hash", "full_name", "created_at"],
    "properties": {
        "email": {"bsonType": "string", "pattern": "^.+@.+$"},
        "password_hash": {"bsonType": "string"},
        "full_name": {"bsonType": "string", "minLength": 1, "maxLength": 120},
        "created_at": {"bsonType": "date"},
    },
}

PAPERS_SCHEMA = {
    "bsonType": "object",
    "required": ["owner_id", "title", "authors", "reading_status",
                 "processing_status", "created_at", "updated_at"],
    "properties": {
        "owner_id": {"bsonType": "objectId"},
        "title": {"bsonType": "string", "minLength": 1, "maxLength": 500},
        "authors": {"bsonType": "array", "items": {"bsonType": "string"}},
        "abstract": {"bsonType": ["string", "null"]},
        "publication": {"bsonType": ["string", "null"]},
        "year": {"bsonType": ["int", "null"], "minimum": 1800, "maximum": 2100},
        "doi": {"bsonType": ["string", "null"]},
        "keywords": {"bsonType": "array", "items": {"bsonType": "string"}},
        "tags": {"bsonType": "array", "items": {"bsonType": "string"}},
        "domain": {"bsonType": ["string", "null"]},
        "notes": {"bsonType": ["string", "null"]},
        "file_hash": {"bsonType": ["string", "null"]},
        "file_path": {"bsonType": ["string", "null"]},
        "reading_status": {"enum": ["unread", "reading", "read"]},
        "processing_status": {"enum": ["none", "uploaded", "processing", "completed", "failed"]},
        "needs_review": {"bsonType": "bool"},
        "created_at": {"bsonType": "date"},
        "updated_at": {"bsonType": "date"},
    },
}

CHUNKS_SCHEMA = {
    "bsonType": "object",
    "required": ["paper_id", "owner_id", "chunk_index", "text"],
    "properties": {
        "paper_id": {"bsonType": "objectId"},
        "owner_id": {"bsonType": "objectId"},
        "chunk_index": {"bsonType": "int"},
        "page": {"bsonType": ["int", "null"]},
        "text": {"bsonType": "string"},
    },
}

SESSIONS_SCHEMA = {
    "bsonType": "object",
    "required": ["owner_id", "title", "paper_ids", "created_at", "updated_at"],
    "properties": {
        "owner_id": {"bsonType": "objectId"},
        "title": {"bsonType": "string", "minLength": 1, "maxLength": 200},
        "paper_ids": {"bsonType": "array", "items": {"bsonType": "objectId"}},
        "created_at": {"bsonType": "date"},
        "updated_at": {"bsonType": "date"},
    },
}

MESSAGES_SCHEMA = {
    "bsonType": "object",
    "required": ["session_id", "owner_id", "role", "content", "created_at"],
    "properties": {
        "session_id": {"bsonType": "objectId"},
        "owner_id": {"bsonType": "objectId"},
        "role": {"enum": ["user", "assistant"]},
        "content": {"bsonType": "string"},
        "supported": {"bsonType": ["bool", "null"]},
        "citations": {"bsonType": "array"},
        "created_at": {"bsonType": "date"},
    },
}

COMPARISONS_SCHEMA = {
    "bsonType": "object",
    "required": ["owner_id", "title", "paper_ids", "columns", "rows", "created_at"],
    "properties": {
        "owner_id": {"bsonType": "objectId"},
        "title": {"bsonType": "string", "minLength": 1, "maxLength": 200},
        "paper_ids": {"bsonType": "array", "minItems": 2, "maxItems": 5, "items": {"bsonType": "objectId"}},
        "columns": {"bsonType": "array"},
        "rows": {"bsonType": "array"},
        "observations": {"bsonType": "array"},
        "created_at": {"bsonType": "date"},
    },
}

GAPS_SCHEMA = {
    "bsonType": "object",
    "required": ["owner_id", "title", "paper_ids", "gaps", "created_at"],
    "properties": {
        "owner_id": {"bsonType": "objectId"},
        "title": {"bsonType": "string", "minLength": 1, "maxLength": 200},
        "paper_ids": {"bsonType": "array", "items": {"bsonType": "objectId"}},
        "gaps": {"bsonType": "array"},
        "created_at": {"bsonType": "date"},
    },
}

VALIDATORS = {"research_comparisons": COMPARISONS_SCHEMA, "research_gaps": GAPS_SCHEMA, "chat_sessions": SESSIONS_SCHEMA, "chat_messages": MESSAGES_SCHEMA, "users": USERS_SCHEMA, "papers": PAPERS_SCHEMA, "document_chunks": CHUNKS_SCHEMA}


def apply_validation(db) -> None:
    """Create collections with strict validators, or update existing ones via collMod."""
    existing = set(db.list_collection_names())
    for name, schema in VALIDATORS.items():
        validator = {"$jsonSchema": schema}
        if name in existing:
            db.command("collMod", name, validator=validator,
                       validationLevel="strict", validationAction="error")
        else:
            db.create_collection(name, validator=validator,
                                 validationLevel="strict", validationAction="error")
