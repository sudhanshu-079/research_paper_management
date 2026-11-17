from pymongo import MongoClient
from pymongo.database import Database
from app.config import settings

_client: MongoClient | None = None
_db: Database | None = None


def get_db() -> Database:
    global _client, _db
    if _db is None:
        _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        _db = _client[settings.db_name]
    return _db


def set_db(db) -> None:
    """Swap the database (used by tests)."""
    global _db
    _db = db


def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
    _client = _db = None
