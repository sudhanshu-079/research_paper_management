import mongomock
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import connection
from app.database.indexes import create_indexes
from app.main import app
from app.services import registry
from app.services.embedder import HashingEmbedder
from app.services.vector_store import FaissStore


@pytest.fixture()
def db():
    # mongomock is an in-memory fake. It does NOT enforce $jsonSchema validators,
    # so schema enforcement is only checked against a real mongod (see docs).
    database = mongomock.MongoClient()["test_db"]
    create_indexes(database)
    connection.set_db(database)
    yield database
    connection.set_db(None)


@pytest.fixture()
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    embedder = HashingEmbedder()
    registry.override(embedder=embedder, store=FaissStore(tmp_path / "vectors", embedder.name))
    yield TestClient(app)
    registry.override(None, None)  # not used as a context manager, so lifespan (real Mongo) is skipped


def register(client, email="a@example.com", password="Password1", name="Alice"):
    r = client.post("/auth/register", json={"email": email, "password": password, "full_name": name})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def auth(client):
    return register(client)


@pytest.fixture()
def other_auth(client):
    return register(client, email="b@example.com", name="Bob")


def make_pdf(title="Deep Learning for Citation Analysis", doi="10.1234/abc.2021.001",
             with_text=True, body=None, author_line=None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if with_text:
        page.insert_text((72, 80), title, fontsize=22)
        if author_line:
            page.insert_text((72, 112), author_line, fontsize=11)
        metadata_y = 132 if author_line else 120
        body_y = 160 if author_line else 150
        page.insert_text((72, metadata_y), f"Published 2021. DOI: {doi}", fontsize=10)
        body = body or ("Abstract: We study how neural networks can classify scientific citations. "
                "Our experiments on a public dataset show clear improvements over baselines, "
                "and we discuss limitations and directions for future work in detail here.\n"
                "Keywords: deep learning; citations; NLP\n"
                "1. Introduction\nScientific literature is growing quickly.")
        page.insert_textbox(pymupdf.Rect(72, body_y, 520, 400), body, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data
