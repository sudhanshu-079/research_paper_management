import numpy as np
from bson import ObjectId

from app.services import metrics
from app.services.registry import get_store
from app.services.search_service import bm25_rank, tokenize
from app.services.vector_store import FaissStore
from tests.conftest import make_pdf

GENOMICS = ("Abstract: We sequence tumour genomes and measure gene expression across patient cohorts. "
            "Protein pathways and mutation rates are compared between treatment groups in this clinical study.\n"
            "Keywords: genomics; cancer\n1. Introduction\nSequencing costs have dropped.")
VISION = ("Abstract: We train convolutional networks for image classification and object detection. "
          "Data augmentation and residual connections improve accuracy on benchmark image datasets considerably.\n"
          "Keywords: vision; cnn\n1. Introduction\nImages are everywhere.")


def upload(client, headers, title, doi, body):
    r = client.post("/papers/upload", files={"file": (f"{doi}.pdf", make_pdf(title, doi, body=body), "application/pdf")},
                    headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def two_papers(client, auth):
    g = upload(client, auth, "Tumour Genomics Study", "10.1/genomics", GENOMICS)
    v = upload(client, auth, "Image Classification with CNNs", "10.1/vision", VISION)
    return g, v


# ---- metrics ----
def test_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], {"a", "c"}
    assert metrics.precision_at_k(ranked, rel, 2) == 0.5
    assert metrics.recall_at_k(ranked, rel, 3) == 1.0
    assert metrics.reciprocal_rank(["x", "y", "a"], rel) == 1 / 3
    assert metrics.reciprocal_rank(["x"], rel) == 0.0
    assert abs(metrics.ndcg_at_k(ranked, rel, 3) - 0.9197) < 1e-3  # (1 + 0.5) / (1 + 1/log2(3))
    assert metrics.ndcg_at_k(["a", "c"], rel, 2) == 1.0


# ---- bm25 ----
def test_bm25_prefers_chunk_with_rare_query_terms():
    chunks = [{"text": "the cat sat on the mat"}, {"text": "genome sequencing of tumour cells"},
              {"text": "the the the the cat cat"}]
    ranked = bm25_rank("tumour genome", chunks)
    assert ranked[0][0] is chunks[1] and len(ranked) == 1
    assert bm25_rank("zzz", chunks) == []
    assert "the" not in tokenize("the genome")


# ---- vector store ----
def test_faiss_store_add_search_remove_persist(tmp_path):
    store = FaissStore(tmp_path, "m")
    vecs = np.eye(4, dtype="float32")
    store.add("u1", [10, 11, 12, 13], vecs)
    assert store.search("u1", vecs[2], 1)[0][0] == 12
    assert abs(store.search("u1", vecs[2], 1)[0][1] - 1.0) < 1e-6
    store.remove("u1", [12])
    assert store.count("u1") == 3
    assert 12 not in [i for i, _ in store.search("u1", vecs[2], 4)]
    assert FaissStore(tmp_path, "m").count("u1") == 3         # survives a restart
    assert store.search("someone-else", vecs[0], 3) == []     # per-user separation


# ---- indexing through the API ----
def test_upload_indexes_chunks(client, auth, db):
    pid = upload(client, auth, "Tumour Genomics Study", "10.1/g", GENOMICS)
    paper = client.get(f"/papers/{pid}", headers=auth).json()
    assert paper["index_status"] == "indexed"
    assert paper["embedding_model"].startswith("hashing")
    chunks = list(db.document_chunks.find({"paper_id": ObjectId(pid)}))
    assert chunks and all("vec_id" in c for c in chunks)
    assert len({c["vec_id"] for c in chunks}) == len(chunks)
    owner = str(db.users.find_one({})["_id"])
    assert get_store().count(owner) == len(chunks)


def test_reindex_does_not_duplicate_vectors(client, auth, db):
    pid = upload(client, auth, "Tumour Genomics Study", "10.1/g", GENOMICS)
    owner = str(db.users.find_one({})["_id"])
    before = get_store().count(owner)
    assert client.post(f"/papers/{pid}/index", headers=auth).status_code == 202
    assert get_store().count(owner) == before


def test_embedding_failure_is_recorded(client, auth, monkeypatch):
    from app.services import registry

    class Broken:
        name, dim = "broken", 8
        def encode(self, texts): raise RuntimeError("model missing")
    monkeypatch.setattr(registry, "_embedder", Broken())
    pid = upload(client, auth, "Tumour Genomics Study", "10.1/g", GENOMICS)
    p = client.get(f"/papers/{pid}", headers=auth).json()
    assert p["index_status"] == "failed"
    assert p["processing_status"] == "completed"   # extraction still succeeded


# ---- search ----
def test_all_modes_find_the_right_paper(client, auth):
    g, v = two_papers(client, auth)
    for mode in ("keyword", "semantic", "hybrid"):
        r = client.get("/search", params={"q": "gene expression in tumour", "mode": mode}, headers=auth).json()
        assert r["mode"] == mode
        assert r["results"][0]["paper_id"] == g, mode
        r = client.get("/search", params={"q": "image classification convolutional", "mode": mode}, headers=auth).json()
        assert r["results"][0]["paper_id"] == v, mode


def test_result_fields_and_matched_by(client, auth):
    two_papers(client, auth)
    hit = client.get("/search", params={"q": "tumour gene expression", "mode": "hybrid"}, headers=auth).json()["results"][0]
    assert set(hit["matched_by"]) == {"keyword", "semantic"}
    assert hit["keyword_score"] > 0 and -1 <= hit["semantic_score"] <= 1
    assert hit["page"] == 1 and "tumour" in hit["snippet"].lower()
    kw = client.get("/search", params={"q": "tumour", "mode": "keyword"}, headers=auth).json()["results"][0]
    assert kw["matched_by"] == ["keyword"] and kw["semantic_score"] is None


def test_each_paper_appears_once(client, auth):
    two_papers(client, auth)
    ids = [h["paper_id"] for h in client.get("/search", params={"q": "networks data genes", "mode": "hybrid"}, headers=auth).json()["results"]]
    assert len(ids) == len(set(ids))


def test_keyword_search_with_no_match_returns_nothing(client, auth):
    two_papers(client, auth)
    assert client.get("/search", params={"q": "blockchain", "mode": "keyword"}, headers=auth).json()["results"] == []


def test_search_is_isolated_per_user(client, auth, other_auth):
    two_papers(client, auth)
    for mode in ("keyword", "semantic", "hybrid"):
        r = client.get("/search", params={"q": "tumour gene expression", "mode": mode}, headers=other_auth)
        assert r.status_code == 200 and r.json()["results"] == [], mode


def test_deleted_paper_disappears_from_search(client, auth, db):
    g, v = two_papers(client, auth)
    owner = str(db.users.find_one({})["_id"])
    client.delete(f"/papers/{g}", headers=auth)
    for mode in ("keyword", "semantic", "hybrid"):
        ids = [h["paper_id"] for h in client.get("/search", params={"q": "tumour gene expression", "mode": mode}, headers=auth).json()["results"]]
        assert g not in ids, mode
    assert get_store().count(owner) == db.document_chunks.count_documents({"owner_id": db.users.find_one({})["_id"]})


def test_search_validation_and_auth(client, auth):
    assert client.get("/search", params={"q": "x"}, headers=auth).status_code == 422
    assert client.get("/search", params={"q": "ok", "mode": "magic"}, headers=auth).status_code == 422
    assert client.get("/search", params={"q": "ok"}).status_code == 401


def test_midway_failure_leaves_no_orphan_vectors(client, auth, db, monkeypatch):
    from app.services import indexing, registry

    class FailsSecondBatch:
        name, dim = "flaky", 256
        calls = 0
        def encode(self, texts):
            FailsSecondBatch.calls += 1
            if FailsSecondBatch.calls == 2:
                raise RuntimeError("out of memory")
            return registry.get_embedder.__globals__["HashingEmbedder"]().encode(texts)

    monkeypatch.setattr(indexing, "BATCH", 1)
    monkeypatch.setattr(registry, "_embedder", FailsSecondBatch())
    long_body = "Abstract: " + " ".join(f"sentence number {i} about gene sequencing" for i in range(24))
    pid = upload(client, auth, "Long Genomics Paper", "10.1/long", long_body)

    assert db.document_chunks.count_documents({"paper_id": ObjectId(pid)}) >= 2   # test really had 2+ batches
    assert FailsSecondBatch.calls == 2
    assert client.get(f"/papers/{pid}", headers=auth).json()["index_status"] == "failed"
    owner = str(db.users.find_one({})["_id"])
    assert get_store().count(owner) == 0
