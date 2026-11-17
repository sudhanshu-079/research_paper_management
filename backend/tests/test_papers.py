from tests.conftest import make_pdf

PAPER = {"title": "Attention Is All You Need", "authors": ["Vaswani", "Shazeer"], "year": 2017,
         "doi": "https://doi.org/10.5555/3295222.3295349", "keywords": ["transformer", "nlp"]}


def test_create_get_update_delete(client, auth):
    r = client.post("/papers", json=PAPER, headers=auth)
    assert r.status_code == 201
    paper = r.json()
    assert paper["doi"] == "10.5555/3295222.3295349"  # normalised
    pid = paper["id"]

    assert client.get(f"/papers/{pid}", headers=auth).json()["title"] == PAPER["title"]

    r = client.patch(f"/papers/{pid}", json={"reading_status": "read", "notes": "great"}, headers=auth)
    assert r.json()["reading_status"] == "read" and r.json()["notes"] == "great"
    assert r.json()["title"] == PAPER["title"]  # untouched fields stay

    assert client.delete(f"/papers/{pid}", headers=auth).status_code == 204
    assert client.get(f"/papers/{pid}", headers=auth).status_code == 404


def test_validation_errors(client, auth):
    assert client.post("/papers", json={"title": ""}, headers=auth).status_code == 422
    assert client.post("/papers", json={"title": "x", "year": 1500}, headers=auth).status_code == 422
    assert client.post("/papers", json={"title": "x", "doi": "notadoi"}, headers=auth).status_code == 422
    assert client.post("/papers", json={"title": "x", "reading_status": "done"}, headers=auth).status_code == 422


def test_duplicate_doi_rejected(client, auth):
    assert client.post("/papers", json=PAPER, headers=auth).status_code == 201
    dup = {**PAPER, "title": "Different title", "doi": "10.5555/3295222.3295349"}
    assert client.post("/papers", json=dup, headers=auth).status_code == 409


def test_search_filter_sort_paginate(client, auth):
    for i, (title, year, status) in enumerate([("Graph Neural Networks", 2020, "read"),
                                               ("Vision Transformers", 2021, "unread"),
                                               ("Graph Attention", 2018, "unread")]):
        client.post("/papers", json={"title": title, "year": year, "reading_status": status}, headers=auth)

    r = client.get("/papers", params={"q": "graph"}, headers=auth).json()
    assert r["total"] == 2
    r = client.get("/papers", params={"reading_status": "unread", "sort": "year", "order": "asc"}, headers=auth).json()
    assert [p["year"] for p in r["items"]] == [2018, 2021]
    r = client.get("/papers", params={"page": 2, "page_size": 2}, headers=auth).json()
    assert r["total"] == 3 and len(r["items"]) == 1
    # regex special characters are treated literally
    assert client.get("/papers", params={"q": ".*"}, headers=auth).json()["total"] == 0
    assert client.get("/papers", params={"sort": "owner_id"}, headers=auth).status_code == 422


def test_users_cannot_access_each_others_papers(client, auth, other_auth):
    pid = client.post("/papers", json=PAPER, headers=auth).json()["id"]

    assert client.get(f"/papers/{pid}", headers=other_auth).status_code == 404
    assert client.patch(f"/papers/{pid}", json={"title": "hacked"}, headers=other_auth).status_code == 404
    assert client.delete(f"/papers/{pid}", headers=other_auth).status_code == 404
    assert client.get("/papers", headers=other_auth).json()["total"] == 0
    # still intact for the owner
    assert client.get(f"/papers/{pid}", headers=auth).json()["title"] == PAPER["title"]
    # same DOI is allowed for a different user
    assert client.post("/papers", json=PAPER, headers=other_auth).status_code == 201


def test_invalid_id_is_404(client, auth):
    assert client.get("/papers/not-an-objectid", headers=auth).status_code == 404


def test_upload_pdf_extracts_metadata_and_chunks(client, auth, db):
    r = client.post("/papers/upload", files={"file": ("x.pdf", make_pdf(), "application/pdf")}, headers=auth)
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["processing_status"] == "completed"
    assert p["title"] == "Deep Learning for Citation Analysis"
    assert p["doi"] == "10.1234/abc.2021.001"
    assert p["year"] == 2021
    assert "neural networks" in p["abstract"]
    assert "deep learning" in p["keywords"]
    assert p["needs_review"] is True and p["has_file"] is True
    assert db.document_chunks.count_documents({"paper_id": __import__("bson").ObjectId(p["id"])}) >= 1


def test_upload_pdf_extracts_visible_author_names(client, auth):
    response = client.post(
        "/papers/upload",
        files={"file": ("paper.pdf", make_pdf(author_line="Jane Doe, John Smith"), "application/pdf")},
        headers=auth,
    )
    assert response.status_code == 201
    assert response.json()["authors"] == ["Jane Doe", "John Smith"]


def test_pdf_file_download_is_available_only_to_owner(client, auth, other_auth):
    uploaded = client.post(
        "/papers/upload",
        files={"file": ("paper.pdf", make_pdf(), "application/pdf")},
        headers=auth,
    )
    assert uploaded.status_code == 201
    paper_id = uploaded.json()["id"]

    response = client.get(f"/papers/{paper_id}/file", headers=auth)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    assert client.get(f"/papers/{paper_id}/file", headers=other_auth).status_code == 404


def test_pdf_author_extraction_prefers_metadata_and_checks_front_page():
    from app.services.pdf_service import _extract_authors

    assert _extract_authors("Jane Doe; John Smith", "", None) == ["Jane Doe", "John Smith"]
    assert _extract_authors(
        None,
        "A Study of Research\nJane Doe, John Smith\nDepartment of Computing\nAbstract: Findings",
        "A Study of Research",
    ) == ["Jane Doe", "John Smith"]


def test_refresh_metadata_can_extract_authors_from_existing_pdf(client, auth, db):
    from bson import ObjectId

    uploaded = client.post(
        "/papers/upload",
        files={"file": ("paper.pdf", make_pdf(author_line="Jane Doe, John Smith"), "application/pdf")},
        headers=auth,
    )
    paper_id = uploaded.json()["id"]
    db.papers.update_one({"_id": ObjectId(paper_id)}, {"$set": {"authors": []}})

    refreshed = client.post(f"/papers/{paper_id}/refresh-metadata", headers=auth)
    assert refreshed.status_code == 200
    assert refreshed.json()["authors"] == ["Jane Doe", "John Smith"]


def test_upload_rejects_non_pdf_and_oversize(client, auth, monkeypatch):
    r = client.post("/papers/upload", files={"file": ("a.pdf", b"not a pdf at all", "application/pdf")}, headers=auth)
    assert r.status_code == 400
    from app.config import settings
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    r = client.post("/papers/upload", files={"file": ("a.pdf", make_pdf(), "application/pdf")}, headers=auth)
    assert r.status_code == 413


def test_upload_duplicate_file_rejected(client, auth):
    pdf = make_pdf()
    assert client.post("/papers/upload", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=auth).status_code == 201
    assert client.post("/papers/upload", files={"file": ("b.pdf", pdf, "application/pdf")}, headers=auth).status_code == 409


def test_scanned_pdf_is_flagged_not_silently_empty(client, auth):
    r = client.post("/papers/upload", files={"file": ("s.pdf", make_pdf(with_text=False), "application/pdf")}, headers=auth)
    assert r.status_code == 201
    assert r.json()["processing_status"] == "failed"
    assert "OCR" in r.json()["processing_error"]


def test_malicious_filename_does_not_escape_upload_dir(client, auth, tmp_path):
    r = client.post("/papers/upload", files={"file": ("../../evil.pdf", make_pdf(), "application/pdf")}, headers=auth)
    assert r.status_code == 201
    assert not (tmp_path / "evil.pdf").exists()
    assert all(tmp_path in p.parents for p in (tmp_path / "uploads").rglob("*.pdf"))


def test_delete_removes_file_and_chunks(client, auth, db):
    p = client.post("/papers/upload", files={"file": ("a.pdf", make_pdf(), "application/pdf")}, headers=auth).json()
    from app.config import settings
    assert list(settings.upload_dir.rglob("*.pdf"))
    client.delete(f"/papers/{p['id']}", headers=auth)
    assert not list(settings.upload_dir.rglob("*.pdf"))
    assert db.document_chunks.count_documents({}) == 0
