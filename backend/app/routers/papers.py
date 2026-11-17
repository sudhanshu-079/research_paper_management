import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.config import settings
from app.database.connection import get_db
from app.deps import current_user, to_object_id
from app.schemas.paper import PaperCreate, PaperList, PaperOut, PaperUpdate, normalize_doi
from app.services import indexing, pdf_service, storage

router = APIRouter(prefix="/papers", tags=["papers"])

SORTABLE = {"created_at", "year", "title"}


def _now():
    return datetime.now(timezone.utc)


def _out(p: dict) -> PaperOut:
    return PaperOut(
        id=str(p["_id"]), title=p["title"], authors=p.get("authors", []),
        abstract=p.get("abstract"), publication=p.get("publication"), year=p.get("year"),
        doi=p.get("doi"), keywords=p.get("keywords", []), tags=p.get("tags", []),
        domain=p.get("domain"), notes=p.get("notes"),
        reading_status=p["reading_status"], processing_status=p["processing_status"],
        needs_review=p.get("needs_review", False), has_file=bool(p.get("file_path")),
        processing_error=p.get("processing_error"),
        index_status=p.get("index_status"), embedding_model=p.get("embedding_model"),
        created_at=p["created_at"], updated_at=p["updated_at"],
    )


def _get_owned(paper_id: str, user: dict) -> dict:
    """Fetch a paper only if it belongs to the user. Other people's papers look like 404s."""
    paper = get_db().papers.find_one({"_id": to_object_id(paper_id), "owner_id": user["_id"]})
    if not paper:
        raise HTTPException(404, "Paper not found")
    return paper


def _new_paper_doc(owner_id, data: dict) -> dict:
    now = _now()
    return {
        "owner_id": owner_id, "title": data["title"], "authors": data.get("authors", []),
        "abstract": data.get("abstract"), "publication": data.get("publication"),
        "year": data.get("year"), "doi": data.get("doi"),
        "keywords": data.get("keywords", []), "tags": data.get("tags", []),
        "domain": data.get("domain"), "notes": data.get("notes"),
        "file_hash": None, "file_path": None,
        "reading_status": data.get("reading_status", "unread"),
        "processing_status": "none", "needs_review": False,
        "created_at": now, "updated_at": now,
    }


@router.post("", response_model=PaperOut, status_code=201)
def create_paper(body: PaperCreate, user: dict = Depends(current_user)):
    doc = _new_paper_doc(user["_id"], body.model_dump())
    try:
        doc["_id"] = get_db().papers.insert_one(doc).inserted_id
    except DuplicateKeyError:
        raise HTTPException(409, "You already have a paper with this DOI")
    return _out(doc)


@router.get("", response_model=PaperList)
def list_papers(
    q: str | None = None,
    reading_status: str | None = Query(None, pattern="^(unread|reading|read)$"),
    year: int | None = None,
    tag: str | None = None,
    sort: str = "created_at",
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict = Depends(current_user),
):
    if sort not in SORTABLE:
        raise HTTPException(422, f"sort must be one of {sorted(SORTABLE)}")
    query: dict = {"owner_id": user["_id"]}
    if q:
        rx = {"$regex": re.escape(q.strip()), "$options": "i"}
        query["$or"] = [{f: rx} for f in ("title", "authors", "keywords", "publication", "doi", "domain")]
    if reading_status:
        query["reading_status"] = reading_status
    if year:
        query["year"] = year
    if tag:
        query["tags"] = tag

    papers = get_db().papers
    total = papers.count_documents(query)
    cursor = (papers.find(query)
              .sort(sort, ASCENDING if order == "asc" else DESCENDING)
              .skip((page - 1) * page_size).limit(page_size))
    return PaperList(items=[_out(p) for p in cursor], total=total, page=page, page_size=page_size)


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str, user: dict = Depends(current_user)):
    return _out(_get_owned(paper_id, user))


@router.post("/{paper_id}/refresh-metadata", response_model=PaperOut)
def refresh_metadata(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_owned(paper_id, user)
    path = storage.resolve_pdf_file(paper.get("file_path"))
    if path is None:
        raise HTTPException(409, "This paper has no available PDF file")
    try:
        extracted = pdf_service.extract_pdf(str(path))
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF metadata: {type(exc).__name__}") from exc

    changes = {}
    if extracted.authors and not paper.get("authors"):
        changes["authors"] = extracted.authors
    if extracted.keywords and not paper.get("keywords"):
        changes["keywords"] = extracted.keywords
    if changes:
        changes["updated_at"] = _now()
        get_db().papers.update_one({"_id": paper["_id"]}, {"$set": changes})
    return _out(_get_owned(paper_id, user))


@router.get("/{paper_id}/file")
def get_paper_file(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_owned(paper_id, user)
    path = storage.resolve_pdf_file(paper.get("file_path"))
    if path is None:
        raise HTTPException(404, "PDF file not found")
    filename = re.sub(r"[^A-Za-z0-9._ -]", "", paper["title"]).strip() or "paper"
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{filename}.pdf",
        content_disposition_type="inline",
    )


@router.patch("/{paper_id}", response_model=PaperOut)
def update_paper(paper_id: str, body: PaperUpdate, user: dict = Depends(current_user)):
    paper = _get_owned(paper_id, user)
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        return _out(paper)
    changes["updated_at"] = _now()
    try:
        get_db().papers.update_one({"_id": paper["_id"]}, {"$set": changes})
    except DuplicateKeyError:
        raise HTTPException(409, "You already have a paper with this DOI")
    return _out(_get_owned(paper_id, user))


@router.delete("/{paper_id}", status_code=204)
def delete_paper(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_owned(paper_id, user)
    db = get_db()
    indexing.remove_paper_vectors(db, paper["_id"], paper["owner_id"])
    db.document_chunks.delete_many({"paper_id": paper["_id"]})
    db.papers.delete_one({"_id": paper["_id"]})
    storage.delete_file(paper.get("file_path"))


@router.post("/upload", response_model=PaperOut, status_code=201)
async def upload_paper(background: BackgroundTasks, file: UploadFile = File(...),
                       user: dict = Depends(current_user)):
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    # Check the actual bytes, not just the filename or content-type the client claims
    if not data.startswith(b"%PDF-"):
        raise HTTPException(400, "Only PDF files are accepted")

    owner_id = user["_id"]
    file_hash = hashlib.sha256(data).hexdigest()
    db = get_db()
    if db.papers.find_one({"owner_id": owner_id, "file_hash": file_hash}):
        raise HTTPException(409, "This PDF is already in your library")

    path = storage.save_pdf(str(owner_id), data)
    fallback_title = Path(file.filename or "Untitled").stem[:200] or "Untitled"
    doc = _new_paper_doc(owner_id, {"title": fallback_title})
    doc.update(file_hash=file_hash, file_path=path, processing_status="processing", needs_review=True)

    try:
        extracted = pdf_service.extract_pdf(path)
    except Exception as exc:  # corrupt or encrypted PDF
        doc.update(processing_status="failed", processing_error=f"Could not read PDF: {type(exc).__name__}")
        extracted = None

    chunks = []
    if extracted is not None:
        if extracted.total_chars < 200:
            doc.update(processing_status="failed",
                       processing_error="No extractable text. This looks like a scanned PDF and needs OCR.")
        else:
            doc["title"] = (extracted.title or fallback_title)[:500]
            doc["authors"] = extracted.authors
            doc["abstract"] = extracted.abstract
            doc["year"] = extracted.year
            doc["keywords"] = extracted.keywords
            doi = normalize_doi(extracted.doi)
            if doi and not db.papers.find_one({"owner_id": owner_id, "doi": doi}):
                doc["doi"] = doi
            doc["processing_status"] = "completed"
            chunks = pdf_service.chunk_pages(extracted.pages)

    try:
        doc["_id"] = db.papers.insert_one(doc).inserted_id
    except DuplicateKeyError:
        storage.delete_file(path)
        raise HTTPException(409, "This PDF is already in your library")

    if chunks:
        db.document_chunks.insert_many(
            [{**c, "paper_id": doc["_id"], "owner_id": owner_id} for c in chunks])
        # embedding runs after the response is sent so the upload returns quickly
        background.add_task(indexing.index_paper, db, doc["_id"])
        doc["index_status"] = "indexing"
    return _out(doc)


@router.post("/{paper_id}/index", response_model=PaperOut, status_code=202)
def reindex_paper(paper_id: str, background: BackgroundTasks, user: dict = Depends(current_user)):
    paper = _get_owned(paper_id, user)
    if not get_db().document_chunks.count_documents({"paper_id": paper["_id"]}):
        raise HTTPException(409, "This paper has no extracted text to index")
    background.add_task(indexing.index_paper, get_db(), paper["_id"])
    paper["index_status"] = "indexing"
    return _out(paper)
