from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.deps import current_user, to_object_id
from app.services import gap_service

router = APIRouter(prefix="/gaps", tags=["research gaps"])


class GapRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=30)
    title: Optional[str] = Field(default=None, max_length=200)


class GapSummary(BaseModel):
    id: str
    title: str
    paper_ids: list[str]
    mode: str
    gap_count: int
    created_at: datetime


class GapAnalysisOut(BaseModel):
    id: str
    title: str
    paper_ids: list[str]
    created_at: datetime
    disclaimer: str
    mode: str
    model: Optional[str] = None
    stats: dict
    frequent_terms: list[dict]
    distinctive_terms: list[dict]
    paper_keywords: list[dict]
    topics: list[dict]
    gaps: list[dict]
    warnings: list[str]


def _full(d: dict) -> GapAnalysisOut:
    return GapAnalysisOut(id=str(d["_id"]), title=d["title"], paper_ids=[str(i) for i in d["paper_ids"]],
                          created_at=d["created_at"], disclaimer=d["disclaimer"], mode=d["mode"],
                          model=d.get("model"), stats=d["stats"], frequent_terms=d["frequent_terms"],
                          distinctive_terms=d["distinctive_terms"], paper_keywords=d["paper_keywords"],
                          topics=d["topics"], gaps=d["gaps"], warnings=d["warnings"])


def _get(gap_id: str, user: dict) -> dict:
    d = get_db().research_gaps.find_one({"_id": to_object_id(gap_id), "owner_id": user["_id"]})
    if not d:
        raise HTTPException(404, "Analysis not found")
    return d


@router.post("/analyze", response_model=GapAnalysisOut, status_code=201)
def analyze(body: GapRequest, user: dict = Depends(current_user)):
    db = get_db()
    ids = list(dict.fromkeys(to_object_id(i) for i in body.paper_ids))
    if len(ids) < 2:
        raise HTTPException(422, "Choose at least 2 different papers")
    found = {p["_id"]: p for p in db.papers.find({"_id": {"$in": ids}, "owner_id": user["_id"]})}
    if len(found) != len(ids):
        raise HTTPException(404, "One or more papers were not found")
    papers = [found[i] for i in ids]
    no_text = [p["title"] for p in papers if not db.document_chunks.count_documents({"paper_id": p["_id"]})]
    if no_text:
        raise HTTPException(409, f"These papers have no extracted text: {', '.join(no_text)}")

    result = gap_service.analyze(db, user["_id"], papers)
    doc = {"owner_id": user["_id"], "paper_ids": ids, "created_at": datetime.now(timezone.utc),
           "title": (body.title or f"Gap analysis of {len(papers)} papers")[:200], **result}
    doc["_id"] = db.research_gaps.insert_one(doc).inserted_id
    return _full(doc)


@router.get("", response_model=list[GapSummary])
def list_analyses(user: dict = Depends(current_user)):
    cur = get_db().research_gaps.find({"owner_id": user["_id"]}).sort("created_at", -1).limit(100)
    return [GapSummary(id=str(d["_id"]), title=d["title"], paper_ids=[str(i) for i in d["paper_ids"]],
                       mode=d["mode"], gap_count=len(d["gaps"]), created_at=d["created_at"]) for d in cur]


@router.get("/{gap_id}", response_model=GapAnalysisOut)
def get_analysis(gap_id: str, user: dict = Depends(current_user)):
    return _full(_get(gap_id, user))


@router.delete("/{gap_id}", status_code=204)
def delete_analysis(gap_id: str, user: dict = Depends(current_user)):
    d = _get(gap_id, user)
    get_db().research_gaps.delete_one({"_id": d["_id"]})
