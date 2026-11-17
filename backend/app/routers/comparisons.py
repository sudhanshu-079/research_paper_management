from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.deps import current_user, to_object_id
from app.services import comparison_service
from app.services import llm as llm_module

router = APIRouter(prefix="/comparisons", tags=["comparisons"])


class ComparisonCreate(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    title: Optional[str] = Field(default=None, max_length=200)


class ComparisonSummary(BaseModel):
    id: str
    title: str
    paper_ids: list[str]
    mode: str
    created_at: datetime


class ComparisonOut(ComparisonSummary):
    model: Optional[str] = None
    columns: list[dict]
    rows: list[dict]
    observations: list[dict]
    warnings: list[str]


def _summary(c: dict) -> dict:
    return {"id": str(c["_id"]), "title": c["title"], "paper_ids": [str(i) for i in c["paper_ids"]],
            "mode": c["mode"], "created_at": c["created_at"]}


def _full(c: dict) -> ComparisonOut:
    return ComparisonOut(**_summary(c), model=c.get("model"), columns=c["columns"], rows=c["rows"],
                         observations=c["observations"], warnings=c.get("warnings", []))


def _get(comp_id: str, user: dict) -> dict:
    c = get_db().research_comparisons.find_one({"_id": to_object_id(comp_id), "owner_id": user["_id"]})
    if not c:
        raise HTTPException(404, "Comparison not found")
    return c


@router.post("", response_model=ComparisonOut, status_code=201)
def create_comparison(body: ComparisonCreate, user: dict = Depends(current_user)):
    db = get_db()
    ids = list(dict.fromkeys(to_object_id(i) for i in body.paper_ids))
    if len(ids) < 2:
        raise HTTPException(422, "Choose at least 2 different papers")
    found = {p["_id"]: p for p in db.papers.find({"_id": {"$in": ids}, "owner_id": user["_id"]})}
    if len(found) != len(ids):
        raise HTTPException(404, "One or more papers were not found")
    papers = [found[i] for i in ids]                      # keep the user's order
    no_text = [p["title"] for p in papers if not db.document_chunks.count_documents({"paper_id": p["_id"]})]
    if no_text:
        raise HTTPException(409, f"These papers have no extracted text: {', '.join(no_text)}")
    try:
        result = comparison_service.compare(db, user["_id"], papers)
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The language model failed: {exc}")
    doc = {"owner_id": user["_id"], "paper_ids": ids, "created_at": datetime.now(timezone.utc),
           "title": (body.title or "Comparison: " + " vs ".join(p["title"][:30] for p in papers))[:200],
           **result}
    doc["_id"] = db.research_comparisons.insert_one(doc).inserted_id
    return _full(doc)


@router.get("", response_model=list[ComparisonSummary])
def list_comparisons(user: dict = Depends(current_user)):
    cur = get_db().research_comparisons.find({"owner_id": user["_id"]}).sort("created_at", -1).limit(100)
    return [_summary(c) for c in cur]


@router.get("/{comp_id}", response_model=ComparisonOut)
def get_comparison(comp_id: str, user: dict = Depends(current_user)):
    return _full(_get(comp_id, user))


@router.get("/{comp_id}/export")
def export_comparison(comp_id: str, format: str = Query("markdown", pattern="^(markdown|json)$"),
                      user: dict = Depends(current_user)):
    c = _get(comp_id, user)
    if format == "json":
        return Response(_full(c).model_dump_json(indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="comparison.json"'})
    return Response(comparison_service.to_markdown(c), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="comparison.md"'})


@router.delete("/{comp_id}", status_code=204)
def delete_comparison(comp_id: str, user: dict = Depends(current_user)):
    c = _get(comp_id, user)
    get_db().research_comparisons.delete_one({"_id": c["_id"]})
