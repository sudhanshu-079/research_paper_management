from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.database.connection import get_db
from app.deps import current_user
from app.services import search_service

router = APIRouter(prefix="/search", tags=["search"])


class SearchHit(BaseModel):
    paper_id: str
    title: str
    year: int | None
    authors: list[str]
    score: float                    # ranking score for the chosen mode (see README)
    keyword_score: float | None     # BM25
    semantic_score: float | None    # cosine similarity, -1..1
    matched_by: list[str]
    page: int | None
    snippet: str


class SearchOut(BaseModel):
    query: str
    mode: str
    results: list[SearchHit]


@router.get("", response_model=SearchOut)
def search(
    q: str = Query(min_length=2, max_length=300),
    mode: str = Query("hybrid", pattern="^(keyword|semantic|hybrid)$"),
    limit: int = Query(10, ge=1, le=50),
    min_semantic: float = Query(0.0, ge=-1.0, le=1.0),
    user: dict = Depends(current_user),
):
    results = search_service.search(get_db(), user["_id"], q, mode, limit, min_semantic)
    return SearchOut(query=q, mode=mode, results=results)
