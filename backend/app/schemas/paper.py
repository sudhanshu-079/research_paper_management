from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ReadingStatus = Literal["unread", "reading", "read"]


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


class PaperBase(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    authors: list[str] = []
    abstract: Optional[str] = None
    publication: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2100)
    doi: Optional[str] = None
    keywords: list[str] = []
    tags: list[str] = []
    domain: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("doi")
    @classmethod
    def _doi(cls, v):
        v = normalize_doi(v)
        if v and not v.startswith("10."):
            raise ValueError("DOI must start with '10.'")
        return v


class PaperCreate(PaperBase):
    reading_status: ReadingStatus = "unread"


class PaperUpdate(BaseModel):
    """All fields optional; only the ones sent are changed."""
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    authors: Optional[list[str]] = None
    abstract: Optional[str] = None
    publication: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2100)
    doi: Optional[str] = None
    keywords: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    domain: Optional[str] = None
    notes: Optional[str] = None
    reading_status: Optional[ReadingStatus] = None
    needs_review: Optional[bool] = None

    @field_validator("doi")
    @classmethod
    def _doi(cls, v):
        v = normalize_doi(v)
        if v and not v.startswith("10."):
            raise ValueError("DOI must start with '10.'")
        return v


class PaperOut(PaperBase):
    id: str
    reading_status: ReadingStatus
    processing_status: str
    needs_review: bool = False
    has_file: bool = False
    processing_error: Optional[str] = None
    index_status: Optional[str] = None
    embedding_model: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class PaperList(BaseModel):
    items: list[PaperOut]
    total: int
    page: int
    page_size: int
