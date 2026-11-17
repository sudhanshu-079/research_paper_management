from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.deps import current_user, to_object_id
from app.services import llm as llm_module
from app.services import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    paper_ids: list[str] = []          # empty = search across the whole library


class SessionOut(BaseModel):
    id: str
    title: str
    paper_ids: list[str]
    created_at: datetime
    updated_at: datetime


class MessageIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    supported: Optional[bool] = None
    citations: list[dict] = []
    mode: Optional[str] = None
    model: Optional[str] = None
    warnings: list[str] = []
    latency_ms: Optional[int] = None
    created_at: datetime


class SessionDetail(SessionOut):
    messages: list[MessageOut]


def _now():
    return datetime.now(timezone.utc)


def _session_out(s: dict) -> SessionOut:
    return SessionOut(id=str(s["_id"]), title=s["title"], paper_ids=[str(i) for i in s["paper_ids"]],
                      created_at=s["created_at"], updated_at=s["updated_at"])


def _msg_out(m: dict) -> MessageOut:
    return MessageOut(id=str(m["_id"]), role=m["role"], content=m["content"], supported=m.get("supported"),
                      citations=m.get("citations", []), mode=m.get("mode"), model=m.get("model"),
                      warnings=m.get("warnings", []), latency_ms=m.get("latency_ms"),
                      created_at=m["created_at"])


def _get_session(session_id: str, user: dict) -> dict:
    s = get_db().chat_sessions.find_one({"_id": to_object_id(session_id), "owner_id": user["_id"]})
    if not s:
        raise HTTPException(404, "Chat session not found")
    return s


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate, user: dict = Depends(current_user)):
    ids = list(dict.fromkeys(to_object_id(i) for i in body.paper_ids))  # de-duplicated, order kept
    if ids:
        owned = get_db().papers.count_documents({"_id": {"$in": ids}, "owner_id": user["_id"]})
        if owned != len(ids):
            raise HTTPException(404, "One or more papers were not found")  # same answer for "not yours"
    now = _now()
    doc = {"owner_id": user["_id"], "title": (body.title or "New chat").strip() or "New chat",
           "paper_ids": ids, "created_at": now, "updated_at": now}
    doc["_id"] = get_db().chat_sessions.insert_one(doc).inserted_id
    return _session_out(doc)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(user: dict = Depends(current_user)):
    cur = get_db().chat_sessions.find({"owner_id": user["_id"]}).sort("updated_at", -1).limit(100)
    return [_session_out(s) for s in cur]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, user: dict = Depends(current_user)):
    s = _get_session(session_id, user)
    msgs = get_db().chat_messages.find({"session_id": s["_id"], "owner_id": user["_id"]}).sort("created_at", 1)
    return SessionDetail(**_session_out(s).model_dump(), messages=[_msg_out(m) for m in msgs])


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, user: dict = Depends(current_user)):
    s = _get_session(session_id, user)
    get_db().chat_messages.delete_many({"session_id": s["_id"]})
    get_db().chat_sessions.delete_one({"_id": s["_id"]})


@router.post("/sessions/{session_id}/messages", response_model=MessageOut, status_code=201)
def ask(session_id: str, body: MessageIn, user: dict = Depends(current_user)):
    db = get_db()
    s = _get_session(session_id, user)
    history = [{"role": m["role"], "content": m["content"]}
               for m in db.chat_messages.find({"session_id": s["_id"]}).sort("created_at", 1)]
    try:
        res = rag_service.answer_question(db, user["_id"], body.question.strip(), s["paper_ids"], history)
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The language model failed: {exc}")

    now = _now()
    db.chat_messages.insert_one({"session_id": s["_id"], "owner_id": user["_id"], "role": "user",
                                 "content": body.question.strip(), "created_at": now})
    reply = {"session_id": s["_id"], "owner_id": user["_id"], "role": "assistant", "content": res["answer"],
             "supported": res["supported"], "citations": res["citations"], "mode": res["mode"],
             "model": res["model"], "warnings": res["warnings"], "latency_ms": res["latency_ms"],
             "created_at": _now()}
    reply["_id"] = db.chat_messages.insert_one(reply).inserted_id
    update = {"updated_at": reply["created_at"]}
    if not history and s["title"] == "New chat":
        update["title"] = body.question.strip()[:60]
    db.chat_sessions.update_one({"_id": s["_id"]}, {"$set": update})
    return _msg_out(reply)
