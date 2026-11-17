"""Keyword (BM25), semantic (FAISS cosine) and hybrid (reciprocal rank fusion) search.

All three work on chunks, then results are grouped so each paper appears once,
represented by its best-matching chunk.
"""
import math
import re
from collections import Counter

from app.config import settings
from app.services.registry import get_embedder, get_store

CANDIDATES = 50
STOPWORDS = set("a an and are as at be by for from has have in is it of on or that the this to was were with we our".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def bm25_rank(query: str, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> list[tuple[dict, float]]:
    """Okapi BM25 over the given chunks. Returns chunks with score > 0, best first."""
    q_terms = set(tokenize(query))
    if not q_terms or not chunks:
        return []
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d) if t in q_terms)
    scored = []
    for chunk, doc in zip(chunks, docs):
        tf = Counter(doc)
        score = 0.0
        for t in q_terms:
            if df[t] == 0 or tf[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        if score > 0:
            scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def semantic_rank(db, owner_id, query: str, min_score: float, paper_ids=None,
                  candidates: int = CANDIDATES) -> list[tuple[dict, float]]:
    qvec = get_embedder().encode([query])[0]
    hits = get_store().search(str(owner_id), qvec, candidates)
    hits = [(vid, s) for vid, s in hits if s >= min_score]
    if not hits:
        return []
    flt = {"owner_id": owner_id, "vec_id": {"$in": [v for v, _ in hits]}}  # owner filter: defence in depth
    if paper_ids is not None:
        flt["paper_id"] = {"$in": list(paper_ids)}
    chunks = {c["vec_id"]: c for c in db.document_chunks.find(flt)}
    return [(chunks[v], s) for v, s in hits if v in chunks]  # drops other papers / deleted chunks


def rank_chunks(db, owner_id, query: str, mode: str, min_semantic: float = 0.0,
                paper_ids=None, candidates: int = CANDIDATES) -> list[dict]:
    """Chunk-level ranking, best first. Each item: chunk, score, keyword_score, semantic_score.

    paper_ids (optional) restricts the search to those papers. Used by search (all papers)
    and by the RAG assistant (selected papers).
    """
    kw, sem = [], []
    if mode in ("keyword", "hybrid"):
        flt = {"owner_id": owner_id}
        if paper_ids is not None:
            flt["paper_id"] = {"$in": list(paper_ids)}
        kw = bm25_rank(query, list(db.document_chunks.find(flt)))[:candidates]
    if mode in ("semantic", "hybrid"):
        # when restricted to a few papers, look deeper so their vectors aren't crowded out
        depth = candidates if paper_ids is None else max(candidates, 400)
        sem = semantic_rank(db, owner_id, query, min_semantic, paper_ids, depth)[:candidates]

    kw_score = {c["_id"]: s for c, s in kw}
    sem_score = {c["_id"]: s for c, s in sem}
    by_id = {c["_id"]: c for c, _ in kw + sem}

    # Reciprocal Rank Fusion: sum over methods of 1 / (k + rank). Uses ranks, not raw
    # scores, because BM25 and cosine live on different scales.
    k = settings.rrf_k
    fused: dict = {}
    for ranking in (kw, sem):
        for rank, (chunk, _) in enumerate(ranking, start=1):
            fused[chunk["_id"]] = fused.get(chunk["_id"], 0.0) + 1.0 / (k + rank)

    order = {"keyword": kw_score, "semantic": sem_score, "hybrid": fused}[mode]
    return [{"chunk": by_id[cid], "score": sc,
             "keyword_score": kw_score.get(cid), "semantic_score": sem_score.get(cid)}
            for cid, sc in sorted(order.items(), key=lambda x: x[1], reverse=True)]


def search(db, owner_id, query: str, mode: str, limit: int, min_semantic: float = 0.0) -> list[dict]:
    ranked = rank_chunks(db, owner_id, query, mode, min_semantic)

    best: dict = {}      # paper_id -> best chunk entry (ranked is already best-first)
    k_best: dict = {}
    s_best: dict = {}
    for item in ranked:
        pid = item["chunk"]["paper_id"]
        best.setdefault(pid, item)
        if item["keyword_score"] is not None:
            k_best[pid] = max(k_best.get(pid, 0.0), item["keyword_score"])
        if item["semantic_score"] is not None:
            s_best[pid] = max(s_best.get(pid, -2.0), item["semantic_score"])
    top = list(best.items())[:limit]

    papers = {p["_id"]: p for p in db.papers.find(
        {"_id": {"$in": [pid for pid, _ in top]}, "owner_id": owner_id})}
    results = []
    for pid, item in top:
        paper = papers.get(pid)
        if not paper:
            continue
        kb, sb = k_best.get(pid), s_best.get(pid)
        results.append({
            "paper_id": str(pid), "title": paper["title"], "year": paper.get("year"),
            "authors": paper.get("authors", []),
            "score": round(item["score"], 4),
            "keyword_score": None if kb is None else round(kb, 4),
            "semantic_score": None if sb is None else round(sb, 4),
            "matched_by": [m for m, v in (("keyword", kb), ("semantic", sb)) if v is not None],
            "page": item["chunk"].get("page"), "snippet": item["chunk"]["text"][:300],
        })
    return results
