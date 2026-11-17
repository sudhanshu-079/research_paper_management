"""Compare keyword vs semantic vs hybrid search on a hand-labelled query set.

Usage (from backend/, with MongoDB running and papers already uploaded + indexed):
    python ../scripts/eval_retrieval.py --email you@example.com --queries ../docs/eval_queries.json

eval_queries.json:
    [{"query": "transformer language models", "relevant_titles": ["Attention Is All You Need"]}, ...]

YOU must write the queries and relevance labels by hand. The script only computes numbers
from whatever you give it; it makes none up.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.database.connection import get_db  # noqa: E402
from app.services import metrics, search_service  # noqa: E402
from app.services.registry import get_embedder  # noqa: E402

KS = (1, 3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--queries", required=True)
    args = ap.parse_args()

    db = get_db()
    user = db.users.find_one({"email": args.email.lower()})
    if not user:
        sys.exit("No such user")
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    print(f"Embedding model: {get_embedder().name}; {len(queries)} queries")

    for mode in ("keyword", "semantic", "hybrid"):
        agg = {f"P@{k}": 0.0 for k in KS} | {f"R@{k}": 0.0 for k in KS} | {"MRR": 0.0} | {f"NDCG@{k}": 0.0 for k in KS}
        for item in queries:
            hits = search_service.search(db, user["_id"], item["query"], mode, limit=max(KS))
            ranked = [h["title"] for h in hits]
            rel = set(item["relevant_titles"])
            for k in KS:
                agg[f"P@{k}"] += metrics.precision_at_k(ranked, rel, k)
                agg[f"R@{k}"] += metrics.recall_at_k(ranked, rel, k)
                agg[f"NDCG@{k}"] += metrics.ndcg_at_k(ranked, rel, k)
            agg["MRR"] += metrics.reciprocal_rank(ranked, rel)
        print(f"\n[{mode}]")
        for name, total in agg.items():
            print(f"  {name:8s} {total / len(queries):.3f}")


if __name__ == "__main__":
    main()
