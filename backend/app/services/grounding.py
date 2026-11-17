"""Helpers shared by every feature that asks an LLM to read paper text."""
import json
import re


def sanitize(text: str) -> str:
    """Stop PDF text from closing our prompt blocks or faking structure."""
    return re.sub(r"</?\s*(sources|source|question|history|cells|task)\b[^>]*>", "[tag removed]", text, flags=re.I)


def parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def make_citation(label: str, chunk: dict, paper: dict) -> dict:
    return {"label": label, "paper_id": str(chunk["paper_id"]), "paper_title": paper["title"],
            "chunk_id": str(chunk["_id"]), "chunk_index": chunk.get("chunk_index"),
            "page": chunk.get("page"), "snippet": chunk["text"][:300]}
