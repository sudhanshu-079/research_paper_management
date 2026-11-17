"""PDF text extraction, light metadata extraction, and chunking (PyMuPDF).

Nothing here guesses at facts: anything we can't find is left empty and the
paper is flagged for manual review.
"""
import re
from dataclasses import dataclass, field

import pymupdf

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


@dataclass
class ExtractedPDF:
    pages: list[tuple[int, str]] = field(default_factory=list)  # (page_number starting at 1, text)
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    abstract: str | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(t.strip()) for _, t in self.pages)


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_largest_font(page) -> str | None:
    """The title is usually the biggest text on page 1."""
    best_size, lines = 0.0, []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if len(text) < 3 or not line["spans"]:
                continue
            size = max(s["size"] for s in line["spans"])
            if size > best_size + 0.5:
                best_size, lines = size, [text]
            elif abs(size - best_size) <= 0.5:
                lines.append(text)
    title = " ".join(lines).strip()
    return title if 5 <= len(title) <= 300 else None


def _extract_abstract(first_pages_text: str) -> str | None:
    m = re.search(r"\babstract\b[\s:\-—.]*", first_pages_text, re.IGNORECASE)
    if not m:
        return None
    rest = first_pages_text[m.end():]
    end = re.search(r"\n\s*(keywords?|index terms|1\.?\s+introduction|introduction)\b",
                    rest, re.IGNORECASE)
    abstract = rest[:end.start()] if end else rest[:1500]
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract[:2500] if len(abstract) > 50 else None


def _extract_keywords(text: str) -> list[str]:
    m = re.search(r"\b(?:keywords?|index terms)\b[\s:\-—]*([^\n]+(?:\n[^\n]+)?)", text, re.IGNORECASE)
    if not m:
        return []
    parts = re.split(r"[;,·•]", m.group(1))
    return [p.strip(" .\n") for p in parts if 2 <= len(p.strip()) <= 60][:15]


def _author_candidates(value: str) -> list[str]:
    parts = re.split(r"\s*(?:;|\||\n|\band\b|&)\s*", value, flags=re.IGNORECASE)
    if len(parts) == 1 and "," in value:
        comma_parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(comma_parts) > 1 and all(len(re.findall(r"[A-Za-z]+", part)) >= 2 for part in comma_parts):
            parts = comma_parts
        elif len(comma_parts) == 2:
            parts = [f"{comma_parts[1]} {comma_parts[0]}"]
    return [part.strip(" ,.;") for part in parts if part.strip(" ,.;")]


def _extract_authors(metadata_author: str | None, first_page_text: str, title: str | None) -> list[str]:
    if metadata_author and metadata_author.strip():
        return _author_candidates(metadata_author)[:12]

    title_key = re.sub(r"\W+", "", title or "").lower()
    excluded = re.compile(
        r"\b(abstract|keywords?|doi|published|received|accepted|university|department|"
        r"school|institute|laboratory|laboratories|email|corresponding|copyright|"
        r"conference|journal|www|http)\b",
        re.IGNORECASE,
    )
    authors: list[str] = []
    for line in first_page_text.splitlines():
        line = line.strip()
        normalized = re.sub(r"\W+", "", line).lower()
        if not line or len(line) > 140 or (title_key and normalized == title_key):
            continue
        if re.search(r"\d|@|https?://", line, re.IGNORECASE) or excluded.search(line):
            continue

        candidates = _author_candidates(line)
        if len(candidates) == 1 and "," in line:
            comma_parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(comma_parts) > 1 and all(len(re.findall(r"[A-Za-z]+", part)) >= 2 for part in comma_parts):
                candidates = comma_parts
        valid = []
        for candidate in candidates:
            words = re.findall(r"[A-Za-z][A-Za-z.'’\-]*", candidate)
            if not 2 <= len(words) <= 5:
                continue
            if not all(word[0].isupper() or len(word) == 1 for word in words):
                continue
            valid.append(" ".join(words))
        if valid:
            authors.extend(valid)
            if len(authors) >= 12:
                break
    return authors[:12]


def extract_pdf(path: str) -> ExtractedPDF:
    out = ExtractedPDF()
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            out.pages.append((i, clean_text(page.get_text())))
        if len(doc) == 0:
            return out
        meta_title = (doc.metadata or {}).get("title", "").strip()
        out.title = meta_title if len(meta_title) >= 5 else _title_from_largest_font(doc[0])
        out.authors = _extract_authors((doc.metadata or {}).get("author"), out.pages[0][1], out.title)

    front = "\n".join(t for _, t in out.pages[:2])
    if m := DOI_RE.search(front):
        out.doi = m.group(1).rstrip(".,;)").lower()
    if m := YEAR_RE.search(front):
        out.year = int(m.group(1))
    out.abstract = _extract_abstract(front)
    out.keywords = _extract_keywords(front)
    return out


def chunk_pages(pages: list[tuple[int, str]], size: int = 1000, overlap: int = 150) -> list[dict]:
    """Split each page into overlapping character windows, remembering the page number.

    Chunks don't cross page boundaries so every chunk maps to exactly one page,
    which makes citations simple later on.
    """
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if len(piece) >= 40:
                chunks.append({"page": page_no, "text": piece})
            if start + size >= len(text):
                break
            start += size - overlap
    for i, c in enumerate(chunks):
        c["chunk_index"] = i
    return chunks
