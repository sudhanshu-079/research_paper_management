"""Classic NLP statistics over a set of papers (scikit-learn, no LLM involved).

Everything here describes the papers the user selected. It says nothing about the
wider literature, so results are phrased as "in this collection".
"""
import math
import re

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

# words that dominate academic prose without naming a topic
DOMAIN_STOP = {
    "paper", "papers", "study", "studies", "results", "result", "using", "used", "use", "based", "proposed",
    "propose", "method", "methods", "approach", "approaches", "figure", "table", "et", "al", "abstract",
    "introduction", "conclusion", "section", "work", "works", "show", "shows", "shown", "also", "however",
    "different", "new", "number", "set", "model", "models", "data", "future", "limitation", "limitations",
}
STOP = list(ENGLISH_STOP_WORDS | DOMAIN_STOP)
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"


def paper_keywords(docs: dict[str, str], top_n: int = 8) -> dict[str, list[str]]:
    """Top TF-IDF terms (1-2 words) per paper. A term scores high when it is frequent in
    one paper but not in the others."""
    if not docs:
        return {}
    ids = list(docs)
    try:
        vec = TfidfVectorizer(stop_words=STOP, token_pattern=TOKEN, ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform([docs[i] for i in ids])
    except ValueError:  # empty vocabulary
        return {i: [] for i in ids}
    terms = vec.get_feature_names_out()
    out = {}
    for row, pid in enumerate(ids):
        r = X[row].toarray().ravel()
        top = r.argsort()[::-1][:top_n]
        out[pid] = [str(terms[j]) for j in top if r[j] > 0]
    return out


def term_coverage(docs: dict[str, str], top_n: int = 15) -> dict:
    """frequent: terms that appear in many of the papers.
    distinctive: terms used repeatedly in exactly one paper (candidate under-covered themes
    *within this collection*)."""
    ids = list(docs)
    if len(ids) < 2:
        return {"frequent": [], "distinctive": []}
    try:
        vec = CountVectorizer(stop_words=STOP, token_pattern=TOKEN)
        X = vec.fit_transform([docs[i] for i in ids]).tocsc()
    except ValueError:
        return {"frequent": [], "distinctive": []}
    terms = vec.get_feature_names_out()
    counts = X.toarray()
    df = (counts > 0).sum(axis=0)
    need = max(2, math.ceil(len(ids) / 2))

    freq_idx = [j for j in range(len(terms)) if df[j] >= need]
    freq_idx.sort(key=lambda j: (-df[j], -counts[:, j].sum()))
    frequent = [{"term": str(terms[j]), "paper_count": int(df[j]),
                 "total_mentions": int(counts[:, j].sum())} for j in freq_idx[:top_n]]

    distinctive = []
    for row, pid in enumerate(ids):
        own = [j for j in range(len(terms)) if df[j] == 1 and counts[row, j] >= 2]
        own.sort(key=lambda j: -counts[row, j])
        distinctive += [{"term": str(terms[j]), "paper_id": pid, "mentions": int(counts[row, j])}
                        for j in own[:5]]
    return {"frequent": frequent, "distinctive": distinctive[:top_n * 2]}


def topics(texts: list[str], n_topics: int = 4, top_terms: int = 6) -> list[list[str]]:
    """NMF topic model over TF-IDF. Returns the top terms of each topic."""
    if len(texts) < 2:
        return []
    try:
        vec = TfidfVectorizer(stop_words=STOP, token_pattern=TOKEN, sublinear_tf=True)
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    k = min(n_topics, X.shape[0], X.shape[1])
    if k < 2:
        return []
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
    nmf.fit(X)
    terms = vec.get_feature_names_out()
    return [[str(terms[j]) for j in comp.argsort()[::-1][:top_terms]] for comp in nmf.components_]
