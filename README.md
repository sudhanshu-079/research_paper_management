# Research Paper Intelligence Platform

Status: **backend: auth, papers, PDF upload, hybrid search, RAG assistant, comparison, research gaps (Phases 1-4, part of 6, 7, 8, 9)**. See PROGRESS.md for what is and isn't done.

## Run locally (Windows, PowerShell)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then set JWT_SECRET to a long random string
uvicorn app.main:app --reload
```

MongoDB must be running locally. On startup the app creates the collections with strict
JSON Schema validators and builds the indexes. API docs: http://localhost:8000/docs

If you already have `database/validation.py` and `main.py` at the repo root, keep them as
reference. The versions in `backend/app/database/` replace them (they cover users, papers and
document_chunks).

## Run the tests

```powershell
cd backend
python -m pytest -q
```

## Run the frontend (Windows, PowerShell)

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open http://localhost:5173. The frontend calls `http://localhost:8000` by default; change
`VITE_API_URL` in `frontend/.env.local` only if the backend uses another address. Keep the
frontend origin consistent with `CORS_ORIGINS` in `backend/.env`.

The current UI includes authentication, dashboard, paper library and metadata editing, PDF
upload and full-file preview/download, library search, paper-scoped summaries and chat,
paper deletion, saved comparisons, and saved research-gap analyses. Paper authors and primary
topic can be reviewed and edited; topics also display extracted keywords. A comparison or gap
analysis requires processed PDFs with extracted text.

Collections, standalone 13-section analysis, multi-file upload,
recommendations, analytics, citation-style exports, and password reset/profile management are
not yet supported by the backend API and are not available in the UI.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | /auth/register, /auth/login, /auth/refresh | tokens |
| GET | /auth/me | current user |
| POST/GET | /papers | create / list (search, filter, sort, paginate) |
| GET/PATCH/DELETE | /papers/{id} | read / update / delete |
| POST | /papers/upload | PDF upload, extraction, chunking |
| POST | /papers/{id}/index | (re)build embeddings for one paper (background) |
| GET | /search?q=&mode=keyword\|semantic\|hybrid&limit=&min_semantic= | search the caller's library |
| POST/GET | /chat/sessions | start a chat (optionally scoped to chosen papers) / list |
| GET/DELETE | /chat/sessions/{id} | history with citations / delete |
| POST | /chat/sessions/{id}/messages | ask a question |
| POST/GET | /comparisons | compare 2-5 papers (saved) / list |
| GET/DELETE | /comparisons/{id} | read / delete |
| GET | /comparisons/{id}/export?format=markdown\|json | download |
| POST/GET | /gaps/analyze, /gaps | candidate research-gap analysis of 2-30 papers (saved) / list |
| GET/DELETE | /gaps/{id} | read / delete |
| GET | /health | DB ping |

## How it works (interview notes)

- **Auth**: bcrypt hashes; JWT access (30 min) and refresh (7 d) tokens carrying a `type`
  claim so a refresh token can't be used as an access token. Logout is client-side (discard
  tokens); there is no server-side revocation yet.
- **Data isolation**: every paper query includes `owner_id`. Someone else's paper returns 404,
  not 403, so IDs can't be probed.
- **Duplicates**: partial unique indexes on (owner, doi) and (owner, file_hash). Partial because
  many papers have no DOI and a normal unique index would treat all the nulls as duplicates.
- **Upload**: checks the `%PDF-` magic bytes and size, hashes with SHA-256, stores under a
  server-generated filename, extracts with PyMuPDF. Title = PDF metadata or largest font on
  page 1. DOI/year/abstract/keywords by regex. Author extraction uses PDF metadata and
  best-effort first-page heuristics; verify extracted authors manually. Every extracted paper
  is flagged `needs_review`.
- **Scanned PDFs**: marked `failed` with an OCR message rather than silently empty.
- **Chunks**: 1000-char windows, 150 overlap, never crossing pages, so each chunk has one page
  number for citations. They're stored in `document_chunks` ready for embedding.

## Search (Phase 7)

Setup on your machine (first semantic run downloads about 90 MB for the MiniLM model):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

To run without downloading a model, set `EMBEDDING_BACKEND=hashing` in `.env`. That backend is
plain word overlap, not semantic, so only use it for development.

**Flow:** upload -> chunks saved -> (background) embed chunks in batches of 32 -> vectors added to
the user's FAISS index -> each chunk stores its integer `vec_id`.

| Component | Input | Preprocessing | Algorithm | Output | Limitations |
|---|---|---|---|---|---|
| Embeddings | chunk text | none (model tokenizes) | all-MiniLM-L6-v2, 384-d, L2-normalised | float32 vector | 256-token input limit (about 1000 chars, matching our chunk size); English-centric |
| Vector index | vectors + int ids | normalised | FAISS `IndexIDMap2(IndexFlatIP)`; inner product of unit vectors = cosine | top-k (id, cosine) | exact brute force: fine up to ~100k chunks per user, then switch to an approximate index |
| Keyword | query + user's chunks | lowercase, tokenise, stopwords out | Okapi BM25 (k1=1.5, b=0.75), written by hand | ranked chunks | recomputed per query over all of the user's chunks; no stemming |
| Hybrid | the two rankings | none | Reciprocal Rank Fusion: sum of 1/(60+rank) | ranked chunks | RRF uses ranks only, so it ignores how confident each method was |

Results are grouped per paper (best chunk wins). `score` means BM25 for keyword mode, cosine for
semantic mode and the RRF value for hybrid mode, so don't compare scores across modes.
`matched_by` says which methods found the paper.

Security: one FAISS file per user and model, plus an `owner_id` filter on the chunk lookup.

**Evaluation:** `python ../scripts/eval_retrieval.py --email you@x.com --queries ../docs/eval_queries.json`
prints P@K, Recall@K, MRR and NDCG@K for each mode. You must write the labelled queries yourself
(see `docs/eval_queries.example.json`); the numbers for your paper come only from that.

Interview angle: why RRF over a weighted sum (different score scales); why per-user indexes (isolation
and easy deletion); why record `vec_id` in Mongo before adding to FAISS (so a crash can't leave
orphan vectors); why the index is a rebuildable cache.

## RAG assistant (Phase 8)

Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` in `.env` (never commit it). Check
`GEMINI_MODEL` against Google's current model list, since model names change. With no key the
assistant still works in **extractive mode**: it returns the top passages with citations and says
plainly that it is not an AI-written answer.

**Flow:** question -> (follow-ups borrow the previous question for retrieval) -> hybrid chunk
retrieval limited to the session's papers and the user's own chunks -> relevance floor ->
prompt with labelled sources S1..Sn -> model returns JSON `{supported, answer, citations}` ->
**we validate the output** -> store and return.

What is enforced in code, not just requested in the prompt:
- No relevant chunks -> fixed "couldn't find anything" reply, and the LLM is never called.
- Every cited label must be one of the sources we actually sent. Unknown labels are removed
  with a warning. "Supported" with no valid citation is downgraded to unsupported.
- Unsupported answers come back with no citations.
- Citations resolve to real paper title, page and chunk id from the database.
- PDF text can't close our `<sources>` block or fake structure (tag sanitising), and the system
  prompt tells the model the sources are untrusted. This stops structural tricks only. A model can
  still be influenced by wording inside a document, so it is a mitigation, not a guarantee.
- Another user's papers/sessions return 404; sessions can only be scoped to your own papers.

Limitations: `RAG_MIN_SEMANTIC=0.25` is an untuned starting guess; citation validation proves a cited
passage was *provided*, not that it truly *supports* the claim (that needs a faithfulness
evaluation); answers are not streamed; only the last 6 messages are sent as history.

Interview angle: "don't trust the model's output" (validate citations against retrieved ids),
graceful degradation without an API key, and why the abstain path lives in code.

## Paper comparison (Phase 9a)

`POST /comparisons {"paper_ids": [...2 to 5...]}` builds a 10-row table (objectives, problem, methodology,
algorithms, datasets, metrics, results, strengths, limitations, future scope), one column per paper.

Flow: per paper, gather evidence (whole paper if under `COMPARISON_CONTEXT_CHARS`, otherwise the best
passages for each field) -> **one model call per paper** returns every field as JSON with citations ->
citations are checked against the passages sent -> one more call writes comparative observations that must
point at specific table cells (`paper 2 / methodology`). Observations citing a missing or empty cell are dropped.

Cell status: `found` (text + validated citations), `not_found`, `unsupported` (the model asserted something
with no valid citation, so it was discarded), `extractive` (no LLM: shows the best-matching passage, not a
summary). Each paper's prompt contains only that paper's text. If nothing was found for any paper, no
observation call is made.

## Research gap identification (Phase 9b)

`POST /gaps/analyze {"paper_ids": [...2 to 30...]}`. Every output carries the disclaimer that these are
**candidate** gaps from your own small collection, not proof that a topic is unexplored in the literature.

| Step | Input | Algorithm | Output | Limitations |
|---|---|---|---|---|
| Statement extraction | chunk text | sentence split + cue phrases ("limited to", "future work", ...) | limitation / future-work sentences with page | cue list is hand-written; recall and precision are unmeasured |
| Clustering | sentence embeddings | agglomerative, cosine distance, average linkage, threshold `GAP_CLUSTER_DISTANCE` | groups = candidate gaps | threshold 0.5 is an untuned guess |
| Evidence strength | a group | number of **distinct papers** in it: 1 weak, 2 moderate, 3+ strong | label | a paper repeating itself doesn't count twice |
| Term statistics | per-paper text | CountVectorizer document frequency; TF-IDF (1-2 grams) | frequent terms, terms distinctive to one paper, keywords per paper | "distinctive" means within this collection only |
| Topics | all chunks | TF-IDF + NMF (k = 2..4) | top terms per topic | small corpora give crude topics |
| Wording (optional LLM) | a group's sentences | one call; citations must belong to that specific gap | description, research question, experiment idea | invalid citations => that gap keeps its extractive text |

If the model fails or returns junk, you still get the full extractive analysis plus a warning.

Interview angle: why evidence strength counts papers not sentences; why the gap prompt bans "unexplored";
why clustering sentences (not whole papers) finds *shared* limitations; why failures degrade instead of erroring.
