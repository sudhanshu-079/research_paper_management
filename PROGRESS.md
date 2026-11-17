# Progress

## Implemented and tested (27 pytest tests, run against mongomock)
- Config via .env, Mongo connection, strict JSON Schema validators (users, papers, document_chunks)
- Indexes incl. partial unique DOI / file-hash indexes
- Register / login / refresh / me, bcrypt, JWT, password rules
- Paper CRUD, search, filters, sorting, pagination
- Per-user isolation (verified by a test that fails when the owner filter is removed)
- PDF upload: validation, size limit, dedupe by hash, metadata extraction, chunking, scanned-PDF flag

## Phase 7 implemented and tested (41 tests total, mongomock + hashing embedder)
- Embedder interface (Sentence Transformers, lazy-loaded; hashing fallback), FAISS per-user store with atomic saves
- Background indexing after upload, manual reindex endpoint, cleanup of partial failures (mutation-checked)
- BM25, semantic, hybrid (RRF) search, per-paper grouping, scores and `matched_by` in results
- Retrieval metrics (P@K, Recall@K, MRR, NDCG@K) unit-tested on hand-computed values; eval script

## Phase 8 implemented and tested (66 tests total)
- LLM provider interface, Gemini REST provider (HTTP layer tested against a mock server), extractive fallback
- RAG service with relevance floor, citation validation, abstain path, prompt-injection sanitising
- Chat sessions/messages with ownership checks, paper scoping, history, chat schemas + indexes
- Mutation-checked: citation validation, relevance floor, sanitiser each have a test that fails without them

## Phase 9 implemented and tested (92 tests total)
- Comparison: 10-field table, per-paper extraction, citation validation, observations tied to cells, save / list / delete, markdown + JSON export
- Gap analysis: statement extraction, embedding + clustering, distinct-paper evidence strength, TF-IDF stats, NMF topics, validated LLM drafting, graceful fallback, saved analyses
- Shared grounding helpers (sanitise, parse JSON, citations) now used by RAG, comparison and gaps
- Mutation-checked: citation validation in all three features, observation-cell validation, evidence strength.
  One mutation survived at first (a paper repeating itself counted as 2 papers); I added a test and it now fails as it should.

## Implemented but NOT verified
- **The real MiniLM model.** No HuggingFace access in my sandbox. Tests use the hashing embedder,
  so they prove the plumbing and isolation, NOT that semantic search finds conceptually related papers.
  Try it yourself with a few real papers and the eval script.
- **Real MongoDB behaviour.** My sandbox had no mongod, and mongomock ignores `$jsonSchema`
  validators and partial-index semantics. The offline schema test only checks the definitions.
  Please run the app against your local MongoDB and confirm startup logs "MongoDB ready".
  If `collMod` or index creation errors, send me the traceback.

- **Live Gemini calls.** A local live smoke check confirmed the configured API key is accepted,
  but `gemini-2.5-flash` is no longer available to new users. The project now uses
  `gemini-3.8-flash`; verify with a live chat request.
- **Answer quality / faithfulness.** Tests use a scripted fake LLM. Nothing here measures whether real
  answers are faithful; that is the faithfulness evaluation still to be built.
- **RAG_MIN_SEMANTIC threshold** is a guess until tuned on real data.

- **Gap and comparison quality on real papers.** Tests use hand-written PDFs and a scripted model, with distances
  measured under the hashing embedder. The cue lists, `GAP_CLUSTER_DISTANCE=0.5` and cluster quality with MiniLM are
  untested. Next step is to label gaps in a few real papers and measure precision.
- Field extraction accuracy of the comparison table with a real model.

## Frontend status
- Implemented: authentication, dashboard, library, upload, search, paper metadata editing,
  paper-scoped summary/chat, paper deletion, comparison, gap analysis, comparison export, and
  conversation deletion/copy.
- Remaining UI features: collections, multi-file upload, dedicated
  13-section paper analysis, richer citation management, recommendations, analytics,
  profile/settings, and exportable notes. Several require new backend endpoints.

## Not started
Answer streaming, RAG faithfulness evaluation, rate limiting, Docker, Crossref/OpenAlex
enrichment, OCR, evaluation scripts, deployment.

## Known issues / limits
- Logout is not server-side; refresh tokens are not revocable.
- No rate limiting yet.
- Search uses case-insensitive regex (fine for small libraries; text/vector search comes later).
- Upload reads the whole PDF into memory (capped at MAX_UPLOAD_MB).
- Author extraction is best-effort from PDF metadata and first-page text; verify it manually.
- Indexing uses FastAPI BackgroundTasks (same process). Fine locally; move to a real worker before deploying to Render.
- Abstracts are not embedded separately, only chunks.
- Comparison/gaps run synchronously in the request (up to ~6 model calls); fine locally, needs a job queue for deployment.
- Comparison has no regenerate endpoint (create a new one).
- Gap analysis does not use the dataset/methodology comparison as a signal (only limitations, future work and term statistics).
- No similar-paper endpoint yet (the vectors are ready for it).
- Your existing root-level main.py / database/validation.py were not visible to me.
