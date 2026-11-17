import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.connection import close_db, get_db
from app.database.indexes import create_indexes
from app.database.validation import apply_validation
from app.routers import auth, chat, comparisons, gaps, papers, search

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = get_db()
    apply_validation(db)
    create_indexes(db)
    log.info("MongoDB ready: %s", settings.db_name)
    yield
    close_db()


app = FastAPI(title="Research Paper Intelligence Platform", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(papers.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(comparisons.router)
app.include_router(gaps.router)


@app.get("/health", tags=["meta"])
def health():
    get_db().command("ping")
    return {"status": "ok"}
