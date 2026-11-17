from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "research_paper_db"
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 7
    upload_dir: Path = Path("../storage/local")
    max_upload_mb: int = 25
    cors_origins: str = "http://localhost:5173"
    embedding_backend: str = "sentence-transformers"  # or "hashing" (offline fallback, weak)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dir: Path = Path("../storage/vector_index")
    rrf_k: int = 60
    llm_provider: str = "none"            # "gemini" or "none" (extractive answers only)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    rag_top_k: int = 6
    comparison_context_chars: int = 9000   # papers shorter than this are sent whole
    gap_cluster_distance: float = 0.5      # cosine distance for grouping statements; UNTUNED
    rag_min_semantic: float = 0.25   # starting guess, NOT tuned. Calibrate with scripts/eval_retrieval.py
    rag_max_context_chars: int = 7000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
