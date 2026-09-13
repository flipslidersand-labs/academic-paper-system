from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for academic paper system."""

    embedding_svc_url: str = Field(default="http://<internal-host>:9092", description="Embedding service URL")
    embedding_api_key: str = Field(default="", description="API key for embedding service")
    embedding_timeout: int = Field(
        default=120,
        gt=0,
        description="embedding-svc HTTP timeout in seconds; large batches (up to 256 chunks) can take >30s",
    )
    qdrant_url: str = Field(default="http://<internal-host>:6333", description="Qdrant vector database URL")
    qdrant_api_key: str = Field(default="", description="API key for Qdrant")
    qdrant_timeout: int = Field(
        default=30,
        gt=0,
        description="Qdrant client timeout in seconds; upsert batches are capped at 200 points (#236)",
    )
    academic_db: str = Field(default="/data/academic.db", description="Path to academic database")
    chunk_size: int = Field(default=512, gt=0, description="Size of text chunks for processing")
    chunk_overlap: int = Field(default=64, description="Overlap between consecutive chunks")
    qdrant_collection: str = Field(default="academic-papers", description="Qdrant collection name")
    port: int = Field(default=8020, gt=0, description="Port for API server")
    google_api_key: str = Field(default="", description="Google API key for generative AI")
    gemini_timeout_ms: int = Field(default=60000, gt=0, description="Gemini API HTTP timeout in milliseconds")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama service URL")
    ollama_model: str = Field(default="mistral", description="Ollama model to use")
    ollama_timeout: int = Field(default=300, gt=0, description="Ollama HTTP timeout in seconds")
    otel_endpoint: str = Field(default="", description="OpenTelemetry endpoint")
    log_level: str = Field(default="INFO", description="Root log level (DEBUG/INFO/WARNING/ERROR)")
    log_format: str = Field(default="json", description="Log format: 'json' or 'text'")
    preferred_categories: str = Field(
        default="cs.AI,cs.LG,cs.CL", description="Comma-separated preferred arXiv categories for scoring"
    )
    max_upload_mb: int = Field(default=50, gt=0, description="Maximum PDF upload size in megabytes")
    api_key: str = Field(default="", description="X-API-Key for write and read endpoints; empty = no auth (#241)")
    pdf_extract_timeout: int = Field(
        default=120,
        description=(
            "Max seconds for extract_text() before the ingest job is failed (#238); "
            "there is no page-count cap, only wall-clock, so malformed/huge PDFs can't hang a job forever"
        ),
    )
    llm_generate_timeout: int = Field(
        default=300,
        description=(
            "Ceiling in seconds for RAGSummarizer's llm.generate() call (#237); "
            "must stay >= the slowest configured LLM client timeout (ollama_timeout) "
            "so it never truncates a legitimate in-flight generation"
        ),
    )
    summarize_total_timeout: int = Field(
        default=460,
        gt=0,
        description=(
            "Overall ceiling in seconds for RAGSummarizer.summarize() (#269); the embedding, "
            "Qdrant and LLM wait_for calls inside it are awaited sequentially, so their timeouts "
            "stack in the worst case instead of applying independently. This bounds the whole "
            "call regardless of that stacking — default is embedding_timeout + qdrant_timeout + "
            "llm_generate_timeout with a small margin for the DB fallback path."
        ),
    )

    @field_validator(
        "embedding_svc_url", "qdrant_url", "api_key", "embedding_api_key", "qdrant_api_key", "google_api_key"
    )
    @classmethod
    def reject_placeholder(cls, v: str) -> str:
        if "<" in v:
            raise ValueError(
                f"Invalid value {v!r}: contains placeholder. "
                "Set the corresponding environment variable before starting."
            )
        return v

    @model_validator(mode="after")
    def check_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size}); "
                "otherwise the sliding window step (chunk_size - chunk_overlap) is <= 0 and chunk_pages() "
                "can loop forever. Set CHUNK_OVERLAP and CHUNK_SIZE environment variables consistently."
            )
        return self

    @property
    def preferred_categories_list(self) -> list[str]:
        """Parse preferred_categories CSV into a list."""
        return [c.strip() for c in self.preferred_categories.split(",") if c.strip()]

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
