import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Matches a whole-value placeholder like "<your-api-key>" or "<placeholder>" (#331):
# anchored to the full string so a legitimate secret that merely happens to contain
# a "<" somewhere in the middle (e.g. a random base64/hex string) is not rejected.
_PLACEHOLDER_RE = re.compile(r"^<.+>$")


def _reject_placeholder(value: str, *, whole: bool) -> str:
    """Raise ValueError if ``value`` looks like an unfilled ``<...>`` placeholder (#732).

    whole=False (URLs): any "<" is a placeholder, since the defaults embed "<internal-host>".
    whole=True (secrets): only a full "<...>" value is rejected, because a real random secret
    may legitimately contain "<" somewhere in the middle (#331).
    """
    bad = _PLACEHOLDER_RE.match(value) if whole else "<" in value
    if bad:
        raise ValueError(
            f"Invalid value {value!r}: contains placeholder. Set the corresponding environment variable before starting."
        )
    return value


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
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between consecutive chunks")
    qdrant_collection: str = Field(default="academic-papers", description="Qdrant collection name")
    embedding_dim: int = Field(
        default=768,
        gt=0,
        description="Embedding vector dimension; must match the embedding model and Qdrant collection",
    )
    port: int = Field(default=8020, gt=0, description="Port for API server")
    google_api_key: str = Field(default="", description="Google API key for generative AI")
    llm_provider: Literal["auto", "gemini", "ollama", "none"] = Field(
        default="auto",
        description=(
            "LLM provider: auto = Gemini if GOOGLE_API_KEY is set else Ollama; "
            "gemini/ollama = explicit; none = disable LLM (#498)"
        ),
    )
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name (#498)")
    gemini_timeout_ms: int = Field(default=60000, gt=0, description="Gemini API HTTP timeout in milliseconds")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama service URL")
    ollama_model: str = Field(default="mistral", description="Ollama model to use")
    ollama_timeout: int = Field(
        default=300,
        gt=0,
        description=(
            "Per-attempt Ollama HTTP timeout in seconds. OllamaClient.generate() retries up "
            "to 3x (#311), so llm_generate_timeout must cover ollama_timeout*3+backoff, not "
            "just a single attempt — raising this value requires raising llm_generate_timeout "
            "(and summarize_total_timeout) to match"
        ),
    )
    otel_endpoint: str = Field(default="", description="OpenTelemetry endpoint")
    log_level: str = Field(default="INFO", description="Root log level (DEBUG/INFO/WARNING/ERROR)")
    log_format: str = Field(default="json", description="Log format: 'json' or 'text'")
    preferred_categories: str = Field(
        default="cs.AI,cs.LG,cs.CL", description="Comma-separated preferred arXiv categories for scoring"
    )
    max_upload_mb: int = Field(default=50, gt=0, description="Maximum PDF upload size in megabytes")
    enable_docs: bool = Field(
        default=False,
        description="Expose /docs, /redoc and /openapi.json (ENABLE_DOCS=true). Off by default: they are unauthenticated (#648)",
    )
    api_key: str = Field(default="", description="X-API-Key for write and read endpoints; empty = no auth (#241)")
    auth_disabled: bool = Field(
        default=False,
        description=(
            "Explicitly run without authentication (AUTH_DISABLED=true). Required to start when "
            "api_key and api_keys are both empty; otherwise startup is refused (fail-closed, #646)"
        ),
    )
    api_keys: str = Field(
        default="",
        description=(
            "Comma-separated additional X-API-Keys accepted alongside api_key (key rotation, #601); "
            "empty elements are dropped; api_key and api_keys both empty = no auth"
        ),
    )
    ingest_api_key: str = Field(
        default="",
        description=(
            "X-API-Key limited to the ingest scope (POST /papers/ingest only once scopes are "
            "enforced, #355); empty = not configured. Setting it alone still enables auth"
        ),
    )
    pdf_extract_timeout: int = Field(
        default=120,
        gt=0,
        description=(
            "Max seconds for extract_text() before the ingest job is failed (#238); "
            "there is no page-count cap, only wall-clock, so malformed/huge PDFs can't hang a job forever"
        ),
    )
    llm_generate_timeout: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Ceiling in seconds for RAGSummarizer's llm.generate() call (#237). "
            "OllamaClient.generate() retries up to 3x on transient errors (#311), each "
            "attempt bounded by ollama_timeout, so the worst case is "
            "ollama_timeout * 3 + backoff(1s + 2s). Unset = derived as ollama_timeout*3+3 "
            "(903 with defaults). An explicit value below that is rejected, since a tighter "
            "value silently defeats the retry by cancelling after the first attempt's timeout (#651)."
        ),
    )
    summarize_total_timeout: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Overall ceiling in seconds for RAGSummarizer.summarize() (#269); the embedding, "
            "Qdrant and LLM wait_for calls inside it are awaited sequentially, so their timeouts "
            "stack in the worst case. Unset = derived as embedding_timeout + qdrant_timeout + "
            "llm_generate_timeout + 10 (1063 with defaults); an explicit value below that is rejected (#651)."
        ),
    )

    @field_validator("embedding_svc_url", "qdrant_url")
    @classmethod
    def reject_placeholder_url(cls, v: str) -> str:
        return _reject_placeholder(v, whole=False)

    @field_validator("api_key", "ingest_api_key", "embedding_api_key", "qdrant_api_key", "google_api_key")
    @classmethod
    def reject_placeholder_secret(cls, v: str) -> str:
        return _reject_placeholder(v, whole=True)

    @field_validator("api_keys")
    @classmethod
    def reject_placeholder_api_keys(cls, v: str) -> str:
        # Same whole-"<...>" rule as api_key, applied per element (#601).
        for item in v.split(","):
            _reject_placeholder(item.strip(), whole=True)
        return v

    @model_validator(mode="after")
    def derive_timeouts(self) -> "Settings":
        """Fill unset llm_generate_timeout / summarize_total_timeout; reject explicit values below the floor (#651)."""
        llm_floor = self.ollama_timeout * 3 + 3
        if self.llm_generate_timeout is None:
            self.llm_generate_timeout = llm_floor
        elif self.llm_generate_timeout < llm_floor:
            raise ValueError(
                f"llm_generate_timeout ({self.llm_generate_timeout}) must be >= ollama_timeout*3+3 ({llm_floor}); "
                "otherwise the Ollama retries are cancelled early. Raise LLM_GENERATE_TIMEOUT or unset it."
            )
        total_floor = self.embedding_timeout + self.qdrant_timeout + self.llm_generate_timeout + 10
        if self.summarize_total_timeout is None:
            self.summarize_total_timeout = total_floor
        elif self.summarize_total_timeout < total_floor:
            raise ValueError(
                f"summarize_total_timeout ({self.summarize_total_timeout}) must be >= embedding_timeout + "
                f"qdrant_timeout + llm_generate_timeout + 10 ({total_floor}). "
                "Raise SUMMARIZE_TOTAL_TIMEOUT or unset it."
            )
        return self

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
    def api_keys_list(self) -> list[str]:
        """Parse api_keys CSV into a list, dropping empty / whitespace-only elements (#601)."""
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def accepted_api_keys(self) -> list[str]:
        """All keys that authenticate a request: api_key plus api_keys. Empty = auth disabled (#601)."""
        return ([self.api_key] if self.api_key else []) + self.api_keys_list

    @property
    def auth_enabled(self) -> bool:
        """True if any of api_key / api_keys / ingest_api_key is set (#626)."""
        return bool(self.accepted_api_keys or self.ingest_api_key)

    @property
    def preferred_categories_list(self) -> list[str]:
        """Parse preferred_categories CSV into a list."""
        return [c.strip() for c in self.preferred_categories.split(",") if c.strip()]

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (cached, #389)."""
    return Settings()


# Backward-compatible alias: existing ``from academic_paper.config import settings``
# keeps working and is the same object as ``get_settings()``.
settings = get_settings()
