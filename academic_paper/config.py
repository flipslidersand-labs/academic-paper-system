from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for academic paper system."""

    embedding_svc_url: str = Field(
        default="http://192.168.68.63:9092",
        description="Embedding service URL"
    )
    embedding_api_key: str = Field(
        default="",
        description="API key for embedding service"
    )
    qdrant_url: str = Field(
        default="http://192.168.68.63:6333",
        description="Qdrant vector database URL"
    )
    qdrant_api_key: str = Field(
        default="",
        description="API key for Qdrant"
    )
    academic_db: str = Field(
        default="/data/academic.db",
        description="Path to academic database"
    )
    chunk_size: int = Field(
        default=512,
        description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlap between consecutive chunks"
    )
    qdrant_collection: str = Field(
        default="academic-papers",
        description="Qdrant collection name"
    )
    port: int = Field(
        default=8020,
        description="Port for API server"
    )
    google_api_key: str = Field(
        default="",
        description="Google API key for generative AI"
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama service URL"
    )
    ollama_model: str = Field(
        default="mistral",
        description="Ollama model to use"
    )
    otel_endpoint: str = Field(
        default="",
        description="OpenTelemetry endpoint"
    )

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
