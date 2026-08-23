"""RAG-based paper summarizer using LLM and vector store."""

import json
import logging
import re
import sqlite3

from academic_paper.embedder import EmbedderClient
from academic_paper.llm import BaseLLMClient
from academic_paper.vector_store import QdrantStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert academic paper analyzer.
Your task is to provide a structured summary of academic papers.
Focus on clarity, accuracy, and extracting key information."""


class RAGSummarizer:
    """Summarize academic papers using RAG (Retrieval-Augmented Generation)."""

    def __init__(self, llm_client: BaseLLMClient, qdrant_store: QdrantStore, embedder: EmbedderClient | None = None):
        """Initialize RAGSummarizer.

        Args:
            llm_client: LLM client for generation
            qdrant_store: Qdrant vector store for retrieval
            embedder: Embedding client for semantic chunk retrieval (optional; falls back to zero vector)
        """
        self.llm = llm_client
        self.qdrant = qdrant_store
        self.embedder = embedder

    async def summarize(
        self,
        paper_id: int,
        file_hash: str,
        top_k: int = 5,
        title: str | None = None,
        file_name: str | None = None,
    ) -> dict:
        """Summarize a paper using RAG.

        Args:
            paper_id: ID of the paper to summarize
            file_hash: File hash of the paper (for Qdrant queries)
            top_k: Number of top chunks to use for context (default 5)

        Returns:
            Dictionary with keys: objective, method, results, limitations, keywords

        Raises:
            ValueError: If no chunks found or LLM returns invalid JSON
        """
        # Build a real query vector from title or filename for semantic chunk retrieval
        query_text = title or (file_name.removesuffix(".pdf") if file_name else None) or "academic paper"
        query_vector: list[float]
        if self.embedder is not None:
            try:
                query_vector = await self.embedder.embed_single(query_text, mode="search")
            except Exception:
                logger.warning("embed_single failed for summarize query=%r — falling back to zero vector", query_text)
                query_vector = [0.0] * 768
        else:
            query_vector = [0.0] * 768

        # Try to retrieve relevant chunks from Qdrant first
        try:
            chunks = self.qdrant.search(
                query_vector=query_vector,
                limit=top_k,
                paper_id_filter=paper_id,
            )
        except (AttributeError, TypeError):
            # If Qdrant mock doesn't support search, try alternative approach
            # This handles both real and mocked Qdrant instances
            try:
                from academic_paper.config import settings
                from academic_paper.db import get_chunks, get_connection

                conn = get_connection(settings.academic_db)
                chunks_db = get_chunks(conn, paper_id)
                conn.close()

                if not chunks_db:
                    raise ValueError(f"No chunks found for paper {paper_id}")

                # Convert database chunks to Qdrant-like format
                chunks = []
                for chunk in chunks_db[:top_k]:
                    chunks.append(
                        {
                            "payload": {
                                "paper_id": paper_id,
                                "page_start": chunk.get("page_start", "unknown"),
                                "text": chunk["text"],
                            }
                        }
                    )
            except (sqlite3.OperationalError, FileNotFoundError):
                raise ValueError(f"No chunks found for paper {paper_id}")

        if not chunks:
            raise ValueError(f"No chunks found for paper {paper_id}")

        # Prepare context from chunks with page information
        context_parts = []
        for chunk in chunks[:top_k]:
            payload = chunk.get("payload", {})
            page_start = payload.get("page_start", "unknown")
            text = payload.get("text", "")
            if text:
                context_parts.append(f"Page {page_start}: {text}")

        if not context_parts:
            raise ValueError(f"No valid content found in chunks for paper {paper_id}")

        context = "\n\n".join(context_parts)

        # Generate summary using LLM
        prompt = f"""Please summarize the following academic paper content and provide a structured summary in JSON format.

Paper content:
{context}

Please respond ONLY with valid JSON in this exact format:
{{
    "objective": "Main objective or research question",
    "method": "Methodology used",
    "results": "Key findings and results",
    "limitations": "Study limitations",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        response = await self.llm.generate(prompt, system=SYSTEM_PROMPT)

        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                summary_data = json.loads(json_match.group())
            else:
                summary_data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")

        # Normalize fields to strings (LLM sometimes returns nested dicts)
        for field in ("objective", "method", "results", "limitations"):
            val = summary_data.get(field, "")
            if not isinstance(val, str):
                summary_data[field] = json.dumps(val, ensure_ascii=False)
        keywords = summary_data.get("keywords", [])
        if not isinstance(keywords, list):
            summary_data["keywords"] = [str(keywords)]
        else:
            summary_data["keywords"] = [str(k) for k in keywords]

        return summary_data
