"""RAG-based summarizer for academic papers using LLM and vector store."""

import json
from academic_paper.llm import BaseLLMClient
from academic_paper.vector_store import QdrantStore

SYSTEM_PROMPT = """You are an academic paper summarization assistant.
Analyze the provided paper chunks and return a JSON summary with these exact keys:
- objective: The main research goal or question (1-2 sentences)
- method: The approach, methodology, or techniques used (2-3 sentences)
- results: Key findings and outcomes (2-3 sentences)
- limitations: Acknowledged limitations or future work (1-2 sentences)
- keywords: List of 5-8 key technical terms (array of strings)

Return ONLY valid JSON, no markdown, no explanation."""


class RAGSummarizer:
    """RAG-based summarizer that fetches paper chunks and generates structured summaries."""

    def __init__(self, llm: BaseLLMClient, qdrant: QdrantStore):
        """Initialize RAGSummarizer with LLM client and vector store.

        Args:
            llm: BaseLLMClient instance for generating summaries
            qdrant: QdrantStore instance for retrieving chunks
        """
        self.llm = llm
        self.qdrant = qdrant

    async def summarize(self, paper_id: int, file_hash: str, top_k: int = 8) -> dict:
        """Generate a structured summary of a paper using RAG.

        Args:
            paper_id: ID of the paper in the database
            file_hash: Hash of the paper file
            top_k: Number of top chunks to retrieve (default 8)

        Returns:
            Dictionary with keys: objective, method, results, limitations, keywords

        Raises:
            ValueError: If LLM returns invalid JSON
        """
        # 1. Retrieve top_k chunks using paper_id filter
        # Using zero vector as query since we want all chunks for the paper
        query_vector = [0.0] * 768
        chunks = self.qdrant.search(query_vector, limit=top_k, paper_id_filter=paper_id)

        # 2. Build context string from chunks
        context = self._build_context(chunks)

        # 3. Call LLM to generate summary
        prompt = f"Please summarize the following academic paper:\n\n{context}"
        response = await self.llm.generate(prompt, system=SYSTEM_PROMPT)

        # 4. Parse and validate JSON response
        try:
            summary = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {response}") from e

        # 5. Validate required keys
        required_keys = {"objective", "method", "results", "limitations", "keywords"}
        if not all(key in summary for key in required_keys):
            raise ValueError(f"Missing required keys. Expected {required_keys}, got {set(summary.keys())}")

        return summary

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks.

        Args:
            chunks: List of chunk dicts from Qdrant search results

        Returns:
            Formatted context string with page information
        """
        parts = []
        for chunk in chunks:
            # Extract page number from payload (nested structure from Qdrant)
            page = chunk.get("payload", {}).get("page_start") or chunk.get("page_start", "?")
            text = chunk.get("payload", {}).get("text") or chunk.get("text", "")
            parts.append(f"Page {page}:\n{text}")
        return "\n\n".join(parts)
