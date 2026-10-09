"""Shared data models."""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Field descriptions double as the hints shown in the summarizer prompt's JSON example.


class PaperSummary(BaseModel):
    """Structured summary of a paper (single source of truth for field names)."""

    objective: str = Field("", description="Main objective or research question")
    method: str = Field("", description="Methodology used")
    results: str = Field("", description="Key findings and results")
    limitations: str = Field("", description="Study limitations")
    keywords: list[str] = Field(default_factory=list, description="")

    @field_validator("objective", "method", "results", "limitations", mode="before")
    @classmethod
    def _coerce_text(cls, v: Any) -> Any:
        # LLM sometimes returns nested dicts/lists — keep them as JSON strings.
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(k) for k in v if k is not None]
        return [str(v)]
