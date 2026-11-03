"""Thin FastAPI dependency providers for app-level objects (#360).

Each reads ``request.app.state`` (set in server.py lifespan) so handlers can be tested with
``app.dependency_overrides`` instead of patching ``app.state``. llm / summarizer may be None.
"""

import asyncio
from typing import Any

from fastapi import Request


def get_embedder(request: Request) -> Any:
    return request.app.state.embedder


def get_vector_store(request: Request) -> Any:
    return request.app.state.vector_store


def get_llm(request: Request) -> Any:
    return request.app.state.llm


def get_summarizer(request: Request) -> Any:
    return request.app.state.summarizer


def get_active_ingest_tasks(request: Request) -> set[asyncio.Task] | None:
    return getattr(request.app.state, "active_ingest_tasks", None)
