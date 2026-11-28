"""#865: docs/ is published as GitHub Pages, so no JSON under it may carry internal columns.

test_generate_portfolio_main.py already pins the fields that generate_portfolio.py writes to
papers.json (#856). This test covers every JSON file that is committed under docs/, including
hand-written records such as migration logs, which the generator never touches.
"""

import json
from pathlib import Path

import generate_portfolio  # scripts/ is on sys.path via tests/conftest.py
import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

# Internal columns / identifiers that must never appear as keys in a published JSON file.
INTERNAL_KEYS = {
    "chunk_head",  # excerpt of the paper body
    "db_file_name",
    "qdrant_file_name",
    "new_file_name",
    "file_hash",
    "paper_id",
    "ingested_at",
}


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


@pytest.mark.parametrize(
    "path",
    [pytest.param(p, id=p.relative_to(DOCS_DIR).as_posix()) for p in sorted(DOCS_DIR.rglob("*.json"))],
)
def test_published_json_exposes_no_internal_columns(path):
    found = INTERNAL_KEYS & set(_keys(json.loads(path.read_text(encoding="utf-8"))))
    assert not found, f"internal columns published under docs/: {sorted(found)}"


def test_papers_json_keys_are_allowlisted_only():
    """papers.json must carry only the allowlisted columns (+ arxiv_id), not merely avoid a denylist (#893).

    INTERNAL_KEYS above is a denylist, so a column it does not know (id, file_name, status, ...) would pass.
    The allowlist is the generator's own, so a stale committed file fails here rather than going out unnoticed.
    """
    allowed = set(generate_portfolio.PUBLIC_PAPER_FIELDS) | {"arxiv_id"}
    papers = json.loads((DOCS_DIR / "papers.json").read_text(encoding="utf-8"))["papers"]
    assert papers, "docs/papers.json has no papers"
    extra = {key for paper in papers for key in paper} - allowed
    assert not extra, f"non-allowlisted columns published in docs/papers.json: {sorted(extra)}"
