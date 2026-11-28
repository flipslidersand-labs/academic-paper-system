"""#865: docs/ is published as GitHub Pages, so no JSON under it may carry internal columns.

test_generate_portfolio_main.py already pins the fields that generate_portfolio.py writes to
papers.json (#856). This test covers every JSON file that is committed under docs/, including
hand-written records such as migration logs, which the generator never touches.
"""

import json
from pathlib import Path

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
