"""ruff pin must be identical in pyproject.toml, ci.yml and .pre-commit-config.yaml (#437)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _find(path: str, pattern: str) -> str:
    m = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
    assert m, f"ruff pin not found in {path}"
    return m.group(1)


def test_ruff_version_pins_in_sync():
    pins = {
        "pyproject.toml": _find("pyproject.toml", r'"ruff==([\d.]+)"'),
        ".github/workflows/ci.yml": _find(".github/workflows/ci.yml", r'^\s*ruff-version:\s*"([\d.]+)"'),
        ".pre-commit-config.yaml": _find(".pre-commit-config.yaml", r"^\s*rev:\s*v([\d.]+)"),
    }
    assert len(set(pins.values())) == 1, f"ruff pins drifted, bump all together: {pins}"
