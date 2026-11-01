"""Guard against re-introducing the module-level ``settings`` alias (#634, #389)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("academic_paper", "scripts", "tests")


def _offending_imports(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "academic_paper.config"
        and any(alias.name == "settings" for alias in node.names)
    ]


def test_no_module_imports_the_settings_alias():
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for d in SCAN_DIRS
        for path in (ROOT / d).rglob("*.py")
        for line in _offending_imports(path)
    ]
    assert offenders == [], f"use get_settings() instead of the settings alias: {offenders}"


def test_config_module_has_no_settings_alias():
    from academic_paper import config

    assert not hasattr(config, "settings")
