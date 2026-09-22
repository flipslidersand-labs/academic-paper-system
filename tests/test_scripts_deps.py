"""Guard against #432: a script importing a third-party package that isn't declared.

arxiv-daily.yml only ran `pip install httpx` for three weeks while
scripts/arxiv_collect.py and scripts/pubmed_collect.py imported defusedxml,
which is declared in pyproject.toml but was never actually installed on that
workflow's runner. This can't happen again silently: every top-level import
in scripts/*.py that isn't stdlib, the local script modules, or academic_paper
itself must resolve to a name declared in [project.dependencies].
"""

import ast
import sys
import tomllib
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# Import name -> distribution name, for the (currently empty) cases where they differ.
_IMPORT_TO_DIST = {}

_LOCAL_MODULES = {p.stem for p in SCRIPTS_DIR.glob("*.py")} | {"academic_paper"}


def _declared_dependency_names() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    # "pdfplumber>=0.11" -> "pdfplumber"
    names = set()
    for spec in deps:
        name = spec.split(">=")[0].split("==")[0].split("[")[0].strip()
        names.add(name.lower().replace("-", "_"))
    return names


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_third_party_import_in_scripts_is_a_declared_dependency():
    stdlib = set(sys.stdlib_module_names)
    declared = _declared_dependency_names()

    undeclared: dict[str, set[str]] = {}
    for script in SCRIPTS_DIR.glob("*.py"):
        for name in _top_level_imports(script):
            if name in stdlib or name in _LOCAL_MODULES:
                continue
            dist = _IMPORT_TO_DIST.get(name, name).lower().replace("-", "_")
            if dist not in declared:
                undeclared.setdefault(script.name, set()).add(name)

    assert not undeclared, (
        f"scripts/*.py import packages not declared in pyproject.toml [project.dependencies]: "
        f"{undeclared}. A CI step that installs only some of these (like arxiv-daily.yml did "
        f"for #432) will break at runtime instead of at review time."
    )
