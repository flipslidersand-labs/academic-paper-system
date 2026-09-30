"""Guard against #436: requirements.lock must carry --hash entries.

The Dockerfile installs requirements.lock with `pip install --require-hashes`
so a mirror/proxy swap or cache poisoning of an already-pinned version can't
silently slip a different artifact into the build. That protection only
holds if every single line in the lock actually carries a hash; a manual
re-generation without `--generate-hashes` would drop it silently.
"""

from pathlib import Path

LOCK_FILE = Path(__file__).parent.parent / "requirements.lock"


def _pinned_package_lines() -> list[str]:
    """Top-level `name==version` lines (skip comments, `# via` continuations, hash lines)."""
    lines = []
    for line in LOCK_FILE.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith(" ") or line.startswith("\t"):
            continue
        lines.append(line)
    return lines


def test_header_records_generate_hashes_flag():
    header = LOCK_FILE.read_text().splitlines()[:6]
    assert any("--generate-hashes" in line for line in header), (
        "requirements.lock header must record --generate-hashes so a manual "
        "re-generation (without it) is obvious from the diff."
    )


def test_every_pinned_package_has_at_least_one_hash():
    text = LOCK_FILE.read_text()
    packages = _pinned_package_lines()
    assert packages, "requirements.lock appears empty; sanity check failed."

    missing = []
    for pkg in packages:
        name = pkg.split("==")[0].split(" ")[0]
        # Find this package's block (from its own line up to the next unindented line).
        start = text.index(pkg)
        end = text.find("\n" + name.split("[")[0], start + 1)
        # Fall back to searching from the next top-level entry if name has extras.
        block = text[start : end if end != -1 else len(text)]
        if "--hash=sha256:" not in block:
            missing.append(pkg)

    assert not missing, (
        f"requirements.lock lines missing --hash: {missing}. Regenerate with "
        f"`pip-compile --generate-hashes --no-strip-extras --output-file=requirements.lock "
        f"pyproject.toml` (see #436)."
    )
