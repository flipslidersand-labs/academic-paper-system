"""Guard against #488: frontend CDN scripts must be version-pinned and carry SRI."""

import re
from pathlib import Path

INDEX = Path(__file__).parent.parent / "frontend" / "index.html"


def test_no_tailwind_play_cdn():
    assert "cdn.tailwindcss.com" not in INDEX.read_text()


def test_external_scripts_pinned_with_sri():
    tags = re.findall(r"<script\b[^>]*\bsrc=\"https?://[^\"]+\"[^>]*>", INDEX.read_text())
    assert tags, "expected at least the alpinejs CDN script"
    for tag in tags:
        assert re.search(r"@\d+\.\d+\.\d+/", tag), f"CDN URL must pin an exact version: {tag}"
        assert 'integrity="sha384-' in tag, f"missing SRI: {tag}"
        assert 'crossorigin="anonymous"' in tag, f"missing crossorigin: {tag}"
