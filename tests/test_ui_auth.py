"""#386: the bundled SPA must send X-API-Key; data endpoints stay gated."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.server import app

INDEX = Path(__file__).parent.parent / "frontend" / "index.html"


def _client(temp_db):
    with (
        patch.object(settings, "academic_db", temp_db),
        patch("academic_paper.server.EmbedderClient", return_value=MagicMock(embed=AsyncMock())),
        patch("academic_paper.server.QdrantStore", return_value=MagicMock()),
    ):
        return TestClient(app)


def test_frontend_sends_api_key_on_every_api_call():
    html = INDEX.read_text(encoding="utf-8")
    # The only raw fetch() is inside apiFetch; everything else must go through it.
    assert len(re.findall(r"(?<![\w.])fetch\(", html)) == 1
    assert "headers.set('X-API-Key'" in html
    assert "localStorage.getItem('apiKey')" in html
    assert 'type="password"' in html


def test_api_data_endpoints_require_key_while_ui_shell_is_static(temp_db):
    client = _client(temp_db)
    with patch.object(settings, "api_key", "secret-key"):
        # Every endpoint the SPA calls is gated; with the header it works.
        for path in ("/stats", "/papers", "/search?q=x", "/papers/1/summary"):
            assert client.get(path).status_code == 401, path
            assert client.get(path, headers={"X-API-Key": "secret-key"}).status_code != 401, path
        # The static shell holds no data (browser navigation cannot send X-API-Key).
        assert client.get("/ui/").status_code == 200
