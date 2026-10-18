"""generate_portfolio.main(): empty-fetch guard and output files (#736)."""

import json
import sys
from urllib.error import URLError

# scripts/ is added to sys.path in tests/conftest.py (#272).
import generate_portfolio
import pytest

_PAPERS = [
    {"id": 1, "title": "日本語タイトル", "authors": ["A"], "score": 0.8, "source": "arxiv", "file_name": "x.pdf"},
    {"id": 2, "title": "Second", "authors": ["B"], "score": 0.4, "source": "arxiv", "file_name": "y.pdf"},
]


def _run(monkeypatch, out_dir, fetch):
    monkeypatch.setattr(generate_portfolio, "fetch_all", fetch)
    monkeypatch.setattr(
        sys, "argv", ["generate_portfolio.py", "--api-url", "http://api/", "--output-dir", str(out_dir)]
    )
    generate_portfolio.main()


def _fetch_factory(papers, summaries, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        return papers if "/papers" in url else summaries

    return fetch


def test_empty_papers_exits_1_and_writes_no_files(monkeypatch, tmp_path):
    out = tmp_path / "docs"
    with pytest.raises(SystemExit) as ei:
        _run(monkeypatch, out, _fetch_factory([], []))
    assert ei.value.code == 1
    assert not (out / "index.html").exists()
    assert not (out / "papers.json").exists()


def test_empty_papers_does_not_fetch_summaries(monkeypatch, tmp_path):
    calls = []
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path / "d", _fetch_factory([], [], calls))
    assert calls == ["http://api/papers?sort=score"]


def test_writes_index_html_and_papers_json(monkeypatch, tmp_path):
    out = tmp_path / "nested" / "docs"  # mkdir(parents=True)
    calls = []
    _run(monkeypatch, out, _fetch_factory(_PAPERS, [], calls))

    assert calls == ["http://api/papers?sort=score", "http://api/summaries?"]
    html_text = (out / "index.html").read_text(encoding="utf-8")
    assert "2 papers" in html_text
    assert "日本語タイトル" in html_text

    raw = (out / "papers.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["count"] == len(_PAPERS) == 2
    assert data["papers"] == _PAPERS
    assert data["updated"] and len(data["updated"]) == 10
    assert "日本語タイトル" in raw  # ensure_ascii=False: not \u-escaped
    assert "\\u" not in raw


def test_summary_objective_is_reflected_by_paper_id(monkeypatch, tmp_path):
    summaries = [{"paper_id": 2, "objective": "OBJECTIVE-FOR-SECOND"}, {"paper_id": 99, "objective": "ORPHAN-OBJ"}]
    _run(monkeypatch, tmp_path, _fetch_factory(_PAPERS, summaries))
    html_text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "OBJECTIVE-FOR-SECOND" in html_text
    assert "ORPHAN-OBJ" not in html_text  # summaries for unknown papers are ignored


def test_partial_fetch_failure_is_not_detected_by_main(monkeypatch, tmp_path):
    """Characterization (current behavior, see PR notes): fetch_all swallows URLError/HTTPError
    mid-pagination and returns what it has, so main() publishes partial data and exits 0.
    If this is later hardened (e.g. fetch_all raises / main exits non-zero), update this test.
    """
    calls = {"n": 0}

    def urlopen(req, timeout=15):
        calls["n"] += 1
        if "/summaries" in req.full_url:
            raise URLError("summaries down")

        class _Resp:
            def read(self):
                return json.dumps({"papers": _PAPERS}).encode()

        return _Resp()

    monkeypatch.setattr(generate_portfolio, "urlopen", urlopen)
    monkeypatch.setattr(sys, "argv", ["g", "--api-url", "http://api", "--output-dir", str(tmp_path)])

    generate_portfolio.main()  # no SystemExit

    data = json.loads((tmp_path / "papers.json").read_text(encoding="utf-8"))
    assert data["count"] == 2
    assert "OBJECTIVE" not in (tmp_path / "index.html").read_text(encoding="utf-8")
