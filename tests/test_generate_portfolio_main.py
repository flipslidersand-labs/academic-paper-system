"""generate_portfolio.main(): empty-fetch guard and output files (#736)."""

import json
import sys
from urllib.error import HTTPError, URLError

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
    assert [p["title"] for p in data["papers"]] == ["日本語タイトル", "Second"]
    assert data["updated"] and len(data["updated"]) == 10
    assert "日本語タイトル" in raw  # ensure_ascii=False: not \u-escaped
    assert "\\u" not in raw


def test_summary_objective_is_reflected_by_paper_id(monkeypatch, tmp_path):
    summaries = [{"paper_id": 2, "objective": "OBJECTIVE-FOR-SECOND"}, {"paper_id": 99, "objective": "ORPHAN-OBJ"}]
    _run(monkeypatch, tmp_path, _fetch_factory(_PAPERS, summaries))
    html_text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "OBJECTIVE-FOR-SECOND" in html_text
    assert "ORPHAN-OBJ" not in html_text  # summaries for unknown papers are ignored


@pytest.mark.parametrize(
    "exc",
    [
        URLError("summaries down"),
        HTTPError("http://api/summaries", 503, "Unavailable", {}, None),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
    ],
)
def test_partial_fetch_failure_exits_1_and_keeps_existing_output(monkeypatch, tmp_path, exc):
    """#771: any non-auth fetch error aborts before writing; existing docs/ stays intact."""
    (tmp_path / "index.html").write_text("OLD-HTML", encoding="utf-8")
    (tmp_path / "papers.json").write_text("OLD-JSON", encoding="utf-8")

    def urlopen(req, timeout=15):
        if "/summaries" in req.full_url:
            raise exc

        class _Resp:
            def read(self):
                return json.dumps({"papers": _PAPERS}).encode()

        return _Resp()

    monkeypatch.setattr(generate_portfolio, "urlopen", urlopen)
    monkeypatch.setattr(sys, "argv", ["g", "--api-url", "http://api", "--output-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as ei:
        generate_portfolio.main()

    assert ei.value.code == 1
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "OLD-HTML"
    assert (tmp_path / "papers.json").read_text(encoding="utf-8") == "OLD-JSON"


def test_papers_json_exposes_only_allowlisted_fields(monkeypatch, tmp_path):
    """#856: internal columns (file_hash, ingested_at, status, id, file_name ...) must not be published."""
    papers = [
        {
            "id": 7,
            "file_name": "arxiv_2610.10536v1.pdf",
            "file_hash": "deadbeef" * 8,
            "title": "T",
            "authors": ["A"],
            "year": None,
            "pages": 3,
            "ingested_at": "2026-10-08T05:45:40+00:00",
            "status": "indexed",
            "categories": ["cs.LG"],
            "published_date": "2026-10-07",
            "source": "arxiv",
            "score": 0.9,
        },
        {"id": 8, "file_name": "local.pdf", "file_hash": "cafe" * 16, "title": "L", "source": "local", "score": 0.1},
    ]
    _run(monkeypatch, tmp_path, _fetch_factory(papers, []))
    raw = (tmp_path / "papers.json").read_text(encoding="utf-8")
    out = json.loads(raw)["papers"]
    allowed = {"title", "authors", "categories", "published_date", "source", "score", "arxiv_id"}
    for p in out:
        assert set(p) <= allowed
    assert out[0]["arxiv_id"] == "2610.10536"
    assert "arxiv_id" not in out[1]
    assert "deadbeef" not in raw and "cafe" * 4 not in raw
    assert "file_hash" not in raw and "ingested_at" not in raw
