"""Tests for scripts/fix_file_names.py (#312)."""

import argparse
import json
import sqlite3

import fix_file_names  # scripts/ is on sys.path via tests/conftest.py
import httpx
import pytest
import respx

QDRANT = "http://qdrant.test:6333"
PAYLOAD_URL = f"{QDRANT}/collections/c/points/payload"


def test_detect_arxiv_id_forward():
    assert fix_file_names._detect_arxiv_id("header arXiv:2301.00001v2 [cs.AI] footer") == ("2301.00001", "v2")


def test_detect_arxiv_id_defaults_version():
    assert fix_file_names._detect_arxiv_id("arXiv:2301.00001 [cs.AI]") == ("2301.00001", "v1")


def test_detect_arxiv_id_mirrored():
    # pdfplumber sometimes extracts the rotated watermark reversed.
    assert fix_file_names._detect_arxiv_id("1v17001.0142:viXra") == ("2410.10071", "v1")


def test_detect_arxiv_id_not_found():
    assert fix_file_names._detect_arxiv_id("no watermark here") is None


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, file_name TEXT)")
    conn.executemany("INSERT INTO papers VALUES (?, ?)", [(1, "old1.pdf"), (2, "old2.pdf")])
    conn.commit()
    conn.close()
    return str(path)


def _args(tmp_path, db_path, entries, dry_run=False):
    mapping = tmp_path / "m.json"
    mapping.write_text(json.dumps({"entries": entries}))
    return argparse.Namespace(mapping=str(mapping), db=db_path, qdrant_url=QDRANT, collection="c", dry_run=dry_run)


def _entries():
    return [
        {"paper_id": 1, "db_file_name": "old1.pdf", "new_file_name": "new1.pdf", "status": "rename"},
        {"paper_id": 2, "db_file_name": "old2.pdf", "new_file_name": "new2.pdf", "status": "rename"},
    ]


def _names(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute("SELECT id, file_name FROM papers").fetchall())
    finally:
        conn.close()


@respx.mock
def test_apply_success_updates_db_and_qdrant(tmp_path, db_path):
    route = respx.post(PAYLOAD_URL).mock(return_value=httpx.Response(200, json={"result": {}}))
    assert fix_file_names.cmd_apply(_args(tmp_path, db_path, _entries())) == 0
    assert _names(db_path) == {1: "new1.pdf", 2: "new2.pdf"}
    assert route.call_count == 2


@respx.mock
def test_apply_dry_run_writes_nothing(tmp_path, db_path):
    route = respx.post(PAYLOAD_URL).mock(return_value=httpx.Response(200))
    assert fix_file_names.cmd_apply(_args(tmp_path, db_path, _entries(), dry_run=True)) == 0
    assert _names(db_path) == {1: "old1.pdf", 2: "old2.pdf"}
    assert route.call_count == 0


@respx.mock
def test_apply_detects_stale_mapping(tmp_path, db_path):
    entries = _entries()
    entries[0]["db_file_name"] = "something-else.pdf"
    route = respx.post(PAYLOAD_URL).mock(return_value=httpx.Response(200))
    assert fix_file_names.cmd_apply(_args(tmp_path, db_path, entries)) == 1
    assert _names(db_path) == {1: "old1.pdf", 2: "old2.pdf"}
    assert route.call_count == 0


@respx.mock
def test_apply_qdrant_failure_rolls_back_db_and_reverts_qdrant(tmp_path, db_path):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append((body["filter"]["must"][0]["match"]["value"], body["payload"]["file_name"]))
        # Second paper's forward update fails; reverts succeed.
        if body["payload"]["file_name"] == "new2.pdf":
            return httpx.Response(500)
        return httpx.Response(200)

    respx.post(PAYLOAD_URL).mock(side_effect=handler)
    assert fix_file_names.cmd_apply(_args(tmp_path, db_path, _entries())) == 1

    # DB untouched, Qdrant reverted for every paper that was attempted.
    assert _names(db_path) == {1: "old1.pdf", 2: "old2.pdf"}
    assert (1, "old1.pdf") in sent and (2, "old2.pdf") in sent


@respx.mock
def test_apply_closes_connection_on_failure(tmp_path, db_path, monkeypatch):
    closed = []
    real_connect = sqlite3.connect

    class Spy:
        def __init__(self, conn):
            self._c = conn

        def __getattr__(self, name):
            return getattr(self._c, name)

        def close(self):
            closed.append(True)
            self._c.close()

    monkeypatch.setattr(fix_file_names.sqlite3, "connect", lambda *a, **k: Spy(real_connect(*a, **k)))
    respx.post(PAYLOAD_URL).mock(side_effect=httpx.ConnectError("down"))
    assert fix_file_names.cmd_apply(_args(tmp_path, db_path, _entries())) == 1
    assert closed
    assert _names(db_path) == {1: "old1.pdf", 2: "old2.pdf"}
