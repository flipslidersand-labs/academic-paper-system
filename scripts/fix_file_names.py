#!/usr/bin/env python3
"""Fix mismatched file_name values for legacy papers (#161 / #162).

Papers ingested before arxiv_collect.py existed have bare `NNNN.NNNNN.pdf`
file names, and two of them (papers 2 and 3) point at the wrong arXiv ID
entirely — the DB was reset and re-seeded while Qdrant kept the original
chunks, so the chunk text is the source of truth.

This script recovers the true arXiv ID from each paper's first Qdrant chunk:
arXiv PDFs carry an "arXiv:NNNN.NNNNNvX" watermark that pdfplumber extracts
either as-is or mirrored (e.g. "1v17001.0142:viXra"), so both directions are
scanned.

Usage:
    # 1. Build a mapping proposal from Qdrant chunk text
    python scripts/fix_file_names.py resolve --out mapping.json

    # 2. Preview the changes
    python scripts/fix_file_names.py apply --mapping mapping.json --dry-run

    # 3. Apply (backs up the DB first, updates DB + Qdrant payloads)
    python scripts/fix_file_names.py apply --mapping mapping.json

Environment:
    QDRANT_URL         (default http://localhost:6333)
    QDRANT_COLLECTION  (default academic-papers)
    ACADEMIC_DB        (default ./data/academic.db)
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

import httpx

ARXIV_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?")
DEFAULT_VERSION = "v1"  # legacy collects fetched the initial submission


def _qdrant_head_chunks(
    client: httpx.Client, qdrant_url: str, collection: str, paper_id: int, count: int
) -> list[dict]:
    """Return payloads of the paper's first `count` chunks, ordered by chunk_index.

    Only the head of the paper is scanned: the arXiv watermark sits on page 1,
    while later chunks contain bibliography entries with *other* papers' IDs.
    """
    payloads = []
    for chunk_index in range(count):
        resp = client.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json={
                "filter": {
                    "must": [
                        {"key": "paper_id", "match": {"value": paper_id}},
                        {"key": "chunk_index", "match": {"value": chunk_index}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
            },
        )
        resp.raise_for_status()
        points = resp.json()["result"]["points"]
        if points:
            payloads.append(points[0]["payload"])
    return payloads


def _detect_arxiv_id(text: str) -> tuple[str, str] | None:
    """Extract (arxiv_id, version) from chunk text, scanning forward and mirrored."""
    m = ARXIV_ID_RE.search(text)
    if m:
        return m.group(1), m.group(2) or DEFAULT_VERSION
    # Mirrored extraction: reversing the text restores "arXiv:NNNN.NNNNNvX" reading order.
    m = ARXIV_ID_RE.search(text[::-1])
    if m:
        return m.group(1), m.group(2) or DEFAULT_VERSION
    return None


def cmd_resolve(args: argparse.Namespace) -> int:
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, file_name FROM papers WHERE id BETWEEN ? AND ? ORDER BY id",
        (args.from_id, args.to_id),
    ).fetchall()
    conn.close()

    entries = []
    with httpx.Client(timeout=15) as client:
        for paper_id, db_file_name in rows:
            payloads = _qdrant_head_chunks(client, args.qdrant_url, args.collection, paper_id, args.scan_chunks)
            payload = payloads[0] if payloads else {}
            text = payload.get("text", "")
            detected = next((d for p in payloads if (d := _detect_arxiv_id(p.get("text", "")))), None)
            entry = {
                "paper_id": paper_id,
                "db_file_name": db_file_name,
                "qdrant_file_name": payload.get("file_name"),
                "chunk_head": text[:120],
            }
            if detected is None:
                entry["status"] = "unresolved"
                entry["new_file_name"] = None
                print(f"  ?? paper {paper_id}: no arXiv ID found in chunk text — resolve manually", file=sys.stderr)
            else:
                arxiv_id, version = detected
                entry["arxiv_id"] = arxiv_id
                entry["new_file_name"] = f"arxiv_{arxiv_id}{version}.pdf"
                db_id = db_file_name.removeprefix("arxiv_").removesuffix(".pdf")
                entry["status"] = "rename" if db_id.split("v")[0] == arxiv_id else "wrong_id"
            entries.append(entry)

    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "qdrant chunk_index=0 text (arXiv watermark, forward+mirrored scan)",
        "entries": entries,
    }
    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    for e in entries:
        print(f"  paper {e['paper_id']}: {e['db_file_name']} -> {e['new_file_name']} [{e['status']}]")
    print(f"[resolve] mapping written to {args.out}")
    return 1 if any(e["status"] == "unresolved" for e in entries) else 0


def cmd_apply(args: argparse.Namespace) -> int:
    with open(args.mapping) as f:
        mapping = json.load(f)
    entries = [e for e in mapping["entries"] if e.get("new_file_name")]
    unresolved = [e for e in mapping["entries"] if not e.get("new_file_name")]
    if unresolved:
        print(f"[apply] ERROR: {len(unresolved)} unresolved entries — fix the mapping first", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    changes = []
    for e in entries:
        row = conn.execute("SELECT file_name FROM papers WHERE id = ?", (e["paper_id"],)).fetchone()
        if row is None:
            print(f"[apply] ERROR: paper {e['paper_id']} not found in DB", file=sys.stderr)
            conn.close()
            return 1
        if row[0] != e["db_file_name"]:
            print(
                f"[apply] ERROR: paper {e['paper_id']} file_name is {row[0]!r}, "
                f"mapping expects {e['db_file_name']!r} — stale mapping, re-run resolve",
                file=sys.stderr,
            )
            conn.close()
            return 1
        if row[0] != e["new_file_name"]:
            changes.append(e)

    if not changes:
        print("[apply] nothing to do — all file names already correct")
        conn.close()
        return 0

    for e in changes:
        print(f"  paper {e['paper_id']}: {e['db_file_name']} -> {e['new_file_name']} [{e['status']}]")

    if args.dry_run:
        print(f"[apply] dry-run: {len(changes)} papers would be updated (DB + Qdrant payloads)")
        conn.close()
        return 0

    backup = f"{args.db}.bak-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(args.db, backup)
    print(f"[apply] DB backed up to {backup}")

    with httpx.Client(timeout=30) as client:
        for e in changes:
            conn.execute("UPDATE papers SET file_name = ? WHERE id = ?", (e["new_file_name"], e["paper_id"]))
            resp = client.post(
                f"{args.qdrant_url}/collections/{args.collection}/points/payload?wait=true",
                json={
                    "payload": {"file_name": e["new_file_name"]},
                    "filter": {"must": [{"key": "paper_id", "match": {"value": e["paper_id"]}}]},
                },
            )
            resp.raise_for_status()
    conn.commit()
    conn.close()
    print(f"[apply] updated {len(changes)} papers in DB and Qdrant")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix legacy paper file names from Qdrant chunk text")
    parser.add_argument("--db", default=os.environ.get("ACADEMIC_DB", "./data/academic.db"))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "academic-papers"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="build mapping proposal from Qdrant chunk text")
    p_resolve.add_argument("--from-id", type=int, default=1)
    p_resolve.add_argument("--to-id", type=int, default=10)
    p_resolve.add_argument("--out", default="file_name_mapping.json")
    p_resolve.add_argument(
        "--scan-chunks", type=int, default=3, help="How many head chunks to scan for the watermark (default: 3)"
    )
    p_resolve.set_defaults(func=cmd_resolve)

    p_apply = sub.add_parser("apply", help="apply a mapping to DB + Qdrant (backs up DB first)")
    p_apply.add_argument("--mapping", required=True)
    p_apply.add_argument("--dry-run", action="store_true", help="print planned changes without writing")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
