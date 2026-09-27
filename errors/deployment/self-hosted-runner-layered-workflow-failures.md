---
title: "self-hosted runner の cron workflow が層状に壊れ、1つ直すと次の層が露見する"
tags: [github-actions, self-hosted-runner, ci, layered-failure]
severity: high
date: "2026-09-22"
---

## 症状

`arxiv-daily.yml`（日次）と `portfolio-publish.yml`（週次）が minipc の永続
self-hosted runner 上で長期間（それぞれ3週間・6週間）連続失敗していたが、
`gh run list` のログだけを見て1つ直すたびに「直った」と誤認しかけた。
実際には layer が3つ重なっており、1層直すと即座に次の層のエラーが露見した:

1. `ModuleNotFoundError: No module named 'defusedxml'` — 依存インストール漏れ
2. 1を直すと `pydantic_core.ValidationError` — `Settings()` がインポート時に
   `EMBEDDING_SVC_URL`/`QDRANT_URL` のプレースホルダ既定値を拒否
3. `portfolio-publish.yml` は別workflowだが同根: `python: command not found`
   （setup-python が無い）

## 原因

- 永続 self-hosted runner はチェックアウトのたびにクリーンな venv にならず、
  「グローバル site-packages に何が入っているか」がその時点の状態に依存する。
  `pip install httpx` のような不完全なインストールコマンドは、たまたま動いて
  いた過去の残留パッケージに頼って何週間も気づかれないことがある。
- `academic_paper.config.py` は import 時に `Settings()` をモジュールレベルで
  インスタンス化する。`scripts/_collect_common.py` 経由でこれを import する
  スクリプトは、Qdrant/embedding-svc に一切接続しなくても、この検証だけで
  落ちる。CI（`ci.yml`）はこの罠を docker smoke test 用のダミー環境変数で
  回避済みだったが、cron ワークフロー側は真似していなかった。
- ログの「exit code」だけを見て「直った」と判断せず、実際に
  `workflow_dispatch` で最後まで完走するかを毎回確認しないと、次の層が
  隠れていることに気づけない。

## 解決策

各修正後に必ず `gh workflow run <file>.yml --ref main -f ...` で実環境実行し、
`gh run view <id> --json conclusion,jobs` で `success` になるまで検証を続けた
（1回の修正 → 1回の実環境検証、を3回繰り返してようやく完走）。

- 依存: `pip install httpx` → `pip install -r requirements.lock -e .`
- Settings: job env に `EMBEDDING_SVC_URL=http://localhost:9092` /
  `QDRANT_URL=http://localhost:6333`（ci.yml と同じダミー値）を追加
- python: `actions/setup-python@v5` を追加

## 予防

- 永続 self-hosted runner を使う cron workflow は、他の workflow
  （特に CI）が既に踏んだ同種の罠（プレースホルダ拒否・python 未解決・
  依存不足）を横展開で確認する。1つの workflow で直した罠は、同じランナー・
  同じライブラリを使う他の workflow にも大抵潜んでいる。
- 「ログのエラーが消えた」＝「直った」ではない。`workflow_dispatch` で
  実際に最後のステップ（本件では ingest 成功や commit push）まで到達する
  ことを確認するまで、次の層が隠れている前提で疑う。
- scripts/*.py が import する サードパーティパッケージが pyproject.toml の
  dependencies に宣言されているかを AST で検証する回帰テストを追加すると
  （このセッションで `tests/test_scripts_deps.py` として実施）、layer 1
  の再発は機械的に防げる。layer 2（Settings 検証）は import 副作用自体が
  設計上の弱点なので、根本対応は遅延初期化か DI 化。
