# API 仕様

対象: `academic-paper-system` の `main`(1174fe1 時点)。正本はコード(`academic_paper/server.py`、`academic_paper/routers/`)です。
実行時の OpenAPI は `ENABLE_DOCS=true` のときだけ `/docs`・`/redoc`・`/openapi.json` で公開されます(既定は無効で、`/docs` は 404)。

この文書の確かさの区分:

- **実測**: 2026-11-27 に、Qdrant・embedding-svc に接続しない状態(`EMBEDDING_SVC_URL` と `QDRANT_URL` を閉じたポートへ向け、`LLM_PROVIDER=none`)で起動したサーバーへ実際に送った結果です。
- **コードから**: embedding・Qdrant・LLM が要る成功応答(取り込み完了、hybrid / vector / nugget 検索、要約生成)です。コードの返却値から書いており、この文書のために実行はしていません。

## 認証とスコープ

リクエストヘッダー `X-API-Key` で認証します(`academic_paper/auth.py`)。キーの種類とスコープは次のとおりです。

| 設定 | 付与されるスコープ |
| --- | --- |
| `API_KEY`、`API_KEYS`(カンマ区切り、ローテーション用) | `ingest` + `read` + `admin` |
| `INGEST_API_KEY` | `ingest` のみ |

- キーが無い・一致しない: **401** `{"detail": "Invalid or missing API key"}`
- キーは有効だがスコープが足りない: **403** `{"detail": "Insufficient scope"}`
- `API_KEY`・`API_KEYS`・`INGEST_API_KEY` がすべて空で `AUTH_DISABLED=true` でもない場合、サーバーは起動を拒否します(fail-closed、`RuntimeError`)。認証なしで動かすには `AUTH_DISABLED=true` を明示します。この場合、認証は全エンドポイントで素通りになります。
- `/health` と `/ui` は認証不要です。

## エンドポイント一覧

| Method | Path | スコープ | 概要 |
| --- | --- | --- | --- |
| POST | `/papers/ingest` | `ingest` | PDF を取り込む(既定は非同期 202、`wait=true` で同期 200) |
| GET | `/papers` | `read` | 論文一覧 |
| GET | `/papers/{paper_id}` | `read` | 論文詳細 |
| GET | `/papers/{paper_id}/summary` | `read` | 保存済み要約の取得(生成はしない) |
| POST | `/papers/{paper_id}/summary` | `admin` | 要約の生成 |
| POST | `/papers/score-all` | `admin` | 全論文の関連度スコアを計算・保存 |
| POST | `/papers/{paper_id}/score` | `admin` | 1 論文の関連度スコアを計算・保存 |
| GET | `/summaries` | `read` | 要約一覧(論文メタデータ付き) |
| POST | `/jobs/summarize-all` | `admin` | 要約未生成の論文を一括要約するジョブを開始 |
| GET | `/jobs` | `read` | ジョブ一覧 |
| GET | `/jobs/{job_id}` | `read` | ジョブ状態 |
| GET | `/search` | `read` | 検索 |
| GET | `/stats` | `read` | 件数の統計 |
| GET | `/metrics` | `admin` | Prometheus メトリクス |
| GET | `/health` | なし | Qdrant・embedding-svc の疎通確認 |
| (mount) | `/ui` | なし | `frontend/` の静的 UI(`frontend/` が存在するときだけ) |

全応答に `X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer` が付きます。`/ui` 配下にはさらに `Content-Security-Policy` が付きます。

## エラー応答の形式

アプリが返すエラーは `{"detail": "<メッセージ>"}` です。次の 2 つは形が違います。

- クエリ・パスの型や範囲の検証エラー(例: `mode` の不正値、`limit` の範囲外)は FastAPI 標準の **422** で、`detail` が配列になります(実測)。

  ```json
  {"detail":[{"type":"string_pattern_mismatch","loc":["query","mode"],"msg":"String should match pattern '^(vector|keyword|hybrid|nugget)$'","input":"bogus","ctx":{"pattern":"^(vector|keyword|hybrid|nugget)$"}}]}
  ```

- 分類できなかった例外は **500** で `{"detail": "Internal error [xxxxxxxx]"}` になります。`xxxxxxxx` はログと突き合わせる ID で、例外の本文は返しません。

例外から HTTP ステータスへの対応(`academic_paper/errors.py` の `classify_error`):

| 例外 | ステータス |
| --- | --- |
| `NotFoundError` | 404 |
| `NotConfiguredError`(LLM 未設定など) | 503 |
| 上流(embedding-svc・Ollama・Gemini)が 401 / 403 | 502(キー設定の誤り。キーの値は返さない) |
| 上流が 5xx / 429、接続失敗、ネットワーク・タイムアウト | 503 |
| 上流がそれ以外の 4xx、不正な応答(ベクトル数の不一致、LLM 出力不正、Qdrant の想定外応答) | 502 |
| `TimeoutError`(`asyncio.wait_for` の期限切れ) | 504 |
| SQLite の一意制約違反(`file_hash`) | 409 |
| 入力検証の `ValueError`、PDF 抽出失敗 | 400 |
| 要約に使えるチャンクが無い | 500(固定文言) |
| 上記以外 | 500 |

## `POST /papers/ingest`

PDF をアップロードしてインデックス化します。

リクエスト: `multipart/form-data`

| フィールド | 必須 | 制約 |
| --- | --- | --- |
| `file` | 必須 | PDF。先頭が `%PDF-` であること。サイズ上限は `MAX_UPLOAD_MB`(既定 50) |
| `title` | 任意 | 最大 1000 文字 |
| `authors` | 任意 | 最大 10000 文字。JSON 文字列配列、またはカンマ区切り |
| `categories` | 任意 | 最大 2000 文字。`authors` と同じ形式 |
| `published_date` | 任意 | `YYYY-MM-DD`。未来日と 1900-01-01 より前は不可 |
| `source` | 任意 | 最大 100 文字 |

クエリ `wait`(bool、既定 false)が true のとき、取り込み完了まで待って 200 を返します。

メタデータの制御文字は空白に置換されます。ファイルの重複はファイルハッシュで判定します。

**202(既定)** — 受理しました。バックグラウンドで 抽出 → チャンク化 → embedding → Qdrant upsert を行います(実測)。

```json
{"job_id":"b6c329fc-ae16-4376-a34e-d97637dbbf10","paper_id":1,"status":"pending"}
```

完了は `GET /jobs/{job_id}` で確認します。`done` のとき、ジョブの `result` に次が入ります(コードから)。

```json
{"paper_id": 1, "file_name": "paper.pdf", "chunks": 42, "status": "indexed"}
```

**200(`wait=true`)** — 同じ内容を直接返します(コードから)。

```json
{"paper_id": 1, "file_name": "paper.pdf", "chunks": 42, "status": "indexed"}
```

エラー:

| ステータス | 条件 | `detail`(実測) |
| --- | --- | --- |
| 409 | 同じファイルが取り込み済み / 取り込み中 | `File already ingested` / `File is already being ingested`(コードから) |
| 413 | サイズ上限超過 | `File too large (max N MB)`(コードから) |
| 415 | PDF ではない | `Not a PDF file (missing %PDF- header)` |
| 422 | `published_date` が不正 | `published_date must be an ISO date (YYYY-MM-DD), got '20260101'` |
| 422 | `published_date` が未来日 | `published_date must not be in the future, got '2099-01-01'` |
| 422 | `authors` / `categories` が文字列配列でない | `authors must be a JSON array of strings or a comma-separated string` |
| 400 ほか | 抽出・チャンク化・embedding・Qdrant の失敗(`wait=true` のみ。非同期ではジョブが `failed` になる) | 上の対応表のとおり |

論文の `status` は `pending` → `indexed` または `failed` と遷移します。`failed` の論文は同じファイルを再アップロードできます(実測では、再アップロードに新しい `paper_id` が付きました)。サーバー起動時に `pending` のまま残った論文は `failed` に直し、対応する Qdrant のベクトルを削除します。

## `GET /papers`

クエリ: `limit`(1–100、既定 20)、`offset`(既定 0)、`author`(部分一致)、`category`(完全一致。例 `cs.AI`)、`sort`(`ingested_at`〔新しい順、既定〕または `score`〔高い順。未スコアは末尾〕)。

`total` はフィルタ適用後の件数です。実測の応答(取り込みに失敗した論文 1 件):

```json
{
  "total": 1,
  "papers": [
    {
      "id": 1, "file_name": "min.pdf", "file_hash": "14bcd090…", "title": null, "authors": null,
      "year": null, "pages": null, "ingested_at": "2026-11-27T14:54:14.916738+00:00", "status": "failed",
      "categories": null, "published_date": null, "source": null, "score": null, "arxiv_id": null
    }
  ]
}
```

`authors` と `categories` は文字列配列(未設定は `null`)です。`GET /papers/{paper_id}` は同じ形の論文 1 件を返し、存在しなければ 404 `Paper not found` です。

## 要約

`GET /papers/{paper_id}/summary` は保存済みの要約だけを返し、LLM 生成も DB 書き込みもしません。

- 論文が無い: 404 `Paper not found`(実測)
- 要約が未生成: 404(`Summary not generated yet — POST /papers/{paper_id}/summary to generate`)

`POST /papers/{paper_id}/summary` は要約を生成して保存します。キャッシュがあればそれを返し、`force=true` で再生成します。LLM が未設定なら 503 `LLM not configured`、論文が無ければ 404 です。

応答(コードから):

```json
{
  "paper_id": 1,
  "model": "gemini-2.0-flash",
  "objective": "…", "method": "…", "results": "…", "limitations": "…",
  "keywords": ["…"],
  "cached": false
}
```

`model` は使った LLM の表示名です。`cached` はキャッシュから返したときだけ true です。

`GET /summaries` は `limit`(1–100、既定 20)・`offset` で要約を新しい順に返します。応答は `{"total": N, "summaries": [...]}` で(実測は空配列)、各要素は `paper_id`・`model`・`objective`・`method`・`results`・`limitations`・`keywords`・`created_at` と、論文の `file_name`・`title`・`authors`・`categories`・`published_date`・`source` を持ちます。

## スコア

`POST /papers/score-all` は全論文を採点し、論文単位の失敗は全体を止めず `failed`・`errors` に集計します。実測の応答(論文 0 件):

```json
{"total":0,"scored":0,"failed":0,"errors":[],"preferred_categories":["cs.AI","cs.LG","cs.CL"]}
```

`POST /papers/{paper_id}/score` は `{"paper_id": 1, "score": 0.73}` の形で返します(例の値はコードの返却形を示すためのもの)。論文が無ければ 404 です。
スコアの式は `README.md` の「Scoring & Portfolio」を参照してください。

## ジョブ

`POST /jobs/summarize-all` は、`indexed` で要約が無い論文をすべて要約するジョブを始めます。202 で `{"job_id": "…", "status": "pending"}` を返します。LLM が未設定なら 503 `LLM not configured`(実測)、同種のジョブが実行中なら 409 `A summarize-all job is already running` です。

`GET /jobs/{job_id}` はジョブ 1 件を返します。存在しなければ 404 `Job not found`(実測)。メモリから退避された完了ジョブも SQLite から読みます。

`GET /jobs` は `limit`(1–100、既定 20)・`offset` でジョブを開始が新しい順に返し、`{"total": N, "jobs": [...]}` の形です(実測)。メモリ上のジョブだけが対象で、退避された完了ジョブは含みません。

ジョブオブジェクト(実測。取り込みが `Failed to extract text from PDF` で失敗した例):

```json
{
  "id": "b6c329fc-ae16-4376-a34e-d97637dbbf10",
  "status": "failed",
  "kind": "ingest",
  "total": 1, "processed": 0, "failed": 1,
  "errors": ["paper_id=1: Failed to extract text from PDF"],
  "started_at": 1795791254.917029,
  "finished_at": 1795791254.923997,
  "result": null
}
```

- ID のキー名は、作成時の応答では `job_id`、ジョブオブジェクトでは `id` です。
- `status` は `pending` / `running` / `done` / `failed` です。`kind` は `ingest` または `summarize-all` です。
- `errors` は先頭 10 件だけ返します。`started_at` と `finished_at` は UNIX 時刻(秒)です。
- サーバー再起動時に `pending` / `running` だったジョブは `failed` になり、`errors` に `Server restarted while job was in flight` が入ります。
- ジョブ状態の正本はプロセス内のメモリです(単一プロセス前提。[ADR](../adr/20261127-single-process-job-store.md))。完了ジョブはメモリに最大 1000 件まで保持し、超えた分は SQLite にだけ残ります。

## `GET /search`

| パラメータ | 既定 | 説明 |
| --- | --- | --- |
| `q` | 必須 | 検索語。1–1000 文字 |
| `mode` | `hybrid` | `hybrid` / `vector` / `keyword` / `nugget`(下記) |
| `limit` | 10 | 1–100 |
| `paper_id` | なし | 論文 ID で絞る |
| `snippet_length` | 200 | スニペットの最大文字数。0 で全文。`nugget` では無効 |
| `nuggets_per_chunk` | 3 | チャンクあたりの文数(1–10)。`nugget` のみ |
| `nugget_embed_weight` | 0.7 | 文の採点で embedding に置く重み(0.0–1.0)。0 で BM25 のみ、1 で embedding のみ。`nugget` のみ |

モード:

- `keyword`: SQLite FTS5 の `bm25()` だけ。クエリは各語を引用符で囲んで文字列リテラルとして扱うため、`AND` や `*` などの演算子として解釈されません。embedding・Qdrant は使いません。
- `vector`: クエリを embedding し、Qdrant の類似度で検索します。
- `hybrid`: `keyword` と `vector` の結果を RRF(`1 / (60 + 順位)`)で統合し、上位 `limit` 件を返します。
- `nugget`: `hybrid` で取ったチャンクから、クエリに近い文を `nuggets_per_chunk` 件ずつ抜き出して `snippet` にします。

応答。`keyword` は実測、他のモードは同じ形です(コードから)。

```json
{
  "mode": "keyword",
  "query": "attention",
  "results": [
    {"rank": 1, "score": -1.23, "paper_id": 1, "chunk_index": 5, "page_start": 2, "snippet": "…"}
  ]
}
```

`results` が空の実測は `{"mode":"keyword","query":"attention","results":[]}` です(例の `score` はコードの返却形を示すための値)。`score` の意味はモードで違います。`keyword` は FTS5 の `bm25()` の値(小さいほど上位)、`vector` は Qdrant の類似度、`hybrid` と `nugget` は RRF スコアです。`page_start` は不明なら `null` です。

検索中の失敗は、上の対応表で 400 / 502 / 503 / 504 などになります。

## `GET /health`

認証不要です。Qdrant と embedding-svc に疎通し、両方正常なら 200、どちらかが失敗なら **503** です。実測(両方に接続できない状態):

```json
{"status":"degraded","qdrant":"error","embedding_svc":"error","embedding_svc_reason":"unreachable"}
```

`embedding_svc_reason` は `ok` / `unauthorized`(401・403) / `unreachable`(接続失敗) / `error` のいずれかです。

## `GET /stats`

`{"papers": N, "chunks": N, "qdrant_points": N}`(実測は `{"papers":0,"chunks":0,"qdrant_points":-1}`)。Qdrant の件数を取れないときは `qdrant_points` が `-1` になります。

## `GET /metrics`

`prometheus-fastapi-instrumentator` が公開する Prometheus 形式のメトリクスです。`admin` スコープが必要で、フルキー(`API_KEY` / `API_KEYS`)は 200、`INGEST_API_KEY` は 403 でした(実測)。
