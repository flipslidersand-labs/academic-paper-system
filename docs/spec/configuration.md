# 設定仕様

対象: `main`(2026-11-27 時点)。正本は `academic_paper/config.py` の `Settings` です。値は環境変数、またはカレントディレクトリの `.env` から読みます(pydantic-settings。環境変数が優先)。名前の大文字小文字は区別しません。

## 起動時の検証

次のどれかに当たると、サーバーは起動しません(`Settings()` の生成時または lifespan で `ValueError` / `RuntimeError`)。

| 条件 | 内容 |
| --- | --- |
| プレースホルダーの URL | `EMBEDDING_SVC_URL` / `QDRANT_URL` に `<` を含む値。既定値は `http://<internal-host>:…` なので、**この 2 つは必ず設定が要る** |
| プレースホルダーの秘密値 | `API_KEY` / `API_KEYS`(各要素) / `INGEST_API_KEY` / `EMBEDDING_API_KEY` / `QDRANT_API_KEY` / `GOOGLE_API_KEY` が、値全体で `<…>` の形 |
| 認証キーが無い | `API_KEY`・`API_KEYS`・`INGEST_API_KEY` がすべて空で、`AUTH_DISABLED=true` でもない(fail-closed。#646) |
| チャンク設定 | `CHUNK_OVERLAP` ≥ `CHUNK_SIZE` |
| LLM のタイムアウト | `LLM_GENERATE_TIMEOUT` < `OLLAMA_TIMEOUT` × 3 + 3 |
| 要約全体のタイムアウト | `SUMMARIZE_TOTAL_TIMEOUT` < `EMBEDDING_TIMEOUT` + `QDRANT_TIMEOUT` + `LLM_GENERATE_TIMEOUT` + 10 |

実測(2026-11-27): `API_KEY` を渡さずに起動すると、次のエラーで終了しました。

```text
RuntimeError: API_KEY / API_KEYS is not set. Refusing to start with unauthenticated endpoints; set API_KEY (or API_KEYS), or set AUTH_DISABLED=true to run without authentication explicitly (#646).
ERROR:    Application startup failed. Exiting.
```

## 環境変数

### 接続先

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `EMBEDDING_SVC_URL` | `http://<internal-host>:9092`(プレースホルダー。要設定) | embedding-svc の URL |
| `EMBEDDING_API_KEY` | 空 | embedding-svc の API キー |
| `EMBEDDING_TIMEOUT` | 120 | embedding-svc の HTTP タイムアウト(秒)。256 チャンクまでのバッチは 30 秒を超えうる |
| `QDRANT_URL` | `http://<internal-host>:6333`(プレースホルダー。要設定) | Qdrant の URL |
| `QDRANT_API_KEY` | 空 | Qdrant の API キー |
| `QDRANT_TIMEOUT` | 30 | Qdrant クライアントのタイムアウト(秒)。upsert は 200 点ずつ |
| `QDRANT_COLLECTION` | `academic-papers` | コレクション名 |
| `EMBEDDING_DIM` | 768 | ベクトル次元。embedding モデルと Qdrant コレクションの次元に合わせる |
| `ACADEMIC_DB` | `/data/academic.db` | SQLite のパス |

### 取り込み

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `CHUNK_SIZE` | 512 | チャンクの大きさ。単位は語数(空白区切り) |
| `CHUNK_OVERLAP` | 64 | チャンクの重なり(語数)。`CHUNK_SIZE` 未満であること |
| `MAX_UPLOAD_MB` | 50 | PDF アップロードの上限(MB) |
| `PDF_EXTRACT_TIMEOUT` | 120 | `extract_text()` の上限(秒)。超えると取り込みジョブが失敗する。ページ数の上限は無い |

### LLM(要約)

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `auto` | `auto` / `gemini` / `ollama` / `none`。`auto` は `GOOGLE_API_KEY` があれば Gemini、無ければ Ollama。`gemini` を明示してキーが空ならエラー。`none` は LLM 無効(要約 API は 503) |
| `GOOGLE_API_KEY` | 空 | Gemini の API キー |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini のモデル名 |
| `GEMINI_TIMEOUT_MS` | 60000 | Gemini の HTTP タイムアウト(ミリ秒) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama の URL |
| `OLLAMA_MODEL` | `mistral` | Ollama のモデル名 |
| `OLLAMA_TIMEOUT` | 300 | Ollama の 1 回あたりの HTTP タイムアウト(秒)。`generate()` は最大 3 回試す |
| `LLM_GENERATE_TIMEOUT` | 未設定(`OLLAMA_TIMEOUT` × 3 + 3 = 903 を採用) | 要約時の `llm.generate()` の上限(秒) |
| `SUMMARIZE_TOTAL_TIMEOUT` | 未設定(`EMBEDDING_TIMEOUT` + `QDRANT_TIMEOUT` + `LLM_GENERATE_TIMEOUT` + 10 = 1063 を採用) | `summarize()` 全体の上限(秒)。embedding・Qdrant・LLM の待ちは直列に積み上がるため |

`OLLAMA_TIMEOUT` を上げるときは、`LLM_GENERATE_TIMEOUT` と `SUMMARIZE_TOTAL_TIMEOUT` を未設定のままにする(自動で追従する)か、下限以上に合わせて上げます。

### 認証

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `API_KEY` | 空 | 全スコープ(`ingest` / `read` / `admin`)の `X-API-Key` |
| `API_KEYS` | 空 | 追加の全スコープキー(カンマ区切り。鍵のローテーション用。空要素は無視) |
| `INGEST_API_KEY` | 空 | `ingest` スコープだけのキー。単独で設定しても認証は有効になる |
| `AUTH_DISABLED` | `false` | `true` で、認証キーが無い状態での起動を許可する。キーを 1 つでも設定していれば認証は有効のまま(`auth_enabled`、`config.py`) |

詳細は [API 仕様](api.md) の「認証とスコープ」を参照してください。

### 運用・可観測性

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `PORT` | 8020 | `Settings` に定義があるが、`academic_paper/` のコードからは参照されない(`grep` で確認)。待ち受けポートは `uvicorn` の `--port` で決まる(Dockerfile は 8020 固定) |
| `ENABLE_DOCS` | `false` | `true` で `/docs`・`/redoc`・`/openapi.json` を公開する。これらは認証対象外 |
| `LOG_LEVEL` | `INFO` | ルートのログレベル |
| `LOG_FORMAT` | `json` | `json` または `text` |
| `OTEL_ENDPOINT` | 空 | OpenTelemetry コレクタのエンドポイント。空なら無効 |
| `PREFERRED_CATEGORIES` | `cs.AI,cs.LG,cs.CL` | スコアリングで優先する arXiv カテゴリ(カンマ区切り) |

### サーバー設定ではないもの

次はコレクタースクリプトが使う値で、`Settings` には含まれません。

| 変数 | 説明 |
| --- | --- |
| `PAPER_API_KEY` | `scripts/` のコレクターが `X-API-Key` として送る値。サーバーの `API_KEY` と同じ値にする |
| `SEMANTIC_SCHOLAR_API_KEY` | `scripts/semantic_scholar_collect.py` 用(`--api-key` でも指定できる) |
| `NCBI_API_KEY` | `scripts/pubmed_collect.py` 用(任意。`--api-key` でも指定できるが、引数は `ps` に見えるため環境変数を推奨する) |

## Docker Compose の追加変数

`docker-compose.yml` は `.env` を `paper-rag` に渡します。加えて、Qdrant コンテナには `QDRANT_API_KEY` が**必須**です(`${QDRANT_API_KEY:?…}`。空だと `docker compose` が起動前に失敗する)。同じ値を `paper-rag` にも渡す前提で、1 つの値を Qdrant サーバーと `paper-rag` の両方に使います。
