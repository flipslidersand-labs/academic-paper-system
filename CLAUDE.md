# academic-paper-system

論文ナレッジベース: PDFテキスト抽出→ベクトル検索→構造化要約の RAG システム

## アーキテクチャ

```
PDF → pdfplumber抽出 → テキストチャンク分割(512/64)
    → embedding-svc:9092 (multilingual-e5-base 768-d) → Qdrant:6333
    → SQLite FTS5 + BM25
    → RRF ハイブリッド検索
    → Gemini / Ollama で構造化要約
```

**スタック**:
- **Backend**: FastAPI + uvicorn
- **PDF処理**: pdfplumber
- **ベクトル化**: embedding-svc (multilingual-e5-base 768次元。Qdrant コレクションは768次元固定のため、モデルを変える場合はコレクションの作り直しが必要)
- **ベクトルDB**: Qdrant
- **検索**: SQLite FTS5 + BM25 + RRF
- **要約LLM**: Google Generative AI (Gemini) / Ollama (Mistral)
- **テレメトリ**: OpenTelemetry SDK + FastAPI instrumentation

**運用前提（単一プロセス）**: 現状は単一プロセス・単一ワーカー前提。`JobStore` はプロセスローカルの dict をジョブ読み取り（`get()` / `list_all()` / `has_running()`）の正本にしており、SQLite は永続化のみ。`uvicorn --workers>1` やレプリカ増設ではジョブ作成と照会が別プロセスになり `get()` が None（404）になるため未対応（#387）。Dockerfile の CMD は `--workers` 未指定（=1）。

## セットアップ

### 環境変数設定

```bash
cp .env.example .env
# .env を編集して各サービスの URL と API キーを設定
```

### Docker での実行

```bash
cp .env.example .env   # embedding-svc / Qdrant / Gemini API キーを設定
docker compose up -d
curl http://localhost:8020/health
```

`qdrant_storage`/`qdrant_snapshots` は Compose が初回起動時に自動作成する
named volume。既存の external volume を使い回したい場合は
`docker-compose.override.yml` (gitignore 済み) で `name`/`external: true` を
上書きする。

### ローカル開発環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn academic_paper.server:app --reload --port 8020
```

## API 一覧

詳細なスキーマ (リクエスト/レスポンスの全フィールド) は起動中サーバーの `/docs` (OpenAPI) を参照。正本は `academic_paper/server.py`。
以下の表は全エンドポイントの一覧、続く節は注意点のみを記す。

認証: `API_KEY` 環境変数が設定されている場合、「要」のエンドポイントは `X-API-Key` ヘッダーが必須 (不一致は 401)。空なら認証無効。

| Method | Path | 認証 | 概要 |
|--------|------|------|------|
| POST | `/papers/ingest` | 要 | PDF 論文をアップロード・インデックス化 (既定は非同期 202、`wait=true` で同期 200) |
| GET | `/papers` | 要 | 論文一覧 (limit/offset・author/category フィルタ・sort) |
| GET | `/papers/{paper_id}` | 要 | 論文詳細 |
| GET | `/papers/{paper_id}/summary` | 要 | 保存済み要約の取得 (キャッシュのみ。未生成は 404) |
| POST | `/papers/{paper_id}/summary` | 要 | 要約の生成 (キャッシュがあればそれを返す。`force=true` で再生成。LLM 未設定は 503) |
| POST | `/papers/score-all` | 要 | 全論文の関連度スコアを計算・保存 (論文単位の失敗は `failed`/`errors` に集計) |
| POST | `/papers/{paper_id}/score` | 要 | 1 論文の関連度スコアを計算・保存 (`{paper_id, score}`) |
| GET | `/summaries` | 要 | 要約一覧 (論文メタデータ付き、limit 1-100 / offset) |
| POST | `/jobs/summarize-all` | 要 | 要約未生成の indexed 論文を一括要約するバックグラウンドジョブ開始 (202 `{job_id, status}`) |
| GET | `/jobs/{job_id}` | 要 | ジョブ状態取得 |
| GET | `/jobs` | 要 | ジョブ一覧 |
| GET | `/search` | 要 | 検索 (`mode=hybrid/vector/keyword/nugget`) |
| GET | `/health` | 不要 | Qdrant / embedding-svc 疎通確認 (両方正常で 200、障害時 503) |
| GET | `/stats` | 要 | DB 統計 (`papers`/`chunks`/`qdrant_points`) |
| (mount) | `/ui` | 不要 | `frontend/` の静的 UI (`frontend/` ディレクトリが存在する場合のみ、全 API ルートの後に mount) |

### 非同期処理・エラーコードの注意点

- 非同期: `POST /papers/ingest` (既定) と `POST /jobs/summarize-all` は 202 + `job_id` を返す。`GET /jobs/{job_id}` で `status` (pending/running/done/failed)・`processed`/`failed`/`errors`・`result` を確認する。
- `POST /jobs/summarize-all`: 実行中の同種ジョブがあれば 409、LLM 未設定なら 503。
- `POST /papers/{paper_id}/summary`: 論文が無ければ 404、LLM / summarizer 未設定は 503、生成失敗は上流障害 502/503 または 500。
- `/health` のみ認証不要。`/ui` も認証対象外 (静的ファイル)。

### `/papers/ingest` (POST)

PDF ファイルをアップロードしてインデックス化します。既定ではファイルハッシュで重複チェックし、`pending` の論文行を保存したうえで
バックグラウンド処理 (抽出 → チャンク化 → Embedding → Qdrant upsert) に回し、**202** を返します。
完了は `GET /jobs/{job_id}` をポーリングして確認します (`done` 時、job の `result` に `paper_id`/`chunks`)。

**Request**:
```
Content-Type: multipart/form-data
file: <PDF file>
title / authors / categories / published_date / source: 任意のメタデータ (form)
```

- `title` (最大 1000 文字)、`authors` (最大 10000)、`categories` (最大 2000)、`published_date` (ISO 日付、最大 10)、`source` (最大 100)
- Query `wait` (bool, デフォルト false): `true` で同期処理し、インデックス完了後に 200 を返す

**Response** (202, 既定):
```json
{
  "job_id": "...",
  "paper_id": 1,
  "status": "pending"
}
```

**Response** (200, `wait=true` のときのみ):
```json
{
  "paper_id": 1,
  "file_name": "paper.pdf",
  "chunks": 42,
  "status": "indexed"
}
```

**Errors**:
- 409: ファイルが既にインジェスト済み (ファイルハッシュで重複チェック)
- 413: ファイルサイズが上限 (`max_upload_mb`) 超過
- 415: PDF ではない (`%PDF-` マジックバイト無し)
- 422: メタデータ不正 (`published_date` が ISO 形式でない等)
- 400: PDF抽出失敗 / チャンク生成失敗 / Embedding/Qdrant エラー (`wait=true` のみ。非同期時は job が failed になる)

### `/papers` (GET)

論文一覧を取得します。

**Query params**:
- `limit`: 返す件数 (1-100, デフォルト 20)
- `offset`: スキップ件数 (デフォルト 0)
- `author`: 著者名で絞り込み (部分一致, オプション)
- `category`: カテゴリコードで絞り込み (完全一致, 例 `cs.AI`, オプション)
- `sort`: `ingested_at` (デフォルト, 新しい順) / `score` (スコア降順, 未スコアは末尾)

`total` はフィルタ適用後の件数です。

**Response** (200):
```json
{
  "total": 42,
  "papers": [
    {
      "id": 1,
      "file_name": "paper.pdf",
      "file_hash": "abc123...",
      "status": "indexed",
      "ingested_at": "2026-07-22T10:00:00Z"
    }
  ]
}
```

### `/papers/{paper_id}` (GET)

論文詳細を取得します。

**Response** (200):
```json
{
  "id": 1,
  "file_name": "paper.pdf",
  "file_hash": "abc123...",
  "status": "indexed",
  "ingested_at": "2026-07-22T10:00:00Z"
}
```

**Errors**:
- 404: 論文が見つからない

### `/papers/{paper_id}/summary` (GET)

保存済みの構造化要約をキャッシュから返します。GET は安全 (冪等) で、LLM 生成や DB 書き込みは行いません (#140)。
要約の生成は `POST /papers/{paper_id}/summary` で行います (`force` クエリ引数はそちら。デフォルト false、`true` でキャッシュを無視して再生成)。

**Response** (200):
```json
{
  "paper_id": 1,
  "model": "gemini-2.0-flash",
  "objective": "研究目的...",
  "method": "研究方法...",
  "results": "結果...",
  "limitations": "制限事項...",
  "keywords": ["keyword1", "keyword2"],
  "cached": true
}
```

**Errors**:
- 404: 論文が見つからない / 要約が未生成 (`POST /papers/{paper_id}/summary` で生成)

### `/search` (GET)

論文を検索します (ハイブリッド / ベクトル / キーワード / nugget)。

**Query params**:
- `q`: 検索クエリ (必須, 1-1000 文字)
- `mode`: 検索モード (デフォルト `hybrid`)
  - `hybrid`: FTS5 (BM25) + ベクトル検索を RRF で統合
  - `keyword`: FTS5 (BM25) のみ
  - `vector`: ベクトル検索のみ
  - `nugget`: hybrid 検索後、各チャンクからクエリに最も関連する上位 N 文 (nugget) を抽出して返す (コンテキスト長削減用)
- `limit`: 返す件数 (1-100, デフォルト 10)
- `paper_id`: 特定の論文 ID に限定 (オプション)
- `snippet_length`: スニペット最大長 (0 以上, デフォルト 200, 0=全文。keyword/vector/hybrid のみ有効)
- `nuggets_per_chunk`: チャンクあたりの文数 (1-10, デフォルト 3。nugget モードのみ)
- `nugget_embed_weight`: nugget スコアリングの Embedding 重み (0.0-1.0, デフォルト 0.7, 0=BM25 のみ / 1=Embedding のみ。nugget モードのみ)

**Response** (200):
```json
{
  "mode": "hybrid",
  "query": "deep learning",
  "results": [
    {
      "rank": 1,
      "score": 0.95,
      "paper_id": 1,
      "chunk_index": 5,
      "page_start": 2,
      "snippet": "Deep learning is a subset of machine learning..."
    }
  ]
}
```

## 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `EMBEDDING_SVC_URL` | embedding-svc URL | `http://<internal-host>:9092` |
| `EMBEDDING_API_KEY` | embedding-svc APIキー | (空) |
| `EMBEDDING_TIMEOUT` | embedding-svc HTTP タイムアウト秒（大バッチは 30s 超） | `120` |
| `QDRANT_URL` | Qdrant URL | `http://<internal-host>:6333` |
| `QDRANT_API_KEY` | Qdrant APIキー | (空) |
| `QDRANT_TIMEOUT` | Qdrant クライアントタイムアウト秒 | `30` |
| `QDRANT_COLLECTION` | Qdrant コレクション名 | `academic-papers` |
| `ACADEMIC_DB` | SQLite DB パス | `/data/academic.db` |
| `CHUNK_SIZE` | テキストチャンクサイズ | `512` |
| `CHUNK_OVERLAP` | チャンク間のオーバーラップ | `64` |
| `GOOGLE_API_KEY` | Gemini APIキー (要約用) | (空) |
| `GEMINI_TIMEOUT_MS` | Gemini API HTTP タイムアウト (ミリ秒) | `60000` |
| `GEMINI_MODEL` | Gemini モデル名 | `gemini-2.0-flash` |
| `LLM_PROVIDER` | LLM プロバイダ `auto`/`gemini`/`ollama`/`none`。auto は Google キーがあれば Gemini、なければ Ollama。gemini 明示でキー空はエラー。none は LLM 無効(要約 API は 503) | `auto` |
| `OLLAMA_URL` | Ollama URL (フォールバック) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama モデル | `mistral` |
| `OLLAMA_TIMEOUT` | Ollama 1回あたりの HTTP タイムアウト秒（generate は最大3回リトライ） | `300` |
| `OTEL_ENDPOINT` | OpenTelemetry コレクタエンドポイント | (空) |
| `LOG_LEVEL` | ルートログレベル (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `LOG_FORMAT` | ログ形式 (`json` / `text`) | `json` |
| `PREFERRED_CATEGORIES` | スコアリングで優先する arXiv カテゴリ（カンマ区切り） | `cs.AI,cs.LG,cs.CL` |
| `MAX_UPLOAD_MB` | PDF アップロード上限 (MB) | `50` |
| `PDF_EXTRACT_TIMEOUT` | `extract_text()` の上限秒（超過で ingest ジョブ失敗） | `120` |
| `LLM_GENERATE_TIMEOUT` | 要約時の `llm.generate()` 上限秒 | `903` |
| `SUMMARIZE_TOTAL_TIMEOUT` | `summarize()` 全体の上限秒 | `1063` |
| `PORT` | API サーバーポート | `8020` |
| `API_KEY` | `/health` 以外の全エンドポイントの X-API-Key（読み取り系含む。空=認証無効） | (空) |
| `INGEST_API_KEY` | ingest スコープ限定の X-API-Key（#355。単独設定でも認証有効。エンドポイントへのスコープ適用は #627） | (空) |
| `PAPER_API_KEY` | コレクタ側が送る X-API-Key（cron は repo secret 経由） | (空) |
| `SEMANTIC_SCHOLAR_API_KEY` | `scripts/semantic_scholar_collect.py` 用 API キー（`--api-key` でも指定可。サーバー設定ではない） | (空) |
| `DISCORD_WEBHOOK_URL` | arxiv-daily.yml の失敗/結果通知先（**repo secret 必須**。未設定だと通知が無効化され、schedule 実行の失敗は Notify ステップが exit 1 で表面化する。登録はオペレーター手動作業 #481） | (未設定) |

**タイムアウトの連動制約** (`academic_paper/config.py`):
- `OLLAMA_TIMEOUT` × 3 + 3 ≤ `LLM_GENERATE_TIMEOUT`（Ollama は最大3回リトライ + backoff 1s+2s。既定 300×3+3=903）
- `SUMMARIZE_TOTAL_TIMEOUT` ≥ `EMBEDDING_TIMEOUT` + `QDRANT_TIMEOUT` + `LLM_GENERATE_TIMEOUT`（各 wait_for は逐次実行され累積するため。既定 120+30+903=1053 + 余裕）
- `OLLAMA_TIMEOUT` を上げる場合は `LLM_GENERATE_TIMEOUT` と `SUMMARIZE_TOTAL_TIMEOUT` も合わせて上げる

## テスト

```bash
# 全テスト実行 (カバレッジ付き)
pytest tests/ -v --cov=academic_paper

# 特定のテストのみ実行
pytest tests/test_search.py -v

# HTML カバレッジレポート生成
pytest tests/ --cov=academic_paper --cov-report=html
# report は htmlcov/index.html
```

**テストスイート**:
- `test_extractor.py` — PDF テキスト抽出・ファイルハッシング
- `test_chunker.py` — テキストチャンク分割
- `test_embedder.py` — embedding-svc クライアント
- `test_vector_store.py` — Qdrant クライアント
- `test_db.py` — SQLite 初期化・CRUD
- `test_db_fts.py` — FTS5 索引・検索
- `test_search.py` — ハイブリッド検索・RRF
- `test_summarizer.py` — RAG 要約ロジック
- `test_llm.py` — LLM クライアント (Gemini / Ollama)
- `test_summary_endpoint.py` — `/papers/{id}/summary` エンドポイント
- `test_server.py` — FastAPI エンドポイント統合テスト
- `test_nugget.py` — nugget 抽出 (クエリ関連文の選別)
- `test_retry.py` — リトライユーティリティ
- `test_jobs.py` — `/jobs/summarize-all`・`/jobs`・`/jobs/{job_id}` とジョブ永続化
- `test_logging_config.py` — 構造化 JSON ロギング設定
- `test_telemetry.py` — OpenTelemetry テレメトリ設定
- `test_scorer.py` — 論文スコアリング (鮮度・カテゴリ関連度)
- `test_metrics.py` — Prometheus `/metrics` エンドポイント
- `test_startup_probe.py` — 起動時ヘルスチェック (`_probe_startup_health`)
- `test_metadata_filter.py` — author/category フィルタ・メタデータ ingest・`/summaries`
- `test_collect_common.py` — `scripts/_collect_common.py` の共通ヘルパー
- `test_collect_scripts.py` — 各コレクタースクリプトの fetch/parse ロジック
- `test_arxiv_collect.py` — `scripts/arxiv_collect.py` のウォーターマーク検証
- `test_ingest_client.py` — `scripts/ingest_client.py` の認証ヘッダー
- `test_config.py` — 設定値のプレースホルダー URL 拒否
- `test_hybrid.py` — ハイブリッド検索 RRF マージ
- `test_cli_utils.py` — `scripts/cli_utils.py` の argparse バリデータ
- `test_http_client.py` — `academic_paper/http_client.py` の注入/一時 AsyncClient ヘルパー
- `test_scripts_deps.py` — scripts/ が import するサードパーティ依存の宣言漏れ検出
- `test_search_service.py` — `academic_paper/services/search_service.py` (FastAPI 非起動の単体テスト)
- `conftest.py` — テスト共通 fixture（`temp_db` など）・環境変数の初期値設定

## 開発

### コード品質

```bash
# Lint (ruff)
ruff check academic_paper/ tests/

# Format (ruff format)
ruff format academic_paper/ tests/
```

**ruff設定** (`pyproject.toml`):
- Line length: 120
- Target version: Python 3.12
- Rules: E F W I N (import sort 含む)

### requirements.lock / requirements-dev.lock の再生成

Docker・CI・arxiv-daily は `requirements.lock` (ハッシュ付き) を正本として使う。
`pyproject.toml` の依存を変えたとき、または依存を最新化したいときに再生成する (Python 3.12 + `pip install pip-tools` が必要)。

```bash
scripts/update-lock.sh             # 現在のピンを維持して再生成
scripts/update-lock.sh --upgrade   # 全依存を最新版に更新して再生成
pytest tests/test_requirements_lock_hashes.py   # 再生成後の検証
```

`requirements-dev.lock` (CI のテスト環境用、dev extras) も同スクリプトが `requirements.lock` を制約にして同時に再生成する。
Python 3.12 以外では失敗する (lock ヘッダが 3.12 固定のため)。

### ディレクトリ構造

```
academic-paper-system/
├── academic_paper/          # ソースコード
│   ├── __init__.py
│   ├── config.py            # Pydantic Settings (環境変数)
│   ├── server.py            # FastAPI アプリケーション
│   ├── extractor.py         # PDF テキスト抽出
│   ├── chunker.py           # テキストチャンク分割
│   ├── embedder.py          # embedding-svc クライアント
│   ├── vector_store.py      # Qdrant クライアント
│   ├── db.py                # SQLite (FTS5 含む)
│   ├── hybrid.py            # RRF マージ
│   ├── llm.py               # LLM クライアント (Gemini / Ollama)
│   ├── summarizer.py        # RAG 要約
│   ├── nugget.py            # nugget 抽出 (クエリ関連文の選別)
│   ├── jobs.py               # バックグラウンドジョブ管理 (bulk 操作)
│   ├── retry.py              # 一時的な通信失敗のリトライ
│   ├── scorer.py             # 論文スコアリング (鮮度・カテゴリ関連度)
│   ├── http_client.py        # 注入 or 一時 AsyncClient ヘルパー
│   ├── services/             # サービス層
│   │   ├── __init__.py
│   │   └── search_service.py # 検索ロジック (FastAPI 非依存)
│   ├── logging_config.py     # 構造化 JSON ロギング設定
│   └── telemetry.py          # OpenTelemetry 計装
├── scripts/                  # 論文収集・運用スクリプト
│   ├── _collect_common.py    # 収集スクリプト共通ヘルパー
│   ├── cli_utils.py          # 収集スクリプト共通 argparse バリデータ
│   ├── ingest_client.py      # `/papers/ingest` 非同期送信クライアント
│   ├── arxiv_collect.py      # arXiv 論文収集
│   ├── openalex_collect.py   # OpenAlex 論文収集
│   ├── pubmed_collect.py     # PubMed Central 論文収集
│   ├── semantic_scholar_collect.py # Semantic Scholar 論文収集
│   ├── update-lock.sh        # requirements.lock 再生成 (--upgrade で最新化)
│   ├── fix_file_names.py     # 既存論文の file_name 不整合修正
│   └── generate_portfolio.py # ポートフォリオ静的ページ生成
├── tests/                   # テストスイート
│   ├── test_*.py
│   └── __init__.py
├── frontend/                # フロントエンド (index.html)
├── docs/                    # ポートフォリオ公開ページ・マイグレーション記録
├── data/                    # SQLite DB (docker-compose mount)
├── pyproject.toml           # プロジェクト設定・依存関係
├── requirements.lock        # ハッシュ固定した依存ロック
├── renovate.json            # Renovate 設定
├── dep-policy.yaml          # 依存更新ポリシー
├── docker-compose.yml       # コンテナオーケストレーション
├── Dockerfile               # コンテナイメージ
├── .github/workflows/
│   ├── ci.yml                # GitHub Actions CI
│   ├── arxiv-daily.yml       # arXiv 論文の日次自動収集
│   └── portfolio-publish.yml # ポートフォリオページの自動公開
├── README.md                # プロジェクト説明
├── .env.example             # 環境変数テンプレート
└── CLAUDE.md               # このファイル
```

## フェーズ構成

| # | Issue | 説明 | 完了 |
|---|-------|------|------|
| Phase 1 | #1 | PDF Ingest + SQLite | ✅ |
| Phase 2 | #2 | Embedding + Qdrant | ✅ |
| Phase 3 | #3 | ハイブリッド検索 RRF | ✅ |
| Phase 4 | #4 | 構造化要約 LLM RAG | ✅ |
| Phase 5 | #5 | CI/OTel/仕上げ | ✅ |

## 関連インフラ

- **embedding-svc**: MINIPC `:9092` (intfloat/multilingual-e5-base, 768次元)
- **Qdrant**: MINIPC `:6333`
- **OTel Collector**: dev-infrastructure `:4317` (オプション)
- **API ポート**: `:8020` (search-engine が `:8010` 使用中)

## トラブルシューティング

### Embedding サービスが接続できない

```bash
# embedding-svc の状態確認 (MINIPC)
curl http://<internal-host>:9092/health

# または MINIPC への SSH トンネル
ssh -N -L 9092:localhost:9092 <minipc-host>
curl http://localhost:9092/health
```

### Qdrant が接続できない

```bash
# Qdrant の状態確認 (MINIPC)
curl http://<internal-host>:6333/health

# コレクション確認
curl http://<internal-host>:6333/collections
```

### SQLite DB が見つからない

```bash
# data/ ディレクトリが存在することを確認
mkdir -p ./data

# ディレクトリパーミッション確認
ls -la ./data/
```

### LLM が利用できない (503)

```bash
# Gemini API キーを設定したか確認
echo $GOOGLE_API_KEY

# Ollama がローカルで起動しているか確認
curl http://localhost:11434/api/tags
```

## デプロイ

### Docker コンテナとしてデプロイ

```bash
# イメージをビルド
docker build -t academic-paper-system:latest .

# コンテナを実行
docker run -d \
  -p 8020:8020 \
  -v data:/data \
  --env-file .env \
  academic-paper-system:latest
```

### Kubernetes としてデプロイ

対象外（本リポジトリは Kubernetes マニフェストを提供しない）。

## Last Updated

2026-10-01 — ディレクトリ構造・テスト一覧を実ファイルに合わせて更新。

2026-08-23 — Phase 5 完了。
全 11 Issue (#96-#106) 実装・PR 化（#110-#119）。
N+1 修正 / upload 制限 / embed 実装化 / DB コンテキストマネージャ /
バッチ embedding / startup probe / JSON logging / Prometheus /metrics /
Docker CI / HEALTHCHECK / マルチステージビルド + requirements.lock
