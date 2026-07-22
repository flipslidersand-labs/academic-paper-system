# academic-paper-system

論文ナレッジベース: PDFテキスト抽出→ベクトル検索→構造化要約の RAG システム

## アーキテクチャ

```
PDF → pdfplumber抽出 → テキストチャンク分割(512/64)
    → embedding-svc:9092 (e5 768-d) → Qdrant:6333
    → SQLite FTS5 + BM25
    → RRF ハイブリッド検索
    → Gemini / Ollama で構造化要約
```

**スタック**:
- **Backend**: FastAPI + uvicorn
- **PDF処理**: pdfplumber
- **ベクトル化**: embedding-svc (e5-large-v2 768次元)
- **ベクトルDB**: Qdrant
- **検索**: SQLite FTS5 + BM25 + RRF
- **要約LLM**: Google Generative AI (Gemini) / Ollama (Mistral)
- **テレメトリ**: OpenTelemetry SDK + FastAPI instrumentation

## セットアップ

### 環境変数設定

```bash
cp .env.example .env
# .env を編集して各サービスの URL と API キーを設定
```

### Docker での実行

```bash
docker-compose up -d
```

### ローカル開発環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn academic_paper.server:app --reload --port 8020
```

## API 一覧

| Method | Path | 説明 |
|--------|------|------|
| POST | `/papers/ingest` | PDF 論文をアップロード・インデックス化 |
| GET | `/papers` | 論文一覧 (ページネーション対応) |
| GET | `/papers/{paper_id}` | 論文詳細 |
| GET | `/papers/{paper_id}/summary` | 構造化要約取得 (LLM RAG) |
| GET | `/search` | ハイブリッド検索 (`mode=hybrid/vector/keyword`) |

### `/papers/ingest` (POST)

PDF ファイルをアップロードしてインデックス化します。

**Request**:
```
Content-Type: multipart/form-data
file: <PDF file>
```

**Response** (200):
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
- 400: PDF抽出失敗 / チャンク生成失敗 / Embedding/Qdrant エラー

### `/papers` (GET)

論文一覧を取得します。

**Query params**:
- `limit`: 返す件数 (1-100, デフォルト 20)
- `offset`: スキップ件数 (デフォルト 0)

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

論文の構造化要約を取得します。キャッシュ機能付き。

**Query params**:
- `force`: キャッシュを無視して再生成 (デフォルト false)

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
  "cached": false
}
```

**Errors**:
- 404: 論文が見つからない
- 503: LLM未設定

### `/search` (GET)

論文をハイブリッド検索します。

**Query params**:
- `q`: 検索クエリ (必須, 1文字以上)
- `mode`: 検索モード (デフォルト `hybrid`)
  - `hybrid`: FTS5 (BM25) + ベクトル検索を RRF で統合
  - `keyword`: FTS5 (BM25) のみ
  - `vector`: ベクトル検索のみ
- `limit`: 返す件数 (1-100, デフォルト 10)
- `paper_id`: 特定の論文 ID に限定 (オプション)

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
| `EMBEDDING_SVC_URL` | embedding-svc URL | `http://192.168.68.63:9092` |
| `EMBEDDING_API_KEY` | embedding-svc APIキー | (空) |
| `QDRANT_URL` | Qdrant URL | `http://192.168.68.63:6333` |
| `QDRANT_API_KEY` | Qdrant APIキー | (空) |
| `QDRANT_COLLECTION` | Qdrant コレクション名 | `academic-papers` |
| `ACADEMIC_DB` | SQLite DB パス | `/data/academic.db` |
| `CHUNK_SIZE` | テキストチャンクサイズ | `512` |
| `CHUNK_OVERLAP` | チャンク間のオーバーラップ | `64` |
| `GOOGLE_API_KEY` | Gemini APIキー (要約用) | (空) |
| `OLLAMA_URL` | Ollama URL (フォールバック) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama モデル | `mistral` |
| `OTEL_ENDPOINT` | OpenTelemetry コレクタエンドポイント | (空) |
| `PORT` | API サーバーポート | `8020` |

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
│   └── summarizer.py        # RAG 要約
├── tests/                   # テストスイート
│   ├── test_*.py
│   └── __init__.py
├── data/                    # SQLite DB (docker-compose mount)
├── pyproject.toml           # プロジェクト設定・依存関係
├── docker-compose.yml       # コンテナオーケストレーション
├── Dockerfile               # コンテナイメージ
├── .github/workflows/ci.yml # GitHub Actions CI
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
| Phase 5 | #5 | CI/OTel/仕上げ | 進行中 |

## 関連インフラ

- **embedding-svc**: MINIPC `:9092` (e5-large-v2, 768次元)
- **Qdrant**: MINIPC `:6333`
- **OTel Collector**: dev-infrastructure `:4317` (オプション)
- **API ポート**: `:8020` (search-engine が `:8010` 使用中)

## トラブルシューティング

### Embedding サービスが接続できない

```bash
# embedding-svc の状態確認 (MINIPC)
curl http://192.168.68.63:9092/health

# または MINIPC への SSH トンネル
ssh -N -L 9092:localhost:9092 <minipc-host>
curl http://localhost:9092/health
```

### Qdrant が接続できない

```bash
# Qdrant の状態確認 (MINIPC)
curl http://192.168.68.63:6333/health

# コレクション確認
curl http://192.168.68.63:6333/collections
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

仕様は TBD (Phase 5 の scope)。

## Last Updated

2026-07-22 — Phase 1-4 完了。API エンドポイント・テストスイート・Docker セットアップ完成。
