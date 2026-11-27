# 利用マニュアル

論文 PDF を取り込み、検索し、要約を作るまでの手順です。各コマンドの検証状況は、手順ごとに次の印で示します。

- **確認済み**: 2026-11-27 に、この文書のコマンドを実際に実行して、書いてある結果を確かめました(Python 3.12、Linux)。
- **未検証**: 実行していません。embedding-svc・Qdrant・LLM・Docker が要る手順です。

API の全項目は [API 仕様](spec/api.md)、環境変数は[設定仕様](spec/configuration.md)にあります。

## 1. 用意するもの

| もの | 用途 |
| --- | --- |
| Python 3.12 | ローカル実行(`requirements.lock` が 3.12 固定) |
| embedding-svc(768 次元の埋め込みを返すサービス) | 取り込みと、`vector` / `hybrid` / `nugget` 検索、要約 |
| Qdrant | 同上 |
| Gemini の API キー、または Ollama | 要約(LLM が不要なら `LLM_PROVIDER=none`) |
| Docker と Docker Compose | コンテナで動かすとき |

`keyword` 検索と、論文・ジョブの参照は、embedding-svc と Qdrant が無くても動きます。

## 2. 導入

### 2.1 ローカルで動かす(確認済み。ただしポートと接続先を変えた実行で、embedding-svc / Qdrant には接続していない)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

起動には、次の環境変数が要ります。

- `API_KEY`: 認証キー。無いと起動しません。試すだけなら `AUTH_DISABLED=true` で無認証にもできます。
- `EMBEDDING_SVC_URL`、`QDRANT_URL`: 既定値はプレースホルダー(`http://<internal-host>:…`)で、そのままだと起動しません。

```bash
mkdir -p data
export API_KEY="$(openssl rand -hex 32)"
export ACADEMIC_DB=./data/academic.db
export EMBEDDING_SVC_URL=http://localhost:9092
export QDRANT_URL=http://localhost:6333
export LLM_PROVIDER=none          # 要約を使うときは「2.3 要約を使う」を参照
uvicorn academic_paper.server:app --port 8020
```

`.env` ファイルに同じ名前で書いても読まれます(`.env.example` に雛形があります)。

### 2.2 Docker Compose で動かす(未検証)

```bash
mkdir -p data
sudo chown -R 10001:10001 data/      # コンテナは uid 10001 の非 root で動く
cp .env.example .env                 # EMBEDDING_SVC_URL・QDRANT_API_KEY などを設定する
docker compose up -d
curl http://localhost:8020/health
```

- `QDRANT_API_KEY` は必須です。空だと `docker compose` が起動前に失敗します。1 つの値を Qdrant と `paper-rag` の両方に使います。`openssl rand -hex 32` で作れます。
- `paper-rag` は `127.0.0.1:8020` にだけ公開されます。外から使うには SSH トンネルかリバースプロキシを使います。
- Qdrant の保存先(`qdrant_storage` / `qdrant_snapshots`)は、初回に Compose が自動で作ります。既存のボリュームを使うときは `docker-compose.override.yml`(gitignore 済み)で `name` と `external: true` を指定します。
- `docker compose up` が `qdrant` の `restart` を含めて動くのは未検証です。`/health` が 200 を返すことを確かめてください。

### 2.3 要約を使う

`GOOGLE_API_KEY` を設定すると Gemini を使います。無ければ Ollama(`OLLAMA_URL`・`OLLAMA_MODEL`)を使います。どちらを使うかを固定するには `LLM_PROVIDER=gemini` または `ollama` にします。LLM を使わないなら `none` で、要約の API は 503 になります。この手順(Gemini・Ollama への実接続)は未検証です。

## 3. 動作確認(確認済み)

以降の例は、2.1 の環境変数が設定済みで、サーバーが `localhost:8020` で動いている前提です。

```bash
curl -s -w '\n[HTTP %{http_code}]\n' http://localhost:8020/health
```

embedding-svc と Qdrant に接続できないときの実際の出力です。

```text
{"status":"degraded","qdrant":"error","embedding_svc":"error","embedding_svc_reason":"unreachable"}
[HTTP 503]
```

両方につながっていれば 200 で `"status"` が `ok` になります(この形は未検証)。

```bash
curl -s -H "X-API-Key: $API_KEY" http://localhost:8020/stats
```

```text
{"papers":0,"chunks":0,"qdrant_points":-1}
```

`qdrant_points` が `-1` なのは、Qdrant の件数が取れていないことを表します。

## 4. 日常の操作

### 4.1 論文を取り込む

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -F "file=@paper.pdf" \
  --form-string "title=Attention Is All You Need" \
  --form-string "categories=cs.CL,cs.LG" \
  --form-string "published_date=2017-06-12" \
  http://localhost:8020/papers/ingest
```

成功すると 202 で `job_id` が返ります(形は確認済み。成功まで通した実行は、embedding-svc と Qdrant が要るため未検証)。

```text
{"job_id":"b6c329fc-ae16-4376-a34e-d97637dbbf10","paper_id":1,"status":"pending"}
```

完了を待ちます。`status` が `done` か `failed` になるまで繰り返します。

```bash
curl -s -H "X-API-Key: $API_KEY" http://localhost:8020/jobs/<job_id>
```

(`<job_id>` は、上で返った値に置き換えます。)`done` のとき `result.chunks` にチャンク数が入ります。小さなファイルなら、`?wait=true` を URL に付けると完了まで待って 200 で返ります。

取り込みが失敗した場合の実際の出力です(中身が PDF として読めないファイルを送った場合。確認済み)。

```text
{"id":"b6c329fc-ae16-4376-a34e-d97637dbbf10","status":"failed","kind":"ingest","total":1,"processed":0,"failed":1,"errors":["paper_id=1: Failed to extract text from PDF"],"started_at":1795791254.917029,"finished_at":1795791254.923997,"result":null}
```

メタデータの制約(`published_date` は `YYYY-MM-DD`、`authors` / `categories` は JSON 配列かカンマ区切り)は [API 仕様](spec/api.md)にあります。

### 4.2 検索する

```bash
curl -s -G -H "X-API-Key: $API_KEY" \
  --data-urlencode "q=attention" \
  --data-urlencode "mode=keyword" \
  http://localhost:8020/search
```

論文が無いときの実際の出力です(確認済み)。

```text
{"mode":"keyword","query":"attention","results":[]}
```

`mode` は `hybrid`(既定)・`vector`・`keyword`・`nugget` です。`keyword` 以外は embedding-svc と Qdrant が要ります(未検証)。`paper_id` で 1 本の論文に絞れます。

### 4.3 論文の一覧と要約

```bash
curl -s -H "X-API-Key: $API_KEY" "http://localhost:8020/papers?limit=5&sort=score"
curl -s -H "X-API-Key: $API_KEY" http://localhost:8020/papers/1/summary           # 保存済みの要約だけ。未生成は 404
curl -s -X POST -H "X-API-Key: $API_KEY" http://localhost:8020/papers/1/summary   # 要約を生成(LLM が要る・未検証)
curl -s -X POST -H "X-API-Key: $API_KEY" http://localhost:8020/jobs/summarize-all # 要約が無い論文をまとめて要約(LLM が要る・未検証)
```

`POST` は `admin` スコープの鍵(`API_KEY`)が要ります。`INGEST_API_KEY` では 403 です。LLM が未設定のとき、`POST /jobs/summarize-all` は 503 です(確認済み: `{"detail":"LLM not configured"}`)。`POST .../summary` は、論文が無ければ先に 404 `Paper not found` を返します(確認済み)。論文があって LLM が未設定のときの 503 は、コードから書いたもので未検証です。

### 4.4 関連度スコア

```bash
curl -s -X POST -H "X-API-Key: $API_KEY" http://localhost:8020/papers/score-all
```

論文が無いときの実際の出力です(確認済み)。

```text
{"total":0,"scored":0,"failed":0,"errors":[],"preferred_categories":["cs.AI","cs.LG","cs.CL"]}
```

### 4.5 論文を自動収集する

`scripts/` のコレクターが、外部の論文ソースから PDF を集めて `POST /papers/ingest` へ送ります。サーバーが認証付きのときは、コレクターに同じキーを `PAPER_API_KEY` で渡します。

```bash
export PAPER_API_KEY="$API_KEY"
python scripts/arxiv_collect.py --api-url http://localhost:8020 --categories cs.AI --max 5
```

オプションは `--help` で確認できます(`--help` の出力は確認済み。外部サイトへの実際の収集は未検証)。

| スクリプト | ソース |
| --- | --- |
| `scripts/arxiv_collect.py` | arXiv(`--categories`、`--max`、`--from-date`、`--until-date`) |
| `scripts/pubmed_collect.py` | PubMed Central(`--terms`、`--max`、日付の範囲) |
| `scripts/openalex_collect.py` | OpenAlex |
| `scripts/semantic_scholar_collect.py` | Semantic Scholar |

## 5. うまくいかないとき

| 症状 | 原因と対処 |
| --- | --- |
| 起動時に `API_KEY / API_KEYS is not set. Refusing to start…`(確認済み) | 認証キーが無い。`API_KEY` を設定する。認証なしで試すだけなら `AUTH_DISABLED=true` |
| 起動時に `Invalid value '…': contains placeholder` | `EMBEDDING_SVC_URL` / `QDRANT_URL`(または各 API キー)に `<…>` の値が残っている。実際の値を設定する |
| 起動時に `llm_generate_timeout … must be >=` などのエラー | タイムアウトの下限を下回っている。値を上げるか、未設定に戻す([設定仕様](spec/configuration.md)) |
| 401 `Invalid or missing API key`(確認済み) | `X-API-Key` が無い、または一致しない |
| 403 `Insufficient scope`(確認済み) | `INGEST_API_KEY` で `ingest` 以外を呼んだ。`API_KEY` を使う |
| `/health` が 503(確認済み) | `qdrant` または `embedding_svc` が `error`。`embedding_svc_reason` が `unreachable` なら接続できていない、`unauthorized` なら `EMBEDDING_API_KEY` が違う。URL とキーを確認する |
| 取り込みが 415(確認済み) | PDF として読めないファイル(先頭が `%PDF-` ではない) |
| 取り込みが 409 | 同じファイルが取り込み済み、または取り込み中(`file_hash` で判定)。`failed` の論文は、同じファイルを再度アップロードできる |
| 取り込みのジョブが `failed` | `GET /jobs/{job_id}` の `errors` を見る。`Failed to extract text from PDF` なら PDF が壊れている、暗号化されている |
| `/docs` が 404(確認済み) | 既定で無効。`ENABLE_DOCS=true` で公開できる(認証なしで全エンドポイントが見えるため、公開は慎重に) |
| `--workers` を増やしたらジョブが 404 になる | 単一プロセス前提で未対応([ADR](adr/20261127-single-process-job-store.md)) |
| Docker で `/data` に書けない(未検証) | `sudo chown -R 10001:10001 data/` |
