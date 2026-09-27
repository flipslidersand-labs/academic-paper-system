---
title: "リポジトリ直下の .env が pytest 全体を汚染し78件が誤って401で落ちる"
tags: [pydantic-settings, pytest, test-isolation]
severity: medium
date: "2026-09-27"
---

## 症状

CLAUDE.md のセットアップ手順通り `cp .env.example .env` して実際の値を
書き込んだ状態で `pytest` を実行すると、78 件が失敗する
（`.env` を退避すると同じコードで 403 passed）。失敗の大半は期待した
ステータスコードの代わりに `401` が返るというもの。

## 原因

`academic_paper/config.py` の `Settings` は
`model_config = SettingsConfigDict(env_file=".env")` で `.env` を読む。
`tests/conftest.py` にこれを無効化・上書きする autouse fixture が無いため、
`API_KEY` を明示的に `patch.object(settings, "api_key", ...)` していない
テストは、ローカルの `.env` に設定した本物の `API_KEY` の影響を受けて
「auth 有効」状態でテストされ、期待した 4xx/2xx の代わりに 401 になる。

CI には `.env` が存在しないため、この現象は CI では絶対に再現しない
（=リポジトリのテストスイート自体は健全）。純粋にローカル環境依存。

## 解決策

このセッションでは修正せず、issue化した（#505）。応急対応としては:

```bash
mv .env /tmp/x && pytest -q; mv /tmp/x .env
```

## 予防

`model_config = SettingsConfigDict(env_file=...)` を使う pydantic-settings
プロジェクトでは、`tests/conftest.py` に autouse fixture を1つ用意して
テスト実行中は `.env` の影響を受けない既知の値に固定するのが定石。
これが無いプロジェクトでは、「ローカルで pytest を回したら大量に落ちた」が
最初に疑うべきは自分の変更ではなくこの罠。
