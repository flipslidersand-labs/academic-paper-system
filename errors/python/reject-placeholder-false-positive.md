---
title: "reject_placeholder バリデータが '<' を含む正規のsecretを誤検知する"
tags: [pydantic, pydantic-settings, security, false-positive]
severity: medium
date: "2026-09-13"
---

## 症状

`academic_paper/config.py` の `Settings()` インスタンス化時に以下で起動時クラッシュする。

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
api_key
  Value error, Invalid value "...<...": contains placeholder.
```

`.env` の `API_KEY` にランダム生成した文字列を入れただけなのに、文字列中に偶然 `<` が
含まれていると起動不能になる。

## 原因

`reject_placeholder` フィールドバリデータ（`#239`/`#241` 由来）は
`http://<internal-host>:9092` のようなドキュメント用プレースホルダを検知する目的で
`"<" in v` だけをチェックしている。プレースホルダかどうかではなく `<` という1文字の
有無だけを見ているため、`<` を含む可能性のある任意の正規シークレット
（ランダム文字列・base64派生の一部記号セットなど）を無差別に拒否する。

## 解決策

このセッションでは `.env` の値をこの specific なチェックに引っかからない値に
差し替えることは行っていない（秘密情報を書き換えないため）。テスト時は
`API_KEY=""` で上書きして回避した。

恒久修正は未実施。対応案:
- `embedding_svc_url` / `qdrant_url` など「URL固定パターン」フィールドと
  `api_key` / `*_api_key` など「任意文字列」フィールドを同じバリデータで
  扱わない（例: URL系は `<host>` 形式の正規表現、キー系は既知の固定
  プレースホルダ文字列との完全一致に変更する）。

## 予防

汎用シークレット/APIキー系フィールドに「プレースホルダ検知」を入れる場合、
「1文字の存在チェック」ではなく「既知のプレースホルダ値との完全一致」または
「URL専用の構文チェック」に限定する。ランダム生成された正規の値を弾く可能性が
あるバリデータは、生成後に実際にSettings()をロードして疎通確認するまで気づけない。
