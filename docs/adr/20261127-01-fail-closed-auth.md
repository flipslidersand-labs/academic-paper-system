# ADR: 認証キーが無いときはサーバーの起動を拒否する(fail-closed)

**ステータス:** Draft(遡及記録)  
**決定日:** 2026-10-12(実装 PR のマージ日。承認者・当時の合意日は NEEDS_CONFIRMATION)  
**review_by:** 2026-12-27

> この ADR は、すでに `main` に入っている変更を後から整理した記録です。当時の比較検討の経緯は、Issue・PR に残っている範囲だけを書き、推定で補っていません。

---

## 課題

`API_KEY` が空のとき、認証が素通りしていました。サーバーは警告ログを出すだけで起動を続けたため、設定漏れがあると、取り込み・要約生成(外部 LLM の課金が発生)・本文検索の全エンドポイントが無認証で公開されました(Issue [#646](https://github.com/flipslidersand-labs/academic-paper-system/issues/646) の「背景」)。

このリポは FastAPI の API サーバーで、PDF の本文と LLM による要約を扱います。

## 決定

認証キー(`API_KEY` または `API_KEYS`、`INGEST_API_KEY`)がどれも設定されておらず、`AUTH_DISABLED=true` も明示されていないとき、起動時(lifespan)に `RuntimeError` を出して起動を拒否します。認証なしで動かすときは `AUTH_DISABLED=true`(既定 `false`)を明示します。その場合は警告ログを出して起動します。

実装: PR [#692](https://github.com/flipslidersand-labs/academic-paper-system/pull/692)(`academic_paper/server.py` の lifespan、`academic_paper/config.py` の `auth_disabled`)。

## 理由

- 設定漏れを、無認証で動き続ける形ではなく、起動の失敗という目に見える形にするため。
- **却下した代替案**
  - **警告ログだけで起動を続ける(従来の挙動)**: Issue #646 が問題として挙げた状態そのもの。ログを見ないと気づけず、気づくまで無認証になるため採らなかった。
  - **リクエストごとにも拒否する(実行時の fail-closed)**: PR #692 は採っておらず、「実行時にも拒否すべきかは要判断」と未決のまま書いている。採らなかった理由は NEEDS_CONFIRMATION。
  - ほかの代替案(例: 認証を常に必須にして無効化の手段を作らない)を比較した記録は、Issue・PR に見つからなかった。NEEDS_CONFIRMATION。
- 数字による比較は行っていない。

## 影響

- `API_KEY` を設定せずに運用していた既存のデプロイは、この変更の後は起動しません。移行には `AUTH_DISABLED=true` が要ります(PR #692 の「NEEDS_CONFIRMATION」に同じ記述がある)。
- 起動時に 1 回だけ検査します。設定オブジェクトが起動後に直接書き換えられても、リクエスト時には再検査しません(PR #692 に同じ記述がある)。
- 起動拒否は、2026-11-27 に実機で確認しました(`API_KEY` を渡さずに起動して `RuntimeError` で終了。手順と出力は [設定仕様](../spec/configuration.md))。

## 再検討条件

実行時にも拒否する要件が出たとき、または認証を外部のゲートウェイへ移すとき。それ以外は未定。

## 関連

- Issue [#646](https://github.com/flipslidersand-labs/academic-paper-system/issues/646)、PR [#692](https://github.com/flipslidersand-labs/academic-paper-system/pull/692)
- [API 仕様](../spec/api.md)の「認証とスコープ」、[設定仕様](../spec/configuration.md)
