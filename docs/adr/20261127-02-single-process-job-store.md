# ADR: ジョブ状態はプロセス内メモリを正本とし、単一プロセス運用を前提にする

**ステータス:** Draft(遡及記録)  
**決定日:** 2026-10-08(前提を文書に明記した PR のマージ日。承認者・当時の合意日は NEEDS_CONFIRMATION)  
**review_by:** 2026-12-27

> この ADR は、すでに `main` にある設計を後から整理した記録です。当時の比較検討の経緯は、Issue・PR に残っている範囲だけを書き、推定で補っていません。

---

## 課題

取り込みと一括要約は、202 と `job_id` を返してバックグラウンドで動き、利用者は `GET /jobs/{job_id}` で結果を取りにいきます。ジョブの状態をどこに置くかを決める必要がありました。

`JobStore`(`academic_paper/jobs.py`)は、ジョブの読み取り(`get()`・`list_all()`)をプロセス内の `dict` から行い、SQLite へは書き込みで永続化します。このままだと、`uvicorn --workers>1` やレプリカ増設で、ジョブを作ったプロセスと問い合わせを受けたプロセスが別になった時点で、SQLite にはあるのに `get()` が `None`(404)を返します(Issue [#387](https://github.com/flipslidersand-labs/academic-paper-system/issues/387) の「背景」)。

## 決定

現状は、単一プロセス・単一ワーカーでの運用を前提にします。`--workers>1` やレプリカ増設には対応しません。Dockerfile の `CMD` は `--workers` を指定しない(=1)ままにし、この前提を `CLAUDE.md` に明記しました(Issue [#603](https://github.com/flipslidersand-labs/academic-paper-system/issues/603)、PR [#610](https://github.com/flipslidersand-labs/academic-paper-system/pull/610))。

あわせて、メモリを膨らませないために、完了済みジョブをメモリに最大 1000 件までしか保持しません(超えた分は SQLite にだけ残る。`DEFAULT_MAX_FINISHED_JOBS`)。退避された完了ジョブは、`GET /jobs/{job_id}` が SQLite から読みます(`get_or_load`)。`GET /jobs`(一覧)はメモリ上のジョブだけを返します。

## 理由

- Issue #603 は「恒久対応の方針は未確定のため、まず運用前提を明文化する」としている。今の規模では問題にならない、とも Issue #387 に書かれている。
- **却下した代替案**: Issue #387 の「対応案」に次の 3 つが並んでいる。いずれも、このリポジトリではまだ採っておらず、却下の理由を書いた記録も見つからなかった(採否は NEEDS_CONFIRMATION)。
  - 読み取りを毎回 SQLite から行う。
  - メモリで見つからないときだけ SQLite にフォールバックする 2 段構成にする。完了ジョブについてだけは、`get_or_load` として入っている。実行中ジョブと一覧、「実行中の同種ジョブがあるか」の判定はメモリだけを見る。
  - ジョブ状態の正本を SQLite、将来は共有ストアに一本化し、メモリはキャッシュにする。
- 数字による比較は行っていない。1000 件の上限は設定値(`jobs.py`)で、測定にもとづく値ではない。NEEDS_CONFIRMATION。

## 影響

- 水平にスケールできません。`--workers>1` にすると、ジョブの照会が 404 になります。
- サーバーを再起動すると、`pending` / `running` のジョブは `failed`(`Server restarted while job was in flight`)になります。
- 「同種ジョブが実行中か」の判定(`create_if_not_running`)もプロセス内の判定です。複数プロセスでは重複起動を防げません。

## 再検討条件

複数ワーカー・複数レプリカで動かす要件が出たとき。それ以外は未定。

## 関連

- Issue [#387](https://github.com/flipslidersand-labs/academic-paper-system/issues/387)、Issue [#603](https://github.com/flipslidersand-labs/academic-paper-system/issues/603)、PR [#610](https://github.com/flipslidersand-labs/academic-paper-system/pull/610)
- [API 仕様](../spec/api.md)の「ジョブ」、[設計書](../design.md)
