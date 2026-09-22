---
title: "MINIPC再起動後、ARC linux-general scale-setがGitHub側ID再割当てで409クラッシュループ"
tags: [arc, github-actions, kubernetes, argocd, nfs, ufw]
severity: high
date: "2026-09-13"
---

## 症状

MINIPC再起動後、academic-paper-system含む org 全体の CI が `test`/`gitleaks` ジョブで
永久 `queued` のまま進まなくなった。`linux-general` runner listener pod のログに:

```
failed to create message session: ... 409 Conflict: ...
RunnerScaleSetSessionConflictException: The actions runner scaleset linux-heavy
already has an active session.
```

listener 自体を復旧させた後も、今度は runner pod が `ContainerCreating` のまま
進まず、`kubectl describe pod` で以下が続いた:

```
Warning  FailedMount  mount.nfs: Connection timed out
（その後）
Warning  FailedMount  mount.nfs: access denied by server while mounting ...
```

## 原因

3層の独立した障害が重なっていた。

1. `AutoscalingRunnerSet`（k8s上、arc-runners namespace）のメタデータに
   GitHub側の scale-set ID（例: `5`）がキャッシュされている。再起動を挟んで
   GitHub側でそのIDが別の scale-set（例: `linux-heavy`）に再割当てされると、
   古いIDのままセッションを張ろうとして 409 Conflict でクラッシュループする。
2. runner pod は kind クラスタの docker bridge（例: `172.23.0.0/16`）上で動くが、
   ホストの UFW ファイアウォールは NFS ポート(111/2049/20048)を LAN サブネット
   （例: `192.168.68.0/22`）にしか許可していなかった。
3. ファイアウォールを通過しても、`/etc/exports` のクライアントACLが同様に
   LANサブネットのみだったため "access denied by server" で拒否され続けた。

## 解決策

```bash
# 1. ARC scale-set ID 衝突: 該当 AutoscalingRunnerSet を削除し ArgoCD に再作成させる
kubectl delete autoscalingrunnerset linux-general -n arc-runners
# ArgoCD が Synced/Healthy なら自動で再作成 → GitHub側から新しいIDを取得し直す

# 2. UFW: kindブリッジ範囲からのNFSアクセスを許可
sudo ufw allow from <kind-bridge-cidr> to any port 111 proto tcp
sudo ufw allow from <kind-bridge-cidr> to any port 111 proto udp
sudo ufw allow from <kind-bridge-cidr> to any port 2049 proto tcp
sudo ufw allow from <kind-bridge-cidr> to any port 20048 proto tcp
sudo ufw allow from <kind-bridge-cidr> to any port 20048 proto udp

# 3. /etc/exports にも同じCIDRを追記して再エクスポート
sudo exportfs -ra

# 最後に、詰まっていた runner pod を強制削除して新規マウントを試させる
kubectl delete pod -n arc-runners -l actions.github.com/scale-set-name=linux-general --grace-period=0 --force
```

## 予防

- kind ブリッジの CIDR は `docker network inspect kind` で確認できる。ホスト再構築や
  Docker 再起動でこの値が変わる可能性があるため、UFW/exports 側の許可ルールを
  Ansible/dotfiles 等で IaC 化しておくと再発時の復旧が速くなる（未対応）。
- ARC scale-set の ID 衝突は再起動のたびに起きうる恒常的なリスク。
  `AutoscalingRunnerSet` 再作成手順を runbook 化しておくと良い（未対応）。
- 障害の切り分け順序: (1) ホスト到達性(ping/ssh) → (2) ARC listener ログ
  (409/404等) → (3) 実際の runner pod イベント(`kubectl describe pod`)。
  レイヤーごとに1つずつ確認しないと「listenerは直ったのにまだCIが動かない」で
  混乱する（本件で実際に起きた）。
