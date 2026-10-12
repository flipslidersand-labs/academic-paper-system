#!/usr/bin/env bash
# requirements.lock と requirements-dev.lock を pyproject.toml から再生成する (#520, #660)。
#
# Usage:
#   scripts/update-lock.sh             # 現在のピンを維持したまま再生成
#   scripts/update-lock.sh --upgrade   # 全依存を最新版へ更新して再生成
#
# requirements-dev.lock は requirements.lock を制約 (-c) にして生成するため、
# 必ず runtime lock の再生成後に dev lock を生成する。
# lock ヘッダが "Python 3.12" 固定のため、3.12 以外では失敗させる。
# pip-compile は pip-tools (pip install pip-tools) が提供する。
set -euo pipefail

upgrade=()
case "${1:-}" in
  "") ;;
  --upgrade) upgrade=(--upgrade) ;;
  *)
    echo "Usage: $0 [--upgrade]" >&2
    exit 2
    ;;
esac

py_ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$py_ver" != "3.12" ]; then
  echo "ERROR: Python 3.12 required (found $py_ver); requirements.lock is pinned to 3.12" >&2
  exit 1
fi

if ! command -v pip-compile >/dev/null 2>&1; then
  echo "ERROR: pip-compile not found (pip install pip-tools)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# 現行ヘッダの `--no-index` は付けない: pip-tools 7.x では pip に「PyPI を引かない」
# として渡され、依存解決が DistributionNotFound で失敗する。
# 付け外しで lock 本体は変わらず、ヘッダのコマンド行のみ変わる (#520)。
pip-compile --generate-hashes --no-strip-extras \
  --output-file=requirements.lock "${upgrade[@]}" pyproject.toml

# dev extras (pytest 等) のロック。ランタイムのピンは requirements.lock に揃える (#487, #660)。
# --upgrade 時も dev 側のみ上がり、ランタイムは直前に再生成した lock が制約になる。
pip-compile --generate-hashes --extra dev --no-strip-extras --allow-unsafe \
  -c requirements.lock --output-file=requirements-dev.lock "${upgrade[@]}" pyproject.toml
