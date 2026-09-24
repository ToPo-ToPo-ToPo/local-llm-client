#!/usr/bin/env bash
# 公開前の確認: wheel を作り、openai の版ごとに**空の venv** へ入れて import とテストを回す。
#
# リポの venv（uv.lock で openai を固定）では、依存の宣言漏れや openai の版替わりに気づけない。
# 0.10.0 は httpx を宣言せずに import していて、openai 3.x（httpx2 に依存し httpx を入れない）と
# 組んだ空の環境で ``import local_llm_client`` が落ちた。ここでは利用者と同じく wheel だけを入れ、
# 依存は pyproject の宣言だけから解かせる（リポのソースは sys.path に載せない）。
#
#   scripts/check-fresh-venv.sh                 # 既定の版の組（下限・2.x の最後・3.0・最新）
#   PYTHON_VERSION=3.12 scripts/check-fresh-venv.sh   # Python の版を替える（既定は下限の 3.11）
#   scripts/check-fresh-venv.sh 2.54.0 latest   # 版を指定（latest = 宣言の範囲で解ける最新）
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON_VERSION:-3.11}
VERSIONS=("$@")
if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=(1.55.3 2.54.0 3.0.0 latest)
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

uv build -q --wheel -o "$WORK/dist"
WHEEL=$(ls "$WORK"/dist/*.whl)
# テストはソースツリーの外へ写して回す（リポの local_llm_client を拾わないように）。
cp -R "$ROOT/tests" "$WORK/tests"
find "$WORK/tests" -name __pycache__ -prune -exec rm -rf {} +
echo "wheel: $(basename "$WHEEL")  python: $PY"

fail=0
for v in "${VERSIONS[@]}"; do
    venv="$WORK/venv-$v"
    uv venv -q -p "$PY" "$venv"
    if [ "$v" = latest ]; then
        VIRTUAL_ENV="$venv" uv pip install -q "$WHEEL" pytest
    else
        VIRTUAL_ENV="$venv" uv pip install -q "$WHEEL" "openai==$v" pytest
    fi
    got=$("$venv/bin/python" -c 'import openai; print(openai.__version__)')
    layer=$("$venv/bin/python" -c '
import importlib, openai
for n in ("httpx2", "httpx"):
    try:
        m = importlib.import_module(n)
    except ImportError:
        continue
    if m.Timeout is openai.Timeout:
        print(n, m.__version__)
')
    echo "== openai $got（HTTP 層: $layer）"
    # 1) import できる・ソースツリーではなく入れた wheel から来ている
    if ! (cd "$WORK" && "$venv/bin/python" -c '
import sys, local_llm_client
assert "site-packages" in local_llm_client.__file__, local_llm_client.__file__
print("  import ok:", local_llm_client.__file__)
print("  TIMEOUT_ERRORS:", [f"{e.__module__}.{e.__qualname__}" for e in local_llm_client.TIMEOUT_ERRORS])
'); then
        echo "  NG: import に失敗"; fail=1; continue
    fi
    # 2) テスト一式（実ソケットの無応答 → LLMTimeoutError の経路を含む）
    if (cd "$WORK" && "$venv/bin/pytest" -q -p no:cacheprovider tests 2>&1 | tail -1 | sed 's/^/  /'
        exit "${PIPESTATUS[0]}"); then
        :
    else
        echo "  NG: テストが落ちた"; fail=1
    fi
done
exit $fail
