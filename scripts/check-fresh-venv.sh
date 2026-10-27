#!/usr/bin/env bash
# 公開前の確認: wheel を作り、openai の版ごとに**空の venv** へ入れて import とテストを回す。
#
# リポの venv（uv.lock で openai を固定）では、依存の宣言漏れや openai の版替わりに気づけない。
# 0.10.0 は httpx を宣言せずに import していて、openai 3.x（httpx2 に依存し httpx を入れない）と
# 組んだ空の環境で ``import local_llm_client`` が落ちた。ここでは利用者と同じく wheel だけを入れ、
# 依存は pyproject の宣言だけから解かせる（リポのソースは sys.path に載せない）。
#
#   scripts/check-fresh-venv.sh                 # 既定の版の組（下限・2.x の最後・3.0・最新）
#   PYTHON_VERSION=3.12.16 scripts/check-fresh-venv.sh  # Python の版を替える（既定は 3.11.18）
#   scripts/check-fresh-venv.sh 2.54.0 latest   # 版を指定（latest = 宣言の範囲で解ける最新）
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON_VERSION:-3.11.18}
CERTIFI_VERSION=${CERTIFI_VERSION:-2026.10.19}
LOG_DIR=${CHECK_LOG_DIR:-$ROOT/.tools/validation/python-$PY}
mkdir -p "$LOG_DIR"
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
        VIRTUAL_ENV="$venv" uv pip install -q "$WHEEL" pytest "certifi==$CERTIFI_VERSION"
    else
        VIRTUAL_ENV="$venv" uv pip install -q "$WHEEL" "openai==$v" pytest "certifi==$CERTIFI_VERSION"
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
    echo "== openai ${got}（HTTP 層: ${layer}）"
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
    uv pip check --python "$venv/bin/python" > "$LOG_DIR/$v-dependencies.log" 2>&1
    if (cd "$WORK" && PYTHONDONTWRITEBYTECODE=1 "$venv/bin/python" -B -m pytest -q -p no:cacheprovider tests
        ) > "$LOG_DIR/$v-tests.log" 2>&1; then
        tail -1 "$LOG_DIR/$v-tests.log"
    else
        cat "$LOG_DIR/$v-tests.log"
        echo "  NG: テストが落ちた"; fail=1
    fi
    if [ "${CHECK_TLS:-0}" = 1 ]; then
        if ! PYTHONDONTWRITEBYTECODE=1 "$venv/bin/python" -B "$ROOT/scripts/check-tls.py" \
            --output "$LOG_DIR/$v-tls.json" > "$LOG_DIR/$v-tls.log" 2>&1; then
            cat "$LOG_DIR/$v-tls.log"
            echo "  NG: TLS の拒否を確認できない経路がある"; fail=1
        else
            echo "  TLS: 正常対照成功、各 HTTP 実装で 14/14 拒否（$LOG_DIR/$v-tls.json）"
        fi
    fi
done
exit $fail
