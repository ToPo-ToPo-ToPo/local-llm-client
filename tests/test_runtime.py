"""インストーラーが Python 除外条件を無視しても、未修正版では起動させない。"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("version", [
    (3, 11, 17), (3, 12, 15), (3, 13, 17), (3, 14, 9), (3, 15, 1),
])
def test_import_rejects_unpatched_python(version):
    # 別プロセスで初回 import を検査する。HTTP SDK を読み込む前に停止する必要がある。
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.version_info = {version!r}; import local_llm_client"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "RuntimeError: local-llm-client requires Python" in result.stderr
    assert "eMudhra TLS security fixes" in result.stderr
    assert f"found {'.'.join(map(str, version))}" in result.stderr
