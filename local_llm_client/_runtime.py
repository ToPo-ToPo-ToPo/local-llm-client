"""Requires-Python の除外を無視するインストーラーでも、未修正の Python で実行させない。"""

import sys


def require_supported_python() -> None:
    """eMudhra 侵害鍵拒否を含む、使用中の系列の最小パッチを確認する。"""
    version = sys.version_info[:3]
    minimum = {
        (3, 11): (3, 11, 18),
        (3, 12): (3, 12, 16),
        (3, 13): (3, 13, 18),
        (3, 14): (3, 14, 10),
        (3, 15): (3, 15, 2),
    }.get(version[:2], (3, 11, 18))
    if version < minimum:
        required = ".".join(map(str, minimum))
        actual = ".".join(map(str, version))
        raise RuntimeError(
            f"local-llm-client requires Python {required} or later in that series "
            f"for the eMudhra TLS security fixes; found {actual}. Update Python "
            "before running this package."
        )
