"""llm-gateway-client — ローカル LLM ゲートウェイ（OpenAI 互換）に繋ぐ高レベルクライアント。

サーバー（ゲートウェイ）は別パッケージ `local-llm-server`。本パッケージは**接続する側**
（エージェント共通のクライアント）だけを提供し、各エージェントが openai のボイラープレートを
再実装しなくて済むようにする。依存は公式 `openai` SDK のみ。

    from llm_gateway_client import LLMClient, connect

    llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                    base_url="http://127.0.0.1:8799/v1")
    print(llm.respond("こんにちは", images=["photo.jpg"]))
"""
from __future__ import annotations

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    ServerNotRunningError,
    build_user_content,
    connect,
    is_ready,
    thinking_extra_body,
    to_image_url,
)

__all__ = [
    "LLMClient",
    "connect",
    "ServerNotRunningError",
    "is_ready",
    "to_image_url",
    "build_user_content",
    "thinking_extra_body",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
]
