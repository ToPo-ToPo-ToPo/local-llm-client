"""local-llm-client — ローカル LLM ゲートウェイ（OpenAI 互換）に繋ぐ高レベルクライアント。

サーバー（ゲートウェイ）は別パッケージ `local-llm-server`。本パッケージは**接続する側**
（エージェント共通のクライアント）だけを提供し、各エージェントが openai のボイラープレートを
再実装しなくて済むようにする。依存は公式 `openai` SDK のみ。

    from local_llm_client import LLMClient, connect

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
    TextSink,
    build_tool_spec,
    build_user_content,
    check_model_served,
    connect,
    is_ready,
    list_models,
    models_match,
    parse_host_port,
    parse_prompt_tool_calls,
    thinking_extra_body,
    to_image_url,
    transform_messages_for_prompt,
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
    # 接続・モデル確認ヘルパ（ゲートウェイの死活/カタログ確認）
    "list_models",
    "models_match",
    "parse_host_port",
    "check_model_served",
    # tool-calling プロトコル（prompt-mode）
    "TextSink",
    "build_tool_spec",
    "transform_messages_for_prompt",
    "parse_prompt_tool_calls",
]
