# llm-gateway-client

ローカル LLM ゲートウェイ（[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server) / OpenAI 互換）に繋ぐ**高レベルクライアント**。

- サーバー（ゲートウェイ）は別パッケージ。これは**接続する側**（エージェント共通のクライアント）。
- 各エージェントが openai のボイラープレート（メッセージ整形・画像入力・thinking 切替・ストリーム）を
  再実装しなくて済む。
- 依存は公式 `openai` SDK のみ。

## インストール

```bash
uv add llm-gateway-client
```

## 使い方

ゲートウェイ（`local-llm-server`）を起動しておき、公開ポートに繋ぐだけ。

```python
from llm_gateway_client import LLMClient

llm = LLMClient(
    model="mlx-community/Qwen3.6-27B-4bit",
    base_url="http://127.0.0.1:8799/v1",
)
print(llm.respond("ローカル LLM の利点を3つ。"))

# 画像入力・ストリーム
print(llm.respond("これは何？", images=["photo.jpg"]))
for piece in llm.respond("長い説明を", stream=True):
    print(piece, end="", flush=True)
```

起動確認付きのワンライナー（未起動なら親切なエラー。サーバーは起動しない）:

```python
from llm_gateway_client import connect, ServerNotRunningError

try:
    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
except ServerNotRunningError:
    print("先にゲートウェイ（local-llm-server）を起動してください")
```

高度な操作（embeddings / tool calling / 構造化出力 / async など）は、土台の openai クライアントに
`llm.openai` で直接アクセスできる。素の `openai` SDK で `base_url` を指してもよい。

## ライセンス

Apache-2.0
