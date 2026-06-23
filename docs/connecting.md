# ゲートウェイへの接続

ゲートウェイ（[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)）を起動したら、
クライアントは公開ポート（既定 `http://127.0.0.1:8799/v1`）に繋ぎ、リクエストの `model` で使う
モデルを選ぶ。`model` は `gateway.toml` に登録済みのものを指定する（初回リクエストで遅延ロード）。
`api_key` はローカルなので任意（`"not-needed"` 等で可）。

## `LLMClient`（推奨）

公式 `openai` SDK を土台にした高レベルクライアント。`respond()` は非ストリームで生成テキスト
（`str`）、`stream=True` で断片の `Iterator[str]` を返す。

```python
from local_llm_client import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")

print(llm.respond("ローカルLLMの利点を3つ。"))                 # 非ストリーム → str

for piece in llm.respond("もっと詳しく", stream=True):          # ストリーム → Iterator[str]
    print(piece, end="", flush=True)

llm.respond("これは何？", images=["plot.png"])                  # 画像（マルチモーダル）
```

主な引数: `model` / `base_url` / `api_key` / `temperature` / `max_tokens` / `timeout`。
`respond()` は `system_prompt` / `images` / `stream` のほか、追加の `**kwargs` を
`chat.completions.create` にそのまま渡す。

### 起動確認付きのワンライナー `connect()`

`connect()` は接続前にゲートウェイの死活を確認し、繋がった `LLMClient` を返す。未起動なら
`ServerNotRunningError` を投げる（**サーバーは自前で起動しない**）。

```python
from local_llm_client import connect, ServerNotRunningError

try:
    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
except ServerNotRunningError:
    print("先にゲートウェイ（local-llm-server）を起動してください")
```

### 高度な操作（`llm.openai`）

`llm.openai` で土台の openai クライアントに直接アクセスできる。embeddings / tool calling /
構造化出力（`response_format`）/ async など、`respond()` に無い操作はこちらを使う。

```python
emb = llm.openai.embeddings.create(model="...", input="...")
```

## 他の OpenAI 互換クライアント

ゲートウェイは標準的な OpenAI 互換 API なので、`openai` SDK を直接使ったり、他言語の
クライアント、`curl` でもそのまま繋がる（このパッケージを入れなくてもよい）。

### `openai` SDK を直接

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mlx-community/Qwen3.6-27B-4bit",
    messages=[{"role": "user", "content": "こんにちは"}],
)
print(resp.choices[0].message.content)
```

### `curl`

```bash
curl -s http://127.0.0.1:8799/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```
