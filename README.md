# local-llm-client

ローカル LLM ゲートウェイ（[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server) / OpenAI 互換）に繋ぐ**高レベルクライアント**。

- サーバー（ゲートウェイ）は別パッケージ。これは**接続する側**（エージェント共通のクライアント）。
- 各エージェントが openai のボイラープレート（メッセージ整形・画像入力・thinking 切替・ストリーム）を
  再実装しなくて済む。
- 依存は公式 `openai` SDK のみ。

## インストール

```bash
uv add local-llm-client
```

## 使い方

ゲートウェイ（`local-llm-server`）を起動しておき、公開ポートに繋ぐだけ。

```python
from local_llm_client import LLMClient

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
from local_llm_client import connect, ServerNotRunningError

try:
    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
except ServerNotRunningError:
    print("先にゲートウェイ（local-llm-server）を起動してください")
```

高度な操作（embeddings / tool calling / 構造化出力 / async など）は、土台の openai クライアントに
`llm.openai` で直接アクセスできる。素の `openai` SDK で `base_url` を指してもよい。

### 別PC（ネットワーク越し）から繋ぐ

ゲートウェイを `host = "0.0.0.0"` ＋ `api_key` で公開している場合は、`base_url` をそのPCのLAN IP、
`api_key` をそのキーに合わせる（chat も在席セッションも自動でキーが載る）。詳細は
[docs/connecting.md](docs/connecting.md#別pcネットワーク越しから繋ぐ)。

```python
llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://192.168.1.5:8799/v1", api_key="＜キー＞")
```

## 在席セッション（使い終わったら即メモリ解放）

`LLMClient` は既定で、ゲートウェイに「このモデルを使う」と登録し、定期ハートビートを送る。
クライアントを破棄（`close()` / `with` ブロック終了 / プロセス終了）すると利用終了を通知し、
**そのモデルを使うエージェントが他に居なければ、ゲートウェイがそのモデルを即アンロードして
メモリを解放する**（`idle_timeout` の20分を待たない）。GPU/RAM が逼迫する環境で、使い終わった
モデルをすぐ片付けたいときに効く。

確実に即解放させるには `with` で囲むか、使い終わりに `close()` を呼ぶ:

```python
with LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
               base_url="http://127.0.0.1:8799/v1") as llm:
    print(llm.respond("..."))
# ブロックを抜けた瞬間、他に同モデル利用者が居なければメモリが即解放される
```

- **明示しなくても安全**: `close()` を呼ばずに落ちても、ゲートウェイ側がハートビート途絶を
  検出して回収する（`gateway.toml` の `session_ttl`、既定90秒）。`with`/`close()` はそれを
  待たず即座に解放するための最短手段。
- **オフにする**: `LLMClient(..., session=False)` で完全に無効化（従来どおり `idle_timeout`
  まかせ）。ゲートウェイが未対応/未起動でも自動で無効化されるだけで、エラーにはならない。
- **任意指定**: `agent_id`（既定は自動採番）、`heartbeat_interval`（既定30秒）。

> サーバー側の仕組みは local-llm-server の
> [在席ベースの即時アンロード](https://github.com/ToPo-ToPo-ToPo/local-llm-server/blob/main/docs/gateway.md#在席ベースの即時アンロード)
> を参照。

## ライセンス

Apache-2.0
