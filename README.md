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

# 動画入力（ゲートウェイが ffmpeg でフレーム抽出して画像として渡す。llama-cpp / mlx-vlm 共通）
print(llm.respond("この動画で何が起きている？", videos=["clip.mp4"]))
```

音声認識（STT）は `transcribe()`。ゲートウェイの whisper バックエンドへ音声を送る
（エージェント側に mlx-whisper は不要。model に whisper 系 ID を指定するだけ）:

```python
stt = LLMClient(model="mlx-community/whisper-large-v3-turbo",
                base_url="http://127.0.0.1:8799/v1")
print(stt.transcribe("input.wav", language="ja"))        # → 文字起こし（文字列）
print(stt.transcribe(audio_bytes, filename="clip.mp3"))  # バイトでも可
print(stt.transcribe("speech.wav", translate=True))      # 英訳
seg = stt.transcribe("input.wav", response_format="verbose_json")  # 区間・言語つき
```

> STT を使うにはゲートウェイ側に ffmpeg CLI が必要（→ local-llm-server の
> [音声認識（STT / whisper）](https://github.com/ToPo-ToPo-ToPo/local-llm-server/blob/main/docs/gateway.md#音声認識stt--whisper)）。

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

## タイムアウトと無応答（ハング）対策

`timeout` を指定しなくても、`LLMClient` は**有限の既定タイムアウト**（`DEFAULT_TIMEOUT` =
read 300 秒 / connect 10 秒）を使う。無指定でも 1 回の呼び出しがプロセスを**無期限にブロック
しない**ようにするため。read タイムアウトは「次のトークンが届くまでの最大待ち時間」＝無応答
（stall）検知として働き、超えると `LLMTimeoutError` を投げる。

```python
from local_llm_client import LLMClient, LLMTimeoutError

llm = LLMClient(model="...", base_url="http://127.0.0.1:8799/v1")
try:
    print(llm.respond("これは何？", images=["photo.jpg"]))
except LLMTimeoutError as e:
    print("無応答:", e)   # 原因の当たり（画像入り時は MTP×images の既知バグ）を添えたメッセージ
```

- **長く/無制限にしたいとき**は明示する（vision のプリフィルが長いモデル等）:
  ```python
  import httpx
  LLMClient(..., timeout=httpx.Timeout(600.0, connect=10.0))  # read 10 分
  LLMClient(..., timeout=httpx.Timeout(None))                 # 無制限（自己責任）
  LLMClient(..., timeout=240)                                 # 数値なら全操作一律
  ```
- openai SDK は timeout エラーを既定で再試行する（`max_retries`、既定 2）ため、実効の最悪
  待ち時間は read × 試行回数になり得る。

> **注意: 画像入力（`images=`）× MTP（投機的デコーディング）は要注意。**
> MTP を有効にしたモデルへ画像を送ると、ゲートウェイが依存する `mlx_vlm` の既知バグで
> **エラーにならずハングする**ことがある。新しめの `local-llm-server` ゲートウェイはこの
> 組み合わせを HTTP 400 で即拒否するが、古いゲートウェイでは `LLMTimeoutError`（上記の
> 有限タイムアウト）で打ち切られる。画像を扱うなら MTP 無しのモデルを使うか、そのモデルの
> `draft_model = "off"` で MTP を切る（→ local-llm-server の
> [MTP ドキュメント](https://github.com/ToPo-ToPo-ToPo/local-llm-server/blob/main/docs/mtp.md)）。

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

## ツール呼び出しの生成中テキストを受け取る（`on_tool_args`、0.8.0）

`LLMClient(stream_tool_calls=True)`（0.9.0）でゲートウェイに頼むと、モデルが
ツール呼び出しを**生成している最中**の生テキスト（Qwen なら `<tool_call><function=…>` の形）が
ストリームの content として届く。頼んだクライアントのリクエストにだけヘッダー
`X-Stream-Tool-Calls: 1` が付き、ゲートウェイ（local-llm-server 0.38.19+）がそのリクエストだけ流すので、
同じモデルを共有するほかのクライアントには影響しない（ゲートウェイ側でモデルの既定
`stream_tool_calls = true` にしても届く。その場合は全クライアントに流れる）。`chat()` はこれを本文（`on_text`）から剥がし、途中経過を
`on_tool_args(raw_text, done)` に渡す。最終的なツール呼び出し（`.tool_calls`）は従来どおり
最後の解析済みチャンクから作る。

```python
from local_llm_client.tool_call_stream import parse_partial_tool_call

def on_tool_args(raw, done):
    p = parse_partial_tool_call(raw)   # {"name", "arguments", "partial", "arguments_text"}
    if p["name"] == "write_file" and p["partial"] and p["partial"][0] == "content":
        editor.show(p["arguments"].get("path"), p["partial"][1])   # 書きかけの本文を表示

llm = LLMClient(model="…", stream_tool_calls=True)
msg = llm.chat(messages, tools, on_text=print, on_tool_args=on_tool_args)
```

頼まないとき（既定）はマーカーが来ないので、挙動は従来と同じ。

## 思考（thinking）の本文を受け取る（`on_reasoning`、0.10.0）

`LLMClient(enable_thinking=True)` で思考が有効なモデルは、推論バックエンド（mlx-vlm 等）が思考を本文と
分けて `reasoning_content`（実装によっては `reasoning`）で返す。`chat()` に `on_reasoning` を渡すと、その
断片を受け取れる。本文（`on_text`）には流れない。渡さなければ従来どおり捨てる。

```python
llm = LLMClient(model="…", enable_thinking=True)
msg = llm.chat(messages, tools, on_text=print, on_reasoning=lambda t: thinking_panel.append(t))
```

思考がバックエンドで分離されずに本文へ混ざってきた場合（`<think>…</think>` など）は、これまでどおり
本文から剥がして捨てる（`on_reasoning` には渡さない）。
