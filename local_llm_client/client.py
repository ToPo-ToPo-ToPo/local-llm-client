"""ローカル LLM ゲートウェイ（OpenAI 互換）へ繋ぐ高レベルクライアント。

各エージェントが個別に書いていた「system / user / 画像をまとめて chat.completions に
投げ、テキスト（or ストリーム）を受け取る」ラッパー。公式 `openai` SDK を土台にするため
自動リトライ・型付きレスポンス・ツール呼び出し/構造化出力などの高度機能もそのまま使える。

    from local_llm_client import LLMClient

    llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                    base_url="http://127.0.0.1:8799/v1")
    print(llm.respond("俳句を一つ詠んでください。"))

起動中のゲートウェイに繋ぐワンライナー（未起動なら親切なエラー。サーバーは起動しない）
なら connect():

    from local_llm_client import connect

    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
    for piece in llm.respond("ローカル LLM の利点は？", stream=True):
        print(piece, end="", flush=True)

より高度な操作（embeddings / tool calling / 構造化出力 / async など）は、`LLMClient.openai`
で土台の openai クライアントに直接アクセスできる。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import httpx  # openai の依存。timeout 既定を read（無応答）中心に組むために使う
from openai import APITimeoutError, OpenAI

from .tool_call_stream import ToolCallStreamFilter
from .reasoning_filter import ReasoningStreamFilter, strip_reasoning

TextSink = Callable[[str], None]


def _noop_text(_text: str) -> None:  # pragma: no cover
    pass

# ゲートウェイ（local-llm-server）の既定値。クライアントは公開ポートへ繋ぐだけ。
DEFAULT_MODEL = "mlx-community/Qwen3.6-27B-4bit"
DEFAULT_BASE_URL = "http://127.0.0.1:8799/v1"
# 既定は "not-needed"（認証なしのローカルゲートウェイ向け）。認証ありのゲートウェイに繋ぐときは
# LLMClient(..., api_key="＜キー＞") で実際のキーを渡す。認証なしなら送られても無視される。
DEFAULT_API_KEY = "not-needed"

# respond()/chat() の既定タイムアウト（timeout 未指定時に使う）。**有限**にすることが重要 ——
# 無指定でも 1 回の呼び出しがプロセスを無期限にブロックしないための自衛。httpx は「総リクエスト
# 時間」ではなく操作ごとの上限を持ち、ストリーミングでは read が「次のトークンが届くまでの最大
# 待ち時間」＝無応答(stall)検知として働く。vision（画像入力）のプリフィルは数十秒かかり得るので
# read は長め（既定 300s）にするが、無限ハングは必ずここで打ち切る。connect は短く（10s）して
# 未起動サーバーには素早く失敗する。より長く/無制限にしたいときは LLMClient(..., timeout=...) で
# 明示（例: httpx.Timeout(None) で無制限、数値で全操作一律、httpx.Timeout(read=..., connect=...)）。
# 注意: openai SDK は timeout エラーを既定で再試行する（max_retries、既定 2）ため、実効の最悪
# 待ち時間は read × 試行回数になり得る。
DEFAULT_READ_TIMEOUT = 300.0
DEFAULT_TIMEOUT = httpx.Timeout(DEFAULT_READ_TIMEOUT, connect=10.0)


class LLMTimeoutError(TimeoutError):
    """生成が client の timeout 内に応答しなかった（無応答/ハングの自衛用の明確なエラー）。

    openai の `APITimeoutError`（httpx の read/connect タイムアウト）を、原因の当たりを付けた
    メッセージに翻訳して投げ直す。特に画像入力（images）併用でのハングは、ゲートウェイ側の
    依存 `mlx_vlm` の既知バグ（MTP 投機的デコーディング有効時に images でハング）が原因である
    ことが多いため、その切り分けを助ける文言を添える。`TimeoutError` の subclass なので、
    従来 `TimeoutError` を捕捉していたコードでもそのまま拾える。
    """


def _auth_headers(api_key: str | None) -> dict:
    """api_key があれば Authorization: Bearer ヘッダを作る（None/空なら付けない）。

    chat（openai SDK 経由）だけでなく、raw HTTP で叩く /v1/models や在席セッション
    （/admin/sessions/*）にも同じキーを載せるための共通ヘルパー。
    """
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}

# 在席ハートビートの既定間隔（秒）。ゲートウェイの session_ttl（既定 90s）より十分短くする。
# テストはこれを 0 にしてバックグラウンドスレッドを止められる（0/負で無効）。
SESSION_HEARTBEAT_INTERVAL = 30.0


class ServerNotRunningError(RuntimeError):
    """接続先（ゲートウェイ）が応答しなかった。"""


def _post_session(base_url: str, path: str, payload: dict, timeout: float = 5.0,
                  api_key: str | None = None):
    """ゲートウェイのセッション管理エンドポイントへ POST する（best-effort）。

    ゲートウェイが未起動／未対応（旧版）でも例外を投げず None を返す（エージェント本体を
    止めない）。base_url は公開ポート（…/v1 等）でよい—ゲートウェイは末尾一致で
    `/admin/sessions/*` を拾う。api_key があれば Authorization を付ける（認証ありゲートウェイ
    では必須）。成功時は応答 JSON（dict）、失敗時は None。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}", data=data,
        headers={"Content-Type": "application/json", **_auth_headers(api_key)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _heartbeat_loop(base_url: str, agent_id: str, model: str,
                    stop: threading.Event, interval: float,
                    api_key: str | None = None) -> None:
    """stop がセットされるまで、interval ごとに在席ハートビートを送る（デーモンスレッド）。

    自身（LLMClient）を参照しないので、クライアントが GC されればハートビートも止まる
    （finalizer が stop をセットする）。ハートビートが弾かれた（未知セッション＝TTL で
    回収済み、またはゲートウェイ復帰）ら登録し直す（best-effort の自己修復）。
    """
    while not stop.wait(interval):
        if _post_session(base_url, "/admin/sessions/heartbeat", {"agent_id": agent_id},
                         api_key=api_key) is None:
            _post_session(base_url, "/admin/sessions/register",
                          {"agent_id": agent_id, "model": model}, api_key=api_key)


def _release_session(base_url: str, agent_id: str,
                     stop: threading.Event, enabled: bool,
                     api_key: str | None = None) -> None:
    """ハートビートを止め、利用終了を通知する（finalizer / close から一度だけ呼ばれる）。"""
    stop.set()
    if enabled:
        _post_session(base_url, "/admin/sessions/release", {"agent_id": agent_id},
                      api_key=api_key)


def is_ready(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.0,
             api_key: str | None = None) -> bool:
    """OpenAI 互換サーバー（ゲートウェイ）が応答可能かを判定する。

    認証ありゲートウェイでは /v1/models もキーを要求するため、api_key があれば載せる
    （無いと 401 になり False になってしまう）。
    """
    req = urllib.request.Request(f"{base_url}/models", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0,
                api_key: str | None = None) -> list[str]:
    """ゲートウェイが公開する全モデル id を /v1/models から返す（取得失敗時は []）。

    ゲートウェイはカタログとして複数モデルを並べる（先頭が必ずしもアクティブとは限らない）
    ので、モデルの提供有無は「リストに含まれるか」で判定する（→ check_model_served）。
    認証ありゲートウェイでは api_key を載せる（無いと 401 で空リストになる）。
    """
    req = urllib.request.Request(f"{base_url}/models", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            mid = it.get("id")
            if isinstance(mid, str) and mid:
                out.append(mid)
    return out


def models_match(a: str | None, b: str | None) -> bool:
    """2つのモデル名が同じものを指すかを大まかに判定する。

    パス指定とリポジトリ名のゆれ（例 /abs/path/Foo と org/Foo）を吸収するため、末尾要素
    （basename）を小文字で比較する。どちらか不明なら True（誤検知を避ける）。
    """
    if not a or not b:
        return True
    if a == b:
        return True
    base = lambda s: s.rstrip("/").split("/")[-1].lower()  # noqa: E731
    return base(a) == base(b)


def parse_host_port(base_url: str, default_port: int = 8799) -> tuple[str, int]:
    """base_url（例 http://127.0.0.1:8799/v1）から host と port を取り出す。"""
    parsed = urllib.parse.urlparse(base_url)
    return parsed.hostname or "127.0.0.1", parsed.port or default_port


def check_model_served(
    base_url: str = DEFAULT_BASE_URL, model: str | None = None, *, timeout: float = 5.0,
    api_key: str | None = None,
) -> list[str]:
    """接続先が設定モデルを提供しているか確認し、警告メッセージ群を返す（取り違え防止）。

    - 単一モデルでロード済みが食い違う → そのサーバーのモデルが使われる旨。
    - 多モデル（ゲートウェイ）でカタログに無い → リクエストが失敗しうる旨。
    一覧が取れない/モデル未指定なら警告なし（空リスト）。
    """
    if not model:
        return []
    models = list_models(base_url, timeout, api_key=api_key)
    if not models or any(models_match(m, model) for m in models):
        return []
    if len(models) == 1:
        return [
            f"the running server has loaded '{models[0]}', but '{model}' was requested. "
            "The existing server's model will be used; stop it and restart to use the "
            "configured model."
        ]
    return [
        f"the server at {base_url} does not offer '{model}' ({len(models)} models "
        "available). The request may fail; point base_url at a server that serves it."
    ]


def _is_url(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:"))


def to_image_url(ref: str) -> str:
    """画像参照（ローカルパス or URL）を OpenAI 互換の image_url 文字列に変換する。

    URL / データURI はそのまま、ローカルファイルは base64 のデータURIにする。
    """
    if _is_url(ref):
        return ref
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {ref}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def to_video_url(ref: str) -> str:
    """動画参照（ローカルパス or URL）を data URI / URL 文字列に変換する。

    URL / データURI はそのまま、ローカルファイルは base64 のデータURIにする。ゲートウェイが
    受け取って ffmpeg でフレーム抽出し、画像として上流モデルへ渡す（→ local-llm-server の動画入力）。
    """
    if _is_url(ref):
        return ref
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {ref}")
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_user_content(
    text: str, images: list[str] | None = None, videos: list[str] | None = None
) -> str | list[dict[str, Any]]:
    """テキスト（＋画像＋動画）を OpenAI 互換の user メッセージ content に組み立てる。

    画像も動画も無ければ素の文字列、あれば text パート＋image_url / video_url パートの配列を返す。
    動画（video_url）はゲートウェイ側でフレーム画像列に展開される（バックエンド非依存）。
    """
    if not images and not videos:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for ref in images or []:
        parts.append({"type": "image_url", "image_url": {"url": to_image_url(ref)}})
    for ref in videos or []:
        parts.append({"type": "video_url", "video_url": {"url": to_video_url(ref)}})
    return parts


def _messages_have_images(messages: list[dict[str, Any]] | None) -> bool:
    """メッセージ列に画像入力（image_url パート）が含まれるか。

    タイムアウト時のエラーメッセージで「画像入力を含む＝MTP×images の既知ハングかも」の
    ヒントを出すかどうかの判定に使う。build_user_content が作る OpenAI vision 形式
    （content 配列の {"type": "image_url", ...}）を検出する。
    """
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "image" in (part.get("type") or ""):
                    return True
    return False


def thinking_extra_body(enable: bool) -> dict[str, Any]:
    """思考(thinking)モードの ON/OFF をサーバーへ渡す extra_body を作る。

    バックエンドによって解釈するキーが異なるため両形式を併記する（未知キーは無視される）:
      - mlx-vlm      … トップレベル enable_thinking
      - mlx_lm/llama … chat_template_kwargs.enable_thinking

    chat.completions.create(..., extra_body=thinking_extra_body(False)) のように渡す。
    """
    return {
        "enable_thinking": enable,
        "chat_template_kwargs": {"enable_thinking": enable},
    }


# --- prompt-mode tool-calling プロトコル ------------------------------------
# ネイティブの function calling に対応しないバックエンド向けに、ツール仕様をシステム
# プロンプトへ注入し、応答中の ```tool {...}``` ブロックを解析する方式。提示（表示整形）は
# 含めない（クライアントは on_text に生テキストを流し、整形はフロントエンドの責務）。
_TOOL_FENCE = re.compile(r"```(?:tool|json)?\s*\n(.*?)```", re.DOTALL)


def build_tool_spec(tools: list[dict[str, Any]]) -> str:
    """ツール定義から、プロンプトに注入する仕様テキストを作る（prompt-mode 用）。"""
    lines = []
    for tool in tools:
        fn = tool["function"]
        params = fn.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []))
        arg_parts = [
            f'"{name}": {prop.get("type", "any")}'
            + ("" if name in required else " (任意)")
            for name, prop in props.items()
        ]
        lines.append(
            f"- {fn['name']}: {fn.get('description', '')}"
            f" 引数 {{{', '.join(arg_parts)}}}"
        )
    return (
        "## 使えるツール\n"
        "ツールを使うときは、本文の最後に次の形式のコードブロックだけを出力してください"
        "（1ブロックにつき1ツール。複数呼ぶなら複数ブロック）:\n"
        '```tool\n{"name": "<ツール名>", "arguments": {<引数>}}\n```\n'
        "例:\n"
        '```tool\n{"name": "run_command", "arguments": {"command": "ls -la"}}\n```\n'
        "重要:\n"
        '- 厳密な JSON で書くこと。引数は必ず "arguments" オブジェクトの中に入れる。\n'
        "- 「ツールを使います」と述べるだけで終わらせず、実際に上記ブロックを出力すること。\n"
        "- ツールを呼ぶと、その結果が次のメッセージで「ツール実行結果:」として渡される。\n"
        "- タスクが完了したら、ツールを使わず通常の文章で最終回答すること。\n\n"
        "利用可能なツール:\n" + "\n".join(lines)
    )


def transform_messages_for_prompt(
    messages: list[dict[str, Any]], spec: str
) -> list[dict[str, Any]]:
    """ネイティブ形式の履歴を prompt-mode 用に変換する（先頭 system に仕様注入、tool→user）。"""
    out: list[dict[str, Any]] = []
    spec_added = False
    for message in messages:
        role = message["role"]
        if role == "system" and not spec_added:
            out.append({"role": "system", "content": f"{message['content']}\n\n{spec}"})
            spec_added = True
        elif role == "tool":
            out.append(
                {"role": "user", "content": "ツール実行結果:\n" + (message.get("content") or "")}
            )
        elif role == "assistant":
            out.append({"role": "assistant", "content": message.get("content") or ""})
        else:
            out.append(dict(message))
    if not spec_added:
        out.insert(0, {"role": "system", "content": spec})
    return out


def parse_prompt_tool_calls(content: str | None) -> list[SimpleNamespace]:
    """応答テキストから ```tool ... ``` ブロックのツール呼び出しを取り出す。"""
    calls: list[SimpleNamespace] = []
    if not content:
        return calls
    for i, block in enumerate(_TOOL_FENCE.findall(content)):
        try:
            obj = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool")
        if not name:
            continue
        args = obj.get("arguments")
        if args is None:
            args = obj.get("args")
        if not isinstance(args, dict):
            # 寛容: arguments が無ければ name 以外のトップレベルキーを引数とみなす
            args = {
                k: v for k, v in obj.items() if k not in ("name", "tool", "arguments", "args")
            }
        calls.append(
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(
                    name=name, arguments=json.dumps(args, ensure_ascii=False)
                ),
            )
        )
    return calls


class LLMClient:
    """OpenAI 互換エンドポイント（ゲートウェイ）へ繋ぐクライアント。

    respond() は非ストリームでは生成テキスト（str）を、stream=True ではテキスト断片の
    Iterator[str] を返す。マルチモーダルは images にローカルパス/URL を渡す。土台の
    openai クライアントは ``self.openai`` で直接使える（embeddings / tool calling など）。

    **タイムアウト（既定は有限）**: ``timeout`` 未指定でも既定 ``DEFAULT_TIMEOUT``
    （read=300s / connect=10s の有限値）を使うので、1 回の呼び出しがプロセスを無期限に
    ブロックすることはない。read が「次のトークンが届くまでの上限」＝無応答検知として働き、
    超えると ``LLMTimeoutError`` を投げる。**画像入力 × MTP（投機的デコーディング）は上流
    ``mlx_vlm`` の既知バグでハングし得る**ため、画像を含むリクエストの timeout ではその旨の
    ヒントをエラーに添える。より長く/無制限にしたいときは ``timeout=httpx.Timeout(None)`` 等を
    明示する（openai SDK は timeout を既定で再試行するので実効の最悪待ち時間は read × 試行回数）。

    **在席セッション（既定 ON）**: 生成時にゲートウェイへ「このモデルを使う」と登録し、
    バックグラウンドで定期ハートビートを送る。クライアントを ``close()``（または ``with``
    ブロック終了 / GC / プロセス終了）で破棄すると利用終了を通知し、そのモデルを使う
    エージェントが他に居なければゲートウェイが即アンロードしてメモリを解放する。ゲートウェイが
    未対応/未起動でも黙って無効化されるだけ（エージェント本体は止めない）。``session=False``
    で完全に無効化できる。確実に解放するため、使い終わりに ``close()`` するか ``with`` を推奨:

        with LLMClient(model="org/Model:Q4") as llm:
            print(llm.respond("..."))
        # ブロックを抜けた瞬間、他に同モデル利用者が居なければメモリが即解放される
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: Any = None,
        tool_mode: str = "native",
        enable_thinking: bool = False,
        stream: bool = True,
        session: bool = True,
        agent_id: str | None = None,
        heartbeat_interval: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        # 認証ありゲートウェイ用のキー。chat（openai SDK）だけでなく、raw HTTP の在席セッション
        # にも同じキーを載せる。認証なしゲートウェイでは送られても無視される。
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        # ツール呼び出しの方式: native（API の function calling）/ prompt（仕様注入＋JSON解析）。
        self.tool_mode = tool_mode
        self.enable_thinking = enable_thinking
        self.stream = stream
        # timeout は float / httpx.Timeout / None。None のときは**有限の既定**（DEFAULT_TIMEOUT）を
        # 使う —— 無指定でも 1 回の呼び出しがプロセスを無期限にブロックしないための自衛。read が
        # 無応答(stall)検知として働く（vision のプリフィルを見込んで長めだが必ず有限）。長く/無制限に
        # したいときは明示（例 httpx.Timeout(None)）。
        resolved_timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        self.timeout = resolved_timeout
        client_kwargs: dict[str, Any] = {
            "base_url": base_url, "api_key": api_key, "timeout": resolved_timeout,
        }
        self.openai = OpenAI(**client_kwargs)

        # --- 在席セッション（即時アンロード用） ---
        self._session_enabled = bool(session)
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:12]}"
        self._hb_stop = threading.Event()
        # finalizer: 明示 close() / GC / プロセス終了 のいずれでも一度だけ release を送る。
        # _release_session は self を参照しない（=GC を妨げない）よう値だけを渡す。
        self._finalizer = weakref.finalize(
            self, _release_session, self.base_url, self.agent_id,
            self._hb_stop, self._session_enabled, self.api_key,
        )
        if self._session_enabled:
            _post_session(self.base_url, "/admin/sessions/register",
                          {"agent_id": self.agent_id, "model": self.model},
                          api_key=self.api_key)
            interval = (
                SESSION_HEARTBEAT_INTERVAL if heartbeat_interval is None
                else float(heartbeat_interval)
            )
            if interval > 0:
                threading.Thread(
                    target=_heartbeat_loop,
                    args=(self.base_url, self.agent_id, self.model, self._hb_stop,
                          interval, self.api_key),
                    daemon=True,
                ).start()

    def close(self) -> None:
        """利用終了をゲートウェイへ通知し、ハートビートを止める（冪等）。

        他に同じモデルを使うエージェントが居なければ、ゲートウェイがそのモデルを即アンロード
        してメモリを解放する。``with`` 文や GC / プロセス終了でも自動的に呼ばれるが、確実に
        即解放させたいなら使い終わりに明示的に呼ぶのが最も確実。
        """
        self._finalizer()  # weakref.finalize は一度だけ実行（再呼び出しは no-op）

    @property
    def closed(self) -> bool:
        """close 済み（セッション解除済み）なら True。"""
        return not self._finalizer.alive

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- タイムアウトの明確化（無応答ハングの自衛） --------------------------
    def _timeout_desc(self) -> str:
        """設定タイムアウトの人間可読な要約（エラーメッセージ用）。"""
        t = self.timeout
        if isinstance(t, httpx.Timeout):
            read = t.read
            return f"read={read:g}s" if read is not None else "no read limit"
        return f"{t}s"

    def _timeout_error(self, messages: list[dict[str, Any]] | None) -> LLMTimeoutError:
        """APITimeoutError を、原因の当たりを付けた LLMTimeoutError に翻訳する。"""
        msg = (
            f"the model did not respond within the client timeout "
            f"({self._timeout_desc()})."
        )
        if _messages_have_images(messages):
            msg += (
                " This request includes image input; if the target model has MTP "
                "(speculative decoding) enabled, image inputs are a known mlx-vlm hang "
                "(newer local-llm-server gateways reject that combination with HTTP 400). "
                "Send images to a model without MTP, or disable MTP for that model."
            )
        msg += (
            " Increase timeout= if the model is simply slow to prefill, or check the "
            "gateway / model server."
        )
        return LLMTimeoutError(msg)

    def respond(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """1 ターン生成する。stream=True なら断片の Iterator[str] を返す。

        videos を渡すと video_url パーツとして送り、ゲートウェイが ffmpeg でフレーム抽出して
        画像入力に展開する（llama-cpp / mlx-vlm 共通。→ local-llm-server の動画入力）。
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {"role": "user",
             "content": build_user_content(user_text, images, videos)})

        # kwargs を先に展開し、既定は setdefault で補う。こうすると呼び出し側が temperature /
        # max_tokens 等をこのターンだけ上書きでき（衝突での TypeError や既定の黙殺を防ぐ）、
        # model / messages を渡した場合も衝突しない。
        params: dict[str, Any] = {"model": self.model, "messages": messages, **kwargs}
        params.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            params.setdefault("max_tokens", self.max_tokens)

        if stream:
            return self._stream(params)
        try:
            resp = self.openai.chat.completions.create(**params)
        except APITimeoutError as exc:
            raise self._timeout_error(params.get("messages")) from exc
        # 思考チャネルが content に混ざっていれば剥がす（chat() と同じ扱い）。
        return strip_reasoning(resp.choices[0].message.content)

    def _stream(self, params: dict[str, Any]) -> Iterator[str]:
        # タイムアウト（無応答）はストリーム生成中に起きる。read タイムアウトが「次のトークンが
        # 来るまでの上限」として働き、超えると APITimeoutError → 明確な LLMTimeoutError に翻訳する。
        # 思考チャネルはチャンク境界で割れても扱えるフィルタで剥がしてから yield する。
        rf = ReasoningStreamFilter()
        try:
            stream = self.openai.chat.completions.create(stream=True, **params)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    cleaned = rf.feed(delta)
                    if cleaned:
                        yield cleaned
            tail = rf.flush()
            if tail:
                yield tail
        except APITimeoutError as exc:
            raise self._timeout_error(params.get("messages")) from exc

    # --- 音声認識（STT。ゲートウェイの whisper バックエンド経由）------------------
    def transcribe(
        self,
        audio: str | os.PathLike | bytes | bytearray | Any,
        *,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        response_format: str = "text",
        translate: bool = False,
        filename: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """音声を文字起こしする（`self.model` を STT モデルとしてゲートウェイに送る）。

        `respond()` の音声版。エージェントが openai の音声ボイラープレート（ファイル整形・
        multipart）を再実装しなくて済む。

        引数:
          audio           音声ファイルのパス（str/PathLike）、開いたバイナリファイルオブジェクト、
                          または生バイト（bytes）。
          language        入力言語のヒント（例 "ja"）。translate=True のときは無視される。
          prompt          文脈/固有名詞のヒント（whisper の initial_prompt）。
          temperature     デコード温度（省略時はサーバ既定）。
          response_format "text"（既定）/ "json" / "verbose_json" / "srt" / "vtt"。
          translate       True で英訳（/v1/audio/translations）。既定は文字起こし。
          filename        audio が bytes のときにサーバへ渡すファイル名（拡張子で音声形式を判定
                          させる。既定 "audio.wav"）。

        戻り値: response_format="text"（既定）は文字列。それ以外は openai SDK の戻り値を
        そのまま返す（"json"/"verbose_json" は `.text` を持つオブジェクト、"srt"/"vtt" は文字列）。
        """
        params: dict[str, Any] = {
            "model": self.model,
            "response_format": response_format,
            **kwargs,
        }
        if prompt:
            params["prompt"] = prompt
        if temperature is not None:
            params["temperature"] = temperature
        # translations エンドポイントは language を受け付けない（常に英訳）。
        if language and not translate:
            params["language"] = language

        endpoint = (
            self.openai.audio.translations if translate
            else self.openai.audio.transcriptions
        )
        opened = None
        try:
            if isinstance(audio, (str, os.PathLike)):
                opened = open(audio, "rb")
                file_arg: Any = opened
            elif isinstance(audio, (bytes, bytearray)):
                # openai SDK は (filename, bytes) タプルを受け付ける。拡張子でサーバが形式判定する。
                file_arg = (filename or "audio.wav", bytes(audio))
            else:
                file_arg = audio  # 既に開いたファイルオブジェクト等
            return endpoint.create(file=file_arg, **params)
        finally:
            if opened is not None:
                opened.close()

    # --- エージェントループ向け: tool-calling 付き 1 ターン -----------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: TextSink = _noop_text,
        on_tool_args: "Callable[[str, bool], None] | None" = None,
    ) -> SimpleNamespace:
        """1 ターン分の応答を受信し、`.content` / `.tool_calls`（/ `.parse_error`）を返す。

        on_tool_args(raw_text, done): ツール呼び出しの**生成中テキスト**の途中経過
        (ゲートウェイの stream_tool_calls が有効なときだけ届く)。raw_text は
        `<tool_call>` 以降に生成された生テキストの累積、done は区間が閉じた合図。
        本文(on_text)にはこの区間は流れない。途中解析は tool_call_stream.parse_partial_tool_call。

        本文は on_text に**生テキスト断片**として流す（表示整形・スピナー等の提示は呼び出し側
        ＝フロントエンドの責務。本クライアントは提示を持たない）。

        tool_mode="prompt" のときは API の tools を使わず、ツール仕様をシステムプロンプトに
        注入し、応答中の ```tool {...}``` を解析してツール呼び出しを取り出す。
        """
        tools = tools or []
        extra: dict[str, Any] = {"extra_body": thinking_extra_body(self.enable_thinking)}

        if tools and self.tool_mode == "prompt":
            spec = build_tool_spec(tools)
            converted = transform_messages_for_prompt(messages, spec)
            if self.stream:
                result = self._chat_stream(converted, [], on_text, extra)
            else:
                result = self._chat_once(converted, [], on_text, extra)
            calls = parse_prompt_tool_calls(result.content)
            parse_error = None
            if not calls and result.content and _TOOL_FENCE.search(result.content):
                parse_error = (
                    "Could not parse the tool-call JSON. "
                    'Correct format: ```tool\n{"name":"<tool_name>","arguments":{...}}\n``` '
                    '(strict JSON; put arguments inside "arguments"). Please output it again.'
                )
            return SimpleNamespace(
                content=result.content, tool_calls=calls or None, parse_error=parse_error
            )
        if self.stream:
            return self._chat_stream(messages, tools, on_text, extra, on_tool_args)
        return self._chat_once(messages, tools, on_text, extra)

    def _chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextSink,
        extra: dict[str, Any],
        on_tool_args: "Callable[[str, bool], None] | None" = None,
    ) -> SimpleNamespace:
        """ストリーミングで応答を取得する（本文断片を on_text へ、tool_calls を蓄積）。"""
        try:
            return self._chat_stream_inner(messages, tools, on_text, extra, on_tool_args)
        except APITimeoutError as exc:
            raise self._timeout_error(messages) from exc

    def _chat_stream_inner(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextSink,
        extra: dict[str, Any],
        on_tool_args: "Callable[[str, bool], None] | None" = None,
    ) -> SimpleNamespace:
        stream = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **extra,
        )
        content_parts: list[str] = []
        partial_calls: dict[int, dict[str, str]] = {}
        emitted = False
        started = False  # 本文の先頭の空白を捨てる
        pending_ws = ""  # 末尾の空白を保留し、後続テキストが来たときだけ出す
        # 思考チャネル（<think> / Harmony <|channel|>analysis…）がバックエンドで分離
        # されず content に混ざったときに剥がす。マーカーがチャンク境界で割れても扱える。
        rf = ReasoningStreamFilter()
        # ツール呼び出しの生テキスト（ゲートウェイの stream_tool_calls 有効時に content へ流れる）
        # を本文から剥がし、途中経過を on_tool_args へ渡す。マーカーが来なければ完全な素通し。
        tf = ToolCallStreamFilter()

        def _tool_progress() -> None:
            if on_tool_args is not None and tf.changed:
                on_tool_args(tf.raw, tf.closed)

        def feed_content(piece: str) -> None:
            nonlocal started, pending_ws, emitted
            if not started:
                piece = piece.lstrip()
                if not piece:
                    return
                started = True
            combined = pending_ws + piece
            stripped = combined.rstrip()
            pending_ws = combined[len(stripped):]
            if stripped:
                content_parts.append(stripped)
                on_text(stripped)
                emitted = True

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                cleaned = rf.feed(delta.content)
                if cleaned:
                    visible = tf.feed(cleaned)
                    _tool_progress()
                    if visible:
                        feed_content(visible)
            for tc in delta.tool_calls or []:
                entry = partial_calls.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    entry["id"] = tc.id
                if tc.function and tc.function.name:
                    entry["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
        # 保留していたマーカー断片・ドロップ中でない残りを掃き出す。
        tail = rf.flush()
        if tail:
            visible = tf.feed(tail)
            _tool_progress()
            if visible:
                feed_content(visible)
        tail2 = tf.flush()
        _tool_progress()
        if tail2:
            feed_content(tail2)
        if emitted:
            on_text("\n")

        tool_calls = [
            SimpleNamespace(
                id=partial_calls[i]["id"],
                function=SimpleNamespace(
                    name=partial_calls[i]["name"], arguments=partial_calls[i]["arguments"]
                ),
            )
            for i in sorted(partial_calls)
        ]
        return SimpleNamespace(
            content="".join(content_parts) or None, tool_calls=tool_calls or None
        )

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextSink,
        extra: dict[str, Any],
    ) -> SimpleNamespace:
        """非ストリーミングで 1 回応答を取得する。"""
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                **extra,
            )
        except APITimeoutError as exc:
            raise self._timeout_error(messages) from exc
        message = response.choices[0].message
        # 思考チャネルが content に混ざっていれば剥がす（ストリームと同じ扱い）。
        content = strip_reasoning(message.content).strip() or None
        if content:
            on_text(content)
            on_text("\n")
        return SimpleNamespace(content=content, tool_calls=message.tool_calls or None)

    def suggest_title(self, task: str, *, max_chars_hint: int = 15) -> str:
        """タスク文から短いタイトルを生成する（失敗時は空文字。例外は投げない）。"""
        messages = [
            {
                "role": "system",
                "content": (
                    "ユーザーのタスクを表す、フォルダ名向けの簡潔なタイトルを作ってください。"
                    f"日本語で{max_chars_hint}文字以内、記号・引用符・拡張子・前置きは付けず、"
                    "タイトルだけを1行で出力します。"
                ),
            },
            {"role": "user", "content": task},
        ]
        try:
            resp = self.openai.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3, max_tokens=64,
                stream=False, extra_body=thinking_extra_body(False),
            )
            content = strip_reasoning(resp.choices[0].message.content)
        except Exception:  # noqa: BLE001 - 補助機能。失敗時は呼び出し側がフォールバック
            return ""
        lines = content.strip().splitlines()
        return lines[0].strip() if lines else ""


def connect(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: Any = None,
    session: bool = True,
    agent_id: str | None = None,
    heartbeat_interval: float | None = None,
) -> LLMClient:
    """**起動中のゲートウェイ**に繋いだ LLMClient を返す（サーバーは起動しない）。

    挙動は 2 つだけ:
      1. base_url が応答する … そのゲートウェイに繋いだ LLMClient を返す。
      2. 応答しない          … ServerNotRunningError を投げる（先にゲートウェイを起動して）。

    繋ぐ前に死活確認したいときの薄いワンライナー。挙動は `LLMClient(model, base_url=...)` と
    同じだが、未起動なら早めに親切なエラーを出す。在席セッション（即時アンロード）は既定で
    有効—不要なら `session=False`。
    """
    if not is_ready(base_url, api_key=api_key):
        raise ServerNotRunningError(
            f"No gateway responding at {base_url}. Start it first (the local-llm-server "
            "gateway / its app), then connect again."
        )
    return LLMClient(
        model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        session=session,
        agent_id=agent_id,
        heartbeat_interval=heartbeat_interval,
    )
