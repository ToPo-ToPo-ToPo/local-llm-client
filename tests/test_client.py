"""LLMClient / connect / 整形ヘルパのテスト（openai をモック、実サーバーには繋がない）。"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from local_llm_client import ServerNotRunningError
from local_llm_client import client as client_mod
from local_llm_client.client import (
    LLMClient,
    build_user_content,
    connect,
    thinking_extra_body,
    to_image_url,
    to_video_url,
)


# --- マルチモーダル content 構築 -------------------------------------------
def test_build_user_content_text_only():
    assert build_user_content("hello") == "hello"


def test_build_user_content_with_image_passthrough_url():
    content = build_user_content("見て", images=["https://example.com/a.png"])
    assert content[0] == {"type": "text", "text": "見て"}
    assert content[1]["image_url"]["url"] == "https://example.com/a.png"


def test_to_image_url_local_file_becomes_data_uri(tmp_path):
    p = tmp_path / "pix.png"
    p.write_bytes(b"\x89PNG\r\n")
    url = to_image_url(str(p))
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n"


def test_to_image_url_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        to_image_url("/no/such/file.png")


def test_build_user_content_with_video():
    # 動画は video_url パーツで送る（ゲートウェイがフレーム展開する）。
    content = build_user_content("この動画は？", videos=["https://example.com/v.mp4"])
    assert content[0] == {"type": "text", "text": "この動画は？"}
    assert content[1] == {"type": "video_url",
                          "video_url": {"url": "https://example.com/v.mp4"}}


def test_build_user_content_images_and_videos_together():
    content = build_user_content("見て", images=["http://x/a.png"],
                                 videos=["http://x/v.mp4"])
    types = [p["type"] for p in content]
    assert types == ["text", "image_url", "video_url"]


def test_to_video_url_local_file_becomes_data_uri(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp4")
    url = to_video_url(str(p))
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x00\x00\x00\x18ftypmp4"


def test_to_video_url_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        to_video_url("/no/such/clip.mp4")


# --- respond（openai クライアントをフェイクに差し替え） --------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_FakeStreamChunk("こん"), _FakeStreamChunk("にちは")])
        return _FakeResp("done")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeTranscriptions:
    """audio.transcriptions / translations のフェイク。呼び出しを記録する。"""

    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # response_format="text" は openai SDK では素の文字列を返す。
        if kwargs.get("response_format") == "text":
            return f"{self.kind}-text"
        return SimpleNamespace(text=f"{self.kind}-obj")


class _FakeAudio:
    def __init__(self):
        self.transcriptions = _FakeTranscriptions("transcribe")
        self.translations = _FakeTranscriptions("translate")


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.chat = _FakeChat()
        self.audio = _FakeAudio()


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(client_mod, "OpenAI", _FakeOpenAI)
    # 在席セッションのネットワーク I/O とハートビートスレッドを無効化（単体テストを隔離）。
    # 呼び出しは記録して、必要なテストが検証できるようにする。
    calls: list[tuple] = []
    monkeypatch.setattr(client_mod, "_post_session",
                        lambda base, path, payload, **k: calls.append((path, payload)) or {})
    monkeypatch.setattr(client_mod, "SESSION_HEARTBEAT_INTERVAL", 0.0)
    return calls


def test_respond_non_stream_returns_text(fake_openai):
    llm = LLMClient(model="m")
    assert llm.respond("hi") == "done"
    sent = llm.openai.chat.completions.calls[0]
    assert sent["messages"][-1] == {"role": "user", "content": "hi"}


def test_respond_includes_system_prompt(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("hi", system_prompt="be brief")
    msgs = llm.openai.chat.completions.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}


def test_respond_stream_yields_pieces(fake_openai):
    llm = LLMClient(model="m")
    assert list(llm.respond("hi", stream=True)) == ["こん", "にちは"]


def test_respond_passes_images(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("見て", images=["https://example.com/a.png"])
    content = llm.openai.chat.completions.calls[0]["messages"][-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "image_url"


def test_respond_passes_videos(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("この動画は？", videos=["https://example.com/v.mp4"])
    content = llm.openai.chat.completions.calls[0]["messages"][-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "video_url"
    assert content[1]["video_url"]["url"] == "https://example.com/v.mp4"


def test_max_tokens_forwarded(fake_openai):
    LLMClient(model="m", max_tokens=128).respond("hi")
    assert LLMClient(model="m", max_tokens=128).max_tokens == 128


# --- 思考チャネル除去（バックエンドが content に混ぜてきたものを剥がす） ----------

def _chat_chunk(content):
    """chat() 経路用のストリームチャンク（tool_calls を持つ delta）。"""
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_respond_non_stream_strips_reasoning(fake_openai):
    llm = LLMClient(model="m")
    leaked = "<|channel|>analysis<|message|>考える<|end|>答えです"
    llm.openai.chat.completions.create = lambda **k: _FakeResp(leaked)
    assert llm.respond("hi") == "答えです"


def test_respond_stream_strips_reasoning(fake_openai):
    llm = LLMClient(model="m")
    pieces = ["<|chan", "nel|>analysis<|mess", "age|>内部思考", "<|end|>本当の答え"]
    llm.openai.chat.completions.create = lambda **k: iter(
        [_FakeStreamChunk(p) for p in pieces]
    )
    assert "".join(llm.respond("hi", stream=True)) == "本当の答え"


def test_chat_stream_strips_reasoning_content_and_on_text(fake_openai):
    llm = LLMClient(model="m")   # 既定 stream=True → _chat_stream 経路
    pieces = ["<think>ひみつの", "推論</think>", "できたよ"]
    llm.openai.chat.completions.create = lambda **k: iter(
        [_chat_chunk(p) for p in pieces]
    )
    shown = []
    m = llm.chat([{"role": "user", "content": "x"}], [], shown.append)
    assert m.content == "できたよ"                  # 戻り値がクリーン
    assert "".join(shown).strip() == "できたよ"      # ストリーム表示もクリーン


def test_chat_clean_text_and_toolcalls_untouched(fake_openai):
    # 制御トークンの無い正常応答は一切変えない（過剰除去なし）。
    llm = LLMClient(model="m")
    llm.openai.chat.completions.create = lambda **k: iter(
        [_chat_chunk("if x < y: pass")]
    )
    shown = []
    m = llm.chat([{"role": "user", "content": "x"}], [], shown.append)
    assert m.content == "if x < y: pass"


def test_openai_client_accessible(fake_openai):
    # 土台の openai クライアントに直接アクセスできる（高度操作用）。
    llm = LLMClient(model="m", base_url="http://127.0.0.1:8080/v1")
    assert llm.openai.init_kwargs["base_url"] == "http://127.0.0.1:8080/v1"


def test_timeout_default_is_finite(fake_openai):
    # timeout 未指定でも**有限の既定**（DEFAULT_TIMEOUT）を openai クライアントへ渡す
    # （無指定でプロセスが無期限ブロックしないための自衛）。
    import httpx
    from local_llm_client.client import DEFAULT_TIMEOUT

    passed = LLMClient(model="m").openai.init_kwargs["timeout"]
    assert passed is DEFAULT_TIMEOUT
    assert isinstance(passed, httpx.Timeout)
    assert passed.read == 300.0 and passed.connect == 10.0  # 有限
    # self.timeout にも保持する（エラーメッセージ用）。
    assert LLMClient(model="m").timeout is DEFAULT_TIMEOUT


def test_timeout_explicit_passed_to_openai(fake_openai):
    # 明示 timeout はそのまま openai クライアントへ伝える。
    assert LLMClient(model="m", timeout=42.0).openai.init_kwargs["timeout"] == 42.0
    assert LLMClient(model="m", timeout=42.0).timeout == 42.0


def test_stream_tool_calls_adds_the_request_header(fake_openai):
    """stream_tool_calls=True のクライアントだけ、ゲートウェイへ「ツール呼び出しの生成中テキストを
    流して」と頼むヘッダーを全リクエストに付ける。既定は付けない（ほかのクライアントに影響しない）。"""
    from local_llm_client.client import STREAM_TOOL_CALLS_HEADER

    on = LLMClient(model="m", stream_tool_calls=True)
    assert on.stream_tool_calls is True
    assert on.openai.init_kwargs["default_headers"] == {STREAM_TOOL_CALLS_HEADER: "1"}
    off = LLMClient(model="m")
    assert off.stream_tool_calls is False
    assert "default_headers" not in off.openai.init_kwargs


def test_stream_tool_calls_header_reaches_the_wire():
    """本物の openai クライアントで、ヘッダーが実際の HTTP リクエストに乗ることを確かめる。"""
    import httpx

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-stream-tool-calls"))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}]})

    from openai import OpenAI as RealOpenAI
    c = LLMClient(model="m", stream_tool_calls=True, session=False, stream=False)
    c.openai = RealOpenAI(base_url="http://gw/v1", api_key="x",
                          default_headers=c.openai.default_headers,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.respond("hi") == "ok"
    assert seen == ["1"]


# --- タイムアウト（無応答ハング）を明確なエラーに翻訳する --------------------
def _timeout_raiser():
    import httpx
    from openai import APITimeoutError

    def create(**kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "http://gw/v1/chat"))

    return create


def test_respond_timeout_raises_llm_timeout_error(fake_openai, monkeypatch):
    from local_llm_client import LLMTimeoutError

    llm = LLMClient(model="m")
    monkeypatch.setattr(llm.openai.chat.completions, "create", _timeout_raiser())
    with pytest.raises(LLMTimeoutError) as exc:
        llm.respond("hi")
    # 有限タイムアウトの記述が載る。画像なしなので MTP ヒントは付かない。
    assert "did not respond within the client timeout" in str(exc.value)
    assert "image input" not in str(exc.value)


def test_respond_timeout_with_images_mentions_mtp(fake_openai, monkeypatch):
    from local_llm_client import LLMTimeoutError

    llm = LLMClient(model="m")
    monkeypatch.setattr(llm.openai.chat.completions, "create", _timeout_raiser())
    with pytest.raises(LLMTimeoutError) as exc:
        llm.respond("これは？", images=["https://example.com/a.png"])
    # 画像入りのハングは MTP×images の既知バグの可能性を示す。
    assert "image input" in str(exc.value) and "MTP" in str(exc.value)


def test_stream_timeout_raises_llm_timeout_error(fake_openai, monkeypatch):
    from local_llm_client import LLMTimeoutError

    llm = LLMClient(model="m")
    monkeypatch.setattr(llm.openai.chat.completions, "create", _timeout_raiser())
    with pytest.raises(LLMTimeoutError):
        list(llm.respond("hi", stream=True))


def test_llm_timeout_error_is_timeout_error():
    # 従来 TimeoutError を捕捉していたコードでもそのまま拾える。
    from local_llm_client import LLMTimeoutError
    assert issubclass(LLMTimeoutError, TimeoutError)


# --- transcribe（STT） -----------------------------------------------------
def test_transcribe_from_path_returns_text(fake_openai, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    llm = LLMClient(model="mlx-community/whisper-large-v3-turbo")
    out = llm.transcribe(str(wav), language="ja")
    assert out == "transcribe-text"  # response_format 既定 "text" → 文字列
    sent = llm.openai.audio.transcriptions.calls[0]
    assert sent["model"] == "mlx-community/whisper-large-v3-turbo"
    assert sent["language"] == "ja"
    assert sent["response_format"] == "text"
    # ファイルはバイナリで開いて渡す。
    assert hasattr(sent["file"], "read")


def test_transcribe_from_bytes_uses_filename_tuple(fake_openai):
    llm = LLMClient(model="whisper")
    llm.transcribe(b"\x00\x01audio", filename="clip.mp3")
    sent = llm.openai.audio.transcriptions.calls[0]
    assert sent["file"] == ("clip.mp3", b"\x00\x01audio")


def test_transcribe_bytes_default_filename(fake_openai):
    llm = LLMClient(model="whisper")
    llm.transcribe(bytearray(b"xyz"))
    assert llm.openai.audio.transcriptions.calls[0]["file"] == ("audio.wav", b"xyz")


def test_transcribe_translate_uses_translations_and_drops_language(fake_openai):
    llm = LLMClient(model="whisper")
    out = llm.transcribe(b"snd", translate=True, language="ja", prompt="hint")
    assert out == "translate-text"
    # translations 側が呼ばれ、transcriptions は未使用。
    assert len(llm.openai.audio.translations.calls) == 1
    assert llm.openai.audio.transcriptions.calls == []
    sent = llm.openai.audio.translations.calls[0]
    assert "language" not in sent          # 英訳は language 非対応
    assert sent["prompt"] == "hint"


def test_transcribe_verbose_json_returns_object(fake_openai):
    llm = LLMClient(model="whisper")
    out = llm.transcribe(b"snd", response_format="verbose_json", temperature=0.2)
    assert out.text == "transcribe-obj"     # 非 text は SDK 戻り値をそのまま返す
    sent = llm.openai.audio.transcriptions.calls[0]
    assert sent["response_format"] == "verbose_json"
    assert sent["temperature"] == 0.2


# --- connect（起動中ゲートウェイに繋ぐだけ。自動起動しない） ----------------
def test_connect_returns_client_when_gateway_ready(fake_openai, monkeypatch):
    monkeypatch.setattr(client_mod, "is_ready", lambda url, *a, **k: True)
    llm = connect(model="m", base_url="http://127.0.0.1:8799/v1")
    assert isinstance(llm, LLMClient)
    assert llm.base_url == "http://127.0.0.1:8799/v1"


def test_connect_raises_when_gateway_down(monkeypatch):
    # 未起動なら自前で立てず、親切なエラーを投げる。
    monkeypatch.setattr(client_mod, "is_ready", lambda url, *a, **k: False)
    with pytest.raises(ServerNotRunningError):
        connect(model="m", base_url="http://127.0.0.1:8799/v1")


# --- thinking_extra_body（バックエンド protocol ヘルパ） --------------------
def test_thinking_extra_body_emits_both_forms():
    on = thinking_extra_body(True)
    assert on["enable_thinking"] is True                       # mlx-vlm 形式
    assert on["chat_template_kwargs"]["enable_thinking"] is True  # mlx_lm/llama 形式
    off = thinking_extra_body(False)
    assert off["enable_thinking"] is False
    assert off["chat_template_kwargs"]["enable_thinking"] is False


# --- chat()（tool-calling）---------------------------------------------------
class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChoice:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


def _fake_stream(pieces):
    for p in pieces:
        yield _StreamChoice(_Delta(content=p))


def test_chat_prompt_mode_parses_tool_calls(monkeypatch):
    import local_llm_client.client as c

    class FakeChat:
        def create(self, **kw):
            # prompt-mode はストリーム。tool ブロックを含む本文を返す。
            return _fake_stream(['実行します\n', '```tool\n{"name":"run_command",',
                                 '"arguments":{"command":"ls"}}\n```'])
    class FakeOpenAI:
        def __init__(self, *a, **k): self.chat = type("X", (), {"completions": FakeChat()})()
    monkeypatch.setattr(c, "OpenAI", FakeOpenAI)

    llm = c.LLMClient(model="m", tool_mode="prompt", stream=True)
    out = []
    res = llm.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "ls して"}],
        tools=[{"function": {"name": "run_command", "description": "", "parameters": {}}}],
        on_text=out.append,
    )
    assert res.tool_calls and res.tool_calls[0].function.name == "run_command"
    assert "ls" in res.tool_calls[0].function.arguments
    assert "".join(out).strip().startswith("実行します")  # 生テキストを on_text に流す


# --- 接続・モデル確認ヘルパ -------------------------------------------------
def test_models_match_basename_and_unknown():
    from local_llm_client import models_match
    assert models_match("org/Foo", "/abs/path/Foo") is True
    assert models_match("a/X", "b/Y") is False
    assert models_match(None, "x") is True   # 不明なら誤検知しない


def test_parse_host_port():
    from local_llm_client import parse_host_port
    assert parse_host_port("http://127.0.0.1:8799/v1") == ("127.0.0.1", 8799)
    assert parse_host_port("http://host/v1") == ("host", 8799)  # 既定 8799


def test_list_models_and_check_served(monkeypatch):
    import io, json as _json
    import local_llm_client.client as c
    from local_llm_client import list_models, check_model_served

    def fake_urlopen(url, timeout=5.0):
        body = _json.dumps({"data": [{"id": "org/A"}, {"id": "org/B"}]}).encode()
        r = io.BytesIO(body); r.status = 200
        r.__enter__ = lambda s=r: s; r.__exit__ = lambda *a: False
        return r
    monkeypatch.setattr(c.urllib.request, "urlopen", fake_urlopen)

    assert list_models("http://x/v1") == ["org/A", "org/B"]
    assert check_model_served("http://x/v1", "org/A") == []          # 提供あり→警告なし
    warns = check_model_served("http://x/v1", "org/Z")               # カタログに無い
    assert warns and "does not offer" in warns[0]


# --- 在席セッション（即時アンロード） --------------------------------------
def test_session_registers_on_init(fake_openai):
    # 既定で register が送られ、agent_id と model が乗る。
    llm = LLMClient(model="m", base_url="http://gw/v1")
    paths = [c[0] for c in fake_openai]
    assert "/admin/sessions/register" in paths
    reg = next(p for p in fake_openai if p[0] == "/admin/sessions/register")
    assert reg[1]["model"] == "m" and reg[1]["agent_id"] == llm.agent_id


def test_session_disabled_sends_nothing(fake_openai):
    LLMClient(model="m", session=False)
    assert fake_openai == []


def test_session_custom_agent_id(fake_openai):
    llm = LLMClient(model="m", agent_id="agent-7")
    assert llm.agent_id == "agent-7"
    reg = next(p for p in fake_openai if p[0] == "/admin/sessions/register")
    assert reg[1]["agent_id"] == "agent-7"


def test_close_releases_session(fake_openai):
    llm = LLMClient(model="m", agent_id="agent-7")
    assert llm.closed is False
    llm.close()
    assert llm.closed is True
    rel = [p for p in fake_openai if p[0] == "/admin/sessions/release"]
    assert rel and rel[-1][1] == {"agent_id": "agent-7"}


def test_close_is_idempotent(fake_openai):
    llm = LLMClient(model="m", agent_id="agent-7")
    llm.close()
    llm.close()
    rel = [p for p in fake_openai if p[0] == "/admin/sessions/release"]
    assert len(rel) == 1  # 2 回目は no-op


def test_context_manager_releases_on_exit(fake_openai):
    with LLMClient(model="m", agent_id="agent-7") as llm:
        assert llm.respond("hi") == "done"
    rel = [p for p in fake_openai if p[0] == "/admin/sessions/release"]
    assert rel and rel[-1][1] == {"agent_id": "agent-7"}


def test_connect_passes_session_through(fake_openai, monkeypatch):
    monkeypatch.setattr(client_mod, "is_ready", lambda url, *a, **k: True)
    llm = connect(model="m", base_url="http://127.0.0.1:8799/v1", agent_id="agent-9")
    assert llm.agent_id == "agent-9"
    assert any(p[0] == "/admin/sessions/register" for p in fake_openai)


def test_post_session_swallows_errors(monkeypatch):
    # ゲートウェイ未起動/未対応でも例外を投げず None（エージェント本体を止めない）。
    import urllib.error
    def boom(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", boom)
    assert client_mod._post_session("http://gw/v1", "/admin/sessions/register", {"x": 1}) is None


# --- API キー認証（ネットワーク公開ゲートウェイ向け） ------------------------
def test_auth_headers():
    from local_llm_client.client import _auth_headers
    assert _auth_headers("k") == {"Authorization": "Bearer k"}
    assert _auth_headers("") == {}
    assert _auth_headers(None) == {}


def _capturing_urlopen(store, body=b"{}"):
    import io
    def fake(req, timeout=5.0):
        # Request のヘッダはキー先頭大文字（"Authorization"）で格納される。
        store["auth"] = req.headers.get("Authorization")
        r = io.BytesIO(body); r.status = 200
        r.__enter__ = lambda s=r: s; r.__exit__ = lambda *a: False
        return r
    return fake


def test_post_session_sends_authorization(monkeypatch):
    seen = {}
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", _capturing_urlopen(seen))
    client_mod._post_session("http://gw/v1", "/admin/sessions/register",
                             {"x": 1}, api_key="secret")
    assert seen["auth"] == "Bearer secret"
    # キー無しなら Authorization を付けない。
    client_mod._post_session("http://gw/v1", "/admin/sessions/register", {"x": 1})
    assert seen["auth"] is None


def test_is_ready_and_list_models_send_key(monkeypatch):
    import json as _json
    body = _json.dumps({"data": [{"id": "m"}]}).encode()
    seen = {}
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", _capturing_urlopen(seen, body))
    assert client_mod.is_ready("http://x/v1", api_key="secret") is True
    assert seen["auth"] == "Bearer secret"
    assert client_mod.list_models("http://x/v1", api_key="secret") == ["m"]
    assert seen["auth"] == "Bearer secret"


def test_client_threads_api_key_to_openai_and_session(monkeypatch):
    monkeypatch.setattr(client_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(client_mod, "SESSION_HEARTBEAT_INTERVAL", 0.0)
    seen: list[tuple] = []
    monkeypatch.setattr(
        client_mod, "_post_session",
        lambda base, path, payload, **k: seen.append((path, k.get("api_key"))) or {},
    )
    llm = LLMClient(model="m", base_url="http://gw/v1", api_key="secret")
    assert llm.api_key == "secret"
    assert llm.openai.init_kwargs["api_key"] == "secret"   # chat（openai SDK）へ渡る
    reg = next(a for a in seen if a[0] == "/admin/sessions/register")
    assert reg[1] == "secret"                              # 在席セッションにも同じキー


def test_client_default_api_key(fake_openai):
    from local_llm_client.client import DEFAULT_API_KEY
    llm = LLMClient(model="m", base_url="http://gw/v1")
    assert llm.api_key == DEFAULT_API_KEY  # 既定は "not-needed"（認証なしゲートウェイ向け）


# --- 思考（thinking）の本文を on_reasoning へ（本文には混ぜない） --------------------

def _thinking_chat(monkeypatch, deltas, *, stream=True, message=None):
    import local_llm_client.client as c

    class FakeChat:
        def create(self, **kw):
            if kw.get("stream"):
                return (_StreamChoice(d) for d in deltas)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = type("X", (), {"completions": FakeChat()})()

    monkeypatch.setattr(c, "OpenAI", FakeOpenAI)
    return c.LLMClient(model="m", stream=stream)


def test_chat_stream_passes_reasoning_to_on_reasoning(monkeypatch):
    """ゲートウェイが本文と分けて返す思考（reasoning_content）は on_reasoning へ。本文には混ぜない。"""
    def think(text):
        d = _Delta()
        d.reasoning_content = text
        d.reasoning = text              # 両方来ても二重に渡さない
        return d

    llm = _thinking_chat(monkeypatch, [think("17×23 を"), think("計算する。"), _Delta(content="391")])
    text, thought = [], []
    res = llm.chat([{"role": "user", "content": "17×23"}], on_text=text.append, on_reasoning=thought.append)
    assert thought == ["17×23 を", "計算する。"]
    assert res.content == "391" and "計算" not in "".join(text)


def test_chat_stream_reads_reasoning_from_model_extra(monkeypatch):
    """openai の型に無い項目は model_extra に入る。そこからも読む（reasoning だけの実装も）。"""
    d = _Delta()
    d.model_extra = {"reasoning": "考え中"}
    llm = _thinking_chat(monkeypatch, [d, _Delta(content="答え")])
    thought = []
    llm.chat([{"role": "user", "content": "q"}], on_reasoning=thought.append)
    assert thought == ["考え中"]


def test_chat_without_on_reasoning_drops_thinking(monkeypatch):
    """on_reasoning を渡さなければ従来どおり思考は捨てる（本文にも出ない）。"""
    d = _Delta()
    d.reasoning_content = "内緒の思考"
    llm = _thinking_chat(monkeypatch, [d, _Delta(content="答え")])
    text = []
    res = llm.chat([{"role": "user", "content": "q"}], on_text=text.append)
    assert res.content == "答え" and "内緒" not in "".join(text)


def test_chat_once_passes_reasoning(monkeypatch):
    """ストリーミングしない経路でも、message の思考を on_reasoning へ渡す。"""
    msg = SimpleNamespace(content="答え", tool_calls=None, reasoning_content="まとめて考えた")
    llm = _thinking_chat(monkeypatch, [], stream=False, message=msg)
    thought = []
    res = llm.chat([{"role": "user", "content": "q"}], on_reasoning=thought.append)
    assert thought == ["まとめて考えた"] and res.content == "答え"
