"""LLMClient / connect / 整形ヘルパのテスト（openai をモック、実サーバーには繋がない）。"""
from __future__ import annotations

import base64

import pytest

from local_llm_client import ServerNotRunningError
from local_llm_client import client as client_mod
from local_llm_client.client import (
    LLMClient,
    build_user_content,
    connect,
    thinking_extra_body,
    to_image_url,
)


@pytest.fixture(autouse=True)
def _hermetic_cwd(tmp_path, monkeypatch):
    """既定の設定探索を空の一時ディレクトリから始めさせる（ユーザーの実 local-llm-client.toml を
    拾わせない）。明示 start を渡す発見テストや load_client_config を差し替えるテストは影響なし。"""
    d = tmp_path / "_no_config"
    d.mkdir()
    monkeypatch.chdir(d)


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


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.chat = _FakeChat()


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


def test_max_tokens_forwarded(fake_openai):
    LLMClient(model="m", max_tokens=128).respond("hi")
    assert LLMClient(model="m", max_tokens=128).max_tokens == 128


def test_openai_client_accessible(fake_openai):
    # 土台の openai クライアントに直接アクセスできる（高度操作用）。
    llm = LLMClient(model="m", base_url="http://127.0.0.1:8080/v1")
    assert llm.openai.init_kwargs["base_url"] == "http://127.0.0.1:8080/v1"


def test_timeout_passed_to_openai(fake_openai):
    # timeout を渡したときだけ openai クライアントへ伝える（None なら既定に任せる）。
    assert "timeout" not in LLMClient(model="m").openai.init_kwargs
    assert LLMClient(model="m", timeout=42.0).openai.init_kwargs["timeout"] == 42.0


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
    assert llm.api_key == DEFAULT_API_KEY  # 既定（環境変数 LOCAL_LLM_API_KEY か "not-needed"）


# --- 共有設定ファイル（local-llm-client.toml）の発見・解決 -------------------
def test_find_config_file_walks_up_nearest_wins(tmp_path):
    from local_llm_client.client import _find_config_file, CONFIG_FILENAME
    (tmp_path / CONFIG_FILENAME).write_text('api_key = "root"\n', encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    # 近くに無ければ親（root）を拾う
    assert _find_config_file(deep) == tmp_path / CONFIG_FILENAME
    # 近い方があればそちらが優先
    (tmp_path / "a" / CONFIG_FILENAME).write_text('api_key = "near"\n', encoding="utf-8")
    assert _find_config_file(deep) == tmp_path / "a" / CONFIG_FILENAME


def test_load_client_config_reads_and_missing(tmp_path):
    from local_llm_client.client import load_client_config, CONFIG_FILENAME
    sub = tmp_path / "proj"
    sub.mkdir()
    assert load_client_config(sub) == {}  # まだ設定ファイルが無い
    (tmp_path / CONFIG_FILENAME).write_text(
        'api_key = "K"\nbase_url = "http://x/v1"\n', encoding="utf-8")
    cfg = load_client_config(sub)          # 親に置いたファイルを拾う
    assert cfg["api_key"] == "K" and cfg["base_url"] == "http://x/v1"


def test_load_client_config_broken_toml_is_empty(tmp_path):
    from local_llm_client.client import load_client_config, CONFIG_FILENAME
    (tmp_path / CONFIG_FILENAME).write_text("this is : not valid = toml [", encoding="utf-8")
    assert load_client_config(tmp_path) == {}  # 壊れていても例外を投げず {}


def test_resolve_endpoint_precedence(monkeypatch):
    from local_llm_client.client import _resolve_endpoint, DEFAULT_BASE_URL, DEFAULT_API_KEY
    # 設定ファイルに base_url / api_key がある想定
    monkeypatch.setattr(client_mod, "load_client_config",
                        lambda *a, **k: {"base_url": "http://file/v1", "api_key": "K"})
    # 未指定 → ファイルの値
    assert _resolve_endpoint(None, None) == ("http://file/v1", "K")
    # 明示引数はファイルより優先
    assert _resolve_endpoint("http://arg/v1", "ARG") == ("http://arg/v1", "ARG")
    # ファイルが空なら既定
    monkeypatch.setattr(client_mod, "load_client_config", lambda *a, **k: {})
    assert _resolve_endpoint(None, None) == (DEFAULT_BASE_URL, DEFAULT_API_KEY)


def test_client_picks_up_discovered_api_key(fake_openai, monkeypatch):
    # エージェントが api_key を渡さなくても、共有設定ファイルのキーが使われる（コード変更ゼロ）。
    monkeypatch.setattr(client_mod, "load_client_config", lambda *a, **k: {"api_key": "shared"})
    llm = LLMClient(model="m", base_url="http://gw/v1")   # api_key は渡さない
    assert llm.api_key == "shared"
    assert llm.openai.init_kwargs["api_key"] == "shared"


def test_explicit_api_key_overrides_config(fake_openai, monkeypatch):
    monkeypatch.setattr(client_mod, "load_client_config", lambda *a, **k: {"api_key": "shared"})
    llm = LLMClient(model="m", base_url="http://gw/v1", api_key="explicit")
    assert llm.api_key == "explicit"
