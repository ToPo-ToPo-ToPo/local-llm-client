"""openai の HTTP 層（2.x まで httpx、3.x から httpx2）の入れ替わりに耐えることのテスト。

0.10.0 は client.py が httpx を直接 import していたため、openai 3.x（httpx2 に依存し httpx を
入れない）と組むと ``import local_llm_client`` が ModuleNotFoundError で落ちた。ここでは
(1) openai が使わない方の HTTP 層が無くても import できること、(2) 実際のソケット越しの
無応答（非ストリーム・ストリーム途中の両方）が LLMTimeoutError になることを、入っている
openai の版のまま確かめる。版をまたいだ確認は scripts/check-fresh-venv.sh で回す。
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading

import openai
import pytest

from local_llm_client import TIMEOUT_ERRORS, LLMClient, LLMTimeoutError


def _openai_http_name() -> str:
    for name in ("httpx2", "httpx"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        if mod.Timeout is openai.Timeout:
            return name
    raise RuntimeError("openai の HTTP 層が見つからない")


def test_import_does_not_need_the_other_http_layer():
    # openai が使っていない方の HTTP 層を import 不能にしても、本パッケージは import できる。
    # （0.10.0 は openai 3.x で httpx が無いと落ちた。その再発を、どの版の openai でも検出する）
    other = "httpx" if _openai_http_name() == "httpx2" else "httpx2"
    code = (
        f"import sys; sys.modules[{other!r}] = None\n"
        "import local_llm_client\n"
        "print(local_llm_client.DEFAULT_TIMEOUT)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_timeout_errors_include_the_http_layer_timeout():
    # ストリーム途中の read タイムアウトは版によって生の HTTP 層の例外で飛んでくるので、
    # APITimeoutError だけでなくその版の TimeoutException も翻訳の対象にする。
    http = __import__(_openai_http_name())
    assert openai.APITimeoutError in TIMEOUT_ERRORS
    assert http.TimeoutException in TIMEOUT_ERRORS
    assert issubclass(http.ReadTimeout, TIMEOUT_ERRORS)


# --- 実ソケットでの無応答 ---------------------------------------------------
_CHUNK = {
    "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "こん"},
                 "finish_reason": None}],
}


class _StallServer:
    """リクエストを読んだあと黙り込む HTTP サーバー。

    stream=True なら SSE のヘッダーと最初の 1 チャンクだけ返してから黙る（生成の途中で
    モデルが止まった状態）。False なら何も返さず黙る（プリフィルで止まった状態）。
    """

    def __init__(self, stream: bool) -> None:
        self.stream = stream
        self._stop = threading.Event()
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen()
        self._sock.settimeout(0.1)
        self.base_url = f"http://127.0.0.1:{self._sock.getsockname()[1]}/v1"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, body = data.split(b"\r\n\r\n", 1)
            length = 0
            for line in head.split(b"\r\n")[1:]:
                key, _, value = line.partition(b":")
                if key.strip().lower() == b"content-length":
                    length = int(value.strip())
            while len(body) < length:
                body += conn.recv(65536)
            if self.stream:
                piece = f"data: {json.dumps(_CHUNK)}\n\n".encode()
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                    + f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
                )
            self._stop.wait()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sock.close()


@pytest.fixture
def stall_server(request):
    server = _StallServer(stream=request.param)
    yield server
    server.close()


def _client(base_url: str, stream: bool) -> LLMClient:
    llm = LLMClient(model="m", base_url=base_url, session=False, stream=stream,
                    timeout=openai.Timeout(0.5, connect=2.0))
    # openai は timeout を既定で再試行する。テストを速くするため再試行しない。
    llm.openai = llm.openai.with_options(max_retries=0)
    return llm


@pytest.mark.parametrize("stall_server", [False], indirect=True)
def test_real_socket_non_stream_stall_raises_llm_timeout_error(stall_server):
    llm = _client(stall_server.base_url, stream=False)
    with pytest.raises(LLMTimeoutError) as exc:
        llm.respond("hi")
    assert "read=0.5s" in str(exc.value)
    assert isinstance(exc.value.__cause__, TIMEOUT_ERRORS)


@pytest.mark.parametrize("stall_server", [True], indirect=True)
def test_real_socket_stream_stall_mid_generation_raises_llm_timeout_error(stall_server):
    # 最初のトークンのあとで止まる。openai 2.x / 3.0〜3.14 はここで HTTP 層の生の ReadTimeout を
    # 投げる（APITimeoutError に包まない）ので、それも LLMTimeoutError になることを確かめる。
    llm = _client(stall_server.base_url, stream=True)
    pieces: list[str] = []
    with pytest.raises(LLMTimeoutError) as exc:
        for piece in llm.respond("hi", stream=True):
            pieces.append(piece)
    assert pieces == ["こん"]
    assert isinstance(exc.value.__cause__, TIMEOUT_ERRORS)


@pytest.mark.parametrize("stall_server", [True], indirect=True)
def test_real_socket_chat_stream_stall_raises_llm_timeout_error(stall_server):
    llm = _client(stall_server.base_url, stream=True)
    seen: list[str] = []
    with pytest.raises(LLMTimeoutError) as exc:
        llm.chat([{"role": "user", "content": "hi"}], [], seen.append)
    assert "こん" in "".join(seen)
    assert isinstance(exc.value.__cause__, TIMEOUT_ERRORS)
