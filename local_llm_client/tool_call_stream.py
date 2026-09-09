"""ツール呼び出しの生成中テキスト(ストリーム)を本文から剥がし、途中経過として取り出す。

ゲートウェイ(local-llm-server 0.38.13+)の `stream_tool_calls` を有効にすると、mlx-vlm が
捨てていた `<tool_call>…</tool_call>` の生テキストが `delta.content` として逐次届く
(最後の解析済み `tool_calls` チャンクは従来どおり届く)。このモジュールは:

- `ToolCallStreamFilter`: マーカーで囲まれた区間を本文(on_text)から除き、区間内の生テキストを
  蓄積する。マーカーがチャンク境界で割れても扱う。マーカーが一切来なければ完全な素通し
  (=ゲートウェイの設定が off のときは挙動が変わらない)。
- `parse_partial_tool_call`: 途中の生テキストから「関数名・確定した引数・書きかけの引数」を
  取り出す。Qwen3 系の XML 風(`<function=NAME><parameter=K>…</parameter>`)を主対象にし、
  JSON 風(`{"name": …, "arguments": {…}}`)は名前と生の引数文字列まで(最善努力)。

対応マーカー: Qwen `<tool_call>`/`</tool_call>`、Gemma 4 `<|tool_call>`/`<tool_call|>`。
"""
from __future__ import annotations

import re
from typing import Any

TOOL_CALL_MARKERS: tuple[tuple[str, str], ...] = (
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call>", "<tool_call|>"),
)

_FUNC_RE = re.compile(r"<function=([A-Za-z0-9_.\-]+)>")
_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_.\-]+)>\n?(.*?)\n?</parameter>", re.S)
_OPEN_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_.\-]+)>\n?(.*)\Z", re.S)
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_JSON_ARGS_RE = re.compile(r'"arguments"\s*:\s*(.*)\Z', re.S)


def _partial_prefix_len(text: str, marker: str) -> int:
    """text の末尾が marker の先頭 n 文字(n<len)と一致する最大 n(境界で割れた候補)。"""
    for n in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


class ToolCallStreamFilter:
    """本文からツール呼び出し区間を除き、区間内の生テキストを蓄積する(チャンク境界に強い)。"""

    def __init__(self, markers: tuple[tuple[str, str], ...] = TOOL_CALL_MARKERS) -> None:
        self._markers = markers
        self._buf = ""            # 未判定の本文(マーカー断片の保留を含む)
        self._end: str | None = None   # 区間内のとき、待っている終了マーカー
        self.raw = ""             # 現在(または直近)の区間内の生テキスト
        self.calls: list[str] = []      # 閉じた区間の生テキスト
        self.changed = False      # 直前の feed で raw が伸びた/閉じた(on_tool_args を呼ぶ合図)
        self.closed = False       # 直前の feed で区間が閉じた

    @property
    def in_call(self) -> bool:
        return self._end is not None

    def feed(self, chunk: str | None) -> str:
        """本文断片を受け取り、表示してよい本文だけを返す。"""
        self.changed = False
        self.closed = False
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._end is not None:
                i = self._buf.find(self._end)
                if i >= 0:
                    self.raw += self._buf[:i]
                    self._buf = self._buf[i + len(self._end):]
                    self.calls.append(self.raw)
                    self._end = None
                    self.changed = True
                    self.closed = True
                    continue
                keep = _partial_prefix_len(self._buf, self._end)
                body, self._buf = self._buf[: len(self._buf) - keep], self._buf[len(self._buf) - keep:]
                if body:
                    self.raw += body
                    self.changed = True
                return "".join(out)
            # 区間外: 開始マーカーを探す
            best = None
            for start, end in self._markers:
                i = self._buf.find(start)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, start, end)
            if best is not None:
                i, start, end = best
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(start):]
                self._end = end
                self.raw = ""
                self.changed = True
                continue
            keep = max(_partial_prefix_len(self._buf, s) for s, _ in self._markers)
            visible, self._buf = self._buf[: len(self._buf) - keep], self._buf[len(self._buf) - keep:]
            out.append(visible)
            return "".join(out)
        return "".join(out)

    def flush(self) -> str:
        """終端処理。区間外の保留(マーカー断片)は本文として返す。区間内の残りは raw に足す。"""
        rest, self._buf = self._buf, ""
        if self._end is not None:
            if rest:
                self.raw += rest
                self.changed = True
            return ""
        return rest


def parse_partial_tool_call(raw: str) -> dict[str, Any]:
    """途中の生テキストから {name, arguments, partial} を取り出す。

    - name: 関数名(未確定なら None)
    - arguments: 閉じた引数 {名前: 値}(XML 風のみ。JSON 風は空)
    - partial: 書きかけの引数 (名前, ここまでの値) か None
    - arguments_text: JSON 風のとき "arguments": 以降の生文字列(最善努力)。XML 風は ""
    """
    text = raw.lstrip("\n")
    m = _FUNC_RE.search(text)
    if m:
        name = m.group(1)
        body = text[m.end():]
        args = {k: v for k, v in _PARAM_RE.findall(body)}
        # 閉じた <parameter> を取り除いた残りに、開いたままの <parameter> があれば書きかけ
        rest = _PARAM_RE.sub("", body)
        pm = _OPEN_PARAM_RE.search(rest)
        partial = (pm.group(1), pm.group(2)) if pm else None
        return {"name": name, "arguments": args, "partial": partial, "arguments_text": ""}
    stripped = text.lstrip()
    if stripped.startswith("{"):
        nm = _JSON_NAME_RE.search(stripped)
        am = _JSON_ARGS_RE.search(stripped)
        return {"name": nm.group(1) if nm else None, "arguments": {}, "partial": None,
                "arguments_text": am.group(1) if am else ""}
    return {"name": None, "arguments": {}, "partial": None, "arguments_text": ""}
