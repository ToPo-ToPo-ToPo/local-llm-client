"""思考チャネル（reasoning channel）の制御トークンを応答テキストから剥がすフィルタ。

ローカル LLM（Qwen 系の ``<think>…</think>`` や gpt-oss/Harmony 系の
``<|channel|>analysis<|message|>…`` など）は、推論・思考の中間出力を専用チャネルに
分けて出す。本来は推論バックエンド（mlx-vlm / vLLM 等）が最終テキストから分離して
``reasoning_content`` に振り分けるが、モデルの chat template が思考抑制フラグ
（``enable_thinking`` は Qwen 系の慣習で、Gemma/Harmony 系テンプレートは解釈しない）を
無視したり、バックエンドに reasoning パーサが無かったりすると、制御トークンや思考本文が
``delta.content`` に混ざって素通ししてくる。とくに native tool_mode でツールを呼ぶ
ターンで顕在化する（analysis/commentary チャネルの前置きが content 側に載る）。

このクライアントは content を受け取る境界なので、ここで剥がせば全ゲートウェイ利用者を
1 箇所で守れる。剥がすのは「モデル内部の制御トークン」であって表示整形ではない
（本来 content に入るべきでないものを取り除く）ため、"整形はフロント責務" の原則とは
別物として扱う。

方針:
- ``<think>…</think>`` / ``<thinking>…</thinking>`` / ``<reasoning>…</reasoning>`` は
  ブロックごと（中身も）削除する。
- Harmony の analysis / commentary / thought チャネルは本文ごと削除する。final
  チャネルの本文だけを残す（マーカーは剥がす）。
- 崩れたマーカー（``<channel|>`` のように ``|`` が片方だけ等）にも寛容にする。
- ストリーミングでマーカーがチャンク境界で分断されても扱えるよう、末尾の未確定
  部分は次のチャンクまで保留する（ReasoningStreamFilter）。

判断が難しいのは「本物のテキストを食わないこと」。制御は必ず ``<`` 始まりの
マーカーとして現れるので、マーカーに隣接した既知ラベル（analysis/commentary/
final/thought）だけを対象にし、素の英単語（"I thought…" の thought や
"final answer" の final）には触れない。
"""
from __future__ import annotations

import re

# ``<think>`` / ``<thinking>`` / ``<reasoning>`` の開閉（Qwen 系の思考ブロック）。
_THINK_OPEN = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)

# Harmony のチャネルヘッダ ``<|channel|>LABEL`` （``<|message|>`` は任意で続く）。
# ``|`` が欠けた崩れ（``<channel|>`` / ``<|channel>`` / ``<channel>``）にも寛容。
_CHANNEL_HDR = re.compile(
    r"<\|?channel\|?>\s*(?P<label>[a-zA-Z]+)?\s*(?:<\|?message\|?>)?",
    re.IGNORECASE,
)
# analysis / commentary / thought チャネルは「本文ごと」落とす（final は残す）。
_DROP_LABELS = {"analysis", "commentary", "thought", "reasoning"}
# 単体で現れる Harmony 制御トークン（本文は伴わない）。``<|start|>role`` の role 名は
# 直後の非マーカー文字なので、start のみ後続の1語を一緒に食う。
_CTRL_START = re.compile(r"<\|?start\|?>[ \t]*[a-zA-Z0-9_.-]*", re.IGNORECASE)
_CTRL_OTHER = re.compile(
    r"<\|?(?:end|message|constrain|return|call)\|?>", re.IGNORECASE
)
# ドロップ中のチャネル本文を終わらせるマーカー（次のチャネル/開始/終了/返却）。
_CHANNEL_END = re.compile(
    r"<\|?(?:channel|start|end|return)\|?>", re.IGNORECASE
)
# チャンク末尾の「マーカーになりかけ」。``<`` 始まりで英字/``|``/``/`` だけが続き、
# まだ閉じていない短い断片なら次のチャンクまで保留する。
_PARTIAL_TAIL = re.compile(r"<\|?/?[a-zA-Z|]{0,12}$")


class ReasoningStreamFilter:
    """細切れチャンクから思考チャネルを剥がすステートフルなフィルタ。

    feed(chunk) は「いま確定して出してよい」クリーンテキストを返す。マーカーが
    チャンク境界で分断された場合は末尾を保留し、次の feed で続きを処理する。
    1 応答の終端で flush() を呼ぶと、保留していた残りをクリーンにして返す。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._drop: str | None = None  # None / "think" / "channel": 本文ドロップ中の種別

    # ------------------------------------------------------------------
    def feed(self, chunk: str) -> str:
        self._buf += chunk or ""
        out: list[str] = []
        self._consume(out, hold_partial=True)
        return "".join(out)

    def flush(self) -> str:
        """終端処理: 保留していた末尾も含めて残り全部をクリーンにして返す。"""
        out: list[str] = []
        self._consume(out, hold_partial=False)
        rest = self._buf
        self._buf = ""
        # ドロップ中に終端したら残りは思考本文なので捨てる。
        if self._drop is not None:
            self._drop = None
            return "".join(out)
        return "".join(out)

    # ------------------------------------------------------------------
    def _consume(self, out: list[str], *, hold_partial: bool) -> None:
        while True:
            if self._drop == "think":
                m = _THINK_CLOSE.search(self._buf)
                if not m:
                    # 閉じ待ち。末尾の閉じタグ断片だけ残して本文は捨てる。
                    self._buf = _tail_after_partial(self._buf, hold_partial)
                    return
                self._buf = self._buf[m.end():]
                self._drop = None
                continue
            if self._drop == "channel":
                m = _CHANNEL_END.search(self._buf)
                if not m:
                    self._buf = _tail_after_partial(self._buf, hold_partial)
                    return
                # 終端マーカーは次段で解釈するため残す。
                self._buf = self._buf[m.start():]
                self._drop = None
                continue

            # NORMAL: 次に現れる制御構造を探す。
            think = _THINK_OPEN.search(self._buf)
            chan = _CHANNEL_HDR.search(self._buf)
            start = _CTRL_START.search(self._buf)
            other = _CTRL_OTHER.search(self._buf)
            stray_close = _THINK_CLOSE.search(self._buf)
            cands = [m for m in (think, chan, start, other, stray_close) if m]
            if not cands:
                # 制御なし。末尾のマーカー断片だけ保留して残りを出す。
                cut = _partial_start(self._buf) if hold_partial else len(self._buf)
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                return
            m = min(cands, key=lambda x: x.start())
            # マーカーがバッファ末尾でちょうど終わる = ラベルや role 名がまだ伸びる
            # 可能性がある（チャネルラベルがチャンク境界で分断）。次のチャンクまで保留し、
            # 部分ラベルを誤って確定しない（"analysis" が "ana" で切れる事故を防ぐ）。
            if hold_partial and m.end() == len(self._buf):
                out.append(self._buf[:m.start()])
                self._buf = self._buf[m.start():]
                return
            out.append(self._buf[:m.start()])

            if m is think:
                self._buf = self._buf[m.end():]
                self._drop = "think"
            elif m is stray_close:
                # 開始のない閉じタグは単に捨てる。
                self._buf = self._buf[m.end():]
            elif m is chan:
                label = (m.group("label") or "").lower()
                self._buf = self._buf[m.end():]
                if label in _DROP_LABELS:
                    self._drop = "channel"
                # final / 未知ラベルはヘッダだけ剥がし本文は残す。
            else:  # start / other 制御トークン
                self._buf = self._buf[m.end():]


def _partial_start(s: str) -> int:
    """末尾の「マーカーになりかけ」断片の開始位置。無ければ len(s)。"""
    m = _PARTIAL_TAIL.search(s)
    if m and m.start() < len(s):  # 実際に "<" を含む断片が末尾にある
        return m.start()
    return len(s)


def _tail_after_partial(s: str, hold_partial: bool) -> str:
    """ドロップ中の本文を捨て、末尾の閉じ/終端マーカー断片だけ残す。"""
    if not hold_partial:
        return ""
    cut = _partial_start(s)
    return s[cut:]


def strip_reasoning(text: str | None) -> str:
    """完成テキスト（非ストリーム）から思考チャネルを剥がす。

    保存履歴・戻り値（office/CLI）・要約など、テキストが1本そろっている用途向け。
    """
    if not text:
        return ""
    f = ReasoningStreamFilter()
    return (f.feed(text) + f.flush())
