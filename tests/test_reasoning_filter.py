"""思考チャネル・サニタイザ（reasoning_filter）の単体テスト。

<think>…</think> や Harmony の <|channel|>analysis… がゲートウェイで取りこぼされ
最終出力に混ざる不具合を、当リポ側で剥がすフィルタの検証。

実行例:
    cd local-llm-client && uv run --with pytest pytest tests/test_reasoning_filter.py -v
"""
from local_llm_client.reasoning_filter import (
    ReasoningStreamFilter,
    strip_reasoning,
)


def stream(chunks):
    """チャンク列を流し、結合出力を返す（ストリーミング経路の検証）。"""
    f = ReasoningStreamFilter()
    out = []
    for c in chunks:
        out.append(f.feed(c))
    out.append(f.flush())
    return "".join(out)


# ----------------------------------------------------------------------
# 完成テキスト（strip_reasoning）: 制御構造を剥がす
# ----------------------------------------------------------------------

def test_think_block_is_removed_with_body():
    assert strip_reasoning("<think>ここは思考</think>本題だよ") == "本題だよ"


def test_thinking_and_reasoning_variants():
    assert strip_reasoning("<thinking>x</thinking>A") == "A"
    assert strip_reasoning("<reasoning>y</reasoning>B") == "B"


def test_harmony_analysis_channel_body_removed():
    t = "<|channel|>analysis<|message|>ユーザーの意図を考える<|end|>答えです"
    assert strip_reasoning(t) == "答えです"


def test_harmony_final_channel_kept_without_markers():
    t = "<|channel|>final<|message|>これが最終回答"
    assert strip_reasoning(t) == "これが最終回答"


def test_malformed_channel_marker_is_tolerated():
    # ユーザー報告の崩れた形 <channel|>
    t = "<channel|>analysis<message|>内部思考<channel|>final<message|>回答"
    assert strip_reasoning(t) == "回答"


def test_start_role_header_removed():
    t = "<|start|>assistant<|channel|>final<|message|>やあ<|return|>"
    assert strip_reasoning(t) == "やあ"


def test_stray_control_tokens_removed():
    assert strip_reasoning("答え<|end|>") == "答え"
    assert strip_reasoning("<|return|>done") == "done"


def test_stray_close_think_removed():
    assert strip_reasoning("本文</think>続き") == "本文続き"


def test_analysis_then_final_full_harmony():
    t = ("<|start|>assistant<|channel|>analysis<|message|>まず考える。"
         "次にこうする。<|end|><|start|>assistant<|channel|>final<|message|>"
         "完成した図面を保存しました。<|return|>")
    assert strip_reasoning(t) == "完成した図面を保存しました。"


# ----------------------------------------------------------------------
# 本物のテキストを食わない（過剰除去の防止）
# ----------------------------------------------------------------------

def test_plain_text_untouched():
    s = "普通の回答です。特に制御トークンはありません。"
    assert strip_reasoning(s) == s


def test_word_thought_in_prose_is_kept():
    s = "I thought about it and the final answer is 42."
    assert strip_reasoning(s) == s


def test_less_than_in_code_is_kept():
    s = "if x < y and y > z: pass"
    assert strip_reasoning(s) == s


def test_word_final_and_analysis_in_prose_kept():
    s = "The final analysis shows a commentary on the results."
    assert strip_reasoning(s) == s


# ----------------------------------------------------------------------
# ストリーミング: マーカーがチャンク境界で分断されても剥がれる
# ----------------------------------------------------------------------

def test_stream_think_split_across_chunks():
    assert stream(["前<thi", "nk>ひみ", "つ</thi", "nk>後"]) == "前後"


def test_stream_channel_split_across_chunks():
    chunks = ["<|chan", "nel|>ana", "lysis<|mess", "age|>考え中", "<|end|>結論"]
    assert stream(chunks) == "結論"


def test_stream_char_by_char_harmony():
    src = "<|channel|>analysis<|message|>思考<|end|>回答テキスト"
    assert stream(list(src)) == "回答テキスト"


def test_stream_plain_text_with_lessthan_flushes():
    # マーカーでない '<' を含む素のテキストが保留されっぱなしにならない
    assert stream(["5 < ", "10 です"]) == "5 < 10 です"


def test_stream_trailing_partial_marker_flushed_if_incomplete():
    # 応答が未完のマーカーで終わった場合、flush で残りを出す（消えない）
    out = stream(["回答おわり", " 5 <"])
    assert out == "回答おわり 5 <"


def test_empty_and_none():
    assert strip_reasoning("") == ""
    assert strip_reasoning(None) == ""
    assert stream([]) == ""


# ----------------------------------------------------------------------
# 先頭の「裸のチャネルラベル」崩れ（上流が開始 <|channel> だけを食った場合）
# ----------------------------------------------------------------------

def test_orphan_leading_label_with_close_is_dropped():
    # gemma-4 で実際に観測: 開始マーカーが消え、ラベル＋本文＋終了 <channel|> が残る。
    raw = "thought\nいろいろ考える\n<channel|>ペンギン、できました。"
    assert strip_reasoning(raw) == "ペンギン、できました。"


def test_orphan_leading_label_minimal():
    assert strip_reasoning("thought\n<channel|>できました。") == "できました。"


def test_orphan_leading_label_without_close_kept_as_prose():
    # 終端マーカーが無ければ思考と断定できない → 素のテキストとして残す（誤食しない）。
    assert strip_reasoning("thought\nそのまま本文だけ。") == "thought\nそのまま本文だけ。"


def test_leading_label_word_in_real_prose_is_not_eaten():
    # 素の英語で "Thought"/"Analysis" 始まりの文は絶対に食わない。
    assert strip_reasoning("Thought about it. Here is the plan.") \
        == "Thought about it. Here is the plan."
    assert strip_reasoning("Analysis complete — all good.") \
        == "Analysis complete — all good."


def test_stream_orphan_leading_label_close_in_same_chunk():
    # 終端が同じチャンクで見えていれば、ストリーミングでもラベル〜終端を落とす。
    assert stream(["thought\n本文\n<channel|>回答"]) == "回答"
