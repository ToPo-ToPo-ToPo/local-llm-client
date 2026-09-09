"""ツール呼び出しの生成中テキスト: フィルタ(本文から剥がす)と途中解析、chat() の on_tool_args。"""
from types import SimpleNamespace

from local_llm_client import LLMClient

from test_client import fake_openai  # noqa: F401 - 偽 openai クライアントの fixture を共有
from local_llm_client.tool_call_stream import ToolCallStreamFilter, parse_partial_tool_call

QWEN = "<tool_call>\n<function=write_file>\n<parameter=path>\npoem.txt\n</parameter>\n<parameter=content>\nThe river\nwinds\n</parameter>\n</function>\n</tool_call>"


# ---- フィルタ --------------------------------------------------------------

def test_filter_passthrough_without_markers():
    f = ToolCallStreamFilter()
    assert f.feed("hello ") + f.feed("world") + f.flush() == "hello world"
    assert f.raw == "" and f.calls == [] and not f.changed


def test_filter_strips_call_and_accumulates_raw_across_chunks():
    f = ToolCallStreamFilter()
    pieces = ["before <tool_", "call>\n<function=write_file>\n<parameter=content>\nab", "c\n</parameter>\n</func", "tion>\n</tool_call> after"]
    shown = "".join(f.feed(p) for p in pieces) + f.flush()
    assert shown == "before  after"
    assert f.calls == ["\n<function=write_file>\n<parameter=content>\nabc\n</parameter>\n</function>\n"]


def test_filter_reports_progress_and_close():
    f = ToolCallStreamFilter()
    f.feed("<tool_call>\n<function=f>\n<parameter=x>\n1")
    assert f.in_call and f.changed and not f.closed and f.raw.endswith("1")
    f.feed("2")
    assert f.changed and f.raw.endswith("12")
    f.feed("\n</parameter>\n</function>\n</tool_call>")
    assert f.closed and not f.in_call and len(f.calls) == 1


def test_filter_holds_partial_marker_then_releases_as_text():
    f = ToolCallStreamFilter()
    assert f.feed("a <tool") == "a "       # 開始マーカーの断片は保留
    assert f.feed("bar") == "<toolbar"     # マーカーでなかった → 本文として出す
    assert f.flush() == ""


def test_filter_gemma_markers():
    f = ToolCallStreamFilter()
    assert f.feed("x <|tool_call>{\"name\":\"f\"}<tool_call|> y") == "x  y"
    assert f.calls == ['{"name":"f"}']


def test_filter_unclosed_call_at_end_goes_to_raw_not_text():
    f = ToolCallStreamFilter()
    f.feed("<tool_call>\n<function=f>\n<parameter=c>\nhalf")
    assert f.flush() == "" and f.raw.endswith("half")


# ---- 途中解析 ---------------------------------------------------------------

def test_parse_partial_xml_name_complete_and_open_param():
    raw = "\n<function=write_file>\n<parameter=path>\npoem.txt\n</parameter>\n<parameter=content>\nThe river\nwinds"
    p = parse_partial_tool_call(raw)
    assert p["name"] == "write_file"
    assert p["arguments"] == {"path": "poem.txt"}
    assert p["partial"] == ("content", "The river\nwinds")


def test_parse_partial_xml_complete():
    p = parse_partial_tool_call(QWEN.replace("<tool_call>", "").replace("</tool_call>", ""))
    assert p["name"] == "write_file" and p["arguments"] == {"path": "poem.txt", "content": "The river\nwinds"}
    assert p["partial"] is None


def test_parse_partial_json_best_effort():
    p = parse_partial_tool_call('{"name": "write_file", "arguments": {"path": "a.txt", "content": "he')
    assert p["name"] == "write_file" and p["arguments_text"].startswith('{"path"')
    assert parse_partial_tool_call("<function=")["name"] is None


# ---- chat() 経路 ------------------------------------------------------------

def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _final_tool_chunk():
    tc = SimpleNamespace(index=0, id="c1", function=SimpleNamespace(
        name="write_file", arguments='{"path": "poem.txt", "content": "The river\\nwinds"}'))
    return _chunk(tool_calls=[tc])


def test_chat_strips_tool_text_from_content_and_reports_progress(fake_openai):
    llm = LLMClient(model="m")
    pieces = ["了解、書きます。", "<tool_call>\n<function=write_file>\n<parameter=path>\npoem.txt\n</parameter>\n<parameter=content>\nThe ri", "ver\nwinds\n</parameter>\n</function>\n</tool_call>"]
    llm.openai.chat.completions.create = lambda **k: iter([_chunk(p) for p in pieces] + [_final_tool_chunk()])
    shown, progress = [], []
    m = llm.chat([{"role": "user", "content": "x"}], [], shown.append,
                 on_tool_args=lambda raw, done: progress.append((raw, done)))
    assert "".join(shown).strip() == "了解、書きます。"          # 本文にツール本文は出ない
    assert "<tool_call>" not in (m.content or "")
    assert m.tool_calls and m.tool_calls[0].function.name == "write_file"  # 最終解析は従来どおり
    assert progress and progress[-1][1] is True                     # 閉じた合図
    assert any("The ri" in raw and not done for raw, done in progress)  # 途中経過が届く
    assert progress[-1][0].endswith("</function>\n")


def test_chat_without_markers_is_unchanged_and_no_progress(fake_openai):
    llm = LLMClient(model="m")
    llm.openai.chat.completions.create = lambda **k: iter([_chunk("できたよ"), _final_tool_chunk()])
    shown, progress = [], []
    m = llm.chat([{"role": "user", "content": "x"}], [], shown.append,
                 on_tool_args=lambda raw, done: progress.append((raw, done)))
    assert m.content == "できたよ" and progress == []
    assert m.tool_calls[0].function.name == "write_file"


def test_chat_default_signature_still_works(fake_openai):
    llm = LLMClient(model="m")
    llm.openai.chat.completions.create = lambda **k: iter([_chunk("hi")])
    assert llm.chat([{"role": "user", "content": "x"}], [], lambda t: None).content == "hi"
