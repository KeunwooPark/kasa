"""The packer trades a little unused context for never overflowing.

Every assertion here is about that trade: estimates may be loose, but they must
be loose in the safe direction.
"""

from __future__ import annotations

from kasa.llm.tokens import HeuristicTokenizer, count_message, count_messages, default_tokenizer
from kasa.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def test_empty_text_costs_nothing() -> None:
    assert HeuristicTokenizer().count("") == 0


def test_english_prose_lands_in_the_right_order_of_magnitude() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    count = HeuristicTokenizer().count(text)
    assert 150 < count < 350  # ~220 real tokens


def test_dense_scripts_are_not_under_counted() -> None:
    """CJK tokenizes near one token per character.

    Estimating it like Latin text would under-count by ~4x, and the first
    symptom would be a context overflow mid-conversation.
    """
    text = "메모리를 관리하는 서버"
    wide = sum(1 for ch in text if not ch.isascii())
    assert HeuristicTokenizer().count(text) >= wide


def test_counting_is_monotone_in_length() -> None:
    tok = HeuristicTokenizer()
    assert tok.count("abc") <= tok.count("abcdef") <= tok.count("abcdefghi")


def test_message_counting_includes_framing_overhead() -> None:
    tok = HeuristicTokenizer()
    assert count_message(Message.user(""), tok) > 0


def test_tool_blocks_are_counted() -> None:
    tok = HeuristicTokenizer()
    plain = count_message(Message.assistant("checking"), tok)
    with_call = count_message(
        Message(
            role="assistant",
            content=(
                TextBlock(text="checking"),
                ToolUseBlock(id="t1", name="get_weather", input={"city": "Seoul"}),
            ),
        ),
        tok,
    )
    assert with_call > plain


def test_tool_results_are_counted() -> None:
    tok = HeuristicTokenizer()
    message = Message.tool_results([ToolResultBlock(tool_use_id="t1", content="x" * 400)])
    assert count_message(message, tok) > 50


def test_totals_sum() -> None:
    tok = HeuristicTokenizer()
    messages = [Message.user("one"), Message.assistant("two")]
    assert count_messages(messages, tok) == sum(count_message(m, tok) for m in messages)


def test_default_tokenizer_always_returns_something() -> None:
    """A missing optional dependency must never be a startup failure."""
    assert default_tokenizer().count("hello") > 0
