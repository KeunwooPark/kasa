"""Token counting for the context packer.

The packer needs an estimate that is *conservative* — over-counting costs a
little unused context, under-counting costs a `ContextOverflowError` mid-turn.
Every heuristic here therefore rounds against us.
"""

from __future__ import annotations

from typing import Protocol

from kasa.llm.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock

#: Rough per-message framing cost (role markers, delimiters). Both provider
#: families add something in this range.
MESSAGE_OVERHEAD_TOKENS = 4

#: Tool definitions are serialized into the prompt; their JSON schema costs
#: roughly this much per character.
_ASCII_TOKENS_PER_CHAR = 0.25

#: Non-ASCII text (CJK especially) tokenizes far denser than Latin script.
#: One token per character over-estimates for accented Latin and under-estimates
#: for nothing, which is the direction we want.
_WIDE_TOKENS_PER_CHAR = 1.0


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """Dependency-free estimator. The default.

    Accurate to roughly ±20% on English prose and code, and deliberately
    pessimistic on non-Latin scripts.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for ch in text if ch.isascii())
        wide_chars = len(text) - ascii_chars
        estimate = ascii_chars * _ASCII_TOKENS_PER_CHAR + wide_chars * _WIDE_TOKENS_PER_CHAR
        return max(1, int(estimate + 0.5))


class TiktokenTokenizer:
    """Exact counts for OpenAI models, when the optional extra is installed."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        import tiktoken  # imported lazily; the dependency is optional

        self.name = f"tiktoken:{encoding_name}"
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


def default_tokenizer() -> Tokenizer:
    """Best available tokenizer.

    Falls back silently: an exact count is nice, but the packer is designed to
    be correct with an estimate, so a missing optional dependency must never be
    a startup failure.
    """
    try:
        return TiktokenTokenizer()
    except Exception:
        return HeuristicTokenizer()


def count_message(msg: Message, tokenizer: Tokenizer) -> int:
    total = MESSAGE_OVERHEAD_TOKENS
    for block in msg.content:
        match block:
            case TextBlock():
                total += tokenizer.count(block.text)
            case ThinkingBlock():
                total += tokenizer.count(block.thinking)
            case ToolUseBlock():
                total += tokenizer.count(block.name) + tokenizer.count(repr(block.input))
            case ToolResultBlock():
                total += tokenizer.count(block.content)
    return total


def count_messages(messages: tuple[Message, ...] | list[Message], tokenizer: Tokenizer) -> int:
    return sum(count_message(m, tokenizer) for m in messages)
