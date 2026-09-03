from __future__ import annotations

import io
import logging

import pytest

from kasa.config import Config
from kasa.core.tools import Tool, ToolContext, ToolRegistry
from kasa.llm.types import ToolUseBlock
from kasa.redact import MIN_SECRET_LENGTH, Redactor


def test_known_secrets_are_replaced_with_their_variable_name() -> None:
    redactor = Redactor({"ANTHROPIC_API_KEY": "sk-ant-abcdefghijklmnop"})
    assert redactor.scrub("the key is sk-ant-abcdefghijklmnop ok") == (
        "the key is [redacted:ANTHROPIC_API_KEY] ok"
    )


def test_short_values_are_left_alone() -> None:
    """A one-character test key would otherwise redact half the transcript."""
    redactor = Redactor({"KEY": "k"})
    assert redactor.scrub("a knapsack of knowledge") == "a knapsack of knowledge"
    assert len("k") < MIN_SECRET_LENGTH


@pytest.mark.parametrize(
    "text",
    [
        "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA",
        "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "github_pat_AAAAAAAAAAAAAAAAAAAAAAAA",
        "xoxb-1234567890-abcdefghij",
        "xapp-1-A0123456789-abcdef",
        "AKIAIOSFODNN7EXAMPLE",
        "ASIAY34FZKBOKMSXQWER",
    ],
)
def test_token_shapes_are_caught_even_when_unknown(text: str) -> None:
    """A key Kasa was never told about, pasted into a message or echoed by a tool."""
    scrubbed = Redactor().scrub(f"here: {text} <-")
    assert text not in scrubbed
    assert "[redacted]" in scrubbed


def test_credentials_in_a_url_are_stripped() -> None:
    scrubbed = Redactor().scrub("remote https://user:ghp_tokenvalue@github.com/a/b.git failed")
    assert "ghp_tokenvalue" not in scrubbed
    assert "github.com/a/b.git" in scrubbed, "the useful half of the message survives"


def test_iam_principal_ids_are_left_alone() -> None:
    """They share the AWS key shape but are identifiers, not credentials — and
    they are the thing an IAM question is about."""
    text = "role AROAEXAMPLEID1234567 denied s3:GetObject to user AIDAEXAMPLEID1234567"
    assert Redactor().scrub(text) == text


PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpAIBAAKCAQEA1n2Xa9wZk3v4Qb8sT0pLmNoPqRsTuVwXyZ0123456789abcd\n"
    "efGHIJKLmnopQRSTuvwxYZ0123456789+/abcdefghijklmnopqrstuvwxyz012=\n"
    "-----END RSA PRIVATE KEY-----"
)


def test_a_private_key_block_is_replaced_whole() -> None:
    """The only multi-line pattern: the secret is the body, not the header."""
    scrubbed = Redactor().scrub(f"the deploy key is\n{PEM}\nput it on the runner")

    assert "MIIEpAIB" not in scrubbed
    assert "PRIVATE KEY" not in scrubbed
    assert scrubbed == "the deploy key is\n[redacted]\nput it on the runner"


@pytest.mark.parametrize(
    "header",
    [
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----",
    ],
)
def test_every_pem_flavour_is_covered(header: str) -> None:
    end = header.replace("BEGIN", "END")
    assert Redactor().scrub(f"{header}\nMIIEpAIBAAKCAQEA\n{end}") == "[redacted]"


def test_a_key_whose_end_marker_was_cut_off_is_still_replaced() -> None:
    """A log truncation or a partial paste leaves the body just as exposed."""
    scrubbed = Redactor().scrub(f"log tail:\n{PEM.split('-----END')[0]}(output truncated)")

    assert "MIIEpAIB" not in scrubbed
    assert "PRIVATE KEY" not in scrubbed
    assert scrubbed.endswith("(output truncated)"), "the rest of the log survives"


def test_ordinary_text_is_untouched() -> None:
    text = "The deploy pipeline is owned by the infra team. See projects/deploy.md."
    assert Redactor().scrub(text) == text


def test_a_secret_containing_another_is_fully_replaced() -> None:
    redactor = Redactor({"OUTER": "abcdefghijkl-innerlongvalue", "INNER": "innerlongvalue"})
    assert redactor.scrub("abcdefghijkl-innerlongvalue") == "[redacted:OUTER]"


def test_from_config_reads_every_referenced_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_CHAT_KEY", "sk-chatsecretvalue123")
    monkeypatch.setenv("MY_GH_TOKEN", "gh-tokensecretvalue123")
    cfg = Config.model_validate(
        {
            "ltm": {"repo": "a/b", "token_env": "MY_GH_TOKEN"},
            "llm": {"chat": {"kind": "openai", "model": "m", "key_env": "MY_CHAT_KEY"}},
        }
    )
    scrubbed = Redactor.from_config(cfg).scrub("sk-chatsecretvalue123 gh-tokensecretvalue123")

    assert "secretvalue" not in scrubbed
    assert "[redacted:MY_CHAT_KEY]" in scrubbed
    assert "[redacted:MY_GH_TOKEN]" in scrubbed


def test_fallback_provider_keys_are_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_KEY", "sk-fallbacksecretvalue")
    cfg = Config.model_validate(
        {
            "llm": {
                "chat": {
                    "kind": "anthropic",
                    "model": "m",
                    "fallbacks": [{"kind": "openai", "model": "n", "key_env": "FALLBACK_KEY"}],
                }
            }
        }
    )
    assert "fallbacksecret" not in Redactor.from_config(cfg).scrub("sk-fallbacksecretvalue")


# -- the two places it is installed -------------------------------------------


def test_log_records_are_scrubbed() -> None:
    stream = io.StringIO()
    parent = logging.getLogger("kasa.test_redact")
    parent.addHandler(logging.StreamHandler(stream))
    parent.setLevel(logging.WARNING)
    try:
        Redactor({"TOKEN": "ghp_averylongtokenvalue"}).install(parent)
        # Logged on a *child*, the way every module in Kasa logs. A filter on
        # the parent logger would never see this record.
        logging.getLogger("kasa.test_redact.child").warning(
            "push failed with %s", "ghp_averylongtokenvalue"
        )
    finally:
        parent.handlers.clear()

    written = stream.getvalue()
    assert "ghp_averylongtokenvalue" not in written
    assert "[redacted:TOKEN]" in written


async def test_tool_results_are_scrubbed_before_reaching_the_model() -> None:
    async def leaky(args: dict[str, object], context: ToolContext) -> str:
        return "fetched with token ghp_averylongtokenvalue000"

    registry = ToolRegistry(
        [
            Tool(
                name="leak",
                description="",
                input_schema={"type": "object", "properties": {}},
                handler=leaky,
            )
        ],
        scrub=Redactor({"TOKEN": "ghp_averylongtokenvalue000"}).scrub,
    )
    result = await registry.dispatch(ToolUseBlock(id="t1", name="leak", input={}))

    assert "ghp_averylongtokenvalue000" not in result.content
    assert "[redacted:TOKEN]" in result.content


async def test_tool_error_messages_are_scrubbed_too() -> None:
    async def explode(args: dict[str, object], context: ToolContext) -> str:
        raise RuntimeError("bad remote https://x:ghp_averylongtokenvalue@github.com/a/b")

    registry = ToolRegistry(
        [
            Tool(
                name="boom",
                description="",
                input_schema={"type": "object", "properties": {}},
                handler=explode,
            )
        ],
        scrub=Redactor().scrub,
    )
    result = await registry.dispatch(ToolUseBlock(id="t1", name="boom", input={}))

    assert result.is_error
    assert "ghp_averylongtokenvalue" not in result.content


async def test_dispatch_without_a_scrubber_is_unchanged() -> None:
    async def plain(args: dict[str, object], context: ToolContext) -> str:
        return "just text"

    registry = ToolRegistry(
        [
            Tool(
                name="plain",
                description="",
                input_schema={"type": "object", "properties": {}},
                handler=plain,
            )
        ]
    )
    result = await registry.dispatch(ToolUseBlock(id="t1", name="plain", input={}))
    assert result.content == "just text"
