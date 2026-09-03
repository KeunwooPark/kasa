"""Asking a model for a typed answer, and refusing to believe it until it is one.

Every consolidation job needs the same thing: a model call whose result is a
validated Pydantic object rather than prose somebody has to parse. The schema
goes in the system prompt — stable across calls, so it stays cacheable — and
the material goes in the user turn.

Two things this deliberately does not do. It does not use any provider's
native JSON mode: those are spelled differently on each side of
`kasa/llm/*_compat.py`, and a provider's representation leaking past that
boundary is the one thing `kasa/llm/types.py` exists to prevent. And it does
not trust the result. `parse_json_object` is lenient about what surrounds the
JSON — models fence it, preface it, apologize after it — and the Pydantic model
is strict about what is inside it, which is the split that matters: the
leniency is about formatting, never about content.

One repair attempt, and only one. A model that cannot produce the shape twice
is not going to produce it on the third try, and the caller has a decision to
make about the episode rather than a retry to spend.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from kasa.errors import KasaError
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.types import ChatRequest, Message

log = logging.getLogger(__name__)

#: How much of an unusable reply goes in the error. Enough to see what the
#: model did instead; not the whole thing, because this is logged.
SNIPPET = 400

_SCHEMA_INSTRUCTION = """Reply with a single JSON object and nothing else. No
prose before it, no explanation after it, no Markdown code fence around it.

It must validate against this JSON Schema:

{schema}"""


class StructuredOutputError(KasaError):
    """The model did not return the shape it was asked for."""


def parse_json_object(text: str) -> Any:
    """The JSON object in a reply, however the model chose to wrap it.

    Tries the whole string first, so a well-behaved reply costs one parse. Only
    then goes looking for the outermost braces, which is what recovers ```json
    fences, a leading "Here is the JSON:", and a trailing paragraph of
    commentary.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        raise StructuredOutputError(f"no JSON object in the reply: {stripped[:SNIPPET]!r}")
    try:
        return json.loads(stripped[start : end + 1])
    except ValueError as exc:
        raise StructuredOutputError(
            f"the reply was not valid JSON ({exc}): {stripped[:SNIPPET]!r}"
        ) from exc


async def complete_json[T: BaseModel](
    registry: ProviderRegistry,
    role: ModelRole,
    schema: type[T],
    *,
    system: str,
    prompt: str,
    tag: str,
    max_tokens: int = 2048,
) -> T:
    """Call `role` and return its reply as a validated `schema`.

    Raises `StructuredOutputError` if it never becomes one. That is a statement
    about this request, not about the provider — the caller should decide what
    to do with the material, not retry the infrastructure.
    """
    request = ChatRequest(
        messages=(Message.user(prompt),),
        system=f"{system}\n\n{_SCHEMA_INSTRUCTION.format(schema=_schema_text(schema))}",
        max_tokens=max_tokens,
        # Extraction and classification are not places to be creative, and a
        # deterministic reply is one a rejected plan can be reproduced from.
        temperature=0.0,
    )

    response = await registry.complete(role, request, tag=tag)
    try:
        return schema.model_validate(parse_json_object(response.text))
    except (StructuredOutputError, ValidationError) as first:
        log.info("%s: the reply did not validate, asking once more (%s)", tag, _brief(first))

    repair = request.model_copy(
        update={
            "messages": (
                Message.user(prompt),
                Message.assistant(response.text or "(nothing)"),
                Message.user(
                    "That did not validate against the schema. Reply again with the "
                    "JSON object alone — no fence, no commentary."
                ),
            )
        }
    )
    second = await registry.complete(role, repair, tag=f"{tag}.repair")
    try:
        return schema.model_validate(parse_json_object(second.text))
    except (StructuredOutputError, ValidationError) as exc:
        raise StructuredOutputError(
            f"{tag}: the model did not return a valid {schema.__name__} in two attempts "
            f"({_brief(exc)}); last reply was {second.text[:SNIPPET]!r}"
        ) from exc


def _schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)


def _brief(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
