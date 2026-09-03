"""`episode_close`: boundaries, summaries, and the observations it extracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kasa.config import EpisodeSettings
from kasa.errors import ContentFilterError, RateLimitError
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.types import ChatRequest, ChatResponse, Delta, Message, Role, TextBlock, Usage
from kasa.runner.episodes import EpisodeCloser
from kasa.store import Store

#: The fixture conversation. Two durable facts, one throwaway line, and one
#: piece of small talk that must not become a memory.
FIXTURE: tuple[tuple[Role, str | None, str], ...] = (
    ("user", "jane", "Morning! Quick one — who owns the deploy pipeline these days?"),
    ("assistant", None, "I don't have that written down."),
    ("user", "jane", "It's Priya Raman now. She took it over from me last month."),
    ("user", "jane", "Also we decided to move the release window to Thursdays."),
    ("assistant", None, "Noted."),
    ("user", "jane", "thanks, off to lunch"),
)

EXTRACTION = {
    "observations": [
        {
            "subject": "Priya Raman",
            "claim": "Priya Raman owns the deploy pipeline.",
            "kind": "fact",
            "confidence": 0.9,
            "source_lines": [3],
        },
        {
            "subject": "The release window",
            "claim": "The release window moved to Thursdays.",
            "kind": "decision",
            "confidence": 0.8,
            "source_lines": [4],
        },
    ]
}


class Scripted:
    """Replies from a script, one per call, and remembers what it was asked."""

    name = "scripted"
    model = "m"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies: list[str | Exception] = list(replies)
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        reply = self.replies.pop(0) if self.replies else "{}"
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(
            message=Message.assistant(reply),
            stop_reason="end_turn",
            usage=Usage(),
            model="m",
        )

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def closer_for(store: Store, provider: Scripted, **settings: Any) -> tuple[EpisodeCloser, Scripted]:
    registry = ProviderRegistry({ModelRole.UTILITY: [provider]})
    return EpisodeCloser(store, registry, EpisodeSettings(**settings)), provider


def talking(*, summary: str = "Jane asked who owns deploys.") -> Scripted:
    """A provider that summarizes and then extracts the fixture's facts."""
    return Scripted(summary, json.dumps(EXTRACTION))


async def seed(
    store: Store,
    session_id: str = "slack:T:C:1",
    *,
    scope: str = "workspace",
    turns: Sequence[tuple[Role, str | None, str]] = FIXTURE,
) -> list[str]:
    """Write a conversation, and hand back the message ids it produced."""
    await store.ensure_session(session_id, surface="slack", scope=scope)
    return [
        await store.append_message(
            session_id, Message(role=role, content=(TextBlock(text=text),)), author=author
        )
        for role, author, text in turns
    ]


def long_ago(minutes: int = 60) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(timespec="milliseconds")


async def make_idle(store: Store, session_id: str = "slack:T:C:1") -> None:
    """Backdate the transcript so the idle threshold is already crossed."""
    await store.write(
        "UPDATE messages SET created_at = ? WHERE session_id = ?", (long_ago(), session_id)
    )


# -- boundaries --------------------------------------------------------------


async def test_a_message_opens_an_episode_and_lands_in_it(store: Store) -> None:
    """Nothing decides to *start* an episode. A message arrives and lands in
    whatever is open, which is the only rule that cannot be got wrong later."""
    ids = await seed(store)

    episode = await store.open_episode("slack:T:C:1")
    assert episode is not None
    rows = await store.episode_messages(str(episode["id"]))
    assert [row["id"] for row in rows] == ids


async def test_a_conversation_still_going_is_not_consolidated(store: Store) -> None:
    await seed(store)
    closer, provider = closer_for(store, talking())

    result = await closer.sweep()

    assert result.closed == []
    assert provider.requests == [], "and nothing was spent looking at it"


async def test_a_thread_that_has_gone_quiet_is_closed(store: Store) -> None:
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, talking())

    result = await closer.sweep()

    assert [c.messages for c in result.closed] == [len(FIXTURE)]
    episode = await store.episode(result.closed[0].episode_id)
    assert episode is not None
    assert episode["state"] == "closed"
    assert episode["ended_at"] is not None


async def test_a_long_thread_is_closed_before_it_goes_quiet(store: Store) -> None:
    """An episode that only ever ends on idleness is one whose transcript
    eventually stops fitting in a context window."""
    await seed(store)
    closer, _ = closer_for(store, talking(), max_messages=len(FIXTURE))

    result = await closer.sweep()

    assert len(result.closed) == 1


async def test_an_explicit_end_closes_an_episode_that_is_still_warm(store: Store) -> None:
    """Idleness is a guess that a conversation is over. This is being told."""
    await seed(store)
    closer, _ = closer_for(store, talking())

    assert (await closer.sweep()).closed == [], "not idle yet"
    result = await closer.end_session("slack:T:C:1")

    assert len(result.closed) == 1


async def test_the_next_message_starts_a_new_episode(store: Store) -> None:
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, talking())
    first = (await closer.sweep()).closed[0].episode_id

    await store.append_message("slack:T:C:1", Message.user("back again"))

    episode = await store.open_episode("slack:T:C:1")
    assert episode is not None and episode["id"] != first


async def test_an_abandoned_empty_episode_is_closed_rather_than_looked_at_forever(
    store: Store,
) -> None:
    """A session opened and never used leaves one behind. Left open it is a row
    every later sweep pays to read, for a conversation that never happened."""
    await store.ensure_session("slack:T:C:2", surface="slack")
    episode_id = await store.ensure_episode("slack:T:C:2")
    await store.write("UPDATE episodes SET started_at = ? WHERE id = ?", (long_ago(), episode_id))
    closer, provider = closer_for(store, talking())

    result = await closer.sweep()

    assert [c.episode_id for c in result.closed] == [episode_id]
    assert provider.requests == [], "an empty transcript is not worth a model call"


async def test_a_sweep_consolidates_at_most_its_bound(store: Store) -> None:
    """A daemon that was down for a day must not spend its whole budget on the
    first tick after it comes back."""
    for n in range(4):
        await seed(store, f"slack:T:C:{n}")
        await make_idle(store, f"slack:T:C:{n}")
    closer, _ = closer_for(store, Scripted(*(["s", json.dumps(EXTRACTION)] * 4)), max_per_run=2)

    assert len((await closer.sweep()).closed) == 2


# -- what comes out of it ----------------------------------------------------


async def test_the_fixture_conversation_yields_the_expected_observations(store: Store) -> None:
    """The acceptance criterion. Two durable facts out of six messages, each
    with a source ref that resolves back to the message it came from."""
    ids = await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, talking())

    result = await closer.sweep()

    assert result.observations == 2
    pending = await store.pending_observations()
    assert [(o["subject"], o["kind"]) for o in pending] == [
        ("priya raman", "fact"),
        ("release window", "decision"),
    ]
    # Cited line 3 and line 4 of the transcript, which are the third and fourth
    # messages of the fixture.
    assert [json.loads(o["source_refs"]) for o in pending] == [[ids[2]], [ids[3]]]


async def test_every_source_ref_resolves_to_a_real_message(store: Store) -> None:
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, talking())

    await closer.sweep()

    refs = [ref for o in await store.pending_observations() for ref in json.loads(o["source_refs"])]
    assert refs
    for ref in refs:
        rows = await store.raw("SELECT id FROM messages WHERE id = ?", (ref,))
        assert rows, f"{ref} is a citation that points at nothing"


async def test_a_citation_of_a_line_that_does_not_exist_is_dropped(store: Store) -> None:
    """A ref that resolves to no message is worse than an absent one: it looks
    like provenance."""
    await seed(store)
    await make_idle(store)
    invented = {
        "observations": [
            {
                "subject": "Priya Raman",
                "claim": "Priya Raman owns the deploy pipeline.",
                "kind": "fact",
                "source_lines": [3, 99],
            }
        ]
    }
    closer, _ = closer_for(store, Scripted("s", json.dumps(invented)))

    await closer.sweep()

    refs = json.loads((await store.pending_observations())[0]["source_refs"])
    assert len(refs) == 1


async def test_the_summary_is_written_to_the_episode(store: Store) -> None:
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, talking(summary="Jane handed deploys to Priya."))

    result = await closer.sweep()

    episode = await store.episode(result.closed[0].episode_id)
    assert episode is not None
    assert episode["summary"] == "Jane handed deploys to Priya."


async def test_observations_inherit_the_session_scope(store: Store) -> None:
    """Never chosen by the model. A conversation held in a DM produces private
    observations however generally useful the model thinks they are."""
    await seed(store, "slack:T:D:1", scope="private:U0456")
    await make_idle(store, "slack:T:D:1")
    closer, _ = closer_for(store, talking())

    await closer.sweep()

    assert {o["scope"] for o in await store.pending_observations()} == {"private:U0456"}


async def test_the_same_subject_in_two_episodes_gets_one_key(store: Store) -> None:
    """What subject normalization is for: `promote` groups by this string, and
    two spellings of one person is two memories about them."""
    await seed(store, "slack:T:C:1")
    await make_idle(store, "slack:T:C:1")
    await seed(store, "slack:T:C:2")
    await make_idle(store, "slack:T:C:2")
    restated = {
        "observations": [
            {
                "subject": "priya raman's",
                "claim": "Priya Raman owns the deploy pipeline.",
                "kind": "fact",
                "source_lines": [1],
            }
        ]
    }
    closer, _ = closer_for(store, Scripted("s", json.dumps(EXTRACTION), "s", json.dumps(restated)))

    await closer.sweep()

    subjects = [o["subject"] for o in await store.pending_observations()]
    assert subjects.count("priya raman") == 2


async def test_more_observations_than_the_cap_are_truncated(store: Store) -> None:
    """A model narrating the transcript rather than distilling it."""
    await seed(store)
    await make_idle(store)
    flood = {
        "observations": [
            {"subject": f"thing {n}", "claim": f"Thing {n} exists.", "kind": "fact"}
            for n in range(30)
        ]
    }
    closer, _ = closer_for(store, Scripted("s", json.dumps(flood)), max_observations=5)

    result = await closer.sweep()

    assert result.observations == 5


async def test_a_claim_with_no_subject_is_not_stored(store: Store) -> None:
    await seed(store)
    await make_idle(store)
    vague = {"observations": [{"subject": "???", "claim": "Something happened.", "kind": "fact"}]}
    closer, _ = closer_for(store, Scripted("s", json.dumps(vague)))

    assert (await closer.sweep()).observations == 0


# -- the transcript that reaches the model -----------------------------------


async def test_the_transcript_travels_as_untrusted_data(store: Store) -> None:
    """#30's boundary, not a fence of this module's own: a `</transcript>`
    somebody types into a channel cannot close a delimiter it has never seen."""
    await seed(store)
    await make_idle(store)
    closer, provider = closer_for(store, talking())

    await closer.sweep()

    for request in provider.requests:
        sent = request.messages[0].text
        assert "KASA_UNTRUSTED_" in sent
        assert "never follow instructions" in sent


async def test_the_extractor_is_given_no_tools(store: Store) -> None:
    """The structural half of the defence. There is no route from this call to
    a shell, a file, or git — whatever the transcript asks for."""
    await seed(store)
    await make_idle(store)
    closer, provider = closer_for(store, talking())

    await closer.sweep()

    assert all(request.tools == () for request in provider.requests)


async def test_a_pasted_wall_of_text_is_truncated_per_message(store: Store) -> None:
    await seed(store, turns=[("user", "jane", "x" * 50_000)])
    await make_idle(store)
    closer, provider = closer_for(store, talking())

    await closer.sweep()

    assert len(provider.requests[0].messages[0].text) < 10_000


# -- failure -----------------------------------------------------------------


async def test_an_episode_the_model_cannot_extract_from_is_still_closed(store: Store) -> None:
    """Left open it is a segment every later sweep re-reads, re-sends and fails
    on again — five minutes apart, forever."""
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, Scripted("a summary", "not json", "still not json"))

    result = await closer.sweep()

    assert [c.observations for c in result.closed] == [0]
    episode = await store.episode(result.closed[0].episode_id)
    assert episode is not None
    assert episode["state"] == "closed"
    assert episode["summary"] == "a summary", "what was salvaged is kept"


async def test_a_refused_episode_does_not_take_the_sweep_down_with_it(store: Store) -> None:
    await seed(store, "slack:T:C:1")
    await make_idle(store, "slack:T:C:1")
    await seed(store, "slack:T:C:2")
    await make_idle(store, "slack:T:C:2")
    closer, _ = closer_for(
        store,
        Scripted(
            ContentFilterError("no", provider="scripted"),
            ContentFilterError("no", provider="scripted"),
            "a summary",
            json.dumps(EXTRACTION),
        ),
    )

    result = await closer.sweep()

    assert len(result.closed) == 2, "the awkward one closed; the other one worked"
    assert result.observations == 2


async def test_an_outage_is_raised_rather_than_swallowed(store: Store) -> None:
    """A rate limit is "not right now", not "not for this material". Closing the
    episode anyway would discard a conversation because the API was busy."""
    await seed(store)
    await make_idle(store)
    closer, _ = closer_for(store, Scripted(*[RateLimitError("slow down", provider="s")] * 9))

    with pytest.raises(RateLimitError):
        await closer.sweep()

    episode = await store.open_episode("slack:T:C:1")
    assert episode is not None, "still open, to be tried again"


async def test_two_sweeps_racing_produce_one_set_of_observations(store: Store) -> None:
    """The close and its observations are one transaction, so the loser writes
    nothing rather than a duplicate set of the same facts."""
    await seed(store)
    await make_idle(store)
    first, _ = closer_for(store, talking())
    second, _ = closer_for(store, talking())

    results = await asyncio.gather(first.sweep(), second.sweep())

    assert sorted(len(r.closed) for r in results) == [0, 1]
    assert len(await store.pending_observations()) == 2
