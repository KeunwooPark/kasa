"""The directory: one `users.info` per person, and names in the transcript.

No `slack_bolt` here, and no import guard, because `identity.py` has neither.
The whole of what it decides — when a cached profile is still good, what a
mention renders as, what happens when Slack does not answer — is reachable
with a callable and a database.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from kasa.adapters.slack.identity import Directory, SlackUser, read_user, user_ref
from kasa.core.events import InboundEvent
from kasa.store import Store

TEAM = "T0TEAM"
JANE = "U0JANE"
RAJ = "U0RAJ"


class Fake:
    """A workspace directory, and a count of how often it was asked."""

    def __init__(self, users: dict[str, dict[str, Any]] | None = None) -> None:
        self.users = users if users is not None else {JANE: profile("jane")}
        self.calls: list[str] = []
        self.fails = False

    async def __call__(self, user_id: str) -> dict[str, Any]:
        self.calls.append(user_id)
        if self.fails:
            raise RuntimeError("slack is having a day")
        return self.users.get(user_id, {})


def profile(display: str, *, real: str = "", **fields: Any) -> dict[str, Any]:
    return {"name": display, "profile": {"display_name": display, "real_name": real}, **fields}


def directory(store: Store, lookup: Fake, *, ttl: timedelta | None = None) -> Directory:
    # `is not None`, because `timedelta(0)` is falsy and it is the TTL half
    # these tests care most about.
    return Directory(store, lookup, team_id=TEAM, ttl=timedelta(hours=24) if ttl is None else ttl)


def event(text: str, *, author: str = JANE) -> InboundEvent:
    return InboundEvent(
        source="slack",
        external_id=f"slack:{TEAM}:C0DEPLOY:1700000000.000100",
        session_id=f"slack:{TEAM}:C0DEPLOY:1700000000.000100",
        text=text,
        author=author,
        channel="C0DEPLOY",
    )


# -- resolving ----------------------------------------------------------------


async def test_a_mention_reaches_the_model_as_a_name(store: Store) -> None:
    lookup = Fake()

    hydrated = await directory(store, lookup).hydrate(event(f"ask <@{JANE}> about it"))

    assert hydrated.text == "ask @jane about it"


async def test_the_gap_around_a_mention_is_the_gap_it_had(store: Store) -> None:
    """`MENTION` eats the whitespace either side of the id, so a substitution
    that forgot to put it back would run the sentence together."""
    lookup = Fake()

    hydrated = await directory(store, lookup).hydrate(event(f"before <@{JANE}> after"))

    assert hydrated.text == "before @jane after"


async def test_a_profile_is_fetched_once_and_then_remembered(store: Store) -> None:
    lookup = Fake()
    resolver = directory(store, lookup)

    await resolver.hydrate(event(f"<@{JANE}> ping"))
    await resolver.hydrate(event(f"<@{JANE}> pong"))

    assert lookup.calls == [JANE], "the second turn should have cost nothing"


async def test_a_second_directory_reads_the_first_one_s_cache(store: Store) -> None:
    """The cache is the database, not the object: a restarted daemon must not
    re-fetch the whole workspace."""
    lookup = Fake()
    await directory(store, lookup).hydrate(event(f"<@{JANE}> ping"))

    await directory(store, lookup).hydrate(event(f"<@{JANE}> ping again"))

    assert lookup.calls == [JANE]


async def test_an_id_nobody_answers_for_is_left_exactly_as_it_was(store: Store) -> None:
    """An id that resolves to nothing is still a participant in the sentence.
    Rewriting it to a bare `@U0RAJ` would read as a name nobody can act on."""
    lookup = Fake(users={})

    hydrated = await directory(store, lookup).hydrate(event(f"what about <@{RAJ}>?"))

    assert hydrated.text == f"what about <@{RAJ}>?"


async def test_the_author_is_recorded_even_when_they_mention_nobody(store: Store) -> None:
    """A DM has no mentions in it at all, and its one participant is exactly
    the person the `people/` mapping exists for."""
    lookup = Fake()

    await directory(store, lookup).hydrate(event("morning", author=JANE))

    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None
    assert row["display_name"] == "jane"
    assert row["memory_id"] is None, "linking is the identity job's job, not this one's"


async def test_several_mentions_are_resolved_in_one_turn(store: Store) -> None:
    lookup = Fake({JANE: profile("jane"), RAJ: profile("raj")})

    hydrated = await directory(store, lookup).hydrate(event(f"<@{JANE}> and <@{RAJ}> agreed"))

    assert hydrated.text == "@jane and @raj agreed"


# -- names that change --------------------------------------------------------


async def test_a_rename_is_picked_up_once_the_cache_is_stale(store: Store) -> None:
    lookup = Fake()
    stale_at_once = directory(store, lookup, ttl=timedelta(0))

    await stale_at_once.hydrate(event(f"<@{JANE}> ping"))
    lookup.users[JANE] = profile("jane-doe")
    hydrated = await stale_at_once.hydrate(event(f"<@{JANE}> ping"))

    assert hydrated.text == "@jane-doe ping"
    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None and row["display_name"] == "jane-doe"


async def test_a_rename_does_not_disturb_an_existing_link(store: Store) -> None:
    """The row is what stops the same person becoming two `people/` memories,
    so a refresh may update the name and must never drop the mapping."""
    lookup = Fake()
    stale_at_once = directory(store, lookup, ttl=timedelta(0))
    await stale_at_once.hydrate(event(f"<@{JANE}> ping"))
    await store.link_slack_user(
        team_id=TEAM, user_id=JANE, memory_id="mem_01J0", memory_name="jane"
    )

    lookup.users[JANE] = profile("jane-doe")
    await stale_at_once.hydrate(event(f"<@{JANE}> ping"))

    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None
    assert row["memory_id"] == "mem_01J0", "still the same person"
    assert (row["display_name"], row["memory_name"]) == ("jane-doe", "jane"), "and now due a write"


# -- when Slack does not answer ----------------------------------------------


async def test_a_failed_lookup_falls_back_to_the_last_known_name(store: Store) -> None:
    lookup = Fake()
    stale_at_once = directory(store, lookup, ttl=timedelta(0))
    await stale_at_once.hydrate(event(f"<@{JANE}> ping"))

    lookup.fails = True
    hydrated = await stale_at_once.hydrate(event(f"<@{JANE}> ping"))

    assert hydrated.text == "@jane ping", "yesterday's name beats an id"


async def test_a_failed_first_lookup_caches_nothing(store: Store) -> None:
    """A gap must not be cached: the next message is the retry."""
    lookup = Fake()
    lookup.fails = True
    resolver = directory(store, lookup)

    hydrated = await resolver.hydrate(event(f"<@{JANE}> ping"))

    assert hydrated.text == f"<@{JANE}> ping"
    assert await store.get_slack_user(TEAM, JANE) is None


async def test_the_turn_survives_a_directory_that_is_broken_outright(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hydrate` sits between the queue and the agent. Raising here would fail
    the turn and have the whole message redelivered — for a rendering step."""

    async def explode(*_: object, **__: object) -> None:
        raise RuntimeError("the database is gone")

    monkeypatch.setattr(store, "get_slack_user", explode)
    original = event(f"<@{JANE}> ping")

    assert await directory(store, Fake()).hydrate(original) == original


# -- reading a profile --------------------------------------------------------


def test_a_display_name_is_preferred_to_a_real_one() -> None:
    user = read_user(JANE, profile("jane", real="Jane Q. Doe"))

    assert (user.display_name, user.real_name) == ("jane", "Jane Q. Doe")
    assert user.name == "jane"


def test_a_profile_with_no_display_name_falls_back_to_the_real_one() -> None:
    user = read_user(JANE, {"profile": {"real_name": "Jane Q. Doe"}})

    assert user.name == "Jane Q. Doe"


def test_a_profile_with_nothing_in_it_falls_back_to_the_id() -> None:
    assert read_user(JANE, {"profile": {}}).name == JANE


def test_a_payload_of_the_wrong_shape_is_read_rather_than_raised() -> None:
    """This runs on the turn path, where an unexpected shape has to degrade."""
    user = read_user(JANE, {"profile": "not an object", "name": 17})

    assert user == SlackUser(user_id=JANE)


def test_a_bot_is_recognized_as_one() -> None:
    assert read_user("B0APP", profile("deploybot", is_bot=True)).is_bot


def test_the_ref_is_the_shape_both_halves_agree_on() -> None:
    assert user_ref(TEAM, JANE) == f"slack://{TEAM}/{JANE}"
