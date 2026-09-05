"""One Slack message, rewritten while the turn that produces it runs.

A long turn is indistinguishable from a broken one. The thread sits there, and
the person who asked either waits or asks again — and asking again is a second
turn, a second model call, and two answers to one question. A message that
visibly changes is the cheapest fix there is.

The rate at which it changes is the whole design. Slack allows roughly one
`chat.update` per second per channel; a token-by-token rewrite would be
hundreds, so every one after the first would be refused, the client would be
backing off during the part of the turn that matters, and the message would
flicker in every reader's client on the way there. So deltas never call Slack.
They land in a buffer, and a painter task redraws at a fixed interval — the
cost of a turn is the same whether the model emits four tokens or four thousand.

When Slack refuses anyway, the answer is to stop redrawing rather than to try
harder: intermediate frames are a nicety and the final one is the answer, so
under rate-limit pressure this degrades to exactly what it replaced, a single
message with the answer in it.

Frames are droppable but not reorderable. Only one write is ever in flight, and
the answer is written after the last frame has landed rather than on top of one
still on its way — see `_stop_painting` for why cancelling is not enough (#192).

No `slack_sdk` import. What it needs is two calls — post one message, rewrite
one message — and taking them as a protocol is what lets the throttling and
the degradation be tested without a socket or the `slack` extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from kasa.errors import KasaError
from kasa.llm.types import Delta, MessageStop, TextDelta, ToolUseStart

log = logging.getLogger(__name__)

#: How often the message is redrawn while the turn runs. Slack's own guidance
#: for `chat.update` is about one per second per channel, and the design doc
#: asks for "roughly 1/sec — never per token" (§10.1).
DEFAULT_INTERVAL = 1.0

#: How many refusals before intermediate frames are abandoned for this turn.
#: One is a burst — several turns in one channel painting on the same second —
#: and worth riding out. A second says the channel is genuinely over budget,
#: and the useful thing to protect at that point is the final update.
MAX_REFUSALS = 2

#: Attempts at the *final* update, which is the answer rather than a frame. It
#: is worth waiting out a refusal for, and not worth failing the turn over
#: until it has been tried: the alternative is re-running the model call.
FINAL_ATTEMPTS = 3

#: What the message says before there is anything to say. Slack will not post
#: an empty message, and this is the frame most likely to be seen — it goes up
#: before the model has produced a single token.
THINKING = "_thinking…_"


class SlackRateLimited(KasaError):
    """Slack refused a write and said when to come back."""

    def __init__(self, retry_after: float = 1.0) -> None:
        super().__init__(f"rate limited by Slack; retry after {retry_after}s")
        self.retry_after = retry_after


class Poster(Protocol):
    """The two writes a live reply needs, and nothing else."""

    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str:
        """Post a message and return its `ts`."""
        ...

    async def update(self, *, channel: str, ts: str, text: str) -> None: ...


class LiveMessage:
    """A reply that is posted once and rewritten until the turn is done."""

    def __init__(
        self,
        poster: Poster,
        *,
        channel: str,
        thread_ts: str | None,
        interval: float = DEFAULT_INTERVAL,
        ts: str | None = None,
    ) -> None:
        self._poster = poster
        self._channel = channel
        self._thread_ts = thread_ts
        self._interval = interval
        #: A `ts` handed in is a message from an earlier attempt at this same
        #: turn, which is reused rather than added to. Without it a turn that
        #: failed and was redelivered leaves a "thinking…" message behind in
        #: the thread for every attempt it took.
        self._ts = ts
        self._painter: asyncio.Task[None] | None = None
        self._refusals = 0
        #: Held for the duration of one write, so a frame and the answer can
        #: never be at Slack at the same time.
        self._writing = asyncio.Lock()
        #: Set once the answer is what should be written. A frame that was
        #: waiting for the lock when it went up is dropped rather than sent.
        self._sealed = False

        self._text: list[str] = []
        self._tools: list[str] = []
        #: Set at the end of one model call, cleared by the first delta of the
        #: next. It is what tells "these tools are running" from "these tools
        #: ran, and the model is now answering" — there is no delta for a tool
        #: result coming back, so the next message starting is the signal.
        self._between = False
        self._painted: str | None = None

    @property
    def ts(self) -> str | None:
        """The message being rewritten, once there is one."""
        return self._ts

    @property
    def live(self) -> bool:
        """Whether anything is being rewritten at all."""
        return self._ts is not None

    async def open(self) -> None:
        """Put the placeholder up and start redrawing it.

        Failure is not fatal and is not raised: a turn whose placeholder could
        not be posted still has an answer to give, and `finish` will post it as
        one message. Slack being unwritable for real surfaces there instead,
        where it is the answer that is lost and the turn is worth retrying.
        """
        if self._ts is None:
            try:
                self._ts = await self._poster.post(
                    channel=self._channel, thread_ts=self._thread_ts, text=THINKING
                )
                self._painted = THINKING
            except Exception as exc:
                log.warning("could not open a live reply in %s: %s", self._channel, exc)
                return
        self._painter = asyncio.create_task(self._paint())

    async def delta(self, delta: Delta) -> None:
        """Take one delta. Never calls Slack, and never blocks on anything."""
        match delta:
            case TextDelta():
                self._turn_over()
                self._text.append(delta.text)
            case ToolUseStart():
                self._turn_over()
                self._tools.append(delta.name)
            case MessageStop():
                self._between = True
            case _:
                pass

    async def finish(self, text: str) -> None:
        """Stop redrawing and write the answer, whatever happened before it."""
        await self._stop_painting()
        final = text.strip()
        if not final:
            # `AgentResult.note` fills this in for every stop reason there is,
            # so an empty answer here means something upstream changed. The
            # placeholder must still be replaced: a message reading "thinking…"
            # forever is worse than one saying nothing came back.
            final = "_the turn produced no answer._"
        if self._ts is None:
            await self._poster.post(channel=self._channel, thread_ts=self._thread_ts, text=final)
            return
        await self._write_final(final)

    async def aclose(self) -> None:
        """Give up on the painter without writing anything.

        For a turn that raised: the queue will run it again, and this message
        is handed to that attempt rather than abandoned mid-sentence.
        """
        await self._stop_painting()

    # -- internals -----------------------------------------------------------

    def _turn_over(self) -> None:
        if self._between:
            self._between = False
            self._tools.clear()

    def render(self) -> str:
        """What the message says right now.

        Text accumulates across every model call in the turn, tool preamble
        included — that is what the person watching actually saw being written.
        The final update replaces all of it with the answer, which is the last
        call's text and the only part that was ever going to be kept.
        """
        parts = []
        if text := "".join(self._text).strip():
            parts.append(text)
        if self._tools:
            running = ", ".join(f"`{name}`" for name in dict.fromkeys(self._tools))
            parts.append(f"_running {running}…_")
        return "\n\n".join(parts) or THINKING

    async def _paint(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            frame = self.render()
            if frame == self._painted or self._ts is None:
                continue
            try:
                async with self._writing:
                    if self._sealed:
                        return
                    await self._poster.update(channel=self._channel, ts=self._ts, text=frame)
            except SlackRateLimited as limit:
                self._refusals += 1
                if self._refusals >= MAX_REFUSALS:
                    log.info("slack is rate limiting %s; showing the answer only", self._channel)
                    return
                await asyncio.sleep(limit.retry_after)
            except Exception as exc:
                # One frame, not the answer. The next tick redraws whatever the
                # message says by then, and `finish` is the write that matters.
                log.debug("a live frame did not land in %s: %s", self._channel, exc)
            else:
                self._painted = frame

    async def _stop_painting(self) -> None:
        """Stop redrawing, without interrupting a redraw already under way.

        Cancelling a painter that is mid-`chat.update` abandons this side of a
        write Slack has already received. Slack still applies it, and nothing
        then orders it against the answer that goes out a moment later — so the
        frame can land last and leave the thread holding a mid-sentence prefix
        of a reply that was delivered in full (#192).

        Taking the lock first is what rules that out: an in-flight frame is
        waited out, the message is sealed against any further one, and only
        then is the painter cancelled. It costs nothing in the ordinary case,
        where the painter is asleep between frames and the lock is free.
        """
        painter, self._painter = self._painter, None
        async with self._writing:
            self._sealed = True
            if painter is not None:
                painter.cancel()
        if painter is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await painter

    async def _write_final(self, final: str) -> None:
        assert self._ts is not None
        for attempt in range(1, FINAL_ATTEMPTS + 1):
            try:
                # Uncontended by now — `_stop_painting` sealed the message —
                # but taken all the same, so "one write at a time" is a
                # property of the class rather than of the order it is called in.
                async with self._writing:
                    await self._poster.update(channel=self._channel, ts=self._ts, text=final)
            except SlackRateLimited as limit:
                if attempt == FINAL_ATTEMPTS:
                    raise
                await asyncio.sleep(limit.retry_after)
            else:
                self._painted = final
                return
