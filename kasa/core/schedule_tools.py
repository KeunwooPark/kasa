"""Setting up a standing task by asking for one.

#179 gave Kasa a table of standing tasks and a CLI to manage it. That is the
wrong surface for the person it is for: somebody in a Slack thread who says
"do this every weekday morning" should not be told to open a terminal and
write five cron fields.

The translation is the one part of this a model is genuinely good at — "every
weekday at 9am Seoul time" into `0 9 * * 1-5` and `Asia/Seoul` — so the model
does it and the tool validates rather than trusts. `Cron.parse` with the zone,
the interval floor, the per-owner cap: every refusal comes back as a tool error
the model can read and correct, which is what `ToolRegistry` already gives
these for free.

Two constraints shape all three, and they are the same ones the memory tools
work under.

**The session supplies the destination, never the model.** `session_id`,
`channel`, `reply_to`, `scope` and `owner` come off `ToolContext`, which the
turn built from the event. There is no argument for where a task posts, so no
instruction smuggled into text Kasa read can create one that posts anywhere but
the thread it was created in. A task created in a DM stays in the DM (§11.1).

**Listing and cancelling are scoped to the calling session**, for the same
reason: text arriving in one channel must not be able to enumerate or delete
another channel's schedules (§7.1). The narrowing is in the query, not applied
to the results — a tool that read every row and then dropped the ones it should
not show has already had them.

What `schedule_create` returns is the next three fire times, rendered in the
task's own zone. That is the whole confirmation story: nobody can check
`0 9 * * 1-5`, and anybody can check "Mon 08 Sep 09:00 Asia/Seoul".
"""

from __future__ import annotations

import logging
from typing import Any

from kasa.core.tools import Tool, ToolContext
from kasa.runner.cron import CronError
from kasa.runner.tasks import ACTIVE, Task, TaskError, Tasks, render_fires

log = logging.getLogger(__name__)

#: How many fire times to read back. Three is enough to show the shape of a
#: weekly or a weekday schedule — one is not: "Mon 08 Sep 09:00" alone is
#: equally consistent with every day, every weekday and every Monday.
CONFIRM_FIRES = 3

#: A task with no owner is one nothing can cap, cancel or notify. Only reachable
#: from a surface that has no notion of who is speaking, which today is the
#: terminal — and the terminal has `kasa task add`.
NO_OWNER = (
    "This conversation has no user identity, so it cannot own a schedule. "
    "Standing tasks are set up from a chat surface, or with `kasa task add`."
)


def schedule_tools(tasks: Tasks) -> list[Tool]:
    """The three scheduling tools, bound to one tasks table."""
    return [_create_tool(tasks), _list_tool(tasks), _cancel_tool(tasks)]


# -- schedule_create ---------------------------------------------------------


def _create_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        try:
            task = await tasks.create(
                owner=context.author,
                # Every one of these is the session's. None is an argument.
                surface=_surface_of(context.session_id),
                session_id=context.session_id,
                channel=context.channel,
                reply_to=context.reply_to,
                scope=context.scope,
                prompt=str(args["prompt"]),
                cron=str(args["cron"]),
                timezone=_timezone(args),
                fire_once=bool(args.get("fire_once", False)),
            )
        except TaskError as exc:
            # A refusal, not a crash: the model wrote the expression, and it is
            # what can correct it. The floor and the cap both name their number
            # in the message for exactly that reason.
            return f"That schedule was not created. {exc}"
        fires = render_fires(task.next_fires(CONFIRM_FIRES), task.timezone)
        listed = "\n".join(f"- {fire}" for fire in fires)
        return f"Created schedule {task.id}. It next runs:\n{listed}"

    return Tool(
        name="schedule_create",
        description=(
            "Set up a standing task: something to do again on a schedule, in "
            "this conversation. Translate what the person asked for into a "
            "five-field cron expression and an IANA time zone — 'every weekday "
            "at 9am Seoul time' is `0 9 * * 1-5` with `Asia/Seoul`. Use the "
            "zone the person named, or the one you already know they are in; "
            "omit it only when they meant UTC. The prompt is what you will be "
            "asked each time it fires, so write it as an instruction that will "
            "still make sense with none of this conversation around it. "
            "Returns the next few fire times: state them back before treating "
            "the task as set up, because that is the only part of this the "
            "person can actually check."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to do each time, written to stand on its own — "
                        "'search for what happened in AI overnight and give me "
                        "the five that matter', not 'do that again'."
                    ),
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "Five fields: minute hour day-of-month month day-of-week. "
                        "`0 9 * * 1-5` is every weekday at nine."
                    ),
                },
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA zone the hour is read in, e.g. 'Asia/Seoul'. Not an "
                        "abbreviation: 'KST' is not a zone. Omit for UTC."
                    ),
                },
                "fire_once": {
                    "type": "boolean",
                    "description": "Run it once and finish, rather than every time it matches.",
                },
            },
            "required": ["prompt", "cron"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _timezone(args: dict[str, Any]) -> str | None:
    zone = str(args.get("timezone") or "").strip()
    return zone or None


def _surface_of(session_id: str) -> str:
    """Which surface a session key belongs to.

    Session ids are built by the adapter as `<surface>:...`, so the prefix is
    the surface. A task's own `surface` is what its fire is delivered as, and
    getting it wrong would enqueue an event no adapter answers.
    """
    surface, _, _ = session_id.partition(":")
    return surface or "cli"


# -- schedule_list -----------------------------------------------------------


def _list_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        found = await tasks.all(owner=context.author, session_id=context.session_id)
        if not found:
            return "There are no standing tasks in this conversation."
        return "\n".join(_describe(task) for task in found)

    return Tool(
        name="schedule_list",
        description=(
            "List the standing tasks you set up in this conversation, with "
            "their schedules and when each next runs. Use it before cancelling "
            "one, and to answer 'what have you got scheduled?'."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
    )


def _describe(task: Task) -> str:
    return f"{task.id} — {task.prompt!r} ({task.label}), {task.state}, next: {_next(task)}"


def _next(task: Task) -> str:
    """When it fires next, or why it never will.

    A schedule that stopped parsing has to say so here. Answering "what have
    you got scheduled?" with a row and no next time reads as though it is
    fine, and the person would go on believing they get their nine o'clock.
    """
    if task.state != ACTIVE:
        return f"not while it is {task.state}"
    try:
        return render_fires(task.next_fires(1), task.timezone)[0]
    except CronError as exc:
        return f"never — {exc}"


# -- schedule_cancel ---------------------------------------------------------


def _cancel_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        task_id = str(args["id"]).strip()
        # Found under the caller's own narrowing rather than by id and then
        # checked: an id from another channel must come back as "no such
        # schedule", not as a refusal that confirms it exists.
        visible = await tasks.all(owner=context.author, session_id=context.session_id)
        if task_id not in {task.id for task in visible}:
            return f"There is no schedule {task_id!r} in this conversation."
        await tasks.cancel(task_id)
        return f"Cancelled schedule {task_id}. It will not run again."

    return Tool(
        name="schedule_cancel",
        description=(
            "Cancel a standing task in this conversation, by the id "
            "`schedule_list` gives. Stopping it is immediate and permanent; "
            "there is nothing to undo it with."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The schedule's id."},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        handler=handler,
    )
