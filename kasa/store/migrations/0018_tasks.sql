-- v5: schedules a person created, rather than ones compiled in.
--
-- Every recurring thing Kasa did before this was a `JobSpec` in
-- `default_specs` — written by whoever built the binary, and unchangeable by
-- whoever uses it. A row here is the other kind: someone said "every weekday at
-- nine, tell me what happened overnight", and this is where that lives.
--
-- The table is deliberately not `jobs`. A job is one run; a task is the
-- standing intent that produces runs, and it outlives every one of them. The
-- clock reads this table and writes `jobs` rows, so the two keep the roles they
-- already had: this one remembers, that one executes.
--
-- What is *not* here is anywhere for the task to post. `channel`, `reply_to`,
-- `scope` and `session_id` are copied from the conversation that created the
-- task and are never settable afterwards (§11.1). A task inherits the
-- visibility of the thread it was asked for in, so text arriving in a DM
-- cannot arrange for anything to be said in a public channel.

CREATE TABLE tasks (
    id           TEXT PRIMARY KEY,      -- ULID
    -- The platform id of whoever asked for it. The cap on how many a person may
    -- create is counted on this, and it is who gets told when one is paused.
    owner        TEXT NOT NULL,
    surface      TEXT NOT NULL,         -- 'slack' | 'cli' | 'http'
    -- Where the turn happens and where the answer goes. Fixed at creation.
    session_id   TEXT NOT NULL,
    channel      TEXT,
    reply_to     TEXT,
    scope        TEXT NOT NULL DEFAULT 'workspace',
    -- What to ask. It becomes the text of an ordinary inbound event, so a task
    -- is answered by the same loop, with the same memory and the same tools, as
    -- the person typing it would have been.
    prompt       TEXT NOT NULL,
    cron         TEXT NOT NULL,         -- five fields
    -- IANA, or NULL for UTC. Stored beside the expression rather than folded
    -- into it because "9am Seoul" moves twice a year and the fields do not.
    timezone     TEXT,
    state        TEXT NOT NULL DEFAULT 'active',  -- active | paused | done
    -- A one-shot: fires once and ends `done`. "Remind me on Friday" is a
    -- schedule with an end, not a different mechanism.
    fire_once    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    last_run_at  TEXT,
    last_job_id  TEXT,
    last_error   TEXT,
    -- Reset by a run that worked. A task that fails forever in silence is worse
    -- than one that stops and says so, and this is what decides when to stop.
    consecutive_failures INTEGER NOT NULL DEFAULT 0
);

-- The clock's query: every task that is still due to fire, oldest first so the
-- order a person created them in is the order they are queued in.
CREATE INDEX tasks_active ON tasks (state, created_at);

-- `schedule_list` and the per-owner cap both ask "whose, and where".
CREATE INDEX tasks_owner ON tasks (owner, session_id);
