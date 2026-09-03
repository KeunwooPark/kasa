-- v2: the durable ingress queue.
--
-- Slack gives an adapter three seconds to acknowledge an event. If the agent
-- loop sits on that path a slow turn becomes a dropped event, and a dropped
-- event becomes a retry storm, so an adapter does exactly one thing: write a
-- row here, ack, return. A dispatcher drains it.
--
-- The payoff beyond latency is that the queue is durable. A process killed
-- mid-turn replays from this table instead of losing the message.
--
-- `UNIQUE (source, external_id)` is the whole dedupe story, and it is the
-- reason this is a table rather than an `asyncio.Queue`: Slack re-sends the
-- same event id aggressively, and a re-sent event that reaches the agent twice
-- produces two answers.

CREATE TABLE inbox (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,          -- 'slack' | 'cli' | 'http'
    external_id TEXT NOT NULL,          -- the provider's own event id
    payload     TEXT NOT NULL,          -- the normalized InboundEvent, as JSON
    received_at TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',  -- pending | leased | done | failed
    -- On a pending row: do not deliver before this, which is how a retry backs
    -- off. On a leased row: when the lease expires and the row becomes
    -- deliverable again, which is how a crash replays. Both readings are
    -- "leave this alone until", so one column carries both and one index
    -- serves the query that finds work.
    lease_until TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    UNIQUE (source, external_id)
);

CREATE INDEX inbox_ready ON inbox (state, lease_until, id);
