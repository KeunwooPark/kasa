-- v2: things a person has to look at, because Kasa may not decide them alone.
--
-- The case this exists for: somebody deletes a message whose claim is already
-- a file in the memory repo. Kasa cannot quietly edit the corpus over it — a
-- retraction is not the same as a correction, the file may have been merged,
-- rewritten or built on since, and a background job that silently deletes what
-- a person wrote is exactly the behaviour the whole patch-plan pipeline exists
-- to prevent. So it says so, and stops.
--
-- Deliberately not the `jobs` table. A job is work Kasa knows how to do; a
-- review is work it has decided it should not do. Reusing `jobs` would put
-- rows in it that no handler will ever run and no scheduler should retry.
--
-- `key` is the dedupe. Slack re-sends a `message_deleted` it did not hear an
-- ack for, and three retries of one deletion is one review — the ingress path
-- handles these inline and its idempotence rests on this constraint.

CREATE TABLE reviews (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- source_edited | source_deleted
    -- What to go and look at, in the words a person would search for.
    subject     TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    -- Observation ids, memory ids: whatever names the thing to check, as JSON.
    refs        TEXT NOT NULL DEFAULT '[]',
    -- Inherited from whatever raised it. A review that quotes a DM is as
    -- private as the DM was.
    scope       TEXT NOT NULL DEFAULT 'workspace',
    state       TEXT NOT NULL DEFAULT 'open',  -- open | done
    key         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE (kind, key)
);

CREATE INDEX reviews_open ON reviews (state, created_at);
