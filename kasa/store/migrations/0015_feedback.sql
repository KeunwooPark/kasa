-- v4: which memories produced an answer, and what somebody thought of it.
--
-- The cheapest quality signal there is. A 👍 on an answer costs the person one
-- click and says something no amount of retrieval tuning can work out on its
-- own: that what was recalled was the right thing to recall. An ❌ says the
-- opposite, and long-term memory has no other way to find out that one of its
-- files is wrong.
--
-- It only works if the answer is still connected to the memories behind it by
-- the time the reaction arrives, which may be days later and is certainly
-- after the process that produced it has forgotten everything. Hence `answers`:
-- one row per reply, keyed by the surface's own id for the message somebody
-- will later react to.

CREATE TABLE answers (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,                  -- 'slack' | 'cli' | 'http'
    -- The surface's key for the *answer*, not the question. On Slack that is
    -- the ts of the message Kasa posted, which is what a `reaction_added`
    -- names.
    external_id TEXT NOT NULL,
    session_id  TEXT,
    -- Inherited from the conversation. Feedback on a DM is as private as the
    -- DM, and the review an ❌ raises quotes it.
    scope       TEXT NOT NULL DEFAULT 'workspace',
    memory_ids  TEXT NOT NULL DEFAULT '[]',     -- JSON array, in rank order
    created_at  TEXT NOT NULL,
    UNIQUE (source, external_id)
);

-- Two consumers, two shapes, one table.
--
-- Salience reads `up` votes the way it reads recalls: a count within a window,
-- recomputed every night. That is what keeps `reflect` idempotent — see
-- `kasa/memory/salience.py`, where the whole argument for recomputing rather
-- than adjusting is made.
--
-- Confidence cannot work that way. It is not derived from anything; it is a
-- number a model set and nothing recomputes, so lowering it every night for
-- the same ❌ would walk it to zero over a fortnight. So a `down` vote is an
-- event that is applied exactly once, and `applied_at` is what says so.
CREATE TABLE memory_feedback (
    id         INTEGER PRIMARY KEY,
    memory_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,                   -- up | down
    answer_id  TEXT REFERENCES answers(id) ON DELETE CASCADE,
    -- Who reacted. One person is one vote per answer: without this a channel
    -- of forty people clicking 👍 on one reply would outweigh a month of
    -- everything else.
    author     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    applied_at TEXT,
    UNIQUE (memory_id, answer_id, author, kind)
);

CREATE INDEX memory_feedback_at ON memory_feedback (kind, created_at);
CREATE INDEX memory_feedback_pending ON memory_feedback (kind) WHERE applied_at IS NULL;
