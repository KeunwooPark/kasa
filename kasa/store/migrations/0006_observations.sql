-- v1: candidate facts awaiting promotion.
--
-- `memory_write` does not write a file. It appends here, and the `promote` job
-- (#29) turns pending observations into a validated patch plan. The interactive
-- path and the background path therefore share one write path, which is the
-- only one that has been through the patch validator.
--
-- `episodes` is created here because `observations` references it. Its lifecycle
-- — opening, closing, summarizing — belongs to #27; for now an observation
-- written by a tool has a session but no episode yet, so the reference is
-- nullable.

CREATE TABLE episodes (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    summary      TEXT,
    state        TEXT NOT NULL DEFAULT 'open',   -- open | closed | consolidated
    signal_score REAL
);

CREATE INDEX episodes_session ON episodes (session_id, state);

CREATE TABLE observations (
    id          TEXT PRIMARY KEY,
    episode_id  TEXT REFERENCES episodes(id) ON DELETE SET NULL,
    session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    subject     TEXT NOT NULL,
    claim       TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- fact | preference | decision | task | relation
    confidence  REAL NOT NULL DEFAULT 0.7,
    -- Inherited from the session that produced it, never chosen by the model.
    scope       TEXT NOT NULL,
    source_refs TEXT NOT NULL DEFAULT '[]',
    state       TEXT NOT NULL DEFAULT 'pending', -- pending | promoted | discarded
    created_at  TEXT NOT NULL
);

CREATE INDEX observations_state ON observations (state, created_at);
CREATE INDEX observations_subject ON observations (subject);
