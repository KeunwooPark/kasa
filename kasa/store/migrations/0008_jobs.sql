-- v3: durable background execution.
--
-- Jobs are rows rather than in-memory timers, which is the whole point: a
-- restart loses nothing, a crashed job's lease expires and it runs again, and
-- the same table extends to an out-of-process worker without a redesign —
-- a second worker is another drainer over these rows.
--
-- The same lease/attempt/dead-letter shape as `inbox`, deliberately, but not
-- the same table. The inbox dedupes on a provider's event id and never
-- schedules; a job schedules and never dedupes on anything external. What the
-- two genuinely share is the loop over them, and that lives in
-- `kasa/core/drain.py` rather than in a table one of them does not fit.

CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,      -- ULID for a one-shot; `kind@time` for a scheduled run
    kind        TEXT NOT NULL,         -- episode_close | promote | reflect | reorganize | forget | reindex
    payload     TEXT,                  -- JSON, or NULL
    -- When it becomes runnable. A recurring job's next fire time, or now for a
    -- one-shot, or now-plus-backoff after a failure — one column, because from
    -- the drainer's side those are the same statement.
    run_after   TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',  -- pending | leased | done | failed
    lease_until TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_at  TEXT NOT NULL,
    -- What `kasa job list` answers with: when this kind last actually ran.
    -- Also what retention is measured from, since a job may have been created
    -- long before it was due.
    finished_at TEXT
);

-- The drainer's query is "runnable now, of a kind I can run", oldest first.
CREATE INDEX jobs_ready ON jobs (state, kind, run_after);
