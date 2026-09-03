-- v0: conversations and call accounting.
--
-- Note what is deliberately absent: anything durable. Long-term memory lives in
-- the git repo (see docs/DESIGN.md §2); this database is a hot buffer and a
-- rebuildable index, and every table added here must keep that true.

CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    surface     TEXT NOT NULL,
    scope       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    last_active TEXT NOT NULL
);

CREATE TABLE messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    author     TEXT,
    content    TEXT NOT NULL,  -- JSON array of canonical content blocks
    tokens     INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);

CREATE INDEX messages_session_seq ON messages (session_id, seq);

CREATE TABLE llm_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    role        TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    tag         TEXT,
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    ok          INTEGER NOT NULL DEFAULT 1,
    error       TEXT
);

CREATE INDEX llm_calls_created ON llm_calls (created_at);
