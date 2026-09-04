-- Vector indexes are disposable generations. A replacement is built under a
-- new vec0 table and made active only when complete, so model changes retain
-- lexical retrieval throughout the rebuild.

CREATE TABLE vector_indexes (
    version    TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    table_name TEXT NOT NULL UNIQUE,
    active     INTEGER NOT NULL DEFAULT 0,
    built_at   TEXT NOT NULL
);

CREATE UNIQUE INDEX vector_indexes_active ON vector_indexes(active) WHERE active = 1;
