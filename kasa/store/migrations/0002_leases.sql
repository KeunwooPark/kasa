-- v1: the single-writer lease over the long-term memory repo.
--
-- Two daemons pushing to the memory repo concurrently is the one way to lose
-- data in this system, so the lease is mandatory rather than advisory. This
-- table is the *visible* half: who holds it, since when, and for what job. The
-- enforcing half is an flock on the clone, which the kernel releases when the
-- holder dies — that is what tells a later run whether a row like this one is
-- live or merely left behind by a crash.

CREATE TABLE leases (
    name        TEXT PRIMARY KEY,
    holder      TEXT NOT NULL,   -- host:pid
    job         TEXT,
    acquired_at TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
