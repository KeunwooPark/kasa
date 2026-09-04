-- v4: when a memory was actually recalled.
--
-- `salience` is described as access-weighted, and until now nothing recorded
-- access. `reflect` decays salience nightly and boosts it for memories that
-- earned their place by being retrieved into a real conversation, which needs
-- a record of that happening.
--
-- Deliberately here and not in the repo. A hit is not a fact about the world;
-- it is telemetry about this installation, and it is the input to a number
-- that *is* durable — the `salience` in the frontmatter. Losing this table to
-- a rebuilt database loses the last few days of weighting and nothing else,
-- which is the right amount of durability for it.
--
-- One row per recall rather than a counter, so `reflect` can ask "since when"
-- and a run that fails does not double-count on the next one.

CREATE TABLE memory_hits (
    id        INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL,
    hit_at    TEXT NOT NULL
);

CREATE INDEX memory_hits_at ON memory_hits (hit_at);
CREATE INDEX memory_hits_memory ON memory_hits (memory_id, hit_at);
