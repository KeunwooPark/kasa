-- v1: pinned, denormalized onto chunks.
--
-- Pinned memories are an always-on candidate source: they go into every
-- retrieval regardless of what was asked. Finding them by opening files would
-- put a filesystem walk on the hot path of every turn, and joining against the
-- manifest would put a second source of truth next to the one in the frontmatter.
-- It sits beside `scope` and `salience` for the same reason those do.

ALTER TABLE chunks ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;

CREATE INDEX chunks_pinned ON chunks (pinned) WHERE pinned = 1;
