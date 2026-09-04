-- v2: who a Slack user id actually is, cached.
--
-- `users.info` is a network call, and the two places that need a name cannot
-- make one: ingress has three seconds to acknowledge an event, and rendering a
-- message for the model happens once per turn. A workspace's membership barely
-- changes, so it is cached here and refreshed on a TTL.
--
-- Derived, and deliberately so. The durable half of an identity is the
-- `source_refs: [slack://<team>/<user>]` on the `people/` memory in the repo;
-- this table is a lookup in front of it, and losing it to a rebuilt database
-- costs one `users.info` per person and re-links from those refs.
--
-- `memory_name` is what the linked memory was last written with, which is what
-- makes a display-name change detectable without diffing the corpus: a row
-- whose `display_name` has moved on from its `memory_name` is one the identity
-- job has to update — *update*, never a second file for the same person.

CREATE TABLE slack_users (
    team_id      TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    display_name TEXT NOT NULL,
    real_name    TEXT NOT NULL DEFAULT '',
    is_bot       INTEGER NOT NULL DEFAULT 0,
    deleted      INTEGER NOT NULL DEFAULT 0,
    -- The `people/` memory this uid resolves to, once the identity job has
    -- written or adopted one. NULL means "seen, not yet linked".
    memory_id    TEXT,
    memory_name  TEXT,
    fetched_at   TEXT NOT NULL,
    linked_at    TEXT,
    PRIMARY KEY (team_id, user_id)
);

-- The identity job's whole query: rows never linked, and rows whose person has
-- since been renamed. Small table, but the sweep runs every few minutes.
CREATE INDEX slack_users_unlinked ON slack_users (fetched_at)
    WHERE is_bot = 0 AND (memory_id IS NULL OR memory_name IS NOT display_name);
