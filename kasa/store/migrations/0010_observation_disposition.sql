-- v3: what happened to a candidate fact, and why.
--
-- `promote` moves an observation out of `pending`, and until now the row said
-- only which of the two it moved to. "Discarded" with no reason is a decision
-- nobody can audit and nobody can learn from — and the whole reason
-- observations are rows rather than a queue in memory is that a person should
-- be able to read what the consolidator decided and disagree with it.
--
-- `attempts` is the bound. A subject whose plan the validator keeps rejecting
-- costs a model call every hour forever, and the observations behind it are
-- never promoted anyway; after a few tries they are discarded, with the
-- rejection recorded as the reason.

ALTER TABLE observations ADD COLUMN reason TEXT;
ALTER TABLE observations ADD COLUMN resolved_at TEXT;
ALTER TABLE observations ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;

-- `promote` reads pending rows and groups them by the pair it reconciles as a
-- unit. Grouping by subject alone would put two visibility scopes in one call.
CREATE INDEX observations_pending ON observations (scope, subject, created_at)
    WHERE state = 'pending';
