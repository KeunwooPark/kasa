-- v3: a message belongs to the episode that was open when it arrived.
--
-- `episodes` has existed since 0006, but nothing opened one and nothing said
-- which messages were in one. Membership by time window — everything in this
-- session between `started_at` and `ended_at` — would have needed no column at
-- all, and it is wrong the moment a clock moves or a message is backfilled.
-- The link a consolidation job reasons about should be the one the write path
-- recorded, not one reconstructed afterwards from timestamps.
--
-- Nullable, and `ON DELETE SET NULL`: every message written before this
-- migration has no episode, and losing a transcript because its episode row
-- went away would be the wrong trade in the other direction.

ALTER TABLE messages ADD COLUMN episode_id TEXT REFERENCES episodes(id) ON DELETE SET NULL;

CREATE INDEX messages_episode ON messages (episode_id, seq);

-- `episode_close` sweeps by state across every session, which the existing
-- (session_id, state) index cannot serve.
CREATE INDEX episodes_state ON episodes (state, started_at);
