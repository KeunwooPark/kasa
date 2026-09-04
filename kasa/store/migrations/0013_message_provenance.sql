-- v2: which message on which surface a stored message actually was.
--
-- Short-term memory is a transcript, and a transcript is only honest while the
-- conversation it copied still says the same thing. Slack lets anybody edit or
-- delete what they said an hour ago, and until now nothing connected that
-- event back to the row it invalidates: `messages.id` is a ULID this process
-- minted, and the only other identifier on the row is a sequence number.
--
-- `external_id` is the same key the inbox dedupes on — `slack:<team>:<channel>:<ts>`
-- — so a revision arriving for a message finds the row without knowing
-- anything about how it was stored. Nullable: every message written before
-- this migration has no such link, as does every assistant and tool message,
-- which nobody outside this process can revise.
--
-- `state` is what a reader has to respect. A tombstoned message is not
-- content any more, and serving one back to the model as if the person still
-- said it is the failure this table exists to prevent.

ALTER TABLE messages ADD COLUMN external_id TEXT;
ALTER TABLE messages ADD COLUMN state TEXT NOT NULL DEFAULT 'live';  -- live|edited|deleted
ALTER TABLE messages ADD COLUMN revised_at TEXT;

-- Partial: only inbound messages carry one, and the lookup is always by a
-- value that is present.
CREATE INDEX messages_external ON messages (external_id) WHERE external_id IS NOT NULL;
