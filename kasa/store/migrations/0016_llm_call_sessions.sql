-- Attribute conversational model calls to their session so cache performance
-- can be inspected per stable prompt prefix.

ALTER TABLE llm_calls ADD COLUMN session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL;
CREATE INDEX llm_calls_session ON llm_calls (session_id);
