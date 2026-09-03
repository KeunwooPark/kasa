-- v1: the derived search index.
--
-- Everything in this file is disposable. `kasa reindex --full` deletes all of it
-- and rebuilds it by walking the repo, and that has to stay true: the moment
-- something durable lives only here, the repo stops being the source of truth
-- and the whole design of this system stops holding.
--
-- `chunks_vec` (embeddings) arrives with #31; lexical search comes first.

CREATE TABLE chunks (
    id         TEXT PRIMARY KEY,   -- <memory_id>:<ordinal>, so rebuilds are stable
    memory_id  TEXT NOT NULL,
    path       TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    scope      TEXT NOT NULL,      -- denormalized from frontmatter, to filter before ranking
    salience   REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX chunks_memory ON chunks (memory_id);
CREATE INDEX chunks_path   ON chunks (path);
CREATE INDEX chunks_scope  ON chunks (scope);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid');

-- External-content FTS5 does not track its table on its own. These keep the two
-- in step, so no code path can insert a chunk and forget to index it.
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE index_state (
    path       TEXT PRIMARY KEY,
    blob_sha   TEXT NOT NULL,      -- git's own blob hash; unchanged means skip
    indexed_at TEXT NOT NULL
);
