-- v1: stem the search index.
--
-- The golden set found this. "Who should I ask about deploys?" missed a memory
-- that says "deploy pipeline", and "Is credential rotation automatic?" missed
-- one that says "credentials rotate" — both because the default tokenizer
-- matches whole words and people do not conjugate their questions to match
-- their notes. Porter stemming turns that class of miss into a hit.
--
-- The table is dropped and rebuilt rather than migrated: FTS5 stores tokens, so
-- changing the tokenizer means every token in it is now wrong. `rebuild` reads
-- back through the content table, so nothing is lost.

DROP TABLE chunks_fts;

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize="porter unicode61"
);

INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild');
