# Prepared demo data

The two LFS files in this directory are reproducible snapshots of the prepared
demo state:

- `search-db.pgdump`: Postgres/ParadeDB documents, chunks, BM25 metadata and
  vector embeddings. Runtime job/event/cache rows are intentionally excluded.
- `neo4j.dump`: Neo4j knowledge graph.

Docker Compose restores both snapshots automatically only when their named
volumes are empty. Existing local volumes are never overwritten.
