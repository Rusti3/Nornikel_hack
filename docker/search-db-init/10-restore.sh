#!/usr/bin/env bash
set -euo pipefail

seed="/seed/search-db.pgdump"

if [[ ! -f "$seed" ]]; then
  echo "Postgres demo seed is missing; starting with an empty database."
  exit 0
fi

echo "Restoring prepared Postgres vector/BM25 demo data..."
pg_restore \
  --exit-on-error \
  --no-owner \
  --no-privileges \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  "$seed"
echo "Postgres demo seed restored."
