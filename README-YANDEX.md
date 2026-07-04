# Local Graph Builder with Yandex AI Studio

This workspace runs an isolated Docker Compose project named
`nornikel-knowledge-graph`. It does not reuse or remove the older `graph`
containers or volumes found on this machine. The initial build reuses the
compatible Python and static-web dependency images already cached locally, but
creates separate final images containing this workspace's Yandex integration.

## Start and stop

```powershell
.\start.ps1
.\stop.ps1
```

- Graph Builder: <http://localhost:8080>
- Backend health: <http://localhost:8000/health>
- Neo4j Browser: <http://localhost:7474>
- PostgreSQL/ParadeDB: `localhost:5433` (database and password are in `.env`)

Neo4j uses the username `neo4j`; its generated password is stored only in the
local `.env` file. The Yandex API key is also stored only there and must be
rotated because it was previously shared in chat.

## Configuration

The frontend exposes `yandex_aliceai`. Graph extraction and chat use the model
in `LLM_MODEL`. Stored graph content uses `EMBED_DOC_MODEL`; searches use
`EMBED_QUERY_MODEL`. Both vectors are configured for 768 dimensions.

To inspect service state or logs:

```powershell
docker compose ps
docker compose logs --tail 100 backend
```

## MEKG pilot corpus

The checked-in pilot manifest contains exactly five documents from each of the
five source folders (25 documents total). Run the evidence-oriented sample with:

```powershell
.\run-mekg-pilot.ps1 -Reset -NoVision -MaxExtractElementsPerDoc 6 -MaxConcurrency 3 -EmbeddingConcurrency 2
```

`-NoVision` is intentional for this corpus because the selected PDF, Word, and
PowerPoint files already have a text layer. The cap ranks narrative fragments
and table rows with measurements, units, conditions, processes, and conclusions.
Documents processed this way are marked `completed_sampled`; this is a quality
pilot, not a claim that every fact from every selected document was extracted.

The resulting quality report is copied to `artifacts\mekg-pilot\qa-report.md`.
For a structure-only parser check, add `-ParseOnly` and omit the extraction cap.
Evidence-linked chunks are embedded with the document model after extraction;
`semantic_search` embeds the user's request with the query model. The default
embedding concurrency is deliberately two to stay below Yandex rate limits.

Example semantic retrieval:

```powershell
$body = @{ text = 'электроэкстракция никеля циркуляция католита'; limit = 5 } |
    ConvertTo-Json
Invoke-RestMethod http://localhost:8000/api/mekg/v1/semantic_search `
    -Method Post -ContentType 'application/json; charset=utf-8' `
    -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

To remove only this new stack while keeping its data:

```powershell
docker compose down
```

Do not add `-v` unless the new Neo4j data is intentionally being deleted.

## Full resumable corpus and vector search

The fast-full runner indexes every text chunk in ParadeDB/pgvector and commits
each completed document to Neo4j in one transaction. PDF OCR, VLM, images, and
`find_tables()` are disabled; native Word, PowerPoint, and Excel tables remain.

```powershell
# Optional 100-document stratified benchmark
.\run-mekg-full.ps1 -Action Start -BenchmarkLimit 100 -DeadlineHours 12

# Full 1,353-document run
.\run-mekg-full.ps1 -Action Start -DeadlineHours 12

# Inspect the latest run or resume a particular run after interruption
.\run-mekg-full.ps1 -Action Status
.\run-mekg-full.ps1 -Action Resume -RunId '<run-uuid>' -DeadlineHours 12

# Stop the backend worker without deleting Neo4j or PostgreSQL volumes
.\run-mekg-full.ps1 -Action Stop
```

The search API is `POST /api/mekg/v1/cross_corpus_search`; the same operation
is exposed over MCP and in the MEKG panel. The older `semantic_search` endpoint
now performs dense-only retrieval from Postgres with the Yandex query model.

## Agentic RAG and public web search

The MEKG panel now contains a deterministic evidence loop based on `arch.md`.
It persists every run and trace event in PostgreSQL and executes it in the
separate `agent-worker` service, so long research requests do not occupy the
HTTP workers. The state machine performs at most three iterations and returns
`full_answer`, `partial_answer_with_gaps`, or an explicit no-data mode.

External Yandex Web Search is off by default. Enable it for an individual query
with the consent checkbox in the UI or `allow_external_web: true` in the API.
Only sanitized query rewrites leave the local stack; document chunks, local
paths, evidence packs, and internal identifiers are never included in web
prompts. Add sensitive project terms to the comma-separated `WEB_REDACT_TERMS`
setting. Closed sources such as Scopus, SciFinder, Reaxys, Total Materia and
Knovel are treated as metadata-only unless a future licensed connector supplies
the full text.

```powershell
$body = @{
  query = 'Какие технологии выщелачивания никеля подтверждены источниками?'
  allow_external_web = $false
  max_iterations = 3
  include_debug = $true
} | ConvertTo-Json

$job = Invoke-RestMethod http://localhost:8000/api/mekg/v1/agentic_rag/jobs `
  -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))

Invoke-RestMethod "http://localhost:8000/api/mekg/v1/agentic_rag/jobs/$($job.job_id)"
```

Other endpoints:

- `GET /api/mekg/v1/agentic_rag/jobs/{id}/events` — reconnectable SSE trace;
- `POST /api/mekg/v1/agentic_rag/jobs/{id}/cancel` — cooperative cancellation;
- `GET /api/mekg/v1/web_search/profiles` — curated source profiles;
- `POST /api/mekg/v1/web_search` — direct public web-search tool;
- `POST /api/mekg/v1/qa/run` — queue the expensive graph QA scan;
- `GET /api/mekg/v1/qa` — return the latest cached QA snapshot quickly.

If Yandex returns `403 PermissionDenied`, the agent opens a circuit breaker and
continues with Neo4j and Postgres BM25. Dense retrieval, web search, reranking,
and final Alice synthesis remain disabled for that run, and the system will not
claim a full generated answer. Grant the service account the required AI Studio
and Search API roles, verify billing/quotas, then rotate the API key that was
previously shared in chat.
