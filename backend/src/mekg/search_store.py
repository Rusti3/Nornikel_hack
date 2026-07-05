from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import MEKGConfig
from .models import ParsedDocument, SourceElement


CORPORA = (
    ("internal_reports", "Доклады", 1.15, "Внутренние технические доклады и презентации"),
    ("scientific_journals", "Журналы", 1.00, "Выпуски научных и отраслевых журналов"),
    ("conference_materials", "Материалы конференций", 1.05, "Материалы научных конференций"),
    ("reviews", "Обзоры", 1.10, "Тематические и аналитические обзоры"),
    ("scientific_articles", "Статьи", 1.05, "Отдельные научные статьи"),
)

CATEGORY_TO_CORPUS = {
    "доклады": "internal_reports",
    "журналы": "scientific_journals",
    "материалы конференций": "conference_materials",
    "обзоры": "reviews",
    "статьи": "scientific_articles",
}

STAGES = (
    "discovered",
    "parsed",
    "chunks_loaded",
    "first_extraction_done",
    "second_extraction_done",
    "second_extraction_skipped",
    "neo4j_committed",
    "complete",
    "complete_with_warnings",
    "failed",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def corpus_for_category(category: str | None) -> str:
    return CATEGORY_TO_CORPUS.get((category or "").strip().casefold(), "scientific_articles")


def vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


class SearchStore:
    """Primary retrieval store and durable full-corpus ingestion checkpoint."""

    def __init__(self, config: MEKGConfig | None = None, *, pool: ConnectionPool | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        if not self.config.postgres_password and pool is None:
            raise ValueError("POSTGRES_PASSWORD is required for the MEKG search store")
        self.pool = pool or ConnectionPool(
            conninfo=self.config.postgres_dsn,
            min_size=self.config.postgres_pool_min,
            max_size=self.config.postgres_pool_max,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=False,
        )

    def open(self) -> None:
        if self.pool.closed:
            self.pool.open(wait=True, timeout=30)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def connection(self):
        self.open()
        with self.pool.connection() as conn:
            yield conn

    def initialize_schema(self) -> dict[str, Any]:
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE EXTENSION IF NOT EXISTS pg_search",
            """
            CREATE TABLE IF NOT EXISTS corpora (
                id text PRIMARY KEY, display_name text NOT NULL, weight double precision NOT NULL,
                description text NOT NULL, enabled boolean NOT NULL DEFAULT true,
                routing_terms jsonb NOT NULL DEFAULT '[]'::jsonb, updated_at timestamptz NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id uuid PRIMARY KEY, corpus_root text NOT NULL, mode text NOT NULL,
                deadline_at timestamptz NOT NULL, status text NOT NULL DEFAULT 'running',
                settings jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                id text PRIMARY KEY, version_id text NOT NULL, source_locator text NOT NULL UNIQUE,
                file_name text NOT NULL, file_type text NOT NULL, sha256 text NOT NULL,
                size_bytes bigint NOT NULL, category text, corpus_id text NOT NULL REFERENCES corpora(id),
                title text, language text, year integer, geography text[], authors text[],
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb, stage text NOT NULL DEFAULT 'discovered',
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS document_jobs (
                run_id uuid NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
                document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                source_path text NOT NULL, stage text NOT NULL DEFAULT 'discovered', attempts integer NOT NULL DEFAULT 0,
                lease_owner text, lease_until timestamptz, error text, timings jsonb NOT NULL DEFAULT '{}'::jsonb,
                extraction_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                selected_element_ids text[] NOT NULL DEFAULT '{}'::text[],
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (run_id, document_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id text PRIMARY KEY, document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                version_id text NOT NULL, corpus_id text NOT NULL REFERENCES corpora(id),
                ordinal integer NOT NULL, kind text NOT NULL, text text NOT NULL, title text,
                page_number integer, slide_number integer, sheet_name text, row_number integer,
                language text, source_type text, evidence_linked boolean NOT NULL DEFAULT false,
                confidence double precision, candidate_entities jsonb NOT NULL DEFAULT '[]'::jsonb,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb, embedding vector(768),
                embedding_model text, embedded_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(), UNIQUE(document_id, version_id, ordinal, kind)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunk_numeric_facts (
                id bigserial PRIMARY KEY, chunk_id text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                parameter text NOT NULL, value_min double precision, value_max double precision,
                comparator text, unit_original text, unit_normalized text, dimension text,
                confidence double precision, metadata jsonb NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key text PRIMARY KEY, kind text NOT NULL, payload jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id uuid PRIMARY KEY, run_type text NOT NULL DEFAULT 'agentic_rag',
                status text NOT NULL DEFAULT 'queued', request_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                state_json jsonb NOT NULL DEFAULT '{}'::jsonb, result_json jsonb,
                error text, attempts integer NOT NULL DEFAULT 0, cancel_requested boolean NOT NULL DEFAULT false,
                lease_owner text, lease_until timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
                started_at timestamptz, updated_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_events (
                id bigserial PRIMARY KEY, run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                event_type text NOT NULL, iteration integer, payload jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """,
            "CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents(corpus_id)",
            "CREATE INDEX IF NOT EXISTS document_jobs_stage_idx ON document_jobs(run_id, stage, lease_until)",
            "CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks(corpus_id)",
            "CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING gin(metadata)",
            "CREATE INDEX IF NOT EXISTS numeric_parameter_idx ON chunk_numeric_facts(lower(parameter))",
            "CREATE INDEX IF NOT EXISTS numeric_range_idx ON chunk_numeric_facts(value_min, value_max)",
            "CREATE INDEX IF NOT EXISTS agent_runs_claim_idx ON agent_runs(status, lease_until, created_at)",
            "CREATE INDEX IF NOT EXISTS agent_events_run_idx ON agent_events(run_id, id)",
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
                cur.executemany(
                    """
                    INSERT INTO corpora(id,display_name,weight,description) VALUES(%s,%s,%s,%s)
                    ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name,
                        weight=excluded.weight,description=excluded.description,updated_at=now()
                    """,
                    CORPORA,
                )
            conn.commit()
        return self.ensure_search_indexes()

    def ensure_search_indexes(self) -> dict[str, Any]:
        created: list[str] = []
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks
                    USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100)
                    """
                )
                created.append("hnsw")
                cur.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM pg_indexes
                                  WHERE schemaname=current_schema() AND indexname='chunks_bm25_idx') AS present
                    """
                )
                present = cur.fetchone()["present"]
                if not present:
                    cur.execute(
                        """
                        CREATE INDEX chunks_bm25_idx ON chunks USING bm25
                        (id, text, title, corpus_id, source_type, language)
                        WITH (key_field='id')
                        """
                    )
                created.append("bm25")
            conn.commit()
        return {"extensions": ["vector", "pg_search"], "indexes": created}

    def create_or_resume_run(
        self, corpus_root: str, *, mode: str, deadline_hours: float, run_id: str | None = None
    ) -> str:
        with self.connection() as conn:
            with conn.cursor() as cur:
                if run_id:
                    cur.execute("SELECT id FROM pipeline_runs WHERE id=%s", (run_id,))
                    if not cur.fetchone():
                        raise KeyError(f"Pipeline run not found: {run_id}")
                    cur.execute(
                        "UPDATE pipeline_runs SET status='running',deadline_at=%s,updated_at=now() WHERE id=%s RETURNING id::text",
                        (utcnow() + timedelta(hours=deadline_hours), run_id),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id::text FROM pipeline_runs
                        WHERE corpus_root=%s AND mode=%s AND status IN ('running','interrupted')
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (corpus_root, mode),
                    )
                    found = cur.fetchone()
                    if found:
                        cur.execute(
                            """
                            UPDATE pipeline_runs SET status='running',deadline_at=%s,updated_at=now()
                            WHERE id=%s
                            """,
                            (utcnow() + timedelta(hours=deadline_hours), found["id"]),
                        )
                        conn.commit()
                        return found["id"]
                    cur.execute(
                        """
                        INSERT INTO pipeline_runs(id,corpus_root,mode,deadline_at,settings)
                        VALUES(gen_random_uuid(),%s,%s,%s,%s::jsonb) RETURNING id::text
                        """,
                        (
                            corpus_root,
                            mode,
                            utcnow() + timedelta(hours=deadline_hours),
                            json.dumps({"deadline_hours": deadline_hours}),
                        ),
                    )
                value = cur.fetchone()["id"]
            conn.commit()
        return value

    def ensure_pipeline_run(
        self, run_id: str, corpus_root: str, *, mode: str = "upload", deadline_hours: float = 24
    ) -> str:
        """Create the durable pipeline row backing an upload job, idempotently."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs(id,corpus_root,mode,deadline_at,status,settings)
                VALUES(%s,%s,%s,%s,'running',%s::jsonb)
                ON CONFLICT(id) DO UPDATE SET status='running',updated_at=now()
                RETURNING id::text
                """,
                (
                    run_id,
                    corpus_root,
                    mode,
                    utcnow() + timedelta(hours=deadline_hours),
                    json.dumps({"source": "interactive_upload"}),
                ),
            )
            value = cur.fetchone()["id"]
            conn.commit()
            return value

    def register_document(self, run_id: str, document: ParsedDocument, source_path: str) -> None:
        corpus_id = corpus_for_category(document.category)
        year_match = re.search(r"(?:19|20)\d{2}", document.source_locator)
        year = int(year_match.group()) if year_match else None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents(id,version_id,source_locator,file_name,file_type,sha256,size_bytes,
                                          category,corpus_id,title,language,year,stage)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'discovered')
                    ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id,sha256=excluded.sha256,
                        size_bytes=excluded.size_bytes,category=excluded.category,corpus_id=excluded.corpus_id,
                        title=COALESCE(excluded.title,documents.title),language=COALESCE(excluded.language,documents.language),
                        year=COALESCE(excluded.year,documents.year),
                        updated_at=now()
                    """,
                    (
                        document.document_id, document.version_id, document.source_locator, document.file_name,
                        document.file_type, document.sha256, document.size_bytes, document.category, corpus_id,
                        document.title, document.language, year,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO document_jobs(run_id,document_id,source_path)
                    VALUES(%s,%s,%s) ON CONFLICT(run_id,document_id) DO UPDATE SET source_path=excluded.source_path
                    """,
                    (run_id, document.document_id, source_path),
                )
            conn.commit()

    def upsert_document_chunks(self, run_id: str, document: ParsedDocument) -> int:
        searchable = [element for element in document.elements if element.text.strip() and element.kind.value in {"text", "table_row"}]
        corpus_id = corpus_for_category(document.category)
        rows = []
        for ordinal, element in enumerate(searchable):
            rows.append(
                (
                    element.id, document.document_id, document.version_id, corpus_id, ordinal,
                    element.kind.value, element.text, document.title, element.page_number, element.slide_number,
                    element.sheet_name, element.row_number, document.language, document.file_type,
                        json.dumps(element.metadata, ensure_ascii=True, default=str),
                )
            )
        with self.connection() as conn:
            with conn.cursor() as cur:
                if rows:
                    for start in range(0, len(rows), self.config.chunk_write_batch):
                        cur.executemany(
                        """
                        INSERT INTO chunks(id,document_id,version_id,corpus_id,ordinal,kind,text,title,
                                           page_number,slide_number,sheet_name,row_number,language,source_type,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT(id) DO UPDATE SET text=excluded.text,title=excluded.title,
                            page_number=excluded.page_number,slide_number=excluded.slide_number,
                            sheet_name=excluded.sheet_name,row_number=excluded.row_number,
                            language=excluded.language,source_type=excluded.source_type,metadata=excluded.metadata,
                            updated_at=now()
                        """,
                            rows[start : start + self.config.chunk_write_batch],
                        )
                cur.execute(
                    "UPDATE documents SET stage='chunks_loaded',title=%s,language=%s,updated_at=now() WHERE id=%s",
                    (document.title, document.language, document.document_id),
                )
                cur.execute(
                    "UPDATE document_jobs SET stage='chunks_loaded',attempts=0,lease_owner=NULL,lease_until=NULL,updated_at=now() WHERE run_id=%s AND document_id=%s",
                    (run_id, document.document_id),
                )
            conn.commit()
        return len(rows)

    def chunks_without_embeddings(self, run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id,c.text FROM chunks c JOIN document_jobs j ON j.document_id=c.document_id
                WHERE j.run_id=%s AND c.embedding IS NULL ORDER BY c.id LIMIT %s
                """,
                (run_id, limit),
            )
            return list(cur.fetchall())

    def set_embedding(self, chunk_id: str, vector: list[float], model: str) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET embedding=%s::vector,embedding_model=%s,embedded_at=now(),updated_at=now() WHERE id=%s",
                (vector_literal(vector), model, chunk_id),
            )
            conn.commit()

    def document_chunks_without_embeddings(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,text FROM chunks WHERE document_id=%s AND embedding IS NULL ORDER BY ordinal",
                (document_id,),
            )
            return list(cur.fetchall())

    def document_embedding_counts(self, document_id: str) -> dict[str, int]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*)::int AS chunks,count(embedding)::int AS embedded FROM chunks WHERE document_id=%s",
                (document_id,),
            )
            return cur.fetchone() or {"chunks": 0, "embedded": 0}

    def find_document_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE sha256=%s ORDER BY updated_at DESC LIMIT 1",
                (sha256,),
            )
            return cur.fetchone()

    def corpus_version(self) -> str:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT coalesce(max(updated_at)::text,'empty') AS version,count(*)::text AS count FROM documents"
            )
            row = cur.fetchone() or {"version": "empty", "count": "0"}
            return f"{row['version']}:{row['count']}"

    def save_extraction(
        self, run_id: str, document_id: str, *, pass_number: int, element_ids: list[str], payload: dict[str, Any]
    ) -> None:
        key = "first" if pass_number == 1 else "second"
        stage = "first_extraction_done" if pass_number == 1 else "second_extraction_done"
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_jobs SET extraction_json=jsonb_set(extraction_json,%s,%s::jsonb,true),
                    selected_element_ids=(SELECT ARRAY(SELECT DISTINCT unnest(selected_element_ids || %s::text[]))),
                    stage=%s,attempts=0,lease_owner=NULL,lease_until=NULL,updated_at=now(),error=NULL
                    WHERE run_id=%s AND document_id=%s
                """,
                        ([key], json.dumps(payload, ensure_ascii=True), element_ids, stage, run_id, document_id),
            )
            cur.execute("UPDATE documents SET stage=%s,updated_at=now() WHERE id=%s", (stage, document_id))
            extraction_map = payload.get("extractions", {})
            for chunk_id, extraction in extraction_map.items():
                entities = [
                    {"name": item.get("canonical_name"), "type": item.get("entity_type")}
                    for item in extraction.get("entities", [])
                    if item.get("canonical_name")
                ]
                cur.execute(
                    "UPDATE chunks SET evidence_linked=true,candidate_entities=%s::jsonb,updated_at=now() WHERE id=%s",
                (json.dumps(entities, ensure_ascii=True), chunk_id),
                )
                cur.execute("DELETE FROM chunk_numeric_facts WHERE chunk_id=%s", (chunk_id,))
                numeric_rows = []
                for item in [*extraction.get("conditions", []), *extraction.get("measurements", [])]:
                    numeric_rows.append(
                        (
                            chunk_id,
                            item.get("name") or item.get("property_name") or "value",
                            item.get("value_min") if item.get("value_min") is not None else item.get("value"),
                            item.get("value_max") if item.get("value_max") is not None else item.get("value"),
                            item.get("comparator"), item.get("unit_original"), None, None,
                        item.get("confidence"), json.dumps({"pass": pass_number}, ensure_ascii=True),
                        )
                    )
                if numeric_rows:
                    cur.executemany(
                        """
                        INSERT INTO chunk_numeric_facts(chunk_id,parameter,value_min,value_max,comparator,
                            unit_original,unit_normalized,dimension,confidence,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        numeric_rows,
                    )
            conn.commit()

    def claim_jobs(
        self, run_id: str, *, stage: str, owner: str, limit: int = 1
    ) -> list[dict[str, Any]]:
        """Lease jobs safely for multiple local or container workers."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH candidates AS (
                    SELECT run_id,document_id FROM document_jobs
                    WHERE run_id=%s AND stage=%s AND attempts < %s
                      AND (lease_until IS NULL OR lease_until < now())
                    ORDER BY document_id FOR UPDATE SKIP LOCKED LIMIT %s
                )
                UPDATE document_jobs j SET lease_owner=%s,
                    lease_until=now()+(%s || ' seconds')::interval,
                    attempts=j.attempts+1,updated_at=now()
                FROM candidates c WHERE j.run_id=c.run_id AND j.document_id=c.document_id
                RETURNING j.*
                """,
                (
                    run_id, stage, self.config.job_max_attempts, limit, owner,
                    self.config.job_lease_seconds,
                ),
            )
            rows = list(cur.fetchall())
            conn.commit()
            return rows

    def heartbeat(self, run_id: str, document_id: str, owner: str) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_jobs SET lease_until=now()+(%s || ' seconds')::interval,updated_at=now()
                WHERE run_id=%s AND document_id=%s AND lease_owner=%s
                """,
                (self.config.job_lease_seconds, run_id, document_id, owner),
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT *,id::text AS run_id FROM pipeline_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Pipeline run not found: {run_id}")
            return row

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE pipeline_runs SET status=%s,updated_at=now(),finished_at=now() WHERE id=%s",
                (status, run_id),
            )
            conn.commit()

    def mark_stage(self, run_id: str, document_id: str, stage: str, *, error: str | None = None) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_jobs SET attempts=CASE WHEN stage<>%s THEN 0 ELSE attempts END,
                    stage=%s,error=%s,updated_at=now(),lease_owner=NULL,lease_until=NULL
                WHERE run_id=%s AND document_id=%s
                """,
                (stage, stage, error, run_id, document_id),
            )
            cur.execute("UPDATE documents SET stage=%s,updated_at=now() WHERE id=%s", (stage, document_id))
            conn.commit()

    def load_job(self, run_id: str, document_id: str) -> dict[str, Any] | None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.*,d.source_locator,d.file_name,d.category,d.corpus_id,d.version_id
                FROM document_jobs j JOIN documents d ON d.id=j.document_id
                WHERE j.run_id=%s AND j.document_id=%s
                """,
                (run_id, document_id),
            )
            return cur.fetchone()

    def list_jobs(self, run_id: str, stages: Iterable[str] | None = None) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            if stages:
                cur.execute(
                    "SELECT * FROM document_jobs WHERE run_id=%s AND stage=ANY(%s) ORDER BY document_id",
                    (run_id, list(stages)),
                )
            else:
                cur.execute("SELECT * FROM document_jobs WHERE run_id=%s ORDER BY document_id", (run_id,))
            return list(cur.fetchall())

    def status(self, run_id: str | None = None) -> dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            if not run_id:
                cur.execute("SELECT id::text FROM pipeline_runs ORDER BY created_at DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return {"run_id": None, "stages": {}, "chunks": 0, "embedded": 0}
                run_id = row["id"]
            cur.execute(
                "SELECT stage,count(*)::int AS count FROM document_jobs WHERE run_id=%s GROUP BY stage",
                (run_id,),
            )
            stages = {row["stage"]: row["count"] for row in cur.fetchall()}
            cur.execute(
                """
                SELECT count(*)::int AS chunks,count(embedding)::int AS embedded
                FROM chunks c WHERE EXISTS(SELECT 1 FROM document_jobs j WHERE j.run_id=%s AND j.document_id=c.document_id)
                """,
                (run_id,),
            )
            counts = cur.fetchone()
            return {"run_id": run_id, "stages": stages, **counts}

    def dense_search(
        self, vector: list[float], *, corpora: list[str], limit: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        clauses = ["c.embedding IS NOT NULL", "c.corpus_id=ANY(%s)"]
        filter_params: list[Any] = [corpora]
        self._append_filters(clauses, filter_params, filters)
        encoded = vector_literal(vector)
        params: list[Any] = [encoded, *filter_params, encoded, limit]
        sql = f"""
            SELECT c.id AS chunk_id,c.document_id,c.corpus_id,c.text,c.title,c.page_number,c.slide_number,
                   c.sheet_name,c.row_number,c.source_type,c.language,c.metadata,c.candidate_entities,
                   d.file_name,d.source_locator,1-(c.embedding <=> %s::vector) AS dense_score
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE {' AND '.join(clauses)} ORDER BY c.embedding <=> %s::vector LIMIT %s
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search=100")
            cur.execute("SET LOCAL hnsw.iterative_scan='relaxed_order'")
            cur.execute(sql, params)
            return list(cur.fetchall())

    def bm25_search(
        self, query: str, *, corpora: list[str], limit: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        clauses = ["c.text @@@ %s", "c.corpus_id=ANY(%s)"]
        params: list[Any] = [query, corpora]
        self._append_filters(clauses, params, filters)
        score_functions = ("paradedb.score(c.id)", "pdb.score(c.id)")
        last_error: Exception | None = None
        for score_expression in score_functions:
            sql = f"""
                SELECT c.id AS chunk_id,c.document_id,c.corpus_id,c.text,c.title,c.page_number,c.slide_number,
                       c.sheet_name,c.row_number,c.source_type,c.language,c.metadata,c.candidate_entities,
                       d.file_name,d.source_locator,{score_expression} AS bm25_score
                FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE {' AND '.join(clauses)} ORDER BY bm25_score DESC LIMIT %s
            """
            try:
                with self.connection() as conn, conn.cursor() as cur:
                    cur.execute(sql, [*params, limit])
                    return list(cur.fetchall())
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"pg_search BM25 query failed: {last_error}")

    @staticmethod
    def _append_filters(clauses: list[str], params: list[Any], filters: dict[str, Any]) -> None:
        if filters.get("document_ids"):
            clauses.append("c.document_id=ANY(%s)")
            params.append(list(dict.fromkeys(filters["document_ids"]))[:100])
        if filters.get("source_type"):
            clauses.append("c.source_type=ANY(%s)")
            params.append(filters["source_type"] if isinstance(filters["source_type"], list) else [filters["source_type"]])
        if filters.get("language"):
            clauses.append("c.language=%s")
            params.append(filters["language"])
        if filters.get("year_min") is not None:
            clauses.append("d.year >= %s")
            params.append(filters["year_min"])
        if filters.get("year_max") is not None:
            clauses.append("d.year <= %s")
            params.append(filters["year_max"])
        if filters.get("geography"):
            clauses.append("%s=ANY(COALESCE(d.geography,'{}'::text[]))")
            params.append(filters["geography"])
        if filters.get("domain"):
            clauses.append("COALESCE(c.metadata->>'domain','') ILIKE %s")
            params.append(f"%{filters['domain']}%")
        if filters.get("confidence_min") is not None:
            clauses.append("COALESCE(c.confidence,0) >= %s")
            params.append(filters["confidence_min"])

    def cache_get(self, kind: str, value: str) -> dict[str, Any] | None:
        key = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM query_cache WHERE cache_key=%s AND kind=%s AND expires_at>now()",
                (key, kind),
            )
            row = cur.fetchone()
            return row["payload"] if row else None

    def cache_set(self, kind: str, value: str, payload: dict[str, Any], hours: int = 24) -> None:
        key = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_cache(cache_key,kind,payload,expires_at)
                VALUES(%s,%s,%s::jsonb,now()+(%s || ' hours')::interval)
                ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,created_at=now(),expires_at=excluded.expires_at
                """,
                    (key, kind, json.dumps(payload, ensure_ascii=True), hours),
            )
            conn.commit()

    def create_agent_run(self, request: dict[str, Any], *, run_type: str = "agentic_rag") -> str:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_runs(id,run_type,request_json)
                VALUES(gen_random_uuid(),%s,%s::jsonb) RETURNING id::text
                """,
                (run_type, json.dumps(request, ensure_ascii=True, default=str)),
            )
            run_id = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO agent_events(run_id,event_type,payload) VALUES(%s,'queued',%s::jsonb)",
                (run_id, json.dumps({"status": "queued", "run_type": run_type}, ensure_ascii=True)),
            )
            conn.commit()
            return run_id

    def claim_agent_run(
        self,
        owner: str,
        *,
        run_type: str | None = None,
        run_types: list[str] | None = None,
    ) -> dict[str, Any] | None:
        allowed_types = run_types or ([run_type] if run_type else None)
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT id FROM agent_runs
                    WHERE status IN ('queued','running') AND cancel_requested=false
                      AND attempts < %s AND (lease_until IS NULL OR lease_until < now())
                      AND (%s::text[] IS NULL OR run_type=ANY(%s::text[]))
                    ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE agent_runs r SET status='running',lease_owner=%s,
                    lease_until=now()+(%s || ' seconds')::interval,
                    attempts=r.attempts+1,started_at=COALESCE(r.started_at,now()),updated_at=now()
                FROM candidate c WHERE r.id=c.id
                RETURNING r.*,r.id::text AS run_id
                """,
                (
                    self.config.job_max_attempts, allowed_types, allowed_types, owner,
                    self.config.agent_lease_seconds,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return row

    def heartbeat_agent_run(self, run_id: str, owner: str) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_runs SET lease_until=now()+(%s || ' seconds')::interval,updated_at=now()
                WHERE id=%s AND lease_owner=%s AND status='running'
                """,
                (self.config.agent_lease_seconds, run_id, owner),
            )
            conn.commit()

    def append_agent_event(
        self, run_id: str, event_type: str, payload: dict[str, Any], *, iteration: int | None = None
    ) -> int:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_events(run_id,event_type,iteration,payload)
                VALUES(%s,%s,%s,%s::jsonb) RETURNING id
                """,
                (run_id, event_type, iteration, json.dumps(payload, ensure_ascii=True, default=str)),
            )
            event_id = int(cur.fetchone()["id"])
            conn.commit()
            return event_id

    def update_agent_state(self, run_id: str, state: dict[str, Any], *, status: str | None = None) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_runs SET state_json=%s::jsonb,status=COALESCE(%s,status),updated_at=now()
                WHERE id=%s
                """,
                (json.dumps(state, ensure_ascii=True, default=str), status, run_id),
            )
            conn.commit()

    def complete_agent_run(self, run_id: str, result: dict[str, Any], state: dict[str, Any]) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_runs SET status='complete',result_json=%s::jsonb,state_json=%s::jsonb,
                    lease_owner=NULL,lease_until=NULL,updated_at=now(),finished_at=now(),error=NULL
                WHERE id=%s
                """,
                (
                    json.dumps(result, ensure_ascii=True, default=str),
                    json.dumps(state, ensure_ascii=True, default=str),
                    run_id,
                ),
            )
            conn.commit()

    def fail_agent_run(self, run_id: str, error: str, *, retryable: bool = False) -> str:
        safe_error = error[:1000]
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT attempts,cancel_requested FROM agent_runs WHERE id=%s FOR UPDATE", (run_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Agent run not found: {run_id}")
            if row["cancel_requested"]:
                status = "cancelled"
            elif retryable and row["attempts"] < self.config.job_max_attempts:
                status = "queued"
            else:
                status = "failed"
            cur.execute(
                """
                UPDATE agent_runs SET status=%s,error=%s,lease_owner=NULL,lease_until=NULL,
                    updated_at=now(),finished_at=CASE WHEN %s='queued' THEN NULL ELSE now() END
                WHERE id=%s
                """,
                (status, safe_error, status, run_id),
            )
            conn.commit()
            return status

    def request_agent_cancel(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_runs SET cancel_requested=true,
                    status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                    finished_at=CASE WHEN status='queued' THEN now() ELSE finished_at END,updated_at=now()
                WHERE id=%s RETURNING status,cancel_requested
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Agent run not found: {run_id}")
            conn.commit()
            return row

    def agent_cancel_requested(self, run_id: str) -> bool:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT cancel_requested FROM agent_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
            return bool(row and row["cancel_requested"])

    def get_agent_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT *,id::text AS run_id FROM agent_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Agent run not found: {run_id}")
            return row

    def list_agent_runs(self, *, run_type: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT *,id::text AS run_id FROM agent_runs
                WHERE run_type=%s ORDER BY created_at DESC LIMIT %s
                """,
                (run_type, max(1, min(limit, 200))),
            )
            return list(cur.fetchall())

    def list_agent_events(self, run_id: str, *, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,event_type,iteration,payload,created_at FROM agent_events
                WHERE run_id=%s AND id>%s ORDER BY id LIMIT %s
                """,
                (run_id, after_id, limit),
            )
            return list(cur.fetchall())

    def latest_qa_result(self) -> dict[str, Any] | None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text AS job_id,status,result_json,error,created_at,finished_at
                FROM agent_runs WHERE run_type='graph_qa'
                ORDER BY created_at DESC LIMIT 1
                """
            )
            return cur.fetchone()
