from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mekg.config import MEKGConfig
from src.mekg.models import ChunkExtraction, ParsedDocument
from src.mekg.repository import MEKGRepository
from src.mekg.search_store import SearchStore


def _read_spool(spool_dir: Path, document_id: str) -> ParsedDocument:
    with gzip.open(spool_dir / f"{document_id}.json.gz", "rt", encoding="utf-8") as stream:
        return ParsedDocument.model_validate_json(stream.read())


def _parse_extractions(payload: Any) -> tuple[dict[str, ChunkExtraction], list[dict[str, Any]], int]:
    if isinstance(payload, str):
        payload = json.loads(payload) if payload else {}
    payload = payload or {}

    extractions: dict[str, ChunkExtraction] = {}
    rejected: list[dict[str, Any]] = []
    invalid = 0
    for pass_name in ("first", "second"):
        part = payload.get(pass_name) or {}
        rejected.extend(part.get("rejected") or [])
        for element_id, extraction in (part.get("extractions") or {}).items():
            try:
                extractions[element_id] = ChunkExtraction.model_validate(extraction)
            except Exception as exc:  # keep source graph even if one LLM item is malformed
                invalid += 1
                rejected.append(
                    {
                        "kind": "invalid_checkpoint_extraction",
                        "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                        "payload": {"element_id": element_id},
                    }
                )
    return extractions, rejected, invalid


def _load_targets(store: SearchStore, run_id: str, limit: int | None) -> list[dict[str, Any]]:
    sql = """
        SELECT
            j.document_id,
            j.stage,
            j.extraction_json,
            count(c.id)::int AS chunk_count,
            count(c.embedding)::int AS embedded_count
        FROM document_jobs j
        JOIN chunks c ON c.document_id = j.document_id
        WHERE j.run_id = %s
        GROUP BY j.document_id, j.stage, j.extraction_json, j.updated_at
        HAVING count(c.id) > 0
        ORDER BY
            CASE WHEN j.stage = 'first_extraction_done' THEN 0 ELSE 1 END,
            j.updated_at,
            j.document_id
    """
    params: list[Any] = [run_id]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with store.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge already parsed Postgres chunks/checkpointed extractions into Neo4j without touching pipeline state."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spool-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()

    config = MEKGConfig.from_env()
    spool_dir = args.spool_dir or config.artifacts_dir / "full-pipeline" / args.run_id
    progress_file = args.progress_file or spool_dir / "partial-neo4j-finalize.progress.json"

    started = time.time()
    stats: dict[str, Any] = {
        "run_id": args.run_id,
        "spool_dir": str(spool_dir),
        "total_targets": 0,
        "completed": 0,
        "failed": 0,
        "missing_spool": 0,
        "with_extractions": 0,
        "invalid_extractions": 0,
        "nodes_merged": 0,
        "relationships_merged": 0,
        "started_at_epoch": started,
        "last_document_id": None,
        "last_error": None,
    }

    store = SearchStore(config)
    repo = MEKGRepository()
    try:
        store.open()
        repo.initialize_schema()
        targets = _load_targets(store, args.run_id, args.limit)
        stats["total_targets"] = len(targets)
        _write_progress(progress_file, stats)
        print(
            json.dumps(
                {"event": "start", "run_id": args.run_id, "targets": len(targets), "spool_dir": str(spool_dir)},
                ensure_ascii=False,
            ),
            flush=True,
        )

        for index, row in enumerate(targets, start=1):
            document_id = row["document_id"]
            stats["last_document_id"] = document_id
            if not (spool_dir / f"{document_id}.json.gz").exists():
                stats["missing_spool"] += 1
                print(
                    json.dumps({"event": "missing_spool", "index": index, "document_id": document_id}, ensure_ascii=False),
                    flush=True,
                )
                _write_progress(progress_file, stats)
                continue

            try:
                document = _read_spool(spool_dir, document_id)
                extractions, rejected, invalid = _parse_extractions(row.get("extraction_json"))
                if extractions:
                    stats["with_extractions"] += 1
                stats["invalid_extractions"] += invalid
                summary = repo.store_document_bundle(document, extractions, rejected)
                repo.write(
                    """
                    MATCH (d:MEKG:Document {id:$id})
                    SET d.partial_graph_run=$run_id,
                        d.partial_graph_stage=$stage,
                        d.partial_graph_chunks=$chunk_count,
                        d.partial_graph_embedded=$embedded_count,
                        d.partial_graph_finalized_at=datetime()
                    """,
                    {
                        "id": document_id,
                        "run_id": args.run_id,
                        "stage": row["stage"],
                        "chunk_count": row["chunk_count"],
                        "embedded_count": row["embedded_count"],
                    },
                )
                stats["completed"] += 1
                stats["nodes_merged"] += int(summary.get("node_count") or 0)
                stats["relationships_merged"] += int(summary.get("relationship_count") or 0)
                if index == 1 or index % 10 == 0 or index == len(targets):
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "index": index,
                                "targets": len(targets),
                                "completed": stats["completed"],
                                "failed": stats["failed"],
                                "missing_spool": stats["missing_spool"],
                                "document_id": document_id,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                _write_progress(progress_file, stats)
            except Exception as exc:
                stats["failed"] += 1
                stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
                print(
                    json.dumps(
                        {
                            "event": "failed_document",
                            "index": index,
                            "document_id": document_id,
                            "error": stats["last_error"],
                            "traceback": traceback.format_exc(limit=3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                _write_progress(progress_file, stats)

        stats["finished_at_epoch"] = time.time()
        stats["elapsed_seconds"] = round(stats["finished_at_epoch"] - started, 3)
        print(json.dumps({"event": "finish", **stats}, ensure_ascii=False), flush=True)
        _write_progress(progress_file, stats)
        return 0 if stats["failed"] == 0 else 2
    finally:
        try:
            store.close()
        finally:
            repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
