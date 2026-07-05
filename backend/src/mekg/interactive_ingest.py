from __future__ import annotations

import asyncio
import logging
import re
import socket
from pathlib import Path
from typing import Any

from src.yandex_embeddings import YandexEmbeddings

from .config import MEKGConfig
from .extractor import MEKGExtractor
from .models import ChunkExtraction, ParsedDocument
from .parsers import DocumentParser
from .pipeline import AdaptiveRateLimiter, FullCorpusPipeline
from .repository import MEKGRepository
from .search_store import SearchStore


logger = logging.getLogger("mekg.interactive_ingest")

CATEGORY_BY_CORPUS = {
    "internal_reports": "Доклады",
    "scientific_journals": "Журналы",
    "conference_materials": "Материалы конференций",
    "reviews": "Обзоры",
    "scientific_articles": "Статьи",
}


def infer_upload_category(file_name: str, requested: str | None) -> str:
    if requested and requested not in {"auto", "Авто"}:
        return CATEGORY_BY_CORPUS.get(requested, requested)
    folded = file_name.casefold()
    rules = (
        (("обзор", "review"), "Обзоры"),
        (("журнал", "journal", "вестник"), "Журналы"),
        (("конференц", "conference", "proceedings"), "Материалы конференций"),
        (("доклад", "report", "презентац", "protocol", "протокол"), "Доклады"),
    )
    for tokens, category in rules:
        if any(token in folded for token in tokens):
            return category
    return "Статьи"


class IngestCancelled(RuntimeError):
    pass


class InteractiveIngestor:
    """Durable fast-full ingestion for files uploaded through the demo UI."""

    def __init__(
        self,
        config: MEKGConfig | None = None,
        *,
        store: SearchStore | None = None,
        repository: MEKGRepository | None = None,
    ) -> None:
        self.config = config or MEKGConfig.from_env()
        self.store = store or SearchStore(self.config)
        self.repository = repository or MEKGRepository()
        self.parser = DocumentParser(self.config)
        self.extractor = MEKGExtractor(self.config)
        self.embeddings = YandexEmbeddings.from_env()
        self.limiter = AdaptiveRateLimiter(self.config.embed_rate)
        self.owner = f"{socket.gethostname()}:ingest"

    def _check_cancel(self, run_id: str) -> None:
        if self.store.agent_cancel_requested(run_id):
            raise IngestCancelled("Interactive ingestion cancelled")

    def _update(
        self,
        run_id: str,
        state: dict[str, Any],
        event: str,
        payload: dict[str, Any],
    ) -> None:
        state["stage"] = event
        state["last_event"] = payload
        self.store.update_agent_state(run_id, state, status="running")
        self.store.append_agent_event(run_id, event, payload)
        self.store.heartbeat_agent_run(run_id, self.owner)

    async def run(
        self,
        run_id: str,
        request: dict[str, Any],
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        files = request.get("files") or []
        if not files:
            raise ValueError("No uploaded files were supplied")
        state: dict[str, Any] = initial_state or {
            "stage": "queued",
            "files": {},
            "progress": 0,
            "warnings": [],
        }
        upload_root = self.config.artifacts_dir / "uploads"
        self.store.ensure_pipeline_run(run_id, str(upload_root), mode="interactive-upload")
        self.repository.initialize_schema()
        self.store.initialize_schema()
        results: list[dict[str, Any]] = []
        total = len(files)
        for index, item in enumerate(files, start=1):
            self._check_cancel(run_id)
            file_key = item["sha256"]
            previous = (state.get("files") or {}).get(file_key) or {}
            if previous.get("outcome") in {"complete", "complete_with_warnings", "deduplicated"}:
                results.append(previous)
                continue
            state["current_file"] = item["file_name"]
            state["file_index"] = index
            state["file_total"] = total
            self._update(run_id, state, "file_started", {
                "file_name": item["file_name"], "index": index, "total": total,
            })
            result = await self._ingest_one(run_id, item, state)
            state.setdefault("files", {})[file_key] = result
            state["progress"] = round(index / total, 4)
            self.store.update_agent_state(run_id, state, status="running")
            results.append(result)

        document_ids = list(dict.fromkeys(
            result["document_id"] for result in results if result.get("document_id")
        ))
        followup_job_id = None
        question = str(request.get("question") or "").strip()
        if question and document_ids:
            followup_job_id = self.store.create_agent_run(
                {
                    "query": question,
                    "allow_external_web": bool(request.get("allow_external_web", False)),
                    "web_profile_ids": request.get("web_profile_ids") or [],
                    "corpora": [],
                    "filters": {},
                    "focus_document_ids": document_ids,
                    "max_iterations": 3,
                    "answer_language": "ru",
                    "include_debug": True,
                },
                run_type="agentic_rag",
            )
            self.store.append_agent_event(
                run_id, "followup_queued", {"job_id": followup_job_id, "document_ids": document_ids}
            )
        warnings = list(dict.fromkeys(
            warning for result in results for warning in (result.get("warnings") or [])
        ))
        outcome = "complete_with_warnings" if warnings else "complete"
        result = {
            "job_id": run_id,
            "status": "complete",
            "outcome": outcome,
            "documents": results,
            "document_ids": document_ids,
            "followup_job_id": followup_job_id,
            "warnings": warnings,
        }
        state.update({"stage": outcome, "progress": 1, "result": result})
        self.store.finish_run(run_id, outcome)
        self.store.append_agent_event(run_id, outcome, {
            "documents": len(document_ids), "warnings": len(warnings),
            "followup_job_id": followup_job_id,
        })
        return result, state

    async def _ingest_one(
        self, run_id: str, item: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        existing = self.store.find_document_by_sha(item["sha256"])
        if existing:
            counts = self.store.document_embedding_counts(existing["id"])
            if existing.get("stage") == "complete" and counts["chunks"] == counts["embedded"]:
                value = {
                    "file_name": item["file_name"],
                    "document_id": existing["id"],
                    "outcome": "deduplicated",
                    "chunks": counts["chunks"],
                    "embedded": counts["embedded"],
                    "warnings": [],
                }
                self._update(run_id, state, "file_deduplicated", value)
                return value
            source_locator = existing["source_locator"]
            category = existing.get("category") or infer_upload_category(item["file_name"], item.get("category"))
        else:
            source_locator = f"upload://{item['sha256']}/{item['file_name']}"
            category = infer_upload_category(item["file_name"], item.get("category"))

        self._update(run_id, state, "parsing", {"file_name": item["file_name"], "progress": 0.08})
        document = await asyncio.to_thread(
            self.parser.parse,
            path,
            source_locator=source_locator,
            category=category,
            fast=True,
        )
        searchable = [
            element for element in document.elements
            if element.text.strip() and element.kind.value in {"text", "table_row"}
        ]
        if not searchable:
            raise ValueError("The document has no readable text layer; OCR is disabled in fast upload mode")
        await asyncio.to_thread(self.store.register_document, run_id, document, str(path))
        spool_dir = self.config.artifacts_dir / "uploads" / "jobs" / run_id / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(FullCorpusPipeline._write_spool, spool_dir, document)
        await asyncio.to_thread(self.store.mark_stage, run_id, document.document_id, "parsed")
        chunks = await asyncio.to_thread(self.store.upsert_document_chunks, run_id, document)
        self._update(run_id, state, "chunks_loaded", {
            "file_name": item["file_name"], "document_id": document.document_id,
            "chunks": chunks, "progress": 0.22,
        })

        missing = await asyncio.to_thread(
            self.store.document_chunks_without_embeddings, document.document_id
        )
        embedding_failures = await self._embed_chunks(run_id, document, missing, state)
        counts = await asyncio.to_thread(self.store.document_embedding_counts, document.document_id)

        job = await asyncio.to_thread(self.store.load_job, run_id, document.document_id)
        extraction_json = (job or {}).get("extraction_json") or {}
        extractions: dict[str, ChunkExtraction] = {}
        rejected: list[dict[str, Any]] = []
        for pass_name in ("first", "second"):
            part = extraction_json.get(pass_name) or {}
            rejected.extend(part.get("rejected") or [])
            for element_id, payload in (part.get("extractions") or {}).items():
                extractions[element_id] = ChunkExtraction.model_validate(payload)

        eligible = FullCorpusPipeline._evidence_elements(document)[:6]
        extraction_warning = None
        try:
            if "first" not in extraction_json:
                self._check_cancel(run_id)
                self._update(run_id, state, "extracting_first", {
                    "file_name": item["file_name"], "elements": len(eligible[:3]), "progress": 0.65,
                })
                first, first_rejected = await self.extractor.extract_batch(eligible[:3])
                payload = {
                    "extractions": {key: value.model_dump(mode="json") for key, value in first.items()},
                    "rejected": first_rejected,
                }
                await asyncio.to_thread(
                    self.store.save_extraction, run_id, document.document_id,
                    pass_number=1, element_ids=[value.id for value in eligible[:3]], payload=payload,
                )
                extractions.update(first)
                rejected.extend(first_rejected)
            if len(eligible) > 3:
                if "second" not in extraction_json:
                    self._check_cancel(run_id)
                    self._update(run_id, state, "extracting_second", {
                        "file_name": item["file_name"], "elements": len(eligible[3:6]), "progress": 0.78,
                    })
                    second, second_rejected = await self.extractor.extract_batch(eligible[3:6])
                    payload = {
                        "extractions": {key: value.model_dump(mode="json") for key, value in second.items()},
                        "rejected": second_rejected,
                    }
                    await asyncio.to_thread(
                        self.store.save_extraction, run_id, document.document_id,
                        pass_number=2, element_ids=[value.id for value in eligible[3:6]], payload=payload,
                    )
                    extractions.update(second)
                    rejected.extend(second_rejected)
            else:
                await asyncio.to_thread(
                    self.store.mark_stage, run_id, document.document_id, "second_extraction_skipped"
                )
        except Exception as exc:
            extraction_warning = f"LLM extraction degraded: {type(exc).__name__}"
            logger.warning("Interactive extraction degraded for %s", document.file_name, exc_info=True)
            await asyncio.to_thread(
                self.store.mark_stage, run_id, document.document_id, "second_extraction_skipped",
                error=extraction_warning,
            )

        self._check_cancel(run_id)
        self._update(run_id, state, "neo4j_finalizing", {
            "file_name": item["file_name"], "facts": len(extractions), "progress": 0.9,
        })
        graph = await asyncio.to_thread(
            self.repository.store_document_bundle, document, extractions, rejected
        )
        await asyncio.to_thread(
            self.store.mark_stage, run_id, document.document_id, "neo4j_committed"
        )
        warnings = []
        if embedding_failures:
            warnings.append(f"Embeddings missing for {embedding_failures} chunks; BM25 and graph are available")
        if extraction_warning:
            warnings.append(extraction_warning)
        outcome = "complete_with_warnings" if warnings else "complete"
        await asyncio.to_thread(
            self.store.mark_stage, run_id, document.document_id, outcome
        )
        value = {
            "file_name": item["file_name"],
            "document_id": document.document_id,
            "outcome": outcome,
            "chunks": counts["chunks"],
            "embedded": counts["embedded"],
            "extraction_elements": len(extractions),
            "graph": graph,
            "warnings": warnings,
        }
        self._update(run_id, state, "file_completed", value)
        return value

    async def _embed_chunks(
        self,
        run_id: str,
        document: ParsedDocument,
        rows: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> int:
        if not rows:
            return 0
        completed = 0
        failures = 0
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, min(4, self.config.embed_workers)))
        total = len(rows)

        async def embed(row: dict[str, Any]) -> None:
            nonlocal completed, failures
            async with semaphore:
                success = False
                for attempt in range(3):
                    self._check_cancel(run_id)
                    try:
                        await self.limiter.wait()
                        vector = await asyncio.to_thread(self.embeddings.embed_documents, [row["text"]])
                        await asyncio.to_thread(
                            self.store.set_embedding, row["id"], vector[0], self.embeddings.doc_model
                        )
                        self.limiter.recover()
                        success = True
                        break
                    except Exception as exc:
                        if "429" in str(exc):
                            self.limiter.throttle()
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                async with lock:
                    completed += 1
                    failures += 0 if success else 1
                    if completed == total or completed == 1 or completed % max(1, total // 10) == 0:
                        self._update(run_id, state, "embedding_progress", {
                            "file_name": document.file_name,
                            "completed": completed,
                            "total": total,
                            "failed": failures,
                            "progress": round(0.22 + 0.4 * completed / total, 3),
                        })

        await asyncio.gather(*(embed(row) for row in rows))
        return failures
