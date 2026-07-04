from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import socket
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .analytics import MEKGAnalytics
from .config import MEKGConfig
from .extractor import MEKGExtractor
from .models import ChunkExtraction, ElementKind, ParsedDocument, SourceElement
from .parsers import DocumentParser, file_sha256, stable_id
from .qa import MEKGQualityAuditor
from .repository import MEKGRepository
from .search_store import SearchStore, corpus_for_category
from src.yandex_embeddings import YandexEmbeddings


def parse_fast_document(path: str, source_locator: str, category: str | None) -> ParsedDocument:
    """Process-pool entrypoint; it deliberately builds no OCR/VLM work."""
    return DocumentParser(MEKGConfig.from_env()).parse(
        path, source_locator=source_locator, category=category, fast=True
    )


async def benchmark_fast_parser(
    corpus_root: str | Path, *, limit: int = 100, workers: int = 4
) -> dict[str, Any]:
    root = Path(corpus_root).resolve()
    paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DocumentParser.SUPPORTED
    ]
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        relative = path.relative_to(root)
        groups[relative.parts[0] if relative.parts else "other"].append(path)
    selected: list[Path] = []
    # Include large files and legacy Word before balanced round-robin sampling.
    for category in sorted(groups):
        largest = sorted(groups[category], key=lambda item: item.stat().st_size, reverse=True)[:3]
        legacy = [item for item in groups[category] if item.suffix.lower() == ".doc"][:2]
        selected.extend(item for item in [*largest, *legacy] if item not in selected)
    queues = {key: deque(sorted(values, key=lambda item: str(item).casefold())) for key, values in groups.items()}
    while len(selected) < limit and any(queues.values()):
        for category in sorted(queues):
            while queues[category] and queues[category][0] in selected:
                queues[category].popleft()
            if queues[category] and len(selected) < limit:
                selected.append(queues[category].popleft())
    selected = selected[:limit]
    started = time.monotonic()
    errors: list[dict[str, str]] = []
    pages = 0
    chunks = 0
    semaphore = asyncio.Semaphore(max(1, workers))
    pool = ProcessPoolExecutor(max_workers=max(1, workers))

    async def parse_one(path: Path) -> ParsedDocument | None:
        relative = path.relative_to(root).as_posix()
        category = Path(relative).parts[0]
        async with semaphore:
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    pool, parse_fast_document, str(path), relative, category
                )
            except Exception as exc:
                errors.append({"path": relative, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
                return None

    try:
        documents = await asyncio.gather(*(parse_one(path) for path in selected))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for document in (item for item in documents if item is not None):
        chunks += sum(element.kind in {ElementKind.TEXT, ElementKind.TABLE_ROW} for element in document.elements)
        locations = {
            element.slide_number or element.page_number or (element.sheet_name, element.row_number)
            for element in document.elements
        }
        pages += len(locations)
    elapsed = max(0.001, time.monotonic() - started)
    return {
        "documents": len(selected), "succeeded": len(selected) - len(errors), "errors": errors,
        "searchable_chunks": chunks, "logical_pages": pages, "elapsed_seconds": round(elapsed, 3),
        "documents_per_minute": round((len(selected) - len(errors)) * 60 / elapsed, 2),
        "pages_per_minute": round(pages * 60 / elapsed, 2),
        "workers": workers,
    }


class AdaptiveRateLimiter:
    def __init__(self, rate: float) -> None:
        self.rate = max(0.1, rate)
        self._interval = 1.0 / self.rate
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(self._next, now) + self._interval
        if delay:
            await asyncio.sleep(delay)

    def throttle(self) -> None:
        self._interval = min(5.0, self._interval * 1.5)

    def recover(self) -> None:
        self._interval = max(1.0 / self.rate, self._interval * 0.98)


class FullCorpusPipeline:
    """Bounded, resumable text-to-vector and evidence-graph pipeline."""

    def __init__(
        self,
        config: MEKGConfig | None = None,
        repository: MEKGRepository | None = None,
        search_store: SearchStore | None = None,
    ) -> None:
        self.config = config or MEKGConfig.from_env()
        self.repository = repository or MEKGRepository()
        self.search = search_store or SearchStore(self.config)
        self.parser = DocumentParser(self.config)
        self.extractor = MEKGExtractor(self.config)
        self.embeddings = YandexEmbeddings.from_env()
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        self.limiter = AdaptiveRateLimiter(self.config.embed_rate)

    async def run(
        self,
        corpus_root: str | Path,
        *,
        deadline_hours: float = 12,
        run_id: str | None = None,
        benchmark_limit: int = 0,
    ) -> dict[str, Any]:
        root = Path(corpus_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.repository.initialize_schema()
        search_schema = await asyncio.to_thread(self.search.initialize_schema)
        run_id = await asyncio.to_thread(
            self.search.create_or_resume_run,
            str(root),
            mode="fast-full",
            deadline_hours=deadline_hours,
            run_id=run_id,
        )
        spool_dir = self.config.artifacts_dir / "full-pipeline" / run_id
        spool_dir.mkdir(parents=True, exist_ok=True)
        inventory = await self._inventory(root, run_id, limit=benchmark_limit)

        embedding_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=2000)
        embedding_workers = [
            asyncio.create_task(self._embedding_worker(embedding_queue), name=f"embedding-{index}")
            for index in range(self.config.embed_workers)
        ]
        await self._enqueue_existing_embeddings(run_id, embedding_queue)
        parsing_done = asyncio.Event()
        first_pass_task = asyncio.create_task(
            self._extract_first_pass(run_id, spool_dir, parsing_done), name="first-extraction-pass"
        )
        try:
            await self._parse_all(run_id, spool_dir, embedding_queue)
        finally:
            parsing_done.set()
        first_pass = await first_pass_task
        second_pass = await self._extract_second_pass(run_id, spool_dir)
        finalized = await self._finalize_all(run_id, spool_dir)
        await embedding_queue.join()
        for _ in embedding_workers:
            await embedding_queue.put(None)
        await asyncio.gather(*embedding_workers)
        indexes = await asyncio.to_thread(self.search.ensure_search_indexes)
        analytics = await asyncio.to_thread(MEKGAnalytics(self.repository).run)
        auditor = MEKGQualityAuditor(self.repository, self._exporter())
        report_dir = self.config.artifacts_dir / "full-report" / run_id
        qa_json, qa_markdown = await asyncio.to_thread(auditor.write_report, report_dir)
        status = await asyncio.to_thread(self.search.status, run_id)
        incomplete = sum(
            count for stage, count in status["stages"].items() if stage not in {"complete"}
        )
        final_status = "complete" if not incomplete and status["chunks"] == status["embedded"] else "interrupted"
        await asyncio.to_thread(self.search.finish_run, run_id, final_status)
        return {
            "run_id": run_id,
            "status": final_status,
            "inventory": inventory,
            "first_pass": first_pass,
            "second_pass": second_pass,
            "finalized": finalized,
            "search": status,
            "schema": search_schema,
            "indexes": indexes,
            "analytics": analytics,
            "qa": auditor.run(),
            "reports": {"json": str(qa_json), "markdown": str(qa_markdown)},
        }

    def _exporter(self):
        from .rdf import RDFExporter

        return RDFExporter(self.repository, self.config.ontology_dir)

    async def _inventory(self, root: Path, run_id: str, *, limit: int = 0) -> dict[str, Any]:
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in DocumentParser.SUPPORTED),
            key=lambda value: str(value).casefold(),
        )
        if limit:
            paths = self._stratified(paths, root, limit)
        semaphore = asyncio.Semaphore(self.config.parse_workers)

        async def register(path: Path) -> str:
            async with semaphore:
                relative = path.relative_to(root).as_posix()
                category = Path(relative).parts[0] if Path(relative).parts else "Статьи"
                sha = await asyncio.to_thread(file_sha256, path)
                document = ParsedDocument(
                    document_id=stable_id("doc", relative.casefold()),
                    version_id=f"docver_{sha}",
                    source_locator=relative,
                    file_name=path.name,
                    file_type=path.suffix.lower().lstrip("."),
                    sha256=sha,
                    size_bytes=path.stat().st_size,
                    category=category,
                )
                await asyncio.to_thread(self.search.register_document, run_id, document, str(path))
                return corpus_for_category(category)

        corpora = await asyncio.gather(*(register(path) for path in paths))
        counts: dict[str, int] = defaultdict(int)
        for corpus in corpora:
            counts[corpus] += 1
        return {"documents": len(paths), "corpora": dict(counts), "root": str(root)}

    @staticmethod
    def _stratified(paths: list[Path], root: Path, limit: int) -> list[Path]:
        groups: dict[str, deque[Path]] = defaultdict(deque)
        for path in paths:
            relative = path.relative_to(root)
            groups[relative.parts[0] if relative.parts else "other"].append(path)
        selected: list[Path] = []
        while len(selected) < limit and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(selected) < limit:
                    selected.append(groups[key].popleft())
        return selected

    async def _enqueue_existing_embeddings(
        self, run_id: str, queue: asyncio.Queue[tuple[str, str] | None]
    ) -> None:
        offset_guard: set[str] = set()
        while True:
            rows = await asyncio.to_thread(self.search.chunks_without_embeddings, run_id, 1000)
            new_rows = [row for row in rows if row["id"] not in offset_guard]
            if not new_rows:
                break
            for row in new_rows:
                offset_guard.add(row["id"])
                await queue.put((row["id"], row["text"]))
            if len(rows) < 1000:
                break

    async def _parse_all(
        self,
        run_id: str,
        spool_dir: Path,
        embedding_queue: asyncio.Queue[tuple[str, str] | None],
    ) -> None:
        process_pool = ProcessPoolExecutor(max_workers=self.config.parse_workers)

        async def worker(index: int) -> None:
            owner = f"{self.owner}:parse:{index}"
            while True:
                claimed = await asyncio.to_thread(
                    self.search.claim_jobs, run_id, stage="discovered", owner=owner, limit=1
                )
                if not claimed:
                    return
                job = claimed[0]
                started = time.monotonic()
                try:
                    loop = asyncio.get_running_loop()
                    document = await loop.run_in_executor(
                        process_pool,
                        parse_fast_document,
                        job["source_path"],
                        self._source_locator(job["document_id"], run_id),
                        self._category(job["document_id"], run_id),
                    )
                    await asyncio.to_thread(self._write_spool, spool_dir, document)
                    await asyncio.to_thread(self.search.mark_stage, run_id, document.document_id, "parsed")
                    await asyncio.to_thread(self.search.upsert_document_chunks, run_id, document)
                    for element in document.elements:
                        if element.text.strip() and element.kind in {ElementKind.TEXT, ElementKind.TABLE_ROW}:
                            await embedding_queue.put((element.id, element.text))
                    logging.info("Parsed %s in %.2fs", document.file_name, time.monotonic() - started)
                except Exception as exc:
                    logging.exception("Fast parsing failed for %s", job["source_path"])
                    retry_stage = "discovered" if job["attempts"] < self.config.job_max_attempts else "failed"
                    await asyncio.to_thread(
                        self.search.mark_stage,
                        run_id,
                        job["document_id"],
                        retry_stage,
                        error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                    )

        try:
            await asyncio.gather(*(worker(index) for index in range(self.config.parse_workers)))
        finally:
            process_pool.shutdown(wait=True, cancel_futures=True)

    def _source_locator(self, document_id: str, run_id: str) -> str:
        row = self.search.load_job(run_id, document_id)
        return row["source_locator"] if row else document_id

    def _category(self, document_id: str, run_id: str) -> str | None:
        row = self.search.load_job(run_id, document_id)
        return row["category"] if row else None

    async def _embedding_worker(self, queue: asyncio.Queue[tuple[str, str] | None]) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            chunk_id, text = item
            try:
                for attempt in range(3):
                    try:
                        await self.limiter.wait()
                        vector = await asyncio.to_thread(self.embeddings.embed_documents, [text])
                        await asyncio.to_thread(
                            self.search.set_embedding, chunk_id, vector[0], self.embeddings.doc_model
                        )
                        self.limiter.recover()
                        break
                    except Exception as exc:
                        if "429" in str(exc):
                            self.limiter.throttle()
                        if attempt == 2:
                            logging.error("Embedding failed for %s: %s", chunk_id, exc)
                        else:
                            await asyncio.sleep(2 ** attempt)
            finally:
                queue.task_done()

    async def _extract_first_pass(
        self, run_id: str, spool_dir: Path, producers_done: asyncio.Event | None = None
    ) -> dict[str, int]:
        completed = 0
        failed = 0
        lock = asyncio.Lock()

        async def worker(index: int) -> None:
            nonlocal completed, failed
            owner = f"{self.owner}:llm:{index}"
            while True:
                jobs = await asyncio.to_thread(
                    self.search.claim_jobs, run_id, stage="chunks_loaded", owner=owner, limit=1
                )
                if not jobs:
                    if producers_done is not None and not producers_done.is_set():
                        await asyncio.sleep(1)
                        continue
                    return
                job = jobs[0]
                try:
                    document = await asyncio.to_thread(self._read_spool, spool_dir, job["document_id"])
                    eligible = self._evidence_elements(document)[:6]
                    extracted, rejected = await self.extractor.extract_batch(eligible[:3])
                    payload = {
                        "extractions": {key: value.model_dump(mode="json") for key, value in extracted.items()},
                        "rejected": rejected,
                    }
                    await asyncio.to_thread(
                        self.search.save_extraction,
                        run_id,
                        document.document_id,
                        pass_number=1,
                        element_ids=[item.id for item in eligible[:3]],
                        payload=payload,
                    )
                    async with lock:
                        completed += 1
                except Exception as exc:
                    logging.exception("First extraction failed for %s", job["document_id"])
                    retry_stage = "chunks_loaded" if job["attempts"] < self.config.job_max_attempts else "failed"
                    await asyncio.to_thread(
                        self.search.mark_stage, run_id, job["document_id"], retry_stage,
                        error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                    )
                    async with lock:
                        failed += 1

        await asyncio.gather(*(worker(index) for index in range(self.config.llm_workers)))
        return {"completed": completed, "failed_attempts": failed}

    async def _extract_second_pass(self, run_id: str, spool_dir: Path) -> dict[str, int]:
        run = await asyncio.to_thread(self.search.get_run, run_id)
        cutoff = run["deadline_at"] - timedelta(hours=1.5)
        jobs = await asyncio.to_thread(self.search.list_jobs, run_id, ["first_extraction_done"])
        by_corpus: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for job in jobs:
            full = await asyncio.to_thread(self.search.load_job, run_id, job["document_id"])
            by_corpus[(full or {}).get("corpus_id", "other")].append(job)
        ordered: list[dict[str, Any]] = []
        while any(by_corpus.values()):
            for corpus in sorted(by_corpus):
                if by_corpus[corpus]:
                    ordered.append(by_corpus[corpus].popleft())
        seconds = max(0.0, (cutoff - datetime.now(timezone.utc)).total_seconds())
        capacity = min(len(ordered), int(seconds * 3 / 60))
        selected = ordered[:capacity]
        skipped = ordered[capacity:]
        for job in skipped:
            await asyncio.to_thread(
                self.search.mark_stage, run_id, job["document_id"], "second_extraction_skipped"
            )

        semaphore = asyncio.Semaphore(self.config.llm_workers)
        completed = 0

        async def process(job: dict[str, Any]) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    document = await asyncio.to_thread(self._read_spool, spool_dir, job["document_id"])
                    eligible = self._evidence_elements(document)[:6]
                    extracted, rejected = await self.extractor.extract_batch(eligible[3:6])
                    payload = {
                        "extractions": {key: value.model_dump(mode="json") for key, value in extracted.items()},
                        "rejected": rejected,
                    }
                    await asyncio.to_thread(
                        self.search.save_extraction, run_id, document.document_id,
                        pass_number=2, element_ids=[item.id for item in eligible[3:6]], payload=payload,
                    )
                    completed += 1
                except Exception as exc:
                    logging.exception("Second extraction failed for %s", job["document_id"])
                    await asyncio.to_thread(
                        self.search.mark_stage, run_id, job["document_id"], "second_extraction_skipped",
                        error=f"second pass skipped after error: {type(exc).__name__}: {str(exc)[:500]}",
                    )

        await asyncio.gather(*(process(job) for job in selected))
        return {"selected": len(selected), "completed": completed, "skipped": len(skipped)}

    async def _finalize_all(self, run_id: str, spool_dir: Path) -> dict[str, int]:
        jobs = await asyncio.to_thread(
            self.search.list_jobs, run_id, ["second_extraction_done", "second_extraction_skipped"]
        )
        completed = 0
        failed = 0
        for job in jobs:
            try:
                document = await asyncio.to_thread(self._read_spool, spool_dir, job["document_id"])
                payload = job.get("extraction_json") or {}
                extractions: dict[str, ChunkExtraction] = {}
                rejected: list[dict[str, Any]] = []
                for pass_name in ("first", "second"):
                    part = payload.get(pass_name) or {}
                    rejected.extend(part.get("rejected") or [])
                    for element_id, extraction in (part.get("extractions") or {}).items():
                        extractions[element_id] = ChunkExtraction.model_validate(extraction)
                await asyncio.to_thread(
                    self.repository.store_document_bundle, document, extractions, rejected
                )
                await asyncio.to_thread(
                    self.search.mark_stage, run_id, document.document_id, "neo4j_committed"
                )
                await asyncio.to_thread(self.search.mark_stage, run_id, document.document_id, "complete")
                completed += 1
            except Exception as exc:
                logging.exception("Neo4j finalization failed for %s", job["document_id"])
                await asyncio.to_thread(
                    self.search.mark_stage, run_id, job["document_id"], "failed",
                    error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                )
                failed += 1
        return {"completed": completed, "failed": failed}

    @staticmethod
    def _evidence_elements(document: ParsedDocument) -> list[SourceElement]:
        eligible = [
            item for item in document.elements
            if item.text.strip() and len(item.text.strip()) >= 40
            and item.kind in {ElementKind.TEXT, ElementKind.TABLE_ROW}
        ]

        def score(item: SourceElement) -> tuple[int, int]:
            text = item.text.casefold()
            numeric = sum(character.isdigit() for character in text)
            terms = sum(term in text for term in (
                "experiment", "исслед", "temperature", "температур", "concentration", "концентрац",
                "recovery", "извлеч", "process", "процесс", "technology", "технолог", "вывод",
                "conclusion", "recommend", "рекоменд", "%", "mg/l", "мг/л",
            ))
            return (terms * 10 + min(numeric, 20) + (5 if item.kind == ElementKind.TABLE_ROW else 2), len(text))

        return sorted(eligible, key=score, reverse=True)

    @staticmethod
    def _write_spool(directory: Path, document: ParsedDocument) -> None:
        target = directory / f"{document.document_id}.json.gz"
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            stream.write(document.model_dump_json())
        temporary.replace(target)

    @staticmethod
    def _read_spool(directory: Path, document_id: str) -> ParsedDocument:
        with gzip.open(directory / f"{document_id}.json.gz", "rt", encoding="utf-8") as stream:
            return ParsedDocument.model_validate_json(stream.read())
