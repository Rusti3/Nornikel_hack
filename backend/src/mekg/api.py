from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from .config import MEKGConfig
from .models import (
    FindContradictionsRequest,
    FindExpertsRequest,
    FindExperimentsRequest,
    FindKnowledgeGapsRequest,
    FindTechnologiesRequest,
    CrossCorpusSearchRequest,
    AgenticRAGRequest,
    AgentJobAccepted,
    GetEvidencePackRequest,
    ResolveEntityRequest,
    SemanticSearchRequest,
    ReviewDecision,
    WebSearchRequest,
)
from .service import MEKGService
from .parsers import DocumentParser
from .search_store import CORPORA, SearchStore
from .web_search import YandexWebSearchClient


router = APIRouter(prefix="/api/mekg/v1", tags=["MEKG"])
_service: MEKGService | None = None
_lock = threading.Lock()


def get_service() -> MEKGService:
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = MEKGService()
    return _service


@router.get("/profile")
def profile():
    config = MEKGConfig.from_env()
    return {
        "profile": "mekg",
        "ontology_version": "1.0.0",
        "llm_model": config.llm_model,
        "vision_model": config.vision_model,
        "embedding_dimensions": int(os.getenv("EMBED_DIMENSIONS", "768")),
        "data_logging": config.data_logging,
        "supported_formats": sorted(DocumentParser.SUPPORTED),
    }


@router.post("/initialize")
def initialize():
    return get_service().initialize()


@router.post("/preflight")
async def preflight(remote: bool = True):
    return await get_service().preflight(remote=remote)


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    source_locator: str | None = Form(None),
    category: str | None = Form(None),
    enable_vision: bool = Form(True),
    enable_extraction: bool = Form(True),
):
    suffix = Path(file.filename or "source").suffix.lower()
    if suffix not in get_service().parser.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported MEKG source format: {suffix}")
    upload_dir = get_service().config.artifacts_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}{suffix}"
    target = upload_dir / safe_name
    try:
        with target.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        return await get_service().ingest_path(
            target,
            source_locator=source_locator or file.filename,
            category=category,
            enable_vision=enable_vision,
            enable_extraction=enable_extraction,
        )
    finally:
        target.unlink(missing_ok=True)


@router.post("/resolve_entity")
def resolve_entity(request: ResolveEntityRequest):
    return get_service().resolve_entity(request).model_dump()


@router.post("/find_experiments")
def find_experiments(request: FindExperimentsRequest):
    return get_service().find_experiments(request).model_dump()


@router.post("/find_technologies")
def find_technologies(request: FindTechnologiesRequest):
    return get_service().find_technologies(request).model_dump()


@router.post("/get_evidence_pack")
def get_evidence_pack(request: GetEvidencePackRequest):
    return get_service().get_evidence_pack(request).model_dump()


@router.post("/find_contradictions")
def find_contradictions(request: FindContradictionsRequest):
    return get_service().find_contradictions(request).model_dump()


@router.post("/find_knowledge_gaps")
def find_knowledge_gaps(request: FindKnowledgeGapsRequest):
    return get_service().find_knowledge_gaps(request).model_dump()


@router.post("/find_experts")
def find_experts(request: FindExpertsRequest):
    return get_service().find_experts(request).model_dump()


@router.post("/semantic_search")
def semantic_search(request: SemanticSearchRequest):
    return get_service().semantic_search(request).model_dump()


@router.post("/cross_corpus_search")
def cross_corpus_search(request: CrossCorpusSearchRequest):
    return get_service().cross_corpus_search(request).model_dump()


@router.get("/graph/neighbors/{node_id}")
def graph_neighbors(node_id: str):
    rows = get_service().repository.query(
        """
        MATCH (n:MEKG {id: $node_id})
        OPTIONAL MATCH (n)-[r]-(m:MEKG)
        WITH n,
             [node IN collect(DISTINCT m) WHERE node IS NOT NULL] AS neighbours,
             [rel IN collect(DISTINCT r) WHERE rel IS NOT NULL] AS relationships
        RETURN [node IN [n] + neighbours | {
            element_id: elementId(node),
            labels: [label IN labels(node) WHERE NOT label IN ['MEKG', 'CanonicalEntity']],
            properties: apoc.map.removeKeys(properties(node), ['embedding', 'text', 'summary'])
        }] AS nodes,
        [rel IN relationships | {
            element_id: elementId(rel),
            start_node_element_id: elementId(startNode(rel)),
            end_node_element_id: elementId(endNode(rel)),
            type: type(rel)
        }] AS relationships
        """,
        {"node_id": node_id},
    )
    return rows[0] if rows else {"nodes": [], "relationships": []}


@router.get("/web_search/profiles")
def web_search_profiles():
    return {"profiles": YandexWebSearchClient.profiles(), "max_domains_per_call": 5, "region": "213"}


@router.post("/web_search")
def web_search(request: WebSearchRequest):
    return YandexWebSearchClient(MEKGConfig.from_env()).search(request).model_dump(mode="json")


def _public_agent_run(row: dict) -> dict:
    return {
        "job_id": row["run_id"],
        "run_type": row["run_type"],
        "status": row["status"],
        "state": row.get("state_json") or {},
        "result": row.get("result_json"),
        "error": row.get("error"),
        "attempts": row.get("attempts", 0),
        "cancel_requested": row.get("cancel_requested", False),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "updated_at": row.get("updated_at"),
        "finished_at": row.get("finished_at"),
    }


@router.post("/agentic_rag/jobs", response_model=AgentJobAccepted, status_code=202)
def start_agentic_rag(request: AgenticRAGRequest):
    store = SearchStore(MEKGConfig.from_env())
    try:
        store.initialize_schema()
        job_id = store.create_agent_run(request.model_dump(mode="json"), run_type="agentic_rag")
        return AgentJobAccepted(
            job_id=job_id,
            status="queued",
            events_url=f"/api/mekg/v1/agentic_rag/jobs/{job_id}/events",
            status_url=f"/api/mekg/v1/agentic_rag/jobs/{job_id}",
        )
    finally:
        store.close()


@router.get("/agentic_rag/jobs/{job_id}")
def get_agentic_rag_job(job_id: uuid.UUID):
    job_key = str(job_id)
    store = SearchStore(MEKGConfig.from_env())
    try:
        return _public_agent_run(store.get_agent_run(job_key))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        store.close()


@router.post("/agentic_rag/jobs/{job_id}/cancel")
def cancel_agentic_rag_job(job_id: uuid.UUID):
    job_key = str(job_id)
    store = SearchStore(MEKGConfig.from_env())
    try:
        value = store.request_agent_cancel(job_key)
        store.append_agent_event(job_key, "cancel_requested", {"status": value["status"]})
        return {"job_id": job_key, **value}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        store.close()


@router.get("/agentic_rag/jobs/{job_id}/events")
async def agentic_rag_events(job_id: uuid.UUID, request: Request, after: int = 0):
    job_key = str(job_id)
    try:
        header_id = int(request.headers.get("last-event-id", "0") or 0)
    except ValueError:
        header_id = 0
    cursor = max(after, header_id)

    async def generate():
        nonlocal cursor
        store = SearchStore(MEKGConfig.from_env())
        try:
            await asyncio.to_thread(store.get_agent_run, job_key)
            while not await request.is_disconnected():
                events = await asyncio.to_thread(store.list_agent_events, job_key, after_id=cursor)
                for item in events:
                    cursor = int(item["id"])
                    yield {
                        "id": str(cursor),
                        "event": item["event_type"],
                        "data": json.dumps(
                            {
                                "id": cursor,
                                "type": item["event_type"],
                                "iteration": item.get("iteration"),
                                "payload": item.get("payload") or {},
                                "created_at": item["created_at"].isoformat(),
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                run = await asyncio.to_thread(store.get_agent_run, job_key)
                if run["status"] in {"complete", "failed", "cancelled"} and not events:
                    break
                await asyncio.sleep(0.6)
        except KeyError:
            yield {"event": "error", "data": json.dumps({"detail": "Agent job not found"})}
        finally:
            store.close()

    return EventSourceResponse(generate(), ping=20)


@router.get("/pipeline/status")
def pipeline_status(run_id: str | None = None):
    store = SearchStore(MEKGConfig.from_env())
    try:
        return store.status(run_id)
    finally:
        store.close()


@router.get("/readiness")
def readiness(run_id: str | None = None):
    warnings: list[str] = []
    store = SearchStore(MEKGConfig.from_env())
    try:
        pipeline = store.status(run_id)
        stages = pipeline.get("stages") or {}
        failed_jobs = int(stages.get("failed") or 0)
        discovered_jobs = int(stages.get("discovered") or 0)
        chunks = int(pipeline.get("chunks") or 0)
        embedded = int(pipeline.get("embedded") or 0)
        if failed_jobs:
            warnings.append(f"Индексация неполная: {failed_jobs} документов завершились ошибкой.")
        if discovered_jobs:
            warnings.append(f"Индексация неполная: {discovered_jobs} документов ожидают обработки/resume.")
        if embedded < chunks:
            warnings.append(f"Embedding coverage: {embedded} из {chunks} чанков; dense search покрывает не весь корпус.")
    except Exception as exc:
        pipeline = {"run_id": run_id, "stages": {}, "chunks": 0, "embedded": 0}
        warnings.append(f"Postgres readiness unavailable: {type(exc).__name__}")
    finally:
        store.close()

    graph: dict[str, object] = {"metrics": {}, "top": {}, "coverage": []}
    try:
        repository = get_service().repository
        metrics = {
            "documents": "MATCH (n:MEKG:Document) RETURN count(n) AS value",
            "chunks": "MATCH (n:MEKG:Chunk) RETURN count(n) AS value",
            "nodes": "MATCH (n:MEKG) RETURN count(n) AS value",
            "relationships": "MATCH (:MEKG)-[r]->(:MEKG) RETURN count(r) AS value",
            "claims": "MATCH (n:MEKG:Claim) RETURN count(n) AS value",
            "experiments": "MATCH (n:MEKG:Experiment) RETURN count(n) AS value",
            "measurements": "MATCH (n:MEKG:Measurement) RETURN count(n) AS value",
            "conditions": "MATCH (n:MEKG:Condition) RETURN count(n) AS value",
            "experts": "MATCH (n:MEKG:Expert) RETURN count(n) AS value",
            "contradictions": "MATCH (n:MEKG:Contradiction) RETURN count(n) AS value",
            "knowledge_gaps": "MATCH (n:MEKG:KnowledgeGap) RETURN count(n) AS value",
            "review_items": (
                "MATCH (n) WHERE (n:MEKGStaging) OR (n:MEKG AND n.validation_status IN ['raw_extracted','needs_review']) "
                "RETURN count(n) AS value"
            ),
        }
        graph["metrics"] = {
            name: int((repository.query(cypher)[0] or {}).get("value") or 0)
            for name, cypher in metrics.items()
        }
        top_queries = {
            "materials": (
                "MATCH (n:MEKG:CanonicalEntity:Material) "
                "OPTIONAL MATCH (n)-[r]-() RETURN n.canonical_name AS name,count(r) AS links ORDER BY links DESC LIMIT 8"
            ),
            "processes": (
                "MATCH (n:MEKG:CanonicalEntity:Process) "
                "OPTIONAL MATCH (n)-[r]-() RETURN n.canonical_name AS name,count(r) AS links ORDER BY links DESC LIMIT 8"
            ),
            "equipment": (
                "MATCH (n:MEKG:CanonicalEntity:Equipment) "
                "OPTIONAL MATCH (n)-[r]-() RETURN n.canonical_name AS name,count(r) AS links ORDER BY links DESC LIMIT 8"
            ),
            "experts": (
                "MATCH (n:MEKG:Expert) "
                "OPTIONAL MATCH (n)-[r]-() RETURN coalesce(n.canonical_name,n.name,n.id) AS name,count(r) AS links ORDER BY links DESC LIMIT 8"
            ),
        }
        graph["top"] = {name: repository.query(cypher) for name, cypher in top_queries.items()}
        graph["coverage"] = repository.query(
            """
            MATCH (d:MEKG:Document)
            WITH coalesce(d.category,'unknown') AS category,count(d) AS documents
            RETURN category,documents ORDER BY documents DESC
            """
        )
        total_jobs = sum(int(value or 0) for value in (pipeline.get("stages") or {}).values())
        graph_documents = int((graph.get("metrics") or {}).get("documents") or 0)
        if total_jobs and graph_documents < total_jobs:
            warnings.append(f"Neo4j source graph: {graph_documents} из {total_jobs} обнаруженных документов.")
    except Exception as exc:
        warnings.append(f"Neo4j readiness unavailable: {type(exc).__name__}")

    store = SearchStore(MEKGConfig.from_env())
    try:
        qa = store.latest_qa_result() or {"status": "not_run", "result_json": None}
    except Exception as exc:
        warnings.append(f"QA snapshot unavailable: {type(exc).__name__}")
        qa = {"status": "unknown", "result_json": None}
    finally:
        store.close()

    checklist = [
        {"id": "multiparameter_queries", "title": "Многопараметрические запросы", "status": "implemented", "evidence": "Agentic analyzer + filters + hard gates"},
        {"id": "ru_en_synonyms", "title": "RU/EN синонимы", "status": "implemented", "evidence": "Terms, aliases, bilingual rewrites"},
        {"id": "numeric_units", "title": "Числовые ограничения с единицами", "status": "implemented", "evidence": "Unit normalization + numeric hard gate"},
        {"id": "geography", "title": "РФ vs зарубежная практика", "status": "implemented", "evidence": "Geography filters and answer gates"},
        {"id": "recent_years", "title": "Последние годы / год источника", "status": "implemented", "evidence": "Year filters and time hard gate"},
        {"id": "experts", "title": "Эксперты и лаборатории", "status": "partial", "evidence": "Expert extraction/search exists; final coverage depends on corpus completion"},
        {"id": "gaps_contradictions", "title": "Пробелы и противоречия", "status": "implemented", "evidence": "Agentic gaps + graph analytics/QA"},
        {"id": "exports", "title": "Экспорт Markdown/JSON-LD", "status": "partial", "evidence": "Markdown answer + JSON-LD/Turtle export; PDF report is roadmap"},
    ]
    return {
        "pipeline": pipeline,
        "graph": graph,
        "qa": qa,
        "corpora": [{"id": item[0], "title": item[1], "weight": item[2], "description": item[3]} for item in CORPORA],
        "checklist": checklist,
        "warnings": warnings,
    }


@router.post("/review/{fact_id}")
def review(fact_id: str, decision: ReviewDecision):
    try:
        return get_service().review(fact_id, decision)
    except KeyError:
        raise HTTPException(status_code=404, detail="MEKG fact not found")


@router.get("/qa")
def qa():
    store = SearchStore(MEKGConfig.from_env())
    try:
        snapshot = store.latest_qa_result()
        if snapshot and snapshot.get("status") == "complete" and snapshot.get("result_json"):
            return {**snapshot["result_json"], "job_id": snapshot["job_id"], "cached": True}
        return {
            "passed": False,
            "pending": bool(snapshot and snapshot.get("status") in {"queued", "running"}),
            "job_id": snapshot.get("job_id") if snapshot else None,
            "status": snapshot.get("status") if snapshot else "not_run",
            "metrics": {},
            "shacl": {"violations": None},
        }
    finally:
        store.close()


@router.post("/qa/run", status_code=202)
def run_qa():
    store = SearchStore(MEKGConfig.from_env())
    try:
        store.initialize_schema()
        current = store.latest_qa_result()
        if current and current.get("status") in {"queued", "running"}:
            return {"job_id": current["job_id"], "status": current["status"]}
        job_id = store.create_agent_run({}, run_type="graph_qa")
        return {"job_id": job_id, "status": "queued"}
    finally:
        store.close()


@router.get("/review-queue")
def review_queue(limit: int = 100):
    return {"items": get_service().review_queue(limit=max(1, min(limit, 500)))}


@router.get("/export/{format}", response_class=PlainTextResponse)
def export(format: str):
    try:
        value = get_service().export(format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    media_type = "application/ld+json" if format in {"jsonld", "json-ld"} else "text/turtle"
    return PlainTextResponse(value, media_type=media_type)
