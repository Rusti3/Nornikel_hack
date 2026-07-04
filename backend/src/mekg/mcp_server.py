from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from .api import get_service
from .models import (
    AgenticRAGRequest,
    ConditionFilter,
    CrossCorpusSearchRequest,
    FindContradictionsRequest,
    FindExpertsRequest,
    FindExperimentsRequest,
    FindKnowledgeGapsRequest,
    FindTechnologiesRequest,
    GetEvidencePackRequest,
    InputStream,
    ResolveEntityRequest,
    TechnologyTarget,
    WebSearchRequest,
)
from .config import MEKGConfig
from .search_store import SearchStore
from .web_search import YandexWebSearchClient


mcp = FastMCP(
    "Metallurgical Evidence Knowledge Graph",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)


@mcp.tool()
def resolve_entity(text: str, type_hint: str | None = None, language: str | None = None) -> dict[str, Any]:
    """Resolve a Russian or English term to a canonical MEKG entity."""
    return get_service().resolve_entity(ResolveEntityRequest(text=text, type_hint=type_hint, language=language)).model_dump()


@mcp.tool()
def find_experiments(
    material: str | None = None,
    process: str | None = None,
    conditions: list[dict[str, Any]] | None = None,
    property: str | None = None,
    geo_scope: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Find evidence-backed experiments by material, process, numeric conditions and property."""
    request = FindExperimentsRequest(
        material=material,
        process=process,
        conditions=[ConditionFilter.model_validate(item) for item in (conditions or [])],
        property=property,
        geo_scope=geo_scope,
        year_min=year_min,
        year_max=year_max,
        limit=limit,
    )
    return get_service().find_experiments(request).model_dump()


@mcp.tool()
def find_technologies(
    problem: str,
    input_stream: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    geo_scope: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Find applicable technologies with limitations, metrics and evidence."""
    request = FindTechnologiesRequest(
        problem=problem,
        input_stream=InputStream.model_validate(input_stream) if input_stream else None,
        target=TechnologyTarget.model_validate(target) if target else None,
        geo_scope=geo_scope,
        limit=limit,
    )
    return get_service().find_technologies(request).model_dump()


@mcp.tool()
def get_evidence_pack(claim_id: str) -> dict[str, Any]:
    """Return a claim with supporting and contradicting source evidence."""
    return get_service().get_evidence_pack(GetEvidencePackRequest(claim_id=claim_id)).model_dump()


@mcp.tool()
def find_contradictions(
    topic: str | None = None,
    entity_ids: list[str] | None = None,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Find explicit contradiction objects and their involved claims."""
    return get_service().find_contradictions(
        FindContradictionsRequest(topic=topic, entity_ids=entity_ids or [], status=status, limit=limit)
    ).model_dump()


@mcp.tool()
def find_knowledge_gaps(
    material: str | None = None,
    process: str | None = None,
    condition: str | None = None,
    property: str | None = None,
    geo_region: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Find persisted or query-derived gaps in experimental evidence."""
    return get_service().find_knowledge_gaps(
        FindKnowledgeGapsRequest(
            material=material,
            process=process,
            condition=condition,
            property=property,
            geo_region=geo_region,
            limit=limit,
        )
    ).model_dump()


@mcp.tool()
def find_experts(topic: str, process: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Find evidence-derived experts and explain the expertise score."""
    return get_service().find_experts(FindExpertsRequest(topic=topic, process=process, limit=limit)).model_dump()


@mcp.tool()
def cross_corpus_search(
    query: str,
    intent: str = "research",
    target_slots: list[str] | None = None,
    corpora: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    numeric_mode: str = "boost",
    max_corpora: int = 4,
    k_per_corpus: int = 12,
    final_k: int = 20,
) -> dict[str, Any]:
    """Retrieve cross-corpus evidence with dense, BM25, routing, rerank and diagnostics; never answers the query."""
    request = CrossCorpusSearchRequest.model_validate({
        "query": query,
        "intent": intent,
        "target_slots": target_slots or [],
        "corpora": corpora or [],
        "filters": filters or {},
        "numeric_mode": numeric_mode,
        "max_corpora": max_corpora,
        "k_per_corpus": k_per_corpus,
        "final_k": final_k,
    })
    return get_service().cross_corpus_search(request).model_dump()


@mcp.tool()
def web_search_research(
    query: str,
    profile_ids: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    search_context_size: str = "high",
    region: str = "213",
) -> dict[str, Any]:
    """Search public web sources through Yandex; a profile or custom allowlist is limited to five domains."""
    request = WebSearchRequest.model_validate({
        "query": query,
        "profile_ids": profile_ids or [],
        "allowed_domains": allowed_domains or [],
        "search_context_size": search_context_size,
        "region": region,
    })
    return YandexWebSearchClient(MEKGConfig.from_env()).search(request).model_dump(mode="json")


@mcp.tool()
def start_agentic_rag(
    query: str,
    allow_external_web: bool = False,
    web_profile_ids: list[str] | None = None,
    corpora: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Start the durable evidence state machine and return a job id; web requires explicit consent."""
    request = AgenticRAGRequest.model_validate({
        "query": query,
        "allow_external_web": allow_external_web,
        "web_profile_ids": web_profile_ids or [],
        "corpora": corpora or [],
        "filters": filters or {},
        "max_iterations": max_iterations,
    })
    store = SearchStore(MEKGConfig.from_env())
    try:
        store.initialize_schema()
        job_id = store.create_agent_run(request.model_dump(mode="json"))
        return {"job_id": job_id, "status": "queued"}
    finally:
        store.close()


@mcp.tool()
def get_agentic_rag_job(job_id: str) -> dict[str, Any]:
    """Get persisted state, result, warnings and status for an agentic RAG job."""
    store = SearchStore(MEKGConfig.from_env())
    try:
        row = store.get_agent_run(job_id)
        return {
            "job_id": row["run_id"], "status": row["status"], "state": row.get("state_json") or {},
            "result": row.get("result_json"), "error": row.get("error"),
        }
    finally:
        store.close()


@mcp.tool()
def cancel_agentic_rag_job(job_id: str) -> dict[str, Any]:
    """Request cooperative cancellation of an agentic RAG job."""
    store = SearchStore(MEKGConfig.from_env())
    try:
        return {"job_id": job_id, **store.request_agent_cancel(job_id)}
    finally:
        store.close()


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def mcp_lifespan(_app):
    async with mcp.session_manager.run():
        yield
