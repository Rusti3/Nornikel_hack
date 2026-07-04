from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .config import MEKGConfig
from .llm_provider import build_agent_llm
from .models import (
    AgentEvidenceItem,
    AgentEvidencePack,
    AgentState,
    AgenticRAGRequest,
    CrossCorpusSearchRequest,
    FindContradictionsRequest,
    FindExpertsRequest,
    FindExperimentsRequest,
    FindKnowledgeGapsRequest,
    FindTechnologiesRequest,
    GetEvidencePackRequest,
    ParsedAgentQuery,
    QueryConstraints,
    ResolveEntityRequest,
    RetrievalPlan,
    SufficiencyVerdict,
    WebSearchRequest,
)
from .search_store import SearchStore
from .service import MEKGService
from .web_search import WEB_SOURCE_PROFILES, YandexWebSearchClient, select_web_profiles


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AgentCancelled(RuntimeError):
    pass


class AgenticRAG:
    """Controlled evidence state machine. The LLM fills schemas; Python owns control flow."""

    def __init__(
        self,
        config: MEKGConfig | None = None,
        *,
        store: SearchStore | None = None,
        service: MEKGService | None = None,
        web_client: YandexWebSearchClient | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = config or MEKGConfig.from_env()
        self.store = store or SearchStore(self.config)
        self.service = service or MEKGService(self.config)
        self._web_client = web_client
        self.llm = llm or build_agent_llm(self.config)

    @property
    def web_client(self) -> YandexWebSearchClient:
        if self._web_client is None:
            self._web_client = YandexWebSearchClient(self.config)
        return self._web_client

    def run(
        self,
        run_id: str,
        request: AgenticRAGRequest,
        *,
        owner: str,
        initial_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], AgentState]:
        state = AgentState.model_validate(initial_state) if initial_state else AgentState(
            query_id=run_id, original_query=request.query
        )
        self._check_cancel(run_id)
        if state.parsed_query is None:
            self._stage(run_id, state, owner, "analyzing", "analysis_started", {"query_id": run_id})
            state.parsed_query = self._analyze(request.query, state)
            self._emit(run_id, "query_analyzed", state.parsed_query.model_dump(mode="json"))
            self.store.update_agent_state(run_id, state.model_dump(mode="json"), status="analyzing")

        start_iteration = max(1, state.iteration + 1)
        for iteration in range(start_iteration, request.max_iterations + 1):
            self._check_cancel(run_id)
            state.iteration = iteration
            plan = self._plan(request, state, iteration)
            self._stage(
                run_id, state, owner, "retrieving", "retrieval_planned",
                plan.model_dump(mode="json"), iteration=iteration,
            )
            retrieval = self._retrieve(request, state, plan)
            pack = self._build_evidence_pack(state, plan, retrieval)
            state.evidence_pack = pack
            history = {
                "iteration": iteration,
                "strategy": plan.strategy,
                "queries": plan.queries,
                "results_count": retrieval["results_count"],
                "useful_results_count": len(pack.items),
                "covered_slots": pack.covered_slots,
                "missing_slots": pack.missing_slots,
                "warnings": retrieval["warnings"],
            }
            state.search_history.append(history)
            state.warnings.extend(item for item in retrieval["warnings"] if item not in state.warnings)
            self._emit(run_id, "evidence_collected", history, iteration=iteration)
            self.store.update_agent_state(run_id, state.model_dump(mode="json"), status="retrieving")

            rough_draft = self._rough_draft(state)
            self._stage(
                run_id, state, owner, "judging", "sufficiency_started",
                {"evidence_items": len(pack.items)}, iteration=iteration,
            )
            verdict = self._judge(state, rough_draft, final_iteration=iteration >= request.max_iterations)
            state.sufficiency = verdict
            self._emit(run_id, "sufficiency_decided", verdict.model_dump(mode="json"), iteration=iteration)
            self.store.update_agent_state(run_id, state.model_dump(mode="json"), status="judging")
            if verdict.action == "answer_full":
                break
            if verdict.action == "search_more" and iteration < request.max_iterations:
                self._stage(
                    run_id, state, owner, "retrying", "targeted_retry",
                    {"focus": verdict.next_search_focus}, iteration=iteration,
                )
                continue
            break

        self._check_cancel(run_id)
        self._stage(run_id, state, owner, "finalizing", "final_synthesis_started", {})
        result = self._synthesize(request, state)
        state.final_mode = result["mode"]
        self._emit(
            run_id, "completed",
            {"mode": result["mode"], "confidence": result["confidence"], "sources": len(result["sources"])},
            iteration=state.iteration,
        )
        return result, state

    def _stage(
        self,
        run_id: str,
        state: AgentState,
        owner: str,
        status: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        iteration: int | None = None,
    ) -> None:
        self.store.heartbeat_agent_run(run_id, owner)
        self.store.update_agent_state(run_id, state.model_dump(mode="json"), status=status)
        self._emit(run_id, event_type, payload, iteration=iteration)

    def _emit(self, run_id: str, event_type: str, payload: dict[str, Any], iteration: int | None = None) -> None:
        self.store.append_agent_event(run_id, event_type, payload, iteration=iteration)

    def _check_cancel(self, run_id: str) -> None:
        if self.store.agent_cancel_requested(run_id):
            raise AgentCancelled("Agent run cancelled")

    def _structured(
        self,
        schema: type[SchemaT],
        system: str,
        payload: dict[str, Any],
        state: AgentState,
    ) -> SchemaT | None:
        if not state.llm_available:
            return None
        try:
            runnable = self.llm.with_structured_output(schema, method="json_schema")
            result = runnable.invoke([
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ])
            return result if isinstance(result, schema) else schema.model_validate(result)
        except Exception as exc:
            state.llm_available = False
            warning = f"Agent LLM circuit breaker: {type(exc).__name__}"
            if warning not in state.warnings:
                state.warnings.append(warning)
            return None

    def _analyze(self, query: str, state: AgentState) -> ParsedAgentQuery:
        result = self._structured(
            ParsedAgentQuery,
            """Analyze a mining/metallurgical R&D question. Return only the requested schema. Identify
            critical slots conservatively. A numeric optimum requires value/range, unit, system, source and
            basis_for_optimality. Geography comparison and time filters must be explicit.""",
            {"query": query},
            state,
        )
        return self._normalize_analysis(result or self._fallback_analysis(query), query)

    @classmethod
    def _normalize_analysis(cls, parsed: ParsedAgentQuery, query: str) -> ParsedAgentQuery:
        """Keep LLM analysis inside the evidence slot vocabulary used by deterministic gates."""
        allowed = {
            "technology_solution", "applicability_condition", "source", "limitations",
            "experiment", "material", "process", "result", "numeric_value_or_range", "unit",
            "material_or_system", "basis_for_optimality", "geography", "practice_type", "year",
            "target_combination", "searched_entities", "missing_slots", "nearest_evidence",
        }
        fallback = cls._fallback_analysis(query)
        required = [slot for slot in parsed.required_slots if slot in allowed]
        folded_slots = " ".join(parsed.required_slots).casefold()
        if any(token in folded_slots for token in ("источник", "source", "citation")):
            required.append("source")
        if any(token in folded_slots for token in ("значен", "процент", "извлеч", "value", "range")):
            required.append("numeric_value_or_range")
        if any(token in folded_slots for token in ("единиц", "unit", "процент")):
            required.append("unit")
        if any(token in folded_slots for token in ("операц", "процесс", "process", "схем")):
            required.append("process")
        parsed.requires_numeric_answer = parsed.requires_numeric_answer or fallback.requires_numeric_answer
        parsed.requires_geography_comparison = (
            parsed.requires_geography_comparison or fallback.requires_geography_comparison
        )
        parsed.requires_time_filter = parsed.requires_time_filter or fallback.requires_time_filter
        if parsed.requires_numeric_answer:
            required.extend(["numeric_value_or_range", "unit", "process", "material_or_system"])
        if parsed.requires_geography_comparison:
            required.extend(["geography", "practice_type"])
        if parsed.requires_time_filter:
            required.append("year")
        if not required:
            required.extend(fallback.required_slots)
        required.append("source")
        parsed.required_slots = list(dict.fromkeys(required))
        parsed.optional_slots = [slot for slot in parsed.optional_slots if slot in allowed]
        return parsed

    @staticmethod
    def _fallback_analysis(query: str) -> ParsedAgentQuery:
        folded = query.casefold()
        numeric = any(word in folded for word in (
            "скорост", "температур", "концентрац", "давлен", "расход", "оптим", "диапазон",
            "извлеч", "выход", "содержан", "%",
            "velocity", "temperature", "concentration", "pressure", "flow rate", "optimal", "range",
        ))
        geography = any(word in folded for word in (
            "рф", "росси", "зарубеж", "миров", "world", "foreign", "international", "domestic",
        )) and any(word in folded for word in ("сравн", "vs", "versus", "compare", "рф", "росси"))
        world_scope = any(word in folded for word in ("зарубеж", "миров", "world", "foreign", "international"))
        time_filter = bool(re.search(r"(?:последн\w*\s+\d+\s+лет|last\s+\d+\s+years|20\d{2})", folded))
        if any(word in folded for word in ("пробел", "не хватает", "gap")):
            intent, answer_type = "knowledge_gap", "gap_analysis"
        elif any(word in folded for word in ("эксперимент", "experiment", "опыт")):
            intent, answer_type = "experiments", "experiment_list"
        elif any(word in folded for word in ("эксперт", "автор", "expert")):
            intent, answer_type = "experts", "review"
        elif any(word in folded for word in ("метод", "технолог", "решени", "method", "technology")):
            intent, answer_type = "technology_review", "table"
        else:
            intent, answer_type = "research", "review"
        required = ["source"]
        if intent == "technology_review":
            required = ["technology_solution", "applicability_condition", "source", "limitations"]
        if intent == "experiments":
            required = ["experiment", "material", "process", "result", "source"]
        if intent == "knowledge_gap":
            required = ["target_combination", "searched_entities", "missing_slots", "nearest_evidence"]
        if numeric:
            required.extend(["numeric_value_or_range", "unit", "process", "material_or_system"])
            if "оптим" in folded or "optimal" in folded:
                required.append("basis_for_optimality")
        if geography or world_scope:
            required.extend(["geography", "practice_type"])
        if time_filter:
            required.append("year")
        material_groups = {
            "nickel": ("никел", "nickel"), "copper": ("мед", "copper"),
            "palladium": ("паллади", "palladium"), "platinum": ("платин", "platinum"),
            "cobalt": ("кобальт", "cobalt"), "iron": ("желез", "iron"),
        }
        process_groups = {
            "electrowinning": ("электроэкстрак", "electrowinning"),
            "leaching": ("выщелач", "leach"), "flotation": ("флотац", "flotation"),
            "roasting": ("обжиг", "roasting"),
            "electrorefining": ("электрорафинир", "electrorefining"),
        }
        materials = [name for name, aliases in material_groups.items() if any(alias in folded for alias in aliases)]
        processes = [name for name, aliases in process_groups.items() if any(alias in folded for alias in aliases)]
        return ParsedAgentQuery(
            intent=intent,
            entities={"materials": materials, "processes": processes, "equipment": [], "properties": [], "substances": []},
            constraints=QueryConstraints(
                geography="comparison" if geography else ("world_practice" if world_scope else None),
                time_range="explicit" if time_filter else None,
            ),
            required_slots=list(dict.fromkeys(required)),
            optional_slots=["equipment_type", "industrial_or_lab_scale", "economic_metrics"],
            requires_numeric_answer=numeric,
            requires_geography_comparison=geography,
            requires_time_filter=time_filter,
            answer_type=answer_type,
        )

    def _plan(self, request: AgenticRAGRequest, state: AgentState, iteration: int) -> RetrievalPlan:
        parsed = state.parsed_query or self._fallback_analysis(request.query)
        strategy = ("broad", "missing_slots", "fallback")[iteration - 1]
        result = self._structured(
            RetrievalPlan,
            """Create bounded retrieval rewrites for the specified iteration. Use Russian and English terms.
            Iteration 1 is broad, iteration 2 targets missing slots, iteration 3 uses synonyms and adjacent
            processes without changing the core material/process. Choose at most two supplied web profile ids.""",
            {
                "iteration": iteration,
                "strategy": strategy,
                "query": request.query,
                "parsed_query": parsed.model_dump(mode="json"),
                "missing_slots": state.sufficiency.missing_slots or parsed.required_slots,
                "available_web_profiles": list(WEB_SOURCE_PROFILES),
            },
            state,
        )
        if result:
            result.iteration = iteration
            result.strategy = strategy
            result.queries = self._bounded_queries(result.queries, request.query)
            result.web_profiles = select_web_profiles(
                " ".join(result.queries), request.web_profile_ids or result.web_profiles, limit=2
            )
            result.graph_tools = self._graph_tools(parsed.intent, iteration)
            return result
        return self._fallback_plan(request, state, iteration, strategy)

    def _fallback_plan(
        self, request: AgenticRAGRequest, state: AgentState, iteration: int, strategy: str
    ) -> RetrievalPlan:
        query = request.query
        missing = state.sufficiency.missing_slots or (state.parsed_query.required_slots if state.parsed_query else [])
        if iteration == 1:
            queries = [query, self._translate_terms(query)]
            goal = "find candidate evidence"
        elif iteration == 2:
            focus = " ".join(missing[:5])
            queries = [f"{query} {focus}", f"{self._translate_terms(query)} {focus} numeric value unit industrial"]
            goal = "fill missing evidence slots"
        else:
            queries = [
                self._translate_terms(query),
                f"{query} synonyms adjacent process mechanism",
                f"{self._translate_terms(query)} hydrodynamics mass transfer industrial practice",
            ]
            goal = "relaxed synonym search with drift guard"
        queries = self._bounded_queries(queries, query)
        parsed = state.parsed_query or self._fallback_analysis(query)
        return RetrievalPlan(
            iteration=iteration,
            strategy=strategy,
            goal=goal,
            queries=queries,
            graph_tools=self._graph_tools(parsed.intent, iteration),
            web_profiles=select_web_profiles(" ".join(queries), request.web_profile_ids, limit=2),
            drift_notes=["Adjacent-process evidence is analogy-only"] if iteration == 3 else [],
        )

    @staticmethod
    def _bounded_queries(values: Iterable[str], original: str) -> list[str]:
        clean = [" ".join(value.split())[:1000] for value in values if value and value.strip()]
        if not clean:
            clean = [original]
        return list(dict.fromkeys(clean))[:6]

    @staticmethod
    def _translate_terms(value: str) -> str:
        replacements = {
            "электроэкстракция": "electrowinning", "никеля": "nickel", "никель": "nickel",
            "циркуляция": "circulation", "католита": "catholyte", "скорость": "velocity",
            "выщелачивание": "leaching", "меди": "copper", "температура": "temperature",
            "концентрация": "concentration", "мировая практика": "world practice",
        }
        result = value.casefold()
        for source, target in replacements.items():
            result = result.replace(source, target)
        return result

    @staticmethod
    def _graph_tools(intent: str, iteration: int) -> list[str]:
        tools = ["resolve_entity"]
        if intent in {"technology_review", "research", "recommendation"}:
            tools.append("find_technologies")
        if intent == "experiments":
            tools.append("find_experiments")
        if intent == "experts":
            tools.append("find_experts")
        if intent == "knowledge_gap" or iteration >= 2:
            tools.append("find_knowledge_gaps")
        if iteration >= 2:
            tools.append("find_contradictions")
        return tools

    def _retrieve(self, request: AgenticRAGRequest, state: AgentState, plan: RetrievalPlan) -> dict[str, Any]:
        parsed = state.parsed_query or self._fallback_analysis(request.query)
        warnings: list[str] = []
        outputs: dict[str, Any] = {"vector": None, "graph": [], "web": None}
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            search_request = CrossCorpusSearchRequest(
                query=plan.queries[0],
                intent=parsed.intent,
                target_slots=parsed.required_slots,
                corpora=request.corpora,
                filters=request.filters,
                numeric_mode="boost",
                final_k=24,
                include_debug=request.include_debug,
                allow_remote=state.llm_available,
            )
            futures[pool.submit(self.service.cross_corpus_search, search_request)] = "vector"
            futures[pool.submit(self._execute_graph_tools, request, state, plan)] = "graph"
            use_web = self._should_use_web(request, state, plan)
            if use_web:
                state.web_calls += len(plan.web_profiles)
                web_request = WebSearchRequest(
                    queries=plan.queries[:4],
                    profile_ids=plan.web_profiles,
                    search_context_size="high",
                    region=self.config.web_search_region,
                )
                futures[pool.submit(self.web_client.search, web_request)] = "web"
            for future in as_completed(futures):
                channel = futures[future]
                try:
                    value = future.result()
                    outputs[channel] = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
                except Exception as exc:
                    warnings.append(f"{channel}: {type(exc).__name__}")
        for channel in ("vector", "web"):
            value = outputs.get(channel)
            if isinstance(value, dict):
                warnings.extend(value.get("warnings") or [])
        graph_rows = outputs.get("graph") or []
        for row in graph_rows:
            warnings.extend((row.get("response") or {}).get("warnings") or [])
        count = 0
        vector_data = (outputs.get("vector") or {}).get("data") or {}
        count += len(vector_data.get("results") or [])
        count += sum(len((row.get("response") or {}).get("evidence") or []) for row in graph_rows)
        count += sum(len(block.get("sources") or []) for block in ((outputs.get("web") or {}).get("data") or {}).get("blocks", []))
        outputs["warnings"] = list(dict.fromkeys(warnings))
        outputs["results_count"] = count
        return outputs

    def _should_use_web(self, request: AgenticRAGRequest, state: AgentState, plan: RetrievalPlan) -> bool:
        if not state.llm_available or not request.allow_external_web or state.web_calls >= 6 or not plan.web_profiles:
            return False
        folded = request.query.casefold()
        external_intent = any(term in folded for term in (
            "миров", "зарубеж", "последн", "свеж", "патент", "world", "foreign", "latest", "patent",
        ))
        return external_intent or plan.iteration > 1

    def _execute_graph_tools(
        self, request: AgenticRAGRequest, state: AgentState, plan: RetrievalPlan
    ) -> list[dict[str, Any]]:
        parsed = state.parsed_query or self._fallback_analysis(request.query)
        materials = parsed.entities.get("materials", [])
        processes = parsed.entities.get("processes", [])
        calls: list[tuple[str, Any]] = []
        for name in [*materials, *processes][:4]:
            calls.append(("resolve_entity", ResolveEntityRequest(text=name)))
        if "find_technologies" in plan.graph_tools:
            calls.append(("find_technologies", FindTechnologiesRequest(problem=request.query, limit=12)))
        if "find_experiments" in plan.graph_tools:
            calls.append(("find_experiments", FindExperimentsRequest(
                material=materials[0] if materials else None,
                process=processes[0] if processes else None,
                limit=15,
            )))
        if "find_experts" in plan.graph_tools:
            calls.append(("find_experts", FindExpertsRequest(
                topic=request.query, process=processes[0] if processes else None, limit=12,
            )))
        if "find_knowledge_gaps" in plan.graph_tools:
            calls.append(("find_knowledge_gaps", FindKnowledgeGapsRequest(
                material=materials[0] if materials else None,
                process=processes[0] if processes else None,
                limit=12,
            )))
        if "find_contradictions" in plan.graph_tools:
            calls.append(("find_contradictions", FindContradictionsRequest(
                topic=processes[0] if processes else request.query[:200], limit=12,
            )))

        result = []
        try:
            source_response = self.service.search_source_evidence(
                plan.queries[0], corpora=request.corpora, limit=24
            )
            result.append({
                "tool": "search_source_evidence",
                "response": source_response.model_dump(mode="json"),
            })
        except Exception as exc:
            result.append({
                "tool": "search_source_evidence",
                "response": {"data": None, "evidence": [], "warnings": [type(exc).__name__]},
            })
        method_map = {
            "resolve_entity": self.service.resolve_entity,
            "find_technologies": self.service.find_technologies,
            "find_experiments": self.service.find_experiments,
            "find_experts": self.service.find_experts,
            "find_knowledge_gaps": self.service.find_knowledge_gaps,
            "find_contradictions": self.service.find_contradictions,
        }
        for tool, tool_request in calls:
            try:
                response = method_map[tool](tool_request)
                result.append({"tool": tool, "response": response.model_dump(mode="json")})
            except Exception as exc:
                result.append({"tool": tool, "response": {"data": None, "evidence": [], "warnings": [type(exc).__name__]}})
        claim_ids = self._claim_ids(result)
        for claim_id in claim_ids[:5]:
            try:
                response = self.service.get_evidence_pack(GetEvidencePackRequest(claim_id=claim_id))
                result.append({"tool": "get_evidence_pack", "response": response.model_dump(mode="json")})
            except Exception:
                continue
        return result

    @staticmethod
    def _claim_ids(value: Any) -> list[str]:
        found = []

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if item.get("id") and item.get("text"):
                    found.append(str(item["id"]))
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return list(dict.fromkeys(found))

    def _build_evidence_pack(
        self, state: AgentState, plan: RetrievalPlan, retrieval: dict[str, Any]
    ) -> AgentEvidencePack:
        parsed = state.parsed_query or self._fallback_analysis(state.original_query)
        merged: dict[str, AgentEvidenceItem] = {item.id: item for item in state.evidence_pack.items}
        vector = retrieval.get("vector") or {}
        for item in (vector.get("data") or {}).get("results", []) or []:
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            text = str(item.get("text") or "")[:2500]
            evidence = AgentEvidenceItem(
                id=f"local:{chunk_id}",
                source_id=str(item.get("document_id") or chunk_id),
                source_type="local_chunk",
                title=item.get("title") or item.get("file_name"),
                snippet=text,
                file_name=item.get("file_name"),
                page_number=item.get("page_number"),
                slide_number=item.get("slide_number"),
                year=self._extract_year(item, text),
                geography=self._geography(text),
                supports_slots=self._supports_slots(parsed.required_slots, text, item.get("matched_slots") or []),
                numeric_facts=self._numeric_facts(text),
                confidence=max(0.0, min(1.0, float(item.get("score") or item.get("dense_score") or 0.45))),
                direct=self._is_direct(state.original_query, text, plan),
            )
            merged[evidence.id] = evidence

        graph_rows = retrieval.get("graph") or []
        contradictions: list[dict[str, Any]] = list(state.evidence_pack.contradictions)
        gaps: list[dict[str, Any]] = list(state.evidence_pack.gaps)
        for row in graph_rows:
            response = row.get("response") or {}
            data = response.get("data") or {}
            if row.get("tool") == "find_contradictions":
                contradictions.extend(data.get("contradictions") or [])
            if row.get("tool") == "find_knowledge_gaps":
                gaps.extend(data.get("gaps") or [])
            for index, item in enumerate(response.get("evidence") or []):
                text = str(item.get("quote") or item.get("text") or "")[:2500]
                if not text:
                    continue
                source_id = str(item.get("document_id") or item.get("element_id") or item.get("id") or f"graph-{index}")
                evidence_id = f"graph:{source_id}:{item.get('page') or item.get('page_number') or index}"
                merged[evidence_id] = AgentEvidenceItem(
                    id=evidence_id,
                    source_id=source_id,
                    source_type=item.get("source_type") or "graph_claim",
                    title=item.get("file_name") or row.get("tool"),
                    snippet=text,
                    file_name=item.get("file_name"),
                    page_number=item.get("page") or item.get("page_number"),
                    slide_number=item.get("slide") or item.get("slide_number"),
                    year=self._extract_year(item, text),
                    geography=item.get("geo_scope") or self._geography(text),
                    supports_slots=self._supports_slots(parsed.required_slots, text),
                    numeric_facts=self._numeric_facts(text),
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or response.get("confidence") or 0.6))),
                    direct=self._is_direct(state.original_query, text, plan),
                )

        web = retrieval.get("web") or {}
        for block in (web.get("data") or {}).get("blocks", []) or []:
            text = str(block.get("text") or "")[:3500]
            for index, source in enumerate(block.get("sources") or []):
                url = source.get("url")
                if not url:
                    continue
                evidence_id = f"web:{url}"
                merged[evidence_id] = AgentEvidenceItem(
                    id=evidence_id,
                    source_id=url,
                    source_type=source.get("source_type") or "web",
                    title=source.get("title") or block.get("profile_id"),
                    snippet=text,
                    url=url,
                    year=self._extract_year(source, text),
                    geography=self._geography(text),
                    supports_slots=self._supports_slots(parsed.required_slots, text),
                    numeric_facts=self._numeric_facts(text),
                    confidence=float(block.get("confidence") or 0.45),
                    direct=self._is_direct(state.original_query, text, plan),
                    metadata_only=bool(source.get("metadata_only") or block.get("metadata_only")),
                )

        items = sorted(merged.values(), key=lambda item: (item.direct, not item.metadata_only, item.confidence), reverse=True)[:80]
        for index, item in enumerate(items, start=1):
            item.citation_label = f"S{index}"
        eligible = [item for item in items if item.direct and not item.metadata_only]
        covered = sorted({slot for item in eligible for slot in item.supports_slots})
        if eligible and "source" in parsed.required_slots:
            covered.append("source")
        covered = list(dict.fromkeys(covered))
        missing = [slot for slot in parsed.required_slots if slot not in covered]
        return AgentEvidencePack(
            items=items,
            covered_slots=covered,
            missing_slots=missing,
            contradictions=self._dedupe_dicts(contradictions),
            gaps=self._dedupe_dicts(gaps),
        )

    @staticmethod
    def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result, seen = [], set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result[:50]

    @staticmethod
    def _supports_slots(slots: list[str], text: str, hinted: list[str] | None = None) -> list[str]:
        folded = text.casefold()
        matched = list(hinted or [])
        aliases = {
            "technology_solution": ("технолог", "метод", "схем", "technology", "method", "solution"),
            "applicability_condition": ("услов", "примен", "при ", "condition", "applic"),
            "limitations": ("огранич", "недостат", "limit", "disadvantage"),
            "experiment": ("эксперимент", "опыт", "experiment", "trial"),
            "material": ("материал", "никел", "мед", "руд", "material", "nickel", "copper", "ore"),
            "material_or_system": ("систем", "материал", "никел", "мед", "system", "material", "nickel"),
            "process": ("процесс", "экстрак", "выщелач", "process", "electrowinning", "leaching"),
            "result": ("результ", "извлеч", "выход", "result", "recovery", "yield"),
            "basis_for_optimality": ("оптим", "рекоменд", "максим", "массоперенос", "optimal", "recommended", "maximum", "mass transfer"),
            "geography": ("росси", "рф", "зарубеж", "миров", "russia", "foreign", "world"),
            "practice_type": ("промышлен", "лаборатор", "industrial", "laboratory", "pilot"),
            "year": ("202", "201", "199"),
            "nearest_evidence": ("evidence", "источник", "данн"),
            "searched_entities": ("процесс", "материал", "entity"),
            "missing_slots": ("нет данн", "не найден", "missing", "gap"),
            "target_combination": ("комбинац", "услов", "combination"),
        }
        numeric = AgenticRAG._numeric_facts(text)
        for slot in slots:
            if slot in matched:
                continue
            if slot in {"numeric_value_or_range", "unit"} and numeric:
                if slot == "numeric_value_or_range" or any(item.get("unit") for item in numeric):
                    matched.append(slot)
                continue
            tokens = aliases.get(slot) or tuple(re.findall(r"\w{3,}", slot.casefold()))
            if any(token in folded for token in tokens):
                matched.append(slot)
        return list(dict.fromkeys(matched))

    @staticmethod
    def _numeric_facts(text: str) -> list[dict[str, Any]]:
        pattern = (
            r"(?P<min>\d+(?:[.,]\d+)?)"
            r"(?:\s*[-–—]\s*(?P<max>\d+(?:[.,]\d+)?))?\s*"
            r"(?P<unit>%|°\s*C|℃|мг/л|мг/дм3|г/л|кг/м3|м/с|л/мин|м3/ч|МПа|кПа|Па|ppm|"
            r"mg/l|g/l|kg/m3|m/s|l/min|m3/h|MPa|kPa|Pa)?"
        )
        result = []
        for match in re.finditer(pattern, text, flags=re.I):
            unit = match.group("unit")
            if not unit and len(result) >= 5:
                continue
            result.append({
                "value_min": float(match.group("min").replace(",", ".")),
                "value_max": float(match.group("max").replace(",", ".")) if match.group("max") else None,
                "unit": unit,
            })
            if len(result) >= 20:
                break
        return result

    @staticmethod
    def _extract_year(item: dict[str, Any], text: str) -> int | None:
        value = item.get("year")
        if isinstance(value, int) and 1900 <= value <= 2100:
            return value
        match = re.search(r"\b(?:19|20)\d{2}\b", text)
        return int(match.group()) if match else None

    @staticmethod
    def _geography(text: str) -> str | None:
        folded = text.casefold()
        if any(term in folded for term in ("россия", "российск", " рф", "russia", "russian")):
            return "domestic"
        if any(term in folded for term in ("зарубеж", "международ", "world", "foreign", "international", "global")):
            return "foreign"
        return None

    @staticmethod
    def _is_direct(original: str, evidence: str, plan: RetrievalPlan) -> bool:
        original_folded = original.casefold()
        evidence_folded = evidence.casefold()
        symbols = set(re.findall(
            r"\b(?:au|ag|pt|pd|rh|ru|ir|os|cu|ni|co|mn|fe|se|te|zn|pb|al|ti)\b",
            original_folded,
        ))
        if len(symbols) >= 2:
            evidence_symbols = set(re.findall(
                r"\b(?:au|ag|pt|pd|rh|ru|ir|os|cu|ni|co|mn|fe|se|te|zn|pb|al|ti)\b",
                evidence_folded,
            ))
            if len(symbols & evidence_symbols) < min(2, len(symbols)):
                return False
        latin_anchors = {
            token.casefold()
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9+.-]{3,}\b", original)
            if token.casefold() not in {
                "what", "which", "where", "with", "from", "process", "method", "using",
                "compare", "between", "answer", "data", "technology",
            }
        }
        acronyms = {
            token.casefold()
            for token in re.findall(r"\b[A-ZА-ЯЁ]{2,5}\b", original)
            if token.casefold() not in {"рф"}
        }
        named_anchors = latin_anchors | acronyms
        if named_anchors and not any(anchor in evidence_folded for anchor in named_anchors):
            return False
        anchored = bool(named_anchors)
        entity_groups = (
            ("material:nickel", ("никел", "nickel")),
            ("material:copper", ("мед", "copper")),
            ("material:gold", ("золот", "gold")),
            ("material:cobalt", ("кобальт", "cobalt")),
            ("material:palladium", ("паллади", "palladium")),
            ("material:platinum", ("платин", "platinum")),
            ("process:leaching", ("выщелач", "leach")),
            ("process:electrowinning", ("электроэкстрак", "electrowinning")),
            ("process:electrorefining", ("электрорафинир", "electrorefining")),
            ("process:flotation", ("флотац", "flotation")),
            ("process:roasting", ("обжиг", "roasting")),
        )
        required_groups = [
            aliases for _name, aliases in entity_groups if any(alias in original_folded for alias in aliases)
        ]
        if required_groups and not all(
            any(alias in evidence_folded for alias in aliases) for aliases in required_groups
        ):
            return False
        core = {token for token in re.findall(r"[a-zа-яё]{4,}", original.casefold()) if token not in {
            "какие", "какая", "который", "описаны", "практике", "what", "which", "where", "world",
        }}
        found = set(re.findall(r"[a-zа-яё]{4,}", evidence.casefold()))
        overlap = len(core & found)
        if plan.strategy != "fallback":
            return overlap >= min(2, len(core)) or bool(required_groups) or anchored or not core
        return overlap >= 2 or anchored

    def _rough_draft(self, state: AgentState) -> str:
        items = state.evidence_pack.items[:20]
        if not items:
            return "Evidence не найдено; ответ невозможен."
        if state.llm_available:
            try:
                context = [
                    {"source": item.citation_label, "text": item.snippet[:900], "slots": item.supports_slots}
                    for item in items
                ]
                response = self.llm.invoke([
                    SystemMessage(content="Create a short internal draft only from supplied evidence. Identify missing facts; do not invent."),
                    HumanMessage(content=json.dumps({"question": state.original_query, "evidence": context}, ensure_ascii=False)),
                ])
                return str(response.content)
            except Exception as exc:
                state.llm_available = False
                state.warnings.append(f"Agent LLM circuit breaker: {type(exc).__name__}")
        covered = ", ".join(state.evidence_pack.covered_slots) or "нет"
        missing = ", ".join(state.evidence_pack.missing_slots) or "нет"
        return f"Найдено источников: {len(items)}. Покрыто: {covered}. Не хватает: {missing}."

    def _judge(self, state: AgentState, rough_draft: str, *, final_iteration: bool) -> SufficiencyVerdict:
        parsed = state.parsed_query or self._fallback_analysis(state.original_query)
        pack = state.evidence_pack
        eligible = [item for item in pack.items if item.direct and not item.metadata_only]
        covered = set(pack.covered_slots)
        numeric_with_unit = any(fact.get("unit") for item in eligible for fact in item.numeric_facts)
        optimality_required = "basis_for_optimality" in parsed.required_slots
        optimality_supported = not optimality_required or any(
            "basis_for_optimality" in item.supports_slots and any(fact.get("unit") for fact in item.numeric_facts)
            for item in eligible
        )
        geographies = {item.geography for item in eligible if item.geography}
        geography_supported = not parsed.requires_geography_comparison or {"domestic", "foreign"}.issubset(geographies)
        time_supported = not parsed.requires_time_filter or any(item.year for item in eligible)
        source_supported = bool(eligible)
        contradictions_clear = not pack.contradictions
        numeric_supported = not parsed.requires_numeric_answer or numeric_with_unit
        hard_gates = {
            "numeric_value_and_unit": numeric_supported,
            "basis_for_optimality": optimality_supported,
            "geography_comparison": geography_supported,
            "time_filter": time_supported,
            "claim_sources": source_supported,
            "contradictions_resolved": contradictions_clear,
        }
        missing = [slot for slot in parsed.required_slots if slot not in covered]
        if parsed.requires_numeric_answer and not numeric_with_unit:
            missing = list(dict.fromkeys([*missing, "numeric_value_or_range", "unit"]))
        if optimality_required and not optimality_supported:
            missing = list(dict.fromkeys([*missing, "basis_for_optimality"]))
        if parsed.requires_geography_comparison and not geography_supported:
            missing = list(dict.fromkeys([*missing, "geography"]))
        if parsed.requires_time_filter and not time_supported:
            missing = list(dict.fromkeys([*missing, "year"]))

        all_slots = not missing
        score = 0
        score += 25 if all_slots else round(25 * (len(covered) / max(1, len(parsed.required_slots))))
        score += 20 if numeric_supported and optimality_supported else 0
        score += 15 if source_supported else 0
        score += 15 if len({item.source_id for item in eligible}) >= 2 else 0
        score += 10 if geography_supported and time_supported else 0
        score += 10 if contradictions_clear else 0
        score += 5 if eligible else 0
        score = max(0, min(100, int(score)))

        llm_verdict = self._structured(
            SufficiencyVerdict,
            """Act as a conservative engineering evidence reviewer. Never answer the question. Missing numeric
            units, source attribution, geography, year, basis for optimality, or unresolved contradictions must
            make full sufficiency false. Metadata-only and analogy evidence cannot close critical slots.""",
            {
                "question": state.original_query,
                "required_slots": parsed.required_slots,
                "evidence_pack": pack.model_dump(mode="json"),
                "rough_draft": rough_draft,
                "search_history": state.search_history,
                "hard_gates": hard_gates,
            },
            state,
        )
        gates_pass = all(hard_gates.values())
        if llm_verdict:
            llm_hard_gates = llm_verdict.hard_gates or {}
            relevant_llm_missing = set(parsed.required_slots) | {"claim_sources", "contradictions_resolved"}
            if parsed.requires_numeric_answer:
                relevant_llm_missing.update({"numeric_value_and_unit", "numeric_value_or_range", "unit"})
            if optimality_required:
                relevant_llm_missing.update({"basis_for_optimality"})
            if parsed.requires_geography_comparison:
                relevant_llm_missing.update({"geography_comparison", "geography"})
            if parsed.requires_time_filter:
                relevant_llm_missing.update({"time_filter", "year"})
            llm_failed_gates = [
                name for name, passed in llm_hard_gates.items()
                if passed is False and name in relevant_llm_missing
            ]
            llm_missing = list(dict.fromkeys([
                *[slot for slot in (llm_verdict.missing_slots or []) if slot in relevant_llm_missing],
                *[slot for slot in (llm_verdict.critical_missing or []) if slot in relevant_llm_missing],
                *llm_failed_gates,
            ]))
            if llm_missing:
                missing = list(dict.fromkeys([*missing, *llm_missing]))
                for gate in llm_failed_gates:
                    if gate in hard_gates:
                        hard_gates[gate] = False
                score = min(score, llm_verdict.score)
                gates_pass = all(hard_gates.values())
                all_slots = not missing
            elif llm_verdict.score > 0:
                if gates_pass and all_slots:
                    score = min(score, max(80, llm_verdict.score))
                else:
                    score = min(score, llm_verdict.score)
        if gates_pass and all_slots and score >= 80:
            action = "answer_full"
        elif not final_iteration:
            action = "search_more"
        elif score >= 60 and eligible:
            action = "answer_partial"
        else:
            action = "no_data"
        reason = (
            "All critical evidence gates pass."
            if action == "answer_full"
            else f"Missing or unverified: {', '.join(missing) or 'hard evidence gates'}."
        )
        return SufficiencyVerdict(
            sufficient=action == "answer_full",
            score=score,
            action=action,
            covered_slots={slot: ("covered" if slot in covered else "missing") for slot in parsed.required_slots},
            missing_slots=missing,
            critical_missing=missing,
            contradictions=pack.contradictions,
            reason=reason,
            next_search_focus=missing[:6],
            can_answer_partially=score >= 60 and bool(eligible),
            hard_gates=hard_gates,
        )

    def _synthesize(self, request: AgenticRAGRequest, state: AgentState) -> dict[str, Any]:
        verdict = state.sufficiency
        sources = [
            {
                "label": item.citation_label,
                "source_id": item.source_id,
                "title": item.title,
                "url": item.url,
                "file_name": item.file_name,
                "page": item.page_number,
                "slide": item.slide_number,
                "source_type": item.source_type,
                "metadata_only": item.metadata_only,
                "direct": item.direct,
            }
            for item in state.evidence_pack.items
        ]
        if verdict.action == "answer_full" and state.llm_available:
            mode = "full_answer"
        elif verdict.can_answer_partially or state.evidence_pack.items:
            mode = "partial_answer_with_gaps"
        elif (state.parsed_query and state.parsed_query.requires_numeric_answer):
            mode = "NO_NUMERIC_DATA"
        else:
            mode = "NO_EVIDENCE_FOUND"

        answer = ""
        if state.llm_available and mode in {"full_answer", "partial_answer_with_gaps"}:
            compact = [
                {
                    "source": item.citation_label,
                    "text": item.snippet[:1200],
                    "numeric_facts": item.numeric_facts,
                    "direct": item.direct,
                    "metadata_only": item.metadata_only,
                }
                for item in state.evidence_pack.items[:30]
            ]
            try:
                response = self.llm.invoke([
                    SystemMessage(content=(
                        "Answer only from the supplied evidence. Cite every important claim as [S1]. "
                        "Never present analogy or metadata-only records as direct proof. Keep all numeric units. "
                        "Structure: conclusion, comparison table when useful, sources, contradictions, gaps, confidence."
                    )),
                    HumanMessage(content=json.dumps({
                        "language": request.answer_language,
                        "question": request.query,
                        "mode": mode,
                        "evidence": compact,
                        "verdict": verdict.model_dump(mode="json"),
                    }, ensure_ascii=False)),
                ])
                answer = str(response.content)
            except Exception as exc:
                state.llm_available = False
                state.warnings.append(f"Agent LLM synthesis fallback: {type(exc).__name__}")
                mode = "partial_answer_with_gaps" if state.evidence_pack.items else mode
        if not answer:
            answer = self._fallback_answer(state, mode)
        confidence = verdict.score / 100
        if mode == "partial_answer_with_gaps":
            confidence = min(confidence, 0.79)
        elif mode in {"NO_EVIDENCE_FOUND", "NO_NUMERIC_DATA"}:
            confidence = min(confidence, 0.49)
        return {
            "job_id": state.query_id,
            "status": "complete",
            "mode": mode,
            "answer_markdown": answer,
            "confidence": round(confidence, 3),
            "sources": sources,
            "gaps": verdict.missing_slots,
            "contradictions": state.evidence_pack.contradictions,
            "warnings": list(dict.fromkeys(state.warnings)),
            "searched_iterations": state.iteration,
            "state": state.model_dump(mode="json") if request.include_debug else None,
        }

    @staticmethod
    def _fallback_answer(state: AgentState, mode: str) -> str:
        verdict = state.sufficiency
        if mode in {"NO_EVIDENCE_FOUND", "NO_NUMERIC_DATA"}:
            nearest = state.evidence_pack.items[:5]
            lines = [
                "## Результат",
                "В доступных источниках системы не найдено достаточно прямых доказательств для полного ответа.",
                f"Проверено итераций: {state.iteration}.",
                f"Не подтверждено: {', '.join(verdict.missing_slots) or 'ключевые слоты'}.",
            ]
            if nearest:
                lines.append("\n## Ближайшие материалы")
                lines.extend(f"- [{item.citation_label}] {item.title or item.source_id}" for item in nearest)
            return "\n".join(lines)
        lines = [
            "## Частичный результат" if mode != "full_answer" else "## Результат",
            "Ниже перечислено только то, что присутствует в собранном Evidence Pack.",
        ]
        direct_items = [
            item for item in state.evidence_pack.items if item.direct and not item.metadata_only
        ]
        chosen = direct_items[:6] or state.evidence_pack.items[:3]
        seen_snippets: set[str] = set()
        for item in chosen:
            normalized = " ".join(item.snippet.split())[:700]
            if not normalized or normalized in seen_snippets:
                continue
            seen_snippets.add(normalized)
            kind = "метаданные" if item.metadata_only else ("аналогия" if not item.direct else "evidence")
            lines.append(f"- [{item.citation_label}] ({kind}) {normalized}")
        if verdict.missing_slots:
            lines.extend(["\n## Пробелы", *[f"- {slot}" for slot in verdict.missing_slots]])
        lines.append(f"\nУверенность: {verdict.score}/100.")
        return "\n".join(lines)
