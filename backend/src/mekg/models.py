from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .schema import ENTITY_LABELS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ValidationStatus(StrEnum):
    RAW_EXTRACTED = "raw_extracted"
    MACHINE_VALIDATED = "machine_validated"
    EXPERT_VALIDATED = "expert_validated"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    CONFLICTING = "conflicting"


class ElementKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    TABLE_ROW = "table_row"
    FIGURE = "figure"
    FORMULA = "formula"


class SourceElement(BaseModel):
    id: str
    kind: ElementKind
    text: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    bbox: list[float] | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublicationBoundary(BaseModel):
    title: str
    start_page: int
    end_page: int
    authors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)
    needs_review: bool = False


class ParsedDocument(BaseModel):
    document_id: str
    version_id: str
    source_locator: str
    file_name: str
    file_type: str
    sha256: str
    size_bytes: int
    category: str | None = None
    title: str | None = None
    language: str | None = None
    elements: list[SourceElement] = Field(default_factory=list)
    publications: list[PublicationBoundary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    element_id: str
    quote: str
    page_number: int | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)


class EntityCandidate(BaseModel):
    local_id: str
    entity_type: str
    canonical_name: str
    name_ru: str | None = None
    name_en: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    evidence: EvidenceRef
    confidence: float = Field(default=0.75, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def fill_canonical_name(cls, data):
        if isinstance(data, dict) and not data.get("canonical_name"):
            fallback = data.get("name_ru") or data.get("name_en")
            if fallback:
                data = {**data, "canonical_name": fallback}
        return data

    @field_validator("entity_type")
    @classmethod
    def clean_type(cls, value: str) -> str:
        compact = "".join(character for character in value if character.isalnum())
        canonical = {label.casefold(): label for label in ENTITY_LABELS}
        return canonical.get(compact.casefold(), value.strip())


class NumericFact(BaseModel):
    local_id: str
    name: str
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    comparator: Literal["=", "<", "<=", ">", ">=", "range"] = "="
    unit_original: str
    method: str | None = None
    material_ref: str | None = None
    substance_ref: str | None = None
    evidence: EvidenceRef
    confidence: float = Field(default=0.75, ge=0, le=1)
    approximate: bool = False

    @field_validator("comparator", mode="before")
    @classmethod
    def normalize_comparator(cls, value):
        return {
            "≈": "=",
            "~": "=",
            "≃": "=",
            "≤": "<=",
            "≥": ">=",
            "=<": "<=",
            "=>": ">=",
        }.get(value, value)

    @field_validator("value", "value_min", "value_max", mode="before")
    @classmethod
    def parse_numeric_value(cls, value):
        """Keep a malformed LLM item rejectable without losing its siblings."""
        if value is None or isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            candidate = value.strip().replace("−", "-").replace(",", ".")
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", candidate):
                return float(candidate)
        return None


class ConditionCandidate(NumericFact):
    pass


class MeasurementCandidate(NumericFact):
    property_name: str


class ExperimentCandidate(BaseModel):
    local_id: str
    name: str
    experiment_type: str = "Experiment"
    material_refs: list[str] = Field(default_factory=list)
    process_refs: list[str] = Field(default_factory=list)
    equipment_refs: list[str] = Field(default_factory=list)
    condition_refs: list[str] = Field(default_factory=list)
    measurement_refs: list[str] = Field(default_factory=list)
    effect: str | None = None
    evidence: EvidenceRef
    confidence: float = Field(default=0.75, ge=0, le=1)


class ClaimCandidate(BaseModel):
    local_id: str
    text: str
    claim_type: str = "Claim"
    entity_refs: list[str] = Field(default_factory=list)
    experiment_refs: list[str] = Field(default_factory=list)
    measurement_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    geo_scope: Literal["DomesticPractice", "ForeignPractice", "GlobalPractice", "UnknownPractice"] = (
        "UnknownPractice"
    )
    countries: list[str] = Field(default_factory=list)
    evidence: EvidenceRef
    confidence: float = Field(default=0.75, ge=0, le=1)


class ExpertCandidate(BaseModel):
    local_id: str
    name: str
    organization: str | None = None
    lab: str | None = None
    topics: list[str] = Field(default_factory=list)
    evidence: EvidenceRef
    confidence: float = Field(default=0.75, ge=0, le=1)


class ChunkExtraction(BaseModel):
    entities: list[EntityCandidate] = Field(default_factory=list)
    conditions: list[ConditionCandidate] = Field(default_factory=list)
    measurements: list[MeasurementCandidate] = Field(default_factory=list)
    experiments: list[ExperimentCandidate] = Field(default_factory=list)
    claims: list[ClaimCandidate] = Field(default_factory=list)
    experts: list[ExpertCandidate] = Field(default_factory=list)


class BatchElementExtraction(BaseModel):
    element_id: str
    extraction: ChunkExtraction = Field(default_factory=ChunkExtraction)


class BatchExtraction(BaseModel):
    items: list[BatchElementExtraction] = Field(default_factory=list)


class PublicationSplit(BaseModel):
    publications: list[PublicationBoundary] = Field(default_factory=list)


class NormalizedUnit(BaseModel):
    original: str
    symbol: str
    dimension: str
    factor_to_base: float = 1.0
    valid: bool = True
    error: str | None = None


class ComponentConstraint(BaseModel):
    name: str
    min: float | None = None
    max: float | None = None
    unit: str | None = None


class ConditionFilter(BaseModel):
    parameter: str
    min: float | None = None
    max: float | None = None
    unit: str | None = None


class ResolveEntityRequest(BaseModel):
    text: str
    type_hint: str | None = None
    language: str | None = None


class FindExperimentsRequest(BaseModel):
    material: str | None = None
    process: str | None = None
    conditions: list[ConditionFilter] = Field(default_factory=list)
    property: str | None = None
    geo_scope: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    limit: int = Field(default=20, ge=1, le=100)


class InputStream(BaseModel):
    type: str = "water"
    components: list[ComponentConstraint] = Field(default_factory=list)


class TechnologyTarget(BaseModel):
    parameter: str
    min: float | None = None
    max: float | None = None
    unit: str | None = None


class FindTechnologiesRequest(BaseModel):
    problem: str
    input_stream: InputStream | None = None
    target: TechnologyTarget | None = None
    geo_scope: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class GetEvidencePackRequest(BaseModel):
    claim_id: str


class FindContradictionsRequest(BaseModel):
    topic: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    status: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class FindKnowledgeGapsRequest(BaseModel):
    material: str | None = None
    process: str | None = None
    condition: str | None = None
    property: str | None = None
    geo_region: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class FindExpertsRequest(BaseModel):
    topic: str
    process: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class SemanticSearchRequest(BaseModel):
    text: str = Field(min_length=2)
    limit: int = Field(default=10, ge=1, le=50)


class SearchFilters(BaseModel):
    source_type: list[str] = Field(default_factory=list)
    language: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    geography: str | None = None
    domain: str | None = None
    confidence_min: float | None = Field(default=None, ge=0, le=1)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class CrossCorpusSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    intent: str = "research"
    target_slots: list[str] = Field(default_factory=list)
    corpora: list[str] = Field(default_factory=list)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    numeric_mode: Literal["boost", "strict"] = "boost"
    max_corpora: int = Field(default=4, ge=1, le=5)
    k_per_corpus: int = Field(default=12, ge=1, le=50)
    final_k: int = Field(default=20, ge=1, le=50)
    include_debug: bool = False
    allow_remote: bool = True


class CorpusRoute(BaseModel):
    corpus_id: str
    reason: str
    confidence: float = Field(default=0.7, ge=0, le=1)
    rewrites: list[str] = Field(default_factory=list)


class SearchRouting(BaseModel):
    routes: list[CorpusRoute] = Field(default_factory=list)
    target_slots: list[str] = Field(default_factory=list)


class RerankItem(BaseModel):
    chunk_id: str
    score: float = Field(ge=0, le=1)
    matched_slots: list[str] = Field(default_factory=list)
    reason: str = ""


class RerankResponse(BaseModel):
    items: list[RerankItem] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    status: Literal["expert_validated", "rejected", "deprecated", "superseded"]
    reviewer: str
    comment: str | None = None
    supersedes_id: str | None = None


class ToolResponse(BaseModel):
    data: Any
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_now_iso)


class WebSearchRequest(BaseModel):
    query: str | None = Field(default=None, min_length=2, max_length=4000)
    queries: list[str] = Field(default_factory=list, max_length=8)
    profile_ids: list[str] = Field(default_factory=list, max_length=2)
    allowed_domains: list[str] = Field(default_factory=list, max_length=5)
    search_context_size: Literal["low", "medium", "high"] = "high"
    region: str = Field(default="213", min_length=1, max_length=32)
    max_output_tokens: int = Field(default=1800, ge=200, le=5000)

    @model_validator(mode="after")
    def require_query(self):
        values = [value.strip() for value in self.queries if value.strip()]
        if self.query and self.query.strip():
            values.insert(0, self.query.strip())
        self.queries = list(dict.fromkeys(values))[:8]
        if not self.queries:
            raise ValueError("query or queries must contain text")
        return self


class AgenticRAGRequest(BaseModel):
    query: str = Field(min_length=2, max_length=8000)
    allow_external_web: bool = False
    web_profile_ids: list[str] = Field(default_factory=list, max_length=9)
    corpora: list[str] = Field(default_factory=list, max_length=5)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    max_iterations: int = Field(default=3, ge=1, le=3)
    answer_language: Literal["ru", "en"] = "ru"
    include_debug: bool = True
    focus_document_ids: list[str] = Field(default_factory=list, max_length=100)


class GraphPathwaysRequest(BaseModel):
    query: str | None = Field(default=None, max_length=2000)
    agent_job_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    entity_ids: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=50, ge=1, le=200)
    include_incomplete: bool = True


class QueryConstraints(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    geography: str | None = None
    time_range: str | None = None


class ParsedAgentQuery(BaseModel):
    intent: str = "research"
    domain: str = "mining_metallurgy"
    entities: dict[str, list[str]] = Field(default_factory=dict)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    required_slots: list[str] = Field(default_factory=list)
    optional_slots: list[str] = Field(default_factory=list)
    requires_numeric_answer: bool = False
    requires_geography_comparison: bool = False
    requires_time_filter: bool = False
    answer_type: Literal["table", "review", "experiment_list", "recommendation", "gap_analysis"] = "review"


class RetrievalPlan(BaseModel):
    iteration: int = Field(ge=1, le=3)
    strategy: Literal["broad", "missing_slots", "fallback"]
    goal: str
    queries: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    graph_tools: list[str] = Field(default_factory=list)
    web_profiles: list[str] = Field(default_factory=list, max_length=2)
    drift_notes: list[str] = Field(default_factory=list)


class AgentEvidenceItem(BaseModel):
    id: str
    source_id: str
    source_type: str
    title: str | None = None
    snippet: str
    url: str | None = None
    file_name: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    year: int | None = None
    geography: str | None = None
    supports_slots: list[str] = Field(default_factory=list)
    numeric_facts: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    direct: bool = True
    metadata_only: bool = False
    citation_label: str | None = None


class AgentEvidencePack(BaseModel):
    items: list[AgentEvidenceItem] = Field(default_factory=list)
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)


class SufficiencyVerdict(BaseModel):
    sufficient: bool = False
    score: int = Field(default=0, ge=0, le=100)
    action: Literal["answer_full", "search_more", "answer_partial", "no_data"] = "search_more"
    covered_slots: dict[str, str] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    critical_missing: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    next_search_focus: list[str] = Field(default_factory=list)
    can_answer_partially: bool = False
    hard_gates: dict[str, bool] = Field(default_factory=dict)


class AgentState(BaseModel):
    query_id: str
    original_query: str
    parsed_query: ParsedAgentQuery | None = None
    iteration: int = 0
    search_history: list[dict[str, Any]] = Field(default_factory=list)
    evidence_pack: AgentEvidencePack = Field(default_factory=AgentEvidencePack)
    sufficiency: SufficiencyVerdict = Field(default_factory=SufficiencyVerdict)
    final_mode: str | None = None
    warnings: list[str] = Field(default_factory=list)
    llm_available: bool = True
    web_calls: int = 0


class AgentJobAccepted(BaseModel):
    job_id: str
    status: str
    events_url: str
    status_url: str
