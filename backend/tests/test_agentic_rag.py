from __future__ import annotations

from types import SimpleNamespace

from src.mekg.agentic import AgenticRAG
from src.mekg.models import (
    AgentEvidenceItem,
    AgentEvidencePack,
    AgentState,
    CrossCorpusSearchRequest,
    ParsedAgentQuery,
    QueryConstraints,
    RerankResponse,
    RetrievalPlan,
    SufficiencyVerdict,
    WebSearchRequest,
)
from src.mekg.retrieval import CrossCorpusRetriever
from src.mekg.web_search import (
    WEB_SOURCE_PROFILES,
    YandexWebSearchClient,
    sanitize_web_query,
    select_web_profiles,
)

from .test_mekg import config


class FakeResponse:
    output_text = "A verified public summary with nickel velocity 0.05 m/s."

    def model_dump(self):
        return {
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": self.output_text,
                    "annotations": [
                        {"type": "url_citation", "url": "https://arxiv.org/abs/1234", "title": "Paper"},
                        {"type": "url_citation", "url": "https://example.com/not-allowed", "title": "Bad"},
                    ],
                }],
            }]
        }


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


def test_web_profiles_respect_five_domain_contract():
    assert WEB_SOURCE_PROFILES
    assert all(1 <= len(profile.domains) <= 5 for profile in WEB_SOURCE_PROFILES.values())


def test_web_query_redacts_paths_ids_emails_and_configured_terms():
    value = sanitize_web_query(
        r"C:\secret\report.pdf doc_123456789 user@example.com Norilsk-secret nickel",
        ("Norilsk-secret",),
    )
    assert "report.pdf" not in value
    assert "doc_123456789" not in value
    assert "user@example.com" not in value
    assert "Norilsk-secret" not in value
    assert "nickel" in value


def test_web_search_parses_only_allowlisted_annotations(tmp_path):
    responses = FakeResponses()
    client = YandexWebSearchClient(
        config(tmp_path), client=SimpleNamespace(responses=responses)
    )
    result = client.search(WebSearchRequest(
        query="nickel hydrometallurgy",
        allowed_domains=["arxiv.org"],
        search_context_size="high",
    ))
    assert [item["url"] for item in result.evidence] == ["https://arxiv.org/abs/1234"]
    tool = responses.calls[0]["tools"][0]
    assert tool["filters"]["allowed_domains"] == ["arxiv.org"]
    assert tool["user_location"]["region"] == "213"


def test_adaptive_profile_router_prefers_patents_and_mining():
    selected = select_web_profiles("патент на выщелачивание никелевой руды")
    assert selected == ["patents", "mining_metals"]


def test_closed_custom_domain_is_still_metadata_only():
    payload = {
        "annotations": [{"type": "url_citation", "url": "https://www.scopus.com/record/1"}]
    }
    sources = YandexWebSearchClient._extract_sources(
        payload, ("scopus.com",), "custom", False, "custom"
    )
    assert sources[0]["metadata_only"] is True
    assert sources[0]["evidence_tier"] == "metadata"


def test_numeric_hard_gate_rejects_value_without_unit():
    rag = AgenticRAG.__new__(AgenticRAG)
    state = AgentState(
        query_id="q",
        original_query="Какая скорость оптимальна?",
        parsed_query=ParsedAgentQuery(
            required_slots=["numeric_value_or_range", "unit", "basis_for_optimality", "source"],
            constraints=QueryConstraints(),
            requires_numeric_answer=True,
        ),
        llm_available=False,
        evidence_pack=AgentEvidencePack(items=[AgentEvidenceItem(
            id="e1", source_id="d1", source_type="local_chunk", snippet="Optimal velocity is 5",
            supports_slots=["numeric_value_or_range", "basis_for_optimality"],
            numeric_facts=[{"value_min": 5, "unit": None}], direct=True,
        )]),
    )
    verdict = rag._judge(state, "draft", final_iteration=True)
    assert verdict.action != "answer_full"
    assert verdict.hard_gates["numeric_value_and_unit"] is False
    assert "unit" in verdict.missing_slots


def test_rule_judge_allows_full_answer_only_with_direct_sourced_evidence():
    rag = AgenticRAG.__new__(AgenticRAG)
    items = [
        AgentEvidenceItem(
            id=f"e{index}", source_id=f"d{index}", source_type="local_chunk",
            snippet="Optimal velocity is 0.05 m/s because mass transfer reaches a maximum.",
            supports_slots=["numeric_value_or_range", "unit", "basis_for_optimality", "source"],
            numeric_facts=[{"value_min": 0.05, "unit": "m/s"}], direct=True, confidence=0.9,
        )
        for index in (1, 2)
    ]
    state = AgentState(
        query_id="q", original_query="What velocity is optimal?",
        parsed_query=ParsedAgentQuery(
            required_slots=["numeric_value_or_range", "unit", "basis_for_optimality", "source"],
            constraints=QueryConstraints(), requires_numeric_answer=True,
        ),
        llm_available=False,
        evidence_pack=AgentEvidencePack(
            items=items,
            covered_slots=["numeric_value_or_range", "unit", "basis_for_optimality", "source"],
        ),
    )
    verdict = rag._judge(state, "draft", final_iteration=False)
    assert verdict.action == "answer_full"
    assert verdict.score >= 80


def test_llm_judge_zero_without_missing_does_not_override_passed_hard_gates():
    rag = AgenticRAG.__new__(AgenticRAG)
    rag._structured = lambda *args, **kwargs: SufficiencyVerdict(
        score=0,
        action="no_data",
        missing_slots=["year", "geography", "basis_for_optimality"],
        critical_missing=["time_filter", "geography_comparison"],
        hard_gates={
            "time_filter": False,
            "geography_comparison": False,
            "basis_for_optimality": False,
        },
        reason="invalid low-confidence judge output with irrelevant missing gates",
    )
    state = AgentState(
        query_id="q",
        original_query="What recoveries are stated for Cuprion Co Ni Cu Mn?",
        parsed_query=ParsedAgentQuery(
            required_slots=["numeric_value_or_range", "unit", "process", "material_or_system", "source"],
            constraints=QueryConstraints(),
            requires_numeric_answer=True,
        ),
        llm_available=True,
        evidence_pack=AgentEvidencePack(
            items=[
                AgentEvidenceItem(
                    id="e1",
                    source_id="d1",
                    source_type="graph_source",
                    snippet="Cuprion process recoveries: Co 91%, Ni 85%, Cu 61%, Mn 60%.",
                    supports_slots=["numeric_value_or_range", "unit", "process", "material_or_system"],
                    numeric_facts=[
                        {"value_min": 91, "unit": "%"},
                        {"value_min": 85, "unit": "%"},
                    ],
                    direct=True,
                    confidence=0.9,
                )
            ],
            covered_slots=["numeric_value_or_range", "unit", "process", "material_or_system", "source"],
        ),
    )

    verdict = rag._judge(state, "draft", final_iteration=True)

    assert verdict.action == "answer_full"
    assert verdict.score >= 80
    assert verdict.missing_slots == []


def test_fallback_iteration_requires_more_core_overlap_for_direct_evidence():
    plan = RetrievalPlan(
        iteration=3, strategy="fallback", goal="analogy", queries=["q"],
    )
    assert AgenticRAG._is_direct(
        "nickel electrowinning catholyte circulation",
        "nickel electrowinning flow measurements",
        plan,
    )
    assert not AgenticRAG._is_direct(
        "nickel electrowinning catholyte circulation",
        "copper electrorefining hydrodynamics",
        plan,
    )


def test_direct_guard_requires_material_and_process_not_only_shared_leaching_term():
    plan = RetrievalPlan(iteration=1, strategy="broad", goal="search", queries=["q"])
    assert not AgenticRAG._is_direct(
        "технологии выщелачивания никеля",
        "электрогидравлическая обработка повышает выщелачивание золота",
        plan,
    )
    assert AgenticRAG._is_direct(
        "технологии выщелачивания никеля",
        "сернокислотное выщелачивание никеля при 80 °C",
        plan,
    )


class BM25OnlyStore:
    def initialize_schema(self):
        return {}

    def bm25_search(self, query, *, corpora, limit, filters):
        return [{
            "chunk_id": "c1", "document_id": "d1", "corpus_id": corpora[0],
            "text": "Nickel recovery evidence", "file_name": "paper.pdf", "source_type": "pdf",
            "bm25_score": 2.5, "candidate_entities": [], "page_number": 1,
        }]

    def dense_search(self, *args, **kwargs):
        raise AssertionError("dense search must not run after embedding failure")


def test_cross_corpus_search_degrades_to_bm25_when_query_embedding_fails():
    retriever = CrossCorpusRetriever.__new__(CrossCorpusRetriever)
    retriever.store = BM25OnlyStore()
    retriever.embeddings = SimpleNamespace(embed_query=lambda _query: (_ for _ in ()).throw(PermissionError()))
    retriever._rerank = lambda request, candidates: (RerankResponse(), "rerank skipped")
    result = retriever.search(CrossCorpusSearchRequest(
        query="nickel recovery", corpora=["scientific_articles"], final_k=5,
    ))
    assert result.data["results"][0]["chunk_id"] == "c1"
    assert any("BM25 fallback" in warning for warning in result.warnings)


def test_retrieval_readability_gate_rejects_broken_pdf_font_maps():
    broken = "\x06\x08\x11\x0b" * 30 + " золото 50 % "
    readable = "Извлечение золота составило 92,7 % при центробежной концентрации материала."
    assert CrossCorpusRetriever._is_readable(readable)
    assert not CrossCorpusRetriever._is_readable(broken)


def test_retrieval_text_cleaner_removes_controls_without_losing_coordinates():
    assert CrossCorpusRetriever._clean_text("Cu\x12 91 %\nNi 85 %") == "Cu 91 %\nNi 85 %"


def test_analysis_normalizes_model_specific_numeric_slots_to_controlled_vocabulary():
    parsed = ParsedAgentQuery(
        required_slots=["извлечение_Co_процент", "операции_технологической_схемы"],
        requires_numeric_answer=True,
    )
    normalized = AgenticRAG._normalize_analysis(
        parsed, "Какие извлечения Co, Ni, Cu и Mn заявлены для процесса Cuprion?"
    )
    assert normalized.required_slots == [
        "numeric_value_or_range", "unit", "process", "material_or_system", "source"
    ]
    assert all("Co" not in slot for slot in normalized.required_slots)


def test_direct_guard_requires_unique_process_name_and_chemical_symbols():
    plan = RetrievalPlan(iteration=1, strategy="broad", goal="search", queries=["q"])
    query = "Какие извлечения Co, Ni, Cu и Mn заявлены для процесса Cuprion?"
    assert AgenticRAG._is_direct(
        query,
        "Процесс Cuprion: Co 91%, Ni 85%, Cu 61%, Mn 60% при аммиачном выщелачивании.",
        plan,
    )
    assert not AgenticRAG._is_direct(
        query,
        "Курс содержит модули: геология, добыча, обогащение и металлургия.",
        plan,
    )
    assert not AgenticRAG._is_direct(
        query,
        "Процесс ЦНИГРИ: Co 80%, Ni 83%, Mn 71%.",
        plan,
    )
