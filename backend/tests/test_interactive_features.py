from __future__ import annotations

import json
from pathlib import Path

from src.mekg.graph_pathways import build_pathways
from src.mekg.interactive_ingest import infer_upload_category
from src.mekg.models import AgenticRAGRequest, GraphPathwaysRequest, SearchFilters
from src.mekg.reports import build_markdown_report
from src.mekg.api import _safe_upload_name


class FakePathRepository:
    def query(self, _cypher, _parameters):
        return [{
            "experiment": {"id": "e1", "labels": ["MEKG", "Experiment"], "name": "Опыт 1", "confidence": 0.9, "validation_status": "machine_validated"},
            "materials": [{"id": "m1", "labels": ["Material"], "name": "Никелевый раствор", "confidence": 0.9, "validation_status": "machine_validated"}],
            "processes": [{"id": "p1", "labels": ["Process"], "name": "Электроэкстракция", "confidence": 0.85, "validation_status": "machine_validated"}],
            "equipment": [{"id": "q1", "labels": ["Equipment"], "name": "Электролизная ванна", "confidence": 0.8, "validation_status": "machine_validated", "relationship": "USES_EQUIPMENT"}],
            "measurements": [{"id": "r1", "labels": ["Measurement"], "name": "Извлечение", "numeric_value": 91.0, "unit_normalized": "%", "confidence": 0.9, "validation_status": "machine_validated"}],
            "claims": [],
            "evidence": {"id": "chunk1", "page": 19, "slide": None, "text": "Извлечение никеля 91%."},
            "documents": [{"id": "d1", "file_name": "report.pdf", "category": "Доклады"}],
        }, {
            "experiment": {"id": "e2", "labels": ["MEKG", "Experiment"], "name": "Опыт без оборудования", "confidence": 0.6, "validation_status": "machine_validated"},
            "materials": [{"id": "m2", "labels": ["Material"], "name": "Шахтная вода", "confidence": 0.7, "validation_status": "machine_validated"}],
            "processes": [{"id": "p2", "labels": ["Process"], "name": "Очистка", "confidence": 0.7, "validation_status": "machine_validated"}],
            "equipment": [], "measurements": [], "claims": [],
            "evidence": {"id": "chunk2", "page": 5, "text": "Пилотный опыт очистки."},
            "documents": [{"id": "d2", "file_name": "water.docx", "category": "Обзоры"}],
        }]


def test_graph_pathways_project_real_experiment_relationships_and_gaps():
    value = build_pathways(FakePathRepository(), GraphPathwaysRequest(limit=10))
    assert value["coverage"] == {"total": 2, "complete": 1, "incomplete": 1}
    full = value["pathways"][0]
    assert full["result_title"] == "Извлечение: 91 %"
    assert full["evidence"]["document"]["file_name"] == "report.pdf"
    incomplete = value["pathways"][1]
    assert set(incomplete["missing_stages"]) == {"equipment", "result"}
    relation_types = {item["type"] for item in value["relationships"]}
    assert {"USES_MATERIAL", "STUDIES_PROCESS", "USES_EQUIPMENT", "PRODUCED_MEASUREMENT"} <= relation_types


def test_graph_pathways_can_hide_incomplete_chains():
    value = build_pathways(
        FakePathRepository(), GraphPathwaysRequest(limit=10, include_incomplete=False)
    )
    assert value["coverage"]["total"] == 1
    assert value["pathways"][0]["complete"] is True


def test_upload_category_defaults_and_explicit_corpus():
    assert infer_upload_category("Литературный обзор гипса.docx", "auto") == "Обзоры"
    assert infer_upload_category("unknown.pdf", "internal_reports") == "Доклады"
    assert infer_upload_category("article.pdf", None) == "Статьи"


def test_upload_name_repairs_utf8_multipart_mojibake():
    broken = "Использование.docx".encode("utf-8").decode("latin-1")
    assert _safe_upload_name(broken) == "Использование.docx"


def test_focus_documents_are_bounded_in_public_contract():
    request = AgenticRAGRequest(query="Что показал приложенный отчёт?", focus_document_ids=["doc_1"])
    assert request.focus_document_ids == ["doc_1"]
    assert SearchFilters(document_ids=["doc_1"]).document_ids == ["doc_1"]


def test_markdown_report_contains_answer_sources_and_no_internal_state():
    report = build_markdown_report({
        "status": "complete",
        "request_json": {"query": "Какова скорость?", "private_prompt": "do not export"},
        "result_json": {
            "mode": "partial_answer_with_gaps",
            "confidence": 0.72,
            "answer_markdown": "Скорость **0,05 м/с** [S1].",
            "sources": [{"label": "S1", "file_name": "review.pdf", "page": 7}],
            "gaps": ["basis_for_optimality"],
            "warnings": [],
            "state": {"reasoning": "never export"},
        },
    })
    assert "0,05 м/с" in report
    assert "review.pdf" in report
    assert "стр. 7" in report
    assert "never export" not in report
    assert "do not export" not in report


def test_expert_fixture_has_ten_decision_complete_questions():
    path = Path(__file__).resolve().parents[1] / "pilot" / "expert_questions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert len(value["questions"]) == 10
    assert all(item["required_slots"] and item["acceptable_modes"] for item in value["questions"])


def test_agent_worker_has_json_cache_serializer():
    from src.mekg import agent_worker

    assert agent_worker.json.loads(agent_worker.json.dumps({"ok": True})) == {"ok": True}
