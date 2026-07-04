from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.mekg.config import MEKGConfig


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_demo_fixture_has_25_documents_and_two_questions_per_document():
    value = json.loads((BACKEND_ROOT / "pilot" / "demo_questions.json").read_text(encoding="utf-8"))
    documents = value["documents"]
    questions = value["questions"]
    assert len(documents) == 25
    assert len(questions) == 50
    assert len({item["document_id"] for item in documents}) == 25
    assert len({item["id"] for item in questions}) == 50
    assert set(Counter(item["corpus_id"] for item in documents).values()) == {5}
    assert set(Counter(item["document_id"] for item in questions).values()) == {2}


def test_demo_gold_contracts_have_coordinates_and_anti_hallucination_rules():
    value = json.loads((BACKEND_ROOT / "pilot" / "demo_questions.json").read_text(encoding="utf-8"))
    document_ids = {item["document_id"] for item in value["documents"]}
    for item in value["questions"]:
        assert item["document_id"] in document_ids
        assert item["gold_answer"].strip()
        assert item["required_facts"]
        assert item["source_coordinates"]
        assert all(source.get("element_id", "").startswith(("chunk_", "row_")) for source in item["source_coordinates"])
        assert item["forbidden_claims"]
        assert set(item["acceptable_modes"]) <= {"full_answer", "partial_answer_with_gaps"}


def test_secret_fields_are_not_exposed_by_config_repr(tmp_path):
    config = MEKGConfig(
        ontology_dir=tmp_path,
        artifacts_dir=tmp_path,
        llm_model="model",
        vision_model="vision",
        yandex_api_key="yandex-secret",
        yandex_folder_id="folder",
        yandex_base_url="https://example.invalid/v1",
        yandex_ocr_url="https://example.invalid/ocr",
        data_logging=False,
        max_concurrency=1,
        chunk_chars=1000,
        chunk_overlap=0,
        ocr_min_chars=0,
        postgres_password="postgres-secret",
        openrouter_api_keys=("openrouter-secret",),
    )
    rendered = repr(config)
    assert "yandex-secret" not in rendered
    assert "postgres-secret" not in rendered
    assert "openrouter-secret" not in rendered
