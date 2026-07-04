from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
sys.path.insert(0, str(BACKEND_ROOT))

from src.mekg.service import MEKGService


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = re.sub(r"\s*%", "%", value)
    return re.sub(r"\s+", " ", value)


def fact_matches(fact: str, text: str) -> bool:
    wanted = re.findall(r"[a-zа-я0-9]+", normalize(fact))
    actual = re.findall(r"[a-zа-я0-9]+", normalize(text))
    if not wanted:
        return False
    matched = 0
    for token in wanted:
        if token.isdigit() or len(token) <= 3:
            ok = token in actual
        else:
            prefix = token[:5] if re.search(r"[а-я]", token) else token[:4]
            ok = any(value.startswith(prefix) for value in actual)
        matched += int(ok)
    return matched / len(wanted) >= 0.7


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Neo4j retrieval and gold-source audit")
    parser.add_argument("--fixture", type=Path, default=BACKEND_ROOT / "pilot" / "demo_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "demo-eval" / "retrieval-audit.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in fixture["documents"]}
    questions = fixture["questions"][: args.limit or None]
    element_ids = list(dict.fromkeys(
        source["element_id"] for question in questions for source in question["source_coordinates"]
    ))
    service = MEKGService()
    source_rows = service.repository.query(
        """
        UNWIND $ids AS id
        OPTIONAL MATCH (doc:Document)-[:HAS_VERSION]->(version:DocumentVersion)-[*1..3]->(ev:MEKG {id:id})
        RETURN id,doc.id AS document_id,doc.fileName AS file_name,ev.text AS text,
               ev.page_number AS page,ev.slide_number AS slide
        """,
        {"ids": element_ids},
    )
    source_by_id = {row["id"]: row for row in source_rows if row.get("text") is not None}
    results: dict[str, Any] = {}
    for index, question in enumerate(questions, start=1):
        document = documents[question["document_id"]]
        response = service.search_source_evidence(
            question["query"], corpora=[document["corpus_id"]], limit=30
        )
        evidence = response.evidence
        target_ranks = [
            rank for rank, item in enumerate(evidence, start=1)
            if item.get("document_id") == question["document_id"]
        ]
        expected_ids = {item["element_id"] for item in question["source_coordinates"]}
        retrieved_ids = {item.get("element_id") for item in evidence}
        gold_rows = [source_by_id[item] for item in expected_ids if item in source_by_id]
        gold_text = "\n".join(str(item.get("text") or "") for item in gold_rows)
        fact_checks = [
            {"fact": fact, "matched": fact_matches(fact, gold_text)}
            for fact in question["required_facts"]
        ]
        results[question["id"]] = {
            "document_id": question["document_id"],
            "file_name": document["file_name"],
            "gold_sources_found": len(gold_rows),
            "gold_sources_expected": len(expected_ids),
            "gold_fact_recall": round(
                sum(item["matched"] for item in fact_checks) / max(1, len(fact_checks)), 3
            ),
            "retrieval_document_hit": bool(target_ranks),
            "retrieval_best_document_rank": min(target_ranks) if target_ranks else None,
            "retrieval_exact_element_hit": bool(expected_ids & retrieved_ids),
            "returned": len(evidence),
            "warnings": response.warnings,
            "fact_checks": fact_checks,
        }
        print(json.dumps({
            "progress": f"{index}/{len(questions)}",
            "question_id": question["id"],
            "document_hit": bool(target_ranks),
            "rank": min(target_ranks) if target_ranks else None,
            "gold_fact_recall": results[question["id"]]["gold_fact_recall"],
        }, ensure_ascii=False), flush=True)

    values = list(results.values())
    summary = {
        "questions": len(values),
        "gold_source_coverage": round(
            sum(item["gold_sources_found"] == item["gold_sources_expected"] for item in values)
            / max(1, len(values)), 3
        ),
        "mean_gold_fact_recall": round(
            sum(item["gold_fact_recall"] for item in values) / max(1, len(values)), 3
        ),
        "retrieval_document_recall_at_30": round(
            sum(item["retrieval_document_hit"] for item in values) / max(1, len(values)), 3
        ),
        "retrieval_exact_element_recall_at_30": round(
            sum(item["retrieval_exact_element_hit"] for item in values) / max(1, len(values)), 3
        ),
        "retrieval_document_recall_at_5": round(
            sum(bool(item["retrieval_best_document_rank"] and item["retrieval_best_document_rank"] <= 5) for item in values)
            / max(1, len(values)), 3
        ),
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    service.repository.close()


if __name__ == "__main__":
    main()
