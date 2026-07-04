from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
DEFAULT_FIXTURE = BACKEND_ROOT / "pilot" / "demo_questions.json"
DEFAULT_OUTPUT = ROOT / "questions.md"


def _json_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _coord(value: dict[str, Any]) -> str:
    parts = [value.get("element_id") or "unknown"]
    if value.get("page") is not None:
        parts.append(f"стр. {value['page']}")
    if value.get("slide") is not None:
        parts.append(f"слайд {value['slide']}")
    return ", ".join(parts)


def render(
    fixture: dict[str, Any],
    readiness: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> str:
    qa = ((readiness.get("qa") or {}).get("result_json") or {})
    inventory = qa.get("documents") or []
    inventory.sort(key=lambda item: (item.get("category") or "", item.get("file_name") or ""))
    selected = {item["document_id"]: item for item in fixture["documents"]}
    results = (evaluation or {}).get("results") or {}
    evidence_ready = sum(int(item.get("evidence_linked_facts") or 0) > 0 for item in inventory)
    categories = Counter(item.get("category") or "Без категории" for item in inventory)

    lines = [
        "# Демо-вопросы по обработанным рабочим документам",
        "",
        "> Файл генерируется из `backend/pilot/demo_questions.json` и актуального QA snapshot. "
        "Внешний Web Search для этого набора выключен.",
        "",
        "## Покрытие",
        "",
        f"- Документов в Neo4j: **{len(inventory)}**.",
        f"- Документов с evidence-фактами: **{evidence_ready}**.",
        f"- Демо-набор: **{len(fixture['documents'])} документов / {len(fixture['questions'])} вопросов**.",
        f"- QA: **{'PASS' if qa.get('passed') else 'NOT PASS'}**, SHACL нарушений: "
        f"**{((qa.get('shacl') or {}).get('violations', 'n/a'))}**.",
        "- Корпусы: " + ", ".join(f"{name} — {count}" for name, count in sorted(categories.items())) + ".",
        "",
        "### Правило выбора",
        "",
        fixture["selection_policy"],
        "`ОР 01_26.pdf` исключен только из демо-набора: его текстовый слой содержит управляющие "
        "символы и не проходит readability gate. В полном инвентаре документ сохранен.",
        "",
        "## 50 вопросов и эталоны",
        "",
    ]
    by_document: dict[str, list[dict[str, Any]]] = {}
    for question in fixture["questions"]:
        by_document.setdefault(question["document_id"], []).append(question)

    for number, document in enumerate(fixture["documents"], start=1):
        lines.extend([
            f"### {number}. {document['file_name']}",
            "",
            f"Корпус: `{document['corpus_id']}` · document id: `{document['document_id']}` · "
            f"evidence-фактов: {document['evidence_linked_facts']}.",
            "",
        ])
        if document.get("selection_note"):
            lines.extend([f"Примечание: {document['selection_note']}.", ""])
        for question in by_document.get(document["document_id"], []):
            outcome = results.get(question["id"]) or {}
            auto = outcome.get("auto_review") or {}
            result = outcome.get("response") or {}
            actual = ((result.get("result") or {}).get("answer_markdown") or "")
            mode = (result.get("result") or {}).get("mode") or "NOT_RUN"
            confidence = (result.get("result") or {}).get("confidence")
            verdict = outcome.get("manual_verdict") or (
                "AUTO_PASS" if auto.get("pass") else "NOT_RUN" if not outcome else "NEEDS_REVIEW"
            )
            lines.extend([
                f"#### {question['id']} · {question['kind']}",
                "",
                f"**Вопрос:** {question['query']}",
                "",
                f"**Эталон:** {question['gold_answer']}",
                "",
                "**Обязательные факты:** " + "; ".join(question["required_facts"]),
                "",
                "**Координаты:** " + "; ".join(_coord(item) for item in question["source_coordinates"]),
                "",
                "**Нельзя утверждать:** " + "; ".join(question.get("forbidden_claims") or []),
                "",
                f"**Результат:** `{verdict}` · mode `{mode}` · confidence "
                f"`{confidence if confidence is not None else 'n/a'}`.",
                "",
            ])
            if actual:
                lines.extend(["**Фактический ответ системы:**", "", actual, ""])

    lines.extend([
        "## Полный инвентарь обработанных документов",
        "",
        "`demo` означает включение в набор 50 вопросов; `evidence-ready` — наличие хотя бы одного "
        "связанного факта. Наличие документа без evidence еще не означает возможность доказательного ответа.",
        "",
        "| № | Корпус | Документ | Статус | Source elements | Evidence facts | Demo |",
        "|---:|---|---|---|---:|---:|:---:|",
    ])
    for index, item in enumerate(inventory, start=1):
        facts = int(item.get("evidence_linked_facts") or 0)
        lines.append(
            f"| {index} | {_escape(item.get('category'))} | {_escape(item.get('file_name'))} | "
            f"{_escape(item.get('status'))}{' · evidence-ready' if facts else ''} | "
            f"{int(item.get('source_elements') or 0)} | {facts} | "
            f"{'✓' if item.get('document_id') in selected else ''} |"
        )
    lines.extend(["", f"Сформировано по QA snapshot: `{qa.get('generated_at', 'unknown')}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--readiness-url", default="http://localhost:8000/api/mekg/v1/readiness")
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    readiness = (
        json.loads(args.readiness_json.read_text(encoding="utf-8"))
        if args.readiness_json else _json_url(args.readiness_url)
    )
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else None
    args.output.write_text(render(fixture, readiness, evaluation), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "documents": len(fixture["documents"]),
             "questions": len(fixture["questions"])},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
