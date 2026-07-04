from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rdf import RDFExporter
from .repository import MEKGRepository
from .schema import ALLOWED_RELATIONSHIPS


class MEKGQualityAuditor:
    def __init__(self, repository: MEKGRepository, exporter: RDFExporter) -> None:
        self.repository = repository
        self.exporter = exporter

    def run(self) -> dict[str, Any]:
        metrics = {}
        metrics["documents"] = self._scalar("MATCH (n:MEKG:Document) RETURN count(n) AS value")
        metrics["nodes"] = self._scalar("MATCH (n:MEKG) RETURN count(n) AS value")
        metrics["relationships"] = self._scalar("MATCH (:MEKG)-[r]->(:MEKG) RETURN count(r) AS value")
        metrics["staging_candidates"] = self._scalar("MATCH (n:MEKGStaging) RETURN count(n) AS value")
        metrics["facts_without_evidence"] = self._scalar(
            "MATCH (n:MEKG) WHERE (n:Claim OR n:Measurement OR n:Condition OR n:RelationshipAssertion) "
            "AND NOT (n)-[:EVIDENCED_BY]->() RETURN count(n) AS value"
        )
        metrics["measurements_invalid"] = self._scalar(
            "MATCH (n:MEKG:Measurement) WHERE "
            "(n.numeric_value IS NULL AND n.value_min IS NULL AND n.value_max IS NULL) "
            "OR NOT (n)-[:MEASURES_PROPERTY]->(:Property) "
            "OR NOT (n)-[:HAS_UNIT]->(:Unit) RETURN count(n) AS value"
        )
        metrics["conditions_invalid"] = self._scalar(
            "MATCH (n:MEKG:Condition) WHERE n.comparator IS NULL OR NOT (n)-[:HAS_PARAMETER]->(:Parameter) "
            "OR NOT (n)-[:HAS_UNIT]->(:Unit) RETURN count(n) AS value"
        )
        metrics["claims_without_pack"] = self._scalar(
            "MATCH (n:MEKG:Claim) WHERE NOT (n)-[:SUPPORTED_BY]->(:EvidencePack) RETURN count(n) AS value"
        )
        metrics["orphan_facts"] = self._scalar(
            "MATCH (n:MEKG) WHERE (n:Claim OR n:Measurement OR n:Condition OR n:Experiment) "
            "AND NOT (n)--() RETURN count(n) AS value"
        )
        relationship_types = {
            row["type"] for row in self.repository.query("MATCH (:MEKG)-[r]->(:MEKG) RETURN DISTINCT type(r) AS type")
        }
        disallowed = sorted(relationship_types - ALLOWED_RELATIONSHIPS)
        shacl = self.exporter.validate()
        passed = (
            metrics["facts_without_evidence"] == 0
            and metrics["measurements_invalid"] == 0
            and metrics["conditions_invalid"] == 0
            and metrics["claims_without_pack"] == 0
            and metrics["orphan_facts"] == 0
            and not disallowed
            and shacl["conforms"]
        )
        return {
            "passed": passed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "disallowed_relationship_types": disallowed,
            "shacl": shacl,
            "documents": self.document_coverage(),
        }

    def document_coverage(self) -> list[dict[str, Any]]:
        return self.repository.query(
            """
            MATCH (d:MEKG:Document)-[:HAS_VERSION]->(v:DocumentVersion)
            OPTIONAL MATCH (v)-[:HAS_PAGE|HAS_CHUNK|HAS_PUBLICATION*1..2]->(source:MEKG)
            OPTIONAL MATCH (fact:MEKG)-[:EVIDENCED_BY]->(source)
            RETURN d.id AS document_id,d.fileName AS file_name,d.category AS category,d.status AS status,
                   count(DISTINCT source) AS source_elements,count(DISTINCT fact) AS evidence_linked_facts
            ORDER BY category,file_name
            """
        )

    def write_report(self, directory: str | Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report = self.run()
        json_path = directory / "qa-report.json"
        markdown_path = directory / "qa-report.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# MEKG pilot QA report",
            "",
            f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
            "",
            "## Metrics",
            "",
            *[f"- {key}: {value}" for key, value in report["metrics"].items()],
            "",
            f"- SHACL conforms: {report['shacl']['conforms']}",
            f"- SHACL violations: {report['shacl']['violations']}",
            f"- Disallowed relationship types: {report['disallowed_relationship_types'] or 'none'}",
            "",
            "## Document coverage",
            "",
            "| Category | File | Status | Source elements | Evidence-linked facts |",
            "|---|---|---:|---:|---:|",
        ]
        lines.extend(
            f"| {row['category'] or ''} | {row['file_name']} | {row['status']} | {row['source_elements']} | {row['evidence_linked_facts']} |"
            for row in report["documents"]
        )
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, markdown_path

    def _scalar(self, query: str) -> int:
        rows = self.repository.query(query)
        return int(rows[0]["value"]) if rows else 0
