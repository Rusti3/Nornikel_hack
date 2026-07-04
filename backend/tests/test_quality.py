from pathlib import Path

from src.mekg.qa import MEKGQualityAuditor
from src.mekg.rdf import RDFExporter


class RangeSnapshotRepository:
    def graph_snapshot(self, *, verified_only=True):
        nodes = [
            {
                "id": "measurement_range",
                "labels": ["MEKG", "Measurement"],
                "properties": {
                    "id": "measurement_range",
                    "value_min": 95.0,
                    "value_max": 97.0,
                    "validation_status": "machine_validated",
                },
            },
            {"id": "property_recovery", "labels": ["MEKG", "Property"], "properties": {"id": "property_recovery"}},
            {"id": "unit_percent", "labels": ["MEKG", "Unit"], "properties": {"id": "unit_percent"}},
            {"id": "chunk_1", "labels": ["MEKG", "Chunk"], "properties": {"id": "chunk_1"}},
        ]
        relationships = [
            {"source": "measurement_range", "type": "MEASURES_PROPERTY", "target": "property_recovery"},
            {"source": "measurement_range", "type": "HAS_UNIT", "target": "unit_percent"},
            {"source": "measurement_range", "type": "EVIDENCED_BY", "target": "chunk_1"},
        ]
        return {"nodes": nodes, "relationships": relationships}


class QueryRecordingRepository:
    def __init__(self):
        self.queries = []

    def query(self, query):
        self.queries.append(query)
        if "RETURN DISTINCT type(r)" in query:
            return []
        if "document_id" in query:
            return []
        return [{"value": 0}]


class ConformingExporter:
    def validate(self):
        return {"conforms": True, "violations": 0, "report": ""}


def test_shacl_accepts_measurement_range_without_scalar_value():
    ontology_dir = Path(__file__).resolve().parents[1] / "ontology"
    result = RDFExporter(RangeSnapshotRepository(), ontology_dir).validate()
    assert result["conforms"] is True
    assert result["violations"] == 0


def test_quality_metric_accepts_scalar_bound_or_range():
    repository = QueryRecordingRepository()
    report = MEKGQualityAuditor(repository, ConformingExporter()).run()
    measurement_query = next(query for query in repository.queries if "MEKG:Measurement" in query)
    assert "n.numeric_value IS NULL AND n.value_min IS NULL AND n.value_max IS NULL" in measurement_query
    assert report["metrics"]["measurements_invalid"] == 0
