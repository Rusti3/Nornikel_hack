from __future__ import annotations

from pathlib import Path
from decimal import Decimal
from typing import Any
from urllib.parse import quote

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from .repository import MEKGRepository


MEKG = Namespace("https://nornickel.example/mekg/")


def _uri(identifier: str) -> URIRef:
    return MEKG[quote(str(identifier), safe="-_.~")]


class RDFExporter:
    def __init__(self, repository: MEKGRepository, ontology_dir: str | Path) -> None:
        self.repository = repository
        self.ontology_dir = Path(ontology_dir)

    def graph(self, *, verified_only: bool = True) -> Graph:
        snapshot = self.repository.graph_snapshot(verified_only=verified_only)
        graph = Graph()
        graph.bind("mekg", MEKG)
        graph.parse(self.ontology_dir / "mekg.ttl", format="turtle")
        graph.parse(self.ontology_dir / "vocab.ttl", format="turtle")
        for row in snapshot["nodes"]:
            subject = _uri(row["id"])
            for label in row["labels"]:
                if label not in {"MEKG", "CanonicalEntity"}:
                    graph.add((subject, RDF.type, MEKG[label]))
            for key, value in row["properties"].items():
                if key in {"embedding", "metadata_json"} or value is None:
                    continue
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if isinstance(item, (str, int, float, bool)):
                        if key in {"numeric_value", "value_min", "value_max", "confidence"} and isinstance(item, (int, float)):
                            graph.add((subject, MEKG[key], Literal(Decimal(str(item)), datatype=XSD.decimal)))
                        else:
                            graph.add((subject, MEKG[key], Literal(item)))
        for row in snapshot["relationships"]:
            graph.add((_uri(row["source"]), MEKG[row["type"]], _uri(row["target"])))
        return graph

    def serialize(self, format: str = "turtle", *, verified_only: bool = True) -> str:
        mapping = {"turtle": "turtle", "ttl": "turtle", "jsonld": "json-ld", "json-ld": "json-ld"}
        if format not in mapping:
            raise ValueError("format must be turtle or jsonld")
        value = self.graph(verified_only=verified_only).serialize(format=mapping[format], indent=2)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def validate(self) -> dict[str, Any]:
        data_graph = self.graph(verified_only=True)
        shapes = Graph().parse(self.ontology_dir / "shapes.ttl", format="turtle")
        ontology = Graph().parse(self.ontology_dir / "mekg.ttl", format="turtle")
        conforms, report_graph, report_text = validate(
            data_graph=data_graph,
            shacl_graph=shapes,
            ont_graph=ontology,
            inference="rdfs",
            abort_on_first=False,
            allow_infos=True,
            allow_warnings=True,
        )
        violations = list(report_graph.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#ValidationResult")))
        return {"conforms": bool(conforms), "violations": len(set(violations)), "report": report_text}
