from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from neo4j import GraphDatabase, RoutingControl
from rdflib import Graph, OWL, RDF, RDFS

from .models import ChunkExtraction, ParsedDocument, ReviewDecision, ValidationStatus
from .parsers import stable_id
from .schema import ALLOWED_LABELS, ALLOWED_RELATIONSHIPS, ENTITY_LABELS
from .units import UnitNormalizer


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(label: str, default: str) -> str:
    normalized = "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", label) if part)
    return normalized if normalized in ALLOWED_LABELS else default


class MEKGRepository:
    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j://neo4j:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD is required for MEKG")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self.units = UnitNormalizer()

    def close(self) -> None:
        self.driver.close()

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
            routing_=RoutingControl.READ,
        )
        return [record.data() for record in records]

    def write(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
            routing_=RoutingControl.WRITE,
        )
        return [record.data() for record in records]

    def initialize_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT mekg_id IF NOT EXISTS FOR (n:MEKG) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT mekg_staging_id IF NOT EXISTS FOR (n:MEKGStaging) REQUIRE n.id IS UNIQUE",
            "CREATE TEXT INDEX mekg_canonical_name IF NOT EXISTS FOR (n:CanonicalEntity) ON (n.canonical_name)",
            "CREATE TEXT INDEX mekg_claim_text IF NOT EXISTS FOR (n:Claim) ON (n.text)",
            "CREATE RANGE INDEX mekg_condition_min IF NOT EXISTS FOR (n:Condition) ON (n.value_min)",
            "CREATE RANGE INDEX mekg_condition_max IF NOT EXISTS FOR (n:Condition) ON (n.value_max)",
            "CREATE RANGE INDEX mekg_measurement_value IF NOT EXISTS FOR (n:Measurement) ON (n.numeric_value)",
            "CREATE RANGE INDEX mekg_confidence IF NOT EXISTS FOR (n:MEKG) ON (n.confidence)",
        ]
        for statement in statements:
            self.write(statement)

    def load_ontology_projection(self, ontology_dir: str) -> dict[str, int]:
        graph = Graph()
        for name in ("mekg.ttl", "vocab.ttl"):
            graph.parse(str(os.path.join(ontology_dir, name)), format="turtle")
        classes = sorted({str(subject) for subject in graph.subjects(RDF.type, OWL.Class)})
        properties = sorted(
            {str(subject) for subject in graph.subjects(RDF.type, OWL.ObjectProperty)}
            | {str(subject) for subject in graph.subjects(RDF.type, OWL.DatatypeProperty)}
        )
        class_rows = [{"id": stable_id("ontology_class", iri), "iri": iri, "name": iri.rsplit("/", 1)[-1]} for iri in classes]
        property_rows = [{"id": stable_id("ontology_property", iri), "iri": iri, "name": iri.rsplit("/", 1)[-1]} for iri in properties]
        self.write(
            "UNWIND $rows AS row MERGE (n:MEKG:OntologyClass {id:row.id}) SET n.iri=row.iri, n.name=row.name, n.validation_status='machine_validated'",
            {"rows": class_rows},
        )
        self.write(
            "UNWIND $rows AS row MERGE (n:MEKG:OntologyProperty {id:row.id}) SET n.iri=row.iri, n.name=row.name, n.validation_status='machine_validated'",
            {"rows": property_rows},
        )
        subclass_rows = []
        for child, parent in graph.subject_objects(RDFS.subClassOf):
            child_iri, parent_iri = str(child), str(parent)
            if child_iri in classes and parent_iri in classes:
                subclass_rows.append({"child": stable_id("ontology_class", child_iri), "parent": stable_id("ontology_class", parent_iri)})
        if subclass_rows:
            self.write(
                "UNWIND $rows AS row MATCH (a:OntologyClass {id:row.child}) MATCH (b:OntologyClass {id:row.parent}) MERGE (a)-[:SUBCLASS_OF]->(b)",
                {"rows": subclass_rows},
            )
        return {"classes": len(class_rows), "properties": len(property_rows), "subclasses": len(subclass_rows)}

    def reset_pilot(self) -> int:
        result = self.query("MATCH (n) WHERE n:MEKG OR n:MEKGStaging RETURN count(n) AS count")
        count = int(result[0]["count"]) if result else 0
        self.write("MATCH (n) WHERE n:MEKG OR n:MEKGStaging DETACH DELETE n")
        return count

    def store_source(self, document: ParsedDocument) -> None:
        now = _now()
        self.write(
            """
            MERGE (d:MEKG:Document {id: $document.id})
            SET d += $document.props
            MERGE (v:MEKG:DocumentVersion {id: $version.id})
            SET v += $version.props
            MERGE (d)-[:HAS_VERSION]->(v)
            """,
            {
                "document": {
                    "id": document.document_id,
                    "props": {
                        "fileName": document.file_name,
                        "file_type": document.file_type,
                        "source_locator": document.source_locator,
                        "category": document.category,
                        "title": document.title,
                        "language": document.language,
                        "created_at": now,
                        "updated_at": now,
                        "status": "parsed",
                        "validation_status": ValidationStatus.MACHINE_VALIDATED.value,
                    },
                },
                "version": {
                    "id": document.version_id,
                    "props": {
                        "sha256": document.sha256,
                        "size_bytes": document.size_bytes,
                        "created_at": now,
                        "warnings": document.warnings,
                        "validation_status": ValidationStatus.MACHINE_VALIDATED.value,
                    },
                },
            },
        )
        page_keys: set[tuple[str, int]] = set()
        rows = []
        for element in document.elements:
            page_kind = "Slide" if element.slide_number else "Page"
            page_no = element.slide_number or element.page_number
            page_id = None
            if page_no:
                page_id = stable_id(page_kind.lower(), f"{document.version_id}:{page_kind}:{page_no}")
                page_keys.add((page_kind, page_no))
            label = {
                "text": "Chunk",
                "table": "Table",
                "table_row": "TableRow",
                "figure": "Figure",
                "formula": "Formula",
            }[element.kind.value]
            relation = {
                "Chunk": "HAS_CHUNK",
                "Table": "HAS_TABLE",
                "TableRow": "HAS_ROW",
                "Figure": "HAS_FIGURE",
                "Formula": "HAS_FORMULA",
            }[label]
            parent_id = element.metadata.get("table_id") or element.metadata.get("derived_from_figure") or page_id or document.version_id
            rows.append(
                {
                    "id": element.id,
                    "label": label,
                    "page_id": page_id,
                    "parent_id": parent_id,
                    "relation": relation,
                    "props": {
                        "text": element.text,
                        "page_number": element.page_number,
                        "slide_number": element.slide_number,
                        "sheet_name": element.sheet_name,
                        "row_number": element.row_number,
                        "bbox": element.bbox,
                        "image_path": element.image_path,
                        "metadata_json": json.dumps(element.metadata, ensure_ascii=False),
                        "validation_status": ValidationStatus.RAW_EXTRACTED.value,
                        "created_at": now,
                    },
                }
            )
        pages = [
            {
                "id": stable_id(kind.lower(), f"{document.version_id}:{kind}:{number}"),
                "label": kind,
                "number": number,
            }
            for kind, number in sorted(page_keys, key=lambda item: item[1])
        ]
        if pages:
            self.write(
                """
                MATCH (v:MEKG:DocumentVersion {id: $version_id})
                UNWIND $pages AS row
                CALL apoc.merge.node(['MEKG', row.label], {id: row.id}, {number: row.number}, {}) YIELD node
                MERGE (v)-[:HAS_PAGE]->(node)
                """,
                {"version_id": document.version_id, "pages": pages},
            )
        if rows:
            self.write(
                """
                UNWIND $rows AS row
                CALL apoc.merge.node(['MEKG', row.label], {id: row.id}, row.props, row.props) YIELD node
                RETURN count(node) AS created
                """,
                {"version_id": document.version_id, "rows": rows},
            )
            self.write(
                """
                UNWIND $rows AS row
                MATCH (parent:MEKG {id:row.parent_id})
                MATCH (node:MEKG {id:row.id})
                CALL apoc.merge.relationship(parent, row.relation, {}, {}, node, {}) YIELD rel
                RETURN count(rel) AS linked
                """,
                {"rows": rows},
            )
        for publication in document.publications:
            pub_id = stable_id(
                "pub", f"{document.version_id}:{publication.start_page}:{publication.end_page}:{publication.title.casefold()}"
            )
            self.write(
                """
                MATCH (v:MEKG:DocumentVersion {id: $version_id})
                MERGE (p:MEKG:Publication {id: $id})
                SET p.title=$title, p.start_page=$start_page, p.end_page=$end_page,
                    p.authors=$authors, p.confidence=$confidence, p.needs_review=$needs_review,
                    p.validation_status=$status
                MERGE (v)-[:HAS_PUBLICATION]->(p)
                """,
                {
                    "version_id": document.version_id,
                    "id": pub_id,
                    "title": publication.title,
                    "start_page": publication.start_page,
                    "end_page": publication.end_page,
                    "authors": publication.authors,
                    "confidence": publication.confidence,
                    "needs_review": publication.needs_review,
                    "status": (
                        ValidationStatus.RAW_EXTRACTED.value
                        if publication.needs_review
                        else ValidationStatus.MACHINE_VALIDATED.value
                    ),
                },
            )

    def store_document_bundle(
        self,
        document: ParsedDocument,
        extractions: dict[str, ChunkExtraction],
        rejected: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Atomically merge a complete source document and its extracted facts.

        The full-corpus pipeline checkpoints LLM results in PostgreSQL and calls
        this method once. All node and relationship rows are consumed by two
        UNWIND clauses inside one Neo4j managed transaction.
        """
        now = _now()
        nodes: dict[str, dict[str, Any]] = {}
        relationships: dict[tuple[str, str, str], dict[str, Any]] = {}

        def node(identifier: str, labels: list[str], **props: Any) -> None:
            clean = {key: value for key, value in props.items() if value is not None}
            clean.setdefault("updated_at", now)
            current = nodes.setdefault(identifier, {"id": identifier, "labels": labels, "props": {}})
            current["props"].update(clean)

        def rel(source: str, relation: str, target: str, **props: Any) -> None:
            if relation not in ALLOWED_RELATIONSHIPS:
                raise ValueError(f"Relationship is not allowed by MEKG ontology: {relation}")
            relationships[(source, relation, target)] = {
                "source": source,
                "relation": relation,
                "target": target,
                "props": {key: value for key, value in props.items() if value is not None},
            }

        node(
            document.document_id,
            ["MEKG", "Document"],
            fileName=document.file_name,
            file_type=document.file_type,
            source_locator=document.source_locator,
            category=document.category,
            title=document.title,
            language=document.language,
            status="complete",
            validation_status=ValidationStatus.MACHINE_VALIDATED.value,
            created_at=now,
        )
        node(
            document.version_id,
            ["MEKG", "DocumentVersion"],
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            warnings=document.warnings,
            validation_status=ValidationStatus.MACHINE_VALIDATED.value,
            created_at=now,
        )
        rel(document.document_id, "HAS_VERSION", document.version_id)

        pages: set[tuple[str, int, str]] = set()
        for element in document.elements:
            page_kind = "Slide" if element.slide_number else "Page"
            page_number = element.slide_number or element.page_number
            page_id = None
            if page_number:
                page_id = stable_id(page_kind.lower(), f"{document.version_id}:{page_kind}:{page_number}")
                pages.add((page_kind, page_number, page_id))
            label = {
                "text": "Chunk", "table": "Table", "table_row": "TableRow",
                "figure": "Figure", "formula": "Formula",
            }[element.kind.value]
            relation = {
                "Chunk": "HAS_CHUNK", "Table": "HAS_TABLE", "TableRow": "HAS_ROW",
                "Figure": "HAS_FIGURE", "Formula": "HAS_FORMULA",
            }[label]
            node(
                element.id,
                ["MEKG", label],
                text=element.text,
                page_number=element.page_number,
                slide_number=element.slide_number,
                sheet_name=element.sheet_name,
                row_number=element.row_number,
                bbox=element.bbox,
                image_path=element.image_path,
                metadata_json=json.dumps(element.metadata, ensure_ascii=False, default=str),
                validation_status=ValidationStatus.RAW_EXTRACTED.value,
                created_at=now,
            )
            parent_id = element.metadata.get("table_id") or element.metadata.get("derived_from_figure") or page_id or document.version_id
            rel(parent_id, relation, element.id)
        for page_kind, number, page_id in pages:
            node(page_id, ["MEKG", page_kind], number=number, validation_status="machine_validated")
            rel(document.version_id, "HAS_PAGE", page_id)
        for publication in document.publications:
            publication_id = stable_id(
                "pub", f"{document.version_id}:{publication.start_page}:{publication.end_page}:{publication.title.casefold()}"
            )
            node(
                publication_id,
                ["MEKG", "Publication"],
                title=publication.title,
                start_page=publication.start_page,
                end_page=publication.end_page,
                authors=publication.authors,
                confidence=publication.confidence,
                needs_review=publication.needs_review,
                validation_status=("raw_extracted" if publication.needs_review else "machine_validated"),
            )
            rel(document.version_id, "HAS_PUBLICATION", publication_id)

        counts = {"entities": 0, "conditions": 0, "measurements": 0, "experiments": 0, "claims": 0, "experts": 0}
        for evidence_id, extraction in extractions.items():
            local: dict[str, str] = {}
            for item in extraction.entities:
                label = _safe_label(item.entity_type, "TopicTag")
                identifier = stable_id("entity", f"{label}:{item.canonical_name.casefold().strip()}")
                node(
                    identifier,
                    ["MEKG", "CanonicalEntity", label],
                    canonical_name=item.canonical_name.strip(), name_ru=item.name_ru, name_en=item.name_en,
                    aliases=sorted(set(item.aliases)), description=item.description, source="extracted",
                    confidence=item.confidence, validation_status="machine_validated", created_at=now,
                )
                rel(identifier, "EVIDENCED_BY", evidence_id)
                for alias in sorted({item.canonical_name, *item.aliases, item.name_ru, item.name_en} - {None, ""}):
                    term_id = stable_id("term", f"{identifier}:{alias.casefold()}")
                    node(term_id, ["MEKG", "Term"], text=alias, language=("ru" if re.search(r"[А-Яа-яЁё]", alias) else "en"))
                    rel(identifier, "HAS_TERM", term_id)
                local[item.local_id] = identifier
                counts["entities"] += 1
            for item in extraction.conditions:
                normalized = self.units.normalize(item.unit_original)
                identifier = stable_id("condition", f"{document.version_id}:{item.local_id}:{evidence_id}")
                parameter_id = stable_id("parameter", item.name.casefold())
                unit_id = stable_id("unit", normalized.symbol.casefold())
                node(
                    identifier, ["MEKG", "Condition"], parameter=item.name,
                    numeric_value=self.units.convert_value(item.value, normalized),
                    value_min=self.units.convert_value(item.value_min, normalized),
                    value_max=self.units.convert_value(item.value_max, normalized),
                    unit_original=item.unit_original, unit_normalized=normalized.symbol,
                    dimension=normalized.dimension, comparator=item.comparator,
                    source_text=item.evidence.quote, confidence=item.confidence,
                    approximate=item.approximate, validation_status="machine_validated", created_at=now,
                )
                node(parameter_id, ["MEKG", "CanonicalEntity", "Parameter"], canonical_name=item.name, validation_status="machine_validated")
                node(unit_id, ["MEKG", "CanonicalEntity", "Unit"], canonical_name=normalized.symbol, symbol=normalized.symbol, dimension=normalized.dimension, validation_status="machine_validated")
                rel(identifier, "HAS_PARAMETER", parameter_id)
                rel(identifier, "HAS_UNIT", unit_id)
                rel(identifier, "EVIDENCED_BY", evidence_id)
                local[item.local_id] = identifier
                counts["conditions"] += 1
            for item in extraction.measurements:
                normalized = self.units.normalize(item.unit_original)
                identifier = stable_id("measurement", f"{document.version_id}:{item.local_id}:{evidence_id}")
                property_id = stable_id("property", item.property_name.casefold())
                unit_id = stable_id("unit", normalized.symbol.casefold())
                node(
                    identifier, ["MEKG", "Measurement"], name=item.name, property_name=item.property_name,
                    numeric_value=self.units.convert_value(item.value, normalized),
                    value_min=self.units.convert_value(item.value_min, normalized),
                    value_max=self.units.convert_value(item.value_max, normalized),
                    comparator=item.comparator, unit_original=item.unit_original,
                    unit_normalized=normalized.symbol, dimension=normalized.dimension,
                    method=item.method, source_text=item.evidence.quote, confidence=item.confidence,
                    validation_status="machine_validated", created_at=now,
                )
                node(property_id, ["MEKG", "CanonicalEntity", "Property"], canonical_name=item.property_name, validation_status="machine_validated")
                node(unit_id, ["MEKG", "CanonicalEntity", "Unit"], canonical_name=normalized.symbol, symbol=normalized.symbol, dimension=normalized.dimension, validation_status="machine_validated")
                rel(identifier, "MEASURES_PROPERTY", property_id)
                rel(identifier, "HAS_UNIT", unit_id)
                rel(identifier, "EVIDENCED_BY", evidence_id)
                local[item.local_id] = identifier
                counts["measurements"] += 1
            for item in extraction.experiments:
                identifier = stable_id("experiment", f"{document.version_id}:{item.local_id}:{evidence_id}")
                node(identifier, ["MEKG", "Experiment"], name=item.name, experiment_type=item.experiment_type, effect=item.effect, confidence=item.confidence, source_text=item.evidence.quote, validation_status="machine_validated", created_at=now)
                rel(identifier, "EVIDENCED_BY", evidence_id)
                for field, relation in ((item.material_refs, "USES_MATERIAL"), (item.process_refs, "STUDIES_PROCESS"), (item.equipment_refs, "USES_EQUIPMENT"), (item.condition_refs, "HAS_CONDITION"), (item.measurement_refs, "PRODUCED_MEASUREMENT")):
                    for reference in field:
                        if reference in local:
                            rel(identifier, relation, local[reference], confidence=item.confidence)
                local[item.local_id] = identifier
                counts["experiments"] += 1
            for item in extraction.claims:
                identifier = stable_id("claim", f"{document.version_id}:{item.local_id}:{evidence_id}")
                pack_id = stable_id("evidence_pack", f"{document.version_id}:{evidence_id}:{identifier}")
                node(identifier, ["MEKG", "Claim"], text=item.text, claim_type=item.claim_type, limitations=item.limitations, recommendation=item.recommendation, geo_scope=item.geo_scope, countries=item.countries, source_text=item.evidence.quote, confidence=item.confidence, validation_status="machine_validated", created_at=now)
                node(pack_id, ["MEKG", "EvidencePack"], confidence=item.confidence, validation_status="machine_validated", created_at=now)
                rel(identifier, "EVIDENCED_BY", evidence_id)
                rel(identifier, "SUPPORTED_BY", pack_id)
                rel(pack_id, "HAS_EVIDENCE", evidence_id)
                rel(document.version_id, "EVIDENCED_BY", pack_id)
                for reference in [*item.entity_refs, *item.experiment_refs, *item.measurement_refs]:
                    if reference in local:
                        rel(identifier, "GENERALIZES", local[reference], confidence=item.confidence, evidence_id=evidence_id)
                local[item.local_id] = identifier
                counts["claims"] += 1
            for item in extraction.experts:
                identifier = stable_id("expert", item.name.casefold().strip())
                node(identifier, ["MEKG", "CanonicalEntity", "Expert", "Author"], canonical_name=item.name, organization=item.organization, lab=item.lab, topics=item.topics, confidence=item.confidence, validation_status="machine_validated")
                rel(identifier, "EVIDENCED_BY", evidence_id)
                local[item.local_id] = identifier
                counts["experts"] += 1

        for index, item in enumerate(rejected or []):
            payload = item.get("payload", {})
            candidate_id = stable_id("candidate", f"{document.version_id}:{index}:{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}")
            node(candidate_id, ["MEKGStaging", "ExtractionCandidate"], kind=item.get("kind"), payload_json=json.dumps(payload, ensure_ascii=False, default=str), reason=item.get("reason"), validation_status="raw_extracted", created_at=now)
            rel(candidate_id, "EVIDENCED_BY", document.version_id)

        parameters = {"nodes": list(nodes.values()), "relationships": list(relationships.values())}
        cypher = """
        CALL () {
          UNWIND $nodes AS row
          CALL apoc.merge.node(row.labels, {id:row.id}, row.props, row.props) YIELD node
          RETURN count(node) AS node_count
        }
        CALL () {
          UNWIND $relationships AS row
          MATCH (source {id:row.source}) MATCH (target {id:row.target})
          CALL apoc.merge.relationship(source,row.relation,{},row.props,target,row.props) YIELD rel
          RETURN count(rel) AS relationship_count
        }
        RETURN node_count,relationship_count
        """

        def write_bundle(tx):
            record = tx.run(cypher, parameters).single()
            return record.data() if record else {"node_count": 0, "relationship_count": 0}

        with self.driver.session(database=self.database) as session:
            summary = session.execute_write(write_bundle)
        return {**counts, **summary}

    def update_document_stage(self, document_id: str, stage: str, **properties: Any) -> None:
        safe = {key: value for key, value in properties.items() if isinstance(key, str)}
        self.write(
            "MATCH (d:MEKG:Document {id:$id}) SET d.status=$stage, d.updated_at=$now, d += $props",
            {"id": document_id, "stage": stage, "now": _now(), "props": safe},
        )

    def store_candidate(self, document: ParsedDocument, kind: str, payload: dict[str, Any], reason: str) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        candidate_id = stable_id("candidate", f"{document.version_id}:{kind}:{serialized}")
        self.write(
            """
            MATCH (v:MEKG:DocumentVersion {id:$version_id})
            MERGE (c:MEKGStaging:ExtractionCandidate {id:$id})
            SET c.kind=$kind, c.payload_json=$payload, c.reason=$reason,
                c.validation_status='raw_extracted', c.created_at=$now
            MERGE (c)-[:EVIDENCED_BY]->(v)
            """,
            {"version_id": document.version_id, "id": candidate_id, "kind": kind, "payload": serialized, "reason": reason, "now": _now()},
        )
        return candidate_id

    def delete_candidate(self, candidate_id: str) -> None:
        self.write("MATCH (candidate:MEKGStaging {id:$id}) DETACH DELETE candidate", {"id": candidate_id})

    def _merge_entity(self, candidate: Any, document: ParsedDocument) -> str:
        label = _safe_label(candidate.entity_type, "TopicTag")
        entity_id = stable_id("entity", f"{label}:{candidate.canonical_name.casefold().strip()}")
        props = {
            "canonical_name": candidate.canonical_name.strip(),
            "name_ru": candidate.name_ru,
            "name_en": candidate.name_en,
            "aliases": sorted(set(candidate.aliases)),
            "description": candidate.description,
            "source": "extracted",
            "confidence": candidate.confidence,
            "validation_status": ValidationStatus.MACHINE_VALIDATED.value,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.write(
            """
            CALL apoc.merge.node(['MEKG','CanonicalEntity',$label], {id:$id}, $props, $props) YIELD node
            WITH node
            MATCH (ev:MEKG {id:$evidence_id})
            MERGE (node)-[:EVIDENCED_BY]->(ev)
            """,
            {"label": label, "id": entity_id, "props": props, "evidence_id": candidate.evidence.element_id},
        )
        for alias in sorted({candidate.canonical_name, *candidate.aliases, candidate.name_ru, candidate.name_en} - {None, ""}):
            term_id = stable_id("term", f"{entity_id}:{alias.casefold()}")
            language = "ru" if re.search(r"[А-Яа-яЁё]", alias) else "en"
            self.write(
                """
                MATCH (entity:MEKG {id:$entity_id})
                MERGE (term:MEKG:Term {id:$term_id})
                SET term.text=$text, term.language=$language, term.term_type=$term_type
                MERGE (entity)-[:HAS_TERM]->(term)
                """,
                {"entity_id": entity_id, "term_id": term_id, "text": alias, "language": language, "term_type": "preferred_label" if alias == candidate.canonical_name else "alias"},
            )
        return entity_id

    def store_extraction(self, document: ParsedDocument, extraction: ChunkExtraction) -> dict[str, int]:
        local_map: dict[str, str] = {}
        counts = {"entities": 0, "conditions": 0, "measurements": 0, "experiments": 0, "claims": 0, "experts": 0}
        for entity in extraction.entities:
            local_map[entity.local_id] = self._merge_entity(entity, document)
            counts["entities"] += 1
        for condition in extraction.conditions:
            normalized = self.units.normalize(condition.unit_original)
            condition_id = stable_id("condition", f"{document.version_id}:{condition.local_id}:{condition.evidence.element_id}")
            parameter_id = stable_id("parameter", condition.name.casefold())
            unit_id = stable_id("unit", normalized.symbol.casefold())
            props = {
                "parameter": condition.name,
                "numeric_value": self.units.convert_value(condition.value, normalized),
                "value_min": self.units.convert_value(condition.value_min, normalized),
                "value_max": self.units.convert_value(condition.value_max, normalized),
                "unit_original": condition.unit_original,
                "unit_normalized": normalized.symbol,
                "dimension": normalized.dimension,
                "comparator": condition.comparator,
                "source_text": condition.evidence.quote,
                "confidence": condition.confidence,
                "approximate": condition.approximate,
                "validation_status": ValidationStatus.MACHINE_VALIDATED.value,
                "created_at": _now(),
            }
            self.write(
                """
                MATCH (ev:MEKG {id:$evidence_id})
                MERGE (c:MEKG:Condition {id:$id}) SET c += $props
                MERGE (p:MEKG:CanonicalEntity:Parameter {id:$parameter_id})
                SET p.canonical_name=$parameter, p.validation_status='machine_validated'
                MERGE (u:MEKG:CanonicalEntity:Unit {id:$unit_id})
                SET u.canonical_name=$unit, u.symbol=$unit, u.dimension=$dimension,
                    u.validation_status='machine_validated'
                MERGE (c)-[:HAS_PARAMETER]->(p)
                MERGE (c)-[:HAS_UNIT]->(u)
                MERGE (c)-[:EVIDENCED_BY]->(ev)
                """,
                {"evidence_id": condition.evidence.element_id, "id": condition_id, "props": props, "parameter_id": parameter_id, "parameter": condition.name, "unit_id": unit_id, "unit": normalized.symbol, "dimension": normalized.dimension},
            )
            local_map[condition.local_id] = condition_id
            counts["conditions"] += 1
        for measurement in extraction.measurements:
            normalized = self.units.normalize(measurement.unit_original)
            measurement_id = stable_id("measurement", f"{document.version_id}:{measurement.local_id}:{measurement.evidence.element_id}")
            property_id = stable_id("property", measurement.property_name.casefold())
            unit_id = stable_id("unit", normalized.symbol.casefold())
            props = {
                "name": measurement.name,
                "numeric_value": self.units.convert_value(
                    measurement.value if measurement.value is not None else (
                        measurement.value_min if measurement.value_min is not None else measurement.value_max
                    ),
                    normalized,
                ),
                "value_min": self.units.convert_value(measurement.value_min, normalized),
                "value_max": self.units.convert_value(measurement.value_max, normalized),
                "unit_original": measurement.unit_original,
                "unit_normalized": normalized.symbol,
                "dimension": normalized.dimension,
                "comparator": measurement.comparator,
                "method": measurement.method,
                "source_text": measurement.evidence.quote,
                "confidence": measurement.confidence,
                "approximate": measurement.approximate,
                "validation_status": ValidationStatus.MACHINE_VALIDATED.value,
                "created_at": _now(),
            }
            self.write(
                """
                MATCH (ev:MEKG {id:$evidence_id})
                MERGE (m:MEKG:Measurement {id:$id}) SET m += $props
                MERGE (p:MEKG:CanonicalEntity:Property {id:$property_id})
                SET p.canonical_name=$property, p.validation_status='machine_validated'
                MERGE (u:MEKG:CanonicalEntity:Unit {id:$unit_id})
                SET u.canonical_name=$unit, u.symbol=$unit, u.dimension=$dimension,
                    u.validation_status='machine_validated'
                MERGE (m)-[:MEASURES_PROPERTY]->(p)
                MERGE (m)-[:HAS_UNIT]->(u)
                MERGE (m)-[:EVIDENCED_BY]->(ev)
                """,
                {"evidence_id": measurement.evidence.element_id, "id": measurement_id, "props": props, "property_id": property_id, "property": measurement.property_name, "unit_id": unit_id, "unit": normalized.symbol, "dimension": normalized.dimension},
            )
            local_map[measurement.local_id] = measurement_id
            counts["measurements"] += 1
        for experiment in extraction.experiments:
            label = _safe_label(experiment.experiment_type, "Experiment")
            experiment_id = stable_id("experiment", f"{document.version_id}:{experiment.local_id}:{experiment.evidence.element_id}")
            self.write(
                """
                MATCH (ev:MEKG {id:$evidence_id})
                CALL apoc.merge.node(['MEKG','Experiment',$label], {id:$id}, $props, $props) YIELD node
                MERGE (node)-[:EVIDENCED_BY]->(ev)
                """,
                {"label": label, "evidence_id": experiment.evidence.element_id, "id": experiment_id, "props": {"name": experiment.name, "effect": experiment.effect, "source_text": experiment.evidence.quote, "confidence": experiment.confidence, "validation_status": ValidationStatus.MACHINE_VALIDATED.value, "created_at": _now()}},
            )
            refs = [
                ("USES_MATERIAL", experiment.material_refs),
                ("STUDIES_PROCESS", experiment.process_refs),
                ("RUN_ON", experiment.equipment_refs),
                ("HAS_CONDITION", experiment.condition_refs),
                ("PRODUCED_MEASUREMENT", experiment.measurement_refs),
            ]
            for relation, local_refs in refs:
                for local_ref in local_refs:
                    target = local_map.get(local_ref)
                    if target:
                        self.merge_asserted_relationship(experiment_id, relation, target, experiment.evidence.element_id, experiment.confidence)
            local_map[experiment.local_id] = experiment_id
            counts["experiments"] += 1
        for claim in extraction.claims:
            label = _safe_label(claim.claim_type, "Claim")
            claim_id = stable_id("claim", f"{document.version_id}:{claim.local_id}:{claim.text.casefold()}")
            pack_id = stable_id("evidence", claim_id)
            version_id = stable_id("claimver", f"{claim_id}:1:{claim.text}")
            self.write(
                """
                MATCH (ev:MEKG {id:$evidence_id})
                CALL apoc.merge.node(['MEKG','Claim',$label], {id:$claim_id}, $props, $props) YIELD node AS claim
                MERGE (pack:MEKG:EvidencePack {id:$pack_id})
                SET pack.supporting_sources_count=1, pack.contradicting_sources_count=0,
                    pack.evidence_quality=CASE WHEN $confidence >= 0.85 THEN 'high' WHEN $confidence >= 0.65 THEN 'medium' ELSE 'low' END,
                    pack.confidence=$confidence, pack.last_updated=$now, pack.validation_status='machine_validated'
                MERGE (version:MEKG:ClaimVersion {id:$version_id})
                SET version.version=1, version.text=$text, version.created_at=$now, version.status='active'
                MERGE (claim)-[:SUPPORTED_BY]->(pack)
                MERGE (claim)-[:HAS_VERSION]->(version)
                MERGE (version)-[:EVIDENCED_BY]->(pack)
                MERGE (pack)-[:HAS_EVIDENCE]->(ev)
                MERGE (claim)-[:EVIDENCED_BY]->(ev)
                """,
                {"label": label, "evidence_id": claim.evidence.element_id, "claim_id": claim_id, "pack_id": pack_id, "version_id": version_id, "text": claim.text, "confidence": claim.confidence, "now": _now(), "props": {"text": claim.text, "source_text": claim.evidence.quote, "geo_scope": claim.geo_scope, "countries": claim.countries, "recommendation": claim.recommendation, "limitations": claim.limitations, "confidence": claim.confidence, "validation_status": ValidationStatus.MACHINE_VALIDATED.value, "created_at": _now(), "updated_at": _now()}},
            )
            for ref in [*claim.entity_refs, *claim.experiment_refs, *claim.measurement_refs]:
                target = local_map.get(ref)
                if target:
                    self.merge_asserted_relationship(claim_id, "GENERALIZES", target, claim.evidence.element_id, claim.confidence)
            local_map[claim.local_id] = claim_id
            counts["claims"] += 1
        for expert in extraction.experts:
            expert_id = stable_id("expert", expert.name.casefold().strip())
            self.write(
                """
                MATCH (ev:MEKG {id:$evidence_id})
                MERGE (e:MEKG:CanonicalEntity:Expert:Author {id:$id})
                SET e.canonical_name=$name, e.organization=$organization, e.lab=$lab,
                    e.topics=$topics, e.confidence=$confidence,
                    e.validation_status='machine_validated', e.updated_at=$now
                MERGE (e)-[:EVIDENCED_BY]->(ev)
                """,
                {"evidence_id": expert.evidence.element_id, "id": expert_id, "name": expert.name, "organization": expert.organization, "lab": expert.lab, "topics": expert.topics, "confidence": expert.confidence, "now": _now()},
            )
            local_map[expert.local_id] = expert_id
            counts["experts"] += 1
        return counts

    def merge_asserted_relationship(self, source_id: str, relation: str, target_id: str, evidence_id: str, confidence: float) -> str:
        if relation not in ALLOWED_RELATIONSHIPS:
            raise ValueError(f"Relationship is not allowed by MEKG ontology: {relation}")
        assertion_id = stable_id("assertion", f"{source_id}:{relation}:{target_id}:{evidence_id}")
        props = {"validation_status": ValidationStatus.MACHINE_VALIDATED.value, "confidence": confidence, "evidence_id": evidence_id}
        self.write(
            """
            MATCH (source:MEKG {id:$source_id})
            MATCH (target:MEKG {id:$target_id})
            MATCH (ev:MEKG {id:$evidence_id})
            CALL apoc.merge.relationship(source, $relation, {}, $props, target, $props) YIELD rel
            MERGE (a:MEKG:RelationshipAssertion {id:$assertion_id})
            SET a.relation_type=$relation, a.confidence=$confidence,
                a.validation_status='machine_validated', a.created_at=$now
            MERGE (a)-[:ASSERTS_SOURCE]->(source)
            MERGE (a)-[:ASSERTS_TARGET]->(target)
            MERGE (a)-[:EVIDENCED_BY]->(ev)
            RETURN a.id AS id
            """,
            {"source_id": source_id, "target_id": target_id, "evidence_id": evidence_id, "relation": relation, "props": props, "assertion_id": assertion_id, "confidence": confidence, "now": _now()},
        )
        return assertion_id

    def review(self, fact_id: str, decision: ReviewDecision) -> dict[str, Any]:
        validation_id = stable_id("validation", f"{fact_id}:{decision.reviewer}:{_now()}:{decision.status}")
        result = self.write(
            """
            MATCH (fact {id:$fact_id}) WHERE fact:MEKG OR fact:MEKGStaging
            SET fact.validation_status=$status, fact.updated_at=$now
            MERGE (reviewer:MEKG:CanonicalEntity:Expert {id:$reviewer_id})
            SET reviewer.canonical_name=$reviewer
            CREATE (record:MEKG:ValidationRecord {id:$validation_id, status:$status, comment:$comment, validated_at:$now})
            MERGE (record)-[:VALIDATED_BY]->(reviewer)
            MERGE (fact)-[:VALIDATED_BY]->(record)
            RETURN fact.id AS id, fact.validation_status AS status, record.id AS validation_id
            """,
            {"fact_id": fact_id, "status": decision.status, "now": _now(), "reviewer": decision.reviewer, "reviewer_id": stable_id("expert", decision.reviewer.casefold()), "validation_id": validation_id, "comment": decision.comment},
        )
        if not result:
            raise KeyError(fact_id)
        if decision.supersedes_id:
            self.merge_asserted_relationship(fact_id, "SUPERSEDES", decision.supersedes_id, validation_id, 1.0)
        return result[0]

    def graph_snapshot(self, *, verified_only: bool = True) -> dict[str, list[dict[str, Any]]]:
        status_filter = (
            "AND (n.validation_status IN ['machine_validated','expert_validated','conflicting'] "
            "OR any(label IN labels(n) WHERE label IN "
            "['Document','DocumentVersion','Publication','Page','Slide','Chunk','Table','TableRow','Figure','Formula']))"
            if verified_only
            else ""
        )
        nodes = self.query(
            f"MATCH (n:MEKG) WHERE NOT n:Session {status_filter} RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties"
        )
        rels = self.query(
            "MATCH (a:MEKG)-[r]->(b:MEKG) "
            + ("WHERE coalesce(r.validation_status,'machine_validated') IN ['machine_validated','expert_validated','conflicting'] " if verified_only else "")
            + "RETURN a.id AS source, type(r) AS type, b.id AS target, properties(r) AS properties"
        )
        return {"nodes": nodes, "relationships": rels}
