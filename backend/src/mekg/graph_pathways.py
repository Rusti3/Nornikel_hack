from __future__ import annotations

import itertools
import re
from typing import Any

from .models import GraphPathwaysRequest
from .repository import MEKGRepository


MATERIAL_LABELS = [
    "Material", "Substance", "ChemicalCompound", "Element", "Ion", "Solution",
    "Ore", "Concentrate", "Waste", "GasComponent", "Phase",
]
PROCESS_LABELS = [
    "Process", "ProcessStep", "Technology", "Method", "Regime", "UnitOperation",
    "HydrometallurgicalProcess", "PyrometallurgicalProcess", "EnvironmentalProcess",
    "WasteProcessingProcess", "ElectrowinningProcess", "LeachingProcess",
    "SmeltingProcess", "DesalinationProcess", "GasCleaningProcess",
    "MineWaterInjectionProcess",
]
EQUIPMENT_LABELS = [
    "Equipment", "Facility", "Cell", "Furnace", "Reactor", "Filter", "Pump",
    "Tank", "Pipeline", "Electrolyzer", "DiaphragmCell", "Scrubber",
]


def _tokens(value: str | None) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[\wА-Яа-яЁё-]{3,}", (value or "").casefold())))[:16]


def _present(value: dict[str, Any] | None) -> bool:
    return bool(value and value.get("id"))


def _node_title(value: dict[str, Any]) -> str:
    return str(
        value.get("name")
        or value.get("canonical_name")
        or value.get("property_name")
        or value.get("text")
        or value.get("id")
        or "—"
    )


def _result_title(value: dict[str, Any]) -> str:
    title = _node_title(value)
    numeric = value.get("numeric_value")
    if numeric is None:
        numeric = value.get("value_min")
    unit = value.get("unit_normalized") or value.get("unit_original")
    if numeric is not None:
        title = f"{title}: {numeric:g}" if isinstance(numeric, (int, float)) else f"{title}: {numeric}"
        if unit:
            title += f" {unit}"
    return title


def build_pathways(repository: MEKGRepository, request: GraphPathwaysRequest) -> dict[str, Any]:
    rows = repository.query(
        """
        MATCH (experiment:MEKG:Experiment)
        OPTIONAL MATCH (experiment)-[:USES_MATERIAL]->(material:MEKG)
        WITH experiment,
             [item IN collect(DISTINCT {
                id:material.id,labels:labels(material),name:coalesce(material.canonical_name,material.name),
                confidence:material.confidence,validation_status:material.validation_status
             }) WHERE item.id IS NOT NULL] AS materials
        OPTIONAL MATCH (experiment)-[:STUDIES_PROCESS]->(process:MEKG)
        WITH experiment,materials,
             [item IN collect(DISTINCT {
                id:process.id,labels:labels(process),name:coalesce(process.canonical_name,process.name),
                confidence:process.confidence,validation_status:process.validation_status
             }) WHERE item.id IS NOT NULL] AS processes
        OPTIONAL MATCH (experiment)-[equipment_rel:USES_EQUIPMENT|RUN_ON]->(equipment:MEKG)
        WITH experiment,materials,processes,
             [item IN collect(DISTINCT {
                id:equipment.id,labels:labels(equipment),name:coalesce(equipment.canonical_name,equipment.name),
                confidence:equipment.confidence,validation_status:equipment.validation_status,
                relationship:type(equipment_rel)
             }) WHERE item.id IS NOT NULL] AS equipment
        OPTIONAL MATCH (experiment)-[:PRODUCED_MEASUREMENT]->(measurement:MEKG:Measurement)
        WITH experiment,materials,processes,equipment,
             [item IN collect(DISTINCT {
                id:measurement.id,labels:labels(measurement),name:coalesce(measurement.name,measurement.property_name),
                property_name:measurement.property_name,numeric_value:measurement.numeric_value,
                value_min:measurement.value_min,value_max:measurement.value_max,
                unit_original:measurement.unit_original,unit_normalized:measurement.unit_normalized,
                confidence:measurement.confidence,validation_status:measurement.validation_status,
                source_text:measurement.source_text
             }) WHERE item.id IS NOT NULL] AS measurements
        OPTIONAL MATCH (claim:MEKG:Claim)-[:GENERALIZES]->(experiment)
        WITH experiment,materials,processes,equipment,measurements,
             [item IN collect(DISTINCT {
                id:claim.id,labels:labels(claim),text:claim.text,name:claim.text,
                confidence:claim.confidence,validation_status:claim.validation_status,
                source_text:claim.source_text
             }) WHERE item.id IS NOT NULL] AS claims
        OPTIONAL MATCH (experiment)-[:EVIDENCED_BY]->(evidence:MEKG)
        CALL (evidence) {
          OPTIONAL MATCH (document:MEKG:Document)-[:HAS_VERSION]->(version:MEKG:DocumentVersion)
          WHERE evidence IS NOT NULL AND EXISTS {
            MATCH (version)-[:HAS_PAGE|HAS_CHUNK|HAS_TABLE|HAS_ROW|HAS_FIGURE|HAS_FORMULA*1..3]->(evidence)
          }
          RETURN [item IN collect(DISTINCT {
            id:document.id,file_name:document.fileName,category:document.category,
            source_locator:document.source_locator
          }) WHERE item.id IS NOT NULL] AS documents
        }
        WITH experiment,materials,processes,equipment,measurements,claims,evidence,documents,
             toLower(coalesce(experiment.name,'')+' '+coalesce(experiment.effect,'')+' '+
                     coalesce(experiment.source_text,'')+' '+coalesce(evidence.text,'')) AS searchable
        WHERE (size($document_ids)=0 OR any(document IN documents WHERE document.id IN $document_ids))
          AND (size($entity_ids)=0 OR experiment.id IN $entity_ids
               OR any(item IN materials WHERE item.id IN $entity_ids)
               OR any(item IN processes WHERE item.id IN $entity_ids)
               OR any(item IN equipment WHERE item.id IN $entity_ids)
               OR any(item IN measurements WHERE item.id IN $entity_ids)
               OR any(item IN claims WHERE item.id IN $entity_ids))
          AND (size($tokens)=0 OR any(token IN $tokens WHERE searchable CONTAINS token
               OR any(item IN materials WHERE toLower(coalesce(item.name,'')) CONTAINS token)
               OR any(item IN processes WHERE toLower(coalesce(item.name,'')) CONTAINS token)
               OR any(item IN equipment WHERE toLower(coalesce(item.name,'')) CONTAINS token)
               OR any(item IN measurements WHERE toLower(coalesce(item.name,'')) CONTAINS token)
               OR any(item IN claims WHERE toLower(coalesce(item.text,'')) CONTAINS token)))
        RETURN {
          id:experiment.id,labels:labels(experiment),name:experiment.name,effect:experiment.effect,
          confidence:experiment.confidence,validation_status:experiment.validation_status,
          source_text:experiment.source_text
        } AS experiment,
        materials,processes,equipment,measurements,claims,
        CASE WHEN evidence IS NULL THEN null ELSE {
          id:evidence.id,page:evidence.page_number,slide:evidence.slide_number,
          sheet:evidence.sheet_name,row:evidence.row_number,text:evidence.text
        } END AS evidence,
        documents
        ORDER BY coalesce(experiment.confidence,0) DESC,experiment.id
        LIMIT $row_limit
        """,
        {
            "tokens": _tokens(request.query),
            "document_ids": list(dict.fromkeys(request.document_ids))[:100],
            "entity_ids": list(dict.fromkeys(request.entity_ids))[:100],
            "row_limit": min(500, max(request.limit * 4, request.limit)),
        },
    )

    pathways: list[dict[str, Any]] = []
    graph_nodes: dict[str, dict[str, Any]] = {}
    graph_relationships: dict[str, dict[str, Any]] = {}
    seen_paths: set[tuple[Any, ...]] = set()

    def add_node(value: dict[str, Any], role: str, title: str | None = None) -> None:
        if not _present(value):
            return
        graph_nodes[value["id"]] = {
            "id": value["id"],
            "labels": value.get("labels") or [role],
            "role": role,
            "title": title or _node_title(value),
            "confidence": value.get("confidence"),
            "validation_status": value.get("validation_status"),
            "properties": value,
        }

    def add_relationship(source: str, relation: str, target: str) -> None:
        identifier = f"{source}:{relation}:{target}"
        graph_relationships[identifier] = {
            "id": identifier,
            "from": source,
            "to": target,
            "type": relation,
        }

    for row in rows:
        experiment = row.get("experiment") or {}
        materials = row.get("materials") or [None]
        processes = row.get("processes") or [None]
        equipment = row.get("equipment") or [None]
        results = row.get("measurements") or row.get("claims") or [None]
        evidence = row.get("evidence") or {}
        documents = row.get("documents") or []
        if not request.include_incomplete and not all((row.get("materials"), row.get("processes"), row.get("equipment"), results and results != [None])):
            continue
        combinations = itertools.product(materials, processes, equipment, results)
        for material, process, machine, result in combinations:
            path_key = (
                experiment.get("id"),
                (material or {}).get("id"),
                (process or {}).get("id"),
                (machine or {}).get("id"),
                (result or {}).get("id"),
                ((documents[0] if documents else {}) or {}).get("id"),
                evidence.get("id"),
            )
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            missing = [
                role for role, value in (
                    ("material", material), ("process", process),
                    ("equipment", machine), ("result", result),
                ) if not _present(value)
            ]
            confidence_values = [
                float(value.get("confidence"))
                for value in (experiment, material, process, machine, result)
                if _present(value) and value.get("confidence") is not None
            ]
            confidence = min(confidence_values) if confidence_values else 0.5
            statuses = {
                str(value.get("validation_status") or "")
                for value in (experiment, material, process, machine, result)
                if _present(value)
            }
            status = "conflicting" if "conflicting" in statuses else (
                "low_confidence" if confidence < 0.65 else "verified"
            )
            pathway = {
                "id": f"path:{experiment.get('id')}:{len(pathways)}",
                "experiment": experiment,
                "material": material,
                "process": process,
                "equipment": machine,
                "result": result,
                "result_title": _result_title(result) if _present(result) else None,
                "missing_stages": missing,
                "complete": not missing,
                "confidence": round(confidence, 3),
                "status": status,
                "evidence": {
                    **evidence,
                    "document": documents[0] if documents else None,
                },
            }
            pathways.append(pathway)

            add_node(experiment, "experiment")
            for value, role, relation in (
                (material, "material", "USES_MATERIAL"),
                (process, "process", "STUDIES_PROCESS"),
                (machine, "equipment", (machine or {}).get("relationship") or "USES_EQUIPMENT"),
                (result, "result", "PRODUCED_MEASUREMENT" if "Measurement" in ((result or {}).get("labels") or []) else "GENERALIZES"),
            ):
                if _present(value):
                    add_node(value, role, _result_title(value) if role == "result" else None)
                    if relation == "GENERALIZES":
                        add_relationship(value["id"], relation, experiment["id"])
                    else:
                        add_relationship(experiment["id"], relation, value["id"])
            if len(pathways) >= request.limit:
                break
        if len(pathways) >= request.limit:
            break

    complete = sum(bool(item["complete"]) for item in pathways)
    return {
        "pathways": pathways,
        "nodes": list(graph_nodes.values()),
        "relationships": list(graph_relationships.values()),
        "coverage": {
            "total": len(pathways),
            "complete": complete,
            "incomplete": len(pathways) - complete,
        },
        "query": request.query,
    }
