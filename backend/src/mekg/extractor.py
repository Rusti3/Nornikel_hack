from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rapidfuzz.fuzz import partial_ratio

from .config import MEKGConfig
from .models import (
    BatchExtraction,
    ChunkExtraction,
    ElementKind,
    ParsedDocument,
    PublicationBoundary,
    PublicationSplit,
    SourceElement,
)
from .schema import ENTITY_LABELS
from .units import UnitNormalizer


SYSTEM_PROMPT = """You extract evidence-bound facts for a Metallurgical Evidence Knowledge Graph.
Return only facts explicitly present in the supplied source element. Never infer missing numbers, units,
authors, geography, causality, or validation. Every item must include evidence.element_id exactly equal to
the supplied ELEMENT_ID and evidence.quote copied verbatim from SOURCE_TEXT. Use stable local ids within the
response so experiments and claims can reference entities, conditions and measurements.

Use entity types from this controlled set: {entity_types}.
Conditions and measurements must be numeric and must include the original unit. A Claim is a statement made
by the source, not your summary. Keep Russian and English names and aliases when explicitly present.
Set geo_scope to DomesticPractice only for Russian practice, ForeignPractice for a specific non-Russian
practice, GlobalPractice only when the source explicitly makes a world/global statement, otherwise
UnknownPractice. Confidence reflects extraction certainty, not scientific truth.
"""


class MEKGExtractor:
    def __init__(self, config: MEKGConfig | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self.config.validate_yandex()
        self.llm = ChatOpenAI(
            api_key=self.config.yandex_api_key,
            base_url=self.config.yandex_base_url,
            model=self.config.llm_model,
            temperature=0,
            timeout=120,
            max_retries=2,
            default_headers={"OpenAI-Project": self.config.yandex_folder_id},
        )
        self.units = UnitNormalizer()
        self.system_prompt = SYSTEM_PROMPT.format(entity_types=", ".join(sorted(ENTITY_LABELS)))

    async def extract_element(self, element: SourceElement) -> tuple[ChunkExtraction, list[dict[str, Any]]]:
        if not element.text or len(element.text.strip()) < 40:
            return ChunkExtraction(), []
        user = f"ELEMENT_ID: {element.id}\nSOURCE_TEXT:\n{element.text}"
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user)]
        try:
            runnable = self.llm.with_structured_output(ChunkExtraction, method="json_schema")
            extraction = await runnable.ainvoke(messages)
            if not isinstance(extraction, ChunkExtraction):
                extraction = ChunkExtraction.model_validate(extraction)
        except Exception as structured_error:
            logging.warning(
                "MEKG structured extraction failed for element %s (%s); retrying JSON mode",
                element.id,
                type(structured_error).__name__,
            )
            schema = json.dumps(ChunkExtraction.model_json_schema(), ensure_ascii=False)
            fallback_messages = [
                SystemMessage(content=self.system_prompt + "\nReturn JSON matching this schema:\n" + schema),
                HumanMessage(content=user),
            ]
            response = await self.llm.ainvoke(fallback_messages)
            raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
            raw = self._extract_json(raw)
            extraction = ChunkExtraction.model_validate_json(raw)
        return self.validate(extraction, element)

    async def extract_batch(
        self, elements: list[SourceElement]
    ) -> tuple[dict[str, ChunkExtraction], list[dict[str, Any]]]:
        """Extract up to three evidence elements in one Alice request.

        Yandex structured output is attempted first, then exactly one JSON-text
        fallback. Elements absent from an otherwise valid batch are retried once
        through the proven single-element path.
        """
        elements = [item for item in elements if item.text.strip()][:3]
        if not elements:
            return {}, []
        sources = "\n\n".join(
            f"ELEMENT_ID: {item.id}\nSOURCE_TEXT:\n{item.text[:4000]}" for item in elements
        )
        prompt = (
            "Extract every source independently. Return one items entry per ELEMENT_ID; "
            "never move evidence between elements.\n\n" + sources
        )
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        batch: BatchExtraction
        try:
            runnable = self.llm.with_structured_output(BatchExtraction, method="json_schema")
            batch = await runnable.ainvoke(messages)
            if not isinstance(batch, BatchExtraction):
                batch = BatchExtraction.model_validate(batch)
        except Exception as structured_error:
            logging.warning(
                "MEKG structured batch extraction failed (%s); retrying JSON mode",
                type(structured_error).__name__,
            )
            schema = json.dumps(BatchExtraction.model_json_schema(), ensure_ascii=False)
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=self.system_prompt + "\nReturn JSON matching this schema:\n" + schema),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
            batch = BatchExtraction.model_validate_json(self._extract_json(raw))

        by_id = {element.id: element for element in elements}
        valid: dict[str, ChunkExtraction] = {}
        rejected: list[dict[str, Any]] = []
        for item in batch.items:
            element = by_id.get(item.element_id)
            if element is None or item.element_id in valid:
                rejected.append(
                    {
                        "kind": "batch_element",
                        "payload": item.model_dump(),
                        "reason": "unknown or duplicate element_id",
                    }
                )
                continue
            extraction, item_rejected = self.validate(item.extraction, element)
            valid[item.element_id] = extraction
            rejected.extend(item_rejected)

        missing = [element for element in elements if element.id not in valid]
        if missing:
            retries = await asyncio.gather(*(self.extract_element(element) for element in missing), return_exceptions=True)
            for element, result in zip(missing, retries):
                if isinstance(result, Exception):
                    rejected.append(
                        {
                            "kind": "extraction_error",
                            "payload": {"element_id": element.id},
                            "reason": f"{type(result).__name__}: {str(result)[:500]}",
                        }
                    )
                else:
                    valid[element.id], item_rejected = result
                    rejected.extend(item_rejected)
        return valid, rejected

    def validate(self, extraction: ChunkExtraction, element: SourceElement) -> tuple[ChunkExtraction, list[dict[str, Any]]]:
        valid = ChunkExtraction()
        rejected: list[dict[str, Any]] = []
        source = self._normalize(element.text)

        def evidence_ok(item: Any) -> bool:
            if item.evidence.element_id != element.id:
                return False
            quote = self._normalize(item.evidence.quote)
            if not quote:
                return False
            if quote in source or partial_ratio(quote, source) >= 95:
                return True

            # Text layers from presentation tables often serialize headers,
            # columns, and values on separate lines. Accept a reconstructed
            # citation only when its meaningful tokens (including every
            # number) are present in the same anchored source element.
            tokens = re.findall(r"[^\W_]+(?:[.,]\d+)?", quote, flags=re.UNICODE)
            source_tokens = set(re.findall(r"[^\W_]+(?:[.,]\d+)?", source, flags=re.UNICODE))
            numbers = re.findall(r"\d+(?:[.,]\d+)?", quote)
            source_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", source))
            coverage = sum(token in source_tokens for token in tokens) / max(len(tokens), 1)
            if len(tokens) >= 4 and coverage >= 0.85 and all(number in source_numbers for number in numbers):
                item.evidence.confidence = min(item.evidence.confidence, 0.7)
                item.confidence = min(item.confidence, 0.7)
                return True
            return False

        for item in extraction.entities:
            if evidence_ok(item) and item.entity_type in ENTITY_LABELS and item.canonical_name.strip():
                valid.entities.append(item)
            else:
                rejected.append({"kind": "entity", "payload": item.model_dump(), "reason": "invalid evidence, type, or name"})
        for field in ("conditions", "measurements"):
            for item in getattr(extraction, field):
                unit = self.units.normalize(item.unit_original)
                numeric_present = any(
                    value is not None for value in (item.value, item.value_min, item.value_max)
                )
                if evidence_ok(item) and unit.valid and numeric_present:
                    getattr(valid, field).append(item)
                else:
                    reason = unit.error or ("non-numeric value" if not numeric_present else "invalid evidence")
                    rejected.append({"kind": field[:-1], "payload": item.model_dump(), "reason": reason})
        known_ids = {item.local_id for item in [*valid.entities, *valid.conditions, *valid.measurements]}
        for item in extraction.experiments:
            if evidence_ok(item):
                item.material_refs = [ref for ref in item.material_refs if ref in known_ids]
                item.process_refs = [ref for ref in item.process_refs if ref in known_ids]
                item.equipment_refs = [ref for ref in item.equipment_refs if ref in known_ids]
                item.condition_refs = [ref for ref in item.condition_refs if ref in known_ids]
                item.measurement_refs = [ref for ref in item.measurement_refs if ref in known_ids]
                valid.experiments.append(item)
                known_ids.add(item.local_id)
            else:
                rejected.append({"kind": "experiment", "payload": item.model_dump(), "reason": "invalid evidence"})
        for item in extraction.claims:
            if evidence_ok(item):
                item.entity_refs = [ref for ref in item.entity_refs if ref in known_ids]
                item.experiment_refs = [ref for ref in item.experiment_refs if ref in known_ids]
                item.measurement_refs = [ref for ref in item.measurement_refs if ref in known_ids]
                valid.claims.append(item)
            else:
                rejected.append({"kind": "claim", "payload": item.model_dump(), "reason": "invalid evidence"})
        for item in extraction.experts:
            if evidence_ok(item) and item.name.strip():
                valid.experts.append(item)
            else:
                rejected.append({"kind": "expert", "payload": item.model_dump(), "reason": "invalid evidence"})
        return valid, rejected

    async def split_publications(self, document: ParsedDocument) -> list[PublicationBoundary]:
        if not any(item.needs_review for item in document.publications):
            return document.publications
        page_text: dict[int, str] = {}
        for element in document.elements:
            if element.kind == ElementKind.TEXT and element.page_number and element.page_number <= 15:
                page_text.setdefault(element.page_number, "")
                page_text[element.page_number] += "\n" + element.text[:5000]
        max_page = max((element.page_number or 0 for element in document.elements), default=1)
        overview = "\n\n".join(f"PAGE {number}\n{text[:5000]}" for number, text in sorted(page_text.items()))
        prompt = (
            f"This file has {max_page} pages and contains multiple scientific publications. "
            "Use the table of contents and headings to return non-overlapping publication boundaries. "
            "Do not invent authors. Set needs_review=true when a boundary is uncertain.\n" + overview[:50000]
        )
        try:
            runnable = self.llm.with_structured_output(PublicationSplit, method="json_schema")
            result = await runnable.ainvoke([HumanMessage(content=prompt)])
            if not isinstance(result, PublicationSplit):
                result = PublicationSplit.model_validate(result)
            boundaries = sorted(result.publications, key=lambda item: item.start_page)
            valid: list[PublicationBoundary] = []
            previous_end = 0
            for boundary in boundaries:
                if boundary.start_page < 1 or boundary.end_page > max_page or boundary.start_page > boundary.end_page:
                    continue
                if boundary.start_page <= previous_end:
                    boundary.needs_review = True
                    continue
                valid.append(boundary)
                previous_end = boundary.end_page
            return valid or document.publications
        except Exception as exc:
            logging.warning("Publication splitting failed for %s: %s", document.file_name, exc)
            return document.publications

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    @staticmethod
    def _extract_json(value: str) -> str:
        value = value.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM did not return a JSON object")
        return value[start : end + 1]
