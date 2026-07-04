from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz.fuzz import WRatio
from PIL import Image, ImageDraw

from .analytics import MEKGAnalytics
from .config import MEKGConfig
from .extractor import MEKGExtractor
from .models import (
    ChunkExtraction,
    CrossCorpusSearchRequest,
    ElementKind,
    FindContradictionsRequest,
    FindExpertsRequest,
    FindExperimentsRequest,
    FindKnowledgeGapsRequest,
    FindTechnologiesRequest,
    GetEvidencePackRequest,
    ResolveEntityRequest,
    ReviewDecision,
    SemanticSearchRequest,
    ToolResponse,
)
from .parsers import DocumentParser, stable_id, validate_manifest
from .qa import MEKGQualityAuditor
from .rdf import RDFExporter
from .repository import MEKGRepository
from .retrieval import CrossCorpusRetriever
from .units import UnitNormalizer
from .vision import YandexVisionClient
from src.yandex_embeddings import YandexEmbeddings
from .models import SourceElement


class MEKGService:
    def __init__(self, config: MEKGConfig | None = None, repository: MEKGRepository | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self.repository = repository or MEKGRepository()
        self.vision = YandexVisionClient(self.config)
        self.parser = DocumentParser(self.config, self.vision)
        self._extractor: MEKGExtractor | None = None
        self._retriever: CrossCorpusRetriever | None = None
        self.exporter = RDFExporter(self.repository, self.config.ontology_dir)
        self.auditor = MEKGQualityAuditor(self.repository, self.exporter)

    @property
    def extractor(self) -> MEKGExtractor:
        if self._extractor is None:
            self._extractor = MEKGExtractor(self.config)
        return self._extractor

    @property
    def retriever(self) -> CrossCorpusRetriever:
        if self._retriever is None:
            self._retriever = CrossCorpusRetriever(self.config)
        return self._retriever

    def initialize(self) -> dict[str, Any]:
        self.repository.initialize_schema()
        ontology = self.repository.load_ontology_projection(str(self.config.ontology_dir))
        return {"ontology": ontology, "vision": self.vision.preflight()}

    @staticmethod
    def _graph_query_tokens(query: str) -> list[str]:
        stopwords = {
            "какие", "какой", "какова", "как", "для", "при", "или", "это", "чем", "что",
            "между", "заявлены", "схема", "процесс", "процесса", "который", "которые",
            "what", "which", "with", "from", "that", "this", "process", "using", "and", "the",
        }
        tokens = [
            token.casefold()[:6]
            if re.fullmatch(r"[А-Яа-яЁё]{8,}", token) else token.casefold()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9+.-]{1,}", query)
            if token.casefold() not in stopwords
        ]
        # Exact alphanumeric terms and numbers are usually the best anchors in technical corpora.
        tokens.sort(key=lambda item: (not any(char.isdigit() for char in item), -len(item)))
        folded = query.casefold().replace("ё", "е")
        bilingual = {
            "никел": ("nickel",), "кобальт": ("cobalt",), "марган": ("manganese",),
            "медн": ("copper",), "золот": ("gold",), "сереб": ("silver",),
            "извлеч": ("recovery", "extraction"), "выщелач": ("leaching",),
            "обогащ": ("beneficiation", "concentration"), "шла": ("slag",),
            "технолог": ("technology",), "мощност": ("capacity",), "энерг": ("energy",),
            "солнеч": ("solar",), "ветр": ("wind",), "гидро": ("hydro",),
            "геотерм": ("geothermal",), "улавлив": ("capture",),
            "хранен": ("storage",), "транспорт": ("transport",),
            "лист": ("sheet", "plate"), "полос": ("strip",), "фольг": ("foil",),
            "производ": ("production", "output"), "квартал": ("quarter",),
            "латун": ("brass",), "прут": ("rod",), "профил": ("section",),
            "регион": ("region", "regional"), "сортност": ("grade",),
            "руд": ("ore",), "складир": ("stockpile",),
            "температур": ("temperature",), "давлен": ("pressure",),
            "допир": ("doping", "doped"), "катод": ("cathode",),
            "прекурсор": ("precursor",), "емкост": ("capacity",), "цикл": ("cycle",),
        }
        expanded = list(tokens)
        for stem, values in bilingual.items():
            if stem in folded:
                expanded.extend(values)
        return list(dict.fromkeys(expanded))[:28]

    def search_source_evidence(
        self,
        query: str,
        *,
        corpora: list[str] | None = None,
        limit: int = 24,
    ) -> ToolResponse:
        """Lexical fallback over Neo4j source elements, including docs absent from Postgres chunks."""
        tokens = self._graph_query_tokens(query)
        if not tokens:
            return ToolResponse(data={"results": []}, evidence=[], confidence=0, warnings=["no graph query tokens"])
        category_map = {
            "internal_reports": "Доклады",
            "scientific_journals": "Журналы",
            "conference_materials": "Материалы конференций",
            "reviews": "Обзоры",
            "scientific_articles": "Статьи",
        }
        categories = [category_map[item] for item in (corpora or []) if item in category_map]
        rows = self.repository.query(
            """
            MATCH (doc:MEKG:Document)-[:HAS_VERSION]->(version:DocumentVersion)
            WHERE size($categories)=0 OR doc.category IN $categories
            MATCH (version)-[*1..3]->(ev:MEKG)
            WHERE (ev:Chunk OR ev:TableRow OR ev:Table) AND ev.text IS NOT NULL
            WITH doc,ev,[token IN $tokens WHERE toLower(ev.text) CONTAINS token] AS matched
            WHERE size(matched) >= CASE WHEN size($tokens) <= 2 THEN 1 ELSE 2 END
            OPTIONAL MATCH (fact:MEKG)-[:EVIDENCED_BY]->(ev)
            WITH doc,ev,matched,count(DISTINCT fact) AS fact_count
            RETURN doc.id AS document_id,doc.fileName AS file_name,doc.category AS category,
                   ev.id AS element_id,ev.text AS text,ev.page_number AS page,
                   ev.slide_number AS slide,matched,fact_count,
                   toFloat(size(matched))/toFloat(size($tokens)) AS lexical_score
            ORDER BY size(matched) DESC,fact_count DESC,lexical_score DESC
            LIMIT $limit
            """,
            {"tokens": tokens, "categories": categories, "limit": max(1, min(100, limit))},
        )
        readable = [item for item in rows if CrossCorpusRetriever._is_readable(str(item.get("text") or ""))]
        dropped = len(rows) - len(readable)
        evidence = [
            {
                "document_id": item["document_id"],
                "file_name": item.get("file_name"),
                "element_id": item["element_id"],
                "quote": CrossCorpusRetriever._clean_text(str(item.get("text") or "")),
                "page": item.get("page"),
                "slide": item.get("slide"),
                "matched_tokens": item.get("matched") or [],
                "confidence": min(1.0, 0.45 + float(item.get("lexical_score") or 0) * 0.4),
                "source_type": "graph_source",
            }
            for item in readable
        ]
        return ToolResponse(
            data={"results": evidence, "query_tokens": tokens},
            evidence=evidence,
            confidence=max((float(item.get("confidence") or 0) for item in evidence), default=0),
            warnings=[f"dropped {dropped} unreadable Neo4j source elements"] if dropped else [],
        )

    async def preflight(self, *, remote: bool = True) -> dict[str, Any]:
        self.config.validate_yandex(vision=True)
        self.initialize()
        neo4j = self.repository.query("RETURN 1 AS ok")[0]["ok"] == 1
        apoc = self.repository.query("RETURN apoc.version() AS version")[0]["version"]
        result: dict[str, Any] = {"neo4j": neo4j, "apoc": apoc, "remote": remote}
        if not remote:
            return result
        headers = {
            "Authorization": f"Bearer {self.config.yandex_api_key}",
            "OpenAI-Project": self.config.yandex_folder_id,
        }
        models_response = requests.get(f"{self.config.yandex_base_url}/models", headers=headers, timeout=30)
        models_response.raise_for_status()
        models = {item["id"] for item in models_response.json().get("data", [])}
        result["models"] = {
            "alice": self.config.llm_model in models,
            "vision": self.config.vision_model in models,
        }
        embeddings = YandexEmbeddings.from_env()
        doc_vector = await asyncio.to_thread(embeddings.embed_documents, ["MEKG document smoke test"])
        query_vector = await asyncio.to_thread(embeddings.embed_query, "MEKG query smoke test")
        result["embeddings"] = {"document": len(doc_vector[0]), "query": len(query_vector)}
        image_path = self.config.artifacts_dir / "preflight.png"
        image = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(image).text((30, 50), "MEKG OCR test: Ni 25 mg/L", fill="black")
        image.save(image_path)
        ocr_text = await asyncio.to_thread(self.vision.recognize_text, image_path.read_bytes(), model="page", mime_type="PNG")
        vision = await asyncio.to_thread(self.vision.analyze_figure, image_path, "Simple OCR preflight image")
        result["ocr"] = {"ok": bool(ocr_text), "chars": len(ocr_text)}
        result["vision"] = {"ok": bool(vision), "figure_type": vision.get("figure_type")}
        sample = SourceElement(
            id="preflight_element",
            kind=ElementKind.TEXT,
            text="At 25 °C the nickel concentration was 200 mg/L.",
        )
        extraction, rejected = await self.extractor.extract_element(sample)
        result["alice"] = {
            "ok": bool(extraction.entities or extraction.conditions or extraction.measurements or extraction.claims),
            "rejected": len(rejected),
        }
        return result

    async def ingest_path(
        self,
        path: str | Path,
        *,
        source_locator: str | None = None,
        category: str | None = None,
        enable_vision: bool = True,
        enable_extraction: bool = True,
    ) -> dict[str, Any]:
        document = await asyncio.to_thread(
            self.parser.parse, path, source_locator=source_locator, category=category
        )
        if any(publication.needs_review for publication in document.publications) and enable_extraction:
            document.publications = await self.extractor.split_publications(document)
        if enable_vision and self.vision.enabled:
            await self._enrich_figures(document)
        await asyncio.to_thread(self.repository.store_source, document)
        self.repository.update_document_stage(document.document_id, "source_stored")
        totals = {"entities": 0, "conditions": 0, "measurements": 0, "experiments": 0, "claims": 0, "experts": 0}
        rejected_count = 0
        errors: list[str] = []
        if enable_extraction:
            semaphore = asyncio.Semaphore(self.config.max_concurrency)
            eligible = [
                element
                for element in document.elements
                if element.text.strip()
                and element.kind in {ElementKind.TEXT, ElementKind.TABLE_ROW, ElementKind.FIGURE, ElementKind.FORMULA}
                and len(element.text.strip()) >= 40
            ]
            eligible_total = len(eligible)
            max_extract = max(0, int(os.getenv("MEKG_MAX_EXTRACT_ELEMENTS_PER_DOC", "0")))
            if max_extract and len(eligible) > max_extract:
                eligible = self._sample_extraction_elements(eligible, max_extract)
                document.warnings.append(
                    f"LLM extraction sampled {len(eligible)} of {eligible_total} eligible source elements"
                )

            async def process(element):
                async with semaphore:
                    try:
                        extraction, rejected = await self.extractor.extract_element(element)
                        counts = await asyncio.to_thread(self.repository.store_extraction, document, extraction)
                        for item in rejected:
                            await asyncio.to_thread(
                                self.repository.store_candidate,
                                document,
                                item["kind"],
                                item["payload"],
                                item["reason"],
                            )
                        return counts, len(rejected), None
                    except Exception as exc:
                        logging.exception("MEKG extraction failed for element %s", element.id)
                        await asyncio.to_thread(
                            self.repository.store_candidate,
                            document,
                            "extraction_error",
                            {"element_id": element.id},
                            f"{type(exc).__name__}: {str(exc)[:500]}",
                        )
                        return {}, 1, f"{element.id}: {type(exc).__name__}"

            results = await asyncio.gather(*(process(element) for element in eligible))
            for counts, rejected, error in results:
                for key, value in counts.items():
                    totals[key] += value
                rejected_count += rejected
                if error:
                    errors.append(error)
            self.repository.update_document_stage(
                document.document_id,
                (
                    "completed_sampled"
                    if max_extract and eligible_total > len(eligible) and not errors
                    else ("completed" if not errors else "completed_with_warnings")
                ),
                extracted_elements=len(eligible),
                eligible_elements=eligible_total,
                sampled_extraction=bool(max_extract and eligible_total > len(eligible)),
                rejected_candidates=rejected_count,
                extraction_errors=len(errors),
            )
        else:
            self.repository.update_document_stage(document.document_id, "parsed_only")
        return {
            "document_id": document.document_id,
            "version_id": document.version_id,
            "file_name": document.file_name,
            "elements": len(document.elements),
            "publications": len(document.publications),
            "extracted": totals,
            "rejected_candidates": rejected_count,
            "warnings": [*document.warnings, *errors],
        }

    async def ingest_manifest(
        self,
        manifest_path: str | Path,
        *,
        corpus_root: str | Path | None = None,
        reset: bool = False,
        enable_vision: bool = True,
        enable_extraction: bool = True,
    ) -> dict[str, Any]:
        self.initialize()
        if reset:
            self.repository.reset_pilot()
            self.initialize()
        entries = validate_manifest(manifest_path, corpus_root)
        results = []
        for index, entry in enumerate(entries, start=1):
            logging.info("MEKG pilot document %s/%s: %s", index, len(entries), entry["path"])
            result = await self.ingest_path(
                entry["resolved_path"],
                source_locator=entry["path"],
                category=entry["category"],
                enable_vision=enable_vision,
                enable_extraction=enable_extraction,
            )
            results.append(result)
        analytics = await asyncio.to_thread(MEKGAnalytics(self.repository).run) if enable_extraction else {}
        report_dir = self.config.artifacts_dir / "pilot-report"
        qa_json, qa_markdown = await asyncio.to_thread(self.auditor.write_report, report_dir)
        return {
            "documents": results,
            "analytics": analytics,
            "qa": self.auditor.run(),
            "reports": {"json": str(qa_json), "markdown": str(qa_markdown)},
        }

    async def retry_extraction_errors(
        self,
        path: str | Path,
        *,
        source_locator: str,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Retry only source elements previously rejected by a technical extraction error."""
        document = await asyncio.to_thread(
            self.parser.parse,
            path,
            source_locator=source_locator,
            category=category,
        )
        rows = self.repository.query(
            """
            MATCH (candidate:MEKGStaging:ExtractionCandidate)-[:EVIDENCED_BY]->
                  (:MEKG:DocumentVersion {id:$version_id})
            WHERE candidate.kind='extraction_error'
            RETURN candidate.id AS candidate_id,candidate.payload_json AS payload_json
            """,
            {"version_id": document.version_id},
        )
        elements = {element.id: element for element in document.elements}
        retried = 0
        resolved = 0
        rejected_count = 0
        remaining: list[dict[str, str]] = []
        for row in rows:
            retried += 1
            payload = json.loads(row["payload_json"] or "{}")
            element_id = payload.get("element_id")
            element = elements.get(element_id)
            if element is None:
                remaining.append({"element_id": str(element_id), "reason": "source element not reconstructed"})
                continue
            try:
                extraction, rejected = await self.extractor.extract_element(element)
                await asyncio.to_thread(self.repository.store_extraction, document, extraction)
                for item in rejected:
                    await asyncio.to_thread(
                        self.repository.store_candidate,
                        document,
                        item["kind"],
                        item["payload"],
                        item["reason"],
                    )
                rejected_count += len(rejected)
                await asyncio.to_thread(self.repository.delete_candidate, row["candidate_id"])
                resolved += 1
            except Exception as exc:
                remaining.append({"element_id": element_id, "reason": f"{type(exc).__name__}: {str(exc)[:500]}"})
        self.repository.update_document_stage(
            document.document_id,
            "completed_sampled" if not remaining else "completed_with_warnings",
            extraction_errors=len(remaining),
        )
        return {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "retried": retried,
            "resolved": resolved,
            "new_rejected_candidates": rejected_count,
            "remaining": remaining,
        }

    async def _enrich_figures(self, document) -> None:
        max_figures = int(os.getenv("MEKG_MAX_VLM_FIGURES_PER_DOC", "20"))
        figures = [element for element in document.elements if element.kind == ElementKind.FIGURE and element.image_path]
        page_context: dict[int, str] = {}
        for element in document.elements:
            if element.kind == ElementKind.TEXT and element.page_number:
                page_context.setdefault(element.page_number, "")
                page_context[element.page_number] += "\n" + element.text[:3000]
        processed = 0
        for figure in figures:
            if processed >= max_figures:
                break
            path = Path(figure.image_path)
            if not path.is_file():
                continue
            context = page_context.get(figure.page_number or figure.slide_number or 0, "")[:4000]
            try:
                content = path.read_bytes()
                ocr = await asyncio.to_thread(self.vision.recognize_text, content, model="markdown", mime_type=path.suffix.lstrip(".") or "PNG")
            except Exception as exc:
                document.warnings.append(f"figure {figure.id}: OCR failed ({type(exc).__name__})")
                ocr = ""
            technical = (ocr + " " + context).casefold()
            relevant = any(
                token in technical
                for token in (
                    "рис.", "figure", "схем", "график", "chart", "process", "процесс", "температур",
                    "concentration", "концентрац", "recovery", "извлечен", "flow", "поток", "slag", "шлак",
                )
            )
            analysis: dict[str, Any] = {}
            if relevant:
                try:
                    analysis = await asyncio.to_thread(self.vision.analyze_figure, path, context)
                except Exception as exc:
                    document.warnings.append(f"figure {figure.id}: VLM failed ({type(exc).__name__})")
            figure.text = "\n".join(
                part for part in [ocr, analysis.get("title", ""), analysis.get("description", ""), *analysis.get("qualitative_claims", [])] if part
            )
            figure.metadata["vision_json"] = analysis
            figure.metadata["numeric_points_approximate"] = analysis.get("numeric_points", [])
            if ocr and ("$" in ocr or re.search(r"[=±∑∫√]", ocr)) and len(ocr) < 3000:
                try:
                    formula = await asyncio.to_thread(self.vision.recognize_text, path.read_bytes(), model="math-markdown", mime_type=path.suffix.lstrip(".") or "PNG")
                    if formula:
                        figure.metadata["formula_markdown"] = formula
                        document.elements.append(
                            SourceElement(
                                id=stable_id("formula", f"{document.version_id}:{figure.id}"),
                                kind=ElementKind.FORMULA,
                                text=formula,
                                page_number=figure.page_number,
                                slide_number=figure.slide_number,
                                image_path=figure.image_path,
                                metadata={"derived_from_figure": figure.id, "ocr_model": "math-markdown"},
                            )
                        )
                except Exception:
                    pass
            processed += 1

    @staticmethod
    def _sample_extraction_elements(elements: list[SourceElement], limit: int) -> list[SourceElement]:
        """Pick evidence-rich elements for a bounded pilot extraction.

        Journals and conference proceedings can contain hundreds of chunks and
        table rows.  For a pilot graph we still want high-signal facts: numeric
        measurements, units, process/equipment words, conclusions, and at least
        one narrative text chunk for document context.
        """
        if limit <= 0 or len(elements) <= limit:
            return elements

        selected: list[SourceElement] = []
        first_text = next((item for item in elements if item.kind == ElementKind.TEXT), None)
        if first_text is not None:
            selected.append(first_text)

        domain_terms = (
            "experiment", "эксперимент", "исслед", "measurement", "измер", "режим", "condition", "услов",
            "temperature", "температур", "concentration", "концентрац", "mg/l", "мг/л", "мг/дм",
            "recovery", "извлеч", "efficiency", "эффектив", "process", "процесс", "technology", "технолог",
            "equipment", "оборуд", "nickel", "никел", "copper", "мед", "slag", "шлак", "matte", "штейн",
            "pgm", "мпг", "au", "ag", "pt", "pd", "conclusion", "вывод", "recommend", "рекоменд",
        )

        def score(index_and_element: tuple[int, SourceElement]) -> tuple[int, int]:
            index, element = index_and_element
            text = element.text.casefold()
            value = 0
            if element.kind == ElementKind.TABLE_ROW:
                value += 8
            elif element.kind == ElementKind.TEXT:
                value += 3
            elif element.kind in {ElementKind.FIGURE, ElementKind.FORMULA}:
                value += 2
            if re.search(r"\d+(?:[,.]\d+)?\s*(?:%|°c|℃|мг/л|мг/дм|г/л|кг/м|м/с|а/м|mpa|kpa|mg/l|g/l|ppm)", text, re.I):
                value += 10
            if re.search(r"(?:<=|>=|≤|≥|=|от\s+\d|до\s+\d|\d+\s*[-–]\s*\d+)", text):
                value += 4
            value += sum(2 for term in domain_terms if term in text)
            return value, -index

        ranked = sorted(enumerate(elements), key=score, reverse=True)
        seen = {item.id for item in selected}
        for _, element in ranked:
            if len(selected) >= limit:
                break
            if element.id in seen:
                continue
            selected.append(element)
            seen.add(element.id)
        order = {element.id: index for index, element in enumerate(elements)}
        selected.sort(key=lambda item: order[item.id])
        return selected

    def resolve_entity(self, request: ResolveEntityRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (entity:MEKG:CanonicalEntity)
            OPTIONAL MATCH (entity)-[:HAS_TERM]->(term:Term)
            WITH entity,collect(DISTINCT term.text) AS terms
            WHERE $type_hint IS NULL OR $type_hint IN labels(entity)
            RETURN entity.id AS entity_id,entity.canonical_name AS canonical_name,
                   [label IN labels(entity) WHERE label <> 'MEKG' AND label <> 'CanonicalEntity'][0] AS type,
                   entity.aliases AS aliases,terms,entity.confidence AS confidence
            LIMIT 500
            """,
            {"type_hint": request.type_hint},
        )
        query = request.text.casefold()
        scored = []
        for row in rows:
            names = [row.get("canonical_name") or "", *(row.get("aliases") or []), *(row.get("terms") or [])]
            score = max((WRatio(query, name.casefold()) for name in names if name), default=0) / 100
            if score >= 0.5:
                scored.append({**row, "match_confidence": score})
        scored.sort(key=lambda item: item["match_confidence"], reverse=True)
        if not scored:
            return ToolResponse(data=None, confidence=0, warnings=["No canonical entity found"])
        best = scored[0]
        best["aliases"] = sorted(set((best.get("aliases") or []) + (best.get("terms") or [])))
        best.pop("terms", None)
        return ToolResponse(data=best, confidence=best["match_confidence"])

    def find_experiments(self, request: FindExperimentsRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (e:MEKG:Experiment)
            OPTIONAL MATCH (e)-[:USES_MATERIAL]->(material:CanonicalEntity)
            OPTIONAL MATCH (e)-[:STUDIES_PROCESS]->(process:CanonicalEntity)
            OPTIONAL MATCH (e)-[:HAS_CONDITION]->(condition:Condition)-[:HAS_PARAMETER]->(parameter:Parameter)
            OPTIONAL MATCH (e)-[:PRODUCED_MEASUREMENT]->(measurement:Measurement)-[:MEASURES_PROPERTY]->(property:Property)
            OPTIONAL MATCH (claim:Claim)-[:GENERALIZES]->(e)
            OPTIONAL MATCH (e)-[:EVIDENCED_BY]->(ev:MEKG)
            WITH e,collect(DISTINCT material.canonical_name) AS materials,
                 collect(DISTINCT process.canonical_name) AS processes,
                 collect(DISTINCT condition{.*,parameter:parameter.canonical_name}) AS conditions,
                 collect(DISTINCT measurement{.*,property:property.canonical_name}) AS measurements,
                 collect(DISTINCT claim{.id,.text,.confidence}) AS claims,
                 collect(DISTINCT ev{.id,.page_number,.slide_number,.text}) AS evidence
            RETURN e{.*,experiment_id:e.id,materials:materials,processes:processes,conditions:conditions,
                     measurements:measurements,claims:claims,evidence:evidence} AS experiment
            ORDER BY e.confidence DESC LIMIT 1000
            """,
            {},
        )
        experiments = []
        for row in rows:
            experiment = row["experiment"]
            name = experiment.get("name") or ""
            scores = []
            if request.material:
                scores.append(self._text_match_score(request.material, [name, *(experiment.get("materials") or [])]))
            if request.process:
                scores.append(self._text_match_score(request.process, [name, *(experiment.get("processes") or [])]))
            if request.property:
                properties = [item.get("property") or item.get("name") or "" for item in experiment.get("measurements") or []]
                scores.append(self._text_match_score(request.property, properties))
            if scores and any(score < 0.45 for score in scores):
                continue
            if request.conditions and not self._conditions_match(request.conditions, experiment.get("conditions") or []):
                continue
            if request.geo_scope:
                scopes = {claim.get("geo_scope") for claim in experiment.get("claims") or []}
                if request.geo_scope not in scopes:
                    continue
            # Publication year is not yet a normalized Experiment property.
            # A requested year range must fail closed rather than be ignored.
            if request.year_min is not None or request.year_max is not None:
                continue
            experiment["match_score"] = sum(scores) / len(scores) if scores else 1.0
            experiments.append(experiment)
        experiments.sort(key=lambda item: (item.get("match_score") or 0, item.get("confidence") or 0), reverse=True)
        experiments = experiments[: request.limit]
        if not experiments:
            return ToolResponse(data={"experiments": [], "knowledge_gap": self._synthetic_gap(request.model_dump())}, confidence=0, warnings=["No verified experiment matched all filters"])
        warnings = []
        if request.year_min is not None or request.year_max is not None:
            warnings.append("Experiment publication years are not normalized in this pilot graph")
        return ToolResponse(
            data={"experiments": experiments},
            evidence=[item for experiment in experiments for item in (experiment.get("evidence") or [])],
            confidence=max(item.get("confidence") or 0 for item in experiments),
            warnings=warnings,
        )

    def find_technologies(self, request: FindTechnologiesRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (t:MEKG:CanonicalEntity)
            WHERE t:Technology OR t:Process OR t:Method
            OPTIONAL MATCH (direct_claim:Claim)-[:GENERALIZES]->(t)
            OPTIONAL MATCH (t)-[:EVIDENCED_BY]->(entity_evidence:MEKG)
            OPTIONAL MATCH (shared_claim:Claim)-[:EVIDENCED_BY]->(entity_evidence)
            WITH t,collect(DISTINCT direct_claim.id) AS direct_claim_ids,
                 apoc.coll.toSet(collect(DISTINCT direct_claim)+collect(DISTINCT shared_claim)) AS claim_nodes,
                 collect(DISTINCT entity_evidence{.id,.page_number,.slide_number,.text}) AS entity_evidence
            RETURN t{.id,.canonical_name,
                     direct_claim_ids:direct_claim_ids,
                     claims:[claim IN claim_nodes WHERE claim IS NOT NULL |
                             claim{.id,.text,.geo_scope,.limitations,.confidence}],
                     entity_evidence:entity_evidence} AS technology
            LIMIT 3000
            """,
            {},
        )
        claim_rows = self.repository.query(
            """
            MATCH (claim:MEKG:Claim)-[:SUPPORTED_BY]->(pack:EvidencePack)-[:HAS_EVIDENCE]->(ev:MEKG)
            RETURN claim{.id,.text,.geo_scope,.limitations,.confidence} AS claim,
                   pack.confidence AS evidence_confidence,
                   collect(DISTINCT ev{.id,.page_number,.slide_number,.text}) AS evidence
            LIMIT 5000
            """
        )
        technologies = []
        used_claims: set[str] = set()
        constraint_queries = []
        if request.input_stream:
            constraint_queries.extend(item.name for item in request.input_stream.components)
        if request.target:
            constraint_queries.append(request.target.parameter)

        def add_candidate(technology: dict[str, Any], score: float) -> None:
            claims = technology.get("claims") or []
            texts = [technology.get("canonical_name") or "", *(claim.get("text") or "" for claim in claims)]
            if request.geo_scope:
                scopes = {claim.get("geo_scope") for claim in claims}
                if request.geo_scope not in scopes:
                    return
            constraint_scores = [self._text_match_score(query, texts) for query in constraint_queries]
            technology["match_score"] = score
            technology["constraint_status"] = (
                "textually_supported"
                if constraint_scores and all(item >= 0.45 for item in constraint_scores)
                else ("not_requested" if not constraint_scores else "unverified")
            )
            technology["matched_constraint_terms"] = sum(item >= 0.45 for item in constraint_scores)
            technology["total_constraint_terms"] = len(constraint_scores)
            technology["evidence_confidence"] = technology.get("evidence_confidence") or max(
                (claim.get("confidence") or 0 for claim in claims), default=0
            )
            used_claims.update(claim.get("id") for claim in claims if claim.get("id"))
            technologies.append(technology)

        for row in rows:
            technology = row["technology"]
            direct_claim_ids = set(technology.pop("direct_claim_ids", []) or [])
            name = technology.get("canonical_name") or ""
            claims = [
                claim
                for claim in (technology.get("claims") or [])
                if claim.get("id") in direct_claim_ids
                or self._text_match_score(name, [claim.get("text") or ""]) >= 0.45
            ]
            technology["claims"] = claims
            texts = [technology.get("canonical_name") or "", *(claim.get("text") or "" for claim in claims)]
            problem_score = self._text_match_score(request.problem, texts, ignore_generic=True)
            constraint_max = max(
                (self._text_match_score(query, texts) for query in constraint_queries), default=0.0
            )
            context_score = self._text_match_score(
                request.problem,
                [item.get("text") or "" for item in technology.get("entity_evidence") or []],
                ignore_generic=True,
            )
            if problem_score < 0.30 and not (constraint_max >= 0.80 and context_score >= 0.30):
                continue
            score = min(1.0, problem_score + 0.15 * constraint_max + 0.05 * context_score)
            if score < 0.30:
                continue
            technology["result_type"] = "canonical_entity"
            add_candidate(technology, score)

        for row in claim_rows:
            claim = row["claim"]
            if claim.get("id") in used_claims:
                continue
            if not self._is_technology_claim(claim.get("text") or ""):
                continue
            texts = [claim.get("text") or ""]
            problem_score = self._text_match_score(request.problem, texts, ignore_generic=True)
            constraint_max = max(
                (self._text_match_score(query, texts) for query in constraint_queries), default=0.0
            )
            context_score = self._text_match_score(
                request.problem,
                [item.get("text") or "" for item in row.get("evidence") or []],
                ignore_generic=True,
            )
            if problem_score < 0.30 and not (constraint_max >= 0.80 and context_score >= 0.30):
                continue
            score = min(1.0, problem_score + 0.15 * constraint_max + 0.05 * context_score)
            if score < 0.30:
                continue
            add_candidate(
                {
                    "id": claim.get("id"),
                    "canonical_name": self._technology_label_from_claim(claim.get("text") or ""),
                    "result_type": "evidence_claim",
                    "claims": [claim],
                    "entity_evidence": row.get("evidence") or [],
                    "evidence_confidence": row.get("evidence_confidence"),
                },
                score,
            )
        technologies.sort(
            key=lambda item: (item.get("match_score") or 0, item.get("evidence_confidence") or 0),
            reverse=True,
        )
        technologies = technologies[: request.limit]
        if not technologies:
            return ToolResponse(data={"technologies": [], "knowledge_gap": self._synthetic_gap(request.model_dump())}, confidence=0, warnings=["No verified technology matched the problem"])
        warnings = []
        if constraint_queries and any(item["constraint_status"] != "textually_supported" for item in technologies):
            warnings.append(
                "Technology candidates match the problem, but the pilot evidence does not verify every input/target constraint"
            )
        knowledge_gap = (
            self._synthetic_gap(request.model_dump())
            if constraint_queries and not any(item["constraint_status"] == "textually_supported" for item in technologies)
            else None
        )
        return ToolResponse(
            data={"technologies": technologies, "knowledge_gap": knowledge_gap},
            evidence=[
                evidence
                for technology in technologies
                for evidence in [*(technology.get("claims") or []), *(technology.get("entity_evidence") or [])]
            ],
            confidence=max(item.get("evidence_confidence") or 0 for item in technologies),
            warnings=warnings,
        )

    def get_evidence_pack(self, request: GetEvidencePackRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (claim:MEKG:Claim {id:$claim_id})-[:SUPPORTED_BY]->(pack:EvidencePack)
            OPTIONAL MATCH (pack)-[:HAS_EVIDENCE]->(ev:MEKG)
            OPTIONAL MATCH (doc:MEKG:Document)-[:HAS_VERSION]->(version:DocumentVersion)
            WHERE ev.id=version.id OR EXISTS { MATCH (version)-[*1..3]->(ev) }
            RETURN claim{.*} AS claim,pack{.*} AS evidence_pack,
                   collect(DISTINCT {document_id:doc.id,file_name:doc.fileName,page:ev.page_number,
                                     slide:ev.slide_number,evidence_id:ev.id,quote:ev.text}) AS sources
            """,
            {"claim_id": request.claim_id},
        )
        if not rows:
            return ToolResponse(data=None, confidence=0, warnings=["Claim or evidence pack not found"])
        row = rows[0]
        return ToolResponse(data=row, evidence=row["sources"], confidence=row["evidence_pack"].get("confidence"))

    def find_contradictions(self, request: FindContradictionsRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (c:MEKG:Contradiction)-[:INVOLVES]->(claim:Claim)
            WITH c,collect(DISTINCT claim{.id,.text,.confidence}) AS claims
            WHERE ($topic IS NULL OR toLower(coalesce(c.reason,'')) CONTAINS toLower($topic)
                  OR any(claim IN claims WHERE toLower(claim.text) CONTAINS toLower($topic)))
              AND ($status IS NULL OR c.status=$status)
            RETURN c{.*,claims:claims} AS contradiction ORDER BY c.severity DESC LIMIT $limit
            """,
            {"topic": request.topic, "status": request.status, "limit": request.limit},
        )
        return ToolResponse(data={"contradictions": [row["contradiction"] for row in rows]})

    def find_knowledge_gaps(self, request: FindKnowledgeGapsRequest) -> ToolResponse:
        terms = [value.casefold() for value in (request.material, request.process, request.condition, request.property, request.geo_region) if value]
        rows = self.repository.query(
            """
            MATCH (g:MEKG:KnowledgeGap)
            WHERE size($terms)=0 OR any(term IN $terms WHERE toLower(g.description) CONTAINS term)
            RETURN g{.*} AS gap ORDER BY g.severity DESC LIMIT $limit
            """,
            {"terms": terms, "limit": request.limit},
        )
        gaps = [row["gap"] for row in rows]
        if not gaps:
            gaps = [self._synthetic_gap(request.model_dump())]
        return ToolResponse(data={"gaps": gaps})

    def find_experts(self, request: FindExpertsRequest) -> ToolResponse:
        rows = self.repository.query(
            """
            MATCH (expert:MEKG:Expert)
            OPTIONAL MATCH (score:ExpertiseScore)-[:FOR_EXPERT]->(expert)
            OPTIONAL MATCH (score)-[:IN_TOPIC]->(topic:TopicTag)
            WITH expert,collect(DISTINCT {topic:topic.canonical_name,score:score.score,evidence_count:score.evidence_count}) AS scores
            RETURN expert{.id,.canonical_name,.organization,.lab,.topics,scores:scores} AS expert
            LIMIT 2000
            """,
            {},
        )
        experts = []
        for row in rows:
            expert = row["expert"]
            topics = [*(expert.get("topics") or []), *(item.get("topic") or "" for item in expert.get("scores") or [])]
            score = self._text_match_score(request.topic, topics)
            if request.process:
                score = min(score, self._text_match_score(request.process, topics))
            if score < 0.45:
                continue
            expert["match_score"] = score
            experts.append(expert)
        experts.sort(
            key=lambda item: (item.get("match_score") or 0, max((score.get("score") or 0 for score in item.get("scores") or []), default=0)),
            reverse=True,
        )
        return ToolResponse(data={"experts": experts[: request.limit]})

    def embed_evidence_chunks(self, *, limit: int = 0, concurrency: int = 3) -> dict[str, Any]:
        rows = self.repository.query(
            """
            MATCH (chunk:MEKG:Chunk)
            WHERE chunk.embedding IS NULL
              AND EXISTS { MATCH (:MEKG)-[:EVIDENCED_BY]->(chunk) }
              AND trim(coalesce(chunk.text,'')) <> ''
            RETURN chunk.id AS id,chunk.text AS text
            ORDER BY chunk.id
            """
        )
        if limit > 0:
            rows = rows[:limit]
        embeddings = YandexEmbeddings.from_env()

        def embed(row: dict[str, Any]) -> tuple[str, list[float]]:
            return row["id"], embeddings.embed_documents([row["text"]])[0]

        completed = 0
        errors = []
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, 8))) as pool:
            futures = {pool.submit(embed, row): row["id"] for row in rows}
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    embedded_id, vector = future.result()
                    self.repository.write(
                        "MATCH (chunk:MEKG:Chunk {id:$id}) SET chunk.embedding=$embedding",
                        {"id": embedded_id, "embedding": vector},
                    )
                    completed += 1
                except Exception as exc:
                    errors.append({"chunk_id": chunk_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return {"eligible": len(rows), "embedded": completed, "errors": errors}

    def semantic_search(self, request: SemanticSearchRequest) -> ToolResponse:
        return self.retriever.semantic_search(request.text, request.limit)

    def cross_corpus_search(self, request: CrossCorpusSearchRequest) -> ToolResponse:
        return self.retriever.search(request)

    def review(self, fact_id: str, decision: ReviewDecision) -> dict[str, Any]:
        return self.repository.review(fact_id, decision)

    def qa(self) -> dict[str, Any]:
        return self.auditor.run()

    def review_queue(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.repository.query(
            """
            MATCH (item)
            WHERE item:MEKGStaging
               OR (item:MEKG AND item.validation_status='raw_extracted')
               OR (item:Publication AND item.needs_review=true)
            RETURN item.id AS id,labels(item) AS labels,item.validation_status AS status,
                   item.reason AS reason,item.title AS title,item.payload_json AS payload_json,
                   item.source_text AS source_text
            ORDER BY item.created_at DESC LIMIT $limit
            """,
            {"limit": limit},
        )

    def export(self, format: str) -> str:
        return self.exporter.serialize(format=format, verified_only=True)

    @staticmethod
    def _synthetic_gap(filters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "missing_evidence",
            "description": "No verified evidence matches the requested combination",
            "filters": {key: value for key, value in filters.items() if value not in (None, [], "")},
            "severity": "high",
            "detected_by": "tool_query",
            "persisted": False,
        }

    @classmethod
    def _text_match_score(
        cls,
        query: str,
        candidates: Iterable[str],
        *,
        ignore_generic: bool = False,
    ) -> float:
        generic = {
            "технология", "технологии", "технологический", "technology", "technologies",
            "метод", "методы", "method", "process", "процесс", "решение", "solutions",
        }

        def tokens(value: str) -> list[str]:
            result = re.findall(r"[a-zа-яё0-9]+", value.casefold().replace("ё", "е"))
            return [item for item in result if len(item) >= 3 and (not ignore_generic or item not in generic)]

        def stem(value: str) -> str:
            if len(value) <= 4:
                return value
            if len(value) <= 7:
                return value[:-1]
            return value[:6]

        query_tokens = tokens(query)
        if not query_tokens:
            return 0.0
        best = 0.0
        for candidate in candidates:
            candidate_tokens = tokens(candidate or "")
            if not candidate_tokens:
                continue
            matched = 0
            for query_token in query_tokens:
                query_stem = stem(query_token)
                short_forms = {
                    "вод": {"вод", "вода", "воды", "воде", "воду", "водой", "водах"},
                }.get(query_stem, {query_token})
                if any(
                    (
                        candidate_token in short_forms
                        if len(query_stem) < 4
                        else (
                            query_stem == stem(candidate_token)
                            or query_stem.startswith(stem(candidate_token))
                            or stem(candidate_token).startswith(query_stem)
                            or WRatio(query_token, candidate_token) >= 90
                        )
                    )
                    for candidate_token in candidate_tokens
                ):
                    matched += 1
            token_score = matched / len(query_tokens)
            phrase_score = WRatio(query.casefold(), (candidate or "").casefold()) / 100
            # Long technical claims can obtain a deceptively high WRatio from
            # generic prose. Phrase similarity is supporting evidence only
            # after at least one query token (or a one-token query) matches.
            fuzzy_score = phrase_score * 0.5 if len(query_tokens) == 1 else 0.0
            best = max(best, token_score, fuzzy_score)
        return min(best, 1.0)

    @staticmethod
    def _technology_label_from_claim(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        method = re.match(
            r"^(?:технический результат\s+)?(способ(?:а)?\s+.{3,65}?)(?::|\s+(?:состоит|заключается))",
            compact,
            flags=re.I,
        )
        if method:
            return method.group(1).strip(" .,:;—-")
        match = re.match(
            r"^(.{3,80}?)(?:\s+(?:удаляет|позволяет|обеспечивает|предлагает|эффектив(?:ен|на|ны)|"
            r"может|предназначен(?:а)?|removes|enables|provides|offers|is effective)\b)",
            compact,
            flags=re.I,
        )
        return (match.group(1) if match else compact[:80]).strip(" .,:;—-")

    @staticmethod
    def _is_technology_claim(text: str) -> bool:
        normalized = text.casefold().replace("ё", "е")
        return any(
            token in normalized
            for token in (
                "технолог", "метод", "способ", "процесс", "установк", "систем", "очист",
                "обработ", "осмос", "мембран", "удаля", "извлеч", "фильтр", "флотац",
                "сепарац", "дистилл", "кристаллиз", "обессол", "technology", "method",
                "process", "system", "treatment", "remov", "filtration", "membrane",
            )
        )

    @classmethod
    def _conditions_match(cls, filters: Iterable[Any], conditions: list[dict[str, Any]]) -> bool:
        normalizer = UnitNormalizer()
        for requested in filters:
            matching = [
                condition
                for condition in conditions
                if cls._text_match_score(
                    requested.parameter,
                    [condition.get("parameter") or "", condition.get("name") or ""],
                )
                >= 0.45
            ]
            if requested.unit:
                normalized = normalizer.normalize(requested.unit)
                matching = [
                    condition
                    for condition in matching
                    if condition.get("unit_normalized") == normalized.symbol
                    or condition.get("unit_original") == requested.unit
                ]
            in_range = False
            for condition in matching:
                values = [
                    value
                    for value in (
                        condition.get("value_min"),
                        condition.get("numeric_value"),
                        condition.get("value_max"),
                    )
                    if value is not None
                ]
                if not values:
                    continue
                low, high = min(values), max(values)
                if requested.min is not None and high < requested.min:
                    continue
                if requested.max is not None and low > requested.max:
                    continue
                in_range = True
                break
            if not in_range:
                return False
        return True
