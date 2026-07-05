from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from .config import MEKGConfig
from .llm_provider import build_agent_llm
from .models import (
    CorpusRoute,
    CrossCorpusSearchRequest,
    RerankResponse,
    SearchRouting,
    ToolResponse,
)
from .search_store import CORPORA, SearchStore
from src.yandex_embeddings import YandexEmbeddings


CORPUS_CARDS = {row[0]: {"name": row[1], "weight": row[2], "description": row[3]} for row in CORPORA}


class CrossCorpusRetriever:
    def __init__(
        self,
        config: MEKGConfig | None = None,
        store: SearchStore | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        self.config = config or MEKGConfig.from_env()
        self.store = store or SearchStore(self.config)
        self.embeddings = YandexEmbeddings.from_env()
        self.llm = llm or build_agent_llm(self.config)

    def search(self, request: CrossCorpusSearchRequest) -> ToolResponse:
        self.store.initialize_schema()
        if request.allow_remote:
            routing, route_warning = self._route(request)
        else:
            routing, route_warning = self._fallback_route(request), "remote models disabled; deterministic BM25 mode"
        filters = request.filters.model_dump(exclude_none=True)
        rankings: list[tuple[str, str, list[dict[str, Any]]]] = []
        futures = {}
        search_warnings: list[str] = []
        dense_available = request.allow_remote
        if not dense_available:
            search_warnings.append("dense search disabled; BM25-only degraded mode")
        with ThreadPoolExecutor(max_workers=min(8, max(2, len(routing.routes) * 2))) as pool:
            for route in routing.routes:
                rewrite = (route.rewrites or [request.query])[0]
                if dense_available:
                    try:
                        vector = self._query_embedding(rewrite)
                        futures[pool.submit(
                            self.store.dense_search,
                            vector,
                            corpora=[route.corpus_id],
                            limit=max(50, request.k_per_corpus * 3),
                            filters=filters,
                        )] = (route.corpus_id, "dense")
                    except Exception as exc:
                        dense_available = False
                        search_warnings.append(
                            f"dense search disabled; BM25 fallback: {type(exc).__name__}"
                        )
                futures[pool.submit(
                    self.store.bm25_search,
                    self._bm25_query(rewrite),
                    corpora=[route.corpus_id],
                    limit=max(50, request.k_per_corpus * 3),
                    filters=filters,
                )] = (route.corpus_id, "bm25")
            for future in as_completed(futures):
                corpus_id, channel = futures[future]
                try:
                    rows = future.result()
                    readable = [
                        {**row, "text": self._clean_text(str(row.get("text") or ""))}
                        for row in rows if self._is_readable(str(row.get("text") or ""))
                    ]
                    dropped = len(rows) - len(readable)
                    if dropped:
                        search_warnings.append(
                            f"{corpus_id} {channel}: dropped {dropped} unreadable text-layer chunks"
                        )
                    rankings.append((corpus_id, channel, readable))
                except Exception as exc:
                    search_warnings.append(f"{corpus_id} {channel}: {type(exc).__name__}")

        candidates = self._fuse(rankings, request)
        if request.allow_remote:
            rerank, rerank_warning = self._rerank(request, candidates[:40])
        else:
            rerank, rerank_warning = RerankResponse(), "Agent LLM rerank disabled in degraded mode"
        rerank_map = {item.chunk_id: item for item in rerank.items}
        numeric_query = self._numeric_mentions(request.query)
        corpus_confidence = {route.corpus_id: route.confidence for route in routing.routes}
        results = []
        for candidate in candidates:
            reranked = rerank_map.get(candidate["chunk_id"])
            dense = candidate.get("dense_norm", 0.0)
            bm25 = candidate.get("bm25_norm", 0.0)
            rrf = candidate.get("rrf_norm", 0.0)
            slots = reranked.matched_slots if reranked else self._matched_slots(request.target_slots, candidate["text"])
            numeric_score = self._numeric_score(numeric_query, candidate["text"])
            if request.numeric_mode == "strict" and numeric_query and numeric_score == 0:
                continue
            source_quality = 1.0 if candidate.get("source_type") in {"pdf", "docx"} else 0.8
            corpus_weight = CORPUS_CARDS.get(candidate["corpus_id"], {}).get("weight", 1.0)
            if reranked:
                final_score = (
                    0.45 * reranked.score + 0.20 * dense + 0.15 * bm25
                    + 0.10 * min(1.0, len(slots) / max(1, len(request.target_slots)))
                    + 0.05 * source_quality + 0.05 * numeric_score
                ) * corpus_weight
            else:
                final_score = (
                    0.50 * rrf + 0.20 * dense + 0.15 * bm25
                    + 0.10 * min(1.0, len(slots) / max(1, len(request.target_slots)))
                    + 0.05 * numeric_score
                ) * corpus_weight
            results.append({
                **candidate,
                "score": round(final_score, 6),
                "rerank_score": reranked.score if reranked else None,
                "matched_slots": slots,
                "numeric_mentions": self._numeric_mentions(candidate["text"]),
                "route_confidence": corpus_confidence.get(candidate["corpus_id"], 0.5),
                "rerank_reason": reranked.reason if reranked else None,
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        results = self._diversify(results, request.final_k, request.intent)
        coverage = self._coverage(request.target_slots, results)
        warnings = [warning for warning in (route_warning, rerank_warning, *search_warnings) if warning]
        evidence = [
            {
                "chunk_id": item["chunk_id"], "document_id": item["document_id"],
                "file_name": item.get("file_name"), "page": item.get("page_number"),
                "slide": item.get("slide_number"), "score": item["score"],
            }
            for item in results
        ]
        return ToolResponse(
            data={
                "selected_corpora": [route.model_dump() for route in routing.routes],
                "rewritten_queries": {route.corpus_id: route.rewrites for route in routing.routes},
                "results": results,
                "coverage_hint": coverage,
                "candidate_entities": self._candidate_entities(results),
                **({"rankings": {"channels": len(rankings), "candidates": len(candidates)}} if request.include_debug else {}),
            },
            evidence=evidence,
            confidence=max((item["score"] for item in results), default=0.0),
            warnings=warnings,
        )

    def semantic_search(self, text: str, limit: int) -> ToolResponse:
        vector = self._query_embedding(text)
        rows = self.store.dense_search(
            vector, corpora=list(CORPUS_CARDS), limit=limit, filters={}
        )
        return ToolResponse(
            data={"results": rows},
            evidence=[{
                "chunk_id": row["chunk_id"], "document_id": row["document_id"],
                "file_name": row.get("file_name"), "page": row.get("page_number"),
                "score": row.get("dense_score"),
            } for row in rows],
            confidence=max((row.get("dense_score") or 0 for row in rows), default=0),
            warnings=[] if rows else ["No Postgres vector chunks matched the query"],
        )

    def _route(self, request: CrossCorpusSearchRequest) -> tuple[SearchRouting, str | None]:
        if request.filters.document_ids and not request.corpora:
            return SearchRouting(
                routes=[
                    CorpusRoute(
                        corpus_id=corpus_id,
                        reason="attached-document focus",
                        confidence=1.0,
                        rewrites=[request.query],
                    )
                    for corpus_id in CORPUS_CARDS
                ],
                target_slots=request.target_slots,
            ), None
        if request.corpora:
            valid = [item for item in request.corpora if item in CORPUS_CARDS][: request.max_corpora]
            return SearchRouting(routes=[CorpusRoute(corpus_id=item, reason="explicit", confidence=1, rewrites=[request.query]) for item in valid], target_slots=request.target_slots), None
        cache_value = json.dumps({"q": request.query, "intent": request.intent, "slots": request.target_slots, "max": request.max_corpora}, ensure_ascii=False, sort_keys=True)
        cached = self.store.cache_get("routing", cache_value)
        if cached:
            return SearchRouting.model_validate(cached), None
        cards = json.dumps(CORPUS_CARDS, ensure_ascii=False)
        prompt = (
            f"Select at most {request.max_corpora} corpora for retrieval and produce one concise corpus-aware "
            f"Russian or English search rewrite per route. Intent={request.intent}; slots={request.target_slots}. "
            f"Corpus cards: {cards}. Query: {request.query}"
        )
        try:
            runnable = self.llm.with_structured_output(SearchRouting, method="json_schema")
            result = runnable.invoke([
                SystemMessage(content="You are a retrieval router. Use only corpus ids supplied by the user."),
                HumanMessage(content=prompt),
            ])
            routing = result if isinstance(result, SearchRouting) else SearchRouting.model_validate(result)
            seen = set()
            routes = []
            for route in routing.routes:
                if route.corpus_id in CORPUS_CARDS and route.corpus_id not in seen:
                    seen.add(route.corpus_id)
                    route.rewrites = (route.rewrites or [request.query])[:2]
                    routes.append(route)
            if not routes:
                raise ValueError("router selected no valid corpus")
            if request.intent in {"internal", "practice"} and "internal_reports" not in seen:
                routes.insert(0, CorpusRoute(
                    corpus_id="internal_reports",
                    reason="required for internal-practice intent",
                    confidence=0.9,
                    rewrites=[request.query],
                ))
            if request.intent in {"compare", "comparative"} and len(routes) < 2:
                fallback_id = next(item for item in CORPUS_CARDS if item not in {route.corpus_id for route in routes})
                routes.append(CorpusRoute(
                    corpus_id=fallback_id,
                    reason="second source type required for comparative intent",
                    confidence=0.6,
                    rewrites=[request.query],
                ))
            routing.routes = routes[: request.max_corpora]
            self.store.cache_set("routing", cache_value, routing.model_dump(mode="json"))
            return routing, None
        except Exception as exc:
            return self._fallback_route(request), f"Agent LLM routing fallback: {type(exc).__name__}"

    @staticmethod
    def _fallback_route(request: CrossCorpusSearchRequest) -> SearchRouting:
        query = request.query.casefold()
        order = []
        if request.intent in {"internal", "practice"} or any(word in query for word in ("норникель", "рудник", "предприят", "доклад")):
            order.append("internal_reports")
        if any(word in query for word in ("обзор", "сравн", "review", "миров")):
            order.append("reviews")
        if any(word in query for word in ("конференц", "докладчик", "conference")):
            order.append("conference_materials")
        order.extend(["scientific_articles", "scientific_journals", "conference_materials", "reviews", "internal_reports"])
        unique = list(dict.fromkeys(order))[: request.max_corpora]
        return SearchRouting(
            routes=[CorpusRoute(corpus_id=item, reason="deterministic keyword routing", confidence=0.6, rewrites=[request.query]) for item in unique],
            target_slots=request.target_slots,
        )

    def _query_embedding(self, query: str) -> list[float]:
        cached = self.store.cache_get("query_embedding", query)
        if cached and len(cached.get("vector", [])) == self.embeddings.dimensions:
            return cached["vector"]
        vector = self.embeddings.embed_query(query)
        self.store.cache_set("query_embedding", query, {"vector": vector})
        return vector

    def _rerank(
        self, request: CrossCorpusSearchRequest, candidates: list[dict[str, Any]]
    ) -> tuple[RerankResponse, str | None]:
        if not candidates:
            return RerankResponse(), None
        rows = [{"chunk_id": item["chunk_id"], "corpus": item["corpus_id"], "text": item["text"][:800]} for item in candidates]
        prompt = json.dumps({"query": request.query, "intent": request.intent, "target_slots": request.target_slots, "candidates": rows}, ensure_ascii=False)
        try:
            runnable = self.llm.with_structured_output(RerankResponse, method="json_schema")
            result = runnable.invoke([
                SystemMessage(content="Rerank retrieval evidence. Score only explicit relevance; do not answer the query."),
                HumanMessage(content=prompt),
            ])
            rerank = result if isinstance(result, RerankResponse) else RerankResponse.model_validate(result)
            allowed = {item["chunk_id"] for item in candidates}
            rerank.items = [item for item in rerank.items if item.chunk_id in allowed]
            return rerank, None
        except Exception as exc:
            return RerankResponse(), f"Agent LLM rerank fallback: {type(exc).__name__}"

    @staticmethod
    def _fuse(rankings: list[tuple[str, str, list[dict[str, Any]]]], request: CrossCorpusSearchRequest) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for corpus_id, channel, rows in rankings:
            values = [float(row.get(f"{channel}_score") or 0) for row in rows]
            low, high = (min(values), max(values)) if values else (0.0, 0.0)
            for rank, row in enumerate(rows, start=1):
                item = candidates.setdefault(row["chunk_id"], {**row, "rrf": 0.0, "dense_norm": 0.0, "bm25_norm": 0.0})
                raw = float(row.get(f"{channel}_score") or 0)
                normalized = (raw - low) / (high - low) if high > low else (1.0 if raw else 0.0)
                item[f"{channel}_score"] = raw
                item[f"{channel}_norm"] = max(item.get(f"{channel}_norm", 0.0), normalized)
                item["rrf"] += 1.0 / (60 + rank)
                item["corpus_id"] = corpus_id
        max_rrf = max((item["rrf"] for item in candidates.values()), default=1.0)
        for item in candidates.values():
            item["rrf_norm"] = item["rrf"] / max_rrf
        return sorted(candidates.values(), key=lambda item: item["rrf"], reverse=True)[: max(40, request.final_k * 4)]

    @staticmethod
    def _bm25_query(value: str) -> str:
        tokens = re.findall(r"[\wА-Яа-яЁё-]{2,}", value, flags=re.UNICODE)
        return " OR ".join(tokens[:24]) or value

    @staticmethod
    def _matched_slots(slots: list[str], text: str) -> list[str]:
        folded = text.casefold()
        return [slot for slot in slots if any(token in folded for token in re.findall(r"\w{3,}", slot.casefold()))]

    @staticmethod
    def _numeric_mentions(text: str) -> list[dict[str, Any]]:
        pattern = r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>%|°C|℃|мг/л|мг/дм3|г/л|МПа|кПа|ppm|mg/l|g/l|mpa|kpa)?"
        return [
            {"value": float(match.group("value").replace(",", ".")), "unit": match.group("unit")}
            for match in re.finditer(pattern, text, flags=re.I)
        ][:20]

    @classmethod
    def _numeric_score(cls, requested: list[dict[str, Any]], text: str) -> float:
        if not requested:
            return 0.0
        found = cls._numeric_mentions(text)
        for wanted in requested:
            for actual in found:
                same_unit = not wanted.get("unit") or not actual.get("unit") or wanted["unit"].casefold() == actual["unit"].casefold()
                tolerance = max(1e-9, abs(wanted["value"]) * 0.05)
                if same_unit and abs(wanted["value"] - actual["value"]) <= tolerance:
                    return 1.0
        return 0.0

    @staticmethod
    def _diversify(results: list[dict[str, Any]], limit: int, intent: str) -> list[dict[str, Any]]:
        selected = []
        documents: dict[str, int] = defaultdict(int)
        corpora: dict[str, int] = defaultdict(int)
        for item in results:
            if documents[item["document_id"]] >= 3 or corpora[item["corpus_id"]] >= 8:
                continue
            selected.append(item)
            documents[item["document_id"]] += 1
            corpora[item["corpus_id"]] += 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _coverage(slots: list[str], results: list[dict[str, Any]]) -> dict[str, Any]:
        if not slots:
            return {"covered": [], "weak": [], "missing": []}
        counts = {slot: sum(slot in item.get("matched_slots", []) for item in results) for slot in slots}
        return {
            "covered": [slot for slot, count in counts.items() if count >= 2],
            "weak": [slot for slot, count in counts.items() if count == 1],
            "missing": [slot for slot, count in counts.items() if count == 0],
        }

    @staticmethod
    def _candidate_entities(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for item in results:
            for entity in item.get("candidate_entities") or []:
                key = (entity.get("type") or "", entity.get("name") or "")
                if key[1]:
                    merged[key] = entity
        return list(merged.values())[:100]

    @staticmethod
    def _is_readable(text: str) -> bool:
        """Reject broken PDF font maps/control-character payloads before ranking evidence."""
        value = text.strip()
        if len(value) < 20:
            return False
        controls = sum(ord(char) < 32 and char not in "\n\r\t" for char in value)
        letters_or_digits = sum(char.isalnum() for char in value)
        if controls / len(value) > 0.05:
            return False
        return letters_or_digits / len(value) >= 0.2

    @staticmethod
    def _clean_text(text: str) -> str:
        value = "".join(
            char if ord(char) >= 32 or char in "\n\r\t" else " " for char in text
        )
        return re.sub(r"[ \t]+", " ", value).strip()
