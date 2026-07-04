from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .config import MEKGConfig
from .models import ToolResponse, WebSearchRequest


@dataclass(frozen=True)
class WebSourceProfile:
    id: str
    title: str
    domains: tuple[str, ...]
    description: str
    metadata_only: bool = False
    evidence_tier: str = "primary"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "domains": list(self.domains),
            "description": self.description,
            "metadata_only": self.metadata_only,
            "evidence_tier": self.evidence_tier,
        }


WEB_SOURCE_PROFILES: dict[str, WebSourceProfile] = {
    "indexes": WebSourceProfile(
        "indexes", "Научные индексы", ("scopus.com", "webofscience.com"),
        "Scopus и Web of Science: публичные карточки и метаданные.", True, "metadata",
    ),
    "journals": WebSourceProfile(
        "journals", "Журнальные платформы",
        ("sciencedirect.com", "link.springer.com", "onlinelibrary.wiley.com", "elsevier.com"),
        "ScienceDirect, Springer, Wiley и Elsevier.", False, "primary",
    ),
    "mining_metals": WebSourceProfile(
        "mining_metals", "Mining и металлы",
        ("onemine.org", "asminternational.org", "dl.asminternational.org"),
        "OneMine и ASM International.", False, "primary",
    ),
    "chemistry": WebSourceProfile(
        "chemistry", "Химические базы", ("cas.org", "scifinder-n.cas.org", "reaxys.com"),
        "CAS/SciFinder и Reaxys: публичные карточки и метаданные.", True, "metadata",
    ),
    "patents": WebSourceProfile(
        "patents", "Патенты",
        ("worldwide.espacenet.com", "patents.google.com", "patentscope.wipo.int"),
        "Espacenet, Google Patents и WIPO PATENTSCOPE.", False, "primary",
    ),
    "russian": WebSourceProfile(
        "russian", "Российские публикации", ("elibrary.ru", "cyberleninka.ru"),
        "eLIBRARY/РИНЦ и КиберЛенинка.", False, "primary",
    ),
    "properties": WebSourceProfile(
        "properties", "Свойства материалов", ("matweb.com", "totalmateria.com", "app.knovel.com"),
        "MatWeb, Total Materia и Knovel; закрытые записи используются как метаданные.", True, "metadata",
    ),
    "materials_data": WebSourceProfile(
        "materials_data", "FAIR и computational data",
        ("materialsproject.org", "nomad-lab.eu", "materialscloud.org"),
        "Materials Project, NOMAD и Materials Cloud.", False, "dataset",
    ),
    "preprints": WebSourceProfile(
        "preprints", "Препринты", ("arxiv.org", "chemrxiv.org"),
        "Свежие препринты; не основной уровень evidence.", False, "preprint",
    ),
}

METADATA_ONLY_DOMAINS = {
    domain
    for profile in WEB_SOURCE_PROFILES.values()
    if profile.metadata_only
    for domain in profile.domains
}


_PATH_PATTERNS = (
    r"(?:[A-Za-z]:\\|/code/|/corpus/|\\\\)[^\s\"']+",
    r"\b(?:doc|chunk|version|run)_[0-9A-Za-z_-]{8,}\b",
    r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
)


def sanitize_web_query(value: str, redact_terms: tuple[str, ...] = ()) -> str:
    """Minimize the external prompt and remove local paths and internal identifiers."""
    text = " ".join(value.replace("\x00", " ").split())
    for pattern in _PATH_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.I)
    for term in redact_terms:
        text = re.sub(re.escape(term), "[REDACTED]", text, flags=re.I)
    return text[:1800].strip()


def select_web_profiles(query: str, requested: list[str] | None = None, limit: int = 2) -> list[str]:
    if requested:
        return [item for item in dict.fromkeys(requested) if item in WEB_SOURCE_PROFILES][:limit]
    folded = query.casefold()
    selected: list[str] = []
    rules = (
        ("patents", ("патент", "patent", "espacenet", "wipo")),
        ("materials_data", ("materials project", "nomad", "dft", "computational", "расчёт")),
        ("properties", ("свойств", "property", "прочност", "плотност", "сплав")),
        ("chemistry", ("реагент", "reaction", "compound", "раствор", "chem")),
        ("mining_metals", ("mining", "руд", "металлург", "electrowinning", "выщелач")),
        ("russian", ("росси", "рф", "russia", "отечествен")),
        ("preprints", ("препринт", "preprint", "последн", "latest", "свеж")),
    )
    for profile_id, terms in rules:
        if any(term in folded for term in terms):
            selected.append(profile_id)
        if len(selected) >= limit:
            break
    for fallback in ("journals", "indexes"):
        if len(selected) >= limit:
            break
        if fallback not in selected:
            selected.append(fallback)
    return selected[:limit]


class YandexWebSearchClient:
    def __init__(self, config: MEKGConfig | None = None, *, client: OpenAI | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self.client = client or OpenAI(
            api_key=self.config.yandex_api_key,
            base_url=self.config.yandex_base_url,
            project=self.config.yandex_folder_id,
            timeout=100.0,
            max_retries=1,
        )
        self._rate_lock = threading.Lock()
        self._last_call = 0.0

    @staticmethod
    def profiles() -> list[dict[str, Any]]:
        return [profile.as_dict() for profile in WEB_SOURCE_PROFILES.values()]

    def search(self, request: WebSearchRequest) -> ToolResponse:
        profile_ids = select_web_profiles(" ".join(request.queries), request.profile_ids, limit=2)
        if request.allowed_domains:
            custom_domains = tuple(self._validate_domains(request.allowed_domains))
            custom_metadata_only = bool(custom_domains) and all(
                any(domain == item or domain.endswith("." + item) for item in METADATA_ONLY_DOMAINS)
                for domain in custom_domains
            )
            jobs = [(
                "custom", custom_domains, custom_metadata_only,
                "metadata" if custom_metadata_only else "custom",
            )]
            effective_profiles = ["custom"]
        else:
            jobs = [
                (
                    profile_id,
                    WEB_SOURCE_PROFILES[profile_id].domains,
                    WEB_SOURCE_PROFILES[profile_id].metadata_only,
                    WEB_SOURCE_PROFILES[profile_id].evidence_tier,
                )
                for profile_id in profile_ids
            ]
            effective_profiles = profile_ids
        sanitized = [sanitize_web_query(item, self.config.web_redact_terms) for item in request.queries]
        sanitized = [item for item in sanitized if item]
        if not sanitized:
            raise ValueError("The web query became empty after redaction")

        blocks: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        warnings: list[str] = []
        for profile_id, domains, metadata_only, tier in jobs:
            try:
                block = self._search_profile(
                    profile_id=profile_id,
                    domains=domains,
                    queries=sanitized,
                    metadata_only=metadata_only,
                    tier=tier,
                    request=request,
                )
                blocks.append(block)
                evidence.extend(block["sources"])
                if not block["sources"]:
                    warnings.append(f"{profile_id}: Yandex returned no URL annotations")
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                suffix = f" HTTP {status}" if status else ""
                warnings.append(f"{profile_id}: {type(exc).__name__}{suffix}")
        return ToolResponse(
            data={"blocks": blocks, "profiles": effective_profiles, "queries": sanitized},
            evidence=evidence,
            confidence=max((block.get("confidence", 0.0) for block in blocks), default=0.0),
            warnings=warnings,
        )

    def _search_profile(
        self,
        *,
        profile_id: str,
        domains: tuple[str, ...],
        queries: list[str],
        metadata_only: bool,
        tier: str,
        request: WebSearchRequest,
    ) -> dict[str, Any]:
        self._wait_for_rate_limit()
        prompt = (
            "Выполни веб-поиск по указанным техническим запросам. Возвращай только проверяемые факты, "
            "числа вместе с единицами и явно отмечай, если доступна лишь карточка/аннотация. "
            "Не делай выводов, которых нет в найденных источниках. Запросы:\n- "
            + "\n- ".join(queries)
        )
        response = self.client.responses.create(
            model=self.config.llm_model,
            input=prompt,
            tools=[{
                "type": "web_search",
                "filters": {"allowed_domains": list(domains)},
                "user_location": {"region": request.region or self.config.web_search_region},
                "search_context_size": request.search_context_size,
            }],
            temperature=0.1,
            max_output_tokens=request.max_output_tokens,
        )
        payload = response.model_dump() if hasattr(response, "model_dump") else response
        text = getattr(response, "output_text", None) or self._find_output_text(payload)
        sources = self._extract_sources(payload, domains, profile_id, metadata_only, tier)
        return {
            "profile_id": profile_id,
            "allowed_domains": list(domains),
            "text": text or "",
            "sources": sources,
            "metadata_only": metadata_only,
            "evidence_tier": tier,
            "confidence": 0.45 if metadata_only else (0.62 if sources else 0.25),
        }

    def _wait_for_rate_limit(self) -> None:
        with self._rate_lock:
            remaining = self.config.web_search_min_interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()

    @staticmethod
    def _validate_domains(domains: list[str]) -> list[str]:
        if len(domains) > 5:
            raise ValueError("Yandex Web Search supports at most five allowed domains")
        clean = []
        for domain in domains:
            value = domain.strip().casefold().lstrip(".")
            if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", value):
                raise ValueError(f"Invalid allowed domain: {domain}")
            clean.append(value)
        return list(dict.fromkeys(clean))

    @classmethod
    def _extract_sources(
        cls,
        payload: Any,
        allowed_domains: tuple[str, ...],
        profile_id: str,
        metadata_only: bool,
        tier: str,
    ) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "url_citation" and value.get("url"):
                    annotations.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        sources = []
        seen = set()
        for item in annotations:
            url = str(item.get("url", "")).strip()
            if url and not re.match(r"https?://", url, re.I):
                url = "https://" + url
            hostname = (urlparse(url).hostname or "").casefold()
            if not any(hostname == domain or hostname.endswith("." + domain) for domain in allowed_domains):
                continue
            if url in seen:
                continue
            seen.add(url)
            source_metadata_only = metadata_only or any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in METADATA_ONLY_DOMAINS
            )
            sources.append({
                "url": url,
                "title": item.get("title") or hostname,
                "source_type": "web_metadata" if source_metadata_only else "web",
                "profile_id": profile_id,
                "metadata_only": source_metadata_only,
                "evidence_tier": "metadata" if source_metadata_only else tier,
            })
        return sources

    @staticmethod
    def _find_output_text(payload: Any) -> str:
        if isinstance(payload, dict):
            if payload.get("type") == "output_text" and isinstance(payload.get("text"), str):
                return payload["text"]
            for value in payload.values():
                found = YandexWebSearchClient._find_output_text(value)
                if found:
                    return found
        if isinstance(payload, list):
            for value in payload:
                found = YandexWebSearchClient._find_output_text(value)
                if found:
                    return found
        return ""
