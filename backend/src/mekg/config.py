from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _secret_list(*names: str) -> tuple[str, ...]:
    """Read a de-duplicated secret list without formatting it for logs."""
    values: list[str] = []
    for name in names:
        raw = os.getenv(name, "")
        for item in raw.replace(";", ",").replace("\n", ",").split(","):
            value = item.strip()
            if value and value not in values:
                values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class MEKGConfig:
    ontology_dir: Path
    artifacts_dir: Path
    llm_model: str
    vision_model: str
    yandex_api_key: str = field(repr=False)
    yandex_folder_id: str
    yandex_base_url: str
    yandex_ocr_url: str
    data_logging: bool
    max_concurrency: int
    chunk_chars: int
    chunk_overlap: int
    ocr_min_chars: int
    postgres_host: str = "search-db"
    postgres_port: int = 5432
    postgres_user: str = "mekg"
    postgres_password: str = field(default="", repr=False)
    postgres_db: str = "mekg_search"
    postgres_pool_min: int = 1
    postgres_pool_max: int = 8
    parse_workers: int = 4
    embed_workers: int = 4
    llm_workers: int = 5
    embed_rate: float = 4.0
    chunk_write_batch: int = 750
    job_lease_seconds: int = 300
    job_max_attempts: int = 3
    agent_poll_seconds: float = 1.0
    agent_lease_seconds: int = 600
    web_search_region: str = "213"
    web_search_min_interval: float = 1.05
    web_redact_terms: tuple[str, ...] = ()
    agent_llm_provider: str = "yandex"
    openrouter_api_keys: tuple[str, ...] = field(default=(), repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_reasoning: bool = True
    openrouter_timeout: float = 180.0
    openrouter_min_interval: float = 3.1
    openrouter_max_output_tokens: int = 3000

    @classmethod
    def from_env(cls) -> "MEKGConfig":
        package_root = Path(__file__).resolve().parents[2]
        folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        return cls(
            ontology_dir=Path(os.getenv("MEKG_ONTOLOGY_DIR", package_root / "ontology")),
            artifacts_dir=Path(os.getenv("MEKG_ARTIFACTS_DIR", package_root / "artifacts")),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            vision_model=os.getenv(
                "VISION_MODEL",
                f"gpt://{folder_id}/qwen3.6-35b-a3b/latest" if folder_id else "",
            ).strip(),
            yandex_api_key=os.getenv("YANDEX_API_KEY", "").strip(),
            yandex_folder_id=folder_id,
            yandex_base_url=os.getenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1").rstrip("/"),
            yandex_ocr_url=os.getenv(
                "YANDEX_OCR_URL", "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
            ).strip(),
            data_logging=_bool("YANDEX_DATA_LOGGING", False),
            max_concurrency=max(1, int(os.getenv("MEKG_MAX_CONCURRENCY", "3"))),
            chunk_chars=max(1000, int(os.getenv("MEKG_CHUNK_CHARS", "10000"))),
            chunk_overlap=max(0, int(os.getenv("MEKG_CHUNK_OVERLAP", "800"))),
            ocr_min_chars=max(0, int(os.getenv("MEKG_OCR_MIN_CHARS", "100"))),
            postgres_host=os.getenv("POSTGRES_HOST", "search-db").strip(),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_user=os.getenv("POSTGRES_USER", "mekg").strip(),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "").strip(),
            postgres_db=os.getenv("POSTGRES_DB", "mekg_search").strip(),
            postgres_pool_min=max(1, int(os.getenv("POSTGRES_POOL_MIN", "1"))),
            postgres_pool_max=max(1, int(os.getenv("POSTGRES_POOL_MAX", "8"))),
            parse_workers=max(1, int(os.getenv("MEKG_PARSE_WORKERS", "4"))),
            embed_workers=max(1, int(os.getenv("MEKG_EMBED_WORKERS", "4"))),
            llm_workers=max(1, int(os.getenv("MEKG_LLM_WORKERS", "5"))),
            embed_rate=max(0.1, float(os.getenv("MEKG_EMBED_RATE", "4"))),
            chunk_write_batch=max(1, int(os.getenv("MEKG_CHUNK_WRITE_BATCH", "750"))),
            job_lease_seconds=max(30, int(os.getenv("MEKG_JOB_LEASE_SECONDS", "300"))),
            job_max_attempts=max(1, int(os.getenv("MEKG_JOB_MAX_ATTEMPTS", "3"))),
            agent_poll_seconds=max(0.2, float(os.getenv("MEKG_AGENT_POLL_SECONDS", "1"))),
            agent_lease_seconds=max(120, int(os.getenv("MEKG_AGENT_LEASE_SECONDS", "600"))),
            web_search_region=os.getenv("YANDEX_WEB_SEARCH_REGION", "213").strip() or "213",
            web_search_min_interval=max(0.1, float(os.getenv("YANDEX_WEB_SEARCH_MIN_INTERVAL", "1.05"))),
            web_redact_terms=tuple(
                item.strip() for item in os.getenv("WEB_REDACT_TERMS", "").split(",") if item.strip()
            ),
            agent_llm_provider=os.getenv("AGENT_LLM_PROVIDER", "yandex").strip().casefold() or "yandex",
            openrouter_api_keys=_secret_list("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
            ).strip(),
            openrouter_reasoning=_bool("OPENROUTER_REASONING", True),
            openrouter_timeout=max(10.0, float(os.getenv("OPENROUTER_TIMEOUT", "180"))),
            openrouter_min_interval=max(
                3.0, float(os.getenv("OPENROUTER_MIN_INTERVAL", "3.1"))
            ),
            openrouter_max_output_tokens=max(
                256, int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "3000"))
            ),
        )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    def validate_yandex(self, *, vision: bool = False) -> None:
        missing = []
        if not self.yandex_api_key:
            missing.append("YANDEX_API_KEY")
        if not self.yandex_folder_id:
            missing.append("YANDEX_FOLDER_ID")
        if not self.llm_model:
            missing.append("LLM_MODEL")
        if vision and not self.vision_model:
            missing.append("VISION_MODEL")
        if missing:
            raise ValueError(f"Missing MEKG settings: {', '.join(missing)}")

    @property
    def agent_model(self) -> str:
        return self.openrouter_model if self.agent_llm_provider == "openrouter" else self.llm_model

    def validate_agent_llm(self) -> None:
        if self.agent_llm_provider not in {"yandex", "openrouter"}:
            raise ValueError("AGENT_LLM_PROVIDER must be 'yandex' or 'openrouter'")
        if self.agent_llm_provider == "openrouter":
            missing = []
            if not self.openrouter_api_keys:
                missing.append("OPENROUTER_API_KEY or OPENROUTER_API_KEYS")
            if not self.openrouter_model:
                missing.append("OPENROUTER_MODEL")
            if missing:
                raise ValueError(f"Missing agent LLM settings: {', '.join(missing)}")
        else:
            self.validate_yandex()
