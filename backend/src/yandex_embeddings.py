import os
from typing import Iterable, List, Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI


YANDEX_AI_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


class YandexEmbeddings(Embeddings):
    """LangChain embeddings adapter for Yandex AI Studio.

    Yandex exposes distinct models for documents and search queries and accepts
    a single input string per OpenAI-compatible embeddings request.  This
    adapter keeps those semantics explicit while presenting LangChain's normal
    ``Embeddings`` interface to Graph Builder.
    """

    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        doc_model: str,
        query_model: str,
        dimensions: int = 768,
        base_url: str = YANDEX_AI_BASE_URL,
        client: Optional[OpenAI] = None,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("YANDEX_API_KEY", api_key),
                ("YANDEX_FOLDER_ID", folder_id),
                ("EMBED_DOC_MODEL", doc_model),
                ("EMBED_QUERY_MODEL", query_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Yandex AI Studio settings: {', '.join(missing)}")
        if dimensions <= 0:
            raise ValueError("EMBED_DIMENSIONS must be a positive integer")

        self.doc_model = doc_model
        self.query_model = query_model
        self.dimensions = dimensions
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            project=folder_id,
            timeout=60.0,
            max_retries=3,
        )

    @classmethod
    def from_env(cls) -> "YandexEmbeddings":
        try:
            dimensions = int(os.getenv("EMBED_DIMENSIONS", "768"))
        except ValueError as exc:
            raise ValueError("EMBED_DIMENSIONS must be an integer") from exc

        return cls(
            api_key=os.getenv("YANDEX_API_KEY", ""),
            folder_id=os.getenv("YANDEX_FOLDER_ID", ""),
            doc_model=os.getenv("EMBED_DOC_MODEL", ""),
            query_model=os.getenv("EMBED_QUERY_MODEL", ""),
            dimensions=dimensions,
            base_url=os.getenv("YANDEX_BASE_URL", YANDEX_AI_BASE_URL),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("Yandex embeddings input cannot be empty")
        return normalized

    def _embed_one(self, text: str, model: str) -> List[float]:
        response = self._client.embeddings.create(
            input=self._normalize_text(text),
            model=model,
            encoding_format="float",
            dimensions=self.dimensions,
        )
        if not response.data:
            raise ValueError("Yandex embeddings API returned no vectors")

        vector = response.data[0].embedding
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Yandex embeddings dimension mismatch: expected {self.dimensions}, got {len(vector)}"
            )
        return vector

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_one(text, self.doc_model) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, self.query_model)
