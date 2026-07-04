from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from .config import MEKGConfig


class YandexVisionClient:
    """OCR and multimodal adapter that never logs document payloads."""

    def __init__(self, config: MEKGConfig | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self._vision_client: OpenAI | None = None
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.yandex_api_key and self.config.yandex_folder_id)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.config.yandex_api_key}",
            "x-folder-id": self.config.yandex_folder_id,
            "x-data-logging-enabled": "true" if self.config.data_logging else "false",
            "Content-Type": "application/json",
        }

    def recognize(self, content: bytes, *, model: str = "page-column-sort", mime_type: str = "PNG") -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Yandex Vision OCR is not configured")
        digest = hashlib.sha256(model.encode() + content).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        normalized_mime = mime_type.upper()
        if normalized_mime == "JPG":
            normalized_mime = "JPEG"
        payload = {
            "mimeType": normalized_mime,
            "languageCodes": ["ru", "en"],
            "model": model,
            "content": base64.b64encode(content).decode("ascii"),
        }
        response = requests.post(
            self.config.yandex_ocr_url,
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        self._cache[digest] = data
        return data

    def recognize_text(self, content: bytes, *, model: str = "page-column-sort", mime_type: str = "PNG") -> str:
        data = self.recognize(content, model=model, mime_type=mime_type)
        return (
            data.get("result", {})
            .get("textAnnotation", {})
            .get("markdown")
            or data.get("result", {}).get("textAnnotation", {}).get("fullText")
            or ""
        ).strip()

    def analyze_figure(self, image_path: str | Path, context: str = "") -> dict[str, Any]:
        if not self.enabled or not self.config.vision_model:
            raise RuntimeError("Yandex multimodal model is not configured")
        image_path = Path(image_path)
        content = image_path.read_bytes()
        digest = hashlib.sha256(b"vlm" + content + context.encode("utf-8")).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        if self._vision_client is None:
            self._vision_client = OpenAI(
                api_key=self.config.yandex_api_key,
                base_url=self.config.yandex_base_url,
                project=self.config.yandex_folder_id,
            )
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"
        prompt = (
            "Analyze this metallurgical or mining figure. Return JSON only with keys: "
            "figure_type, title, description, axes, series, qualitative_claims, numeric_points. "
            "numeric_points must contain value, unit, label and approximate=true. Never invent unreadable values. "
            f"Nearby source text: {context[:3000]}"
        )
        response = self._vision_client.chat.completions.create(
            model=self.config.vision_model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logging.warning("VLM returned non-JSON output for image hash %s", digest[:12])
            result = {"description": text, "numeric_points": [], "parse_warning": "non_json"}
        self._cache[digest] = result
        return result

    def preflight(self) -> dict[str, Any]:
        return {
            "configured": self.enabled,
            "ocr_url": self.config.yandex_ocr_url,
            "vision_model": self.config.vision_model,
            "data_logging": self.config.data_logging,
        }
