"""LLM provider abstraction. Add a provider by implementing `generate`."""
import logging
import time
from typing import Protocol

import httpx

from app.config import settings

log = logging.getLogger("llm")

_MAX_ATTEMPTS = 3
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str:
        """Return the model's raw text reply."""


class GeminiProvider:
    """Google Gemini over plain REST (no SDK, so one less dependency)."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, client: httpx.Client | None = None):
        self.name = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=60)

    def generate(self, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        resp: httpx.Response | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                # key goes in a header, not the URL, so it can't leak into logs
                resp = self._client.post(self.URL.format(model=self.name), json=body,
                                         headers={"x-goog-api-key": self._key})
            except httpx.HTTPError as exc:
                if attempt + 1 == _MAX_ATTEMPTS:
                    raise LLMError(f"Could not reach Gemini: {type(exc).__name__}") from exc
                time.sleep(0.5 * (2 ** attempt))
                continue
            if resp.status_code == 200:
                break
            if resp.status_code in _RETRYABLE_STATUSES and attempt + 1 < _MAX_ATTEMPTS:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = min(max(float(retry_after), 0), 3) if retry_after is not None else 0.5 * (2 ** attempt)
                except ValueError:
                    delay = 0.5 * (2 ** attempt)
                time.sleep(delay)
                continue
            break

        if resp is None:
            raise LLMError("Gemini did not return a response")
        if resp.status_code != 200:
            detail = ""
            try:
                payload = resp.json()
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
                detail = error.get("message", "") if isinstance(error, dict) else ""
            except ValueError:
                pass
            if detail:
                detail = str(detail)
                if self._key:
                    detail = detail.replace(self._key, "[redacted]")
                detail = detail[:500]
                raise LLMError(f"Gemini returned HTTP {resp.status_code}: {detail}")
            raise LLMError(f"Gemini returned HTTP {resp.status_code}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Gemini returned an unexpected response (possibly blocked or empty)") from exc


_provider: LLMProvider | None = None
_resolved = False


def get_llm() -> LLMProvider | None:
    """None means 'no LLM configured': the assistant then answers extractively."""
    global _provider, _resolved
    if not _resolved:
        if settings.llm_provider == "gemini":
            if settings.gemini_api_key:
                _provider = GeminiProvider(settings.gemini_api_key, settings.gemini_model)
            else:
                log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty; using extractive mode")
        _resolved = True
    return _provider


def override(provider: LLMProvider | None) -> None:
    """Used by tests."""
    global _provider, _resolved
    _provider, _resolved = provider, True
