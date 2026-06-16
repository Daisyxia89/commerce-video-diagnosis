from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMProviderConfig:
    endpoint: str = ""
    api_key: str = ""
    model: str = "doubao-1.5-pro-32k-250115"
    timeout: int = 60
    provider: str = "openai_compatible"
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)


def resolve_llm_config(
    *,
    endpoint: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> LLMProviderConfig:
    resolved_endpoint = (endpoint or base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    resolved_model = model or os.environ.get("OPENAI_MODEL") or "doubao-1.5-pro-32k-250115"
    resolved_timeout = timeout if timeout is not None else int(os.environ.get("OPENAI_TIMEOUT", "60"))
    resolved_provider = provider or os.environ.get("UNDERSTANDING_LLM_PROVIDER") or "openai_compatible"
    return LLMProviderConfig(
        endpoint=resolved_endpoint,
        api_key=resolved_api_key,
        model=resolved_model,
        timeout=resolved_timeout,
        provider=resolved_provider,
        extra_headers=dict(extra_headers or {}),
    )


def build_chat_headers(config: LLMProviderConfig, *, llm_tag: str = "") -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    if llm_tag:
        headers["X-LLM-TAG"] = llm_tag
    headers.update(dict(config.extra_headers or {}))
    return headers


def require_llm_config(config: LLMProviderConfig, *, purpose: str) -> LLMProviderConfig:
    if not config.is_configured:
        raise RuntimeError(f"缺少 LLM provider 配置，无法调用{purpose}。请显式配置 endpoint/api_key/model，或设置 OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL。")
    return config
