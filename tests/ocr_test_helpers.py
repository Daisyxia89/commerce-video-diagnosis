from __future__ import annotations

import os
from pathlib import Path

from extractor.config.loader import load_config
from extractor.models.config_models import ProviderConfig

SKILL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


_DEMO_API_KEYS = {"", "demo-key"}
_DEMO_ENDPOINTS = {
    "",
    "https://example.com/v1/chat/completions",
    "https://example.com/v1/audio/transcriptions",
}


def _provider_looks_configured(provider: ProviderConfig) -> bool:
    if not provider.enabled:
        return False
    if provider.path:
        return True
    endpoint = str(provider.endpoint or "").strip()
    api_key = str(provider.api_key or "").strip()
    if endpoint in _DEMO_ENDPOINTS:
        return False
    if api_key in _DEMO_API_KEYS:
        return False
    return bool(provider.provider and endpoint and api_key and provider.model)


def _resolve_config_path(config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    suffix = candidate.as_posix()
    if suffix.startswith("user_skills/commerce-video-diagnosis/"):
        suffix = suffix.removeprefix("user_skills/commerce-video-diagnosis/")
        return SKILL_ROOT / suffix
    return WORKSPACE_ROOT / candidate


def is_ocr_provider_configured(config_path: str) -> bool:
    config = load_config(str(_resolve_config_path(config_path)))
    return _provider_looks_configured(config.providers.ocr)


def require_ocr_provider_or_skip(config_path: str, *, reason_prefix: str = "") -> None:
    resolved = _resolve_config_path(config_path).resolve()
    if is_ocr_provider_configured(config_path):
        return
    message = f"OCR provider 未配置或仍为公开仓库 demo 占位配置，跳过真实 OCR integration 用例: {resolved}"
    if reason_prefix:
        message = f"{reason_prefix}: {message}"
    import pytest

    pytest.skip(message)


def is_runtime_ocr_env_configured() -> bool:
    endpoint = os.environ.get("VIDEO_FACTPACK_OCR_ENDPOINT", os.environ.get("VIDEO_FACTPACK_VLM_ENDPOINT", "")).strip()
    api_key = os.environ.get("VIDEO_FACTPACK_OCR_API_KEY", os.environ.get("VIDEO_FACTPACK_VLM_API_KEY", "")).strip()
    model = os.environ.get("VIDEO_FACTPACK_OCR_MODEL", os.environ.get("VIDEO_FACTPACK_VLM_MODEL", "")).strip()
    if endpoint in _DEMO_ENDPOINTS:
        return False
    if api_key in _DEMO_API_KEYS:
        return False
    return bool(endpoint and api_key and model)
