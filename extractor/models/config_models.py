from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeConfig:
    source_platform: str = ""
    trace_artifacts: bool = True
    enable_real_ocr_feedback: bool = False
    ocr_feedback_top_k: int = 4
    provider_runtime_max_retries: int = 2
    provider_runtime_backoff_sec: int = 2
    provider_runtime_max_requests_per_run: int = 0
    provider_fallback_mode: str = "force_off"


@dataclass
class LocalToolConfig:
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    workspace_dir: str = ""


@dataclass
class InputConfig:
    factpack_path: str = ""
    video_path: str = ""
    video_id: str = ""
    source_product_id: str = ""
    request_id: str = ""


@dataclass
class ProviderRetryPolicy:
    max_retries: int = 0
    backoff_sec: int = 0


@dataclass
class ProviderConfig:
    enabled: bool = False
    provider: str = ""
    path: str = ""
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    timeout_sec: int = 60
    required: bool = False
    adapter: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    retry_policy: ProviderRetryPolicy = field(default_factory=ProviderRetryPolicy)


@dataclass
class ProvidersConfig:
    vlm: ProviderConfig = field(default_factory=ProviderConfig)
    asr: ProviderConfig = field(default_factory=ProviderConfig)
    ocr: ProviderConfig = field(default_factory=ProviderConfig)


@dataclass
class OutputConfig:
    factpack_path: str = ""
    request_path: str = ""
    result_path: str = ""


@dataclass
class ExtractorConfig:
    runtime: RuntimeConfig
    local_tools: LocalToolConfig
    input: InputConfig
    providers: ProvidersConfig
    output: OutputConfig



def _provider_from_dict(data: dict[str, Any]) -> ProviderConfig:
    retry = data.get("retry_policy") or {}
    return ProviderConfig(
        enabled=bool(data.get("enabled", False)),
        provider=str(data.get("provider") or ""),
        path=str(data.get("path") or ""),
        endpoint=str(data.get("endpoint") or ""),
        model=str(data.get("model") or ""),
        api_key=str(data.get("api_key") or ""),
        timeout_sec=int(data.get("timeout_sec", 60) or 60),
        required=bool(data.get("required", False)),
        adapter=str(data.get("adapter") or ""),
        extra=dict(data.get("extra") or {}),
        retry_policy=ProviderRetryPolicy(
            max_retries=int(retry.get("max_retries", 0) or 0),
            backoff_sec=int(retry.get("backoff_sec", 0) or 0),
        ),
    )



def extractor_config_from_dict(data: dict[str, Any]) -> ExtractorConfig:
    runtime = RuntimeConfig(**(data.get("runtime") or {}))
    local_tools = LocalToolConfig(**(data.get("local_tools") or {}))
    input_cfg = InputConfig(**(data.get("input") or {}))
    providers_raw = data.get("providers") or {}
    providers = ProvidersConfig(
        vlm=_provider_from_dict(providers_raw.get("vlm") or {}),
        asr=_provider_from_dict(providers_raw.get("asr") or {}),
        ocr=_provider_from_dict(providers_raw.get("ocr") or {}),
    )
    output = OutputConfig(**(data.get("output") or {}))
    return ExtractorConfig(runtime=runtime, local_tools=local_tools, input=input_cfg, providers=providers, output=output)
