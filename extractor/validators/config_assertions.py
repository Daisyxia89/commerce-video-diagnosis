from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import ConfigViolation, DependencyViolation
from ..models.config_models import ExtractorConfig, ProviderConfig
from ..utils.paths import resolve_resource_path



def _validate_provider(name: str, cfg: ProviderConfig) -> None:
    if cfg.required and not cfg.enabled:
        raise ConfigViolation(f"provider {name} required=true 但 enabled=false")
    if not cfg.enabled:
        return
    if cfg.provider == "fixture_file":
        if not cfg.path:
            raise ConfigViolation(f"provider {name} 缺少 path")
        resolved = resolve_resource_path(cfg.path)
        if not resolved.exists():
            raise ConfigViolation(
                f"provider {name} path 不存在: {cfg.path} (已尝试解析为 {resolved})"
            )
        return
    missing = [field for field in ("provider", "adapter", "model", "timeout_sec") if not getattr(cfg, field)]
    if missing:
        raise ConfigViolation(f"provider {name} 缺少字段: {missing}")
    if not cfg.endpoint:
        raise ConfigViolation(f"provider {name} 缺少 endpoint")
    if not cfg.api_key:
        raise ConfigViolation(f"provider {name} 缺少 api_key")



def assert_config_valid(config: ExtractorConfig) -> None:
    if not config.input.video_id:
        raise ConfigViolation("input.video_id 缺失")
    if not config.input.source_product_id:
        raise ConfigViolation("input.source_product_id 缺失")
    if config.runtime.provider_runtime_max_retries < 0:
        raise ConfigViolation("runtime.provider_runtime_max_retries 不能小于 0")
    if config.runtime.provider_runtime_backoff_sec < 0:
        raise ConfigViolation("runtime.provider_runtime_backoff_sec 不能小于 0")
    if config.runtime.provider_runtime_max_requests_per_run < 0:
        raise ConfigViolation("runtime.provider_runtime_max_requests_per_run 不能小于 0")
    # auto / force_on 为协议保留位：当前公开版无内置 fallback 实现，仅用于表达调用方的配置意图。
    if config.runtime.provider_fallback_mode not in {"auto", "force_on", "force_off"}:
        raise ConfigViolation("runtime.provider_fallback_mode 只允许 auto / force_on / force_off；其中 auto / force_on 为协议保留位，当前公开版无内置 fallback 实现")
    if not config.input.factpack_path and not config.input.video_path and not any(
        [config.providers.vlm.enabled, config.providers.asr.enabled, config.providers.ocr.enabled]
    ):
        raise ConfigViolation("必须提供 factpack_path、video_path，或启用 provider")
    _validate_provider("vlm", config.providers.vlm)
    _validate_provider("asr", config.providers.asr)
    _validate_provider("ocr", config.providers.ocr)



def assert_local_dependencies(config: ExtractorConfig) -> None:
    ffmpeg = config.local_tools.ffmpeg_path or shutil.which("ffmpeg")
    ffprobe = config.local_tools.ffprobe_path or shutil.which("ffprobe")
    if not ffmpeg:
        raise DependencyViolation("缺少 ffmpeg")
    if not ffprobe:
        raise DependencyViolation("缺少 ffprobe")
    workspace_dir = Path(config.local_tools.workspace_dir or "output/commerce_video_diagnosis_runtime")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if not workspace_dir.exists() or not workspace_dir.is_dir():
        raise DependencyViolation(f"workspace_dir 不可用: {workspace_dir}")
    if config.input.video_path and not Path(config.input.video_path).exists():
        raise DependencyViolation(f"video_path 不存在: {config.input.video_path}")
