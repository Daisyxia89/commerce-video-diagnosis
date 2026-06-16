from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(TESTS_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.config.loader import load_config
from extractor.errors import ConfigViolation, PreprocessViolation, ProviderExecutionViolation
from extractor.models.config_models import (
    ExtractorConfig,
    InputConfig,
    LocalToolConfig,
    OutputConfig,
    ProviderConfig,
    ProvidersConfig,
    RuntimeConfig,
)
from extractor.preprocess.pipeline import _preprocess_provider_mode, _run_real_ocr_feedback
from extractor.providers.orchestrator import ProviderOrchestrator
from extractor.validators.config_assertions import assert_config_valid



def _base_config(tmp_path: Path) -> ExtractorConfig:
    return ExtractorConfig(
        runtime=RuntimeConfig(),
        local_tools=LocalToolConfig(workspace_dir=str(tmp_path)),
        input=InputConfig(video_path="demo.mp4", video_id="vid-demo", source_product_id="prod-demo"),
        providers=ProvidersConfig(),
        output=OutputConfig(),
    )



def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def _dummy_preproc(tmp_path: Path) -> dict:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    return {
        "audio_path": str(audio_path),
        "segments": [
            {"segment_id": "SEG01", "segment_type": "main", "start_sec": 0.0, "end_sec": 1.0},
        ],
    }


@pytest.mark.unit
def test_load_config_accepts_provider_fallback_mode(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "runtime": {"provider_fallback_mode": "force_on"},
            "local_tools": {"workspace_dir": str(tmp_path)},
            "input": {"video_id": "vid-demo", "source_product_id": "prod-demo", "video_path": "demo.mp4"},
            "providers": {},
            "output": {},
        },
    )

    config = load_config(str(config_path))

    assert config.runtime.provider_fallback_mode == "force_on"


@pytest.mark.unit
def test_legacy_internal_fallback_mode_config_crashes_early(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "runtime": {"internal_fallback_mode": "force_on"},
            "local_tools": {"workspace_dir": str(tmp_path)},
            "input": {"video_id": "vid-demo", "source_product_id": "prod-demo", "video_path": "demo.mp4"},
            "providers": {},
            "output": {},
        },
    )

    with pytest.raises(TypeError, match="internal_fallback_mode"):
        load_config(str(config_path))


@pytest.mark.unit
def test_provider_fallback_mode_validation_rejects_unknown_enum(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.runtime.provider_fallback_mode = "fallback_everything"

    with pytest.raises(ConfigViolation, match="runtime.provider_fallback_mode"):
        assert_config_valid(config)


@pytest.mark.unit
def test_orchestrator_trace_defaults_to_external_public(tmp_path: Path) -> None:
    orchestrator = ProviderOrchestrator(_base_config(tmp_path))

    assert orchestrator.provider_resolution_trace["environment_mode"] == "external_public"
    assert orchestrator.provider_resolution_trace["fallback_protocol_mode"] == "force_off"
    trace_json = json.dumps(orchestrator.provider_resolution_trace, ensure_ascii=False)
    assert "internal" not in trace_json.lower()


@pytest.mark.unit
def test_orchestrator_force_on_keeps_protocol_trace_but_crashes_without_provider(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.runtime.provider_fallback_mode = "force_on"
    config.providers.asr = ProviderConfig(enabled=True, provider="openai_whisper")
    orchestrator = ProviderOrchestrator(config)

    with pytest.raises(ProviderExecutionViolation, match="公开仓库未内置 public_fallback_stub 实现"):
        orchestrator._run_asr_provider(_dummy_preproc(tmp_path))

    trace = orchestrator.provider_resolution_trace
    assert trace["environment_mode"] == "fallback_requested"
    assert trace["fallback_protocol_mode"] == "force_on"
    assert trace["asr"] == {
        "selected_provider_mode": "error",
        "provider_name": "unconfigured",
        "fallback_used": False,
        "fallback_reason": "provider_not_configured",
    }


@pytest.mark.unit
def test_orchestrator_force_off_requires_explicit_byok_guidance(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.vlm = ProviderConfig(enabled=True, provider="openai_gpt4o_vision")
    orchestrator = ProviderOrchestrator(config)
    preproc = {
        "segments": [
            {
                "segment_id": "SEG01",
                "segment_type": "main",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "frame_path": str(tmp_path / "frame.jpg"),
                "frames": [],
            }
        ]
    }
    Path(preproc["segments"][0]["frame_path"]).write_bytes(b"frame")

    with pytest.raises(ProviderExecutionViolation, match="请在 providers.vlm 中配置可用的视觉 provider"):
        orchestrator._run_vlm_provider(preproc)

    assert orchestrator.provider_resolution_trace["vlm"] == {
        "selected_provider_mode": "error",
        "provider_name": "unconfigured",
        "fallback_used": False,
        "fallback_reason": "provider_not_configured",
    }


@pytest.mark.unit
def test_preprocess_provider_mode_and_error_messages_match_public_release_contract() -> None:
    assert _preprocess_provider_mode("force_off") == "external_public"
    assert _preprocess_provider_mode("force_on") == "fallback_requested"
    assert _preprocess_provider_mode("auto") == "external_public"

    with pytest.raises(PreprocessViolation, match="公开仓库仅保留 provider fallback 协议位"):
        _run_real_ocr_feedback("demo.jpg", provider_fallback_mode="force_off")

    with pytest.raises(PreprocessViolation, match="公开仓库未包含任何内置实现"):
        _run_real_ocr_feedback("demo.jpg", provider_fallback_mode="force_on")
