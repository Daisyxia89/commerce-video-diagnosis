from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

SKILL_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(TESTS_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _repo_path(path_value: str | Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _materialize_config(config_path: str | Path, output_path: Path) -> Path:
    resolved_config_path = _repo_path(config_path)
    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))

    local_tools = payload.get("local_tools") or {}
    workspace_dir = local_tools.get("workspace_dir")
    if workspace_dir:
        local_tools["workspace_dir"] = str(_repo_path(workspace_dir))

    input_payload = payload.get("input") or {}
    for key in ("video_path", "factpack_path"):
        path_value = input_payload.get(key)
        if path_value:
            input_payload[key] = str(_repo_path(path_value))

    providers = payload.get("providers") or {}
    for provider_cfg in providers.values():
        if not isinstance(provider_cfg, dict):
            continue
        provider_path = provider_cfg.get("path")
        if provider_path:
            provider_cfg["path"] = str(_repo_path(provider_path))

    output_payload = payload.get("output") or {}
    for key in ("factpack_path", "request_path", "result_path"):
        path_value = output_payload.get(key)
        if path_value:
            output_payload[key] = str(_repo_path(path_value))

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path

from extractor.entry import run_extractor
from extractor.errors import FactPackViolation
from extractor.providers.orchestrator import ProviderOrchestrator
from extractor.models.config_models import (
    ExtractorConfig,
    InputConfig,
    LocalToolConfig,
    OutputConfig,
    ProviderConfig,
    ProvidersConfig,
    RuntimeConfig,
)
from extractor.validators.factpack_assertions import assert_factpack_schema
from full_smoke_assertions import assert_full_smoke_workspace
from scripts.run_raw_video_smoke import build_ocr_regression_summary
from tests.factpack_protocol_negative_cases import (
    apply_mutation,
    build_main_skeleton_assertion_negative_cases,
    build_main_skeleton_parse_negative_cases,
)
from tests.ocr_test_helpers import require_ocr_provider_or_skip
from commerce_video_diagnosis.understanding.core import FactPack

CONFIG = _repo_path("user_skills/commerce-video-diagnosis/fixtures/p0_fixture_config.json")
RAW_VIDEO_REGRESSION_CONFIG = _repo_path("user_skills/commerce-video-diagnosis/fixtures/raw_video_regression_config.json")


SECOND_FILTER_TRACE_TYPE_BOUNDARY_CASES = [
    {
        "label": "candidate_score_string_on_decision_candidate",
        "path": ("second_filter_trace", "decisions", 0, "decision_context", "candidate_score"),
        "invalid_value": "0.91",
        "expected_path": "second_filter_trace -> decisions -> 0 -> decision_context -> candidate_score",
    },
    {
        "label": "adjacent_count_float_on_candidate",
        "path": ("second_filter_trace", "candidates", 0, "decision_context", "adjacent_protected_count_10s"),
        "invalid_value": 1.5,
        "expected_path": "second_filter_trace -> candidates -> 0 -> decision_context -> adjacent_protected_count_10s",
    },
    {
        "label": "same_chain_string_on_decision",
        "path": ("second_filter_trace", "decisions", 0, "same_chain"),
        "invalid_value": "false",
        "expected_path": "second_filter_trace -> decisions -> 0 -> same_chain",
    },
    {
        "label": "decision_illegal_enum_on_decision",
        "path": ("second_filter_trace", "decisions", 0, "decision"),
        "invalid_value": "hold",
        "expected_path": "second_filter_trace -> decisions -> 0 -> decision",
    },
]



@pytest.mark.unit
def test_validate_only(tmp_path: Path) -> None:
    config_path = _materialize_config(CONFIG, tmp_path / "p0_fixture_config.resolved.json")
    result = run_extractor(str(config_path), mode="validate-only")
    assert result["status"] == "validated"



@pytest.mark.unit
def test_build_request(tmp_path: Path) -> None:
    config_path = _materialize_config(CONFIG, tmp_path / "p0_fixture_config.resolved.json")
    result = run_extractor(str(config_path), mode="build-request")
    request = result["request"]
    assert request["video_id"] == "query-1780911885"
    assert "fact_pack" in request
    assert request["fact_pack"]["segments"][0]["segment_id"] == "SEG01"


@pytest.mark.unit
def test_seg02_seg03_seg04_high_ocr_boundary_backfills_seg03_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = ExtractorConfig(
        runtime=RuntimeConfig(),
        local_tools=LocalToolConfig(workspace_dir=str(tmp_path)),
        input=InputConfig(),
        providers=ProvidersConfig(
            vlm=ProviderConfig(enabled=True, provider="openai_compatible_vlm"),
            asr=ProviderConfig(enabled=True, provider="openai_compatible_asr"),
            ocr=ProviderConfig(enabled=True, provider="openai_compatible_ocr"),
        ),
        output=OutputConfig(),
    )
    orchestrator = ProviderOrchestrator(config)

    primary_frame = tmp_path / "seg03_f02.jpg"
    dense_text_frame = tmp_path / "seg03_f04.jpg"
    back_frame = tmp_path / "seg03_f03.jpg"
    for frame in (primary_frame, dense_text_frame, back_frame):
        frame.write_bytes(b"frame")

    calls: list[tuple[str, str]] = []

    def _fake_analyze_single_frame(frame_path: str, segment_id: str, role: str) -> dict:
        calls.append((segment_id, role))
        if role == "middle":
            return {
                "visual_subject": "young woman with flushed face",
                "shot_size": "close-up",
                "camera_movement": "static",
                "lighting_tone": "bright natural daylight",
                "key_objects": ["woman"],
                "actions": [{"action_name": "looking at camera", "physical_intensity": "low"}],
                "ocr_facts": [],
            }
        if role == "dense_text":
            return {
                "visual_subject": "unused fallback",
                "shot_size": "close-up",
                "camera_movement": "static",
                "lighting_tone": "bright natural daylight",
                "key_objects": [],
                "actions": [],
                "ocr_facts": [
                    {"text": "真实效果测评 具体因人而异", "color": "white", "font_family": "Source Han Sans", "font_weight": "bold", "font_size_level": "large", "stroke_style": "none", "text_effect_style": "solid_fill"},
                    {"text": "来你凑近拍！", "color": "white", "font_family": "Source Han Sans", "font_weight": "bold", "font_size_level": "large", "stroke_style": "none", "text_effect_style": "solid_fill"},
                ],
            }
        return {
            "visual_subject": "unused fallback",
            "shot_size": "close-up",
            "camera_movement": "static",
            "lighting_tone": "bright natural daylight",
            "key_objects": [],
            "actions": [],
            "ocr_facts": [{"text": "不应命中", "color": "white", "font_family": "Source Han Sans", "font_weight": "regular", "font_size_level": "medium", "stroke_style": "none", "text_effect_style": "solid_fill"}],
        }

    monkeypatch.setattr(orchestrator, "_run_public_fallback_frame_stub", _fake_analyze_single_frame)

    preproc = {
        "segments": [
            {
                "segment_id": "SEG02",
                "segment_type": "main",
                "start_sec": 7.0,
                "end_sec": 34.0,
                "frame_path": str(tmp_path / "seg02_f02.jpg"),
                "frames": [{"frame_path": str(tmp_path / "seg02_f02.jpg"), "sampling_role": "middle", "frame_second": 18.5}],
                "frame_plan": {"upsampling_triggers": [], "metrics": {"avg_text_density": 0.02, "max_text_boxes": 5}},
            },
            {
                "segment_id": "SEG03",
                "segment_type": "main",
                "start_sec": 34.0,
                "end_sec": 38.5,
                "frame_path": str(primary_frame),
                "frames": [
                    {"frame_path": str(tmp_path / "seg03_f01.jpg"), "sampling_role": "front", "frame_second": 34.0, "representative_score": 121.0},
                    {"frame_path": str(primary_frame), "sampling_role": "middle", "frame_second": 36.0, "representative_score": 111.5},
                    {"frame_path": str(dense_text_frame), "sampling_role": "dense_text", "frame_second": 37.0, "representative_score": 110.1},
                    {"frame_path": str(back_frame), "sampling_role": "back", "frame_second": 38.45, "representative_score": 117.9},
                ],
                "frame_plan": {
                    "upsampling_triggers": ["ACTION_STATE_CHANGE", "RESULT_STATE", "TEXT_CHANGE_DENSE"],
                    "metrics": {"avg_text_density": 0.603032, "max_text_boxes": 244},
                },
            },
            {
                "segment_id": "SEG04",
                "segment_type": "main",
                "start_sec": 38.5,
                "end_sec": 44.5,
                "frame_path": str(tmp_path / "seg04_f02.jpg"),
                "frames": [{"frame_path": str(tmp_path / "seg04_f02.jpg"), "sampling_role": "middle", "frame_second": 41.0}],
                "frame_plan": {"upsampling_triggers": [], "metrics": {"avg_text_density": 0.02, "max_text_boxes": 3}},
            },
        ]
    }

    visual_rows, ocr_rows = orchestrator._run_public_fallback_image_stub(preproc)

    seg03_visual = next(item for item in visual_rows if item["segment_id"] == "SEG03")
    seg03_ocr = next(item for item in ocr_rows if item["segment_id"] == "SEG03")

    assert seg03_visual["visual_facts"]["visual_subject"] == "young woman with flushed face"
    assert [item["text"] for item in seg03_ocr["ocr_facts"]] == ["真实效果测评 具体因人而异", "来你凑近拍！"]
    assert calls == [("SEG02", "middle"), ("SEG03", "middle"), ("SEG03", "dense_text"), ("SEG04", "middle")]


@pytest.mark.unit
def test_two_stage_run(tmp_path: Path) -> None:
    config_path = _materialize_config(CONFIG, tmp_path / "p0_fixture_config.resolved.json")
    result = run_extractor(str(config_path), mode="two-stage-run")
    assert result["status"] == "two_stage_done"
    assert "blueprint" in result["result"]
    written = json.loads(_repo_path("output/p0_fixture_result.json").read_text(encoding="utf-8"))
    assert "blueprint" in written



@pytest.mark.integration
def test_two_stage_run_raw_video_with_real_ocr_feedback(tmp_path: Path) -> None:
    config_path = _materialize_config(RAW_VIDEO_REGRESSION_CONFIG, tmp_path / "raw_video_regression_config.resolved.json")
    require_ocr_provider_or_skip(str(config_path), reason_prefix="raw_video_regression")
    result = run_extractor(str(config_path), mode="two-stage-run")
    assert result["status"] == "two_stage_done"
    assert "blueprint" in result["result"]

    factpack = json.loads(_repo_path("output/raw_video_regression/factpack.json").read_text(encoding="utf-8"))
    assert factpack["segments"]

    parsed_factpack = FactPack.parse_obj(factpack)
    assert parsed_factpack.second_filter_trace.candidates
    assert parsed_factpack.second_filter_trace.decisions

    for case in SECOND_FILTER_TRACE_TYPE_BOUNDARY_CASES:
        mutated_factpack = apply_mutation(
            factpack,
            {"mode": "set", "path": case["path"], "value": case["invalid_value"]},
        )
        with pytest.raises(ValidationError) as exc_info:
            FactPack.parse_obj(mutated_factpack)
        assert case["expected_path"] in str(exc_info.value), case["label"]

    for schema_case in build_main_skeleton_parse_negative_cases(factpack):
        mutated_factpack = apply_mutation(factpack, schema_case["mutation"])
        with pytest.raises(ValidationError) as exc_info:
            FactPack.parse_obj(mutated_factpack)
        assert schema_case["expected_path"] in str(exc_info.value), schema_case["label"]

    for assertion_case in build_main_skeleton_assertion_negative_cases(factpack):
        mutated_factpack = apply_mutation(factpack, assertion_case["mutation"])
        with pytest.raises(FactPackViolation) as exc_info:
            assert_factpack_schema(mutated_factpack)
        assert assertion_case["expected_error"] in str(exc_info.value), assertion_case["label"]

    written_result = json.loads(_repo_path("output/raw_video_regression/result.json").read_text(encoding="utf-8"))
    assert "blueprint" in written_result

    preprocess_payload = json.loads(
        _repo_path("output/raw_video_regression/runtime/preprocess.json").read_text(encoding="utf-8")
    )
    decision_payload = json.loads(
        _repo_path("output/raw_video_regression/runtime/decision_report.json").read_text(encoding="utf-8")
    )

    decision_report = preprocess_payload.get("decision_report", [])
    assert decision_report
    assert decision_payload.get("decision_report") == decision_report

    summary_entry = next(item for item in decision_report if item.get("reason_code") == "DECISION_SUMMARY")
    assert summary_entry["summary"]["ocr_feedback_enabled"] is True
    assert summary_entry["summary"]["ocr_hit_count"] > 0

    dropped_after_rescoring = [
        item for item in decision_report if item.get("reason_code") == "DROPPED_AFTER_OCR_RESCORING"
    ]
    if dropped_after_rescoring:
        assert all("OCR_FEEDBACK_RESCORING" in item.get("reason_codes", []) for item in dropped_after_rescoring)
        assert all(isinstance(item.get("rank_after_ocr"), int) for item in dropped_after_rescoring)

    regression_summary = build_ocr_regression_summary(
        workspace=_repo_path("output/raw_video_regression"),
        result=result,
    )
    assert regression_summary["ocr_feedback_enabled"] is True
    assert regression_summary["ocr_hit_count"] > 0
    assert regression_summary["dropped_after_ocr_rescoring_count"] == len(dropped_after_rescoring)
    assert regression_summary["decision_report_synced"] is True



@pytest.mark.integration
def test_full_smoke_outputs_from_workspace() -> None:
    workspace_value = os.environ.get("AIME_SMOKE_WORKSPACE", "").strip()
    if not workspace_value:
        pytest.skip("AIME_SMOKE_WORKSPACE not set; full smoke workspace assertion skipped")

    workspace = Path(workspace_value)
    if not workspace.is_dir():
        pytest.skip(f"AIME_SMOKE_WORKSPACE does not exist: {workspace}")

    assert_full_smoke_workspace(workspace)
