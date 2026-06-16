from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.assembly.factpack_builder import build_factpack
from commerce_video_diagnosis.understanding.core import FactPack


def _segment(segment_id: str, start_sec: float, end_sec: float, *, asr_text: str, visual_subject: str, ocr_text: str = "") -> dict:
    return {
        "segment_id": segment_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "visual_facts": {
            "shot_size": "close_up",
            "camera_movement": "static",
            "visual_subject": visual_subject,
            "lighting_tone": "bright_natural_daylight",
            "key_objects": ["产品", "脸部"],
            "actions": [{"action_name": "speaking", "physical_intensity": "low"}],
        },
        "audio_facts": {"asr_text": asr_text, "sfx_events": [], "bgm_events": []},
        "ocr_facts": [
            {
                "text": ocr_text,
                "position": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.05},
                "color": "white",
                "font_family": "Source Han Sans",
                "font_weight": "regular",
                "font_size_level": "medium",
                "stroke_style": "none",
                "text_effect_style": "solid_fill",
            }
        ] if ocr_text else [],
        "rhythm_facts": {"transition_type": "hard_cut", "pace_marker": "normal"},
    }


def _normalized_stream(segments: list[dict]) -> dict:
    return {
        "vlm": [
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "visual_facts": seg["visual_facts"],
                "rhythm_facts": seg["rhythm_facts"],
            }
            for seg in segments
        ],
        "asr": [
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "audio_facts": seg["audio_facts"],
            }
            for seg in segments
        ],
        "ocr": [
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "ocr_facts": seg["ocr_facts"],
            }
            for seg in segments
        ],
    }


def _preproc_payload() -> dict:
    return {
        "segments": [
            {"segment_id": "SEG01", "segment_type": "main", "frames": [{"frame_id": "SEG01_F01"}]},
            {"segment_id": "SEG02", "segment_type": "main", "frames": [{"frame_id": "SEG02_F01"}]},
        ],
        "decision_report": [
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
                "cluster_id": "CLUSTER01",
                "representative_sec": 5.0,
                "representative_metrics": {"score": 0.91},
                "protected_representative_sec": 5.0,
                "protected_representative_reason": "soft_cut_structural_peak_protected",
            },
            {
                "reason_code": "SHOT_BOUNDARY_REP_PROTECTED",
                "cluster_id": "CLUSTER01",
                "protected_representative_sec": 5.0,
                "protected_representative_reason": "soft_cut_structural_peak_protected",
                "trigger_signals": ["layout_migration", "ocr_structure_jump"],
            },
        ],
    }


def _valid_factpack() -> dict:
    left = _segment(
        "SEG01",
        0.0,
        5.0,
        asr_text="这个面霜现在还是持续舒缓，我脸上已经没那么烫了。",
        visual_subject="主播本人脸部近景",
        ocr_text="持续舒缓",
    )
    right = _segment(
        "SEG02",
        5.0,
        10.0,
        asr_text="继续说这个面霜的持续舒缓，还是这一个结果，没有换论证目标。",
        visual_subject="主播本人脸部近景",
        ocr_text="持续舒缓",
    )
    return build_factpack(
        _normalized_stream([left, right]),
        {"source_platform": "抖音", "duration_sec": 10.0, "fps": 30.0, "resolution": "720x1280"},
        preproc=_preproc_payload(),
    )


def _set_nested(payload: dict, path: tuple[str | int, ...], value: object) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


@pytest.mark.unit
def test_second_filter_trace_schema_accepts_valid_factpack() -> None:
    factpack = _valid_factpack()

    parsed = FactPack.parse_obj(factpack)

    assert parsed.second_filter_trace.candidates
    assert parsed.second_filter_trace.decisions
    assert parsed.second_filter_trace.decisions[0].decision_context.candidate_score == 0.91


@pytest.mark.unit
def test_second_filter_trace_schema_rejects_candidate_extra_field() -> None:
    factpack = copy.deepcopy(_valid_factpack())
    factpack["second_filter_trace"]["candidates"][0]["unexpected_flag"] = True

    with pytest.raises(ValidationError) as exc_info:
        FactPack.parse_obj(factpack)

    assert "second_filter_trace -> candidates -> 0 -> unexpected_flag" in str(exc_info.value)


@pytest.mark.unit
def test_second_filter_trace_schema_rejects_missing_decision_context_field() -> None:
    factpack = copy.deepcopy(_valid_factpack())
    del factpack["second_filter_trace"]["decisions"][0]["decision_context"]["same_bundle_relation"]

    with pytest.raises(ValidationError) as exc_info:
        FactPack.parse_obj(factpack)

    assert "second_filter_trace -> decisions -> 0 -> decision_context -> same_bundle_relation" in str(exc_info.value)


@pytest.mark.unit
def test_second_filter_trace_schema_rejects_trace_level_extra_field() -> None:
    factpack = copy.deepcopy(_valid_factpack())
    factpack["second_filter_trace"]["rogue_bucket"] = []

    with pytest.raises(ValidationError) as exc_info:
        FactPack.parse_obj(factpack)

    assert "second_filter_trace -> rogue_bucket" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "invalid_value", "expected_path"),
    [
        (
            ("second_filter_trace", "candidates", 0, "decision_context", "candidate_score"),
            "0.91",
            "second_filter_trace -> candidates -> 0 -> decision_context -> candidate_score",
        ),
        (
            ("second_filter_trace", "decisions", 0, "decision_context", "candidate_score"),
            -0.01,
            "second_filter_trace -> decisions -> 0 -> decision_context -> candidate_score",
        ),
        (
            ("second_filter_trace", "candidates", 0, "decision_context", "adjacent_protected_count_10s"),
            1.5,
            "second_filter_trace -> candidates -> 0 -> decision_context -> adjacent_protected_count_10s",
        ),
        (
            ("second_filter_trace", "decisions", 0, "decision_context", "adjacent_protected_count_10s"),
            "1",
            "second_filter_trace -> decisions -> 0 -> decision_context -> adjacent_protected_count_10s",
        ),
        (
            ("second_filter_trace", "candidates", 0, "high_ocr_scene"),
            "true",
            "second_filter_trace -> candidates -> 0 -> high_ocr_scene",
        ),
        (
            ("second_filter_trace", "decisions", 0, "same_chain"),
            "false",
            "second_filter_trace -> decisions -> 0 -> same_chain",
        ),
        (
            ("second_filter_trace", "decisions", 0, "cta"),
            1,
            "second_filter_trace -> decisions -> 0 -> cta",
        ),
        (
            ("second_filter_trace", "decisions", 0, "decision"),
            "hold",
            "second_filter_trace -> decisions -> 0 -> decision",
        ),
    ],
    ids=[
        "candidate_score_string_on_candidate",
        "candidate_score_negative_on_decision",
        "adjacent_count_float_on_candidate",
        "adjacent_count_string_on_decision",
        "high_ocr_scene_string",
        "same_chain_string",
        "cta_int",
        "decision_illegal_enum",
    ],
)
def test_second_filter_trace_schema_rejects_type_boundary_violations(
    path: tuple[str | int, ...],
    invalid_value: object,
    expected_path: str,
) -> None:
    factpack = copy.deepcopy(_valid_factpack())
    _set_nested(factpack, path, invalid_value)

    with pytest.raises(ValidationError) as exc_info:
        FactPack.parse_obj(factpack)

    assert expected_path in str(exc_info.value)
