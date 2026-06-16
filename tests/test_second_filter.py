from __future__ import annotations

from functools import partial
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.assembly.factpack_builder import build_factpack
from extractor.assembly.second_filter import (  # noqa: E402
    SecondFilterContractViolation,
    build_second_filter_candidate,
    is_stage_shift,
    second_filter,
)
from extractor.validators.factpack_assertions import assert_factpack_schema
from tests.case_meta_helpers import (
    assert_contains_with_case_context,
    assert_equal_with_case_context,
)

FIXTURE_PATH = SKILL_ROOT / "tests/fixtures/second_filter_cases.json"


def _load_fixture_cases() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


FIXTURE_CASES = _load_fixture_cases()
DECISION_CASES = FIXTURE_CASES["decision_cases"]
ERROR_CASES = FIXTURE_CASES["error_cases"]
CASE_META = FIXTURE_CASES.get("case_meta") or {}
_assert_equal_with_case_context = partial(assert_equal_with_case_context, case_meta=CASE_META)
_assert_contains_with_case_context = partial(assert_contains_with_case_context, case_meta=CASE_META)



def _semantic_payload(**overrides) -> dict:
    payload = {
        "argument_chain_id": "CHAIN_RELIEF_01",
        "task_stage": "result_explanation",
        "proof_goal": "验证持续舒缓是否成立",
        "subject_entity": "主播本人",
        "target_object": "婴适孩童面霜",
        "action_type": "talking_head_explanation",
        "visual_carrier_type": "talking_head",
        "contains_new_test_start": False,
        "contains_new_subject_switch": False,
        "contains_new_goal_switch": False,
        "contains_cta_transition": False,
        "semantic_summary": "继续解释持续舒缓结果",
        "evidence_refs": ["ASR", "VISUAL"],
    }
    payload.update(overrides)
    return payload


DEFAULT_CONTEXT = {
    "candidate_score": 0.91,
    "adjacent_protected_count_10s": 1,
    "same_bundle_relation": "same_bundle",
    "ocr_jump_strength": 0.8,
    "layout_migration_strength": 0.9,
}



def _candidate(
    *,
    prev_overrides: dict | None = None,
    next_overrides: dict | None = None,
    decision_context_overrides: dict | None = None,
    **candidate_overrides,
) -> dict:
    payload = {
        "boundary_id": "BOUNDARY_SEG18_SEG19",
        "protected_sec": 147.0,
        "prev_segment_id": "SEG18",
        "next_segment_id": "SEG19",
        "trigger_signals": ["layout_migration", "ocr_structure_jump"],
        "high_ocr_scene": True,
        "prev_segment_semantics": _semantic_payload(evidence_refs=["ASR", "OCR", "VISUAL"]),
        "next_segment_semantics": _semantic_payload(
            visual_carrier_type="broll",
            evidence_refs=["ASR", "OCR", "VISUAL"],
        ),
        "decision_context": dict(DEFAULT_CONTEXT),
    }
    if prev_overrides:
        merged_prev = {"evidence_refs": ["ASR", "OCR", "VISUAL"]}
        merged_prev.update(prev_overrides)
        payload["prev_segment_semantics"] = _semantic_payload(**merged_prev)
    if next_overrides:
        merged_next = {"visual_carrier_type": "broll", "evidence_refs": ["ASR", "OCR", "VISUAL"]}
        merged_next.update(next_overrides)
        payload["next_segment_semantics"] = _semantic_payload(**merged_next)
    if decision_context_overrides:
        payload["decision_context"].update(decision_context_overrides)
    payload.update(candidate_overrides)
    return payload



def _build_case_candidate(spec: dict) -> dict:
    return _candidate(**(spec or {}))



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
        ]
        if ocr_text
        else [],
        "rhythm_facts": {"transition_type": "hard_cut", "pace_marker": "normal"},
    }



def _normalized_stream(segments: list[dict]) -> dict:
    vlm, asr, ocr = [], [], []
    for seg in segments:
        vlm.append(
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "visual_facts": seg["visual_facts"],
                "rhythm_facts": seg["rhythm_facts"],
            }
        )
        asr.append(
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "audio_facts": seg["audio_facts"],
            }
        )
        ocr.append(
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "ocr_facts": seg["ocr_facts"],
            }
        )
    return {"vlm": vlm, "asr": asr, "ocr": ocr}



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



def _assert_decision(candidate: dict, expected_decision: str, expected_reason: str, *, case_id: str = "UNKNOWN") -> dict:
    result = second_filter(candidate)
    _assert_equal_with_case_context(result["decision"], expected_decision, case_id=case_id, field_name="decision")
    _assert_equal_with_case_context(result["reason_code"], expected_reason, case_id=case_id, field_name="reason_code")
    return result


@pytest.mark.unit
@pytest.mark.parametrize("case", DECISION_CASES, ids=[item["case_id"] for item in DECISION_CASES])
def test_second_filter_fixture_decision_cases(case: dict) -> None:
    candidate = _build_case_candidate(case["candidate"])
    expected = case["expected"]
    _assert_decision(candidate, expected["decision"], expected["reason_code"], case_id=case["case_id"])


@pytest.mark.unit
@pytest.mark.parametrize("case", ERROR_CASES, ids=[item["case_id"] for item in ERROR_CASES])
def test_second_filter_fixture_error_cases(case: dict) -> None:
    candidate = _build_case_candidate(case["candidate"])
    with pytest.raises(SecondFilterContractViolation) as exc_info:
        second_filter(candidate)
    _assert_contains_with_case_context(
        str(exc_info.value),
        case["expected_error"],
        case_id=case["case_id"],
        field_name="exception_message",
    )


@pytest.mark.unit
def test_decision_context_candidate_score_change_does_not_flip_decision() -> None:
    low_score = _assert_decision(
        _candidate(
            high_ocr_scene=False,
            prev_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_DC01", "target_object": "成分背书", "visual_carrier_type": "talking_head", "action_type": "instrument_test"},
            next_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_DC01", "target_object": "成分背书", "visual_carrier_type": "report_card", "action_type": "instrument_test"},
            decision_context_overrides={"candidate_score": 0.11},
        ),
        "drop",
        "same_argument_chain_continuous_shot",
        case_id="DC01",
    )
    high_score = _assert_decision(
        _candidate(
            high_ocr_scene=False,
            prev_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_DC01", "target_object": "成分背书", "visual_carrier_type": "talking_head", "action_type": "instrument_test"},
            next_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_DC01", "target_object": "成分背书", "visual_carrier_type": "report_card", "action_type": "instrument_test"},
            decision_context_overrides={"candidate_score": 0.99},
        ),
        "drop",
        "same_argument_chain_continuous_shot",
        case_id="DC01",
    )
    assert low_score["reason_code"] == high_score["reason_code"]


@pytest.mark.unit
def test_decision_context_ocr_strength_change_does_not_flip_decision() -> None:
    low_signal = _assert_decision(
        _candidate(
            prev_overrides={"task_stage": "social_proof", "proof_goal": "建立图卡社证", "argument_chain_id": "CHAIN_DC02", "visual_carrier_type": "comment_page", "action_type": "social_proof_insert"},
            next_overrides={"task_stage": "social_proof", "proof_goal": "建立图卡报告页", "argument_chain_id": "CHAIN_DC02", "visual_carrier_type": "report_card", "action_type": "social_proof_insert", "contains_new_goal_switch": False, "contains_new_subject_switch": False, "contains_cta_transition": False},
            decision_context_overrides={"ocr_jump_strength": 0.1, "layout_migration_strength": 0.1},
        ),
        "drop",
        "high_ocr_structure_jump_without_semantic_shift",
        case_id="DC02",
    )
    high_signal = _assert_decision(
        _candidate(
            prev_overrides={"task_stage": "social_proof", "proof_goal": "建立图卡社证", "argument_chain_id": "CHAIN_DC02", "visual_carrier_type": "comment_page", "action_type": "social_proof_insert"},
            next_overrides={"task_stage": "social_proof", "proof_goal": "建立图卡报告页", "argument_chain_id": "CHAIN_DC02", "visual_carrier_type": "report_card", "action_type": "social_proof_insert", "contains_new_goal_switch": False, "contains_new_subject_switch": False, "contains_cta_transition": False},
            decision_context_overrides={"ocr_jump_strength": 0.95, "layout_migration_strength": 0.95},
        ),
        "drop",
        "high_ocr_structure_jump_without_semantic_shift",
        case_id="DC02",
    )
    assert low_signal["reason_code"] == high_signal["reason_code"]


@pytest.mark.unit
def test_log01_decision_log_contains_required_fields() -> None:
    result = _assert_decision(
        _candidate(
            high_ocr_scene=False,
            prev_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_LOG01", "target_object": "成分背书", "visual_carrier_type": "talking_head", "action_type": "instrument_test"},
            next_overrides={"task_stage": "ingredient_backing", "proof_goal": "建立成分背书", "argument_chain_id": "CHAIN_LOG01", "target_object": "成分背书", "visual_carrier_type": "report_card", "action_type": "instrument_test"},
        ),
        "drop",
        "same_argument_chain_continuous_shot",
        case_id="LOG01",
    )
    assert result["same_chain"] is True
    assert result["same_goal"] is True
    assert result["prev_carrier"]
    assert result["next_carrier"]
    assert result["decision_context"] == DEFAULT_CONTEXT


@pytest.mark.unit
def test_stage_shift_whitelist_matches_prd_examples() -> None:
    assert is_stage_shift("result_explanation", "social_proof") is True
    assert is_stage_shift("social_proof", "cta_close") is True
    assert is_stage_shift("test_execution", "result_explanation") is True
    assert is_stage_shift("social_proof", "test_execution") is False


@pytest.mark.unit
def test_build_second_filter_candidate_and_factpack_trace() -> None:
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
    candidate = build_second_filter_candidate(
        left,
        right,
        boundary_info={
            "boundary_id": "BOUNDARY_SEG01_SEG02",
            "protected_sec": 5.0,
            "trigger_signals": ["layout_migration", "ocr_structure_jump"],
            "representative_metrics": {"score": 0.91},
        },
        adjacent_protected_count_10s=0,
    )
    decision = second_filter(candidate)
    assert decision["boundary_id"] == "BOUNDARY_SEG01_SEG02"
    assert decision["decision"] == "drop"
    assert decision["decision_context"]["candidate_score"] == 0.91
    assert decision["decision_context"]["same_bundle_relation"] == "same_bundle"

    factpack = build_factpack(
        _normalized_stream([left, right]),
        {"source_platform": "抖音", "duration_sec": 10.0, "fps": 30.0, "resolution": "720x1280"},
        preproc=_preproc_payload(),
    )
    assert_factpack_schema(factpack)
    assert factpack["second_filter_trace"]["candidates"]
    assert factpack["second_filter_trace"]["decisions"]
    assert factpack["second_filter_trace"]["decisions"][0]["reason_code"] == "same_step_micro_cut"
    assert factpack["second_filter_trace"]["decisions"][0]["decision_context"]["candidate_score"] == 0.91
    assert len(factpack["semantic_bundles"]) == 1
    assert "same_step_micro_cut" in factpack["semantic_bundles"][0]["aggregation_reason"]
