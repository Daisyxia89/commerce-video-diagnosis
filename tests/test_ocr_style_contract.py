from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.adapters.ocr_adapter import adapt_ocr
from extractor.errors import AdapterViolation, FactPackViolation
from extractor.validators.factpack_assertions import assert_factpack_schema


def _ocr_fact() -> dict:
    return {
        "text": "真实效果测评",
        "position": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
        "color": "white",
        "font_family": "Source Han Sans",
        "font_weight": "bold",
        "font_size_level": "large",
        "stroke_style": "none",
        "text_effect_style": "solid_fill",
    }


def _factpack() -> dict:
    return {
        "video_meta": {
            "source_platform": "抖音",
            "duration_sec": 2.0,
            "fps": 30.0,
            "resolution": "720x1280",
        },
        "segments": [
            {
                "segment_id": "SEG01",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "visual_facts": {
                    "shot_size": "close_up",
                    "camera_movement": "static",
                    "visual_subject": "主播脸部近景",
                    "lighting_tone": "bright_natural_daylight",
                    "key_objects": ["面部"],
                    "actions": [{"action_name": "speaking", "physical_intensity": "low"}],
                },
                "audio_facts": {"asr_text": "这是测试口播", "sfx": [], "bgm_tone": "unknown"},
                "ocr_facts": [_ocr_fact()],
                "rhythm_facts": {"transition_type": "hard_cut", "pace_marker": "normal"},
            }
        ],
        "semantic_bundles": [
            {
                "bundle_id": "BUNDLE_01",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "segment_ids": ["SEG01"],
                "bundle_role": "narrative_unit",
                "aggregation_reason": ["single_physical_segment_no_further_merge"],
                "blocked_boundary_ids": [],
                "coverage_frame_refs": ["SEG01_F01"],
            }
        ],
        "segment_to_bundle_map": {"SEG01": "BUNDLE_01"},
        "bundle_to_segment_range": {
            "BUNDLE_01": {
                "start_segment_index": 0,
                "end_segment_index": 0,
                "start_segment_id": "SEG01",
                "end_segment_id": "SEG01",
            }
        },
        "second_filter_trace": {
            "candidates": [
                {
                    "boundary_id": "BOUNDARY_SEG01_SEG02",
                    "trigger_signals": [],
                    "high_ocr_scene": False,
                    "prev_segment_semantics": {},
                    "next_segment_semantics": {},
                    "decision_context": {
                        "candidate_score": 0.1,
                        "adjacent_protected_count_10s": 0,
                        "same_bundle_relation": "same_bundle",
                        "ocr_jump_strength": 0,
                        "layout_migration_strength": 0,
                    },
                }
            ],
            "decisions": [
                {
                    "boundary_id": "BOUNDARY_SEG01_SEG02",
                    "decision": "keep",
                    "reason_code": "unit_test",
                    "decision_context": {
                        "candidate_score": 0.1,
                        "adjacent_protected_count_10s": 0,
                        "same_bundle_relation": "same_bundle",
                        "ocr_jump_strength": 0,
                        "layout_migration_strength": 0,
                    },
                }
            ],
        },
        "storyboard_source": "semantic_bundles",
    }


@pytest.mark.unit
def test_adapt_ocr_rejects_missing_style_field() -> None:
    raw = [
        {
            "segment_id": "SEG01",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "ocr_facts": [
                {
                    "text": "首屏字幕",
                    "position": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
                    "color": "white",
                    "font_family": "Source Han Sans",
                    "font_weight": "bold",
                    "font_size_level": "large",
                    "stroke_style": "none",
                }
            ],
        }
    ]

    with pytest.raises(AdapterViolation, match="text_effect_style"):
        adapt_ocr(raw)


@pytest.mark.unit
def test_factpack_assertions_reject_missing_style_field() -> None:
    factpack = _factpack()
    broken = copy.deepcopy(factpack)
    del broken["segments"][0]["ocr_facts"][0]["font_weight"]

    with pytest.raises(FactPackViolation, match="ocr_facts.font_weight 缺失"):
        assert_factpack_schema(broken)
