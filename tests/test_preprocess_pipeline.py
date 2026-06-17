from __future__ import annotations

from functools import partial
import json
import sys
from pathlib import Path

from tests.ocr_test_helpers import require_ocr_provider_or_skip

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pytest

# 集成用例依赖的视频切片由 smoke/回归阶段生成，不随公开版发布。
# 优先相对 SKILL_ROOT 解析，回退到 REPO_ROOT；缺失时由 _skip_if_clip_missing 跳过。
_CLIP_REL = "output/raw_video_regression/clip_8s.mp4"
CLIP_8S = SKILL_ROOT / _CLIP_REL if (SKILL_ROOT / _CLIP_REL).exists() else REPO_ROOT / _CLIP_REL


def _skip_if_clip_missing() -> str:
    if not Path(CLIP_8S).is_file():
        pytest.skip(f"集成用例依赖 smoke 阶段视频切片，未生成则跳过: {CLIP_8S}")
    return str(CLIP_8S)

from extractor.errors import PreprocessViolation
from extractor.preprocess.pipeline import (
    _annotate_segment_types,
    _apply_frame_budget,
    _build_boundary_clusters,
    _build_segment_frames,
    _collect_raw_boundary_hits,
    _determine_frame_plan,
    _merge_semantic_continuity,
    _select_cluster_representatives,
    run_preprocess,
)
from tests.case_meta_helpers import (
    assert_equal_with_case_context,
    assert_true_with_case_context,
)

PREPROCESS_CASE_META = {

    "P_INT_01": {

        "source_section": "integration",

        "title": "真实视频多帧预处理结构校验",

        "acceptance_focus": "确认 clip_8s.mp4 的 preprocess 产物结构完整，segments / frames / has_tail 等关键字段稳定输出。",

    },

    "P_INT_02": {

        "source_section": "integration",

        "title": "decision_report 汇总与 reason_code 校验",

        "acceptance_focus": "确认 DECISION_SUMMARY 和核心 reason_code 集合稳定输出，摘要字段完整且类型正确。",

    },

    "P_INT_03": {

        "source_section": "integration",

        "title": "真实 OCR rescoring 回放校验",

        "acceptance_focus": "确认开启真实 OCR feedback 后，decision_report 与 candidate_scores 对 rescoring 的记录完整且自洽。",

    },

}

_assert_true_with_preprocess_context = partial(assert_true_with_case_context, case_meta=PREPROCESS_CASE_META)

_assert_equal_with_preprocess_context = partial(assert_equal_with_case_context, case_meta=PREPROCESS_CASE_META)





from extractor.validators.preprocess_assertions import assert_preprocess_output





def _make_sample(
    sec: float,
    *,
    score: float,
    diff_score: float,
    hist_diff: float,
    text_density: float,
    text_boxes: int,
) -> dict:
    return {
        "sec": sec,
        "score": score,
        "diff_score": diff_score,
        "hist_diff": hist_diff,
        "text_density": text_density,
        "text_boxes": text_boxes,
        "reason_codes": [],
    }



def _build_segment(segment_id: str, start_sec: float, end_sec: float, segment_type: str = "main", frame_count: int = 3) -> dict:
    role_templates = [
        ("front", round(start_sec + (end_sec - start_sec) * 0.2, 3)),
        ("middle", round(start_sec + (end_sec - start_sec) * 0.5, 3)),
        ("back", round(start_sec + (end_sec - start_sec) * 0.8, 3)),
        ("dense_text", round(start_sec + (end_sec - start_sec) * 0.6, 3)),
        ("result_state", round(start_sec + (end_sec - start_sec) * 0.9, 3)),
    ]
    frames = []
    for idx, (role, sec) in enumerate(role_templates[:frame_count], start=1):
        frames.append(
            {
                "frame_id": f"{segment_id}_F{idx:02d}",
                "frame_second": sec,
                "sampling_role": role,
                "frame_path": f"output/{segment_id.lower()}_f{idx:02d}.jpg",
            }
        )
    middle = next((frame for frame in frames if frame["sampling_role"] == "middle"), frames[0])
    return {
        "segment_id": segment_id,
        "shot_id": f"SHOT_{segment_id}",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frame_path": middle["frame_path"],
        "frame_second": middle["frame_second"],
        "segment_strategy": "shot_multiframe_coverage",
        "segment_type": segment_type,
        "frames": frames,
        "frame_plan": {
            "default_frame_count": 2 if frame_count <= 2 else 3,
            "final_frame_count": frame_count,
            "final_frame_count_before_budget": frame_count,
            "upsampling_triggers": ["TEXT_CHANGE_DENSE"] if frame_count >= 4 else [],
            "downgrade_reason": "",
            "min_keep_frames": 1 if frame_count <= 2 else 3,
            "metrics": {
                "shot_len": round(end_sec - start_sec, 3),
                "sample_count": max(frame_count, 1),
                "avg_diff_score": 8.0,
                "max_diff_score": 12.0,
                "avg_hist_diff": 0.2,
                "max_hist_diff": 0.3,
                "avg_text_density": 0.02,
                "max_text_density": 0.04,
                "max_text_boxes": 5,
                "text_change_events": 1,
                "focus_switches": 0,
                "late_result_signal": False,
                "stable_short_shot": frame_count <= 2,
            },
        },
    }



def _build_preprocess_payload(*, duration_sec: float, segments: list[dict], has_tail: bool) -> dict:
    frame_paths = [frame["frame_path"] for seg in segments for frame in seg["frames"]]
    frame_seconds = [float(frame["frame_second"]) for seg in segments for frame in seg["frames"]]
    total_selected_frames = sum(len(seg["frames"]) for seg in segments)
    decision_report = [
        {
            "reason_code": "SELECTED_CUT_POINTS",
            "cut_points": [10.0],
            "boundaries_after_merge": [0.0, 10.0, duration_sec],
            "selected_frame_seconds": frame_seconds,
            "raw_boundary_hit_secs": [8.0, 10.0, 25.0],
            "representative_cut_points": [10.0, 25.0],
            "protected_representative_secs": [],
            "provisional_boundaries": [0.0, 10.0, 25.0, duration_sec],
        },
        {
            "reason_code": "DECISION_SUMMARY",
            "summary": {
                "shot_count": len(segments),
                "final_segment_count": len(segments),
                "tail_segment_count": sum(1 for seg in segments if seg.get("segment_type") == "tail"),
                "has_tail": has_tail,
                "total_candidates": max(total_selected_frames, 1),
                "total_selected_frames": total_selected_frames,
                "max_frames": 256,
                "budget_recovery_count": 0,
                "raw_boundary_hit_count": 3,
                "hard_boundary_hit_count": 3,
                "soft_cut_recall_count": 0,
                "boundary_cluster_count": 2,
                "protected_representative_count": 0,
                "representative_boundary_count": 2,
                "semantic_merge_count": 0,
                "final_cut_point_count": len(segments) - 1,
                "cluster_window_sec": 2.5,
                "soft_cut_window_sec": 1.0,
                "ocr_feedback_enabled": False,
                "ocr_hit_count": 0,
                "long_shot_split_count": 0,
            },
        },
        {
            "reason_code": "FRAME_BUDGET_STATUS",
            "max_frames": 256,
            "total_frames_before_budget": total_selected_frames,
            "total_frames_after_budget": total_selected_frames,
            "budget_recovery_count": 0,
            "budget_exceeded": False,
        },
    ]
    for seg in segments:
        decision_report.extend(
            [
                {
                    "reason_code": "DEFAULT_FRAME_COUNT_DECISION",
                    "segment_id": seg["segment_id"],
                    "shot_id": seg["shot_id"],
                    "shot_start": seg["start_sec"],
                    "shot_end": seg["end_sec"],
                    "shot_len": round(seg["end_sec"] - seg["start_sec"], 3),
                    "default_frame_count": seg["frame_plan"]["default_frame_count"],
                    "final_frame_count_before_budget": seg["frame_plan"]["final_frame_count_before_budget"],
                    "stable_short_shot": seg["frame_plan"]["metrics"]["stable_short_shot"],
                    "downgrade_reason": seg["frame_plan"]["downgrade_reason"],
                    "metrics": seg["frame_plan"]["metrics"],
                },
                {
                    "reason_code": "UPSAMPLING_TRIGGER_STATUS",
                    "segment_id": seg["segment_id"],
                    "shot_id": seg["shot_id"],
                    "upsampling_triggers": seg["frame_plan"]["upsampling_triggers"],
                    "upsampled": bool(seg["frame_plan"]["upsampling_triggers"]),
                },
                {
                    "reason_code": "FRAME_SELECTION_APPLIED",
                    "segment_id": seg["segment_id"],
                    "shot_id": seg["shot_id"],
                    "frame_count": len(seg["frames"]),
                    "frames": [
                        {
                            "frame_id": frame["frame_id"],
                            "frame_second": frame["frame_second"],
                            "sampling_role": frame.get("sampling_role"),
                        }
                        for frame in seg["frames"]
                    ],
                },
            ]
        )
    decision_report.extend(
        [
            {
                "reason_code": "SHOT_BOUNDARY_HARD_HIT_DETECTED",
                "sec": 8.0,
                "hist_diff": 0.6,
                "frame_diff": 60.0,
                "trigger": "hist_diff",
                "hit_type": "hard_cut_hit",
                "trigger_signals": ["hist_diff_threshold"],
                "support_signal_count": 1,
                "local_peak_rank": 2,
                "boundary_strength": 108.0,
                "text_density": 0.02,
                "text_boxes": 3,
                "score": 55.0,
            },
            {
                "reason_code": "SHOT_BOUNDARY_HARD_HIT_DETECTED",
                "sec": 10.0,
                "hist_diff": 0.8,
                "frame_diff": 70.0,
                "trigger": "hist_diff",
                "hit_type": "hard_cut_hit",
                "trigger_signals": ["hist_diff_threshold"],
                "support_signal_count": 1,
                "local_peak_rank": 1,
                "boundary_strength": 136.0,
                "text_density": 0.03,
                "text_boxes": 4,
                "score": 80.0,
            },
            {
                "reason_code": "SHOT_BOUNDARY_HARD_HIT_DETECTED",
                "sec": 25.0,
                "hist_diff": 0.7,
                "frame_diff": 60.0,
                "trigger": "hist_diff",
                "hit_type": "hard_cut_hit",
                "trigger_signals": ["hist_diff_threshold"],
                "support_signal_count": 1,
                "local_peak_rank": 1,
                "boundary_strength": 118.0,
                "text_density": 0.04,
                "text_boxes": 5,
                "score": 75.0,
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTERED",
                "cluster_id": "CLUSTER01",
                "cluster_start_sec": 8.0,
                "cluster_end_sec": 10.0,
                "member_secs": [8.0, 10.0],
                "member_types": ["hard_cut_hit", "hard_cut_hit"],
                "protected_candidate_secs": [],
                "protected_candidate_reasons": {},
                "member_count": 2,
                "cluster_window_sec": 2.5,
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTERED",
                "cluster_id": "CLUSTER02",
                "cluster_start_sec": 25.0,
                "cluster_end_sec": 25.0,
                "member_secs": [25.0],
                "member_types": ["hard_cut_hit"],
                "protected_candidate_secs": [],
                "protected_candidate_reasons": {},
                "member_count": 1,
                "cluster_window_sec": 2.5,
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
                "cluster_id": "CLUSTER01",
                "representative_sec": 10.0,
                "representative_reason": "cluster_local_peak_hist_diff_priority",
                "representative_metrics": {"hit_type": "hard_cut_hit", "hist_diff": 0.8, "frame_diff": 70.0, "boundary_strength": 136.0, "score": 80.0, "support_signal_count": 1},
                "peak_window_sec": 1.0,
                "protected_representative_sec": None,
                "protected_representative_reason": None,
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
                "cluster_id": "CLUSTER02",
                "representative_sec": 25.0,
                "representative_reason": "cluster_local_peak_hist_diff_priority",
                "representative_metrics": {"hit_type": "hard_cut_hit", "hist_diff": 0.7, "frame_diff": 60.0, "boundary_strength": 118.0, "score": 75.0, "support_signal_count": 1},
                "peak_window_sec": 1.0,
                "protected_representative_sec": None,
                "protected_representative_reason": None,
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED",
                "cluster_id": "CLUSTER01",
                "dropped_secs": [8.0],
                "drop_reason": "cluster_consolidation_non_representative_members",
            },
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED",
                "cluster_id": "CLUSTER02",
                "dropped_secs": [],
                "drop_reason": "cluster_consolidation_non_representative_members",
            },
            {
                "reason_code": "SHOT_BOUNDARY_MERGE_STATUS",
                "semantic_merge_count": 0,
                "merge_actions": 0,
                "merge_block_count": 0,
                "final_boundary_count_before_short_merge": len(segments) - 1,
                "duration_sec": duration_sec,
            },
        ]
    )
    decision_report.append(
        {
            "reason_code": "TAIL_STATUS_RECORDED",
            "has_tail": has_tail,
            "tail_segment_ids": [seg["segment_id"] for seg in segments if seg.get("segment_type") == "tail"],
        }
    )
    return {
        "video_path": "fake.mp4",
        "video_meta": {
            "source_platform": "抖音",
            "duration_sec": duration_sec,
            "fps": 30.0,
            "resolution": "720x1280",
        },
        "audio_path": "output/fake.wav",
        "frame_paths": frame_paths,
        "frame_seconds": frame_seconds,
        "has_tail": has_tail,
        "segments": segments,
        "decision_report": decision_report,
    }


@pytest.mark.integration
def test_run_preprocess_multi_frames() -> None:
    case_id = "P_INT_01"
    workspace = "output/test_preprocess_multi_frames"
    clip = _skip_if_clip_missing()
    result = run_preprocess(
        video_path=clip,
        workspace_dir=workspace,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        source_platform="抖音",
    )
    _assert_true_with_preprocess_context(
        len(result["frame_paths"]) >= len(result["segments"]),
        case_id=case_id,
        field_name="frame_paths_vs_segments",
        detail=f"frame_paths={len(result['frame_paths'])}, segments={len(result['segments'])}",
    )
    _assert_equal_with_preprocess_context(
        len(result["frame_paths"]),
        len(result["frame_seconds"]),
        case_id=case_id,
        field_name="frame_paths_vs_frame_seconds",
    )
    _assert_true_with_preprocess_context(
        len(result["frame_paths"]) <= 256,
        case_id=case_id,
        field_name="frame_paths_budget",
        detail=f"frame_paths={len(result['frame_paths'])}",
    )
    _assert_true_with_preprocess_context(
        len(result["segments"]) >= 1,
        case_id=case_id,
        field_name="segments_count",
        detail=f"segments={len(result['segments'])}",
    )
    payload = json.loads(Path(workspace, "preprocess.json").read_text(encoding="utf-8"))
    _assert_true_with_preprocess_context(
        payload["video_meta"]["duration_sec"] > 0,
        case_id=case_id,
        field_name="video_meta.duration_sec",
        detail=f"duration_sec={payload['video_meta']['duration_sec']}",
    )
    _assert_equal_with_preprocess_context(
        len(payload["segments"]),
        len(result["segments"]),
        case_id=case_id,
        field_name="payload.segments_vs_result.segments",
    )
    _assert_true_with_preprocess_context(
        isinstance(payload["has_tail"], bool),
        case_id=case_id,
        field_name="has_tail_type",
        detail=f"type={type(payload['has_tail']).__name__}",
    )
    _assert_true_with_preprocess_context(
        all("frame_path" in seg for seg in payload["segments"]),
        case_id=case_id,
        field_name="segments.frame_path",
        detail="each segment must include frame_path",
    )
    _assert_true_with_preprocess_context(
        all("shot_id" in seg for seg in payload["segments"]),
        case_id=case_id,
        field_name="segments.shot_id",
        detail="each segment must include shot_id",
    )
    _assert_true_with_preprocess_context(
        all("segment_strategy" in seg for seg in payload["segments"]),
        case_id=case_id,
        field_name="segments.segment_strategy",
        detail="each segment must include segment_strategy",
    )
    _assert_true_with_preprocess_context(
        all(seg.get("segment_type") in {"main", "tail"} for seg in payload["segments"]),
        case_id=case_id,
        field_name="segments.segment_type",
        detail="segment_type must be main or tail",
    )
    _assert_true_with_preprocess_context(
        all(isinstance(seg.get("frames"), list) and seg["frames"] for seg in payload["segments"]),
        case_id=case_id,
        field_name="segments.frames",
        detail="each segment must contain non-empty frames list",
    )
    _assert_true_with_preprocess_context(
        all(all("sampling_role" in frame for frame in seg["frames"]) for seg in payload["segments"]),
        case_id=case_id,
        field_name="frames.sampling_role",
        detail="each frame must include sampling_role",
    )
    _assert_equal_with_preprocess_context(
        payload["has_tail"],
        any(seg.get("segment_type") == "tail" for seg in payload["segments"]),
        case_id=case_id,
        field_name="has_tail_consistency",
    )


@pytest.mark.integration
def test_decision_report_summary_and_reason_codes() -> None:
    case_id = "P_INT_02"
    workspace = "output/test_preprocess_reason_codes"
    clip = _skip_if_clip_missing()
    run_preprocess(
        video_path=clip,
        workspace_dir=workspace,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        source_platform="抖音",
    )
    payload = json.loads(Path(workspace, "preprocess.json").read_text(encoding="utf-8"))
    decision_report = payload.get("decision_report", [])
    _assert_true_with_preprocess_context(
        bool(decision_report),
        case_id=case_id,
        field_name="decision_report",
        detail="decision_report must be non-empty",
    )

    summary_entry = next(item for item in decision_report if item.get("reason_code") == "DECISION_SUMMARY")
    summary = summary_entry["summary"]
    for key in (
        "total_candidates",
        "ocr_hit_count",
        "shot_count",
        "final_segment_count",
        "long_shot_split_count",
        "tail_segment_count",
        "total_selected_frames",
        "max_frames",
        "budget_recovery_count",
        "raw_boundary_hit_count",
        "hard_boundary_hit_count",
        "soft_cut_recall_count",
        "boundary_cluster_count",
        "protected_representative_count",
        "representative_boundary_count",
        "semantic_merge_count",
        "final_cut_point_count",
    ):
        _assert_true_with_preprocess_context(
            key in summary,
            case_id=case_id,
            field_name=f"summary.{key}",
            detail="missing summary key",
        )
        _assert_true_with_preprocess_context(
            isinstance(summary[key], int),
            case_id=case_id,
            field_name=f"summary.{key}",
            detail=f"type={type(summary[key]).__name__}",
        )
        _assert_true_with_preprocess_context(
            summary[key] >= 0,
            case_id=case_id,
            field_name=f"summary.{key}",
            detail=f"value={summary[key]}",
        )
    _assert_true_with_preprocess_context(
        "cluster_window_sec" in summary,
        case_id=case_id,
        field_name="summary.cluster_window_sec",
        detail="missing summary key",
    )
    _assert_true_with_preprocess_context(
        "soft_cut_window_sec" in summary,
        case_id=case_id,
        field_name="summary.soft_cut_window_sec",
        detail="missing summary key",
    )
    _assert_true_with_preprocess_context(
        isinstance(summary["cluster_window_sec"], (int, float)),
        case_id=case_id,
        field_name="summary.cluster_window_sec",
        detail=f"type={type(summary['cluster_window_sec']).__name__}",
    )
    _assert_true_with_preprocess_context(
        isinstance(summary["soft_cut_window_sec"], (int, float)),
        case_id=case_id,
        field_name="summary.soft_cut_window_sec",
        detail=f"type={type(summary['soft_cut_window_sec']).__name__}",
    )
    _assert_true_with_preprocess_context(
        summary["cluster_window_sec"] > 0,
        case_id=case_id,
        field_name="summary.cluster_window_sec",
        detail=f"value={summary['cluster_window_sec']}",
    )
    _assert_true_with_preprocess_context(
        summary["soft_cut_window_sec"] > 0,
        case_id=case_id,
        field_name="summary.soft_cut_window_sec",
        detail=f"value={summary['soft_cut_window_sec']}",
    )
    _assert_equal_with_preprocess_context(
        summary["max_frames"],
        256,
        case_id=case_id,
        field_name="summary.max_frames",
    )
    _assert_true_with_preprocess_context(
        isinstance(summary["has_tail"], bool),
        case_id=case_id,
        field_name="summary.has_tail",
        detail=f"type={type(summary['has_tail']).__name__}",
    )

    reason_codes = {item.get("reason_code") for item in decision_report if isinstance(item, dict)}
    required_reason_codes = {
        "DECISION_SUMMARY",
        "FRAME_BUDGET_STATUS",
        "DEFAULT_FRAME_COUNT_DECISION",
        "UPSAMPLING_TRIGGER_STATUS",
        "FRAME_SELECTION_APPLIED",
        "SHOT_BOUNDARY_CLUSTERED",
        "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
        "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED",
        "SHOT_BOUNDARY_MERGE_STATUS",
        "TAIL_RULES_APPLIED",
        "TAIL_STATUS_RECORDED",
    }
    _assert_true_with_preprocess_context(
        required_reason_codes.issubset(reason_codes),
        case_id=case_id,
        field_name="reason_codes.required_subset",
        detail=f"missing={sorted(required_reason_codes - reason_codes)}",
    )
    _assert_true_with_preprocess_context(
        "SELECTED_CUT_POINTS" in reason_codes or "FALLBACK_UNIFORM" in reason_codes,
        case_id=case_id,
        field_name="reason_codes.cut_point_strategy",
        detail=f"reason_codes={sorted(code for code in reason_codes if code)}",
    )


@pytest.mark.integration
def test_decision_report_marks_dropped_after_real_ocr_rescoring() -> None:
    require_ocr_provider_or_skip(
        "user_skills/commerce-video-diagnosis/fixtures/raw_video_regression_config.json",
        reason_prefix="P_INT_03",
    )
    case_id = "P_INT_03"
    workspace = "output/test_preprocess_real_ocr_feedback"
    clip = _skip_if_clip_missing()
    run_preprocess(
        video_path=clip,
        workspace_dir=workspace,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        source_platform="抖音",
        enable_real_ocr_feedback=True,
        ocr_feedback_top_k=4,
    )
    payload = json.loads(Path(workspace, "preprocess.json").read_text(encoding="utf-8"))
    decision_report = payload.get("decision_report", [])

    summary_entry = next(item for item in decision_report if item.get("reason_code") == "DECISION_SUMMARY")
    _assert_true_with_preprocess_context(
        summary_entry["summary"]["ocr_feedback_enabled"] is True,
        case_id=case_id,
        field_name="summary.ocr_feedback_enabled",
        detail=f"value={summary_entry['summary']['ocr_feedback_enabled']}",
    )
    _assert_true_with_preprocess_context(
        summary_entry["summary"]["ocr_hit_count"] > 0,
        case_id=case_id,
        field_name="summary.ocr_hit_count",
        detail=f"value={summary_entry['summary']['ocr_hit_count']}",
    )

    dropped_after_rescoring = [
        item for item in decision_report if item.get("reason_code") == "DROPPED_AFTER_OCR_RESCORING"
    ]
    rescored_candidates = [
        item for item in payload.get("candidate_scores", []) if "OCR_FEEDBACK_RESCORING" in item.get("reason_codes", [])
    ]
    _assert_true_with_preprocess_context(
        bool(rescored_candidates),
        case_id=case_id,
        field_name="rescored_candidates",
        detail="expected OCR rescoring to mark at least one candidate",
    )
    if dropped_after_rescoring:
        _assert_true_with_preprocess_context(
            all("OCR_FEEDBACK_RESCORING" in item.get("reason_codes", []) for item in dropped_after_rescoring),
            case_id=case_id,
            field_name="dropped_after_rescoring.reason_codes",
            detail="all dropped_after_rescoring items must include OCR_FEEDBACK_RESCORING",
        )
        _assert_true_with_preprocess_context(
            all(isinstance(item.get("rank_after_ocr"), int) for item in dropped_after_rescoring),
            case_id=case_id,
            field_name="dropped_after_rescoring.rank_after_ocr",
            detail="rank_after_ocr must be int for all dropped items",
        )
    else:
        _assert_true_with_preprocess_context(
            any(item.get("selected_as_frame") for item in rescored_candidates),
            case_id=case_id,
            field_name="rescored_candidates.selected_as_frame",
            detail="expected at least one rescored candidate to remain selected",
        )


@pytest.mark.unit
def test_collect_raw_boundary_hits_uses_time_order_not_score_order() -> None:
    samples = [
        _make_sample(9.0, score=99.0, diff_score=90.0, hist_diff=0.95, text_density=0.01, text_boxes=1),
        _make_sample(3.0, score=10.0, diff_score=90.0, hist_diff=0.95, text_density=0.01, text_boxes=1),
        _make_sample(5.5, score=20.0, diff_score=90.0, hist_diff=0.95, text_density=0.01, text_boxes=1),
        _make_sample(0.0, score=1.0, diff_score=0.0, hist_diff=0.0, text_density=0.0, text_boxes=0),
    ]
    raw_hits, decisions, _ = _collect_raw_boundary_hits(samples, 60.0)

    assert [round(item["sec"], 3) for item in raw_hits] == [3.0, 5.5, 9.0]
    detected_secs = [item["sec"] for item in decisions if item.get("reason_code") == "SHOT_BOUNDARY_HARD_HIT_DETECTED"]
    assert detected_secs == [3.0, 5.5, 9.0]


@pytest.mark.unit
def test_boundary_clusters_consolidate_dense_hits_and_keep_one_representative() -> None:
    raw_hits = [
        {"sec": 10.0, "hist_diff": 0.60, "frame_diff": 70.0, "boundary_strength": 116.0, "score": 80.0},
        {"sec": 11.0, "hist_diff": 0.92, "frame_diff": 68.0, "boundary_strength": 146.4, "score": 60.0},
        {"sec": 12.0, "hist_diff": 0.70, "frame_diff": 88.0, "boundary_strength": 140.4, "score": 95.0},
        {"sec": 20.0, "hist_diff": 0.80, "frame_diff": 66.0, "boundary_strength": 132.8, "score": 70.0},
    ]
    clusters, cluster_decisions = _build_boundary_clusters(raw_hits, 2.5)
    representatives, rep_decisions = _select_cluster_representatives(clusters)

    assert len(clusters) == 2
    assert [item["sec"] for item in representatives] == [11.0, 20.0]
    assert any(item.get("reason_code") == "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED" for item in rep_decisions)
    assert len(cluster_decisions) == 2


@pytest.mark.unit
def test_boundary_soft_cut_recall_and_protected_representative() -> None:
    samples = [
        _make_sample(0.0, score=1.0, diff_score=0.0, hist_diff=0.0, text_density=0.0, text_boxes=0),
        _make_sample(1.0, score=8.0, diff_score=6.0, hist_diff=0.08, text_density=0.01, text_boxes=1),
        _make_sample(2.0, score=22.0, diff_score=14.0, hist_diff=0.40, text_density=0.08, text_boxes=8),
        _make_sample(3.0, score=7.0, diff_score=5.0, hist_diff=0.07, text_density=0.012, text_boxes=1),
    ]

    raw_hits, decisions, _ = _collect_raw_boundary_hits(samples, 10.0)
    clusters, _ = _build_boundary_clusters(raw_hits, 2.5)
    representatives, rep_decisions = _select_cluster_representatives(clusters)

    assert any(item["sec"] == 2.0 and item["hit_type"] == "soft_cut_hit" for item in raw_hits)
    assert any(item.get("reason_code") == "SHOT_BOUNDARY_SOFT_CUT_RECALLED" and item.get("sec") == 2.0 for item in decisions)
    assert representatives[0]["protected"] is True
    assert representatives[0]["sec"] == 2.0
    assert any(item.get("reason_code") == "SHOT_BOUNDARY_REP_PROTECTED" for item in rep_decisions)


@pytest.mark.unit
def test_merge_semantic_continuity_blocks_protected_representative() -> None:
    samples = [
        _make_sample(2.0, score=30.0, diff_score=10.0, hist_diff=0.15, text_density=0.03, text_boxes=5),
        _make_sample(6.0, score=32.0, diff_score=11.0, hist_diff=0.16, text_density=0.031, text_boxes=5),
        _make_sample(10.0, score=28.0, diff_score=12.0, hist_diff=0.42, text_density=0.08, text_boxes=8),
        _make_sample(12.0, score=29.0, diff_score=10.5, hist_diff=0.14, text_density=0.029, text_boxes=6),
        _make_sample(17.0, score=31.0, diff_score=11.5, hist_diff=0.17, text_density=0.032, text_boxes=5),
    ]
    protected_boundary = {
        "sec": 10.0,
        "cluster_id": "CLUSTER01",
        "hit_type": "soft_cut_hit",
        "trigger_signals": ["local_peak_near_hard_threshold", "ocr_structure_jump"],
        "protected": True,
        "protected_reason": "soft_cut_structural_peak_protected",
    }

    boundaries, decisions, merge_count = _merge_semantic_continuity(
        [0.0, 10.0, 20.0],
        samples,
        20.0,
        boundary_representatives=[protected_boundary],
    )

    assert merge_count == 0
    assert boundaries == [0.0, 10.0, 20.0]
    blocked = [item for item in decisions if item.get("reason_code") == "SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP"]
    assert blocked
    assert blocked[0]["protected_representative_sec"] == 10.0


@pytest.mark.unit
def test_merge_semantic_continuity_merges_adjacent_similar_segments() -> None:
    samples = [
        _make_sample(2.0, score=30.0, diff_score=10.0, hist_diff=0.15, text_density=0.03, text_boxes=5),
        _make_sample(6.0, score=32.0, diff_score=11.0, hist_diff=0.16, text_density=0.031, text_boxes=5),
        _make_sample(12.0, score=29.0, diff_score=10.5, hist_diff=0.14, text_density=0.029, text_boxes=6),
        _make_sample(17.0, score=31.0, diff_score=11.5, hist_diff=0.17, text_density=0.032, text_boxes=5),
        _make_sample(24.0, score=50.0, diff_score=35.0, hist_diff=0.50, text_density=0.12, text_boxes=14),
        _make_sample(28.0, score=48.0, diff_score=38.0, hist_diff=0.52, text_density=0.13, text_boxes=15),
    ]
    boundaries, decisions, merge_count = _merge_semantic_continuity([0.0, 10.0, 20.0, 30.0], samples, 30.0)

    assert merge_count == 1
    assert boundaries == [0.0, 20.0, 30.0]
    assert any(item.get("reason_code") == "SHOT_BOUNDARY_MERGE_PASSED_CONTINUITY_GATE" for item in decisions)
    assert any(item.get("reason_code") == "SHOT_BOUNDARY_MERGED_BY_SEMANTIC_CONTINUITY" for item in decisions)


@pytest.mark.unit
def test_merge_semantic_continuity_blocks_result_page_and_strong_boundary_peak() -> None:
    samples = [
        _make_sample(2.0, score=18.0, diff_score=6.0, hist_diff=0.08, text_density=0.01, text_boxes=1),
        _make_sample(8.0, score=20.0, diff_score=7.0, hist_diff=0.09, text_density=0.012, text_boxes=1),
        _make_sample(10.0, score=95.0, diff_score=72.0, hist_diff=0.93, text_density=0.02, text_boxes=2),
        _make_sample(14.0, score=40.0, diff_score=4.0, hist_diff=0.06, text_density=0.14, text_boxes=12),
        _make_sample(18.0, score=42.0, diff_score=5.0, hist_diff=0.07, text_density=0.13, text_boxes=11),
    ]
    boundaries, decisions, merge_count = _merge_semantic_continuity([0.0, 10.0, 20.0], samples, 20.0)

    assert merge_count == 0
    assert boundaries == [0.0, 10.0, 20.0]
    blocked = [item for item in decisions if item.get("reason_code") == "SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE"]
    assert blocked
    assert {"strong_boundary_peak", "right_segment_result_or_cta_page"}.issubset(set(blocked[0]["block_reasons"]))


@pytest.mark.unit
def test_merge_semantic_continuity_applies_chain_brake_after_one_anchor_merge() -> None:
    samples = [
        _make_sample(2.0, score=20.0, diff_score=9.0, hist_diff=0.10, text_density=0.02, text_boxes=3),
        _make_sample(8.0, score=21.0, diff_score=9.5, hist_diff=0.11, text_density=0.021, text_boxes=3),
        _make_sample(12.0, score=22.0, diff_score=9.0, hist_diff=0.10, text_density=0.022, text_boxes=3),
        _make_sample(18.0, score=22.5, diff_score=9.2, hist_diff=0.11, text_density=0.021, text_boxes=3),
        _make_sample(22.0, score=23.0, diff_score=8.8, hist_diff=0.09, text_density=0.02, text_boxes=3),
        _make_sample(28.0, score=23.5, diff_score=9.1, hist_diff=0.1, text_density=0.021, text_boxes=3),
    ]
    boundaries, decisions, merge_count = _merge_semantic_continuity([0.0, 10.0, 20.0, 30.0], samples, 30.0)

    assert merge_count == 1
    assert boundaries == [0.0, 20.0, 30.0]
    blocked = [item for item in decisions if item.get("reason_code") == "SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE"]
    assert blocked
    assert "chain_merge_limit_reached" in blocked[0]["block_reasons"]


@pytest.mark.unit
def test_annotate_segment_types_marks_tail_for_long_video_tail_window() -> None:
    segments = [
        _build_segment("SEG01", 0.0, 20.0),
        _build_segment("SEG02", 20.0, 36.0),
        _build_segment("SEG03", 36.0, 42.0),
    ]
    samples = [
        {"sec": 5.0, "text_density": 0.01, "text_boxes": 1, "diff_score": 24.0, "hist_diff": 0.33, "reason_codes": []},
        {"sec": 33.0, "text_density": 0.02, "text_boxes": 2, "diff_score": 18.0, "hist_diff": 0.22, "reason_codes": []},
        {"sec": 38.0, "text_density": 0.12, "text_boxes": 10, "diff_score": 6.0, "hist_diff": 0.08, "reason_codes": ["TAIL_WINDOW_HIT", "TAIL_TEXT_DENSITY_HIT"]},
    ]

    annotated, decisions = _annotate_segment_types(segments, samples, 42.0)

    assert [seg["segment_type"] for seg in annotated] == ["main", "main", "tail"]
    tail_status = next(item for item in decisions if item.get("reason_code") == "TAIL_STATUS_RECORDED")
    assert tail_status["has_tail"] is True
    assert tail_status["tail_segment_ids"] == ["SEG03"]


@pytest.mark.unit
def test_determine_frame_plan_short_stable_segment_uses_two_frames() -> None:
    samples = [
        _make_sample(0.5, score=10.0, diff_score=4.0, hist_diff=0.08, text_density=0.005, text_boxes=1),
        _make_sample(1.0, score=12.0, diff_score=5.0, hist_diff=0.09, text_density=0.006, text_boxes=1),
        _make_sample(1.5, score=11.0, diff_score=4.5, hist_diff=0.07, text_density=0.005, text_boxes=1),
    ]
    plan = _determine_frame_plan(samples, 0.0, 2.0)
    frames = _build_segment_frames("SEG01", samples, 0.0, 2.0, plan)

    assert plan["default_frame_count"] == 2
    assert plan["final_frame_count"] == 2
    assert [frame["sampling_role"] for frame in frames][0] == "middle"
    assert len(frames) == 2


@pytest.mark.unit
def test_determine_frame_plan_complex_segment_upsamples_to_four_or_five_frames() -> None:
    samples = [
        _make_sample(0.5, score=12.0, diff_score=8.0, hist_diff=0.12, text_density=0.01, text_boxes=1),
        _make_sample(1.0, score=18.0, diff_score=26.0, hist_diff=0.62, text_density=0.02, text_boxes=2),
        _make_sample(1.5, score=16.0, diff_score=14.0, hist_diff=0.22, text_density=0.07, text_boxes=9),
        _make_sample(2.0, score=22.0, diff_score=28.0, hist_diff=0.58, text_density=0.03, text_boxes=3),
        _make_sample(2.5, score=17.0, diff_score=10.0, hist_diff=0.18, text_density=0.08, text_boxes=10),
        _make_sample(3.0, score=19.0, diff_score=12.0, hist_diff=0.2, text_density=0.02, text_boxes=2),
        _make_sample(3.5, score=23.0, diff_score=20.0, hist_diff=0.25, text_density=0.09, text_boxes=11),
    ]
    plan = _determine_frame_plan(samples, 0.0, 4.0)
    frames = _build_segment_frames("SEG02", samples, 0.0, 4.0, plan)
    roles = {frame["sampling_role"] for frame in frames}

    assert plan["final_frame_count"] in {4, 5}
    assert len(frames) == plan["final_frame_count"]
    assert {"front", "middle", "back"}.issubset(roles)
    assert roles & {"dense_text", "action_peak", "result_state"}


@pytest.mark.unit
def test_apply_frame_budget_recovers_simple_segments_first() -> None:
    simple_a = _build_segment("SEG01", 0.0, 2.0, frame_count=2)
    simple_b = _build_segment("SEG02", 2.0, 4.0, frame_count=2)
    complex_seg = _build_segment("SEG03", 4.0, 8.0, frame_count=5)
    complex_seg["frame_plan"]["default_frame_count"] = 3
    complex_seg["frame_plan"]["min_keep_frames"] = 3
    complex_seg["frame_plan"]["upsampling_triggers"] = ["TEXT_CHANGE_DENSE", "ACTION_STATE_CHANGE"]
    complex_seg["frame_plan"]["metrics"]["avg_text_density"] = 0.09
    complex_seg["frame_plan"]["metrics"]["avg_diff_score"] = 22.0

    segments, decisions = _apply_frame_budget([simple_a, simple_b, complex_seg], max_frames=6)
    budget_status = next(item for item in decisions if item.get("reason_code") == "FRAME_BUDGET_STATUS")
    recovery_actions = [item for item in decisions if item.get("reason_code") == "BUDGET_RECOVERY_ACTION"]

    assert budget_status["total_frames_before_budget"] == 9
    assert budget_status["total_frames_after_budget"] == 6
    assert recovery_actions
    by_segment = {segment["segment_id"]: segment for segment in segments}
    assert len(by_segment["SEG03"]["frames"]) >= 3
    assert len(by_segment["SEG01"]["frames"]) == 1
    assert len(by_segment["SEG02"]["frames"]) == 1


@pytest.mark.unit
def test_assert_preprocess_output_allows_long_video_has_tail_false() -> None:
    segments = [
        _build_segment("SEG01", 0.0, 18.0, "main"),
        _build_segment("SEG02", 18.0, 36.0, "main"),
    ]
    payload = _build_preprocess_payload(duration_sec=36.0, segments=segments, has_tail=False)

    assert_preprocess_output(payload)


@pytest.mark.unit
def test_assert_preprocess_output_blocks_missing_sampling_role() -> None:
    segments = [
        _build_segment("SEG01", 0.0, 18.0, "main"),
        _build_segment("SEG02", 18.0, 36.0, "tail"),
    ]
    del segments[1]["frames"][0]["sampling_role"]
    payload = _build_preprocess_payload(duration_sec=36.0, segments=segments, has_tail=True)

    with pytest.raises(PreprocessViolation, match="sampling_role"):
        assert_preprocess_output(payload)


@pytest.mark.unit
def test_assert_preprocess_output_blocks_missing_budget_status() -> None:
    segments = [
        _build_segment("SEG01", 0.0, 18.0, "main"),
        _build_segment("SEG02", 18.0, 36.0, "tail"),
    ]
    payload = _build_preprocess_payload(duration_sec=36.0, segments=segments, has_tail=True)
    payload["decision_report"] = [
        item for item in payload["decision_report"] if item.get("reason_code") != "FRAME_BUDGET_STATUS"
    ]

    with pytest.raises(PreprocessViolation, match="FRAME_BUDGET_STATUS"):
        assert_preprocess_output(payload)
