from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def has_reason_code(entries: list[dict], reason_code: str) -> bool:
    return any(item.get("reason_code") == reason_code for item in entries)


def assert_full_smoke_workspace(workspace: Path) -> None:
    combined_path = workspace / "combined_smoke_result.json"
    preprocess_path = workspace / "runtime" / "preprocess.json"
    decision_path = workspace / "runtime" / "decision_report.json"
    factpack_path = workspace / "factpack.json"
    result_path = workspace / "result.json"

    combined = read_json(combined_path)
    preprocess = read_json(preprocess_path)
    decision_payload = read_json(decision_path)
    factpack = read_json(factpack_path)
    result = read_json(result_path)

    smoke_result = combined.get("smoke_result")
    assert isinstance(smoke_result, dict), "combined output must contain smoke_result"
    assert smoke_result.get("status") == "two_stage_done", "smoke_result.status must be two_stage_done"

    smoke_blueprint = smoke_result.get("result", {}).get("blueprint", {})
    assert smoke_blueprint.get("video_id"), "combined smoke_result blueprint.video_id missing"
    assert smoke_blueprint.get("source_product_id"), "combined smoke_result blueprint.source_product_id missing"
    storyboard_segments = smoke_blueprint.get("storyboard_segments") or []
    assert storyboard_segments, "combined smoke_result blueprint.storyboard_segments must be non-empty"

    regression = combined.get("ocr_feedback_regression")
    assert isinstance(regression, dict), "combined output must contain ocr_feedback_regression"
    assert regression.get("status") == "two_stage_done", "ocr_feedback_regression.status must be two_stage_done"
    assert regression.get("ocr_feedback_enabled") is True, "ocr_feedback_enabled must be true"
    assert regression.get("ocr_hit_count", 0) > 0, "ocr_hit_count must be > 0"
    assert regression.get("dropped_after_ocr_rescoring_count", 0) >= 0, "dropped_after_ocr_rescoring_count must be >= 0"
    assert regression.get("decision_report_synced") is True, "decision_report_synced must be true"

    preprocess_segments = preprocess.get("segments") or []
    preprocess_frame_paths = preprocess.get("frame_paths") or []
    preprocess_decision_report = preprocess.get("decision_report") or []
    assert preprocess.get("video_path"), "preprocess.video_path missing"
    assert preprocess.get("audio_path"), "preprocess.audio_path missing"
    assert preprocess_segments, "preprocess.segments must be non-empty"
    assert preprocess_frame_paths, "preprocess.frame_paths must be non-empty"
    assert preprocess_decision_report, "preprocess.decision_report must be non-empty"
    assert isinstance(preprocess.get("has_tail"), bool), "preprocess.has_tail must be bool"
    assert has_reason_code(preprocess_decision_report, "DECISION_SUMMARY"), "preprocess decision_report must contain DECISION_SUMMARY"
    assert has_reason_code(preprocess_decision_report, "SELECTED_CUT_POINTS") or has_reason_code(preprocess_decision_report, "FALLBACK_UNIFORM"), "preprocess decision_report must contain shot selection or fallback"
    assert has_reason_code(preprocess_decision_report, "FRAME_BUDGET_STATUS"), "preprocess decision_report must contain FRAME_BUDGET_STATUS"
    assert has_reason_code(preprocess_decision_report, "DEFAULT_FRAME_COUNT_DECISION"), "preprocess decision_report must contain DEFAULT_FRAME_COUNT_DECISION"
    assert has_reason_code(preprocess_decision_report, "UPSAMPLING_TRIGGER_STATUS"), "preprocess decision_report must contain UPSAMPLING_TRIGGER_STATUS"
    assert has_reason_code(preprocess_decision_report, "FRAME_SELECTION_APPLIED"), "preprocess decision_report must contain FRAME_SELECTION_APPLIED"
    assert has_reason_code(preprocess_decision_report, "SHOT_BOUNDARY_CLUSTERED"), "preprocess decision_report must contain SHOT_BOUNDARY_CLUSTERED"
    assert has_reason_code(preprocess_decision_report, "SHOT_BOUNDARY_CLUSTER_REP_SELECTED"), "preprocess decision_report must contain SHOT_BOUNDARY_CLUSTER_REP_SELECTED"
    assert has_reason_code(preprocess_decision_report, "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED"), "preprocess decision_report must contain SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED"
    assert has_reason_code(preprocess_decision_report, "SHOT_BOUNDARY_MERGE_STATUS"), "preprocess decision_report must contain SHOT_BOUNDARY_MERGE_STATUS"
    assert has_reason_code(preprocess_decision_report, "TAIL_STATUS_RECORDED"), "preprocess decision_report must contain tail status"
    assert all(seg.get("shot_id") for seg in preprocess_segments), "preprocess.segments[*].shot_id must be present"
    assert all(seg.get("segment_strategy") for seg in preprocess_segments), "preprocess.segments[*].segment_strategy must be present"
    assert all(seg.get("segment_type") in {"main", "tail"} for seg in preprocess_segments), "preprocess.segments[*].segment_type must be main/tail"
    assert all(isinstance(seg.get("frames"), list) and seg.get("frames") for seg in preprocess_segments), "preprocess.segments[*].frames must be non-empty"
    assert all(all(frame.get("sampling_role") for frame in seg.get("frames") or []) for seg in preprocess_segments), "preprocess.segments[*].frames[*].sampling_role must be present"
    assert preprocess.get("has_tail") == any(seg.get("segment_type") == "tail" for seg in preprocess_segments), "preprocess.has_tail must match preprocess.segments[*].segment_type"

    decision_report = decision_payload.get("decision_report") or []
    candidate_scores = decision_payload.get("candidate_scores") or []
    assert decision_report, "decision_report.json decision_report must be non-empty"
    assert candidate_scores or has_reason_code(decision_report, "FALLBACK_UNIFORM"), "decision_report.json candidate_scores must be non-empty unless fallback"
    assert decision_report == preprocess_decision_report, "decision_report.json must stay synced with preprocess.decision_report"
    assert has_reason_code(decision_report, "DECISION_SUMMARY"), "decision_report.json must contain DECISION_SUMMARY"

    factpack_segments = factpack.get("segments") or []
    factpack_meta = factpack.get("video_meta") or {}
    segments = factpack.get("segments") or []
    segment_to_bundle_map = factpack.get("segment_to_bundle_map") or {}
    bundle_to_segment_range = factpack.get("bundle_to_segment_range") or {}
    assert factpack_segments, "factpack.segments must be non-empty"
    assert segments, "factpack.segments must be non-empty"
    assert segment_to_bundle_map, "factpack.segment_to_bundle_map must be non-empty"
    assert bundle_to_segment_range, "factpack.bundle_to_segment_range must be non-empty"
    assert factpack.get("storyboard_source") == "segments", "factpack.storyboard_source must be segments"
    assert factpack_meta.get("source_platform"), "factpack.video_meta.source_platform missing"
    assert factpack_meta.get("duration_sec") is not None, "factpack.video_meta.duration_sec missing"

    result_blueprint = result.get("blueprint") or {}
    result_segments = result_blueprint.get("storyboard_segments") or []
    workflow_report = result.get("workflow_report") or {}
    assert result_blueprint.get("video_id"), "result blueprint.video_id missing"
    assert result_blueprint.get("source_product_id"), "result blueprint.source_product_id missing"
    assert result_blueprint.get("storyboard_source") == "segments", "result blueprint.storyboard_source must be segments"
    assert result_blueprint.get("segments"), "result blueprint.segments must be non-empty"
    assert result_segments, "result blueprint.storyboard_segments must be non-empty"
    assert workflow_report.get("video_id"), "result workflow_report.video_id missing"
    assert workflow_report.get("source_product_id"), "result workflow_report.source_product_id missing"
