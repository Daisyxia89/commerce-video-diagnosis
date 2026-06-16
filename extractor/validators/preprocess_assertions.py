from __future__ import annotations

from ..errors import PreprocessViolation


LONG_VIDEO_TAIL_REQUIRED_SEC = 30.0
ALLOWED_SEGMENT_TYPES = {"main", "tail"}
ALLOWED_SAMPLING_ROLES = {"front", "middle", "back", "dense_text", "action_peak", "result_state"}
MAX_FRAMES_CAP = 256
REQUIRED_REASON_CODES = {
    "DECISION_SUMMARY",
    "TAIL_STATUS_RECORDED",
    "FRAME_BUDGET_STATUS",
    "DEFAULT_FRAME_COUNT_DECISION",
    "UPSAMPLING_TRIGGER_STATUS",
    "FRAME_SELECTION_APPLIED",
    "SHOT_BOUNDARY_CLUSTERED",
    "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
    "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED",
    "SHOT_BOUNDARY_MERGE_STATUS",
}


def assert_preprocess_output(preproc: dict) -> None:
    if not preproc.get("audio_path"):
        raise PreprocessViolation("preprocess 缺少 audio_path")
    frame_paths = preproc.get("frame_paths") or []
    if not frame_paths:
        raise PreprocessViolation("preprocess 缺少 frame_paths")
    frame_seconds = preproc.get("frame_seconds") or []
    if len(frame_paths) != len(frame_seconds):
        raise PreprocessViolation("preprocess.frame_paths 与 preprocess.frame_seconds 长度不一致")
    if len(frame_paths) > MAX_FRAMES_CAP:
        raise PreprocessViolation("preprocess.frame_paths 超出 max_frames=256 预算上限")
    segments = preproc.get("segments") or []
    if not segments:
        raise PreprocessViolation("preprocess 缺少 segments")
    if "has_tail" not in preproc or not isinstance(preproc.get("has_tail"), bool):
        raise PreprocessViolation("preprocess 缺少 has_tail 或 has_tail 不是 bool")

    video_meta = preproc.get("video_meta") or {}
    for key in ("source_platform", "duration_sec", "fps", "resolution"):
        if key not in video_meta:
            raise PreprocessViolation(f"preprocess.video_meta 缺少 {key}")
    duration_sec = float(video_meta.get("duration_sec") or 0.0)

    decision_report = preproc.get("decision_report") or []
    if not decision_report:
        raise PreprocessViolation("preprocess 缺少 decision_report")
    reason_codes = {item.get("reason_code") for item in decision_report if isinstance(item, dict)}
    if "DECISION_SUMMARY" not in reason_codes:
        raise PreprocessViolation("preprocess.decision_report 缺少 DECISION_SUMMARY")
    if not ({"SELECTED_CUT_POINTS", "FALLBACK_UNIFORM"} & reason_codes):
        raise PreprocessViolation("preprocess.decision_report 缺少 Shot 决策结果")
    if "TAIL_STATUS_RECORDED" not in reason_codes:
        raise PreprocessViolation("preprocess.decision_report 缺少尾部分离状态")
    missing_reason_codes = REQUIRED_REASON_CODES - reason_codes
    if missing_reason_codes:
        missing = ", ".join(sorted(missing_reason_codes))
        raise PreprocessViolation(f"preprocess.decision_report 缺少关键决策节点: {missing}")

    summary_entry = next(
        (item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "DECISION_SUMMARY"),
        None,
    )
    summary = (summary_entry or {}).get("summary") or {}
    for key in (
        "shot_count",
        "final_segment_count",
        "tail_segment_count",
        "total_candidates",
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
        "cluster_window_sec",
        "soft_cut_window_sec",
    ):
        if key not in summary:
            raise PreprocessViolation(f"preprocess.decision_report.DECISION_SUMMARY.summary 缺少 {key}")
    if int(summary.get("max_frames") or 0) != MAX_FRAMES_CAP:
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.max_frames 必须为 256")
    if int(summary.get("total_selected_frames") or 0) > MAX_FRAMES_CAP:
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.total_selected_frames 超出预算")
    if int(summary.get("hard_boundary_hit_count") or 0) + int(summary.get("soft_cut_recall_count") or 0) != int(summary.get("raw_boundary_hit_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.hard_boundary_hit_count + soft_cut_recall_count 必须等于 raw_boundary_hit_count")
    if int(summary.get("boundary_cluster_count") or 0) > int(summary.get("raw_boundary_hit_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.boundary_cluster_count 不能大于 raw_boundary_hit_count")
    if int(summary.get("protected_representative_count") or 0) > int(summary.get("representative_boundary_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.protected_representative_count 不能大于 representative_boundary_count")
    if int(summary.get("representative_boundary_count") or 0) > int(summary.get("boundary_cluster_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.representative_boundary_count 不能大于 boundary_cluster_count")
    if int(summary.get("final_cut_point_count") or 0) > int(summary.get("representative_boundary_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.final_cut_point_count 不能大于 representative_boundary_count")

    if int(summary.get("raw_boundary_hit_count") or 0) > 0 and not ({"SHOT_BOUNDARY_HARD_HIT_DETECTED", "SHOT_BOUNDARY_SOFT_CUT_RECALLED"} & reason_codes):
        raise PreprocessViolation("preprocess.decision_report 缺少边界命中记录")
    if int(summary.get("hard_boundary_hit_count") or 0) > 0 and "SHOT_BOUNDARY_HARD_HIT_DETECTED" not in reason_codes:
        raise PreprocessViolation("preprocess.decision_report 缺少 SHOT_BOUNDARY_HARD_HIT_DETECTED")
    if int(summary.get("soft_cut_recall_count") or 0) > 0 and "SHOT_BOUNDARY_SOFT_CUT_RECALLED" not in reason_codes:
        raise PreprocessViolation("preprocess.decision_report 缺少 SHOT_BOUNDARY_SOFT_CUT_RECALLED")

    budget_status = next(
        (item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "FRAME_BUDGET_STATUS"),
        None,
    )
    if budget_status is None:
        raise PreprocessViolation("preprocess.decision_report 缺少 FRAME_BUDGET_STATUS")
    if int(budget_status.get("max_frames") or 0) != MAX_FRAMES_CAP:
        raise PreprocessViolation("preprocess.decision_report.FRAME_BUDGET_STATUS.max_frames 必须为 256")
    if int(budget_status.get("total_frames_after_budget") or 0) > MAX_FRAMES_CAP:
        raise PreprocessViolation("preprocess.decision_report.FRAME_BUDGET_STATUS.total_frames_after_budget 超出预算")

    selection_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "FRAME_SELECTION_APPLIED"
    ]
    if len(selection_entries) != len(segments):
        raise PreprocessViolation("preprocess.decision_report.FRAME_SELECTION_APPLIED 数量必须与 segments 一致")

    default_frame_decisions = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "DEFAULT_FRAME_COUNT_DECISION"
    ]
    if len(default_frame_decisions) != len(segments):
        raise PreprocessViolation("preprocess.decision_report.DEFAULT_FRAME_COUNT_DECISION 数量必须与 segments 一致")

    trigger_status_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "UPSAMPLING_TRIGGER_STATUS"
    ]
    if len(trigger_status_entries) != len(segments):
        raise PreprocessViolation("preprocess.decision_report.UPSAMPLING_TRIGGER_STATUS 数量必须与 segments 一致")

    clustered_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_CLUSTERED"
    ]
    rep_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_CLUSTER_REP_SELECTED"
    ]
    dropped_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED"
    ]
    protected_rep_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_REP_PROTECTED"
    ]
    merge_status = next(
        (item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_MERGE_STATUS"),
        None,
    )
    if merge_status is None:
        raise PreprocessViolation("preprocess.decision_report 缺少 SHOT_BOUNDARY_MERGE_STATUS")
    if len(clustered_entries) != int(summary.get("boundary_cluster_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTERED 数量必须与 summary.boundary_cluster_count 一致")
    if len(rep_entries) != int(summary.get("representative_boundary_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTER_REP_SELECTED 数量必须与 summary.representative_boundary_count 一致")
    if protected_rep_entries and int(summary.get("protected_representative_count") or 0) != len(protected_rep_entries):
        raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_REP_PROTECTED 数量必须与 summary.protected_representative_count 一致")
    rep_by_cluster = {item.get("cluster_id"): item for item in rep_entries}
    dropped_by_cluster = {item.get("cluster_id"): item for item in dropped_entries}
    for cluster in clustered_entries:
        cluster_id = cluster.get("cluster_id")
        member_secs = cluster.get("member_secs") or []
        if not member_secs:
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTERED.member_secs 不能为空")
        member_types = cluster.get("member_types") or []
        if member_types and len(member_types) != len(member_secs):
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTERED.member_types 必须与 member_secs 一致")
        protected_candidate_secs = [round(float(sec), 3) for sec in (cluster.get("protected_candidate_secs") or [])]
        if any(sec not in [round(float(item), 3) for item in member_secs] for sec in protected_candidate_secs):
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTERED.protected_candidate_secs 必须属于 member_secs")
        rep = rep_by_cluster.get(cluster_id)
        if rep is None:
            raise PreprocessViolation("preprocess.decision_report 缺少 cluster 对应的代表边界记录")
        representative_sec = round(float(rep.get("representative_sec") or 0.0), 3)
        if representative_sec not in [round(float(sec), 3) for sec in member_secs]:
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTER_REP_SELECTED.representative_sec 必须属于 member_secs")
        protected_representative_sec = rep.get("protected_representative_sec")
        if protected_representative_sec is not None and round(float(protected_representative_sec), 3) not in [round(float(sec), 3) for sec in member_secs]:
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTER_REP_SELECTED.protected_representative_sec 必须属于 member_secs")
        dropped = dropped_by_cluster.get(cluster_id)
        expected_dropped = max(len(member_secs) - 1, 0)
        actual_dropped = len((dropped or {}).get("dropped_secs") or [])
        if expected_dropped != actual_dropped:
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED 与 cluster 成员数不一致")

    merge_entries = [
        item
        for item in decision_report
        if isinstance(item, dict)
        and item.get("reason_code") in {"SHOT_BOUNDARY_MERGED_BY_SEMANTIC_CONTINUITY", "SHOT_BOUNDARY_MERGED_BY_VISUAL_SIMILARITY"}
    ]
    blocked_merge_entries = [
        item
        for item in decision_report
        if isinstance(item, dict)
        and item.get("reason_code") in {"SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE", "SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP"}
    ]
    passed_gate_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_MERGE_PASSED_CONTINUITY_GATE"
    ]
    tie_breaker_entries = [
        item for item in decision_report if isinstance(item, dict) and item.get("reason_code") == "SHOT_BOUNDARY_MERGE_TIE_BREAKER_APPLIED"
    ]
    if int(merge_status.get("semantic_merge_count") or 0) != len(merge_entries):
        raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_STATUS.semantic_merge_count 必须与实际 merge 记录数量一致")
    if int(merge_status.get("merge_actions") or 0) != len(merge_entries):
        raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_STATUS.merge_actions 必须与实际 merge 记录数量一致")
    for entry in merge_entries + blocked_merge_entries + passed_gate_entries + tie_breaker_entries:
        for key in ("left_shot_id", "right_shot_id", "shared_boundary_sec", "evidence_summary"):
            if key not in entry:
                raise PreprocessViolation(f"preprocess.decision_report.{entry.get('reason_code')} 缺少 {key}")
    for entry in blocked_merge_entries:
        if entry.get("reason_code") == "SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE" and not (entry.get("block_reasons") or []):
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE.block_reasons 不能为空")
        if entry.get("reason_code") == "SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP":
            if "protected_representative_sec" not in entry or "block_reason" not in entry:
                raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP 缺少 protected 字段")
    for entry in passed_gate_entries:
        if not (entry.get("continuity_signals") or []):
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_PASSED_CONTINUITY_GATE.continuity_signals 不能为空")
    for entry in tie_breaker_entries:
        if "tie_breaker_reason" not in entry or "tie_breaker_score" not in entry:
            raise PreprocessViolation("preprocess.decision_report.SHOT_BOUNDARY_MERGE_TIE_BREAKER_APPLIED 缺少 tie_breaker 字段")

    protected_rep_secs = [
        round(float(item.get("protected_representative_sec")), 3)
        for item in rep_entries
        if item.get("protected_representative_sec") is not None
    ]
    if len(protected_rep_secs) != int(summary.get("protected_representative_count") or 0):
        raise PreprocessViolation("preprocess.decision_report.DECISION_SUMMARY.summary.protected_representative_count 与代表点记录不一致")

    selected_cut_points_entry = next(
        (item for item in decision_report if isinstance(item, dict) and item.get("reason_code") in {"SELECTED_CUT_POINTS", "FALLBACK_UNIFORM"}),
        None,
    )
    if selected_cut_points_entry is None:
        raise PreprocessViolation("preprocess.decision_report 缺少最终 cut point 记录")
    boundaries_after_merge = selected_cut_points_entry.get("boundaries_after_merge") or []
    if boundaries_after_merge:
        rounded_boundaries = [round(float(x), 3) for x in boundaries_after_merge]
        if rounded_boundaries[0] != 0.0:
            raise PreprocessViolation("preprocess.decision_report.boundaries_after_merge 首边界必须为 0.0")
        if rounded_boundaries[-1] != round(duration_sec, 3):
            raise PreprocessViolation("preprocess.decision_report.boundaries_after_merge 末边界必须等于 duration_sec")
        if rounded_boundaries != sorted(rounded_boundaries):
            raise PreprocessViolation("preprocess.decision_report.boundaries_after_merge 必须严格递增")
        if len(rounded_boundaries) - 1 != len(segments):
            raise PreprocessViolation("preprocess.decision_report.boundaries_after_merge 与 segments 数量不一致")
        final_cut_points = set(rounded_boundaries[1:-1])
        for sec in protected_rep_secs:
            if sec in final_cut_points:
                continue
            has_override = any(
                isinstance(item, dict)
                and item.get("reason_code") in {"SHOT_BOUNDARY_MERGED_BY_SEMANTIC_CONTINUITY", "SHOT_BOUNDARY_MERGED_BY_VISUAL_SIMILARITY"}
                and round(float(item.get("shared_boundary_sec") or -1.0), 3) == sec
                for item in decision_report
            )
            if not has_override:
                raise PreprocessViolation("protected_representative_sec 未进入最终 cut points 且缺少明确 override 记录")

    flattened_frame_paths: list[str] = []
    flattened_frame_seconds: list[float] = []
    last_start = -1.0
    tail_started = False
    tail_count = 0
    for seg in segments:
        for key in ("segment_id", "shot_id", "start_sec", "end_sec", "frame_path", "frame_second", "segment_strategy", "segment_type", "frames", "frame_plan"):
            if key not in seg:
                raise PreprocessViolation(f"preprocess.segments[*] 缺少 {key}")
        start_sec = seg.get("start_sec")
        end_sec = seg.get("end_sec")
        if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)):
            raise PreprocessViolation("preprocess.segments[*] start_sec/end_sec 必须为数值")
        if float(start_sec) >= float(end_sec):
            raise PreprocessViolation("preprocess.segments[*] 要求 start_sec < end_sec")
        if float(start_sec) < last_start:
            raise PreprocessViolation("preprocess.segments[*] 时间轴必须按 start_sec 递增")
        if float(end_sec) > duration_sec + 1e-6:
            raise PreprocessViolation("preprocess.segments[*].end_sec 超出视频时长")
        last_start = float(start_sec)

        segment_type = seg.get("segment_type")
        if segment_type not in ALLOWED_SEGMENT_TYPES:
            raise PreprocessViolation("preprocess.segments[*].segment_type 仅允许 main/tail")
        if segment_type == "tail":
            tail_started = True
            tail_count += 1
        elif tail_started:
            raise PreprocessViolation("preprocess.segments[*] 不允许在 tail 后重新出现 main")

        frame_plan = seg.get("frame_plan") or {}
        for key in ("default_frame_count", "final_frame_count", "final_frame_count_before_budget", "upsampling_triggers", "min_keep_frames", "metrics"):
            if key not in frame_plan:
                raise PreprocessViolation(f"preprocess.segments[*].frame_plan 缺少 {key}")

        frames = seg.get("frames") or []
        if not isinstance(frames, list) or not frames:
            raise PreprocessViolation("preprocess.segments[*].frames 不能为空")
        if int(frame_plan.get("final_frame_count") or 0) != len(frames):
            raise PreprocessViolation("preprocess.segments[*].frame_plan.final_frame_count 必须与 frames 长度一致")

        primary_frame_path = str(seg.get("frame_path") or "")
        primary_frame_second = float(seg.get("frame_second") or 0.0)
        primary_hit = False
        for frame in frames:
            for key in ("frame_id", "frame_second", "sampling_role", "frame_path"):
                if key not in frame:
                    raise PreprocessViolation(f"preprocess.segments[*].frames[*] 缺少 {key}")
            role = str(frame.get("sampling_role") or "")
            if role not in ALLOWED_SAMPLING_ROLES:
                raise PreprocessViolation("preprocess.segments[*].frames[*].sampling_role 非法")
            frame_second = frame.get("frame_second")
            if not isinstance(frame_second, (int, float)):
                raise PreprocessViolation("preprocess.segments[*].frames[*].frame_second 必须为数值")
            if not (float(start_sec) <= float(frame_second) <= float(end_sec) + 1e-6):
                raise PreprocessViolation("preprocess.segments[*].frames[*].frame_second 超出 segment 时间窗")
            frame_path = str(frame.get("frame_path") or "")
            if not frame_path:
                raise PreprocessViolation("preprocess.segments[*].frames[*].frame_path 不能为空")
            flattened_frame_paths.append(frame_path)
            flattened_frame_seconds.append(float(frame_second))
            if frame_path == primary_frame_path and abs(float(frame_second) - primary_frame_second) < 1e-6:
                primary_hit = True
        if not primary_hit:
            raise PreprocessViolation("preprocess.segments[*].frame_path/frame_second 必须命中 frames[*]")

    has_tail = bool(preproc.get("has_tail"))
    if has_tail != (tail_count > 0):
        raise PreprocessViolation("preprocess.has_tail 与 segments[*].segment_type 不一致")
    if tail_count >= len(segments):
        raise PreprocessViolation("preprocess.segments[*] 至少需要保留一个 main 段")
    if duration_sec > LONG_VIDEO_TAIL_REQUIRED_SEC and tail_count == 0 and has_tail is not False:
        raise PreprocessViolation("长视频缺少 tail 且未显式标记 has_tail=false")

    if flattened_frame_paths != frame_paths:
        raise PreprocessViolation("preprocess.frame_paths 必须与 segments[*].frames[*].frame_path 扁平聚合一致")
    if [round(x, 3) for x in flattened_frame_seconds] != [round(float(x), 3) for x in frame_seconds]:
        raise PreprocessViolation("preprocess.frame_seconds 必须与 segments[*].frames[*].frame_second 扁平聚合一致")
