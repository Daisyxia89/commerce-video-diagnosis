from __future__ import annotations

from ..errors import FactPackViolation



def assert_factpack_schema(factpack: dict) -> None:
    if not isinstance(factpack, dict):
        raise FactPackViolation("FactPack 顶层必须是对象")
    video_meta = factpack.get("video_meta")
    segments = factpack.get("segments")
    semantic_bundles = factpack.get("semantic_bundles")
    segment_to_bundle_map = factpack.get("segment_to_bundle_map")
    bundle_to_segment_range = factpack.get("bundle_to_segment_range")
    second_filter_trace = factpack.get("second_filter_trace")
    storyboard_source = factpack.get("storyboard_source")
    if not isinstance(video_meta, dict):
        raise FactPackViolation("FactPack 缺少 video_meta")
    if not isinstance(segments, list) or not segments:
        raise FactPackViolation("FactPack 缺少 segments 或 segments 为空")
    if storyboard_source != "semantic_bundles":
        raise FactPackViolation("FactPack.storyboard_source 必须显式声明为 semantic_bundles")
    if not isinstance(semantic_bundles, list) or not semantic_bundles:
        raise FactPackViolation("FactPack 缺少 semantic_bundles 或 semantic_bundles 为空")
    if not isinstance(segment_to_bundle_map, dict) or not segment_to_bundle_map:
        raise FactPackViolation("FactPack 缺少 segment_to_bundle_map")
    if not isinstance(bundle_to_segment_range, dict) or not bundle_to_segment_range:
        raise FactPackViolation("FactPack 缺少 bundle_to_segment_range")
    if not isinstance(second_filter_trace, dict):
        raise FactPackViolation("FactPack 缺少 second_filter_trace")
    if not isinstance(second_filter_trace.get("candidates"), list):
        raise FactPackViolation("FactPack.second_filter_trace.candidates 必须为列表")
    if not isinstance(second_filter_trace.get("decisions"), list):
        raise FactPackViolation("FactPack.second_filter_trace.decisions 必须为列表")
    for key in ("source_platform", "duration_sec", "fps", "resolution"):
        if key not in video_meta:
            raise FactPackViolation(f"video_meta 缺少 {key}")
    seen: set[str] = set()
    ordered_segment_ids: list[str] = []
    last_end = -1.0
    for seg in segments:
        segment_id = str(seg.get("segment_id") or "").strip()
        if not segment_id:
            raise FactPackViolation("segment_id 缺失")
        if segment_id in seen:
            raise FactPackViolation(f"segment_id 重复: {segment_id}")
        seen.add(segment_id)
        ordered_segment_ids.append(segment_id)
        start_sec = seg.get("start_sec")
        end_sec = seg.get("end_sec")
        if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)):
            raise FactPackViolation(f"{segment_id} 时间轴缺失或非法")
        if start_sec >= end_sec:
            raise FactPackViolation(f"{segment_id} 时间轴非法，要求 start_sec < end_sec")
        if start_sec < last_end:
            raise FactPackViolation(f"{segment_id} 时间轴未按顺序递增")
        last_end = float(end_sec)
        vf = seg.get("visual_facts") or {}
        af = seg.get("audio_facts") or {}
        rf = seg.get("rhythm_facts") or {}
        for field in ("shot_size", "camera_movement", "visual_subject", "lighting_tone"):
            if not str(vf.get(field) or "").strip():
                raise FactPackViolation(f"{segment_id} 缺少 visual_facts.{field}")
        if af.get("asr_text") is None or not isinstance(af.get("asr_text"), str):
            raise FactPackViolation(f"{segment_id} audio_facts.asr_text 必须为字符串")
        for field in ("transition_type", "pace_marker"):
            if not str(rf.get(field) or "").strip():
                raise FactPackViolation(f"{segment_id} 缺少 rhythm_facts.{field}")
        for action in vf.get("actions") or []:
            if not str(action.get("physical_intensity") or "").strip():
                raise FactPackViolation(f"{segment_id} actions[*].physical_intensity 缺失")
        for ocr in seg.get("ocr_facts") or []:
            position = ocr.get("position")
            if not isinstance(position, dict) or not {"x", "y", "w", "h"}.issubset(position.keys()):
                raise FactPackViolation(f"{segment_id} ocr_facts.position 必须含 x/y/w/h")
            for axis in ("x", "y", "w", "h"):
                value = position.get(axis)
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise FactPackViolation(f"{segment_id} ocr_facts.position.{axis} 必须为 0-1 数值")
            for field in ("color", "font_family", "font_weight", "font_size_level", "stroke_style", "text_effect_style"):
                if not str(ocr.get(field) or "").strip():
                    raise FactPackViolation(f"{segment_id} ocr_facts.{field} 缺失")

    covered_segment_ids: list[str] = []
    seen_bundle_ids: set[str] = set()
    for index, bundle in enumerate(semantic_bundles):
        bundle_id = str(bundle.get("bundle_id") or "").strip()
        if not bundle_id:
            raise FactPackViolation(f"semantic_bundles[{index}].bundle_id 缺失")
        if bundle_id in seen_bundle_ids:
            raise FactPackViolation(f"semantic_bundles.bundle_id 重复: {bundle_id}")
        seen_bundle_ids.add(bundle_id)
        segment_ids = bundle.get("segment_ids") or []
        aggregation_reason = bundle.get("aggregation_reason") or []
        coverage_frame_refs = bundle.get("coverage_frame_refs") or []
        blocked_boundary_ids = bundle.get("blocked_boundary_ids") or []
        if not isinstance(segment_ids, list) or not segment_ids:
            raise FactPackViolation(f"{bundle_id} 缺少 segment_ids")
        if not isinstance(aggregation_reason, list) or not aggregation_reason:
            raise FactPackViolation(f"{bundle_id} 缺少 aggregation_reason")
        if not isinstance(coverage_frame_refs, list) or not coverage_frame_refs:
            raise FactPackViolation(f"{bundle_id} 缺少 coverage_frame_refs")
        if not isinstance(blocked_boundary_ids, list):
            raise FactPackViolation(f"{bundle_id}.blocked_boundary_ids 必须为列表")
        if float(bundle.get("start_sec") or 0.0) >= float(bundle.get("end_sec") or 0.0):
            raise FactPackViolation(f"{bundle_id} 时间轴非法，要求 start_sec < end_sec")
        bundle_indexes = []
        for segment_id in segment_ids:
            if segment_id not in seen:
                raise FactPackViolation(f"{bundle_id} 引用了不存在的 segment_id: {segment_id}")
            bundle_indexes.append(ordered_segment_ids.index(segment_id))
            mapped_bundle_id = str(segment_to_bundle_map.get(segment_id) or "").strip()
            if mapped_bundle_id != bundle_id:
                raise FactPackViolation(f"segment_to_bundle_map[{segment_id}] 与 {bundle_id} 不一致")
        if bundle_indexes != list(range(bundle_indexes[0], bundle_indexes[-1] + 1)):
            raise FactPackViolation(f"{bundle_id}.segment_ids 必须连续，不允许跳段聚合")
        range_payload = bundle_to_segment_range.get(bundle_id)
        if not isinstance(range_payload, dict):
            raise FactPackViolation(f"bundle_to_segment_range 缺少 {bundle_id}")
        if int(range_payload.get("start_segment_index", -1)) != bundle_indexes[0]:
            raise FactPackViolation(f"bundle_to_segment_range[{bundle_id}].start_segment_index 不正确")
        if int(range_payload.get("end_segment_index", -1)) != bundle_indexes[-1]:
            raise FactPackViolation(f"bundle_to_segment_range[{bundle_id}].end_segment_index 不正确")
        if str(range_payload.get("start_segment_id") or "") != segment_ids[0]:
            raise FactPackViolation(f"bundle_to_segment_range[{bundle_id}].start_segment_id 不正确")
        if str(range_payload.get("end_segment_id") or "") != segment_ids[-1]:
            raise FactPackViolation(f"bundle_to_segment_range[{bundle_id}].end_segment_id 不正确")
        covered_segment_ids.extend(segment_ids)

    if sorted(covered_segment_ids) != sorted(ordered_segment_ids):
        raise FactPackViolation("semantic_bundles 必须完整且唯一覆盖全部 segments")
    if len(semantic_bundles) > len(segments):
        raise FactPackViolation("semantic_bundles 数量不能大于 segments 数量")

    for index, candidate in enumerate(second_filter_trace.get("candidates") or []):
        boundary_id = str(candidate.get("boundary_id") or "").strip()
        if not boundary_id:
            raise FactPackViolation(f"second_filter_trace.candidates[{index}].boundary_id 缺失")
        if not isinstance(candidate.get("trigger_signals"), list):
            raise FactPackViolation(f"{boundary_id}.trigger_signals 必须为列表")
        if not isinstance(candidate.get("high_ocr_scene"), bool):
            raise FactPackViolation(f"{boundary_id}.high_ocr_scene 必须为 bool")
        for field in ("prev_segment_semantics", "next_segment_semantics", "decision_context"):
            if not isinstance(candidate.get(field), dict):
                raise FactPackViolation(f"{boundary_id}.{field} 必须为对象")

    for index, decision in enumerate(second_filter_trace.get("decisions") or []):
        boundary_id = str(decision.get("boundary_id") or "").strip()
        if not boundary_id:
            raise FactPackViolation(f"second_filter_trace.decisions[{index}].boundary_id 缺失")
        if decision.get("decision") not in {"keep", "drop"}:
            raise FactPackViolation(f"{boundary_id}.decision 非法")
        if not str(decision.get("reason_code") or "").strip():
            raise FactPackViolation(f"{boundary_id}.reason_code 缺失")
        decision_context = decision.get("decision_context")
        if not isinstance(decision_context, dict):
            raise FactPackViolation(f"{boundary_id}.decision_context 必须为对象")
        for field in (
            "candidate_score",
            "adjacent_protected_count_10s",
            "same_bundle_relation",
            "ocr_jump_strength",
            "layout_migration_strength",
        ):
            if field not in decision_context:
                raise FactPackViolation(f"{boundary_id}.decision_context.{field} 缺失")
