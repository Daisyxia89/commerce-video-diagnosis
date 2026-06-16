from __future__ import annotations

import re
from collections import OrderedDict

from ..errors import FactPackViolation
from .second_filter import build_second_filter_candidate, second_filter

CTA_TOKENS = {
    "下单",
    "购买",
    "立即购买",
    "马上买",
    "点链接",
    "点击链接",
    "购物车",
    "橱窗",
    "领券",
    "直播间",
    "拍",
    "抢",
    "私信",
    "咨询",
    "客服",
    "关注",
}
DISCLAIMER_TOKENS = {
    "效果因人而异",
    "仅供参考",
    "示意",
    "请遵医嘱",
    "谨慎购买",
}


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", "", text)



def _extract_tokens(*parts: object) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        text = _normalize_text(part)
        if not text:
            continue
        ascii_tokens = re.findall(r"[a-z0-9_]+", text)
        tokens.update(token for token in ascii_tokens if len(token) >= 2)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        tokens.update(chinese_tokens)
    return tokens



def _token_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    return bool(left & right)



def _segment_text_payload(segment: dict) -> str:
    visual = (segment.get("visual_facts") or {}).get("visual_subject", "")
    asr_text = (segment.get("audio_facts") or {}).get("asr_text", "")
    ocr_text = " ".join(str(item.get("text") or "") for item in segment.get("ocr_facts") or [])
    action_text = " ".join(str(item.get("action_name") or "") for item in (segment.get("visual_facts") or {}).get("actions") or [])
    return " ".join(part for part in (visual, asr_text, ocr_text, action_text) if str(part).strip())



def _is_cta_segment(segment: dict) -> bool:
    text = _normalize_text(_segment_text_payload(segment))
    return any(token in text for token in CTA_TOKENS)



def _is_disclaimer_segment(segment: dict) -> bool:
    text = _normalize_text(_segment_text_payload(segment))
    return any(token in text for token in DISCLAIMER_TOKENS)



def _pick_primary_frame_refs(preproc_segments: list[dict] | None) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for item in preproc_segments or []:
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            continue
        frames = item.get("frames") or []
        frame_ids = [str(frame.get("frame_id") or "").strip() for frame in frames if str(frame.get("frame_id") or "").strip()]
        refs[segment_id] = frame_ids[:1] or [f"{segment_id}:PRIMARY"]
    return refs



def _protected_boundary_context(decision_report: list[dict] | None) -> dict[float, dict[str, object]]:
    protected: dict[float, dict[str, object]] = {}
    for item in decision_report or []:
        if not isinstance(item, dict):
            continue
        sec = item.get("protected_representative_sec")
        if sec is None:
            continue
        sec_key = round(float(sec), 3)
        current = protected.setdefault(sec_key, {"protected_sec": sec_key, "trigger_signals": [], "representative_metrics": {}})
        reason_code = str(item.get("reason_code") or "").strip()
        if reason_code == "SHOT_BOUNDARY_REP_PROTECTED":
            current["trigger_signals"] = list(item.get("trigger_signals") or [])
            current["protected_reason"] = item.get("protected_representative_reason")
        elif reason_code == "SHOT_BOUNDARY_CLUSTER_REP_SELECTED":
            current["representative_metrics"] = dict(item.get("representative_metrics") or {})
            current["protected_reason"] = item.get("protected_representative_reason")
    return protected



def _build_boundary_ids(segments: list[dict], protected_context: dict[float, dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    boundary_map: dict[tuple[str, str], dict[str, object]] = {}
    for left, right in zip(segments, segments[1:]):
        boundary_sec = round(float(left["end_sec"]), 3)
        boundary_id = f"BOUNDARY_{left['segment_id']}_{right['segment_id']}"
        protected_payload = protected_context.get(boundary_sec, {})
        boundary_map[(left["segment_id"], right["segment_id"])] = {
            "boundary_id": boundary_id,
            "boundary_sec": boundary_sec,
            "protected_sec": boundary_sec,
            "protected": boundary_sec in protected_context,
            "trigger_signals": list(protected_payload.get("trigger_signals") or []),
            "representative_metrics": dict(protected_payload.get("representative_metrics") or {}),
            "protected_reason": protected_payload.get("protected_reason"),
        }
    return boundary_map



def _build_second_filter_trace(segments: list[dict], boundary_map: dict[tuple[str, str], dict[str, object]]) -> dict[str, list[dict]]:
    protected_secs = sorted(
        round(float(boundary_info.get("protected_sec") or 0.0), 3)
        for boundary_info in boundary_map.values()
        if boundary_info.get("protected")
    )
    candidates: list[dict] = []
    decisions: list[dict] = []
    decision_by_boundary_id: dict[str, dict] = {}
    for left_segment, right_segment in zip(segments, segments[1:]):
        boundary_info = boundary_map[(left_segment["segment_id"], right_segment["segment_id"])]
        if not boundary_info.get("protected"):
            continue
        protected_sec = round(float(boundary_info.get("protected_sec") or 0.0), 3)
        adjacent_protected_count_10s = sum(1 for sec in protected_secs if abs(sec - protected_sec) <= 10.0) - 1
        candidate = build_second_filter_candidate(
            left_segment,
            right_segment,
            boundary_info=boundary_info,
            adjacent_protected_count_10s=max(adjacent_protected_count_10s, 0),
        )
        decision = second_filter(candidate)
        candidates.append(candidate)
        decisions.append(decision)
        decision_by_boundary_id[str(decision["boundary_id"])] = decision
    return {
        "candidates": candidates,
        "decisions": decisions,
        "decision_by_boundary_id": decision_by_boundary_id,
    }



def _continuity_reasons(bundle_segments: list[dict], next_segment: dict) -> list[str]:
    last_segment = bundle_segments[-1]
    last_visual = last_segment.get("visual_facts") or {}
    next_visual = next_segment.get("visual_facts") or {}
    last_audio = last_segment.get("audio_facts") or {}
    next_audio = next_segment.get("audio_facts") or {}

    reasons: list[str] = []

    subject_tokens_left = _extract_tokens(last_visual.get("visual_subject"))
    subject_tokens_right = _extract_tokens(next_visual.get("visual_subject"))
    if _token_overlap(subject_tokens_left, subject_tokens_right):
        reasons.append("same_primary_subject")

    action_tokens_left = _extract_tokens(*(item.get("action_name") for item in last_visual.get("actions") or []))
    action_tokens_right = _extract_tokens(*(item.get("action_name") for item in next_visual.get("actions") or []))
    if _token_overlap(action_tokens_left, action_tokens_right):
        reasons.append("continuous_action_chain")

    text_tokens_left = _extract_tokens(last_audio.get("asr_text"), *(item.get("text") for item in last_segment.get("ocr_facts") or []))
    text_tokens_right = _extract_tokens(next_audio.get("asr_text"), *(item.get("text") for item in next_segment.get("ocr_facts") or []))
    if _token_overlap(text_tokens_left, text_tokens_right):
        reasons.append("same_communication_goal")

    ocr_tokens_left = _extract_tokens(*(item.get("text") for item in last_segment.get("ocr_facts") or []))
    ocr_tokens_right = _extract_tokens(*(item.get("text") for item in next_segment.get("ocr_facts") or []))
    if _token_overlap(ocr_tokens_left, ocr_tokens_right):
        reasons.append("ocr_rollover_only")

    if (
        str(last_visual.get("shot_size") or "").strip()
        and str(last_visual.get("shot_size") or "").strip() == str(next_visual.get("shot_size") or "").strip()
        and str(last_visual.get("camera_movement") or "").strip() == str(next_visual.get("camera_movement") or "").strip()
    ):
        reasons.append("same_shot_task_continuity")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped



def _hard_block_reasons(
    left_segment: dict,
    right_segment: dict,
    boundary_info: dict[str, object],
    *,
    second_filter_decision: dict | None = None,
) -> list[str]:
    reasons: list[str] = []
    if boundary_info.get("protected") and (second_filter_decision or {}).get("decision") != "drop":
        reasons.append("protected_representative_boundary")

    left_cta = _is_cta_segment(left_segment)
    right_cta = _is_cta_segment(right_segment)
    if left_cta != right_cta:
        reasons.append("cta_transition")

    left_disclaimer = _is_disclaimer_segment(left_segment)
    right_disclaimer = _is_disclaimer_segment(right_segment)
    if left_disclaimer != right_disclaimer:
        reasons.append("independent_disclaimer_unit")

    left_type = str(left_segment.get("segment_type") or "main")
    right_type = str(right_segment.get("segment_type") or "main")
    if left_type != right_type and "tail" in {left_type, right_type}:
        reasons.append("tail_structure_boundary")

    return reasons



def _build_semantic_bundles(
    segments: list[dict],
    preproc_segments: list[dict] | None = None,
    decision_report: list[dict] | None = None,
) -> tuple[list[dict], dict[str, str], dict[str, dict[str, object]], dict[str, list[dict]]]:
    if not segments:
        return [], {}, {}, {"candidates": [], "decisions": []}

    frame_refs = _pick_primary_frame_refs(preproc_segments)
    protected_context = _protected_boundary_context(decision_report)
    boundary_map = _build_boundary_ids(segments, protected_context)
    second_filter_trace = _build_second_filter_trace(segments, boundary_map)
    second_filter_decision_by_boundary = second_filter_trace.get("decision_by_boundary_id") or {}

    bundles: list[dict] = []
    segment_to_bundle_map: dict[str, str] = {}
    bundle_to_segment_range: dict[str, dict[str, object]] = {}

    current_bundle_segments: list[dict] = [segments[0]]
    current_crossed_boundaries: list[str] = []
    current_reasons: list[str] = []

    def flush_bundle() -> None:
        bundle_index = len(bundles) + 1
        bundle_id = f"BUNDLE_{bundle_index:02d}"
        member_ids = [segment["segment_id"] for segment in current_bundle_segments]
        aggregation_reason = current_reasons[:] or ["single_physical_segment_no_further_merge"]
        coverage_refs: list[str] = []
        for segment_id in member_ids:
            refs = frame_refs.get(segment_id)
            if refs:
                coverage_refs.extend(refs[:1])
            else:
                coverage_refs.append(f"{segment_id}:PRIMARY")
        bundle = {
            "bundle_id": bundle_id,
            "start_sec": float(current_bundle_segments[0]["start_sec"]),
            "end_sec": float(current_bundle_segments[-1]["end_sec"]),
            "segment_ids": member_ids,
            "bundle_role": "narrative_unit",
            "aggregation_reason": aggregation_reason,
            "blocked_boundary_ids": current_crossed_boundaries[:],
            "coverage_frame_refs": coverage_refs,
        }
        bundles.append(bundle)
        start_index = segments.index(current_bundle_segments[0])
        end_index = segments.index(current_bundle_segments[-1])
        bundle_to_segment_range[bundle_id] = {
            "start_segment_index": start_index,
            "end_segment_index": end_index,
            "start_segment_id": member_ids[0],
            "end_segment_id": member_ids[-1],
        }
        for segment_id in member_ids:
            segment_to_bundle_map[segment_id] = bundle_id

    for next_segment in segments[1:]:
        left_segment = current_bundle_segments[-1]
        boundary_info = boundary_map[(left_segment["segment_id"], next_segment["segment_id"])]
        boundary_id = str(boundary_info["boundary_id"])
        second_filter_decision = second_filter_decision_by_boundary.get(boundary_id)
        hard_block_reasons = _hard_block_reasons(
            left_segment,
            next_segment,
            boundary_info,
            second_filter_decision=second_filter_decision,
        )
        continuity_reasons = _continuity_reasons(current_bundle_segments, next_segment)

        if hard_block_reasons or not continuity_reasons:
            flush_bundle()
            current_bundle_segments = [next_segment]
            current_crossed_boundaries = []
            current_reasons = []
            continue

        current_bundle_segments.append(next_segment)
        if boundary_id not in current_crossed_boundaries:
            current_crossed_boundaries.append(boundary_id)
        if second_filter_decision and second_filter_decision.get("decision") == "drop":
            reason_code = str(second_filter_decision.get("reason_code") or "").strip()
            if reason_code and reason_code not in current_reasons:
                current_reasons.append(reason_code)
        for reason in continuity_reasons:
            if reason not in current_reasons:
                current_reasons.append(reason)

    flush_bundle()
    second_filter_trace.pop("decision_by_boundary_id", None)
    return bundles, segment_to_bundle_map, bundle_to_segment_range, second_filter_trace



def build_factpack(normalized: dict, video_meta: dict, preproc: dict | None = None) -> dict:
    if not normalized.get("vlm"):
        raise FactPackViolation("P0 最小闭环要求 vlm fixture 必须存在")
    by_segment: OrderedDict[str, dict] = OrderedDict()
    preproc_segment_map = {
        str(item.get("segment_id") or "").strip(): item
        for item in ((preproc or {}).get("segments") or [])
        if str(item.get("segment_id") or "").strip()
    }
    for stream_name in ("vlm", "asr", "ocr"):
        for item in normalized.get(stream_name, []):
            segment_id = item["segment_id"]
            preproc_segment = preproc_segment_map.get(segment_id, {})
            current = by_segment.setdefault(
                segment_id,
                {
                    "segment_id": segment_id,
                    "start_sec": item["start_sec"],
                    "end_sec": item["end_sec"],
                    "visual_facts": {},
                    "audio_facts": {"asr_text": "", "sfx_events": [], "bgm_events": []},
                    "ocr_facts": [],
                    "rhythm_facts": {},
                    "segment_type": preproc_segment.get("segment_type", "main"),
                },
            )
            if stream_name == "vlm":
                current["visual_facts"] = item["visual_facts"]
                current["rhythm_facts"] = item["rhythm_facts"]
            elif stream_name == "asr":
                current["audio_facts"] = item["audio_facts"]
            elif stream_name == "ocr":
                current["ocr_facts"] = item["ocr_facts"]
    segments = sorted(by_segment.values(), key=lambda x: (x["start_sec"], x["segment_id"]))
    semantic_bundles, segment_to_bundle_map, bundle_to_segment_range, second_filter_trace = _build_semantic_bundles(
        segments,
        preproc_segments=(preproc or {}).get("segments") or [],
        decision_report=(preproc or {}).get("decision_report") or [],
    )
    for segment in segments:
        segment.pop("segment_type", None)
    return {
        "video_meta": video_meta,
        "segments": segments,
        "semantic_bundles": semantic_bundles,
        "segment_to_bundle_map": segment_to_bundle_map,
        "bundle_to_segment_range": bundle_to_segment_range,
        "second_filter_trace": second_filter_trace,
        "storyboard_source": "semantic_bundles",
    }
