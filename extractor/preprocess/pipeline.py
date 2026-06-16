from __future__ import annotations

import json
import math
from pathlib import Path

import cv2

from ..errors import PreprocessViolation
from ..providers.runtime_governance import ProviderRuntimeGovernance
from .metadata import extract_audio, extract_frame, probe_video_meta

SAMPLE_INTERVAL_SEC = 0.5
MIN_SHOT_LENGTH_SEC = 0.8
HIST_DIFF_BASE_THRESHOLD = 0.45
FRAME_DIFF_BASE_THRESHOLD = 16.0
ADAPTIVE_HIST_ZSCORE = 0.8
ADAPTIVE_FRAME_ZSCORE = 1.2
MAX_FRAMES_CAP = 256
SHORT_STABLE_SHOT_SEC = 2.5
BOUNDARY_CLUSTER_WINDOW_SEC = 2.5
BOUNDARY_PEAK_WINDOW_SEC = 1.0
SEMANTIC_MERGE_TEXT_DENSITY_DELTA = 0.02
SEMANTIC_MERGE_TEXT_BOX_DELTA = 4
SEMANTIC_MERGE_DIFF_DELTA = 12.0
SEMANTIC_MERGE_HIST_DELTA = 0.12
SEMANTIC_BLOCK_TEXT_DENSITY_DELTA = 0.045
SEMANTIC_BLOCK_TEXT_BOX_DELTA = 8
SEMANTIC_BLOCK_BOUNDARY_STRENGTH = 110.0
SEMANTIC_BLOCK_BOUNDARY_HIST_DIFF = 0.72
SEMANTIC_CHAIN_MAX_MERGES = 1
SEMANTIC_CHAIN_MAX_DURATION_SEC = 12.0
SEMANTIC_CONTINUITY_MAX_DURATION_RATIO = 2.5
SEMANTIC_CONTINUITY_MAX_SAMPLE_RATIO = 3.0
SEMANTIC_CONTINUITY_MIN_SIGNALS = 2
SEMANTIC_TIE_BREAKER_MIN_SCORE = 3
TAIL_RATIO = 0.10
TAIL_MIN_WINDOW_SEC = 3.0
TAIL_MAX_WINDOW_SEC = 12.0
LONG_VIDEO_TAIL_REQUIRED_SEC = 30.0
TAIL_TEXT_DENSITY_THRESHOLD = 0.08
TAIL_TEXT_BOXES_THRESHOLD = 8
TAIL_STATIC_FRAME_DIFF_THRESHOLD = 10.0
TAIL_STATIC_HIST_DIFF_THRESHOLD = 0.15
TAIL_STATIC_TEXT_DENSITY_THRESHOLD = 0.03
TAIL_STATIC_TEXT_BOXES_THRESHOLD = 4
UPSAMPLE_TEXT_DENSITY_THRESHOLD = 0.05
UPSAMPLE_TEXT_BOXES_THRESHOLD = 8
UPSAMPLE_FRAME_DIFF_THRESHOLD = 24.0
UPSAMPLE_HIST_DIFF_THRESHOLD = 0.55
UPSAMPLE_TEXT_CHANGE_DELTA = 0.012
UPSAMPLE_BOX_CHANGE_DELTA = 2
FRAME_DEDUP_MIN_GAP_SEC = 0.35


def _fallback_segment_count(duration: float) -> int:
    if duration <= 1.0:
        return 1
    return max(1, min(12, math.ceil(duration / 4.5)))



def _clamp_frame_second(sec: float, start: float, end: float) -> float:
    upper = max(end - 0.05, start)
    return round(min(max(sec, start), upper), 3)



def _new_frame_item(segment_id: str, index: int, sec: float, role: str, representative: dict | None = None) -> dict:
    item = {
        "frame_id": f"{segment_id}_F{index:02d}",
        "frame_second": round(sec, 3),
        "sampling_role": role,
    }
    if representative is not None:
        item["representative_score"] = representative.get("score")
        item["diff_score"] = representative.get("diff_score")
        item["hist_diff"] = representative.get("hist_diff")
        item["text_density"] = representative.get("text_density")
        item["text_boxes"] = representative.get("text_boxes")
    return item



def _build_uniform_segments(duration: float) -> list[dict]:
    count = _fallback_segment_count(duration)
    step = duration / count
    segments: list[dict] = []
    for idx in range(count):
        start = round(idx * step, 3)
        end = round(duration if idx == count - 1 else (idx + 1) * step, 3)
        mid = _clamp_frame_second(round((start + end) / 2.0, 3), start, end)
        segment_id = f"SEG{idx + 1:02d}"
        frames = [_new_frame_item(segment_id, 1, mid, "middle")]
        segments.append(
            {
                "segment_id": segment_id,
                "shot_id": f"SHOT{idx + 1:02d}",
                "start_sec": start,
                "end_sec": end,
                "frame_second": mid,
                "segment_strategy": "uniform_fallback",
                "frames": frames,
            }
        )
    return segments



def _compute_text_density(frame) -> tuple[float, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    canvas_area = max(height * width, 1)
    total_area = 0.0
    text_boxes = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 20 or w < 6 or h < 6:
            continue
        if w > width * 0.95 or h > height * 0.5:
            continue
        total_area += area
        text_boxes += 1
    density = total_area / canvas_area
    return density, text_boxes



def _compute_color_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist



def _position_bonus(sec: float, duration: float) -> float:
    bonus = 0.0
    if sec <= 3.0:
        bonus += 4.0
    if duration - sec <= 15.0:
        bonus += 8.0
    return bonus



def _preprocess_provider_mode(provider_fallback_mode: str) -> str:
    # auto / force_on 为协议保留位：当前公开版无内置 fallback 实现，仅用于表达调用方配置意图。
    mode = str(provider_fallback_mode or "force_off").strip().lower() or "force_off"
    if mode == "force_on":
        return "fallback_requested"
    return "external_public"



def _allow_preprocess_provider_fallback(provider_fallback_mode: str) -> bool:
    _ = _preprocess_provider_mode(provider_fallback_mode)
    return False



def _raise_preprocess_provider_fallback_error(provider_fallback_mode: str) -> None:
    mode = str(provider_fallback_mode or "force_off").strip().lower() or "force_off"
    if mode == "force_on":
        raise PreprocessViolation(
            "OCR feedback provider fallback 协议位已被显式请求，但公开仓库未包含任何内置实现。请通过 providers.ocr 注入可执行 OCR provider，或关闭 enable_real_ocr_feedback。"
        )
    raise PreprocessViolation(
        "OCR feedback 已启用，但当前未配置可用 OCR provider。公开仓库仅保留 provider fallback 协议位，不再内置任何默认执行路径；请配置 providers.ocr，或关闭 enable_real_ocr_feedback。"
    )



def _run_real_ocr_feedback(
    frame_path: str,
    *,
    runtime_governance: ProviderRuntimeGovernance | None = None,
    operation_key: str = "",
    provider_fallback_mode: str = "force_off",
) -> tuple[int, float, list[str]]:
    _ = frame_path
    _ = runtime_governance
    _ = operation_key
    if not _allow_preprocess_provider_fallback(provider_fallback_mode):
        _raise_preprocess_provider_fallback_error(provider_fallback_mode)
    raise PreprocessViolation("OCR feedback fallback protocol stub reached unexpectedly.")



def _cta_tail_bonus(sec: float, duration: float, text_density: float, text_boxes: int) -> tuple[float, list[str]]:
    remaining = duration - sec
    if remaining > 8.0:
        return 0.0, []
    bonus = 6.0
    reasons = ["TAIL_WINDOW_HIT"]
    if text_density >= 0.08:
        bonus += 6.0
        reasons.append("TAIL_TEXT_DENSITY_HIT")
    if text_boxes >= 8:
        bonus += 4.0
        reasons.append("TAIL_TEXT_BOXES_HIT")
    if remaining <= 3.0:
        bonus += 6.0
        reasons.append("TAIL_LAST3S_HIT")
    return bonus, reasons



def _adaptive_threshold(values: list[float], base: float, zscore: float) -> float:
    if not values:
        return round(base, 4)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(variance, 0.0))
    return round(max(base, mean + std * zscore), 4)



def _collect_frame_samples(video_path: str, duration: float, probe_dir: str = "") -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return [], {}

    stride = max(1, int(round(fps * SAMPLE_INTERVAL_SEC)))
    frame_idx = 0
    samples: list[dict] = []
    prev_small = None
    prev_hist = None
    probe_path = Path(probe_dir) if probe_dir else None
    if probe_path:
        probe_path.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        sec = min(round(frame_idx / fps, 3), round(duration, 3))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        hist = _compute_color_histogram(frame)
        text_density, text_boxes = _compute_text_density(frame)

        diff_score = 0.0
        hist_diff = 0.0
        if prev_small is not None:
            diff = cv2.absdiff(small, prev_small)
            diff_score = float(diff.mean())
        if prev_hist is not None:
            hist_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

        tail_bonus, tail_reason_codes = _cta_tail_bonus(sec, duration, text_density, text_boxes)
        score = (
            diff_score * 1.2
            + hist_diff * 60.0
            + min(text_density * 1200, 30.0)
            + min(text_boxes * 1.5, 12.0)
            + _position_bonus(sec, duration)
            + tail_bonus
        )

        frame_path = ""
        if probe_path:
            frame_path = str(probe_path / f"probe_{int(sec * 1000):05d}.jpg")
            cv2.imwrite(frame_path, frame)

        samples.append(
            {
                "sec": sec,
                "diff_score": round(diff_score, 4),
                "hist_diff": round(hist_diff, 4),
                "text_density": round(text_density, 6),
                "text_boxes": int(text_boxes),
                "score": round(score, 4),
                "frame_path": frame_path,
                "reason_codes": tail_reason_codes,
            }
        )

        prev_small = small
        prev_hist = hist
        frame_idx += 1

    cap.release()
    return samples, {"fps": fps, "sample_stride_frames": stride, "sample_interval_sec": SAMPLE_INTERVAL_SEC}



def _rank_samples_with_ocr_feedback(
    samples: list[dict],
    *,
    enable_real_ocr_feedback: bool,
    ocr_feedback_top_k: int,
    runtime_governance: ProviderRuntimeGovernance | None = None,
    provider_fallback_mode: str = "force_off",
) -> tuple[list[dict], dict[float, float], dict[float, int], set[float]]:
    ranked = sorted(samples, key=lambda x: float(x["score"]), reverse=True)
    initial_score_by_sec = {round(float(item["sec"]), 3): float(item["score"]) for item in ranked}
    rescored_secs: set[float] = set()

    if enable_real_ocr_feedback:
        rescore_limit = max(0, ocr_feedback_top_k) + 2
        for sample in ranked[:rescore_limit]:
            frame_path = sample.get("frame_path")
            if not frame_path:
                continue
            sample_sec = round(float(sample.get("sec", 0.0)), 3)
            real_ocr_count, real_ocr_chars, real_ocr_preview = _run_real_ocr_feedback(
                str(frame_path),
                runtime_governance=runtime_governance,
                operation_key=f"ocr_feedback:{sample_sec:07.3f}:{Path(str(frame_path)).name}",
                provider_fallback_mode=provider_fallback_mode,
            )
            sample["real_ocr_count"] = real_ocr_count
            sample["real_ocr_chars"] = real_ocr_chars
            sample["real_ocr_preview"] = real_ocr_preview
            sample.setdefault("reason_codes", []).append("OCR_FEEDBACK_RESCORING")
            sample["score"] = round(
                float(sample["score"])
                + min(real_ocr_count * 2.0, 10.0)
                + min(real_ocr_chars * 0.35, 12.0),
                4,
            )
            rescored_secs.add(round(float(sample["sec"]), 3))
        ranked = sorted(ranked, key=lambda x: float(x["score"]), reverse=True)

    final_rank_positions = {round(float(item["sec"]), 3): idx for idx, item in enumerate(ranked)}
    return ranked, initial_score_by_sec, final_rank_positions, rescored_secs



def _boundary_strength(sample: dict) -> float:
    hist_diff = float(sample.get("hist_diff", 0.0))
    frame_diff = float(sample.get("diff_score", 0.0))
    return round(hist_diff * 100.0 + frame_diff * 0.8, 4)



def _local_peak_rank(prev_sample: dict | None, sample: dict, next_sample: dict | None) -> int:
    current_strength = _boundary_strength(sample)
    stronger_neighbors = 0
    for neighbor in (prev_sample, next_sample):
        if neighbor is None:
            continue
        if _boundary_strength(neighbor) > current_strength + 1e-6:
            stronger_neighbors += 1
    return stronger_neighbors + 1



def _build_boundary_hit(sample: dict, *, hit_type: str, trigger: str, trigger_signals: list[str], local_peak_rank: int) -> dict:
    return {
        "sec": round(float(sample.get("sec", 0.0)), 3),
        "hist_diff": round(float(sample.get("hist_diff", 0.0)), 4),
        "frame_diff": round(float(sample.get("diff_score", 0.0)), 4),
        "trigger": trigger,
        "hit_type": hit_type,
        "trigger_signals": trigger_signals,
        "support_signal_count": len(trigger_signals),
        "local_peak_rank": int(local_peak_rank),
        "boundary_strength": _boundary_strength(sample),
        "text_density": round(float(sample.get("text_density", 0.0)), 6),
        "text_boxes": int(sample.get("text_boxes", 0)),
        "score": round(float(sample.get("score", 0.0)), 4),
    }



def _soft_cut_signals(
    prev_sample: dict | None,
    sample: dict,
    next_sample: dict | None,
    *,
    hist_threshold: float,
    frame_threshold: float,
) -> tuple[list[str], int]:
    hist_diff = float(sample.get("hist_diff", 0.0))
    frame_diff = float(sample.get("diff_score", 0.0))
    local_peak_rank = _local_peak_rank(prev_sample, sample, next_sample)
    support_signals: list[str] = []

    near_hist_threshold = hist_diff >= hist_threshold * 0.85
    near_frame_threshold = frame_diff >= frame_threshold * 0.82
    if local_peak_rank == 1 and (near_hist_threshold or near_frame_threshold):
        support_signals.append("local_peak_near_hard_threshold")

    prev_density_delta = abs(float(sample.get("text_density", 0.0)) - float((prev_sample or {}).get("text_density", 0.0)))
    next_density_delta = abs(float(sample.get("text_density", 0.0)) - float((next_sample or {}).get("text_density", 0.0)))
    prev_box_delta = abs(int(sample.get("text_boxes", 0)) - int((prev_sample or {}).get("text_boxes", 0)))
    next_box_delta = abs(int(sample.get("text_boxes", 0)) - int((next_sample or {}).get("text_boxes", 0)))
    max_density_delta = max(prev_density_delta, next_density_delta)
    max_box_delta = max(prev_box_delta, next_box_delta)
    if local_peak_rank == 1 and (max_density_delta >= 0.03 or max_box_delta >= 5):
        support_signals.append("ocr_structure_jump")
    if local_peak_rank == 1 and max_density_delta >= 0.02 and max_box_delta >= 3:
        support_signals.append("layout_migration")

    prev_stable = prev_sample is not None and float(prev_sample.get("diff_score", 0.0)) <= frame_threshold * 0.55 and float(prev_sample.get("hist_diff", 0.0)) <= hist_threshold * 0.55
    next_stable = next_sample is not None and float(next_sample.get("diff_score", 0.0)) <= frame_threshold * 0.55 and float(next_sample.get("hist_diff", 0.0)) <= hist_threshold * 0.55
    if local_peak_rank == 1 and prev_stable and next_stable and any(signal in support_signals for signal in {"ocr_structure_jump", "layout_migration"}):
        support_signals.append("stable_to_new_stable_state_switch")

    deduped_signals: list[str] = []
    for signal in support_signals:
        if signal not in deduped_signals:
            deduped_signals.append(signal)
    return deduped_signals, local_peak_rank



def _collect_raw_boundary_hits(samples: list[dict], duration: float) -> tuple[list[dict], list[dict], dict]:
    """Collect raw boundary hits in time order with hard/soft distinction.

    This step is responsible for chronological scanning and soft-cut recall only.
    It does **not** try to control final segment count.
    """
    soft_cut_window_sec = max(BOUNDARY_PEAK_WINDOW_SEC, 1.0)
    if len(samples) <= 1:
        return [], [], {
            "sample_interval_sec": SAMPLE_INTERVAL_SEC,
            "hist_diff_threshold": HIST_DIFF_BASE_THRESHOLD,
            "frame_diff_threshold": FRAME_DIFF_BASE_THRESHOLD,
            "min_shot_length_sec": max(MIN_SHOT_LENGTH_SEC, min(2.0, duration * 0.06)),
            "cluster_window_sec": BOUNDARY_CLUSTER_WINDOW_SEC,
            "boundary_peak_window_sec": BOUNDARY_PEAK_WINDOW_SEC,
            "soft_cut_window_sec": soft_cut_window_sec,
            "hard_boundary_hit_count": 0,
            "soft_cut_recall_count": 0,
        }

    ordered_samples = sorted(samples, key=lambda item: float(item.get("sec", 0.0)))
    hist_values = [float(sample.get("hist_diff", 0.0)) for sample in ordered_samples[1:]]
    frame_values = [float(sample.get("diff_score", 0.0)) for sample in ordered_samples[1:]]
    hist_threshold = _adaptive_threshold(hist_values, HIST_DIFF_BASE_THRESHOLD, ADAPTIVE_HIST_ZSCORE)
    frame_threshold = _adaptive_threshold(frame_values, FRAME_DIFF_BASE_THRESHOLD, ADAPTIVE_FRAME_ZSCORE)
    min_shot_length_sec = round(max(MIN_SHOT_LENGTH_SEC, min(2.0, duration * 0.06)), 3)

    raw_hits: list[dict] = []
    decisions: list[dict] = []
    hard_hit_count = 0
    soft_cut_count = 0
    last_detected_boundary_sec: float | None = None
    for idx, sample in enumerate(ordered_samples[1:], start=1):
        prev_sample = ordered_samples[idx - 1] if idx - 1 >= 0 else None
        next_sample = ordered_samples[idx + 1] if idx + 1 < len(ordered_samples) else None
        sec = round(float(sample.get("sec", 0.0)), 3)
        hist_diff = float(sample.get("hist_diff", 0.0))
        frame_diff = float(sample.get("diff_score", 0.0))
        hist_hit = hist_diff >= hist_threshold
        composite_hit = frame_diff >= frame_threshold and hist_diff >= max(hist_threshold * 0.7, 0.2)
        soft_cut_signals, local_peak_rank = _soft_cut_signals(
            prev_sample,
            sample,
            next_sample,
            hist_threshold=hist_threshold,
            frame_threshold=frame_threshold,
        )
        soft_cut_hit = (
            not (hist_hit or composite_hit)
            and local_peak_rank == 1
            and "local_peak_near_hard_threshold" in soft_cut_signals
            and len(soft_cut_signals) >= 2
        )
        if not (hist_hit or composite_hit or soft_cut_hit):
            continue

        if hist_hit or composite_hit:
            trigger_signals = ["hist_diff_threshold"] if hist_hit else ["hist_plus_frame_diff_threshold"]
            if hist_hit and composite_hit:
                trigger_signals = ["hist_diff_threshold", "hist_plus_frame_diff_threshold"]
            boundary_hit = _build_boundary_hit(
                sample,
                hit_type="hard_cut_hit",
                trigger="hist_diff" if hist_hit else "hist_plus_frame_diff",
                trigger_signals=trigger_signals,
                local_peak_rank=local_peak_rank,
            )
            sample.setdefault("reason_codes", []).append("SHOT_BOUNDARY_HARD_HIT_DETECTED")
            decisions.append(
                {
                    "reason_code": "SHOT_BOUNDARY_HARD_HIT_DETECTED",
                    **boundary_hit,
                }
            )
            hard_hit_count += 1
        else:
            boundary_hit = _build_boundary_hit(
                sample,
                hit_type="soft_cut_hit",
                trigger="soft_cut_recall",
                trigger_signals=soft_cut_signals,
                local_peak_rank=local_peak_rank,
            )
            sample.setdefault("reason_codes", []).append("SHOT_BOUNDARY_SOFT_CUT_RECALLED")
            decisions.append(
                {
                    "reason_code": "SHOT_BOUNDARY_SOFT_CUT_RECALLED",
                    **boundary_hit,
                }
            )
            soft_cut_count += 1

        raw_hits.append(boundary_hit)
        if last_detected_boundary_sec is not None and sec - last_detected_boundary_sec < min_shot_length_sec:
            decisions.append(
                {
                    "reason_code": "SHOT_BOUNDARY_SKIPPED_TOO_CLOSE",
                    "sec": sec,
                    "hit_type": boundary_hit["hit_type"],
                    "min_shot_length_sec": min_shot_length_sec,
                    "previous_boundary_sec": round(last_detected_boundary_sec, 3),
                }
            )
        last_detected_boundary_sec = sec

    return raw_hits, decisions, {
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "hist_diff_threshold": hist_threshold,
        "frame_diff_threshold": frame_threshold,
        "min_shot_length_sec": min_shot_length_sec,
        "cluster_window_sec": BOUNDARY_CLUSTER_WINDOW_SEC,
        "boundary_peak_window_sec": BOUNDARY_PEAK_WINDOW_SEC,
        "soft_cut_window_sec": soft_cut_window_sec,
        "hard_boundary_hit_count": hard_hit_count,
        "soft_cut_recall_count": soft_cut_count,
    }



def _build_boundary_clusters(raw_hits: list[dict], cluster_window_sec: float = BOUNDARY_CLUSTER_WINDOW_SEC) -> tuple[list[dict], list[dict]]:
    if not raw_hits:
        return [], []

    ordered_hits = sorted(raw_hits, key=lambda item: float(item.get("sec", 0.0)))
    clusters: list[list[dict]] = []
    current_members: list[dict] = [ordered_hits[0]]
    for item in ordered_hits[1:]:
        prev_sec = float(current_members[-1].get("sec", 0.0))
        sec = float(item.get("sec", 0.0))
        if sec - prev_sec <= cluster_window_sec:
            current_members.append(item)
            continue
        clusters.append(current_members)
        current_members = [item]
    clusters.append(current_members)

    outputs: list[dict] = []
    decisions: list[dict] = []
    for idx, members in enumerate(clusters, start=1):
        member_secs = [round(float(member.get("sec", 0.0)), 3) for member in members]
        member_types = [str(member.get("hit_type", "unknown")) for member in members]
        protected_candidates = [
            member for member in members if member.get("hit_type") == "soft_cut_hit" and int(member.get("support_signal_count", 0)) >= 2
        ]
        protected_candidate_secs = [round(float(member.get("sec", 0.0)), 3) for member in protected_candidates]
        protected_candidate_reasons = {
            round(float(member.get("sec", 0.0)), 3): list(member.get("trigger_signals") or []) for member in protected_candidates
        }
        cluster = {
            "cluster_id": f"CLUSTER{idx:02d}",
            "cluster_start_sec": member_secs[0],
            "cluster_end_sec": member_secs[-1],
            "member_secs": member_secs,
            "member_types": member_types,
            "member_count": len(member_secs),
            "protected_candidate_secs": protected_candidate_secs,
            "protected_candidate_reasons": protected_candidate_reasons,
            "members": members,
        }
        outputs.append(cluster)
        decisions.append(
            {
                "reason_code": "SHOT_BOUNDARY_CLUSTERED",
                "cluster_id": cluster["cluster_id"],
                "cluster_start_sec": cluster["cluster_start_sec"],
                "cluster_end_sec": cluster["cluster_end_sec"],
                "member_secs": member_secs,
                "member_types": member_types,
                "protected_candidate_secs": protected_candidate_secs,
                "protected_candidate_reasons": protected_candidate_reasons,
                "member_count": len(member_secs),
                "cluster_window_sec": round(cluster_window_sec, 3),
            }
        )
    return outputs, decisions



def _pick_cluster_representative(cluster: dict, peak_window_sec: float = BOUNDARY_PEAK_WINDOW_SEC) -> tuple[dict, list[dict]]:
    members = list(cluster.get("members") or [])
    center = (float(cluster.get("cluster_start_sec", 0.0)) + float(cluster.get("cluster_end_sec", 0.0))) / 2.0
    protected_candidate_secs = {round(float(sec), 3) for sec in (cluster.get("protected_candidate_secs") or [])}

    def protected_rank(member: dict) -> tuple:
        sec = float(member.get("sec", 0.0))
        return (
            int(member.get("support_signal_count", 0)),
            float(member.get("hist_diff", 0.0)),
            float(member.get("boundary_strength", 0.0)),
            -abs(sec - center),
            -int(member.get("local_peak_rank", 9)),
        )

    def representative_rank(member: dict) -> tuple:
        sec = float(member.get("sec", 0.0))
        return (
            float(member.get("hist_diff", 0.0)),
            float(member.get("boundary_strength", 0.0)),
            -int(member.get("local_peak_rank", 9)),
            -abs(sec - center),
            -float(member.get("score", 0.0)),
        )

    protected_members = [member for member in members if round(float(member.get("sec", 0.0)), 3) in protected_candidate_secs]
    protected_reason = None
    if protected_members:
        representative = max(protected_members, key=protected_rank)
        protected_reason = "soft_cut_structural_peak_protected"
        representative_reason = "protected_soft_cut_structural_peak"
    else:
        representative = max(members, key=representative_rank)
        representative_reason = "cluster_local_peak_hist_diff_priority"

    representative_sec = round(float(representative.get("sec", 0.0)), 3)
    rep_output = {
        "sec": representative_sec,
        "cluster_id": cluster["cluster_id"],
        "hit_type": representative.get("hit_type"),
        "trigger_signals": list(representative.get("trigger_signals") or []),
        "protected": protected_reason is not None,
        "protected_reason": protected_reason,
    }
    decisions = [
        {
            "reason_code": "SHOT_BOUNDARY_CLUSTER_REP_SELECTED",
            "cluster_id": cluster["cluster_id"],
            "representative_sec": representative_sec,
            "representative_reason": representative_reason,
            "representative_metrics": {
                "hit_type": representative.get("hit_type"),
                "hist_diff": round(float(representative.get("hist_diff", 0.0)), 4),
                "frame_diff": round(float(representative.get("frame_diff", 0.0)), 4),
                "boundary_strength": round(float(representative.get("boundary_strength", 0.0)), 4),
                "score": round(float(representative.get("score", 0.0)), 4),
                "support_signal_count": int(representative.get("support_signal_count", 0)),
            },
            "peak_window_sec": round(peak_window_sec, 3),
            "protected_representative_sec": representative_sec if protected_reason else None,
            "protected_representative_reason": protected_reason,
        }
    ]
    if protected_reason:
        decisions.append(
            {
                "reason_code": "SHOT_BOUNDARY_REP_PROTECTED",
                "cluster_id": cluster["cluster_id"],
                "protected_representative_sec": representative_sec,
                "protected_representative_reason": protected_reason,
                "trigger_signals": list(representative.get("trigger_signals") or []),
            }
        )
    dropped_secs = [round(float(member.get("sec", 0.0)), 3) for member in members if member is not representative]
    decisions.append(
        {
            "reason_code": "SHOT_BOUNDARY_CLUSTER_MEMBER_DROPPED",
            "cluster_id": cluster["cluster_id"],
            "dropped_secs": dropped_secs,
            "drop_reason": "cluster_consolidation_non_representative_members",
        }
    )
    return rep_output, decisions



def _select_cluster_representatives(clusters: list[dict]) -> tuple[list[dict], list[dict]]:
    representatives: list[dict] = []
    decisions: list[dict] = []
    for cluster in clusters:
        representative, cluster_decisions = _pick_cluster_representative(cluster)
        representatives.append(representative)
        decisions.extend(cluster_decisions)
    return representatives, decisions



def _compute_segment_semantic_signature(samples: list[dict], start: float, end: float) -> dict:
    segment_samples = sorted(_collect_segment_samples(samples, start, end), key=lambda item: float(item.get("sec", 0.0)))
    if not segment_samples:
        return {
            "duration_sec": round(max(end - start, 0.0), 3),
            "avg_text_density": 0.0,
            "avg_text_boxes": 0.0,
            "avg_diff_score": 0.0,
            "avg_hist_diff": 0.0,
            "max_text_density": 0.0,
            "max_text_boxes": 0,
            "start_text_density": 0.0,
            "start_text_boxes": 0,
            "start_diff_score": 0.0,
            "start_hist_diff": 0.0,
            "end_text_density": 0.0,
            "end_text_boxes": 0,
            "end_diff_score": 0.0,
            "end_hist_diff": 0.0,
            "static_dense_text_frames": 0,
            "sample_count": 0,
        }
    text_densities = [float(sample.get("text_density", 0.0)) for sample in segment_samples]
    text_boxes = [int(sample.get("text_boxes", 0)) for sample in segment_samples]
    diff_scores = [float(sample.get("diff_score", 0.0)) for sample in segment_samples]
    hist_diffs = [float(sample.get("hist_diff", 0.0)) for sample in segment_samples]
    first_sample = segment_samples[0]
    last_sample = segment_samples[-1]
    static_dense_text_frames = sum(
        1
        for sample in segment_samples
        if (
            float(sample.get("text_density", 0.0)) >= TAIL_TEXT_DENSITY_THRESHOLD
            or int(sample.get("text_boxes", 0)) >= TAIL_TEXT_BOXES_THRESHOLD
        )
        and float(sample.get("diff_score", 0.0)) <= TAIL_STATIC_FRAME_DIFF_THRESHOLD + 2.0
        and float(sample.get("hist_diff", 0.0)) <= TAIL_STATIC_HIST_DIFF_THRESHOLD + 0.05
    )
    return {
        "duration_sec": round(max(end - start, 0.0), 3),
        "avg_text_density": round(sum(text_densities) / len(text_densities), 6),
        "avg_text_boxes": round(sum(text_boxes) / len(text_boxes), 4),
        "avg_diff_score": round(sum(diff_scores) / len(diff_scores), 4),
        "avg_hist_diff": round(sum(hist_diffs) / len(hist_diffs), 4),
        "max_text_density": round(max(text_densities, default=0.0), 6),
        "max_text_boxes": max(text_boxes, default=0),
        "start_text_density": round(float(first_sample.get("text_density", 0.0)), 6),
        "start_text_boxes": int(first_sample.get("text_boxes", 0)),
        "start_diff_score": round(float(first_sample.get("diff_score", 0.0)), 4),
        "start_hist_diff": round(float(first_sample.get("hist_diff", 0.0)), 4),
        "end_text_density": round(float(last_sample.get("text_density", 0.0)), 6),
        "end_text_boxes": int(last_sample.get("text_boxes", 0)),
        "end_diff_score": round(float(last_sample.get("diff_score", 0.0)), 4),
        "end_hist_diff": round(float(last_sample.get("hist_diff", 0.0)), 4),
        "static_dense_text_frames": static_dense_text_frames,
        "sample_count": len(segment_samples),
    }



def _nearest_sample(samples: list[dict], sec: float) -> dict | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(float(sample.get("sec", sec)) - sec), default=None)



def _is_result_like_signature(signature: dict) -> bool:
    low_motion_tail = (
        float(signature.get("end_diff_score", 0.0)) <= TAIL_STATIC_FRAME_DIFF_THRESHOLD + 2.0
        and float(signature.get("end_hist_diff", 0.0)) <= TAIL_STATIC_HIST_DIFF_THRESHOLD + 0.05
    )
    low_motion_average = (
        float(signature.get("avg_diff_score", 0.0)) <= TAIL_STATIC_FRAME_DIFF_THRESHOLD + 2.0
        and float(signature.get("avg_hist_diff", 0.0)) <= TAIL_STATIC_HIST_DIFF_THRESHOLD + 0.05
    )
    return (
        int(signature.get("static_dense_text_frames", 0)) >= 1
        and (
            float(signature.get("max_text_density", 0.0)) >= TAIL_TEXT_DENSITY_THRESHOLD
            or int(signature.get("max_text_boxes", 0)) >= TAIL_TEXT_BOXES_THRESHOLD
        )
        and (low_motion_tail or low_motion_average)
    )



def _duration_ratio(left_signature: dict, right_signature: dict) -> float:
    left_duration = max(float(left_signature.get("duration_sec", 0.0)), 0.001)
    right_duration = max(float(right_signature.get("duration_sec", 0.0)), 0.001)
    return round(max(left_duration, right_duration) / min(left_duration, right_duration), 4)



def _sample_ratio(left_signature: dict, right_signature: dict) -> float:
    left_count = max(int(left_signature.get("sample_count", 0)), 1)
    right_count = max(int(right_signature.get("sample_count", 0)), 1)
    return round(max(left_count, right_count) / min(left_count, right_count), 4)



def _build_boundary_evidence(boundary_sample: dict | None) -> dict:
    if boundary_sample is None:
        return {}
    return {
        "sec": round(float(boundary_sample.get("sec", 0.0)), 3),
        "hist_diff": round(float(boundary_sample.get("hist_diff", 0.0)), 4),
        "frame_diff": round(float(boundary_sample.get("diff_score", 0.0)), 4),
        "boundary_strength": round(_boundary_strength(boundary_sample), 4),
        "text_density": round(float(boundary_sample.get("text_density", 0.0)), 6),
        "text_boxes": int(boundary_sample.get("text_boxes", 0)),
    }



def _build_boundary_representative_evidence(boundary_rep: dict | None) -> dict:
    if boundary_rep is None:
        return {}
    return {
        "sec": round(float(boundary_rep.get("sec", 0.0)), 3),
        "cluster_id": boundary_rep.get("cluster_id"),
        "hit_type": boundary_rep.get("hit_type"),
        "protected": bool(boundary_rep.get("protected")),
        "protected_reason": boundary_rep.get("protected_reason"),
        "trigger_signals": list(boundary_rep.get("trigger_signals") or []),
    }



def _evaluate_semantic_merge(
    left_signature: dict,
    right_signature: dict,
    boundary_sample: dict | None,
    boundary_rep: dict | None,
    *,
    left_shot_id: str,
    right_shot_id: str,
    shared_boundary_sec: float,
    chain_merge_count: int,
    merged_left_duration_sec: float,
) -> tuple[bool, str | None, list[dict]]:
    decisions: list[dict] = []
    boundary_evidence = _build_boundary_evidence(boundary_sample)
    boundary_rep_evidence = _build_boundary_representative_evidence(boundary_rep)
    text_density_delta = abs(float(left_signature.get("end_text_density", 0.0)) - float(right_signature.get("start_text_density", 0.0)))
    text_boxes_delta = abs(int(left_signature.get("end_text_boxes", 0)) - int(right_signature.get("start_text_boxes", 0)))
    avg_text_density_delta = abs(float(left_signature.get("avg_text_density", 0.0)) - float(right_signature.get("avg_text_density", 0.0)))
    avg_text_boxes_delta = abs(float(left_signature.get("avg_text_boxes", 0.0)) - float(right_signature.get("avg_text_boxes", 0.0)))
    diff_delta = abs(float(left_signature.get("avg_diff_score", 0.0)) - float(right_signature.get("avg_diff_score", 0.0)))
    hist_delta = abs(float(left_signature.get("avg_hist_diff", 0.0)) - float(right_signature.get("avg_hist_diff", 0.0)))
    duration_ratio = _duration_ratio(left_signature, right_signature)
    sample_ratio = _sample_ratio(left_signature, right_signature)
    boundary_is_soft = boundary_rep_evidence.get("hit_type") == "soft_cut_hit"
    boundary_is_protected = bool(boundary_rep_evidence.get("protected"))

    evidence_summary = {
        "left_signature": left_signature,
        "right_signature": right_signature,
        "boundary_evidence": boundary_evidence,
        "boundary_representative": boundary_rep_evidence,
        "merged_left_duration_sec": round(merged_left_duration_sec, 3),
        "chain_merge_count": chain_merge_count,
        "duration_ratio": duration_ratio,
        "sample_ratio": sample_ratio,
        "text_density_delta": round(text_density_delta, 6),
        "text_boxes_delta": text_boxes_delta,
        "avg_text_density_delta": round(avg_text_density_delta, 6),
        "avg_text_boxes_delta": round(avg_text_boxes_delta, 4),
        "diff_delta": round(diff_delta, 4),
        "hist_delta": round(hist_delta, 4),
    }

    if boundary_is_protected:
        decisions.append(
            {
                "reason_code": "SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP",
                "left_shot_id": left_shot_id,
                "right_shot_id": right_shot_id,
                "shared_boundary_sec": round(shared_boundary_sec, 3),
                "protected_representative_sec": round(float(boundary_rep_evidence.get("sec", shared_boundary_sec)), 3),
                "block_reason": boundary_rep_evidence.get("protected_reason") or "protected_representative_active",
                "evidence_summary": evidence_summary,
            }
        )
        return False, None, decisions

    block_reasons: list[str] = []
    if text_density_delta >= SEMANTIC_BLOCK_TEXT_DENSITY_DELTA or text_boxes_delta >= SEMANTIC_BLOCK_TEXT_BOX_DELTA:
        block_reasons.append("ocr_structure_jump")
    if _is_result_like_signature(right_signature):
        block_reasons.append("right_segment_result_or_cta_page")
    if _is_result_like_signature(left_signature) and not _is_result_like_signature(right_signature):
        block_reasons.append("left_segment_already_result_page")
    if boundary_evidence and (
        float(boundary_evidence.get("boundary_strength", 0.0)) >= SEMANTIC_BLOCK_BOUNDARY_STRENGTH
        or float(boundary_evidence.get("hist_diff", 0.0)) >= SEMANTIC_BLOCK_BOUNDARY_HIST_DIFF
    ):
        block_reasons.append("strong_boundary_peak")
    if chain_merge_count >= SEMANTIC_CHAIN_MAX_MERGES:
        block_reasons.append("chain_merge_limit_reached")
    if merged_left_duration_sec >= SEMANTIC_CHAIN_MAX_DURATION_SEC:
        block_reasons.append("left_anchor_duration_cap_reached")
    if block_reasons:
        decisions.append(
            {
                "reason_code": "SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE",
                "left_shot_id": left_shot_id,
                "right_shot_id": right_shot_id,
                "shared_boundary_sec": round(shared_boundary_sec, 3),
                "block_reasons": block_reasons,
                "evidence_summary": evidence_summary,
            }
        )
        return False, None, decisions

    continuity_signals: list[str] = []
    structural_signals = 0
    if avg_text_density_delta <= SEMANTIC_MERGE_TEXT_DENSITY_DELTA and avg_text_boxes_delta <= SEMANTIC_MERGE_TEXT_BOX_DELTA:
        continuity_signals.append("ocr_density_and_box_count_stable")
        structural_signals += 1
    if text_density_delta <= SEMANTIC_MERGE_TEXT_DENSITY_DELTA and text_boxes_delta <= SEMANTIC_MERGE_TEXT_BOX_DELTA:
        continuity_signals.append("boundary_entry_exit_text_stable")
        structural_signals += 1
    if diff_delta <= SEMANTIC_MERGE_DIFF_DELTA and hist_delta <= SEMANTIC_MERGE_HIST_DELTA:
        continuity_signals.append("visual_motion_stable")
        structural_signals += 1
    if duration_ratio <= SEMANTIC_CONTINUITY_MAX_DURATION_RATIO and sample_ratio <= SEMANTIC_CONTINUITY_MAX_SAMPLE_RATIO:
        continuity_signals.append("segment_scale_compatible")

    required_signal_count = SEMANTIC_CONTINUITY_MIN_SIGNALS + (1 if boundary_is_soft else 0)
    required_structural_signals = 2 if boundary_is_soft else 1
    if len(continuity_signals) >= required_signal_count and structural_signals >= required_structural_signals:
        decisions.append(
            {
                "reason_code": "SHOT_BOUNDARY_MERGE_PASSED_CONTINUITY_GATE",
                "left_shot_id": left_shot_id,
                "right_shot_id": right_shot_id,
                "shared_boundary_sec": round(shared_boundary_sec, 3),
                "continuity_signals": continuity_signals,
                "required_signal_count": required_signal_count,
                "required_structural_signals": required_structural_signals,
                "evidence_summary": evidence_summary,
            }
        )
        merge_reason_code = (
            "SHOT_BOUNDARY_MERGED_BY_SEMANTIC_CONTINUITY"
            if any(signal in continuity_signals for signal in {"ocr_density_and_box_count_stable", "boundary_entry_exit_text_stable"})
            else "SHOT_BOUNDARY_MERGED_BY_VISUAL_SIMILARITY"
        )
        return True, merge_reason_code, decisions

    tie_breaker_signals: list[str] = []
    tie_breaker_score = 0
    if avg_text_density_delta <= SEMANTIC_MERGE_TEXT_DENSITY_DELTA + 0.008:
        tie_breaker_score += 1
        tie_breaker_signals.append("avg_text_density_nearly_stable")
    if avg_text_boxes_delta <= SEMANTIC_MERGE_TEXT_BOX_DELTA + 1:
        tie_breaker_score += 1
        tie_breaker_signals.append("avg_text_boxes_nearly_stable")
    if diff_delta <= SEMANTIC_MERGE_DIFF_DELTA + 2.0:
        tie_breaker_score += 1
        tie_breaker_signals.append("avg_diff_score_nearly_stable")
    if hist_delta <= SEMANTIC_MERGE_HIST_DELTA + 0.02:
        tie_breaker_score += 1
        tie_breaker_signals.append("avg_hist_diff_nearly_stable")

    allow_tie_break_merge = (
        not boundary_is_soft
        and tie_breaker_score >= SEMANTIC_TIE_BREAKER_MIN_SCORE
        and duration_ratio <= 1.8
        and sample_ratio <= 1.8
        and float(boundary_evidence.get("boundary_strength", 0.0)) < SEMANTIC_BLOCK_BOUNDARY_STRENGTH
    )
    decisions.append(
        {
            "reason_code": "SHOT_BOUNDARY_MERGE_TIE_BREAKER_APPLIED",
            "left_shot_id": left_shot_id,
            "right_shot_id": right_shot_id,
            "shared_boundary_sec": round(shared_boundary_sec, 3),
            "tie_breaker_reason": (
                "soft_cut_boundary_prefers_keep"
                if boundary_is_soft
                else "weak_stability_supports_merge" if allow_tie_break_merge else "tie_breaker_keeps_boundary"
            ),
            "tie_breaker_score": tie_breaker_score,
            "tie_breaker_signals": tie_breaker_signals,
            "evidence_summary": evidence_summary,
        }
    )
    if not allow_tie_break_merge:
        return False, None, decisions
    merge_reason_code = (
        "SHOT_BOUNDARY_MERGED_BY_SEMANTIC_CONTINUITY"
        if tie_breaker_score >= 3 and len([signal for signal in tie_breaker_signals if "text" in signal]) >= 2
        else "SHOT_BOUNDARY_MERGED_BY_VISUAL_SIMILARITY"
    )
    return True, merge_reason_code, decisions



def _merge_semantic_continuity(
    boundaries: list[float],
    samples: list[dict],
    duration: float,
    boundary_representatives: list[dict] | None = None,
) -> tuple[list[float], list[dict], int]:
    if len(boundaries) <= 2:
        return boundaries, [{"reason_code": "SHOT_BOUNDARY_MERGE_STATUS", "semantic_merge_count": 0, "merge_actions": 0, "merge_block_count": 0}], 0

    representative_lookup = {
        round(float(item.get("sec", 0.0)), 3): item for item in (boundary_representatives or []) if isinstance(item, dict)
    }
    provisional_shots = []
    for idx in range(len(boundaries) - 1):
        start = round(float(boundaries[idx]), 3)
        end = round(float(boundaries[idx + 1]), 3)
        provisional_shots.append(
            {
                "shot_id": f"PSHOT{idx + 1:02d}",
                "start_sec": start,
                "end_sec": end,
                "signature": _compute_segment_semantic_signature(samples, start, end),
            }
        )

    decisions: list[dict] = []
    kept_boundaries = [round(float(boundaries[0]), 3)]
    merge_actions = 0
    merge_block_count = 0

    anchor_shot = provisional_shots[0]
    merged_start = float(anchor_shot["start_sec"])
    merged_end = float(anchor_shot["end_sec"])
    chain_merge_count = 0

    for right_shot in provisional_shots[1:]:
        shared_boundary_sec = round(merged_end, 3)
        boundary_sample = _nearest_sample(samples, shared_boundary_sec)
        boundary_rep = representative_lookup.get(shared_boundary_sec)
        should_merge, merge_reason_code, gate_decisions = _evaluate_semantic_merge(
            anchor_shot["signature"],
            right_shot["signature"],
            boundary_sample,
            boundary_rep,
            left_shot_id=anchor_shot["shot_id"],
            right_shot_id=right_shot["shot_id"],
            shared_boundary_sec=shared_boundary_sec,
            chain_merge_count=chain_merge_count,
            merged_left_duration_sec=merged_end - merged_start,
        )
        decisions.extend(gate_decisions)
        if should_merge and merge_reason_code:
            decisions.append(
                {
                    "reason_code": merge_reason_code,
                    "left_shot_id": anchor_shot["shot_id"],
                    "right_shot_id": right_shot["shot_id"],
                    "left_segment_start_sec": round(merged_start, 3),
                    "shared_boundary_sec": shared_boundary_sec,
                    "right_segment_end_sec": round(float(right_shot["end_sec"]), 3),
                    "merge_reason": "three_layer_gate_continuity",
                    "evidence_summary": {
                        "left_signature": anchor_shot["signature"],
                        "right_signature": right_shot["signature"],
                        "boundary_evidence": _build_boundary_evidence(boundary_sample),
                        "boundary_representative": _build_boundary_representative_evidence(boundary_rep),
                        "chain_merge_count_before_merge": chain_merge_count,
                    },
                }
            )
            merged_end = float(right_shot["end_sec"])
            chain_merge_count += 1
            merge_actions += 1
            continue

        if any(item.get("reason_code") in {"SHOT_BOUNDARY_MERGE_BLOCKED_BY_EVIDENCE", "SHOT_BOUNDARY_MERGE_BLOCKED_BY_PROTECTED_REP"} for item in gate_decisions):
            merge_block_count += 1
        kept_boundaries.append(shared_boundary_sec)
        anchor_shot = right_shot
        merged_start = float(right_shot["start_sec"])
        merged_end = float(right_shot["end_sec"])
        chain_merge_count = 0

    kept_boundaries.append(round(float(duration), 3))
    decisions.append(
        {
            "reason_code": "SHOT_BOUNDARY_MERGE_STATUS",
            "semantic_merge_count": merge_actions,
            "merge_actions": merge_actions,
            "merge_block_count": merge_block_count,
            "final_boundary_count_before_short_merge": max(0, len(kept_boundaries) - 2),
            "duration_sec": round(duration, 3),
        }
    )
    return kept_boundaries, decisions, merge_actions



def _merge_short_segments(
    boundaries: list[float],
    duration: float,
    protected_boundary_secs: set[float] | None = None,
) -> tuple[list[float], list[dict]]:
    protected_boundary_secs = {round(float(sec), 3) for sec in (protected_boundary_secs or set())}
    min_len = max(MIN_SHOT_LENGTH_SEC, min(2.5, duration * 0.15))
    decisions: list[dict] = []
    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            seg_len = end - start
            if seg_len >= min_len:
                continue
            if idx == 0:
                candidate = round(float(boundaries[idx + 1]), 3)
                if candidate in protected_boundary_secs:
                    decisions.append(
                        {
                            "reason_code": "MERGE_SHORT_SEGMENT_SKIPPED_PROTECTED",
                            "segment_start": start,
                            "segment_end": end,
                            "segment_len": round(seg_len, 3),
                            "protected_boundary": candidate,
                            "min_len": round(min_len, 3),
                        }
                    )
                    continue
                changed = True
                removed = boundaries[idx + 1]
                del boundaries[idx + 1]
                decisions.append(
                    {
                        "reason_code": "MERGE_SHORT_SEGMENT_HEAD",
                        "segment_start": start,
                        "segment_end": end,
                        "segment_len": round(seg_len, 3),
                        "removed_boundary": removed,
                        "min_len": round(min_len, 3),
                    }
                )
            elif idx == len(boundaries) - 2:
                candidate = round(float(boundaries[idx]), 3)
                if candidate in protected_boundary_secs:
                    decisions.append(
                        {
                            "reason_code": "MERGE_SHORT_SEGMENT_SKIPPED_PROTECTED",
                            "segment_start": start,
                            "segment_end": end,
                            "segment_len": round(seg_len, 3),
                            "protected_boundary": candidate,
                            "min_len": round(min_len, 3),
                        }
                    )
                    continue
                changed = True
                removed = boundaries[idx]
                del boundaries[idx]
                decisions.append(
                    {
                        "reason_code": "MERGE_SHORT_SEGMENT_TAIL",
                        "segment_start": start,
                        "segment_end": end,
                        "segment_len": round(seg_len, 3),
                        "removed_boundary": removed,
                        "min_len": round(min_len, 3),
                    }
                )
            else:
                left_len = boundaries[idx] - boundaries[idx - 1]
                right_len = boundaries[idx + 2] - boundaries[idx + 1]
                left_boundary = round(float(boundaries[idx]), 3)
                right_boundary = round(float(boundaries[idx + 1]), 3)
                prefer_left = left_len >= right_len
                candidate_order = [left_boundary, right_boundary] if prefer_left else [right_boundary, left_boundary]
                removable_boundary = next((candidate for candidate in candidate_order if candidate not in protected_boundary_secs), None)
                if removable_boundary is None:
                    decisions.append(
                        {
                            "reason_code": "MERGE_SHORT_SEGMENT_SKIPPED_PROTECTED",
                            "segment_start": start,
                            "segment_end": end,
                            "segment_len": round(seg_len, 3),
                            "protected_boundary": sorted({left_boundary, right_boundary}),
                            "min_len": round(min_len, 3),
                        }
                    )
                    continue
                changed = True
                if removable_boundary == left_boundary:
                    removed = boundaries[idx]
                    del boundaries[idx]
                    decisions.append(
                        {
                            "reason_code": "MERGE_SHORT_SEGMENT_TO_LEFT",
                            "segment_start": start,
                            "segment_end": end,
                            "segment_len": round(seg_len, 3),
                            "removed_boundary": removed,
                            "left_len": round(left_len, 3),
                            "right_len": round(right_len, 3),
                            "min_len": round(min_len, 3),
                        }
                    )
                else:
                    removed = boundaries[idx + 1]
                    del boundaries[idx + 1]
                    decisions.append(
                        {
                            "reason_code": "MERGE_SHORT_SEGMENT_TO_RIGHT",
                            "segment_start": start,
                            "segment_end": end,
                            "segment_len": round(seg_len, 3),
                            "removed_boundary": removed,
                            "left_len": round(left_len, 3),
                            "right_len": round(right_len, 3),
                            "min_len": round(min_len, 3),
                        }
                    )
            break
    return boundaries, decisions



def _collect_segment_samples(samples: list[dict], start: float, end: float) -> list[dict]:
    segment_samples = [sample for sample in samples if start <= float(sample.get("sec", 0.0)) <= end]
    if segment_samples:
        return segment_samples
    midpoint = (start + end) / 2.0
    nearest = min(samples, key=lambda sample: abs(float(sample.get("sec", midpoint)) - midpoint), default=None)
    return [nearest] if nearest is not None else []



def _compute_focus_switches(segment_samples: list[dict]) -> int:
    switches = 0
    last_bucket = None
    for sample in segment_samples:
        text_signal = float(sample.get("text_density", 0.0)) * 1200 + int(sample.get("text_boxes", 0)) * 2.0
        motion_signal = float(sample.get("diff_score", 0.0)) * 1.2 + float(sample.get("hist_diff", 0.0)) * 60.0
        bucket = "text" if text_signal >= motion_signal else "motion"
        if last_bucket is not None and bucket != last_bucket:
            switches += 1
        last_bucket = bucket
    return switches



def _compute_shot_complexity(segment_samples: list[dict], start: float, end: float) -> dict:
    shot_len = round(end - start, 3)
    if not segment_samples:
        return {
            "shot_len": shot_len,
            "sample_count": 0,
            "avg_diff_score": 0.0,
            "max_diff_score": 0.0,
            "avg_hist_diff": 0.0,
            "max_hist_diff": 0.0,
            "avg_text_density": 0.0,
            "max_text_density": 0.0,
            "max_text_boxes": 0,
            "text_change_events": 0,
            "focus_switches": 0,
            "late_result_signal": False,
            "stable_short_shot": shot_len <= SHORT_STABLE_SHOT_SEC,
        }

    diff_values = [float(sample.get("diff_score", 0.0)) for sample in segment_samples]
    hist_values = [float(sample.get("hist_diff", 0.0)) for sample in segment_samples]
    density_values = [float(sample.get("text_density", 0.0)) for sample in segment_samples]
    box_values = [int(sample.get("text_boxes", 0)) for sample in segment_samples]

    text_change_events = 0
    for idx in range(1, len(segment_samples)):
        prev = segment_samples[idx - 1]
        current = segment_samples[idx]
        density_delta = abs(float(current.get("text_density", 0.0)) - float(prev.get("text_density", 0.0)))
        box_delta = abs(int(current.get("text_boxes", 0)) - int(prev.get("text_boxes", 0)))
        if density_delta >= UPSAMPLE_TEXT_CHANGE_DELTA or box_delta >= UPSAMPLE_BOX_CHANGE_DELTA:
            text_change_events += 1

    focus_switches = _compute_focus_switches(segment_samples)
    late_start = start + max(shot_len * 0.67, 0.0)
    late_samples = [sample for sample in segment_samples if float(sample.get("sec", 0.0)) >= late_start]
    late_result_signal = any(
        float(sample.get("text_density", 0.0)) >= max(UPSAMPLE_TEXT_DENSITY_THRESHOLD, 0.03)
        or int(sample.get("text_boxes", 0)) >= max(UPSAMPLE_TEXT_BOXES_THRESHOLD - 2, 4)
        for sample in late_samples
    )

    avg_diff_score = round(sum(diff_values) / max(len(diff_values), 1), 4)
    avg_hist_diff = round(sum(hist_values) / max(len(hist_values), 1), 4)
    avg_text_density = round(sum(density_values) / max(len(density_values), 1), 6)
    max_diff_score = round(max(diff_values, default=0.0), 4)
    max_hist_diff = round(max(hist_values, default=0.0), 4)
    max_text_density = round(max(density_values, default=0.0), 6)
    max_text_boxes = max(box_values, default=0)

    stable_short_shot = (
        shot_len <= SHORT_STABLE_SHOT_SEC
        and avg_diff_score < FRAME_DIFF_BASE_THRESHOLD
        and avg_hist_diff < HIST_DIFF_BASE_THRESHOLD
        and text_change_events <= 1
        and max_text_density < 0.03
        and max_text_boxes < 5
    )
    return {
        "shot_len": shot_len,
        "sample_count": len(segment_samples),
        "avg_diff_score": avg_diff_score,
        "max_diff_score": max_diff_score,
        "avg_hist_diff": avg_hist_diff,
        "max_hist_diff": max_hist_diff,
        "avg_text_density": avg_text_density,
        "max_text_density": max_text_density,
        "max_text_boxes": max_text_boxes,
        "text_change_events": text_change_events,
        "focus_switches": focus_switches,
        "late_result_signal": late_result_signal,
        "stable_short_shot": stable_short_shot,
    }



def _determine_frame_plan(segment_samples: list[dict], start: float, end: float) -> dict:
    metrics = _compute_shot_complexity(segment_samples, start, end)
    shot_len = float(metrics["shot_len"])
    triggers: list[str] = []

    if metrics["text_change_events"] >= 2 or metrics["max_text_density"] >= UPSAMPLE_TEXT_DENSITY_THRESHOLD or metrics["max_text_boxes"] >= UPSAMPLE_TEXT_BOXES_THRESHOLD:
        triggers.append("TEXT_CHANGE_DENSE")
    if metrics["max_diff_score"] >= UPSAMPLE_FRAME_DIFF_THRESHOLD or metrics["max_hist_diff"] >= UPSAMPLE_HIST_DIFF_THRESHOLD:
        triggers.append("ACTION_STATE_CHANGE")
    if metrics["focus_switches"] >= 2:
        triggers.append("FOCUS_SWITCH")
    if metrics["late_result_signal"]:
        triggers.append("RESULT_STATE")

    downgrade_reason = ""
    if shot_len <= 1.0:
        default_frame_count = 1
        final_frame_count = 1
        downgrade_reason = "EXTREME_SHORT_SHOT_SINGLE_FRAME"
    else:
        default_frame_count = 2 if metrics["stable_short_shot"] else 3
        if triggers:
            final_frame_count = 5 if len(set(triggers)) >= 2 and shot_len >= 2.0 else 4
        else:
            final_frame_count = default_frame_count

    min_keep_frames = 1 if default_frame_count <= 2 else 3
    return {
        "default_frame_count": default_frame_count,
        "final_frame_count": final_frame_count,
        "upsampling_triggers": sorted(set(triggers)),
        "downgrade_reason": downgrade_reason,
        "min_keep_frames": min_keep_frames,
        "metrics": metrics,
    }



def _pick_best_sample_for_window(
    segment_samples: list[dict],
    window_start: float,
    window_end: float,
    *,
    target_sec: float,
    excluded_secs: set[float],
) -> dict | None:
    tight_candidates = [
        sample
        for sample in segment_samples
        if window_start <= float(sample.get("sec", 0.0)) <= window_end and round(float(sample.get("sec", 0.0)), 3) not in excluded_secs
    ]
    candidates = tight_candidates or [
        sample for sample in segment_samples if round(float(sample.get("sec", 0.0)), 3) not in excluded_secs
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item.get("score", 0.0)),
            -abs(float(item.get("sec", target_sec)) - target_sec),
        ),
    )



def _pick_role_specific_sample(
    segment_samples: list[dict],
    role: str,
    *,
    start: float,
    end: float,
    excluded_secs: set[float],
) -> dict | None:
    candidates = [sample for sample in segment_samples if round(float(sample.get("sec", 0.0)), 3) not in excluded_secs]
    if not candidates:
        return None

    if role == "dense_text":
        return max(
            candidates,
            key=lambda item: (
                float(item.get("text_density", 0.0)) * 1200 + int(item.get("text_boxes", 0)) * 3.0,
                float(item.get("score", 0.0)),
            ),
        )
    if role == "action_peak":
        return max(
            candidates,
            key=lambda item: (
                float(item.get("diff_score", 0.0)) * 1.2 + float(item.get("hist_diff", 0.0)) * 60.0,
                float(item.get("score", 0.0)),
            ),
        )
    if role == "result_state":
        late_start = start + max((end - start) * 0.67, 0.0)
        late_candidates = [sample for sample in candidates if float(sample.get("sec", 0.0)) >= late_start]
        target = late_candidates or candidates
        return max(
            target,
            key=lambda item: (
                float(item.get("text_density", 0.0)) * 1200 + int(item.get("text_boxes", 0)) * 3.0 + float(item.get("score", 0.0)),
                float(item.get("sec", 0.0)),
            ),
        )
    return max(candidates, key=lambda item: float(item.get("score", 0.0)))



def _append_selected_frame(
    selected_frames: list[dict],
    segment_id: str,
    candidate: dict | None,
    role: str,
    *,
    start: float,
    end: float,
    excluded_secs: set[float],
) -> None:
    if candidate is None:
        fallback_sec = _clamp_frame_second(round((start + end) / 2.0, 3), start, end)
        sec = fallback_sec
        if sec in excluded_secs:
            return
        selected_frames.append(_new_frame_item(segment_id, len(selected_frames) + 1, sec, role))
        excluded_secs.add(sec)
        return
    sec = _clamp_frame_second(float(candidate.get("sec", (start + end) / 2.0)), start, end)
    rounded_sec = round(sec, 3)
    if rounded_sec in excluded_secs:
        return
    selected_frames.append(_new_frame_item(segment_id, len(selected_frames) + 1, rounded_sec, role, candidate))
    excluded_secs.add(rounded_sec)



def _build_segment_frames(segment_id: str, segment_samples: list[dict], start: float, end: float, plan: dict) -> list[dict]:
    if not segment_samples:
        mid = _clamp_frame_second(round((start + end) / 2.0, 3), start, end)
        return [_new_frame_item(segment_id, 1, mid, "middle")]

    target_count = int(plan["final_frame_count"])
    selected_frames: list[dict] = []
    excluded_secs: set[float] = set()
    length = max(end - start, 0.01)

    def window(frac_start: float, frac_end: float) -> tuple[float, float, float]:
        window_start = start + length * frac_start
        window_end = start + length * frac_end
        target = (window_start + window_end) / 2.0
        return window_start, window_end, target

    if target_count == 1:
        middle_start, middle_end, middle_target = window(0.2, 0.8)
        candidate = _pick_best_sample_for_window(
            segment_samples,
            middle_start,
            middle_end,
            target_sec=middle_target,
            excluded_secs=excluded_secs,
        )
        _append_selected_frame(selected_frames, segment_id, candidate, "middle", start=start, end=end, excluded_secs=excluded_secs)
        return selected_frames

    if target_count == 2:
        middle_start, middle_end, middle_target = window(0.25, 0.75)
        middle = _pick_best_sample_for_window(
            segment_samples,
            middle_start,
            middle_end,
            target_sec=middle_target,
            excluded_secs=excluded_secs,
        )
        _append_selected_frame(selected_frames, segment_id, middle, "middle", start=start, end=end, excluded_secs=excluded_secs)
        supplemental_roles = ["dense_text", "action_peak", "result_state"]
        supplemental = None
        supplemental_role = "action_peak"
        for role in supplemental_roles:
            candidate = _pick_role_specific_sample(
                segment_samples,
                role,
                start=start,
                end=end,
                excluded_secs=excluded_secs,
            )
            if candidate is None:
                continue
            if middle is not None and abs(float(candidate.get("sec", 0.0)) - float(middle.get("sec", 0.0))) < FRAME_DEDUP_MIN_GAP_SEC:
                continue
            supplemental = candidate
            supplemental_role = role
            break
        _append_selected_frame(
            selected_frames,
            segment_id,
            supplemental,
            supplemental_role,
            start=start,
            end=end,
            excluded_secs=excluded_secs,
        )
        return sorted(selected_frames, key=lambda item: float(item["frame_second"]))

    base_roles = [
        ("front", *window(0.0, 0.34)),
        ("middle", *window(0.33, 0.67)),
        ("back", *window(0.66, 1.0)),
    ]
    for role, window_start, window_end, target in base_roles:
        candidate = _pick_best_sample_for_window(
            segment_samples,
            window_start,
            window_end,
            target_sec=target,
            excluded_secs=excluded_secs,
        )
        _append_selected_frame(selected_frames, segment_id, candidate, role, start=start, end=end, excluded_secs=excluded_secs)

    supplemental_role_order: list[str] = []
    triggers = set(plan.get("upsampling_triggers") or [])
    if "TEXT_CHANGE_DENSE" in triggers:
        supplemental_role_order.append("dense_text")
    if "ACTION_STATE_CHANGE" in triggers or "FOCUS_SWITCH" in triggers:
        supplemental_role_order.append("action_peak")
    if "RESULT_STATE" in triggers:
        supplemental_role_order.append("result_state")
    for fallback_role in ("dense_text", "action_peak", "result_state"):
        if fallback_role not in supplemental_role_order:
            supplemental_role_order.append(fallback_role)

    extras_needed = max(target_count - 3, 0)
    for role in supplemental_role_order:
        if extras_needed <= 0:
            break
        candidate = _pick_role_specific_sample(
            segment_samples,
            role,
            start=start,
            end=end,
            excluded_secs=excluded_secs,
        )
        if candidate is None:
            continue
        _append_selected_frame(selected_frames, segment_id, candidate, role, start=start, end=end, excluded_secs=excluded_secs)
        if len(selected_frames) > 3:
            extras_needed -= 1

    if len(selected_frames) < target_count:
        remaining = [
            item for item in sorted(segment_samples, key=lambda sample: float(sample.get("score", 0.0)), reverse=True)
            if round(float(item.get("sec", 0.0)), 3) not in excluded_secs
        ]
        for sample in remaining:
            if len(selected_frames) >= target_count:
                break
            _append_selected_frame(
                selected_frames,
                segment_id,
                sample,
                "action_peak",
                start=start,
                end=end,
                excluded_secs=excluded_secs,
            )

    return sorted(selected_frames, key=lambda item: float(item["frame_second"]))



def _primary_frame_from_segment_frames(frames: list[dict]) -> dict:
    if not frames:
        return {"frame_second": 0.0, "sampling_role": "middle"}
    by_role = {item.get("sampling_role"): item for item in frames}
    return by_role.get("middle") or by_role.get("front") or frames[0]



def _build_segments_from_boundaries(boundaries: list[float], samples: list[dict]) -> tuple[list[dict], set[float], list[dict]]:
    segments: list[dict] = []
    selected_secs: set[float] = set()
    selection_decisions: list[dict] = []

    for shot_idx in range(len(boundaries) - 1):
        shot_start = round(boundaries[shot_idx], 3)
        shot_end = round(boundaries[shot_idx + 1], 3)
        if shot_end - shot_start <= 0.0:
            continue
        shot_id = f"SHOT{shot_idx + 1:02d}"
        segment_id = f"SEG{len(segments) + 1:02d}"
        segment_samples = _collect_segment_samples(samples, shot_start, shot_end)
        plan = _determine_frame_plan(segment_samples, shot_start, shot_end)
        frames = _build_segment_frames(segment_id, segment_samples, shot_start, shot_end, plan)
        primary = _primary_frame_from_segment_frames(frames)

        segment = {
            "segment_id": segment_id,
            "shot_id": shot_id,
            "start_sec": shot_start,
            "end_sec": shot_end,
            "frame_second": _clamp_frame_second(float(primary.get("frame_second", shot_start)), shot_start, shot_end),
            "segment_strategy": "shot_multiframe_coverage",
            "frames": frames,
            "frame_plan": {
                "default_frame_count": plan["default_frame_count"],
                "final_frame_count": len(frames),
                "final_frame_count_before_budget": plan["final_frame_count"],
                "upsampling_triggers": plan["upsampling_triggers"],
                "downgrade_reason": plan["downgrade_reason"],
                "min_keep_frames": plan["min_keep_frames"],
                "metrics": plan["metrics"],
            },
        }
        if primary.get("representative_score") is not None:
            segment["representative_score"] = primary.get("representative_score")
        segments.append(segment)
        selected_secs.update(round(float(frame["frame_second"]), 3) for frame in frames)

        selection_decisions.append(
            {
                "reason_code": "DEFAULT_FRAME_COUNT_DECISION",
                "segment_id": segment_id,
                "shot_id": shot_id,
                "shot_start": shot_start,
                "shot_end": shot_end,
                "shot_len": round(shot_end - shot_start, 3),
                "default_frame_count": plan["default_frame_count"],
                "final_frame_count_before_budget": plan["final_frame_count"],
                "stable_short_shot": bool(plan["metrics"].get("stable_short_shot")),
                "downgrade_reason": plan["downgrade_reason"],
                "metrics": plan["metrics"],
            }
        )
        selection_decisions.append(
            {
                "reason_code": "UPSAMPLING_TRIGGER_STATUS",
                "segment_id": segment_id,
                "shot_id": shot_id,
                "upsampling_triggers": plan["upsampling_triggers"],
                "upsampled": bool(plan["upsampling_triggers"]),
            }
        )
        selection_decisions.append(
            {
                "reason_code": "FRAME_SELECTION_APPLIED",
                "segment_id": segment_id,
                "shot_id": shot_id,
                "frame_count": len(frames),
                "frames": [
                    {
                        "frame_id": frame["frame_id"],
                        "frame_second": frame["frame_second"],
                        "sampling_role": frame["sampling_role"],
                    }
                    for frame in frames
                ],
            }
        )
    return segments, selected_secs, selection_decisions



def _apply_frame_budget(segments: list[dict], max_frames: int = MAX_FRAMES_CAP) -> tuple[list[dict], list[dict]]:
    decisions: list[dict] = []
    total_before = sum(len(segment.get("frames") or []) for segment in segments)
    total_after = total_before

    decisions.append(
        {
            "reason_code": "FRAME_BUDGET_STATUS",
            "max_frames": max_frames,
            "total_frames_before_budget": total_before,
            "total_frames_after_budget": total_after,
            "budget_recovery_count": 0,
            "budget_exceeded": total_before > max_frames,
        }
    )

    if total_before <= max_frames:
        return segments, decisions

    recovery_count = 0
    while total_after > max_frames:
        candidate_index = None
        candidate_priority = None
        for idx, segment in enumerate(segments):
            frames = segment.get("frames") or []
            plan = segment.get("frame_plan") or {}
            min_keep = int(plan.get("min_keep_frames") or 1)
            if len(frames) <= min_keep:
                continue
            metrics = (plan.get("metrics") or {})
            priority = (
                0 if int(plan.get("default_frame_count") or 1) <= 2 else 1,
                0 if not plan.get("upsampling_triggers") else 1,
                float(metrics.get("avg_text_density", 0.0)),
                float(metrics.get("avg_diff_score", 0.0)),
                len(frames),
                idx,
            )
            if candidate_priority is None or priority < candidate_priority:
                candidate_priority = priority
                candidate_index = idx
        if candidate_index is None:
            break

        segment = segments[candidate_index]
        frames = list(segment.get("frames") or [])
        removable_indexes = [
            idx
            for idx, frame in enumerate(frames)
            if frame.get("sampling_role") not in {"front", "middle", "back"}
        ]
        remove_idx = removable_indexes[-1] if removable_indexes else len(frames) - 1
        removed = frames.pop(remove_idx)
        segment["frames"] = frames
        primary = _primary_frame_from_segment_frames(frames)
        segment["frame_second"] = float(primary.get("frame_second", segment["start_sec"]))
        segment["frame_plan"]["final_frame_count"] = len(frames)
        total_after -= 1
        recovery_count += 1
        decisions.append(
            {
                "reason_code": "BUDGET_RECOVERY_ACTION",
                "segment_id": segment["segment_id"],
                "shot_id": segment["shot_id"],
                "removed_frame_id": removed.get("frame_id"),
                "removed_frame_second": removed.get("frame_second"),
                "removed_sampling_role": removed.get("sampling_role"),
                "remaining_frame_count": len(frames),
                "recovery_strategy": "simple_shot_compress" if int(segment.get("frame_plan", {}).get("default_frame_count") or 1) <= 2 else "supplemental_frame_trim",
            }
        )

    decisions[0]["total_frames_after_budget"] = total_after
    decisions[0]["budget_recovery_count"] = recovery_count
    decisions[0]["budget_exceeded"] = total_before > max_frames
    return segments, decisions



def _tail_candidate_window_sec(duration: float) -> float:
    return round(max(TAIL_MIN_WINDOW_SEC, min(TAIL_MAX_WINDOW_SEC, duration * TAIL_RATIO)), 3)



def _annotate_segment_types(segments: list[dict], samples: list[dict], duration: float) -> tuple[list[dict], list[dict]]:
    if not segments:
        return [], []

    tail_window_sec = _tail_candidate_window_sec(duration)
    tail_start_sec = round(max(0.0, duration - tail_window_sec), 3)
    decisions: list[dict] = [
        {
            "reason_code": "TAIL_RULES_APPLIED",
            "tail_ratio": TAIL_RATIO,
            "tail_candidate_window_sec": tail_window_sec,
            "tail_start_sec": tail_start_sec,
            "long_video_tail_required_sec": LONG_VIDEO_TAIL_REQUIRED_SEC,
        }
    ]

    if len(segments) == 1 or duration < TAIL_MIN_WINDOW_SEC:
        segments[0]["segment_type"] = "main"
        decisions.append(
            {
                "reason_code": "TAIL_SEGMENT_RETAINED_MAIN",
                "segment_id": segments[0]["segment_id"],
                "tail_candidate": False,
                "reason": "single_segment_or_short_video",
            }
        )
        decisions.append(
            {
                "reason_code": "TAIL_STATUS_RECORDED",
                "has_tail": False,
                "tail_segment_ids": [],
            }
        )
        return segments, decisions

    for segment in segments:
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        segment_samples = _collect_segment_samples(samples, start_sec, end_sec)
        tail_candidate = end_sec > tail_start_sec
        max_text_density = max((float(sample.get("text_density", 0.0)) for sample in segment_samples), default=0.0)
        max_text_boxes = max((int(sample.get("text_boxes", 0)) for sample in segment_samples), default=0)
        avg_diff_score = round(
            sum(float(sample.get("diff_score", 0.0)) for sample in segment_samples) / max(len(segment_samples), 1),
            4,
        )
        avg_hist_diff = round(
            sum(float(sample.get("hist_diff", 0.0)) for sample in segment_samples) / max(len(segment_samples), 1),
            4,
        )
        pattern_hits: list[str] = []
        if max_text_density >= TAIL_TEXT_DENSITY_THRESHOLD:
            pattern_hits.append("TEXT_DENSITY")
        if max_text_boxes >= TAIL_TEXT_BOXES_THRESHOLD:
            pattern_hits.append("TEXT_BOXES")
        if (
            avg_diff_score <= TAIL_STATIC_FRAME_DIFF_THRESHOLD
            and avg_hist_diff <= TAIL_STATIC_HIST_DIFF_THRESHOLD
            and (
                max_text_density >= TAIL_STATIC_TEXT_DENSITY_THRESHOLD
                or max_text_boxes >= TAIL_STATIC_TEXT_BOXES_THRESHOLD
            )
        ):
            pattern_hits.append("STATIC_ENDCARD")

        is_tail = tail_candidate and bool(pattern_hits)
        segment["segment_type"] = "tail" if is_tail else "main"
        decisions.append(
            {
                "reason_code": "TAIL_SEGMENT_CLASSIFIED" if is_tail else "TAIL_SEGMENT_RETAINED_MAIN",
                "segment_id": segment["segment_id"],
                "segment_start_sec": round(start_sec, 3),
                "segment_end_sec": round(end_sec, 3),
                "tail_candidate": tail_candidate,
                "pattern_hits": pattern_hits,
                "max_text_density": round(max_text_density, 6),
                "max_text_boxes": max_text_boxes,
                "avg_diff_score": avg_diff_score,
                "avg_hist_diff": avg_hist_diff,
                "sample_reason_codes": sorted(
                    {
                        code
                        for sample in segment_samples
                        for code in sample.get("reason_codes", [])
                        if isinstance(code, str) and code
                    }
                ),
            }
        )

    tail_indexes = [idx for idx, segment in enumerate(segments) if segment.get("segment_type") == "tail"]
    if tail_indexes:
        first_tail_idx = tail_indexes[0]
        if first_tail_idx == 0:
            for segment in segments:
                segment["segment_type"] = "main"
            decisions.append(
                {
                    "reason_code": "TAIL_ALL_SEGMENTS_REVERTED_TO_MAIN",
                    "reason": "tail_suffix_must_keep_at_least_one_main_segment",
                }
            )
        else:
            forced_tail_ids: list[str] = []
            for segment in segments[first_tail_idx:]:
                if segment.get("segment_type") != "tail":
                    segment["segment_type"] = "tail"
                    forced_tail_ids.append(segment["segment_id"])
            if forced_tail_ids:
                decisions.append(
                    {
                        "reason_code": "TAIL_SUFFIX_FORCED_CONTIGUOUS",
                        "tail_segment_ids": forced_tail_ids,
                    }
                )

    tail_segment_ids = [segment["segment_id"] for segment in segments if segment.get("segment_type") == "tail"]
    decisions.append(
        {
            "reason_code": "TAIL_STATUS_RECORDED",
            "has_tail": bool(tail_segment_ids),
            "tail_segment_ids": tail_segment_ids,
        }
    )
    return segments, decisions



def _build_loser_decisions(
    ranked: list[dict],
    *,
    selected_secs: set[float],
    initial_score_by_sec: dict[float, float],
    final_rank_positions: dict[float, int],
    rescored_secs: set[float],
) -> list[dict]:
    loser_decisions: list[dict] = []
    for sample in ranked[: min(20, len(ranked))]:
        sec = round(float(sample.get("sec", 0.0)), 3)
        if sec in selected_secs:
            continue
        reason_code = "DROPPED_AFTER_OCR_RESCORING" if sec in rescored_secs else "LOWER_RANK_NOT_SELECTED"
        extra_fields: dict = {}
        if sec in rescored_secs:
            extra_fields["rank_after_ocr"] = final_rank_positions.get(sec)
        loser_decisions.append(
            {
                "reason_code": reason_code,
                "sec": sec,
                "score": sample.get("score"),
                "initial_score": round(initial_score_by_sec.get(sec, float(sample.get("score", 0.0))), 4),
                "diff_score": sample.get("diff_score"),
                "hist_diff": sample.get("hist_diff"),
                "text_density": sample.get("text_density"),
                "text_boxes": sample.get("text_boxes"),
                "reason_codes": sample.get("reason_codes", []),
                **extra_fields,
            }
        )
    return loser_decisions



def _build_segments(
    video_path: str,
    duration: float,
    enable_real_ocr_feedback: bool = False,
    ocr_feedback_top_k: int = 4,
    probe_dir: str = "",
    runtime_governance: ProviderRuntimeGovernance | None = None,
    provider_fallback_mode: str = "force_off",
) -> tuple[list[dict], list[dict], list[dict]]:
    samples, sample_debug = _collect_frame_samples(video_path, duration, probe_dir=probe_dir)
    if not samples:
        segments = _build_uniform_segments(duration)
        segments, budget_decisions = _apply_frame_budget(segments, MAX_FRAMES_CAP)
        segments, tail_decisions = _annotate_segment_types(segments, samples, duration)
        has_tail = any(segment.get("segment_type") == "tail" for segment in segments)
        total_selected_frames = sum(len(segment.get("frames") or []) for segment in segments)
        summary = {
            "strategy": "uniform_fallback",
            "selected_cut_point_count": 0,
            "shot_count": len(segments),
            "final_segment_count": len(segments),
            "ocr_feedback_enabled": enable_real_ocr_feedback,
            "ocr_hit_count": 0,
            "long_shot_split_count": 0,
            "tail_segment_count": sum(1 for segment in segments if segment.get("segment_type") == "tail"),
            "has_tail": has_tail,
            "total_candidates": 0,
            "total_selected_frames": total_selected_frames,
            "max_frames": MAX_FRAMES_CAP,
            "budget_recovery_count": budget_decisions[0].get("budget_recovery_count", 0),
            "raw_boundary_hit_count": 0,
            "hard_boundary_hit_count": 0,
            "soft_cut_recall_count": 0,
            "boundary_cluster_count": 0,
            "protected_representative_count": 0,
            "representative_boundary_count": 0,
            "semantic_merge_count": 0,
            "final_cut_point_count": 0,
            "cluster_window_sec": BOUNDARY_CLUSTER_WINDOW_SEC,
            "soft_cut_window_sec": max(BOUNDARY_PEAK_WINDOW_SEC, 1.0),
            **sample_debug,
        }
        decision_report = [
            {
                "reason_code": "FALLBACK_UNIFORM",
                "decision": "fallback_uniform",
                "reason": "unreadable_or_empty_video_samples",
                "cut_points": [],
            },
            {"reason_code": "DECISION_SUMMARY", "summary": summary},
            {"reason_code": "SHOT_BOUNDARY_MERGE_STATUS", "semantic_merge_count": 0, "merge_actions": 0, "merge_block_count": 0},
            *budget_decisions,
            *tail_decisions,
        ]
        return segments, [], decision_report

    ranked, initial_score_by_sec, final_rank_positions, rescored_secs = _rank_samples_with_ocr_feedback(
        samples,
        enable_real_ocr_feedback=enable_real_ocr_feedback,
        ocr_feedback_top_k=ocr_feedback_top_k,
        runtime_governance=runtime_governance,
        provider_fallback_mode=provider_fallback_mode,
    )
    raw_boundary_hits, boundary_hit_decisions, threshold_debug = _collect_raw_boundary_hits(samples, duration)
    clusters, cluster_decisions = _build_boundary_clusters(raw_boundary_hits, BOUNDARY_CLUSTER_WINDOW_SEC)
    representative_boundaries, representative_decisions = _select_cluster_representatives(clusters)
    representative_cut_points = [round(float(item.get("sec", 0.0)), 3) for item in representative_boundaries]
    protected_representative_secs = {
        round(float(item.get("sec", 0.0)), 3) for item in representative_boundaries if item.get("protected")
    }
    provisional_boundaries = [0.0] + representative_cut_points + [round(duration, 3)]
    semantic_boundaries, semantic_merge_decisions, semantic_merge_count = _merge_semantic_continuity(
        provisional_boundaries,
        samples,
        duration,
        boundary_representatives=representative_boundaries,
    )
    boundaries, short_merge_decisions = _merge_short_segments(
        semantic_boundaries,
        duration,
        protected_boundary_secs=protected_representative_secs,
    )
    cut_points = [round(sec, 3) for sec in boundaries[1:-1]]
    segments, selected_secs, selection_decisions = _build_segments_from_boundaries(boundaries, ranked)
    segments, budget_decisions = _apply_frame_budget(segments, MAX_FRAMES_CAP)
    selected_secs = {
        round(float(frame["frame_second"]), 3)
        for segment in segments
        for frame in segment.get("frames") or []
    }
    segments, tail_decisions = _annotate_segment_types(segments, ranked, duration)
    loser_decisions = _build_loser_decisions(
        ranked,
        selected_secs=selected_secs,
        initial_score_by_sec=initial_score_by_sec,
        final_rank_positions=final_rank_positions,
        rescored_secs=rescored_secs,
    )

    if not segments:
        segments = _build_uniform_segments(duration)
        segments, budget_decisions = _apply_frame_budget(segments, MAX_FRAMES_CAP)
        segments, tail_decisions = _annotate_segment_types(segments, ranked, duration)
        has_tail = any(segment.get("segment_type") == "tail" for segment in segments)
        total_selected_frames = sum(len(segment.get("frames") or []) for segment in segments)
        summary = {
            "strategy": "uniform_fallback",
            "selected_cut_point_count": len(cut_points),
            "shot_count": len(segments),
            "final_segment_count": len(segments),
            "ocr_feedback_enabled": enable_real_ocr_feedback,
            "ocr_hit_count": len(rescored_secs),
            "long_shot_split_count": 0,
            "tail_segment_count": sum(1 for segment in segments if segment.get("segment_type") == "tail"),
            "has_tail": has_tail,
            "total_candidates": len(samples),
            "total_selected_frames": total_selected_frames,
            "max_frames": MAX_FRAMES_CAP,
            "budget_recovery_count": budget_decisions[0].get("budget_recovery_count", 0),
            "raw_boundary_hit_count": len(raw_boundary_hits),
            "hard_boundary_hit_count": int(threshold_debug.get("hard_boundary_hit_count", 0)),
            "soft_cut_recall_count": int(threshold_debug.get("soft_cut_recall_count", 0)),
            "boundary_cluster_count": len(clusters),
            "protected_representative_count": sum(1 for item in representative_boundaries if item.get("protected")),
            "representative_boundary_count": len(representative_boundaries),
            "semantic_merge_count": semantic_merge_count,
            "final_cut_point_count": len(cut_points),
            "cluster_window_sec": BOUNDARY_CLUSTER_WINDOW_SEC,
            "soft_cut_window_sec": float(threshold_debug.get("soft_cut_window_sec", max(BOUNDARY_PEAK_WINDOW_SEC, 1.0))),
            **sample_debug,
            **threshold_debug,
        }
        decision_report = [
            {
                "reason_code": "FALLBACK_UNIFORM",
                "decision": "fallback_uniform",
                "reason": "empty_segments_after_boundary_build",
                "cut_points": cut_points,
            },
            {"reason_code": "DECISION_SUMMARY", "summary": summary},
            *boundary_hit_decisions,
            *cluster_decisions,
            *representative_decisions,
            *semantic_merge_decisions,
            *short_merge_decisions,
            *budget_decisions,
            *tail_decisions,
            *loser_decisions,
        ]
        return segments, ranked[: min(20, len(ranked))], decision_report

    candidate_scores = []
    selected_cut_points = {round(float(sec), 3) for sec in cut_points}
    for sample in ranked[: min(20, len(ranked))]:
        sec = round(float(sample.get("sec", 0.0)), 3)
        item = dict(sample)
        item["selected_as_cut_point"] = sec in selected_cut_points
        item["selected_as_frame"] = sec in selected_secs
        candidate_scores.append(item)

    has_tail = any(segment.get("segment_type") == "tail" for segment in segments)
    total_selected_frames = sum(len(segment.get("frames") or []) for segment in segments)
    summary = {
        "strategy": "shot_multiframe_coverage",
        "selected_cut_point_count": len(cut_points),
        "shot_count": max(0, len(boundaries) - 1),
        "final_segment_count": len(segments),
        "ocr_feedback_enabled": enable_real_ocr_feedback,
        "ocr_hit_count": len(rescored_secs),
        "long_shot_split_count": 0,
        "tail_segment_count": sum(1 for segment in segments if segment.get("segment_type") == "tail"),
        "has_tail": has_tail,
        "total_candidates": len(samples),
        "total_selected_frames": total_selected_frames,
        "max_frames": MAX_FRAMES_CAP,
        "budget_recovery_count": budget_decisions[0].get("budget_recovery_count", 0),
        "raw_boundary_hit_count": len(raw_boundary_hits),
        "hard_boundary_hit_count": int(threshold_debug.get("hard_boundary_hit_count", 0)),
        "soft_cut_recall_count": int(threshold_debug.get("soft_cut_recall_count", 0)),
        "boundary_cluster_count": len(clusters),
        "protected_representative_count": sum(1 for item in representative_boundaries if item.get("protected")),
        "representative_boundary_count": len(representative_boundaries),
        "semantic_merge_count": semantic_merge_count,
        "final_cut_point_count": len(cut_points),
        "cluster_window_sec": BOUNDARY_CLUSTER_WINDOW_SEC,
        "soft_cut_window_sec": float(threshold_debug.get("soft_cut_window_sec", max(BOUNDARY_PEAK_WINDOW_SEC, 1.0))),
        **sample_debug,
        **threshold_debug,
    }
    decision_report = [
        {
            "reason_code": "SELECTED_CUT_POINTS",
            "decision": "selected_cut_points",
            "cut_points": cut_points,
            "boundaries_after_merge": boundaries,
            "selected_frame_seconds": sorted(selected_secs),
            "raw_boundary_hit_secs": [round(float(item.get("sec", 0.0)), 3) for item in raw_boundary_hits],
            "representative_cut_points": representative_cut_points,
            "protected_representative_secs": sorted(protected_representative_secs),
            "provisional_boundaries": provisional_boundaries,
        },
        {"reason_code": "DECISION_SUMMARY", "summary": summary},
        *boundary_hit_decisions,
        *cluster_decisions,
        *representative_decisions,
        *semantic_merge_decisions,
        *selection_decisions,
        *short_merge_decisions,
        *budget_decisions,
        *tail_decisions,
        *loser_decisions,
    ]
    return segments, candidate_scores, decision_report



def run_preprocess(
    video_path: str,
    workspace_dir: str,
    ffmpeg_path: str,
    ffprobe_path: str,
    source_platform: str,
    enable_real_ocr_feedback: bool = False,
    ocr_feedback_top_k: int = 4,
    provider_runtime_max_retries: int = 2,
    provider_runtime_backoff_sec: int = 2,
    provider_runtime_max_requests_per_run: int = 0,
    provider_fallback_mode: str = "force_off",
) -> dict:
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    audio_path = workspace / "audio.wav"
    meta = probe_video_meta(video_path, ffprobe_path, source_platform)
    extract_audio(video_path, str(audio_path), ffmpeg_path)
    ocr_feedback_runtime = ProviderRuntimeGovernance(
        workspace_dir=str(workspace),
        provider_name="preprocess_ocr_feedback",
        max_retries=provider_runtime_max_retries,
        backoff_sec=provider_runtime_backoff_sec,
        max_requests_per_run=provider_runtime_max_requests_per_run,
    )

    segments, candidate_scores, decision_report = _build_segments(
        video_path,
        float(meta["duration_sec"]),
        enable_real_ocr_feedback=enable_real_ocr_feedback,
        ocr_feedback_top_k=ocr_feedback_top_k,
        probe_dir=str(workspace / "probe_frames"),
        runtime_governance=ocr_feedback_runtime,
        provider_fallback_mode=provider_fallback_mode,
    )
    frame_paths: list[str] = []
    frame_seconds: list[float] = []
    for seg in segments:
        for frame in seg.get("frames") or []:
            frame_path = workspace / f"{str(frame['frame_id']).lower()}.jpg"
            extract_frame(video_path, str(frame_path), float(frame["frame_second"]), ffmpeg_path)
            frame["frame_path"] = str(frame_path)
            frame_paths.append(str(frame_path))
            frame_seconds.append(float(frame["frame_second"]))

        primary = _primary_frame_from_segment_frames(seg.get("frames") or [])
        seg["frame_second"] = float(primary.get("frame_second", seg["start_sec"]))
        seg["frame_path"] = str(primary.get("frame_path") or "")

    has_tail = any(seg.get("segment_type") == "tail" for seg in segments)
    payload = {
        "video_path": video_path,
        "video_meta": meta,
        "audio_path": str(audio_path),
        "frame_paths": frame_paths,
        "frame_seconds": frame_seconds,
        "has_tail": has_tail,
        "segments": segments,
        "candidate_scores": candidate_scores,
        "decision_report": decision_report,
        "runtime_governance": {
            "preprocess_ocr_feedback": ocr_feedback_runtime.state.to_dict(),
        },
    }
    (workspace / "preprocess.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (workspace / "decision_report.json").write_text(json.dumps({"decision_report": decision_report, "candidate_scores": candidate_scores}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
