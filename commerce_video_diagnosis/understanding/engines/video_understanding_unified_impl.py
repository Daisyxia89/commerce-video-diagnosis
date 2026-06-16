from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from commerce_video_diagnosis.understanding.engines.product_diagnosis_engine import DiagnosticInput, ProductDiagnosisEngine, _build_price_band_lookup
from commerce_video_diagnosis.understanding.engines.product_diagnoser import DIFFERENTIATOR_DOMAIN_TYPES
from commerce_video_diagnosis.understanding.engines.triad_asset_repository import (
    TriadAssetPersistenceError,
    TriadAssetPersistenceSummary,
    TriadAssetRepository,
    build_blueprint_id,
    build_segment_record_id,
)
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from commerce_video_diagnosis.understanding.llm_provider import build_chat_headers, require_llm_config, resolve_llm_config

class ProtocolViolation(RuntimeError):
    """输入协议违规：必须 Crash Early。"""


class StrictBaseModel(BaseModel):
    """所有协议对象默认禁止 extra 字段。

    PRD v2 明确要求：不做运行时向下兼容；污染字段必须直接阻断。
    """

    class Config:
        extra = "forbid"


def _utc_now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _flatten_keys(value: Any, *, prefix: str = "") -> list[str]:
    """递归收集所有 key path，用于检测答案字段藏匿。"""
    keys: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            keys.append(path)
            keys.extend(_flatten_keys(v, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            keys.extend(_flatten_keys(item, prefix=f"{prefix}[{idx}]" if prefix else f"[{idx}]"))
    return keys


def _find_forbidden_field_paths(payload: Any, forbidden_keys: set[str]) -> list[str]:
    hit_paths: list[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k) in forbidden_keys:
                hit_paths.append(str(k))
            for nested in _find_forbidden_field_paths(v, forbidden_keys):
                hit_paths.append(f"{k}.{nested}")
        return hit_paths
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            for nested in _find_forbidden_field_paths(item, forbidden_keys):
                hit_paths.append(f"[{i}].{nested}")
        return hit_paths
    return []


FORBIDDEN_ANSWER_KEYS_IN_FACTPACK = {
    # 全局答案
    "primary_hec",
    "slider_signature",
    "segment_tags",
    "triad_assets",
    # 策略/意图
    "jtbd",
    "original_jtbd",
    "original_jtbd",
    "category_strategy_intent",
    "product_strategy_intent",
    "weapon_tags",
    # 分镜业务标签答案
    "hook_label",
    "effect_label",
    "cta_label",
    # 字段藏匿典型容器
    "metadata_overrides",
    "extensions",
    "metadata_overrides",
}

DEPRECATED_OLD_FIELDS = {
    "blueprint_path",
    "blueprint_payload",
    "product_diagnosis_payload",
    "raw_blueprint",
    "raw_blueprint_payload",
}

HOOK_ENUM = {
    "H1 痛点/焦虑直击",
    "H2 利益/价格前置",
    "H3 反差结果前置",
    "H4 即时操作展示",
    "H5 反常识与悬念",
    "H6 场景/人群代入",
    "H7 明星/权威同款",
}
EFFECT_ENUM = {
    "E0 单点演示",
    "E1 效果测评",
    "E2 暴力实测",
    "E3 对比/拉踩",
    "E4 感官实证",
    "E5 保姆级教程",
    "E6 成分/参数科普",
    "E7 产地溯源/工厂实录",
}
CTA_ENUM = {
    "C1 利益/价格逼单",
    "C2 福利/保障机制",
    "C3 指令行动",
    "C4 人群/场景总结",
    "C5 效果留白/情绪定格",
}
SECONDARY_EFFECT_LABELS = {f"E{i}" for i in range(8)}


class BlueprintBuilder:
    """保留给离线分析测试使用的最小辅助类；不参与任何运行时兼容。"""

    def _derive_emotional_tone(self, segment: dict[str, Any], visual_description: str, spoken_lines: str) -> str:
        text = " ".join(
            str(value or "")
            for value in (
                segment.get("persuasion_function"),
                visual_description,
                spoken_lines,
            )
        )
        if any(keyword in text for keyword in ("节日", "送礼", "礼物", "惊喜", "仪式感")):
            return "营造温暖、惊喜或节日仪式感。"
        if any(keyword in text for keyword in ("结果", "证明", "对比", "擦净", "利落")):
            return "强调结果可信与动作利落。"
        return "自然克制，保持客观说明。"


def _summarize_action_names(actions: list[dict[str, Any]]) -> str:
    names = [str(action.get("action_name") or "").strip() for action in actions if str(action.get("action_name") or "").strip()]
    return "、".join(names)



def _derive_action_intensity(seg: FactPackSegment) -> str:
    intensities = [str(action.get("physical_intensity") or "").strip().lower() for action in seg.visual_facts.actions]
    if any(level in {"high", "strong", "violent"} for level in intensities):
        return "高强度，动作发力明确，节奏压强高。"
    if any(level in {"medium", "mid", "moderate"} for level in intensities):
        return "中强度，动作清晰，存在明确推进感。"
    if any(level in {"low", "light", "gentle"} for level in intensities):
        return "低强度，动作克制，以稳定展示为主。"
    return "中低强度，动作自然，按信息表达平稳推进。"



def _derive_action_mechanics(seg: FactPackSegment) -> str:
    action_summary = _summarize_action_names(seg.visual_facts.actions)
    if action_summary:
        return f"围绕{action_summary}展开，按镜头信息点逐步推进。"
    if seg.visual_facts.camera_movement:
        return f"以{seg.visual_facts.camera_movement}镜头配合主体展示推进信息。"
    return "围绕主体展示与口播说明做自然动作推进。"



def _derive_emotion_tension(seg: FactPackSegment) -> str:
    bgm_tones = " ".join(event.tone for event in seg.audio_facts.bgm_events)
    sfx_names = " ".join(event.event_name for event in seg.audio_facts.sfx_events)
    text = " ".join(
        part
        for part in (
            seg.audio_facts.asr_text,
            bgm_tones,
            sfx_names,
            seg.visual_facts.camera_movement,
            seg.visual_facts.lighting_tone,
            _summarize_action_names(seg.visual_facts.actions),
            seg.rhythm_facts.pace_marker,
        )
        if part
    )
    lowered = text.lower()
    intensities = [str(action.get("physical_intensity") or "").strip().lower() for action in seg.visual_facts.actions]
    if any(token in text for token in ("爆表", "至于吗", "快看", "立刻", "马上", "撑住", "扛住")) or any(
        token in lowered for token in ("tense", "urgent", "fast", "push_in")
    ) or any(level in {"high", "strong", "violent"} for level in intensities):
        return "张力偏高，表演要绷紧并持续向结果逼近。"
    if any(token in text for token in ("对比", "结果", "证明", "终于", "一下", "看清")) or any(
        token in lowered for token in ("medium", "moderate", "normal")
    ) or any(level in {"medium", "mid", "moderate"} for level in intensities):
        return "张力中等，先稳住信息，再把注意力推向关键结果。"
    return "张力偏低，以克制、平稳的表达承接信息。"



def _derive_performance_emotion(seg: FactPackSegment, persuasion_function: str) -> dict[str, str]:
    action_mechanics = _derive_action_mechanics(seg)
    action_intensity = _derive_action_intensity(seg)
    action_summary = _summarize_action_names(seg.visual_facts.actions)
    acting_focus = action_summary or seg.visual_facts.visual_subject or "主体展示"
    acting_instructions = f"围绕{acting_focus}做清晰表演，服务于{persuasion_function}。"
    emotional_tone = BlueprintBuilder()._derive_emotional_tone(
        {"persuasion_function": persuasion_function},
        seg.visual_facts.visual_subject,
        seg.audio_facts.asr_text,
    )
    return {
        "acting_instructions": acting_instructions,
        "emotion_tension": _derive_emotion_tension(seg),
        "emotional_tone": emotional_tone,
        "action_mechanics": action_mechanics,
        "action_intensity": action_intensity,
    }



def _build_segment_source_evidence(seg: FactPackSegment) -> list[str]:
    evidence: list[str] = []
    if str(seg.audio_facts.asr_text).strip():
        evidence.append(f"{seg.segment_id}.audio_facts.asr_text")
    for index, _ in enumerate(seg.ocr_facts):
        evidence.append(f"{seg.segment_id}.ocr_facts[{index}].text")
    for index, action in enumerate(seg.visual_facts.actions):
        if str(action.get("action_name") or "").strip():
            evidence.append(f"{seg.segment_id}.visual_facts.actions[{index}].action_name")
    for index, _ in enumerate(seg.audio_facts.sfx_events):
        evidence.append(f"{seg.segment_id}.audio_facts.sfx_events[{index}]")
    for index, _ in enumerate(seg.audio_facts.bgm_events):
        evidence.append(f"{seg.segment_id}.audio_facts.bgm_events[{index}]")
    if str(seg.rhythm_facts.pace_marker).strip():
        evidence.append(f"{seg.segment_id}.rhythm_facts.pace_marker")
    if str(seg.visual_facts.visual_subject).strip():
        evidence.append(f"{seg.segment_id}.visual_facts.visual_subject")
    return evidence



def _derive_audio_business_role(local_hec_tag: str, persuasion_function: str, event_kind: Literal["sfx", "bgm"]) -> str:
    if local_hec_tag.startswith("C"):
        stage_role = "收口推进"
    elif local_hec_tag.startswith("H"):
        stage_role = "开场造势"
    elif "辅助举证" in persuasion_function:
        stage_role = "辅助举证"
    elif local_hec_tag.startswith("E"):
        stage_role = "结果举证"
    else:
        stage_role = "信息承接"
    detail = "音效触发点" if event_kind == "sfx" else "BGM 节奏切换"
    return f"{stage_role}：承担{detail}与情绪推进。"



def _derive_audio_event_projection(
    seg: FactPackSegment,
    *,
    local_hec_tag: str,
    persuasion_function: str,
) -> dict[str, Any]:
    shared_evidence = _build_segment_source_evidence(seg)
    return AudioEventProjection.parse_obj(
        {
            "sfx_events": [
                {
                    "event_name": event.event_name,
                    "start_sec": event.start_sec,
                    "end_sec": event.end_sec,
                    "trigger_sec": event.start_sec,
                    "business_role": _derive_audio_business_role(local_hec_tag, persuasion_function, "sfx"),
                    "source_evidence": shared_evidence
                    or [f"{seg.segment_id}.audio_facts.sfx_events[{index}]"] ,
                }
                for index, event in enumerate(seg.audio_facts.sfx_events)
            ],
            "bgm_events": [
                {
                    "tone": event.tone,
                    "start_sec": event.start_sec,
                    "end_sec": event.end_sec,
                    "trigger_sec": event.start_sec,
                    "business_role": _derive_audio_business_role(local_hec_tag, persuasion_function, "bgm"),
                    "source_evidence": shared_evidence
                    or [f"{seg.segment_id}.audio_facts.bgm_events[{index}]"] ,
                }
                for index, event in enumerate(seg.audio_facts.bgm_events)
            ],
        }
    ).dict()



def _derive_reusable_clip_notes(
    seg: FactPackSegment,
    *,
    local_hec_tag: str,
    persuasion_function: str,
    audio_event_projection: dict[str, Any],
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    action_summary = _summarize_action_names(seg.visual_facts.actions)
    if action_summary or str(seg.visual_facts.visual_subject).strip():
        notes.append(
            {
                "note": (
                    f"可拆为{local_hec_tag}片段：保留{seg.visual_facts.shot_size}景别下的"
                    f"{action_summary or seg.visual_facts.visual_subject}，直接服务于{persuasion_function}。"
                ),
                "source_evidence": [
                    f"{seg.segment_id}.visual_facts.shot_size",
                    f"{seg.segment_id}.visual_facts.visual_subject",
                    *[
                        f"{seg.segment_id}.visual_facts.actions[{index}].action_name"
                        for index, action in enumerate(seg.visual_facts.actions)
                        if str(action.get("action_name") or "").strip()
                    ],
                ],
            }
        )
    if seg.ocr_facts:
        notes.append(
            {
                "note": f"画面花字可复用：{seg.ocr_facts[0].text}，适合直接作为该桥段的字幕锚点。",
                "source_evidence": [f"{seg.segment_id}.ocr_facts[0].text"],
            }
        )
    elif audio_event_projection.get("sfx_events") or audio_event_projection.get("bgm_events"):
        notes.append(
            {
                "note": "声音节奏点可复用：保留当前音效/BGM 起点，可直接复刻情绪推进节拍。",
                "source_evidence": [
                    *[
                        f"{seg.segment_id}.audio_event_projection.sfx_events[{index}]"
                        for index, _ in enumerate(audio_event_projection.get("sfx_events") or [])
                    ],
                    *[
                        f"{seg.segment_id}.audio_event_projection.bgm_events[{index}]"
                        for index, _ in enumerate(audio_event_projection.get("bgm_events") or [])
                    ],
                ],
            }
        )
    normalized_notes: list[dict[str, Any]] = []
    for item in notes[:2]:
        normalized_notes.append(ReusableClipNote.parse_obj(item).dict())
    return normalized_notes



def _extract_segment_risk_notes(seg: FactPackSegment) -> list[dict[str, Any]]:
    segment_text = _collect_segment_text(seg)
    notes: list[dict[str, Any]] = []
    if _H5_EXTREME_SCENARIO_PATTERN.search(segment_text) or _E2_STRESS_TEST_PATTERN.search(segment_text):
        notes.append(
            RiskBridgeNote.parse_obj(
                {
                    "risk_type": "extreme_scene",
                    "risk_level": "high",
                    "note": "存在极端环境/高压测试桥段，复用时必须补齐安全边界，禁止按常规使用场景直接照搬。",
                    "source_evidence": _build_segment_source_evidence(seg) or [f"{seg.segment_id}.audio_facts.asr_text"],
                }
            ).dict()
        )
    if _MALICIOUS_COMPARISON_PATTERN.search(segment_text) or _MALICIOUS_OLD_SOLUTION_PATTERN.search(segment_text):
        notes.append(
            RiskBridgeNote.parse_obj(
                {
                    "risk_type": "malicious_comparison",
                    "risk_level": "high",
                    "note": "出现妖魔化旧方案/合法成分的风险表达，必须走人工复核，不得直接复用。",
                    "source_evidence": _build_segment_source_evidence(seg) or [f"{seg.segment_id}.audio_facts.asr_text"],
                }
            ).dict()
        )
    if _PRICE_OR_BENEFIT_PATTERN.search(segment_text) and any(token in segment_text for token in ("下单", "购买", "点击", "链接")):
        notes.append(
            RiskBridgeNote.parse_obj(
                {
                    "risk_type": "cta_pressure",
                    "risk_level": "medium",
                    "note": "收口存在价格/动作联动，复用时需避免违规逼单或过度承诺。",
                    "source_evidence": _build_segment_source_evidence(seg) or [f"{seg.segment_id}.audio_facts.asr_text"],
                }
            ).dict()
        )
    return notes



def _derive_fourth_layer_assets(
    seg: FactPackSegment,
    *,
    local_hec_tag: str,
    persuasion_function: str,
    audio_event_projection: dict[str, Any],
) -> dict[str, Any]:
    reusable_clip_notes = _derive_reusable_clip_notes(
        seg,
        local_hec_tag=local_hec_tag,
        persuasion_function=persuasion_function,
        audio_event_projection=audio_event_projection,
    )
    risk_bridge_notes = _extract_segment_risk_notes(seg)
    is_key_bridge = bool(
        local_hec_tag.startswith(("E", "C"))
        or reusable_clip_notes
        or risk_bridge_notes
        or str(seg.audio_facts.asr_text).strip()
        or seg.ocr_facts
    )
    return {
        "is_key_bridge": is_key_bridge,
        "reusable_clip_notes": reusable_clip_notes,
        "risk_bridge_notes": risk_bridge_notes,
    }


class RequestProvenance(StrictBaseModel):
    producer_type: Literal[
        "system_native_inference",
        "external_vlm",
        "human_annotator",
        "external_pipeline",
        "ssot_lookup",
    ]
    generator_version: str | None = None
    generated_at: str | None = None


class OCRFact(StrictBaseModel):
    text: str
    position: dict[str, float]
    color: str
    font_family: str
    font_weight: str
    font_size_level: str
    stroke_style: str
    text_effect_style: str


class VisualFacts(StrictBaseModel):
    shot_size: str
    camera_movement: str
    visual_subject: str
    lighting_tone: str
    key_objects: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)


class AudioEventBase(StrictBaseModel):
    start_sec: float
    end_sec: float

    @validator("start_sec", "end_sec", pre=True)
    def _validate_time_value(cls, value: Any, field: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field.name} 必须为数值")
        return float(value)


class SFXEvent(AudioEventBase):
    event_name: str

    @validator("event_name")
    def _validate_event_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("event_name 不能为空")
        return normalized


class BGMEvent(AudioEventBase):
    tone: str

    @validator("tone")
    def _validate_tone(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("tone 不能为空")
        return normalized


class AudioFacts(StrictBaseModel):
    asr_text: str
    sfx_events: list[SFXEvent] = Field(default_factory=list)
    bgm_events: list[BGMEvent] = Field(default_factory=list)


class RhythmFacts(StrictBaseModel):
    transition_type: str
    pace_marker: str


class FactPackSegment(StrictBaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    visual_facts: VisualFacts
    audio_facts: AudioFacts
    ocr_facts: list[OCRFact] = Field(default_factory=list)
    rhythm_facts: RhythmFacts


class VideoMeta(StrictBaseModel):
    source_platform: str
    duration_sec: float
    fps: float
    resolution: str


class SemanticBundle(StrictBaseModel):
    bundle_id: str
    start_sec: float
    end_sec: float
    segment_ids: list[str]
    bundle_role: Literal["narrative_unit"] = "narrative_unit"
    aggregation_reason: list[str]
    blocked_boundary_ids: list[str] = Field(default_factory=list)
    coverage_frame_refs: list[str] = Field(default_factory=list)

    @validator("bundle_id", pre=True)
    def _validate_bundle_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("bundle_id 必须为非空字符串")
        return value

    @validator("start_sec", "end_sec", pre=True)
    def _validate_bundle_time(cls, value: Any, field: Any) -> float | int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field.name} 必须为数值")
        return value

    @validator("segment_ids", "aggregation_reason", "blocked_boundary_ids", "coverage_frame_refs", pre=True)
    def _validate_string_lists(cls, value: Any, field: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field.name} 必须为字符串列表")
        if field.name in {"segment_ids", "aggregation_reason", "coverage_frame_refs"} and not value:
            raise ValueError(f"{field.name} 不能为空")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field.name} 必须为非空字符串列表")
        return value


class BundleSegmentRange(StrictBaseModel):
    start_segment_index: int
    end_segment_index: int
    start_segment_id: str
    end_segment_id: str

    @validator("start_segment_index", "end_segment_index", pre=True)
    def _validate_segment_index(cls, value: Any, field: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field.name} 必须为整数")
        return value

    @validator("start_segment_id", "end_segment_id", pre=True)
    def _validate_segment_id(cls, value: Any, field: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} 必须为非空字符串")
        return value


class SecondFilterDecisionContext(StrictBaseModel):
    candidate_score: float | int | None = None
    adjacent_protected_count_10s: int
    same_bundle_relation: str
    ocr_jump_strength: float | int
    layout_migration_strength: float | int

    @validator("candidate_score", pre=True)
    def _validate_candidate_score(cls, value: Any) -> float | int | None:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("candidate_score 必须为非负数字或 null")
        if value < 0:
            raise ValueError("candidate_score 不能为负数")
        return value

    @validator("adjacent_protected_count_10s", pre=True)
    def _validate_adjacent_protected_count(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("adjacent_protected_count_10s 必须为非负整数")
        if value < 0:
            raise ValueError("adjacent_protected_count_10s 不能为负数")
        return value


class SecondFilterCandidate(StrictBaseModel):
    boundary_id: str
    protected_sec: float
    prev_segment_id: str
    next_segment_id: str
    trigger_signals: list[str] = Field(default_factory=list)
    high_ocr_scene: bool
    prev_segment_semantics: dict[str, Any] = Field(default_factory=dict)
    next_segment_semantics: dict[str, Any] = Field(default_factory=dict)
    decision_context: SecondFilterDecisionContext

    @validator("high_ocr_scene", pre=True)
    def _validate_high_ocr_scene(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("high_ocr_scene 必须为 bool")
        return value


class SecondFilterDecision(StrictBaseModel):
    boundary_id: str
    protected_sec: float
    decision: Literal["keep", "drop"]
    reason_code: str
    same_chain: bool
    same_goal: bool
    new_test: bool
    new_subject: bool
    new_goal: bool
    cta: bool
    prev_carrier: str | None = None
    next_carrier: str | None = None
    decision_context: SecondFilterDecisionContext

    @validator("same_chain", "same_goal", "new_test", "new_subject", "new_goal", "cta", pre=True)
    def _validate_decision_flags(cls, value: Any, field: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{field.name} 必须为 bool")
        return value


class SecondFilterTrace(StrictBaseModel):
    candidates: list[SecondFilterCandidate] = Field(default_factory=list)
    decisions: list[SecondFilterDecision] = Field(default_factory=list)


class FactPack(StrictBaseModel):
    video_meta: VideoMeta
    segments: list[FactPackSegment]
    semantic_bundles: list[SemanticBundle] = Field(default_factory=list)
    segment_to_bundle_map: dict[str, str] = Field(default_factory=dict)
    bundle_to_segment_range: dict[str, BundleSegmentRange] = Field(default_factory=dict)
    second_filter_trace: SecondFilterTrace = Field(default_factory=SecondFilterTrace)
    storyboard_source: Literal["semantic_bundles"] = "semantic_bundles"


class SliderDimension(StrictBaseModel):
    score: int
    business_judgment: str

    @validator("score", pre=True)
    def _validate_score(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("score 必须为 0-10 的整数")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("score 必须为 0-10 的整数")
        if numeric < 0 or numeric > 10:
            raise ValueError("score 超出范围 [0,10]")
        if abs(numeric - round(numeric)) > 1e-6:
            raise ValueError("score 必须为整数")
        return int(round(numeric))

    @validator("business_judgment")
    def _validate_business_judgment(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("business_judgment 不能为空")
        return normalized


SLIDER_DEFAULT_JUDGMENTS = {
    "visual": "视觉表达强度待补充业务定性。",
    "audio": "音频表达强度待补充业务定性。",
    "proof": "举证表达强度待补充业务定性。",
    "cta": "收口表达强度待补充业务定性。",
}

SLIDER_INGEST_JUDGMENT_HINTS = {
    "visual": ("偏收敛", "偏强"),
    "audio": ("偏收敛", "偏强"),
    "proof": ("偏弱", "偏强"),
    "cta": ("偏弱", "偏强"),
}


def _build_ingest_slider_business_judgment(dimension: str, score: Any) -> str:
    numeric_score = int(float(score))
    low_hint, high_hint = SLIDER_INGEST_JUDGMENT_HINTS[dimension]
    tendency = high_hint if numeric_score >= 6 else low_hint
    dim_cn = {
        "visual": "视觉",
        "audio": "音频",
        "proof": "举证",
        "cta": "收口",
    }[dimension]
    return f"导入轨沿用历史评分 {numeric_score}，当前按新协议补齐为 {dim_cn}表达{tendency}。"


def _normalize_slider_signature_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolViolation("AssetIngest 失败：blueprint.slider_signature 必须为对象")
    normalized: dict[str, Any] = {}
    for dimension in ("visual", "audio", "proof", "cta"):
        item = value.get(dimension)
        if isinstance(item, dict):
            if "business_judgment" in item and str(item.get("business_judgment") or "").strip():
                normalized[dimension] = dict(item)
            else:
                normalized[dimension] = {
                    **dict(item),
                    "business_judgment": _build_ingest_slider_business_judgment(dimension, item.get("score")),
                }
        else:
            normalized[dimension] = {
                "score": item,
                "business_judgment": _build_ingest_slider_business_judgment(dimension, item),
            }
    return SliderSignature.parse_obj(normalized).dict()


class SliderSignature(StrictBaseModel):
    visual: SliderDimension
    audio: SliderDimension
    proof: SliderDimension
    cta: SliderDimension

    @validator("visual", "audio", "proof", "cta", pre=True)
    def _normalize_dimension(cls, value: Any, field: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{field.name} 必须为包含 score 与 business_judgment 的对象")
        score = value.get("score")
        business_judgment = value.get("business_judgment")
        if not isinstance(business_judgment, str) or not business_judgment.strip():
            raise ValueError(f"{field.name}.business_judgment 必填且不能为空")
        normalized_judgment = business_judgment.strip()
        if normalized_judgment == SLIDER_DEFAULT_JUDGMENTS[field.name]:
            raise ValueError(f"{field.name}.business_judgment 不允许使用默认占位文案")
        return {"score": score, "business_judgment": normalized_judgment}


class PerformanceEmotion(StrictBaseModel):
    acting_instructions: str
    emotion_tension: str
    emotional_tone: str
    action_mechanics: str
    action_intensity: str

    @validator("acting_instructions", "emotion_tension", "emotional_tone", "action_mechanics", "action_intensity")
    def _validate_non_empty(cls, value: str, field: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field.name} 不能为空")
        return normalized

    @root_validator
    def _validate_emotion_distinction(cls, values: dict[str, Any]) -> dict[str, Any]:
        emotion_tension = str(values.get("emotion_tension") or "").strip()
        emotional_tone = str(values.get("emotional_tone") or "").strip()
        if emotion_tension and emotional_tone and emotion_tension == emotional_tone:
            raise ValueError("emotion_tension 不得与 emotional_tone 完全相同")
        return values


class ProjectedSFXEvent(StrictBaseModel):
    event_name: str
    start_sec: float
    end_sec: float
    trigger_sec: float
    business_role: str
    source_evidence: list[str] = Field(default_factory=list)

    @validator("event_name", "business_role")
    def _validate_non_empty_text(cls, value: str, field: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field.name} 不能为空")
        return normalized

    @validator("source_evidence", each_item=True)
    def _validate_source_evidence_item(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("source_evidence 不能为空字符串")
        return normalized


class ProjectedBGMEvent(StrictBaseModel):
    tone: str
    start_sec: float
    end_sec: float
    trigger_sec: float
    business_role: str
    source_evidence: list[str] = Field(default_factory=list)

    @validator("tone", "business_role")
    def _validate_non_empty_text(cls, value: str, field: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field.name} 不能为空")
        return normalized

    @validator("source_evidence", each_item=True)
    def _validate_source_evidence_item(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("source_evidence 不能为空字符串")
        return normalized


class AudioEventProjection(StrictBaseModel):
    sfx_events: list[ProjectedSFXEvent] = Field(default_factory=list)
    bgm_events: list[ProjectedBGMEvent] = Field(default_factory=list)


class ReusableClipNote(StrictBaseModel):
    note: str
    source_evidence: list[str] = Field(default_factory=list)

    @validator("note")
    def _validate_note(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("note 不能为空")
        return normalized

    @validator("source_evidence", each_item=True)
    def _validate_source_evidence_item(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("source_evidence 不能为空字符串")
        return normalized


class RiskBridgeNote(StrictBaseModel):
    risk_type: str
    note: str
    risk_level: Literal["low", "medium", "high"]
    source_evidence: list[str] = Field(default_factory=list)

    @validator("risk_type", "note")
    def _validate_text(cls, value: str, field: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field.name} 不能为空")
        return normalized

    @validator("source_evidence", each_item=True)
    def _validate_source_evidence_item(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("source_evidence 不能为空字符串")
        return normalized


class StoryboardSegment(StrictBaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    visual_facts: VisualFacts
    audio_facts: AudioFacts
    ocr_facts: list[OCRFact] = Field(default_factory=list)
    rhythm_facts: RhythmFacts
    local_hec_tag: str
    persuasion_function: str
    performance_emotion: PerformanceEmotion
    audio_event_projection: AudioEventProjection = Field(default_factory=AudioEventProjection)
    is_key_bridge: bool = False
    reusable_clip_notes: list[ReusableClipNote] = Field(default_factory=list)
    risk_bridge_notes: list[RiskBridgeNote] = Field(default_factory=list)
    member_segment_ids: list[str] = Field(default_factory=list)
    aggregation_reason: list[str] = Field(default_factory=list)
    coverage_frame_refs: list[str] = Field(default_factory=list)
    blocked_boundary_ids: list[str] = Field(default_factory=list)

    @validator("local_hec_tag")
    def _validate_local_hec_tag(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("local_hec_tag 不能为空")
        if not re.fullmatch(r"(?:H[1-7]|E[0-7]|C[1-5])", normalized):
            raise ValueError("local_hec_tag 必须是合法的 H/E/C 标签")
        return normalized

    @validator("persuasion_function")
    def _validate_persuasion_function(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("persuasion_function 不能为空")
        return normalized


class PrimaryHEC(StrictBaseModel):
    hook_label: str
    effect_label: str
    cta_label: str

    @validator("hook_label", pre=True)
    def _validate_hook_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("hook_label 必须为字符串")
        normalized = value.strip().upper()
        if not re.fullmatch(r"H[1-7]", normalized):
            raise ValueError("hook_label 必须是 H1-H7")
        return normalized

    @validator("effect_label", pre=True)
    def _validate_effect_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("effect_label 必须为字符串")
        normalized = value.strip().upper()
        if normalized not in SECONDARY_EFFECT_LABELS:
            raise ValueError("effect_label 必须是 E0-E7")
        return normalized

    @validator("cta_label", pre=True)
    def _validate_cta_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("cta_label 必须为字符串")
        normalized = value.strip().upper()
        if not re.fullmatch(r"C[1-5]", normalized):
            raise ValueError("cta_label 必须是 C1-C5")
        return normalized


class SecondaryEffect(StrictBaseModel):
    effect_label: str
    evidence_segment_ids: list[str]
    reason: str

    @validator("effect_label", pre=True)
    def _validate_effect_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("effect_label 必须为字符串")
        normalized = value.strip().upper()
        if normalized not in SECONDARY_EFFECT_LABELS:
            raise ValueError("effect_label 必须是 E0-E7")
        return normalized

    @validator("evidence_segment_ids", pre=True)
    def _validate_evidence_segment_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("evidence_segment_ids 必须为非空数组")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("evidence_segment_ids 必须为非空字符串数组")
        return [item.strip() for item in value]

    @validator("reason", pre=True)
    def _validate_reason(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("reason 不能为空")
        return value.strip()


class RiskFlags(StrictBaseModel):
    inference_mode: str
    hec_reason: str
    hec_evidence_segment_ids: list[str]
    secondary_effects_present: bool

    @validator("inference_mode", "hec_reason", pre=True)
    def _validate_non_empty_text(cls, value: Any, field: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} 不能为空")
        return value.strip()

    @validator("hec_evidence_segment_ids", pre=True)
    def _validate_hec_evidence_segment_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("hec_evidence_segment_ids 必须为非空数组")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("hec_evidence_segment_ids 必须为非空字符串数组")
        return [item.strip() for item in value]

    @validator("secondary_effects_present", pre=True)
    def _validate_secondary_effects_present(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("secondary_effects_present 必须为布尔值")
        return value


class StoryboardFactLayer(StrictBaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    visual_facts: VisualFacts
    audio_facts: AudioFacts
    ocr_facts: list[OCRFact] = Field(default_factory=list)
    rhythm_facts: RhythmFacts
    member_segment_ids: list[str]
    aggregation_reason: list[str]
    coverage_frame_refs: list[str]
    blocked_boundary_ids: list[str] = Field(default_factory=list)

    @validator("segment_id", pre=True)
    def _validate_segment_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("segment_id 不能为空")
        return value.strip()

    @validator("member_segment_ids", "aggregation_reason", "coverage_frame_refs", pre=True)
    def _validate_required_string_lists(cls, value: Any, field: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field.name} 必须为非空字符串数组")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field.name} 必须为非空字符串数组")
        return [item.strip() for item in value]

    @validator("blocked_boundary_ids", pre=True)
    def _validate_optional_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("blocked_boundary_ids 必须为字符串数组")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("blocked_boundary_ids 必须为非空字符串数组")
        return [item.strip() for item in value]


class StoryboardStyleLayer(StrictBaseModel):
    local_hec_tag: str
    persuasion_function: str
    performance_emotion: PerformanceEmotion

    @validator("local_hec_tag", pre=True)
    def _validate_local_hec_tag(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("local_hec_tag 必须为字符串")
        normalized = value.strip().upper()
        if not re.fullmatch(r"(?:H[1-7]|E[0-7]|C[1-5])", normalized):
            raise ValueError("local_hec_tag 必须是合法的 H/E/C 标签")
        return normalized

    @validator("persuasion_function", pre=True)
    def _validate_persuasion_function(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("persuasion_function 不能为空")
        return value.strip()


class StoryboardExecutionLayer(StrictBaseModel):
    audio_event_projection: AudioEventProjection = Field(default_factory=AudioEventProjection)
    is_key_bridge: bool
    reusable_clip_notes: list[ReusableClipNote] = Field(default_factory=list)
    risk_bridge_notes: list[RiskBridgeNote] = Field(default_factory=list)

    @validator("is_key_bridge", pre=True)
    def _validate_is_key_bridge(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("is_key_bridge 必须为布尔值")
        return value


class BlueprintL1Envelope(StrictBaseModel):
    blueprint_id: str
    video_id: str
    source_product_id: str
    storyboard_source: Literal["segments"]
    semantic_bundles: list[SemanticBundle]
    segment_to_bundle_map: dict[str, str]
    bundle_to_segment_range: dict[str, BundleSegmentRange]

    @validator("blueprint_id", "video_id", "source_product_id", pre=True)
    def _validate_non_empty_text(cls, value: Any, field: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} 不能为空")
        return value.strip()

    @validator("semantic_bundles", pre=True)
    def _validate_semantic_bundles(cls, value: Any) -> list[Any]:
        if not isinstance(value, list) or not value:
            raise ValueError("semantic_bundles 必须为非空数组")
        return value

    @validator("segment_to_bundle_map", "bundle_to_segment_range", pre=True)
    def _validate_non_empty_dict(cls, value: Any, field: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{field.name} 必须为非空对象")
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError(f"{field.name} 的 key 必须为非空字符串")
        if field.name == "segment_to_bundle_map":
            if any(not isinstance(item, str) or not item.strip() for item in value.values()):
                raise ValueError("segment_to_bundle_map 的 value 必须为非空字符串")
            return {str(key).strip(): str(item).strip() for key, item in value.items()}
        return value


class BlueprintL2Envelope(StrictBaseModel):
    primary_hec: PrimaryHEC
    secondary_effects: list[SecondaryEffect] = Field(default_factory=list)
    slider_signature: SliderSignature
    risk_flags: RiskFlags


class VideoUnderstandingRequest(StrictBaseModel):
    request_id: str
    video_id: str
    source_product_id: str
    video_url: str = ""
    item_name: str = ""
    shop_name: str = ""
    leaf_category: str = ""
    price: str = ""
    core_selling_points: list[str] = Field(default_factory=list)
    fact_pack: FactPack
    provenance: RequestProvenance
    options: dict[str, Any] = Field(default_factory=dict)

    @validator("core_selling_points", pre=True)
    def _normalize_core_selling_points(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized
        text = str(value).strip()
        if not text:
            return []
        return [item.strip() for item in re.split(r"[;；\n]+", text) if item.strip()]

    @root_validator
    def _validate_caller_product_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        required_fields = ("item_name", "shop_name", "leaf_category", "price", "core_selling_points")
        has_any = any(values.get(field) for field in ("video_url", *required_fields))
        if not has_any:
            return values
        missing = [field for field in required_fields if not values.get(field)]
        if missing:
            raise ValueError(f"caller 直传商品字段不完整，缺少：{', '.join(missing)}")
        return values


class AssetIngestRequest(StrictBaseModel):
    request_id: str
    video_id: str
    source_product_id: str
    asset_package: dict[str, Any]
    provenance: RequestProvenance
    options: dict[str, Any] = Field(default_factory=dict)


class FieldProvenance(StrictBaseModel):
    field_path: str
    producer_type: Literal[
        "system_native_inference",
        "external_vlm",
        "human_annotator",
        "external_pipeline",
        "ssot_lookup",
    ]
    source_type: str
    source_refs: list[str]
    generated_at: str
    generator_version: str


class VideoUnderstandingResult(StrictBaseModel):
    blueprint: dict[str, Any]
    workflow_report: dict[str, Any]
    phase_4_output: dict[str, Any]
    phase_5_output: dict[str, Any]
    triad_assets: dict[str, Any]
    provenance_report: list[FieldProvenance]
    video_coverage_gap_report: dict[str, Any] = Field(default_factory=dict)


class AssetIngestResult(StrictBaseModel):
    ingested_assets: dict[str, Any]
    validation_report: dict[str, Any]
    provenance_report: list[FieldProvenance]
    import_mode: Literal["asset_accumulation"] = "asset_accumulation"


class ProductSnapshot(StrictBaseModel):
    source_product_id: str
    leaf_category_id: str
    leaf_category_name: str
    product_name: str
    brand_name: str
    shop_name: str = ""
    brand_asset_level: Literal["high", "low"]
    price_band: Literal["high", "low"]
    price_source: str = "legacy_ssot"
    financial_risk_level: Literal["high", "low"]
    core_jtbd: str
    trust_barrier_level: Literal["low", "high"]
    cognitive_barrier_level: Literal["low", "high"]
    habit_switch_barrier_level: Literal["low", "high"]
    diagnosis_version: str
    diagnosis_generated_at: str


@dataclass(frozen=True)
class FileSSOTClient:
    path: Path

    def get_product_snapshot(self, source_product_id: str) -> ProductSnapshot:
        if not self.path.exists():
            raise ProtocolViolation(f"SSOT 文件不存在: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        record = payload.get(source_product_id)
        if not isinstance(record, dict):
            raise ProtocolViolation(f"SSOT 查表失败: source_product_id={source_product_id}")
        try:
            return ProductSnapshot.parse_obj(record)
        except ValidationError as e:
            raise ProtocolViolation(f"SSOT 返回缺字段或非法: {e}")

    def upsert_product_snapshot(self, snapshot: ProductSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ProtocolViolation(f"SSOT 文件格式非法: {self.path}")
        else:
            payload = {}
        payload[snapshot.source_product_id] = snapshot.dict()
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slugify_category(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "").upper()).strip("_")
    return slug or "UNKNOWN"


def _normalize_brand_name(item_name: str, shop_name: str) -> str:
    del item_name
    return str(shop_name or "").strip() or "未命名店铺"


def _derive_price_band(leaf_category: str, price_text: str) -> tuple[Literal["high", "low"], Literal["高水位", "低水位"]]:
    lookup = _build_price_band_lookup()
    if leaf_category not in lookup:
        raise ProtocolViolation(f"price_band_dict.csv 不存在类目 {leaf_category}，无法派生价格水位。")
    try:
        price_value = float(str(price_text).strip())
    except ValueError as exc:
        raise ProtocolViolation(f"caller 直传价格非法：{price_text}") from exc
    median = float(lookup[leaf_category])
    if price_value >= median:
        return "high", "高水位"
    return "low", "低水位"


def _build_provisional_product_snapshot(request: VideoUnderstandingRequest) -> ProductSnapshot:
    price_band, relative_price_level = _derive_price_band(request.leaf_category, request.price)
    return ProductSnapshot(
        source_product_id=request.source_product_id,
        leaf_category_id=f"LC_{_slugify_category(request.leaf_category)}_{request.source_product_id}",
        leaf_category_name=request.leaf_category,
        product_name=request.item_name,
        brand_name=_normalize_brand_name(request.item_name, request.shop_name),
        shop_name=request.shop_name,
        brand_asset_level="low",
        price_band=price_band,
        price_source="caller_provided",
        financial_risk_level="high" if relative_price_level == "高水位" else "low",
        core_jtbd="待商品诊断回填",
        trust_barrier_level="high",
        cognitive_barrier_level="high",
        habit_switch_barrier_level="high",
        diagnosis_version="caller_input_pending_product_diagnosis",
        diagnosis_generated_at=_utc_now_rfc3339(),
    )


def _infer_difference_domain_from_type(difference_type: str) -> str:
    normalized_difference_type = str(difference_type or "").strip()
    for difference_domain, difference_types in DIFFERENTIATOR_DOMAIN_TYPES.items():
        if normalized_difference_type in difference_types:
            return difference_domain
    raise ProtocolViolation(f"caller 直传 difference_type 缺少合法 difference_domain：{normalized_difference_type}")



def _build_caller_diagnostic_payload(request: VideoUnderstandingRequest) -> dict[str, Any]:
    core_selling_point = "；".join(request.core_selling_points)
    _, relative_price_level = _derive_price_band(request.leaf_category, request.price)
    difference_type = "自身卖点陈述"
    difference_domain = _infer_difference_domain_from_type(difference_type)
    return {
        "product_id": request.source_product_id,
        "leaf_category": request.leaf_category,
        "shop_name": request.shop_name,
        "product_name": request.item_name,
        "price": request.price,
        "core_selling_point": core_selling_point,
        "core_selling_point_source": "caller_provided.core_selling_points",
        "target_people": "",
        "differentiator": {
            "comparison_object": "",
            "comparison_object_evidence_type": "null",
            "difference_domain": difference_domain,
            "difference_type": difference_type,
            "conclusion": core_selling_point,
            "evidence_chain": [
                {
                    "evidence_source": "caller_provided.core_selling_points",
                    "evidence_text": point,
                }
                for point in request.core_selling_points
            ],
        },
        "engine_node": {"relative_price_level": relative_price_level},
    }


def _run_product_diagnosis_from_caller_input(request: VideoUnderstandingRequest) -> dict[str, Any]:
    payload = _build_caller_diagnostic_payload(request)
    diagnosis = ProductDiagnosisEngine().diagnose(DiagnosticInput.from_payload(payload))
    return diagnosis.to_dict()


def _map_diagnosis_to_product_snapshot(
    request: VideoUnderstandingRequest,
    diagnosis: dict[str, Any],
) -> ProductSnapshot:
    resistance_profile = diagnosis.get("resistance_profile") or {}
    category_matrix = diagnosis.get("category_intent_matrix") or {}
    product_matrix = diagnosis.get("product_intent_matrix") or {}
    trust_barrier = str(product_matrix.get("trust_barrier") or "").strip()
    relative_price_level = str(product_matrix.get("relative_price_level") or "").strip()
    ocean = str(category_matrix.get("ocean") or "").strip()
    frequency = str(category_matrix.get("frequency") or "").strip()
    return ProductSnapshot(
        source_product_id=request.source_product_id,
        leaf_category_id=f"LC_{_slugify_category(request.leaf_category)}_{request.source_product_id}",
        leaf_category_name=request.leaf_category,
        product_name=request.item_name,
        brand_name=_normalize_brand_name(request.item_name, request.shop_name),
        shop_name=request.shop_name,
        brand_asset_level="high" if trust_barrier == "极低" else "low",
        price_band="high" if relative_price_level == "高水位" else "low",
        price_source="caller_provided",
        financial_risk_level="high" if str(resistance_profile.get("financial_risk") or "") == "高" else "low",
        core_jtbd=str(diagnosis.get("jtbd") or "").strip() or "待商品诊断回填",
        trust_barrier_level="low" if trust_barrier == "极低" else "high",
        cognitive_barrier_level="high" if ocean == "蓝海" else "low",
        habit_switch_barrier_level="high" if frequency == "快消" else "low",
        diagnosis_version="caller_product_diagnosis_v1",
        diagnosis_generated_at=_utc_now_rfc3339(),
    )


def _build_product_snapshot_from_caller_input(
    request: VideoUnderstandingRequest,
    *,
    ssot: FileSSOTClient,
    triad_repo: TriadAssetRepository | None = None,
) -> dict[str, Any]:
    diagnosis = _run_product_diagnosis_from_caller_input(request)
    snapshot = _map_diagnosis_to_product_snapshot(request, diagnosis)
    ssot.upsert_product_snapshot(snapshot)
    if triad_repo is not None:
        triad_repo.upsert_product_snapshot(
            snapshot.dict(),
            {
                "producer_type": "external_pipeline",
                "source_refs": [
                    f"source_product_id:{snapshot.source_product_id}",
                    "caller_provided:item_name/shop_name/leaf_category/price/core_selling_points",
                ],
            },
        )
    return {"snapshot": snapshot, "diagnosis": diagnosis}


def _patch_video_understanding_result_with_product_snapshot(
    result: VideoUnderstandingResult,
    snapshot: ProductSnapshot,
    route_b_diagnosis: dict[str, Any],
) -> VideoUnderstandingResult:
    patched = result.dict()
    patched["triad_assets"]["product_master_snapshot"] = {
        **snapshot.dict(),
        "provenance": {
            "producer_type": "external_pipeline",
            "source_refs": [
                f"source_product_id:{snapshot.source_product_id}",
                "caller_provided:item_name/shop_name/leaf_category/price/core_selling_points",
            ],
        },
    }
    patched["triad_assets"]["product_diagnosis_result"] = route_b_diagnosis
    patched["workflow_report"]["product_context_source"] = "caller_product_diagnosis"
    patched["workflow_report"]["parallel_branches"] = {
        "route_a_video_understanding": "completed",
        "route_b_product_diagnosis": "completed",
        "join_status": "completed",
    }
    for item in patched.get("provenance_report", []):
        if item.get("field_path") == "triad_assets.product_master_snapshot":
            item["producer_type"] = "external_pipeline"
            item["source_type"] = "product_diagnosis.caller_input"
            item["source_refs"] = [
                f"source_product_id:{snapshot.source_product_id}",
                "caller_provided:item_name/shop_name/leaf_category/price/core_selling_points",
            ]
            item["generated_at"] = snapshot.diagnosis_generated_at
            item["generator_version"] = snapshot.diagnosis_version
    return VideoUnderstandingResult.parse_obj(patched)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)



def _is_db_persistence_enabled(options: dict[str, Any] | None, db_path: str | Path | None) -> bool:
    if db_path is not None:
        return True
    if not isinstance(options, dict):
        return False
    return _is_truthy(options.get("db_persistence_enabled"))



def _default_triad_assets_db_path(ssot_path: str | Path | None) -> Path:
    if ssot_path is not None:
        return Path(ssot_path).with_name("triad_assets.db")
    return Path(__file__).parent / "data" / "triad_assets.db"



def _resolve_triad_assets_db_engine(*, options: dict[str, Any] | None, request_id: str, db_path: str | Path | None) -> str:
    if db_path is not None:
        return "sqlite"
    raw_engine = None
    if isinstance(options, dict):
        raw_engine = options.get("triad_assets_db_engine") or options.get("db_engine")
    normalized = str(raw_engine or "sqlite").strip().lower()
    if normalized in {"sqlite", "mysql"}:
        return normalized
    if normalized != "auto":
        raise ProtocolViolation(f"triad_assets_db_engine 非法：{normalized}")
    rollout_percent = _resolve_mysql_rollout_percent(options)
    return "mysql" if _request_id_rolls_into_mysql(request_id=request_id, rollout_percent=rollout_percent) else "sqlite"



def _resolve_mysql_rollout_percent(options: dict[str, Any] | None) -> int:
    raw_value = None
    if isinstance(options, dict):
        raw_value = options.get("triad_assets_mysql_rollout_percent")
    if raw_value in (None, ""):
        return 0
    try:
        percent = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation("triad_assets_mysql_rollout_percent 必须是 0~100 的整数") from exc
    if percent < 0 or percent > 100:
        raise ProtocolViolation("triad_assets_mysql_rollout_percent 必须是 0~100 的整数")
    return percent



def _request_id_rolls_into_mysql(*, request_id: str, rollout_percent: int) -> bool:
    if rollout_percent <= 0:
        return False
    if rollout_percent >= 100:
        return True
    bucket = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < rollout_percent



def _resolve_mysql_config(options: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if isinstance(options, dict):
        raw_config = (
            options.get("triad_assets_mysql")
            or options.get("triad_assets_mysql_config")
            or options.get("triad_assets_db_config")
        )
        if raw_config is not None and not isinstance(raw_config, dict):
            raise ProtocolViolation("triad_assets_mysql_config 必须是 dict")
        if isinstance(raw_config, dict):
            for key, value in raw_config.items():
                if value not in (None, ""):
                    merged[key] = value
    required_keys = ("host", "user", "password", "database")
    if not all(str(merged.get(key) or "").strip() for key in required_keys):
        return None
    return merged



def _resolve_triad_asset_repository(
    *,
    options: dict[str, Any] | None,
    ssot_path: str | Path | None,
    db_path: str | Path | None,
    request_id: str,
) -> TriadAssetRepository | None:
    if not _is_db_persistence_enabled(options, db_path):
        return None
    resolved_engine = _resolve_triad_assets_db_engine(options=options, request_id=request_id, db_path=db_path)
    if resolved_engine == "mysql":
        mysql_config = _resolve_mysql_config(options)
        if mysql_config is None:
            raise ProtocolViolation("已启用 MySQL adapter，但缺少 triad_assets_mysql_config")
        return TriadAssetRepository(engine="mysql", mysql_config=mysql_config)
    raw_db_path = db_path
    if raw_db_path is None and isinstance(options, dict):
        raw_db_path = options.get("triad_assets_db_path") or options.get("db_path")
    return TriadAssetRepository(Path(raw_db_path) if raw_db_path is not None else _default_triad_assets_db_path(ssot_path))



def _build_triad_asset_persistence_summary(
    *,
    triad_repo: TriadAssetRepository,
    request_id: str,
    video_id: str,
    source_product_id: str,
    workflow_version: str,
    generator_version: str,
    triad_assets: dict[str, Any],
) -> TriadAssetPersistenceSummary:
    product_snapshot = dict(triad_assets.get("product_master_snapshot") or {})
    product_provenance = dict(product_snapshot.get("provenance") or {})
    blueprint = dict(triad_assets.get("video_blueprint_master") or {})
    segment_records = list(triad_assets.get("video_segment_fact_table") or [])
    product_write = triad_repo.upsert_product_snapshot(product_snapshot, product_provenance)
    blueprint_write = triad_repo.persist_blueprint_with_segments(
        product_snapshot_id=product_write.product_snapshot_id,
        request_id=request_id,
        video_id=video_id,
        source_product_id=source_product_id,
        generator_version=generator_version,
        workflow_version=workflow_version,
        blueprint=blueprint,
        segment_records=segment_records,
    )
    table_counts = triad_repo.get_table_counts()
    if blueprint_write.segment_count != len(segment_records):
        raise ProtocolViolation(
            f"TriadAssets 落库后置断言失败：expected_segment_count={len(segment_records)} actual={blueprint_write.segment_count}"
        )
    return TriadAssetPersistenceSummary(
        product_snapshot_id=product_write.product_snapshot_id,
        snapshot_hash=product_write.snapshot_hash,
        blueprint_id=blueprint_write.blueprint_id,
        idempotency_key=blueprint_write.idempotency_key,
        segment_count=blueprint_write.segment_count,
        table_counts=table_counts,
        product_snapshot_inserted=product_write.inserted,
        blueprint_inserted=blueprint_write.inserted,
    )



def _build_db_persistence_payload(
    summary: TriadAssetPersistenceSummary,
    *,
    triad_repo: TriadAssetRepository,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "db_engine": triad_repo.engine,
        "db_locator": triad_repo.locator,
        "db_path": str(triad_repo.db_path) if triad_repo.db_path is not None else None,
        "product_snapshot_id": summary.product_snapshot_id,
        "snapshot_hash": summary.snapshot_hash,
        "blueprint_id": summary.blueprint_id,
        "idempotency_key": summary.idempotency_key,
        "segment_count": summary.segment_count,
        "table_counts": {
            "product_master_snapshot": summary.table_counts.product_snapshot_count,
            "video_blueprint_master": summary.table_counts.video_blueprint_count,
            "video_segment_fact_table": summary.table_counts.video_segment_count,
        },
        "write_result": {
            "product_master_snapshot": "inserted" if summary.product_snapshot_inserted else "deduplicated",
            "video_blueprint_master": "inserted" if summary.blueprint_inserted else "deduplicated",
            "video_segment_fact_table": "inserted" if summary.blueprint_inserted else "deduplicated",
        },
    }



def _attach_db_persistence_to_video_result(
    result: VideoUnderstandingResult,
    *,
    summary: TriadAssetPersistenceSummary,
    triad_repo: TriadAssetRepository,
) -> VideoUnderstandingResult:
    patched = result.dict()
    persistence_payload = _build_db_persistence_payload(summary, triad_repo=triad_repo)
    patched.setdefault("workflow_report", {})["db_persistence"] = persistence_payload
    patched.setdefault("phase_5_output", {}).setdefault("summary", {})["db_persistence"] = persistence_payload
    discipline_checks = list(patched["phase_5_output"].get("discipline_checks") or [])
    discipline_checks.append({"name": "triad_assets_db_persisted", "status": "pass"})
    patched["phase_5_output"]["discipline_checks"] = discipline_checks
    return VideoUnderstandingResult.parse_obj(patched)



def _attach_db_persistence_to_asset_ingest_result(
    result: AssetIngestResult,
    *,
    summary: TriadAssetPersistenceSummary,
    triad_repo: TriadAssetRepository,
) -> AssetIngestResult:
    patched = result.dict()
    persistence_payload = _build_db_persistence_payload(summary, triad_repo=triad_repo)
    patched.setdefault("validation_report", {})["db_persistence"] = persistence_payload
    patched.setdefault("ingested_assets", {})["db_persistence"] = persistence_payload
    return AssetIngestResult.parse_obj(patched)



def _assert_factpack_purity(request: VideoUnderstandingRequest) -> None:
    # Gate 1：FactPack 纯净性校验
    hit = _find_forbidden_field_paths(request.fact_pack.dict(), FORBIDDEN_ANSWER_KEYS_IN_FACTPACK)
    if hit:
        raise ProtocolViolation(f"Gate1 失败：FactPack 出现答案/策略字段：{hit}")


def _assert_director_ready_schema(request: VideoUnderstandingRequest) -> None:


    # Gate 1：Director-ready 字段完整性（PRD C.6）


    if not request.fact_pack.segments:


        raise ProtocolViolation("Gate1 失败：fact_pack.segments 不能为空")





    seen_segment_ids: set[str] = set()


    last_end_sec = -1.0


    for seg in request.fact_pack.segments:


        vf = seg.visual_facts


        af = seg.audio_facts


        rf = seg.rhythm_facts





        if not str(seg.segment_id).strip():


            raise ProtocolViolation("Gate1 失败：segment_id 缺失")


        if seg.segment_id in seen_segment_ids:


            raise ProtocolViolation(f"Gate1 失败：segment_id 重复：{seg.segment_id}")


        seen_segment_ids.add(seg.segment_id)


        if seg.start_sec >= seg.end_sec:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 时间轴非法，要求 start_sec < end_sec")


        if seg.start_sec < last_end_sec:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 时间轴未按顺序递增")


        last_end_sec = seg.end_sec





        if not vf.shot_size:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 shot_size")


        if not vf.lighting_tone:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 lighting_tone")


        if not vf.camera_movement:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 camera_movement")


        if not vf.visual_subject:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 visual_subject")


        if af.asr_text is None or not isinstance(af.asr_text, str):


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} audio_facts.asr_text 必须为字符串，可为空串")

        for sfx_event in af.sfx_events:
            if sfx_event.start_sec >= sfx_event.end_sec:
                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} sfx_events[*] 要求 start_sec < end_sec")
            if sfx_event.start_sec < seg.start_sec or sfx_event.end_sec > seg.end_sec:
                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} sfx_events[*] 必须锚定在分镜时间轴内")

        for bgm_event in af.bgm_events:
            if bgm_event.start_sec >= bgm_event.end_sec:
                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} bgm_events[*] 要求 start_sec < end_sec")
            if bgm_event.start_sec < seg.start_sec or bgm_event.end_sec > seg.end_sec:
                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} bgm_events[*] 必须锚定在分镜时间轴内")

        if not rf.transition_type:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 rhythm_facts.transition_type")


        if not rf.pace_marker:


            raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} 缺少 rhythm_facts.pace_marker")


        for act in vf.actions:


            if not str(act.get("action_name") or "").strip():


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} actions[*].action_name 缺失")


            if not str(act.get("physical_intensity") or "").strip():


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} actions[*].physical_intensity 缺失")


        for ocr in seg.ocr_facts:


            if not isinstance(ocr.position, dict) or not {"x", "y", "w", "h"}.issubset(ocr.position.keys()):


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.position 必须含 x/y/w/h")


            for axis in ("x", "y", "w", "h"):


                value = ocr.position.get(axis)


                if not isinstance(value, (int, float)) or value < 0 or value > 1:


                    raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.position.{axis} 必须为 0-1 数值")


            if ocr.position.get("w", 0) <= 0 or ocr.position.get("h", 0) <= 0:


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.position.w/h 必须大于 0")


            if ocr.position.get("x", 0) + ocr.position.get("w", 0) > 1 or ocr.position.get("y", 0) + ocr.position.get("h", 0) > 1:


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.position 边界越出画面")


            if not ocr.color:


                raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.color 缺失")


            for style_key in ("font_family", "font_weight", "font_size_level", "stroke_style", "text_effect_style"):


                if not str(getattr(ocr, style_key, "") or "").strip():


                    raise ProtocolViolation(f"Gate1 失败：{seg.segment_id} ocr_facts.{style_key} 缺失")





    fact_pack = request.fact_pack


    if fact_pack.storyboard_source != "semantic_bundles":


        raise ProtocolViolation("Gate1 失败：storyboard_source 必须显式声明为 semantic_bundles")


    if not fact_pack.semantic_bundles:


        raise ProtocolViolation("Gate1 失败：fact_pack.semantic_bundles 不能为空")


    if not fact_pack.segment_to_bundle_map:


        raise ProtocolViolation("Gate1 失败：fact_pack.segment_to_bundle_map 缺失")


    if not fact_pack.bundle_to_segment_range:


        raise ProtocolViolation("Gate1 失败：fact_pack.bundle_to_segment_range 缺失")





    all_segment_ids = [seg.segment_id for seg in fact_pack.segments]


    covered_segment_ids: list[str] = []


    seen_bundle_ids: set[str] = set()


    for bundle in fact_pack.semantic_bundles:


        if bundle.bundle_id in seen_bundle_ids:


            raise ProtocolViolation(f"Gate1 失败：bundle_id 重复：{bundle.bundle_id}")


        seen_bundle_ids.add(bundle.bundle_id)


        if not bundle.segment_ids:


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 缺少 segment_ids")


        if not bundle.aggregation_reason:


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 缺少 aggregation_reason")


        if not bundle.coverage_frame_refs:


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 缺少 coverage_frame_refs")


        bundle_indexes: list[int] = []


        for segment_id in bundle.segment_ids:


            if segment_id not in all_segment_ids:


                raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 引用了不存在的 segment_id={segment_id}")


            mapped_bundle_id = fact_pack.segment_to_bundle_map.get(segment_id)


            if mapped_bundle_id != bundle.bundle_id:


                raise ProtocolViolation(f"Gate1 失败：segment_to_bundle_map[{segment_id}] 未正确回链 {bundle.bundle_id}")


            bundle_indexes.append(all_segment_ids.index(segment_id))


        if bundle_indexes != list(range(bundle_indexes[0], bundle_indexes[-1] + 1)):


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id}.segment_ids 必须连续，不允许跳段聚合")


        bundle_range = fact_pack.bundle_to_segment_range.get(bundle.bundle_id)


        if bundle_range is None:


            raise ProtocolViolation(f"Gate1 失败：bundle_to_segment_range 缺少 {bundle.bundle_id}")


        if bundle_range.start_segment_index != bundle_indexes[0] or bundle_range.end_segment_index != bundle_indexes[-1]:


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 的 bundle_to_segment_range 序号不一致")


        if bundle_range.start_segment_id != bundle.segment_ids[0] or bundle_range.end_segment_id != bundle.segment_ids[-1]:


            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 的 bundle_to_segment_range segment_id 不一致")


        expected_start_sec = request.fact_pack.segments[bundle_indexes[0]].start_sec
        expected_end_sec = request.fact_pack.segments[bundle_indexes[-1]].end_sec
        if bundle.start_sec != expected_start_sec or bundle.end_sec != expected_end_sec:
            raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 的起止时间与物理 segments 不一致")

        for coverage_ref in bundle.coverage_frame_refs:
            normalized_ref = str(coverage_ref).strip()
            if not any(re.match(rf"^{re.escape(segment_id)}(?:[:_]|$)", normalized_ref) for segment_id in bundle.segment_ids):
                raise ProtocolViolation(f"Gate1 失败：{bundle.bundle_id} 的 coverage_frame_refs 超出 bundle segment 范围")

        covered_segment_ids.extend(bundle.segment_ids)


    if sorted(covered_segment_ids) != sorted(all_segment_ids):


        raise ProtocolViolation("Gate1 失败：semantic_bundles 必须完整且唯一覆盖全部 segments")











def _assert_provenance_input(request: VideoUnderstandingRequest | AssetIngestRequest) -> None:



    # Gate 2：输入 provenance 完整性 + 路由一致性
    if request.provenance.producer_type == "system_native_inference":
        raise ProtocolViolation("Gate2 失败：输入 provenance 伪装为 system_native_inference")
    if not request.provenance.generator_version or not str(request.provenance.generator_version).strip():
        raise ProtocolViolation("Gate2 失败：provenance.generator_version 缺失")
    if not request.provenance.generated_at or not str(request.provenance.generated_at).strip():
        raise ProtocolViolation("Gate2 失败：provenance.generated_at 缺失")


def _assert_gate0_arbitration(raw_payload: dict[str, Any]) -> Literal["fact_pack", "asset_package"]:
    # Gate 0：输入形态仲裁（必须先做）
    hit_deprecated = [k for k in raw_payload.keys() if k in DEPRECATED_OLD_FIELDS]
    if hit_deprecated:
        raise ProtocolViolation(f"Gate0 失败：检测到旧字段（已废弃，不允许运行时兼容）：{hit_deprecated}")

    has_fact = "fact_pack" in raw_payload
    has_asset = "asset_package" in raw_payload
    if has_fact and has_asset:
        raise ProtocolViolation("Gate0 失败：同一请求同时混带 fact_pack 与 asset_package")
    if has_fact:
        # PRD 7.1/7.4：FactPack 推理轨要求“全 payload 纯净”，禁止答案字段藏匿到任何角落。
        hit_anywhere = _find_forbidden_field_paths(raw_payload, FORBIDDEN_ANSWER_KEYS_IN_FACTPACK)
        if hit_anywhere:
            raise ProtocolViolation(f"Gate0 失败：检测到答案/策略字段（含藏匿），禁止进入推理轨：{hit_anywhere}")
        return "fact_pack"

    if has_asset:
        # PRD 7.1：禁止同一请求在 asset_package 之外再混带答案字段/策略字段。
        payload_without_asset = {k: v for k, v in raw_payload.items() if k != "asset_package"}
        hit_outside_asset = _find_forbidden_field_paths(payload_without_asset, FORBIDDEN_ANSWER_KEYS_IN_FACTPACK)
        if hit_outside_asset:
            raise ProtocolViolation(
                f"Gate0 失败：asset_package 轨道外出现答案/策略字段，属于混带：{hit_outside_asset}"
            )
        return "asset_package"

    raise ProtocolViolation("Gate0 失败：无法仲裁 payload_kind；必须显式提供 fact_pack 或 asset_package")


_MALICIOUS_COMPARISON_PATTERN = re.compile(r"(有毒|致癌|毁头皮|烂脸|别再用|千万别用)")
_MALICIOUS_OLD_SOLUTION_PATTERN = re.compile(
    r"((外面那种|市面上.*那种|旧方案|老方案|普通款|别家|其他家).*(全是添加剂|都是添加剂|一堆添加剂|化学成分|科技与狠活|谁敢|不敢给.*吃|不能给.*吃))|((全是添加剂|都是添加剂|一堆添加剂).*(谁敢|不敢给.*吃|不能给.*吃))"
)
_E4_BLACKLIST_CATEGORY_PATTERN = re.compile(r"(洗发水|牙膏|抗老|防晒|粉底|底妆)")
_PRICE_OR_BENEFIT_PATTERN = re.compile(r"(¥|￥|元|到手|买一送一|领券|券后|立减|直降|第2件|第二件|半价|优惠)")
_COGNITIVE_CONFLICT_PATTERN = re.compile(r"(很多人|以为|其实|真相|误区|别被|都错了)")
_H5_EXTREME_SCENARIO_PATTERN = re.compile(r"(火焰山|暴晒|地表温度|超过60度|高温|极端环境|无遮挡|没做任何防晒|没有任何遮挡)")
_H5_CHALLENGE_PATTERN = re.compile(r"(挑战|测试|扛得住|能不能扛住|这都敢|这种情况下|会怎样|现场.*测)")
_H6_SCENE_PATTERN = re.compile(r"(下班|好累|想放松|治愈|熬夜|通勤|约会|出差)")
_H6_PROBLEM_PATTERN = re.compile(r"(问题|残留|泛红|出油|黑头|毛孔|卡粉|脱妆|异味|头屑|干痒)")
_SUBJECTIVE_SENSORY_PATTERN = re.compile(r"(摸起来|很丝滑|一抹就化|很清爽|有痛感|肤感|闻起来)")
_OBJECTIVE_SENSORY_EQUIVALENCE_PATTERN = re.compile(r"(爆汁|拉丝|酥脆声|咔嚓|起泡声音|香味扑鼻|颗粒感清晰|烟雾可视)")
_PROOF_SIGNAL_PATTERN = re.compile(r"(对比|证明|实测|结果|前后|变化|检测)")

# taxonomy v2：H7 明星/权威同款
_H7_ENDORSEMENT_PATTERN = re.compile(
    r"(明星|女明星|男明星|同款|同款妆|同款穿搭|大V|博主同款|权威|专家|医生|医师|主任|教授|院士|央视|认证|皮肤科|牙医)"
)

# taxonomy v2：H3 反差结果前置（最小信号，避免滥打）
_H3_BEFORE_AFTER_PATTERN = re.compile(r"(前后|before|after|对比|左边|右边|半边|一边)")

# taxonomy v2：H4 即时操作展示（动作即看点）
_H4_OPERATION_PATTERN = re.compile(r"(一镜到底|现在就|直接|马上)(.*)(画|涂|抹|用|操作|开箱|拆|挤|倒|喷)")

# Effect v2：教程/成分/溯源
_E5_TUTORIAL_PATTERN = re.compile(r"(教程|手把手|步骤|怎么用|教你|新手|跟着做|正确方法|错误方法)")
_E6_INGREDIENT_PATTERN = re.compile(r"(成分|配方|配料|含量|浓度|参数|技术|专利|数据|检测报告|证书|标准)")
_E7_ORIGIN_PATTERN = re.compile(r"(产地|原产地|溯源|工厂|车间|生产线|基地|牧场|仓库|无菌)")
_E2_STRESS_TEST_PATTERN = re.compile(r"(暴力|极限|摔|砸|电钻|承重|浸水|高温|低温|喷火|强酸|强碱)")
_E2_HARD_STRESS_PATTERN = re.compile(r"(暴力|极限|摔|砸|电钻|承重|浸水|喷火|强酸|强碱)")
_E1_TEST_PATTERN = re.compile(r"(防水|防汗|耐磨|持妆|测|测试|实测|验证)")

# CTA v2
_C2_MECHANISM_PATTERN = re.compile(r"(运费险|包退|退换|无理由|买一送|赠|送.*(正装|小样)|保障|兜底)")
_C3_DIRECTIVE_PATTERN = re.compile(r"(下单|拍下|点击|小黄车|链接|冲|赶紧|直接买|备起来)")
_C4_AUDIENCE_PATTERN = re.compile(r"(姐妹|宝妈|打工人|学生|油皮|干皮|混油|敏感肌|黄黑皮|手残党)")


def _load_taxonomy_dictionary_v2() -> str:
    """加载字典文本，用于 LLM Prompt。

    这里不做强依赖；文件缺失时返回空串。
    """

    search_roots = [Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]]
    path = None
    for base in search_roots:
        candidate = base / "memory" / "topics" / "taxonomy_dictionary_v2.md"
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _is_llm_enabled() -> bool:
    return resolve_llm_config().is_configured and requests is not None


def _call_openai_chat_json(
    *,
    system: str,
    user: dict[str, Any],
    model: str = "doubao-1.5-pro-32k-250115",
    llm_tag: str = "video_understanding_module_v2_hec_judge",
) -> dict[str, Any]:
    """最小 OpenAI-compatible JSON 调用。

    本地未配置 provider 时请在外层做 Gate，避免在这里吞错。
    """

    llm_config = require_llm_config(resolve_llm_config(model=model), purpose="视频理解 HEC Judge")
    base_url = llm_config.endpoint
    if requests is None:
        raise RuntimeError("缺少 requests 依赖")

    headers = build_chat_headers(llm_config, llm_tag=llm_tag)
    payload = {
        "model": llm_config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=llm_config.timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\\w*\\n?", "", cleaned)
        cleaned = re.sub(r"\\n?```$", "", cleaned)
    return json.loads(cleaned)



def _serialize_factpack_segment(seg: FactPackSegment) -> dict[str, Any]:
    return {
        "segment_id": seg.segment_id,
        "start_sec": seg.start_sec,
        "end_sec": seg.end_sec,
        "visual_facts": seg.visual_facts.dict(),
        "audio_facts": seg.audio_facts.dict(),
        "ocr_facts": [ocr.dict() for ocr in seg.ocr_facts],
        "rhythm_facts": seg.rhythm_facts.dict(),
    }



def _merge_unique_strings(values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return merged



def _select_segment_level_refs(segment_id: str, refs: list[str]) -> list[str]:
    matched = [str(ref) for ref in refs if str(ref).startswith(segment_id)]
    if matched:
        return matched
    return []



def _select_segment_level_boundary_ids(segment_id: str, boundary_ids: list[str]) -> list[str]:
    return [str(boundary_id) for boundary_id in boundary_ids if segment_id in str(boundary_id)]



def _build_segment_storyboard_unit(segment: FactPackSegment, bundle: SemanticBundle) -> dict[str, Any]:
    coverage_frame_refs = _select_segment_level_refs(segment.segment_id, bundle.coverage_frame_refs)
    if not coverage_frame_refs:
        if len(bundle.segment_ids) == 1:
            coverage_frame_refs = [str(ref) for ref in bundle.coverage_frame_refs]
        else:
            raise ProtocolViolation(
                f"Stage4 失败：{bundle.bundle_id} 缺少 {segment.segment_id} 的独立 coverage_frame_refs，无法输出严格 segment 级 storyboard"
            )

    return {
        "segment_id": segment.segment_id,
        "start_sec": segment.start_sec,
        "end_sec": segment.end_sec,
        "visual_facts": segment.visual_facts.dict(),
        "audio_facts": segment.audio_facts.dict(),
        "ocr_facts": [ocr.dict() for ocr in segment.ocr_facts],
        "rhythm_facts": segment.rhythm_facts.dict(),
        "member_segment_ids": [segment.segment_id],
        "aggregation_reason": ["segment_level_storyboard_unit"],
        "coverage_frame_refs": coverage_frame_refs,
        "blocked_boundary_ids": _select_segment_level_boundary_ids(segment.segment_id, bundle.blocked_boundary_ids),
    }



def _storyboard_units_from_fact_pack(fact_pack: FactPack) -> list[dict[str, Any]]:
    if fact_pack.storyboard_source != "semantic_bundles":
        raise ProtocolViolation("storyboard_source 非 semantic_bundles，禁止回退到 fact_pack.segments 生成 storyboard")
    if not fact_pack.semantic_bundles:
        raise ProtocolViolation("storyboard_source=semantic_bundles 但 fact_pack.semantic_bundles 缺失")

    bundles_by_id = {bundle.bundle_id: bundle for bundle in fact_pack.semantic_bundles}
    storyboard_units: list[dict[str, Any]] = []
    for segment in fact_pack.segments:
        bundle_id = fact_pack.segment_to_bundle_map.get(segment.segment_id)
        if not bundle_id:
            raise ProtocolViolation(f"Stage4 失败：{segment.segment_id} 缺少 segment_to_bundle_map 回链")
        bundle = bundles_by_id.get(bundle_id)
        if bundle is None:
            raise ProtocolViolation(f"Stage4 失败：{segment.segment_id} 回链的 bundle_id={bundle_id} 不存在")
        storyboard_units.append(_build_segment_storyboard_unit(segment, bundle))
    return storyboard_units



def _assert_bridge_asset_contract(payload: dict[str, Any], *, error_prefix: str) -> None:
    audio_event_projection = payload.get("audio_event_projection") or {}
    if not isinstance(audio_event_projection, dict):
        raise ProtocolViolation(f"{error_prefix} audio_event_projection 必须为对象")
    for field_name in ("sfx_events", "bgm_events"):
        events = audio_event_projection.get(field_name)
        if not isinstance(events, list):
            raise ProtocolViolation(f"{error_prefix} audio_event_projection.{field_name} 必须为数组")
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ProtocolViolation(f"{error_prefix} audio_event_projection.{field_name}[{index}] 必须为对象")
            required_keys = {"start_sec", "end_sec", "trigger_sec", "business_role", "source_evidence"}
            required_keys.add("event_name" if field_name == "sfx_events" else "tone")
            missing_keys = [key for key in required_keys if key not in event]
            if missing_keys:
                raise ProtocolViolation(
                    f"{error_prefix} audio_event_projection.{field_name}[{index}] 缺少字段 {missing_keys}"
                )
            if float(event["start_sec"]) >= float(event["end_sec"]):
                raise ProtocolViolation(
                    f"{error_prefix} audio_event_projection.{field_name}[{index}] 要求 start_sec < end_sec"
                )
            trigger_sec = float(event["trigger_sec"])
            if trigger_sec < float(event["start_sec"]) or trigger_sec > float(event["end_sec"]):
                raise ProtocolViolation(
                    f"{error_prefix} audio_event_projection.{field_name}[{index}].trigger_sec 必须落在事件时间窗内"
                )
            if not str(event.get("business_role") or "").strip():
                raise ProtocolViolation(
                    f"{error_prefix} audio_event_projection.{field_name}[{index}].business_role 不能为空"
                )
            source_evidence = event.get("source_evidence")
            if not isinstance(source_evidence, list) or not all(str(item).strip() for item in source_evidence):
                raise ProtocolViolation(
                    f"{error_prefix} audio_event_projection.{field_name}[{index}].source_evidence 必须为非空字符串数组"
                )

    if not isinstance(payload.get("is_key_bridge"), bool):
        raise ProtocolViolation(f"{error_prefix} is_key_bridge 必须为布尔值")
    for field_name in ("reusable_clip_notes", "risk_bridge_notes"):
        notes = payload.get(field_name)
        if not isinstance(notes, list):
            raise ProtocolViolation(f"{error_prefix} {field_name} 必须为数组")
        for index, note in enumerate(notes):
            if not isinstance(note, dict):
                raise ProtocolViolation(f"{error_prefix} {field_name}[{index}] 必须为对象")
            if not str(note.get("note") or "").strip():
                raise ProtocolViolation(f"{error_prefix} {field_name}[{index}].note 不能为空")
            source_evidence = note.get("source_evidence")
            if not isinstance(source_evidence, list) or not all(str(item).strip() for item in source_evidence):
                raise ProtocolViolation(
                    f"{error_prefix} {field_name}[{index}].source_evidence 必须为非空字符串数组"
                )
            if field_name == "risk_bridge_notes":
                if not str(note.get("risk_type") or "").strip():
                    raise ProtocolViolation(f"{error_prefix} risk_bridge_notes[{index}].risk_type 不能为空")
                if str(note.get("risk_level") or "").strip() not in {"low", "medium", "high"}:
                    raise ProtocolViolation(f"{error_prefix} risk_bridge_notes[{index}].risk_level 非法")


def _assert_segment_level_storyboard_output(
    fact_pack: FactPack,
    storyboard_segments: list[dict[str, Any]],
) -> None:
    expected_segments = list(fact_pack.segments)
    if len(storyboard_segments) != len(expected_segments):
        raise ProtocolViolation("Stage4 失败：storyboard_segments 数量必须与 fact_pack.segments 完全一致")

    expected_ids = [segment.segment_id for segment in expected_segments]
    actual_ids = [str(segment.get("segment_id") or "").strip() for segment in storyboard_segments]
    if actual_ids != expected_ids:
        raise ProtocolViolation("Stage4 失败：storyboard_segments.segment_id 顺序必须与 fact_pack.segments 严格一致")

    seen_segment_ids: set[str] = set()
    for expected_segment, storyboard_segment in zip(expected_segments, storyboard_segments):
        segment_id = storyboard_segment["segment_id"]
        if segment_id in seen_segment_ids:
            raise ProtocolViolation(f"Stage4 失败：storyboard_segments.segment_id 重复：{segment_id}")
        seen_segment_ids.add(segment_id)

        member_segment_ids = list(storyboard_segment.get("member_segment_ids") or [])
        if member_segment_ids != [expected_segment.segment_id]:
            raise ProtocolViolation(
                f"Stage4 失败：{segment_id} 必须严格一对一回落到单个物理 segment，禁止 bundle 聚合残留"
            )

        start_sec = float(storyboard_segment.get("start_sec"))
        end_sec = float(storyboard_segment.get("end_sec"))
        if abs(start_sec - float(expected_segment.start_sec)) > 1e-6 or abs(end_sec - float(expected_segment.end_sec)) > 1e-6:
            raise ProtocolViolation(f"Stage4 失败：{segment_id} 时间轴必须与输入 segment 严格对齐")

        coverage_frame_refs = list(storyboard_segment.get("coverage_frame_refs") or [])
        if not coverage_frame_refs:
            raise ProtocolViolation(f"Stage4 失败：{segment_id} 缺少 coverage_frame_refs")
        if any(not str(ref).startswith(expected_segment.segment_id) for ref in coverage_frame_refs):
            raise ProtocolViolation(f"Stage4 失败：{segment_id} coverage_frame_refs 必须只回链本 segment")

        _assert_bridge_asset_contract(storyboard_segment, error_prefix=f"Stage4 失败：{segment_id}")



def _storyboard_unit_to_factpack_segment(unit: dict[str, Any]) -> FactPackSegment:
    return FactPackSegment.parse_obj(
        {
            "segment_id": unit["segment_id"],
            "start_sec": unit["start_sec"],
            "end_sec": unit["end_sec"],
            "visual_facts": unit["visual_facts"],
            "audio_facts": unit["audio_facts"],
            "ocr_facts": unit.get("ocr_facts", []),
            "rhythm_facts": unit["rhythm_facts"],
        }
    )



def _collect_segment_text(seg: FactPackSegment) -> str:
    action_names = "、".join(str(action.get("action_name") or "") for action in seg.visual_facts.actions if action.get("action_name"))
    ocr_text = "\n".join(ocr.text for ocr in seg.ocr_facts)
    return "\n".join(
        part
        for part in (
            seg.audio_facts.asr_text or "",
            ocr_text,
            seg.visual_facts.visual_subject or "",
            action_names,
        )
        if part
    )


def _segment_overlap_sec(seg: FactPackSegment, start_sec: float, end_sec: float) -> float:
    return max(0.0, min(seg.end_sec, end_sec) - max(seg.start_sec, start_sec))


def _is_primary_window_segment(
    seg: FactPackSegment,
    *,
    start_sec: float,
    end_sec: float,
    min_overlap_sec: float = 3.0,
    max_spill_sec: float = 1.0,
    min_overlap_ratio: float = 0.6,
) -> bool:
    overlap_sec = _segment_overlap_sec(seg, start_sec, end_sec)
    if overlap_sec <= 0:
        return False
    seg_duration = max(seg.end_sec - seg.start_sec, 1e-6)
    spill_before = max(start_sec - seg.start_sec, 0.0)
    spill_after = max(seg.end_sec - end_sec, 0.0)
    if spill_before <= 1e-6 and spill_after <= 1e-6:
        return True
    return (
        overlap_sec >= min_overlap_sec
        and overlap_sec / seg_duration >= min_overlap_ratio
        and max(spill_before, spill_after) <= max_spill_sec
    )


def _strip_price_benefit_signal(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if _PRICE_OR_BENEFIT_PATTERN.search(line):
            line = _PRICE_OR_BENEFIT_PATTERN.sub("", line).strip(" ，,。；;：:")
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _collect_factpack_texts(fact_pack: FactPack) -> tuple[str, str, str, str]:
    segments = fact_pack.segments
    if not segments:
        return "", "", "", ""

    segments = sorted(segments, key=lambda seg: (seg.start_sec, seg.end_sec, seg.segment_id))
    duration = max(seg.end_sec for seg in segments)
    hook_cut = 5.0
    cta_cut = max(duration - 5.0, duration * 0.8)

    primary_hook_segments = [
        seg
        for seg in segments
        if _is_primary_window_segment(seg, start_sec=0.0, end_sec=hook_cut)
    ]
    hook_segments = primary_hook_segments
    if not hook_segments:
        hook_segments = [seg for seg in segments if _segment_overlap_sec(seg, 0.0, hook_cut) > 0]

    cta_segments = [
        seg
        for seg in segments
        if _is_primary_window_segment(seg, start_sec=cta_cut, end_sec=duration)
    ]
    if not cta_segments:
        cta_segments = [seg for seg in segments if _segment_overlap_sec(seg, cta_cut, duration) > 0]

    mid_segments = [
        seg
        for seg in segments
        if _segment_overlap_sec(seg, hook_cut, cta_cut) > 0
    ]
    if not mid_segments:
        mid_segments = segments

    hook_text = "\n".join(
        (
            _collect_segment_text(seg)
            if seg in primary_hook_segments
            else _strip_price_benefit_signal(_collect_segment_text(seg))
        )
        for seg in hook_segments
    )
    mid_text = "\n".join(_collect_segment_text(seg) for seg in mid_segments)
    cta_text = "\n".join(_collect_segment_text(seg) for seg in cta_segments)
    all_text = "\n".join(_collect_segment_text(seg) for seg in segments)
    return hook_text, mid_text, cta_text, all_text


def _append_hec_reason_note(risk_flags: dict[str, Any], note: str) -> dict[str, Any]:
    patched_flags = dict(risk_flags)
    normalized_note = str(note or "").strip()
    if not normalized_note:
        return patched_flags
    base_reason = str(patched_flags.get("hec_reason") or "").strip()
    patched_flags["hec_reason"] = f"{base_reason} {normalized_note}".strip() if base_reason else normalized_note
    return patched_flags



def _apply_hook_guardrails(
    primary_hec: dict[str, str],
    risk_flags: dict[str, Any],
    *,
    fact_pack: FactPack,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Hook 后置仲裁：若 Hook 被复核改写，risk_flags.hec_reason 必须显式对齐最终 hook_label。"""

    hook_text, _, _, _ = _collect_factpack_texts(fact_pack)
    has_scene_signal = bool(_H6_SCENE_PATTERN.search(hook_text))
    has_problem_signal = bool(_H6_PROBLEM_PATTERN.search(hook_text))
    has_audience_signal = bool(_C4_AUDIENCE_PATTERN.search(hook_text))
    h6_allowed = has_audience_signal or (has_scene_signal and has_problem_signal)
    has_price_or_benefit_signal = bool(_PRICE_OR_BENEFIT_PATTERN.search(hook_text))
    has_endorsement_signal = bool(_H7_ENDORSEMENT_PATTERN.search(hook_text))
    has_before_after_signal = bool(_H3_BEFORE_AFTER_PATTERN.search(hook_text))
    has_cognitive_conflict_signal = bool(_COGNITIVE_CONFLICT_PATTERN.search(hook_text))

    patched_hec = dict(primary_hec)
    patched_flags = dict(risk_flags)
    original_hook = str(patched_hec.get("hook_label") or "").strip()
    final_hook = original_hook
    guardrail_note = ""

    if h6_allowed:
        if final_hook != "H6":
            final_hook = "H6"
            guardrail_note = "Hook 仲裁复核：命中合法场景/人群代入证据，最终 Hook=H6。"
    else:
        demoted_from_h6 = final_hook == "H6"
        if demoted_from_h6:
            final_hook = "H1"
        if (
            has_cognitive_conflict_signal
            and not has_price_or_benefit_signal
            and not has_endorsement_signal
            and not has_before_after_signal
            and final_hook not in {"H2", "H3", "H6", "H7"}
        ):
            if final_hook != "H5":
                final_hook = "H5"
                if demoted_from_h6:
                    guardrail_note = "Hook 仲裁复核：原始 H6 缺少合法场景/人群代入证据，且命中反常识/强冲突信号，最终 Hook=H5。"
                else:
                    guardrail_note = "Hook 仲裁复核：命中反常识/强冲突信号，最终 Hook=H5。"
        elif demoted_from_h6:
            guardrail_note = "Hook 仲裁复核：原始 H6 缺少合法场景/人群代入证据，最终 Hook=H1。"

    if final_hook:
        patched_hec["hook_label"] = final_hook
    if guardrail_note:
        patched_flags = _append_hec_reason_note(patched_flags, guardrail_note)
    return patched_hec, patched_flags


def _apply_gate4_script_level_review(
    primary_hec: dict[str, str],
    risk_flags: dict[str, Any],
    *,
    fact_pack: FactPack,
) -> tuple[dict[str, str], dict[str, Any]]:
    """脚本级强制审查：只负责补充标准化 risk_flags 文案，不允许引入动态字段。"""

    _, _, _, all_text = _collect_factpack_texts(fact_pack)
    if not all_text:
        return primary_hec, risk_flags

    malicious_hit = bool(
        _MALICIOUS_COMPARISON_PATTERN.search(all_text)
        or _MALICIOUS_OLD_SOLUTION_PATTERN.search(all_text)
    )
    if not malicious_hit:
        return primary_hec, risk_flags

    patched_flags = dict(risk_flags)
    base_reason = str(patched_flags.get("hec_reason") or "").strip()
    gate4_note = "Gate4 命中恶意拉踩风险，需人工复核脚本级表述。"
    patched_flags["hec_reason"] = f"{base_reason} {gate4_note}".strip() if base_reason else gate4_note
    return dict(primary_hec), patched_flags


def _risk_flag_default_evidence_segment_ids(fact_pack: FactPack) -> list[str]:
    return [seg.segment_id for seg in fact_pack.segments if str(seg.segment_id).strip()]



def _finalize_risk_flags(
    fact_pack: FactPack,
    primary_hec: dict[str, str],
    secondary_effects: list[dict[str, Any]],
    risk_flags: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(risk_flags or {})
    inference_mode = str(raw.get("inference_mode") or "unknown").strip() or "unknown"
    hec_reason = str(raw.get("hec_reason") or "").strip()
    if not hec_reason:
        hook = str(primary_hec.get("hook_label") or "").strip() or "H0"
        effect = str(primary_hec.get("effect_label") or "").strip() or "E0"
        cta = str(primary_hec.get("cta_label") or "").strip() or "C5"
        hec_reason = f"{inference_mode} 裁决输出主骨架 {hook}/{effect}/{cta}。"
    evidence_segment_ids = raw.get("hec_evidence_segment_ids")
    if not isinstance(evidence_segment_ids, list) or not evidence_segment_ids:
        evidence_segment_ids = _risk_flag_default_evidence_segment_ids(fact_pack)
    normalized_ids: list[str] = []
    seen: set[str] = set()
    valid_segment_ids = {seg.segment_id for seg in fact_pack.segments}
    for item in evidence_segment_ids:
        segment_id = str(item).strip()
        if not segment_id or segment_id in seen or segment_id not in valid_segment_ids:
            continue
        seen.add(segment_id)
        normalized_ids.append(segment_id)
    if not normalized_ids:
        normalized_ids = _risk_flag_default_evidence_segment_ids(fact_pack)
    if not normalized_ids:
        raise ProtocolViolation("risk_flags.hec_evidence_segment_ids 无法归一化为合法 segment_id")
    return RiskFlags.parse_obj(
        {
            "inference_mode": inference_mode,
            "hec_reason": hec_reason,
            "hec_evidence_segment_ids": normalized_ids,
            "secondary_effects_present": bool(secondary_effects),
        }
    ).dict()


SECONDARY_EFFECT_REASON_HINTS = {
    "E1": "局部片段存在明确实测/结果验证，但整片主说服骨架由主 E 承担，因此作为副 E 保留。",
    "E3": "局部片段出现对比/替代论证，但没有接管整片主说服闭环，因此仅作为副 E。",
    "E4": "局部片段存在客观感官实证，但证据承担度不足以升格为主 E，因此仅作为副 E。",
    "E5": "局部片段承担了教程/步骤说明，但主要购买说服仍由主 E 完成，因此仅作为副 E。",
    "E6": "局部片段存在成分/机理/参数说明，但没有接管整片主证据承担，因此仅作为副 E。",
    "E7": "局部片段出现产地/工厂/溯源信息，但不足以构成整片主骨架，因此仅作为副 E。",
}


def _build_secondary_effect_reason(effect_label: str) -> str:
    return SECONDARY_EFFECT_REASON_HINTS.get(
        effect_label,
        "局部片段存在可回链的辅助 Effect 证据，但不足以升格为主 E。",
    )



def _normalize_secondary_effects_payload(secondary_effects: Any) -> list[dict[str, Any]]:
    if secondary_effects is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in secondary_effects:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        normalized_item: dict[str, Any] = {
            "effect_label": str(item.get("effect_label") or "").strip().upper(),
            "evidence_segment_ids": [
                str(segment_id or "").strip() for segment_id in list(item.get("evidence_segment_ids") or [])
            ],
        }
        reason = item.get("reason")
        if reason is not None and str(reason).strip():
            normalized_item["reason"] = str(reason).strip()
        normalized.append(normalized_item)
    return normalized



def _clean_secondary_effects_before_assertion(
    secondary_effects: Any,
    *,
    primary_effect_label: str | None,
) -> list[dict[str, Any]]:
    normalized = _normalize_secondary_effects_payload(secondary_effects)
    primary_effect = str(primary_effect_label or "").strip().upper()
    seen_labels: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for item in normalized:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        effect_label = str(item.get("effect_label") or "").strip().upper()
        if effect_label and (effect_label == primary_effect or effect_label in seen_labels):
            continue
        if effect_label:
            seen_labels.add(effect_label)
        cleaned.append(item)
    return cleaned



def _finalize_secondary_effects(
    secondary_effects: Any,
    *,
    fact_pack: FactPack,
    primary_effect_label: str | None,
) -> list[dict[str, Any]]:
    normalized = _normalize_secondary_effects_payload(secondary_effects)
    segment_duration_map = {
        seg.segment_id: max(seg.end_sec - seg.start_sec, 0.0)
        for seg in fact_pack.segments
    }
    total_duration = sum(segment_duration_map.values()) or 0.0
    seen_labels: set[str] = set()
    finalized: list[dict[str, Any]] = []
    primary_effect = str(primary_effect_label or "").strip().upper()

    for item in normalized:
        if not isinstance(item, dict):
            continue
        effect_label = str(item.get("effect_label") or "").strip().upper()
        if not effect_label or effect_label == primary_effect or effect_label in seen_labels:
            continue
        evidence_segment_ids = [
            segment_id
            for segment_id in list(item.get("evidence_segment_ids") or [])
            if segment_id in segment_duration_map
        ]
        if not evidence_segment_ids:
            continue
        evidence_duration = sum(segment_duration_map[segment_id] for segment_id in evidence_segment_ids)
        evidence_share = (evidence_duration / total_duration) if total_duration > 0 else 0.0
        if effect_label == "E3" and len(evidence_segment_ids) < 3 and evidence_share < 0.18:
            continue
        seen_labels.add(effect_label)
        finalized_item = {
            "effect_label": effect_label,
            "evidence_segment_ids": evidence_segment_ids,
            "reason": str(item.get("reason") or "").strip() or _build_secondary_effect_reason(effect_label),
        }
        finalized.append(finalized_item)
    return finalized



def _infer_secondary_effects_rule(
    fact_pack: FactPack,
    product: ProductSnapshot,
    *,
    primary_effect_label: str | None,
) -> list[dict[str, Any]]:
    segments = sorted(fact_pack.segments, key=lambda seg: (seg.start_sec, seg.end_sec, seg.segment_id))
    if not segments:
        return []

    total_duration = sum(max(seg.end_sec - seg.start_sec, 0.0) for seg in segments) or 0.0
    candidate_map: dict[str, dict[str, Any]] = {}
    primary_effect = str(primary_effect_label or "").strip().upper()

    for index, seg in enumerate(segments):
        seg_text = _collect_segment_text(seg)
        if not seg_text:
            continue
        seg_duration = max(seg.end_sec - seg.start_sec, 0.0)
        action_names = "、".join(
            str(action.get("action_name") or "") for action in seg.visual_facts.actions if action.get("action_name")
        )
        segment_labels: list[str] = []

        if (
            _MALICIOUS_COMPARISON_PATTERN.search(seg_text)
            or _MALICIOUS_OLD_SOLUTION_PATTERN.search(seg_text)
            or any(token in seg_text for token in ("旧款", "以前", "替换", "竞品", "别家", "外面那种", "对比"))
        ):
            segment_labels.append("E3")
        if _E7_ORIGIN_PATTERN.search(seg_text):
            segment_labels.append("E7")
        if _E6_INGREDIENT_PATTERN.search(seg_text):
            segment_labels.append("E6")
        if _E5_TUTORIAL_PATTERN.search(seg_text):
            segment_labels.append("E5")
        if _OBJECTIVE_SENSORY_EQUIVALENCE_PATTERN.search(seg_text) and not _E4_BLACKLIST_CATEGORY_PATTERN.search(
            product.leaf_category_name
        ):
            segment_labels.append("E4")
        if (
            _E1_TEST_PATTERN.search(seg_text)
            or any(token in seg_text for token in ("演示", "结果", "吸收", "验证", "测试", "实测", "测", "水分测试仪", "体温计"))
            or any(token in action_names for token in ("演示", "滴落", "测试", "验证"))
        ):
            segment_labels.append("E1")

        for label in dict.fromkeys(segment_labels):
            if label == primary_effect:
                continue
            entry = candidate_map.setdefault(
                label,
                {
                    "effect_label": label,
                    "evidence_segment_ids": [],
                    "_duration": 0.0,
                    "_first_index": index,
                },
            )
            if seg.segment_id not in entry["evidence_segment_ids"]:
                entry["evidence_segment_ids"].append(seg.segment_id)
            entry["_duration"] += seg_duration

    min_duration = max(2.0, total_duration * 0.10) if total_duration > 0 else 2.0
    ordered_candidates = sorted(
        candidate_map.values(),
        key=lambda item: (-float(item.get("_duration") or 0.0), int(item.get("_first_index") or 0)),
    )

    secondary_effects: list[dict[str, Any]] = []
    for candidate in ordered_candidates:
        evidence_segment_ids = list(candidate.get("evidence_segment_ids") or [])
        duration = float(candidate.get("_duration") or 0.0)
        if not evidence_segment_ids:
            continue
        if duration + 1e-6 < min_duration and len(evidence_segment_ids) < 2:
            continue
        effect_label = str(candidate.get("effect_label") or "").strip().upper()
        secondary_effects.append(
            {
                "effect_label": effect_label,
                "evidence_segment_ids": evidence_segment_ids,
                "reason": _build_secondary_effect_reason(effect_label),
            }
        )
    return secondary_effects



def _infer_primary_hec_rule_v2(fact_pack: FactPack, product: ProductSnapshot) -> tuple[dict[str, str], dict[str, Any]]:
    """无 LLM 环境时的最小可跑通 fallback（taxonomy v2 对齐）。"""

    segments = fact_pack.segments
    if not segments:
        return {"hook_label": "H1", "effect_label": "E0", "cta_label": "C5"}, {"inference_mode": "rule_fallback"}

    hook_text, mid_text, cta_text, all_text = _collect_factpack_texts(fact_pack)

    risk_flags: dict[str, Any] = {
        "inference_mode": "rule_fallback",
        "secondary_effects_present": False,
    }
    review_notes: list[str] = []

    # Hook（taxonomy v2）
    hook_cut = 5.0
    hook_overlap_segments = [
        seg
        for seg in sorted(segments, key=lambda seg: (seg.start_sec, seg.end_sec, seg.segment_id))
        if _segment_overlap_sec(seg, 0.0, hook_cut) > 0
    ]
    hook_price_segments = [
        seg
        for seg in hook_overlap_segments
        if _is_primary_window_segment(seg, start_sec=0.0, end_sec=hook_cut)
        and _PRICE_OR_BENEFIT_PATTERN.search(_collect_segment_text(seg))
    ]
    first_hook_segment_text = _collect_segment_text(hook_overlap_segments[0]) if hook_overlap_segments else ""
    has_primary_problem_narrative = bool(
        _H6_PROBLEM_PATTERN.search(first_hook_segment_text)
        or _COGNITIVE_CONFLICT_PATTERN.search(first_hook_segment_text)
    )
    has_hook_problem_signal = any(
        _H6_PROBLEM_PATTERN.search(_collect_segment_text(seg))
        or _COGNITIVE_CONFLICT_PATTERN.search(_collect_segment_text(seg))
        for seg in hook_overlap_segments
    )
    has_scene_signal = bool(_H6_SCENE_PATTERN.search(hook_text))
    has_problem_signal = bool(_H6_PROBLEM_PATTERN.search(hook_text))
    has_audience_signal = bool(_C4_AUDIENCE_PATTERN.search(hook_text))
    has_eligible_price_signal = bool(hook_price_segments)
    h2_blocked_by_primary_narrative = has_eligible_price_signal and has_primary_problem_narrative and has_hook_problem_signal
    has_extreme_challenge_signal = bool(
        _H5_EXTREME_SCENARIO_PATTERN.search(first_hook_segment_text)
        and (
            _H5_CHALLENGE_PATTERN.search(hook_text)
            or _E2_STRESS_TEST_PATTERN.search(hook_text)
            or _COGNITIVE_CONFLICT_PATTERN.search(hook_text)
            or any(token in first_hook_segment_text for token in ("吗", "？", "?"))
        )
    )

    if h2_blocked_by_primary_narrative:
        hook = "H1"
        review_notes.append("价格型 Hook 因主叙事已形成强问题冲突，被规则降级为 H1。")
    elif has_eligible_price_signal:
        hook = "H2"
    elif _H7_ENDORSEMENT_PATTERN.search(hook_text):
        hook = "H7"
    elif has_audience_signal or (has_scene_signal and has_problem_signal):
        hook = "H6"
    elif _H3_BEFORE_AFTER_PATTERN.search(hook_text):
        hook = "H3"
    elif has_extreme_challenge_signal or _COGNITIVE_CONFLICT_PATTERN.search(hook_text):
        hook = "H5"
    else:
        has_action = any(seg.visual_facts.actions for seg in segments if seg.start_sec < 5.0)
        if has_action or _H4_OPERATION_PATTERN.search(hook_text):
            hook = "H4"
        else:
            hook = "H1"

    # Effect（taxonomy v2）
    total_duration = sum(max(seg.end_sec - seg.start_sec, 0.0) for seg in segments) or 0.0
    e1_demo_duration = 0.0
    e3_comparison_duration = 0.0
    for seg in segments:
        seg_text = _collect_segment_text(seg)
        seg_duration = max(seg.end_sec - seg.start_sec, 0.0)
        action_names = "、".join(
            str(action.get("action_name") or "") for action in seg.visual_facts.actions if action.get("action_name")
        )
        if (
            _MALICIOUS_COMPARISON_PATTERN.search(seg_text)
            or _MALICIOUS_OLD_SOLUTION_PATTERN.search(seg_text)
            or any(token in seg_text for token in ("旧款", "以前", "替换", "竞品", "别家", "外面那种"))
        ):
            e3_comparison_duration += seg_duration
        if (
            _E1_TEST_PATTERN.search(seg_text)
            or any(token in seg_text for token in ("演示", "结果", "吸收", "验证", "测试", "实测", "滴落"))
            or any(token in action_names for token in ("演示", "滴落", "测试", "验证"))
        ):
            e1_demo_duration += seg_duration

    e1_is_primary_bone = (
        total_duration > 0
        and e1_demo_duration / total_duration >= 0.55
        and e1_demo_duration > e3_comparison_duration
    )

    if e1_is_primary_bone:
        effect = "E1"
    elif _MALICIOUS_COMPARISON_PATTERN.search(all_text):
        effect = "E3"
        review_notes.append("检测到疑似恶意拉踩语义，Effect 维度按 E3 处理，需 Gate4 复核。")
    elif _E7_ORIGIN_PATTERN.search(mid_text):
        effect = "E7"
    elif _E6_INGREDIENT_PATTERN.search(mid_text):
        effect = "E6"
    elif _E5_TUTORIAL_PATTERN.search(mid_text):
        effect = "E5"
    elif any(token in mid_text for token in ("对比", "旧款", "以前", "替换", "竞品")):
        effect = "E3"
    elif _E2_STRESS_TEST_PATTERN.search(mid_text):
        effect = "E2"
    elif _SUBJECTIVE_SENSORY_PATTERN.search(mid_text):
        # taxonomy v2：主观肤感不等价于功效 -> 降级 E0
        effect = "E0"
        review_notes.append("主观感受表达不构成 E4 客观感官实证，已按规则降级。")
    elif _OBJECTIVE_SENSORY_EQUIVALENCE_PATTERN.search(mid_text):
        if _E4_BLACKLIST_CATEGORY_PATTERN.search(product.leaf_category_name):
            effect = "E0"
            review_notes.append("命中 E4 类目黑名单，客观感官等价表达不允许升格为 E4。")
        else:
            effect = "E4"
    elif _E1_TEST_PATTERN.search(mid_text) or e1_demo_duration > 0:
        effect = "E1"
    else:
        effect = "E0"

    # CTA（taxonomy v2）
    if _PRICE_OR_BENEFIT_PATTERN.search(cta_text):
        cta = "C1"
    elif _C2_MECHANISM_PATTERN.search(cta_text):
        cta = "C2"
    elif _C3_DIRECTIVE_PATTERN.search(cta_text):
        cta = "C3"
    elif _C4_AUDIENCE_PATTERN.search(cta_text):
        cta = "C4"
    else:
        cta = "C5"

    risk_flags["hec_reason"] = " ".join(
        [
            f"rule_fallback 裁决输出主骨架 {hook}/{effect}/{cta}。",
            *review_notes,
        ]
    ).strip()
    risk_flags["hec_evidence_segment_ids"] = _risk_flag_default_evidence_segment_ids(fact_pack)
    return {"hook_label": hook, "effect_label": effect, "cta_label": cta}, risk_flags


def _infer_primary_hec_and_risks(
    fact_pack: FactPack,
    product: ProductSnapshot,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """HEC 裁决（LLM-first）。

    - 若配置了 OPENAI_BASE_URL/OPENAI_API_KEY：走大模型裁决，输出 taxonomy v2 的 H/E/C 编号。
    - 否则：走本地最小规则 fallback（同样对齐 taxonomy v2），保证链路可跑通。
    """

    if not fact_pack.segments:
        return (
            {"hook_label": "H1", "effect_label": "E0", "cta_label": "C5"},
            [],
            {"inference_mode": "empty_factpack"},
        )

    if not _is_llm_enabled():
        primary_hec, risk_flags = _infer_primary_hec_rule_v2(fact_pack, product)
        secondary_effects = _finalize_secondary_effects(
            _infer_secondary_effects_rule(
                fact_pack,
                product,
                primary_effect_label=primary_hec.get("effect_label"),
            ),
            fact_pack=fact_pack,
            primary_effect_label=primary_hec.get("effect_label"),
        )
        return primary_hec, secondary_effects, _finalize_risk_flags(fact_pack, primary_hec, secondary_effects, risk_flags)

    taxonomy_text = _load_taxonomy_dictionary_v2()
    segments_payload = [
        {
            "segment_id": seg.segment_id,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "asr_text": seg.audio_facts.asr_text,
            "ocr_text": [o.text for o in seg.ocr_facts],
            "visual_subject": seg.visual_facts.visual_subject,
            "key_objects": seg.visual_facts.key_objects,
            "actions": seg.visual_facts.actions,
        }
        for seg in fact_pack.segments
    ]

    system = (
        "你是短视频三段式 HEC 标签判定器（taxonomy v2）。\n"
        "只允许输出 Hook=H1-H7、Effect=E0-E7、CTA=C1-C5 的编号，不得输出旧版标签或自造标签。\n"
        "输出必须是严格 JSON 对象，字段：hook_label,effect_label,cta_label,reason,evidence_segment_ids,secondary_effects。\n"
        "其中 evidence_segment_ids 必须是数组，元素必须来自输入的 segment_id。\n"
        "secondary_effects 必须是数组；每个 item 只能包含 effect_label,evidence_segment_ids,reason。\n"
        "secondary_effects 的正确定义：它是去重后的副 Effect 类型列表；每个 item 代表一种副 E 类型及其最强证据，不是把同一种 Effect 的多个证据块逐条罗列，更不是主 E 证据补充区。\n"
        "主副 E 裁决必须遵守：先判主 E，再判副 E；primary effect 只能有一个，secondary_effects 只允许承接局部/辅助/补强型 Effect，绝不能接管整片主说服骨架。\n"
        "生成顺序必须固定为三步：1）先基于整片闭环选出唯一主 E，写入 effect_label；2）再只从剩余 Effect 类型里筛选副 E；3）每种副 E 最多输出一次，并为该类型挑选最能支撑它的 evidence_segment_ids。\n"
        "主 E 判断标准：以完成购买说服为准、以主证据承担关系为准、以整片闭环为准。\n"
        "如果 E1 与 E6 并存，必须判断哪个 Evidence 真正解决用户下单前的关键顾虑；承担主骨架者留在 effect_label，另一个若证据充分则进入 secondary_effects。\n"
        "E1 与 E2 的边界：自然场景下的前后对照、仪器测值、补水/降温/吸收等效果验证，归 E1；只有对产品或效果进行破坏性、极限承压、超常损伤测试（如摔砸、承重、浸水、喷火、强酸强碱）时，才归 E2。单纯高温/暴晒环境本身不等于 E2。\n"
        "secondary_effects[*].reason 用一句话解释它为什么成立，以及为什么没有升格为主 E。\n"
        "secondary_effects 只保留形成稳定局部说服单元的副 E；一两句顺带提及、没有独立证据块的弱噪音不要输出。\n"
        "输出前强制自检：若 secondary_effects 中任一 effect_label 与主 effect_label 重复，或 secondary_effects 内部出现重复 effect_label，必须删除重复项后再输出，绝不允许主副 E 重复出现在最终 JSON。\n"
        "Effect 判定必须遵守主体边界铁律：\n"
        "1. `E6` 只用于讲解本产品自身的成分、参数、证书、标准、机理。\n"
        "2. 如果文案在讲外部方案、旧方案、普通款、他牌、外面那种产品的成分/添加剂/参数/安全性，并借此证明对方不好、旧方案不行或本品更值得替代，本质属于 `E3`，即使出现大量成分词，也绝对不能判成 `E6`。\n"
        "3. Gate 4 风险标记只负责审查合规风险，不负责替你改标签；标签必须先按语义本身判对。\n"
        "4. 判定 Effect 主骨架时，必须基于全片的篇幅占比和信息密度，先看哪一种表达承担了大部分说服任务。\n"
        "5. 严禁被局部的拉踩、妖魔化、刺激词汇绑架全局标签；如果全片大部分仍是单点演示/结果展示，就必须判为 `E1`，局部拉踩只算 `E3` 辅助噪音。\n"
        "6. 只有当旧方案对比/替换论证本身占据主要篇幅，且承担主说服任务时，才把全局 Effect 判为 `E3`。\n"
        "\n[taxonomy_dictionary_v2]\n" + (taxonomy_text[:8000] if taxonomy_text else "(missing)")
    )
    user = {
        "task": "根据 FactPack 分镜事实判定整条视频的主 H/E/C，并在不改变主 E 唯一性的前提下补充 secondary_effects。",
        "product_snapshot": {
            "leaf_category_name": product.leaf_category_name,
            "product_name": product.product_name,
            "brand_name": product.brand_name,
        },
        "fact_pack": {
            "video_meta": fact_pack.video_meta.dict(),
            "segments": segments_payload,
        },
    }

    try:
        parsed = _call_openai_chat_json(system=system, user=user)
        hook = str(parsed.get("hook_label") or "").strip()
        effect = str(parsed.get("effect_label") or "").strip().upper()
        cta = str(parsed.get("cta_label") or "").strip()
        evidence_segment_ids = parsed.get("evidence_segment_ids")

        allowed_hook = {f"H{i}" for i in range(1, 8)}
        allowed_effect = {f"E{i}" for i in range(0, 8)}
        allowed_cta = {f"C{i}" for i in range(1, 6)}
        if hook not in allowed_hook or effect not in allowed_effect or cta not in allowed_cta:
            raise ProtocolViolation(f"LLM HEC 输出非法：hook={hook!r}, effect={effect!r}, cta={cta!r}")

        valid_segment_ids = {seg.segment_id for seg in fact_pack.segments}
        _, mid_text, _, _ = _collect_factpack_texts(fact_pack)
        e1_demo_duration = sum(
            max(seg.end_sec - seg.start_sec, 0.0)
            for seg in fact_pack.segments
            if _E1_TEST_PATTERN.search(_collect_segment_text(seg))
            or any(
                token in "、".join(
                    str(action.get("action_name") or "") for action in seg.visual_facts.actions if action.get("action_name")
                )
                for token in ("演示", "测试", "验证", "测量")
            )
        )
        effect_guardrail_e2_to_e1 = False
        if effect == "E2" and not _E2_HARD_STRESS_PATTERN.search(mid_text) and (
            _E1_TEST_PATTERN.search(mid_text) or e1_demo_duration > 0
        ):
            effect = "E1"
            effect_guardrail_e2_to_e1 = True
        if not isinstance(evidence_segment_ids, list) or not evidence_segment_ids:
            raise ProtocolViolation("LLM HEC 输出缺少 evidence_segment_ids")
        if any((not isinstance(x, str)) or x not in valid_segment_ids for x in evidence_segment_ids):
            raise ProtocolViolation(f"LLM HEC 输出 evidence_segment_ids 非法：{evidence_segment_ids}")

        secondary_effects = _clean_secondary_effects_before_assertion(
            parsed.get("secondary_effects"),
            primary_effect_label=effect,
        )
        _assert_secondary_effects_payload(
            secondary_effects,
            primary_effect_label=effect,
            valid_segment_ids=valid_segment_ids,
            error_prefix="LLM HEC 输出非法",
        )
        if not secondary_effects:
            secondary_effects = _infer_secondary_effects_rule(
                fact_pack,
                product,
                primary_effect_label=effect,
            )
        secondary_effects = _finalize_secondary_effects(
            secondary_effects,
            fact_pack=fact_pack,
            primary_effect_label=effect,
        )

        risk_flags: dict[str, Any] = {
            "inference_mode": "llm",
            "hec_reason": parsed.get("reason"),
            "hec_evidence_segment_ids": evidence_segment_ids,
        }
        if effect_guardrail_e2_to_e1:
            base_reason = str(risk_flags.get("hec_reason") or "").strip()
            override_reason = "规则复核将主 Effect 从 E2 降级为 E1。"
            risk_flags["hec_reason"] = f"{base_reason} {override_reason}".strip() if base_reason else override_reason
        primary_hec = {"hook_label": hook, "effect_label": effect, "cta_label": cta}
        primary_hec, risk_flags = _apply_gate4_script_level_review(primary_hec, risk_flags, fact_pack=fact_pack)
        return primary_hec, secondary_effects, _finalize_risk_flags(fact_pack, primary_hec, secondary_effects, risk_flags)
    except Exception as exc:
        primary_hec, risk_flags = _infer_primary_hec_rule_v2(fact_pack, product)
        secondary_effects = _finalize_secondary_effects(
            _infer_secondary_effects_rule(
                fact_pack,
                product,
                primary_effect_label=primary_hec.get("effect_label"),
            ),
            fact_pack=fact_pack,
            primary_effect_label=primary_hec.get("effect_label"),
        )
        patched_flags = dict(risk_flags)
        fallback_note = f"LLM 裁决失败，已回退规则裁决：{str(exc).strip()}"
        base_reason = str(patched_flags.get("hec_reason") or "").strip()
        patched_flags["hec_reason"] = f"{base_reason} {fallback_note}".strip() if base_reason else fallback_note
        primary_hec, patched_flags = _apply_gate4_script_level_review(primary_hec, patched_flags, fact_pack=fact_pack)
        return primary_hec, secondary_effects, _finalize_risk_flags(fact_pack, primary_hec, secondary_effects, patched_flags)


def _normalize_slider_score(raw_score: Any, *, dimension: str) -> int:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        raise ProtocolViolation(f"LLM Slider 输出 {dimension}.score 非法：{raw_score!r}")
    if score < 0 or score > 10:
        raise ProtocolViolation(f"LLM Slider 输出 {dimension}.score 超出范围 [0,10]：{score}")
    if abs(score - round(score)) > 1e-6:
        raise ProtocolViolation(f"LLM Slider 输出 {dimension}.score 必须为整数：{score}")
    return int(round(score))



def _get_slider_dimension_score(slider_signature: dict[str, Any] | None, dimension: str) -> int:
    item = (slider_signature or {}).get(dimension, 0)
    if isinstance(item, dict):
        item = item.get("score", 0)
    try:
        return int(item)
    except (TypeError, ValueError):
        return 0


def _build_secondary_effect_segment_lookup(
    secondary_effects: list[dict[str, Any]] | None,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in secondary_effects or []:
        if not isinstance(item, dict):
            continue
        effect_label = str(item.get("effect_label") or "").strip().upper()
        if not effect_label:
            continue
        for segment_id in list(item.get("evidence_segment_ids") or []):
            normalized_segment_id = str(segment_id or "").strip()
            if normalized_segment_id and normalized_segment_id not in lookup:
                lookup[normalized_segment_id] = effect_label
    return lookup


def _get_segment_secondary_effect_label(
    seg: FactPackSegment,
    secondary_effects: list[dict[str, Any]] | None,
) -> str | None:
    return _build_secondary_effect_segment_lookup(secondary_effects).get(seg.segment_id)



def _neutral_slider_signature_meta(
    *,
    segments: list[FactPackSegment],
    inference_mode: str,
    fallback_reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    neutral_signature = SliderSignature.parse_obj(
        {
            "visual": {"score": 5, "business_judgment": "视觉表达处于中位占位，既不是粗暴直给，也谈不上精致质感。"},
            "audio": {"score": 5, "business_judgment": "音频表达处于中位占位，介于生活经验式表达与克制专业表达之间。"},
            "proof": {"score": 5, "business_judgment": "举证表达处于中位占位，介于看结果就信与讲机理才信之间。"},
            "cta": {"score": 5, "business_judgment": "收口表达处于中位占位，介于强逼单与顺势收单之间。"},
        }
    ).dict()
    fallback_segment_ids = [seg.segment_id for seg in segments[:2]] or []
    meta: dict[str, Any] = {
        "inference_mode": inference_mode,
        "summary": "Slider 未取得有效 LLM 结果，已回退到 10 分制中性四维光谱占位。",
        "dimension_reasons": {
            "visual": neutral_signature["visual"]["business_judgment"],
            "audio": neutral_signature["audio"]["business_judgment"],
            "proof": neutral_signature["proof"]["business_judgment"],
            "cta": neutral_signature["cta"]["business_judgment"],
        },
        "dimension_evidence_segment_ids": {
            "visual": fallback_segment_ids,
            "audio": fallback_segment_ids,
            "proof": fallback_segment_ids,
            "cta": fallback_segment_ids,
        },
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    return neutral_signature, meta



def _infer_slider_signature(
    fact_pack: FactPack,
    *,
    primary_hec: dict[str, str] | None = None,
    secondary_effects: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """全片级四维光谱裁决（LLM-first，10 分制整数）。"""
    segments = fact_pack.segments
    if not segments:
        return (
            SliderSignature.parse_obj(
                {
                    "visual": {"score": 0, "business_judgment": "无分镜输入，无法判断视觉是粗暴直给还是精致质感。"},
                    "audio": {"score": 0, "business_judgment": "无分镜输入，无法判断音频是生活经验还是克制专业。"},
                    "proof": {"score": 0, "business_judgment": "无分镜输入，无法判断举证更偏结果直给还是机理解释。"},
                    "cta": {"score": 0, "business_judgment": "无分镜输入，无法判断收口更偏强逼单还是顺势收单。"},
                }
            ).dict(),
            {
                "inference_mode": "empty_factpack",
                "summary": "FactPack 为空，返回零值四维光谱。",
                "dimension_reasons": {
                    "visual": "无分镜输入，无法判断视觉是粗暴直给还是精致质感。",
                    "audio": "无分镜输入，无法判断音频是生活经验还是克制专业。",
                    "proof": "无分镜输入，无法判断举证更偏结果直给还是机理解释。",
                    "cta": "无分镜输入，无法判断收口更偏强逼单还是顺势收单。",
                },
                "dimension_evidence_segment_ids": {"visual": [], "audio": [], "proof": [], "cta": []},
            },
        )

    if not _is_llm_enabled():
        return _neutral_slider_signature_meta(segments=segments, inference_mode="llm_unavailable_neutral_fallback")

    taxonomy_text = _load_taxonomy_dictionary_v2()
    segments_payload = [_serialize_factpack_segment(seg) for seg in segments]
    system = (
        "你是短视频四维光谱（Slider Signature）裁决器。\n"
        "你只负责判定 how-to-say 的表现强度，绝不能改写 HEC 主骨架，且只允许输出四个固定维度。\n"
        "四个维度固定为：visual（视觉表现强度）、audio（听觉/口播承载强度）、proof（证据支撑强度）、cta（号召/收口强度）。\n"
        "判分规则：\n"
        "1. 必须基于整条视频的主表达方式做定性裁决，禁止把 segment 命中次数、长短、关键词数量当作公式。\n"
        "2. score 必须是 [0,10] 的整数；0 表示几乎不存在，10 表示该维度极强地承担主说服任务。\n"
        "3. 每个维度都必须给出 `business_judgment` 和 evidence_segment_ids；evidence_segment_ids 只能引用输入里的 segment_id。\n"
        "4. visual 的 business_judgment 必须对照 PRD 5.2，明确判断更偏「粗暴直给」还是「精致质感」。\n"
        "5. audio 的 business_judgment 必须对照 PRD 5.2，明确判断更偏「生活经验」还是「克制专业」。\n"
        "6. proof 的 business_judgment 必须对照 PRD 5.2，明确判断更偏「看结果就信」还是「讲机理才信」。\n"
        "7. cta 的 business_judgment 必须对照 PRD 5.2，明确判断更偏「强逼单」还是「顺势收单」。\n"
        "8. secondary_effects 仅作为辅助语境，帮助你识别局部副 E；绝对禁止因为 secondary_effects 存在就反向抬高 proof.score。proof 仍必须围绕主说服骨架裁决。\n"
        "9. 如果主 E 为 E1、secondary_effects 含 E6，proof.business_judgment 仍应优先判断整片更偏『看结果就信』还是『讲机理才信』，不能把局部成分补充误写成整片主证据。\n"
        "10. 输出必须是严格 JSON 对象，字段固定为 visual、audio、proof、cta、summary。\n"
        "每个维度下必须是对象：{score,business_judgment,evidence_segment_ids}。\n"
        "\n[taxonomy_dictionary_v2]\n" + (taxonomy_text[:8000] if taxonomy_text else "(missing)")
    )
    user = {
        "task": "请根据 FactPack 和已裁决主 HEC，输出整条视频的四维光谱强度（10 分制整数）。",
        "primary_hec": primary_hec or {},
        "secondary_effects": secondary_effects or [],
        "fact_pack": {
            "video_meta": fact_pack.video_meta.dict(),
            "segments": segments_payload,
        },
    }

    try:
        parsed = _call_openai_chat_json(
            system=system,
            user=user,
            llm_tag="video_understanding_module_v2_slider_signature",
        )
        valid_segment_ids = {seg.segment_id for seg in segments}
        signature_raw: dict[str, dict[str, Any]] = {}
        dimension_reasons: dict[str, str] = {}
        dimension_evidence_segment_ids: dict[str, list[str]] = {}

        for dimension in ("visual", "audio", "proof", "cta"):
            item = parsed.get(dimension)
            if not isinstance(item, dict):
                raise ProtocolViolation(f"LLM Slider 输出缺少 {dimension} 对象")
            business_judgment = str(item.get("business_judgment") or item.get("reason") or "").strip()
            if not business_judgment:
                raise ProtocolViolation(f"LLM Slider 输出缺少 {dimension}.business_judgment")
            signature_raw[dimension] = {
                "score": _normalize_slider_score(item.get("score"), dimension=dimension),
                "business_judgment": business_judgment,
            }
            evidence_segment_ids = item.get("evidence_segment_ids")
            if not isinstance(evidence_segment_ids, list) or not evidence_segment_ids:
                raise ProtocolViolation(f"LLM Slider 输出缺少 {dimension}.evidence_segment_ids")
            if any((not isinstance(x, str)) or x not in valid_segment_ids for x in evidence_segment_ids):
                raise ProtocolViolation(
                    f"LLM Slider 输出 {dimension}.evidence_segment_ids 非法：{evidence_segment_ids}"
                )
            dimension_reasons[dimension] = business_judgment
            dimension_evidence_segment_ids[dimension] = evidence_segment_ids

        signature = SliderSignature.parse_obj(signature_raw).dict()
        return signature, {
            "inference_mode": "llm",
            "summary": str(parsed.get("summary") or "").strip() or "已完成 10 分制四维光谱裁决。",
            "dimension_reasons": dimension_reasons,
            "dimension_evidence_segment_ids": dimension_evidence_segment_ids,
        }
    except Exception as exc:
        return _neutral_slider_signature_meta(
            segments=segments,
            inference_mode="llm_invalid_neutral_fallback",
            fallback_reason=str(exc),
        )



def _infer_storyboard_persuasion_rule(
    seg: FactPackSegment,
    *,
    primary_hec: dict[str, str] | None = None,
    slider_signature: dict[str, Any] | None = None,
    secondary_effects: list[dict[str, Any]] | None = None,
) -> str:
    segment_text = _collect_segment_text(seg)
    secondary_effect_label = _get_segment_secondary_effect_label(seg, secondary_effects)
    action_names = [
        str(action.get("action_name") or "").strip()
        for action in seg.visual_facts.actions
        if str(action.get("action_name") or "").strip()
    ]
    actions_text = "、".join(action_names)
    if _C3_DIRECTIVE_PATTERN.search(segment_text) or "搜索" in segment_text:
        return "行动收口：引导用户搜索、点击或下单"
    if secondary_effect_label == "E6":
        return "辅助举证：补充成分机理支撑主效果"
    if secondary_effect_label == "E3":
        return "辅助举证：局部对比替代，强化替换动机"
    if secondary_effect_label == "E5":
        return "辅助举证：补充步骤说明降低理解门槛"
    if secondary_effect_label == "E7":
        return "辅助举证：补充产地工厂信息增加可信度"
    if secondary_effect_label == "E1":
        return "辅助举证：局部结果验证补强主卖点"
    if _MALICIOUS_OLD_SOLUTION_PATTERN.search(segment_text) or _MALICIOUS_COMPARISON_PATTERN.search(segment_text):
        return "旧方案对比：放大外部方案风险以制造替换动机"
    if _PROOF_SIGNAL_PATTERN.search(segment_text) or _E6_INGREDIENT_PATTERN.search(segment_text) or any(
        token in segment_text for token in ("标准", "原果汁", "配料", "参数", "证书", "检测", "99%", "结果", "实测")
    ):
        return "证据举证：用标准、成分或结果建立可信度"
    if actions_text and any(token in actions_text for token in ("展示", "揉搓", "倒", "喷", "抹", "涂", "开箱", "拆", "拿", "放", "掰")):
        return f"动作演示：通过{actions_text}把卖点画面化"
    if _H6_SCENE_PATTERN.search(segment_text):
        return "场景代入：把产品带入具体使用情境"
    if _COGNITIVE_CONFLICT_PATTERN.search(segment_text) or any(token in segment_text for token in ("其实", "误区", "别被")):
        return "认知破题：先打破旧认知再引出卖点"
    if seg.ocr_facts:
        return "信息补充：用花字同步强化当前表达重点"
    if _get_slider_dimension_score(slider_signature, "visual") >= 7 and seg.visual_facts.visual_subject:
        return "视觉承接：用主体画面维持卖点记忆"
    effect_label = (primary_hec or {}).get("effect_label") or "主卖点"
    return f"信息承接：围绕{effect_label}补充当前分镜卖点"



def _infer_storyboard_local_hec_tag(
    seg: FactPackSegment,
    *,
    primary_hec: dict[str, str] | None = None,
    slider_signature: dict[str, Any] | None = None,
    secondary_effects: list[dict[str, Any]] | None = None,
) -> str:
    segment_text = _collect_segment_text(seg)
    secondary_effect_label = _get_segment_secondary_effect_label(seg, secondary_effects)
    action_names = "、".join(
        str(action.get("action_name") or "") for action in seg.visual_facts.actions if action.get("action_name")
    )
    if _C3_DIRECTIVE_PATTERN.search(segment_text) or _PRICE_OR_BENEFIT_PATTERN.search(segment_text) or "搜索" in segment_text:
        return (primary_hec or {}).get("cta_label") or "C3"
    if secondary_effect_label:
        return secondary_effect_label
    if (
        _MALICIOUS_COMPARISON_PATTERN.search(segment_text)
        or _MALICIOUS_OLD_SOLUTION_PATTERN.search(segment_text)
        or any(token in segment_text for token in ("旧款", "以前", "替换", "竞品", "别家", "外面那种"))
    ):
        return "E3"
    if _E6_INGREDIENT_PATTERN.search(segment_text):
        return "E6"
    if _E2_STRESS_TEST_PATTERN.search(segment_text):
        return "E2"
    if (
        _E1_TEST_PATTERN.search(segment_text)
        or any(token in segment_text for token in ("演示", "结果", "吸收", "验证", "测试", "实测", "滴落"))
        or any(token in action_names for token in ("演示", "展示", "滴落", "测试", "验证", "揉搓"))
        or _get_slider_dimension_score(slider_signature, "visual") >= 7
    ):
        return "E1"
    if _H6_SCENE_PATTERN.search(segment_text):
        return "H6"
    if _PRICE_OR_BENEFIT_PATTERN.search(segment_text):
        return "H2"
    if _COGNITIVE_CONFLICT_PATTERN.search(segment_text):
        return "H5"
    for fallback in (
        (primary_hec or {}).get("effect_label"),
        (primary_hec or {}).get("hook_label"),
        (primary_hec or {}).get("cta_label"),
        "H1",
    ):
        normalized = str(fallback or "").strip()
        if re.fullmatch(r"(?:H[1-7]|E[0-7]|C[1-5])", normalized):
            return normalized
    return "H1"



def _build_storyboard_segments_rule(
    fact_pack: FactPack,
    *,
    primary_hec: dict[str, str] | None = None,
    slider_signature: dict[str, Any] | None = None,
    secondary_effects: list[dict[str, Any]] | None = None,
    inference_mode: str,
    fallback_reason: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    storyboard_units = _storyboard_units_from_fact_pack(fact_pack)
    storyboard_segments: list[dict[str, Any]] = []
    for unit in storyboard_units:
        fact_segment = _storyboard_unit_to_factpack_segment(unit)
        local_hec_tag = _infer_storyboard_local_hec_tag(
            fact_segment,
            primary_hec=primary_hec,
            slider_signature=slider_signature,
            secondary_effects=secondary_effects,
        )
        persuasion_function = _infer_storyboard_persuasion_rule(
            fact_segment,
            primary_hec=primary_hec,
            slider_signature=slider_signature,
            secondary_effects=secondary_effects,
        )
        audio_event_projection = _derive_audio_event_projection(
            fact_segment,
            local_hec_tag=local_hec_tag,
            persuasion_function=persuasion_function,
        )
        fourth_layer_assets = _derive_fourth_layer_assets(
            fact_segment,
            local_hec_tag=local_hec_tag,
            persuasion_function=persuasion_function,
            audio_event_projection=audio_event_projection,
        )
        storyboard_segments.append(
            StoryboardSegment.parse_obj(
                {
                    **unit,
                    "local_hec_tag": local_hec_tag,
                    "persuasion_function": persuasion_function,
                    "performance_emotion": _derive_performance_emotion(fact_segment, persuasion_function),
                    "audio_event_projection": audio_event_projection,
                    **fourth_layer_assets,
                }
            ).dict()
        )
    _assert_segment_level_storyboard_output(fact_pack, storyboard_segments)
    meta: dict[str, Any] = {
        "inference_mode": inference_mode,
        "summary": "Stage 4 未取得有效 LLM 结果，已根据 segment 级事实规则生成业务意图。",
        "segment_ids": [segment["segment_id"] for segment in storyboard_segments],
        "storyboard_source": "segments",
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    return storyboard_segments, meta



def _infer_storyboard_segments(
    fact_pack: FactPack,
    *,
    primary_hec: dict[str, str] | None = None,
    slider_signature: dict[str, Any] | None = None,
    secondary_effects: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    storyboard_units = _storyboard_units_from_fact_pack(fact_pack)
    if not storyboard_units:
        return [], {"inference_mode": "empty_factpack", "summary": "FactPack 为空，无分镜可注释。", "segment_ids": []}

    if not _is_llm_enabled():
        return _build_storyboard_segments_rule(
            fact_pack,
            primary_hec=primary_hec,
            slider_signature=slider_signature,
            secondary_effects=secondary_effects,
            inference_mode="llm_unavailable_rule_fallback",
        )

    segments_payload = [
        {
            **unit,
            "member_segment_ids": list(unit.get("member_segment_ids") or []),
            "aggregation_reason": list(unit.get("aggregation_reason") or []),
            "coverage_frame_refs": list(unit.get("coverage_frame_refs") or []),
            "blocked_boundary_ids": list(unit.get("blocked_boundary_ids") or []),
            "semantic_bundle_id": fact_pack.segment_to_bundle_map.get(unit["segment_id"], ""),
        }
        for unit in storyboard_units
    ]
    system = (
        "你是短视频局部分镜业务意图注释器。\n"
        "你要为每个物理 segment 生成 `local_hec_tag` 与 `persuasion_function`，描述这个 segment 在整条视频说服链路中的局部挂靠标签和业务作用。\n"
        "必须遵守：\n"
        "1. 只能基于输入的 visual/audio/actions/ocr/rhythm 事实，以及已给定的 primary_hec / slider_signature 推断，不得脑补未出现的信息。\n"
        "2. 每个 segment 都必须输出 1 个 `local_hec_tag`，取值只能是 H1-H7、E0-E7、C1-C5 之一。\n"
        "3. 每个 segment 都必须输出 1 条简洁中文 `persuasion_function`，长度建议 8-30 个字。\n"
        "4. 必须保持 segment 粒度，不得把多个物理 segments 聚合成一条输出。\n"
        "5. 禁止输出空泛占位语，例如“基于分镜事实生成的注释”“MVP 占位”等。\n"
        "6. 输出严格 JSON：{\"segments\":[{\"segment_id\":\"SEG01\",\"local_hec_tag\":\"E1\",\"persuasion_function\":\"...\"}],\"summary\":\"...\"}。\n"
        "7. segments 数量必须与输入完全一致，segment_id 必须一一对应，不得遗漏、不重排、不新增。\n"
        "8. 若某个 segment 命中 secondary_effects.evidence_segment_ids，则该 segment 的 local_hec_tag 优先挂靠对应副 E；persuasion_function 必须体现‘辅助举证’定位，禁止继续只围绕主 E 复读。\n"
    )
    user = {
        "task": "请根据 FactPack segment 级分镜事实、主 HEC 与四维光谱，逐 segment 生成业务意图注释。",
        "primary_hec": primary_hec or {},
        "slider_signature": slider_signature or {},
        "secondary_effects": secondary_effects or [],
        "fact_pack": {
            "video_meta": fact_pack.video_meta.dict(),
            "storyboard_source": "segments",
            "storyboard_segments": segments_payload,
        },
    }

    try:
        parsed = _call_openai_chat_json(
            system=system,
            user=user,
            llm_tag="video_understanding_module_v2_storyboard_segments",
        )
        items = parsed.get("segments")
        if not isinstance(items, list) or len(items) != len(storyboard_units):
            raise ProtocolViolation("LLM Stage4 输出 segments 数量非法")

        valid_segment_ids = {unit["segment_id"] for unit in storyboard_units}
        allowed_local_tags = {*(f"H{i}" for i in range(1, 8)), *(f"E{i}" for i in range(0, 8)), *(f"C{i}" for i in range(1, 6))}
        annotations_by_id: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ProtocolViolation("LLM Stage4 输出 segments 元素必须为对象")
            segment_id = str(item.get("segment_id") or "").strip()
            if not segment_id or segment_id not in valid_segment_ids:
                raise ProtocolViolation(f"LLM Stage4 输出 segment_id 非法：{segment_id!r}")
            if segment_id in annotations_by_id:
                raise ProtocolViolation(f"LLM Stage4 输出 segment_id 重复：{segment_id}")
            local_hec_tag = str(item.get("local_hec_tag") or "").strip()
            if local_hec_tag not in allowed_local_tags:
                raise ProtocolViolation(f"LLM Stage4 输出 {segment_id}.local_hec_tag 非法：{local_hec_tag!r}")
            persuasion_function = str(item.get("persuasion_function") or "").strip()
            if not persuasion_function:
                raise ProtocolViolation(f"LLM Stage4 输出 {segment_id}.persuasion_function 缺失")
            if "MVP 占位" in persuasion_function or "基于分镜事实生成的注释" in persuasion_function:
                raise ProtocolViolation(f"LLM Stage4 输出 {segment_id}.persuasion_function 仍为占位文案")
            annotations_by_id[segment_id] = {
                "local_hec_tag": local_hec_tag,
                "persuasion_function": persuasion_function,
            }

        if set(annotations_by_id.keys()) != valid_segment_ids:
            raise ProtocolViolation("LLM Stage4 输出 segment 覆盖不完整")

        storyboard_segments = []
        for unit in storyboard_units:
            annotation = annotations_by_id[unit["segment_id"]]
            fact_segment = _storyboard_unit_to_factpack_segment(unit)
            audio_event_projection = _derive_audio_event_projection(
                fact_segment,
                local_hec_tag=annotation["local_hec_tag"],
                persuasion_function=annotation["persuasion_function"],
            )
            fourth_layer_assets = _derive_fourth_layer_assets(
                fact_segment,
                local_hec_tag=annotation["local_hec_tag"],
                persuasion_function=annotation["persuasion_function"],
                audio_event_projection=audio_event_projection,
            )
            storyboard_segments.append(
                StoryboardSegment.parse_obj(
                    {
                        **unit,
                        **annotation,
                        "performance_emotion": _derive_performance_emotion(
                            fact_segment,
                            annotation["persuasion_function"],
                        ),
                        "audio_event_projection": audio_event_projection,
                        **fourth_layer_assets,
                    }
                ).dict()
            )
        _assert_segment_level_storyboard_output(fact_pack, storyboard_segments)
        return storyboard_segments, {
            "inference_mode": "llm",
            "summary": str(parsed.get("summary") or "").strip() or "已完成逐 segment 业务意图语义推导。",
            "segment_ids": [segment["segment_id"] for segment in storyboard_segments],
            "storyboard_source": "segments",
        }
    except Exception as exc:
        return _build_storyboard_segments_rule(
            fact_pack,
            primary_hec=primary_hec,
            slider_signature=slider_signature,
            secondary_effects=secondary_effects,
            inference_mode="llm_invalid_rule_fallback",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def _build_provenance_report(
    *,
    blueprint_id: str,
    source_product_id: str,
    primary_hec_source_refs: list[str],
    slider_source_refs: list[str],
    secondary_effects: list[dict[str, Any]] | None,
    generator_version: str,
) -> list[FieldProvenance]:
    now = _utc_now_rfc3339()
    provenance_items: list[FieldProvenance] = [
        FieldProvenance(
            field_path="blueprint.primary_hec",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*]",
            source_refs=primary_hec_source_refs,
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="blueprint.slider_signature",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*]",
            source_refs=slider_source_refs,
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="blueprint.storyboard_segments[*].persuasion_function",
            producer_type="system_native_inference",
            source_type="fact_pack.semantic_bundles[*] -> fact_pack.segments[*].visual_facts/audio_facts/ocr_facts/rhythm_facts",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="blueprint.storyboard_segments[*].audio_event_projection",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].audio_facts/rhythm_facts -> storyboard annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="blueprint.storyboard_segments[*].reusable_clip_notes",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].visual_facts/audio_facts/ocr_facts -> storyboard annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="blueprint.storyboard_segments[*].risk_bridge_notes",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].visual_facts/audio_facts/ocr_facts -> compliance annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.product_master_snapshot",
            producer_type="ssot_lookup",
            source_type="ssot.product_snapshot",
            source_refs=[f"source_product_id:{source_product_id}"],
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.video_blueprint_master.blueprint_id",
            producer_type="system_native_inference",
            source_type="blueprint.blueprint_id",
            source_refs=[blueprint_id],
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.video_segment_fact_table[*].annotation.audio_event_projection",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].audio_facts/rhythm_facts -> segment annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.video_segment_fact_table[*].annotation.reusable_clip_notes",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].visual_facts/audio_facts/ocr_facts -> segment annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.video_segment_fact_table[*].annotation.risk_bridge_notes",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*].visual_facts/audio_facts/ocr_facts -> compliance annotation",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
        FieldProvenance(
            field_path="triad_assets.video_segment_fact_table[*].annotation",
            producer_type="system_native_inference",
            source_type="fact_pack.segments[*]",
            source_refs=sorted(set(primary_hec_source_refs)),
            generated_at=now,
            generator_version=generator_version,
        ),
    ]
    for index, secondary_effect in enumerate(secondary_effects or []):
        source_refs = sorted(
            {
                str(segment_id).strip()
                for segment_id in list(secondary_effect.get("evidence_segment_ids") or [])
                if str(segment_id).strip()
            }
        )
        if not source_refs:
            continue
        provenance_items.append(
            FieldProvenance(
                field_path=f"blueprint.secondary_effects[{index}].effect_label",
                producer_type="system_native_inference",
                source_type="fact_pack.segments[*]",
                source_refs=source_refs,
                generated_at=now,
                generator_version=generator_version,
            )
        )
        provenance_items.append(
            FieldProvenance(
                field_path=f"blueprint.secondary_effects[{index}].evidence_segment_ids",
                producer_type="system_native_inference",
                source_type="fact_pack.segments[*]",
                source_refs=source_refs,
                generated_at=now,
                generator_version=generator_version,
            )
        )
    return provenance_items


def _assert_provenance_report_evidence_refs(
    provenance_report: list[FieldProvenance],
    *,
    valid_segment_ids: set[str],
) -> None:
    """PRD 6.5：evidence refs 指向不存在的 segment 必须判非法。"""

    invalid: list[tuple[str, str]] = []
    for item in provenance_report:
        if item.producer_type != "system_native_inference":
            continue
        if "fact_pack.segments" not in item.source_type:
            continue
        for ref in item.source_refs:
            if ref not in valid_segment_ids:
                invalid.append((item.field_path, ref))
    if invalid:
        raise ProtocolViolation(f"Gate2 失败：provenance_report 证据锚点失效：{invalid}")


def _normalize_field_path(field_path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", field_path)


def _assert_secondary_effects_payload(


    secondary_effects: Any,


    *,


    primary_effect_label: str | None,


    valid_segment_ids: set[str],


    error_prefix: str,


) -> None:


    if not isinstance(secondary_effects, list):


        raise ProtocolViolation(f"{error_prefix}：blueprint.secondary_effects 必须为数组")


    primary_effect = str(primary_effect_label or "").strip().upper()


    seen_effect_labels: set[str] = set()


    for index, item in enumerate(secondary_effects):


        item_prefix = f"{error_prefix}：blueprint.secondary_effects[{index}]"


        if not isinstance(item, dict):


            raise ProtocolViolation(f"{item_prefix} 必须为对象")


        effect_label = str(item.get("effect_label") or "").strip().upper()


        if effect_label not in SECONDARY_EFFECT_LABELS:


            raise ProtocolViolation(f"{item_prefix}.effect_label 非法，必须属于 E0-E7")


        if primary_effect and effect_label == primary_effect:


            raise ProtocolViolation(f"{item_prefix}.effect_label 不得与 blueprint.primary_hec.effect_label 重复")


        if effect_label in seen_effect_labels:


            raise ProtocolViolation(f"{item_prefix}.effect_label 不允许重复")


        seen_effect_labels.add(effect_label)





        evidence_segment_ids = item.get("evidence_segment_ids")


        if not isinstance(evidence_segment_ids, list) or not evidence_segment_ids:


            raise ProtocolViolation(f"{item_prefix}.evidence_segment_ids 必须为非空数组")


        normalized_segment_ids: list[str] = []


        for seg_index, segment_id in enumerate(evidence_segment_ids):


            normalized_segment_id = str(segment_id or "").strip()


            if not normalized_segment_id:


                raise ProtocolViolation(


                    f"{item_prefix}.evidence_segment_ids[{seg_index}] 必须为非空字符串"


                )


            normalized_segment_ids.append(normalized_segment_id)


        if len(set(normalized_segment_ids)) != len(normalized_segment_ids):


            raise ProtocolViolation(f"{item_prefix}.evidence_segment_ids 不允许重复 segment_id")


        invalid_segment_ids = sorted(set(normalized_segment_ids) - valid_segment_ids)


        if invalid_segment_ids:


            raise ProtocolViolation(f"{item_prefix}.evidence_segment_ids 存在无效 segment_id: {invalid_segment_ids}")


        reason = item.get("reason")


        if reason is not None and not str(reason).strip():


            raise ProtocolViolation(f"{item_prefix}.reason 若存在则不能为空字符串")











def _format_validation_error(exc: ValidationError) -> str:


    first_error = exc.errors()[0] if exc.errors() else {}


    loc = ".".join(str(item) for item in first_error.get("loc", ()))


    msg = str(first_error.get("msg") or str(exc))


    return f"{loc}: {msg}" if loc else msg











def _assert_blueprint_l1_contract(blueprint: dict[str, Any], fact_pack: FactPack) -> None:
    try:
        l1_envelope = BlueprintL1Envelope.parse_obj(
            {
                "blueprint_id": blueprint.get("blueprint_id"),
                "video_id": blueprint.get("video_id"),
                "source_product_id": blueprint.get("source_product_id"),
                "storyboard_source": blueprint.get("storyboard_source"),
                "semantic_bundles": blueprint.get("semantic_bundles"),
                "segment_to_bundle_map": blueprint.get("segment_to_bundle_map"),
                "bundle_to_segment_range": blueprint.get("bundle_to_segment_range"),
            }
        )
    except ValidationError as exc:
        raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：{_format_validation_error(exc)}") from exc

    storyboard_segments = blueprint.get("storyboard_segments")
    if not isinstance(storyboard_segments, list) or not storyboard_segments:
        raise ProtocolViolation("Blueprint 四层字段断言失败（L1）：blueprint.storyboard_segments 必须为非空数组")

    valid_segment_ids = [segment.segment_id for segment in fact_pack.segments]
    valid_segment_id_set = set(valid_segment_ids)

    bundle_ids: list[str] = []
    for index, bundle in enumerate(l1_envelope.semantic_bundles):
        if bundle.bundle_id in bundle_ids:
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：semantic_bundles.bundle_id 重复：{bundle.bundle_id}")
        bundle_ids.append(bundle.bundle_id)
        if len(set(bundle.segment_ids)) != len(bundle.segment_ids):
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：semantic_bundles[{index}].segment_ids 不允许重复"
            )
        invalid_segment_ids = sorted(set(bundle.segment_ids) - valid_segment_id_set)
        if invalid_segment_ids:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：semantic_bundles[{index}] 存在无效 segment_id: {invalid_segment_ids}"
            )

    bundle_id_set = set(bundle_ids)
    segment_to_bundle_keys = set(l1_envelope.segment_to_bundle_map)
    if segment_to_bundle_keys != valid_segment_id_set:
        missing_keys = sorted(valid_segment_id_set - segment_to_bundle_keys)
        extra_keys = sorted(segment_to_bundle_keys - valid_segment_id_set)
        raise ProtocolViolation(
            "Blueprint 四层字段断言失败（L1）：segment_to_bundle_map 必须与 fact_pack.segments 一一对应"
            f"；missing={missing_keys}；extra={extra_keys}"
        )
    for segment_id, bundle_id in l1_envelope.segment_to_bundle_map.items():
        if bundle_id not in bundle_id_set:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：segment_to_bundle_map.{segment_id} 指向未知 bundle_id={bundle_id}"
            )

    range_keys = set(l1_envelope.bundle_to_segment_range)
    if range_keys != bundle_id_set:
        missing_keys = sorted(bundle_id_set - range_keys)
        extra_keys = sorted(range_keys - bundle_id_set)
        raise ProtocolViolation(
            "Blueprint 四层字段断言失败（L1）：bundle_to_segment_range 必须与 semantic_bundles 一一对应"
            f"；missing={missing_keys}；extra={extra_keys}"
        )

    segment_index_map = {segment_id: index for index, segment_id in enumerate(valid_segment_ids)}
    for index, bundle in enumerate(l1_envelope.semantic_bundles):
        bundle_range = l1_envelope.bundle_to_segment_range[bundle.bundle_id]
        expected_start_segment_id = bundle.segment_ids[0]
        expected_end_segment_id = bundle.segment_ids[-1]
        expected_start_index = segment_index_map[expected_start_segment_id]
        expected_end_index = segment_index_map[expected_end_segment_id]
        if bundle_range.start_segment_id != expected_start_segment_id or bundle_range.end_segment_id != expected_end_segment_id:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：bundle_to_segment_range.{bundle.bundle_id} 的起止 segment 与 semantic_bundles[{index}] 不一致"
            )
        if bundle_range.start_segment_index != expected_start_index or bundle_range.end_segment_index != expected_end_index:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：bundle_to_segment_range.{bundle.bundle_id} 的起止下标与 semantic_bundles[{index}] 不一致"
            )
        expected_chain = valid_segment_ids[expected_start_index : expected_end_index + 1]
        if bundle.segment_ids != expected_chain:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：semantic_bundles[{index}].segment_ids 必须在全片时间轴上连续"
            )
        for segment_id in bundle.segment_ids:
            if l1_envelope.segment_to_bundle_map.get(segment_id) != bundle.bundle_id:
                raise ProtocolViolation(
                    f"Blueprint 四层字段断言失败（L1）：segment_to_bundle_map.{segment_id} 与 semantic_bundles[{index}] 回链不一致"
                )

    normalized_storyboard_segments: list[dict[str, Any]] = []
    for index, storyboard_segment in enumerate(storyboard_segments):
        if not isinstance(storyboard_segment, dict):
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：storyboard_segments[{index}] 必须为对象")
        try:
            fact_layer = StoryboardFactLayer.parse_obj(
                {
                    "segment_id": storyboard_segment.get("segment_id"),
                    "start_sec": storyboard_segment.get("start_sec"),
                    "end_sec": storyboard_segment.get("end_sec"),
                    "visual_facts": storyboard_segment.get("visual_facts"),
                    "audio_facts": storyboard_segment.get("audio_facts"),
                    "ocr_facts": storyboard_segment.get("ocr_facts", []),
                    "rhythm_facts": storyboard_segment.get("rhythm_facts"),
                    "member_segment_ids": storyboard_segment.get("member_segment_ids"),
                    "aggregation_reason": storyboard_segment.get("aggregation_reason"),
                    "coverage_frame_refs": storyboard_segment.get("coverage_frame_refs"),
                    "blocked_boundary_ids": storyboard_segment.get("blocked_boundary_ids", []),
                }
            )
        except ValidationError as exc:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：storyboard_segments[{index}] {_format_validation_error(exc)}"
            ) from exc
        normalized_storyboard_segments.append({**storyboard_segment, **fact_layer.dict()})

    expected_segments = list(fact_pack.segments)
    if len(normalized_storyboard_segments) != len(expected_segments):
        raise ProtocolViolation("Blueprint 四层字段断言失败（L1）：storyboard_segments 数量必须与 fact_pack.segments 完全一致")

    expected_ids = [segment.segment_id for segment in expected_segments]
    actual_ids = [segment["segment_id"] for segment in normalized_storyboard_segments]
    if actual_ids != expected_ids:
        raise ProtocolViolation(
            "Blueprint 四层字段断言失败（L1）：storyboard_segments.segment_id 顺序必须与 fact_pack.segments 严格一致"
        )

    seen_segment_ids: set[str] = set()
    for expected_segment, storyboard_segment in zip(expected_segments, normalized_storyboard_segments):
        segment_id = storyboard_segment["segment_id"]
        if segment_id in seen_segment_ids:
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：storyboard_segments.segment_id 重复：{segment_id}")
        seen_segment_ids.add(segment_id)

        member_segment_ids = list(storyboard_segment.get("member_segment_ids") or [])
        if member_segment_ids != [expected_segment.segment_id]:
            raise ProtocolViolation(
                f"Blueprint 四层字段断言失败（L1）：{segment_id} 必须严格一对一回落到单个物理 segment，禁止 bundle 聚合残留"
            )

        start_sec = float(storyboard_segment.get("start_sec"))
        end_sec = float(storyboard_segment.get("end_sec"))
        if abs(start_sec - float(expected_segment.start_sec)) > 1e-6 or abs(end_sec - float(expected_segment.end_sec)) > 1e-6:
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：{segment_id} 时间轴必须与输入 segment 严格对齐")

        coverage_frame_refs = list(storyboard_segment.get("coverage_frame_refs") or [])
        if not coverage_frame_refs:
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：{segment_id} 缺少 coverage_frame_refs")
        if any(not str(ref).startswith(expected_segment.segment_id) for ref in coverage_frame_refs):
            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L1）：{segment_id} coverage_frame_refs 必须只回链本 segment")



def _assert_blueprint_l2_contract(blueprint: dict[str, Any], *, valid_segment_ids: set[str]) -> None:











    try:











        l2_envelope = BlueprintL2Envelope.parse_obj(











            {











                "primary_hec": blueprint.get("primary_hec"),











                "secondary_effects": blueprint.get("secondary_effects", []),











                "slider_signature": blueprint.get("slider_signature"),











                "risk_flags": blueprint.get("risk_flags"),











            }











        )











    except ValidationError as exc:











        raise ProtocolViolation(f"Blueprint 四层字段断言失败（L2）：{_format_validation_error(exc)}") from exc























    _assert_secondary_effects_payload(











        [item.dict() for item in l2_envelope.secondary_effects],











        primary_effect_label=l2_envelope.primary_hec.effect_label,











        valid_segment_ids=valid_segment_ids,











        error_prefix="Blueprint 四层字段断言失败（L2）",











    )























    evidence_segment_ids = list(l2_envelope.risk_flags.hec_evidence_segment_ids)











    if len(set(evidence_segment_ids)) != len(evidence_segment_ids):











        raise ProtocolViolation("Blueprint 四层字段断言失败（L2）：risk_flags.hec_evidence_segment_ids 不允许重复")











    invalid_segment_ids = sorted(set(evidence_segment_ids) - valid_segment_ids)











    if invalid_segment_ids:











        raise ProtocolViolation(











            f"Blueprint 四层字段断言失败（L2）：risk_flags.hec_evidence_segment_ids 存在无效 segment_id: {invalid_segment_ids}"











        )











    if l2_envelope.risk_flags.secondary_effects_present != bool(l2_envelope.secondary_effects):











        raise ProtocolViolation(











            "Blueprint 四层字段断言失败（L2）：risk_flags.secondary_effects_present 必须与 secondary_effects 是否为空保持一致"











        )















































def _assert_blueprint_l3_contract(blueprint: dict[str, Any]) -> None:














    storyboard_segments = blueprint.get("storyboard_segments") or []


    for index, storyboard_segment in enumerate(storyboard_segments):


        if not isinstance(storyboard_segment, dict):


            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L3）：storyboard_segments[{index}] 必须为对象")


        try:


            StoryboardStyleLayer.parse_obj(


                {


                    "local_hec_tag": storyboard_segment.get("local_hec_tag"),


                    "persuasion_function": storyboard_segment.get("persuasion_function"),


                    "performance_emotion": storyboard_segment.get("performance_emotion"),


                }


            )


        except ValidationError as exc:


            raise ProtocolViolation(


                f"Blueprint 四层字段断言失败（L3）：storyboard_segments[{index}] {_format_validation_error(exc)}"


            ) from exc











def _assert_blueprint_l4_contract(blueprint: dict[str, Any]) -> None:


    storyboard_segments = blueprint.get("storyboard_segments") or []


    for index, storyboard_segment in enumerate(storyboard_segments):


        if not isinstance(storyboard_segment, dict):


            raise ProtocolViolation(f"Blueprint 四层字段断言失败（L4）：storyboard_segments[{index}] 必须为对象")


        try:


            StoryboardExecutionLayer.parse_obj(


                {


                    "audio_event_projection": storyboard_segment.get("audio_event_projection", {}),


                    "is_key_bridge": storyboard_segment.get("is_key_bridge"),


                    "reusable_clip_notes": storyboard_segment.get("reusable_clip_notes", []),


                    "risk_bridge_notes": storyboard_segment.get("risk_bridge_notes", []),


                }


            )


        except ValidationError as exc:


            raise ProtocolViolation(


                f"Blueprint 四层字段断言失败（L4）：storyboard_segments[{index}] {_format_validation_error(exc)}"


            ) from exc


        _assert_bridge_asset_contract(


            storyboard_segment,


            error_prefix=f"Blueprint 四层字段断言失败（L4）：storyboard_segments[{index}]",


        )











def _assert_blueprint_four_layer_contract(blueprint: dict[str, Any], *, fact_pack: FactPack) -> None:


    if not isinstance(blueprint, dict):


        raise ProtocolViolation("Blueprint 四层字段断言失败：blueprint 必须为对象")


    valid_segment_ids = {segment.segment_id for segment in fact_pack.segments}


    _assert_blueprint_l1_contract(blueprint, fact_pack)


    _assert_blueprint_l2_contract(blueprint, valid_segment_ids=valid_segment_ids)


    _assert_blueprint_l3_contract(blueprint)


    _assert_blueprint_l4_contract(blueprint)











def _assert_asset_package_storyboard_contract(


    blueprint: dict[str, Any],


    triad_assets: dict[str, Any],


    *,


    valid_segment_ids: set[str],


) -> None:



    storyboard_segments = blueprint.get("storyboard_segments")
    if not isinstance(storyboard_segments, list) or not storyboard_segments:
        raise ProtocolViolation("AssetIngest 失败：asset_package.blueprint.storyboard_segments 必须为非空数组")

    storyboard_by_id: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(storyboard_segments):
        if not isinstance(segment, dict):
            raise ProtocolViolation(f"AssetIngest 失败：blueprint.storyboard_segments[{index}] 必须为对象")
        segment_id = str(segment.get("segment_id") or "").strip()
        if not segment_id:
            raise ProtocolViolation(f"AssetIngest 失败：blueprint.storyboard_segments[{index}].segment_id 不能为空")
        if segment_id in storyboard_by_id:
            raise ProtocolViolation(f"AssetIngest 失败：blueprint.storyboard_segments.segment_id 重复：{segment_id}")
        storyboard_by_id[segment_id] = segment
        _assert_bridge_asset_contract(segment, error_prefix=f"AssetIngest 失败：blueprint.storyboard_segments[{index}]")

    missing_storyboard_segments = sorted(valid_segment_ids - set(storyboard_by_id))
    if missing_storyboard_segments:
        raise ProtocolViolation(
            "AssetIngest 失败：blueprint.storyboard_segments 未覆盖全部 triad_assets.video_segment_fact_table segment："
            f"{missing_storyboard_segments}"
        )
    extra_storyboard_segments = sorted(set(storyboard_by_id) - valid_segment_ids)
    if extra_storyboard_segments:
        raise ProtocolViolation(
            f"AssetIngest 失败：blueprint.storyboard_segments 存在无效 segment_id: {extra_storyboard_segments}"
        )

    for index, row in enumerate(triad_assets.get("video_segment_fact_table") or []):
        if not isinstance(row, dict):
            raise ProtocolViolation(f"AssetIngest 失败：triad_assets.video_segment_fact_table[{index}] 必须为对象")
        segment_id = str(row.get("segment_id") or "").strip()
        if not segment_id:
            raise ProtocolViolation(f"AssetIngest 失败：triad_assets.video_segment_fact_table[{index}].segment_id 不能为空")
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            raise ProtocolViolation(
                f"AssetIngest 失败：triad_assets.video_segment_fact_table[{index}].annotation 必须为对象"
            )
        _assert_bridge_asset_contract(
            annotation,
            error_prefix=f"AssetIngest 失败：triad_assets.video_segment_fact_table[{index}].annotation",
        )
        storyboard_segment = storyboard_by_id.get(segment_id)
        if storyboard_segment is None:
            raise ProtocolViolation(
                f"AssetIngest 失败：triad_assets.video_segment_fact_table[{index}].segment_id={segment_id} 未在 blueprint.storyboard_segments 中声明"
            )
        for field_name in (
            "audio_event_projection",
            "is_key_bridge",
            "reusable_clip_notes",
            "risk_bridge_notes",
        ):
            if annotation.get(field_name) != storyboard_segment.get(field_name):
                raise ProtocolViolation(
                    "AssetIngest 失败：triad_assets.video_segment_fact_table[*].annotation 必须与 "
                    "blueprint.storyboard_segments[*] 同构透传 P1-4/P1-5 字段"
                )


def _assert_required_asset_field_provenance(
    provenance_report: list[FieldProvenance],
    *,
    valid_segment_ids: set[str],
    secondary_effects: list[dict[str, Any]] | None = None,
) -> None:
    required_field_paths = {
        "blueprint.primary_hec",
        "blueprint.slider_signature",
        "blueprint.storyboard_segments[*].audio_event_projection",
        "blueprint.storyboard_segments[*].is_key_bridge",
        "blueprint.storyboard_segments[*].reusable_clip_notes",
        "blueprint.storyboard_segments[*].risk_bridge_notes",
        "triad_assets.video_segment_fact_table[*].annotation",
        "triad_assets.video_segment_fact_table[*].annotation.audio_event_projection",
        "triad_assets.video_segment_fact_table[*].annotation.is_key_bridge",
        "triad_assets.video_segment_fact_table[*].annotation.reusable_clip_notes",
        "triad_assets.video_segment_fact_table[*].annotation.risk_bridge_notes",
    }
    normalized_to_items: dict[str, list[FieldProvenance]] = {}
    for item in provenance_report:
        normalized = _normalize_field_path(item.field_path)
        normalized_to_items.setdefault(normalized, []).append(item)

    missing = sorted(path for path in required_field_paths if path not in normalized_to_items)
    if missing:
        raise ProtocolViolation(f"AssetIngest 失败：缺少关键字段 provenance 覆盖：{missing}")

    secondary_effects = secondary_effects or []
    for index, secondary_effect in enumerate(secondary_effects):
        effect_path = f"blueprint.secondary_effects[{index}].effect_label"
        evidence_path = f"blueprint.secondary_effects[{index}].evidence_segment_ids"
        normalized_effect_path = _normalize_field_path(effect_path)
        normalized_evidence_path = _normalize_field_path(evidence_path)
        if normalized_effect_path not in normalized_to_items:
            raise ProtocolViolation(f"AssetIngest 失败：缺少关键字段 provenance 覆盖：['{effect_path}']")
        if normalized_evidence_path not in normalized_to_items:
            raise ProtocolViolation(f"AssetIngest 失败：缺少关键字段 provenance 覆盖：['{evidence_path}']")

        declared_refs = {
            str(segment_id).strip() for segment_id in secondary_effect.get("evidence_segment_ids", []) if str(segment_id).strip()
        }
        for path_name, normalized_path in ((effect_path, normalized_effect_path), (evidence_path, normalized_evidence_path)):
            aggregated_refs: set[str] = set()
            for item in normalized_to_items[normalized_path]:
                if not item.source_refs:
                    raise ProtocolViolation(f"AssetIngest 失败：{path_name} provenance source_refs 不允许为空")
                aggregated_refs.update(item.source_refs)
            missing_refs = sorted(declared_refs - aggregated_refs)
            if missing_refs:
                raise ProtocolViolation(f"AssetIngest 失败：{path_name} provenance 未覆盖声明证据：{missing_refs}")

    for path in (
        "blueprint.primary_hec",
        "blueprint.slider_signature",
        "blueprint.storyboard_segments[*].audio_event_projection",
        "blueprint.storyboard_segments[*].is_key_bridge",
        "blueprint.storyboard_segments[*].reusable_clip_notes",
        "blueprint.storyboard_segments[*].risk_bridge_notes",
        "triad_assets.video_segment_fact_table[*].annotation",
        "triad_assets.video_segment_fact_table[*].annotation.audio_event_projection",
        "triad_assets.video_segment_fact_table[*].annotation.is_key_bridge",
        "triad_assets.video_segment_fact_table[*].annotation.reusable_clip_notes",
        "triad_assets.video_segment_fact_table[*].annotation.risk_bridge_notes",
    ):
        aggregated_refs: set[str] = set()
        for item in normalized_to_items[path]:
            if not item.source_refs:
                raise ProtocolViolation(f"AssetIngest 失败：{path} provenance source_refs 不允许为空")
            aggregated_refs.update(item.source_refs)
        missing_refs = sorted(valid_segment_ids - aggregated_refs)
        if missing_refs:
            raise ProtocolViolation(f"AssetIngest 失败：{path} provenance 未覆盖全部 segment：{missing_refs}")


def _build_asset_bus_manifest(
    *,
    bus_id: str,
    bus_role: Literal["producer", "consumer"],
    blueprint_id: str,
    source_product_id: str,
    video_id: str,
    segment_count: int,
    semantic_bundle_count: int,
    phase_4_annotation_count: int,
    triad_assets: dict[str, Any],
    required_field_paths: list[str],
) -> dict[str, Any]:
    channel_manifest = [
        {
            "channel_name": "product_master_snapshot",
            "record_count": 1 if triad_assets.get("product_master_snapshot") else 0,
            "primary_keys": [source_product_id],
        },
        {
            "channel_name": "video_blueprint_master",
            "record_count": 1 if triad_assets.get("video_blueprint_master") else 0,
            "primary_keys": [blueprint_id],
        },
        {
            "channel_name": "video_segment_fact_table",
            "record_count": len(triad_assets.get("video_segment_fact_table", [])),
            "primary_keys": [
                str(item.get("segment_id") or "")
                for item in triad_assets.get("video_segment_fact_table", [])
                if str(item.get("segment_id") or "").strip()
            ],
        },
    ]
    return {
        "bus_id": bus_id,
        "bus_role": bus_role,
        "cascade": "1:N:N",
        "channel_manifest": channel_manifest,
        "handoff_contract": {
            "video_id": video_id,
            "blueprint_id": blueprint_id,
            "source_product_id": source_product_id,
            "required_field_provenance": required_field_paths,
        },
        "lineage_snapshot": {
            "segment_count": segment_count,
            "semantic_bundle_count": semantic_bundle_count,
            "phase_4_annotation_count": phase_4_annotation_count,
        },
    }


def _build_asset_bus_import_summary(
    *,
    request_id: str,
    video_id: str,
    source_product_id: str,
    evidence_refs: list[str],
    triad_assets: dict[str, Any],
    validation_checks: list[dict[str, str]],
    required_field_paths: list[str],
) -> dict[str, Any]:
    accepted_channels = [
        {
            "channel_name": "product_master_snapshot",
            "record_count": 1,
            "primary_keys": [source_product_id],
        },
        {
            "channel_name": "asset_package.blueprint",
            "record_count": 1,
            "primary_keys": [str(triad_assets.get("video_blueprint_master", {}).get("blueprint_id") or "external_blueprint")],
        },
        {
            "channel_name": "video_segment_fact_table",
            "record_count": len(triad_assets.get("video_segment_fact_table", [])),
            "primary_keys": [
                str(item.get("segment_id") or "")
                for item in triad_assets.get("video_segment_fact_table", [])
                if str(item.get("segment_id") or "").strip()
            ],
        },
    ]
    return {
        "bus_id": f"ASSETBUS_IMPORT_{request_id}",
        "bus_role": "consumer",
        "import_mode": "asset_accumulation",
        "accepted_channels": accepted_channels,
        "validation_snapshot": {
            item["name"]: item["status"] for item in validation_checks
        },
        "handoff_contract": {
            "request_id": request_id,
            "video_id": video_id,
            "source_product_id": source_product_id,
            "evidence_refs": list(evidence_refs),
            "required_field_provenance": list(required_field_paths),
        },
    }


def _video_understanding_engine(
    request: VideoUnderstandingRequest,
    *,
    ssot: FileSSOTClient,
    generator_version: str = "vu_module_v2_mvp",
    product_override: ProductSnapshot | None = None,
) -> VideoUnderstandingResult:
    # Stage 0：输入仲裁与 QA（此处认为已经经过 Gate0）
    _assert_factpack_purity(request)
    _assert_director_ready_schema(request)
    _assert_provenance_input(request)

    # Crash Early：source_product_id 必填 + SSOT 必须可查
    if not request.source_product_id.strip():
        raise ProtocolViolation("source_product_id 缺失")
    product = product_override or ssot.get_product_snapshot(request.source_product_id)

    # Stage 1：事实归一化（MVP 直接沿用输入，不做补全/脑补）
    fact_pack = request.fact_pack

    # Stage 2：全局 HEC 骨架裁决
    primary_hec, secondary_effects, risk_flags = _infer_primary_hec_and_risks(fact_pack, product)
    primary_hec, risk_flags = _apply_hook_guardrails(primary_hec, risk_flags, fact_pack=fact_pack)
    primary_hec, risk_flags = _apply_gate4_script_level_review(primary_hec, risk_flags, fact_pack=fact_pack)

    # Stage 3：四维光谱
    slider_signature, slider_meta = _infer_slider_signature(
        fact_pack,
        primary_hec=primary_hec,
        secondary_effects=secondary_effects,
    )

    # Stage 4：局部分镜注释层
    segment_annotations, storyboard_meta = _infer_storyboard_segments(
        fact_pack,
        primary_hec=primary_hec,
        slider_signature=slider_signature,
        secondary_effects=secondary_effects,
    )
    annotation_by_segment_id = {
        segment["segment_id"]: {
            "local_hec_tag": segment["local_hec_tag"],
            "persuasion_function": segment["persuasion_function"],
            "performance_emotion": dict(segment.get("performance_emotion") or {}),
            "audio_event_projection": dict(segment.get("audio_event_projection") or {}),
            "is_key_bridge": bool(segment.get("is_key_bridge")),
            "reusable_clip_notes": list(segment.get("reusable_clip_notes") or []),
            "risk_bridge_notes": list(segment.get("risk_bridge_notes") or []),
            "member_segment_ids": list(segment.get("member_segment_ids") or []),
            "aggregation_reason": list(segment.get("aggregation_reason") or []),
            "coverage_frame_refs": list(segment.get("coverage_frame_refs") or []),
            "blocked_boundary_ids": list(segment.get("blocked_boundary_ids") or []),
        }
        for segment in segment_annotations
    }

    # Stage 5：资产沉淀（1:N:N）
    blueprint_id = build_blueprint_id(
        request_id=request.request_id,
        video_id=request.video_id,
        source_product_id=request.source_product_id,
    )
    blueprint = {
        "blueprint_id": blueprint_id,
        "video_id": request.video_id,
        "source_product_id": request.source_product_id,
        "primary_hec": primary_hec,
        "secondary_effects": secondary_effects,
        "slider_signature": slider_signature,
        "risk_flags": risk_flags,
        "storyboard_source": "segments",
        "semantic_bundles": [bundle.dict() for bundle in fact_pack.semantic_bundles],
        "segment_to_bundle_map": dict(fact_pack.segment_to_bundle_map),
        "bundle_to_segment_range": {
            bundle_id: bundle_range.dict() for bundle_id, bundle_range in fact_pack.bundle_to_segment_range.items()
        },
        "storyboard_segments": segment_annotations,
    }

    # triad_assets: 三表结构
    triad_assets = {
        "product_master_snapshot": {
            **product.dict(),
            "provenance": {
                "producer_type": "ssot_lookup",
                "source_refs": [f"source_product_id:{product.source_product_id}"],
            },
        },
        "video_blueprint_master": {
            "blueprint_id": blueprint_id,
            "video_id": request.video_id,
            "source_product_id": request.source_product_id,
            "primary_hec": primary_hec,
            "secondary_effects": secondary_effects,
            "slider_signature": slider_signature,
            "storyboard_source": "segments",
            "semantic_bundle_count": len(fact_pack.semantic_bundles),
            "provenance": [
                {
                    "field_path": "primary_hec",
                    "producer_type": "system_native_inference",
                    "source_refs": [seg.segment_id for seg in fact_pack.segments],
                },
                *[
                    {
                        "field_path": f"secondary_effects[{index}].effect_label",
                        "producer_type": "system_native_inference",
                        "source_refs": list(secondary_effect.get("evidence_segment_ids") or []),
                    }
                    for index, secondary_effect in enumerate(secondary_effects)
                ],
                *[
                    {
                        "field_path": f"secondary_effects[{index}].evidence_segment_ids",
                        "producer_type": "system_native_inference",
                        "source_refs": list(secondary_effect.get("evidence_segment_ids") or []),
                    }
                    for index, secondary_effect in enumerate(secondary_effects)
                ],
                {
                    "field_path": "source_product_id",
                    "producer_type": "ssot_lookup",
                    "source_refs": [f"source_product_id:{request.source_product_id}"],
                },
            ],
        },
        "video_segment_fact_table": [
            {
                "segment_record_id": build_segment_record_id(
                    blueprint_id=blueprint_id,
                    segment_id=seg.segment_id,
                ),
                "blueprint_id": blueprint_id,
                "segment_id": seg.segment_id,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "shot_size": seg.visual_facts.shot_size,
                "camera_movement": seg.visual_facts.camera_movement,
                "lighting_tone": seg.visual_facts.lighting_tone,
                "visual_subject": seg.visual_facts.visual_subject,
                "key_objects": seg.visual_facts.key_objects,
                "actions": seg.visual_facts.actions,
                "ocr_facts": [o.dict() for o in seg.ocr_facts],
                "audio_facts": seg.audio_facts.dict(),
                "rhythm_facts": seg.rhythm_facts.dict(),
                "bundle_id": fact_pack.segment_to_bundle_map.get(seg.segment_id, ""),
                "annotation": annotation_by_segment_id[seg.segment_id],
                "provenance": {
                    "producer_type": "system_native_inference",
                    "source_refs": [seg.segment_id],
                },
            }
            for seg in fact_pack.segments
        ],
    }

    phase_4_output = {
        "phase_id": "phase_4_segment_annotation",
        "status": "completed",
        "summary": {
            "segment_count": len(segment_annotations),
            "semantic_bundle_count": len(fact_pack.semantic_bundles),
            "storyboard_source": "segments",
            "annotation_coverage": 1.0 if segment_annotations else 0.0,
        },
        "annotation_records": segment_annotations,
        "discipline_checks": [
            {"name": "segment_annotation_complete", "status": "pass"},
            {"name": "no_bundle_aggregation_residue", "status": "pass"},
            {"name": "local_hec_and_persuasion_present", "status": "pass"},
            {"name": "audio_events_eventized", "status": "pass"},
            {"name": "fourth_layer_assets_present", "status": "pass"},
        ],
    }

    required_field_paths = [
        "blueprint.primary_hec",
        *[
            f"blueprint.secondary_effects[{index}].effect_label"
            for index, secondary_effect in enumerate(secondary_effects)
            if list(secondary_effect.get("evidence_segment_ids") or [])
        ],
        *[
            f"blueprint.secondary_effects[{index}].evidence_segment_ids"
            for index, secondary_effect in enumerate(secondary_effects)
            if list(secondary_effect.get("evidence_segment_ids") or [])
        ],
        "blueprint.slider_signature",
        "blueprint.storyboard_segments[*].audio_event_projection",
        "blueprint.storyboard_segments[*].reusable_clip_notes",
        "blueprint.storyboard_segments[*].risk_bridge_notes",
        "triad_assets.video_segment_fact_table[*].annotation",
        "triad_assets.video_segment_fact_table[*].annotation.audio_event_projection",
        "triad_assets.video_segment_fact_table[*].annotation.reusable_clip_notes",
        "triad_assets.video_segment_fact_table[*].annotation.risk_bridge_notes",
    ]
    asset_bus = _build_asset_bus_manifest(
        bus_id=f"ASSETBUS_{blueprint_id}",
        bus_role="producer",
        blueprint_id=blueprint_id,
        source_product_id=request.source_product_id,
        video_id=request.video_id,
        segment_count=len(fact_pack.segments),
        semantic_bundle_count=len(fact_pack.semantic_bundles),
        phase_4_annotation_count=len(segment_annotations),
        triad_assets=triad_assets,
        required_field_paths=required_field_paths,
    )

    phase_5_output = {
        "phase_id": "phase_5_asset_accumulation",
        "status": "completed",
        "summary": {
            "cascade": "1:N:N",
            "blueprint_id": blueprint_id,
            "segment_granular_record_count": len(triad_assets.get("video_segment_fact_table", [])),
            "phase_4_annotation_count": len(segment_annotations),
            "asset_bus_id": asset_bus["bus_id"],
        },
        "asset_tables": triad_assets,
        "asset_bus": asset_bus,
        "discipline_checks": [
            {"name": "phase_4_dependency_locked", "status": "pass"},
            {"name": "triad_asset_cascade_complete", "status": "pass"},
            {"name": "asset_bus_manifest_ready", "status": "pass"},
        ],
    }

    _assert_blueprint_four_layer_contract(blueprint, fact_pack=fact_pack)

    # workflow_report
    workflow_report = {
        "workflow_version": "video_understanding_dual_track_v2",
        "inference_mode": risk_flags.get("inference_mode", "unknown"),
        "request_id": request.request_id,
        "video_id": request.video_id,
        "blueprint_id": blueprint_id,
        "source_product_id": request.source_product_id,
        "gate_checks": [
            {"gate_id": "gate0", "status": "completed"},
            {"gate_id": "gate1", "status": "completed"},
            {"gate_id": "gate2", "status": "completed"},
        ],
        "stage_sequence": [
            {
                "stage_id": "module_1_structured_parsing",
                "status": "completed",
                "output": {
                    "segment_count": len(fact_pack.segments),
                    "semantic_bundle_count": len(fact_pack.semantic_bundles),
                },
            },
            {
                "stage_id": "module_2_global_hec_adjudication",
                "status": "completed",
                "output": {
                    "primary_hec": primary_hec,
                    "secondary_effects": secondary_effects,
                },
            },
            {
                "stage_id": "module_3_slider_extraction",
                "status": "completed",
                "output": {
                    "four_dimensional_signature": slider_signature,
                    **slider_meta,
                },
            },
            {
                "stage_id": "module_4_segment_annotation",
                "status": "completed",
                "output": phase_4_output,
            },
            {
                "stage_id": "module_5_asset_accumulation",
                "status": "completed",
                "output": phase_5_output,
            },
        ],
    }

    provenance_report = _build_provenance_report(
        blueprint_id=blueprint_id,
        source_product_id=request.source_product_id,
        primary_hec_source_refs=[seg.segment_id for seg in fact_pack.segments],
        slider_source_refs=[seg.segment_id for seg in fact_pack.segments],
        secondary_effects=secondary_effects,
        generator_version=generator_version,
    )

    _assert_provenance_report_evidence_refs(
        provenance_report,
        valid_segment_ids={seg.segment_id for seg in fact_pack.segments},
    )

    return VideoUnderstandingResult(
        blueprint=blueprint,
        workflow_report=workflow_report,
        phase_4_output=phase_4_output,
        phase_5_output=phase_5_output,
        triad_assets=triad_assets,
        provenance_report=provenance_report,
        video_coverage_gap_report={
            "checked_product_evidence": list(getattr(product, "core_selling_points", []) or []),
            "covered_by_video_evidence": [],
            "uncovered_product_evidence": list(getattr(product, "core_selling_points", []) or []),
            "blocking": False,
            "reason": "Route A 仅验收 ASR/OCR 中真实出现的 evidence 是否被如实记录进 FactPack；商品侧 evidence 未被视频覆盖只作为信息项透出，不作为阻断项。",
        },
    )


def _asset_ingest(
    request: AssetIngestRequest,
    *,
    ssot: FileSSOTClient,
    generator_version: str = "vu_module_v2_asset_accumulation_mvp",
) -> AssetIngestResult:
    _assert_provenance_input(request)

    # Crash Early：source_product_id 必填 + SSOT 必须可查（用于关系校验，不允许绕过）
    if not request.source_product_id.strip():
        raise ProtocolViolation("source_product_id 缺失")
    product = ssot.get_product_snapshot(request.source_product_id)

    asset_package = request.asset_package

    # Schema 校验（最小要求，见 PRD 5.2.1 / 8.2）
    if not isinstance(asset_package, dict):
        raise ProtocolViolation("AssetIngest 失败：asset_package 必须为对象")

    package_producer = str(asset_package.get("producer_type") or "").strip()
    if not package_producer:
        raise ProtocolViolation("AssetIngest 失败：asset_package.producer_type 缺失")
    if package_producer == "system_native_inference":
        raise ProtocolViolation("AssetIngest 失败：asset_package.producer_type 禁止为 system_native_inference")

    # request.provenance 表示“谁把资产包送进来/由哪条外部链路产出请求”，
    # asset_package.producer_type 表示“资产包本体的生产者”。二者允许不同（见 PRD 6.4）。
    producer_type = request.provenance.producer_type
    if producer_type == "system_native_inference":
        raise ProtocolViolation("AssetIngest 禁止 producer_type=system_native_inference")

    evidence_refs = asset_package.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs or not all(isinstance(x, str) and x.strip() for x in evidence_refs):
        raise ProtocolViolation("AssetIngest 失败：asset_package.evidence_refs 必须为非空字符串数组")

    required_structures = ("blueprint", "triad_assets", "provenance_report")
    missing_structures = [key for key in required_structures if key not in asset_package]
    if missing_structures:
        raise ProtocolViolation(f"AssetIngest 失败：asset_package 缺少核心结构：{missing_structures}")

    blueprint = asset_package.get("blueprint")
    if not isinstance(blueprint, dict):
        raise ProtocolViolation("AssetIngest 失败：asset_package.blueprint 必须为对象")
    missing_blueprint_fields = [
        field_name
        for field_name in ("primary_hec", "secondary_effects", "slider_signature", "storyboard_segments")
        if field_name not in blueprint
    ]
    if missing_blueprint_fields:
        raise ProtocolViolation(
            f"AssetIngest 失败：asset_package.blueprint 缺少必填字段：{missing_blueprint_fields}"
        )
    try:
        normalized_slider_signature = _normalize_slider_signature_payload(blueprint.get("slider_signature"))
    except (ValidationError, ValueError, TypeError, ProtocolViolation) as e:
        raise ProtocolViolation(f"AssetIngest 失败：blueprint.slider_signature 非法：{e}")
    blueprint["slider_signature"] = normalized_slider_signature

    triad_assets = asset_package.get("triad_assets")
    if not isinstance(triad_assets, dict):
        raise ProtocolViolation("AssetIngest 失败：asset_package.triad_assets 必须为对象")
    if "video_segment_fact_table" not in triad_assets or not isinstance(triad_assets["video_segment_fact_table"], list):
        raise ProtocolViolation("AssetIngest 失败：triad_assets.video_segment_fact_table 必须存在且为数组")

    valid_segment_ids = {
        str(item.get("segment_id"))
        for item in triad_assets["video_segment_fact_table"]
        if isinstance(item, dict) and str(item.get("segment_id") or "").strip()
    }
    if not valid_segment_ids:
        raise ProtocolViolation("AssetIngest 失败：triad_assets.video_segment_fact_table 缺少有效 segment_id")

    _assert_secondary_effects_payload(
        blueprint.get("secondary_effects"),
        primary_effect_label=(blueprint.get("primary_hec") or {}).get("effect_label"),
        valid_segment_ids=valid_segment_ids,
        error_prefix="AssetIngest 失败",
    )

    _assert_asset_package_storyboard_contract(
        blueprint,
        triad_assets,
        valid_segment_ids=valid_segment_ids,
    )

    field_level_provenance = asset_package.get("provenance_report")
    if not isinstance(field_level_provenance, list) or not field_level_provenance:
        raise ProtocolViolation("AssetIngest 失败：asset_package.provenance_report 必须为非空数组")

    parsed_field_level_provenance: list[FieldProvenance] = []

    # Provenance 校验：外部资产包内部若含 field_provenance，不得出现 system_native_inference
    nested_hits = []
    if isinstance(asset_package, dict):
        for path in _flatten_keys(asset_package):
            if path.endswith("producer_type"):
                # 仅在真正读值时再检查
                pass
        def _scan(obj: Any, cur: str = "") -> None:
            if isinstance(obj, dict):
                if obj.get("producer_type") == "system_native_inference":
                    nested_hits.append(cur or "<root>")
                for k, v in obj.items():
                    _scan(v, f"{cur}.{k}" if cur else str(k))
            elif isinstance(obj, list):
                for i, it in enumerate(obj):
                    _scan(it, f"{cur}[{i}]")
        _scan(asset_package)
    if nested_hits:
        raise ProtocolViolation(f"AssetIngest provenance 伪装 system_native_inference: {nested_hits}")

    invalid_evidence_refs: list[str] = []
    for item in field_level_provenance:
        if not isinstance(item, dict):
            raise ProtocolViolation("AssetIngest 失败：asset_package.provenance_report 元素必须为对象")
        try:
            parsed_item = FieldProvenance.parse_obj(item)
        except ValidationError as e:
            raise ProtocolViolation(f"AssetIngest 失败：field provenance 非法：{e}")
        refs = parsed_item.source_refs
        for ref in refs:
            if ref not in valid_segment_ids:
                invalid_evidence_refs.append(str(ref))
        parsed_field_level_provenance.append(parsed_item)
    if invalid_evidence_refs:
        raise ProtocolViolation(f"AssetIngest 失败：provenance_report 存在无效 evidence_refs: {sorted(set(invalid_evidence_refs))}")

    _assert_required_asset_field_provenance(
        parsed_field_level_provenance,
        valid_segment_ids=valid_segment_ids,
        secondary_effects=list(blueprint.get("secondary_effects") or []),
    )

    required_field_paths = [
        "blueprint.primary_hec",
        *[
            f"blueprint.secondary_effects[{index}].effect_label"
            for index, secondary_effect in enumerate(list(blueprint.get("secondary_effects") or []))
            if list(secondary_effect.get("evidence_segment_ids") or [])
        ],
        *[
            f"blueprint.secondary_effects[{index}].evidence_segment_ids"
            for index, secondary_effect in enumerate(list(blueprint.get("secondary_effects") or []))
            if list(secondary_effect.get("evidence_segment_ids") or [])
        ],
        "blueprint.slider_signature",
        "blueprint.storyboard_segments[*].audio_event_projection",
        "blueprint.storyboard_segments[*].is_key_bridge",
        "blueprint.storyboard_segments[*].reusable_clip_notes",
        "blueprint.storyboard_segments[*].risk_bridge_notes",
        "triad_assets.video_segment_fact_table[*].annotation",
        "triad_assets.video_segment_fact_table[*].annotation.audio_event_projection",
        "triad_assets.video_segment_fact_table[*].annotation.is_key_bridge",
        "triad_assets.video_segment_fact_table[*].annotation.reusable_clip_notes",
        "triad_assets.video_segment_fact_table[*].annotation.risk_bridge_notes",
    ]

    validation_checks = [
        {"name": "schema", "status": "pass"},
        {"name": "storyboard_annotation_contract", "status": "pass"},
        {"name": "provenance", "status": "pass"},
        {"name": "required_field_level_provenance", "status": "pass"},
        {"name": "full_segment_coverage", "status": "pass"},
        {"name": "ssot_lookup", "status": "pass"},
        {"name": "import_mode_mark", "status": "pass"},
        {"name": "asset_bus_consumer_ready", "status": "pass"},
    ]
    asset_bus = _build_asset_bus_import_summary(
        request_id=request.request_id,
        video_id=request.video_id,
        source_product_id=request.source_product_id,
        evidence_refs=[str(item) for item in evidence_refs],
        triad_assets=triad_assets,
        validation_checks=validation_checks,
        required_field_paths=required_field_paths,
    )

    ingested_assets = {
        "product_master_snapshot": {
            **product.dict(),
            "provenance": {
                "producer_type": "ssot_lookup",
                "source_refs": [f"source_product_id:{product.source_product_id}"],
            },
        },
        "asset_package": asset_package,
        "asset_bus": asset_bus,
        "import_mode": "asset_accumulation",
        "request_producer_type": producer_type,
        "asset_package_producer_type": package_producer,
    }

    validation_report = {
        "workflow_version": "asset_accumulation_prd_v2_mvp",
        "request_id": request.request_id,
        "video_id": request.video_id,
        "source_product_id": request.source_product_id,
        "asset_bus_summary": {
            "bus_id": asset_bus["bus_id"],
            "accepted_channel_count": len(asset_bus["accepted_channels"]),
        },
        "checks": validation_checks,
    }

    provenance_report = [
        FieldProvenance(
            field_path="import_mode",
            producer_type="external_pipeline",
            source_type="asset_accumulation",
            source_refs=["asset_package"],
            generated_at=_utc_now_rfc3339(),
            generator_version=generator_version,
        )
    ]

    return AssetIngestResult(
        ingested_assets=ingested_assets,
        validation_report=validation_report,
        provenance_report=provenance_report,
    )


def handle_request(
    raw_payload: dict[str, Any],
    *,
    ssot_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> VideoUnderstandingResult | AssetIngestResult:
    """路由层：先 Gate0 仲裁，再进入对应轨道。

    - payload_kind=fact_pack -> VideoUnderstandingEngine
    - payload_kind=asset_package -> AssetIngest
    """
    payload_kind = _assert_gate0_arbitration(raw_payload)
    ssot = FileSSOTClient(Path(ssot_path) if ssot_path is not None else Path(__file__).parent / "data" / "ssot.json")

    try:
        if payload_kind == "fact_pack":
            req = VideoUnderstandingRequest.parse_obj(raw_payload)
            triad_repo = _resolve_triad_asset_repository(
                options=req.options,
                ssot_path=ssot_path,
                db_path=db_path,
                request_id=req.request_id,
            )
            if req.item_name:
                provisional_snapshot = _build_provisional_product_snapshot(req)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    route_a_future = executor.submit(
                        _video_understanding_engine,
                        req,
                        ssot=ssot,
                        product_override=provisional_snapshot,
                    )
                    route_b_future = executor.submit(
                        _build_product_snapshot_from_caller_input,
                        req,
                        ssot=ssot,
                        triad_repo=triad_repo,
                    )
                    route_a_result = route_a_future.result()
                    route_b_result = route_b_future.result()
                    route_b_snapshot = route_b_result["snapshot"]
                    route_b_diagnosis = route_b_result["diagnosis"]
                result = _patch_video_understanding_result_with_product_snapshot(
                    route_a_result,
                    route_b_snapshot,
                    route_b_diagnosis,
                )
            else:
                result = _video_understanding_engine(req, ssot=ssot)
            if triad_repo is not None:
                summary = _build_triad_asset_persistence_summary(
                    triad_repo=triad_repo,
                    request_id=req.request_id,
                    video_id=req.video_id,
                    source_product_id=req.source_product_id,
                    workflow_version=str(result.workflow_report.get("workflow_version") or "video_understanding_dual_track_v2"),
                    generator_version=(
                        result.provenance_report[0].generator_version if result.provenance_report else "vu_module_v2_mvp"
                    ),
                    triad_assets=result.triad_assets,
                )
                result = _attach_db_persistence_to_video_result(result, summary=summary, triad_repo=triad_repo)
            return result
        req = AssetIngestRequest.parse_obj(raw_payload)
        triad_repo = _resolve_triad_asset_repository(
            options=req.options,
            ssot_path=ssot_path,
            db_path=db_path,
            request_id=req.request_id,
        )
        result = _asset_ingest(req, ssot=ssot)
        if triad_repo is not None:
            asset_package = dict(result.ingested_assets.get("asset_package") or {})
            triad_assets = dict(asset_package.get("triad_assets") or {})
            summary = _build_triad_asset_persistence_summary(
                triad_repo=triad_repo,
                request_id=req.request_id,
                video_id=req.video_id,
                source_product_id=req.source_product_id,
                workflow_version=str(result.validation_report.get("workflow_version") or "asset_accumulation_prd_v2_mvp"),
                generator_version=(
                    result.provenance_report[0].generator_version
                    if result.provenance_report
                    else "vu_module_v2_asset_accumulation_mvp"
                ),
                triad_assets=triad_assets,
            )
            result = _attach_db_persistence_to_asset_ingest_result(result, summary=summary, triad_repo=triad_repo)
        return result
    except ValidationError as e:
        # 统一转为 Crash Early 的协议错误，避免上层误判为“可兼容的 schema 问题”。
        raise ProtocolViolation(f"输入协议 Schema 校验失败（禁止向下兼容）：{e}")
    except TriadAssetPersistenceError as e:
        raise ProtocolViolation(f"TriadAssets 物理落库失败：{e}") from e


def _coerce_engine_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "dict") and callable(payload.dict):
        return payload.dict()
    if hasattr(payload, "model_dump") and callable(payload.model_dump):
        return payload.model_dump()
    raise ProtocolViolation("VideoUnderstandingEngine 仅接受 dual-track dict payload（fact_pack / asset_package）")


class VideoUnderstandingEngine:
    """统一对外入口：仅接受 dual-track payload。"""

    def __init__(self, ssot_path: str | Path | None = None, db_path: str | Path | None = None) -> None:
        self.ssot_path = Path(ssot_path) if ssot_path is not None else None
        self.db_path = Path(db_path) if db_path is not None else None

    def run_pipeline(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        raw_payload = _coerce_engine_payload(payload)
        result = handle_request(raw_payload, ssot_path=self.ssot_path, db_path=self.db_path)
        if isinstance(result, VideoUnderstandingResult):
            return {
                "blueprint": result.blueprint,
                "fact_pack": raw_payload.get("fact_pack", {}),
                "segment_tagging_output": {
                    "video_id": raw_payload.get("video_id", ""),
                    "blueprint_id": str(result.workflow_report.get("blueprint_id") or result.blueprint.get("blueprint_id") or ""),
                    "segment_records": list(result.phase_4_output.get("annotation_records") or []),
                    "schema_version": "v2_dual_track",
                },
                "triad_assets": result.triad_assets,
                "phase_4_output": result.phase_4_output,
                "phase_5_output": result.phase_5_output,
                "workflow_report": result.workflow_report,
                "provenance_report": [item.dict() for item in result.provenance_report],
            }
        return {
            "ingested_assets": result.ingested_assets,
            "validation_report": result.validation_report,
            "provenance_report": [item.dict() for item in result.provenance_report],
            "import_mode": result.import_mode,
        }

    def understand(self, payload: dict[str, Any] | Any) -> Any:
        pipeline = self.run_pipeline(payload)
        return pipeline.get("blueprint") or pipeline.get("ingested_assets")

    def to_segment_tagging_output(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        return self.run_pipeline(payload).get("segment_tagging_output", {})

    def to_asset_triad_output(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        pipeline = self.run_pipeline(payload)
        return pipeline.get("triad_assets") or pipeline.get("ingested_assets", {})

    def to_phase4_output(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        return self.run_pipeline(payload).get("phase_4_output", {})

    def to_phase5_output(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        pipeline = self.run_pipeline(payload)
        return pipeline.get("phase_5_output") or {
            "phase_id": "phase_5_asset_accumulation",
            "status": "completed",
            "asset_bus": pipeline.get("ingested_assets", {}).get("asset_bus", {}),
            "validation_report": pipeline.get("validation_report", {}),
        }

    def to_workflow_report(self, payload: dict[str, Any] | Any) -> dict[str, Any]:
        pipeline = self.run_pipeline(payload)
        return pipeline.get("workflow_report") or pipeline.get("validation_report", {})
