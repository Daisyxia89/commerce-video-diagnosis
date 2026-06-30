from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, root_validator, validator


JSONDict = dict[str, Any]
SegmentRole = Literal["hook", "effect", "cta", "bridge", "transition", "mixed"]


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        validate_assignment = True


LEGACY_HEC_KEYS = ("hook", "effect", "cta")
NEW_HEC_KEYS = ("hook_tag", "effect_tag", "cta_tag")


def _raise_if_legacy_hec_keys(payload: Any, *, field_name: str) -> None:
    if not isinstance(payload, dict):
        return
    legacy_keys = [key for key in LEGACY_HEC_KEYS if key in payload]
    if legacy_keys:
        raise ValueError(f"{field_name} 检测到旧版 HEC 键残留: {legacy_keys}")


class RawTextRange(StrictBaseModel):
    start_sec: float
    end_sec: float
    text: str

    @validator("text")
    def _validate_text(cls, value: str) -> str:
        if not value:
            raise ValueError("text 不能为空")
        return value


class RawKeyframeRef(StrictBaseModel):
    timestamp_sec: float
    frame_description: str
    image_path: str | None = None

    @validator("frame_description")
    def _validate_frame_description(cls, value: str) -> str:
        if not value:
            raise ValueError("frame_description 不能为空")
        return value


class RawStoryboardTag(StrictBaseModel):
    primary_label: str
    hook_label: str | None = None
    effect_label: str | None = None
    cta_label: str | None = None
    supporting_labels: list[str] = Field(default_factory=list)

    @validator("primary_label")
    def _validate_primary_label(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_label 不能为空")
        return value


class RawTaxonomy(StrictBaseModel):
    hook_label: str | None = None
    effect_label: str | None = None
    cta_label: str | None = None
    supporting_labels: list[str] = Field(default_factory=list)


class RawSliderEvidence(StrictBaseModel):
    score: int
    evidence: str

    @validator("score")
    def _validate_score(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("score 必须在 0-100 之间")
        return value

    @validator("evidence")
    def _validate_evidence(cls, value: str) -> str:
        if not value:
            raise ValueError("evidence 不能为空")
        return value


class RawModule5Sliders(StrictBaseModel):
    visual_slider: RawSliderEvidence
    audio_slider: RawSliderEvidence
    proof_slider: RawSliderEvidence
    cta_slider: RawSliderEvidence


class RawVisualGuidance(StrictBaseModel):
    shot_size: str
    camera_movement: str
    visual_core: str
    lighting_tone: str

    @validator("shot_size", "camera_movement", "visual_core", "lighting_tone")
    def _validate_non_empty(cls, value: str, field):
        if not value:
            raise ValueError(f"{field.name} 不能为空")
        return value


class RawTimedAudioEvent(StrictBaseModel):
    start_sec: float
    end_sec: float

    @validator("start_sec", "end_sec")
    def _validate_non_negative_time(cls, value: float, field):
        if value < 0:
            raise ValueError(f"{field.name} 不能为负数")
        return value


class RawSFXEvent(RawTimedAudioEvent):
    event_name: str

    @validator("event_name")
    def _validate_event_name(cls, value: str) -> str:
        if not value:
            raise ValueError("event_name 不能为空")
        return value


class RawBGMEvent(RawTimedAudioEvent):
    tone: str

    @validator("tone")
    def _validate_tone(cls, value: str) -> str:
        if not value:
            raise ValueError("tone 不能为空")
        return value


class RawAuditoryText(StrictBaseModel):
    asr_text: str
    ocr_text: str
    audio_effects: str
    ocr_color: str
    ocr_position: str
    font_family: str
    font_weight: str
    font_size_level: str
    stroke_style: str
    text_effect_style: str

    @validator(
        "asr_text",
        "ocr_text",
        "audio_effects",
        "ocr_color",
        "ocr_position",
        "font_family",
        "font_weight",
        "font_size_level",
        "stroke_style",
        "text_effect_style",
    )
    def _validate_non_empty(cls, value: str, field):
        if not value:
            raise ValueError(f"{field.name} 不能为空")
        return value


class RawPerformanceEmotion(StrictBaseModel):
    acting_instructions: str
    emotion_tension: str
    emotional_tone: str
    action_mechanics: str
    action_intensity: str

    @validator("acting_instructions", "emotion_tension", "emotional_tone", "action_mechanics", "action_intensity")
    def _validate_non_empty(cls, value: str, field):
        if not value:
            raise ValueError(f"{field.name} 不能为空")
        return value


class RawStoryboardSegment(StrictBaseModel):
    shot_id: str
    duration: float
    role: SegmentRole
    tag: RawStoryboardTag
    visual_description: str
    spoken_lines: str
    keyframe_image: str
    segment_id: str
    start_sec: float
    end_sec: float
    segment_role: SegmentRole
    shot_type: str | None = None
    editing_pattern: str | None = None
    keyframe_refs: list[RawKeyframeRef]
    asr_excerpt: list[RawTextRange]
    ocr_excerpt: list[RawTextRange]
    taxonomy: RawTaxonomy
    persuasion_function: str
    module5_sliders: RawModule5Sliders
    confidence: float
    needs_human_review: bool
    visual_guidance: RawVisualGuidance
    auditory_text: RawAuditoryText
    performance_emotion: RawPerformanceEmotion
    member_segment_ids: list[str] = Field(default_factory=list)
    aggregation_reason: list[str] = Field(default_factory=list)
    coverage_frame_refs: list[str] = Field(default_factory=list)
    blocked_boundary_ids: list[str] = Field(default_factory=list)

    @validator(
        "shot_id",
        "visual_description",
        "spoken_lines",
        "keyframe_image",
        "segment_id",
        "persuasion_function",
    )
    def _validate_required_strings(cls, value: str, field):
        if not value:
            raise ValueError(f"{field.name} 不能为空")
        return value


class RawSemanticBundle(StrictBaseModel):
    bundle_id: str
    start_sec: float
    end_sec: float
    segment_ids: list[str]
    bundle_role: str
    aggregation_reason: list[str]
    blocked_boundary_ids: list[str] = Field(default_factory=list)
    coverage_frame_refs: list[str] = Field(default_factory=list)


class RawBundleSegmentRange(StrictBaseModel):
    start_segment_index: int
    end_segment_index: int
    start_segment_id: str
    end_segment_id: str


class RawBlueprintModel(StrictBaseModel):
    blueprint_id: str
    source_video: str | None = None
    source_product_id: str | None = None
    original_product_name: str | None = None
    original_jtbd: str | None = None
    category_strategy_intent: str | None = None
    product_strategy_intent: str | None = None
    storyboard_segments: list[RawStoryboardSegment]
    semantic_bundles: list[RawSemanticBundle] = Field(default_factory=list)
    segment_to_bundle_map: dict[str, str] = Field(default_factory=dict)
    bundle_to_segment_range: dict[str, RawBundleSegmentRange] = Field(default_factory=dict)
    storyboard_source: str | None = None

    class Config(StrictBaseModel.Config):
        extra = "allow"

    @root_validator(pre=True)
    def _reject_legacy_hec_keys(cls, values: dict[str, Any]) -> dict[str, Any]:
        _raise_if_legacy_hec_keys(values.get("primary_hec"), field_name="primary_hec")
        _raise_if_legacy_hec_keys(values.get("taxonomy_result"), field_name="taxonomy_result")
        return values

    @validator("blueprint_id")
    def _validate_blueprint_id(cls, value: str) -> str:
        if not value:
            raise ValueError("blueprint_id 不能为空")
        return value

    @validator("storyboard_segments")
    def _validate_segments(cls, value: list[RawStoryboardSegment]) -> list[RawStoryboardSegment]:
        if not value:
            raise ValueError("storyboard_segments 不能为空")
        return value


@dataclass(slots=True)
class SliderEvidence:
    score: int
    evidence: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SliderSignature:
    visual: SliderEvidence
    audio: SliderEvidence
    proof: SliderEvidence
    cta: SliderEvidence

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class VisualGuidance:
    shot_size: str
    camera_movement: str
    visual_core: str
    lighting_tone: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class TimedAudioEvent:
    start_sec: float
    end_sec: float

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SFXEvent(TimedAudioEvent):
    event_name: str


@dataclass(slots=True)
class BGMEvent(TimedAudioEvent):
    tone: str


@dataclass(slots=True)
class AuditoryText:
    asr_text: str
    ocr_text: str
    audio_effects: str
    ocr_color: str
    ocr_position: str
    font_family: str
    font_weight: str
    font_size_level: str
    stroke_style: str
    text_effect_style: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class PerformanceEmotion:
    acting_instructions: str
    emotion_tension: str
    emotional_tone: str
    action_mechanics: str
    action_intensity: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class StoryboardSegment:
    shot_id: str
    duration: float
    role: SegmentRole
    tag: JSONDict
    visual_description: str
    spoken_lines: str
    keyframe_image: str
    segment_id: str
    start_sec: float
    end_sec: float
    segment_role: SegmentRole
    shot_type: str | None = None
    editing_pattern: str | None = None
    keyframe_refs: list[JSONDict] = field(default_factory=list)
    asr_excerpt: list[JSONDict] = field(default_factory=list)
    ocr_excerpt: list[JSONDict] = field(default_factory=list)
    taxonomy: JSONDict = field(default_factory=dict)
    persuasion_function: str = ""
    module5_sliders: JSONDict = field(default_factory=dict)
    confidence: float = 0.0
    needs_human_review: bool = False
    visual_guidance: VisualGuidance = field(default_factory=lambda: VisualGuidance("", "", "", ""))
    auditory_text: AuditoryText = field(
        default_factory=lambda: AuditoryText("", "", "", "", "", "", "", "", "", "")
    )
    performance_emotion: PerformanceEmotion = field(default_factory=lambda: PerformanceEmotion("", "", "", "", ""))

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SegmentTagRecord:
    segment_id: str
    video_id: str
    blueprint_id: str
    source_video: str
    start_sec: float
    end_sec: float
    segment_role: SegmentRole
    primary_label: str
    hook_label: str | None = None
    effect_label: str | None = None
    cta_label: str | None = None
    visual_slider: int = 0
    audio_slider: int = 0
    proof_slider: int = 0
    cta_slider: int = 0
    confidence: float = 0.0
    needs_human_review: bool = False
    persuasion_function: str = ""
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.1"

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class VideoFactPack:
    video_id: str
    source_video: str
    duration_sec: float
    keyframe_records: list[JSONDict] = field(default_factory=list)
    asr_records: list[JSONDict] = field(default_factory=list)
    ocr_records: list[JSONDict] = field(default_factory=list)
    rhythm_markers: list[JSONDict] = field(default_factory=list)
    segment_fact_records: list[JSONDict] = field(default_factory=list)
    semantic_bundles: list[JSONDict] = field(default_factory=list)
    segment_to_bundle_map: JSONDict = field(default_factory=dict)
    bundle_to_segment_range: JSONDict = field(default_factory=dict)
    storyboard_source: str = "semantic_bundles"
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.1"

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class HookSoftConstraintContract:
    trigger_cta_tags: list[str] = field(default_factory=list)
    required_effect_capabilities_all: list[str] = field(default_factory=list)
    unmet_risk_flag: str = ""

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class CTAResolution:
    requested_cta_tag: str = ""
    resolved_cta_tag: str = ""
    resolution_type: str = "direct"
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SoftConstraintResult:
    rule_id: str
    status: str
    required_capabilities: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    risk_flag: str | None = None

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class CandidateSet:
    h_list: list[JSONDict]
    effect_list: list[JSONDict]
    cta_list: list[JSONDict]
    schema_version: str = "v0.5"
    jtbd: str = ""
    persuasion_route: str = ""
    r_rule: str = ""
    p_rule: str = ""
    task_domain: str = "functional"

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class ProductECSkeleton:
    effect_tag: str
    cta_tag: str
    schema_version: str = "v0.5"
    effect_label: str = ""
    cta_label: str = ""
    effect_capabilities_snapshot: list[str] = field(default_factory=list)
    cta_resolution: CTAResolution | JSONDict = field(default_factory=CTAResolution)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class ProductHEC:
    hook_tag: str
    effect_tag: str
    cta_tag: str
    variant_id: str = ""
    schema_version: str = "v0.5"
    hook_label: str = ""
    effect_label: str = ""
    cta_label: str = ""
    activation_tags: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    soft_constraint_results: list[SoftConstraintResult | JSONDict] = field(default_factory=list)
    route_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class ProductDiagnosis:
    """商品诊断统一协议。"""

    product_id: str
    product_name: str
    category: str
    jtbd: str
    resistance_profile: JSONDict
    core_intent: JSONDict
    candidate_set: CandidateSet | JSONDict = field(
        default_factory=lambda: CandidateSet(h_list=[], effect_list=[], cta_list=[], schema_version="v0.5")
    )
    product_ec_skeletons: list[ProductECSkeleton | JSONDict] = field(default_factory=list)
    product_hecs: list[ProductHEC | JSONDict] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    evidence: JSONDict = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.5"

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class VideoBlueprint:
    blueprint_id: str
    source_video: str
    primary_hec: JSONDict
    secondary_effects: list[JSONDict] = field(default_factory=list)
    storyboard_segments: list[StoryboardSegment] = field(default_factory=list)
    slider_signature: SliderSignature = field(
        default_factory=lambda: SliderSignature(
            visual=SliderEvidence(score=0, evidence=""),
            audio=SliderEvidence(score=0, evidence=""),
            proof=SliderEvidence(score=0, evidence=""),
            cta=SliderEvidence(score=0, evidence=""),
        )
    )
    evidence_alignment: list[JSONDict] = field(default_factory=list)
    semantic_bundles: list[JSONDict] = field(default_factory=list)
    segment_to_bundle_map: JSONDict = field(default_factory=dict)
    bundle_to_segment_range: JSONDict = field(default_factory=dict)
    storyboard_source: str = "segments"
    source_product_id: str = ""
    original_product_name: str = ""
    original_jtbd: str = ""
    category_strategy_intent: str = ""
    product_strategy_intent: str = ""
    segment_tags: list[SegmentTagRecord] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.5"

    def to_dict(self) -> JSONDict:
        return asdict(self)

    def to_asset_triad(self) -> JSONDict:
        product_helper_record = {
            "source_product_id": self.source_product_id,
            "original_product_name": self.original_product_name,
            "original_jtbd": self.original_jtbd,
            "category_strategy_intent": self.category_strategy_intent,
            "product_strategy_intent": self.product_strategy_intent,
            "intent_source": self.metadata.get("intent_source", "unknown"),
        }
        blueprint_master_record = {
            "blueprint_id": self.blueprint_id,
            "source_product_id": self.source_product_id,
            "source_video": self.source_video,
            "primary_hec_signature": self.primary_hec.get("signature", ""),
            "hook_tag": self.primary_hec.get("hook_tag"),
            "effect_tag": self.primary_hec.get("effect_tag"),
            "cta_tag": self.primary_hec.get("cta_tag"),
            "secondary_effects": self.secondary_effects,
            "slider_signature": self.slider_signature.to_dict(),
            "storyboard_source": self.storyboard_source,
            "semantic_bundle_count": len(self.semantic_bundles),
            "evidence_alignment": self.evidence_alignment,
            "schema_version": self.schema_version,
        }
        if self.segment_tags:
            segment_granular_records = [record.to_dict() for record in self.segment_tags]
        else:
            segment_granular_records = [
                {
                    "segment_id": segment.segment_id,
                    "blueprint_id": self.blueprint_id,
                    "source_video": self.source_video,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "segment_role": segment.segment_role,
                    "primary_label": segment.tag.get("primary_label"),
                    "hook_label": segment.tag.get("hook_label"),
                    "effect_label": segment.tag.get("effect_label"),
                    "cta_label": segment.tag.get("cta_label"),
                    "persuasion_function": segment.persuasion_function,
                    "metadata": {"source_product_id": self.source_product_id},
                    "schema_version": "v0.1",
                }
                for segment in self.storyboard_segments
            ]
        return {
            "cascade": "1:N:N",
            "product_helper_record": product_helper_record,
            "blueprint_master_record": blueprint_master_record,
            "segment_granular_records": segment_granular_records,
        }


@dataclass(slots=True)
class MatchVerdict:
    """模式 B 匹配网关结果。"""

    gate1_pass: bool
    gate2_pass: bool
    gate3a_pass: bool
    gate3b_pass: bool
    patch_required: bool
    risk_flags: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    matched_variant_id: str = ""
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.2"

    @property
    def gate3_pass(self) -> bool:
        return self.gate3a_pass and self.gate3b_pass

    @property
    def status(self) -> Literal["green", "patch_required", "blocked"]:
        if self.gate1_pass and self.gate2_pass and self.gate3_pass and not self.patch_required:
            return "green"
        if self.patch_required:
            return "patch_required"
        return "blocked"

    def to_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["gate3_pass"] = self.gate3_pass
        payload["status"] = self.status
        return payload


@dataclass(slots=True)
class ScriptPackage:
    mode: Literal["mode_a", "mode_b"]
    script_text: str
    storyboard: list[JSONDict]
    used_hec: JSONDict
    used_slider: SliderSignature
    source_assets: list[str] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)
    schema_version: str = "v0.2"

    def to_dict(self) -> JSONDict:
        return asdict(self)


__all__ = [
    "JSONDict",
    "SegmentRole",
    "ValidationError",
    "RawBlueprintModel",
    "RawStoryboardSegment",
    "RawVisualGuidance",
    "RawTimedAudioEvent",
    "RawSFXEvent",
    "RawBGMEvent",
    "RawAuditoryText",
    "RawPerformanceEmotion",
    "SliderEvidence",
    "SliderSignature",
    "VisualGuidance",
    "TimedAudioEvent",
    "SFXEvent",
    "BGMEvent",
    "AuditoryText",
    "PerformanceEmotion",
    "StoryboardSegment",
    "SegmentTagRecord",
    "VideoFactPack",
    "CandidateSet",
    "ProductECSkeleton",
    "ProductHEC",
    "ProductDiagnosis",
    "VideoBlueprint",
    "MatchVerdict",
    "ScriptPackage",
]


# =============================================================================
# 说服要求建模模块（Persuasion Requirement Modeling）协议族 —— V3.1 一期
#
# 关联：PRD1 §7（输出协议）、§8（说服要求框架）、§11（视频诊断契约）、§12（工程断言）。
# 由 core_skill/engines/product_diagnoser.py 反向 import；不在本文件迁移
# ProductDiagnosisOutput（一期边界：保持其原位定义）。
# 所有模型继承 StrictBaseModel（extra="forbid"），任何越界字段一律拦截。
# =============================================================================

# decision_gap 七分类（PRD1 §8.1）
DECISION_GAP_VALUES: tuple[str, ...] = (
    "need_gap",
    "fit_gap",
    "value_gap",
    "proof_gap",
    "trust_gap",
    "risk_gap",
    "action_gap",
)
DecisionGap = Literal[
    "need_gap",
    "fit_gap",
    "value_gap",
    "proof_gap",
    "trust_gap",
    "risk_gap",
    "action_gap",
]

# content_goal 九项闭集（PRD1 §6.2）
CONTENT_GOAL_VALUES: tuple[str, ...] = (
    "conversion",
    "purchase",
    "add_to_cart",
    "coupon_claim",
    "shop_entry",
    "seeding",
    "education",
    "brand_awareness",
    "unknown",
)
ContentGoal = Literal[
    "conversion",
    "purchase",
    "add_to_cart",
    "coupon_claim",
    "shop_entry",
    "seeding",
    "education",
    "brand_awareness",
    "unknown",
]

# action_gap 仅在以下 5 类转化目标下激活（PRD1 §6.2 / §12.3）
ACTION_GOALS: frozenset[str] = frozenset(
    {"conversion", "purchase", "add_to_cart", "coupon_claim", "shop_entry"}
)

# 23 条 active MVP 白名单（PRD1 §8.2 / §12.1.1）——线上唯一合法 requirement 集合
ACTIVE_REQUIREMENT_WHITELIST: tuple[str, ...] = (
    "expose_current_pain",
    "clarify_usage_scenario",
    "identify_target_user",
    "prove_user_fit",
    "prove_scenario_fit",
    "prove_spec_fit",
    "prove_core_benefit",
    "prove_new_solution_efficiency",
    "establish_clear_difference",
    "prove_replacement_value",
    "prove_price_reasonableness",
    "provide_visible_result",
    "prove_effect_not_degraded",
    "prove_quality_stability",
    "establish_basic_trust",
    "prove_source_credibility",
    "provide_authority_endorsement",
    "reduce_trial_risk",
    "resolve_quality_risk",
    "resolve_safety_risk",
    "resolve_value_risk",
    "prove_current_purchase_reason",
    "clarify_purchase_threshold",
)

Priority = Literal["high", "medium", "low"]
JTBDTemplateStatus = Literal["matched", "fallback_generic"]
RequirementCompletionStatus = Literal["completed", "partial", "missing", "not_applicable"]

# requirement_completion_schema.status_enum 固定值（PRD1 §11.1）
REQUIREMENT_STATUS_ENUM: tuple[str, ...] = ("completed", "partial", "missing", "not_applicable")
# 诊断维度固定值（PRD1 §11.1）
DIAGNOSIS_DIMENSIONS: tuple[str, ...] = (
    "whether_requirement_appears",
    "whether_evidence_is_sufficient",
    "whether_sequence_is_reasonable",
    "whether_risk_is_resolved",
)

# 旧字段废弃黑名单（PRD1 §13.3 / §12.1.7）——出现一次即拦截，不允许并存
DEPRECATED_PERSUASION_KEYS: tuple[str, ...] = (
    "persuasion_profile",
    "required_persuasion_tasks",
    "task_id",
    "task_name",
    "task_type",
    "hec_task_mapping",
)


def assert_no_deprecated_persuasion_keys(payload: Any, *, where: str = "payload") -> None:
    """旧说服任务字段强拦截（PRD1 §12.1.7、§13.3）。

    旧 ``persuasion_profile`` 概念已废弃，不再双写；新协议统一使用
    ``persuasion_requirement_profile``。任意旧字段出现一次即抛错，禁止静默放过。
    """
    if not isinstance(payload, dict):
        return
    hit = [key for key in DEPRECATED_PERSUASION_KEYS if key in payload]
    if hit:
        raise ValueError(
            f"{where} 命中已废弃说服任务字段 {hit}；旧 persuasion_profile 概念已废弃，"
            f"请统一使用 persuasion_requirement_profile（PRD1 §12.1.7/§13.3）。"
        )


class PersuasionRequirement(StrictBaseModel):
    """单条说服要求（PRD1 §7.4）。requirement_id 必须命中 23 条 active 白名单。"""

    requirement_id: str
    requirement_name: str
    decision_gap: DecisionGap
    source: list[str] = Field(default_factory=list)
    priority: Priority
    required: bool
    sequence_rank: int = Field(ge=10, le=59)
    success_criteria: str
    related_decision_criteria: list[str] = Field(default_factory=list)
    required_evidence_requirements: list[str] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)

    @validator("requirement_id")
    def _requirement_in_whitelist(cls, value: str) -> str:
        if value not in ACTIVE_REQUIREMENT_WHITELIST:
            raise ValueError(
                f"requirement_id={value} 不在 23 条 active MVP 白名单内（PRD1 §12.1.1）。"
            )
        return value

    @validator("source")
    def _source_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("persuasion_requirement.source 不允许为空。")
        return value


class NotApplicableRequirement(StrictBaseModel):
    """未激活要求（PRD1 §7.4 not_applicable_requirements）。"""

    requirement_id: str
    decision_gap: DecisionGap
    status: Literal["not_applicable"] = "not_applicable"
    reason: str

    @validator("requirement_id")
    def _requirement_in_whitelist(cls, value: str) -> str:
        if value not in ACTIVE_REQUIREMENT_WHITELIST:
            raise ValueError(
                f"not_applicable requirement_id={value} 不在 23 条 active 白名单内（PRD1 §12.1.1）。"
            )
        return value


class PrimaryJTBD(StrictBaseModel):
    level1: str
    level2: str


class CategoryResistance(StrictBaseModel):
    rule: str
    summary: str


class ProductConversionBarrier(StrictBaseModel):
    rule: str


class MainPersuasionRoute(StrictBaseModel):
    """主说服路线（PRD1 §7.2 main_persuasion_route）。"""

    primary_jtbd: PrimaryJTBD
    category_resistance: CategoryResistance
    product_conversion_barrier: ProductConversionBarrier


class ActivatedCategoryRequirements(StrictBaseModel):
    """品类扩展激活结果（PRD1 §7.2 activated_category_requirements）。

    未命中路由字典时三段固定为空、routing_confidence 为空字符串。
    routing_confidence 在此结构内透出（PRD1 §9.4 / TC-CR-005），避免污染顶层协议字段集合。
    """

    category_group: str
    routing_confidence: str = ""
    activated_decision_criteria: list[str] = Field(default_factory=list)
    activated_evidence_requirements: list[str] = Field(default_factory=list)
    activated_risk_points: list[str] = Field(default_factory=list)


class RequirementCompletionSchema(StrictBaseModel):
    """requirement 完成度契约（PRD1 §11.1）。"""

    status_enum: list[RequirementCompletionStatus] = Field(
        default_factory=lambda: list(REQUIREMENT_STATUS_ENUM)
    )
    minimum_required_requirements: list[str] = Field(default_factory=list)
    diagnosis_dimensions: list[str] = Field(
        default_factory=lambda: list(DIAGNOSIS_DIMENSIONS)
    )

    @validator("status_enum")
    def _status_enum_fixed(cls, value: list[str]) -> list[str]:
        if list(value) != list(REQUIREMENT_STATUS_ENUM):
            raise ValueError(
                f"status_enum 必须固定为 {list(REQUIREMENT_STATUS_ENUM)}（PRD1 §11.1）。"
            )
        return value

    @validator("minimum_required_requirements")
    def _minimum_in_whitelist(cls, value: list[str]) -> list[str]:
        illegal = [rid for rid in value if rid not in ACTIVE_REQUIREMENT_WHITELIST]
        if illegal:
            raise ValueError(
                f"minimum_required_requirements 含非白名单项 {illegal}（PRD1 §12.1.1）。"
            )
        return value

    @validator("diagnosis_dimensions")
    def _dimensions_fixed(cls, value: list[str]) -> list[str]:
        if list(value) != list(DIAGNOSIS_DIMENSIONS):
            raise ValueError(
                f"diagnosis_dimensions 必须固定为 {list(DIAGNOSIS_DIMENSIONS)}（PRD1 §11.1）。"
            )
        return value


class DiagnosisContract(StrictBaseModel):
    """视频诊断契约（PRD1 §11.1）。"""

    requirement_completion_schema: RequirementCompletionSchema


class PersuasionRequirementProfile(StrictBaseModel):
    """说服要求 profile 顶层协议（PRD1 §7.2）。

    顶层字段集合严格等于 PRD1 §7.2 的 11 项，extra="forbid" 拦截任何越界字段
    （TC-SC-002）。作为 ProductDiagnosisOutput 的顶层旁路字段输出。
    """

    profile_version: str = "v3.1"
    content_goal: ContentGoal
    category_group: str
    jtbd_template_status: JTBDTemplateStatus
    requirement_dictionary_version: str
    category_purchase_criteria_version: str = ""
    main_persuasion_route: MainPersuasionRoute
    activated_category_requirements: ActivatedCategoryRequirements
    persuasion_requirements: list[PersuasionRequirement] = Field(default_factory=list)
    not_applicable_requirements: list[NotApplicableRequirement] = Field(default_factory=list)
    diagnosis_contract: DiagnosisContract

    @root_validator
    def _action_gap_governance(cls, values: dict[str, Any]) -> dict[str, Any]:
        """action_gap 激活治理（PRD1 §12.1.3）：

        非转化目标下，persuasion_requirements 不得包含 action_gap 要求；
        其必须落入 not_applicable_requirements。
        """
        content_goal = values.get("content_goal")
        requirements = values.get("persuasion_requirements") or []
        if content_goal not in ACTION_GOALS:
            leaked = [
                r.requirement_id
                for r in requirements
                if getattr(r, "decision_gap", None) == "action_gap"
            ]
            if leaked:
                raise ValueError(
                    f"content_goal={content_goal} 非转化目标，action_gap 要求 {leaked} "
                    f"不得进入 persuasion_requirements，必须输出 not_applicable（PRD1 §12.1.3）。"
                )
        return values
