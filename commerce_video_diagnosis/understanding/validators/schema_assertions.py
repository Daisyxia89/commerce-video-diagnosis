from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from commerce_video_diagnosis.understanding.keyword_rules import assert_rule_trace, build_rule_trace, get_string_list

from commerce_video_diagnosis.understanding.schemas.protocols import (
    MatchVerdict,
    ProductDiagnosis,
    ScriptPackage,
    SegmentTagRecord,
    VideoBlueprint,
)


class SchemaAssertionError(AssertionError):
    """统一协议层断言异常。"""


ALLOWED_SCRIPT_MODES = {"mode_a", "mode_b"}
REQUIRED_HEC_KEYS = {"hook_tag", "effect_tag", "cta_tag"}
REQUIRED_CANDIDATE_SET_KEYS = {
    "schema_version",
    "jtbd",
    "persuasion_route",
    "r_rule",
    "p_rule",
    "task_domain",
    "h_list",
    "effect_list",
    "cta_list",
}
REQUIRED_EFFECT_CANDIDATE_KEYS = {"code", "label", "effect_tag", "completion_capabilities", "completion_reason_codes"}
REQUIRED_CTA_CANDIDATE_KEYS = {"code", "label", "cta_tag", "close_strength", "required_effect_capabilities_any", "fallback_priority"}
REQUIRED_HOOK_CANDIDATE_KEYS = {"code", "label", "hook_tag", "soft_constraint_contract"}
REQUIRED_HOOK_SOFT_CONSTRAINT_KEYS = {"trigger_cta_tags", "required_effect_capabilities_all", "unmet_risk_flag"}
REQUIRED_CTA_RESOLUTION_KEYS = {"requested_cta_tag", "resolved_cta_tag", "resolution_type", "reason_codes"}
REQUIRED_SOFT_CONSTRAINT_RESULT_KEYS = {"rule_id", "status", "required_capabilities", "missing_capabilities", "risk_flag"}
REQUIRED_EC_SKELETON_KEYS = {"schema_version", "effect_tag", "cta_tag", "effect_capabilities_snapshot", "cta_resolution"}
ALLOWED_PRODUCT_HEC_KEYS = REQUIRED_HEC_KEYS | {
    "variant_id",
    "schema_version",
    "hook_label",
    "effect_label",
    "cta_label",
    "activation_tags",
    "risk_flags",
    "risk_tag",
    "soft_constraint_results",
    "route_tags",
}
LEGACY_HEC_KEYS = {"hook", "effect", "cta"}
REQUIRED_SLIDER_KEYS = {"visual", "audio", "proof", "cta"}
REQUIRED_SLIDER_EVIDENCE_KEYS = {"score", "evidence"}
SLIDER_AGGREGATION_BANNED_TOKENS = ("平均", "均值", "聚合", "加权", "storyboard segment")
REQUIRED_SEGMENT_TAG_KEYS = {
    "segment_id",
    "video_id",
    "blueprint_id",
    "source_video",
    "start_sec",
    "end_sec",
    "segment_role",
    "primary_label",
    "visual_slider",
    "audio_slider",
    "proof_slider",
    "cta_slider",
}
REQUIRED_STORYBOARD_SEGMENT_KEYS = {
    "shot_id",
    "duration",
    "role",
    "tag",
    "visual_description",
    "spoken_lines",
    "keyframe_image",
    "segment_id",
    "start_sec",
    "end_sec",
    "segment_role",
    "keyframe_refs",
    "asr_excerpt",
    "ocr_excerpt",
    "taxonomy",
    "persuasion_function",
    "module5_sliders",
    "confidence",
    "needs_human_review",
    "visual_guidance",
    "auditory_text",
    "performance_emotion",
}
REQUIRED_VISUAL_GUIDANCE_KEYS = {"shot_size", "camera_movement", "visual_core", "lighting_tone"}
REQUIRED_AUDITORY_TEXT_KEYS = {
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
}
REQUIRED_PERFORMANCE_EMOTION_KEYS = {"acting_instructions", "emotion_tension", "emotional_tone", "action_mechanics", "action_intensity"}
REQUIRED_TAXONOMY_KEYS = {"hook_label", "effect_label", "cta_label", "supporting_labels"}
REQUIRED_KEYFRAME_KEYS = {"timestamp_sec", "frame_description"}
REQUIRED_TAG_KEYS = {"primary_label", "hook_label", "effect_label", "cta_label", "supporting_labels"}
SHOT_SIZE_TOKENS = {"特写", "近景", "中景", "全景", "远景", "大全", "半身", "微距", "怼脸"}
ACTION_TOKENS = {
    "展示",
    "拿",
    "举",
    "说",
    "指",
    "推",
    "拉",
    "切",
    "倒",
    "按",
    "拍",
    "摸",
    "揉",
    "拉伸",
    "翻",
    "走",
    "跑",
    "喝",
    "穿",
    "涂",
    "喷",
    "冲",
    "对比",
    "演示",
    "坐",
    "休息",
}
BANNED_STORYBOARD_JARGON_TOKENS = {"局部图纸资产", "图纸资产", "资产包", "资产位", "XX资产"}
REQUIRED_TEXT_RANGE_KEYS = {"start_sec", "end_sec", "text"}
REQUIRED_SFX_EVENT_KEYS = {"event_name", "start_sec", "end_sec"}
REQUIRED_BGM_EVENT_KEYS = {"tone", "start_sec", "end_sec"}
REQUIRED_SEMANTIC_BUNDLE_KEYS = {
    "bundle_id",
    "start_sec",
    "end_sec",
    "segment_ids",
    "bundle_role",
    "aggregation_reason",
    "blocked_boundary_ids",
    "coverage_frame_refs",
}
REQUIRED_BUNDLE_RANGE_KEYS = {"start_segment_index", "end_segment_index", "start_segment_id", "end_segment_id"}
REQUIRED_SECONDARY_EFFECT_KEYS = {"effect_label", "evidence_segment_ids"}
ALLOWED_SECONDARY_EFFECT_KEYS = REQUIRED_SECONDARY_EFFECT_KEYS | {"reason"}
VALID_SECONDARY_EFFECT_LABELS = {f"E{i}" for i in range(8)}

E4_BLACKLIST_CATEGORY_TOKENS = {
    "洗发",
    "洗护",
    "护发",
    "牙膏",
    "抗老",
    "防晒",
    "底妆",
    "粉底",
    "粉底液",
    "气垫",
    "bb霜",
    "遮瑕",
}
E4_NON_EQUIVALENT_TOKENS = {
    "口播",
    "体验描述",
    "丝滑",
    "无拖拽",
    "质地",
    "微距",
    "怼脸",
    "上脸",
    "涂抹",
    "成膜",
    "持妆",
    "报告",
    "参数",
    "成分",
    "实验室",
    "检测",
    "数值",
}
FOOD_CATEGORY_TOKENS = set(get_string_list("schema_assertions.food_category_tokens"))
E7_FACTORY_VISUAL_TOKENS = {
    "工厂",
    "工厂实录",
    "车间",
    "生产线",
    "流水线",
    "基地",
    "果园",
    "牧场",
    "农场",
    "养殖",
    "鱼塘",
    "采摘",
    "原产地",
    "产地",
    "溯源",
    "加工",
    "包装线",
}
PAIN_EXPOSURE_TOKENS = {
    "脏",
    "污",
    "垢",
    "黄",
    "黑",
    "异味",
    "油",
    "痒",
    "堵",
    "卡",
    "残留",
    "难洗",
    "费力",
    "不会",
    "失手",
    "风险",
    "掉",
    "咬",
    "痛",
    "敏感",
    "干",
    "粗糙",
    "暗沉",
    "脱妆",
    "起皮",
    "不干净",
    "包浆",
}
DEFECT_REPAIR_SPECIFIC_TOKENS = {
    "黄",
    "脏",
    "污",
    "痘",
    "塌",
    "秃",
    "卡",
    "裂",
    "斑",
    "味",
    "异味",
    "污渍",
    "水垢",
    "尿垢",
    "油垢",
    "发黄",
    "暗沉",
    "起皮",
    "脱妆",
    "毛躁",
    "打结",
    "开裂",
    "堵塞",
}
DEFECT_REPAIR_PROOF_TOKENS = {
    "修复",
    "修掉",
    "改善",
    "去除",
    "淡化",
    "清掉",
    "冲掉",
    "遮住",
    "遮掉",
    "补回",
    "恢复",
    "对比",
    "前后",
    "旧方案",
    "新方案",
    "测评",
    "实测",
    "见效",
    "干净",
    "变白",
    "抚平",
    "不卡粉",
    "不脱妆",
}
FUTURE_RISK_TOKENS = {
    "预防",
    "防止",
    "避免",
    "防护",
    "保护",
    "隔离",
    "减少风险",
    "降低风险",
    "别受伤",
    "别出事故",
    "防刮",
    "防晒",
    "紫外线",
}
E2_BOUNDARY_SOFT_SCENE_TOKENS = {
    "高温",
    "暴晒",
    "晒",
    "太阳",
    "紫外线",
    "火焰山",
    "补水",
    "降温",
    "舒缓",
    "吸收",
    "水分测试仪",
    "温度计",
    "测温",
    "实测",
    "测试",
    "验证",
}
E2_BOUNDARY_HARD_STRESS_TOKENS = {
    "暴力",
    "极限",
    "摔",
    "砸",
    "电钻",
    "承重",
    "浸水",
    "喷火",
    "强酸",
    "强碱",
}

VALID_JTBD = {
    "生存/运转维系",
    "缺陷修复/冲突消除",
    "降本增效/懒人替代",
    "物理安全与风险规避",
    "情绪安心/主观降险",
    "新奇探索/瞬时刺激",
    "自我犒赏与秩序掌控",
    "照护与责任履行",
    "礼赠与关系表达",
    "圈层认同（圈层归属/身份锚定）",
    "阶层与审美发信",
}


def _as_dict(payload: Any) -> dict[str, Any]:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return payload
    raise SchemaAssertionError(f"不支持的协议对象类型: {type(payload)!r}")


def require_non_empty(value: Any, field_name: str) -> None:
    if value is None:
        raise SchemaAssertionError(f"字段 {field_name} 不允许为空。")
    if isinstance(value, str) and not value.strip():
        raise SchemaAssertionError(f"字段 {field_name} 不允许为空字符串。")
    if isinstance(value, (list, dict, tuple, set)) and not value:
        raise SchemaAssertionError(f"字段 {field_name} 不允许为空集合。")


def require_keys(payload: dict[str, Any], field_name: str, required_keys: Iterable[str]) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise SchemaAssertionError(f"字段 {field_name} 缺少必需键: {missing}")


def require_no_extra_keys(payload: dict[str, Any], field_name: str, allowed_keys: Iterable[str]) -> None:
    extras = sorted(key for key in payload.keys() if key not in set(allowed_keys))
    if extras:
        raise SchemaAssertionError(f"字段 {field_name} 检测到污染字段注入: {extras}")


def _require_score_range(value: Any, field_name: str) -> None:
    if not isinstance(value, int):
        raise SchemaAssertionError(f"字段 {field_name} 必须是整数。")
    if not 0 <= value <= 100:
        raise SchemaAssertionError(f"字段 {field_name} 必须在 0-100 之间。")


def _require_number(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise SchemaAssertionError(f"字段 {field_name} 必须是数值。")


def _assert_hec_payload(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_HEC_KEYS)
    require_no_extra_keys(payload, field_name, ALLOWED_PRODUCT_HEC_KEYS)
    legacy_keys = LEGACY_HEC_KEYS.intersection(payload.keys())
    if legacy_keys:
        raise SchemaAssertionError(f"字段 {field_name} 检测到旧版 HEC 键残留: {sorted(legacy_keys)}")
    for key in REQUIRED_HEC_KEYS:
        require_non_empty(payload.get(key), f"{field_name}.{key}")
    for label_key in ("hook_label", "effect_label", "cta_label"):
        if label_key in payload:
            require_non_empty(payload.get(label_key), f"{field_name}.{label_key}")
    activation_tags = payload.get("activation_tags", [])
    risk_flags = payload.get("risk_flags", [])
    soft_constraint_results = payload.get("soft_constraint_results", [])
    if not isinstance(activation_tags, list):
        raise SchemaAssertionError(f"字段 {field_name}.activation_tags 必须是列表。")
    if not isinstance(risk_flags, list):
        raise SchemaAssertionError(f"字段 {field_name}.risk_flags 必须是列表。")
    if not isinstance(soft_constraint_results, list):
        raise SchemaAssertionError(f"字段 {field_name}.soft_constraint_results 必须是列表。")
    for index, result in enumerate(soft_constraint_results):
        _assert_soft_constraint_result(result, f"{field_name}.soft_constraint_results[{index}]")


def _assert_secondary_effects_payload(
    payload: Any,
    field_name: str,
    *,
    primary_effect_label: str | None = None,
    valid_segment_ids: set[str] | None = None,
) -> None:
    if not isinstance(payload, list):
        raise SchemaAssertionError(f"字段 {field_name} 必须是列表。")
    primary_effect = str(primary_effect_label or "").strip().upper()
    seen_effect_labels: set[str] = set()
    for index, item in enumerate(payload):
        item_field = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            raise SchemaAssertionError(f"字段 {item_field} 必须是对象。")
        require_keys(item, item_field, REQUIRED_SECONDARY_EFFECT_KEYS)
        require_no_extra_keys(item, item_field, ALLOWED_SECONDARY_EFFECT_KEYS)
        effect_label = str(item.get("effect_label") or "").strip().upper()
        if effect_label not in VALID_SECONDARY_EFFECT_LABELS:
            raise SchemaAssertionError(f"字段 {item_field}.effect_label 非法，必须属于 E0-E7。")
        if primary_effect and effect_label == primary_effect:
            raise SchemaAssertionError(f"字段 {item_field}.effect_label 不得与 primary_hec.effect_label 重复。")
        if effect_label in seen_effect_labels:
            raise SchemaAssertionError(f"字段 {item_field}.effect_label 不允许重复。")
        seen_effect_labels.add(effect_label)

        evidence_segment_ids = item.get("evidence_segment_ids")
        if not isinstance(evidence_segment_ids, list) or not evidence_segment_ids:
            raise SchemaAssertionError(f"字段 {item_field}.evidence_segment_ids 必须是非空列表。")
        normalized_segment_ids: list[str] = []
        for seg_index, segment_id in enumerate(evidence_segment_ids):
            normalized_segment_id = str(segment_id or "").strip()
            if not normalized_segment_id:
                raise SchemaAssertionError(
                    f"字段 {item_field}.evidence_segment_ids[{seg_index}] 必须是非空字符串。"
                )
            normalized_segment_ids.append(normalized_segment_id)
        if len(set(normalized_segment_ids)) != len(normalized_segment_ids):
            raise SchemaAssertionError(f"字段 {item_field}.evidence_segment_ids 不允许重复 segment_id。")
        if valid_segment_ids is not None:
            invalid_segment_ids = sorted(set(normalized_segment_ids) - valid_segment_ids)
            if invalid_segment_ids:
                raise SchemaAssertionError(
                    f"字段 {item_field}.evidence_segment_ids 存在无效 segment_id: {invalid_segment_ids}"
                )
        reason = item.get("reason")
        if reason is not None and not str(reason).strip():
            raise SchemaAssertionError(f"字段 {item_field}.reason 若存在则不能为空字符串。")


def _assert_hook_soft_constraint_contract(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_HOOK_SOFT_CONSTRAINT_KEYS)
    require_no_extra_keys(payload, field_name, REQUIRED_HOOK_SOFT_CONSTRAINT_KEYS)
    trigger_cta_tags = payload.get("trigger_cta_tags")
    required_effect_capabilities_all = payload.get("required_effect_capabilities_all")
    if not isinstance(trigger_cta_tags, list) or not trigger_cta_tags:
        raise SchemaAssertionError(f"字段 {field_name}.trigger_cta_tags 必须是非空列表。")
    if not isinstance(required_effect_capabilities_all, list) or not required_effect_capabilities_all:
        raise SchemaAssertionError(f"字段 {field_name}.required_effect_capabilities_all 必须是非空列表。")
    require_non_empty(payload.get("unmet_risk_flag"), f"{field_name}.unmet_risk_flag")


def _assert_effect_candidate(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_EFFECT_CANDIDATE_KEYS)
    require_non_empty(payload.get("code"), f"{field_name}.code")
    require_non_empty(payload.get("label"), f"{field_name}.label")
    require_non_empty(payload.get("effect_tag"), f"{field_name}.effect_tag")
    completion_capabilities = payload.get("completion_capabilities")
    completion_reason_codes = payload.get("completion_reason_codes")
    if not isinstance(completion_capabilities, list) or not completion_capabilities:
        raise SchemaAssertionError(f"字段 {field_name}.completion_capabilities 必须是非空列表。")
    if not isinstance(completion_reason_codes, list):
        raise SchemaAssertionError(f"字段 {field_name}.completion_reason_codes 必须是列表。")


def _assert_cta_candidate(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_CTA_CANDIDATE_KEYS)
    require_non_empty(payload.get("code"), f"{field_name}.code")
    require_non_empty(payload.get("label"), f"{field_name}.label")
    require_non_empty(payload.get("cta_tag"), f"{field_name}.cta_tag")
    close_strength = str(payload.get("close_strength", "")).strip()
    if close_strength not in {"active_push", "passive_close"}:
        raise SchemaAssertionError(f"字段 {field_name}.close_strength 非法。")
    required_effect_capabilities_any = payload.get("required_effect_capabilities_any")
    fallback_priority = payload.get("fallback_priority")
    if not isinstance(required_effect_capabilities_any, list):
        raise SchemaAssertionError(f"字段 {field_name}.required_effect_capabilities_any 必须是列表。")
    if not isinstance(fallback_priority, list):
        raise SchemaAssertionError(f"字段 {field_name}.fallback_priority 必须是列表。")


def _assert_hook_candidate(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_HOOK_CANDIDATE_KEYS)
    require_non_empty(payload.get("code"), f"{field_name}.code")
    require_non_empty(payload.get("label"), f"{field_name}.label")
    require_non_empty(payload.get("hook_tag"), f"{field_name}.hook_tag")
    hook_tag = str(payload.get("hook_tag", "")).strip().upper()
    contract = payload.get("soft_constraint_contract")
    if hook_tag in {"H5", "H6", "H7"}:
        if not isinstance(contract, dict):
            raise SchemaAssertionError(f"字段 {field_name}.soft_constraint_contract 必须是对象。")
        _assert_hook_soft_constraint_contract(contract, f"{field_name}.soft_constraint_contract")
    elif contract is not None:
        raise SchemaAssertionError(f"字段 {field_name} 不应携带 soft_constraint_contract。")


def _assert_cta_resolution(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_CTA_RESOLUTION_KEYS)
    requested_cta_tag = str(payload.get("requested_cta_tag", "")).strip().upper()
    resolved_cta_tag = str(payload.get("resolved_cta_tag", "")).strip().upper()
    resolution_type = str(payload.get("resolution_type", "")).strip()
    reason_codes = payload.get("reason_codes")
    if not requested_cta_tag or not resolved_cta_tag:
        raise SchemaAssertionError(f"字段 {field_name} 缺少 requested/resolved cta。")
    if resolution_type not in {"direct", "downgrade"}:
        raise SchemaAssertionError(f"字段 {field_name}.resolution_type 非法。")
    if not isinstance(reason_codes, list):
        raise SchemaAssertionError(f"字段 {field_name}.reason_codes 必须是列表。")
    if requested_cta_tag == resolved_cta_tag and resolution_type != "direct":
        raise SchemaAssertionError(f"字段 {field_name} direct/downgrade 状态不一致。")
    if requested_cta_tag != resolved_cta_tag and resolution_type != "downgrade":
        raise SchemaAssertionError(f"字段 {field_name} direct/downgrade 状态不一致。")


def _assert_soft_constraint_result(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_SOFT_CONSTRAINT_RESULT_KEYS)
    require_no_extra_keys(payload, field_name, REQUIRED_SOFT_CONSTRAINT_RESULT_KEYS)
    require_non_empty(payload.get("rule_id"), f"{field_name}.rule_id")
    status = str(payload.get("status", "")).strip()
    if status not in {"satisfied", "risk_marked"}:
        raise SchemaAssertionError(f"字段 {field_name}.status 非法。")
    required_capabilities = payload.get("required_capabilities")
    missing_capabilities = payload.get("missing_capabilities")
    if not isinstance(required_capabilities, list):
        raise SchemaAssertionError(f"字段 {field_name}.required_capabilities 必须是列表。")
    if not isinstance(missing_capabilities, list):
        raise SchemaAssertionError(f"字段 {field_name}.missing_capabilities 必须是列表。")
    risk_flag = payload.get("risk_flag")
    if status == "risk_marked" and not risk_flag:
        raise SchemaAssertionError(f"字段 {field_name}.risk_flag 不允许为空。")
    if status == "satisfied" and risk_flag:
        raise SchemaAssertionError(f"字段 {field_name}.risk_flag 在 satisfied 状态下必须为空。")


def _assert_candidate_set(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_CANDIDATE_SET_KEYS)
    require_non_empty(payload.get("schema_version"), f"{field_name}.schema_version")
    require_non_empty(payload.get("jtbd"), f"{field_name}.jtbd")
    require_non_empty(payload.get("persuasion_route"), f"{field_name}.persuasion_route")
    require_non_empty(payload.get("r_rule"), f"{field_name}.r_rule")
    require_non_empty(payload.get("p_rule"), f"{field_name}.p_rule")
    task_domain = str(payload.get("task_domain", "")).strip()
    if task_domain not in {"functional", "emotion_social"}:
        raise SchemaAssertionError(f"字段 {field_name}.task_domain 非法。")
    for key in REQUIRED_CANDIDATE_SET_KEYS:
        if key.endswith("_list"):
            value = payload.get(key)
            if not isinstance(value, list) or not value:
                raise SchemaAssertionError(f"字段 {field_name}.{key} 必须是非空列表。")
    for index, node in enumerate(payload.get("effect_list", [])):
        if not isinstance(node, dict):
            raise SchemaAssertionError(f"字段 {field_name}.effect_list[{index}] 必须是对象。")
        _assert_effect_candidate(node, f"{field_name}.effect_list[{index}]")
    for index, node in enumerate(payload.get("cta_list", [])):
        if not isinstance(node, dict):
            raise SchemaAssertionError(f"字段 {field_name}.cta_list[{index}] 必须是对象。")
        _assert_cta_candidate(node, f"{field_name}.cta_list[{index}]")
    for index, node in enumerate(payload.get("h_list", [])):
        if not isinstance(node, dict):
            raise SchemaAssertionError(f"字段 {field_name}.h_list[{index}] 必须是对象。")
        _assert_hook_candidate(node, f"{field_name}.h_list[{index}]")


def _derive_expected_ec_resolution_priority(candidate_set: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cta_list = candidate_set.get("cta_list", [])
    effect_list = candidate_set.get("effect_list", [])
    available_cta_tags = {
        str(item.get("cta_tag", "")).strip().upper()
        for item in cta_list
        if str(item.get("cta_tag", "")).strip()
    }
    expected_map: dict[tuple[str, str], dict[str, Any]] = {}
    for effect in effect_list:
        effect_tag = str(effect.get("effect_tag", "")).strip().upper()
        effect_capabilities = {
            str(capability).strip() for capability in effect.get("completion_capabilities", []) if str(capability).strip()
        }
        for index, cta in enumerate(cta_list):
            requested_cta_tag = str(cta.get("cta_tag", "")).strip().upper()
            resolved_cta_tag = requested_cta_tag
            required_any = [
                str(capability).strip()
                for capability in cta.get("required_effect_capabilities_any", [])
                if str(capability).strip()
            ]
            if required_any and not any(capability in effect_capabilities for capability in required_any):
                resolved_cta_tag = _resolve_schema_fallback_cta(
                    cta.get("fallback_priority", []),
                    available_cta_tags,
                    requested_cta_tag,
                )
            combo = (effect_tag, resolved_cta_tag)
            expected_map.setdefault(
                combo,
                {
                    "requested_cta_tag": requested_cta_tag,
                },
            )
    return expected_map


def _resolve_schema_fallback_cta(fallback_priority: list[Any], available_cta_tags: set[str], requested_cta_tag: str) -> str:
    for candidate in fallback_priority:
        normalized = str(candidate).strip().upper()
        if normalized in available_cta_tags:
            return normalized
    raise SchemaAssertionError(f"字段 candidate_set.cta_list 中的 CTA {requested_cta_tag} 准入失败后无可用降级目标。")


def _assert_ec_skeleton_resolution_priority(
    payload: dict[str, Any],
    field_name: str,
    expected_resolution_map: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str]:
    combo = (
        str(payload.get("effect_tag", "")).strip().upper(),
        str(payload.get("cta_tag", "")).strip().upper(),
    )
    expected_meta = expected_resolution_map.get(combo)
    if expected_meta is None:
        raise SchemaAssertionError(f"字段 {field_name} 不是由 CandidateSet 可推导的合法 EC 组合。")
    cta_resolution = payload.get("cta_resolution") or {}
    resolved_cta_tag = str(cta_resolution.get("resolved_cta_tag", "")).strip().upper()
    if resolved_cta_tag != combo[1]:
        raise SchemaAssertionError(f"字段 {field_name}.cta_resolution.resolved_cta_tag 必须与骨架层 cta_tag 对齐。")
    requested_cta_tag = str(cta_resolution.get("requested_cta_tag", "")).strip().upper()
    expected_requested_cta_tag = expected_meta["requested_cta_tag"]
    if requested_cta_tag != expected_requested_cta_tag:
        raise SchemaAssertionError(
            f"字段 {field_name}.cta_resolution.requested_cta_tag 并行 CTA 降级去重顺序异常："
            f"effect_tag={combo[0]}, resolved_cta_tag={combo[1]} 应保留输入顺序更早的 {expected_requested_cta_tag}，"
            f"实际收到 {requested_cta_tag}。"
        )
    return combo


def _assert_ec_skeleton(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_EC_SKELETON_KEYS)
    require_non_empty(payload.get("schema_version"), f"{field_name}.schema_version")
    require_non_empty(payload.get("effect_tag"), f"{field_name}.effect_tag")
    require_non_empty(payload.get("cta_tag"), f"{field_name}.cta_tag")
    for label_key in ("effect_label", "cta_label"):
        if label_key in payload:
            require_non_empty(payload.get(label_key), f"{field_name}.{label_key}")
    effect_capabilities_snapshot = payload.get("effect_capabilities_snapshot")
    if not isinstance(effect_capabilities_snapshot, list):
        raise SchemaAssertionError(f"字段 {field_name}.effect_capabilities_snapshot 必须是列表。")
    cta_resolution = payload.get("cta_resolution")
    if not isinstance(cta_resolution, dict):
        raise SchemaAssertionError(f"字段 {field_name}.cta_resolution 必须是对象。")
    _assert_cta_resolution(cta_resolution, f"{field_name}.cta_resolution")
    hook_tag = str(payload.get("hook_tag", "")).strip()
    if hook_tag:
        raise SchemaAssertionError(f"字段 {field_name} 不允许携带 hook_tag；Product_EC_Skeletons 只能输出 EC 骨架。")


def _assert_slider_evidence(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_SLIDER_EVIDENCE_KEYS)
    _require_score_range(payload.get("score"), f"{field_name}.score")
    require_non_empty(payload.get("evidence"), f"{field_name}.evidence")
    evidence_text = str(payload.get("evidence"))
    if field_name.startswith("slider_signature.") and any(token in evidence_text for token in SLIDER_AGGREGATION_BANNED_TOKENS):
        raise SchemaAssertionError(f"字段 {field_name}.evidence 禁止出现均值/聚合话术。")


def _assert_slider_signature(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_SLIDER_KEYS)
    for key in REQUIRED_SLIDER_KEYS:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise SchemaAssertionError(f"字段 {field_name}.{key} 必须是对象。")
        _assert_slider_evidence(value, f"{field_name}.{key}")


def _assert_visual_guidance(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_VISUAL_GUIDANCE_KEYS)
    for key in REQUIRED_VISUAL_GUIDANCE_KEYS:
        require_non_empty(payload.get(key), f"{field_name}.{key}")


def _assert_auditory_text(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_AUDITORY_TEXT_KEYS)
    for key in REQUIRED_AUDITORY_TEXT_KEYS:
        require_non_empty(payload.get(key), f"{field_name}.{key}")


def _assert_performance_emotion(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_PERFORMANCE_EMOTION_KEYS)
    for key in REQUIRED_PERFORMANCE_EMOTION_KEYS:
        require_non_empty(payload.get(key), f"{field_name}.{key}")
    emotion_tension = str(payload.get("emotion_tension") or "").strip()
    emotional_tone = str(payload.get("emotional_tone") or "").strip()
    if emotion_tension and emotional_tone and emotion_tension == emotional_tone:
        raise SchemaAssertionError(f"{field_name}.emotion_tension 不得与 emotional_tone 完全相同。")


def _assert_taxonomy_payload(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_TAXONOMY_KEYS)
    if not isinstance(payload.get("supporting_labels"), list):
        raise SchemaAssertionError(f"字段 {field_name}.supporting_labels 必须是列表。")



def _assert_storyboard_tag(payload: dict[str, Any], field_name: str) -> None:
    require_keys(payload, field_name, REQUIRED_TAG_KEYS)
    require_non_empty(payload.get("primary_label"), f"{field_name}.primary_label")
    if not isinstance(payload.get("supporting_labels"), list):
        raise SchemaAssertionError(f"字段 {field_name}.supporting_labels 必须是列表。")



def _assert_visual_description(value: Any, field_name: str) -> None:
    require_non_empty(value, field_name)
    text = str(value).strip()
    if not any(token in text for token in SHOT_SIZE_TOKENS):
        raise SchemaAssertionError(f"字段 {field_name} 必须明确写出景别，如特写/近景/中景/全景。")
    if not any(token in text for token in ACTION_TOKENS):
        raise SchemaAssertionError(f"字段 {field_name} 必须明确写出主体动作，不能只做抽象总结。")
    for token in BANNED_STORYBOARD_JARGON_TOKENS:
        if token in text:
            raise SchemaAssertionError(f"字段 {field_name} 禁止出现生造黑话：{token}。")



def _assert_spoken_lines(value: Any, field_name: str) -> None:
    require_non_empty(value, field_name)
    text = str(value).strip()
    for token in BANNED_STORYBOARD_JARGON_TOKENS:
        if token in text:
            raise SchemaAssertionError(f"字段 {field_name} 禁止出现生造黑话：{token}。")


def _assert_keyframe_refs(payload: list[dict[str, Any]], field_name: str) -> None:
    if not isinstance(payload, list):
        raise SchemaAssertionError(f"字段 {field_name} 必须是列表。")
    require_non_empty(payload, field_name)
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SchemaAssertionError(f"{field_name}[{index}] 必须是对象。")
        require_keys(item, f"{field_name}[{index}]", REQUIRED_KEYFRAME_KEYS)
        _require_number(item.get("timestamp_sec"), f"{field_name}[{index}].timestamp_sec")
        require_non_empty(item.get("frame_description"), f"{field_name}[{index}].frame_description")


def _assert_text_ranges(payload: list[dict[str, Any]], field_name: str) -> None:
    if not isinstance(payload, list):
        raise SchemaAssertionError(f"字段 {field_name} 必须是列表。")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SchemaAssertionError(f"{field_name}[{index}] 必须是对象。")
        require_keys(item, f"{field_name}[{index}]", REQUIRED_TEXT_RANGE_KEYS)
        _require_number(item.get("start_sec"), f"{field_name}[{index}].start_sec")
        _require_number(item.get("end_sec"), f"{field_name}[{index}].end_sec")
        if float(item["start_sec"]) > float(item["end_sec"]):
            raise SchemaAssertionError(f"{field_name}[{index}] 的 start_sec 不得大于 end_sec。")
        require_non_empty(item.get("text"), f"{field_name}[{index}].text")


def _assert_timed_audio_events(payload: list[dict[str, Any]], field_name: str, required_keys: set[str]) -> None:
    if not isinstance(payload, list):
        raise SchemaAssertionError(f"字段 {field_name} 必须是列表。")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SchemaAssertionError(f"{field_name}[{index}] 必须是对象。")
        require_keys(item, f"{field_name}[{index}]", required_keys)
        _require_number(item.get("start_sec"), f"{field_name}[{index}].start_sec")
        _require_number(item.get("end_sec"), f"{field_name}[{index}].end_sec")
        if float(item["start_sec"]) >= float(item["end_sec"]):
            raise SchemaAssertionError(f"{field_name}[{index}] 的 start_sec 必须小于 end_sec。")
        for key in required_keys - {"start_sec", "end_sec"}:
            require_non_empty(item.get(key), f"{field_name}[{index}].{key}")


def _assert_storyboard_segments(payload: list[dict[str, Any]]) -> None:
    for index, segment in enumerate(payload):
        if not isinstance(segment, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}] 必须是对象。")
        require_keys(segment, f"storyboard_segments[{index}]", REQUIRED_STORYBOARD_SEGMENT_KEYS)
        require_non_empty(segment.get("shot_id"), f"storyboard_segments[{index}].shot_id")
        require_non_empty(segment.get("segment_id"), f"storyboard_segments[{index}].segment_id")
        require_non_empty(segment.get("role"), f"storyboard_segments[{index}].role")
        require_non_empty(segment.get("segment_role"), f"storyboard_segments[{index}].segment_role")
        if str(segment.get("shot_id")).strip() != str(segment.get("segment_id")).strip():
            raise SchemaAssertionError(f"storyboard_segments[{index}].shot_id 必须与 segment_id 对齐。")
        if str(segment.get("role")).strip() != str(segment.get("segment_role")).strip():
            raise SchemaAssertionError(f"storyboard_segments[{index}].role 必须与 segment_role 对齐。")
        _assert_visual_description(segment.get("visual_description"), f"storyboard_segments[{index}].visual_description")
        _assert_spoken_lines(segment.get("spoken_lines"), f"storyboard_segments[{index}].spoken_lines")
        require_non_empty(segment.get("keyframe_image"), f"storyboard_segments[{index}].keyframe_image")
        _require_number(segment.get("duration"), f"storyboard_segments[{index}].duration")
        _require_number(segment.get("start_sec"), f"storyboard_segments[{index}].start_sec")
        _require_number(segment.get("end_sec"), f"storyboard_segments[{index}].end_sec")
        if float(segment["start_sec"]) >= float(segment["end_sec"]):
            raise SchemaAssertionError(f"storyboard_segments[{index}] 的 start_sec 必须小于 end_sec。")
        expected_duration = float(segment["end_sec"]) - float(segment["start_sec"])
        if abs(float(segment["duration"]) - expected_duration) > 1e-6:
            raise SchemaAssertionError(f"storyboard_segments[{index}].duration 必须等于 end_sec - start_sec。")
        tag = segment.get("tag")
        if not isinstance(tag, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].tag 必须是对象。")
        _assert_storyboard_tag(tag, f"storyboard_segments[{index}].tag")
        require_non_empty(segment.get("persuasion_function"), f"storyboard_segments[{index}].persuasion_function")
        _assert_keyframe_refs(segment.get("keyframe_refs"), f"storyboard_segments[{index}].keyframe_refs")
        _assert_text_ranges(segment.get("asr_excerpt"), f"storyboard_segments[{index}].asr_excerpt")
        _assert_text_ranges(segment.get("ocr_excerpt"), f"storyboard_segments[{index}].ocr_excerpt")
        if not isinstance(segment.get("module5_sliders"), dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].module5_sliders 必须是对象。")
        require_keys(segment["module5_sliders"], f"storyboard_segments[{index}].module5_sliders", {"visual_slider", "audio_slider", "proof_slider", "cta_slider"})
        for slider_key in ("visual_slider", "audio_slider", "proof_slider", "cta_slider"):
            slider = segment["module5_sliders"].get(slider_key)
            if not isinstance(slider, dict):
                raise SchemaAssertionError(f"storyboard_segments[{index}].module5_sliders.{slider_key} 必须是对象。")
            _assert_slider_evidence(slider, f"storyboard_segments[{index}].module5_sliders.{slider_key}")
        taxonomy = segment.get("taxonomy")
        if not isinstance(taxonomy, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].taxonomy 必须是对象。")
        _assert_taxonomy_payload(taxonomy, f"storyboard_segments[{index}].taxonomy")
        visual_guidance = segment.get("visual_guidance")
        if not isinstance(visual_guidance, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].visual_guidance 必须是对象。")
        _assert_visual_guidance(visual_guidance, f"storyboard_segments[{index}].visual_guidance")
        auditory_text = segment.get("auditory_text")
        if not isinstance(auditory_text, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].auditory_text 必须是对象。")
        _assert_auditory_text(auditory_text, f"storyboard_segments[{index}].auditory_text")
        if str(auditory_text.get("asr_text", "")).strip() != str(segment.get("spoken_lines", "")).strip():
            raise SchemaAssertionError(f"storyboard_segments[{index}].spoken_lines 必须与 auditory_text.asr_text 对齐。")
        performance_emotion = segment.get("performance_emotion")
        if not isinstance(performance_emotion, dict):
            raise SchemaAssertionError(f"storyboard_segments[{index}].performance_emotion 必须是对象。")
        _assert_performance_emotion(performance_emotion, f"storyboard_segments[{index}].performance_emotion")
        if not isinstance(segment.get("confidence"), (int, float)):
            raise SchemaAssertionError(f"storyboard_segments[{index}].confidence 必须是数值。")
        if not isinstance(segment.get("needs_human_review"), bool):
            raise SchemaAssertionError(f"storyboard_segments[{index}].needs_human_review 必须是布尔值。")


def _assert_segment_tags(payload: list[dict[str, Any]]) -> None:


    for index, record in enumerate(payload):


        if not isinstance(record, dict):


            raise SchemaAssertionError(f"segment_tags[{index}] 必须是对象。")


        require_keys(record, f"segment_tags[{index}]", REQUIRED_SEGMENT_TAG_KEYS)


        require_non_empty(record.get("segment_id"), f"segment_tags[{index}].segment_id")


        require_non_empty(record.get("video_id"), f"segment_tags[{index}].video_id")


        require_non_empty(record.get("blueprint_id"), f"segment_tags[{index}].blueprint_id")


        require_non_empty(record.get("source_video"), f"segment_tags[{index}].source_video")


        require_non_empty(record.get("primary_label"), f"segment_tags[{index}].primary_label")


        if record.get("start_sec") is None or record.get("end_sec") is None:


            raise SchemaAssertionError(f"segment_tags[{index}] 必须包含起止时间。")


        if float(record["start_sec"]) >= float(record["end_sec"]):


            raise SchemaAssertionError(f"segment_tags[{index}] 的 start_sec 必须小于 end_sec。")


        for slider_key in ("visual_slider", "audio_slider", "proof_slider", "cta_slider"):


            _require_score_range(record.get(slider_key), f"segment_tags[{index}].{slider_key}")











def _assert_semantic_bundles(











    semantic_bundles: list[dict[str, Any]],











    storyboard_segment_ids: list[str],











    segment_to_bundle_map: dict[str, str],











    bundle_to_segment_range: dict[str, dict[str, Any]],











    storyboard_source: str,











) -> None:











    if storyboard_source != "segments":











        raise SchemaAssertionError("storyboard_source 必须显式声明为 segments。")











        











    











    if not semantic_bundles:











        raise SchemaAssertionError("semantic_bundles 不允许为空。")











    if not segment_to_bundle_map:











        raise SchemaAssertionError("segment_to_bundle_map 不允许为空。")











    if not bundle_to_segment_range:











        raise SchemaAssertionError("bundle_to_segment_range 不允许为空。")























    physical_segment_ids = [str(segment_id).strip() for segment_id in segment_to_bundle_map.keys()]











    covered_segment_ids: list[str] = []











    seen_bundle_ids: set[str] = set()











    for index, bundle in enumerate(semantic_bundles):











        if not isinstance(bundle, dict):











            raise SchemaAssertionError(f"semantic_bundles[{index}] 必须是对象。")











        require_keys(bundle, f"semantic_bundles[{index}]", REQUIRED_SEMANTIC_BUNDLE_KEYS)











        bundle_id = str(bundle.get("bundle_id") or "").strip()











        if not bundle_id:











            raise SchemaAssertionError(f"semantic_bundles[{index}].bundle_id 不能为空。")











        if bundle_id in seen_bundle_ids:











            raise SchemaAssertionError(f"semantic_bundles[{index}].bundle_id 重复：{bundle_id}。")











        seen_bundle_ids.add(bundle_id)











        _require_number(bundle.get("start_sec"), f"semantic_bundles[{index}].start_sec")











        _require_number(bundle.get("end_sec"), f"semantic_bundles[{index}].end_sec")











        if float(bundle["start_sec"]) >= float(bundle["end_sec"]):











            raise SchemaAssertionError(f"semantic_bundles[{index}] 的 start_sec 必须小于 end_sec。")











        bundle_segment_ids = bundle.get("segment_ids")











        if not isinstance(bundle_segment_ids, list) or not bundle_segment_ids:











            raise SchemaAssertionError(f"semantic_bundles[{index}].segment_ids 不允许为空。")











        if not isinstance(bundle.get("aggregation_reason"), list) or not bundle.get("aggregation_reason"):











            raise SchemaAssertionError(f"semantic_bundles[{index}].aggregation_reason 不允许为空。")











        if not isinstance(bundle.get("coverage_frame_refs"), list) or not bundle.get("coverage_frame_refs"):











            raise SchemaAssertionError(f"semantic_bundles[{index}].coverage_frame_refs 不允许为空。")











        if not isinstance(bundle.get("blocked_boundary_ids"), list):











            raise SchemaAssertionError(f"semantic_bundles[{index}].blocked_boundary_ids 必须是列表。")























        indexes: list[int] = []











        for segment_id in bundle_segment_ids:











            normalized_segment_id = str(segment_id or "").strip()











            if normalized_segment_id not in physical_segment_ids:











                raise SchemaAssertionError(f"semantic_bundles[{index}] 引用了不存在的 segment_id: {normalized_segment_id}。")











            indexes.append(physical_segment_ids.index(normalized_segment_id))











            if str(segment_to_bundle_map.get(normalized_segment_id) or "").strip() != bundle_id:











                raise SchemaAssertionError(











                    f"segment_to_bundle_map[{normalized_segment_id}] 必须回链到 semantic_bundles[{index}]。"











                )











        if indexes != list(range(indexes[0], indexes[-1] + 1)):











            raise SchemaAssertionError(f"semantic_bundles[{index}].segment_ids 必须连续，不允许跳段聚合。")























        bundle_range = bundle_to_segment_range.get(bundle_id)











        if not isinstance(bundle_range, dict):











            raise SchemaAssertionError(f"bundle_to_segment_range[{bundle_id}] 必须存在且为对象。")











        require_keys(bundle_range, f"bundle_to_segment_range[{bundle_id}]", REQUIRED_BUNDLE_RANGE_KEYS)











        if int(bundle_range.get("start_segment_index", -1)) != indexes[0]:











            raise SchemaAssertionError(f"bundle_to_segment_range[{bundle_id}].start_segment_index 不正确。")











        if int(bundle_range.get("end_segment_index", -1)) != indexes[-1]:











            raise SchemaAssertionError(f"bundle_to_segment_range[{bundle_id}].end_segment_index 不正确。")











        if str(bundle_range.get("start_segment_id") or "").strip() != str(bundle_segment_ids[0]).strip():











            raise SchemaAssertionError(f"bundle_to_segment_range[{bundle_id}].start_segment_id 不正确。")











        if str(bundle_range.get("end_segment_id") or "").strip() != str(bundle_segment_ids[-1]).strip():











            raise SchemaAssertionError(f"bundle_to_segment_range[{bundle_id}].end_segment_id 不正确。")











        covered_segment_ids.extend(str(item).strip() for item in bundle_segment_ids)























    if sorted(covered_segment_ids) != sorted(physical_segment_ids):











        raise SchemaAssertionError("semantic_bundles 必须完整且唯一覆盖全部物理 segments。")











    if len(semantic_bundles) > len(physical_segment_ids):











        raise SchemaAssertionError("semantic_bundles 数量不能大于物理 segments 数量。")











    if sorted(seen_bundle_ids) != sorted(storyboard_segment_ids):











        raise SchemaAssertionError("storyboard_segments 必须与 semantic_bundles 逐条对齐。")























def _flatten_text(value: Any) -> str:



    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def _contains_any(text: str, tokens: set[str]) -> bool:
    return any(token in text for token in tokens)


def _segment_has_label(segment: dict[str, Any], label: str) -> bool:
    taxonomy = segment.get("taxonomy", {})
    if taxonomy.get("hook_label") == label:
        return True
    if taxonomy.get("effect_label") == label:
        return True
    if taxonomy.get("cta_label") == label:
        return True
    supporting_labels = taxonomy.get("supporting_labels") or []
    return isinstance(supporting_labels, list) and label in supporting_labels


def _collect_segment_text(segment: dict[str, Any]) -> str:
    return _flatten_text(
        {
            "visual_description": segment.get("visual_description"),
            "persuasion_function": segment.get("persuasion_function"),
            "visual_guidance": segment.get("visual_guidance"),
            "auditory_text": segment.get("auditory_text"),
            "performance_emotion": segment.get("performance_emotion"),
            "keyframe_refs": segment.get("keyframe_refs"),
            "asr_excerpt": segment.get("asr_excerpt"),
            "ocr_excerpt": segment.get("ocr_excerpt"),
        }
    )


def _extract_product_context(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata") or {}
    product_context = metadata.get("product_context") or {}
    content_summary = metadata.get("content_summary") or {}
    return {
        "category_leaf": _flatten_text(product_context.get("category_leaf")),
        "product_name": _flatten_text(data.get("original_product_name") or product_context.get("product_name")),
        "content_summary": _flatten_text(content_summary),
    }


def _is_defect_repair_task(data: dict[str, Any]) -> bool:
    jtbd_text = _flatten_text(data.get("original_jtbd"))
    return "缺陷修复/冲突消除" in jtbd_text


def _assert_defect_repair_boundaries(data: dict[str, Any]) -> None:
    if not _is_defect_repair_task(data):
        return

    segments = data.get("storyboard_segments") or []
    if not segments:
        return

    product_context = _extract_product_context(data)
    all_text = " ".join(_collect_segment_text(segment) for segment in segments)
    all_text = f"{all_text} {product_context['content_summary']}"
    first_segment_text = _collect_segment_text(segments[0])

    if _contains_any(all_text, FUTURE_RISK_TOKENS) and not _contains_any(all_text, DEFECT_REPAIR_SPECIFIC_TOKENS):
        raise SchemaAssertionError(
            "当前内容更像预防未来风险而非修复既有问题，必须改路由至‘物理安全与风险规避’。"
        )

    if not _contains_any(first_segment_text, DEFECT_REPAIR_SPECIFIC_TOKENS):
        raise SchemaAssertionError(
            "缺陷修复/冲突消除 任务缺少已发生且可感知的具体缺陷暴露，必须明确落到黄、脏、痘、塌、秃、卡、裂、斑、味、污等问题对象。"
        )

    proof_segments = segments[1:] if len(segments) > 1 else segments
    proof_labels = {"E0 单点演示", "E1 效果测评", "E2 暴力实测", "E3 对比/拉踩", "E5 保姆级教程"}
    has_proof_label = any(
        segment.get("taxonomy", {}).get("effect_label") in proof_labels
        or _contains_any(_flatten_text(segment.get("taxonomy", {}).get("supporting_labels") or []), proof_labels)
        for segment in proof_segments
    )
    proof_text = " ".join(_collect_segment_text(segment) for segment in proof_segments)
    if not has_proof_label and not _contains_any(proof_text, DEFECT_REPAIR_PROOF_TOKENS):
        raise SchemaAssertionError(
            "缺陷修复/冲突消除 任务缺少扎实修复证据；中段必须回答为什么真的能修、比旧方案强在哪里。"
        )


def _assert_h6_requires_pain_exposure(data: dict[str, Any]) -> None:
    segments = data.get("storyboard_segments") or []
    h6_segments = [segment for segment in segments if _segment_has_label(segment, "H6 场景/人群代入")]
    if not h6_segments:
        return

    early_segments = segments[:2] if len(segments) >= 2 else segments
    early_text = " ".join(_collect_segment_text(segment) for segment in early_segments)
    summary_text = _extract_product_context(data)["content_summary"]
    combined_text = f"{early_text} {summary_text}"
    if not _contains_any(combined_text, PAIN_EXPOSURE_TOKENS):
        raise SchemaAssertionError(
            "H6 场景/人群代入 缺少紧随其后的具体痛点/缺陷暴露，未满足‘场景后必须挂载烂摊子’的硬校验。"
        )


def _assert_e4_boundary_rules(data: dict[str, Any]) -> None:
    segments = data.get("storyboard_segments") or []
    e4_segments = [segment for segment in segments if _segment_has_label(segment, "E4 感官实证")]
    if not e4_segments:
        return

    product_context = _extract_product_context(data)
    category_text = f"{product_context['category_leaf']} {product_context['product_name']}"
    if _contains_any(category_text, E4_BLACKLIST_CATEGORY_TOKENS):
        raise SchemaAssertionError(
            "E4 感官实证 命中洗护/底妆等黑名单类目，按法典必须硬拦截，不能仅靠 Prompt 约束。"
        )

    violating_segments = []
    for segment in e4_segments:
        segment_text = _collect_segment_text(segment)
        if _contains_any(segment_text, E4_NON_EQUIVALENT_TOKENS):
            violating_segments.append(segment.get("segment_id", "unknown"))
    if violating_segments:
        joined = ", ".join(violating_segments)
        raise SchemaAssertionError(
            f"E4 感官实证 出现非等价证据（如口播体验/微距/上脸/参数报告）: {joined}。"
        )


def _assert_e7_food_requires_factory_evidence(data: dict[str, Any]) -> None:
    segments = data.get("storyboard_segments") or []
    e7_segments = [segment for segment in segments if _segment_has_label(segment, "E7 产地溯源/工厂实录")]
    if not e7_segments:
        return

    product_context = _extract_product_context(data)
    category_text = f"{product_context['category_leaf']} {product_context['product_name']}"
    matched_food_keyword = next((token for token in FOOD_CATEGORY_TOKENS if token in category_text), None)
    if not matched_food_keyword:
        return
    assert_rule_trace(build_rule_trace("schema_assertions.food_category_tokens", matched_food_keyword), "schema_assertions.food_category_tokens")

    evidence_text = " ".join(_collect_segment_text(segment) for segment in e7_segments)
    evidence_text = f"{evidence_text} {product_context['content_summary']}"
    if not _contains_any(evidence_text, E7_FACTORY_VISUAL_TOKENS):
        raise SchemaAssertionError(
            "食品/农产品类目若判定 E7 产地溯源/工厂实录，必须出现工厂/基地/产地/采摘等可见源头画面证据。"
        )


def _assert_e1_e2_boundary_rules(data: dict[str, Any]) -> None:
    primary_hec = data.get("primary_hec") or {}
    primary_effect_label = _flatten_text(primary_hec.get("effect_label")).upper()
    if primary_effect_label != "E2":
        return

    segments = data.get("storyboard_segments") or []
    if not segments:
        return

    combined_text = " ".join(_collect_segment_text(segment) for segment in segments)
    content_summary = _extract_product_context(data)["content_summary"]
    combined_text = f"{combined_text} {content_summary}"
    has_hard_stress = _contains_any(combined_text, E2_BOUNDARY_HARD_STRESS_TOKENS)
    has_soft_scene = _contains_any(combined_text, E2_BOUNDARY_SOFT_SCENE_TOKENS)
    if not has_hard_stress and has_soft_scene:
        raise SchemaAssertionError(
            "E1/E2 边界校验失败：当前更像自然场景下的效果验证（如高温暴晒、补水降温、仪器测值），不得判为 E2 暴力实测。"
        )


def _assert_assembly_blocked_status(status: dict[str, Any]) -> None:
    expected = {
        "status": "assembly_blocked",
        "reason_code": "no_expressible_cta_after_admission",
        "jtbd_level1": "自我犒赏",
        "route_context": "R02xP03",
        "blocked_stage": "module4_cta_admission",
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise SchemaAssertionError(f"assembly_blocked.{key} 必须为 {value}。")
    evidence = status.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise SchemaAssertionError("assembly_blocked.evidence 必须是非空列表。")
    if not status.get("user_facing_message"):
        raise SchemaAssertionError("assembly_blocked.user_facing_message 不能为空。")


def _assert_out_of_scope_for_mvp_status(status: dict[str, Any]) -> None:
    if status.get("status") != "out_of_scope_for_mvp":
        raise SchemaAssertionError("out_of_scope_for_mvp.status 必须为 out_of_scope_for_mvp。")

    required_nonempty = [
        "reason_code",
        "supported_stage",
        "unsupported_stage",
        "user_facing_message",
        # trace（裁决 1）
        "scope_gate_status",
        "jtbd_hint",
    ]
    for key in required_nonempty:
        if not str(status.get(key, "")).strip():
            raise SchemaAssertionError(f"out_of_scope_for_mvp.{key} 不能为空。")

    # trace（裁决 1）：命中 token + 字段来源约束。
    matched_tokens = status.get("matched_tokens")
    matched_fields = status.get("matched_fields")
    if not isinstance(matched_tokens, list) or not matched_tokens:
        raise SchemaAssertionError("out_of_scope_for_mvp.matched_tokens 必须是非空列表。")
    if not isinstance(matched_fields, list) or not matched_fields:
        raise SchemaAssertionError("out_of_scope_for_mvp.matched_fields 必须是非空列表。")

    allowed_product_fields = {
        "leaf_category",
        "category_path",
        "product_name",
        "core_selling_points",
        "brand_name",
        "product_detail_summary",
    }
    extra_fields = sorted({str(x) for x in matched_fields} - allowed_product_fields)
    if extra_fields:
        raise SchemaAssertionError(
            f"scope gate 字段来源违规：matched_fields 出现非商品侧字段 {extra_fields}。"
        )

    # 防视频污染断言：若 scope gate 触发，matched_fields 不得出现任何视频侧字段。
    banned_video_fields = {
        "asr_text",
        "ocr_text",
        "vlm_description",
        "video_factpack",
        "audio_text",
        "auditory_text",
    }
    banned_hit = sorted({str(x) for x in matched_fields}.intersection(banned_video_fields))
    if banned_hit:
        raise SchemaAssertionError(
            f"scope gate 命中 trace 被视频污染：matched_fields 包含视频侧字段 {banned_hit}。"
        )


def assert_product_diagnosis(payload: ProductDiagnosis | dict[str, Any]) -> None:
    data = _as_dict(payload)
    legacy_fields = {"strategy_payload", "ec_skeletons", "hec_variants"}.intersection(data.keys())
    if legacy_fields:
        raise SchemaAssertionError(f"检测到旧协议字段残留：{', '.join(sorted(legacy_fields))}")

    # PRD-1.2：needs_review 为中止态，不触发武器库/HEC/完整诊断，协议层短路放行。
    if str(data.get("jtbd", "")).strip() == "needs_review":
        return

    assembly_status = data.get("assembly_status")
    if isinstance(assembly_status, dict) and assembly_status.get("status") == "out_of_scope_for_mvp":
        _assert_out_of_scope_for_mvp_status(assembly_status)
        if data.get("product_ec_skeletons") or data.get("product_hecs"):
            raise SchemaAssertionError("out_of_scope_for_mvp 状态下不得输出 Product_EC_Skeletons / Product_HECs。")
        return

    if isinstance(assembly_status, dict) and assembly_status.get("status") == "assembly_blocked":
        _assert_assembly_blocked_status(assembly_status)
        if data.get("product_ec_skeletons") or data.get("product_hecs"):
            raise SchemaAssertionError("assembly_blocked 状态下不得输出 Product_EC_Skeletons / Product_HECs。")
        return

    for field_name in (
        "product_id",
        "product_name",
        "category",
        "jtbd",
        "resistance_profile",
        "core_intent",
        "candidate_set",
        "product_ec_skeletons",
        "product_hecs",
    ):
        val = data.get(field_name)
        if not val or (isinstance(val, str) and not val.strip()):
            raise Exception("SSOT Data Missing or Invalid Error")

    jtbd = str(data.get("jtbd", "")).strip()
    if jtbd not in VALID_JTBD:
        raise Exception("SSOT Data Missing or Invalid Error")

    core_intent = data.get("core_intent")
    if not isinstance(core_intent, dict):
        raise SchemaAssertionError("字段 core_intent 必须是对象。")

    category_strategy_intent = str(core_intent.get("category_strategy_intent", "")).strip()
    product_strategy_intent = str(core_intent.get("product_strategy_intent", "")).strip()
    if not category_strategy_intent or not product_strategy_intent:
        raise SchemaAssertionError("字段 core_intent 必须包含 category_strategy_intent 与 product_strategy_intent。")

    resistance_profile = data.get("resistance_profile")
    if not isinstance(resistance_profile, dict):
        raise SchemaAssertionError("字段 resistance_profile 必须是对象。")

    candidate_set = data.get("candidate_set")
    if not isinstance(candidate_set, dict):
        raise SchemaAssertionError("字段 candidate_set 必须是对象。")
    _assert_candidate_set(candidate_set, "candidate_set")
    if str(candidate_set.get("jtbd", "")).strip() != jtbd:
        raise SchemaAssertionError("candidate_set.jtbd 必须与顶层 jtbd 对齐。")

    expected_ec_resolution_map = _derive_expected_ec_resolution_priority(candidate_set)
    product_ec_skeletons = data.get("product_ec_skeletons")
    if not isinstance(product_ec_skeletons, list):
        raise SchemaAssertionError("字段 product_ec_skeletons 必须是列表。")
    if not product_ec_skeletons:
        raise SchemaAssertionError("字段 product_ec_skeletons 不允许为空列表。")
    seen_ec_combos: set[tuple[str, str]] = set()
    for index, skeleton in enumerate(product_ec_skeletons):
        if not isinstance(skeleton, dict):
            raise SchemaAssertionError(f"product_ec_skeletons[{index}] 必须是对象。")
        _assert_ec_skeleton(skeleton, f"product_ec_skeletons[{index}]")
        combo = _assert_ec_skeleton_resolution_priority(
            skeleton,
            f"product_ec_skeletons[{index}]",
            expected_ec_resolution_map,
        )
        if combo in seen_ec_combos:
            raise SchemaAssertionError(
                f"product_ec_skeletons[{index}] 检测到重复 EC 组合：effect_tag={combo[0]}, cta_tag={combo[1]}。"
            )
        seen_ec_combos.add(combo)
        if str(skeleton.get("schema_version", "")).strip() != str(candidate_set.get("schema_version", "")).strip():
            raise SchemaAssertionError("CandidateSet 与 Product_EC_Skeletons 的 schema_version 必须同步。")
    expected_combo_set = set(expected_ec_resolution_map.keys())
    if seen_ec_combos != expected_combo_set:
        missing_combos = sorted(expected_combo_set - seen_ec_combos)
        extra_combos = sorted(seen_ec_combos - expected_combo_set)
        raise SchemaAssertionError(
            "Product_EC_Skeletons 必须与 CandidateSet 推导结果严格一致："
            f"missing={missing_combos}, extra={extra_combos}。"
        )

    brand_tier = str(resistance_profile.get("brand_tier", "")).strip()
    financial_risk = str(resistance_profile.get("financial_risk", "")).strip()
    relative_price_level = str(resistance_profile.get("relative_price_level", "")).strip()
    if financial_risk == "高" and relative_price_level != "高水位":
        raise SchemaAssertionError("financial_risk=高 时，relative_price_level 必须为 高水位。")
    if financial_risk == "低" and relative_price_level != "低水位":
        raise SchemaAssertionError("financial_risk=低 时，relative_price_level 必须为 低水位。")

    product_axis = product_strategy_intent.split("_", 1)[0].strip()
    if financial_risk == "高" and product_axis not in {"P02", "P04"}:
        raise SchemaAssertionError("高财务风险只能映射到 P02/P04，发现商品意图轴不一致。")
    if financial_risk == "低" and product_axis not in {"P01", "P03"}:
        raise SchemaAssertionError("低财务风险只能映射到 P01/P03，发现商品意图轴不一致。")

    channel_risk = str(resistance_profile.get("channel_risk", "")).strip()
    if brand_tier == "白牌" and channel_risk == "有风险":
        raise SchemaAssertionError("白牌商品不应被标记为 channel_risk=有风险；该风险仅适用于大牌经销渠道真伪场景。")
    if brand_tier == "大牌经销" and channel_risk not in {"有风险", ""}:
        raise SchemaAssertionError("大牌经销商品的 channel_risk 应显式标记为 有风险。")
    if brand_tier in {"白牌", "大牌官方"} and channel_risk not in {"无风险", ""}:
        raise SchemaAssertionError("白牌/大牌官方商品的 channel_risk 应为 无风险 或留空。")

    if not isinstance(data["product_hecs"], list):
        raise SchemaAssertionError("字段 product_hecs 必须是列表。")
    for index, variant in enumerate(data["product_hecs"]):
        if not isinstance(variant, dict):
            raise SchemaAssertionError(f"product_hecs[{index}] 必须是对象。")
        _assert_hec_payload(variant, f"product_hecs[{index}]")
        if str(variant.get("schema_version", "")).strip() != str(candidate_set.get("schema_version", "")).strip():
            raise SchemaAssertionError("CandidateSet 与 Product_HECs 的 schema_version 必须同步。")


def assert_video_blueprint(payload: VideoBlueprint | dict[str, Any]) -> None:
    data = _as_dict(payload)
    # 基础必填项
    required_fields = (
        "blueprint_id",
        "source_video",
        "primary_hec",
        "secondary_effects",
        "storyboard_segments",
        "slider_signature",
        "evidence_alignment",
        "source_product_id",
        "semantic_bundles",
        "segment_to_bundle_map",
        "bundle_to_segment_range",
        "storyboard_source",
    )
    for field_name in required_fields:
        val = data.get(field_name)
        if field_name == "secondary_effects":
            if val is None:
                raise SchemaAssertionError("字段 secondary_effects 不允许缺失。")
            continue
        if not val or (isinstance(val, str) and not val.strip()):
            raise SchemaAssertionError(f"字段 {field_name} 不允许为空。")

    # 意图基因 (Intent Genes) 校验 - v0.4 引入
    intent_genes = (
        "original_product_name",
        "original_jtbd",
        "category_strategy_intent",
        "product_strategy_intent",
    )
    for field_name in intent_genes:
        # 在协议层设为必填以确保数据对齐
        val = str(data.get(field_name, "")).strip()
        if not val:
            raise Exception("SSOT Data Missing or Invalid Error")
        if field_name == "original_jtbd" and val not in VALID_JTBD:
            raise Exception("SSOT Data Missing or Invalid Error")

    _assert_hec_payload(data["primary_hec"], "primary_hec")
    if not isinstance(data["storyboard_segments"], list):
        raise SchemaAssertionError("字段 storyboard_segments 必须是列表。")
    _assert_storyboard_segments(data["storyboard_segments"])
    valid_storyboard_segment_ids = {
        str(segment.get("segment_id") or "").strip() for segment in data["storyboard_segments"] if segment.get("segment_id")
    }
    _assert_secondary_effects_payload(
        data["secondary_effects"],
        "secondary_effects",
        primary_effect_label=data["primary_hec"].get("effect_label"),
        valid_segment_ids=valid_storyboard_segment_ids,
    )
    _assert_semantic_bundles(
        data["semantic_bundles"],
        [str(segment.get("segment_id") or "").strip() for segment in data["storyboard_segments"]],
        data["segment_to_bundle_map"],
        data["bundle_to_segment_range"],
        str(data.get("storyboard_source") or ""),
    )
    if not isinstance(data["evidence_alignment"], list):
        raise SchemaAssertionError("字段 evidence_alignment 必须是列表。")
    _assert_slider_signature(data["slider_signature"], "slider_signature")
    segment_tags = data.get("segment_tags", [])
    if segment_tags:
        if not isinstance(segment_tags, list):
            raise SchemaAssertionError("字段 segment_tags 必须是列表。")
        _assert_segment_tags(segment_tags)
        if len(segment_tags) != len(data["storyboard_segments"]):
            raise SchemaAssertionError("segment_tags 数量必须与 storyboard_segments 一致。")
        for index, record in enumerate(segment_tags):
            if _flatten_text(record.get("blueprint_id")) != _flatten_text(data.get("blueprint_id")):
                raise SchemaAssertionError(f"segment_tags[{index}].blueprint_id 必须回链当前 blueprint_id。")
            source_product_id = _flatten_text(record.get("metadata", {}).get("source_product_id"))
            if source_product_id and source_product_id != _flatten_text(data.get("source_product_id")):
                raise SchemaAssertionError(
                    f"segment_tags[{index}].metadata.source_product_id 与 blueprint.source_product_id 不一致。"
                )

    _assert_h6_requires_pain_exposure(data)
    _assert_defect_repair_boundaries(data)
    _assert_e4_boundary_rules(data)
    _assert_e7_food_requires_factory_evidence(data)
    _assert_e1_e2_boundary_rules(data)


def assert_match_verdict(payload: MatchVerdict | dict[str, Any]) -> None:
    data = _as_dict(payload)
    for field_name in ("gate1_pass", "gate2_pass", "gate3_pass", "patch_required"):
        if not isinstance(data.get(field_name), bool):
            raise SchemaAssertionError(f"字段 {field_name} 必须是布尔值。")
    if data["patch_required"] and data["gate3_pass"]:
        raise SchemaAssertionError("gate3_pass=True 时，不应同时标记 patch_required=True。")
    if not (data["gate1_pass"] and data["gate2_pass"]) and not data.get("blocked_reason"):
        raise SchemaAssertionError("前置网关失败时，必须填写 blocked_reason。")


MODE_B_PLACEHOLDER_PATTERNS = (
    "保留原视频这种直给节奏",
    "改写成",
    "这一段按",
    "这里补一个商品证据",
)



def _assert_mode_b_script_has_no_placeholder(text: Any, field_name: str) -> None:
    normalized = _flatten_text(text)
    if not normalized:
        raise SchemaAssertionError(f"字段 {field_name} 不能为空。")
    for pattern in MODE_B_PLACEHOLDER_PATTERNS:
        if pattern in normalized:
            raise SchemaAssertionError(f"字段 {field_name} 命中占位模板文案“{pattern}”，按 PRD 必须 Crash Early。")



def assert_script_package(payload: ScriptPackage | dict[str, Any]) -> None:
    data = _as_dict(payload)
    for field_name in ("mode", "script_text", "storyboard", "used_hec", "used_slider"):
        require_non_empty(data.get(field_name), field_name)
    if data["mode"] not in ALLOWED_SCRIPT_MODES:
        raise SchemaAssertionError(f"字段 mode 非法: {data['mode']}")
    _assert_hec_payload(data["used_hec"], "used_hec")
    _assert_slider_signature(data["used_slider"], "used_slider")
    if not isinstance(data["storyboard"], list):
        raise SchemaAssertionError("字段 storyboard 必须是列表。")
    _assert_mode_b_script_has_no_placeholder(data["script_text"], "script_text")
    for index, segment in enumerate(data["storyboard"], start=1):
        if not isinstance(segment, dict):
            raise SchemaAssertionError(f"storyboard[{index}] 必须是对象。")
        _assert_mode_b_script_has_no_placeholder(segment.get("rewritten_spoken_lines"), f"storyboard[{index}].rewritten_spoken_lines")


# =============================================================================
# 前端消费层输出契约（《电商短视频诊断：前端消费层输出契约》SSOT）后置断言
# -----------------------------------------------------------------------------
# 为 12 类契约对象提供 Crash Early 后置断言，由 response_assembler 在装配后调用。
# 命名约定：assert_contract_<object>。不通过一律 raise SchemaAssertionError。
# =============================================================================

# ---- 契约枚举集合 ----
CONTRACT_TOP_STATUS = {
    "diagnosis_completed", "needs_review", "out_of_scope_for_mvp",
    "assembly_blocked", "schema_error", "provider_not_configured",
}
CONTRACT_PRICE_BAND = {"high", "medium", "low", "unknown"}
CONTRACT_TRUST_BARRIER = {"high", "medium", "low", "unknown"}
CONTRACT_CHANNEL_RISK = {"risk", "no_risk", "unknown"}
CONTRACT_ENDORSEMENT = {"has_endorsement", "no_endorsement", "unknown"}
CONTRACT_BRAND_TIER = {"brand", "white_label", "unknown"}
CONTRACT_SUPPORT_REQ_PRIORITY = {"required", "optional"}
CONTRACT_EVIDENCE_SOURCE = {
    "product_factpack", "video_factpack", "product_understanding",
    "video_understanding", "raw_output",
}
CONTRACT_EVIDENCE_ROLE = {"hook", "proof", "safety", "cta", "transition", "other"}
CONTRACT_OVERVIEW_OVERALL = {"pass", "needs_minor_repair", "needs_repair", "mismatch", "blocked", "needs_review"}
CONTRACT_OVERVIEW_AUDIENCE = {"high_match", "partial_match", "low_match", "too_broad", "data_missing"}
CONTRACT_OVERVIEW_PROFILE = {"completed", "partial", "incomplete", "insufficient_evidence", "data_missing"}
CONTRACT_OVERVIEW_HEC = {"matched", "acceptable_deviation", "weak_match", "mismatch", "data_missing"}
CONTRACT_OVERVIEW_SLIDER = {"matched", "mixed_deviation", "slightly_strong", "slightly_weak", "mismatch", "data_missing"}
CONTRACT_OVERVIEW_REQ_COVERAGE = {"completed", "partial", "failed", "data_missing"}
CONTRACT_PROFILE_STATUS = {"completed", "needs_review", "insufficient_evidence", "data_missing"}
CONTRACT_PROFILE_MATCH_RESULT = {"high_match", "partial", "mismatch"}
CONTRACT_GAP_LEVEL = {"high", "medium", "low"}
CONTRACT_REQ_COVERAGE_STATUS = {"completed", "partial", "failed", "data_missing"}
CONTRACT_COMPLETION_STATUS = {"completed", "weak", "missing", "not_applicable"}
CONTRACT_HEC_STATUS = {"matched", "acceptable_deviation", "weak_match", "mismatch", "data_missing"}
CONTRACT_HEC_DIMENSION = {"hook", "effect", "cta", "chain"}
CONTRACT_HEC_DIM_STATUS = {"matched", "acceptable_deviation", "weak_match", "mismatch"}
CONTRACT_SLIDER_STATUS = {"matched", "mixed_deviation", "slightly_strong", "slightly_weak", "mismatch", "insufficient_evidence", "data_missing"}
CONTRACT_SLIDER_AXIS = {"visual", "audio", "proof", "cta"}
CONTRACT_SLIDER_FIT_STATUS = {"fit", "too_strong", "too_weak", "wrong_direction"}
CONTRACT_ISSUE_SEVERITY = {"low", "medium", "high"}
CONTRACT_MODULE = {"audience_match", "profile_match", "requirement_coverage", "hec_match", "slider_match"}
CONTRACT_SUGGESTION_PRIORITY = {"P0", "P1", "P2"}
CONTRACT_QA_STATUS = {"PASS", "FAIL", "NOT_RUN"}
CONTRACT_E2E_STATUS = {"passed", "failed", "not_run"}


def _contract_require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaAssertionError(f"契约字段 {field_name} 必须是对象（dict），实际为 {type(value)!r}。")
    return value


def _contract_require_list(value: Any, field_name: str) -> list:
    if not isinstance(value, list):
        raise SchemaAssertionError(f"契约字段 {field_name} 必须是数组（list），实际为 {type(value)!r}。")
    return value


def _contract_require_enum(value: Any, allowed: set[str], field_name: str) -> None:
    if value not in allowed:
        raise SchemaAssertionError(f"契约字段 {field_name} 枚举非法：{value!r}，合法集合={sorted(allowed)}。")


def _contract_require_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaAssertionError(f"契约字段 {field_name} 必须为非空字符串。")


def _contract_require_key_present(payload: dict[str, Any], key: str, field_name: str) -> None:
    """允许 null，但 key 必须存在。"""
    if key not in payload:
        raise SchemaAssertionError(f"契约字段 {field_name} 必须存在（允许 null，但 key 不可缺失）。")


def assert_contract_evidence_list(value: Any, field_name: str) -> None:
    """统一 evidence schema（契约第 14 节）：每条含 source/field/value/segment_id/confidence。"""
    items = _contract_require_list(value, field_name)
    for idx, ev in enumerate(items):
        ev_dict = _contract_require_dict(ev, f"{field_name}[{idx}]")
        for key in ("source", "field", "value", "segment_id", "confidence"):
            _contract_require_key_present(ev_dict, key, f"{field_name}[{idx}].{key}")
        _contract_require_enum(ev_dict.get("source"), CONTRACT_EVIDENCE_SOURCE, f"{field_name}[{idx}].source")


# ---- 1. diagnosis_meta ----
def assert_contract_diagnosis_meta(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis_meta")
    for key in (
        "diagnosis_id", "request_id", "created_at", "workflow_version", "model_version",
        "model_provider", "source_product_id", "video_id", "qa_status", "e2e_status",
    ):
        _contract_require_key_present(data, key, f"diagnosis_meta.{key}")
    # 必填（裁决 2）：request_id / video_id / source_product_id / diagnosis_id / created_at
    for key in ("diagnosis_id", "request_id", "created_at", "source_product_id", "video_id"):
        _contract_require_str(data.get(key), f"diagnosis_meta.{key}")
    _contract_require_enum(data.get("qa_status"), CONTRACT_QA_STATUS, "diagnosis_meta.qa_status")
    _contract_require_enum(data.get("e2e_status"), CONTRACT_E2E_STATUS, "diagnosis_meta.e2e_status")


# ---- 2. product_understanding ----
def assert_contract_product_understanding(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "product_understanding")
    require_keys(
        data, "product_understanding",
        ("basic_info", "target_people", "core_selling_points", "jtbd",
         "supporting_requirements", "expected_hec", "candidate_set", "conversion_resistance", "evidence"),
    )
    basic = _contract_require_dict(data["basic_info"], "product_understanding.basic_info")
    _contract_require_str(basic.get("product_name"), "product_understanding.basic_info.product_name")
    _contract_require_str(basic.get("leaf_category"), "product_understanding.basic_info.leaf_category")
    for key in ("brand_name", "shop_name", "price"):
        _contract_require_key_present(basic, key, f"product_understanding.basic_info.{key}")
    _contract_require_enum(basic.get("price_band"), CONTRACT_PRICE_BAND, "product_understanding.basic_info.price_band")

    require_non_empty(data.get("target_people"), "product_understanding.target_people")
    require_non_empty(data.get("core_selling_points"), "product_understanding.core_selling_points")

    jtbd = _contract_require_dict(data["jtbd"], "product_understanding.jtbd")
    _contract_require_str(jtbd.get("domain"), "product_understanding.jtbd.domain")
    _contract_require_str(jtbd.get("primary_task"), "product_understanding.jtbd.primary_task")
    _contract_require_str(jtbd.get("reasoning"), "product_understanding.jtbd.reasoning")
    _contract_require_key_present(jtbd, "sub_task", "product_understanding.jtbd.sub_task")
    _contract_require_list(jtbd.get("evidence_chain"), "product_understanding.jtbd.evidence_chain")

    reqs = _contract_require_list(data["supporting_requirements"], "product_understanding.supporting_requirements")
    require_non_empty(reqs, "product_understanding.supporting_requirements")
    for idx, r in enumerate(reqs):
        rd = _contract_require_dict(r, f"supporting_requirements[{idx}]")
        _contract_require_str(rd.get("requirement_id"), f"supporting_requirements[{idx}].requirement_id")
        _contract_require_str(rd.get("requirement_name"), f"supporting_requirements[{idx}].requirement_name")
        _contract_require_enum(rd.get("priority"), CONTRACT_SUPPORT_REQ_PRIORITY, f"supporting_requirements[{idx}].priority")
        _contract_require_key_present(rd, "description", f"supporting_requirements[{idx}].description")

    hec = _contract_require_dict(data["expected_hec"], "product_understanding.expected_hec")
    for key in ("hook_tag", "effect_tag", "cta_tag"):
        _contract_require_str(hec.get(key), f"product_understanding.expected_hec.{key}")

    cs = _contract_require_dict(data["candidate_set"], "product_understanding.candidate_set")
    for key in ("candidate_h", "core_e", "core_c", "primary_effect", "primary_cta"):
        _contract_require_key_present(cs, key, f"product_understanding.candidate_set.{key}")

    cr = _contract_require_dict(data["conversion_resistance"], "product_understanding.conversion_resistance")
    _contract_require_enum(cr.get("trust_barrier"), CONTRACT_TRUST_BARRIER, "conversion_resistance.trust_barrier")
    _contract_require_enum(cr.get("price_barrier"), CONTRACT_TRUST_BARRIER, "conversion_resistance.price_barrier")
    _contract_require_enum(cr.get("channel_risk"), CONTRACT_CHANNEL_RISK, "conversion_resistance.channel_risk")
    _contract_require_enum(cr.get("endorsement"), CONTRACT_ENDORSEMENT, "conversion_resistance.endorsement")
    _contract_require_enum(cr.get("brand_tier"), CONTRACT_BRAND_TIER, "conversion_resistance.brand_tier")

    assert_contract_evidence_list(data.get("evidence"), "product_understanding.evidence")


# ---- 3. video_understanding（含 visual_segments P0 字段非空）----
def assert_contract_video_understanding(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "video_understanding")
    require_keys(
        data, "video_understanding",
        ("video_meta", "text_stream", "visual_stream", "video_base_fact",
         "video_jtbd", "actual_hec", "slider_signature", "evidence_spans"),
    )
    vm = _contract_require_dict(data["video_meta"], "video_understanding.video_meta")
    _contract_require_str(vm.get("video_id"), "video_understanding.video_meta.video_id")
    for key in ("source_platform", "source_url", "duration_sec"):
        _contract_require_key_present(vm, key, f"video_understanding.video_meta.{key}")

    ts = _contract_require_dict(data["text_stream"], "video_understanding.text_stream")
    for key in ("asr_summary", "asr_segments", "ocr_text", "ocr_texts"):
        _contract_require_key_present(ts, key, f"video_understanding.text_stream.{key}")
    _contract_require_list(ts.get("asr_segments"), "video_understanding.text_stream.asr_segments")
    _contract_require_list(ts.get("ocr_texts"), "video_understanding.text_stream.ocr_texts")

    vs = _contract_require_dict(data["visual_stream"], "video_understanding.visual_stream")
    segments = _contract_require_list(vs.get("visual_segments"), "video_understanding.visual_stream.visual_segments")
    if not segments:
        raise SchemaAssertionError("video_understanding.visual_stream.visual_segments 不得为空（契约 6.3）。")
    for idx, seg in enumerate(segments):
        sd = _contract_require_dict(seg, f"visual_segments[{idx}]")
        # P0 必填非空：segment_id / core_scene_desc / core_action
        _contract_require_str(sd.get("segment_id"), f"visual_segments[{idx}].segment_id")
        _contract_require_str(sd.get("core_scene_desc"), f"visual_segments[{idx}].core_scene_desc")
        _contract_require_str(sd.get("core_action"), f"visual_segments[{idx}].core_action")
        # 允许 null 但 key 必须存在
        for key in ("start_sec", "end_sec", "related_asr_segment_id", "rhythm_change_reason"):
            _contract_require_key_present(sd, key, f"visual_segments[{idx}].{key}")
        _contract_require_list(sd.get("related_ocr_texts"), f"visual_segments[{idx}].related_ocr_texts")
        if not isinstance(sd.get("is_rhythm_change_point"), bool):
            raise SchemaAssertionError(f"visual_segments[{idx}].is_rhythm_change_point 必须为布尔。")
        _contract_require_enum(sd.get("evidence_role"), CONTRACT_EVIDENCE_ROLE, f"visual_segments[{idx}].evidence_role")

    vbf = _contract_require_dict(data["video_base_fact"], "video_understanding.video_base_fact")
    if not isinstance(vbf.get("total_segment_count"), int):
        raise SchemaAssertionError("video_base_fact.total_segment_count 必须为整数。")
    _contract_require_list(vbf.get("rhythm_change_points"), "video_base_fact.rhythm_change_points")
    _contract_require_list(vbf.get("key_evidence_actions"), "video_base_fact.key_evidence_actions")
    _contract_require_str(vbf.get("original_fact_record"), "video_base_fact.original_fact_record")

    vj = _contract_require_dict(data["video_jtbd"], "video_understanding.video_jtbd")
    _contract_require_str(vj.get("primary_task"), "video_understanding.video_jtbd.primary_task")
    _contract_require_key_present(vj, "reasoning", "video_understanding.video_jtbd.reasoning")
    _contract_require_list(vj.get("evidence"), "video_understanding.video_jtbd.evidence")

    ah = _contract_require_dict(data["actual_hec"], "video_understanding.actual_hec")
    for key in ("hook_tag", "effect_tag", "cta_tag"):
        _contract_require_str(ah.get(key), f"video_understanding.actual_hec.{key}")
    _contract_require_key_present(ah, "reason", "video_understanding.actual_hec.reason")

    ss = _contract_require_dict(data["slider_signature"], "video_understanding.slider_signature")
    for key in ("visual", "audio", "proof", "cta", "summary"):
        _contract_require_key_present(ss, key, f"video_understanding.slider_signature.{key}")

    assert_contract_evidence_list(data.get("evidence_spans"), "video_understanding.evidence_spans")


# ---- 4. diagnosis.overview ----
def assert_contract_overview(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis.overview")
    _contract_require_enum(data.get("overall_status"), CONTRACT_OVERVIEW_OVERALL, "overview.overall_status")
    _contract_require_enum(data.get("audience_match_status"), CONTRACT_OVERVIEW_AUDIENCE, "overview.audience_match_status")
    _contract_require_enum(data.get("profile_match_status"), CONTRACT_OVERVIEW_PROFILE, "overview.profile_match_status")
    _contract_require_enum(data.get("hec_match_status"), CONTRACT_OVERVIEW_HEC, "overview.hec_match_status")
    _contract_require_enum(data.get("slider_match_status"), CONTRACT_OVERVIEW_SLIDER, "overview.slider_match_status")
    _contract_require_enum(data.get("requirement_coverage_status"), CONTRACT_OVERVIEW_REQ_COVERAGE, "overview.requirement_coverage_status")
    _contract_require_str(data.get("requirement_coverage_text"), "overview.requirement_coverage_text")
    _contract_require_str(data.get("summary"), "overview.summary")


# ---- 5. diagnosis.profile_match ----
def assert_contract_profile_match(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis.profile_match")
    if "available_for_frontend_mapping" in data:
        raise SchemaAssertionError("profile_match 禁止输出 available_for_frontend_mapping（契约 17）。")
    status = data.get("status")
    _contract_require_enum(status, CONTRACT_PROFILE_STATUS, "profile_match.status")
    _contract_require_enum(data.get("match_result"), CONTRACT_PROFILE_MATCH_RESULT, "profile_match.match_result")
    gap = _contract_require_dict(data.get("gap"), "profile_match.gap")
    _contract_require_enum(gap.get("level"), CONTRACT_GAP_LEVEL, "profile_match.gap.level")
    _contract_require_str(gap.get("description"), "profile_match.gap.description")
    for side in ("product_audience", "video_audience"):
        sd = _contract_require_dict(data.get(side), f"profile_match.{side}")
        for key in ("primary", "scene", "core_need"):
            _contract_require_key_present(sd, key, f"profile_match.{side}.{key}")
    assert_contract_evidence_list(data.get("evidence"), "profile_match.evidence")
    # completed / needs_review 下必填 string 不得为空
    if status in {"completed", "needs_review"}:
        for side in ("product_audience", "video_audience"):
            for key in ("primary", "core_need"):
                _contract_require_str((data.get(side) or {}).get(key), f"profile_match.{side}.{key}")
    # insufficient_evidence 下 summary 必须说明缺失原因
    if status == "insufficient_evidence":
        if "缺" not in str(data.get("summary") or ""):
            raise SchemaAssertionError("profile_match insufficient_evidence 必须在 summary 说明缺失原因。")
    # completed 下 evidence 必须双侧覆盖
    if status == "completed":
        sources = {e.get("source") for e in (data.get("evidence") or []) if isinstance(e, dict)}
        if not ({"product_factpack", "product_understanding"} & sources) or not ({"video_factpack", "video_understanding"} & sources):
            raise SchemaAssertionError("profile_match completed evidence 必须同时覆盖商品侧与视频侧。")


# ---- 6. diagnosis.requirement_coverage ----
def assert_contract_requirement_coverage(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis.requirement_coverage")
    _contract_require_enum(data.get("status"), CONTRACT_REQ_COVERAGE_STATUS, "requirement_coverage.status")
    completed = data.get("completed_count")
    total = data.get("total_count")
    if not isinstance(completed, int) or not isinstance(total, int):
        raise SchemaAssertionError("requirement_coverage.completed_count / total_count 必须为整数。")
    if completed > total:
        raise SchemaAssertionError(f"requirement_coverage.completed_count({completed}) 不得大于 total_count({total})。")
    items = _contract_require_list(data.get("items"), "requirement_coverage.items")
    actual_completed = 0
    for idx, it in enumerate(items):
        itd = _contract_require_dict(it, f"requirement_coverage.items[{idx}]")
        _contract_require_str(itd.get("requirement_id"), f"requirement_coverage.items[{idx}].requirement_id")
        _contract_require_str(itd.get("requirement_name"), f"requirement_coverage.items[{idx}].requirement_name")
        if not isinstance(itd.get("required"), bool):
            raise SchemaAssertionError(f"requirement_coverage.items[{idx}].required 必须为布尔。")
        cstatus = itd.get("completion_status")
        _contract_require_enum(cstatus, CONTRACT_COMPLETION_STATUS, f"requirement_coverage.items[{idx}].completion_status")
        for key in ("expected", "actual", "missing_reason", "repair_direction"):
            _contract_require_key_present(itd, key, f"requirement_coverage.items[{idx}].{key}")
        assert_contract_evidence_list(itd.get("matched_evidence_spans"), f"requirement_coverage.items[{idx}].matched_evidence_spans")
        if cstatus == "completed":
            actual_completed += 1
            # completed 项 evidence 必须双侧覆盖（product + video）
            sources = {e.get("source") for e in (itd.get("matched_evidence_spans") or []) if isinstance(e, dict)}
            has_product = bool({"product_factpack", "product_understanding"} & sources)
            has_video = bool({"video_factpack", "video_understanding"} & sources)
            if not (has_product and has_video):
                raise SchemaAssertionError(
                    f"requirement_coverage.items[{idx}] completed 项 evidence 必须双侧覆盖（product+video），实际 sources={sorted(sources)}。"
                )
        elif cstatus in {"weak", "missing"}:
            _contract_require_str(itd.get("missing_reason"), f"requirement_coverage.items[{idx}].missing_reason")
    if actual_completed != completed:
        raise SchemaAssertionError(
            f"requirement_coverage.completed_count({completed}) 与 items 中 completed 实际计数({actual_completed}) 不一致。"
        )
    _contract_require_list(data.get("missing_required_requirements"), "requirement_coverage.missing_required_requirements")
    _contract_require_list(data.get("weak_requirements"), "requirement_coverage.weak_requirements")
    _contract_require_str(data.get("summary"), "requirement_coverage.summary")


# ---- 7. diagnosis.hec_match ----
def assert_contract_hec_match(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis.hec_match")
    status = data.get("status")
    _contract_require_enum(status, CONTRACT_HEC_STATUS, "hec_match.status")
    for side in ("product_expected", "video_actual"):
        sd = _contract_require_dict(data.get(side), f"hec_match.{side}")
        for key in ("hook_tag", "effect_tag", "cta_tag"):
            _contract_require_key_present(sd, key, f"hec_match.{side}.{key}")
    dims = _contract_require_list(data.get("dimension_results"), "hec_match.dimension_results")
    if status != "data_missing" and not dims:
        raise SchemaAssertionError("hec_match.dimension_results 不得为空（非 data_missing 时必须给出商品应然 vs 视频实然对照）。")
    seen_dims = set()
    for idx, dim in enumerate(dims):
        dd = _contract_require_dict(dim, f"hec_match.dimension_results[{idx}]")
        _contract_require_enum(dd.get("dimension"), CONTRACT_HEC_DIMENSION, f"hec_match.dimension_results[{idx}].dimension")
        _contract_require_enum(dd.get("status"), CONTRACT_HEC_DIM_STATUS, f"hec_match.dimension_results[{idx}].status")
        for key in ("expected", "actual", "impact", "suggestion"):
            _contract_require_key_present(dd, key, f"hec_match.dimension_results[{idx}].{key}")
        seen_dims.add(dd.get("dimension"))
    _contract_require_key_present(data, "acceptable_deviation_reason", "hec_match.acceptable_deviation_reason")
    _contract_require_key_present(data, "hec_gap_summary", "hec_match.hec_gap_summary")
    # acceptable_deviation 必须说明业务影响
    if status == "acceptable_deviation":
        _contract_require_str(data.get("acceptable_deviation_reason"), "hec_match.acceptable_deviation_reason")


# ---- 8. diagnosis.slider_match ----
def assert_contract_slider_match(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "diagnosis.slider_match")
    _contract_require_enum(data.get("status"), CONTRACT_SLIDER_STATUS, "slider_match.status")
    _contract_require_list(data.get("target_audience_reference"), "slider_match.target_audience_reference")
    for key in ("expected_slider_preference", "actual_slider_signature"):
        sd = _contract_require_dict(data.get(key), f"slider_match.{key}")
        for axis in ("visual", "audio", "proof", "cta"):
            _contract_require_key_present(sd, axis, f"slider_match.{key}.{axis}")
    axes = _contract_require_list(data.get("axis_results"), "slider_match.axis_results")
    for idx, ar in enumerate(axes):
        ad = _contract_require_dict(ar, f"slider_match.axis_results[{idx}]")
        _contract_require_enum(ad.get("axis"), CONTRACT_SLIDER_AXIS, f"slider_match.axis_results[{idx}].axis")
        _contract_require_enum(ad.get("fit_status"), CONTRACT_SLIDER_FIT_STATUS, f"slider_match.axis_results[{idx}].fit_status")
        for key in ("expected", "actual", "judgment", "repair_direction"):
            _contract_require_key_present(ad, key, f"slider_match.axis_results[{idx}].{key}")
        assert_contract_evidence_list(ad.get("evidence"), f"slider_match.axis_results[{idx}].evidence")
    _contract_require_key_present(data, "audience_acceptance_judgment", "slider_match.audience_acceptance_judgment")
    _contract_require_key_present(data, "slider_gap_summary", "slider_match.slider_gap_summary")


# ---- 9. diagnosis.top_issues ----
def assert_contract_top_issues(payload: Any) -> None:
    items = _contract_require_list(payload, "diagnosis.top_issues")
    for idx, it in enumerate(items):
        itd = _contract_require_dict(it, f"top_issues[{idx}]")
        _contract_require_str(itd.get("issue_id"), f"top_issues[{idx}].issue_id")
        _contract_require_enum(itd.get("severity"), CONTRACT_ISSUE_SEVERITY, f"top_issues[{idx}].severity")
        _contract_require_enum(itd.get("module"), CONTRACT_MODULE, f"top_issues[{idx}].module")
        _contract_require_str(itd.get("title"), f"top_issues[{idx}].title")
        _contract_require_str(itd.get("description"), f"top_issues[{idx}].description")
        _contract_require_key_present(itd, "related_requirement_id", f"top_issues[{idx}].related_requirement_id")
        assert_contract_evidence_list(itd.get("evidence"), f"top_issues[{idx}].evidence")


# ---- 10. diagnosis.suggestions ----
def assert_contract_suggestions(payload: Any, valid_issue_ids: set[str]) -> None:
    items = _contract_require_list(payload, "diagnosis.suggestions")
    for idx, it in enumerate(items):
        itd = _contract_require_dict(it, f"suggestions[{idx}]")
        _contract_require_str(itd.get("suggestion_id"), f"suggestions[{idx}].suggestion_id")
        _contract_require_enum(itd.get("priority"), CONTRACT_SUGGESTION_PRIORITY, f"suggestions[{idx}].priority")
        _contract_require_enum(itd.get("module"), CONTRACT_MODULE, f"suggestions[{idx}].module")
        _contract_require_str(itd.get("action"), f"suggestions[{idx}].action")
        _contract_require_str(itd.get("reason"), f"suggestions[{idx}].reason")
        _contract_require_str(itd.get("expected_effect"), f"suggestions[{idx}].expected_effect")
        _contract_require_key_present(itd, "related_issue_id", f"suggestions[{idx}].related_issue_id")
        # 必须可回指 issue（契约 12.2）
        related = itd.get("related_issue_id")
        if related is not None and related not in valid_issue_ids:
            raise SchemaAssertionError(f"suggestions[{idx}].related_issue_id={related!r} 无法回指到任何 top_issue。")


# ---- 11. artifacts ----
def assert_contract_artifacts(payload: dict[str, Any]) -> None:
    data = _contract_require_dict(payload, "artifacts")
    for key in ("request_payload", "raw_response", "normalized_response"):
        _contract_require_dict(data.get(key), f"artifacts.{key}")
    _contract_require_list(data.get("source_files"), "artifacts.source_files")
    # raw_response 是 Smoke 验收依据：必须保留原始 video_persuasion_diagnosis_result
    raw = data.get("raw_response") or {}
    if "video_persuasion_diagnosis_result" not in raw:
        raise SchemaAssertionError("artifacts.raw_response 必须保留原始 video_persuasion_diagnosis_result。")


# ---- 12. 顶层响应（status + 组装与编排）----
def assert_frontend_contract_response(payload: dict[str, Any]) -> None:
    """前端消费层契约顶层后置断言：校验顶层结构 + 逐一调用 12 类契约对象断言。"""
    data = _contract_require_dict(payload, "frontend_contract_response")
    require_keys(
        data, "frontend_contract_response",
        ("status", "diagnosis_meta", "product_understanding", "video_understanding", "diagnosis", "artifacts"),
    )
    _contract_require_enum(data.get("status"), CONTRACT_TOP_STATUS, "status")

    assert_contract_diagnosis_meta(data["diagnosis_meta"])
    assert_contract_product_understanding(data["product_understanding"])
    assert_contract_video_understanding(data["video_understanding"])

    diagnosis = _contract_require_dict(data["diagnosis"], "diagnosis")
    require_keys(
        diagnosis, "diagnosis",
        ("overview", "profile_match", "hec_match", "slider_match",
         "requirement_coverage", "top_issues", "suggestions"),
    )
    assert_contract_overview(diagnosis["overview"])
    assert_contract_profile_match(diagnosis["profile_match"])
    assert_contract_hec_match(diagnosis["hec_match"])
    assert_contract_slider_match(diagnosis["slider_match"])
    assert_contract_requirement_coverage(diagnosis["requirement_coverage"])
    assert_contract_top_issues(diagnosis["top_issues"])
    issue_ids = {
        it.get("issue_id") for it in (diagnosis["top_issues"] or []) if isinstance(it, dict)
    }
    assert_contract_suggestions(diagnosis["suggestions"], issue_ids)

    assert_contract_artifacts(data["artifacts"])

    # overview 滚动汇总一致性：requirement_coverage_status 必须与 requirement_coverage.status 对齐
    rc_status = (diagnosis["requirement_coverage"] or {}).get("status")
    ov_rc_status = (diagnosis["overview"] or {}).get("requirement_coverage_status")
    if rc_status != ov_rc_status:
        raise SchemaAssertionError(
            f"overview.requirement_coverage_status({ov_rc_status}) 与 requirement_coverage.status({rc_status}) 不一致。"
        )
