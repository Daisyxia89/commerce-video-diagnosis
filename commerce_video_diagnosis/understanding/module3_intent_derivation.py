"""模块 3：策略意图推导 + 武器库挂载 + 动态寻址。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from commerce_video_diagnosis.understanding.schemas.protocols import CandidateSet

try:
    from commerce_video_diagnosis.understanding.validators.schema_assertions import VALID_JTBD as _VALID_JTBD
except Exception:  # pragma: no cover
    _VALID_JTBD = frozenset()

ROOT = Path(__file__).resolve().parent
WEAPON_LIBRARY_PATH = ROOT / "data" / "hec_weapon_library_snapshot.json"
TAXONOMY_PATH = ROOT / "memory/topics/taxonomy_dictionary_v2.md"

LEGACY_FIXED_TAG_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"红海[-×].*?(?:必然|固定|直接|映射|等于).{0,12}[HEC]\d"),
    re.compile(r"蓝海[-×].*?(?:必然|固定|直接|映射|等于).{0,12}[HEC]\d"),
    re.compile(r"(?:快消|耐消).{0,12}(?:必然|固定|直接|映射|等于).{0,12}[HEC]\d"),
    re.compile(r"(?:四象限|象限).{0,12}(?:固定标签|旧链路|HEC)"),
)

CATEGORY_INTENT_MAP: Final[dict[tuple[str, str], str]] = {
    ("蓝海", "快消"): "R01_蓝海品类教育",
    ("蓝海", "耐消"): "R01_蓝海品类教育",
    ("红海-核心", "快消"): "R02_存量同类替换",
    ("红海-核心", "耐消"): "R03_存量选型决策",
    ("红海-破圈", "快消"): "R04_高频旧习惯迁移",
    ("红海-破圈", "耐消"): "R05_低频惰性替换",
}

PRODUCT_INTENT_MAP: Final[dict[tuple[str, str], str]] = {
    ("大牌", "低价"): "P01_顺滑收单",
    ("大牌", "高价"): "P02_价值证明",
    ("白牌", "低价"): "P03_低门槛试单",
    ("白牌", "高价"): "P04_先建信再成交",
}

BACKBONE_COPY: Final[dict[str, str]] = {
    "R01": "先完成品类教育与认知建模，再把商品放进具体人群/场景，最后推动行动。",
    "R02": "先证明为什么值得替换旧方案，再把价值压到行动决策。",
    "R03": "先降低选型风险与理解门槛，再推动成交。",
    "R04": "先剥离旧 SOP 与使用惯性，再证明新方案更省事、更值得立刻切换。",
    "R05": "先制造不换就继续吃亏的感受，再完成替换论证与收口。",
}

LABEL_MAP: Final[dict[str, str]] = {
    "H1": "H1 痛点/焦虑直击",
    "H2": "H2 利益/价格前置",
    "H3": "H3 反差结果前置",
    "H4": "H4 即时操作展示",
    "H5": "H5 反常识与悬念",
    "H6": "H6 场景/人群代入",
    "H7": "H7 明星/权威同款",
    "E0": "E0 单点演示",
    "E1": "E1 效果测评",
    "E2": "E2 暴力实测",
    "E3": "E3 对比/拉踩",
    "E4": "E4 感官实证",
    "E5": "E5 保姆级教程",
    "E6": "E6 成分/参数科普",
    "E7": "E7 产地溯源/工厂实录",
    "C1": "C1 利益/价格逼单",
    "C2": "C2 福利/保障机制",
    "C3": "C3 指令行动",
    "C4": "C4 人群/场景总结",
    "C5": "C5 效果留白/情绪定格",
}

E_AXIS_PREFERENCES: Final[dict[str, tuple[str, ...]]] = {
    "R01": ("E6", "E0", "E5", "E1"),
    "R02": ("E3", "E1", "E0", "E6"),
    "R03": ("E6", "E1", "E0", "E5"),
    "R04": ("E3", "E1", "E5", "E0"),
    "R05": ("E3", "E1", "E0", "E6"),
    "P01": ("E0", "E3", "E6"),
    "P02": ("E1", "E6", "E3"),
    "P03": ("E0", "E1", "E3", "E6"),
    "P04": ("E6", "E1", "E0", "E3"),
}

C_AXIS_PREFERENCES: Final[dict[str, tuple[str, ...]]] = {
    "R01": ("C4", "C3", "C2"),
    "R02": ("C1", "C3", "C4"),
    "R03": ("C2", "C1", "C4"),
    "R04": ("C3", "C1", "C2"),
    "R05": ("C1", "C2", "C3"),
    "P01": ("C1", "C3", "C4"),
    "P02": ("C1", "C2", "C4"),
    "P03": ("C2", "C3", "C4"),
    "P04": ("C2", "C1", "C3"),
}

H_AXIS_PREFERENCES: Final[dict[str, tuple[str, ...]]] = {
    "R01": ("H5", "H6", "H4", "H1"),
    "R02": ("H1", "H3", "H6", "H2"),
    "R03": ("H6", "H2", "H5", "H1"),
    "R04": ("H5", "H1", "H3", "H6"),
    "R05": ("H1", "H5", "H6", "H3"),
    "P01": ("H2", "H3", "H6"),
    "P02": ("H3", "H2", "H1"),
    "P03": ("H6", "H1", "H2"),
    "P04": ("H1", "H6", "H5", "H7"),
}

MODIFIER_DEFINITIONS: Final[dict[str, dict[str, Any]]] = {
    "channel_risk": {
        "token": "强制渠道防伪自证",
        "inject_e": ("E6",),
        "note": "渠道风险存在，必须在 Core E List 中强制挂载 E6 作为防伪/资质自证动作入口。",
    },
    "has_endorsement": {
        "token": "背书缓释",
        "inject_h": ("H7",),
        "note": "存在外部背书，可降低前期信任解释成本，但不改变主分组。",
    },
}

JTBD_POOL_CONFIG: Final[dict[str, dict[str, str]]] = {
    "生存/运转维系": {
        "pool_id": "jtbd.functional.maintenance",
        "source_record": "功能型任务",
        "primary_path": "先确认商品承担的是基础维系/正常运转任务，再证明它能稳定承接日常刚需，避免表达跳成情绪价值或身份表达。",
    },
    "缺陷修复/冲突消除": {
        "pool_id": "jtbd.functional.defect_repair",
        "source_record": "缺陷修复/冲突消除",
        "primary_path": "先把缺陷、异常或冲突问题具象化，再证明当前商品能把问题修掉、压下去或纠正回来。",
    },
    "降本增效/懒人替代": {
        "pool_id": "jtbd.functional.efficiency_replace",
        "source_record": "功能型任务",
        "primary_path": "先确认旧流程的繁琐、费时或费力，再证明当前商品能用更省事的新方案完成同一任务。",
    },
    "物理安全与风险规避": {
        "pool_id": "jtbd.functional.safety_avoidance",
        "source_record": "功能型任务",
        "primary_path": "先把客观风险说清，再证明当前商品如何通过更稳妥的路径降低真实受伤或事故风险。",
    },
    "情绪安心/主观降险": {
        "pool_id": "jtbd.emotional.reassurance",
        "source_record": "功能型任务",
        "primary_path": "先确认用户的不确定感与主观顾虑，再证明当前商品如何提供可持续的安心感与心理兜底。",
    },
    "新奇探索/瞬时刺激": {
        "pool_id": "jtbd.emotional.novelty",
        "source_record": "功能型任务",
        "primary_path": "先确认用户对新奇刺激的驱动，再证明当前商品为何能提供可被感知的尝鲜价值。",
    },
    "自我犒赏与秩序掌控": {
        "pool_id": "jtbd.emotional.reward_and_control",
        "source_record": "功能型任务",
        "primary_path": "先确认用户的悦己、秩序或掌控诉求，再证明当前商品如何把这种感受稳定兑现。",
    },
    "照护与责任履行": {
        "pool_id": "jtbd.social.care_responsibility",
        "source_record": "功能型任务",
        "primary_path": "先确认照护责任与履责压力，再证明当前商品如何帮助用户把照护任务做对、做到位。",
    },
    "礼赠与关系表达": {
        "pool_id": "jtbd.social.gifting",
        "source_record": "功能型任务",
        "primary_path": "先确认礼赠场景与关系表达诉求，再证明当前商品如何承担体面表达与关系传递。",
    },
    "圈层认同（圈层归属/身份锚定）": {
        "pool_id": "jtbd.social.community_identity",
        "source_record": "功能型任务",
        "primary_path": "先确认圈层共识与同好归属，再证明当前商品为何能成为该圈层的身份锚点。",
    },
    "阶层与审美发信": {
        "pool_id": "jtbd.social.aesthetic_signaling",
        "source_record": "功能型任务",
        "primary_path": "先确认用户希望发出的审美与阶层信号，再证明当前商品如何承担这种外显表达。",
    },
}


FUNCTIONAL_JTBDS: Final[frozenset[str]] = frozenset({
    "生存/运转维系",
    "缺陷修复/冲突消除",
    "降本增效/懒人替代",
    "物理安全与风险规避",
})

EFFECT_COMPLETION_CAPABILITY_MAP: Final[dict[str, tuple[str, ...]]] = {
    "E0": ("scenario_proof_complete",),
    "E1": ("functional_proof_complete",),
    "E2": ("functional_proof_complete",),
    "E3": ("functional_proof_complete", "answer_revealed", "emotion_tension_complete"),
    "E4": ("functional_proof_complete", "emotion_tension_complete"),
    "E5": ("scenario_proof_complete", "answer_revealed"),
    "E6": ("functional_proof_complete", "identity_binding_complete"),
    "E7": ("functional_proof_complete", "identity_binding_complete"),
}

HOOK_SOFT_CONSTRAINTS: Final[dict[str, dict[str, Any]]] = {
    "H5": {
        "trigger_cta_tags": ["C1", "C2"],
        "required_effect_capabilities_all": ["answer_revealed"],
        "unmet_risk_flag": "tempo_discount",
    },
    "H6": {
        "trigger_cta_tags": ["C1", "C2"],
        "required_effect_capabilities_all": ["identity_binding_complete"],
        "unmet_risk_flag": "resonance_break_risk",
    },
    "H7": {
        "trigger_cta_tags": ["C1", "C2"],
        "required_effect_capabilities_all": ["identity_binding_complete"],
        "unmet_risk_flag": "resonance_break_risk",
    },
}

C4_C5_FALLBACK_PRIORITY: Final[tuple[str, ...]] = ("C3", "C1", "C2")


@dataclass(frozen=True, slots=True)
class Module3IntentInput:
    jtbd: str
    cognition_attribute: str
    frequency_attribute: str
    trust_attribute: str
    price_attribute: str
    modifiers: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_weapon_library() -> dict[str, Any]:
    with WEAPON_LIBRARY_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    _assert_weapon_library_clean(payload)
    return payload


def _normalize_label(value: str) -> dict[str, str]:
    code = value.split()[0].strip()
    label_map = _load_taxonomy_label_map()
    if code not in label_map:
        raise ValueError(f"HEC Taxonomy 枚举越界: {code}")
    label = label_map[code]
    return {"code": code, "label": label}


def _scan_for_legacy_fixed_tag_logic(texts: list[str]) -> None:
    for text in texts:
        normalized = str(text or "").strip()
        if not normalized:
            continue
        for pattern in LEGACY_FIXED_TAG_PATTERNS:
            if pattern.search(normalized):
                raise ValueError("检测到旧逻辑残留：存在‘象限 -> 固定标签’或同类脏映射，判定未完成架构切换。")


def _extract_taxonomy_codes() -> dict[str, str]:
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"HEC Taxonomy 缺失: {TAXONOMY_PATH}")
    pattern = re.compile(r"\*\s+\*\*([HEC]\d)\s+([^*]+)\*\*")
    mapping: dict[str, str] = {}
    for line in TAXONOMY_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        code = match.group(1).strip()
        label = f"{code} {match.group(2).strip()}"
        mapping[code] = label
    if not mapping:
        raise ValueError("HEC Taxonomy 为空或格式非法，无法校验候选 H/E/C。")
    return mapping


@lru_cache(maxsize=1)
def _load_taxonomy_label_map() -> dict[str, str]:
    mapping = _extract_taxonomy_codes()
    _scan_for_legacy_fixed_tag_logic(list(mapping.values()))
    return mapping


def _choose_weapon_record(jtbd: str) -> tuple[dict[str, str], dict[str, Any]]:
    payload = _load_weapon_library()
    records = payload["records"]
    config = JTBD_POOL_CONFIG.get(jtbd)
    if not config:
        raise ValueError(f"未配置 JTBD 独立任务池: {jtbd}")
    source_record = config["source_record"]
    if source_record not in records:
        raise ValueError(f"任务武器库缺少记录: {source_record}")
    return config, records[source_record]


def _assert_weapon_library_clean(payload: dict[str, Any]) -> None:
    records = payload.get("records") or {}
    if not isinstance(records, dict) or not records:
        raise ValueError("任务武器库为空或格式非法。")
    texts: list[str] = []
    for record in records.values():
        if not isinstance(record, dict):
            continue
        for key in ("main_line", "constraints"):
            texts.append(str(record.get(key, "")))
        for key in ("candidate_h", "candidate_e", "candidate_c"):
            values = record.get(key) or []
            if isinstance(values, list):
                texts.extend(str(item) for item in values)
    _scan_for_legacy_fixed_tag_logic(texts)


def _rule_code(value: str) -> str:
    return value.split("_", 1)[0]


def _ordered_intersection(pool_codes: list[str], category_preferences: tuple[str, ...], product_preferences: tuple[str, ...]) -> list[str]:
    product_rank = {code: index for index, code in enumerate(product_preferences)}
    category_rank = {code: index for index, code in enumerate(category_preferences)}
    pool_rank = {code: index for index, code in enumerate(pool_codes)}

    def sort_key(code: str) -> tuple[int, int, int, int]:
        in_category = 0 if code in category_rank else 1
        in_product = 0 if code in product_rank else 1
        return (
            in_category,
            category_rank.get(code, len(category_rank) + len(pool_codes)),
            in_product,
            product_rank.get(code, pool_rank.get(code, len(pool_codes))),
        )

    ordered = sorted(dict.fromkeys(pool_codes), key=sort_key)
    return ordered


def _pool_codes(items: list[str]) -> list[str]:
    return [str(item).split()[0].strip() for item in items if str(item).strip()]


def _normalize_modifier_tokens(modifiers: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in modifiers:
        token = str(item).strip()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _resolve_modifier_tokens(modifiers: list[str]) -> list[str]:
    resolved: list[str] = []
    for modifier in _normalize_modifier_tokens(modifiers):
        config = MODIFIER_DEFINITIONS.get(modifier)
        if config:
            token = str(config.get("token") or modifier).strip()
            if token and token not in resolved:
                resolved.append(token)
        elif modifier not in resolved:
            resolved.append(modifier)
    return resolved


def _inject_modifiers(
    *,
    modifiers: list[str],
    core_e_codes: list[str],
    candidate_h_codes: list[str],
    weapon_e_pool: list[str],
    weapon_h_pool: list[str],
) -> tuple[list[str], list[str], list[str]]:
    notes: list[str] = []
    final_e_codes = list(core_e_codes)
    final_h_codes = list(candidate_h_codes)
    weapon_e_set = set(weapon_e_pool)
    weapon_h_set = set(weapon_h_pool)

    for modifier in _normalize_modifier_tokens(modifiers):
        config = MODIFIER_DEFINITIONS.get(modifier)
        if not config:
            continue
        for code in config.get("inject_e", ()):
            if code in weapon_e_set and code not in final_e_codes:
                final_e_codes.append(code)
        for code in config.get("inject_h", ()):
            if code in weapon_h_set and code not in final_h_codes:
                final_h_codes.append(code)
        note = str(config.get("note") or "").strip()
        if note and note not in notes:
            notes.append(note)
    return final_e_codes, final_h_codes, notes


def _collect_cross_pool_intercepts(preferences: tuple[str, ...], pool_codes: list[str], axis_name: str) -> list[str]:
    intercepts: list[str] = []
    pool_set = set(pool_codes)
    for code in dict.fromkeys(preferences):
        if code not in pool_set:
            intercepts.append(f"{axis_name} 轴偏好 {code} 不在当前 JTBD 任务池内，已拦截。")
    return intercepts


def _build_rp_axis_sorting_note(
    *,
    jtbd_pool_id: str,
    category_code: str,
    product_code: str,
    intercept_logs: list[str],
) -> str:
    note = (
        f"先锁定 JTBD 主说服路径对应的任务池 `{jtbd_pool_id}`，再由 R 轴 {category_code} 与 P 轴 {product_code} "
        "仅在池内对 H/E/C 候选做重排，不改写主线。"
    )
    if intercept_logs:
        return note + f" 本轮有 {len(intercept_logs)} 条池外偏好被拦截并留痕。"
    return note + " 本轮未发生跨池越权。"


def _cross_map_weapon_pool(
    *,
    category_code: str,
    product_code: str,
    weapon_record: dict[str, Any],
    modifiers: list[str],
    jtbd_pool_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[str], list[str], str]:
    weapon_h_pool = _pool_codes(weapon_record.get("candidate_h") or [])
    weapon_e_pool = _pool_codes(weapon_record.get("candidate_e") or [])
    weapon_c_pool = _pool_codes(weapon_record.get("candidate_c") or [])

    category_e_preferences = E_AXIS_PREFERENCES.get(category_code, ())
    product_e_preferences = E_AXIS_PREFERENCES.get(product_code, ())
    category_c_preferences = C_AXIS_PREFERENCES.get(category_code, ())
    product_c_preferences = C_AXIS_PREFERENCES.get(product_code, ())
    category_h_preferences = H_AXIS_PREFERENCES.get(category_code, ())
    product_h_preferences = H_AXIS_PREFERENCES.get(product_code, ())

    core_e_codes = _ordered_intersection(weapon_e_pool, category_e_preferences, product_e_preferences)
    core_c_codes = _ordered_intersection(weapon_c_pool, category_c_preferences, product_c_preferences)
    candidate_h_codes = _ordered_intersection(weapon_h_pool, product_h_preferences, category_h_preferences)

    core_e_codes, candidate_h_codes, modifier_notes = _inject_modifiers(
        modifiers=modifiers,
        core_e_codes=core_e_codes,
        candidate_h_codes=candidate_h_codes,
        weapon_e_pool=weapon_e_pool,
        weapon_h_pool=weapon_h_pool,
    )

    intercept_logs = [
        *_collect_cross_pool_intercepts(category_e_preferences + product_e_preferences, weapon_e_pool, "E"),
        *_collect_cross_pool_intercepts(category_c_preferences + product_c_preferences, weapon_c_pool, "C"),
        *_collect_cross_pool_intercepts(category_h_preferences + product_h_preferences, weapon_h_pool, "H"),
    ]
    sorting_note = _build_rp_axis_sorting_note(
        jtbd_pool_id=jtbd_pool_id,
        category_code=category_code,
        product_code=product_code,
        intercept_logs=intercept_logs,
    )

    core_e = [_normalize_label(code) for code in core_e_codes]
    core_c = [_normalize_label(code) for code in core_c_codes]
    candidate_h = [_normalize_label(code) for code in candidate_h_codes]
    return core_e, core_c, candidate_h, modifier_notes, intercept_logs, sorting_note


def _task_domain_bucket(jtbd: str) -> str:
    return "functional" if jtbd in FUNCTIONAL_JTBDS else "emotion_social"


def _effect_completion_capabilities(effect_code: str) -> list[str]:
    return list(EFFECT_COMPLETION_CAPABILITY_MAP.get(effect_code, ()))


def _build_effect_candidate(effect_code: str) -> dict[str, Any]:
    payload = _normalize_label(effect_code)
    capabilities = _effect_completion_capabilities(effect_code)
    payload.update(
        {
            "effect_tag": effect_code,
            "completion_capabilities": capabilities,
            "completion_reason_codes": [f"capability_from_{effect_code.lower()}:{item}" for item in capabilities],
        }
    )
    return payload


def _build_cta_candidate(cta_code: str, *, task_domain: str) -> dict[str, Any]:
    payload = _normalize_label(cta_code)
    required_effect_capabilities_any: list[str] = []
    fallback_priority: list[str] = []
    close_strength = "active_push"
    if cta_code in {"C4", "C5"}:
        close_strength = "passive_close"
        fallback_priority = list(C4_C5_FALLBACK_PRIORITY)
        if task_domain == "functional":
            required_effect_capabilities_any = ["functional_proof_complete", "scenario_proof_complete"]
        else:
            required_effect_capabilities_any = ["identity_binding_complete", "emotion_tension_complete"]
    payload.update(
        {
            "cta_tag": cta_code,
            "close_strength": close_strength,
            "required_effect_capabilities_any": required_effect_capabilities_any,
            "fallback_priority": fallback_priority,
        }
    )
    return payload


def _build_hook_candidate(hook_code: str) -> dict[str, Any]:
    payload = _normalize_label(hook_code)
    payload["hook_tag"] = hook_code
    payload["soft_constraint_contract"] = HOOK_SOFT_CONSTRAINTS.get(hook_code)
    return payload



def derive_category_strategy_intent(cognition_attribute: str, frequency_attribute: str) -> str:
    try:
        return CATEGORY_INTENT_MAP[(cognition_attribute, frequency_attribute)]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"未覆盖的品类意图输入: {(cognition_attribute, frequency_attribute)}") from exc


def derive_product_strategy_intent(trust_attribute: str, price_attribute: str) -> str:
    try:
        return PRODUCT_INTENT_MAP[(trust_attribute, price_attribute)]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"未覆盖的商品意图输入: {(trust_attribute, price_attribute)}") from exc


def derive_candidate_set(input_data: Module3IntentInput) -> CandidateSet:
    if _VALID_JTBD and input_data.jtbd not in _VALID_JTBD:
        raise ValueError(f"非法 JTBD: {input_data.jtbd}")

    category_intent = derive_category_strategy_intent(
        cognition_attribute=input_data.cognition_attribute,
        frequency_attribute=input_data.frequency_attribute,
    )
    product_intent = derive_product_strategy_intent(
        trust_attribute=input_data.trust_attribute,
        price_attribute=input_data.price_attribute,
    )

    pool_config, weapon_record = _choose_weapon_record(input_data.jtbd)
    category_code = _rule_code(category_intent)
    product_code = _rule_code(product_intent)
    core_e, core_c, candidate_h, _, _, _ = _cross_map_weapon_pool(
        category_code=category_code,
        product_code=product_code,
        weapon_record=weapon_record,
        modifiers=input_data.modifiers,
        jtbd_pool_id=pool_config["pool_id"],
    )
    task_domain = _task_domain_bucket(input_data.jtbd)

    return CandidateSet(
        schema_version="v0.5",
        jtbd=input_data.jtbd,
        persuasion_route=pool_config["primary_path"],
        r_rule=category_intent,
        p_rule=product_intent,
        task_domain=task_domain,
        h_list=[_build_hook_candidate(item["code"]) for item in candidate_h],
        effect_list=[_build_effect_candidate(item["code"]) for item in core_e],
        cta_list=[_build_cta_candidate(item["code"], task_domain=task_domain) for item in core_c],
    )


__all__ = [
    "CandidateSet",
    "Module3IntentInput",
    "derive_candidate_set",
    "derive_category_strategy_intent",
    "derive_product_strategy_intent",
]
