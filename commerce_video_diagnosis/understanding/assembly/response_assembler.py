"""前端消费层输出契约装配层（response_assembler）。

契约 SSOT：《电商短视频诊断：前端消费层输出契约》。

设计原则（Crash Early 防御式编程）：
- 所有「无法可靠输出」的契约**必填**字段一律 Crash Early（抛 ContractAssemblyError），
  禁止默认值 / 占位 / 前端 fallback 掩盖。
- 契约明确「可 null」的字段，保留 key 输出 null。
- 不污染现有引擎 raw 输出：raw_response 原样保留 video_persuasion_diagnosis_result。

顶层结构：
    { status, diagnosis_meta, product_understanding, video_understanding,
      diagnosis: { overview, profile_match, hec_match, slider_match,
                   requirement_coverage, top_issues, suggestions },
      artifacts: { request_payload, raw_response, normalized_response, source_files } }
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from commerce_video_diagnosis.understanding.assembly.hec_dictionary import lookup_hec
from commerce_video_diagnosis.understanding.validators.schema_assertions import (
    assert_frontend_contract_response,
)


class ContractAssemblyError(ValueError):
    """契约装配失败（必填字段无可靠来源 → Crash Early）。"""


# ---------------------------------------------------------------------------
# 枚举映射表（以契约文档为准；引擎内部枚举 → 契约枚举）
# ---------------------------------------------------------------------------
# 第二批 contract 治理：商品应然侧（product_understanding）不再输出旧版「转化阻力」裸字段，
# 故 _PRICE_BAND_MAP / _TRUST_BARRIER_MAP / _FINANCIAL_RISK_MAP / _map_channel_risk / _map_brand_tier
# 全部从 contract 输出路径移除（见 build_product_fact_vector）。

# 商品事实向量枚举闭集（严格按 module3 PRD §5.2「商品事实向量枚举约束」；
# 口径复用 product_diagnoser._to_module3_category_attr / _to_module3_trust_attr / _to_module3_price_attr）。
_FACT_COGNITION_ENUM = {"蓝海", "红海-核心", "红海-破圈"}
_FACT_FREQUENCY_ENUM = {"快消", "耐消"}
_FACT_TRUST_ENUM = {"大牌", "白牌"}
_FACT_PRICE_ENUM = {"高", "低"}
# endorsement / channel_risk 允许合法空值 null（不伪造）。

# 引擎诊断枚举 → 契约 overview / 模块枚举
_OVERALL_STATUS_MAP = {
    "good": "pass",
    "needs_minor_repair": "needs_minor_repair",
    "needs_major_repair": "needs_repair",
    "mismatch": "mismatch",
}
_AUDIENCE_MATCH_MAP = {
    "high_match": "high_match",
    "partial_match": "partial_match",
    "low_match": "low_match",
    "too_broad": "too_broad",
}
_PROFILE_LEGACY_TO_OVERVIEW = {
    "completed": "completed",
    "partial": "partial",
    "weak": "partial",
    "missing": "incomplete",
    "not_applicable": "insufficient_evidence",
}
_HEC_STATUS_MAP = {
    "good": "matched",
    "acceptable_deviation": "acceptable_deviation",
    "risky_deviation": "weak_match",
    "mismatch": "mismatch",
}
_SLIDER_STATUS_MAP = {
    "fit": "matched",
    "mixed_deviation": "mixed_deviation",
    "too_strong": "slightly_strong",
    "too_weak": "slightly_weak",
    "wrong_direction": "mismatch",
    "mismatch": "mismatch",
}
# 引擎 requirement completion_status → 契约 completion_status（契约无 partial，partial 归并为 weak）
_COMPLETION_STATUS_MAP = {
    "completed": "completed",
    "partial": "weak",
    "weak": "weak",
    "missing": "missing",
    "not_applicable": "not_applicable",
}
# 引擎 issue_type → 契约 module
_MODULE_MAP = {
    "audience": "audience_match",
    "profile": "requirement_coverage",
    "hec": "hec_match",
    "slider": "slider_match",
}
_SEVERITY_MAP = {"P0": "high", "P1": "medium", "P2": "low"}


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------
def _require(value: Any, field: str) -> Any:
    """必填字段守卫：None / 空串 → Crash Early。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ContractAssemblyError(f"契约必填字段缺失或为空，无可靠来源：{field}")
    return value


def _evidence(
    source: str,
    field: str,
    value: Any,
    segment_id: Optional[str] = None,
    confidence: Optional[float] = None,
) -> dict[str, Any]:
    """统一 evidence schema（契约第 14 节）。"""
    return {
        "source": source,
        "field": field,
        "value": value if value is None or isinstance(value, str) else str(value),
        "segment_id": segment_id,
        "confidence": confidence,
    }


def _split_terms(text: Optional[str]) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    parts: list[str] = [text]
    for sep in ("/", "、", "，", ",", "；", ";", "|"):
        parts = [p for chunk in parts for p in chunk.split(sep)]
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# diagnosis_meta（裁决 2：runner/wrapper 注入，必填缺失 Crash Early）
# ---------------------------------------------------------------------------
def build_diagnosis_meta(meta_input: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta_input, Mapping):
        raise ContractAssemblyError("diagnosis_meta 入参必须为 Mapping。")
    request_id = _require(meta_input.get("request_id"), "diagnosis_meta.request_id")
    video_id = _require(meta_input.get("video_id"), "diagnosis_meta.video_id")
    source_product_id = _require(
        meta_input.get("source_product_id"), "diagnosis_meta.source_product_id"
    )
    diagnosis_id = meta_input.get("diagnosis_id") or f"diag-{uuid.uuid4().hex[:16]}"
    created_at = meta_input.get("created_at") or datetime.now(timezone.utc).isoformat()
    return {
        "diagnosis_id": diagnosis_id,
        "request_id": request_id,
        "created_at": created_at,
        # 允许 null，但 key 必须存在
        "workflow_version": meta_input.get("workflow_version"),
        "model_version": meta_input.get("model_version"),
        "model_provider": meta_input.get("model_provider"),
        "source_product_id": source_product_id,
        "video_id": video_id,
        "qa_status": meta_input.get("qa_status") or "NOT_RUN",
        "e2e_status": meta_input.get("e2e_status") or "not_run",
    }


# ---------------------------------------------------------------------------
# product_fact_vector（商品事实向量，F2：6 维结构化枚举 + 可读 conversion_barriers）
# ---------------------------------------------------------------------------
def build_product_fact_vector(product_diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """F2：独立装配商品事实向量（6 维，枚举严格按 module3 PRD §5.2）。

    来源 ``resistance_profile``，归一到 PRD 中文枚举；口径复用 product_diagnoser 的
    ``_to_module3_category_attr / _to_module3_trust_attr / _to_module3_price_attr``，不新造枚举。
    - 必填维度（认知/频次/信任/价格）缺源数据一律 Crash Early。
    - 合法空值维度（背书/渠道风险）映射不到时输出 null，不伪造。

    6 维（字段名沿用 PRD 推荐结构）：
        cognition_attribute / frequency_attribute / trust_attribute /
        price_attribute / endorsement_attribute / channel_risk_attribute
    另附 ``conversion_barriers``（list[str] 可读解释层，F3，不替代结构化枚举）。
    """
    resistance = product_diagnosis.get("resistance_profile") or {}
    if not isinstance(resistance, Mapping) or not resistance:
        raise ContractAssemblyError("product_fact_vector 无可靠来源（resistance_profile 缺失）。")

    # 认知 cognition：蓝海 / 红海-{核心|破圈}（复用 _to_module3_category_attr 口径）
    ocean = str(resistance.get("ocean") or "").strip()
    competition_focus = str(resistance.get("competition_focus") or "").strip()
    if not ocean:
        raise ContractAssemblyError(
            "product_fact_vector.cognition_attribute 无可靠来源（resistance_profile.ocean 缺失）。"
        )
    if ocean == "蓝海":
        cognition = "蓝海"
    elif ocean == "红海":
        if not competition_focus:
            raise ContractAssemblyError(
                "product_fact_vector.cognition_attribute 缺源：红海须含 competition_focus（核心/破圈）。"
            )
        cognition = f"红海-{competition_focus}"
    else:
        raise ContractAssemblyError(
            f"product_fact_vector.cognition_attribute 源 ocean 非法枚举：{ocean!r}。"
        )
    if cognition not in _FACT_COGNITION_ENUM:
        raise ContractAssemblyError(
            f"product_fact_vector.cognition_attribute 归一结果越界：{cognition!r}（合法集={sorted(_FACT_COGNITION_ENUM)}）。"
        )

    # 频次 frequency：快消 / 耐消
    frequency = str(resistance.get("frequency") or "").strip()
    if frequency not in _FACT_FREQUENCY_ENUM:
        raise ContractAssemblyError(
            f"product_fact_vector.frequency_attribute 缺源或越界：{frequency!r}（合法集={sorted(_FACT_FREQUENCY_ENUM)}）。"
        )

    # 信任 trust：含「大牌」→大牌，否则白牌（复用 _to_module3_trust_attr 口径）
    brand_tier = str(resistance.get("brand_tier") or "").strip()
    if not brand_tier:
        raise ContractAssemblyError(
            "product_fact_vector.trust_attribute 无可靠来源（resistance_profile.brand_tier 缺失）。"
        )
    trust = "大牌" if "大牌" in brand_tier else "白牌"

    # 价格 price：高水位→高，否则→低（复用 _to_module3_price_attr 口径）
    relative_price = str(resistance.get("relative_price_level") or "").strip()
    if not relative_price:
        raise ContractAssemblyError(
            "product_fact_vector.price_attribute 无可靠来源（resistance_profile.relative_price_level 缺失）。"
        )
    price = "高" if relative_price == "高水位" else "低"

    # 背书 endorsement：有背书 / null（None / 空 → null，不伪造）
    endorsement_raw = resistance.get("endorsement")
    endorsement = "有背书" if (endorsement_raw and str(endorsement_raw).strip()) else None

    # 渠道风险 channel_risk：仅 有风险 / null（无风险及其它 → null）
    channel_risk_raw = str(resistance.get("channel_risk") or "").strip()
    channel_risk = "有风险" if channel_risk_raw == "有风险" else None

    fact_vector: dict[str, Any] = {
        "cognition_attribute": cognition,
        "frequency_attribute": frequency,
        "trust_attribute": trust,
        "price_attribute": price,
        "endorsement_attribute": endorsement,
        "channel_risk_attribute": channel_risk,
    }
    # F3：conversion_barriers 仅作可读解释层（list[str]），不替代结构化枚举、不复活 conversion_resistance。
    fact_vector["conversion_barriers"] = _build_conversion_barriers(fact_vector)
    return fact_vector


def _build_conversion_barriers(fact_vector: Mapping[str, Any]) -> list[str]:
    """基于商品事实向量生成可读的「转化卡点」中文解释（解释层，非结构化枚举）。"""
    barriers: list[str] = []
    if fact_vector["trust_attribute"] == "白牌":
        barriers.append("白牌信任存量低，需先建立来源可信/使用证据再促成交。")
    if fact_vector["price_attribute"] == "高":
        barriers.append("价格水位偏高，需强化价值证明与算账说服以化解价格顾虑。")
    if fact_vector["channel_risk_attribute"] == "有风险":
        barriers.append("存在渠道风险，需补充货源可信与防伪/风险消除信息。")
    if fact_vector["cognition_attribute"] == "蓝海":
        barriers.append("蓝海新品类认知空白，需先建立任务意识与新解法正当性。")
    elif fact_vector["cognition_attribute"] == "红海-破圈":
        barriers.append("红海破圈替换，需剥离旧方案使用惯性并证明切换更值。")
    if fact_vector["frequency_attribute"] == "耐消":
        barriers.append("耐消决策链长，需通过参数/适配/避坑提供选型确定性。")
    return barriers


# ---------------------------------------------------------------------------
# product_hec 三元组（F7：code/name/definition；definition 必来自后端权威字典）
# ---------------------------------------------------------------------------
def _hec_name_from_label(raw_label: Any, code: str) -> str:
    """从 ``*_label``（如 "H1 痛点/焦虑直击"）剥离 code 前缀得到业务名称；缺失返回空串。"""
    if not isinstance(raw_label, str) or not raw_label.strip():
        return ""
    parts = raw_label.strip().split(None, 1)
    if parts and parts[0].strip().upper() == code and len(parts) > 1:
        return parts[1].strip()
    # label 不含 code 前缀（或整串即名称）时原样返回
    return raw_label.strip()


def _build_hec_triple(raw_tag: Any, raw_label: Any, dim: str, idx: int) -> dict[str, str]:
    """把单维裸 tag 升级为 {code, name, definition} 三元组（F7）。

    - code：取 tag 的标签代号（兼容 "H1" 纯码或 "H1 痛点/焦虑直击" 整串，统一取首 token 大写）。
    - name：优先用 ``*_label`` 剥离 code 前缀；label 缺失再回退权威字典 name。
    - definition：**必来自后端权威字典（taxonomy_dictionary_v2.md）**；查不到 code → Crash Early。
    """
    raw = _require(raw_tag, f"product_hec[{idx}].{dim}_tag")
    code = str(raw).strip().split()[0].strip().upper()
    if not code:
        raise ContractAssemblyError(f"product_hec[{idx}].{dim}_tag 无法解析 code：{raw!r}。")
    # 权威字典回查（lookup_hec 内部未命中即 Crash Early，禁止编造 definition）
    entry = lookup_hec(code)
    name = _hec_name_from_label(raw_label, code) or entry["name"]
    definition = entry["definition"]
    if not (isinstance(name, str) and name.strip()):
        raise ContractAssemblyError(f"product_hec[{idx}].{dim}.name 无可靠来源（label/字典均缺）。")
    if not (isinstance(definition, str) and definition.strip()):
        # 理论上 lookup_hec 已保证非空；此处再次守卫，禁止空串占位
        raise ContractAssemblyError(f"product_hec[{idx}].{dim}.definition 字典缺失（禁止占位）。")
    return {"code": code, "name": name, "definition": definition}


# ---------------------------------------------------------------------------
# product_understanding（商品侧应然，只读商品诊断产物）
# ---------------------------------------------------------------------------
def build_product_understanding(product_diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """F1：商品理解 6 段固定输出（键集合与顺序固定）：

        basic_info → product_fact_vector → module3 → candidate_set → product_hec → evidence

    第二批结构治理要点：
    - 原 ``target_people``（模块 1 人群线索）迁入 ``basic_info.audience_hint``，不再作顶层段。
    - 原 ``core_selling_points`` 收敛进 ``basic_info``（PRD §5.1 商品基础信息含差异化卖点）。
    - 新增独立 ``product_fact_vector``（见 build_product_fact_vector）。
    - ``module3`` 透传第一批引擎产出的 persuasion_requirement_profile + product_target_audience。
    - ``expected_hec`` → 更名 ``product_hec``；第三批每维裸 tag 升级为 {code,name,definition} 三元组
      （definition 必来自后端权威字典 taxonomy_dictionary_v2.md，前端纯消费，不复活 expected_hec）。
    - ``candidate_set`` 第三批补 ``derived_from``（requirement_ids ⊆ profile / audience_groups ⊆ primary_audiences）。
    - 移除顶层 ``jtbd`` / ``supporting_requirements`` / ``expected_hec`` / ``conversion_resistance`` /
      ``target_people``；其下游均改读 raw product_diagnosis，不受影响。
    """
    pd = product_diagnosis
    inp = ((pd.get("evidence") or {}).get("input")) or {}
    core_intent = pd.get("core_intent") or {}
    profile = pd.get("persuasion_requirement_profile") or {}
    pta = pd.get("product_target_audience") or {}
    product_hecs = pd.get("product_hecs") or []
    assembly_status = pd.get("assembly_status")
    is_out_of_scope_for_mvp = (
        isinstance(assembly_status, Mapping)
        and assembly_status.get("status") == "out_of_scope_for_mvp"
    )

    # --- 1. basic_info（含 audience_hint：原模块 1 target_people 线索；core_selling_points）---
    product_name = _require(pd.get("product_name"), "product_understanding.basic_info.product_name")
    leaf_category = _require(pd.get("leaf_category"), "product_understanding.basic_info.leaf_category")
    audience_hint = _split_terms(inp.get("target_people"))
    if not audience_hint:
        raise ContractAssemblyError(
            "product_understanding.basic_info.audience_hint 无可靠来源（evidence.input.target_people 缺失）。"
        )
    core_selling_points = _split_terms(inp.get("core_selling_point"))
    if not core_selling_points:
        raise ContractAssemblyError(
            "product_understanding.basic_info.core_selling_points 无可靠来源（evidence.input.core_selling_point 缺失）。"
        )
    basic_info = {
        "product_name": product_name,
        "leaf_category": leaf_category,
        "brand_name": inp.get("brand_name") or pd.get("brand_name"),
        "shop_name": pd.get("shop_name"),
        "price": pd.get("price"),
        "audience_hint": audience_hint,
        "core_selling_points": core_selling_points,
    }

    # --- 2. product_fact_vector（F2：独立 6 维事实向量）---
    product_fact_vector = build_product_fact_vector(pd)

    # --- 3. module3（透传第一批引擎产出；非 scope-gate 状态下保持 Crash Early）---
    if not is_out_of_scope_for_mvp:
        if not isinstance(profile, Mapping) or not profile or not profile.get("persuasion_requirements"):
            raise ContractAssemblyError(
                "product_understanding.module3.persuasion_requirement_profile 为空或缺 persuasion_requirements（Crash Early）。"
            )
        if not isinstance(pta, Mapping) or not pta:
            raise ContractAssemblyError(
                "product_understanding.module3.product_target_audience 为空（Crash Early）。"
            )
    module3 = {
        "persuasion_requirement_profile": dict(profile) if isinstance(profile, Mapping) else {},
        "product_target_audience": dict(pta) if isinstance(pta, Mapping) else {},
    }

    # --- 4. candidate_set（沿用 core_intent 来源；F6 补 derived_from 可追溯）---
    candidate_set = {
        "candidate_h": core_intent.get("candidate_h") or [],
        "core_e": core_intent.get("core_e") or [],
        "core_c": core_intent.get("core_c") or [],
        "primary_effect": core_intent.get("primary_effect"),
        "primary_cta": core_intent.get("primary_cta"),
    }
    # F6：derived_from —— requirement_ids 来源 profile（真实 id），audience_groups 来源 primary_audiences
    reqs = []
    if isinstance(profile, Mapping):
        reqs = [r for r in (profile.get("persuasion_requirements") or []) if isinstance(r, Mapping)]
    requirement_ids = [
        r.get("requirement_id")
        for r in reqs
        if r.get("requirement_id") and (bool(r.get("required")) or r.get("priority") == "high")
    ]
    if not requirement_ids:
        # 无 required/high 命中则回退全部真实 requirement_id（仍来源 profile，不另造）
        requirement_ids = [r.get("requirement_id") for r in reqs if r.get("requirement_id")]

    audience_groups: list[str] = []
    if isinstance(pta, Mapping):
        audience_groups = [
            a.get("audience_group")
            for a in (pta.get("primary_audiences") or [])
            if isinstance(a, Mapping) and a.get("audience_group")
        ]

    if not is_out_of_scope_for_mvp:
        if not requirement_ids:
            raise ContractAssemblyError(
                "candidate_set.derived_from.requirement_ids 为空（profile 已保证非空，Crash Early）。"
            )
        if not audience_groups:
            raise ContractAssemblyError(
                "candidate_set.derived_from.audience_groups 为空（primary_audiences 已保证非空，Crash Early）。"
            )

    candidate_set["derived_from"] = {
        "status": "out_of_scope_for_mvp" if is_out_of_scope_for_mvp else "ok",
        "requirement_ids": requirement_ids,
        "audience_groups": audience_groups,
    }

    # --- 5. product_hec（F7：每维裸 tag → {code,name,definition} 三元组；definition 必来自字典）---
    if is_out_of_scope_for_mvp:
        # PRD-0：scope gate 命中时允许 Product_HECs 为空；装配层必须返回「状态化空结构」而非 Crash。
        product_hec = [
            {
                "status": "out_of_scope_for_mvp",
                "reason": "当前商品命中 MVP 暂不支持的场景，未生成推荐 HEC。",
                "hook": None,
                "effect": None,
                "cta": None,
                "code_name_definition_ready": False,
            }
        ]
    else:
        if not product_hecs or not isinstance(product_hecs[0], Mapping):
            raise ContractAssemblyError("product_understanding.product_hec 无可靠来源（product_hecs 为空）。")
        product_hec: list[dict[str, Any]] = []
        for idx, hec in enumerate(product_hecs):
            if not isinstance(hec, Mapping):
                raise ContractAssemblyError(f"product_understanding.product_hec[{idx}] 非法（非对象）。")
            product_hec.append(
                {
                    "variant_id": hec.get("variant_id"),
                    "hook": _build_hec_triple(hec.get("hook_tag"), hec.get("hook_label"), "hook", idx),
                    "effect": _build_hec_triple(hec.get("effect_tag"), hec.get("effect_label"), "effect", idx),
                    "cta": _build_hec_triple(hec.get("cta_tag"), hec.get("cta_label"), "cta", idx),
                }
            )

    # --- 6. evidence（统一 schema；标注 product_fact_vector / product_hec 来源）---
    primary_hec = product_hec[0]
    evidence = [
        _evidence("product_factpack", "basic_info.product_name", product_name),
        _evidence("product_factpack", "basic_info.leaf_category", leaf_category),
        _evidence(
            "product_understanding",
            "product_fact_vector.cognition_attribute",
            product_fact_vector["cognition_attribute"],
        ),
        _evidence(
            "product_understanding",
            "product_hec",
            (
                "out_of_scope_for_mvp"
                if is_out_of_scope_for_mvp
                else f"{primary_hec['hook']['code']}/{primary_hec['effect']['code']}/{primary_hec['cta']['code']}"
            ),
        ),
    ]

    return {
        "basic_info": basic_info,
        "product_fact_vector": product_fact_vector,
        "module3": module3,
        "candidate_set": candidate_set,
        "product_hec": product_hec,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# video_understanding（视频侧实然，只读视频事实 / 视频理解产物）
# ---------------------------------------------------------------------------
def build_video_understanding(
    video_understanding_input: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    vu = video_understanding_input
    segments = [s for s in (vu.get("storyboard_segments") or []) if isinstance(s, Mapping)]
    if not segments:
        raise ContractAssemblyError("video_understanding 无可靠分镜来源（storyboard_segments 为空）。")

    # --- video_meta（runben 样本无 platform/url/duration → null，key 必须存在）---
    video_meta = {
        "video_id": _require(vu.get("video_id"), "video_understanding.video_meta.video_id"),
        "source_platform": vu.get("source_platform"),
        "source_url": vu.get("source_url"),
        "duration_sec": vu.get("duration_sec"),
    }

    # --- 分镜衍生：visual / asr / ocr ---
    visual_segments: list[dict[str, Any]] = []
    asr_segments: list[dict[str, Any]] = []
    ocr_texts: list[dict[str, Any]] = []
    rhythm_change_points: list[dict[str, Any]] = []
    key_evidence_actions: list[dict[str, Any]] = []
    visual_subjects: list[str] = []
    asr_parts: list[str] = []
    ocr_parts: list[str] = []

    for seg in segments:
        seg_id = _require(seg.get("segment_id"), "visual_segments[].segment_id")
        start_sec = seg.get("start_sec")
        end_sec = seg.get("end_sec")
        vf = seg.get("visual_facts") or {}
        rf = seg.get("rhythm_facts") or {}
        core_scene_desc = _require(vf.get("visual_subject"), f"visual_segments[{seg_id}].core_scene_desc")
        actions = [a for a in (vf.get("actions") or []) if isinstance(a, Mapping)]
        core_action = "、".join(str(a.get("action_name") or "").strip() for a in actions if a.get("action_name"))
        if not core_action:
            raise ContractAssemblyError(f"visual_segments[{seg_id}].core_action 无可靠来源（visual_facts.actions 缺失）。")
        evidence_role = seg.get("evidence_role") or _role_to_evidence_role(seg.get("role"))
        is_change = bool(rf.get("is_rhythm_change_point"))
        change_reason = rf.get("rhythm_change_reason")
        ocr_text = seg.get("ocr")

        visual_segments.append(
            {
                "segment_id": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "core_scene_desc": core_scene_desc,
                "core_action": core_action,
                "related_asr_segment_id": seg_id,  # ASR 与画面同段，segment_id 对齐
                "related_ocr_texts": [ocr_text] if ocr_text else [],
                "is_rhythm_change_point": is_change,
                "rhythm_change_reason": change_reason,
                "evidence_role": evidence_role,
            }
        )
        visual_subjects.append(core_scene_desc)

        asr_text = seg.get("asr")
        asr_segments.append(
            {
                "segment_id": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": asr_text or "",
                "text_summary": None,
                "confidence": None,
            }
        )
        if asr_text:
            asr_parts.append(asr_text)
        if ocr_text:
            ocr_texts.append({"segment_id": seg_id, "start_sec": start_sec, "end_sec": end_sec, "text": ocr_text})
            ocr_parts.append(ocr_text)

        if is_change:
            rhythm_change_points.append({"time_sec": start_sec, "desc": change_reason or "节奏变化点"})
        time_range = f"{start_sec}-{end_sec}s" if start_sec is not None and end_sec is not None else None
        key_evidence_actions.append({"segment_id": seg_id, "time_range": time_range, "action_desc": core_action})

    text_stream = {
        "asr_summary": "；".join(asr_parts) if asr_parts else "",
        "asr_segments": asr_segments,
        "ocr_text": " / ".join(ocr_parts) if ocr_parts else "",
        "ocr_texts": ocr_texts,
    }
    visual_stream = {
        "visual_summary": "；".join(visual_subjects) if visual_subjects else "",
        "visual_segments": visual_segments,
    }

    import json as _json

    video_base_fact = {
        "total_segment_count": len(segments),
        "rhythm_change_points": rhythm_change_points,
        "key_evidence_actions": key_evidence_actions,
        "original_fact_record": _json.dumps(segments, ensure_ascii=False),
    }

    # --- video_jtbd（视频实然侧 JTBD，源视频理解 video_jtbd；缺失 Crash Early）---
    video_jtbd_src = vu.get("video_jtbd")
    if not isinstance(video_jtbd_src, Mapping) or not video_jtbd_src.get("primary_task"):
        raise ContractAssemblyError(
            "video_understanding.video_jtbd.primary_task 无可靠来源（视频理解未输出 video_jtbd）。"
        )
    video_jtbd = {
        "primary_task": video_jtbd_src.get("primary_task"),
        "reasoning": video_jtbd_src.get("reasoning"),
        "evidence": video_jtbd_src.get("evidence") or [],
    }

    # --- actual_hec（视频实然 HEC，源引擎归一化的 actual_video_hec + primary_hec.signature）---
    actual_video_hec = (raw_result.get("hec_match_diagnosis") or {}).get("actual_video_hec") or {}
    primary_hec = vu.get("primary_hec") or {}
    actual_hec = {
        "hook_tag": _require(actual_video_hec.get("hook_tag"), "video_understanding.actual_hec.hook_tag"),
        "effect_tag": _require(actual_video_hec.get("effect_tag"), "video_understanding.actual_hec.effect_tag"),
        "cta_tag": _require(actual_video_hec.get("cta_tag"), "video_understanding.actual_hec.cta_tag"),
        "reason": primary_hec.get("signature"),
    }

    # --- slider_signature（视频实然四轴 + summary）---
    slider_src = vu.get("slider_signature") or {}
    slider_signature = {
        "visual": _slider_score(slider_src.get("visual")),
        "audio": _slider_score(slider_src.get("audio")),
        "proof": _slider_score(slider_src.get("proof")),
        "cta": _slider_score(slider_src.get("cta")),
        "summary": None,
    }
    if all(slider_signature[k] is not None for k in ("visual", "audio", "proof", "cta")):
        slider_signature["summary"] = (
            f"视觉{slider_signature['visual']}/音频{slider_signature['audio']}/"
            f"证明{slider_signature['proof']}/CTA{slider_signature['cta']}"
        )

    # --- evidence_spans（视频侧证据，统一 schema）---
    evidence_spans = []
    for span in vu.get("evidence_spans") or []:
        if not isinstance(span, Mapping):
            continue
        evidence_spans.append(
            _evidence("video_factpack", f"evidence_span:{span.get('span_id')}", span.get("text"))
        )

    return {
        "video_meta": video_meta,
        "text_stream": text_stream,
        "visual_stream": visual_stream,
        "video_base_fact": video_base_fact,
        "video_jtbd": video_jtbd,
        "actual_hec": actual_hec,
        "slider_signature": slider_signature,
        "evidence_spans": evidence_spans,
    }


def _slider_score(axis_value: Any) -> Any:
    if isinstance(axis_value, Mapping):
        return axis_value.get("score")
    if isinstance(axis_value, (int, float)):
        return axis_value
    return None


def _role_to_evidence_role(role: Any) -> str:
    s = str(role or "").lower()
    if "hook" in s:
        return "hook"
    if "cta" in s or "close" in s:
        return "cta"
    if "effect" in s or "proof" in s:
        return "proof"
    if "transition" in s:
        return "transition"
    return "other"


# ---------------------------------------------------------------------------
# diagnosis.profile_match（提升引擎 profile_match_diagnosis 的前端字段）
# ---------------------------------------------------------------------------
def _normalize_evidence(raw_evidence: Any) -> list[dict[str, Any]]:
    """把引擎产出的 evidence 列表归一化为契约统一 evidence schema（补齐 segment_id/confidence key）。"""
    normalized: list[dict[str, Any]] = []
    for ev in raw_evidence or []:
        if not isinstance(ev, Mapping):
            continue
        normalized.append(
            {
                "source": ev.get("source"),
                "field": ev.get("field"),
                "value": ev.get("value"),
                "segment_id": ev.get("segment_id"),
                "confidence": ev.get("confidence"),
            }
        )
    return normalized


def build_profile_match(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    pm = raw_result.get("profile_match_diagnosis") or {}
    keys = ("status", "product_audience", "video_audience", "gap", "match_result", "evidence", "summary")
    missing = [k for k in keys if k not in pm]
    if missing:
        raise ContractAssemblyError(f"diagnosis.profile_match 缺少前端字段：{missing}（引擎 profile_match_diagnosis 未对齐契约）。")
    out = {k: pm[k] for k in keys}
    out["evidence"] = _normalize_evidence(pm.get("evidence"))
    return out


# ---------------------------------------------------------------------------
# diagnosis.requirement_coverage（源 profile_match_diagnosis.requirement_results）
# ---------------------------------------------------------------------------
def build_requirement_coverage(
    raw_result: Mapping[str, Any],
    product_diagnosis: Mapping[str, Any],
    video_understanding_input: Mapping[str, Any],
) -> dict[str, Any]:
    pm = raw_result.get("profile_match_diagnosis") or {}
    results = pm.get("requirement_results") or []
    profile = product_diagnosis.get("persuasion_requirement_profile") or {}
    req_map = {
        r.get("requirement_id"): r
        for r in (profile.get("persuasion_requirements") or [])
        if isinstance(r, Mapping)
    }
    span_text = {
        s.get("span_id"): s.get("text")
        for s in (video_understanding_input.get("evidence_spans") or [])
        if isinstance(s, Mapping)
    }

    items: list[dict[str, Any]] = []
    completed_count = 0
    for r in results:
        if not isinstance(r, Mapping):
            continue
        rid = r.get("requirement_id")
        raw_status = str(r.get("completion_status") or "")
        status = _COMPLETION_STATUS_MAP.get(raw_status, "missing")
        expected = (req_map.get(rid) or {}).get("success_criteria") or ""
        actual = r.get("judgment") or ""

        # 统一 evidence：completed 必须双侧覆盖（product + video）
        matched_spans = r.get("matched_evidence_spans") or []
        ev: list[dict[str, Any]] = []
        if expected:
            ev.append(_evidence("product_understanding", "persuasion_requirements.success_criteria", expected))
        if matched_spans:
            for sid in matched_spans:
                ev.append(_evidence("video_factpack", f"evidence_span:{sid}", span_text.get(sid) or sid, segment_id=sid))
        elif actual:
            ev.append(_evidence("video_factpack", "profile_match.judgment", actual))

        if status == "completed":
            completed_count += 1
            missing_reason = ""
        else:
            missing_reason = r.get("repair_direction") or actual or f"要求 {rid} 未完成（{raw_status}）。"

        items.append(
            {
                "requirement_id": rid,
                "requirement_name": r.get("requirement_name") or (req_map.get(rid) or {}).get("requirement_name") or "",
                "required": bool(r.get("required")),
                "completion_status": status,
                "expected": expected,
                "actual": actual,
                "matched_evidence_spans": ev,
                "missing_reason": missing_reason,
                "repair_direction": r.get("repair_direction") or "",
            }
        )

    total_count = len(items)
    missing_required = list(pm.get("missing_required_requirements") or [])
    weak_requirements = [it["requirement_id"] for it in items if it["completion_status"] == "weak"]

    if total_count == 0:
        status = "data_missing"
    elif missing_required:
        status = "failed"
    elif completed_count == total_count:
        status = "completed"
    else:
        status = "partial"

    summary = pm.get("information_miss_summary") or f"{completed_count}/{total_count} 项说服要求完成。"

    return {
        "status": status,
        "completed_count": completed_count,
        "total_count": total_count,
        "items": items,
        "missing_required_requirements": missing_required,
        "weak_requirements": weak_requirements,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# diagnosis.hec_match（布尔 hit 重组为 dimension_results）
# ---------------------------------------------------------------------------
def build_hec_match(raw_result: Mapping[str, Any], product_diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    hm = raw_result.get("hec_match_diagnosis") or {}
    if hm.get("diagnosis_aborted"):
        return {
            "status": "data_missing",
            "product_expected": {"hook_tag": "", "effect_tag": "", "cta_tag": ""},
            "video_actual": {"hook_tag": "", "effect_tag": "", "cta_tag": ""},
            "dimension_results": [],
            "acceptable_deviation_reason": "",
            "hec_gap_summary": "HEC Match 中止（商品任务待补充）。",
        }

    status = _HEC_STATUS_MAP.get(str(hm.get("match_status") or ""), "data_missing")
    product_hecs = product_diagnosis.get("product_hecs") or []
    hec0 = product_hecs[0] if product_hecs and isinstance(product_hecs[0], Mapping) else {}
    product_expected = {
        "hook_tag": _require(hec0.get("hook_tag"), "hec_match.product_expected.hook_tag"),
        "effect_tag": _require(hec0.get("effect_tag"), "hec_match.product_expected.effect_tag"),
        "cta_tag": _require(hec0.get("cta_tag"), "hec_match.product_expected.cta_tag"),
    }
    actual = hm.get("actual_video_hec") or {}
    video_actual = {
        "hook_tag": _require(actual.get("hook_tag"), "hec_match.video_actual.hook_tag"),
        "effect_tag": _require(actual.get("effect_tag"), "hec_match.video_actual.effect_tag"),
        "cta_tag": _require(actual.get("cta_tag"), "hec_match.video_actual.cta_tag"),
    }

    full_hit = bool(hm.get("full_combination_hit"))
    axis_defs = (
        ("hook", "hook_tag", "hook_hit"),
        ("effect", "effect_tag", "effect_hit"),
        ("cta", "cta_tag", "cta_hit"),
    )
    dimension_results: list[dict[str, Any]] = []
    for dim, tag_key, _hit_key in axis_defs:
        exp = product_expected[tag_key]
        act = video_actual[tag_key]
        if exp == act:
            dim_status = "matched"
            impact = f"{dim} 维度商品应然与视频实然一致（{exp}）。"
            suggestion = ""
        elif dim == "effect":
            dim_status = "mismatch"
            impact = f"effect（说服核心）偏移：应然 {exp} → 实然 {act}，影响核心说服闭环。"
            suggestion = f"对齐 effect 标签至 {exp}。"
        else:
            dim_status = "acceptable_deviation"
            impact = f"{dim} 偏移：应然 {exp} → 实然 {act}（effect 一致时属可接受偏差）。"
            suggestion = f"如需精准对齐，可将 {dim} 标签调整为 {exp}。"
        dimension_results.append(
            {
                "dimension": dim,
                "expected": exp,
                "actual": act,
                "status": dim_status,
                "impact": impact,
                "suggestion": suggestion,
            }
        )
    # chain 维度：整组组合一致性
    chain_status = "matched" if full_hit else status
    dimension_results.append(
        {
            "dimension": "chain",
            "expected": f"{product_expected['hook_tag']}/{product_expected['effect_tag']}/{product_expected['cta_tag']}",
            "actual": f"{video_actual['hook_tag']}/{video_actual['effect_tag']}/{video_actual['cta_tag']}",
            "status": chain_status if chain_status in {"matched", "acceptable_deviation", "weak_match", "mismatch"} else "mismatch",
            "impact": hm.get("logic_chain_judgment") or "",
            "suggestion": "" if full_hit else "对齐三轴组合至商品候选 HEC。",
        }
    )

    acceptable_deviation_reason = ""
    if status == "acceptable_deviation":
        acceptable_deviation_reason = (
            "effect（说服核心）命中候选集合，hook/cta 偏移属可接受偏差；建议结合业务判断是否需要精准对齐。"
        )

    return {
        "status": status,
        "product_expected": product_expected,
        "video_actual": video_actual,
        "dimension_results": dimension_results,
        "acceptable_deviation_reason": acceptable_deviation_reason,
        "hec_gap_summary": hm.get("hec_gap_summary") or "",
    }


# ---------------------------------------------------------------------------
# diagnosis.slider_match（axis_results 补 expected/actual/evidence）
# ---------------------------------------------------------------------------
def build_slider_match(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    sm = raw_result.get("slider_match_diagnosis") or {}
    status = _SLIDER_STATUS_MAP.get(str(sm.get("match_status") or ""), "data_missing")
    expected_pref = sm.get("expected_slider_preference") or {}
    actual_sig = sm.get("actual_slider_signature") or {}

    def _pref_str(axis: str) -> Any:
        v = expected_pref.get(axis)
        if isinstance(v, Mapping) and "min" in v and "max" in v:
            return f"{v['min']}-{v['max']}"
        return v

    expected_slider_preference = {axis: _pref_str(axis) for axis in ("visual", "audio", "proof", "cta")}
    actual_slider_signature = {axis: actual_sig.get(axis) for axis in ("visual", "audio", "proof", "cta")}

    axis_results: list[dict[str, Any]] = []
    for ar in sm.get("axis_results") or []:
        if not isinstance(ar, Mapping):
            continue
        axis = ar.get("axis")
        expected = _pref_str(axis)
        actual = actual_sig.get(axis)
        axis_results.append(
            {
                "axis": axis,
                "fit_status": ar.get("fit_status"),
                "expected": str(expected) if expected is not None else "",
                "actual": str(actual) if actual is not None else "",
                "judgment": ar.get("judgment") or "",
                "repair_direction": ar.get("repair_direction") or "",
                "evidence": [_evidence("video_factpack", f"slider_signature.{axis}", actual)],
            }
        )

    return {
        "status": status,
        "target_audience_reference": sm.get("target_audience_reference") or [],
        "expected_slider_preference": expected_slider_preference,
        "actual_slider_signature": actual_slider_signature,
        "axis_results": axis_results,
        "audience_acceptance_judgment": sm.get("audience_acceptance_judgment") or "",
        "slider_gap_summary": sm.get("slider_gap_summary") or "",
    }


# ---------------------------------------------------------------------------
# top_issues / suggestions（从 diagnosis_summary 重建，建立可回指 id）
# ---------------------------------------------------------------------------
def _resolve_related_requirement_id(issue_summary: str, product_diagnosis: Mapping[str, Any]) -> Optional[str]:
    profile = product_diagnosis.get("persuasion_requirement_profile") or {}
    for r in profile.get("persuasion_requirements") or []:
        rid = r.get("requirement_id") if isinstance(r, Mapping) else None
        if rid and rid in (issue_summary or ""):
            return rid
    return None


def build_top_issues_and_suggestions(
    raw_result: Mapping[str, Any],
    product_diagnosis: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = raw_result.get("diagnosis_summary") or {}
    repairs = summary.get("repair_suggestions") or []

    top_issues: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for idx, rs in enumerate(repairs, start=1):
        if not isinstance(rs, Mapping):
            continue
        issue_type = str(rs.get("issue_type") or "")
        module = _MODULE_MAP.get(issue_type, "profile_match")
        priority = str(rs.get("priority") or "P2")
        severity = _SEVERITY_MAP.get(priority, "low")
        issue_summary = rs.get("issue_summary") or ""
        repair_direction = rs.get("repair_direction") or ""
        issue_id = f"ISSUE-{idx:02d}"
        suggestion_id = f"SUGG-{idx:02d}"

        related_req = _resolve_related_requirement_id(issue_summary, product_diagnosis) if module == "requirement_coverage" else None

        issue_evidence = [
            _evidence("raw_output", f"diagnosis_summary.repair_suggestions[{idx - 1}].issue_summary", issue_summary)
        ]
        for sid in rs.get("related_evidence_spans") or []:
            issue_evidence.append(_evidence("video_factpack", f"evidence_span:{sid}", sid, segment_id=sid))

        title = issue_summary.split("：", 1)[0] if "：" in issue_summary else issue_summary[:30]
        top_issues.append(
            {
                "issue_id": issue_id,
                "severity": severity,
                "module": module,
                "title": title or f"{module} 问题",
                "description": issue_summary,
                "evidence": issue_evidence,
                "related_requirement_id": related_req,
            }
        )
        suggestions.append(
            {
                "suggestion_id": suggestion_id,
                "priority": priority,
                "module": module,
                "action": repair_direction or issue_summary,
                "reason": issue_summary,
                "expected_effect": f"修复后预计提升 {module} 的诊断结论（{severity} 级问题收敛）。",
                "related_issue_id": issue_id,
            }
        )
    return top_issues, suggestions


# ---------------------------------------------------------------------------
# diagnosis.overview（滚动汇总各模块状态）
# ---------------------------------------------------------------------------
def build_diagnosis_overview(
    raw_result: Mapping[str, Any],
    requirement_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    summary = raw_result.get("diagnosis_summary") or {}
    audience = raw_result.get("audience_match_diagnosis") or {}
    pm = raw_result.get("profile_match_diagnosis") or {}
    hm = raw_result.get("hec_match_diagnosis") or {}
    sm = raw_result.get("slider_match_diagnosis") or {}

    if summary.get("diagnosis_aborted"):
        overall_status = "needs_review"
    else:
        overall_status = _OVERALL_STATUS_MAP.get(str(summary.get("overall_status") or ""), "needs_review")

    audience_match_status = _AUDIENCE_MATCH_MAP.get(str(audience.get("match_status") or ""), "data_missing")
    profile_match_status = _PROFILE_LEGACY_TO_OVERVIEW.get(str(pm.get("match_status") or ""), "data_missing")
    if hm.get("diagnosis_aborted"):
        hec_match_status = "data_missing"
    else:
        hec_match_status = _HEC_STATUS_MAP.get(str(hm.get("match_status") or ""), "data_missing")
    slider_match_status = _SLIDER_STATUS_MAP.get(str(sm.get("match_status") or ""), "data_missing")

    rc_status = requirement_coverage.get("status") or "data_missing"
    rc_text = (
        f"{requirement_coverage.get('completed_count')}/{requirement_coverage.get('total_count')} completed"
    )

    findings = summary.get("key_findings") or []
    overview_summary = " ".join(findings) if findings else (summary.get("overall_status") or "诊断完成。")

    return {
        "overall_status": overall_status,
        "audience_match_status": audience_match_status,
        "profile_match_status": profile_match_status,
        "hec_match_status": hec_match_status,
        "slider_match_status": slider_match_status,
        "requirement_coverage_status": rc_status,
        "requirement_coverage_text": rc_text,
        "summary": overview_summary,
    }


# ---------------------------------------------------------------------------
# 顶层 status（裁决 4：按契约枚举映射）
# ---------------------------------------------------------------------------
def resolve_status(raw_result: Mapping[str, Any]) -> str:
    pm = raw_result.get("profile_match_diagnosis") or {}
    summary = raw_result.get("diagnosis_summary") or {}
    if pm.get("task_status") == "needs_review" or summary.get("diagnosis_aborted"):
        return "needs_review"
    return "diagnosis_completed"


# ---------------------------------------------------------------------------
# artifacts（裁决 3：raw=原始 video_persuasion_diagnosis_result；normalized=契约 diagnosis）
# ---------------------------------------------------------------------------
def build_artifacts(
    request_payload: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    normalized_diagnosis: Mapping[str, Any],
    source_files: Sequence[str],
) -> dict[str, Any]:
    return {
        "request_payload": dict(request_payload),
        "raw_response": {"video_persuasion_diagnosis_result": dict(raw_result)},
        "normalized_response": dict(normalized_diagnosis),
        "source_files": list(source_files),
    }


# ---------------------------------------------------------------------------
# 顶层装配
# ---------------------------------------------------------------------------
def assemble_frontend_response(
    *,
    product_diagnosis: Mapping[str, Any],
    video_payload: Mapping[str, Any],
    raw_diagnosis_result: Mapping[str, Any],
    diagnosis_meta_input: Mapping[str, Any],
    source_files: Sequence[str],
) -> dict[str, Any]:
    """组装前端 5 Tab 可直接消费的契约响应对象。

    参数：
      product_diagnosis      —— 商品侧完整诊断（runben_full_diagnosis.json 全量 dict）。
      video_payload          —— VideoDiagnosisEngine 的入参 payload
                                （含 product_diagnosis + video_understanding）。
      raw_diagnosis_result   —— 引擎输出的 video_persuasion_diagnosis_result（原始 raw）。
      diagnosis_meta_input   —— runner/wrapper 注入的 meta（request_id/video_id/source_product_id 必填）。
      source_files           —— artifacts.source_files。
    """
    if not isinstance(raw_diagnosis_result, Mapping):
        raise ContractAssemblyError("raw_diagnosis_result 非法。")
    video_understanding_input = video_payload.get("video_understanding")
    if not isinstance(video_understanding_input, Mapping):
        raise ContractAssemblyError("video_payload.video_understanding 缺失或非法。")

    diagnosis_meta = build_diagnosis_meta(diagnosis_meta_input)
    product_understanding = build_product_understanding(product_diagnosis)
    video_understanding = build_video_understanding(video_understanding_input, raw_diagnosis_result)

    profile_match = build_profile_match(raw_diagnosis_result)
    requirement_coverage = build_requirement_coverage(
        raw_diagnosis_result, product_diagnosis, video_understanding_input
    )
    hec_match = build_hec_match(raw_diagnosis_result, product_diagnosis)
    slider_match = build_slider_match(raw_diagnosis_result)
    top_issues, suggestions = build_top_issues_and_suggestions(raw_diagnosis_result, product_diagnosis)
    overview = build_diagnosis_overview(raw_diagnosis_result, requirement_coverage)

    diagnosis = {
        "overview": overview,
        "profile_match": profile_match,
        "hec_match": hec_match,
        "slider_match": slider_match,
        "requirement_coverage": requirement_coverage,
        "top_issues": top_issues,
        "suggestions": suggestions,
    }

    status = resolve_status(raw_diagnosis_result)
    artifacts = build_artifacts(video_payload, raw_diagnosis_result, diagnosis, source_files)

    response = {
        "status": status,
        "diagnosis_meta": diagnosis_meta,
        "product_understanding": product_understanding,
        "video_understanding": video_understanding,
        "diagnosis": diagnosis,
        "artifacts": artifacts,
    }

    # 后置断言（Crash Early）：12 类契约对象 + 顶层校验，不通过则 raise。
    assert_frontend_contract_response(response)
    return response
