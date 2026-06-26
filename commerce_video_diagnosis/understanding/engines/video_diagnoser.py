"""视频说服诊断模块（Block 2）。

入口 `VideoDiagnosisEngine.diagnose(payload) -> dict`，返回
`{"video_persuasion_diagnosis_result": {...}}`，包含 Step0–Step6 七段输出。

设计原则（严格遵循 PRD / IMPL_BRIEF）：
- 纯函数式、确定性：`VideoDiagnosisEngine()` 无外部依赖；slider 偏好字典从文件加载，可注入。
- 所有断言用代码实现（Crash Early），不依赖 LLM 自觉。
- video_target_audience（Step1）独立判定，**不继承** product_target_audience，
  且**代码层不引用 slider_signature**（slider 仅在 Step5 使用）。
- 目标人群相关字段一律不使用 `segments` 命名。

Block 1.2（video_HEC 命名契约）决策记录：
  视频理解侧输出字段为 `primary_hec`，而诊断 PRD 的输入字段名为 `video_HEC`。
  结论：**不修改视频理解侧**。在本模块 Step0 输入层做映射——优先读取 `video_HEC`，
  缺失时回退读取 `primary_hec` 并归一化为 video_HEC，同时记一条 warning。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from commerce_video_diagnosis.understanding.engines.audience_taxonomy import (
    CONSUMPTION_LABELS,
    EIGHT_AUDIENCE_GROUPS,
    compose_audience_group,
    expand_axis,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SLIDER_DICT_PATH = _DATA_DIR / "audience_slider_preference_dictionary.json"

SLIDER_AXES = ("visual", "audio", "proof", "cta")


# =============================================================================
# 合法枚举集合（Crash Early 防止未来枚举漂移）
# =============================================================================
LEGAL_ENUMS: dict[str, frozenset[str]] = {
    "input_validation.status": frozenset({"passed", "failed"}),
    "audience_match_diagnosis.match_status": frozenset(
        {"high_match", "partial_match", "low_match", "too_broad"}
    ),
    "video_target_audience.fit_level": frozenset({"primary", "secondary", "mismatch_risk"}),
    "video_target_audience.age_axis": frozenset({"young", "mature", "mixed"}),
    "video_target_audience.gender_axis": frozenset({"female", "male", "mixed"}),
    "video_target_audience.consumption_power_axis": frozenset({"mid_high", "low", "mixed"}),
    "profile_match_diagnosis.match_status": frozenset(
        {"completed", "partial", "weak", "missing", "not_applicable"}
    ),
    "profile_match_diagnosis.completion_status": frozenset(
        {"completed", "partial", "weak", "missing", "not_applicable"}
    ),
    "hec_match_diagnosis.match_status": frozenset(
        {"good", "acceptable_deviation", "risky_deviation", "mismatch"}
    ),
    "slider_match_diagnosis.match_status": frozenset(
        {"fit", "mixed_deviation", "too_strong", "too_weak", "wrong_direction", "mismatch"}
    ),
    "slider_match_diagnosis.fit_status": frozenset(
        {"fit", "too_strong", "too_weak", "wrong_direction"}
    ),
    "diagnosis_summary.overall_status": frozenset(
        {"good", "needs_minor_repair", "needs_major_repair", "mismatch"}
    ),
    "repair_suggestions.priority": frozenset({"P0", "P1", "P2"}),
    "repair_suggestions.issue_type": frozenset({"audience", "profile", "hec", "slider"}),
}


class VideoDiagnosisEnumError(ValueError):
    """诊断结果枚举越界（Crash Early 防漂移）。"""


class VideoDiagnosisInputError(ValueError):
    """视频诊断输入校验失败（Crash Early）。"""


# =============================================================================
# Step1 video_target_audience 关键词表（视频侧独立判定，与商品侧解耦）
# =============================================================================
VIDEO_FEMALE_KEYWORDS: tuple[str, ...] = (
    "母婴", "婴", "宝宝", "儿童", "小朋友", "孩子", "孕", "妈妈", "家庭", "家用",
    "驱蚊", "防蚊", "蚊", "洗护", "护肤", "面膜", "美妆", "彩妆", "香氛", "女",
    "内衣", "卫生巾", "厨房", "家清", "辅食",
)
VIDEO_MALE_KEYWORDS: tuple[str, ...] = (
    "汽车", "车载", "机油", "工具", "五金", "户外", "钓鱼", "男士", "剃须",
    "电竞", "装备", "摩托",
)
VIDEO_MATURE_KEYWORDS: tuple[str, ...] = (
    "母婴", "宝宝", "儿童", "小朋友", "孩子", "家庭", "家用", "辅食", "老人",
    "养生", "保健", "厨房", "家清", "驱蚊", "防蚊", "照护",
)
VIDEO_YOUNG_KEYWORDS: tuple[str, ...] = (
    "潮玩", "盲盒", "零食", "球鞋", "电竞", "新奇", "学生", "ins", "网红", "二次元",
)
VIDEO_LOW_CONSUMPTION_KEYWORDS: tuple[str, ...] = (
    "便宜", "划算", "优惠", "平价", "性价比", "实惠", "9.9", "买一送", "加量",
    "薅羊毛", "白菜价", "清仓", "低价",
)
VIDEO_MID_HIGH_CONSUMPTION_KEYWORDS: tuple[str, ...] = (
    "官方", "旗舰", "正品", "品牌", "成分", "派卡瑞丁", "品质", "高端", "专利",
    "认证", "检测", "报告", "A级", "温和", "无刺激", "安全", "效果", "测评",
    "实测", "挑战", "无包", "种草", "口碑",
)

# Step1 信号采集用：hook / scene / persona / cta-benefit 触发词
VIDEO_HOOK_KEYWORDS: tuple[str, ...] = (
    "挑战", "痛点", "焦虑", "崩溃", "实测", "测评", "对比", "揭秘", "震惊", "竟然",
    "蚊", "叮", "咬", "包",
)
VIDEO_SCENE_KEYWORDS: tuple[str, ...] = (
    "家用", "家庭", "户外", "居家", "出门", "便携", "夜晚", "睡觉", "露营", "校园",
    "办公室", "厨房", "卧室",
)

# Step3 说服要求 → 视频文本检索关键词（确定性映射，覆盖 12 个 requirement_id）
REQUIREMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "expose_current_pain": ("痛点", "蚊", "叮", "咬", "包", "瘙痒", "红肿", "挑战", "困扰"),
    "prove_user_fit": ("小朋友", "儿童", "宝宝", "孩子", "家庭", "适合", "也能用", "人群"),
    "prove_scenario_fit": ("家用", "户外", "便携", "场景", "出门", "居家", "大瓶", "小瓶"),
    "prove_core_benefit": ("驱蚊", "防蚊", "有效", "无包", "防叮", "效果"),
    "provide_visible_result": ("分钟", "无包", "测评", "挑战", "只", "效果", "实测", "可见"),
    "establish_basic_trust": ("官方", "品牌", "正品", "旗舰", "大牌"),
    "reduce_trial_risk": ("无味道", "无味", "温和", "无刺激", "0刺激", "安全", "敏感肌"),
    "prove_source_credibility": ("检测", "报告", "成分", "派卡瑞丁", "认证", "审查号"),
    "provide_authority_endorsement": ("权威", "检测报告", "认证", "审查号", "专利", "背书"),
    "resolve_safety_risk": ("安全", "温和", "无刺激", "小朋友", "婴幼儿", "儿童", "也能用"),
    "prove_current_purchase_reason": ("现在", "立即", "优惠", "囤", "夏天", "蚊季", "当下"),
    "clarify_purchase_threshold": ("大瓶", "小瓶", "规格", "价格", "套装", "选购", "家用", "便携"),
}


# =============================================================================
# PRD-4：视频诊断 Profile Match / HEC Match 收窄用契约常量
# =============================================================================
# 上游契约：needs_review 为商品诊断 Stage B 空候选/事实不足的协议合法值（字符串），
# 不是第 12 个业务任务标签。诊断侧遇此值 Profile Match / HEC Match 中止。
NEEDS_REVIEW = "needs_review"

# PRD-4.1 Blocked Direction Check：GR-02/03/04 被阻断方向 → 对应 12 标签任务。
# 判断依据为商品诊断 non_selected_task_reasons 中 gate=blocked 且 reason 前缀为
# GR-02/GR-03/GR-04 的方向，这些上提方向不得计入有效说服覆盖。
GR_BLOCKED_DIRECTION_TASK: dict[str, str] = {
    "GR-02": "物理安全与风险规避",
    "GR-03": "情绪安心/主观降险",
    "GR-04": "自我犒赏与秩序掌控",
}

# PRD-4.1 Secondary Coverage / Blocked Direction Check：12 标签任务 → 视频文本claim
# 检索关键词（确定性映射，仅用于判断"视频是否把某任务方向当主诉/次级收益越位"）。
# 与 REQUIREMENT_KEYWORDS 同为判断层关键词表，不改变任何架构。
TASK_CLAIM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "生存/运转维系": ("维持运转", "持续使用", "日常运转", "维系", "保持运行", "正常运转", "续航"),
    "缺陷修复/冲突消除": ("修复", "解决", "消除", "去除", "搞定问题", "修补", "补救", "解决痛点"),
    "降本增效/懒人替代": ("省时", "省力", "高效", "效率", "懒人", "一步到位", "省事", "替代"),
    "物理安全与风险规避": ("物理安全", "防护", "防止伤害", "规避风险", "防伤害", "安全防护", "杜绝风险", "防护屏障"),
    "情绪安心/主观降险": ("安心", "放心", "省心", "踏实", "无忧", "心安", "缓解焦虑", "安全感"),
    "新奇探索/瞬时刺激": ("新奇", "猎奇", "刺激", "新鲜", "好玩", "惊喜", "尝鲜", "新体验"),
    "自我犒赏与秩序掌控": ("犒赏", "悦己", "宠爱自己", "奖励自己", "仪式感", "精致生活", "掌控感", "秩序感"),
    "照护与责任履行": ("照护", "守护", "呵护", "为家人", "为孩子", "尽责", "照顾", "关爱"),
    "礼赠与关系表达": ("送礼", "礼物", "礼盒", "伴手礼", "心意", "送给", "回礼"),
    "圈层认同（圈层归属/身份锚定）": ("圈层", "同款", "身份认同", "归属", "圈子", "潮流人群", "身份锚定"),
    "阶层与审美发信": ("品味", "格调", "审美", "高级感", "彰显身份", "质感", "档次"),
}


# =============================================================================
# slider 偏好字典 loader（Crash Early）
# =============================================================================
def load_audience_slider_preference_dictionary(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """加载八大人群 slider 偏好字典。文件缺失 / 人群缺失 / 轴缺失即 Crash Early。"""
    target = Path(path) if path is not None else DEFAULT_SLIDER_DICT_PATH
    if not target.exists():
        raise VideoDiagnosisInputError(f"audience_slider_preference_dictionary 文件缺失: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    preferences = data.get("preferences")
    if not isinstance(preferences, dict) or not preferences:
        raise VideoDiagnosisInputError("audience_slider_preference_dictionary.preferences 非法或为空。")
    axes = tuple(data.get("axes") or SLIDER_AXES)
    missing_groups = [g for g in EIGHT_AUDIENCE_GROUPS if g not in preferences]
    if missing_groups:
        raise VideoDiagnosisInputError(f"slider 偏好字典缺少人群: {missing_groups}")
    for group, axis_map in preferences.items():
        if not isinstance(axis_map, Mapping):
            raise VideoDiagnosisInputError(f"slider 偏好字典人群 {group} 结构非法。")
        for axis in axes:
            if axis not in axis_map:
                raise VideoDiagnosisInputError(f"slider 偏好字典人群 {group} 缺少轴 {axis}。")
            rng = axis_map[axis]
            if not isinstance(rng, Mapping) or "min" not in rng or "max" not in rng:
                raise VideoDiagnosisInputError(f"slider 偏好字典人群 {group} 轴 {axis} 缺少 min/max。")
    return data


# =============================================================================
# 文本抽取工具（segments / bundles / spans 均为普通 dict）
# =============================================================================
def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_stringify(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value)
    return str(value)


def _segment_role(segment: Mapping[str, Any]) -> str:
    for key in ("role", "segment_role", "bundle_role", "persuasion_function"):
        value = segment.get(key)
        if value:
            return str(value)
    return ""


def _segment_text(segment: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "asr", "ocr", "asr_text", "ocr_text", "text", "spoken_lines",
        "visual_description", "summary", "bundle_text", "evidence", "span_text",
    ):
        if key in segment and segment[key]:
            parts.append(_stringify(segment[key]))
    # 兜底：auditory_text / 嵌套结构
    if "auditory_text" in segment:
        parts.append(_stringify(segment["auditory_text"]))
    if not parts:
        parts.append(_stringify(dict(segment)))
    return " ".join(parts)


def _span_id(span: Mapping[str, Any], index: int) -> str:
    for key in ("span_id", "id", "evidence_id", "segment_id"):
        if span.get(key):
            return str(span[key])
    return f"span_{index}"


def _hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [kw for kw in keywords if kw in text]


class VideoDiagnosisEngine:
    """视频说服诊断引擎。确定性、可注入 slider 字典。"""

    def __init__(self, slider_dictionary: Optional[Mapping[str, Any]] = None) -> None:
        # 允许注入；不注入则在需要时从文件加载（Crash Early 在 loader 内实现）。
        self._slider_dictionary: Optional[Mapping[str, Any]] = slider_dictionary

    # ------------------------------------------------------------------ 入口
    def diagnose(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise VideoDiagnosisInputError("payload 必须是 Mapping。")
        product_diagnosis = payload.get("product_diagnosis")
        video_understanding = payload.get("video_understanding")
        if not isinstance(product_diagnosis, Mapping):
            raise VideoDiagnosisInputError("payload.product_diagnosis 缺失或非法。")
        if not isinstance(video_understanding, Mapping):
            raise VideoDiagnosisInputError("payload.video_understanding 缺失或非法。")

        # Step0：输入校验 + video_HEC 映射（Block 1.2）
        input_validation, video_hec = self._step0_input_validation(
            product_diagnosis, video_understanding
        )

        # 文本语料（供 Step1 / Step3 检索）
        corpus = self._build_corpus(video_understanding)

        # Step1：视频目标人群（独立判定，绝不读取 slider_signature）
        video_target_audience = self._step1_video_target_audience(video_understanding, corpus)

        # Step2：人群匹配
        audience_match = self._step2_audience_match(
            product_diagnosis.get("product_target_audience") or {}, video_target_audience
        )

        # A2：依据 Step2 结果回填 video_target_audience.mismatch_risk_audiences
        # （仅当 match_status 为 low_match / too_broad 时，把 unexpected_video_audiences
        #  作为错配风险补入；high_match / partial_match 不追加。须在最终输出装配前完成。）
        self._backfill_mismatch_risk(video_target_audience, audience_match)

        # PRD-4：商品主任务（jtbd_level1，兼容读取 jtbd）作为 Profile/HEC 收窄基准。
        jtbd_level1 = self._resolve_jtbd_level1(product_diagnosis)
        secondary_benefits = product_diagnosis.get("secondary_benefits") or []
        non_selected_task_reasons = product_diagnosis.get("non_selected_task_reasons") or []

        # PRD-4.2 needs_review 降级：诊断总入口遇 jtbd_level1 == needs_review 时
        # 不做完整诊断（Profile Match 与 HEC Match 均中止），标记"商品任务待补充"，
        # 无 HEC Match 结论。audience / slider 与 jtbd 无关，仍照常产出。
        if jtbd_level1 == NEEDS_REVIEW:
            slider_match = self._step5_slider_match(
                video_target_audience,
                product_diagnosis.get("product_target_audience") or {},
                video_understanding.get("slider_signature") or {},
            )
            result = {
                "video_persuasion_diagnosis_result": {
                    "input_validation": input_validation,
                    "video_target_audience": video_target_audience,
                    "audience_match_diagnosis": audience_match,
                    "profile_match_diagnosis": self._step3_profile_match(
                        product_diagnosis.get("persuasion_requirement_profile") or {},
                        corpus,
                        jtbd_level1=jtbd_level1,
                    ),
                    "hec_match_diagnosis": self._build_needs_review_hec_match(),
                    "slider_match_diagnosis": slider_match,
                    "diagnosis_summary": self._build_needs_review_summary(),
                }
            }
            body = result["video_persuasion_diagnosis_result"]
            # 后置断言（Crash Early）：needs_review 降级结构合法——Profile/HEC/Summary 均须中止。
            if body["profile_match_diagnosis"].get("task_status") != NEEDS_REVIEW:
                raise VideoDiagnosisInputError(
                    "needs_review 降级断言失败：profile_match_diagnosis 未标记 needs_review 中止态。"
                )
            if not body["hec_match_diagnosis"].get("diagnosis_aborted"):
                raise VideoDiagnosisInputError(
                    "needs_review 降级断言失败：hec_match_diagnosis 未中止（应无 HEC Match 结论）。"
                )
            if not body["diagnosis_summary"].get("diagnosis_aborted"):
                raise VideoDiagnosisInputError(
                    "needs_review 降级断言失败：diagnosis_summary 未标记'商品任务待补充'中止态。"
                )
            self._validate_enums(body)
            return result

        # Step3：说服要求匹配（PRD-4.1 收窄：Primary/Secondary/Blocked 三层）
        profile_match = self._step3_profile_match(
            product_diagnosis.get("persuasion_requirement_profile") or {},
            corpus,
            jtbd_level1=jtbd_level1,
            secondary_benefits=secondary_benefits,
            non_selected_task_reasons=non_selected_task_reasons,
        )

        # Step4：HEC 匹配（PRD-4.2 回查：EC Skeletons + CandidateSet 显式回查 source_role）
        hec_match = self._step4_hec_match(
            product_diagnosis.get("product_HEC") or {},
            video_hec,
            ec_skeletons=(
                product_diagnosis.get("Product_EC_Skeletons")
                or product_diagnosis.get("product_ec_skeletons")
            ),
            candidate_set=product_diagnosis.get("candidate_set"),
        )

        # Step5：slider 匹配
        slider_match = self._step5_slider_match(
            video_target_audience,
            product_diagnosis.get("product_target_audience") or {},
            video_understanding.get("slider_signature") or {},
        )

        # Step6：诊断总览
        diagnosis_summary = self._step6_summary(
            audience_match, profile_match, hec_match, slider_match
        )

        result = {
            "video_persuasion_diagnosis_result": {
                "input_validation": input_validation,
                "video_target_audience": video_target_audience,
                "audience_match_diagnosis": audience_match,
                "profile_match_diagnosis": profile_match,
                "hec_match_diagnosis": hec_match,
                "slider_match_diagnosis": slider_match,
                "diagnosis_summary": diagnosis_summary,
            }
        }
        # Crash Early：枚举边界守卫（防止未来漂移）
        self._validate_enums(result["video_persuasion_diagnosis_result"])
        return result

    # ------------------------------------------------------------------ PRD-4 辅助
    @staticmethod
    def _resolve_jtbd_level1(product_diagnosis: Mapping[str, Any]) -> Optional[str]:
        """读取商品诊断主任务（jtbd_level1）。上游契约：主任务字段为 jtbd（即 jtbd_level1），
        兼容读取 jtbd_level1 / jtbd（字符串或 {level1: ...} 结构）及 product_fact_vector 内字段。"""
        for key in ("jtbd_level1", "jtbd"):
            value = product_diagnosis.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, Mapping):
                level1 = value.get("level1") or value.get("jtbd_level1")
                if isinstance(level1, str) and level1.strip():
                    return level1.strip()
        pfv = product_diagnosis.get("product_fact_vector")
        if isinstance(pfv, Mapping):
            for key in ("jtbd_level1", "jtbd"):
                value = pfv.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _build_needs_review_hec_match() -> dict[str, Any]:
        """PRD-4.2：needs_review 降级——HEC Match 中止，无结论（match_status 置空）。"""
        return {
            "diagnosis_aborted": True,
            "abort_reason": "jtbd_level1=needs_review：商品任务待补充，HEC Match 中止，无 HEC Match 结论。",
            "match_status": None,
        }

    @staticmethod
    def _build_needs_review_summary() -> dict[str, Any]:
        """PRD-4.2：needs_review 降级——诊断总览中止，标记"商品任务待补充"。"""
        return {
            "diagnosis_aborted": True,
            "overall_status": None,
            "task_status": "商品任务待补充",
            "abort_reason": (
                "jtbd_level1=needs_review：商品任务待补充（Stage B 空候选/事实不足），"
                "Profile Match 与 HEC Match 均中止，无 HEC Match 结论。"
            ),
            "key_findings": ["商品任务待补充：商品诊断主任务为 needs_review，完整诊断中止。"],
            "priority_issues": [],
            "repair_suggestions": [],
        }

    # ------------------------------------------------------------------ Guard
    @staticmethod
    def _check_enum(field: str, value: Any) -> None:
        legal = LEGAL_ENUMS[field]
        if value not in legal:
            raise VideoDiagnosisEnumError(
                f"非法枚举值：{field}={value!r}，合法集合={sorted(legal)}"
            )

    @classmethod
    def _validate_enums(cls, body: Mapping[str, Any]) -> None:
        iv = body.get("input_validation") or {}
        cls._check_enum("input_validation.status", iv.get("status"))

        vta = body.get("video_target_audience") or {}
        for bucket in ("primary_audiences", "secondary_audiences", "mismatch_risk_audiences"):
            for item in (vta.get(bucket) or []):
                if isinstance(item, Mapping) and "fit_level" in item:
                    cls._check_enum("video_target_audience.fit_level", item.get("fit_level"))
        ax = vta.get("axis_judgment") or {}
        if "age_axis" in ax:
            cls._check_enum("video_target_audience.age_axis", (ax["age_axis"] or {}).get("value"))
        if "gender_axis" in ax:
            cls._check_enum("video_target_audience.gender_axis", (ax["gender_axis"] or {}).get("value"))
        if "consumption_power_axis" in ax:
            cls._check_enum(
                "video_target_audience.consumption_power_axis",
                (ax["consumption_power_axis"] or {}).get("value"),
            )

        am = body.get("audience_match_diagnosis") or {}
        cls._check_enum("audience_match_diagnosis.match_status", am.get("match_status"))

        pm = body.get("profile_match_diagnosis") or {}
        cls._check_enum("profile_match_diagnosis.match_status", pm.get("match_status"))
        for r in (pm.get("requirement_results") or []):
            cls._check_enum(
                "profile_match_diagnosis.completion_status", (r or {}).get("completion_status")
            )

        hm = body.get("hec_match_diagnosis") or {}
        # PRD-4.2：needs_review 降级时 HEC Match 中止、无结论，跳过枚举校验。
        if not hm.get("diagnosis_aborted"):
            cls._check_enum("hec_match_diagnosis.match_status", hm.get("match_status"))

        sm = body.get("slider_match_diagnosis") or {}
        cls._check_enum("slider_match_diagnosis.match_status", sm.get("match_status"))
        for ar in (sm.get("axis_results") or []):
            cls._check_enum("slider_match_diagnosis.fit_status", (ar or {}).get("fit_status"))

        ds = body.get("diagnosis_summary") or {}
        # PRD-4.2：needs_review 降级时诊断总览中止（无 overall_status），跳过枚举校验。
        if not ds.get("diagnosis_aborted"):
            cls._check_enum("diagnosis_summary.overall_status", ds.get("overall_status"))
            for s in (ds.get("repair_suggestions") or []):
                cls._check_enum("repair_suggestions.priority", (s or {}).get("priority"))
                cls._check_enum("repair_suggestions.issue_type", (s or {}).get("issue_type"))

    # ------------------------------------------------------------------ Step0
    def _step0_input_validation(
        self,
        product_diagnosis: Mapping[str, Any],
        video_understanding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        blocking_issues: list[str] = []
        warnings: list[str] = []

        # 1. source_product_id 一致
        p_pid = str(product_diagnosis.get("source_product_id") or "").strip()
        v_pid = str(video_understanding.get("source_product_id") or "").strip()
        if not p_pid or not v_pid:
            blocking_issues.append("source_product_id 缺失（product 或 video 侧）。")
        elif p_pid != v_pid:
            blocking_issues.append(
                f"source_product_id 不一致：product={p_pid} vs video={v_pid}。"
            )

        # 2. product_target_audience.primary_audiences 非空
        pta = product_diagnosis.get("product_target_audience") or {}
        if not (isinstance(pta, Mapping) and pta.get("primary_audiences")):
            blocking_issues.append("product_target_audience.primary_audiences 为空。")

        # 3. persuasion_requirement_profile 存在且 persuasion_requirements 非空
        profile = product_diagnosis.get("persuasion_requirement_profile") or {}
        if not (isinstance(profile, Mapping) and profile.get("persuasion_requirements")):
            blocking_issues.append("persuasion_requirement_profile.persuasion_requirements 为空。")

        # 4. product_HEC 存在且包含非空 candidates 数组（每个 candidate 含 H/E/C）
        product_hec = product_diagnosis.get("product_HEC") or {}
        candidates = product_hec.get("candidates") if isinstance(product_hec, Mapping) else None
        if not candidates or not isinstance(candidates, list):
            blocking_issues.append(
                "product_HEC.candidates 缺失或为空数组（candidates 为空必须 Crash Early）。"
            )
        else:
            for idx, cand in enumerate(candidates):
                if not isinstance(cand, Mapping):
                    blocking_issues.append(f"product_HEC.candidates[{idx}] 非法。")
                    continue
                h = cand.get("H") or cand.get("hook_tag")
                e = cand.get("E") or cand.get("effect_tag")
                c = cand.get("C") or cand.get("cta_tag")
                if not (h and e and c):
                    blocking_issues.append(
                        f"product_HEC.candidates[{idx}] 缺少 H/E/C 三轴标签。"
                    )

        # 5. video_HEC（或 primary_hec 映射）存在且含 hook/effect/cta（Block 1.2）
        video_hec = self._resolve_video_hec(video_understanding, warnings)
        if not self._hec_complete(video_hec):
            blocking_issues.append("video_HEC/primary_hec 缺失或缺少 hook_tag/effect_tag/cta_tag。")

        # 6. slider_signature 四轴齐全且每轴有 score
        slider = video_understanding.get("slider_signature") or {}
        if not isinstance(slider, Mapping):
            blocking_issues.append("slider_signature 非法。")
        else:
            for axis in SLIDER_AXES:
                axis_value = slider.get(axis)
                if not isinstance(axis_value, Mapping) or "score" not in axis_value:
                    blocking_issues.append(f"slider_signature.{axis} 缺失或缺少 score。")

        # 7. evidence_spans / semantic_bundles / storyboard_segments 至少一类可用
        if not any(
            video_understanding.get(key)
            for key in ("evidence_spans", "semantic_bundles", "storyboard_segments")
        ):
            blocking_issues.append(
                "evidence_spans / semantic_bundles / storyboard_segments 至少需要一类非空。"
            )

        if blocking_issues:
            raise VideoDiagnosisInputError("Step0 输入校验失败：" + "；".join(blocking_issues))

        return (
            {"status": "passed", "blocking_issues": [], "warnings": warnings},
            dict(video_hec),
        )

    @staticmethod
    def _hec_complete(hec: Any) -> bool:
        if not isinstance(hec, Mapping):
            return False
        return all(str(hec.get(k) or "").strip() for k in ("hook_tag", "effect_tag", "cta_tag"))

    @staticmethod
    def _resolve_video_hec(
        video_understanding: Mapping[str, Any], warnings: list[str]
    ) -> dict[str, Any]:
        """Block 1.2：优先 video_HEC，缺失时回退 primary_hec 并归一化（记 warning）。"""
        video_hec = video_understanding.get("video_HEC")
        if isinstance(video_hec, Mapping) and video_hec:
            return dict(video_hec)
        primary_hec = video_understanding.get("primary_hec")
        if isinstance(primary_hec, Mapping) and primary_hec:
            warnings.append(
                "video_HEC 缺失，已回退读取 primary_hec 并归一化为 video_HEC（命名契约映射）。"
            )
            return {
                "hook_tag": primary_hec.get("hook_tag"),
                "effect_tag": primary_hec.get("effect_tag"),
                "cta_tag": primary_hec.get("cta_tag"),
                "signature": primary_hec.get("signature", ""),
            }
        return {}

    # ------------------------------------------------------------------ 语料
    def _build_corpus(self, video_understanding: Mapping[str, Any]) -> dict[str, Any]:
        storyboard = [s for s in (video_understanding.get("storyboard_segments") or []) if isinstance(s, Mapping)]
        bundles = [b for b in (video_understanding.get("semantic_bundles") or []) if isinstance(b, Mapping)]
        spans = [e for e in (video_understanding.get("evidence_spans") or []) if isinstance(e, Mapping)]

        role_text: dict[str, list[str]] = {"hook": [], "effect": [], "cta": [], "other": []}
        all_parts: list[str] = []
        span_index: list[tuple[str, str]] = []  # (span_id, text)

        for seg in storyboard:
            role = _segment_role(seg).lower()
            text = _segment_text(seg)
            all_parts.append(text)
            if "hook" in role:
                role_text["hook"].append(text)
            elif "effect" in role or "proof" in role or "body" in role:
                role_text["effect"].append(text)
            elif "cta" in role or "close" in role:
                role_text["cta"].append(text)
            else:
                role_text["other"].append(text)
        for bundle in bundles:
            text = _segment_text(bundle)
            all_parts.append(text)
            role = _segment_role(bundle).lower()
            bucket = "hook" if "hook" in role else "effect" if ("effect" in role or "proof" in role) else "cta" if "cta" in role else "other"
            role_text[bucket].append(text)
        for idx, span in enumerate(spans):
            text = _segment_text(span)
            all_parts.append(text)
            span_index.append((_span_id(span, idx), text))

        return {
            "full_text": " ".join(all_parts),
            "role_text": {k: " ".join(v) for k, v in role_text.items()},
            "span_index": span_index,
            "has_storyboard": bool(storyboard),
            "has_bundles": bool(bundles),
            "has_spans": bool(spans),
        }

    # ------------------------------------------------------------------ Step1
    def _step1_video_target_audience(
        self, video_understanding: Mapping[str, Any], corpus: Mapping[str, Any]
    ) -> dict[str, Any]:
        """独立判定视频目标人群。严禁读取 slider_signature（本方法签名不接收 slider）。"""
        full_text = corpus["full_text"]
        role_text = corpus["role_text"]
        hook_text = " ".join([role_text.get("hook", ""), role_text.get("other", "")])
        cta_text = " ".join([role_text.get("cta", ""), role_text.get("effect", "")])

        # ---- 信号采集（四步链路）----
        hook_signals = sorted(set(_hits(hook_text or full_text, VIDEO_HOOK_KEYWORDS)))
        scene_signals = sorted(set(_hits(full_text, VIDEO_SCENE_KEYWORDS)))
        female_hits = _hits(full_text, VIDEO_FEMALE_KEYWORDS)
        male_hits = _hits(full_text, VIDEO_MALE_KEYWORDS)
        mature_hits = _hits(full_text, VIDEO_MATURE_KEYWORDS)
        young_hits = _hits(full_text, VIDEO_YOUNG_KEYWORDS)
        persona_signals = sorted(set(female_hits + male_hits + mature_hits + young_hits))
        low_hits = _hits(cta_text or full_text, VIDEO_LOW_CONSUMPTION_KEYWORDS)
        mid_high_hits = _hits(cta_text or full_text, VIDEO_MID_HIGH_CONSUMPTION_KEYWORDS)
        cta_benefit_signals = sorted(set(low_hits + mid_high_hits))

        # ---- 性别轴（主值 + 冲突少数信号）----
        gender_conflict: list[str] = []
        if female_hits and male_hits:
            if len(female_hits) > len(male_hits):
                gender_axis = "female"
                gender_conflict = ["male"]
                gender_reason = (
                    f"女性信号占主导（{'、'.join(female_hits[:5])}），"
                    f"存在少数男性信号（{'、'.join(male_hits[:3])}）→ 男性为错配风险"
                )
            elif len(male_hits) > len(female_hits):
                gender_axis = "male"
                gender_conflict = ["female"]
                gender_reason = (
                    f"男性信号占主导（{'、'.join(male_hits[:5])}），"
                    f"存在少数女性信号（{'、'.join(female_hits[:3])}）→ 女性为错配风险"
                )
            else:
                gender_axis = "mixed"
                gender_reason = "男女信号势均力敌，性别判为 mixed"
        elif female_hits:
            gender_axis, gender_reason = "female", f"人设/场景命中女性信号（{'、'.join(female_hits[:5])}）"
        elif male_hits:
            gender_axis, gender_reason = "male", f"人设/场景命中男性信号（{'、'.join(male_hits[:5])}）"
        else:
            gender_axis, gender_reason = "mixed", "性别信号不唯一或缺失，性别判为 mixed"

        # ---- 年龄轴（主值 + 次级邻接年龄）----
        age_secondary: list[str] = []
        if mature_hits and young_hits:
            if len(mature_hits) > len(young_hits):
                age_axis = "mature"
                age_secondary = ["young"]
                age_reason = (
                    f"年长信号占主导（{'、'.join(mature_hits[:5])}），"
                    f"存在少数年轻信号（{'、'.join(young_hits[:3])}）→ 年轻为次级邻接"
                )
            elif len(young_hits) > len(mature_hits):
                age_axis = "young"
                age_secondary = ["mature"]
                age_reason = (
                    f"年轻信号占主导（{'、'.join(young_hits[:5])}），"
                    f"存在少数年长信号（{'、'.join(mature_hits[:3])}）→ 年长为次级邻接"
                )
            else:
                age_axis = "mixed"
                age_reason = "年长与年轻信号势均力敌，年龄判为 mixed"
        elif mature_hits:
            age_axis, age_reason = "mature", f"人设/场景命中年长信号（{'、'.join(mature_hits[:5])}）"
        elif young_hits:
            age_axis, age_reason = "young", f"人设/场景命中年轻信号（{'、'.join(young_hits[:5])}）"
        else:
            age_axis, age_reason = "mixed", "无明确年龄信号，年龄判为 mixed"

        # ---- 消费力轴（CTA/利益点）----
        consumption_neighbor: Optional[str] = None
        if mid_high_hits and low_hits:
            consumption_axis = "mixed"
            consumption_values = ["mid_high", "low"]
            consumption_reason = "CTA 同时出现品质/官方与低价/划算信号，消费力判为 mixed（同人群跨消费力覆盖）"
        elif mid_high_hits:
            consumption_axis = "mid_high"
            consumption_values = ["mid_high"]
            consumption_neighbor = "low"
            consumption_reason = f"CTA/利益点命中品质/官方/成分信号（{'、'.join(mid_high_hits[:5])}）→ 中高消费力"
        elif low_hits:
            consumption_axis = "low"
            consumption_values = ["low"]
            consumption_neighbor = "mid_high"
            consumption_reason = f"CTA/利益点命中低价/划算信号（{'、'.join(low_hits[:5])}）→ 低消费力"
        else:
            consumption_axis = "mixed"
            consumption_values = ["mid_high", "low"]
            consumption_reason = "无明确 CTA/利益点消费力信号，消费力判为 mixed"

        # ---- 八大人群收束（显式主/次/风险分桶，避免朴素笛卡尔积）----
        age_primary = age_axis if age_axis in ("young", "mature") else "mature"
        gender_primary = gender_axis if gender_axis in ("female", "male") else "female"

        evidence_all = sorted(set(hook_signals + scene_signals + persona_signals + cta_benefit_signals))

        def _judgment(group: str, level: str, reason: str) -> dict[str, Any]:
            return {
                "audience_group": group,
                "fit_level": level,
                "reason": reason,
                "evidence": evidence_all,
            }

        seen: set[str] = set()
        # 主目标：主年龄 × 主性别 × 全部消费力档（消费力 mixed 时同人群跨档均为主覆盖）
        primary_audiences: list[dict[str, Any]] = []
        for c in consumption_values:
            group = compose_audience_group(age_primary, gender_primary, c)
            if group in seen:
                continue
            seen.add(group)
            primary_audiences.append(
                _judgment(
                    group,
                    "primary",
                    f"Hook/场景→入场角色；人设→{age_reason}、{gender_reason}；CTA→{consumption_reason}。",
                )
            )

        # 次级目标：跨年龄邻接（主性别 × 邻接年龄 × 全部消费力档）+ 跨消费力邻接
        secondary_audiences: list[dict[str, Any]] = []
        for sa in age_secondary:
            for c in consumption_values:
                group = compose_audience_group(sa, gender_primary, c)
                if group in seen:
                    continue
                seen.add(group)
                secondary_audiences.append(
                    _judgment(group, "secondary", "主性别下相邻年龄层的次级覆盖人群（跨年龄邻接）。")
                )
        if consumption_neighbor is not None:
            group = compose_audience_group(age_primary, gender_primary, consumption_neighbor)
            if group not in seen:
                seen.add(group)
                secondary_audiences.append(
                    _judgment(group, "secondary", "同年龄/性别下相邻消费力档的次级覆盖人群（跨消费力邻接）。")
                )
        secondary_audiences = secondary_audiences[:3]

        # mismatch_risk：Step1 内生冲突信号（少数性别 × 邻接年龄 × 低消费力档）
        mismatch_risk_audiences: list[dict[str, Any]] = []
        risk_age = age_secondary[0] if age_secondary else age_primary
        for cg in gender_conflict:
            group = compose_audience_group(risk_age, cg, "low")
            if group in seen:
                continue
            seen.add(group)
            mismatch_risk_audiences.append(
                _judgment(
                    group,
                    "mismatch_risk",
                    f"视频出现与主目标性别冲突的少数信号，可能误吸引「{group}」，属错配风险人群。",
                )
            )

        caveats: list[str] = []
        if "mixed" in (age_axis, gender_axis, consumption_axis):
            caveats.append("存在 mixed 轴：视频人群信号边界不清，已展开多坐标覆盖。")
        if gender_conflict or age_secondary:
            caveats.append("存在少数/邻接人群信号，已分别归入次级覆盖或错配风险。")

        return {
            "primary_audiences": primary_audiences,
            "secondary_audiences": secondary_audiences,
            "mismatch_risk_audiences": mismatch_risk_audiences,
            "axis_judgment": {
                "age_axis": {"value": age_axis, "evidence": sorted(set(mature_hits + young_hits)), "reason": age_reason},
                "gender_axis": {"value": gender_axis, "evidence": sorted(set(female_hits + male_hits)), "reason": gender_reason},
                "consumption_power_axis": {
                    "value": consumption_axis,
                    "evidence": sorted(set(low_hits + mid_high_hits)),
                    "reason": consumption_reason,
                },
            },
            "reasoning_chain": {
                "hook_scene_to_role": f"Hook 信号（{'、'.join(hook_signals) or '无'}）+ 场景信号（{'、'.join(scene_signals) or '无'}）→ 入场角色与场景。",
                "persona_to_age_gender": f"人设信号 → {age_reason}；{gender_reason}。",
                "cta_benefit_to_consumption_power": f"CTA/利益点信号 → {consumption_reason}。",
            },
            "evidence_summary": {
                "hook_signals": hook_signals,
                "scene_signals": scene_signals,
                "persona_signals": persona_signals,
                "cta_benefit_signals": cta_benefit_signals,
            },
            "caveats": caveats,
        }

    # ------------------------------------------------------------------ Step2
    def _step2_audience_match(
        self, product_target_audience: Mapping[str, Any], video_target_audience: Mapping[str, Any]
    ) -> dict[str, Any]:
        product_primary = self._group_set(product_target_audience.get("primary_audiences"))
        product_secondary = self._group_set(product_target_audience.get("secondary_audiences"))
        video_primary = self._group_set(video_target_audience.get("primary_audiences"))

        matched = sorted(product_primary & video_primary)
        uncovered = sorted(product_primary - video_primary)
        unexpected = sorted(video_primary - (product_primary | product_secondary))

        # match_status（优先判 too_broad）
        if len(video_primary) >= 4 or not video_primary:
            match_status = "too_broad"
        elif product_primary and product_primary <= video_primary:
            match_status = "high_match"
        elif matched:
            match_status = "partial_match"
        else:
            match_status = "low_match"

        judgment = (
            f"商品主目标 P={sorted(product_primary)}；视频主目标 V={sorted(video_primary)}；"
            f"matched={matched}，uncovered={uncovered}，unexpected={unexpected} → {match_status}。"
        )
        return {
            "match_status": match_status,
            "matched_audiences": matched,
            "uncovered_product_audiences": uncovered,
            "unexpected_video_audiences": unexpected,
            "judgment": judgment,
            "evidence": sorted(product_primary | video_primary),
        }

    @staticmethod
    def _group_set(items: Any) -> set[str]:
        result: set[str] = set()
        if isinstance(items, (list, tuple)):
            for it in items:
                if isinstance(it, Mapping) and it.get("audience_group"):
                    result.add(str(it["audience_group"]))
                elif isinstance(it, str):
                    result.add(it)
        return result

    @staticmethod
    def _backfill_mismatch_risk(
        video_target_audience: dict[str, Any], audience_match: Mapping[str, Any]
    ) -> None:
        """A2：当人群匹配为 low_match / too_broad 时，把 Step2 计算出的
        unexpected_video_audiences（视频锁定、商品未覆盖的人群）回填为
        video_target_audience.mismatch_risk_audiences；high_match / partial_match
        不追加。与 Step1 内生的错配风险人群合并去重。"""
        if audience_match.get("match_status") not in ("low_match", "too_broad"):
            return
        existing = video_target_audience.setdefault("mismatch_risk_audiences", [])
        seen = {
            j.get("audience_group")
            for j in existing
            if isinstance(j, Mapping)
        }
        for group in audience_match.get("unexpected_video_audiences") or []:
            if group in seen:
                continue
            seen.add(group)
            existing.append(
                {
                    "audience_group": group,
                    "fit_level": "mismatch_risk",
                    "reason": (
                        f"人群匹配为 {audience_match.get('match_status')}：视频锁定「{group}」"
                        f"但不在商品主/次目标内，属错配风险。"
                    ),
                    "evidence": [],
                }
            )

    # ------------------------------------------------------------------ Step3
    def _step3_profile_match(
        self,
        profile: Mapping[str, Any],
        corpus: Mapping[str, Any],
        *,
        jtbd_level1: Optional[str] = None,
        secondary_benefits: Optional[list] = None,
        non_selected_task_reasons: Optional[list] = None,
    ) -> dict[str, Any]:
        # PRD-4.1 needs_review early return：jtbd_level1 == needs_review 时 Profile Match
        # 直接 early return，只输出"商品任务未确定 / 事实不足，需复核"，不做 Primary/Secondary/Blocked
        # 三层覆盖判断。match_status 取合法枚举 not_applicable（任务未确定，不适用三层判断）。
        if jtbd_level1 == NEEDS_REVIEW:
            result = {
                "match_status": "not_applicable",
                "task_status": NEEDS_REVIEW,
                "requirement_results": [],
                "missing_required_requirements": [],
                "weak_requirements": [],
                "information_miss_summary": "商品任务未确定 / 事实不足，需复核（jtbd_level1=needs_review）。",
                "profile_match_error": {},
                "coverage_layers": {},
                "blocked_direction_hit": False,
                "early_return_reason": NEEDS_REVIEW,
            }
            # 后置断言（Crash Early）：needs_review early return 不得携带三层覆盖结论。
            if result["coverage_layers"] or result["requirement_results"]:
                raise VideoDiagnosisInputError(
                    "Profile Match needs_review early return 断言失败：不应产出 Primary/Secondary/Blocked 三层结论。"
                )
            return result

        full_text = corpus["full_text"]
        span_index = corpus["span_index"]
        requirements = profile.get("persuasion_requirements") or []
        not_applicable_ids = {
            str((r or {}).get("requirement_id"))
            for r in (profile.get("not_applicable_requirements") or [])
            if isinstance(r, Mapping)
        }

        requirement_results: list[dict[str, Any]] = []
        missing_required: list[str] = []
        weak_requirements: list[str] = []

        for req in requirements:
            if not isinstance(req, Mapping):
                continue
            req_id = str(req.get("requirement_id") or "")
            req_name = str(req.get("requirement_name") or "")
            required = bool(req.get("required"))

            keywords = REQUIREMENT_KEYWORDS.get(req_id)
            if keywords is None:
                # 兜底：用 requirement_name 与 success_criteria 切词
                criteria = str(req.get("success_criteria") or "")
                keywords = tuple(t for t in (req_name,) if t)
                matched_kw = [kw for kw in keywords if kw and kw in full_text]
                if criteria and any(tok in full_text for tok in [req_name] if tok):
                    matched_kw = matched_kw or [req_name]
            else:
                matched_kw = [kw for kw in keywords if kw in full_text]

            matched_spans = [sid for sid, text in span_index if any(kw in text for kw in (keywords or ()))]

            if req_id in not_applicable_ids:
                status = "not_applicable"
            else:
                total = max(len(keywords or ()), 1)
                ratio = len(matched_kw) / total
                if len(matched_kw) == 0:
                    status = "missing"
                elif ratio >= 0.5 or len(matched_kw) >= 3:
                    status = "completed"
                elif ratio >= 0.3 or len(matched_kw) == 2:
                    status = "partial"
                else:
                    status = "weak"

            if status == "missing" and required:
                missing_required.append(req_id)
            if status in ("weak", "partial"):
                weak_requirements.append(req_id)

            judgment = (
                f"要求「{req_name}」命中关键词 {matched_kw or '无'} → {status}。"
            )
            repair_direction = ""
            if status == "missing":
                repair_direction = f"视频需补讲「{req_name}」相关内容（参考关键词：{list(keywords or ())[:5]}）。"
            elif status in ("weak", "partial"):
                repair_direction = f"视频对「{req_name}」表达偏弱，建议强化证据与口播。"

            requirement_results.append(
                {
                    "requirement_id": req_id,
                    "requirement_name": req_name,
                    "required": required,
                    "completion_status": status,
                    "matched_evidence_spans": matched_spans,
                    "judgment": judgment,
                    "repair_direction": repair_direction,
                }
            )

        # overall_status（D1 合法枚举：completed/partial/weak/missing/not_applicable）
        required_results = [r for r in requirement_results if r["required"] and r["completion_status"] != "not_applicable"]
        if not required_results:
            # 所有必讲均 not_applicable（或不存在必讲）
            overall_status = "not_applicable"
        elif any(r["completion_status"] == "missing" for r in required_results):
            overall_status = "missing"
        elif all(r["completion_status"] == "completed" for r in required_results):
            overall_status = "completed"
        else:
            # 含 partial/weak（无 missing），按 D1 语义：归为 partial
            overall_status = "partial"

        info_miss = (
            f"必讲缺失 {missing_required}；偏弱/部分 {weak_requirements}。"
            if (missing_required or weak_requirements)
            else "必讲要求均已充分覆盖。"
        )

        # ---------------------------------------------------------------- PRD-4.1 收窄三层
        # profile_match_error：边界违规归因（dict[error_code -> reason]），非空即"不得判 good"。
        profile_match_error: dict[str, str] = {}

        # Primary Coverage 层：jtbd_level1 为主任务基准，primary_requirement = 必讲说服要求集合。
        # 主说服要求"完全未覆盖"（所有必讲均 missing）→ primary_requirement_missing。
        required_ids = [r["requirement_id"] for r in required_results]
        covered_required_ids = [
            r["requirement_id"]
            for r in required_results
            if r["completion_status"] in ("completed", "partial", "weak")
        ]
        primary_covered = bool(covered_required_ids)
        primary_requirement_missing = bool(required_ids) and not primary_covered
        if primary_requirement_missing:
            profile_match_error["primary_requirement_missing"] = (
                f"主说服要求（必讲 {required_ids}）完全未覆盖。"
            )
        primary_layer = {
            "primary_requirement_ref": "primary_requirement",
            "required_requirement_ids": required_ids,
            "covered_requirement_ids": covered_required_ids,
            "missing_required_requirements": list(missing_required),
            "covered": primary_covered,
            "judgment": (
                "主说服要求未覆盖（primary_requirement_missing）。"
                if primary_requirement_missing
                else f"主说服要求覆盖：{covered_required_ids or '无必讲'}。"
            ),
        }

        # Secondary Coverage 层：secondary_benefits 中的任务不得替代 primary_requirement。
        # 视频命中次级收益（关键词）但主说服要求未覆盖 → 次级收益越位替代主链路：
        #   secondary_without_primary=true 且判 secondary_requirement_overclaimed（不得判 good）。
        secondary_benefits = list(secondary_benefits or [])
        secondary_hits: list[str] = [
            task
            for task in secondary_benefits
            if any(kw in full_text for kw in TASK_CLAIM_KEYWORDS.get(task, ()))
        ]
        secondary_without_primary = (
            bool(secondary_hits) and bool(required_ids) and not primary_covered
        )
        secondary_requirement_overclaimed = secondary_without_primary
        if secondary_requirement_overclaimed:
            profile_match_error["secondary_requirement_overclaimed"] = (
                f"次级收益 {secondary_hits} 替代主链路（primary_requirement 未覆盖）。"
            )
        if secondary_without_primary:
            profile_match_error["secondary_without_primary"] = (
                f"次级收益 {secondary_hits} 在主说服要求缺失下越位。"
            )
        secondary_layer = {
            "secondary_benefits": secondary_benefits,
            "secondary_benefits_hit": secondary_hits,
            "overclaimed": secondary_requirement_overclaimed,
            "secondary_without_primary": secondary_without_primary,
            "judgment": (
                f"次级收益越位替代主链路：{secondary_hits}。"
                if secondary_requirement_overclaimed
                else f"次级收益命中 {secondary_hits or '无'}，未越位。"
            ),
        }

        # Blocked Direction Check 层：GR-02/03/04 被阻断方向不得计入有效说服覆盖。
        # 判断依据 = 商品诊断 non_selected_task_reasons 中 gate=blocked 且 reason 前缀 GR-02/03/04 的方向。
        blocked_directions = self._collect_blocked_directions(non_selected_task_reasons)
        blocked_directions_hit: list[str] = [
            task
            for task in blocked_directions
            if any(kw in full_text for kw in TASK_CLAIM_KEYWORDS.get(task, ()))
        ]
        blocked_direction_hit = bool(blocked_directions_hit)
        if blocked_direction_hit:
            profile_match_error["blocked_requirement_hit"] = (
                f"视频把已阻断方向 {blocked_directions_hit} 当主诉（GR-02/03/04），不计入有效说服覆盖。"
            )
            profile_match_error["task_boundary_violation"] = (
                f"任务边界违规：已阻断方向 {blocked_directions_hit} 越界为说服主诉。"
            )
            profile_match_error["blocked_direction_hit"] = (
                f"blocked_direction_hit=true：{blocked_directions_hit}。"
            )
        blocked_layer = {
            "blocked_directions": blocked_directions,
            "blocked_directions_hit": blocked_directions_hit,
            "blocked_direction_hit": blocked_direction_hit,
            "judgment": (
                f"命中已阻断方向（不计入有效覆盖）：{blocked_directions_hit}。"
                if blocked_direction_hit
                else "未把任何已阻断方向（GR-02/03/04）当主诉。"
            ),
        }

        # 后置断言（Crash Early）：三层结论与 profile_match_error 必须结构一致，不得静默吞掉。
        if secondary_requirement_overclaimed and "secondary_requirement_overclaimed" not in profile_match_error:
            raise VideoDiagnosisInputError(
                "Profile Match 断言失败：secondary_requirement_overclaimed 判定未写入 profile_match_error。"
            )
        if primary_requirement_missing and "primary_requirement_missing" not in profile_match_error:
            raise VideoDiagnosisInputError(
                "Profile Match 断言失败：primary_requirement_missing 判定未写入 profile_match_error。"
            )
        if blocked_direction_hit != ("blocked_direction_hit" in profile_match_error):
            raise VideoDiagnosisInputError(
                "Profile Match 断言失败：blocked_direction_hit 与 profile_match_error 不一致。"
            )

        return {
            "match_status": overall_status,
            "requirement_results": requirement_results,
            "missing_required_requirements": missing_required,
            "weak_requirements": weak_requirements,
            "information_miss_summary": info_miss,
            "profile_match_error": profile_match_error,
            "blocked_direction_hit": blocked_direction_hit,
            "coverage_layers": {
                "primary_coverage": primary_layer,
                "secondary_coverage": secondary_layer,
                "blocked_direction_check": blocked_layer,
            },
        }

    @staticmethod
    def _collect_blocked_directions(non_selected_task_reasons: Optional[list]) -> list[str]:
        """PRD-4.1：从商品诊断 non_selected_task_reasons 提取 gate=blocked 且 reason 前缀为
        GR-02/GR-03/GR-04 的被阻断方向（去重保序）。"""
        directions: list[str] = []
        for item in (non_selected_task_reasons or []):
            if not isinstance(item, Mapping):
                continue
            if item.get("gate") != "blocked":
                continue
            reason = str(item.get("reason") or "")
            for code, mapped_task in GR_BLOCKED_DIRECTION_TASK.items():
                if reason.startswith(f"{code}:"):
                    # 优先用 reason 命中的 GR 编号映射方向；与 task 字段取并集（容错）。
                    for task in (mapped_task, str(item.get("task") or "")):
                        if task and task not in directions:
                            directions.append(task)
                    break
        return directions

    # ------------------------------------------------------------------ Step4
    def _step4_hec_match(
        self,
        product_hec: Mapping[str, Any],
        video_hec: Mapping[str, Any],
        *,
        ec_skeletons: Optional[list] = None,
        candidate_set: Optional[Any] = None,
    ) -> dict[str, Any]:
        """HEC Match：与 candidates 集合做三轴交集判定。

        - 三轴整组命中某一候选 → good
        - 命中轴数 == 2 且 E 命中 → acceptable_deviation
        - 命中轴数 == 2 且 E 不命中（H+C 命中、E 错配）→ risky_deviation
        - 命中轴数 <= 1 → mismatch
        - candidates 为空 → Crash Early（应已在 Step0 拦截）

        PRD-4.2 回查：当上游提供 Product_EC_Skeletons / CandidateSet 时，先用 video_HEC 对齐
        Product_HECs（即上方三轴判定），再通过 (effect_tag, cta_tag) 回查 Product_EC_Skeletons，
        并显式回查 source_role / source_requirement_ref，校验 cta_resolution / effect_capabilities_snapshot /
        soft_constraint_results / risk_flags。回查失败 → raise ValueError(source_role_lookup_failed)。
        """
        candidates_raw = product_hec.get("candidates") if isinstance(product_hec, Mapping) else None
        if not candidates_raw:
            raise VideoDiagnosisInputError(
                "product_HEC.candidates 为空，无法进行 HEC Match（Crash Early）。"
            )

        candidates: list[dict[str, str]] = []
        for cand in candidates_raw:
            candidates.append({
                "H": str(cand.get("H") or cand.get("hook_tag") or ""),
                "E": str(cand.get("E") or cand.get("effect_tag") or ""),
                "C": str(cand.get("C") or cand.get("cta_tag") or ""),
            })

        actual = {k: str(video_hec.get(k) or "") for k in ("hook_tag", "effect_tag", "cta_tag")}
        actual_hec = {"H": actual["hook_tag"], "E": actual["effect_tag"], "C": actual["cta_tag"]}

        h_set = {c["H"] for c in candidates}
        e_set = {c["E"] for c in candidates}
        c_set = {c["C"] for c in candidates}

        h_hit = actual_hec["H"] in h_set
        e_hit = actual_hec["E"] in e_set
        c_hit = actual_hec["C"] in c_set

        full_hit = any(
            actual_hec["H"] == c["H"] and actual_hec["E"] == c["E"] and actual_hec["C"] == c["C"]
            for c in candidates
        )

        if full_hit:
            match_status = "good"
        else:
            hit_count = int(h_hit) + int(e_hit) + int(c_hit)
            if hit_count >= 2:
                # E 轴权重硬约束：必须显式区分 E 命中/不命中
                match_status = "acceptable_deviation" if e_hit else "risky_deviation"
            else:
                match_status = "mismatch"

        gap_parts: list[str] = []
        if not h_hit:
            gap_parts.append(f"hook {actual_hec['H']} 不在候选 H 集 {sorted(h_set)}")
        if not e_hit:
            gap_parts.append(f"effect {actual_hec['E']} 不在候选 E 集 {sorted(e_set)}（说服核心）")
        if not c_hit:
            gap_parts.append(f"cta {actual_hec['C']} 不在候选 C 集 {sorted(c_set)}")
        hec_gap_summary = "；".join(gap_parts) if gap_parts else "三轴均命中候选集合。"

        logic_chain_judgment = (
            f"actual={actual_hec}；H_hit={h_hit}, E_hit={e_hit}, C_hit={c_hit}, "
            f"full_combination_hit={full_hit} → {match_status}。"
        )

        result = {
            "match_status": match_status,
            "candidates": candidates,
            "actual_video_hec": actual,
            "hook_hit": h_hit,
            "effect_hit": e_hit,
            "cta_hit": c_hit,
            "full_combination_hit": full_hit,
            "logic_chain_judgment": logic_chain_judgment,
            "hec_gap_summary": hec_gap_summary,
        }

        # ---------------------------------------------------------------- PRD-4.2 回查
        # 仅当上游提供 Product_EC_Skeletons / CandidateSet 时执行（缺省则保持纯三轴判定，向后兼容）。
        if ec_skeletons is not None or candidate_set is not None:
            result["source_role_lookup"] = self._hec_source_role_lookup(
                effect_tag=actual["effect_tag"],
                cta_tag=actual["cta_tag"],
                ec_skeletons=ec_skeletons,
                candidate_set=candidate_set,
                product_hec=product_hec,
                actual_hec=actual,
            )
        return result

    # ------------------------------------------------------------------ PRD-4.2 HEC 回查
    def _hec_source_role_lookup(
        self,
        *,
        effect_tag: str,
        cta_tag: str,
        ec_skeletons: Optional[list],
        candidate_set: Optional[Any],
        product_hec: Mapping[str, Any],
        actual_hec: Mapping[str, str],
    ) -> dict[str, Any]:
        """通过 (effect_tag, cta_tag) 回查 Product_EC_Skeletons 与 CandidateSet 的
        source_role / source_requirement_ref，并校验 cta_resolution / effect_capabilities_snapshot /
        soft_constraint_results / risk_flags。回查失败即 Crash Early（raise ValueError）。"""

        # 1) 回查 Product_EC_Skeletons：通过 (effect_tag, cta_tag) 命中骨架，取 cta_resolution / 能力快照。
        skeleton = None
        for sk in (ec_skeletons or []):
            if not isinstance(sk, Mapping):
                continue
            if str(sk.get("effect_tag") or "") == effect_tag and str(sk.get("cta_tag") or "") == cta_tag:
                skeleton = sk
                break

        # 2) 回查 CandidateSet：effect_list / cta_list 显式回查 source_role / source_requirement_ref。
        effect_list = self._candidate_list(candidate_set, "effect_list")
        cta_list = self._candidate_list(candidate_set, "cta_list")
        effect_candidate = next(
            (c for c in effect_list if isinstance(c, Mapping) and str(c.get("effect_tag") or "") == effect_tag),
            None,
        )
        cta_candidate = next(
            (c for c in cta_list if isinstance(c, Mapping) and str(c.get("cta_tag") or "") == cta_tag),
            None,
        )

        # 后置断言（Crash Early）：(effect_tag, cta_tag) 必须能在 CandidateSet / EC Skeleton 中
        # 回查到 source_role / source_requirement_ref，否则 source_role_lookup_failed。
        roles: dict[str, dict[str, str]] = {}
        if effect_candidate is not None:
            roles["effect"] = {
                "source_role": str(effect_candidate.get("source_role") or ""),
                "source_requirement_ref": str(effect_candidate.get("source_requirement_ref") or ""),
            }
        if cta_candidate is not None:
            roles["cta"] = {
                "source_role": str(cta_candidate.get("source_role") or ""),
                "source_requirement_ref": str(cta_candidate.get("source_requirement_ref") or ""),
            }
        valid_roles = {
            axis: meta
            for axis, meta in roles.items()
            if meta["source_role"] and meta["source_requirement_ref"]
        }
        if not valid_roles or skeleton is None:
            raise ValueError(
                "source_role_lookup_failed：(effect_tag, cta_tag)="
                f"({effect_tag!r}, {cta_tag!r}) 无法在 CandidateSet/EC Skeleton 回查到 "
                f"source_role/source_requirement_ref（ec_skeleton_hit={skeleton is not None}，"
                f"candidate_roles={roles or '无'}）。"
            )

        # 3) 主次错位：仅命中 secondary 来源而 primary 缺失 → secondary_hit_without_primary。
        present_roles = {meta["source_role"] for meta in valid_roles.values()}
        secondary_hit_without_primary = bool(present_roles) and present_roles <= {"secondary"}

        # 4) 校验 cta_resolution（降级越界）/ effect_capabilities_snapshot（证据能力覆盖）。
        cta_resolution = skeleton.get("cta_resolution") or {}
        effect_capabilities_snapshot = list(skeleton.get("effect_capabilities_snapshot") or [])
        cta_downgraded = str((cta_resolution or {}).get("resolution_type") or "") == "downgrade"
        effect_capabilities_missing = not effect_capabilities_snapshot

        # 5) 继承风险：从对齐的 Product_HEC 候选回查 soft_constraint_results / risk_flags。
        matched_variant = self._match_product_hec_variant(product_hec, actual_hec)
        soft_constraint_results = (
            list(matched_variant.get("soft_constraint_results") or []) if matched_variant else []
        )
        risk_flags = (
            list(matched_variant.get("risk_flags") or []) if matched_variant else []
        )

        lookup = {
            "effect_tag": effect_tag,
            "cta_tag": cta_tag,
            "ec_skeleton_hit": True,
            "source_roles": roles,
            "secondary_hit_without_primary": secondary_hit_without_primary,
            "cta_resolution": cta_resolution,
            "cta_downgraded": cta_downgraded,
            "effect_capabilities_snapshot": effect_capabilities_snapshot,
            "effect_capabilities_missing": effect_capabilities_missing,
            "soft_constraint_results": soft_constraint_results,
            "risk_flags": risk_flags,
            "judgment": (
                "回查成功："
                f"source_roles={present_roles}"
                + ("；仅命中 secondary 来源，主次错位 secondary_hit_without_primary。" if secondary_hit_without_primary else "。")
            ),
        }

        # 后置断言（Crash Early）：回查结论结构合法，不得静默吞掉。
        if secondary_hit_without_primary and "primary" in present_roles:
            raise ValueError(
                "HEC Match 回查断言失败：secondary_hit_without_primary 与 present_roles 含 primary 矛盾。"
            )
        return lookup

    @staticmethod
    def _candidate_list(candidate_set: Optional[Any], list_name: str) -> list:
        """从 CandidateSet（dict 或对象）取 h_list/effect_list/cta_list。"""
        if candidate_set is None:
            return []
        if isinstance(candidate_set, Mapping):
            value = candidate_set.get(list_name)
        else:
            value = getattr(candidate_set, list_name, None)
        return list(value) if isinstance(value, (list, tuple)) else []

    @staticmethod
    def _match_product_hec_variant(
        product_hec: Mapping[str, Any], actual_hec: Mapping[str, str]
    ) -> Optional[Mapping[str, Any]]:
        """从 product_HEC.candidates（Product_HECs）回查与 video_HEC 三轴一致的 variant，
        用于继承 soft_constraint_results / risk_flags。"""
        candidates_raw = product_hec.get("candidates") if isinstance(product_hec, Mapping) else None
        for cand in (candidates_raw or []):
            if not isinstance(cand, Mapping):
                continue
            h = str(cand.get("H") or cand.get("hook_tag") or "")
            e = str(cand.get("E") or cand.get("effect_tag") or "")
            c = str(cand.get("C") or cand.get("cta_tag") or "")
            if (
                h == actual_hec.get("hook_tag")
                and e == actual_hec.get("effect_tag")
                and c == actual_hec.get("cta_tag")
            ):
                return cand
        return None

    # ------------------------------------------------------------------ Step5
    def _resolve_slider_dictionary(self) -> Mapping[str, Any]:
        if self._slider_dictionary is None:
            self._slider_dictionary = load_audience_slider_preference_dictionary()
        return self._slider_dictionary

    def _step5_slider_match(
        self,
        video_target_audience: Mapping[str, Any],
        product_target_audience: Mapping[str, Any],
        slider_signature: Mapping[str, Any],
    ) -> dict[str, Any]:
        slider_dict = self._resolve_slider_dictionary()
        preferences = slider_dict["preferences"]

        video_primary = self._group_set(video_target_audience.get("primary_audiences"))
        product_primary = self._group_set(product_target_audience.get("primary_audiences"))
        intersection = video_primary & product_primary
        if intersection:
            reference_groups = sorted(intersection)
        elif video_primary:
            reference_groups = sorted(video_primary)
        else:
            reference_groups = sorted(product_primary)
        if not reference_groups:
            raise VideoDiagnosisInputError("Step5 无法确定参照人群（video/product primary 均为空）。")

        ref_group = reference_groups[0]
        if ref_group not in preferences:
            raise VideoDiagnosisInputError(f"slider 偏好字典缺少参照人群 {ref_group}。")
        ref_pref = preferences[ref_group]

        expected_pref: dict[str, dict[str, float]] = {}
        actual_signature: dict[str, float] = {}
        axis_results: list[dict[str, Any]] = []
        for axis in SLIDER_AXES:
            rng = ref_pref[axis]
            lo, hi = float(rng["min"]), float(rng["max"])
            expected_pref[axis] = {"min": lo, "max": hi}
            score = float((slider_signature.get(axis) or {}).get("score", 0.0))
            actual_signature[axis] = score

            if lo <= score <= hi:
                fit_status = "fit"
                repair = ""
            elif score < lo:
                gap = lo - score
                fit_status = "wrong_direction" if gap >= 0.4 else "too_weak"
                repair = f"{axis} 偏弱（{score:.2f} < {lo:.2f}），需加强表现强度。"
            else:
                gap = score - hi
                fit_status = "wrong_direction" if gap >= 0.4 else "too_strong"
                repair = f"{axis} 偏强（{score:.2f} > {hi:.2f}），需收敛表现强度。"
            axis_results.append(
                {
                    "axis": axis,
                    "fit_status": fit_status,
                    "judgment": f"{axis} score={score:.2f}，参照区间[{lo:.2f},{hi:.2f}] → {fit_status}。",
                    "repair_direction": repair,
                }
            )

        statuses = [r["fit_status"] for r in axis_results]
        non_fit = [s for s in statuses if s != "fit"]
        has_strong = "too_strong" in statuses
        has_weak = "too_weak" in statuses
        has_wrong = "wrong_direction" in statuses
        if not non_fit:
            match_status = "fit"
        elif statuses.count("wrong_direction") >= 2 or (len(non_fit) == len(statuses) and has_wrong):
            match_status = "mismatch"
        elif has_wrong and not has_strong and not has_weak:
            match_status = "wrong_direction"
        elif has_strong and not has_weak and not has_wrong:
            match_status = "too_strong"
        elif has_weak and not has_strong and not has_wrong:
            match_status = "too_weak"
        else:
            match_status = "mixed_deviation"

        acceptance = (
            f"参照人群「{ref_group}」对四轴的接受度评估 → {match_status}。"
        )
        gap_summary_parts = [r["axis"] for r in axis_results if r["fit_status"] != "fit"]
        slider_gap_summary = (
            f"偏离轴：{gap_summary_parts}。" if gap_summary_parts else "四轴均落在参照人群偏好区间内。"
        )
        return {
            "match_status": match_status,
            "target_audience_reference": reference_groups,
            "expected_slider_preference": expected_pref,
            "actual_slider_signature": actual_signature,
            "axis_results": axis_results,
            "audience_acceptance_judgment": acceptance,
            "slider_gap_summary": slider_gap_summary,
        }

    # ------------------------------------------------------------------ Step6
    def _step6_summary(
        self,
        audience_match: Mapping[str, Any],
        profile_match: Mapping[str, Any],
        hec_match: Mapping[str, Any],
        slider_match: Mapping[str, Any],
    ) -> dict[str, Any]:
        repair_suggestions: list[dict[str, Any]] = []
        key_findings: list[str] = []
        priority_issues: list[str] = []

        # ---- audience（P0：商品主目标未覆盖 / low_match / too_broad）----
        a_status = audience_match.get("match_status")
        uncovered = audience_match.get("uncovered_product_audiences") or []
        if uncovered or a_status in ("low_match", "too_broad"):
            repair_suggestions.append(
                {
                    "priority": "P0",
                    "issue_type": "audience",
                    "issue_summary": f"人群匹配 {a_status}，未覆盖商品主目标 {uncovered}。",
                    "repair_direction": "调整 Hook/人设/CTA，使视频主目标对齐商品主目标人群。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P0:audience")
        key_findings.append(f"人群匹配：{a_status}。")

        # ---- profile（P0：必讲 missing；P2：weak/partial）----
        missing_required = profile_match.get("missing_required_requirements") or []
        weak_reqs = profile_match.get("weak_requirements") or []
        # PRD-4.1：Profile Match 边界违规（secondary 越位 / primary 缺失 / blocked 方向命中）
        # 一律记 P0，确保"不得判 good"。
        profile_match_error = profile_match.get("profile_match_error") or {}
        if profile_match_error:
            repair_suggestions.append(
                {
                    "priority": "P0",
                    "issue_type": "profile",
                    "issue_summary": (
                        f"Profile Match 边界违规（profile_match_error）：{sorted(profile_match_error.keys())}。"
                    ),
                    "repair_direction": (
                        "回到主说服要求(primary_requirement)主链路，删除越位的次级收益与被阻断方向(GR-02/03/04)主诉。"
                    ),
                    "related_evidence_spans": [],
                }
            )
            for code in sorted(profile_match_error.keys()):
                priority_issues.append(f"P0:profile:{code}")
        if missing_required:
            repair_suggestions.append(
                {
                    "priority": "P0",
                    "issue_type": "profile",
                    "issue_summary": f"必讲说服要求缺失：{missing_required}。",
                    "repair_direction": "补齐缺失的必讲要求（证据/口播/权威背书）。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P0:profile")
        if weak_reqs:
            repair_suggestions.append(
                {
                    "priority": "P2",
                    "issue_type": "profile",
                    "issue_summary": f"说服要求偏弱/部分满足：{weak_reqs}。",
                    "repair_direction": "强化偏弱要求的证据呈现与表达充分度。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P2:profile")
        key_findings.append(f"说服要求覆盖：{profile_match.get('match_status') or profile_match.get('overall_status')}。")

        # ---- hec（P1：mismatch；P2：acceptable_deviation/risky_deviation）----
        h_status = hec_match.get("match_status")
        if h_status == "mismatch":
            repair_suggestions.append(
                {
                    "priority": "P1",
                    "issue_type": "hec",
                    "issue_summary": f"HEC 失配（effect 断裂）：{hec_match.get('hec_gap_summary')}。",
                    "repair_direction": "重建 effect_tag 主线，使视频说服核心与商品一致。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P1:hec")
        elif h_status in ("acceptable_deviation", "risky_deviation"):
            repair_suggestions.append(
                {
                    "priority": "P2",
                    "issue_type": "hec",
                    "issue_summary": f"HEC {h_status}：{hec_match.get('hec_gap_summary')}。",
                    "repair_direction": "对齐 hook/cta 标签，减少 HEC 偏移。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P2:hec")
        # PRD-4.2：HEC 回查主次错位（仅命中 secondary 来源而 primary 缺失）→ P1。
        hec_lookup = hec_match.get("source_role_lookup") or {}
        if hec_lookup.get("secondary_hit_without_primary"):
            repair_suggestions.append(
                {
                    "priority": "P1",
                    "issue_type": "hec",
                    "issue_summary": (
                        "HEC Match 回查主次错位：(effect_tag, cta_tag) 仅命中 secondary 来源而 primary 缺失"
                        "（secondary_hit_without_primary）。"
                    ),
                    "repair_direction": "回到 primary source_role 的 (effect_tag, cta_tag) 主链路重建 HEC。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P1:hec:secondary_hit_without_primary")
        key_findings.append(f"HEC 匹配：{h_status}。")

        # ---- slider（P1：mismatch；P2：单轴偏强/偏弱）----
        s_status = slider_match.get("match_status")
        if s_status == "mismatch":
            repair_suggestions.append(
                {
                    "priority": "P1",
                    "issue_type": "slider",
                    "issue_summary": f"Slider 关键轴错位：{slider_match.get('slider_gap_summary')}。",
                    "repair_direction": "按参照人群偏好重做四轴强度配比。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P1:slider")
        elif s_status in ("too_strong", "too_weak", "wrong_direction", "mixed_deviation"):
            repair_suggestions.append(
                {
                    "priority": "P2",
                    "issue_type": "slider",
                    "issue_summary": f"Slider {s_status}：{slider_match.get('slider_gap_summary')}。",
                    "repair_direction": "微调偏离轴强度，贴合参照人群偏好区间。",
                    "related_evidence_spans": [],
                }
            )
            priority_issues.append("P2:slider")
        key_findings.append(f"Slider 匹配：{s_status}。")

        # ---- overall_status（D3：good/needs_minor_repair/needs_major_repair/mismatch）----
        priorities = {s["priority"] for s in repair_suggestions}
        if "P0" in priorities:
            overall_status = "mismatch"
        elif "P1" in priorities:
            overall_status = "needs_major_repair"
        elif "P2" in priorities:
            overall_status = "needs_minor_repair"
        else:
            overall_status = "good"

        return {
            "overall_status": overall_status,
            "key_findings": key_findings,
            "priority_issues": priority_issues,
            "repair_suggestions": repair_suggestions,
        }


__all__ = [
    "VideoDiagnosisEngine",
    "VideoDiagnosisInputError",
    "VideoDiagnosisEnumError",
    "LEGAL_ENUMS",
    "load_audience_slider_preference_dictionary",
    "DEFAULT_SLIDER_DICT_PATH",
]
