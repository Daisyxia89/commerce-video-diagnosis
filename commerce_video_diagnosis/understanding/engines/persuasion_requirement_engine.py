"""说服要求建模引擎（Persuasion Requirement Modeling Engine）—— V3.1 一期。

独立旁路引擎：由 product_diagnoser.py 在商品诊断完成后调用，产出
``persuasion_requirement_profile``。一期严格遵守 PRD1 边界：

- 线上 requirement 仅允许 23 条 active MVP 白名单（PRD1 §8.2 / §12.1.1）；
- ``content_goal`` 由调用侧显式传入，LLM 不推断；非转化目标下 action_gap 输出
  not_applicable（PRD1 §6.2 / §12.1.3）；
- ``category_group`` 仅由 category_group_routing_dictionary 路由，未命中固定回落
  ``unknown`` 且不激活品类扩展 requirement（PRD1 §12.1.4 / §12.1.5）；
- JTBD 未命中模板时回退通用 requirement 并标记 ``fallback_generic``（PRD1 §12.1.6）；
- 不生成 HEC / 不输出 H/E/C 枚举 / 不改 CandidateSet 输入接口。

任何违反硬约束的情况一律 Crash Early，不静默放过。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from core_skill.schemas.protocols import (
    ACTION_GOALS,
    ACTIVE_REQUIREMENT_WHITELIST,
    CONTENT_GOAL_VALUES,
    ActivatedCategoryRequirements,
    CategoryResistance,
    DiagnosisContract,
    MainPersuasionRoute,
    NotApplicableRequirement,
    PersuasionRequirement,
    PersuasionRequirementProfile,
    PrimaryJTBD,
    ProductConversionBarrier,
    RequirementCompletionSchema,
    assert_no_deprecated_persuasion_keys,
)

def _locate_project_root() -> Path:
    """从当前文件向上查找包含 ``core_skill/dictionaries`` 的项目根目录。

    迁移到正式目录 ``commerce-video-diagnosis/commerce_video_diagnosis/understanding/engines/``
    后，引擎不再与字典共享父级，必须显式定位项目根，避免相对路径错位。
    """
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "core_skill" / "dictionaries").is_dir():
            return candidate
    raise FileNotFoundError(
        "未能在任何父级目录找到 core_skill/dictionaries，无法定位 v3.1 字典资源。"
    )


PROJECT_ROOT = _locate_project_root()
DICTIONARY_DIR = PROJECT_ROOT / "core_skill" / "dictionaries"

PROFILE_VERSION = "v3.1"
REQUIREMENT_DICTIONARY_VERSION = "v3.1_active_mvp_23"
CATEGORY_CRITERIA_VERSION = "v3.1_phase1_4groups"
FALLBACK_GROUP = "unknown"

# 视频诊断契约 minimum_required_requirements（PRD1 §11.1）
MINIMUM_REQUIRED_REQUIREMENTS: tuple[str, ...] = ("prove_core_benefit", "provide_visible_result")

# 通用骨干要求：覆盖 7 个 decision_gap 的购买判断主路径，与 JTBD / 品类无关，始终输出。
# 包含 2 条 minimum_required（prove_core_benefit / provide_visible_result）。
BASE_GENERIC_REQUIREMENTS: tuple[str, ...] = (
    "expose_current_pain",
    "prove_core_benefit",
    "provide_visible_result",
    "establish_basic_trust",
    "reduce_trial_risk",
    "prove_current_purchase_reason",
    "clarify_purchase_threshold",
)

SRC_GENERIC = "persuasion_requirement_dictionary"
SRC_JTBD = "JTBD_requirement_template_dictionary"
SRC_CATEGORY = "category_purchase_criteria_dictionary"

_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_PRIORITY = {0: "low", 1: "medium", 2: "high"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"字典文件缺失：{path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class PersuasionRequirementEngine:
    """说服要求建模引擎。"""

    def __init__(self, dictionary_dir: Path | str | None = None) -> None:
        self.dictionary_dir = Path(dictionary_dir) if dictionary_dir else DICTIONARY_DIR
        self._load_dictionaries()
        self._validate_dictionaries()

    # ------------------------------------------------------------------ 加载
    def _load_dictionaries(self) -> None:
        d = self.dictionary_dir
        self.requirement_dict = _load_json(d / "persuasion_requirement_dictionary.json")
        self.jtbd_dict = _load_json(d / "JTBD_requirement_template_dictionary.json")
        self.category_dict = _load_json(d / "category_purchase_criteria_dictionary.json")
        self.routing_dict = _load_json(d / "category_group_routing_dictionary.json")

        # 索引：requirement_id -> 记录
        self.requirement_index: dict[str, dict[str, Any]] = {
            r["requirement_id"]: r for r in self.requirement_dict.get("requirements", [])
        }
        self.active_ids: set[str] = {
            rid for rid, r in self.requirement_index.items() if r.get("status") == "active"
        }
        # JTBD 模板：jtbd_level2 -> template
        self.jtbd_index: dict[str, dict[str, Any]] = {
            t["jtbd_level2"]: t for t in self.jtbd_dict.get("templates", [])
        }
        # 品类购买判断：category_group -> criteria
        self.category_index: dict[str, dict[str, Any]] = {
            g["category_group"]: g for g in self.category_dict.get("category_groups", [])
        }
        # 路由：leaf_category -> route
        self.routing_index: dict[str, dict[str, Any]] = {
            r["leaf_category"]: r for r in self.routing_dict.get("routes", [])
        }

    # ------------------------------------------------------ 启动期字典体检（Crash Early）
    def _validate_dictionaries(self) -> None:
        # 1) active 白名单与协议常量严格一致
        if self.active_ids != set(ACTIVE_REQUIREMENT_WHITELIST):
            missing = set(ACTIVE_REQUIREMENT_WHITELIST) - self.active_ids
            extra = self.active_ids - set(ACTIVE_REQUIREMENT_WHITELIST)
            raise ValueError(
                f"通用字典 active 集合与协议 23 条白名单不一致：缺失 {sorted(missing)}，多余 {sorted(extra)}。"
            )
        # 2) JTBD 模板内 requirement_id 必须命中 active 白名单
        for level2, tpl in self.jtbd_index.items():
            for item in tpl.get("requirements", []):
                rid = item["requirement_id"]
                if rid not in self.active_ids:
                    raise ValueError(
                        f"JTBD 模板 [{level2}] 的 requirement_id={rid} 不在 active 白名单内。"
                    )
        # 3) 品类 derived_requirement_id 必须命中 active 白名单（A-DICT-7 / PRD1 §12.1.8）
        for group, conf in self.category_index.items():
            for crit in conf.get("decision_criteria", []):
                rid = crit["derived_requirement_id"]
                if rid not in self.active_ids:
                    raise ValueError(
                        f"品类 [{group}] criterion={crit.get('criterion_id')} 的 "
                        f"derived_requirement_id={rid} 不在 active 白名单内（A-DICT-7）。"
                    )
        # 4) 路由表 category_group 必须在品类字典中（unknown 除外）
        for leaf, route in self.routing_index.items():
            group = route["category_group"]
            if group not in self.category_index:
                raise ValueError(
                    f"路由表 leaf_category={leaf} 指向未登记 category_group={group}。"
                )

    # ----------------------------------------------------------------- 主入口
    def generate_profile(
        self, product_fact: Mapping[str, Any], content_goal: str = "unknown"
    ) -> dict[str, Any]:
        """生成 persuasion_requirement_profile（返回 dict，已通过协议强校验）。"""
        if not isinstance(product_fact, Mapping):
            raise TypeError("product_fact 必须是 Mapping。")
        assert_no_deprecated_persuasion_keys(dict(product_fact), where="product_fact")

        # —— 0. content_goal 闭集校验（PRD1 §6.2，LLM 不推断；越界 Crash Early）——
        if content_goal not in CONTENT_GOAL_VALUES:
            raise ValueError(
                f"content_goal={content_goal} 不在 9 项合法枚举内（PRD1 §6.2）。"
            )
        is_action_goal = content_goal in ACTION_GOALS

        leaf_category = str(product_fact.get("leaf_category", "") or "")
        jtbd_level1 = str(product_fact.get("jtbd_level1", "") or "功能任务")
        jtbd_level2 = str(product_fact.get("jtbd_level2", "") or "")
        risk_points = list(product_fact.get("risk_points", []) or [])

        # —— 1. 类目路由（仅查表，禁止 LLM 推断；未命中回落 unknown）——
        route = self.routing_index.get(leaf_category)
        if route:
            category_group = route["category_group"]
            routing_confidence = route.get("routing_confidence", "")
        else:
            category_group = FALLBACK_GROUP
            routing_confidence = ""

        # —— 2. JTBD 模板召回 ——
        jtbd_template = self.jtbd_index.get(jtbd_level2)
        jtbd_template_status = "matched" if jtbd_template else "fallback_generic"

        # —— 3. 通用 requirement 基础集 ——
        bucket: dict[str, dict[str, Any]] = {}
        for rid in BASE_GENERIC_REQUIREMENTS:
            self._merge_requirement(bucket, rid, source=SRC_GENERIC)

        # —— 4. JTBD 模板实例化（matched 时）——
        # 字段消费：instantiated_requirement_name / instantiated_success_criteria /
        #         required（false 时仅在 activation_condition 满足时才激活）/
        #         activation_condition（存在时需在商品输入中找到对应证据）。
        if jtbd_template:
            for item in jtbd_template["requirements"]:
                rid = item["requirement_id"]
                required_flag = bool(item.get("required", True))
                activation_condition = item.get("activation_condition")
                # required=False 默认不激活；存在 activation_condition 时需校验通过；
                # required=False 且无 activation_condition → 视为不激活。
                if not required_flag:
                    if not activation_condition:
                        continue
                    if not self._check_activation_condition(rid, activation_condition, product_fact):
                        continue
                # required=True 且带 activation_condition 时，同样需要证据校验，
                # 否则降级为不激活（保持字典语义一致：activation_condition 存在即为硬门槛）。
                elif activation_condition and not self._check_activation_condition(
                    rid, activation_condition, product_fact
                ):
                    continue
                self._merge_requirement(
                    bucket,
                    rid,
                    source=SRC_JTBD,
                    priority_override=item.get("default_priority_override"),
                    rank_override=item.get("default_sequence_rank_override"),
                    instantiated_name=item.get("instantiated_requirement_name"),
                    instantiated_criteria=item.get("instantiated_success_criteria"),
                    jtbd_required=required_flag,
                )

        # —— 5. 品类购买判断激活（命中 group 时）——
        activated_criteria: list[str] = []
        activated_evidence: list[str] = []
        activated_risks: list[str] = []
        if category_group != FALLBACK_GROUP:
            conf = self.category_index.get(category_group, {})
            for crit in conf.get("decision_criteria", []):
                rid = crit["derived_requirement_id"]
                self._merge_requirement(
                    bucket,
                    rid,
                    source=SRC_CATEGORY,
                    priority_override=crit.get("default_priority"),
                    related_criteria=[crit["criterion_id"]],
                    evidence=list(crit.get("evidence_requirements", [])),
                    risks=list(crit.get("risk_points", [])),
                    instantiation=f"围绕「{crit['criterion_name']}」提供品类证据，"
                    f"满足购买判断标准 {crit['criterion_id']}。",
                )
                activated_criteria.append(crit["criterion_id"])
                activated_evidence.extend(crit.get("evidence_requirements", []))
                activated_risks.extend(crit.get("risk_points", []))

        # —— 6. R/P 属性补充：价格 / 信任抗性强化 ——
        self._supplement_rp_requirements(bucket, product_fact)

        # —— 7+8. 动态计算 priority / required / sequence_rank + success_criteria ——
        for rid, rec in bucket.items():
            rec["priority"], rec["required"] = self._compute_priority_required(rec, product_fact)
            rec["sequence_rank"] = self._compute_sequence_rank(rec)
            rec["success_criteria"] = self._build_success_criteria(rec, product_fact, jtbd_level2)

        # —— 9. action_gap 激活断言（非转化目标 → not_applicable）——
        persuasion_records: list[dict[str, Any]] = []
        not_applicable: list[NotApplicableRequirement] = []
        for rid, rec in bucket.items():
            if rec["decision_gap"] == "action_gap" and not is_action_goal:
                not_applicable.append(
                    NotApplicableRequirement(
                        requirement_id=rid,
                        decision_gap="action_gap",
                        reason=f"content_goal={content_goal}，不激活转化类要求。",
                    )
                )
                continue
            persuasion_records.append(rec)

        # —— 10. active 白名单兜底校验（Crash Early）——
        for rec in persuasion_records:
            if rec["requirement_id"] not in self.active_ids:
                raise ValueError(
                    f"输出 requirement_id={rec['requirement_id']} 不在 23 条 active 白名单内。"
                )

        # 排序：sequence_rank 升序，稳定输出
        persuasion_records.sort(key=lambda r: (r["sequence_rank"], r["requirement_id"]))
        not_applicable.sort(key=lambda r: r.requirement_id)

        requirements = [
            PersuasionRequirement(
                requirement_id=rec["requirement_id"],
                requirement_name=rec.get("_instantiated_name") or rec["requirement_name"],
                decision_gap=rec["decision_gap"],
                source=sorted(rec["source"]),
                priority=rec["priority"],
                required=rec["required"],
                sequence_rank=rec["sequence_rank"],
                success_criteria=rec["success_criteria"],
                related_decision_criteria=sorted(set(rec["related_decision_criteria"])),
                required_evidence_requirements=sorted(set(rec["required_evidence_requirements"])),
                risk_points=sorted(set(rec["risk_points"])),
            )
            for rec in persuasion_records
        ]

        # —— 11. 组装 profile + 协议强校验 ——
        profile = PersuasionRequirementProfile(
            profile_version=PROFILE_VERSION,
            content_goal=content_goal,
            category_group=category_group,
            jtbd_template_status=jtbd_template_status,
            requirement_dictionary_version=REQUIREMENT_DICTIONARY_VERSION,
            category_purchase_criteria_version=(
                CATEGORY_CRITERIA_VERSION if category_group != FALLBACK_GROUP else ""
            ),
            main_persuasion_route=self._build_main_route(product_fact, jtbd_level1, jtbd_level2),
            activated_category_requirements=ActivatedCategoryRequirements(
                category_group=category_group,
                routing_confidence=routing_confidence,
                activated_decision_criteria=sorted(set(activated_criteria)),
                activated_evidence_requirements=sorted(set(activated_evidence)),
                activated_risk_points=sorted(set(activated_risks)),
            ),
            persuasion_requirements=requirements,
            not_applicable_requirements=not_applicable,
            diagnosis_contract=DiagnosisContract(
                requirement_completion_schema=RequirementCompletionSchema(
                    minimum_required_requirements=list(MINIMUM_REQUIRED_REQUIREMENTS),
                )
            ),
        )
        return profile.dict()

    # --------------------------------------------------------------- 合并去重
    def _merge_requirement(
        self,
        bucket: dict[str, dict[str, Any]],
        requirement_id: str,
        *,
        source: str,
        priority_override: str | None = None,
        rank_override: int | None = None,
        related_criteria: list[str] | None = None,
        evidence: list[str] | None = None,
        risks: list[str] | None = None,
        instantiation: str | None = None,
        instantiated_name: str | None = None,
        instantiated_criteria: str | None = None,
        jtbd_required: bool | None = None,
    ) -> None:
        base = self.requirement_index.get(requirement_id)
        if base is None:
            raise ValueError(f"requirement_id={requirement_id} 未登记于通用字典。")
        if requirement_id not in self.active_ids:
            raise ValueError(
                f"requirement_id={requirement_id} 非 active，不得进入线上 profile（candidate pool 隔离）。"
            )
        rec = bucket.get(requirement_id)
        if rec is None:
            rec = {
                "requirement_id": requirement_id,
                "requirement_name": base["requirement_name"],
                "decision_gap": base["decision_gap"],
                "source": set(),
                "related_decision_criteria": [],
                "required_evidence_requirements": [],
                "risk_points": [],
                "_default_priority": base.get("default_priority", "medium"),
                "_default_required": bool(base.get("default_required", False)),
                "_default_rank": int(base.get("default_sequence_rank", 30)),
                "_priority_overrides": [],
                "_rank_override": None,
                "_instantiations": [],
                "_instantiated_name": None,
                "_instantiated_criteria": None,
                "_jtbd_required": False,
            }
            bucket[requirement_id] = rec
        rec["source"].add(source)
        if priority_override:
            rec["_priority_overrides"].append(priority_override)
        if rank_override is not None:
            rec["_rank_override"] = rank_override
        if related_criteria:
            rec["related_decision_criteria"].extend(related_criteria)
        if evidence:
            rec["required_evidence_requirements"].extend(evidence)
        if risks:
            rec["risk_points"].extend(risks)
        if instantiation:
            rec["_instantiations"].append(instantiation)
        if instantiated_name and not rec["_instantiated_name"]:
            rec["_instantiated_name"] = instantiated_name
        if instantiated_criteria and not rec["_instantiated_criteria"]:
            rec["_instantiated_criteria"] = instantiated_criteria
        if jtbd_required:
            rec["_jtbd_required"] = True

    # ----------------------------------------------- activation_condition 校验
    # 字典中带 activation_condition 的要求必须在商品输入中找到对应可追溯证据，
    # 才允许激活进入 profile。当前一期覆盖以下要求：
    #   - provide_authority_endorsement：检测 / 认证 / 资质 / 标准 / 专业机构
    _ACTIVATION_KEYWORDS: dict[str, tuple[str, ...]] = {
        "provide_authority_endorsement": (
            "检测", "认证", "资质", "标准", "专业机构", "权威", "背书",
            "实验室", "鉴定", "证书", "ISO", "CE", "FDA", "SGS", "GMP",
            "国标", "GB ", "GB/T", "药监", "药品监督", "国家", "官方",
        ),
    }

    # product_fact 中可能承载证据的字段集合（统一文本化后做关键字命中）。
    _ACTIVATION_FACT_KEYS: tuple[str, ...] = (
        "selling_points",
        "certifications",
        "authority_endorsements",
        "evidence",
        "evidence_chain",
        "title",
        "category",
        "leaf_category",
        "description",
        "brand_assets",
        "trust_attribute",
        "extra_evidence",
        "source_evidence",
    )

    def _check_activation_condition(
        self,
        requirement_id: str,
        activation_condition: str,
        product_fact: Mapping[str, Any],
    ) -> bool:
        keywords = self._ACTIVATION_KEYWORDS.get(requirement_id)
        if not keywords:
            # 未注册的 activation_condition：保持保守策略 → 不激活，避免静默放过。
            return False
        haystack = self._collect_fact_text(product_fact)
        return any(kw in haystack for kw in keywords)

    @staticmethod
    def _collect_fact_text(product_fact: Mapping[str, Any]) -> str:
        chunks: list[str] = []

        def _walk(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                chunks.append(value)
            elif isinstance(value, Mapping):
                for v in value.values():
                    _walk(v)
            elif isinstance(value, (list, tuple, set)):
                for v in value:
                    _walk(v)
            else:
                chunks.append(str(value))

        for key in PersuasionRequirementEngine._ACTIVATION_FACT_KEYS:
            if key in product_fact:
                _walk(product_fact[key])
        return "\n".join(chunks)

    # ----------------------------------------------------------- R/P 属性补充
    def _supplement_rp_requirements(
        self, bucket: dict[str, dict[str, Any]], product_fact: Mapping[str, Any]
    ) -> None:
        price_attr = str(product_fact.get("price_attribute", "") or "")
        trust_attr = str(product_fact.get("trust_attribute", "") or "")
        # 价格抗性：低价/价格敏感 → 补充价格合理性证明
        if any(k in price_attr for k in ("低价", "价格敏感", "敏感", "low")):
            self._merge_requirement(bucket, "prove_price_reasonableness", source=SRC_GENERIC)
        # 信任抗性：白牌/低信任 → 补充来源可信
        if any(k in trust_attr for k in ("白牌", "低信任", "低", "white")):
            self._merge_requirement(bucket, "prove_source_credibility", source=SRC_GENERIC)

    # --------------------------------------------- 动态 priority / required 计算
    def _compute_priority_required(
        self, rec: dict[str, Any], product_fact: Mapping[str, Any]
    ) -> tuple[str, bool]:
        level = _PRIORITY_RANK[rec["_default_priority"]]
        for ov in rec["_priority_overrides"]:
            if ov in _PRIORITY_RANK:
                level = max(level, _PRIORITY_RANK[ov])
        gap = rec["decision_gap"]
        has_risk = bool(product_fact.get("risk_points"))
        trust_attr = str(product_fact.get("trust_attribute", "") or "")
        price_attr = str(product_fact.get("price_attribute", "") or "")
        # 事实强化：商品存在风险点 → 风险/证据类要求上调
        if has_risk and gap in ("risk_gap", "proof_gap"):
            level = max(level, _PRIORITY_RANK["high"])
        # 白牌/低信任 → 信任类上调
        if gap == "trust_gap" and any(k in trust_attr for k in ("白牌", "低信任", "低")):
            level = max(level, _PRIORITY_RANK["high"])
        # 价格敏感 → 价格合理性上调
        if rec["requirement_id"] == "prove_price_reasonableness" and any(
            k in price_attr for k in ("低价", "敏感")
        ):
            level = max(level, _PRIORITY_RANK["high"])
        # 多来源命中（JTBD + 品类同时要求）→ 上调
        if len(rec["source"]) >= 2:
            level = max(level, _PRIORITY_RANK["high"])
        priority = _RANK_PRIORITY[level]
        required = (
            rec["_default_required"]
            or rec.get("_jtbd_required", False)
            or priority == "high"
            or SRC_CATEGORY in rec["source"]
        )
        return priority, required

    # -------------------------------------------- 动态 sequence_rank 计算（限带内）
    def _compute_sequence_rank(self, rec: dict[str, Any]) -> int:
        base = rec["_rank_override"] if rec["_rank_override"] is not None else rec["_default_rank"]
        band_lo = (base // 10) * 10
        band_hi = band_lo + 9
        delta = 0
        if rec["priority"] == "high":
            delta -= 1
        elif rec["priority"] == "low":
            delta += 1
        # 多来源强化 → 略微提前
        if len(rec["source"]) >= 2:
            delta -= 1
        rank = base + delta
        rank = max(band_lo, min(band_hi, rank))
        return max(10, min(59, rank))

    # --------------------------------------------------- success_criteria 实例化
    def _build_success_criteria(
        self, rec: dict[str, Any], product_fact: Mapping[str, Any], jtbd_level2: str
    ) -> str:
        # 优先使用 JTBD 字典中的 instantiated_success_criteria 作为主体描述；
        # 未实例化时回退至通用字典 definition。
        primary = rec.get("_instantiated_criteria") or self.requirement_index[
            rec["requirement_id"]
        ].get("definition", "")
        parts = [primary]
        if rec["_instantiations"]:
            parts.append(rec["_instantiations"][0])
        selling_points = [str(s) for s in (product_fact.get("selling_points") or [])][:2]
        context_bits: list[str] = []
        if jtbd_level2:
            context_bits.append(f"JTBD「{jtbd_level2}」")
        if selling_points:
            context_bits.append("卖点：" + "、".join(selling_points))
        if rec["risk_points"]:
            context_bits.append("需回应风险：" + "、".join(sorted(set(rec["risk_points"]))[:2]))
        if context_bits:
            parts.append("结合" + "；".join(context_bits) + "，让用户对该判断点形成确定结论。")
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------ main_persuasion_route
    def _build_main_route(
        self, product_fact: Mapping[str, Any], jtbd_level1: str, jtbd_level2: str
    ) -> MainPersuasionRoute:
        cognitive = str(product_fact.get("cognitive_attribute", "") or "未知认知")
        frequency = str(product_fact.get("frequency_attribute", "") or "未知频次")
        trust = str(product_fact.get("trust_attribute", "") or "未知信任")
        price = str(product_fact.get("price_attribute", "") or "未知价格")
        return MainPersuasionRoute(
            primary_jtbd=PrimaryJTBD(level1=jtbd_level1, level2=jtbd_level2),
            category_resistance=CategoryResistance(
                rule=f"{cognitive} × {frequency}",
                summary=f"在{cognitive}、{frequency}的品类认知下组织购买判断主路径。",
            ),
            product_conversion_barrier=ProductConversionBarrier(
                rule=f"{trust} × {price}",
                summary=f"在{trust}信任存量、{price}价格水位下处理转化阻力。",
            ),
        )


@lru_cache(maxsize=1)
def get_default_engine() -> PersuasionRequirementEngine:
    return PersuasionRequirementEngine()


def build_persuasion_requirement_profile(
    product_fact: Mapping[str, Any], content_goal: str = "unknown"
) -> dict[str, Any]:
    """便捷入口：使用默认字典目录构建 profile。"""
    return get_default_engine().generate_profile(product_fact, content_goal)
