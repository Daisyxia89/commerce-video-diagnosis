from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from commerce_video_diagnosis.understanding.schemas.protocols import (
    CTAResolution,
    CandidateSet,
    HookSoftConstraintContract,
    ProductECSkeleton,
    ProductHEC,
    SoftConstraintResult,
)


BANNED_RAW_FACT_KEYS = {
    "target_people",
    "core_selling_point",
    "product_name",
    "leaf_category",
    "shop_name",
    "price",
    "differentiator",
    "brand_name",
}

BANNED_PRESENTATION_KEYS = {
    "目标人群入口",
    "内容场景",
    "镜头玩法",
    "话术风格",
    "scene",
    "shot",
    "camera",
    "tone",
    "style",
    "script_text",
    "storyboard",
}

LEGACY_HEC_KEYS = {"hook", "effect", "cta"}
HOOK_SOFT_CONSTRAINT_ALLOWED_KEYS = {
    "trigger_cta_tags",
    "required_effect_capabilities_all",
    "unmet_risk_flag",
}
CTA_LABEL_FALLBACK = {
    "C1": "C1 利益/价格逼单",
    "C2": "C2 福利/保障机制",
    "C3": "C3 指令行动",
    "C4": "C4 人群/场景总结",
    "C5": "C5 效果留白/情绪定格",
}


@dataclass(slots=True)
class ProductVariantAssembler:
    """模块 4：先组装 Product_EC_Skeletons，再生成 Product_HECs。"""

    def assemble_product_ec_skeletons(self, candidate_set: CandidateSet | dict[str, Any]) -> list[dict[str, Any]]:
        candidate_payload = self._coerce_candidate_set(candidate_set)
        self._assert_candidate_set_boundary(candidate_payload)
        effect_list = self._normalize_effect_candidates(candidate_payload.get("effect_list"))
        cta_list = self._normalize_cta_candidates(candidate_payload.get("cta_list"))
        available_cta_tags = {item["cta_tag"] for item in cta_list}
        cta_input_rank = self._build_cta_input_rank(cta_list)

        product_ec_skeletons: list[dict[str, Any]] = []
        seen_resolution_meta: dict[tuple[str, str], dict[str, Any]] = {}
        for effect, cta in product(effect_list, cta_list):
            resolved_cta_tag = cta["cta_tag"]
            resolution_type = "direct"
            reason_codes: list[str] = []
            required_any = cta["required_effect_capabilities_any"]
            effect_capabilities = effect["completion_capabilities"]
            if required_any and not any(capability in effect_capabilities for capability in required_any):
                resolved_cta_tag = self._resolve_fallback_cta(cta["fallback_priority"], available_cta_tags, cta["cta_tag"])
                resolution_type = "downgrade"
                reason_codes = [
                    "passive_close_admission_failed",
                    f"missing_any:{'|'.join(required_any)}",
                    f"fallback_to:{resolved_cta_tag}",
                ]
            combo = (effect["effect_tag"], resolved_cta_tag)
            current_rank = cta_input_rank[cta["cta_tag"]]
            existing_meta = seen_resolution_meta.get(combo)
            if existing_meta is not None:
                self._assert_parallel_downgrade_priority(
                    combo=combo,
                    existing_meta=existing_meta,
                    current_requested_cta_tag=cta["cta_tag"],
                    current_rank=current_rank,
                )
                continue
            skeleton = ProductECSkeleton(
                schema_version=str(candidate_payload.get("schema_version") or "v0.5"),
                effect_tag=effect["effect_tag"],
                cta_tag=resolved_cta_tag,
                effect_label=effect["label"],
                cta_label=self._resolve_cta_label(resolved_cta_tag, cta_list),
                effect_capabilities_snapshot=list(effect_capabilities),
                cta_resolution=CTAResolution(
                    requested_cta_tag=cta["cta_tag"],
                    resolved_cta_tag=resolved_cta_tag,
                    resolution_type=resolution_type,
                    reason_codes=reason_codes,
                ).to_dict(),
            ).to_dict()
            self._assert_product_ec_skeleton_boundary(skeleton)
            product_ec_skeletons.append(skeleton)
            seen_resolution_meta[combo] = {
                "requested_cta_tag": cta["cta_tag"],
                "rank": current_rank,
            }

        if not product_ec_skeletons:
            raise ValueError("模块 4.1 组装后无合法 Product_EC_Skeletons。")
        return product_ec_skeletons

    def assemble_product_hecs(
        self,
        jtbd: str,
        product_ec_skeletons: list[ProductECSkeleton | dict[str, Any]],
        core_h_list: list[dict[str, Any]],
        *,
        product_id: str = "",
    ) -> list[dict[str, Any]]:
        normalized_skeletons = self._normalize_product_ec_skeletons(product_ec_skeletons)
        normalized_h_list = self._normalize_hook_candidates(core_h_list)

        product_hecs: list[dict[str, Any]] = []
        variant_index = 1
        for hook, skeleton in product(normalized_h_list, normalized_skeletons):
            candidate_variant = {
                "hook_tag": hook["hook_tag"],
                "effect_tag": skeleton["effect_tag"],
                "cta_tag": skeleton["cta_tag"],
            }
            if self._should_prune(jtbd, candidate_variant):
                continue
            soft_constraint_results, risk_flags = self._evaluate_hook_soft_constraints(hook=hook, skeleton=skeleton)
            activation_tags = self._build_activation_tags(
                hook_tag=hook["hook_tag"],
                effect_tag=skeleton["effect_tag"],
                cta_tag=skeleton["cta_tag"],
            )
            variant = ProductHEC(
                hook_tag=hook["hook_tag"],
                effect_tag=skeleton["effect_tag"],
                cta_tag=skeleton["cta_tag"],
                variant_id=f"{product_id or jtbd}-v{variant_index}",
                schema_version=skeleton["schema_version"],
                hook_label=hook["label"],
                effect_label=skeleton["effect_label"],
                cta_label=skeleton["cta_label"],
                activation_tags=activation_tags,
                risk_flags=risk_flags,
                soft_constraint_results=soft_constraint_results,
                route_tags=list(activation_tags),
            ).to_dict()
            self._assert_product_hec_boundary(variant)
            product_hecs.append(variant)
            variant_index += 1

        if not product_hecs:
            raise ValueError("模块 4.2 装配后无合法 Product_HECs。")
        return product_hecs

    def assemble(self, jtbd: str, candidate_set: CandidateSet | dict[str, Any], *, product_id: str = "") -> list[dict[str, Any]]:
        candidate_payload = self._coerce_candidate_set(candidate_set)
        product_ec_skeletons = self.assemble_product_ec_skeletons(candidate_payload)
        core_h_list = candidate_payload["h_list"]
        return self.assemble_product_hecs(jtbd, product_ec_skeletons, core_h_list, product_id=product_id)

    def _coerce_candidate_set(self, candidate_set: CandidateSet | dict[str, Any]) -> dict[str, Any]:
        if isinstance(candidate_set, CandidateSet):
            return candidate_set.to_dict()
        if not isinstance(candidate_set, dict):
            raise ValueError("模块 4 仅接收 CandidateSet 对象。")
        return candidate_set

    def _assert_candidate_set_boundary(self, candidate_set: dict[str, Any]) -> None:
        leaked_keys = BANNED_RAW_FACT_KEYS.intersection(candidate_set.keys())
        if leaked_keys:
            leaked = ", ".join(sorted(leaked_keys))
            raise ValueError(f"模块 4 禁止透传原始商品事实：{leaked}")
        presentation_keys = BANNED_PRESENTATION_KEYS.intersection(candidate_set.keys())
        if presentation_keys:
            leaked = ", ".join(sorted(presentation_keys))
            raise ValueError(f"模块 4 禁止透传表现层字段：{leaked}")
        required_fields = ("h_list", "effect_list", "cta_list", "jtbd", "persuasion_route", "r_rule", "p_rule", "task_domain")
        missing = [field for field in required_fields if field not in candidate_set]
        if missing:
            raise ValueError(f"CandidateSet 缺少协议字段：{', '.join(missing)}")

    def _normalize_effect_candidates(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("CandidateSet.effect_list 必须是非空列表。")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"CandidateSet.effect_list[{index}] 必须是对象。")
            code = str(item.get("effect_tag") or item.get("code") or "").strip().upper()
            label = str(item.get("label") or item.get("effect_label") or "").strip()
            capabilities = item.get("completion_capabilities")
            reason_codes = item.get("completion_reason_codes")
            if not code or not label:
                raise ValueError(f"CandidateSet.effect_list[{index}] 缺少 effect_tag/label。")
            if not isinstance(capabilities, list):
                raise ValueError(f"CandidateSet.effect_list[{index}] 缺少 completion_capabilities。")
            if not isinstance(reason_codes, list):
                raise ValueError(f"CandidateSet.effect_list[{index}] 缺少 completion_reason_codes。")
            normalized.append(
                {
                    "effect_tag": code,
                    "label": label,
                    "completion_capabilities": [str(cap).strip() for cap in capabilities if str(cap).strip()],
                    "completion_reason_codes": [str(reason).strip() for reason in reason_codes if str(reason).strip()],
                }
            )
        return normalized

    def _normalize_cta_candidates(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("CandidateSet.cta_list 必须是非空列表。")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"CandidateSet.cta_list[{index}] 必须是对象。")
            code = str(item.get("cta_tag") or item.get("code") or "").strip().upper()
            label = str(item.get("label") or item.get("cta_label") or "").strip()
            close_strength = str(item.get("close_strength") or "").strip()
            required_any = item.get("required_effect_capabilities_any")
            fallback_priority = item.get("fallback_priority")
            if not code or not label:
                raise ValueError(f"CandidateSet.cta_list[{index}] 缺少 cta_tag/label。")
            if close_strength not in {"active_push", "passive_close"}:
                raise ValueError(f"CandidateSet.cta_list[{index}].close_strength 非法。")
            if not isinstance(required_any, list):
                raise ValueError(f"CandidateSet.cta_list[{index}] 缺少 required_effect_capabilities_any。")
            if not isinstance(fallback_priority, list):
                raise ValueError(f"CandidateSet.cta_list[{index}] 缺少 fallback_priority。")
            normalized.append(
                {
                    "cta_tag": code,
                    "label": label,
                    "close_strength": close_strength,
                    "required_effect_capabilities_any": [str(cap).strip() for cap in required_any if str(cap).strip()],
                    "fallback_priority": [str(tag).strip().upper() for tag in fallback_priority if str(tag).strip()],
                }
            )
        return normalized

    def _normalize_hook_candidates(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("CandidateSet.h_list 必须是非空列表。")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"CandidateSet.h_list[{index}] 必须是对象。")
            code = str(item.get("hook_tag") or item.get("code") or "").strip().upper()
            label = str(item.get("label") or item.get("hook_label") or "").strip()
            contract = item.get("soft_constraint_contract")
            if not code or not label:
                raise ValueError(f"CandidateSet.h_list[{index}] 缺少 hook_tag/label。")
            if code in {"H5", "H6", "H7"} and not isinstance(contract, dict):
                raise ValueError(f"CandidateSet.h_list[{index}] 缺少 soft_constraint_contract。")
            if code in {"H1", "H2", "H3", "H4"} and contract is not None:
                raise ValueError(f"CandidateSet.h_list[{index}] 不允许携带 soft_constraint_contract。")
            normalized.append(
                {
                    "hook_tag": code,
                    "label": label,
                    "soft_constraint_contract": self._normalize_hook_soft_constraint_contract(
                        contract,
                        field_name=f"CandidateSet.h_list[{index}].soft_constraint_contract",
                    )
                    if isinstance(contract, dict)
                    else None,
                }
            )
        return normalized

    def _normalize_hook_soft_constraint_contract(self, value: Any, *, field_name: str) -> dict[str, Any]:
        payload = value.to_dict() if isinstance(value, HookSoftConstraintContract) else value
        if not isinstance(payload, dict):
            raise ValueError(f"{field_name} 必须是对象。")
        extra_keys = set(payload.keys()) - HOOK_SOFT_CONSTRAINT_ALLOWED_KEYS
        if extra_keys:
            raise ValueError(f"{field_name} 检测到污染字段注入：{', '.join(sorted(extra_keys))}。")
        trigger_cta_tags = payload.get("trigger_cta_tags")
        required_all = payload.get("required_effect_capabilities_all")
        unmet_risk_flag = str(payload.get("unmet_risk_flag") or "").strip()
        if not isinstance(trigger_cta_tags, list) or not trigger_cta_tags:
            raise ValueError(f"{field_name}.trigger_cta_tags 必须是非空列表。")
        if not isinstance(required_all, list) or not required_all:
            raise ValueError(f"{field_name}.required_effect_capabilities_all 必须是非空列表。")
        if not unmet_risk_flag:
            raise ValueError(f"{field_name}.unmet_risk_flag 不能为空。")
        return {
            "trigger_cta_tags": [str(tag).strip().upper() for tag in trigger_cta_tags if str(tag).strip()],
            "required_effect_capabilities_all": [str(cap).strip() for cap in required_all if str(cap).strip()],
            "unmet_risk_flag": unmet_risk_flag,
        }

    def _build_cta_input_rank(self, cta_list: list[dict[str, Any]]) -> dict[str, int]:
        return {item["cta_tag"]: index for index, item in enumerate(cta_list)}

    def _assert_parallel_downgrade_priority(
        self,
        *,
        combo: tuple[str, str],
        existing_meta: dict[str, Any],
        current_requested_cta_tag: str,
        current_rank: int,
    ) -> None:
        existing_rank = existing_meta["rank"]
        if current_rank < existing_rank:
            effect_tag, resolved_cta_tag = combo
            raise ValueError(
                "模块 4.1 并行 CTA 降级去重顺序异常："
                f"effect_tag={effect_tag}, resolved_cta_tag={resolved_cta_tag} 应保留更早出现的 requested_cta_tag；"
                f"当前 {current_requested_cta_tag} 的输入顺序早于已保留的 {existing_meta['requested_cta_tag']}。"
            )

    def _resolve_fallback_cta(self, fallback_priority: list[str], available_cta_tags: set[str], requested_cta_tag: str) -> str:
        for candidate in fallback_priority:
            if candidate in available_cta_tags:
                return candidate
        raise ValueError(f"CTA {requested_cta_tag} 准入失败后无可用降级目标。")

    def _resolve_cta_label(self, cta_tag: str, cta_list: list[dict[str, Any]]) -> str:
        for item in cta_list:
            if item["cta_tag"] == cta_tag:
                return item["label"]
        return CTA_LABEL_FALLBACK.get(cta_tag, cta_tag)

    def _normalize_product_ec_skeletons(
        self,
        product_ec_skeletons: list[ProductECSkeleton | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not product_ec_skeletons:
            raise ValueError("模块 4.2 输入缺少 Product_EC_Skeletons。")

        normalized: list[dict[str, Any]] = []
        for index, skeleton in enumerate(product_ec_skeletons):
            payload = skeleton.to_dict() if isinstance(skeleton, ProductECSkeleton) else skeleton
            if not isinstance(payload, dict):
                raise ValueError(f"模块 4.2 输入的 Product_EC_Skeletons[{index}] 必须是对象。")
            leaked_legacy_keys = LEGACY_HEC_KEYS.intersection(payload.keys())
            if leaked_legacy_keys:
                raise ValueError(
                    f"模块 4.2 输入污染：Product_EC_Skeletons[{index}] 禁止旧版 HEC 字段：{', '.join(sorted(leaked_legacy_keys))}。"
                )
            hook_tag = str(payload.get("hook_tag") or "").strip().upper()
            if hook_tag:
                raise ValueError(f"模块 4.2 输入污染：Product_EC_Skeletons[{index}] 不允许携带 Hook。")
            effect_tag = str(payload.get("effect_tag") or "").strip().upper()
            cta_tag = str(payload.get("cta_tag") or "").strip().upper()
            effect_label = str(payload.get("effect_label") or "").strip()
            cta_label = str(payload.get("cta_label") or "").strip()
            capabilities = payload.get("effect_capabilities_snapshot")
            cta_resolution = payload.get("cta_resolution")
            if not effect_tag or not cta_tag:
                raise ValueError(f"模块 4.2 输入的 Product_EC_Skeletons[{index}] 缺少 effect/cta 标签。")
            if not effect_label or not cta_label:
                raise ValueError(f"模块 4.2 输入的 Product_EC_Skeletons[{index}] 缺少 effect/cta 标签文案。")
            if not isinstance(capabilities, list):
                raise ValueError(f"模块 4.2 输入的 Product_EC_Skeletons[{index}] 缺少 effect_capabilities_snapshot。")
            normalized.append(
                {
                    "schema_version": str(payload.get("schema_version") or "v0.5"),
                    "effect_tag": effect_tag,
                    "cta_tag": cta_tag,
                    "effect_label": effect_label,
                    "cta_label": cta_label,
                    "effect_capabilities_snapshot": [str(cap).strip() for cap in capabilities if str(cap).strip()],
                    "cta_resolution": self._normalize_cta_resolution(
                        cta_resolution,
                        field_name=f"Product_EC_Skeletons[{index}].cta_resolution",
                    ),
                }
            )
        return normalized

    def _normalize_cta_resolution(self, value: Any, *, field_name: str) -> dict[str, Any]:
        payload = value.to_dict() if isinstance(value, CTAResolution) else value
        if not isinstance(payload, dict):
            raise ValueError(f"{field_name} 必须是对象。")
        requested = str(payload.get("requested_cta_tag") or "").strip().upper()
        resolved = str(payload.get("resolved_cta_tag") or "").strip().upper()
        resolution_type = str(payload.get("resolution_type") or "").strip()
        reason_codes = payload.get("reason_codes")
        if not requested or not resolved:
            raise ValueError(f"{field_name} 缺少 requested/resolved cta。")
        if resolution_type not in {"direct", "downgrade"}:
            raise ValueError(f"{field_name}.resolution_type 非法。")
        if not isinstance(reason_codes, list):
            raise ValueError(f"{field_name}.reason_codes 必须是列表。")
        if requested == resolved and resolution_type != "direct":
            raise ValueError(f"{field_name} requested_cta_tag == resolved_cta_tag 时必须为 direct。")
        if requested != resolved and resolution_type != "downgrade":
            raise ValueError(f"{field_name} requested_cta_tag != resolved_cta_tag 时必须为 downgrade。")
        return {
            "requested_cta_tag": requested,
            "resolved_cta_tag": resolved,
            "resolution_type": resolution_type,
            "reason_codes": [str(reason).strip() for reason in reason_codes if str(reason).strip()],
        }

    def _evaluate_hook_soft_constraints(self, *, hook: dict[str, Any], skeleton: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        contract = hook.get("soft_constraint_contract")
        if not contract or skeleton["cta_tag"] not in set(contract["trigger_cta_tags"]):
            return [], []
        required_capabilities = list(contract["required_effect_capabilities_all"])
        capability_snapshot = set(skeleton["effect_capabilities_snapshot"])
        missing_capabilities = [item for item in required_capabilities if item not in capability_snapshot]
        if missing_capabilities:
            risk_flag = contract["unmet_risk_flag"]
            result = SoftConstraintResult(
                rule_id=f"{hook['hook_tag']}_soft_constraint",
                status="risk_marked",
                required_capabilities=required_capabilities,
                missing_capabilities=missing_capabilities,
                risk_flag=risk_flag,
            ).to_dict()
            return [result], [risk_flag]
        result = SoftConstraintResult(
            rule_id=f"{hook['hook_tag']}_soft_constraint",
            status="satisfied",
            required_capabilities=required_capabilities,
            missing_capabilities=[],
            risk_flag=None,
        ).to_dict()
        return [result], []

    def _should_prune(self, jtbd: str, variant: dict[str, Any]) -> bool:
        cta_tag = str(variant.get("cta_tag") or "").strip().upper()
        jtbd_text = str(jtbd or "").strip()
        if "缺陷修复/冲突消除" in jtbd_text and cta_tag == "C5":
            return True
        return False

    def _build_activation_tags(self, *, hook_tag: str, effect_tag: str, cta_tag: str) -> list[str]:
        activation_tags: list[str] = []
        if hook_tag == "H5" or effect_tag in {"E1", "E2"}:
            activation_tags.append("需强测评/打假人设激活")
        if hook_tag in {"H6", "H7"} or cta_tag == "C4":
            activation_tags.append("需特定人群场景共鸣激活")
        return activation_tags

    def _assert_product_ec_skeleton_boundary(self, skeleton: dict[str, Any]) -> None:
        leaked_legacy_keys = LEGACY_HEC_KEYS.intersection(skeleton.keys())
        if leaked_legacy_keys:
            raise ValueError(f"模块 4.1 输出越界，禁止旧版 HEC 字段：{', '.join(sorted(leaked_legacy_keys))}")
        if skeleton.get("hook_tag"):
            raise ValueError("模块 4.1 输出越界：Product_EC_Skeletons 不允许包含 Hook。")
        leaked_keys = BANNED_PRESENTATION_KEYS.intersection(skeleton.keys())
        if leaked_keys:
            raise ValueError(f"模块 4.1 输出越界，禁止表现层字段：{', '.join(sorted(leaked_keys))}")
        if not isinstance(skeleton.get("effect_capabilities_snapshot"), list):
            raise ValueError("模块 4.1 输出越界：缺少 effect_capabilities_snapshot。")
        self._normalize_cta_resolution(skeleton.get("cta_resolution"), field_name="Product_EC_Skeleton.cta_resolution")

    def _assert_product_hec_boundary(self, variant: dict[str, Any]) -> None:
        legacy_hec_keys = LEGACY_HEC_KEYS.intersection(variant.keys())
        if legacy_hec_keys:
            raise ValueError(f"模块 4.2 输出越界，禁止旧版 HEC 字段：{', '.join(sorted(legacy_hec_keys))}")
        for key in BANNED_PRESENTATION_KEYS:
            if key in variant:
                raise ValueError(f"模块 4.2 输出越界，禁止表现层字段：{key}")
        if not isinstance(variant.get("activation_tags"), list):
            raise ValueError("模块 4.2 输出越界：activation_tags 必须是列表。")
        if not isinstance(variant.get("risk_flags"), list):
            raise ValueError("模块 4.2 输出越界：risk_flags 必须是列表。")
        soft_constraint_results = variant.get("soft_constraint_results")
        if not isinstance(soft_constraint_results, list):
            raise ValueError("模块 4.2 输出越界：soft_constraint_results 必须是列表。")
        for index, item in enumerate(soft_constraint_results):
            payload = item.to_dict() if isinstance(item, SoftConstraintResult) else item
            if not isinstance(payload, dict):
                raise ValueError(f"soft_constraint_results[{index}] 必须是对象。")
            status = str(payload.get("status") or "").strip()
            if status not in {"satisfied", "risk_marked"}:
                raise ValueError(f"soft_constraint_results[{index}].status 非法。")
            risk_flag = payload.get("risk_flag")
            if status == "risk_marked" and not risk_flag:
                raise ValueError(f"soft_constraint_results[{index}] 命中风险时必须填写 risk_flag。")
            if status == "satisfied" and risk_flag:
                raise ValueError(f"soft_constraint_results[{index}] satisfied 状态不得携带 risk_flag。")
