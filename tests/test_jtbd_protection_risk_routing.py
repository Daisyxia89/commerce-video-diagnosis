"""驱蚊液等功能型风险规避商品的 JTBD 路由回归测试。

覆盖三处修复，确保功能型风险规避商品（如驱蚊液）在
`difference_type=自身卖点陈述` 时不再被错误降级为「生存/运转维系」：

- P0 `_is_physical_safety_fact`：物理安全前置词表补入防虫/防护类风险词。
- P1 `_supports_maintenance_task`：`自身卖点陈述` 不再单独充分支撑维持任务，
  叠加“无任何风险/防护语义”的硬约束。
- P2 桥接层：防护/防虫/防晒/安全等风险类目命中且证据含风险锚点时，
  自动注入隐式对比对象，不直接降级为纯「自身卖点陈述」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DiagnosticInput,
    DifferentiatorEvidence,
    Module1Output,
    ProductDiagnosisEngine,
    StructuredDifferentiator,
)


def _make_engine() -> ProductDiagnosisEngine:
    return ProductDiagnosisEngine()


def _make_module1(
    *,
    leaf_category: str,
    product_name: str,
    core_selling_point: str,
    difference_type: str = "自身卖点陈述",
) -> Module1Output:
    differentiator = StructuredDifferentiator(
        comparison_object="",
        comparison_object_evidence_type="null",
        difference_domain="functional",
        difference_type=difference_type,
        conclusion="占位结论",
        evidence_chain=[DifferentiatorEvidence("商品信息", core_selling_point)],
        summary="占位结论",
    )
    return Module1Output(
        leaf_category=leaf_category,
        shop_name="示例店铺",
        product_name=product_name,
        price="39",
        core_selling_point=core_selling_point,
        core_selling_point_source="caller_provided.core_selling_points",
        target_people="户外人群",
        differentiator=differentiator,
    )


# ---------------------------------------------------------------------------
# P0：物理安全前置词表覆盖防虫/防护类风险词
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "驱蚊液 有效驱避蚊虫",
        "防止蚊虫叮咬，避免红肿瘙痒",
        "婴幼儿防虫喷雾，降低虫媒风险",
        "户外驱蚊，防蚊更安心",
        "敏感肌易过敏，舒缓红肿",
    ],
)
def test_p0_physical_safety_fact_covers_anti_insect_protection_words(text: str) -> None:
    engine = _make_engine()
    assert engine._is_physical_safety_fact(text) is True


@pytest.mark.unit
def test_p0_legacy_physical_safety_words_still_hit() -> None:
    engine = _make_engine()
    # 回归：原有显性伤害词不能因为词表迁移而丢失。
    for legacy in ("烫伤", "触电", "晒伤", "割伤", "漏电"):
        assert engine._is_physical_safety_fact(f"使用不当可能{legacy}") is True


@pytest.mark.unit
def test_p0_non_risk_text_not_misjudged() -> None:
    engine = _make_engine()
    assert engine._is_physical_safety_fact("快速蓬松定型，无胶感不粘腻") is False


# ---------------------------------------------------------------------------
# P1：自身卖点陈述不再单独支撑维持任务
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_p1_self_statement_with_risk_semantic_blocks_maintenance() -> None:
    engine = _make_engine()
    module1 = _make_module1(
        leaf_category="驱蚊液",
        product_name="户外驱蚊液",
        core_selling_point="有效驱避蚊虫",
    )
    text = "驱蚊液 有效驱避蚊虫 防止蚊虫叮咬 避免红肿瘙痒"
    assert engine._supports_maintenance_task(module1, text) is False


@pytest.mark.unit
def test_p1_self_statement_with_protection_semantic_blocks_maintenance() -> None:
    engine = _make_engine()
    module1 = _make_module1(
        leaf_category="防护手套",
        product_name="作业防护手套",
        core_selling_point="防割保护双手",
    )
    text = "防护手套 防割 保护双手 避免割伤"
    assert engine._supports_maintenance_task(module1, text) is False


@pytest.mark.unit
def test_p1_self_statement_without_risk_still_supports_maintenance() -> None:
    engine = _make_engine()
    # 回归：无风险/防护语义、且非食品/日用/效率语义时，自身卖点陈述仍可回落维持任务。
    module1 = _make_module1(
        leaf_category="发型定型喷雾",
        product_name="蓬松定型喷雾",
        core_selling_point="快速蓬松定型",
    )
    text = "定型喷雾 快速蓬松定型 无胶感不粘腻"
    assert engine._supports_maintenance_task(module1, text) is True


@pytest.mark.unit
def test_p1_food_maintenance_unaffected() -> None:
    engine = _make_engine()
    module1 = _make_module1(
        leaf_category="坚果",
        product_name="每日坚果",
        core_selling_point="配料干净",
    )
    text = "每日坚果 日常补给 口粮 配料干净"
    assert engine._supports_maintenance_task(module1, text) is True


# ---------------------------------------------------------------------------
# P0 + P1 联合：驱蚊液最终不落「生存/运转维系」
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_p0_p1_repellent_fallback_routes_to_physical_safety() -> None:
    engine = _make_engine()
    module1 = _make_module1(
        leaf_category="驱蚊液",
        product_name="户外驱蚊液",
        core_selling_point="有效驱避蚊虫，防止蚊虫叮咬",
    )
    task = engine._fallback_functional_task(module1)
    assert task == "物理安全与风险规避"
    assert task != "生存/运转维系"


# ---------------------------------------------------------------------------
# P2：桥接层隐式对比对象注入
# ---------------------------------------------------------------------------
def _patch_llms(engine: ProductDiagnosisEngine) -> None:
    def _fake_generate(**kwargs: object) -> str:
        # 自身卖点陈述路径返回不含相对语义的纯陈述，避免误触自身卖点残留断言；
        # 其余（如注入后的风险降低）返回带风险锚点的相对结论。
        if kwargs.get("difference_type") == "自身卖点陈述":
            return "主打有效驱避蚊虫的核心卖点"
        return "相对未使用任何驱避产品，降低蚊虫叮咬风险"

    engine.differentiator_conclusion_llm.generate = _fake_generate  # type: ignore[assignment]
    engine.differentiator_semantic_judge_llm.judge = (  # type: ignore[assignment]
        lambda **kwargs: {
            "supports_difference_type": True,
            "supports_conclusion": True,
            "reason": "ok",
        }
    )


@pytest.mark.unit
def test_p2_protection_category_injects_implicit_comparison() -> None:
    engine = _make_engine()
    _patch_llms(engine)
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="户外旗舰店",
        product_name="户外驱蚊液",
        price="39",
        core_selling_point="有效驱避蚊虫",
        differentiator="",
        bridge_difference_domain="functional",
        bridge_difference_type="自身卖点陈述",
        bridge_source_evidence=["驱蚊液有效驱避蚊虫，防止蚊虫叮咬"],
        bridge_comparison_object="",
        bridge_comparison_object_evidence_type="null",
    )
    result = engine._normalize_differentiator(payload)
    assert result.difference_type == "风险降低"
    assert result.difference_domain == "functional"
    assert result.comparison_object == "同类旧方案"
    assert result.comparison_object_evidence_type == "jtbd_inferred"


@pytest.mark.unit
def test_p2_non_protection_category_keeps_self_statement() -> None:
    engine = _make_engine()
    _patch_llms(engine)
    payload = DiagnosticInput(
        leaf_category="发型定型喷雾",
        shop_name="美发旗舰店",
        product_name="蓬松定型喷雾",
        price="59",
        core_selling_point="快速蓬松定型",
        differentiator="",
        bridge_difference_domain="functional",
        bridge_difference_type="自身卖点陈述",
        bridge_source_evidence=["一喷一吹快速做出蓬松发型"],
        bridge_comparison_object="",
        bridge_comparison_object_evidence_type="null",
    )
    result = engine._normalize_differentiator(payload)
    # 非防护风险类目：保持原协议，自身卖点陈述清空 comparison_object。
    assert result.difference_type == "自身卖点陈述"
    assert result.comparison_object == ""
    assert result.comparison_object_evidence_type == "null"


@pytest.mark.unit
def test_p2_protection_category_without_risk_anchor_keeps_self_statement() -> None:
    engine = _make_engine()
    _patch_llms(engine)
    # 类目命中防护风险词，但证据中无风险/防护锚点：不注入，保持原协议，避免下游误判。
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="户外旗舰店",
        product_name="户外驱蚊液",
        price="39",
        core_selling_point="清新草本香味",
        differentiator="",
        bridge_difference_domain="functional",
        bridge_difference_type="自身卖点陈述",
        bridge_source_evidence=["清新草本香味，气味宜人"],
        bridge_comparison_object="",
        bridge_comparison_object_evidence_type="null",
    )
    result = engine._normalize_differentiator(payload)
    assert result.difference_type == "自身卖点陈述"
    assert result.comparison_object == ""
    assert result.comparison_object_evidence_type == "null"


@pytest.mark.unit
def test_p2_caller_provided_comparison_object_not_overridden() -> None:
    engine = _make_engine()
    _patch_llms(engine)
    # 调用方已显式提供 comparison_object（非自身卖点陈述路径），注入逻辑不应触发改写。
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="户外旗舰店",
        product_name="户外驱蚊液",
        price="39",
        core_selling_point="有效驱避蚊虫",
        differentiator="",
        bridge_difference_domain="functional",
        bridge_difference_type="风险降低",
        bridge_source_evidence=["相比普通驱蚊方案，防止蚊虫叮咬更安全"],
        bridge_comparison_object="同类旧方案",
        bridge_comparison_object_evidence_type="user_provided",
    )
    result = engine._normalize_differentiator(payload)
    assert result.difference_type == "风险降低"
    assert result.comparison_object == "同类旧方案"
    assert result.comparison_object_evidence_type == "user_provided"
