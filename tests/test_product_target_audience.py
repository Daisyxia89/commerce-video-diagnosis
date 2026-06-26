"""Block 1.1 商品目标人群 product_target_audience 测试。

测试 `ProductDiagnosisEngine._derive_product_target_audience` 纯函数逻辑：
- 润本锚（D6 口径）：primary/secondary/weak_fit 精确匹配。
- 男性偏向品类 → 含男性人群。
- 四属性缺失 → Crash Early。
- 字段命名：禁止出现 `segments`。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DifferentiatorEvidence,
    JTBDProposal,
    Module1Output,
    ProductDiagnosisEngine,
    ProductIntentMatrix,
    ProductTargetAudience,
    StructuredDifferentiator,
)


def _engine() -> ProductDiagnosisEngine:
    # 仅测试纯函数，无需初始化外部依赖
    return ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)


def _module1(
    *,
    leaf_category: str,
    product_name: str,
    core_selling_point: str = "",
    target_people: str = "",
) -> Module1Output:
    return Module1Output(
        leaf_category=leaf_category,
        shop_name="示例店铺",
        product_name=product_name,
        price="24.9",
        core_selling_point=core_selling_point,
        core_selling_point_source="caller_provided.core_selling_points",
        target_people=target_people,
        differentiator=StructuredDifferentiator(
            comparison_object="",
            difference_domain="functional",
            difference_type="自身卖点陈述",
            conclusion="占位",
            evidence_chain=[DifferentiatorEvidence("商品信息", core_selling_point or "占位")],
            summary="占位",
        ),
    )


def _matrix(*, brand_tier: str, relative_price_level: str, business_category: str = "宝宝防蚊水") -> ProductIntentMatrix:
    return ProductIntentMatrix(
        brand_tier=brand_tier,
        trust_barrier="极低",
        financial_risk="高",
        relative_price_level=relative_price_level,
        matrix_label=f"{brand_tier}×{relative_price_level}",
        business_category=business_category,
        median_price_threshold=13.5,
        price_value=24.9,
        product_intent="占位",
        reasoning=["占位"],
    )


def _proposal(primary_task: str, domain: str = "功能域") -> JTBDProposal:
    return JTBDProposal(domain=domain, primary_task=primary_task, reasoning="占位", reasoning_path=["占位"])


# ---------------------------------------------------------------------------
# 润本锚（D6）：primary/secondary/weak_fit 精确匹配
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_runben_anchor_matches_d6():
    engine = _engine()
    module1 = _module1(
        leaf_category="宝宝防蚊水",
        product_name="【A级驱蚊力】润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水防蚊叮蚊怕花露水",
        core_selling_point="派卡瑞丁A级驱蚊力，长效防蚊驱虫，温和无刺激",
        target_people="婴幼儿人群",
    )
    matrix = _matrix(brand_tier="大牌官方", relative_price_level="高水位")
    result = engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), matrix)

    assert isinstance(result, ProductTargetAudience)
    # primary
    assert [(a.audience_group, a.fit_level) for a in result.primary_audiences] == [
        ("年长中高消费力女性", "primary")
    ]
    # secondary（D6：大牌官方 高水位 → 追加同年龄性别的低消费力 secondary）
    assert [(a.audience_group, a.fit_level) for a in result.secondary_audiences] == [
        ("年长低消费力女性", "secondary")
    ]
    # weak_fit 为空
    assert result.weak_fit_audiences == []
    # reasoning_chain 三段非空
    rc = result.reasoning_chain
    assert rc.task_to_role.strip()
    assert rc.role_category_to_age_gender.strip()
    assert rc.brand_price_to_consumption_power.strip()
    # reason 可解释（含任务/角色/品牌价格依据），不做精确字符串断言
    assert "风险责任人" in result.primary_audiences[0].reason
    # D6 caveat
    assert any("低消费力" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# 男性偏向品类（汽车/工具 + 物理安全）→ 含男性人群
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_male_leaning_category_contains_male_audience():
    engine = _engine()
    module1 = _module1(
        leaf_category="车载工具箱",
        product_name="车载五金工具套装户外应急",
        core_selling_point="户外应急安全防护",
        target_people="车主/越野人群",
    )
    matrix = _matrix(brand_tier="白牌", relative_price_level="低水位", business_category="车载工具箱")
    result = engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), matrix)

    all_groups = [a.audience_group for a in result.primary_audiences + result.secondary_audiences]
    assert any("男性" in g for g in all_groups), all_groups


# ---------------------------------------------------------------------------
# Crash Early：四属性缺失
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_crash_early_missing_primary_task():
    engine = _engine()
    module1 = _module1(leaf_category="宝宝防蚊水", product_name="润本驱蚊液")
    matrix = _matrix(brand_tier="大牌官方", relative_price_level="高水位")
    with pytest.raises(ValueError, match="primary_task"):
        engine._derive_product_target_audience(module1, SimpleNamespace(primary_task=""), matrix)


@pytest.mark.unit
def test_crash_early_missing_brand_tier():
    engine = _engine()
    module1 = _module1(leaf_category="宝宝防蚊水", product_name="润本驱蚊液")
    bad_matrix = SimpleNamespace(brand_tier="", relative_price_level="高水位")
    with pytest.raises(ValueError, match="brand_tier"):
        engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), bad_matrix)


@pytest.mark.unit
def test_crash_early_missing_relative_price_level():
    engine = _engine()
    module1 = _module1(leaf_category="宝宝防蚊水", product_name="润本驱蚊液")
    bad_matrix = SimpleNamespace(brand_tier="大牌官方", relative_price_level="")
    with pytest.raises(ValueError, match="relative_price_level"):
        engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), bad_matrix)


@pytest.mark.unit
def test_crash_early_missing_category_and_product_name():
    engine = _engine()
    module1 = _module1(leaf_category="", product_name="")
    matrix = _matrix(brand_tier="大牌官方", relative_price_level="高水位")
    with pytest.raises(ValueError, match="leaf_category 与 product_name"):
        engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), matrix)


# ---------------------------------------------------------------------------
# 字段命名：禁止出现 segments
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_segments_field_naming():
    engine = _engine()
    module1 = _module1(
        leaf_category="宝宝防蚊水",
        product_name="润本驱蚊液",
        core_selling_point="温和驱蚊",
        target_people="婴幼儿人群",
    )
    matrix = _matrix(brand_tier="大牌官方", relative_price_level="高水位")
    result = engine._derive_product_target_audience(module1, _proposal("物理安全与风险规避"), matrix)

    def _assert_no_segments(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "segments" not in k, f"非法字段名包含 segments: {k}"
                _assert_no_segments(v)
        elif isinstance(obj, list):
            for v in obj:
                _assert_no_segments(v)

    _assert_no_segments(result.dict())
