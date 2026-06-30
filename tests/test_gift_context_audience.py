"""第五批：通用 gift_context 送礼场景识别与双重角色拆分测试。

覆盖：
- gift_context 确定性识别（hla + 通用节日/对象 + 非送礼负样本）；
- profile 引擎 gift_context_rule 激活（映射已有 active requirement，不新造 id）；
- _derive_product_target_audience gift 分支（primary=购买决策者，secondary=受礼者，
  第四段解释购买者与受礼者关系并引用真实 requirement）；
- 非送礼样本完全保持原八大人群派生逻辑不变。
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

from commerce_video_diagnosis.understanding.gift_context import detect_gift_context  # noqa: E402
from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DifferentiatorEvidence,
    JTBDProposal,
    Module1Output,
    ProductDiagnosisEngine,
    ProductIntentMatrix,
    ProductTargetAudience,
    StructuredDifferentiator,
)
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import (  # noqa: E402
    PersuasionRequirementEngine,
)

_PERSUASION_ENGINE = PersuasionRequirementEngine()


def _engine() -> ProductDiagnosisEngine:
    return ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)


def _module1(
    *,
    leaf_category: str,
    product_name: str,
    core_selling_point: str = "",
    target_people: str = "",
    target_people_raw: str = "",
) -> Module1Output:
    return Module1Output(
        leaf_category=leaf_category,
        shop_name="示例店铺",
        product_name=product_name,
        price="98",
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
        target_people_raw=target_people_raw,
    )


def _matrix(*, brand_tier: str = "大牌官方", relative_price_level: str = "低水位") -> ProductIntentMatrix:
    return ProductIntentMatrix(
        brand_tier=brand_tier,
        trust_barrier="极低",
        financial_risk="中",
        relative_price_level=relative_price_level,
        matrix_label=f"{brand_tier}×{relative_price_level}",
        business_category="T恤/Polo衫",
        median_price_threshold=120.0,
        price_value=98.0,
        product_intent="占位",
        reasoning=["占位"],
    )


def _proposal(primary_task: str = "生存/运转维系", domain: str = "功能域") -> JTBDProposal:
    return JTBDProposal(domain=domain, primary_task=primary_task, reasoning="占位", reasoning_path=["占位"])


def _gift_profile(target_people_raw: str, *, source_evidence=None, title="") -> dict:
    product_fact = {
        "leaf_category": "T恤/Polo衫",
        "category": "T恤/Polo衫",
        "title": title,
        "jtbd_level1": "功能域",
        "jtbd_level2": "生存/运转维系",
        "cognition_attribute": "红海-核心",
        "frequency_attribute": "耐用",
        "trust_attribute": "极低",
        "price_attribute": "低水位",
        "selling_points": ["凉感抗菌速干 polo"],
        "source_evidence": source_evidence or [],
        "target_people_raw": target_people_raw,
        "risk_points": [],
    }
    profile = _PERSUASION_ENGINE.generate_profile(product_fact, content_goal="purchase")
    assert profile.get("persuasion_requirements")
    return profile


# ---------------------------------------------------------------------------
# 1) gift_context 确定性识别：通用框架对任意对象/时机/意图生效
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_detect_gift_context_hla_father_day():
    gc = detect_gift_context([
        "HLA海澜之家山不在高短袖polo 凉感抗菌速干",
        "海澜之家品牌官方背书 + 父亲节礼盒装",
        "父亲节给爸爸/父亲选礼物的成年子女；自用的中年男性",
    ])
    assert gc and gc["is_gift"] is True
    assert gc["gift_scene"] == "父亲节"
    assert gc["gift_recipient"] == "爸爸"
    assert gc["recipient_demographic"] == "中年男性"
    assert gc["purchase_decider"] == "成年子女"
    assert gc["relationship"] == "子女送父亲"


@pytest.mark.unit
@pytest.mark.parametrize(
    "texts,scene,recipient,decider",
    [
        (["母亲节送妈妈的护手霜"], "母亲节", "妈妈", "成年子女"),
        (["七夕礼物送女友"], "七夕情人节", "伴侣", "伴侣"),
        (["教师节礼物"], "教师节", "老师", "学生或家长"),
        (["送给客户的高端礼盒"], "通用送礼", "职场关系人", "职场关系人"),
    ],
)
def test_detect_gift_context_generic_scenes(texts, scene, recipient, decider):
    gc = detect_gift_context(texts)
    assert gc and gc["is_gift"] is True
    assert gc["gift_scene"] == scene
    assert gc["gift_recipient"] == recipient
    assert gc["purchase_decider"] == decider


@pytest.mark.unit
def test_detect_gift_context_non_gift_returns_none():
    # 仅命中纯人群词（中年男性），无送礼意图/时机 → 不判为送礼
    assert detect_gift_context(["夏季出汗多的中年男性自用polo"]) is None
    # 含 recipient 关键词「宝宝」但无意图/时机 → 不判为送礼（runben 防回退）
    assert detect_gift_context(["宝宝防蚊水 婴幼儿户外驱蚊"]) is None


# ---------------------------------------------------------------------------
# 2) profile 引擎 gift_context_rule 激活：映射已有 active requirement，不新造 id
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_profile_gift_context_rule_activation():
    profile = _gift_profile(
        "父亲节给爸爸/父亲选礼物的成年子女",
        source_evidence=["父亲节礼盒装"],
    )
    reqs = profile["persuasion_requirements"]
    gift_sourced = {
        r["requirement_id"] for r in reqs if "gift_context_rule" in (r.get("source") or [])
    }
    # 三条映射的已有 active requirement 必须被激活并带 gift_context_rule 来源
    assert {"identify_target_user", "prove_user_fit", "clarify_usage_scenario"} <= gift_sourced
    # 不新造 requirement_id：所有 id 都在 active 白名单内（引擎已强校验，这里再确认无异常 id）
    from core_skill.schemas.protocols import ACTIVE_REQUIREMENT_WHITELIST as W

    assert all(r["requirement_id"] in W for r in reqs)


@pytest.mark.unit
def test_profile_non_gift_has_no_gift_source():
    product_fact = {
        "leaf_category": "宝宝防蚊水",
        "category": "宝宝防蚊水",
        "title": "润本驱蚊液婴幼儿防蚊",
        "jtbd_level1": "功能域",
        "jtbd_level2": "物理安全与风险规避",
        "cognition_attribute": "红海-核心",
        "frequency_attribute": "快消",
        "trust_attribute": "极低",
        "price_attribute": "高水位",
        "selling_points": ["派卡瑞丁A级驱蚊力"],
        "source_evidence": ["第三方检测报告"],
        "target_people_raw": "婴幼儿/儿童/家庭日常户外人群",
        "risk_points": [],
    }
    profile = _PERSUASION_ENGINE.generate_profile(product_fact, content_goal="purchase")
    assert all(
        "gift_context_rule" not in (r.get("source") or [])
        for r in profile["persuasion_requirements"]
    )


# ---------------------------------------------------------------------------
# 3) audience gift 分支：primary=购买决策者，secondary=受礼者
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_audience_gift_dual_role_split():
    engine = _engine()
    module1 = _module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家山不在高短袖polo 凉感抗菌速干",
        core_selling_point="凉感抗菌速干 polo",
        target_people="中年男性人群",
        target_people_raw="父亲节给爸爸/父亲选礼物的成年子女；自用的中年男性",
    )
    profile = _gift_profile(
        "父亲节给爸爸/父亲选礼物的成年子女",
        source_evidence=["父亲节礼盒装"],
    )
    result = engine._derive_product_target_audience(
        module1,
        _proposal("生存/运转维系"),
        _matrix(),
        persuasion_requirement_profile=profile,
    )
    assert isinstance(result, ProductTargetAudience)
    # primary = 购买决策者（成年子女）
    assert [a.audience_group for a in result.primary_audiences] == ["成年子女"]
    # secondary = 受礼者（爸爸/中年男性）
    assert result.secondary_audiences
    assert "爸爸" in result.secondary_audiences[0].audience_group
    assert "中年男性" in result.secondary_audiences[0].audience_group
    # 第四段解释购买者与受礼者关系，并引用真实 requirement（含 requirement_id 括号）
    ppta = result.reasoning_chain.persuasion_profile_to_audience
    assert ppta.strip()
    assert "成年子女" in ppta and "爸爸" in ppta
    assert "(" in ppta and ")" in ppta
    # caveat 显式区分送礼者≠受礼者
    assert any("送礼者≠受礼者" in c for c in result.caveats)


@pytest.mark.unit
def test_audience_gift_detect_via_product_fact_source_evidence():
    """口径一致性：送礼信号仅在 source_evidence（真实 hla payload 形态，
    product_name/target_people 已去送礼词）时，audience 仍能经 product_fact 识别送礼
    并拆双重角色——与 profile 引擎同一组信号口径，不出现「profile 判送礼、audience 走八大人群」漂移。
    """
    engine = _engine()
    module1 = _module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家山不在高短袖polo 凉感抗菌速干",  # 已去送礼词
        core_selling_point="凉感抗菌速干 polo",
        target_people="中年男性人群",
        target_people_raw="夏季出汗多/需要凉感速干 polo 的中年男性",  # 不含送礼语义
    )
    product_fact = {
        "title": "HLA海澜之家山不在高短袖polo 凉感抗菌速干",
        "category": "T恤/Polo衫",
        "leaf_category": "T恤/Polo衫",
        "selling_points": ["凉感抗菌速干 polo"],
        # 送礼信号仅存在于 source_evidence
        "source_evidence": ["海澜之家品牌官方背书 + 父亲节礼盒装 送爸爸"],
        "target_people_raw": "夏季出汗多/需要凉感速干 polo 的中年男性",
    }
    profile = _gift_profile(
        "夏季出汗多/需要凉感速干 polo 的中年男性",
        source_evidence=["海澜之家品牌官方背书 + 父亲节礼盒装 送爸爸"],
    )
    # 无 product_fact 时（仅 module1 字段）识别不到送礼 → 走八大人群
    fallback = engine._derive_product_target_audience(
        module1, _proposal("生存/运转维系"), _matrix(), persuasion_requirement_profile=profile
    )
    assert all("消费力" in a.audience_group for a in fallback.primary_audiences)
    # 传入 product_fact（含 source_evidence）后 → 识别送礼并拆双重角色
    result = engine._derive_product_target_audience(
        module1,
        _proposal("生存/运转维系"),
        _matrix(),
        persuasion_requirement_profile=profile,
        product_fact=product_fact,
    )
    assert [a.audience_group for a in result.primary_audiences] == ["成年子女"]
    assert "爸爸" in result.secondary_audiences[0].audience_group
    assert any("送礼者≠受礼者" in c for c in result.caveats)


@pytest.mark.unit
def test_audience_non_gift_keeps_eight_population_path():
    """非送礼样本：保持原八大人群坐标派生（不走 gift 分支）。"""
    engine = _engine()
    module1 = _module1(
        leaf_category="宝宝防蚊水",
        product_name="润本驱蚊液婴幼儿防蚊",
        core_selling_point="派卡瑞丁A级驱蚊力",
        target_people="婴幼儿人群",
        target_people_raw="婴幼儿/儿童/家庭日常户外人群",
    )
    profile = _PERSUASION_ENGINE.generate_profile(
        {
            "leaf_category": "宝宝防蚊水",
            "category": "宝宝防蚊水",
            "title": "润本驱蚊液婴幼儿防蚊",
            "jtbd_level1": "功能域",
            "jtbd_level2": "物理安全与风险规避",
            "cognition_attribute": "红海-核心",
            "frequency_attribute": "快消",
            "trust_attribute": "极低",
            "price_attribute": "高水位",
            "selling_points": ["派卡瑞丁A级驱蚊力"],
            "source_evidence": ["第三方检测报告"],
            "target_people_raw": "婴幼儿/儿童/家庭日常户外人群",
            "risk_points": [],
        },
        content_goal="purchase",
    )
    result = engine._derive_product_target_audience(
        module1,
        _proposal("物理安全与风险规避"),
        _matrix(brand_tier="大牌官方", relative_price_level="高水位"),
        persuasion_requirement_profile=profile,
    )
    # 八大人群坐标（含「消费力」「女性/男性」），不是 gift 角色标签
    assert all("消费力" in a.audience_group for a in result.primary_audiences)
    assert not any("送礼者≠受礼者" in c for c in result.caveats)
