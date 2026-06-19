# -*- coding: utf-8 -*-
"""/JG 独立 QA 验收：JTBD 功能型风险规避防降级路由（P0 / P1 / P2）。

本测试用例独立依据业务 PRD `docs/product_diagnosis_dictionary.md` 4.3.3 节
「JTBD 路由逻辑补充（功能型风险规避防降级）」第 1/2/3 条编写，
不参考、不复用研发自测文件 tests/test_jtbd_protection_risk_routing.py。

PRD 绝对准则锚点：
- P0（第 1 条 + 字典 line 323）：物理安全前置词表必须覆盖防虫/防护类风险词
  （蚊虫叮咬、叮咬、蚊虫、驱蚊、防蚊、防虫、驱虫、虫媒、驱避、瘙痒、红肿、过敏），
  命中任一即在 Stage A 唯一锁定「物理安全与风险规避」，后续阶段不再改写 primary_task。
- P1（第 2 条）：difference_type=自身卖点陈述 不再单独作为「生存/运转维系」的充分条件；
  一旦出现风险/防护/缺陷语义，必须交回上游任务本体裁决，不得回落基础维系。
- P2（第 3 条）：防护/防虫/防晒/安全等风险类目命中且证据含风险/防护锚点、且调用方未提供
  comparison_object 时，桥接层必须注入隐式对比对象
  （comparison_object=同类旧方案、comparison_object_evidence_type=jtbd_inferred、
  difference_type 改写为 风险降低、difference_domain=functional）；
  否则保持原协议（自身卖点陈述清空 comparison_object）。
- 反向（验收要求 3）：非风险品（纸巾、米、普通饮料）不得被误判为防护/物理安全类。
"""

import pytest

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (
    DifferentiatorEvidence,
    DiagnosticInput,
    Module1Output,
    PHYSICAL_SAFETY_TOKENS,
    PROTECTION_RISK_CATEGORY_TOKENS,
    PROTECTION_SEMANTIC_TOKENS,
    ProductDiagnosisEngine,
    StructuredDifferentiator,
)

pytestmark = pytest.mark.unit

# PRD 字典 line 323 明确列出的防虫/防护类风险词（P0 必须全覆盖）。
PRD_P0_REQUIRED_TOKENS = [
    "蚊虫叮咬",
    "叮咬",
    "蚊虫",
    "驱蚊",
    "防蚊",
    "防虫",
    "驱虫",
    "虫媒",
    "驱避",
    "瘙痒",
    "红肿",
    "过敏",
]

# PRD 字典 line 323 要求保留的显性伤害词（防降级不得削弱原有覆盖）。
LEGACY_HARM_TOKENS = ["烫伤", "触电", "跌落", "晒伤"]

# 验收要求 3：非风险品样本。
NON_RISK_TEXTS = [
    "抽纸 原木浆 柔软亲肤 3层加厚",
    "东北大米 当季新米 粒粒饱满 口感软糯",
    "气泡饮料 0糖0卡 清爽解腻 多口味",
]


@pytest.fixture(scope="module")
def engine() -> ProductDiagnosisEngine:
    return ProductDiagnosisEngine()


def _make_module1(
    *,
    leaf_category: str,
    product_name: str,
    core_selling_point: str,
    difference_type: str = "自身卖点陈述",
    target_people: str = "有相关需求的人群",
    second_level_category: str = "",
    third_level_category: str = "",
) -> Module1Output:
    """构造最小 Module1Output；difference_type 默认走 PRD 关注的自身卖点陈述路径。"""
    differentiator = StructuredDifferentiator(
        comparison_object="",
        comparison_object_evidence_type="null",
        difference_domain="functional",
        difference_type=difference_type,
        conclusion=core_selling_point,
        evidence_chain=[DifferentiatorEvidence(evidence_source="商品信息", evidence_text=core_selling_point)],
        summary=core_selling_point,
    )
    return Module1Output(
        leaf_category=leaf_category,
        shop_name="示例旗舰店",
        product_name=product_name,
        price="59.9",
        core_selling_point=core_selling_point,
        core_selling_point_source="商品信息",
        target_people=target_people,
        differentiator=differentiator,
        second_level_category=second_level_category,
        third_level_category=third_level_category,
    )


# ---------------------------------------------------------------------------
# P0：_is_physical_safety_fact 词表覆盖 + Stage A 锁定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", PRD_P0_REQUIRED_TOKENS)
def test_p0_physical_safety_token_table_covers_protection_words(token):
    """P0：PRD line 323 要求的每个防虫/防护风险词，都必须落在物理安全前置词表内。"""
    assert token in PHYSICAL_SAFETY_TOKENS, f"物理安全词表缺少 PRD 强制要求的风险词：{token}"


@pytest.mark.parametrize("token", PRD_P0_REQUIRED_TOKENS)
def test_p0_is_physical_safety_fact_hits_each_protection_word(engine, token):
    """P0：命中任一防虫/防护风险词，_is_physical_safety_fact 必须判 True。"""
    assert engine._is_physical_safety_fact(f"商品文案包含{token}相关描述") is True


@pytest.mark.parametrize("token", LEGACY_HARM_TOKENS)
def test_p0_legacy_harm_tokens_still_covered(engine, token):
    """P0：防降级补丁不得削弱原有显性伤害词覆盖。"""
    assert engine._is_physical_safety_fact(f"使用不当可能{token}") is True


def test_p0_stage_a_locks_mosquito_repellent_to_physical_safety(engine):
    """P0：驱蚊液在 Stage A 唯一锁定「物理安全与风险规避」，并排除「生存/运转维系」。"""
    module1 = _make_module1(
        leaf_category="驱蚊液",
        product_name="户外驱蚊液 防蚊防虫喷雾",
        core_selling_point="有效驱蚊防虫，减少蚊虫叮咬带来的瘙痒红肿",
    )
    rule_context = engine._build_rule_tree_context(_dummy_payload("驱蚊液"), module1)

    assert rule_context["candidate_tasks"] == ["物理安全与风险规避"], (
        f"驱蚊液 Stage A 应唯一锁定物理安全，实际：{rule_context['candidate_tasks']}"
    )
    assert rule_context["subcategory_context"] == "stage_a_hard_gate"
    # PRD：命中后唯一裁决，生存/运转维系应被显式排除。
    assert "生存/运转维系" in rule_context.get("excluded_tasks", {})
    assert "生存/运转维系" not in rule_context["candidate_tasks"]


def test_p0_stage_a_locks_on_bite_word_without_repellent_word(engine):
    """P0：仅出现「叮咬/红肿」等风险结果词（无驱蚊字样）也应在 Stage A 锁定物理安全。"""
    module1 = _make_module1(
        leaf_category="止痒膏",
        product_name="户外止痒膏",
        core_selling_point="缓解叮咬后的红肿与瘙痒",
    )
    rule_context = engine._build_rule_tree_context(_dummy_payload("止痒膏"), module1)
    assert rule_context["candidate_tasks"] == ["物理安全与风险规避"]


def _dummy_payload(leaf_category: str) -> DiagnosticInput:
    return DiagnosticInput(
        leaf_category=leaf_category,
        shop_name="示例旗舰店",
        product_name=leaf_category,
        price="59.9",
        core_selling_point="",
        target_people="有相关需求的人群",
        differentiator="",
    )


# ---------------------------------------------------------------------------
# P1：_supports_maintenance_task 收紧
# ---------------------------------------------------------------------------


def test_p1_self_statement_with_physical_risk_blocks_maintenance(engine):
    """P1：自身卖点陈述 + 物理安全风险语义（驱蚊），不得回落「生存/运转维系」。"""
    module1 = _make_module1(
        leaf_category="驱蚊液",
        product_name="户外驱蚊液",
        core_selling_point="有效驱蚊防虫",
        difference_type="自身卖点陈述",
    )
    text = engine._module1_joined_text(module1)
    assert engine._supports_maintenance_task(module1, text) is False


def test_p1_self_statement_with_protection_semantic_blocks_maintenance(engine):
    """P1：自身卖点陈述 + 防护语义（防滑，属 protection_semantic_tokens）同样不得回落维系。

    选用「防滑」是为了精准命中 P1 收紧分支：它属于 protection_semantic_tokens，
    但不在 physical_safety_tokens / ordinary_daily_tokens / food / maintenance_supply 中，
    可避免被前置分支提前放行，从而独立验证「防护语义阻断维系兜底」这一收紧逻辑。
    """
    module1 = _make_module1(
        leaf_category="瑜伽垫",
        product_name="加厚防滑瑜伽垫",
        core_selling_point="表面防滑稳固",
        difference_type="自身卖点陈述",
    )
    text = engine._module1_joined_text(module1)
    # 前置确认：该样本不命中 physical_safety，从而走到 P1 收紧分支。
    assert engine._is_physical_safety_fact(text) is False
    assert engine._contains_any(text, PROTECTION_SEMANTIC_TOKENS) is True
    assert engine._supports_maintenance_task(module1, text) is False


def test_p1_self_statement_without_risk_still_allows_maintenance(engine):
    """P1（防过度收紧）：自身卖点陈述 + 无任何风险/防护/缺陷语义，仍允许回落「生存/运转维系」。

    选用普通装饰品类，规避 food / maintenance_supply / ordinary_daily / efficiency / ease
    等前置分支，确保命中 PRD 第 2 条「无风险语义时才允许回落」的正例分支。
    """
    module1 = _make_module1(
        leaf_category="桌面摆件",
        product_name="北欧风陶瓷花瓶",
        core_selling_point="造型独特线条简约",
        difference_type="自身卖点陈述",
    )
    text = engine._module1_joined_text(module1)
    assert engine._is_physical_safety_fact(text) is False
    assert engine._contains_any(text, PROTECTION_SEMANTIC_TOKENS) is False
    assert engine._supports_maintenance_task(module1, text) is True


# ---------------------------------------------------------------------------
# P2：桥接层隐式对比对象注入
# ---------------------------------------------------------------------------


def test_p2_protection_category_with_risk_evidence_injects_implicit_comparison(engine):
    """P2：防护类目 + 证据含风险锚点 + 调用方未提供 comparison_object，必须注入隐式对比对象。"""
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="示例旗舰店",
        product_name="户外驱蚊液",
        bridge_difference_type="自身卖点陈述",
    )
    injected = engine._inject_protection_risk_implicit_comparison(
        payload,
        bridge_source_evidence=["户外有效防蚊，降低蚊虫叮咬风险"],
        bridge_comparison_object="",
    )
    assert injected is not None, "防护类目 + 风险证据应注入隐式对比对象，实际返回 None"
    difference_domain, difference_type, comparison_object, evidence_type = injected
    assert difference_domain == "functional"
    assert difference_type == "风险降低"
    assert comparison_object == "同类旧方案"
    assert evidence_type == "jtbd_inferred"


def test_p2_non_protection_category_keeps_original_protocol(engine):
    """P2：非防护类目（纸巾）不注入，保持原「自身卖点陈述清空 comparison_object」协议。"""
    payload = DiagnosticInput(
        leaf_category="抽纸",
        shop_name="示例旗舰店",
        product_name="原木浆抽纸",
        bridge_difference_type="自身卖点陈述",
    )
    injected = engine._inject_protection_risk_implicit_comparison(
        payload,
        bridge_source_evidence=["3层加厚，柔软亲肤"],
        bridge_comparison_object="",
    )
    assert injected is None


def test_p2_protection_category_without_risk_anchor_no_injection(engine):
    """P2：防护类目命中但证据无风险/防护锚点时，不得注入（PRD：保持原协议）。"""
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="示例旗舰店",
        product_name="驱蚊液",
        bridge_difference_type="自身卖点陈述",
    )
    injected = engine._inject_protection_risk_implicit_comparison(
        payload,
        bridge_source_evidence=["成分天然，味道清新"],  # 无 风险/安全/保护/防 等锚点
        bridge_comparison_object="",
    )
    assert injected is None


def test_p2_caller_provided_comparison_object_is_respected(engine):
    """P2：调用方已显式提供 comparison_object 时，桥接层不得覆盖（尊重上游输入）。"""
    payload = DiagnosticInput(
        leaf_category="驱蚊液",
        shop_name="示例旗舰店",
        product_name="户外驱蚊液",
        bridge_difference_type="自身卖点陈述",
    )
    injected = engine._inject_protection_risk_implicit_comparison(
        payload,
        bridge_source_evidence=["户外有效防蚊，降低叮咬风险"],
        bridge_comparison_object="同赛道竞品",
    )
    assert injected is None


# ---------------------------------------------------------------------------
# 反向验收（要求 3）：非风险品不被误判为防护/物理安全类
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", NON_RISK_TEXTS)
def test_non_risk_products_not_physical_safety(engine, text):
    assert engine._is_physical_safety_fact(text) is False


@pytest.mark.parametrize("text", NON_RISK_TEXTS)
def test_non_risk_products_not_protection_category(engine, text):
    assert engine._find_first_keyword(text, PROTECTION_RISK_CATEGORY_TOKENS) is None


@pytest.mark.parametrize(
    ("leaf_category", "product_name", "core_selling_point"),
    [
        ("抽纸", "原木浆抽纸", "3层加厚 柔软亲肤"),
        ("大米", "东北当季新米", "粒粒饱满 口感软糯"),
        ("气泡饮料", "0糖气泡水", "清爽解腻 多口味"),
    ],
)
def test_non_risk_products_stage_a_not_locked_to_physical_safety(engine, leaf_category, product_name, core_selling_point):
    """非风险品在 Stage A 不应被锁定为物理安全。"""
    module1 = _make_module1(
        leaf_category=leaf_category,
        product_name=product_name,
        core_selling_point=core_selling_point,
    )
    rule_context = engine._build_rule_tree_context(_dummy_payload(leaf_category), module1)
    assert "物理安全与风险规避" not in rule_context["candidate_tasks"], (
        f"非风险品 {leaf_category} 被误锁定物理安全：{rule_context['candidate_tasks']}"
    )
    assert rule_context["subcategory_context"] != "stage_a_hard_gate" or rule_context["candidate_tasks"] != [
        "物理安全与风险规避"
    ]
