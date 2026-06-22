# -*- coding: utf-8 -*-
"""PRD §5.1.5 社会任务证据链回归测试。

覆盖：
- 基础功能型商品（如驱蚊液 / 厨房清洁巾）不得误判为社会任务
- 关键词存在但证据不足时必须拒绝
- 三门槛均满足时才允许判定为 `阶层与审美发信`
- reasoning 不得恒含社会任务专有模板词
"""
from __future__ import annotations

import pytest

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (
    DiagnosticInput,
    JTBDProposal,
    Module1Output,
    ProductDiagnosisEngine,
    StructuredDifferentiator,
    SOCIAL_DOMAIN,
)


def _build_engine() -> ProductDiagnosisEngine:
    return ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)  # 仅测试纯函数，不需要初始化外部依赖


def _module1(**overrides) -> Module1Output:
    base = dict(
        leaf_category="日用百货",
        shop_name="测试店铺",
        product_name="测试商品",
        price="29.9",
        core_selling_point="",
        core_selling_point_source="title_llm_extracted",
        target_people="",
        differentiator=StructuredDifferentiator(comparison_object=""),
        second_level_category="",
        third_level_category="",
    )
    base.update(overrides)
    return Module1Output(**base)


def _payload(**overrides) -> DiagnosticInput:
    base = dict(
        leaf_category="日用百货",
        shop_name="测试店铺",
        product_name="测试商品",
        price="29.9",
        core_selling_point="",
        target_people="",
        differentiator=StructuredDifferentiator(comparison_object=""),
    )
    base.update(overrides)
    return DiagnosticInput(**base)


# -----------------------------
# 1. 基础功能型不得误判
# -----------------------------
def test_basic_functional_repellent_not_social_task():
    engine = _build_engine()
    text = "驱蚊液 户外防蚊 清洁除味 家用"
    task, evidence, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task is None, f"驱蚊液不得被误判为社会任务，但得到 {task}"
    assert any("基础功能型" in r for r in rejections)


def test_basic_functional_with_family_keyword_not_social():
    """文本里出现 '家人' 但商品是基础功能（清洁），不应判为照护任务。"""
    engine = _build_engine()
    text = "厨房清洁巾 给家人更干净的厨房 去油污 除味"
    task, _, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task is None
    assert any("基础功能型" in r or "拒绝" in r for r in rejections)


# -----------------------------
# 2. 关键词存在但证据不足必须拒绝
# -----------------------------
def test_gift_keyword_without_relationship_object_rejected():
    engine = _build_engine()
    text = "精美礼盒装 高级感包装 适合送礼"
    task, _, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task is None
    assert any("缺少明确关系对象" in r for r in rejections)


def test_marketing_only_rhetoric_rejected():
    engine = _build_engine()
    text = "网红爆款 高级感同款 潮流必备"
    task, _, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task is None
    assert any("仅出现营销修辞词" in r for r in rejections)


def test_caregiving_keyword_without_relationship_object_rejected():
    engine = _build_engine()
    text = "全面护理 责任履行 守护每一天"
    task, _, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task is None
    assert rejections, "应记录证据不闭合的拒绝理由"


# -----------------------------
# 3. 合法路由
# -----------------------------
def test_caregiving_with_relationship_object_passes():
    engine = _build_engine()
    text = "婴儿专用护理棉柔巾 妈妈安心照护宝宝肌肤"
    task, evidence, rejections = engine._infer_social_task(_payload(), _module1(), text)
    assert task == "照护与责任履行"
    assert evidence and "关系对象" in evidence[0] and "照护" in evidence[0]


def test_gift_with_relationship_object_passes():
    engine = _build_engine()
    text = "中秋礼盒 送父母送长辈的伴手礼"
    task, evidence, _ = engine._infer_social_task(_payload(), _module1(), text)
    assert task == "礼赠与关系表达"
    assert evidence and "礼赠场景" in evidence[0]


def test_circle_identity_passes():
    engine = _build_engine()
    text = "lo娘穿搭必备 lolita 圈内款"
    task, evidence, _ = engine._infer_social_task(_payload(), _module1(), text)
    assert task == "圈层认同（圈层归属/身份锚定）"
    assert evidence and "具名圈层证据" in evidence[0]


# -----------------------------
# 4. 阶层与审美发信三门槛
# -----------------------------
def test_status_signaling_requires_all_three_gates():
    engine = _build_engine()
    # 仅高外显，无门槛、无共识 → 拒绝
    text_only_visibility = "腕表 商务场合佩戴"
    task, _, rejections = engine._infer_social_task(_payload(), _module1(), text_only_visibility)
    assert task is None
    assert any("三门槛未集齐" in r for r in rejections)

    # 高外显 + 获取门槛 + 圈层共识 → 通过
    text_full = "限量联名腕表 商务场合佩戴 老钱风 经典款"
    task, evidence, _ = engine._infer_social_task(_payload(), _module1(), text_full)
    assert task == "阶层与审美发信"
    assert evidence and "三道硬门槛全部命中" in evidence[0]


# -----------------------------
# 5. reasoning 不得恒含模板词
# -----------------------------
def test_reasoning_does_not_contain_template_keywords_when_not_social():
    """当 _infer_social_task 返回 None 时，调用方不会构造 reasoning 含模板词。"""
    engine = _build_engine()
    text = "驱蚊液 户外防蚊 家用"
    task, evidence, _ = engine._infer_social_task(_payload(), _module1(), text)
    # 不会产出社会任务 evidence
    assert task is None
    # evidence_lines 在拒绝场景里必须为空，不能写"圈层共识/阶层发信/身份表达"
    blob = " ".join(evidence)
    for banned in ("圈层共识", "阶层发信", "身份表达", "社会认同"):
        assert banned not in blob


def test_reasoning_contains_only_real_evidence_on_pass():
    engine = _build_engine()
    text = "婴儿专用护理棉柔巾 妈妈安心照护宝宝肌肤"
    task, evidence, _ = engine._infer_social_task(_payload(), _module1(), text)
    assert task == "照护与责任履行"
    blob = " ".join(evidence)
    # 必须包含真实命中的证据词，且不能仅仅是"圈层共识"模板
    assert "宝宝" in blob or "妈妈" in blob
    assert "照护" in blob


# -----------------------------
# 6. JTBDProposal 后置断言（root_validator）
# -----------------------------
def test_jtbd_proposal_rejects_template_only_reasoning():
    classifier = ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)
    proposal = JTBDProposal(
        domain=SOCIAL_DOMAIN,
        primary_task="照护与责任履行",
        reasoning="规则树直接判定为社会任务。",
        reasoning_path=["规则树直接判定为社会任务"],
        candidate_tasks=["照护与责任履行"],
        evidence_chain=[{"evidence_source": "目标人群", "evidence_text": "宝宝"}],
        gate_reasons=["fake"],
        triggered_rule="social_priority_rule",
    )
    with pytest.raises(ValueError, match="预设模板"):
        classifier._assert_proposal(proposal)


def test_jtbd_proposal_rejects_social_without_gate_reasons():
    classifier = ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)
    proposal = JTBDProposal(
        domain=SOCIAL_DOMAIN,
        primary_task="照护与责任履行",
        reasoning="商品文本同时出现关系对象（宝宝）与照护语义（护理）",
        reasoning_path=["商品文本同时出现关系对象（宝宝）与照护语义（护理）"],
        candidate_tasks=["照护与责任履行"],
        evidence_chain=[{"evidence_source": "目标人群", "evidence_text": "宝宝"}],
        gate_reasons=[],
        triggered_rule="social_priority_rule",
    )
    with pytest.raises(ValueError, match="gate_reasons"):
        classifier._assert_proposal(proposal)


def test_jtbd_proposal_accepts_social_with_real_evidence():
    classifier = ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)
    proposal = JTBDProposal(
        domain=SOCIAL_DOMAIN,
        primary_task="照护与责任履行",
        reasoning="商品文本同时出现关系对象（宝宝）与照护语义（护理）",
        reasoning_path=["商品文本同时出现关系对象（宝宝）与照护语义（护理）"],
        candidate_tasks=["照护与责任履行"],
        evidence_chain=[{"evidence_source": "目标人群", "evidence_text": "宝宝"}],
        gate_reasons=["商品文本同时出现关系对象（宝宝）与照护语义（护理）"],
        triggered_rule="social_priority_rule",
    )
    classifier._assert_proposal(proposal)
    assert proposal.primary_task == "照护与责任履行"
