# -*- coding: utf-8 -*-
"""P1 Hotfix：response_assembler 对 out_of_scope_for_mvp 状态的装配兜底。

目标：
- out_of_scope_for_mvp 下 build_product_understanding 不 Crash Early；
- product_hec 返回状态化空结构；
- hook/effect/cta 不伪造，保持 None；
- 正常样本不受影响。

注：此文件为回归用 targeted test；不修改 CandidateSet / Product_HEC 生成策略，只验证装配层行为。
"""

import json
import os

from commerce_video_diagnosis.understanding.assembly.response_assembler import build_product_understanding


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RUNBEN_FULL_DIAG = os.path.join(_REPO, "outputs", "runben_diagnosis", "runben_full_diagnosis.json")

EXPECTED_PU_ORDER = [
    "basic_info",
    "product_fact_vector",
    "module3",
    "candidate_set",
    "product_hec",
    "evidence",
]


def _fake_out_of_scope_pd():
    """构造一个最小可装配的 out_of_scope_for_mvp 商品诊断输入。

    约束：
    - 不在 assembler 造 HEC 数据：product_hecs 必须为空；
    - 不改 CandidateSet 生成策略：core_intent 保持空列表即可；
    - basic_info / fact_vector 所需字段完整，避免与本 hotfix 无关的 Crash Early。
    """

    return {
        "product_id": "fake_out_of_scope_product",
        "product_name": "【样例】MVP 暂不支持商品",
        "leaf_category": "测试类目",
        "brand_name": "测试品牌",
        "shop_name": "测试店铺",
        "price": 9.9,
        "jtbd": "任意",
        "resistance_profile": {
            "ocean": "红海",
            "competition_focus": "核心",
            "frequency": "快消",
            "brand_tier": "白牌",
            "relative_price_level": "低水位",
            "endorsement": None,
            "channel_risk": "无风险",
        },
        "core_intent": {
            "candidate_h": [],
            "core_e": [],
            "core_c": [],
            "primary_effect": None,
            "primary_cta": None,
        },
        "persuasion_requirement_profile": {},
        "product_target_audience": {},
        "product_hecs": [],
        "assembly_status": {
            "status": "out_of_scope_for_mvp",
            "reason_code": "test_only",
            "user_facing_message": "当前商品暂不支持",
        },
        "evidence": {
            "input": {
                "target_people": "测试人群",
                "core_selling_point": "测试卖点",
                "brand_name": "测试品牌",
            }
        },
    }


def test_tc1_out_of_scope_build_product_understanding_no_crash_and_keep_6_sections():
    pu = build_product_understanding(_fake_out_of_scope_pd())
    assert list(pu.keys()) == EXPECTED_PU_ORDER


def test_tc2_out_of_scope_product_hec_status_and_null_triples():
    pu = build_product_understanding(_fake_out_of_scope_pd())
    assert isinstance(pu.get("product_hec"), list) and pu["product_hec"], "product_hec 必须为非空 list"
    hec0 = pu["product_hec"][0]
    assert hec0.get("status") == "out_of_scope_for_mvp"
    assert hec0.get("hook") is None
    assert hec0.get("effect") is None
    assert hec0.get("cta") is None


def test_tc3_normal_sample_still_returns_hec_triples():
    with open(_RUNBEN_FULL_DIAG, "r", encoding="utf-8") as f:
        pd = json.load(f)
    pu = build_product_understanding(pd)
    assert list(pu.keys()) == EXPECTED_PU_ORDER
    hec0 = pu["product_hec"][0]
    # 正常样本必须是三元组结构（code/name/definition 非空）
    for dim in ("hook", "effect", "cta"):
        triple = hec0.get(dim)
        assert isinstance(triple, dict)
        assert isinstance(triple.get("code"), str) and triple["code"].strip()
        assert isinstance(triple.get("name"), str) and triple["name"].strip()
        assert isinstance(triple.get("definition"), str) and triple["definition"].strip()
