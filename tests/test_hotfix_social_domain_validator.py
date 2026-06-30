from __future__ import annotations

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (
    CategoryIntentMatrix,
    ProductDiagnosisEngine,
    ProductDiagnosisOutput,
    ProductIntentMatrix,
)


HLA_PAYLOAD = {
    "leaf_category": "T恤/Polo衫",
    "shop_name": "HLA海澜之家官方旗舰店",
    "second_level_category": "男装",
    "third_level_category": "POLO衫",
    "brand_name": "HLA海澜之家",
    "product_name": "【父亲节送礼】HLA海澜之家山不在高短袖polo26新凉感抗菌送爸爸",
    "price": "198",
    "core_selling_point": "凉感抗菌速干，父亲节礼盒装，适合送爸爸",
    "core_selling_point_source": "caller_provided.core_selling_points",
    "target_people": "父亲节给爸爸/父亲选礼物的成年子女；自用的中年男性",
    "differentiator": "HLA品牌官方背书，凉感抗菌速干面料，父亲节礼盒装适合作为送爸爸的礼物。",
    "bridge_comparison_object": "同类旧方案",
    "bridge_comparison_object_evidence_type": "jtbd_inferred",
    "bridge_difference_domain": "functional",
    "bridge_difference_type": "自身卖点陈述",
    "bridge_source_evidence": [
        "父亲节礼盒装，送爸爸",
        "凉感科技：接触凉感系数 ≥0.15 J/(cm²·s)，HLA TECH 认证",
        "抗菌速干 POLO，适合中年男性日常通勤穿着",
    ],
    "bridge_evidence_source": "商品信息",
    "product_id": "hotfix_hla_social_gift",
    "engine_node": {"relative_price_level": "低水位"},
}

RUNBEN_PAYLOAD = {
    "leaf_category": "宝宝防蚊水",
    "shop_name": "润本官方旗舰店",
    "second_level_category": "驱蚊用品",
    "third_level_category": "宝宝防蚊水",
    "brand_name": "润本",
    "product_name": "润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水",
    "price": "24.9",
    "core_selling_point": "派卡瑞丁A级驱蚊力，长效防蚊驱虫，温和无刺激",
    "core_selling_point_source": "caller_provided.core_selling_points",
    "target_people": "婴幼儿/儿童/家庭日常户外人群",
    "bridge_comparison_object": "同类旧方案",
    "bridge_comparison_object_evidence_type": "jtbd_inferred",
    "bridge_difference_domain": "functional",
    "bridge_difference_type": "自身卖点陈述",
    "bridge_source_evidence": ["派卡瑞丁A级驱蚊力", "温和无刺激"],
    "bridge_evidence_source": "商品信息",
    "product_id": "hotfix_runben",
    "engine_node": {"relative_price_level": "高水位"},
}

BLUEMOON_PAYLOAD = {
    "leaf_category": "洗衣液",
    "shop_name": "蓝月亮官方旗舰店",
    "second_level_category": "衣物清洁护理",
    "third_level_category": "洗衣液",
    "brand_name": "蓝月亮",
    "product_name": "蓝月亮深层洁净护理洗衣液",
    "price": "39.9",
    "core_selling_point": "深层去污，低泡易漂，洁净护衣",
    "core_selling_point_source": "caller_provided.core_selling_points",
    "target_people": "家庭日常洗衣人群",
    "bridge_comparison_object": "同类旧方案",
    "bridge_comparison_object_evidence_type": "jtbd_inferred",
    "bridge_difference_domain": "functional",
    "bridge_difference_type": "自身卖点陈述",
    "bridge_source_evidence": ["深层去污", "低泡易漂", "洁净护衣"],
    "bridge_evidence_source": "商品信息",
    "product_id": "hotfix_bluemoon",
    "engine_node": {"relative_price_level": "低水位"},
}


def test_hla_social_domain_engine_diagnose_outputs_gift_roles_and_profile():
    out = ProductDiagnosisEngine().diagnose(HLA_PAYLOAD).dict(exclude_none=True)

    assert out["domain"] == "社会域"
    assert out["primary_task"] == "礼赠与关系表达"
    assert out["evidence_chain"]
    assert out["gate_reasons"]
    profile = out["persuasion_requirement_profile"]
    assert profile and profile.get("persuasion_requirements")
    assert out["evidence"]["module1_output"].get("target_people_raw")
    audience = out["product_target_audience"]
    assert audience["primary_audiences"]
    assert audience["secondary_audiences"]
    assert profile.get("persuasion_requirements")


def test_social_output_validator_declares_and_serializes_evidence_fields():
    category_matrix = CategoryIntentMatrix(
        ocean="蓝海", competition_focus="核心", frequency="耐消", domain_route_rule="功能域兜底",
        matrix_label="蓝海×核心×耐消", category_intent="占位", reasoning=["占位"]
    )
    product_matrix = ProductIntentMatrix(
        brand_tier="大牌官方", trust_barrier="极低", financial_risk="中", relative_price_level="低水位",
        matrix_label="大牌官方×低水位", business_category="T恤/Polo衫", median_price_threshold=120.0,
        price_value=198.0, product_intent="占位", reasoning=["占位"]
    )
    out = ProductDiagnosisOutput(
        leaf_category="T恤/Polo衫", shop_name="HLA", product_name="HLA polo", price=198.0,
        domain="社会域", primary_task="礼赠与关系表达", category_intent="占位", product_intent="占位",
        category_intent_matrix=category_matrix, product_intent_matrix=product_matrix,
        reasoning_path=["商品文本同时出现关系对象（爸爸）与礼赠场景（父亲节）"],
        gate_reasons=["商品文本同时出现关系对象（爸爸）与礼赠场景（父亲节）"],
        evidence_chain=[{"evidence_source": "商品信息", "evidence_text": "父亲节送爸爸"}],
        category="T恤/Polo衫", jtbd="礼赠与关系表达", resistance_profile={}, core_intent={},
    )
    serialized = out.dict(exclude_none=True)
    assert serialized["evidence_chain"][0]["evidence_text"] == "父亲节送爸爸"
    assert serialized["gate_reasons"]


def test_non_social_runben_and_bluemoon_still_diagnose():
    for payload in (RUNBEN_PAYLOAD, BLUEMOON_PAYLOAD):
        out = ProductDiagnosisEngine().diagnose(payload).dict(exclude_none=True)
        assert out["domain"] != "社会域"
        assert out["persuasion_requirement_profile"]["persuasion_requirements"]
        assert out["product_target_audience"]["primary_audiences"]
