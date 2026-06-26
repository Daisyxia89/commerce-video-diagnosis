"""HLA 海澜之家 山不在高 短袖 Polo 商品诊断 end-to-end runner。

复用 ProductDiagnosisEngine + persuasion_requirement_engine，输出：
- outputs/hla_diagnosis/hla_full_diagnosis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent))

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DiagnosticInput,
    ProductDiagnosisEngine,
)
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import (  # noqa: E402
    build_persuasion_requirement_profile,
)


PRODUCT_NAME_RAW = "【父亲节送礼】HLA海澜之家山不在高短袖polo26新凉感抗菌送爸爸"
# Stage A 走功能域：使用功能属性强化的 title 喂入引擎（保留原始 title 作为外部展示）
PRODUCT_NAME = "HLA海澜之家山不在高短袖polo 凉感抗菌速干"
SHOP_NAME = "海澜之家官方旗舰店"
LEAF_CATEGORY = "T恤/Polo衫"
PRICE = "98"

# Step 1：白名单查询（已确认）
# memory/topics/brand_whitelist.csv 命中 "海澜之家官方旗舰店,S6,箱包"
# 走白名单路由：trust_attribute=大牌官方，trust_barrier=极低
TRUST_ATTRIBUTE = "大牌官方"
TRUST_BARRIER = "极低"

BRIDGE_SOURCE_EVIDENCE = [
    "凉感科技：接触凉感系数 ≥0.15 J/(cm²·s)，HLA TECH 认证，通过 GB/T 35263-2017 检测合格",
    "吸湿排汗：渗透面吸水速率 ≥3级，通过 GB/T 21655.2-2019、GB/T 8829-2017 检测合格",
    "抗菌功能（短袖 polo 夏季穿着卫生）",
    "合体版型 + 高周波印花山川 logo",
    "尺码 XXS-6XL 全覆盖",
    "海澜之家品牌官方背书 + 父亲节礼盒装",
]
CORE_SELLING_POINT = "凉感系数≥0.15、抗菌、吸湿排汗速干的夏季短袖 polo"

payload = DiagnosticInput(
    leaf_category=LEAF_CATEGORY,
    shop_name=SHOP_NAME,
    second_level_category="男装",
    third_level_category=LEAF_CATEGORY,
    brand_name="HLA海澜之家",
    product_name=PRODUCT_NAME,
    price=PRICE,
    core_selling_point=CORE_SELLING_POINT,
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="夏季出汗多/需要凉感速干 polo 的中年男性",
    differentiator="",
    bridge_comparison_object="",
    bridge_comparison_object_evidence_type="null",
    bridge_difference_domain="functional",
    bridge_difference_type="自身卖点陈述",
    bridge_source_evidence=BRIDGE_SOURCE_EVIDENCE,
    bridge_evidence_source="商品信息",
    product_id="hla_polo_98",
    engine_node={"relative_price_level": "低水位"},
)

engine = ProductDiagnosisEngine()
diagnosis = engine.diagnose(payload)

_cat_matrix = diagnosis.category_intent_matrix
_cognition_attribute = f"{_cat_matrix.ocean}-{_cat_matrix.competition_focus}" if _cat_matrix.competition_focus else _cat_matrix.ocean
product_fact = {
    "leaf_category": LEAF_CATEGORY,
    "category": f"男装 > {LEAF_CATEGORY}",
    "title": PRODUCT_NAME,
    "shop_name": SHOP_NAME,
    "price": float(PRICE),
    "price_attribute": diagnosis.product_intent_matrix.relative_price_level,
    "trust_attribute": diagnosis.product_intent_matrix.trust_barrier,
    "cognition_attribute": _cognition_attribute,
    "frequency_attribute": diagnosis.category_intent_matrix.frequency,
    "endorsement_attribute": "HLA TECH 凉感认证 + GB/T 35263-2017、GB/T 21655.2-2019、GB/T 8829-2017 检测合格",
    "channel_risk_attribute": "低",
    "jtbd_level1": diagnosis.domain,
    "jtbd_level2": diagnosis.primary_task,
    "selling_points": [
        "凉感系数 ≥0.15 J/(cm²·s)，HLA TECH 认证",
        "吸湿排汗渗透面吸水速率 ≥3级",
        "抗菌功能",
        "合体版型 + 胸前山川 logo 父爱如山",
        "XXS-6XL 全尺码覆盖",
        "父亲节礼盒装，父子同款",
    ],
    "certifications": [
        "GB/T 35263-2017 凉感检测合格",
        "GB/T 21655.2-2019、GB/T 8829-2017 吸湿排汗检测合格",
        "HLA TECH 认证",
    ],
    "authority_endorsements": [
        "HLA 海澜之家品牌官方背书（白名单 S6 大牌官方）",
        "国标 GB/T 35263-2017、GB/T 21655.2-2019 第三方检测",
    ],
    "evidence": [
        "凉感系数 ≥0.15 J/(cm²·s)",
        "吸湿排汗 ≥3级",
        "尺码 XXS-6XL",
    ],
    "source_evidence": (
        "商品图：HLA TECH 凉感认证 ≥0.15 / GB T 35263-2017 / GB T 21655.2-2019 / "
        "山不在高系列 / 胸前山川 logo / 父亲节送礼场景"
    ),
    "risk_points": [
        "父亲节后场景紧迫感衰减",
        "中老年爸爸尺码与版型偏好不确定",
    ],
}

profile = build_persuasion_requirement_profile(product_fact, content_goal="purchase")

out = diagnosis.dict(exclude_none=True)
out["persuasion_requirement_profile"] = profile

# 标注白名单路由结果
out["brand_whitelist_routing"] = {
    "shop_name": SHOP_NAME,
    "hit": True,
    "rank": "S6",
    "primary_category_in_whitelist": "箱包",
    "trust_attribute": TRUST_ATTRIBUTE,
    "trust_barrier": TRUST_BARRIER,
    "source": "memory/topics/brand_whitelist.csv",
}

OUTPUT_DIR = ROOT / "outputs" / "hla_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
full_path = OUTPUT_DIR / "hla_full_diagnosis.json"
full_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "full_path": str(full_path.relative_to(ROOT)),
    "domain": diagnosis.domain,
    "primary_task": diagnosis.primary_task,
    "brand_tier": diagnosis.product_intent_matrix.brand_tier,
    "trust_barrier": diagnosis.product_intent_matrix.trust_barrier,
    "relative_price_level": diagnosis.product_intent_matrix.relative_price_level,
    "hec_count": len(out.get("product_hecs", []) or []),
}, ensure_ascii=False, indent=2))
