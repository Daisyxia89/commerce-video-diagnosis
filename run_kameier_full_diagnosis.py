"""KAMEIER 卡玫尔牙刷端到端诊断 runner。"""
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

PRODUCT_NAME = "KAMEIER卡玫尔 4支峰型凸面中硬毛牙刷螺旋深层清洁护龈家用成人适用高档"
SHOP_NAME = "秉晟精品严选店"
LEAF_CATEGORY = "牙刷"
# 价格未知，4 支装白牌牙刷常见 19.9 兜底，仅用于过引擎 price>0 校验，不代表真实价。
PRICE = "19.9"
PRICE_FALLBACK_NOTE = "商品图未提供具体售价；引擎强校验 price>0，使用 19.9 占位（4支装白牌牙刷常见低位价）。该值非真实价格 SSOT。"

# Step 1 白名单：店铺 "秉晟精品严选店" 未命中 brand_whitelist.csv
TRUST_ATTRIBUTE = "白牌"
TRUST_BARRIER = "高"

CORE_SELLING_POINTS = [
    "山峰弧面牙刷，物理磨尖 + 螺旋刷丝",
    "125°峰形峰尖裁切，双效净护2合1（外侧温和护龈/内侧深层清洁）",
    "升级方孔菱形超密植毛：22孔大方孔，磨得圆、植得密、高矮错层",
    "错层双重植毛",
    "清洁牙黄/烟黄/茶渍，不易炸毛",
    "刀峰凸面精准清洁",
    "枫形高矮错层，适配巴氏刷牙法",
    "螺旋刷丝温和护龈",
    "口腔护理专业品牌，官方正品",
    "4支装家庭装",
]
CORE_SELLING_POINT_STR = "\n".join(CORE_SELLING_POINTS)

DIFFERENTIATOR_TEXT = (
    "125°峰尖裁切山峰弧面 + 22孔菱形超密植毛 + 螺旋刷丝护龈，"
    "对比传统平面牙刷实现双效净护、贴合牙窝、深入缝隙不漏刷。"
)

BRIDGE_SOURCE_EVIDENCE = [
    "125°峰尖裁切山峰弧面（商品图）",
    "22孔大方孔菱形超密植毛 + 高矮错层（商品图）",
    "物理磨尖 + 螺旋刷丝（商品图）",
    "适配巴氏刷牙法（卖点描述）",
    "4支家庭装（商品图）",
]

payload = DiagnosticInput(
    leaf_category=LEAF_CATEGORY,
    shop_name=SHOP_NAME,
    second_level_category="口腔护理",
    third_level_category="牙刷",
    brand_name="KAMEIER/卡玫尔",
    product_name=PRODUCT_NAME,
    price=PRICE,
    core_selling_point=CORE_SELLING_POINT_STR,
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="希望深层清洁牙渍/烟茶渍并兼顾护龈不勒嘴的成人",
    differentiator=DIFFERENTIATOR_TEXT,
    bridge_comparison_object="同类旧方案",
    bridge_comparison_object_evidence_type="null",
    bridge_difference_domain="functional",
    bridge_difference_type="自身卖点陈述",
    bridge_source_evidence=BRIDGE_SOURCE_EVIDENCE,
    bridge_evidence_source="商品信息",
    product_id="kameier_brush_4pack",
    engine_node={"relative_price_level": "低水位"},
)

engine = ProductDiagnosisEngine()
diagnosis = engine.diagnose(payload)

# F5/F3：persuasion_requirement_profile 现由 ProductDiagnosisEngine 内部产出并随主输出装配，
# runner 不再后挂 profile。直接从引擎输出读取并断言非空（为空 Crash Early，不兜底生成）。
out = diagnosis.dict(exclude_none=True)
profile = out.get("persuasion_requirement_profile")
if not profile or not profile.get("persuasion_requirements"):
    raise ValueError(
        "引擎输出的 persuasion_requirement_profile 为空或 persuasion_requirements 为空，"
        "停止输出（Crash Early，不做后挂兜底）。"
    )
out["brand_whitelist_routing"] = {
    "shop_name": SHOP_NAME,
    "hit": False,
    "trust_attribute": TRUST_ATTRIBUTE,
    "trust_barrier": TRUST_BARRIER,
    "source": "memory/topics/brand_whitelist.csv",
    "note": "未命中白名单，按白牌路由：trust_attribute=低，trust_barrier=高",
}
out["price_note"] = PRICE_FALLBACK_NOTE

OUTPUT_DIR = ROOT / "outputs" / "kameier_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
full_path = OUTPUT_DIR / "kameier_product_diagnosis.json"
full_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "full_path": str(full_path.relative_to(ROOT)),
    "domain": diagnosis.domain,
    "primary_task": diagnosis.primary_task,
    "brand_tier": diagnosis.product_intent_matrix.brand_tier,
    "trust_barrier_engine": diagnosis.product_intent_matrix.trust_barrier,
    "trust_barrier_route": TRUST_BARRIER,
    "relative_price_level": diagnosis.product_intent_matrix.relative_price_level,
    "hec_count": len(out.get("product_hecs", []) or []),
}, ensure_ascii=False, indent=2))
