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

# F5/F3：persuasion_requirement_profile 现由 ProductDiagnosisEngine 内部产出并随主输出装配，
# runner 不再后挂 profile。直接从引擎输出读取并断言非空（为空 Crash Early，不兜底生成）。
out = diagnosis.dict(exclude_none=True)
profile = out.get("persuasion_requirement_profile")
if not profile or not profile.get("persuasion_requirements"):
    raise ValueError(
        "引擎输出的 persuasion_requirement_profile 为空或 persuasion_requirements 为空，"
        "停止输出（Crash Early，不做后挂兜底）。"
    )

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
