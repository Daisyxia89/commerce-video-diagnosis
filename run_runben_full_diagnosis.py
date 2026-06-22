"""End-to-end product diagnosis for 润本驱蚊液.

Runs the official ProductDiagnosisEngine (Stage A → hard-gate JTBD = 物理安全与风险规避),
then attaches the full persuasion_requirement_profile from the v3.1 engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# core_skill 在仓库根目录下，需要把项目根追加到 path 末尾（不能 insert 0，否则会覆盖本地 commerce_video_diagnosis）
sys.path.append(str(ROOT.parent))

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DiagnosticInput,
    ProductDiagnosisEngine,
)
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import (  # noqa: E402
    build_persuasion_requirement_profile,
)


PRODUCT_NAME = "【A级驱蚊力】润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水防蚊叮蚊怕花露水"
SHOP_NAME = "润本官方旗舰店"
LEAF_CATEGORY = "宝宝防蚊水"

# 桥接层显式提供：driving repellent → 风险降低 / functional / 同类旧方案 / jtbd_inferred
BRIDGE_SOURCE_EVIDENCE = [
    "派卡瑞丁15%/20% A级驱蚊力，防蚊8小时、耐汗保护7h",
    "7%款驱蚊酯日常居家温和不刺激，第三方检测0刺激、减少刺激",
    "干扰蚊子嗅觉识别，皮肤表面形成气味屏障，避险闻不到咬不着，更安心",
    "广告审查号：粤农药广审（视）01260018号；第三方检测报告 2200938-1 / SHG211647 / ET2025-230，安全可溯",
]
CORE_SELLING_POINT = "派卡瑞丁A级驱蚊力，长效防蚊驱虫，温和无刺激"

payload = DiagnosticInput(
    leaf_category=LEAF_CATEGORY,
    shop_name=SHOP_NAME,
    second_level_category="驱蚊用品",
    third_level_category=LEAF_CATEGORY,
    brand_name="润本",
    product_name=PRODUCT_NAME,
    price="24.9",
    core_selling_point=CORE_SELLING_POINT,
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="婴幼儿/儿童/家庭日常户外人群",
    differentiator="",
    bridge_comparison_object="同类旧方案",
    bridge_comparison_object_evidence_type="jtbd_inferred",
    bridge_difference_domain="functional",
    bridge_difference_type="风险降低",
    bridge_source_evidence=BRIDGE_SOURCE_EVIDENCE,
    bridge_evidence_source="商品信息",
    product_id="runben_repellent_24p9",
    engine_node={"relative_price_level": "高水位"},
)

engine = ProductDiagnosisEngine()
diagnosis = engine.diagnose(payload)

# 从 engine 输出回填 product_fact 给 persuasion 引擎（jtbd_level2 必须来自 PRD 枚举）
_cat_matrix = diagnosis.category_intent_matrix
_cognition_attribute = f"{_cat_matrix.ocean}-{_cat_matrix.competition_focus}"
product_fact = {
    "leaf_category": LEAF_CATEGORY,
    "category": f"驱蚊用品 > {LEAF_CATEGORY}",
    "title": PRODUCT_NAME,
    "shop_name": SHOP_NAME,
    "price": 24.9,
    "price_attribute": diagnosis.product_intent_matrix.relative_price_level,
    "trust_attribute": diagnosis.product_intent_matrix.trust_barrier,
    # PRD 8.5.1：使用 cognition_attribute（品类竞争态势），原 cognitive_attribute 保留为兜底兼容
    "cognition_attribute": _cognition_attribute,
    "frequency_attribute": diagnosis.category_intent_matrix.frequency,
    "endorsement_attribute": "多项第三方检测报告与广告审查号",
    "channel_risk_attribute": "低",
    "jtbd_level1": diagnosis.domain,
    "jtbd_level2": diagnosis.primary_task,  # ← Stage A 裁决结果（PRD 枚举）
    "selling_points": [
        "派卡瑞丁15%/20% A级驱蚊力，驱蚊8小时",
        "驱蚊酯7%款日常居家温和不刺激",
        "不含香精/酒精，第三方检测0刺激",
        "可倒喷设计，30ml便携",
        "猫狗环境友好",
    ],
    "certifications": [
        "广告审查号：粤农药广审（视）01260018号",
        "第三方检测报告编号：2200938-1、SHG211647、ET2025-230",
        "0刺激/驱蚊时效/耐汗评价均有第三方检测报告支撑",
    ],
    "authority_endorsements": [
        "多项第三方检测报告",
        "广告审查号备案",
    ],
    "evidence": [
        "7%款驱蚊6.5h",
        "15%款驱蚊8h、耐汗保护7h",
        "20%款驱蚊8h、驱蠓6.5h",
    ],
    "source_evidence": (
        "商品图：A级驱蚊力 派卡瑞丁15%/20% 第三方检测报告编号 2200938-1 SHG211647 ET2025-230 "
        "广告审查号粤农药广审（视）01260018号"
    ),
    "risk_points": [
        "婴幼儿/敏感肌肤是否安全",
        "驱蚊成分浓度选择",
        "实际驱蚊时长是否达标",
    ],
}

profile = build_persuasion_requirement_profile(product_fact, content_goal="purchase")

# 把 profile 挂回 ProductDiagnosisOutput，输出完整协议
out = diagnosis.dict(exclude_none=True)
out["persuasion_requirement_profile"] = profile

OUTPUT_DIR = ROOT / "outputs" / "runben_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

full_path = OUTPUT_DIR / "runben_full_diagnosis.json"
slim_path = OUTPUT_DIR / "runben_slim_diagnosis.json"
md_path = OUTPUT_DIR / "runben_persuasion_profile_report.md"

# 1) full 诊断
full_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# 2) slim 诊断：剥离体量较大的 persuasion_requirement_profile，仅保留 main_persuasion_route 摘要
slim = {k: v for k, v in out.items() if k != "persuasion_requirement_profile"}
slim["persuasion_requirement_profile_summary"] = {
    "main_persuasion_route": profile.get("main_persuasion_route"),
    "profile_version": profile.get("profile_version"),
    "content_goal": profile.get("content_goal"),
    "persuasion_requirements_count": len(profile.get("persuasion_requirements", []) or []),
}
slim_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

# 3) MD 报告：补入 CandidateSet / product_ec_skeletons / product_hecs 三节
def _md_table_h(items):
    lines = ["| code | label | hook_tag |", "| --- | --- | --- |"]
    for it in items:
        lines.append(f"| {it.get('code','')} | {it.get('label','')} | {it.get('hook_tag','')} |")
    return "\n".join(lines)


def _md_table_e(items):
    lines = ["| code | label | effect_tag | completion_capabilities |", "| --- | --- | --- | --- |"]
    for it in items:
        caps = ", ".join(it.get("completion_capabilities", []) or [])
        lines.append(f"| {it.get('code','')} | {it.get('label','')} | {it.get('effect_tag','')} | {caps} |")
    return "\n".join(lines)


def _md_table_c(items):
    lines = ["| code | label | cta_tag | close_strength | fallback_priority |", "| --- | --- | --- | --- | --- |"]
    for it in items:
        fb = ", ".join(it.get("fallback_priority", []) or [])
        lines.append(
            f"| {it.get('code','')} | {it.get('label','')} | {it.get('cta_tag','')} | "
            f"{it.get('close_strength','')} | {fb} |"
        )
    return "\n".join(lines)


candidate_set = out.get("candidate_set", {}) or {}
ec_skeletons = out.get("product_ec_skeletons", []) or []
hecs = out.get("product_hecs", []) or []
mpr = profile.get("main_persuasion_route", {}) or {}
cat_resistance = mpr.get("category_resistance", {}) or {}
prod_barrier = mpr.get("product_conversion_barrier", {}) or {}
primary_jtbd = mpr.get("primary_jtbd", {}) or {}

ec_lines = ["| # | effect_tag | cta_tag | effect_label | cta_label | cta_resolution |", "| --- | --- | --- | --- | --- | --- |"]
for idx, ec in enumerate(ec_skeletons, 1):
    resolution = ec.get("cta_resolution", {}) or {}
    ec_lines.append(
        f"| {idx} | {ec.get('effect_tag','')} | {ec.get('cta_tag','')} | "
        f"{ec.get('effect_label','')} | {ec.get('cta_label','')} | "
        f"{resolution.get('resolution_type','')} |"
    )

hec_lines = ["| # | variant_id | hook_tag | effect_tag | cta_tag | risk_flags |", "| --- | --- | --- | --- | --- | --- |"]
for idx, hec in enumerate(hecs, 1):
    risks = ", ".join(hec.get("risk_flags", []) or [])
    hec_lines.append(
        f"| {idx} | {hec.get('variant_id','')} | {hec.get('hook_tag','')} | "
        f"{hec.get('effect_tag','')} | {hec.get('cta_tag','')} | {risks} |"
    )

md = f"""# 润本驱蚊液 · Persuasion Profile Report

## 1. 商品与诊断概览

- 商品：{PRODUCT_NAME}
- 店铺：{SHOP_NAME}
- 叶子类目：{LEAF_CATEGORY}
- 主任务（JTBD level1 / level2）：{primary_jtbd.get('level1','')} / {primary_jtbd.get('level2','')}
- category_resistance.rule：`{cat_resistance.get('rule','')}`
- category_resistance.summary：{cat_resistance.get('summary','')}
- product_conversion_barrier.rule：`{prod_barrier.get('rule','')}`

## 2. CandidateSet

- jtbd：{candidate_set.get('jtbd','')}
- r_rule：{candidate_set.get('r_rule','')}
- p_rule：{candidate_set.get('p_rule','')}
- task_domain：{candidate_set.get('task_domain','')}
- persuasion_route：{candidate_set.get('persuasion_route','')}

### 2.1 候选 H 库（candidate_set.h_list，{len(candidate_set.get('h_list', []) or [])} 条）

{_md_table_h(candidate_set.get('h_list', []) or [])}

### 2.2 Core E-list（candidate_set.effect_list，{len(candidate_set.get('effect_list', []) or [])} 条）

{_md_table_e(candidate_set.get('effect_list', []) or [])}

### 2.3 Core C-list（candidate_set.cta_list，{len(candidate_set.get('cta_list', []) or [])} 条）

{_md_table_c(candidate_set.get('cta_list', []) or [])}

## 3. product_ec_skeletons（EC 主链，共 {len(ec_skeletons)} 条）

{chr(10).join(ec_lines)}

## 4. product_hecs（HEC variants，共 {len(hecs)} 条）

{chr(10).join(hec_lines)}
"""

md_path.write_text(md, encoding="utf-8")

print(json.dumps({
    "full_path": str(full_path.relative_to(ROOT)),
    "slim_path": str(slim_path.relative_to(ROOT)),
    "md_path": str(md_path.relative_to(ROOT)),
    "core_e_list": [e.get("code") for e in candidate_set.get("effect_list", []) or []],
    "core_c_list": [c.get("code") for c in candidate_set.get("cta_list", []) or []],
    "candidate_h_list": [h.get("code") for h in candidate_set.get("h_list", []) or []],
    "ec_skeletons_count": len(ec_skeletons),
    "hec_variants_count": len(hecs),
    "category_resistance_rule": cat_resistance.get("rule"),
}, ensure_ascii=False, indent=2))
