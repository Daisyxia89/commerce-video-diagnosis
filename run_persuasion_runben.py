"""[引擎单测 / 演示脚本] persuasion_requirement_profile 引擎独立验证。

⚠️ 用途说明（第一批 F5+F3 裁定）：
- 本脚本仅用于直接喂 product_fact、单独验证 PersuasionRequirementEngine 的产出，
  属于 profile 引擎的单测 / 演示，**不代表商品诊断全链路 contract**。
- 商品诊断正式链路一律以 `ProductDiagnosisEngine.diagnose()` 的输出为准，
  profile 由引擎内部产出，禁止用本脚本的旁路 product_fact 当作线上口径。
- 本脚本保留、不纳入第一批主链路改造。
"""
import json, sys
sys.path.insert(0, '.')
sys.path.append('..')
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import build_persuasion_requirement_profile

product_fact = {
    "leaf_category": "宝宝防蚊水",
    "category": "驱蚊用品 > 宝宝防蚊水",
    "title": "【A级驱蚊力】润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水防蚊叮蚊怕花露水",
    "shop_name": "润本官方旗舰店",
    "price": 24.9,
    "price_attribute": "中低价",
    "trust_attribute": "中",
    "cognitive_attribute": "中等认知门槛",
    "frequency_attribute": "季节性中频",
    "endorsement_attribute": "多项第三方检测报告与广告审查号",
    "channel_risk_attribute": "低",
    "jtbd_level1": "功能任务",
    "jtbd_level2": "户外/居家防蚊驱蚊",
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
    "source_evidence": "商品图：A级驱蚊力 派卡瑞丁15%/20% 第三方检测报告编号 2200938-1 SHG211647 ET2025-230 广告审查号粤农药广审（视）01260018号",
    "risk_points": [
        "婴幼儿/敏感肌肤是否安全",
        "驱蚊成分浓度选择",
        "实际驱蚊时长是否达标",
    ],
}

profile = build_persuasion_requirement_profile(product_fact, content_goal="purchase")
print(json.dumps(profile, ensure_ascii=False, indent=2))
