import json, sys, traceback
sys.path.insert(0, ".")
from commerce_video_diagnosis.understanding.engines.product_diagnoser import ProductDiagnosisEngine

base_bridge = dict(
    differentiator="",
    bridge_comparison_object="",
    bridge_comparison_object_evidence_type="null",
    bridge_difference_domain="functional",
    bridge_difference_type="自身卖点陈述",
    bridge_evidence_source="商品信息",
)
samples = {
 "B10_N05": dict(leaf_category="洗发水", shop_name="纯妈官方旗舰店", product_name="纯妈二硫化硒控油去屑洗发精华素", price="65.0", core_selling_point="控油去屑；日常洗护头皮，减少洗头频率省时省力", target_people="有头皮出油、头屑、头皮红痒长痘以及洗头掉发问题的人群"),
 "B20_N14": dict(leaf_category="洗发水", shop_name="侧柏叶旗舰店", product_name="侧柏叶洗发水植物天然草本何首乌皂角养发护发止痒去屑控油", price="33.9", core_selling_point="植物天然草本，养发护发止痒去屑控油，成分天然温和", target_people="有头发油腻头屑多且偏好天然草本成分的人群"),
 "B20_N02": dict(leaf_category="卸妆膏", shop_name="YOUFE旗舰店", product_name="YOUFE净透卸妆膏小金砖快乳化眼唇脸清洁油乳", price="49.9", core_selling_point="快乳化卸妆，省时一步到位清洁眼唇脸，日常基础卸妆", target_people="用过老款的粉丝及有高效卸妆需求的美妆爱好者"),
 "B29_N31": dict(leaf_category="面膜", shop_name="怡末旗舰店", product_name="怡末粉面膜微囊贴片致臻面膜补水保湿紧致抗皱", price="99.0", core_selling_point="补水保湿紧致抗皱，改善松弛暗沉熬夜后状态，效果测评向", target_people="有补水保湿紧致抗皱需求的女性，爱熬夜人群"),
}
eng = ProductDiagnosisEngine()
for sid, inp in samples.items():
    payload = dict(inp); payload.update(base_bridge)
    payload["core_selling_point_source"]="caller_provided.core_selling_points"
    payload["bridge_source_evidence"]=[inp["product_name"], inp["core_selling_point"]]; payload["engine_node"]={"relative_price_level":"低水位"}
    print("="*60); print(sid)
    try:
        out = eng.diagnose(payload)
        d = out.to_protocol_dict() if hasattr(out,"to_protocol_dict") else out.dict()
        print("  jtbd/jtbd_level1:", d.get("jtbd"), "|", d.get("jtbd_level1"))
        print("  secondary_benefits:", d.get("secondary_benefits"))
        print("  non_selected_task_reasons:", json.dumps(d.get("non_selected_task_reasons"), ensure_ascii=False))
        print("  has primary_task key:", "primary_task" in d)
    except Exception as e:
        print("  EXCEPTION:", type(e).__name__, str(e)[:400])
