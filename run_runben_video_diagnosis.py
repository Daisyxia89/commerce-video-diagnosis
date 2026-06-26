"""润本驱蚊液 · 视频说服诊断端到端 runner（Block 2 验收）。

复用商品侧诊断输出（outputs/runben_diagnosis/runben_full_diagnosis.json，由
run_runben_full_diagnosis.py 生成），构造润本视频样本，跑 VideoDiagnosisEngine，
输出 outputs/runben_diagnosis/runben_video_diagnosis.json。

润本视频样本信号（PRD 指定）：
  700只蚊子挑战 / 15分钟无包 / 小朋友也能用 / 无味道 / 大瓶家用+小瓶便携
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent))

from commerce_video_diagnosis.understanding.engines.video_diagnoser import (  # noqa: E402
    VideoDiagnosisEngine,
)

OUTPUT_DIR = ROOT / "outputs" / "runben_diagnosis"
PRODUCT_DIAGNOSIS_PATH = OUTPUT_DIR / "runben_full_diagnosis.json"


def _load_product_diagnosis() -> dict:
    """加载商品诊断输出；若不存在则先跑 run_runben_full_diagnosis.py 生成。"""
    if not PRODUCT_DIAGNOSIS_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "run_runben_full_diagnosis.py")], check=True)
    return json.loads(PRODUCT_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


def build_payload() -> dict:
    diagnosis = _load_product_diagnosis()
    source_product_id = diagnosis.get("product_id") or "runben_repellent_24p9"

    product_hec0 = (diagnosis.get("product_hecs") or [{}])[0]
    product_HEC = {
        "hook_tag": product_hec0.get("hook_tag"),
        "effect_tag": product_hec0.get("effect_tag"),
        "cta_tag": product_hec0.get("cta_tag"),
    }

    product_diagnosis = {
        "source_product_id": source_product_id,
        "product_fact_vector": {
            "leaf_category": diagnosis.get("leaf_category"),
            "primary_task": diagnosis.get("primary_task"),
            "brand_tier": diagnosis.get("product_intent_matrix", {}).get("brand_tier"),
            "relative_price_level": diagnosis.get("product_intent_matrix", {}).get("relative_price_level"),
            "price": diagnosis.get("price"),
        },
        "product_target_audience": diagnosis.get("product_target_audience"),
        "persuasion_requirement_profile": diagnosis.get("persuasion_requirement_profile"),
        "product_HEC": product_HEC,
    }

    # 润本视频样本：用 primary_hec 入参以演示 Block 1.2 命名契约映射（会产生 warning）
    video_understanding = {
        "video_id": "runben_video_001",
        "source_product_id": source_product_id,
        "primary_hec": {
            # B1：HEC hook H1 → H5（物理安全域合法 Hook），effect/cta 不变。
            "hook_tag": "H5",
            "effect_tag": "E1",
            "cta_tag": "C4",
            "signature": "H5安全场景切入→E1效果测评→C4人群场景总结",
        },
        "slider_signature": {
            "visual": {"score": 0.82, "evidence": "700只蚊子挑战强视觉冲击/猎奇画面"},
            "audio": {"score": 0.6, "evidence": "口播稳，讲解清晰"},
            "proof": {"score": 0.75, "evidence": "15分钟无包实测过程"},
            # B2：CTA 0.6 → 0.3（收口偏弱，落到参照人群 cta 区间下界以下）。
            "cta": {"score": 0.3, "evidence": "大瓶家用+小瓶便携场景总结收口偏弱"},
        },
        "storyboard_segments": [
            {
                "segment_id": "seg_hook_1",
                "role": "hook",
                "asr": "实测700只蚊子挑战，看看这瓶驱蚊液到底防不防蚊，被叮一身红肿包真的很困扰",
                "ocr": "700只蚊子挑战",
            },
            {
                "segment_id": "seg_effect_1",
                "role": "effect",
                "asr": "手臂放进蚊箱15分钟拿出来，一个包都没有，真的无包，驱蚊效果肉眼可见",
                "ocr": "15分钟 无包",
            },
            {
                "segment_id": "seg_effect_2",
                "role": "effect",
                "asr": "成分温和，小朋友也能用，喷上去无味道不刺鼻，敏感肌也安全",
                "ocr": "小朋友也能用 · 无味道",
            },
            {
                "segment_id": "seg_effect_3",
                "role": "effect",
                # B3：mid_high 消费力线索（官方/正品/成分/品质/派卡瑞丁）+ 部分权威背书证据
                "asr": "认准官方旗舰店正品，派卡瑞丁成分通过检测报告认证，品质有保障",
                "ocr": "官方正品 · 检测报告 认证",
            },
            {
                "segment_id": "seg_cta_1",
                # B3：low 消费力线索（划算/大瓶家用）+ 年轻(学生)与男性(户外)少数信号
                "role": "cta",
                "asr": "大瓶家用更划算够全家一夏天，小瓶便携出门带，学生党宿舍、户外露营也适合，一套搞定",
                "ocr": "大瓶家用划算+小瓶便携",
            },
        ],
        "semantic_bundles": [
            {
                "bundle_id": "bundle_1",
                "bundle_role": "effect",
                "text": "700只蚊子挑战实测，15分钟无包，驱蚊效果可见，官方品质保障",
            }
        ],
        "evidence_spans": [
            {"span_id": "span_pain", "text": "蚊子太多被叮一身包很困扰"},
            {"span_id": "span_proof", "text": "15分钟无包，700只蚊子挑战实测效果"},
            {"span_id": "span_safety", "text": "小朋友也能用，温和无味道无刺激"},
            {"span_id": "span_cta", "text": "大瓶家用小瓶便携，家庭驱蚊一套搞定更划算"},
        ],
    }

    return {"product_diagnosis": product_diagnosis, "video_understanding": video_understanding}


def main() -> None:
    payload = build_payload()
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(payload)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "runben_video_diagnosis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    vr = result["video_persuasion_diagnosis_result"]
    print(json.dumps(
        {
            "output_path": str(out_path.relative_to(ROOT)),
            "sections": list(vr.keys()),
            "input_validation_warnings": vr["input_validation"]["warnings"],
            "video_primary_audiences": [a["audience_group"] for a in vr["video_target_audience"]["primary_audiences"]],
            "audience_match_status": vr["audience_match_diagnosis"]["match_status"],
            "profile_match_status": vr["profile_match_diagnosis"]["match_status"],
            "hec_match_status": vr["hec_match_diagnosis"]["match_status"],
            "slider_match_status": vr["slider_match_diagnosis"]["match_status"],
            "summary_overall_status": vr["diagnosis_summary"]["overall_status"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
