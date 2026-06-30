"""HLA 视频诊断 runner — 使用真实视频理解结果。

视频源：https://www.douyin.com/video/7645882806262082931
理解工具：inner_skills/analyze_media/analyze_video.py（多模态 LLM）
完整 ASR / OCR / 分段记录见 outputs/hla_diagnosis/hla_video_factpack.md
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

OUTPUT_DIR = ROOT / "outputs" / "hla_diagnosis"
PRODUCT_DIAGNOSIS_PATH = OUTPUT_DIR / "hla_full_diagnosis.json"


def _load_product_diagnosis() -> dict:
    if not PRODUCT_DIAGNOSIS_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "run_hla_full_diagnosis.py")], check=True)
    return json.loads(PRODUCT_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


def build_payload() -> dict:
    diagnosis = _load_product_diagnosis()
    source_product_id = diagnosis.get("product_id") or "hla_polo_98"

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

    # ===== 真实视频理解信号（来自 analyze_video，视频时长 25s，竖屏 1080x1920） =====
    # Hook：00:00-00:05 透明亮片礼盒 + 蓝色礼盒，BGM《父亲》"送给爸爸的父亲节礼物"
    # Effect：00:06-00:21 开箱仪式感、卡片祝福、衣领面料展示、袖口弹性、山川 logo、揉搓抗皱
    # CTA：00:22-00:25 "这么好看又实用的海澜之家短袖，哪位爸爸会拒绝呀～父亲节赶紧给老爸安排上！"
    # 无主播口播，纯花字 + BGM 父亲节情绪叙事
    video_understanding = {
        "video_id": "douyin_7645882806262082931",
        "source_product_id": source_product_id,
        "primary_hec": {
            # H1 情绪共鸣（父亲节情绪 BGM + 仪式开箱）；E6 卖点细节展示（凉感面料/弹性/抗皱/logo）；C1 节日紧迫
            "hook_tag": "H1",
            "effect_tag": "E6",
            "cta_tag": "C1",
            "signature": "H1父亲节情绪开箱→E6面料/弹性/logo细节→C1父亲节紧迫感收口",
        },
        "slider_signature": {
            "visual": {"score": 0.78, "evidence": "亮片透明礼盒+父爱如山卡片，仪式感视觉冲击强；面料特写、揉搓、袖口拉伸均有微距镜头"},
            "audio": {"score": 0.45, "evidence": "无主播口播，仅 BGM《父亲》歌词，audio 信息密度偏弱"},
            "proof": {"score": 0.55, "evidence": "面料揉搓不变形 + 袖口弹性拉伸 + 山川 logo 特写，但凉感系数/检测报告等核心 selling point 未在画面/字幕中露出"},
            "cta": {"score": 0.65, "evidence": "末尾文案『父亲节赶紧给老爸安排上』+ 节日紧迫，但缺少明确购买路径/价格/限时优惠"},
        },
        "storyboard_segments": [
            {
                "segment_id": "seg_hook_1",
                "role": "hook",
                "asr": "（BGM 父亲）总是向你索取，却不曾说谢谢你",
                "ocr": "送给爸爸的父亲节礼物",
            },
            {
                "segment_id": "seg_hook_2",
                "role": "hook",
                "asr": "（BGM 父亲）直到长大以后，才懂得你不容易",
                "ocr": "其实爸爸也是需要被爱的",
            },
            {
                "segment_id": "seg_effect_1",
                "role": "effect",
                "asr": "（BGM）每次离开总是装作轻松的样子",
                "ocr": "偷偷给他准备了海澜之家 父爱如山惊喜礼盒～；打开礼盒仪式感满满",
            },
            {
                "segment_id": "seg_effect_2",
                "role": "effect",
                "asr": "（BGM）微笑说回去吧，转身又泪流满面",
                "ocr": "还有祝福语卡片 把对爸爸的爱也要说出来；海澜之家山不在高短袖系列～",
            },
            {
                "segment_id": "seg_effect_3",
                "role": "effect",
                "asr": "（BGM）我愿用我一切，换你岁月长留",
                "ocr": "简洁翻领设计 不易变形；采用夏天专属凉感面料 爱出汗的爸爸也很合适～；袖口有弹性 不勒手臂",
            },
            {
                "segment_id": "seg_effect_4",
                "role": "effect",
                "asr": "（BGM）一生要强的爸爸",
                "ocr": "胸前精致山川 logo 寓意父爱如山；优选品质面料 揉搓不起球不变形；版型设计 时尚大方",
            },
            {
                "segment_id": "seg_cta_1",
                "role": "cta",
                "asr": "（BGM）我能为你做点什么",
                "ocr": "这么好看又实用的海澜之家短袖；哪位爸爸会拒绝呀～ 父亲节赶紧给老爸安排上！",
            },
        ],
        "semantic_bundles": [
            {
                "bundle_id": "bundle_gift",
                "bundle_role": "hook",
                "text": "父亲节情绪礼盒开箱仪式感，父爱如山祝福卡片",
            },
            {
                "bundle_id": "bundle_fabric",
                "bundle_role": "effect",
                "text": "凉感面料 + 袖口弹性 + 揉搓不起球不变形 + 山川 logo 寓意父爱如山",
            },
            {
                "bundle_id": "bundle_cta",
                "bundle_role": "cta",
                "text": "父亲节赶紧给老爸安排上，海澜之家短袖好看又实用",
            },
        ],
        "evidence_spans": [
            {"span_id": "span_emotion", "text": "送给爸爸的父亲节礼物 / 其实爸爸也是需要被爱的"},
            {"span_id": "span_fabric", "text": "夏天专属凉感面料 爱出汗的爸爸也很合适"},
            {"span_id": "span_logo", "text": "胸前精致山川 logo 寓意父爱如山"},
            {"span_id": "span_durability", "text": "优选品质面料 揉搓不起球不变形 / 袖口有弹性 不勒手臂"},
            {"span_id": "span_cta", "text": "父亲节赶紧给老爸安排上"},
        ],
    }

    return {"product_diagnosis": product_diagnosis, "video_understanding": video_understanding}


def main() -> None:
    payload = build_payload()
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(payload)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hla_video_diagnosis.json"
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
