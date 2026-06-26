"""KAMEIER 卡玫尔牙刷视频诊断 runner（基于 analyze_video 真实理解结果）。"""
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

OUTPUT_DIR = ROOT / "outputs" / "kameier_diagnosis"
PRODUCT_DIAGNOSIS_PATH = OUTPUT_DIR / "kameier_product_diagnosis.json"


def _load_product_diagnosis() -> dict:
    if not PRODUCT_DIAGNOSIS_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "run_kameier_full_diagnosis.py")], check=True)
    return json.loads(PRODUCT_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


# ========== FactPack（来自 inner_skills/analyze_media/analyze_video，
# 视频 https://www.douyin.com/video/7651824813833716901，时长 39s，720x960） ==========
FACT_PACK = {
    "video_meta": {
        "source_platform": "douyin",
        "video_url": "https://www.douyin.com/video/7651824813833716901",
        "duration_sec": 39.6,
        "fps": 30,
        "resolution": "720x960",
    },
    "segments": [
        {
            "segment_id": "seg_01", "start_sec": 0, "end_sec": 3, "role": "hook",
            "visual_facts": "剪刀剪开牙刷包装特写；橙、蓝两支峰形刷头并排对比；洗手池台面。",
            "audio_facts": "口播：我猜这种特别的牙刷你一定没有用过吧。语速平缓有引导感。",
            "ocr_facts": "字幕：我猜这种特别的牙刷",
            "rhythm_facts": "开场即产品特写，节奏紧凑。",
        },
        {
            "segment_id": "seg_02", "start_sec": 3, "end_sec": 6, "role": "hook",
            "visual_facts": "多支不同颜色牙刷交错叠放；将牙刷垂直插入洗漱杯。",
            "audio_facts": "口播：不用牙膏都可以把牙齿刷得干净。牙刷不一定要买很贵的，但要勤换新。",
            "ocr_facts": "字幕：你一定没有用过吧 / 不用牙膏都可以把牙齿刷的干净 / 牙刷不一定要买很贵的",
            "rhythm_facts": "中等速度，多物展示。",
        },
        {
            "segment_id": "seg_03", "start_sec": 6, "end_sec": 9, "role": "hook",
            "visual_facts": "旧蓝牙刷放进透明水杯搅拌，水变浑浊起沫，呈现旧牙刷脏污对比。",
            "audio_facts": "口播：别嫌我啰嗦，长时间不换用起来是真的脏。",
            "ocr_facts": "字幕：但要勤换新 / 别嫌我啰嗦 / 长时间不换用起来是真的脏",
            "rhythm_facts": "对比放慢，强调脏污痛点。",
        },
        {
            "segment_id": "seg_04", "start_sec": 9, "end_sec": 12, "role": "effect",
            "visual_facts": "手持多包封装牙刷（每包两支）展示；切换至峰形刷头近景。",
            "audio_facts": "口播：你就学我备上这款最近特别爆火的凸面中硬毛牙刷。",
            "ocr_facts": "字幕：你就学我 / 备上这款最近特别爆火的 / 凸面中硬毛牙刷",
            "rhythm_facts": "平滑过渡，引入主推单品。",
        },
        {
            "segment_id": "seg_05", "start_sec": 12, "end_sec": 15, "role": "effect",
            "visual_facts": "蓝色牙刷在仿真牙模上演示刷牙；展示刷头菱形植毛设计特写。",
            "audio_facts": "口播：用一次就感觉以前的牙白刷了。它是这种菱形植毛的设计。",
            "ocr_facts": "字幕：用一次就感觉以前的牙白刷了 / 它是这种菱形植毛的设计",
            "rhythm_facts": "演示与口播同步。",
        },
        {
            "segment_id": "seg_06", "start_sec": 15, "end_sec": 20, "role": "effect",
            "visual_facts": "刷头特写：中间凸起四周低；两支刷头上下咬合演示125°峰形弧面。",
            "audio_facts": "口播：中间凸起来四周低。刷面是这种120度尖峰弧面设计。",
            "ocr_facts": "字幕：中间凸起来四周低 / 刷面是这种120度尖峰弧面设计",
            "rhythm_facts": "微距特写，节奏细致。",
        },
        {
            "segment_id": "seg_07", "start_sec": 20, "end_sec": 25, "role": "effect",
            "visual_facts": "刷头在手背皮肤上滑动测试温和度；在牙模内侧/缝隙演示贴合。",
            "audio_facts": "口播：可以更好贴合牙窝深处。符合巴氏刷牙法。平时刷不到的牙齿缝隙都能照顾到。",
            "ocr_facts": "字幕：可以更好的贴合我们的牙窝深处 / 符合巴氏刷牙法 / 平时刷不到的牙齿缝隙都能照顾到",
            "rhythm_facts": "动作演示逻辑性强。",
        },
        {
            "segment_id": "seg_08", "start_sec": 25, "end_sec": 30, "role": "effect",
            "visual_facts": "展示4支装家庭包；手指拨动螺旋刷丝表现弹力。",
            "audio_facts": "口播：新品冲销量，到手四支是真的划算。我超爱它的弹力螺旋刷毛。",
            "ocr_facts": "字幕：新品冲销量 / 到手四支是真的划算 / 我超爱它的弹力螺旋刷毛",
            "rhythm_facts": "强调利益点，节奏加快。",
        },
        {
            "segment_id": "seg_09", "start_sec": 30, "end_sec": 35, "role": "effect",
            "visual_facts": "粉色膏体涂层牙模上清洁演示；展示防滑手柄流线设计。",
            "audio_facts": "口播：刷牙强劲有力，清洁力强同时不伤娇嫩牙龈。符合人体工学的防滑手柄。",
            "ocr_facts": "字幕：刷牙强劲有力 / 清洁力强的同时 / 也不会伤害娇嫩的牙龈 / 符合人体工学的防滑手柄",
            "rhythm_facts": "细节全方位展示。",
        },
        {
            "segment_id": "seg_10", "start_sec": 35, "end_sec": 39.6, "role": "cta",
            "visual_facts": "刷头侧面配色展示；牙刷插杯；最后展示台面整套包装。",
            "audio_facts": "口播：配色好看又很好抓握。总感觉牙齿刷不干净，又不舍得换牙刷的，真的可以试试。",
            "ocr_facts": "字幕：配色好看又很好抓握 / 总感觉牙齿刷不干净 / 又不舍得换牙刷的 / 真的可以试试",
            "rhythm_facts": "收尾稳健，软性试用号召。",
        },
    ],
    "full_asr_transcript": (
        "我猜这种特别的牙刷你一定没有用过吧。不用牙膏都可以把牙齿刷得干净。"
        "牙刷不一定要买很贵的，但要勤换新。别嫌我啰嗦，长时间不换用起来是真的脏。"
        "你就学我备上这款最近特别爆火的凸面中硬毛牙刷。用一次就感觉以前的牙白刷了。"
        "它是这种菱形植毛的设计。中间凸起来四周低。刷面是这种120度尖峰弧面设计。"
        "可以更好的贴合我们的牙窝深处。符合巴氏刷牙法。平时刷不到的牙齿缝隙都能照顾到。"
        "新品冲销量，到手四支是真的划算。我超爱它的弹力螺旋刷毛。刷牙强劲有力。"
        "清洁力强的同时也不会伤害娇嫩的牙龈。符合人体工学的防滑手柄。配色好看又很好抓握。"
        "总感觉牙齿刷不干净，又不舍得换牙刷的，真的可以试试。"
    ),
}

# 视频侧 HEC（从 factpack 推断）：
# H3 反差/对比（旧脏牙刷 vs 新峰形牙刷）；
# E6 卖点细节展示（菱形植毛 / 120°尖峰弧面 / 螺旋刷毛 / 巴氏刷牙法 / 人体工学手柄）；
# C1 促销利益/试用呼吁（新品冲销量4支划算 + 真的可以试试）
VIDEO_PRIMARY_HEC = {
    "hook_tag": "H3",
    "effect_tag": "E6",
    "cta_tag": "C1",
    "signature": "H3旧牙刷脏水反差→E6峰形/螺旋刷丝/巴氏贴合细节展示→C1四支划算+试试看",
}

SLIDER_SIGNATURE = {
    "visual": {"score": 0.78, "evidence": "峰形刷头特写/旧牙刷脏水搅拌/牙模刷牙/螺旋刷丝/手柄流线均有微距与对比镜头，视觉密度高"},
    "audio": {"score": 0.82, "evidence": "全程主播口播叙事，节奏清晰，痛点-机制-利益-收口结构完整，BGM 弱化"},
    "proof": {"score": 0.55, "evidence": "牙模演示+脏水对比+菱形植毛/120°尖峰展示可视证据足；但缺第三方检测、品牌权威背书、用户口碑等社会证据"},
    "cta": {"score": 0.50, "evidence": "末段仅软性 '真的可以试试' + '新品冲销量4支划算'，无明确购买路径/价格/限时/无忧承诺"},
}


def build_payload() -> dict:
    diagnosis = _load_product_diagnosis()
    source_product_id = diagnosis.get("product_id") or "kameier_brush_4pack"

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
            "trust_attribute": diagnosis.get("brand_whitelist_routing", {}).get("trust_attribute"),
            "trust_barrier": diagnosis.get("brand_whitelist_routing", {}).get("trust_barrier"),
        },
        "product_target_audience": diagnosis.get("product_target_audience"),
        "persuasion_requirement_profile": diagnosis.get("persuasion_requirement_profile"),
        "product_HEC": product_HEC,
    }

    storyboard_segments = []
    semantic_bundles_by_role = {"hook": [], "effect": [], "cta": []}
    evidence_spans = []
    for seg in FACT_PACK["segments"]:
        storyboard_segments.append({
            "segment_id": seg["segment_id"],
            "role": seg["role"],
            "asr": seg["audio_facts"],
            "ocr": seg["ocr_facts"],
        })
        semantic_bundles_by_role[seg["role"]].append(seg["visual_facts"] + "｜" + seg["ocr_facts"])
        evidence_spans.append({"span_id": f"span_{seg['segment_id']}", "text": seg["ocr_facts"] + "｜" + seg["audio_facts"]})

    semantic_bundles = [
        {"bundle_id": f"bundle_{role}", "bundle_role": role, "text": " || ".join(items)}
        for role, items in semantic_bundles_by_role.items() if items
    ]

    video_understanding = {
        "video_id": "douyin_7651824813833716901",
        "source_product_id": source_product_id,
        "primary_hec": VIDEO_PRIMARY_HEC,
        "slider_signature": SLIDER_SIGNATURE,
        "storyboard_segments": storyboard_segments,
        "semantic_bundles": semantic_bundles,
        "evidence_spans": evidence_spans,
    }

    return {"product_diagnosis": product_diagnosis, "video_understanding": video_understanding}


def main() -> None:
    payload = build_payload()
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(payload)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "kameier_video_factpack.json").write_text(
        json.dumps(FACT_PACK, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "kameier_video_understanding_blueprint.json").write_text(
        json.dumps({"primary_hec": VIDEO_PRIMARY_HEC, "slider_signature": SLIDER_SIGNATURE}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_path = OUTPUT_DIR / "kameier_video_diagnosis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 合并完整诊断
    product_full = json.loads(PRODUCT_DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    merged = {
        "brand_whitelist_route": product_full.get("brand_whitelist_routing"),
        "product_diagnosis": product_full,
        "video_factpack": FACT_PACK,
        "video_understanding_blueprint": {
            "primary_hec": VIDEO_PRIMARY_HEC,
            "slider_signature": SLIDER_SIGNATURE,
        },
        "video_diagnosis": result,
        "price_note": product_full.get("price_note"),
    }
    (OUTPUT_DIR / "kameier_full_diagnosis.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vr = result["video_persuasion_diagnosis_result"]
    print(json.dumps(
        {
            "output_path": str(out_path.relative_to(ROOT)),
            "input_validation_warnings": vr["input_validation"].get("warnings", []),
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
