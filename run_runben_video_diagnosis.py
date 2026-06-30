"""润本驱蚊液 · 视频说服诊断端到端 runner（Block 2 验收 + 前端消费层契约装配）。

复用商品侧诊断输出（outputs/runben_diagnosis/runben_full_diagnosis.json，由
run_runben_full_diagnosis.py 生成），构造润本视频样本（裁决 1：带 visual_subject/
actions/start_sec/end_sec 的富 FactPack 分镜），跑 VideoDiagnosisEngine，并装配
《电商短视频诊断：前端消费层输出契约》响应对象。

产出：
  outputs/runben_diagnosis/runben_video_diagnosis.json   —— 引擎原始输出（保持稳定，不破坏）
  outputs/runben_diagnosis/runben_contract_response.json —— 前端 5 Tab 契约响应

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
from commerce_video_diagnosis.understanding.assembly.response_assembler import (  # noqa: E402
    assemble_frontend_response,
)

OUTPUT_DIR = ROOT / "outputs" / "runben_diagnosis"
PRODUCT_DIAGNOSIS_PATH = OUTPUT_DIR / "runben_full_diagnosis.json"


def _load_product_diagnosis() -> dict:
    """加载商品诊断输出；若不存在则先跑 run_runben_full_diagnosis.py 生成。"""
    if not PRODUCT_DIAGNOSIS_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "run_runben_full_diagnosis.py")], check=True)
    return json.loads(PRODUCT_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


def _build_rich_storyboard_segments() -> list[dict]:
    """裁决 1：富 FactPack 分镜（与 fixtures/query_1780911885_vlm.json 同构）。

    每段含 segment_id / start_sec / end_sec / visual_facts{visual_subject, actions, ...} /
    ocr / asr，真实描述润本驱蚊液样本画面，禁止编造占位。
    """
    return [
        {
            "segment_id": "seg_hook_1",
            "role": "hook",
            "evidence_role": "hook",
            "start_sec": 0.0,
            "end_sec": 4.0,
            "asr": "实测700只蚊子挑战，看看这瓶驱蚊液到底防不防蚊，被叮一身红肿包真的很困扰",
            "ocr": "700只蚊子挑战",
            "visual_facts": {
                "shot_size": "medium_close_up",
                "camera_movement": "static",
                "visual_subject": "博主举起手臂正对镜头，画面叠加「700只蚊子挑战」字幕，桌上放着装满蚊子的透明蚊箱，手臂上能看到被叮后的红肿包",
                "lighting_tone": "flat_neutral",
                "key_objects": ["手臂", "红肿包", "透明蚊箱", "700只蚊子挑战字幕"],
                "actions": [
                    {"action_name": "举起手臂展示叮咬痛点", "physical_intensity": "medium"},
                    {"action_name": "指向蚊箱引出挑战", "physical_intensity": "low"},
                ],
            },
            "rhythm_facts": {
                "transition_type": "hard_cut",
                "pace_marker": "fast",
                "is_rhythm_change_point": True,
                "rhythm_change_reason": "开场痛点 Hook 切入，强视觉冲击建立挑战悬念",
            },
        },
        {
            "segment_id": "seg_effect_1",
            "role": "effect",
            "evidence_role": "proof",
            "start_sec": 4.0,
            "end_sec": 9.0,
            "asr": "手臂放进蚊箱15分钟拿出来，一个包都没有，真的无包，驱蚊效果肉眼可见",
            "ocr": "15分钟 无包",
            "visual_facts": {
                "shot_size": "close_up",
                "camera_movement": "slow_push_in",
                "visual_subject": "博主把喷过驱蚊液的手臂伸进装满蚊子的透明蚊箱中静置，计时器显示15分钟，取出后特写手臂皮肤完好无叮咬包",
                "lighting_tone": "flat_neutral",
                "key_objects": ["手臂", "透明蚊箱", "15分钟计时器", "无叮咬的皮肤特写"],
                "actions": [
                    {"action_name": "将手臂伸入蚊箱实测", "physical_intensity": "high"},
                    {"action_name": "取出手臂展示无包效果", "physical_intensity": "medium"},
                ],
            },
            "rhythm_facts": {
                "transition_type": "match_cut",
                "pace_marker": "normal",
                "is_rhythm_change_point": True,
                "rhythm_change_reason": "由 Hook 痛点切换到效果实测，进入功效证明段",
            },
        },
        {
            "segment_id": "seg_effect_2",
            "role": "effect",
            "evidence_role": "safety",
            "start_sec": 9.0,
            "end_sec": 13.0,
            "asr": "成分温和，小朋友也能用，喷上去无味道不刺鼻，敏感肌也安全",
            "ocr": "小朋友也能用 · 无味道",
            "visual_facts": {
                "shot_size": "medium",
                "camera_movement": "static",
                "visual_subject": "镜头切到给小朋友手臂上喷驱蚊液的画面，家长凑近闻喷雾后的皮肤表示无味道，画面安静温和",
                "lighting_tone": "warm_soft",
                "key_objects": ["小朋友手臂", "驱蚊喷雾", "家长", "无味道字幕"],
                "actions": [
                    {"action_name": "给小朋友喷涂驱蚊液", "physical_intensity": "low"},
                    {"action_name": "凑近闻喷后皮肤验证无味", "physical_intensity": "low"},
                ],
            },
            "rhythm_facts": {
                "transition_type": "hard_cut",
                "pace_marker": "slow",
                "is_rhythm_change_point": True,
                "rhythm_change_reason": "场景由实测切换到安全背书，语气转柔和",
            },
        },
        {
            "segment_id": "seg_effect_3",
            "role": "effect",
            "evidence_role": "proof",
            "start_sec": 13.0,
            "end_sec": 17.0,
            "asr": "认准官方旗舰店正品，派卡瑞丁成分通过检测报告认证，品质有保障",
            "ocr": "官方正品 · 检测报告 认证",
            "visual_facts": {
                "shot_size": "close_up",
                "camera_movement": "pan",
                "visual_subject": "特写官方旗舰店正品外包装与第三方检测报告、派卡瑞丁成分说明文档，镜头平移逐一展示资质",
                "lighting_tone": "flat_neutral",
                "key_objects": ["官方正品包装", "第三方检测报告", "派卡瑞丁成分说明", "认证标识"],
                "actions": [
                    {"action_name": "展示官方正品包装", "physical_intensity": "low"},
                    {"action_name": "翻开检测报告与成分文档", "physical_intensity": "low"},
                ],
            },
            "rhythm_facts": {
                "transition_type": "hard_cut",
                "pace_marker": "normal",
                "is_rhythm_change_point": False,
                "rhythm_change_reason": None,
            },
        },
        {
            "segment_id": "seg_cta_1",
            "role": "cta",
            "evidence_role": "cta",
            "start_sec": 17.0,
            "end_sec": 22.0,
            "asr": "大瓶家用更划算够全家一夏天，小瓶便携出门带，学生党宿舍、户外露营也适合，一套搞定",
            "ocr": "大瓶家用划算+小瓶便携",
            "visual_facts": {
                "shot_size": "medium",
                "camera_movement": "static",
                "visual_subject": "桌面并排摆放大瓶家用装与小瓶便携装，博主拿起两瓶向镜头介绍家用加便携的组合更划算",
                "lighting_tone": "warm_soft",
                "key_objects": ["大瓶家用装", "小瓶便携装", "组合套装"],
                "actions": [
                    {"action_name": "并排展示大瓶小瓶组合", "physical_intensity": "low"},
                    {"action_name": "拿起产品介绍场景与人群", "physical_intensity": "low"},
                ],
            },
            "rhythm_facts": {
                "transition_type": "hard_cut",
                "pace_marker": "normal",
                "is_rhythm_change_point": True,
                "rhythm_change_reason": "进入 CTA 收口段，落到人群/场景总结",
            },
        },
    ]


def build_payload() -> dict:
    diagnosis = _load_product_diagnosis()
    source_product_id = diagnosis.get("product_id") or "runben_repellent_24p9"

    # product_HEC：以商品主 HEC（product_hecs[0] = H1/E1/C4）为应然候选集合。
    product_hec0 = (diagnosis.get("product_hecs") or [{}])[0]
    product_HEC = {
        "candidates": [
            {
                "H": product_hec0.get("hook_tag"),
                "E": product_hec0.get("effect_tag"),
                "C": product_hec0.get("cta_tag"),
            }
        ]
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
        # video_meta 在 runben 样本无平台/链接/时长来源 → null（契约允许，key 必须存在）
        "source_platform": None,
        "source_url": None,
        "duration_sec": None,
        "primary_hec": {
            # B1：HEC hook H1 → H5（物理安全域合法 Hook），effect/cta 不变。
            "hook_tag": "H5",
            "effect_tag": "E1",
            "cta_tag": "C4",
            "signature": "H5安全场景切入→E1效果测评→C4人群场景总结",
        },
        # 视频实然侧 JTBD（视频理解输出，独立于商品应然，证据来自视频画面/口播）
        "video_jtbd": {
            "primary_task": "驱蚊防护（物理安全与风险规避）",
            "reasoning": "视频以 700 只蚊子实测挑战为 Hook，围绕 15 分钟无包效果、小朋友也能用的安全背书、官方正品检测报告组织内容，整体演示的是为家庭/儿童规避蚊虫叮咬风险的驱蚊防护任务。",
            "evidence": [
                {"source": "video_factpack", "field": "asr", "value": "实测700只蚊子挑战，被叮一身红肿包真的很困扰", "segment_id": "seg_hook_1", "confidence": None},
                {"source": "video_factpack", "field": "asr", "value": "15分钟一个包都没有，真的无包，驱蚊效果肉眼可见", "segment_id": "seg_effect_1", "confidence": None},
                {"source": "video_factpack", "field": "asr", "value": "成分温和，小朋友也能用，敏感肌也安全", "segment_id": "seg_effect_2", "confidence": None},
            ],
        },
        "slider_signature": {
            "visual": {"score": 0.82, "evidence": "700只蚊子挑战强视觉冲击/猎奇画面"},
            "audio": {"score": 0.6, "evidence": "口播稳，讲解清晰"},
            "proof": {"score": 0.75, "evidence": "15分钟无包实测过程"},
            # B2：CTA 0.6 → 0.3（收口偏弱，落到参照人群 cta 区间下界以下）。
            "cta": {"score": 0.3, "evidence": "大瓶家用+小瓶便携场景总结收口偏弱"},
        },
        "storyboard_segments": _build_rich_storyboard_segments(),
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
    full_diagnosis = _load_product_diagnosis()
    payload = build_payload()
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(payload)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 引擎原始输出（保持稳定交付，不破坏）
    out_path = OUTPUT_DIR / "runben_video_diagnosis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_result = result["video_persuasion_diagnosis_result"]

    # 2) 前端消费层契约响应（裁决 2：runner 注入 diagnosis_meta）
    diagnosis_meta_input = {
        "request_id": "req-runben-20260628-001",
        "video_id": payload["video_understanding"]["video_id"],
        "source_product_id": payload["product_diagnosis"]["source_product_id"],
        "workflow_version": "jtbd-diagnosis-mvp-v1",
        "model_version": None,
        "model_provider": None,
    }
    source_files = [
        "outputs/runben_diagnosis/runben_full_diagnosis.json",
        "outputs/runben_diagnosis/runben_video_diagnosis.json",
        "outputs/runben_diagnosis/runben_diagnosis_report.md",
    ]
    contract_response = assemble_frontend_response(
        product_diagnosis=full_diagnosis,
        video_payload=payload,
        raw_diagnosis_result=raw_result,
        diagnosis_meta_input=diagnosis_meta_input,
        source_files=source_files,
    )
    contract_path = OUTPUT_DIR / "runben_contract_response.json"
    contract_path.write_text(json.dumps(contract_response, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "raw_output_path": str(out_path.relative_to(ROOT)),
            "contract_output_path": str(contract_path.relative_to(ROOT)),
            "top_level_keys": list(contract_response.keys()),
            "status": contract_response["status"],
            "diagnosis_keys": list(contract_response["diagnosis"].keys()),
            "visual_segment_count": len(contract_response["video_understanding"]["visual_stream"]["visual_segments"]),
            "requirement_coverage": {
                "status": contract_response["diagnosis"]["requirement_coverage"]["status"],
                "completed_count": contract_response["diagnosis"]["requirement_coverage"]["completed_count"],
                "total_count": contract_response["diagnosis"]["requirement_coverage"]["total_count"],
            },
            "overview": contract_response["diagnosis"]["overview"],
            "input_validation_warnings": raw_result["input_validation"]["warnings"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
