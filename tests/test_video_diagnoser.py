"""Block 2 视频说服诊断 video_diagnoser 测试。

覆盖：
- Step0 Crash Early（source_product_id 不一致 / 缺 slider 轴 / 缺 product_HEC）。
- primary_hec 回填 video_HEC（含 warning）。
- 润本视频样本端到端：七段齐全 / audience high / hec matched(H1/E1/C4) / 结构正确 / 优先级正确。
- video_target_audience 独立于 slider（删除/修改 slider 不影响 Step1）。
- D4 axis_judgment 三轴均存在。
- D1 / D2 / D3 新枚举（禁止废弃枚举）。
- 字段命名：目标人群字段无 `segments`。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from commerce_video_diagnosis.understanding.engines.video_diagnoser import (  # noqa: E402
    VideoDiagnosisEngine,
    VideoDiagnosisInputError,
    VideoDiagnosisEnumError,
    load_audience_slider_preference_dictionary,
)

SOURCE_PID = "runben_repellent_24p9"

PRODUCT_TARGET_AUDIENCE = {
    "primary_audiences": [
        {"audience_group": "年长中高消费力女性", "fit_level": "primary", "reason": "风险责任人 + 大牌官方高水位"}
    ],
    "secondary_audiences": [
        {"audience_group": "年长低消费力女性", "fit_level": "secondary", "reason": "套装价值覆盖低消费力"}
    ],
    "weak_fit_audiences": [],
}

PROFILE = {
    "profile_version": "v3.1",
    "content_goal": "purchase",
    "persuasion_requirements": [
        {"requirement_id": "expose_current_pain", "requirement_name": "暴露当前痛点", "required": True, "success_criteria": "感知当前蚊虫叮咬痛点"},
        {"requirement_id": "prove_user_fit", "requirement_name": "证明人群适配", "required": True, "success_criteria": "小朋友也能用"},
        {"requirement_id": "provide_visible_result", "requirement_name": "提供可见效果", "required": True, "success_criteria": "15分钟无包"},
        {"requirement_id": "reduce_trial_risk", "requirement_name": "降低试错风险", "required": True, "success_criteria": "温和无味道"},
        {"requirement_id": "prove_source_credibility", "requirement_name": "证明来源可信", "required": True, "success_criteria": "检测报告成分"},
        {"requirement_id": "clarify_purchase_threshold", "requirement_name": "明确购买门槛", "required": False, "success_criteria": "大瓶小瓶规格"},
    ],
    "not_applicable_requirements": [],
}

PRODUCT_HEC = {"candidates": [{"H": "H1", "E": "E1", "C": "C4"}]}


def _slider(visual=0.5, audio=0.6, proof=0.8, cta=0.6):
    return {
        "visual": {"score": visual, "evidence": "v"},
        "audio": {"score": audio, "evidence": "a"},
        "proof": {"score": proof, "evidence": "p"},
        "cta": {"score": cta, "evidence": "c"},
    }


def _storyboard():
    return [
        {"segment_id": "h", "role": "hook", "asr": "实测700只蚊子挑战，看看防不防蚊", "ocr": "700只蚊子挑战"},
        {"segment_id": "e1", "role": "effect", "asr": "放进去15分钟，一个包都没有，无包", "ocr": "15分钟 无包"},
        {"segment_id": "e2", "role": "effect", "asr": "成分温和，小朋友也能用，无味道不刺鼻", "ocr": "小朋友也能用 无味道"},
        {"segment_id": "c", "role": "cta", "asr": "大瓶家用，小瓶便携出门带", "ocr": "大瓶家用+小瓶便携"},
    ]


def build_payload(*, use_primary_hec=False, slider=None, source_video_pid=SOURCE_PID):
    video = {
        "video_id": "v1",
        "source_product_id": source_video_pid,
        "slider_signature": slider if slider is not None else _slider(),
        "storyboard_segments": _storyboard(),
        "semantic_bundles": [{"bundle_id": "b1", "bundle_role": "effect", "text": "700只蚊子挑战 15分钟无包 驱蚊"}],
        "evidence_spans": [
            {"span_id": "sp1", "text": "蚊子太多被叮一身包"},
            {"span_id": "sp2", "text": "15分钟无包实测效果"},
            {"span_id": "sp3", "text": "小朋友也能用温和无味道"},
        ],
    }
    if use_primary_hec:
        video["primary_hec"] = {"hook_tag": "H1", "effect_tag": "E1", "cta_tag": "C4", "signature": "sig"}
    else:
        video["video_HEC"] = {"hook_tag": "H1", "effect_tag": "E1", "cta_tag": "C4"}
    return {
        "product_diagnosis": {
            "source_product_id": SOURCE_PID,
            "product_fact_vector": {"leaf_category": "宝宝防蚊水"},
            "product_target_audience": copy.deepcopy(PRODUCT_TARGET_AUDIENCE),
            "persuasion_requirement_profile": copy.deepcopy(PROFILE),
            "product_HEC": copy.deepcopy(PRODUCT_HEC),
        },
        "video_understanding": video,
    }


# ---------------------------------------------------------------------------
# Step0 Crash Early
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_step0_source_product_id_mismatch():
    engine = VideoDiagnosisEngine()
    payload = build_payload(source_video_pid="other_pid")
    with pytest.raises(VideoDiagnosisInputError, match="source_product_id"):
        engine.diagnose(payload)


@pytest.mark.unit
def test_step0_missing_slider_axis():
    engine = VideoDiagnosisEngine()
    payload = build_payload()
    del payload["video_understanding"]["slider_signature"]["proof"]
    with pytest.raises(VideoDiagnosisInputError, match="slider_signature"):
        engine.diagnose(payload)


@pytest.mark.unit
def test_step0_missing_product_hec():
    engine = VideoDiagnosisEngine()
    payload = build_payload()
    payload["product_diagnosis"]["product_HEC"] = {}
    with pytest.raises(VideoDiagnosisInputError, match="product_HEC"):
        engine.diagnose(payload)


# ---------------------------------------------------------------------------
# primary_hec 回填 video_HEC（Block 1.2），走通且有 warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_primary_hec_fallback_with_warning():
    engine = VideoDiagnosisEngine()
    payload = build_payload(use_primary_hec=True)
    result = engine.diagnose(payload)["video_persuasion_diagnosis_result"]
    warnings = result["input_validation"]["warnings"]
    assert any("primary_hec" in w for w in warnings)
    # 映射后 hec 仍可匹配
    assert result["hec_match_diagnosis"]["match_status"] == "good"


# ---------------------------------------------------------------------------
# 润本视频样本端到端
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_runben_video_end_to_end():
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(build_payload())["video_persuasion_diagnosis_result"]

    # 七段齐全
    assert set(result.keys()) == {
        "input_validation",
        "video_target_audience",
        "audience_match_diagnosis",
        "profile_match_diagnosis",
        "hec_match_diagnosis",
        "slider_match_diagnosis",
        "diagnosis_summary",
    }
    assert result["input_validation"]["status"] == "passed"

    # video 主目标 = 年长中高消费力女性
    vta = result["video_target_audience"]
    assert [a["audience_group"] for a in vta["primary_audiences"]] == ["年长中高消费力女性"]

    # audience high_match（或 partial），且覆盖商品主目标
    assert result["audience_match_diagnosis"]["match_status"] in ("high_match", "partial_match")
    assert "年长中高消费力女性" in result["audience_match_diagnosis"]["matched_audiences"]

    # hec good(H1/E1/C4)
    hec = result["hec_match_diagnosis"]
    assert hec["match_status"] == "good"
    assert hec["actual_video_hec"] == {"hook_tag": "H1", "effect_tag": "E1", "cta_tag": "C4"}
    assert hec["full_combination_hit"] is True

    # profile 结构正确（A1/D1：profile 汇总字段更名为 match_status）
    pmd = result["profile_match_diagnosis"]
    assert "overall_status" not in pmd, "profile 不得保留旧字段名 overall_status"
    assert pmd["match_status"] in ("completed", "partial", "weak", "missing", "not_applicable")
    assert pmd["requirement_results"]
    for r in pmd["requirement_results"]:
        assert r["completion_status"] in ("completed", "partial", "weak", "missing", "not_applicable")
        assert "partial_complete" not in r["completion_status"]

    # slider 结构正确
    smd = result["slider_match_diagnosis"]
    assert set(a["axis"] for a in smd["axis_results"]) == {"visual", "audio", "proof", "cta"}
    assert smd["target_audience_reference"]

    # diagnosis_summary 优先级正确（D3 枚举）
    summary = result["diagnosis_summary"]
    assert summary["overall_status"] in ("good", "needs_minor_repair", "needs_major_repair", "mismatch")


# ---------------------------------------------------------------------------
# D4：axis_judgment 三轴均存在
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_axis_judgment_three_axes_present():
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(build_payload())["video_persuasion_diagnosis_result"]
    axis_judgment = result["video_target_audience"]["axis_judgment"]
    for axis in ("age_axis", "gender_axis", "consumption_power_axis"):
        assert axis in axis_judgment
        assert "value" in axis_judgment[axis]
        assert "evidence" in axis_judgment[axis]
        assert "reason" in axis_judgment[axis]
    # 保留原 reasoning_chain 与 evidence_summary
    assert set(result["video_target_audience"]["reasoning_chain"].keys()) == {
        "hook_scene_to_role",
        "persona_to_age_gender",
        "cta_benefit_to_consumption_power",
    }
    assert "evidence_summary" in result["video_target_audience"]


# ---------------------------------------------------------------------------
# video_target_audience 独立于 slider
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_video_target_audience_independent_of_slider():
    engine = VideoDiagnosisEngine()
    weak = engine.diagnose(build_payload(slider=_slider(0.1, 0.1, 0.1, 0.1)))[
        "video_persuasion_diagnosis_result"
    ]["video_target_audience"]
    strong = engine.diagnose(build_payload(slider=_slider(0.9, 0.9, 0.9, 0.9)))[
        "video_persuasion_diagnosis_result"
    ]["video_target_audience"]
    assert weak == strong, "Step1 结果不得随 slider 改变（必须独立于 slider_signature）"


# ---------------------------------------------------------------------------
# 字段命名：目标人群字段无 segments
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_segments_in_audience_fields():
    engine = VideoDiagnosisEngine()
    result = engine.diagnose(build_payload())["video_persuasion_diagnosis_result"]
    vta = result["video_target_audience"]

    def _assert_no_segments(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "segments" not in k, f"目标人群字段非法命名: {k}"
                _assert_no_segments(v)
        elif isinstance(obj, list):
            for v in obj:
                _assert_no_segments(v)

    _assert_no_segments(vta)


# ---------------------------------------------------------------------------
# D1：profile completion_status 新枚举 completed/partial/weak/missing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_d1_profile_completion_status_enum():
    engine = VideoDiagnosisEngine()
    corpus = {
        # 蚊/叮/包/困扰/挑战 → expose_current_pain completed；小朋友/也能用 → prove_user_fit partial；
        # 大牌 → establish_basic_trust weak；credibility 无命中 → missing
        "full_text": "蚊子太多被叮一身包很困扰，700只蚊子挑战，小朋友也能用，大牌出品",
        "span_index": [],
    }
    profile = {
        "persuasion_requirements": [
            {"requirement_id": "expose_current_pain", "requirement_name": "暴露当前痛点", "required": True},
            {"requirement_id": "prove_user_fit", "requirement_name": "证明人群适配", "required": True},
            {"requirement_id": "establish_basic_trust", "requirement_name": "建立基础信任", "required": True},
            {"requirement_id": "prove_source_credibility", "requirement_name": "证明来源可信", "required": True},
        ],
        "not_applicable_requirements": [],
    }
    result = engine._step3_profile_match(profile, corpus)
    status_by_id = {r["requirement_id"]: r["completion_status"] for r in result["requirement_results"]}
    assert status_by_id["expose_current_pain"] == "completed"
    assert status_by_id["prove_user_fit"] == "partial"
    assert status_by_id["establish_basic_trust"] == "weak"
    assert status_by_id["prove_source_credibility"] == "missing"
    assert result["match_status"] == "missing"  # 有必讲 missing
    # 禁止废弃枚举
    assert all(s != "partial_complete" for s in status_by_id.values())


@pytest.mark.unit
def test_d1_profile_overall_partial_and_not_applicable():
    engine = VideoDiagnosisEngine()
    corpus = {"full_text": "小朋友也能用，温和无味道", "span_index": []}
    profile = {
        "persuasion_requirements": [
            {"requirement_id": "prove_user_fit", "requirement_name": "证明人群适配", "required": True},
            {"requirement_id": "reduce_trial_risk", "requirement_name": "降低试错风险", "required": True},
            {"requirement_id": "clarify_purchase_threshold", "requirement_name": "明确购买门槛", "required": False},
        ],
        "not_applicable_requirements": [{"requirement_id": "clarify_purchase_threshold"}],
    }
    result = engine._step3_profile_match(profile, corpus)
    status_by_id = {r["requirement_id"]: r["completion_status"] for r in result["requirement_results"]}
    assert status_by_id["clarify_purchase_threshold"] == "not_applicable"
    # 无必讲 missing、有 partial/weak → overall partial
    assert result["match_status"] == "partial"


# ---------------------------------------------------------------------------
# D2：slider match_status 新枚举（fit/too_strong/too_weak/wrong_direction/mixed_deviation/mismatch）
# ---------------------------------------------------------------------------
def _slider_match(engine, **scores):
    vta = {"primary_audiences": [{"audience_group": "年长中高消费力女性"}]}
    pta = {"primary_audiences": [{"audience_group": "年长中高消费力女性"}]}
    return engine._step5_slider_match(vta, pta, _slider(**scores))


@pytest.mark.unit
def test_d2_slider_enums():
    engine = VideoDiagnosisEngine()
    # 参照 年长中高消费力女性：visual0.3-0.7 audio0.4-0.8 proof0.6-1.0 cta0.4-0.8
    assert _slider_match(engine, visual=0.5, audio=0.6, proof=0.8, cta=0.6)["match_status"] == "fit"
    # 仅偏强（visual 0.75 > 0.7）
    assert _slider_match(engine, visual=0.75, audio=0.6, proof=0.8, cta=0.6)["match_status"] == "too_strong"
    # 仅偏弱（proof 0.45 < 0.6）
    assert _slider_match(engine, visual=0.5, audio=0.6, proof=0.45, cta=0.6)["match_status"] == "too_weak"
    # 偏强+偏弱混合 → mixed_deviation
    assert _slider_match(engine, visual=0.75, audio=0.6, proof=0.45, cta=0.6)["match_status"] == "mixed_deviation"
    # 单轴严重越界（proof 0.1，gap 0.5）→ wrong_direction
    assert _slider_match(engine, visual=0.5, audio=0.6, proof=0.1, cta=0.6)["match_status"] == "wrong_direction"
    # 多轴严重越界 → mismatch（proof 0.1 + cta 0.0 均 wrong_direction）
    assert _slider_match(engine, visual=0.5, audio=0.6, proof=0.1, cta=0.0)["match_status"] == "mismatch"
    # 轴级 fit_status 合法枚举
    res = _slider_match(engine, visual=0.75, audio=0.6, proof=0.45, cta=0.6)
    for r in res["axis_results"]:
        assert r["fit_status"] in ("fit", "too_strong", "too_weak", "wrong_direction")
    # 禁止废弃枚举
    assert res["match_status"] not in ("partial_match", "matched", "slightly_strong", "slightly_weak")


# ---------------------------------------------------------------------------
# D3：diagnosis_summary overall_status 新枚举映射
# ---------------------------------------------------------------------------
def _summary(engine, *, audience, profile, hec, slider):
    return engine._step6_summary(audience, profile, hec, slider)


@pytest.mark.unit
def test_d3_overall_status_mapping():
    engine = VideoDiagnosisEngine()
    clean_audience = {"match_status": "high_match", "uncovered_product_audiences": []}
    clean_profile = {"match_status": "completed", "missing_required_requirements": [], "weak_requirements": []}
    clean_hec = {"match_status": "good", "hec_gap_summary": ""}
    clean_slider = {"match_status": "fit", "slider_gap_summary": ""}

    # 无问题 → good
    s = _summary(engine, audience=clean_audience, profile=clean_profile, hec=clean_hec, slider=clean_slider)
    assert s["overall_status"] == "good"

    # 仅 P2（slider too_strong）→ needs_minor_repair
    s = _summary(engine, audience=clean_audience, profile=clean_profile, hec=clean_hec,
                 slider={"match_status": "too_strong", "slider_gap_summary": "visual"})
    assert s["overall_status"] == "needs_minor_repair"

    # 有 P1（hec mismatch）无 P0 → needs_major_repair
    s = _summary(engine, audience=clean_audience, profile=clean_profile,
                 hec={"match_status": "mismatch", "hec_gap_summary": "effect断裂"}, slider=clean_slider)
    assert s["overall_status"] == "needs_major_repair"

    # 有 P0（profile missing 必讲）→ mismatch
    s = _summary(engine, audience=clean_audience,
                 profile={"match_status": "missing", "missing_required_requirements": ["x"], "weak_requirements": []},
                 hec=clean_hec, slider=clean_slider)
    assert s["overall_status"] == "mismatch"
    # 禁止废弃枚举
    assert s["overall_status"] not in ("pass", "needs_repair")


# ---------------------------------------------------------------------------
# loader Crash Early
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_slider_dictionary_loader_missing_file():
    with pytest.raises(VideoDiagnosisInputError, match="文件缺失"):
        load_audience_slider_preference_dictionary("/nonexistent/path/slider.json")


@pytest.mark.unit
def test_slider_dictionary_loader_covers_all_eight_groups():
    data = load_audience_slider_preference_dictionary()
    from commerce_video_diagnosis.understanding.engines.audience_taxonomy import EIGHT_AUDIENCE_GROUPS

    for group in EIGHT_AUDIENCE_GROUPS:
        assert group in data["preferences"]



# ---------------------------------------------------------------------------
# 枚举边界 Crash Early 守卫（防止未来枚举漂移）
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_enum_guard_rejects_illegal_overall_status(monkeypatch):
    """通过 monkeypatch 让 _step3_profile_match 输出非法 match_status，
    验证 diagnose() 的最终 _validate_enums 抛出 VideoDiagnosisEnumError。"""
    engine = VideoDiagnosisEngine()
    original = engine._step3_profile_match

    def _bad(profile, corpus):
        result = original(profile, corpus)
        result["match_status"] = "incomplete"  # 非法
        return result

    monkeypatch.setattr(engine, "_step3_profile_match", _bad)
    with pytest.raises(VideoDiagnosisEnumError, match="profile_match_diagnosis.match_status"):
        engine.diagnose(build_payload())


@pytest.mark.unit
def test_enum_guard_rejects_illegal_completion_status(monkeypatch):
    engine = VideoDiagnosisEngine()
    original = engine._step3_profile_match

    def _bad(profile, corpus):
        result = original(profile, corpus)
        if result["requirement_results"]:
            result["requirement_results"][0]["completion_status"] = "partial_complete"
        return result

    monkeypatch.setattr(engine, "_step3_profile_match", _bad)
    with pytest.raises(VideoDiagnosisEnumError, match="completion_status"):
        engine.diagnose(build_payload())


@pytest.mark.unit
def test_enum_guard_passes_on_legal_runben():
    engine = VideoDiagnosisEngine()
    # 不应该抛 VideoDiagnosisEnumError
    result = engine.diagnose(build_payload())["video_persuasion_diagnosis_result"]
    assert result["profile_match_diagnosis"]["match_status"] in {
        "completed", "partial", "weak", "missing", "not_applicable"
    }
