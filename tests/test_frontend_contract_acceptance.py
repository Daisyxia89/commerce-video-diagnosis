# -*- coding: utf-8 -*-
"""
独立 QA 验收测试 —— 《电商短视频诊断：前端消费层输出契约》

铁律 / 独立性声明:
- 本文件的「期望 schema」**仅来自契约文档**（电商短视频诊断：前端消费层输出契约.lark_1.md）。
- 本文件**不引用、不导入、不参照**任何实现文件（response_assembler.py / schema_assertions.py）
  及任何研发自测用例。
- 仅加载研发产出结果 JSON（runben_contract_response.json）做断言；断言标准独立来自契约。
- 契约有歧义处一律按字面从严判定。

每个 test_* 函数 = 一条验收用例，pytest -v 会逐条给出 PASS/FAIL。
注释中的 [契约依据] 标注对应契约小节/BLOCK。
"""
import json
import os
import pytest

# ---------------------------------------------------------------------------
# 加载待验收产出
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RESP_PATH = os.path.join(
    _REPO, "outputs", "runben_diagnosis", "runben_contract_response.json"
)


@pytest.fixture(scope="module")
def resp():
    with open(_RESP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 契约枚举集合（均直接抄录自契约 Schema）
# ---------------------------------------------------------------------------
# [契约依据] §3 BLOCK 9
STATUS_ENUM = {
    "diagnosis_completed",
    "needs_review",
    "out_of_scope_for_mvp",
    "assembly_blocked",
    "schema_error",
    "provider_not_configured",
}
TOP_KEYS = {
    "status",
    "diagnosis_meta",
    "product_understanding",
    "video_understanding",
    "diagnosis",
    "artifacts",
}
DIAGNOSIS_KEYS = {
    "overview",
    "profile_match",
    "hec_match",
    "slider_match",
    "requirement_coverage",
    "top_issues",
    "suggestions",
}
ARTIFACTS_KEYS = {"request_payload", "raw_response", "normalized_response", "source_files"}

# [契约依据] §14 BLOCK 123 统一 evidence schema
EVIDENCE_KEYS = {"source", "field", "value", "segment_id", "confidence"}
EVIDENCE_SOURCE_ENUM = {
    "product_factpack",
    "video_factpack",
    "product_understanding",
    "video_understanding",
    "raw_output",
}

# [契约依据] §5 BLOCK 23
PRICE_BAND_ENUM = {"high", "medium", "low", "unknown"}
PRIORITY_ENUM = {"required", "optional"}
HML_UNKNOWN = {"high", "medium", "low", "unknown"}
CHANNEL_RISK_ENUM = {"risk", "no_risk", "unknown"}
ENDORSEMENT_ENUM = {"has_endorsement", "no_endorsement", "unknown"}
BRAND_TIER_ENUM = {"brand", "white_label", "unknown"}

# [契约依据] §6 BLOCK 33 evidence_role
EVIDENCE_ROLE_ENUM = {"hook", "proof", "safety", "cta", "transition", "other"}

# [契约依据] §7 BLOCK 63 overview 各状态枚举
OVERALL_STATUS_ENUM = {
    "pass",
    "needs_minor_repair",
    "needs_repair",
    "mismatch",
    "blocked",
    "needs_review",
}
AUDIENCE_MATCH_STATUS_ENUM = {
    "high_match",
    "partial_match",
    "low_match",
    "too_broad",
    "data_missing",
}
PROFILE_MATCH_STATUS_OV_ENUM = {
    "completed",
    "partial",
    "incomplete",
    "insufficient_evidence",
    "data_missing",
}
HEC_MATCH_STATUS_ENUM = {
    "matched",
    "acceptable_deviation",
    "weak_match",
    "mismatch",
    "data_missing",
}
SLIDER_MATCH_STATUS_OV_ENUM = {
    "matched",
    "mixed_deviation",
    "slightly_strong",
    "slightly_weak",
    "mismatch",
    "data_missing",
}
REQ_COVERAGE_STATUS_OV_ENUM = {"completed", "partial", "failed", "data_missing"}

# [契约依据] §8 BLOCK 78
PROFILE_MATCH_STATUS_ENUM = {
    "completed",
    "needs_review",
    "insufficient_evidence",
    "data_missing",
}
GAP_LEVEL_ENUM = {"high", "medium", "low"}
MATCH_RESULT_ENUM = {"high_match", "partial", "mismatch"}

# [契约依据] §9 BLOCK 88
REQ_COVERAGE_STATUS_ENUM = {"completed", "partial", "failed", "data_missing"}
COMPLETION_STATUS_ENUM = {"completed", "weak", "missing", "not_applicable"}

# [契约依据] §10 BLOCK 97
HEC_DIMENSION_ENUM = {"hook", "effect", "cta", "chain"}
HEC_DIM_STATUS_ENUM = {"matched", "acceptable_deviation", "weak_match", "mismatch"}

# [契约依据] §11 BLOCK 106
SLIDER_STATUS_ENUM = {
    "matched",
    "mixed_deviation",
    "slightly_strong",
    "slightly_weak",
    "mismatch",
    "insufficient_evidence",
    "data_missing",
}
AXIS_ENUM = {"visual", "audio", "proof", "cta"}
FIT_STATUS_ENUM = {"fit", "too_strong", "too_weak", "wrong_direction"}

# [契约依据] §12 BLOCK 109
SEVERITY_ENUM = {"low", "medium", "high"}
SUGGESTION_PRIORITY_ENUM = {"P0", "P1", "P2"}
MODULE_ENUM = {
    "audience_match",
    "profile_match",
    "requirement_coverage",
    "hec_match",
    "slider_match",
}


# ---------------------------------------------------------------------------
# 公共断言工具
# ---------------------------------------------------------------------------
def _nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def _assert_evidence_item(ev, ctx):
    """统一 evidence schema 校验 [契约依据] §14 BLOCK 123"""
    assert isinstance(ev, dict), f"{ctx}: evidence 项必须是对象"
    assert EVIDENCE_KEYS.issubset(ev.keys()), (
        f"{ctx}: evidence 缺少字段 {EVIDENCE_KEYS - set(ev.keys())}"
    )
    assert ev["source"] in EVIDENCE_SOURCE_ENUM, (
        f"{ctx}: source={ev['source']!r} 不在契约枚举 {EVIDENCE_SOURCE_ENUM}"
    )


# ===========================================================================
# 1. 顶层结构 / status  [契约依据] §3 BLOCK 9
# ===========================================================================
def test_top_level_six_keys_present(resp):
    """顶层 6 键齐全 [契约依据] §3 BLOCK 9"""
    assert TOP_KEYS.issubset(resp.keys()), (
        f"顶层缺少键: {TOP_KEYS - set(resp.keys())}"
    )


def test_status_in_contract_enum(resp):
    """status 属于契约枚举集合 [契约依据] §3 BLOCK 9"""
    assert resp["status"] in STATUS_ENUM, (
        f"status={resp['status']!r} 不在契约枚举 {STATUS_ENUM}"
    )


def test_diagnosis_seven_subkeys_present(resp):
    """diagnosis 七个子模块键齐全 [契约依据] §3 BLOCK 9"""
    assert DIAGNOSIS_KEYS.issubset(resp["diagnosis"].keys()), (
        f"diagnosis 缺少: {DIAGNOSIS_KEYS - set(resp['diagnosis'].keys())}"
    )


# ===========================================================================
# 2. diagnosis_meta  [契约依据] §4 BLOCK 14/16/17
# ===========================================================================
def test_meta_p1_required_nonempty(resp):
    """diagnosis_id/request_id/created_at/source_product_id/video_id P1 必填非空 [契约依据] §4.3 BLOCK 16"""
    meta = resp["diagnosis_meta"]
    for k in ["diagnosis_id", "request_id", "created_at", "source_product_id", "video_id"]:
        assert k in meta, f"diagnosis_meta 缺少必填字段 {k}"
        assert _nonempty_str(meta[k]), f"diagnosis_meta.{k} 不得为空: {meta.get(k)!r}"


def test_meta_request_video_sourceproduct_nonempty(resp):
    """request_id / video_id / source_product_id 必填非空（任务强调项）[契约依据] §4.3 BLOCK 16"""
    meta = resp["diagnosis_meta"]
    for k in ["request_id", "video_id", "source_product_id"]:
        assert _nonempty_str(meta.get(k)), f"diagnosis_meta.{k} 必填非空, 实际={meta.get(k)!r}"


def test_meta_version_keys_exist(resp):
    """workflow_version / model_version 两个 key 必须存在（值可为 null）[契约依据] §4.2 BLOCK 14 + §4.3 BLOCK 17

    说明(独立性): 任务清单提及 model_provider, 但契约 §4 Schema 未定义该字段；
    按「期望 schema 只能来自契约」铁律, 本用例仅断言契约定义的 workflow_version / model_version。
    """
    meta = resp["diagnosis_meta"]
    assert "workflow_version" in meta, "diagnosis_meta 缺少 workflow_version"
    assert "model_version" in meta, "diagnosis_meta 缺少 model_version（值可 null 但 key 必须存在）"


def test_meta_qa_e2e_status_enums(resp):
    """qa_status / e2e_status 枚举合法 [契约依据] §4.2 BLOCK 14"""
    meta = resp["diagnosis_meta"]
    assert meta.get("qa_status") in {"PASS", "FAIL", "NOT_RUN"}, (
        f"qa_status={meta.get('qa_status')!r} 非法"
    )
    assert meta.get("e2e_status") in {"passed", "failed", "not_run"}, (
        f"e2e_status={meta.get('e2e_status')!r} 非法"
    )


# ===========================================================================
# 3. product_understanding  [契约依据] §5 BLOCK 23
# ===========================================================================
def test_pu_basic_info_required_fields(resp):
    """basic_info 必填子字段齐全 + price_band 枚举 [契约依据] §5.2 BLOCK 23"""
    bi = resp["product_understanding"]["basic_info"]
    for k in ["product_name", "leaf_category", "brand_name", "shop_name", "price", "price_band"]:
        assert k in bi, f"basic_info 缺少子字段 {k}"
    assert _nonempty_str(bi["product_name"]), "basic_info.product_name 不得为空"
    assert _nonempty_str(bi["leaf_category"]), "basic_info.leaf_category 不得为空"
    assert bi["price_band"] in PRICE_BAND_ENUM, (
        f"price_band={bi['price_band']!r} 不在枚举 {PRICE_BAND_ENUM}"
    )


def test_pu_target_people_nonempty_array(resp):
    """target_people 为非空字符串数组 [契约依据] §5.2 BLOCK 23"""
    tp = resp["product_understanding"]["target_people"]
    assert isinstance(tp, list) and len(tp) > 0, f"target_people 必须非空数组, 实际={tp!r}"
    assert all(_nonempty_str(x) for x in tp), "target_people 各项必须为非空字符串"


def test_pu_core_selling_points_nonempty_array(resp):
    """core_selling_points 为非空字符串数组 [契约依据] §5.2 BLOCK 23"""
    sp = resp["product_understanding"]["core_selling_points"]
    assert isinstance(sp, list) and len(sp) > 0, f"core_selling_points 必须非空数组, 实际={sp!r}"
    assert all(_nonempty_str(x) for x in sp), "core_selling_points 各项必须为非空字符串"


def test_pu_jtbd_required_fields(resp):
    """jtbd 必填字段 domain/primary_task/sub_task/reasoning/evidence_chain [契约依据] §5.2 BLOCK 23"""
    jtbd = resp["product_understanding"]["jtbd"]
    for k in ["domain", "primary_task", "sub_task", "reasoning", "evidence_chain"]:
        assert k in jtbd, f"jtbd 缺少字段 {k}"
    assert _nonempty_str(jtbd["domain"]), "jtbd.domain 不得为空"
    assert _nonempty_str(jtbd["primary_task"]), "jtbd.primary_task 不得为空"
    assert _nonempty_str(jtbd["reasoning"]), "jtbd.reasoning 不得为空"
    assert isinstance(jtbd["evidence_chain"], list), "jtbd.evidence_chain 必须是数组"


def test_pu_supporting_requirements_item_fields(resp):
    """supporting_requirements 各项字段 + priority 枚举 [契约依据] §5.2 BLOCK 23"""
    sr = resp["product_understanding"]["supporting_requirements"]
    assert isinstance(sr, list) and len(sr) > 0, "supporting_requirements 必须非空数组"
    for i, item in enumerate(sr):
        for k in ["requirement_id", "requirement_name", "priority", "description"]:
            assert k in item, f"supporting_requirements[{i}] 缺少 {k}"
        assert item["priority"] in PRIORITY_ENUM, (
            f"supporting_requirements[{i}].priority={item['priority']!r} 不在枚举 {PRIORITY_ENUM}"
        )


def test_pu_expected_hec_three_tags(resp):
    """expected_hec 三标签齐全且非空 [契约依据] §5.2 BLOCK 23"""
    eh = resp["product_understanding"]["expected_hec"]
    for k in ["hook_tag", "effect_tag", "cta_tag"]:
        assert k in eh, f"expected_hec 缺少 {k}"
        assert _nonempty_str(eh[k]), f"expected_hec.{k} 不得为空"


def test_pu_candidate_set_fields(resp):
    """candidate_set 字段齐全 [契约依据] §5.2 BLOCK 23"""
    cs = resp["product_understanding"]["candidate_set"]
    for k in ["candidate_h", "core_e", "core_c", "primary_effect", "primary_cta"]:
        assert k in cs, f"candidate_set 缺少 {k}"
    for k in ["candidate_h", "core_e", "core_c"]:
        assert isinstance(cs[k], list), f"candidate_set.{k} 必须是数组"


def test_pu_conversion_resistance_fields(resp):
    """conversion_resistance 子字段齐全 + 枚举合法 [契约依据] §5.2 BLOCK 23"""
    cr = resp["product_understanding"]["conversion_resistance"]
    assert cr["trust_barrier"] in HML_UNKNOWN, f"trust_barrier={cr.get('trust_barrier')!r} 非法"
    assert cr["price_barrier"] in HML_UNKNOWN, f"price_barrier={cr.get('price_barrier')!r} 非法"
    assert cr["channel_risk"] in CHANNEL_RISK_ENUM, f"channel_risk={cr.get('channel_risk')!r} 非法"
    assert cr["endorsement"] in ENDORSEMENT_ENUM, f"endorsement={cr.get('endorsement')!r} 非法"
    assert cr["brand_tier"] in BRAND_TIER_ENUM, f"brand_tier={cr.get('brand_tier')!r} 非法"


def test_pu_evidence_schema(resp):
    """product_understanding.evidence 统一 evidence schema [契约依据] §14 BLOCK 123"""
    ev = resp["product_understanding"]["evidence"]
    assert isinstance(ev, list), "product_understanding.evidence 必须是数组"
    for i, e in enumerate(ev):
        _assert_evidence_item(e, f"product_understanding.evidence[{i}]")


# ===========================================================================
# 4. video_understanding  [契约依据] §6 BLOCK 33 + 6.3
# ===========================================================================
def test_vu_text_stream_fields(resp):
    """text_stream: asr_summary/asr_segments/ocr 字段齐全 [契约依据] §6.2 BLOCK 33"""
    ts = resp["video_understanding"]["text_stream"]
    for k in ["asr_summary", "asr_segments", "ocr_text", "ocr_texts"]:
        assert k in ts, f"text_stream 缺少 {k}"
    assert isinstance(ts["asr_segments"], list), "asr_segments 必须是数组"
    assert isinstance(ts["ocr_texts"], list), "ocr_texts 必须是数组"
    # asr_segments 各项字段
    for i, seg in enumerate(ts["asr_segments"]):
        for k in ["segment_id", "start_sec", "end_sec", "text", "text_summary", "confidence"]:
            assert k in seg, f"asr_segments[{i}] 缺少 {k}"


def test_vu_visual_segments_nonempty(resp):
    """visual_stream.visual_segments 非空数组（不得长期返回空数组）[契约依据] §6.3 BLOCK 35"""
    vs = resp["video_understanding"]["visual_stream"]["visual_segments"]
    assert isinstance(vs, list) and len(vs) > 0, (
        f"visual_segments 必须为非空数组, 实际={vs!r}"
    )


def test_vu_visual_segment_p0_fields(resp):
    """每个 visual_segment 的 P0 字段齐全, core_scene_desc/core_action 非空, evidence_role 枚举 [契约依据] §6.3 BLOCK 36-43 + §6.2 BLOCK 33"""
    vs = resp["video_understanding"]["visual_stream"]["visual_segments"]
    for i, seg in enumerate(vs):
        for k in ["segment_id", "core_scene_desc", "core_action", "related_ocr_texts", "evidence_role"]:
            assert k in seg, f"visual_segments[{i}] 缺少 P0 字段 {k}"
        assert _nonempty_str(seg["segment_id"]), f"visual_segments[{i}].segment_id 不得为空"
        assert _nonempty_str(seg["core_scene_desc"]), f"visual_segments[{i}].core_scene_desc 不得为空"
        assert _nonempty_str(seg["core_action"]), f"visual_segments[{i}].core_action 不得为空"
        assert isinstance(seg["related_ocr_texts"], list), (
            f"visual_segments[{i}].related_ocr_texts 必须是数组"
        )
        assert seg["evidence_role"] in EVIDENCE_ROLE_ENUM, (
            f"visual_segments[{i}].evidence_role={seg['evidence_role']!r} 不在枚举 {EVIDENCE_ROLE_ENUM}"
        )


def test_vu_actual_hec_fields(resp):
    """actual_hec 三标签齐全 [契约依据] §6.2 BLOCK 33"""
    ah = resp["video_understanding"]["actual_hec"]
    for k in ["hook_tag", "effect_tag", "cta_tag"]:
        assert k in ah, f"actual_hec 缺少 {k}"
        assert _nonempty_str(ah[k]), f"actual_hec.{k} 不得为空"


def test_vu_slider_signature_fields(resp):
    """slider_signature 字段齐全 [契约依据] §6.2 BLOCK 33"""
    ss = resp["video_understanding"]["slider_signature"]
    for k in ["visual", "audio", "proof", "cta", "summary"]:
        assert k in ss, f"slider_signature 缺少 {k}"


def test_vu_total_segment_count_matches(resp):
    """video_base_fact.total_segment_count == len(visual_segments) [契约依据] §6.2 BLOCK 33]"""
    vu = resp["video_understanding"]
    total = vu["video_base_fact"]["total_segment_count"]
    n = len(vu["visual_stream"]["visual_segments"])
    assert total == n, f"total_segment_count={total} 与 visual_segments 数量={n} 不一致"


def test_vu_evidence_spans_schema(resp):
    """video_understanding.evidence_spans 统一 evidence schema [契约依据] §14 BLOCK 123"""
    es = resp["video_understanding"]["evidence_spans"]
    assert isinstance(es, list), "evidence_spans 必须是数组"
    for i, e in enumerate(es):
        _assert_evidence_item(e, f"video_understanding.evidence_spans[{i}]")


# ===========================================================================
# 5. diagnosis.overview  [契约依据] §7 BLOCK 63
# ===========================================================================
def test_overview_all_status_keys_and_summary(resp):
    """overview 各模块 *_status 齐全 + summary [契约依据] §7.2 BLOCK 63"""
    ov = resp["diagnosis"]["overview"]
    required = [
        "overall_status",
        "audience_match_status",
        "profile_match_status",
        "hec_match_status",
        "slider_match_status",
        "requirement_coverage_status",
        "requirement_coverage_text",
        "summary",
    ]
    for k in required:
        assert k in ov, f"overview 缺少 {k}"
    assert _nonempty_str(ov["summary"]), "overview.summary 不得为空"


def test_overview_status_enums(resp):
    """overview 各状态值合法 [契约依据] §7.2 BLOCK 63"""
    ov = resp["diagnosis"]["overview"]
    assert ov["overall_status"] in OVERALL_STATUS_ENUM, f"overall_status={ov['overall_status']!r} 非法"
    assert ov["audience_match_status"] in AUDIENCE_MATCH_STATUS_ENUM, f"audience_match_status 非法"
    assert ov["profile_match_status"] in PROFILE_MATCH_STATUS_OV_ENUM, f"profile_match_status 非法"
    assert ov["hec_match_status"] in HEC_MATCH_STATUS_ENUM, f"hec_match_status 非法"
    assert ov["slider_match_status"] in SLIDER_MATCH_STATUS_OV_ENUM, f"slider_match_status 非法"
    assert ov["requirement_coverage_status"] in REQ_COVERAGE_STATUS_OV_ENUM, f"requirement_coverage_status 非法"


# ===========================================================================
# 6. diagnosis.profile_match  [契约依据] §8 BLOCK 78 + 8.3
# ===========================================================================
def test_profile_match_structure_and_enums(resp):
    """profile_match 结构齐全 + status/match_result/gap.level 枚举 [契约依据] §8.2 BLOCK 78"""
    pm = resp["diagnosis"]["profile_match"]
    for k in ["status", "product_audience", "video_audience", "gap", "match_result", "evidence", "summary"]:
        assert k in pm, f"profile_match 缺少 {k}"
    assert pm["status"] in PROFILE_MATCH_STATUS_ENUM, f"profile_match.status={pm['status']!r} 非法"
    for side in ["product_audience", "video_audience"]:
        for sk in ["primary", "scene", "core_need"]:
            assert sk in pm[side], f"profile_match.{side} 缺少 {sk}"
    assert "level" in pm["gap"] and "description" in pm["gap"], "profile_match.gap 缺少 level/description"
    assert pm["gap"]["level"] in GAP_LEVEL_ENUM, f"gap.level={pm['gap']['level']!r} 非法"
    assert pm["match_result"] in MATCH_RESULT_ENUM, f"match_result={pm['match_result']!r} 非法"
    assert isinstance(pm["evidence"], list), "profile_match.evidence 必须是数组"


def test_profile_match_evidence_schema(resp):
    """profile_match.evidence 各项符合统一 evidence schema [契约依据] §14 BLOCK 123"""
    for i, e in enumerate(resp["diagnosis"]["profile_match"]["evidence"]):
        _assert_evidence_item(e, f"profile_match.evidence[{i}]")


def test_profile_match_completed_needs_review_nonempty(resp):
    """completed/needs_review 下必填 string 不得为空 [契约依据] §8.3 BLOCK 81（条件用例）"""
    pm = resp["diagnosis"]["profile_match"]
    if pm["status"] in {"completed", "needs_review"}:
        for side in ["product_audience", "video_audience"]:
            for sk in ["primary", "scene", "core_need"]:
                assert _nonempty_str(pm[side][sk]), (
                    f"status={pm['status']} 下 profile_match.{side}.{sk} 不得为空"
                )
        assert _nonempty_str(pm["summary"]), f"status={pm['status']} 下 summary 不得为空"
    else:
        pytest.skip(f"profile_match.status={pm['status']}，BLOCK 81 条件不触发")


def test_profile_match_insufficient_evidence_summary(resp):
    """insufficient_evidence 下 summary 必须说明缺失原因（非空）[契约依据] §8.3 BLOCK 82（条件用例）"""
    pm = resp["diagnosis"]["profile_match"]
    if pm["status"] == "insufficient_evidence":
        assert _nonempty_str(pm["summary"]), "insufficient_evidence 下 summary 必须说明缺失原因"
    else:
        pytest.skip(f"profile_match.status={pm['status']}，BLOCK 82 条件不触发")


def test_profile_match_completed_evidence_both_sides(resp):
    """completed 下 evidence 必须覆盖商品侧与视频侧 [契约依据] §8.3 BLOCK 83（条件用例）"""
    pm = resp["diagnosis"]["profile_match"]
    if pm["status"] == "completed":
        sources = {e.get("source") for e in pm["evidence"]}
        assert "product_factpack" in sources, "completed 下 evidence 缺少商品侧(product_factpack)"
        assert "video_factpack" in sources, "completed 下 evidence 缺少视频侧(video_factpack)"
    else:
        pytest.skip(f"profile_match.status={pm['status']}，BLOCK 83 条件不触发")


# ===========================================================================
# 7. diagnosis.requirement_coverage  [契约依据] §9 BLOCK 88
# ===========================================================================
def test_rc_count_and_status(resp):
    """completed_count <= total_count, status 枚举合法 [契约依据] §9.2 BLOCK 88"""
    rc = resp["diagnosis"]["requirement_coverage"]
    assert rc["status"] in REQ_COVERAGE_STATUS_ENUM, f"rc.status={rc['status']!r} 非法"
    assert isinstance(rc["completed_count"], int) and isinstance(rc["total_count"], int)
    assert rc["completed_count"] <= rc["total_count"], (
        f"completed_count({rc['completed_count']}) > total_count({rc['total_count']})"
    )


def test_rc_items_fields_and_enum(resp):
    """items 每项字段齐全 + completion_status 枚举 [契约依据] §9.2 BLOCK 88"""
    rc = resp["diagnosis"]["requirement_coverage"]
    assert isinstance(rc["items"], list) and len(rc["items"]) > 0, "items 必须非空数组"
    for i, it in enumerate(rc["items"]):
        for k in [
            "requirement_id", "requirement_name", "required", "completion_status",
            "expected", "actual", "matched_evidence_spans", "missing_reason", "repair_direction",
        ]:
            assert k in it, f"requirement_coverage.items[{i}] 缺少 {k}"
        assert it["completion_status"] in COMPLETION_STATUS_ENUM, (
            f"items[{i}].completion_status={it['completion_status']!r} 不在枚举 {COMPLETION_STATUS_ENUM}"
        )


def test_rc_completed_items_have_evidence(resp):
    """completed 项 matched_evidence_spans 非空 [契约依据] §9.1 BLOCK 86 + §9.2 BLOCK 88"""
    rc = resp["diagnosis"]["requirement_coverage"]
    for i, it in enumerate(rc["items"]):
        if it["completion_status"] == "completed":
            spans = it["matched_evidence_spans"]
            assert isinstance(spans, list) and len(spans) > 0, (
                f"completed 项 items[{i}]({it['requirement_id']}) 的 matched_evidence_spans 不得为空"
            )
            for j, e in enumerate(spans):
                _assert_evidence_item(e, f"rc.items[{i}].matched_evidence_spans[{j}]")


# ===========================================================================
# 8. diagnosis.hec_match  [契约依据] §10 BLOCK 97
# ===========================================================================
def test_hec_match_structure(resp):
    """hec_match: status + product_expected + video_actual 双侧对照 [契约依据] §10.2 BLOCK 97 + §10.3 BLOCK 100"""
    hm = resp["diagnosis"]["hec_match"]
    assert hm["status"] in HEC_MATCH_STATUS_ENUM, f"hec_match.status={hm['status']!r} 非法"
    for side in ["product_expected", "video_actual"]:
        assert side in hm, f"hec_match 缺少 {side}"
        for k in ["hook_tag", "effect_tag", "cta_tag"]:
            assert k in hm[side], f"hec_match.{side} 缺少 {k}"


def test_hec_dimension_results(resp):
    """dimension_results 每项 dimension/expected/actual/status/impact/suggestion + 枚举 [契约依据] §10.2 BLOCK 97"""
    hm = resp["diagnosis"]["hec_match"]
    drs = hm["dimension_results"]
    assert isinstance(drs, list) and len(drs) > 0, "dimension_results 必须非空数组"
    for i, dr in enumerate(drs):
        for k in ["dimension", "expected", "actual", "status", "impact", "suggestion"]:
            assert k in dr, f"dimension_results[{i}] 缺少 {k}"
        assert dr["dimension"] in HEC_DIMENSION_ENUM, (
            f"dimension_results[{i}].dimension={dr['dimension']!r} 不在枚举 {HEC_DIMENSION_ENUM}"
        )
        assert dr["status"] in HEC_DIM_STATUS_ENUM, (
            f"dimension_results[{i}].status={dr['status']!r} 不在枚举 {HEC_DIM_STATUS_ENUM}"
        )


# ===========================================================================
# 9. diagnosis.slider_match  [契约依据] §11 BLOCK 106
# ===========================================================================
def test_slider_match_status_enum(resp):
    """slider_match.status 枚举合法 [契约依据] §11.2 BLOCK 106"""
    sm = resp["diagnosis"]["slider_match"]
    assert sm["status"] in SLIDER_STATUS_ENUM, f"slider_match.status={sm['status']!r} 非法"


def test_slider_axis_results(resp):
    """axis_results 每项 axis/fit_status/expected/actual/judgment/repair_direction/evidence + 枚举 [契约依据] §11.2 BLOCK 106"""
    sm = resp["diagnosis"]["slider_match"]
    ars = sm["axis_results"]
    assert isinstance(ars, list) and len(ars) > 0, "axis_results 必须非空数组"
    for i, ar in enumerate(ars):
        for k in ["axis", "fit_status", "expected", "actual", "judgment", "repair_direction", "evidence"]:
            assert k in ar, f"axis_results[{i}] 缺少 {k}"
        assert ar["axis"] in AXIS_ENUM, f"axis_results[{i}].axis={ar['axis']!r} 不在枚举 {AXIS_ENUM}"
        assert ar["fit_status"] in FIT_STATUS_ENUM, (
            f"axis_results[{i}].fit_status={ar['fit_status']!r} 不在枚举 {FIT_STATUS_ENUM}"
        )
        assert isinstance(ar["evidence"], list), f"axis_results[{i}].evidence 必须是数组"
        for j, e in enumerate(ar["evidence"]):
            _assert_evidence_item(e, f"slider.axis_results[{i}].evidence[{j}]")


# ===========================================================================
# 10. top_issues / suggestions  [契约依据] §12 BLOCK 109 + 12.2
# ===========================================================================
def test_top_issues_fields_and_enums(resp):
    """top_issues 每项 issue_id/severity/module/title/description + 枚举 [契约依据] §12.1 BLOCK 109"""
    issues = resp["diagnosis"]["top_issues"]
    assert isinstance(issues, list) and len(issues) > 0, "top_issues 必须非空数组"
    for i, t in enumerate(issues):
        for k in ["issue_id", "severity", "module", "title", "description"]:
            assert k in t, f"top_issues[{i}] 缺少 {k}"
            if k != "severity" and k != "module":
                assert _nonempty_str(t[k]), f"top_issues[{i}].{k} 不得为空"
        assert t["severity"] in SEVERITY_ENUM, f"top_issues[{i}].severity={t['severity']!r} 不在枚举 {SEVERITY_ENUM}"
        assert t["module"] in MODULE_ENUM, f"top_issues[{i}].module={t['module']!r} 不在枚举 {MODULE_ENUM}"


def test_suggestions_fields_and_enums(resp):
    """suggestions 每项 suggestion_id/priority/module/action/reason + 枚举 [契约依据] §12.1 BLOCK 109"""
    sugs = resp["diagnosis"]["suggestions"]
    assert isinstance(sugs, list) and len(sugs) > 0, "suggestions 必须非空数组"
    for i, s in enumerate(sugs):
        for k in ["suggestion_id", "priority", "module", "action", "reason"]:
            assert k in s, f"suggestions[{i}] 缺少 {k}"
            if k not in ("priority", "module"):
                assert _nonempty_str(s[k]), f"suggestions[{i}].{k} 不得为空"
        assert s["priority"] in SUGGESTION_PRIORITY_ENUM, (
            f"suggestions[{i}].priority={s['priority']!r} 不在枚举 {SUGGESTION_PRIORITY_ENUM}"
        )
        assert s["module"] in MODULE_ENUM, f"suggestions[{i}].module={s['module']!r} 不在枚举 {MODULE_ENUM}"


def test_suggestions_related_issue_id_traceable(resp):
    """suggestions.related_issue_id 必须能回指到某 top_issues.issue_id [契约依据] §12.2 BLOCK 113"""
    issue_ids = {t["issue_id"] for t in resp["diagnosis"]["top_issues"]}
    for i, s in enumerate(resp["diagnosis"]["suggestions"]):
        rid = s.get("related_issue_id")
        assert rid in issue_ids, (
            f"suggestions[{i}].related_issue_id={rid!r} 无法回指到 top_issues.issue_id {issue_ids}"
        )


# ===========================================================================
# 11. artifacts  [契约依据] §13 BLOCK 116 + 13.2
# ===========================================================================
def test_artifacts_keys_present(resp):
    """artifacts: request_payload/raw_response/normalized_response/source_files 齐全 [契约依据] §13.1 BLOCK 116"""
    art = resp["artifacts"]
    assert ARTIFACTS_KEYS.issubset(art.keys()), f"artifacts 缺少: {ARTIFACTS_KEYS - set(art.keys())}"
    assert isinstance(art["source_files"], list) and len(art["source_files"]) > 0, (
        "source_files 必须非空数组"
    )


def test_artifacts_raw_response_contains_persuasion_result(resp):
    """raw_response 含 video_persuasion_diagnosis_result（Smoke 验收依据）[契约依据] §13.2 BLOCK 118"""
    rr = resp["artifacts"]["raw_response"]
    assert isinstance(rr, dict), "raw_response 必须是对象"
    assert "video_persuasion_diagnosis_result" in rr, (
        "raw_response 缺少 video_persuasion_diagnosis_result"
    )


def test_artifacts_normalized_equals_diagnosis(resp):
    """normalized_response 等价于 diagnosis 对象 [契约依据] §13.1 BLOCK 116 + 最终裁定 BLOCK 144"""
    assert resp["artifacts"]["normalized_response"] == resp["diagnosis"], (
        "normalized_response 与 diagnosis 对象不等价"
    )


# ===========================================================================
# 12. 统一 evidence schema 汇总  [契约依据] §14 BLOCK 123
# ===========================================================================
def test_unified_evidence_schema_all_modules(resp):
    """所有被验收对象的 evidence 项含 source/field/value/segment_id/confidence 五 key 且 source 合法 [契约依据] §14 BLOCK 123"""
    collected = []
    collected += resp["product_understanding"]["evidence"]
    collected += resp["video_understanding"]["evidence_spans"]
    collected += resp["video_understanding"]["video_jtbd"]["evidence"]
    collected += resp["diagnosis"]["profile_match"]["evidence"]
    for it in resp["diagnosis"]["requirement_coverage"]["items"]:
        collected += it["matched_evidence_spans"]
    for ar in resp["diagnosis"]["slider_match"]["axis_results"]:
        collected += ar["evidence"]
    for t in resp["diagnosis"]["top_issues"]:
        collected += t["evidence"]
    assert len(collected) > 0, "未采集到任何 evidence 项"
    for i, e in enumerate(collected):
        _assert_evidence_item(e, f"unified_evidence[{i}]")
