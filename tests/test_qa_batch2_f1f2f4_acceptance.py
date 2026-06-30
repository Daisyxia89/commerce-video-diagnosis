# -*- coding: utf-8 -*-
"""独立 QA 正式验收 —— 第二批 F1+F2+F4（contract 装配层改造）。

独立性声明（强制）:
- 本文件由 /JG 独立节点依据《module3 PRD §5.2 商品事实向量枚举约束》与本批 AC 自行编写。
- **不复用、不导入**研发自写/自改的 tests/test_frontend_contract_acceptance.py、
  tests/test_product_target_audience.py 的任何断言或常量。
- 期望枚举闭集一律在本文件内**独立硬编码**（抄录自 PRD），不从 schema_assertions 导入 CONTRACT_* 常量。
- 验收对象为**真实产物**：
  (a) outputs/runben_diagnosis/runben_full_diagnosis.json —— 真实商品诊断 dict（引擎产出），
      喂给真实实现函数 build_product_understanding / build_product_fact_vector 得到真实输出 dict；
  (b) outputs/runben_diagnosis/runben_contract_response.json —— 真实装配后的完整契约响应。
- 调用实现侧的 build_* / assert_* 属于「对被测实现取真实输出 / 触发其断言」，非复用研发测试断言。
- 发现实现与标准冲突一律如实 Fail，不软化断言、不改实现。

AC 覆盖: AC1~AC12（见各 test_ac* docstring）。
"""
import copy
import json
import os

import pytest

# 被测实现（assembly 装配层 + 契约断言层）—— 这是验收对象本身，非研发测试用例。
from commerce_video_diagnosis.understanding.assembly.response_assembler import (
    ContractAssemblyError,
    build_product_fact_vector,
    build_product_understanding,
)
from commerce_video_diagnosis.understanding.validators.schema_assertions import (
    SchemaAssertionError,
    assert_contract_product_understanding,
    assert_frontend_contract_response,
)

# ---------------------------------------------------------------------------
# 真实产物路径
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_FULL_DIAG = os.path.join(_REPO, "outputs", "runben_diagnosis", "runben_full_diagnosis.json")
_CONTRACT = os.path.join(_REPO, "outputs", "runben_diagnosis", "runben_contract_response.json")

# ---------------------------------------------------------------------------
# 独立硬编码的 PRD §5.2 商品事实向量枚举闭集（不从实现导入）
# ---------------------------------------------------------------------------
PRD_COGNITION = {"蓝海", "红海-核心", "红海-破圈"}      # 认知属性
PRD_FREQUENCY = {"快消", "耐消"}                         # 频次属性
PRD_TRUST = {"大牌", "白牌"}                             # 信任属性
PRD_PRICE = {"高", "低"}                                 # 价格属性
PRD_ENDORSEMENT = {"有背书", None}                       # 背书属性（合法空值 null）
PRD_CHANNEL_RISK = {"有风险", None}                      # 渠道风险（合法空值 null）

# F1 固定 6 段（顺序敏感）
EXPECTED_PU_ORDER = [
    "basic_info", "product_fact_vector", "module3", "candidate_set", "product_hec", "evidence",
]

# F4 禁出键（product_understanding 子树递归不得出现）
BANNED_KEYS = {
    "trust_barrier", "brand_tier", "price_barrier", "financial_risk", "relative_price_level",
    "expected_hec", "supporting_requirements", "conversion_resistance", "target_people", "price_band",
}
# F4 禁出「旧英文/旧工程枚举」字符串值。
# 说明：AC6 列出的 channel_risk 旧枚举含 low/medium/high/低/高，但这些是「generic 通用词」——
#   priority 字段合法使用 high/medium/low，price_attribute 合法使用 高/低。
#   因此对全子树做 blanket 扫描时，只针对「旧系统唯一、无歧义」的 token；
#   channel_risk_attribute / endorsement_attribute 字段本身是否携带旧枚举，
#   由 AC5 闭集断言（{有风险,null} / {有背书,null}）+ 本文件字段级检查共同保证。
BANNED_ENDORSEMENT_VALUES = {"unknown", "no_endorsement", "has_endorsement"}
BANNED_CHANNEL_RISK_UNIQUE = {"no_risk"}
# channel_risk / endorsement 字段绝不允许出现的旧工程枚举（字段级，精确比对）
LEGACY_CHANNEL_RISK_FIELD_VALUES = {"no_risk", "risk", "low", "medium", "high", "低", "高"}
LEGACY_ENDORSEMENT_FIELD_VALUES = {"unknown", "no_endorsement", "has_endorsement"}


# ---------------------------------------------------------------------------
# fixtures：真实产物 + 真实实现输出 + 第二品类增强覆盖
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_pd():
    """真实商品诊断 dict（引擎产出，落盘）。"""
    with open(_FULL_DIAG, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def contract_resp():
    """真实装配后的完整契约响应。"""
    with open(_CONTRACT, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pu(real_pd):
    """对真实商品诊断调用被测实现 build_product_understanding，得到真实 product_understanding。"""
    return build_product_understanding(real_pd)


@pytest.fixture(scope="module")
def pu_json(contract_resp):
    """落盘契约响应中的 product_understanding（验证序列化后顺序/内容一致）。"""
    return contract_resp["product_understanding"]


@pytest.fixture(scope="module")
def fact_vector_alt():
    """第二品类增强覆盖：蓝海 / 耐消 / 白牌 / 低 / 有背书 / 有风险，覆盖 AC4/AC5 的另一支枚举。"""
    pd_alt = {
        "resistance_profile": {
            "ocean": "蓝海",
            "frequency": "耐消",
            "brand_tier": "白牌",
            "relative_price_level": "低水位",
            "endorsement": "某权威机构认证",
            "channel_risk": "有风险",
        }
    }
    return build_product_fact_vector(pd_alt)


# ---------------------------------------------------------------------------
# 通用工具：递归扫描
# ---------------------------------------------------------------------------
def _scan_keys(node, target_keys, path="product_understanding"):
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if k in target_keys:
                hits.append(p)
            hits.extend(_scan_keys(v, target_keys, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_scan_keys(v, target_keys, f"{path}[{i}]"))
    return hits


def _scan_str_values(node, target_values, path="product_understanding"):
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            hits.extend(_scan_str_values(v, target_values, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_scan_str_values(v, target_values, f"{path}[{i}]"))
    elif isinstance(node, str) and node in target_values:
        hits.append((path, node))
    return hits


# ===========================================================================
# AC1（F1 6 段 + 顺序）
# ===========================================================================
def test_ac1_six_segments_ordered(pu, pu_json):
    """product_understanding 顶层键有序等于固定 6 段。"""
    assert list(pu.keys()) == EXPECTED_PU_ORDER, f"实测(build)={list(pu.keys())}"
    # 落盘 JSON 序列化后顺序同样固定
    assert list(pu_json.keys()) == EXPECTED_PU_ORDER, f"实测(json)={list(pu_json.keys())}"


# ===========================================================================
# AC2（F1 module3：含且仅含两对象且非空）
# ===========================================================================
def test_ac2_module3_exactly_two_nonempty(pu):
    m3 = pu["module3"]
    assert set(m3.keys()) == {"persuasion_requirement_profile", "product_target_audience"}, \
        f"module3 键集合={set(m3.keys())}"
    profile = m3["persuasion_requirement_profile"]
    pta = m3["product_target_audience"]
    assert profile and profile.get("persuasion_requirements"), "profile.persuasion_requirements 为空"
    assert pta and pta.get("primary_audiences"), "pta.primary_audiences 为空"


# ===========================================================================
# AC3（F1 product_hec 存在且无 expected_hec）
# ===========================================================================
def test_ac3_product_hec_present_no_expected_hec(pu):
    assert "product_hec" in pu, "缺 product_hec"
    assert isinstance(pu["product_hec"], list) and len(pu["product_hec"]) > 0, "product_hec 非非空 list"
    # product_understanding 子树递归不得出现 expected_hec
    assert _scan_keys(pu, {"expected_hec"}) == [], "出现禁出键 expected_hec"


# ===========================================================================
# AC4（F2 六维齐全）
# ===========================================================================
def test_ac4_fact_vector_six_dimensions(pu, fact_vector_alt):
    fv = pu["product_fact_vector"]
    six = [
        "cognition_attribute", "frequency_attribute", "trust_attribute",
        "price_attribute", "endorsement_attribute", "channel_risk_attribute",
    ]
    for dim in six:
        assert dim in fv, f"product_fact_vector 缺维度 {dim}"
    # 第二品类同样具备 6 维
    for dim in six:
        assert dim in fact_vector_alt, f"第二品类 product_fact_vector 缺维度 {dim}"


# ===========================================================================
# AC5（F2 枚举闭集 + 越界拦截）
# ===========================================================================
def test_ac5_enum_closed_set_real_and_alt(pu, fact_vector_alt):
    """真实样本 + 第二品类样本的六维取值必须落在 PRD 闭集内。"""
    for fv, tag in ((pu["product_fact_vector"], "runben"), (fact_vector_alt, "alt")):
        assert fv["cognition_attribute"] in PRD_COGNITION, f"[{tag}] cognition={fv['cognition_attribute']}"
        assert fv["frequency_attribute"] in PRD_FREQUENCY, f"[{tag}] frequency={fv['frequency_attribute']}"
        assert fv["trust_attribute"] in PRD_TRUST, f"[{tag}] trust={fv['trust_attribute']}"
        assert fv["price_attribute"] in PRD_PRICE, f"[{tag}] price={fv['price_attribute']}"
        assert fv["endorsement_attribute"] in PRD_ENDORSEMENT, f"[{tag}] endorsement={fv['endorsement_attribute']}"
        assert fv["channel_risk_attribute"] in PRD_CHANNEL_RISK, f"[{tag}] channel_risk={fv['channel_risk_attribute']}"
    # 两品类合并需覆盖到全部枚举分支（增强覆盖证据）
    trusts = {pu["product_fact_vector"]["trust_attribute"], fact_vector_alt["trust_attribute"]}
    prices = {pu["product_fact_vector"]["price_attribute"], fact_vector_alt["price_attribute"]}
    assert trusts == PRD_TRUST, f"trust 分支未覆盖全：{trusts}"
    assert prices == PRD_PRICE, f"price 分支未覆盖全：{prices}"


@pytest.mark.parametrize("dim,bad_value", [
    ("trust_attribute", "中牌"),
    ("price_attribute", "中"),
    ("endorsement_attribute", "no_endorsement"),
    ("channel_risk_attribute", "no_risk"),
    ("cognition_attribute", "紫海"),
    ("frequency_attribute", "中频"),
])
def test_ac5_out_of_range_blocked_by_assertion(pu, dim, bad_value):
    """越界枚举注入 → 契约断言必须拦截（SchemaAssertionError），不得放过。"""
    bad = copy.deepcopy(pu)
    bad["product_fact_vector"][dim] = bad_value
    with pytest.raises(SchemaAssertionError):
        assert_contract_product_understanding(bad)


def test_ac5_build_rejects_bad_source_enum():
    """源 ocean 非法 → build_product_fact_vector 必须 Crash Early（ContractAssemblyError）。"""
    with pytest.raises(ContractAssemblyError):
        build_product_fact_vector({"resistance_profile": {
            "ocean": "灰海", "frequency": "快消", "brand_tier": "白牌",
            "relative_price_level": "低水位",
        }})


# ===========================================================================
# AC6（F4 禁出 · product_understanding 子树）
# ===========================================================================
def test_ac6_no_banned_keys_in_subtree(pu, pu_json):
    for tree, tag in ((pu, "build"), (pu_json, "json")):
        key_hits = _scan_keys(tree, BANNED_KEYS)
        assert key_hits == [], f"[{tag}] 命中禁出键：{key_hits}"


def test_ac6_no_banned_legacy_values_in_subtree(pu, pu_json):
    # blanket 扫描仅针对旧系统唯一、无歧义 token（避免误杀 priority=high/medium、price=高/低）
    banned_values = BANNED_ENDORSEMENT_VALUES | BANNED_CHANNEL_RISK_UNIQUE
    for tree, tag in ((pu, "build"), (pu_json, "json")):
        val_hits = _scan_str_values(tree, banned_values)
        assert val_hits == [], f"[{tag}] 命中禁出旧枚举值：{val_hits}"


def test_ac6_channel_risk_endorsement_fields_no_legacy_enum(pu, pu_json, fact_vector_alt):
    """字段级精确比对：channel_risk_attribute / endorsement_attribute 绝不携带旧工程枚举。"""
    for fv, tag in (
        (pu["product_fact_vector"], "build"),
        (pu_json["product_fact_vector"], "json"),
        (fact_vector_alt, "alt"),
    ):
        assert fv["channel_risk_attribute"] not in LEGACY_CHANNEL_RISK_FIELD_VALUES, \
            f"[{tag}] channel_risk_attribute 携带旧枚举：{fv['channel_risk_attribute']}"
        assert fv["endorsement_attribute"] not in LEGACY_ENDORSEMENT_FIELD_VALUES, \
            f"[{tag}] endorsement_attribute 携带旧枚举：{fv['endorsement_attribute']}"


def test_ac6_no_top_level_conversion_resistance_or_target_people(pu):
    assert "conversion_resistance" not in pu
    assert "target_people" not in pu
    # conversion_barriers 不应作为 product_understanding 顶层段（仅可存在于 fact_vector 内）
    assert "conversion_barriers" not in pu


# ===========================================================================
# AC7（F4 target_people 迁移）
# ===========================================================================
def test_ac7_target_people_migrated_to_audience_hint(pu):
    assert "target_people" not in pu, "顶层仍有 target_people"
    audience_hint = pu["basic_info"].get("audience_hint")
    assert audience_hint, "basic_info.audience_hint 为空"
    assert isinstance(audience_hint, list) and all(isinstance(x, str) and x.strip() for x in audience_hint)


# ===========================================================================
# AC8（conversion_barriers 仅解释层，不替代结构化枚举）
# ===========================================================================
def test_ac8_conversion_barriers_is_explanation_layer(pu):
    fv = pu["product_fact_vector"]
    barriers = fv.get("conversion_barriers")
    if barriers is not None:
        assert isinstance(barriers, list) and all(isinstance(x, str) for x in barriers), \
            "conversion_barriers 必须为 list[str] 可读文本"
    # 结构化六维枚举仍独立存在（barriers 不是枚举载体）
    for dim in ("cognition_attribute", "frequency_attribute", "trust_attribute",
                "price_attribute", "endorsement_attribute", "channel_risk_attribute"):
        assert dim in fv, f"结构化枚举 {dim} 缺失，疑被 barriers 替代"


# ===========================================================================
# AC9（断言生效 · Crash Early 干净断言异常，非 KeyError/TypeError）
# ===========================================================================
def _expect_clean_assertion(callable_fn):
    """断言抛出的是干净的 AssertionError（SchemaAssertionError），而非脏 KeyError/TypeError。"""
    try:
        callable_fn()
    except (KeyError, TypeError) as e:  # 脏异常 → Fail
        raise AssertionError(f"抛出脏异常 {type(e).__name__}: {e!r}（应为干净断言异常）")
    except AssertionError:
        return  # 干净断言异常（SchemaAssertionError 是 AssertionError 子类）
    raise AssertionError("未抛出任何异常（违规未被拦截）")


def test_ac9_inject_expected_hec_blocked(pu):
    bad = copy.deepcopy(pu)
    bad["product_hec_legacy_alias"] = None  # 占位避免空操作
    del bad["product_hec_legacy_alias"]
    bad["expected_hec"] = bad["product_hec"]  # 注入禁出键
    _expect_clean_assertion(lambda: assert_contract_product_understanding(bad))


def test_ac9_channel_risk_no_risk_blocked(pu):
    bad = copy.deepcopy(pu)
    bad["product_fact_vector"]["channel_risk_attribute"] = "no_risk"
    _expect_clean_assertion(lambda: assert_contract_product_understanding(bad))


def test_ac9_segment_order_shuffled_blocked(pu):
    bad = copy.deepcopy(pu)
    # 打乱段顺序：把 evidence 提到最前
    reordered = {"evidence": bad["evidence"]}
    for k in EXPECTED_PU_ORDER:
        if k != "evidence":
            reordered[k] = bad[k]
    _expect_clean_assertion(lambda: assert_contract_product_understanding(reordered))


def test_ac9_delete_module3_blocked(pu):
    bad = copy.deepcopy(pu)
    del bad["module3"]
    _expect_clean_assertion(lambda: assert_contract_product_understanding(bad))


def test_ac9_full_response_assertion_blocks_violation(contract_resp):
    """完整契约响应注入违规（expected_hec）→ 顶层 assert_frontend_contract_response 干净拦截。"""
    bad = copy.deepcopy(contract_resp)
    bad["product_understanding"]["expected_hec"] = ["x"]
    _expect_clean_assertion(lambda: assert_frontend_contract_response(bad))


def test_ac9_valid_response_passes(contract_resp):
    """真实产物本身必须通过顶层断言（确认拦截器不是误杀一切）。"""
    assert_frontend_contract_response(copy.deepcopy(contract_resp))


# ===========================================================================
# AC10（第一批不回退）
# ===========================================================================
def test_ac10_batch1_no_regression(pu):
    profile = pu["module3"]["persuasion_requirement_profile"]
    assert profile and profile.get("persuasion_requirements"), "profile 回退为空"
    pta = pu["module3"]["product_target_audience"]
    rc = pta.get("reasoning_chain")
    # reasoning_chain 仍为四段，且含 persuasion_profile_to_audience
    if isinstance(rc, dict):
        assert len(rc) == 4, f"reasoning_chain 段数={len(rc)}（期望 4）：{list(rc.keys())}"
        assert "persuasion_profile_to_audience" in rc, "reasoning_chain 缺 persuasion_profile_to_audience"
    elif isinstance(rc, list):
        assert len(rc) == 4, f"reasoning_chain 段数={len(rc)}（期望 4）"
        joined = json.dumps(rc, ensure_ascii=False)
        assert "persuasion_profile_to_audience" in joined, "reasoning_chain 缺 persuasion_profile_to_audience"
    else:
        raise AssertionError(f"reasoning_chain 类型非法：{type(rc).__name__}")


# ===========================================================================
# AC12（artifacts 回显说明：brand_tier/relative_price_level 仅在请求快照）
# ===========================================================================
def test_ac12_legacy_fields_only_in_request_payload_snapshot(contract_resp, pu):
    # product_understanding 子树不得出现这两个字段
    assert _scan_keys(pu, {"brand_tier", "relative_price_level"}) == []
    assert _scan_keys(contract_resp["product_understanding"], {"brand_tier", "relative_price_level"}) == []

    # 仅允许出现在 artifacts.request_payload 回显快照内
    art = contract_resp["artifacts"]["request_payload"]
    bt = _scan_keys(art, {"brand_tier"}, path="artifacts.request_payload")
    rpl = _scan_keys(art, {"relative_price_level"}, path="artifacts.request_payload")
    assert len(bt) == 1, f"brand_tier 命中路径={bt}"
    assert len(rpl) == 1, f"relative_price_level 命中路径={rpl}"
