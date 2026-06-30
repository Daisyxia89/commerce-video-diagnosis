"""第四批 F9 · 商品理解 E2E 准入闸（contract gate）独立 QA 测试（/JG 节点自编）。

本文件把前三批分散的断言收敛成一套「商品理解 E2E 准入闸」，用例 T1–T14，覆盖
用户下发的 8 项覆盖要求。**不改任何业务逻辑**，仅对真实产出做断言。

环境口径（离线，不依赖在线 LLM Judge）：
- 真实 product_diagnosis 输入 = 已提交的 ``outputs/runben_diagnosis/runben_full_diagnosis.json``；
  直接调用 ``understanding/assembly/response_assembler.build_product_understanding(...)``
  对真实 product_diagnosis dict 装配并断言。
- 需要视频侧的用例（T11）走 ``run_runben_video_diagnosis.py`` 的 ``build_payload()`` 路径，
  跑离线确定性引擎 ``VideoDiagnosisEngine``；并交叉读取已提交的
  ``outputs/runben_diagnosis/runben_contract_response.json``。
- ``run_runben_full_diagnosis.py`` 因在线 LLM Judge 依赖会失败，不纳入本 gate。

断言以 PRD/契约为准，独立编写；不复用研发自写测试（tests/test_frontend_contract_acceptance.py
等）的断言实现，也不导入第一/二/三批 QA 用例文件。
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commerce_video_diagnosis.understanding.assembly.response_assembler import (  # noqa: E402
    ContractAssemblyError,
    build_product_fact_vector,
    build_product_understanding,
)
from commerce_video_diagnosis.understanding.assembly.hec_dictionary import (  # noqa: E402
    HECDictionaryError,
    lookup_hec,
)

FULL_DIAGNOSIS_PATH = ROOT / "outputs" / "runben_diagnosis" / "runben_full_diagnosis.json"
CONTRACT_RESPONSE_PATH = ROOT / "outputs" / "runben_diagnosis" / "runben_contract_response.json"

# --------------------------------------------------------------------------- #
# PRD 枚举闭集（T2；字段名/枚举与 build_product_fact_vector 实现一致核对，按真实 PRD 闭集）
# --------------------------------------------------------------------------- #
# 6 维：认知/频次/信任/价格/背书/渠道风险；背书、渠道风险允许 null。
FACT_VECTOR_ENUMS: dict[str, set] = {
    "cognition_attribute": {"蓝海", "红海-核心", "红海-破圈"},          # 认知
    "frequency_attribute": {"快消", "耐消"},                            # 频次
    "trust_attribute": {"大牌", "白牌"},                               # 信任
    "price_attribute": {"高", "低"},                                  # 价格
    "endorsement_attribute": {"有背书", None},                        # 背书（允许 null）
    "channel_risk_attribute": {"有风险", None},                       # 渠道风险（允许 null）
}

# T3 禁出旧字段（业务子树，递归；不扫 artifacts）。
FORBIDDEN_BARE_KEYS = {
    "expected_hec",
    "conversion_resistance",
    "supporting_requirements",
    "trust_barrier",
    "brand_tier",
    "price_barrier",
    "financial_risk",
    "relative_price_level",
    "endorsement",       # 旧英文裸字段（新字段为 endorsement_attribute）
    "channel_risk",      # 旧工程裸字段（新字段为 channel_risk_attribute）
    "target_people",     # 顶层人群线索旧字段（已迁入 basic_info.audience_hint）
}
# 旧 endorsement 英文枚举值 / 旧工程 channel_risk 枚举值（值级禁出，仅校验对应维度取值）
LEGACY_ENDORSEMENT_VALUES = {"unknown", "no_endorsement", "has_endorsement"}
LEGACY_CHANNEL_RISK_VALUES = {"no_risk", "low", "medium", "high"}

# product_understanding 6 段固定键集合与顺序（T1）
EXPECTED_PU_KEYS = [
    "basic_info",
    "product_fact_vector",
    "module3",
    "candidate_set",
    "product_hec",
    "evidence",
]


# --------------------------------------------------------------------------- #
# fixtures（真实产出，自包含、可重复跑）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def product_diagnosis() -> dict:
    assert FULL_DIAGNOSIS_PATH.exists(), (
        f"真实商品诊断产物缺失：{FULL_DIAGNOSIS_PATH}（请确认已提交 runben_full_diagnosis.json）"
    )
    return json.loads(FULL_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def product_understanding(product_diagnosis) -> dict:
    """直接调用 contract 装配层对真实 product_diagnosis dict 装配（真实 contract response 路径）。"""
    return build_product_understanding(product_diagnosis)


@pytest.fixture(scope="module")
def contract_response() -> dict:
    assert CONTRACT_RESPONSE_PATH.exists(), (
        f"契约响应缺失：{CONTRACT_RESPONSE_PATH}（请先离线跑 run_runben_video_diagnosis.py）"
    )
    return json.loads(CONTRACT_RESPONSE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 工具：递归扫描禁出 key / 抽取 module3 对象
# --------------------------------------------------------------------------- #
def _scan_keys(node, path: str, target_keys: set) -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in target_keys:
                hits.append(f"{path}.{k}")
            hits.extend(_scan_keys(v, f"{path}.{k}", target_keys))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_scan_keys(v, f"{path}[{i}]", target_keys))
    return hits


def _profile_of(pu: dict) -> dict:
    return pu["module3"]["persuasion_requirement_profile"]


def _pta_of(pu: dict) -> dict:
    return pu["module3"]["product_target_audience"]


def _real_requirement_ids(pu: dict) -> set:
    return {
        r.get("requirement_id")
        for r in _profile_of(pu).get("persuasion_requirements") or []
        if isinstance(r, dict) and r.get("requirement_id")
    }


# =========================================================================== #
# T1 — 商品理解 6 段键集合 === {basic_info, product_fact_vector, module3,
#      candidate_set, product_hec, evidence} 且顺序稳定。
# =========================================================================== #
def test_t1_six_sections_keys_and_order(product_understanding):
    keys = list(product_understanding.keys())
    assert keys == EXPECTED_PU_KEYS, f"6 段键集合/顺序不稳定：实得={keys} 应得={EXPECTED_PU_KEYS}"
    assert set(keys) == set(EXPECTED_PU_KEYS), "6 段键集合不相等"


# =========================================================================== #
# T2 — product_fact_vector 含 6 维，每维取值 ∈ PRD 枚举闭集（背书/渠道风险允许 null）。
# =========================================================================== #
def test_t2_fact_vector_six_dims_closed_enum(product_understanding):
    fv = product_understanding["product_fact_vector"]
    # 6 维全部存在
    for dim in FACT_VECTOR_ENUMS:
        assert dim in fv, f"product_fact_vector 缺维度：{dim}"
    # 每维取值落在 PRD 闭集
    for dim, closed in FACT_VECTOR_ENUMS.items():
        assert fv[dim] in closed, f"product_fact_vector.{dim}={fv[dim]!r} 越界，闭集={closed}"
    # 必填四维非空（认知/频次/信任/价格不允许 null）
    for dim in ("cognition_attribute", "frequency_attribute", "trust_attribute", "price_attribute"):
        assert fv[dim] is not None and str(fv[dim]).strip(), f"必填维度 {dim} 为空"


def test_t2_fact_vector_matches_direct_builder(product_diagnosis, product_understanding):
    """6 维取值与 build_product_fact_vector 独立调用结果一致（口径自洽）。"""
    direct = build_product_fact_vector(product_diagnosis)
    fv = product_understanding["product_fact_vector"]
    for dim in FACT_VECTOR_ENUMS:
        assert fv[dim] == direct[dim], f"{dim} 与独立装配不一致：{fv[dim]!r} vs {direct[dim]!r}"


# =========================================================================== #
# T3 — 旧字段禁出（仅扫 product_understanding 与 diagnosis 业务子树，递归；不扫 artifacts）。
# =========================================================================== #
def test_t3_no_forbidden_keys_in_product_understanding(product_understanding):
    hits = _scan_keys(product_understanding, "product_understanding", FORBIDDEN_BARE_KEYS)
    assert not hits, f"product_understanding 出现禁出旧字段：{hits}"


def test_t3_no_forbidden_keys_in_diagnosis_subtree(contract_response):
    pu_hits = _scan_keys(
        contract_response["product_understanding"], "product_understanding", FORBIDDEN_BARE_KEYS
    )
    diag_hits = _scan_keys(contract_response["diagnosis"], "diagnosis", FORBIDDEN_BARE_KEYS)
    assert not pu_hits, f"product_understanding 出现禁出旧字段：{pu_hits}"
    assert not diag_hits, f"diagnosis 业务子树出现禁出旧字段：{diag_hits}"


def test_t3_endorsement_channel_risk_not_legacy_values(product_understanding):
    fv = product_understanding["product_fact_vector"]
    assert fv["endorsement_attribute"] not in LEGACY_ENDORSEMENT_VALUES, (
        f"endorsement_attribute 命中旧英文枚举：{fv['endorsement_attribute']!r}"
    )
    assert fv["channel_risk_attribute"] not in LEGACY_CHANNEL_RISK_VALUES, (
        f"channel_risk_attribute 命中旧工程枚举：{fv['channel_risk_attribute']!r}"
    )


# =========================================================================== #
# T4 — candidate_set.derived_from.requirement_ids ⊆ profile.persuasion_requirements[].requirement_id
# =========================================================================== #
def test_t4_requirement_ids_subset_of_profile(product_understanding):
    df = product_understanding["candidate_set"]["derived_from"]
    rids = df["requirement_ids"]
    assert isinstance(rids, list) and rids, "derived_from.requirement_ids 为空"
    real = _real_requirement_ids(product_understanding)
    extraneous = set(rids) - real
    assert not extraneous, f"requirement_ids 越界（非 profile 真实 id）：{extraneous}"


# =========================================================================== #
# T5 — candidate_set.derived_from.audience_groups ⊆ product_target_audience.primary_audiences[].audience_group
# =========================================================================== #
def test_t5_audience_groups_subset_of_primary_audiences(product_understanding):
    df = product_understanding["candidate_set"]["derived_from"]
    groups = df["audience_groups"]
    assert isinstance(groups, list) and groups, "derived_from.audience_groups 为空"
    real = {
        a.get("audience_group")
        for a in _pta_of(product_understanding).get("primary_audiences") or []
        if isinstance(a, dict) and a.get("audience_group")
    }
    extraneous = set(groups) - real
    assert not extraneous, f"audience_groups 越界（非 primary_audiences）：{extraneous}"


# =========================================================================== #
# T6 — product_hec 每项 hook/effect/cta 均为 {code,name,definition} 三元组，三键非空字符串。
# =========================================================================== #
def test_t6_product_hec_triple_shape(product_understanding):
    hec = product_understanding["product_hec"]
    assert isinstance(hec, list) and hec, "product_hec 为空"
    for i, item in enumerate(hec):
        for dim in ("hook", "effect", "cta"):
            assert dim in item, f"product_hec[{i}] 缺维度 {dim}"
            triple = item[dim]
            assert isinstance(triple, dict), f"product_hec[{i}].{dim} 非对象"
            assert set(triple.keys()) >= {"code", "name", "definition"}, (
                f"product_hec[{i}].{dim} 缺三元组键：{triple.keys()}"
            )
            for key in ("code", "name", "definition"):
                v = triple[key]
                assert isinstance(v, str) and v.strip(), f"product_hec[{i}].{dim}.{key} 空/非字符串"


# =========================================================================== #
# T7 — product_hec 每个 code 命中后端 HEC 字典，name/definition 与 lookup_hec 交叉一致。
# =========================================================================== #
def test_t7_product_hec_code_crossverify_dictionary(product_understanding):
    for i, item in enumerate(product_understanding["product_hec"]):
        for dim in ("hook", "effect", "cta"):
            triple = item[dim]
            entry = lookup_hec(triple["code"])  # 独立查表，命中即字典存在
            assert triple["definition"] == entry["definition"], (
                f"product_hec[{i}].{dim} definition 与字典不一致："
                f"contract={triple['definition']!r} dict={entry['definition']!r}"
            )
            assert triple["name"] == entry["name"], (
                f"product_hec[{i}].{dim} name 与字典不一致："
                f"contract={triple['name']!r} dict={entry['name']!r}"
            )


# =========================================================================== #
# T8 — 非法 code（H9/E8/C9/空串/空格）调用 lookup_hec 抛 HECDictionaryError（Crash Early）。
# =========================================================================== #
@pytest.mark.parametrize("bad_code", ["H9", "E8", "C9", "", "   ", "H 1", "XX", None])
def test_t8_illegal_code_crash_early(bad_code):
    with pytest.raises(HECDictionaryError):
        lookup_hec(bad_code)


# =========================================================================== #
# T9 — reasoning_chain 四段齐全且非空。
# =========================================================================== #
def test_t9_reasoning_chain_four_segments(product_understanding):
    rc = _pta_of(product_understanding).get("reasoning_chain")
    assert isinstance(rc, dict), "product_target_audience.reasoning_chain 缺失或非对象"
    for seg in (
        "task_to_role",
        "role_category_to_age_gender",
        "brand_price_to_consumption_power",
        "persuasion_profile_to_audience",
    ):
        assert isinstance(rc.get(seg), str) and rc[seg].strip(), f"reasoning_chain.{seg} 缺/空"


# =========================================================================== #
# T10 — persuasion_profile_to_audience 中引用的 requirement_id ∈ profile（从文本抽取 id 校验子集）。
# =========================================================================== #
def test_t10_seg4_referenced_requirement_ids_subset(product_understanding):
    rc = _pta_of(product_understanding)["reasoning_chain"]
    seg4 = rc["persuasion_profile_to_audience"]
    # 从文本中抽取 (xxx) / （xxx） 形式的英文 requirement_id
    referenced = set(re.findall(r"[（(]\s*([a-z][a-z_]+)\s*[)）]", seg4))
    assert referenced, f"persuasion_profile_to_audience 未抽到任何 requirement_id 引用：{seg4!r}"
    real = _real_requirement_ids(product_understanding)
    extraneous = referenced - real
    assert not extraneous, f"第四段引用了非 profile 的 requirement_id：{extraneous}\nseg4={seg4!r}"


# =========================================================================== #
# T11 — video_diagnoser 只读 module3：视频诊断不生产/覆盖商品侧 product_target_audience /
#       persuasion_requirement_profile；video_target_audience 为独立字段。
# =========================================================================== #
def test_t11_video_diagnoser_readonly_on_product_objects():
    import run_runben_video_diagnosis as runner
    from commerce_video_diagnosis.understanding.engines.video_diagnoser import (
        VideoDiagnosisEngine,
    )

    payload = runner.build_payload()
    pd = payload["product_diagnosis"]
    # 诊断前快照商品侧两对象
    pta_before = copy.deepcopy(pd.get("product_target_audience"))
    profile_before = copy.deepcopy(pd.get("persuasion_requirement_profile"))
    assert pta_before, "前置：payload.product_diagnosis.product_target_audience 应非空"
    assert profile_before, "前置：payload.product_diagnosis.persuasion_requirement_profile 应非空"

    result = VideoDiagnosisEngine().diagnose(payload)
    body = result["video_persuasion_diagnosis_result"]

    # 1) 视频侧输出不生产商品侧对象（结构上根本不含这两个商品对象）
    assert "product_target_audience" not in body, "视频诊断输出越权产出 product_target_audience"
    assert "persuasion_requirement_profile" not in body, (
        "视频诊断输出越权产出 persuasion_requirement_profile"
    )
    # 2) 视频侧人群为独立字段，不复用商品对象名
    assert "video_target_audience" in body, "视频诊断未输出独立的 video_target_audience"

    # 3) 商品侧两对象未被改写/补算（输入不被污染）
    assert pd.get("product_target_audience") == pta_before, "视频诊断改写了商品侧 product_target_audience"
    assert pd.get("persuasion_requirement_profile") == profile_before, (
        "视频诊断改写了商品侧 persuasion_requirement_profile"
    )

    # 4) video_target_audience 与商品 product_target_audience 是相互独立的对象
    assert body["video_target_audience"] is not pta_before, "video_target_audience 与商品对象非独立"


def test_t11_contract_video_understanding_has_no_product_objects(contract_response):
    """前端契约的 video_understanding 子树同样不含商品侧两对象（视频侧不补算商品对象）。"""
    vu = contract_response["video_understanding"]
    hits = _scan_keys(
        vu, "video_understanding", {"product_target_audience", "persuasion_requirement_profile"}
    )
    assert not hits, f"video_understanding 子树越权包含商品侧对象：{hits}"


# =========================================================================== #
# T12 — basic_info.audience_hint 存在且非空；product_understanding 顶层无旧 target_people。
# =========================================================================== #
def test_t12_audience_hint_present_and_no_target_people(product_understanding):
    basic = product_understanding["basic_info"]
    assert "audience_hint" in basic, "basic_info 缺 audience_hint"
    hint = basic["audience_hint"]
    assert isinstance(hint, list) and hint and all(
        isinstance(x, str) and x.strip() for x in hint
    ), f"basic_info.audience_hint 为空或含空项：{hint!r}"
    assert "target_people" not in product_understanding, "product_understanding 顶层仍存在旧 target_people"


# =========================================================================== #
# T13 — module3 两对象：profile 含非空 persuasion_requirements；pta 含非空 primary_audiences。
# =========================================================================== #
def test_t13_module3_two_objects_non_empty(product_understanding):
    m3 = product_understanding["module3"]
    profile = m3.get("persuasion_requirement_profile")
    pta = m3.get("product_target_audience")
    assert isinstance(profile, dict) and profile, "module3.persuasion_requirement_profile 空"
    reqs = profile.get("persuasion_requirements")
    assert isinstance(reqs, list) and reqs, "persuasion_requirement_profile.persuasion_requirements 空"
    assert isinstance(pta, dict) and pta, "module3.product_target_audience 空"
    pas = pta.get("primary_audiences")
    assert isinstance(pas, list) and pas, "product_target_audience.primary_audiences 空"


# =========================================================================== #
# T14 — profile/audience 缺失或为空时 build_product_understanding Crash Early。
# =========================================================================== #
def _mutate(product_diagnosis: dict, **overrides) -> dict:
    pd = copy.deepcopy(product_diagnosis)
    for k, v in overrides.items():
        pd[k] = v
    return pd


def test_t14_missing_profile_crash_early(product_diagnosis):
    pd = _mutate(product_diagnosis, persuasion_requirement_profile={})
    with pytest.raises(ContractAssemblyError):
        build_product_understanding(pd)


def test_t14_empty_persuasion_requirements_crash_early(product_diagnosis):
    pd = _mutate(
        product_diagnosis, persuasion_requirement_profile={"persuasion_requirements": []}
    )
    with pytest.raises(ContractAssemblyError):
        build_product_understanding(pd)


def test_t14_missing_audience_crash_early(product_diagnosis):
    pd = _mutate(product_diagnosis, product_target_audience={})
    with pytest.raises(ContractAssemblyError):
        build_product_understanding(pd)
