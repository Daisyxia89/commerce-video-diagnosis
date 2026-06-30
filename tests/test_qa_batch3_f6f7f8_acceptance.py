"""独立 QA 第三批 F6+F7+F8 正式验收（/JG 节点自编，不复用研发自写用例）。

验收对象 = 真实产出：
- 直接调用 contract 装配层 ``build_product_understanding(...)`` 对真实 product_diagnosis
  （outputs/runben_diagnosis/runben_full_diagnosis.json，已提交真实诊断产物）做断言；
- 同时加载 ``run_runben_video_diagnosis.py`` 真实生成路径产出的
  outputs/runben_diagnosis/runben_contract_response.json 做整树扫描（diagnosis 业务子树）。

完全依据任务下发的 AC1–AC11 自编断言；definition 真实性用后端字典加载器
``lookup_hec`` 独立查表交叉验证，并验证非法 code Crash Early。
不依赖、不导入 tests/test_frontend_contract_acceptance.py（研发自写用例）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from commerce_video_diagnosis.understanding.assembly.response_assembler import (
    build_product_understanding,
)
from commerce_video_diagnosis.understanding.assembly.hec_dictionary import (
    lookup_hec,
    HECDictionaryError,
)

ROOT = Path(__file__).resolve().parents[1]
FULL_DIAGNOSIS_PATH = ROOT / "outputs" / "runben_diagnosis" / "runben_full_diagnosis.json"
CONTRACT_RESPONSE_PATH = ROOT / "outputs" / "runben_diagnosis" / "runben_contract_response.json"

# AC11 禁出字段（业务子树，不扫 artifacts）
FORBIDDEN_KEYS = {
    "expected_hec",
    "conversion_resistance",
    "supporting_requirements",
    "trust_barrier",
    "brand_tier",
    "price_barrier",
    "financial_risk",
    "relative_price_level",
    "endorsement",      # 旧英文裸字段（区别于新 endorsement_attribute）
    "channel_risk",     # 旧工程裸字段（区别于新 channel_risk_attribute）
    "target_people",
}


# --------------------------------------------------------------------------- #
# fixtures（真实产出）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def product_diagnosis() -> dict:
    assert FULL_DIAGNOSIS_PATH.exists(), f"真实商品诊断产物缺失：{FULL_DIAGNOSIS_PATH}"
    return json.loads(FULL_DIAGNOSIS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def product_understanding(product_diagnosis) -> dict:
    """直接调用 contract 装配层对真实 product_diagnosis dict 装配。"""
    return build_product_understanding(product_diagnosis)


@pytest.fixture(scope="module")
def contract_response() -> dict:
    assert CONTRACT_RESPONSE_PATH.exists(), (
        f"契约响应缺失：{CONTRACT_RESPONSE_PATH}（请先跑 run_runben_video_diagnosis.py）"
    )
    return json.loads(CONTRACT_RESPONSE_PATH.read_text(encoding="utf-8"))


def _scan_forbidden(node, path: str) -> list[str]:
    """递归扫描 dict key，命中 FORBIDDEN_KEYS 即记录路径。"""
    hits: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in FORBIDDEN_KEYS:
                hits.append(f"{path}.{k}")
            hits.extend(_scan_forbidden(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_scan_forbidden(v, f"{path}[{i}]"))
    return hits


# --------------------------------------------------------------------------- #
# AC1 — F6 结构：candidate_set.derived_from 含 requirement_ids / audience_groups 两键，均非空
# --------------------------------------------------------------------------- #
def test_ac1_derived_from_structure(product_understanding):
    cs = product_understanding["candidate_set"]
    assert "derived_from" in cs, "candidate_set 缺 derived_from"
    df = cs["derived_from"]
    assert set(df.keys()) >= {"requirement_ids", "audience_groups"}, df.keys()
    assert isinstance(df["requirement_ids"], list) and df["requirement_ids"], "requirement_ids 空"
    assert isinstance(df["audience_groups"], list) and df["audience_groups"], "audience_groups 空"


# --------------------------------------------------------------------------- #
# AC2 — F6 可追溯-需求：requirement_ids ⊆ profile.persuasion_requirements[].requirement_id
# --------------------------------------------------------------------------- #
def test_ac2_requirement_ids_subset(product_understanding):
    df = product_understanding["candidate_set"]["derived_from"]
    profile = product_understanding["module3"]["persuasion_requirement_profile"]
    real_ids = {
        r.get("requirement_id")
        for r in profile["persuasion_requirements"]
        if isinstance(r, dict) and r.get("requirement_id")
    }
    extraneous = set(df["requirement_ids"]) - real_ids
    assert not extraneous, f"derived_from.requirement_ids 越界（非 profile 真实 id）：{extraneous}"


# --------------------------------------------------------------------------- #
# AC3 — F6 可追溯-人群：audience_groups ⊆ product_target_audience.primary_audiences[].audience_group
# --------------------------------------------------------------------------- #
def test_ac3_audience_groups_subset(product_understanding):
    df = product_understanding["candidate_set"]["derived_from"]
    pta = product_understanding["module3"]["product_target_audience"]
    real_groups = {
        a.get("audience_group")
        for a in pta["primary_audiences"]
        if isinstance(a, dict) and a.get("audience_group")
    }
    extraneous = set(df["audience_groups"]) - real_groups
    assert not extraneous, f"derived_from.audience_groups 越界（非 primary_audiences）：{extraneous}"


# --------------------------------------------------------------------------- #
# AC4 — F6 选取规则：required==true 或 priority=="high"；无命中才回退全部
# --------------------------------------------------------------------------- #
def test_ac4_selection_rule(product_understanding):
    df = product_understanding["candidate_set"]["derived_from"]
    profile = product_understanding["module3"]["persuasion_requirement_profile"]
    reqs = [r for r in profile["persuasion_requirements"] if isinstance(r, dict)]
    expected = [
        r["requirement_id"]
        for r in reqs
        if r.get("requirement_id") and (bool(r.get("required")) or r.get("priority") == "high")
    ]
    if not expected:  # 回退全部
        expected = [r["requirement_id"] for r in reqs if r.get("requirement_id")]
    assert df["requirement_ids"] == expected, (
        f"选取规则不符\n实得={df['requirement_ids']}\n应得={expected}"
    )
    # 被排除条目核验：既非 required 也非 high 的应被排除
    excluded = [
        r["requirement_id"]
        for r in reqs
        if r.get("requirement_id") and not (bool(r.get("required")) or r.get("priority") == "high")
    ]
    for rid in excluded:
        assert rid not in df["requirement_ids"], f"应排除条目仍被纳入：{rid}"


# --------------------------------------------------------------------------- #
# AC5 — F7 三元组：每项 hook/effect/cta 均为对象且含 code/name/definition，三者非空字符串
# --------------------------------------------------------------------------- #
def test_ac5_hec_triple_shape(product_understanding):
    hec = product_understanding["product_hec"]
    assert isinstance(hec, list) and hec, "product_hec 为空"
    for i, item in enumerate(hec):
        for dim in ("hook", "effect", "cta"):
            o = item[dim]
            assert isinstance(o, dict), f"product_hec[{i}].{dim} 非对象"
            assert set(o.keys()) >= {"code", "name", "definition"}, f"[{i}].{dim} 缺键 {o.keys()}"
            for key in ("code", "name", "definition"):
                assert isinstance(o[key], str) and o[key].strip(), f"[{i}].{dim}.{key} 空/非串"


# --------------------------------------------------------------------------- #
# AC6 — F7 definition 真实性：与后端字典 lookup 一致；非法 code Crash Early
# --------------------------------------------------------------------------- #
def test_ac6_definition_from_dictionary(product_understanding):
    for i, item in enumerate(product_understanding["product_hec"]):
        for dim in ("hook", "effect", "cta"):
            o = item[dim]
            entry = lookup_hec(o["code"])  # 独立查表
            assert o["definition"] == entry["definition"], (
                f"[{i}].{dim} definition 与字典不一致：contract={o['definition']!r} dict={entry['definition']!r}"
            )
            # name 若由 label 供给可不等于字典 name，但必须命中字典且非编造定义
            assert o["definition"].strip(), f"[{i}].{dim}.definition 占位/空"


@pytest.mark.parametrize("bad_code", ["H9", "E8", "C9", "", "  ", "XX"])
def test_ac6_illegal_code_crash_early(bad_code):
    with pytest.raises(HECDictionaryError):
        lookup_hec(bad_code)


# --------------------------------------------------------------------------- #
# AC7 — F7 无裸 tag / 无 expected_hec
# --------------------------------------------------------------------------- #
def test_ac7_no_bare_tag_no_expected_hec(product_understanding):
    for i, item in enumerate(product_understanding["product_hec"]):
        for bad in ("hook_tag", "effect_tag", "cta_tag"):
            assert bad not in item, f"product_hec[{i}] 出现裸 {bad}"
    # product_understanding 子树不出现 expected_hec
    hits = _scan_forbidden(product_understanding, "product_understanding")
    expected_hits = [h for h in hits if h.endswith(".expected_hec")]
    assert not expected_hits, f"product_understanding 仍含 expected_hec：{expected_hits}"


# --------------------------------------------------------------------------- #
# AC8 — F7 前端纯消费：name/definition 由后端供给（contract 内即可读）
# --------------------------------------------------------------------------- #
def test_ac8_frontend_pure_consumption(product_understanding):
    for i, item in enumerate(product_understanding["product_hec"]):
        for dim in ("hook", "effect", "cta"):
            o = item[dim]
            # contract 内即含 name + definition，前端无需任何字典即可消费
            assert o.get("name") and o.get("definition"), f"[{i}].{dim} 未由后端供给 name/definition"


# --------------------------------------------------------------------------- #
# AC9 — F8 回归：reasoning_chain 四段齐全；第四段引用 requirement_id ∈ profile
# --------------------------------------------------------------------------- #
def test_ac9_reasoning_chain(product_understanding):
    pta = product_understanding["module3"]["product_target_audience"]
    rc = pta["reasoning_chain"]
    for seg in (
        "task_to_role",
        "role_category_to_age_gender",
        "brand_price_to_consumption_power",
        "persuasion_profile_to_audience",
    ):
        assert isinstance(rc.get(seg), str) and rc[seg].strip(), f"reasoning_chain.{seg} 缺/空"
    profile = product_understanding["module3"]["persuasion_requirement_profile"]
    real_ids = [
        r["requirement_id"]
        for r in profile["persuasion_requirements"]
        if isinstance(r, dict) and r.get("requirement_id")
    ]
    seg4 = rc["persuasion_profile_to_audience"]
    referenced = [rid for rid in real_ids if f"({rid})" in seg4]
    assert referenced, f"第四段未引用任何真实 requirement_id；real_ids={real_ids}\nseg4={seg4}"


# --------------------------------------------------------------------------- #
# AC10 — 不回退：6 段顺序固定；product_fact_vector 在；module3 两对象非空
# --------------------------------------------------------------------------- #
def test_ac10_no_regression_structure(product_understanding):
    assert list(product_understanding.keys()) == [
        "basic_info",
        "product_fact_vector",
        "module3",
        "candidate_set",
        "product_hec",
        "evidence",
    ], f"6 段顺序错误：{list(product_understanding.keys())}"
    assert product_understanding["product_fact_vector"], "product_fact_vector 缺失/空"
    m3 = product_understanding["module3"]
    assert m3["persuasion_requirement_profile"], "module3.persuasion_requirement_profile 空"
    assert m3["product_target_audience"], "module3.product_target_audience 空"


# --------------------------------------------------------------------------- #
# AC11 — 禁出字段（product_understanding + diagnosis 业务子树；不扫 artifacts）
# --------------------------------------------------------------------------- #
def test_ac11_no_forbidden_keys_product_understanding(product_understanding):
    hits = _scan_forbidden(product_understanding, "product_understanding")
    assert not hits, f"product_understanding 出现禁出字段：{hits}"


def test_ac11_no_forbidden_keys_diagnosis_subtree(contract_response):
    # 仅扫 product_understanding 与 diagnosis 业务子树，artifacts 按裁定豁免
    pu_hits = _scan_forbidden(contract_response["product_understanding"], "product_understanding")
    diag_hits = _scan_forbidden(contract_response["diagnosis"], "diagnosis")
    assert not pu_hits, f"product_understanding 出现禁出字段：{pu_hits}"
    assert not diag_hits, f"diagnosis 出现禁出字段：{diag_hits}"
