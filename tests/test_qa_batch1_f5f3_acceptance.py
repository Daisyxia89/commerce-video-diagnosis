"""独立 QA 正式验收：第一批 F5+F3 改造（/JG 仲裁节点编写）。

独立性声明
==========
- 本文件由独立 QA 节点依据《逐条验收标准 AC1–AC10》**从零编写**，
  不复用、不导入研发自写/自改的 ``tests/test_product_target_audience.py`` 的任何断言。
- 验收对象是**真实引擎产出**：所有断言均针对
  ``ProductDiagnosisEngine().diagnose(payload)`` 的真实输出，不 mock、不打桩。
- 独立构造两个不同品类 payload（驱蚊液 / 牙刷）增强覆盖。
- 若实现与验收标准冲突，如实 Fail，不软化断言、不改业务代码。

运行：``python -m pytest tests/test_qa_batch1_f5f3_acceptance.py -q``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# core_skill 在仓库根目录之上一级（与 run_*_full_diagnosis.py 一致的引入方式）。
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL,
    DiagnosticInput,
    ProductDiagnosisEngine,
)
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import (  # noqa: E402
    PersuasionRequirementEngine,
)

# --------------------------------------------------------------------------- #
# 独立构造的真实 payload（两个不同品类）。字段参考 run_runben_full_diagnosis.py 第 1-62 行
# 的构造方式，但内容由 QA 独立编写/复用，断言完全自写。
# --------------------------------------------------------------------------- #
RUNBEN_PAYLOAD = DiagnosticInput(
    leaf_category="宝宝防蚊水",
    shop_name="润本官方旗舰店",
    second_level_category="驱蚊用品",
    third_level_category="宝宝防蚊水",
    brand_name="润本",
    product_name="【A级驱蚊力】润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水防蚊叮蚊怕花露水",
    price="24.9",
    core_selling_point="派卡瑞丁A级驱蚊力，长效防蚊驱虫，温和无刺激",
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="婴幼儿/儿童/家庭日常户外人群",
    differentiator="",
    bridge_comparison_object="同类旧方案",
    bridge_comparison_object_evidence_type="jtbd_inferred",
    bridge_difference_domain="functional",
    bridge_difference_type="风险降低",
    bridge_source_evidence=[
        "派卡瑞丁15%/20% A级驱蚊力，防蚊8小时、耐汗保护7h",
        "7%款驱蚊酯日常居家温和不刺激，第三方检测0刺激、减少刺激",
        "干扰蚊子嗅觉识别，皮肤表面形成气味屏障，避险闻不到咬不着，更安心",
        "广告审查号：粤农药广审（视）01260018号；第三方检测报告 2200938-1 / SHG211647 / ET2025-230，安全可溯",
    ],
    bridge_evidence_source="商品信息",
    product_id="qa_runben_repellent_24p9",
    engine_node={"relative_price_level": "高水位"},
)

# 第二品类：牙刷（参考 run_kameier_full_diagnosis.py 的真实输入，独立复述）。
KAMEIER_PAYLOAD = DiagnosticInput(
    leaf_category="牙刷",
    shop_name="秉晟精品严选店",
    second_level_category="口腔护理",
    third_level_category="牙刷",
    brand_name="KAMEIER/卡玫尔",
    product_name="KAMEIER卡玫尔 4支峰型凸面中硬毛牙刷螺旋深层清洁护龈家用成人适用高档",
    price="19.9",
    core_selling_point=(
        "山峰弧面牙刷，物理磨尖 + 螺旋刷丝\n"
        "125°峰形峰尖裁切，双效净护2合1\n"
        "升级方孔菱形超密植毛\n"
        "清洁牙黄/烟黄/茶渍，不易炸毛\n"
        "螺旋刷丝温和护龈\n"
        "4支装家庭装"
    ),
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="希望深层清洁牙渍/烟茶渍并兼顾护龈不勒嘴的成人",
    differentiator=(
        "125°峰尖裁切山峰弧面 + 22孔菱形超密植毛 + 螺旋刷丝护龈，"
        "对比传统平面牙刷实现双效净护、贴合牙窝、深入缝隙不漏刷。"
    ),
    bridge_comparison_object="同类旧方案",
    bridge_comparison_object_evidence_type="null",
    bridge_difference_domain="functional",
    bridge_difference_type="自身卖点陈述",
    bridge_source_evidence=[
        "125°峰尖裁切山峰弧面（商品图）",
        "22孔大方孔菱形超密植毛 + 高矮错层（商品图）",
        "物理磨尖 + 螺旋刷丝（商品图）",
        "4支家庭装（商品图）",
    ],
    bridge_evidence_source="商品信息",
    product_id="qa_kameier_brush_4pack",
    engine_node={"relative_price_level": "低水位"},
)

PROFILE_REQUIRED_KEYS = {
    "requirement_id",
    "requirement_name",
    "priority",
    "decision_gap",
    "source",
    "related_decision_criteria",
    "required_evidence_requirements",
    "risk_points",
}

# requirement_id 形如 expose_current_pain，文本里以 「name(id)」 形式出现。
_REQ_ID_IN_TEXT = re.compile(r"\(([a-z][a-z_]+)\)")


@pytest.fixture(scope="module")
def runben_diag():
    return ProductDiagnosisEngine().diagnose(RUNBEN_PAYLOAD)


@pytest.fixture(scope="module")
def kameier_diag():
    return ProductDiagnosisEngine().diagnose(KAMEIER_PAYLOAD)


def _profile_dict(diag):
    out = diag.dict(exclude_none=True)
    return out["persuasion_requirement_profile"], out


def _req_id_set(profile: dict) -> set[str]:
    return {r["requirement_id"] for r in profile.get("persuasion_requirements", [])}


# ============================ AC1（F5-顺序）============================ #
@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac1_profile_before_audience_and_referenced(fixture_name, request):
    """profile 非空，且 audience 第四段引用 profile 中真实 requirement → 间接证明
    profile 在 audience 之前生成且 audience 强依赖 profile。"""
    diag = request.getfixturevalue(fixture_name)
    profile, out = _profile_dict(diag)
    assert profile and profile.get("persuasion_requirements"), "profile 为空"
    rc = out["product_target_audience"]["reasoning_chain"]
    text = rc["persuasion_profile_to_audience"]
    ids_in_text = set(_REQ_ID_IN_TEXT.findall(text))
    assert ids_in_text, f"第四段未引用任何 requirement_id：{text!r}"
    assert ids_in_text <= _req_id_set(profile), (
        f"第四段引用了 profile 之外的 requirement_id：{ids_in_text - _req_id_set(profile)}"
    )


# ============================ AC2（F5-引擎内聚）============================ #
@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac2_profile_is_formal_engine_output_field(fixture_name, request):
    diag = request.getfixturevalue(fixture_name)
    # 作为引擎输出对象的正式字段直接可读，非外部后挂。
    assert diag.persuasion_requirement_profile is not None
    profile = diag.persuasion_requirement_profile.dict()
    assert profile.get("persuasion_requirements"), "profile.persuasion_requirements 为空"


# ============================ AC3（保护条件1）============================ #
def test_ac3_engine_inited_once_in_constructor():
    eng = ProductDiagnosisEngine()
    assert hasattr(eng, "persuasion_engine"), "persuasion_engine 未在 __init__ 初始化"
    assert isinstance(eng.persuasion_engine, PersuasionRequirementEngine)
    # 同一引擎实例多次 diagnose 应复用同一个 persuasion_engine（非每次新建）。
    before = id(eng.persuasion_engine)
    eng.diagnose(RUNBEN_PAYLOAD)
    eng.diagnose(KAMEIER_PAYLOAD)
    assert id(eng.persuasion_engine) == before, "diagnose 期间 persuasion_engine 被重建"


def test_ac3_dictionary_load_failure_crash_early(tmp_path):
    """字典缺失/加载失败必须在构造期 Crash Early（无 try/except 吞异常）。"""
    empty_dir = tmp_path / "no_dict"
    empty_dir.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        PersuasionRequirementEngine(dictionary_dir=empty_dir)


# ============================ AC5（保护条件2）============================ #
@pytest.mark.parametrize("bad_profile", [None, {}, {"persuasion_requirements": []}])
def test_ac5_derive_audience_crash_early_on_empty_profile(bad_profile):
    """_derive_product_target_audience 在 profile 空/缺 persuasion_requirements 时 raise。
    profile 校验位于函数开头，先于其它属性推导，故可安全传 None 占位其它入参。"""
    eng = ProductDiagnosisEngine()
    with pytest.raises(ValueError):
        eng._derive_product_target_audience(
            None, None, None, persuasion_requirement_profile=bad_profile
        )


# ============================ AC6（保护条件4 + F3）============================ #
@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac6_referenced_requirement_ids_exist_in_profile(fixture_name, request):
    diag = request.getfixturevalue(fixture_name)
    profile, out = _profile_dict(diag)
    text = out["product_target_audience"]["reasoning_chain"]["persuasion_profile_to_audience"]
    ids_in_text = set(_REQ_ID_IN_TEXT.findall(text))
    real_ids = _req_id_set(profile)
    assert ids_in_text, "第四段未抽取到 requirement_id（疑似自由文本/写死占位）"
    missing = ids_in_text - real_ids
    assert not missing, f"第四段引用了 profile 不存在的 requirement_id：{missing}"


# ============================ AC7（保护条件5：八大人群未被改写）============================ #
@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac7_eight_audience_algorithm_intact(fixture_name, request):
    diag = request.getfixturevalue(fixture_name)
    out = diag.dict(exclude_none=True)
    pta = out["product_target_audience"]
    rc = pta["reasoning_chain"]
    for seg in ("task_to_role", "role_category_to_age_gender", "brand_price_to_consumption_power"):
        assert rc.get(seg) and rc[seg].strip(), f"reasoning_chain.{seg} 缺失或为空"
    assert pta.get("primary_audiences"), "primary_audiences 为空"


# ============================ AC8（content_goal 收尾）============================ #
def test_ac8_content_goal_constant_value():
    assert DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL == "purchase"


@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac8_content_goal_propagated(fixture_name, request):
    diag = request.getfixturevalue(fixture_name)
    out = diag.dict(exclude_none=True)
    assert out["metadata"]["content_goal"] == "purchase"
    assert out["persuasion_requirement_profile"]["content_goal"] == "purchase"


# ============================ AC9（profile 完整性）============================ #
@pytest.mark.parametrize("fixture_name", ["runben_diag", "kameier_diag"])
def test_ac9_each_requirement_has_required_keys(fixture_name, request):
    diag = request.getfixturevalue(fixture_name)
    profile, _ = _profile_dict(diag)
    reqs = profile["persuasion_requirements"]
    assert reqs, "persuasion_requirements 为空"
    for i, req in enumerate(reqs):
        missing = PROFILE_REQUIRED_KEYS - set(req.keys())
        assert not missing, f"第 {i} 条 requirement 缺少字段键：{missing}"
