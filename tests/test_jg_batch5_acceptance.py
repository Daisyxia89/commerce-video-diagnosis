"""/JG 独立验收：第五批 gift_context 通用送礼场景识别与双重角色拆分。

依据 PRD（``module3_prd_dev_checklist (3).md`` §10），由 /JG 节点独立出题，
**绝不复用** 研发自测文件 ``tests/test_gift_context_audience.py``。每条 AC 给
独立断言 + 复现命令 + 证据片段。失败即 Fail，禁止因实现妥协放松断言。

复现命令：
    cd commerce-video-diagnosis
    pytest -q tests/test_jg_batch5_acceptance.py

通用性测试：4 类非父亲节送礼信号 + 1 个负样本。
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# 仅向 sys.path 注入 SKILL_ROOT；REPO_ROOT 由内层 persuasion_requirement_engine
# 的 _bootstrap_core_skill_on_path 在首次 import 时自动补齐（避免外层 stub
# commerce_video_diagnosis 抢占内层真实包）。
SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

# 先 import 内层 commerce_video_diagnosis（触发 bootstrap，使 core_skill 可见）。
from commerce_video_diagnosis.understanding.engines import (  # noqa: E402
    persuasion_requirement_engine as pre_module,
)
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import (  # noqa: E402
    PersuasionRequirementEngine,
)
from commerce_video_diagnosis.understanding.gift_context import (  # noqa: E402
    detect_gift_context,
)
from commerce_video_diagnosis.understanding.engines.product_diagnoser import (  # noqa: E402
    DifferentiatorEvidence,
    JTBDProposal,
    Module1Output,
    ProductDiagnosisEngine,
    ProductIntentMatrix,
    ProductTargetAudience,
    StructuredDifferentiator,
)
from commerce_video_diagnosis.understanding.assembly.response_assembler import (  # noqa: E402
    build_product_understanding,
)
from core_skill.schemas.protocols import (  # noqa: E402
    ACTIVE_REQUIREMENT_WHITELIST,
    DEPRECATED_PERSUASION_KEYS,
)

# ---- 单例 PR 引擎；JG 测试不依赖研发自测引擎实例 ----
_JG_PR_ENGINE = PersuasionRequirementEngine()
_ENGINE_CTOR_BYPASS = lambda: ProductDiagnosisEngine.__new__(ProductDiagnosisEngine)  # noqa: E731


# ---------------------------------------------------------------------------
# 工具：构造合法 Module1Output / ProductIntentMatrix / JTBDProposal
# ---------------------------------------------------------------------------
def _mk_module1(
    *,
    leaf_category: str,
    product_name: str,
    core_selling_point: str = "占位卖点",
    target_people: str = "占位人群",
    target_people_raw: str = "",
) -> Module1Output:
    return Module1Output(
        leaf_category=leaf_category,
        shop_name="JG_QA_店铺",
        product_name=product_name,
        price="128",
        core_selling_point=core_selling_point,
        core_selling_point_source="caller_provided.core_selling_points",
        target_people=target_people,
        differentiator=StructuredDifferentiator(
            comparison_object="",
            difference_domain="functional",
            difference_type="自身卖点陈述",
            conclusion="占位",
            evidence_chain=[DifferentiatorEvidence("商品信息", core_selling_point or "占位")],
            summary="占位",
        ),
        target_people_raw=target_people_raw,
    )


def _mk_matrix(
    *,
    brand_tier: str = "大牌官方",
    relative_price_level: str = "低水位",
    trust_barrier: str = "极低",
    business_category: str = "T恤/Polo衫",
) -> ProductIntentMatrix:
    return ProductIntentMatrix(
        brand_tier=brand_tier,
        trust_barrier=trust_barrier,
        financial_risk="中",
        relative_price_level=relative_price_level,
        matrix_label=f"{brand_tier}×{relative_price_level}",
        business_category=business_category,
        median_price_threshold=120.0,
        price_value=98.0,
        product_intent="占位",
        reasoning=["占位"],
    )


def _mk_proposal(primary_task: str = "生存/运转维系", domain: str = "功能域") -> JTBDProposal:
    return JTBDProposal(
        domain=domain,
        primary_task=primary_task,
        reasoning="占位",
        reasoning_path=["占位"],
    )


def _jg_profile(
    *,
    title: str,
    selling_points: list[str],
    source_evidence: list[str],
    target_people_raw: str,
    leaf_category: str = "T恤/Polo衫",
    jtbd_level1: str = "功能域",
    jtbd_level2: str = "生存/运转维系",
    price_attribute: str = "低水位",
    trust_attribute: str = "极低",
) -> dict[str, Any]:
    pfact = {
        "leaf_category": leaf_category,
        "category": leaf_category,
        "title": title,
        "jtbd_level1": jtbd_level1,
        "jtbd_level2": jtbd_level2,
        "cognition_attribute": "红海-核心",
        "frequency_attribute": "耐用",
        "trust_attribute": trust_attribute,
        "price_attribute": price_attribute,
        "selling_points": selling_points,
        "source_evidence": source_evidence,
        "target_people_raw": target_people_raw,
        "risk_points": [],
    }
    profile = _JG_PR_ENGINE.generate_profile(pfact, content_goal="purchase")
    assert profile.get("persuasion_requirements"), "profile 必须非空"
    return profile


# ===========================================================================
# AC1：最新代码对 hla 样本派生 → 旧人群「年轻中高消费力女性」错判必须消除
# ===========================================================================
@pytest.mark.unit
def test_jg_ac1_hla_old_snapshot_misclassification_eliminated():
    """AC1：用当前最新代码独立构造 hla 派生，primary/secondary 与
    persuasion_profile_to_audience 必须体现双重角色，**不得回到** 旧快照
    （primary=年轻中高消费力女性、未拆双重角色）。

    旧 outputs/hla_diagnosis/hla_full_diagnosis.json 仍保留错判快照（生成
    早于第五批），本用例独立重跑 audience 派生，证明同一组 hla 信号在最新
    代码下旧错判已消除。
    """
    engine = _ENGINE_CTOR_BYPASS()
    module1 = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家山不在高短袖polo 凉感抗菌速干",
        core_selling_point="凉感系数≥0.15、抗菌、吸湿排汗速干的夏季短袖 polo",
        target_people="中年男性人群",
        target_people_raw="父亲节给爸爸/父亲选礼物的成年子女；自用的中年男性",
    )
    product_fact = {
        "leaf_category": "T恤/Polo衫",
        "category": "T恤/Polo衫",
        "title": "HLA海澜之家山不在高短袖polo 凉感抗菌速干",
        "selling_points": ["凉感系数≥0.15、抗菌、吸湿排汗速干的夏季短袖 polo"],
        "source_evidence": [
            "凉感科技：接触凉感系数 ≥0.15 J/(cm²·s)，HLA TECH 认证",
            "海澜之家品牌官方背书 + 父亲节礼盒装",
        ],
        "target_people_raw": "父亲节给爸爸/父亲选礼物的成年子女；自用的中年男性",
    }
    profile = _jg_profile(
        title=product_fact["title"],
        selling_points=product_fact["selling_points"],
        source_evidence=product_fact["source_evidence"],
        target_people_raw=product_fact["target_people_raw"],
    )
    result = engine._derive_product_target_audience(
        module1,
        _mk_proposal("生存/运转维系"),
        _mk_matrix(brand_tier="大牌官方", relative_price_level="低水位"),
        persuasion_requirement_profile=profile,
        product_fact=product_fact,
    )
    assert isinstance(result, ProductTargetAudience)

    # 旧错判：primary 不应再出现「年轻中高消费力女性」
    primary_labels = [a.audience_group for a in result.primary_audiences]
    assert "年轻中高消费力女性" not in primary_labels, (
        f"AC1 FAIL：旧错判仍在 primary={primary_labels}"
    )
    # 新口径：primary 必须等于「成年子女」（购买决策者）
    assert primary_labels == ["成年子女"], f"AC1 FAIL：primary={primary_labels}"
    # secondary 必须含「爸爸」+「中年男性」（受礼者画像）
    secondary_labels = [a.audience_group for a in result.secondary_audiences]
    assert any("爸爸" in s and "中年男性" in s for s in secondary_labels), (
        f"AC1 FAIL：secondary={secondary_labels}"
    )
    # 第四段：必须存在并解释购买者与受礼者关系
    ppta = result.reasoning_chain.persuasion_profile_to_audience
    assert ppta and "成年子女" in ppta and "爸爸" in ppta, f"AC1 FAIL ppta={ppta!r}"


# ===========================================================================
# AC2：接入层送礼信号不被剥离——target_people_raw + raw signal trace
# ===========================================================================
@pytest.mark.unit
def test_jg_ac2_input_layer_raw_signal_preserved_and_traced():
    """AC2：``Module1Output.target_people_raw`` 字段存在且完整保留节日/送礼/
    对象/关系词；``_record_gift_signal_trace`` 把命中写入 keyword_rule_traces。

    本用例绕开 LLM 依赖（不调用 _assert_structured_differentiator 中的语义
    judge），直接构造合法 Module1Output 与 DiagnosticInput 后调用引擎的
    trace 写入函数 —— 这样既验证字段保留口径、又验证 trace 链路。
    """
    from commerce_video_diagnosis.understanding.engines.product_diagnoser import (
        DiagnosticInput,
    )

    # 字段存在性：Module1Output 必须包含 target_people_raw
    module1_fields = set(Module1Output.__dataclass_fields__.keys())
    assert "target_people_raw" in module1_fields, (
        f"AC2 FAIL：Module1Output 缺字段 target_people_raw，fields={module1_fields}"
    )

    raw_signal = "父亲节给爸爸/老爸选礼物的成年子女；送给爸爸"
    module1 = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家 polo",
        core_selling_point="凉感抗菌速干",
        target_people="中年男性人群",
        target_people_raw=raw_signal,
    )
    for token in ("父亲节", "爸爸", "成年子女", "送给", "礼物", "老爸"):
        assert token in module1.target_people_raw, (
            f"AC2 FAIL：target_people_raw 丢失关键词 {token!r}：{module1.target_people_raw!r}"
        )

    payload = DiagnosticInput(
        leaf_category="T恤/Polo衫",
        shop_name="JG_QA_店铺",
        second_level_category="男装",
        third_level_category="T恤/Polo衫",
        brand_name="HLA海澜之家",
        product_name="HLA海澜之家 polo",
        price="98",
        core_selling_point="凉感抗菌速干",
        core_selling_point_source="caller_provided.core_selling_points",
        target_people=raw_signal,
        differentiator="",
        bridge_comparison_object="",
        bridge_comparison_object_evidence_type="null",
        bridge_difference_domain="functional",
        bridge_difference_type="自身卖点陈述",
        bridge_source_evidence=["海澜之家品牌官方背书 + 父亲节礼盒装"],
        bridge_evidence_source="商品信息",
        product_id="jg_ac2",
        engine_node={"relative_price_level": "低水位"},
    )

    # 绕开 __init__ 与 LLM 依赖；直接构造引擎并复用 trace 写入函数。
    engine = _ENGINE_CTOR_BYPASS()
    engine._keyword_rule_traces = []  # 与 _reset_keyword_rule_traces 等价
    engine._record_gift_signal_trace(payload, module1)
    traces = list(engine._keyword_rule_traces)
    gift_traces = [
        t
        for t in traces
        if t.get("source_rule") == "gift_context_raw_signal_trace"
        or t.get("field_name") == "gift_context"
    ]
    assert gift_traces, f"AC2 FAIL：缺少 gift_context raw signal trace，traces={traces}"
    t0 = gift_traces[0]
    assert "is_gift=true" in (t0.get("output_value") or ""), (
        f"AC2 FAIL：trace output_value 异常：{t0}"
    )
    matched_kw = t0.get("matched_keyword") or ""
    assert any(tok in matched_kw for tok in ("父亲节", "爸爸", "礼物", "送给")), (
        f"AC2 FAIL：trace matched_keyword 异常：{matched_kw!r}"
    )

    # bridge_source_evidence 中的送礼词也参与识别（即使 product_name 被去送礼词）
    payload_bridge_only = DiagnosticInput(
        leaf_category="T恤/Polo衫",
        shop_name="JG_QA_店铺",
        second_level_category="男装",
        third_level_category="T恤/Polo衫",
        brand_name="HLA海澜之家",
        product_name="HLA海澜之家 polo",
        price="98",
        core_selling_point="凉感抗菌速干",
        core_selling_point_source="caller_provided.core_selling_points",
        target_people="中年男性",
        differentiator="",
        bridge_comparison_object="",
        bridge_comparison_object_evidence_type="null",
        bridge_difference_domain="functional",
        bridge_difference_type="自身卖点陈述",
        bridge_source_evidence=["父亲节礼盒装 送爸爸"],
        bridge_evidence_source="商品信息",
        product_id="jg_ac2_bridge",
        engine_node={"relative_price_level": "低水位"},
    )
    module1_bridge_only = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家 polo",
        core_selling_point="凉感抗菌速干",
        target_people="中年男性",
        target_people_raw="中年男性",
    )
    engine2 = _ENGINE_CTOR_BYPASS()
    engine2._keyword_rule_traces = []
    engine2._record_gift_signal_trace(payload_bridge_only, module1_bridge_only)
    traces2 = list(engine2._keyword_rule_traces)
    assert any(
        t.get("source_rule") == "gift_context_raw_signal_trace" for t in traces2
    ), f"AC2 FAIL：bridge_source_evidence 中送礼信号未参与 trace 记录：{traces2}"


# ===========================================================================
# AC3：profile 输出可见 gift_context 命中（source=gift_context_rule）
# ===========================================================================
@pytest.mark.unit
def test_jg_ac3_profile_exposes_gift_context_source():
    profile = _jg_profile(
        title="海澜之家短袖 polo",
        selling_points=["凉感抗菌速干"],
        source_evidence=["父亲节礼盒装 送爸爸"],
        target_people_raw="父亲节给爸爸/父亲选礼物的成年子女",
    )
    reqs = profile["persuasion_requirements"]
    gift_sourced = {
        r["requirement_id"]
        for r in reqs
        if "gift_context_rule" in (r.get("source") or [])
    }
    expected = {"identify_target_user", "prove_user_fit", "clarify_usage_scenario"}
    assert expected <= gift_sourced, (
        f"AC3 FAIL：gift_context_rule 来源 requirement 缺失，"
        f"期望 {expected}，实际 {gift_sourced}"
    )


# ===========================================================================
# AC4：gift requirement 必须命中已有 active 白名单 + Crash Early 校验
# ===========================================================================
@pytest.mark.unit
def test_jg_ac4_gift_requirements_within_whitelist():
    profile = _jg_profile(
        title="父亲节礼物 polo",
        selling_points=["凉感抗菌速干"],
        source_evidence=["父亲节礼盒装 送爸爸"],
        target_people_raw="父亲节给爸爸的成年子女",
    )
    for r in profile["persuasion_requirements"]:
        rid = r["requirement_id"]
        assert rid in ACTIVE_REQUIREMENT_WHITELIST, (
            f"AC4 FAIL：profile 出现非 active id={rid}"
        )

    # GIFT_CONTEXT_REQUIREMENTS 配置中每个 id 必须在白名单内
    for rid, _tpl in pre_module.GIFT_CONTEXT_REQUIREMENTS:
        assert rid in ACTIVE_REQUIREMENT_WHITELIST, (
            f"AC4 FAIL：GIFT_CONTEXT_REQUIREMENTS 含非 active id={rid}"
        )


@pytest.mark.unit
def test_jg_ac4_illegal_mapping_crash_early(monkeypatch):
    """构造非法映射（新造 requirement_id）必须 Crash Early，禁止静默兜底。"""
    illegal = (
        ("__not_in_whitelist_xxx__", "非法 id"),
        ("identify_target_user", "锁定受礼者「{recipient}」"),
    )
    monkeypatch.setattr(pre_module, "GIFT_CONTEXT_REQUIREMENTS", illegal, raising=True)
    engine = PersuasionRequirementEngine()
    pfact = {
        "leaf_category": "T恤/Polo衫",
        "category": "T恤/Polo衫",
        "title": "polo",
        "jtbd_level1": "功能域",
        "jtbd_level2": "生存/运转维系",
        "cognition_attribute": "红海-核心",
        "frequency_attribute": "耐用",
        "trust_attribute": "极低",
        "price_attribute": "低水位",
        "selling_points": ["凉感抗菌速干"],
        "source_evidence": ["父亲节礼盒装 送爸爸"],
        "target_people_raw": "父亲节给爸爸的成年子女",
        "risk_points": [],
    }
    with pytest.raises(ValueError, match="不在 active 白名单内"):
        engine.generate_profile(pfact, content_goal="purchase")


# ===========================================================================
# AC5：audience primary=purchase_decider、secondary=gift_recipient
# ===========================================================================
@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,leaf,task,domain,exp_primary,exp_recipient_keywords",
    [
        # 父亲节 → 子女送父亲
        (
            "父亲节给爸爸选礼物",
            "T恤/Polo衫",
            "生存/运转维系",
            "功能域",
            "成年子女",
            ("爸爸", "中年男性"),
        ),
        # 母亲节 → 子女送母亲
        (
            "母亲节送妈妈的护手霜",
            "护手霜",
            "生存/运转维系",
            "功能域",
            "成年子女",
            ("妈妈", "中年女性"),
        ),
        # 七夕 → 伴侣互赠
        (
            "七夕礼物送女友",
            "T恤/Polo衫",
            "生存/运转维系",
            "功能域",
            "伴侣",
            ("伴侣",),
        ),
        # 教师节 → 学生或家长送老师
        (
            "教师节礼物送老师",
            "T恤/Polo衫",
            "生存/运转维系",
            "功能域",
            "学生或家长",
            ("老师",),
        ),
    ],
)
def test_jg_ac5_audience_dual_role_for_various_gift_scenes(
    raw, leaf, task, domain, exp_primary, exp_recipient_keywords
):
    engine = _ENGINE_CTOR_BYPASS()
    module1 = _mk_module1(
        leaf_category=leaf,
        product_name="JG_QA_测试礼品",
        core_selling_point="占位卖点",
        target_people="占位人群",
        target_people_raw=raw,
    )
    pfact = {
        "leaf_category": leaf,
        "category": leaf,
        "title": "JG_QA_测试礼品",
        "selling_points": ["占位卖点"],
        "source_evidence": [raw],
        "target_people_raw": raw,
    }
    profile = _jg_profile(
        title=pfact["title"],
        selling_points=pfact["selling_points"],
        source_evidence=pfact["source_evidence"],
        target_people_raw=raw,
        leaf_category=leaf,
        jtbd_level1=domain,
        jtbd_level2=task,
    )
    result = engine._derive_product_target_audience(
        module1,
        _mk_proposal(task, domain),
        _mk_matrix(brand_tier="大牌官方", relative_price_level="低水位"),
        persuasion_requirement_profile=profile,
        product_fact=pfact,
    )
    assert [a.audience_group for a in result.primary_audiences] == [exp_primary], (
        f"AC5 FAIL：raw={raw!r}, primary={result.primary_audiences}"
    )
    sec_labels = [a.audience_group for a in result.secondary_audiences]
    for tok in exp_recipient_keywords:
        assert any(tok in s for s in sec_labels), (
            f"AC5 FAIL：缺受礼者关键词 {tok!r}，secondary={sec_labels}"
        )


# ===========================================================================
# AC6：persuasion_profile_to_audience 解释 buyer↔recipient 关系并引用真实 requirement
# ===========================================================================
@pytest.mark.unit
def test_jg_ac6_ppta_explains_relationship_and_cites_real_requirement():
    engine = _ENGINE_CTOR_BYPASS()
    raw = "父亲节给爸爸/父亲选礼物的成年子女"
    pfact = {
        "leaf_category": "T恤/Polo衫",
        "category": "T恤/Polo衫",
        "title": "HLA海澜之家 polo",
        "selling_points": ["凉感抗菌速干"],
        "source_evidence": ["父亲节礼盒装"],
        "target_people_raw": raw,
    }
    profile = _jg_profile(
        title=pfact["title"],
        selling_points=pfact["selling_points"],
        source_evidence=pfact["source_evidence"],
        target_people_raw=raw,
    )
    module1 = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name=pfact["title"],
        core_selling_point=pfact["selling_points"][0],
        target_people="中年男性人群",
        target_people_raw=raw,
    )
    result = engine._derive_product_target_audience(
        module1,
        _mk_proposal("生存/运转维系"),
        _mk_matrix(),
        persuasion_requirement_profile=profile,
        product_fact=pfact,
    )
    ppta = result.reasoning_chain.persuasion_profile_to_audience
    assert ppta and ppta.strip(), "AC6 FAIL：ppta 为空"
    # 显式提到购买者与受礼者，且明确「非同一人」
    assert "成年子女" in ppta and "爸爸" in ppta, f"AC6 FAIL ppta={ppta!r}"
    assert ("非同一人" in ppta) or ("≠" in ppta), (
        f"AC6 FAIL：未显式标注非同一人 ppta={ppta!r}"
    )
    # 引用真实 requirement_id（profile 中 gift_context_rule 来源的 id 出现在 ppta）
    gift_ids = [
        r["requirement_id"]
        for r in profile["persuasion_requirements"]
        if "gift_context_rule" in (r.get("source") or [])
    ]
    assert gift_ids, "AC6 FAIL：profile 无 gift_context_rule 来源 requirement"
    assert any(rid in ppta for rid in gift_ids), (
        f"AC6 FAIL：ppta 未引用真实 requirement_id={gift_ids} ppta={ppta!r}"
    )
    # 同时引用了 requirement_name（不是占位文案）
    name_ref = any(
        (r.get("requirement_name") or "") in ppta
        for r in profile["persuasion_requirements"]
        if "gift_context_rule" in (r.get("source") or [])
    )
    assert name_ref, f"AC6 FAIL：ppta 未引用真实 requirement_name ppta={ppta!r}"


# ===========================================================================
# AC7：品牌/价格消费力规则不覆盖显式送礼信号
# ===========================================================================
@pytest.mark.unit
def test_jg_ac7_gift_branch_overrides_brand_price_rule():
    """同 module1 标的，仅切换是否含送礼信号：含送礼 → 走 gift 分支；
    无送礼 → 走品牌价格消费力分支（八大人群）。
    """
    engine = _ENGINE_CTOR_BYPASS()
    base_module1 = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家 polo",
        core_selling_point="凉感抗菌速干",
        target_people="中年男性人群",
        target_people_raw="夏季出汗多/需要凉感速干 polo 的中年男性",
    )
    profile_non_gift = _JG_PR_ENGINE.generate_profile(
        {
            "leaf_category": "T恤/Polo衫",
            "category": "T恤/Polo衫",
            "title": "HLA海澜之家 polo",
            "jtbd_level1": "功能域",
            "jtbd_level2": "生存/运转维系",
            "cognition_attribute": "红海-核心",
            "frequency_attribute": "耐用",
            "trust_attribute": "极低",
            "price_attribute": "低水位",
            "selling_points": ["凉感抗菌速干"],
            "source_evidence": ["第三方检测报告"],
            "target_people_raw": "夏季出汗多/需要凉感速干 polo 的中年男性",
            "risk_points": [],
        },
        content_goal="purchase",
    )
    non_gift_pfact = {
        "leaf_category": "T恤/Polo衫",
        "category": "T恤/Polo衫",
        "title": "HLA海澜之家 polo",
        "selling_points": ["凉感抗菌速干"],
        "source_evidence": ["第三方检测报告"],
        "target_people_raw": "夏季出汗多/需要凉感速干 polo 的中年男性",
    }
    res_non_gift = engine._derive_product_target_audience(
        base_module1,
        _mk_proposal("生存/运转维系"),
        _mk_matrix(brand_tier="大牌官方", relative_price_level="低水位"),
        persuasion_requirement_profile=profile_non_gift,
        product_fact=non_gift_pfact,
    )
    # 八大人群坐标含「消费力」字眼
    assert all("消费力" in a.audience_group for a in res_non_gift.primary_audiences), (
        f"AC7 FAIL：非送礼 primary 未走八大人群，{res_non_gift.primary_audiences}"
    )

    # 同标的但加入送礼信号 → 必须切到 gift 分支（primary != 八大人群标签）
    gift_pfact = dict(non_gift_pfact)
    gift_pfact["source_evidence"] = ["父亲节礼盒装 送爸爸"]
    gift_pfact["target_people_raw"] = "父亲节给爸爸选礼物的成年子女"
    profile_gift = _jg_profile(
        title=gift_pfact["title"],
        selling_points=gift_pfact["selling_points"],
        source_evidence=gift_pfact["source_evidence"],
        target_people_raw=gift_pfact["target_people_raw"],
    )
    module1_gift = _mk_module1(
        leaf_category="T恤/Polo衫",
        product_name="HLA海澜之家 polo",
        core_selling_point="凉感抗菌速干",
        target_people="中年男性人群",
        target_people_raw="父亲节给爸爸选礼物的成年子女",
    )
    res_gift = engine._derive_product_target_audience(
        module1_gift,
        _mk_proposal("生存/运转维系"),
        _mk_matrix(brand_tier="大牌官方", relative_price_level="低水位"),
        persuasion_requirement_profile=profile_gift,
        product_fact=gift_pfact,
    )
    primary_gift = [a.audience_group for a in res_gift.primary_audiences]
    assert primary_gift == ["成年子女"], (
        f"AC7 FAIL：送礼信号未覆盖品牌价格规则，primary={primary_gift}"
    )
    # 显式标注送礼者≠受礼者
    assert any("送礼者≠受礼者" in c for c in res_gift.caveats), (
        f"AC7 FAIL：缺 caveats={res_gift.caveats}"
    )


# ===========================================================================
# AC8：runben（非送礼）样本不回退——audience/ppta 无 gift 污染
# ===========================================================================
@pytest.mark.unit
def test_jg_ac8_runben_non_gift_no_regression():
    runben_path = SKILL_ROOT / "outputs/runben_diagnosis/runben_full_diagnosis.json"
    assert runben_path.exists(), "AC8 FAIL：缺少 runben 输出快照"
    data = json.loads(runben_path.read_text(encoding="utf-8"))

    # 1) profile 中无 gift_context_rule 来源
    prp = data.get("persuasion_requirement_profile") or {}
    for r in prp.get("persuasion_requirements") or []:
        assert "gift_context_rule" not in (r.get("source") or []), (
            f"AC8 FAIL：runben profile 出现 gift_context_rule 污染：{r}"
        )

    # 2) audience 未走 gift 分支：primary 仍为八大人群坐标
    pta = data.get("product_target_audience") or {}
    for a in pta.get("primary_audiences") or []:
        assert "消费力" in (a.get("audience_group") or ""), (
            f"AC8 FAIL：runben audience 退化非八大人群：{a}"
        )
    caveats = pta.get("caveats") or []
    for c in caveats:
        assert "送礼者≠受礼者" not in c, f"AC8 FAIL：runben caveats 出现 gift 污染：{c}"
    chain = pta.get("reasoning_chain") or {}
    for v in chain.values():
        if isinstance(v, str):
            assert "送礼" not in v, f"AC8 FAIL：runben reasoning_chain 出现 gift 污染：{v}"

    # 3) 接入层独立校验：runben 输入信号不触发 gift_context
    runben_raw_segments = [
        "宝宝防蚊水 婴幼儿户外驱蚊",
        "派卡瑞丁A级驱蚊力",
        "婴幼儿/儿童/家庭日常户外人群",
    ]
    assert detect_gift_context(runben_raw_segments) is None, (
        "AC8 FAIL：runben 输入误命中 gift_context"
    )


# ===========================================================================
# AC9：6 段结构、product_fact_vector 独立、product_hec 三元组不回退
# ===========================================================================
@pytest.mark.unit
def test_jg_ac9_six_section_structure_and_hec_triple_no_regression():
    runben_path = SKILL_ROOT / "outputs/runben_diagnosis/runben_full_diagnosis.json"
    data = json.loads(runben_path.read_text(encoding="utf-8"))
    pu = build_product_understanding(data)

    # 6 段固定键集合 + 顺序
    expected_keys = ["basic_info", "product_fact_vector", "module3", "candidate_set", "product_hec", "evidence"]
    assert list(pu.keys()) == expected_keys, (
        f"AC9 FAIL：6 段结构异常，实际 keys={list(pu.keys())}"
    )

    # module3 透传 profile + audience
    assert "persuasion_requirement_profile" in pu["module3"]
    assert "product_target_audience" in pu["module3"]

    # product_fact_vector 独立（不并入 conversion_resistance）
    assert isinstance(pu["product_fact_vector"], dict) and pu["product_fact_vector"]

    # 业务子树不出旧字段：DEPRECATED_PERSUASION_KEYS 不能作为 dict key 出现
    # （recursive 检查；不做朴素子串匹配，避免 persuasion_profile_to_audience 这类
    # 合法长键命中 persuasion_profile 子串）
    def _walk_dict_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_dict_keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _walk_dict_keys(v)

    all_keys = set(_walk_dict_keys(pu))
    for bad in DEPRECATED_PERSUASION_KEYS:
        assert bad not in all_keys, (
            f"AC9 FAIL：旧字段 {bad!r} 作为 dict key 出现在 product_understanding"
        )

    # product_hec 必须为三元组 {code,name,definition}
    assert pu["product_hec"], "AC9 FAIL：product_hec 为空"
    for hec in pu["product_hec"]:
        for axis in ("hook", "effect", "cta"):
            assert axis in hec, f"AC9 FAIL：缺 product_hec.{axis}"
            triple = hec[axis]
            for key in ("code", "name", "definition"):
                assert key in triple and str(triple[key]).strip(), (
                    f"AC9 FAIL：product_hec.{axis}.{key} 缺失：{triple}"
                )


# ===========================================================================
# AC 附加：通用性 —— 4 类非父亲节送礼正样本 + 1 个负样本
# ===========================================================================
@pytest.mark.unit
@pytest.mark.parametrize(
    "label,texts,exp_scene,exp_recipient,exp_decider",
    [
        ("生日礼物送女朋友", ["生日礼物送女朋友"], "生日", "伴侣", "伴侣"),
        ("母亲节送妈妈", ["母亲节送妈妈的护手霜礼盒"], "母亲节", "妈妈", "成年子女"),
        ("七夕送伴侣", ["七夕情人节送老公的礼物"], "七夕情人节", "伴侣", "伴侣"),
        # 过年送长辈：scene=春节过年（无 implies_recipient），recipient 显式命中
        ("过年送长辈", ["过年送长辈的保健礼盒"], "春节过年", "长辈", "晚辈"),
        ("教师节送老师", ["教师节礼物送老师"], "教师节", "老师", "学生或家长"),
        ("给客户送礼", ["送给客户的高端礼盒"], "通用送礼", "职场关系人", "职场关系人"),
    ],
)
def test_jg_acextra_generic_gift_scenarios_positive(
    label, texts, exp_scene, exp_recipient, exp_decider
):
    gc = detect_gift_context(texts)
    assert gc is not None and gc["is_gift"] is True, (
        f"通用性 FAIL[{label}]：未识别 gift_context"
    )
    assert gc["gift_scene"] == exp_scene, (
        f"通用性 FAIL[{label}]：scene={gc['gift_scene']} 期望 {exp_scene}"
    )
    assert gc["gift_recipient"] == exp_recipient, (
        f"通用性 FAIL[{label}]：recipient={gc['gift_recipient']} 期望 {exp_recipient}"
    )
    assert gc["purchase_decider"] == exp_decider, (
        f"通用性 FAIL[{label}]：decider={gc['purchase_decider']} 期望 {exp_decider}"
    )
    # 任何正样本都必须有 evidence 非空（可追溯）
    assert gc.get("evidence"), f"通用性 FAIL[{label}]：evidence 为空"


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,texts",
    [
        ("普通自用 polo", ["夏季出汗多/需要凉感速干 polo 的中年男性"]),
        ("婴幼儿驱蚊", ["宝宝防蚊水 婴幼儿户外驱蚊"]),
        ("普通牛奶", ["家庭日常饮用纯牛奶"]),
        ("纯人群词无意图无时机", ["中年男性", "男士"]),
    ],
)
def test_jg_acextra_non_gift_negative_no_false_trigger(label, texts):
    assert detect_gift_context(texts) is None, f"通用性 FAIL[{label}]：负样本误触发 gift_context"


# ===========================================================================
# AC 附加：禁止 hla/父亲节单样本硬编码（基于源码静态检索）
# ===========================================================================
@pytest.mark.unit
def test_jg_acextra_no_hla_or_father_day_hardcoding_in_engine():
    """框架级断言：识别引擎源码（gift_context.py + product_diagnoser.py 中的
    gift 分支函数 + persuasion_requirement_engine.py 中的 gift 激活）必须保持
    通用，不得硬编码 'hla' / 'HLA' / '海澜之家' 作为判定条件。
    """
    targets = [
        SKILL_ROOT / "commerce_video_diagnosis/understanding/gift_context.py",
        SKILL_ROOT
        / "commerce_video_diagnosis/understanding/engines/persuasion_requirement_engine.py",
    ]
    forbidden = ("hla", "HLA", "海澜之家")
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (
                f"通用性 FAIL：{path.name} 出现 hla 硬编码 token={token!r}"
            )

    # product_diagnoser.py 中 gift 分支函数 _build_gift_context_audience 不得引用 'hla'
    pd_text = (
        SKILL_ROOT / "commerce_video_diagnosis/understanding/engines/product_diagnoser.py"
    ).read_text(encoding="utf-8")
    start = pd_text.find("def _build_gift_context_audience")
    assert start > 0, "通用性 FAIL：未找到 _build_gift_context_audience 函数体"
    end = pd_text.find("\n    def ", start + 1)
    body = pd_text[start:end if end > 0 else len(pd_text)]
    for token in ("hla", "HLA", "海澜之家"):
        assert token not in body, (
            f"通用性 FAIL：_build_gift_context_audience 硬编码 token={token!r}"
        )
