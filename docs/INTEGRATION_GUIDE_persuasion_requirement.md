# Persuasion Requirement 主干接入说明

## 1. 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `product_fact` | object | 是 | 说服要求建模引擎的商品事实输入对象，由 caller 组装后传入。引擎只消费事实，不自行推断调用目标。 |
| `product_fact.leaf_category` / `category` | string | 建议必填 | 商品叶子类目。用于 `category_group_routing_dictionary.json` 查表路由到 `category_group`；未命中时固定回落 `unknown`，不由 LLM 猜测。 |
| `product_fact.jtbd_level1` | string | 否 | 一级 JTBD。缺省按 `功能任务` 处理。 |
| `product_fact.jtbd_level2` | string | 建议必填 | 二级 JTBD。用于 `JTBD_requirement_template_dictionary.json` 召回模板；未命中时 `jtbd_template_status=fallback_generic`。 |
| `product_fact.selling_points` | array[string] | 否 | 商品卖点事实。用于部分 `activation_condition` 的证据匹配。 |
| `product_fact.risk_points` | array[string] | 否 | 商品风险点事实。会进入相关 requirement 的 `risk_points` 与主路径说明。 |
| `product_fact.certifications` | array[string] | 否 | 权威认证、检测、标准等事实。用于权威背书类 requirement 激活判断。 |
| `content_goal` | enum string | 是 | 内容目标，由 caller 显式传入。合法枚举：`conversion`、`purchase`、`add_to_cart`、`coupon_claim`、`shop_entry`、`seeding`、`education`、`brand_awareness`、`unknown`。引擎只做枚举校验，不由 LLM 推断。 |

调用入口：

```python
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import build_persuasion_requirement_profile

profile = build_persuasion_requirement_profile(
    product_fact={
        "leaf_category": "速食拌面",
        "jtbd_level1": "功能任务",
        "jtbd_level2": "降本增效/懒人替代",
        "selling_points": ["3分钟出餐", "酱料丰富"],
        "risk_points": ["担心不好吃", "担心分量不足"],
    },
    content_goal="conversion",
)
```

## 2. 输出

`persuasion_requirement_profile` 是 `ProductDiagnosisOutput` 上的可选旁路字段，类型为 `object | null`。完整结构固定为 11 个顶层字段：

| 字段 | 类型 | 定义 |
|---|---:|---|
| `profile_version` | string | profile 协议版本，当前固定 `v3.1`。 |
| `content_goal` | enum string | caller 传入的内容目标原值，通过 9 项枚举校验。 |
| `category_group` | string | 类目路由结果。命中路由表时为具体品类组，未命中为 `unknown`。 |
| `jtbd_template_status` | enum string | JTBD 模板召回状态：`matched` 或 `fallback_generic`。 |
| `requirement_dictionary_version` | string | 说服要求字典版本，当前固定 `v3.1_active_mvp_23`。 |
| `category_purchase_criteria_version` | string | 品类购买判断字典版本，当前固定 `v3.1_phase1_4groups`。 |
| `main_persuasion_route` | object | 主说服路径说明，包含 JTBD、品类阻力、商品转化阻力。 |
| `activated_category_requirements` | object | 本次被品类判断激活的决策标准、证据要求、风险点。 |
| `persuasion_requirements` | array[object] | 本次应进入诊断的说服要求列表，只允许 23 条 active MVP 白名单内 ID。 |
| `not_applicable_requirements` | array[object] | 本次明确不适用的要求。非转化目标下 action gap 要求必须进入这里。 |
| `diagnosis_contract` | object | 下游诊断契约，包含完成状态枚举、最低必需要求、诊断维度。 |

`persuasion_requirements[]` 字段定义：

| 字段 | 类型 | 定义 |
|---|---:|---|
| `requirement_id` | string | 说服要求 ID，必须在 23 条 active MVP 白名单内。 |
| `requirement_name` | string | 要求名称；JTBD 模板命中时可由 `instantiated_requirement_name` 实例化。 |
| `decision_gap` | enum string | 决策缺口：`need_gap`、`fit_gap`、`value_gap`、`proof_gap`、`trust_gap`、`risk_gap`、`action_gap`。 |
| `source` | array[string] | 来源字典，可包含通用 requirement、JTBD 模板、品类购买判断。 |
| `priority` | enum string | 优先级：`high`、`medium`、`low`。 |
| `required` | boolean | 是否必需。由字典、模板、品类激活与优先级合并后裁决。 |
| `sequence_rank` | integer | 建议诊断顺序，合法范围 10-59。 |
| `success_criteria` | string | 达成标准；JTBD 模板命中时可合入 `instantiated_success_criteria`。 |
| `related_decision_criteria` | array[string] | 关联的品类购买判断 criterion。 |
| `required_evidence_requirements` | array[string] | 需要的视频/内容证据要求。 |
| `risk_points` | array[string] | 关联风险点。 |

示例：

```json
{
  "profile_version": "v3.1",
  "content_goal": "conversion",
  "category_group": "食品生鲜",
  "jtbd_template_status": "matched",
  "requirement_dictionary_version": "v3.1_active_mvp_23",
  "category_purchase_criteria_version": "v3.1_phase1_4groups",
  "main_persuasion_route": {
    "primary_jtbd": {
      "level1": "功能任务",
      "level2": "降本增效/懒人替代"
    },
    "category_resistance": {
      "rule": "category_group_routing_dictionary",
      "summary": "leaf_category=速食拌面 路由到食品生鲜"
    },
    "product_conversion_barrier": {
      "rule": "product_fact_risk_points",
      "summary": "担心不好吃；担心分量不足"
    }
  },
  "activated_category_requirements": {
    "category_group": "食品生鲜",
    "routing_confidence": "high",
    "activated_decision_criteria": ["taste_expectation", "freshness_or_quality"],
    "activated_evidence_requirements": ["口味/口感证明", "品质稳定性证明"],
    "activated_risk_points": ["不好吃", "不新鲜"]
  },
  "persuasion_requirements": [
    {
      "requirement_id": "prove_core_benefit",
      "requirement_name": "证明核心利益",
      "decision_gap": "proof_gap",
      "source": ["persuasion_requirement_dictionary"],
      "priority": "high",
      "required": true,
      "sequence_rank": 30,
      "success_criteria": "观众能看到或理解商品核心利益如何成立。",
      "related_decision_criteria": [],
      "required_evidence_requirements": [],
      "risk_points": []
    }
  ],
  "not_applicable_requirements": [],
  "diagnosis_contract": {
    "requirement_completion_schema": {
      "status_enum": ["completed", "weak", "missing", "not_applicable"],
      "minimum_required_requirements": ["prove_core_benefit", "provide_visible_result"],
      "diagnosis_dimensions": [
        "whether_requirement_appears",
        "whether_evidence_is_sufficient",
        "whether_sequence_is_reasonable",
        "whether_risk_is_resolved"
      ]
    }
  }
}
```

## 3. 字典

四个 V3.1 字典均位于 `core_skill/dictionaries/`：

| 字典路径 | 职责 |
|---|---|
| `core_skill/dictionaries/persuasion_requirement_dictionary.json` | 通用说服要求主字典。维护 active MVP 白名单、candidate pool、决策缺口、默认优先级、顺序和基础达成标准。线上 profile 只允许 active 白名单进入。 |
| `core_skill/dictionaries/JTBD_requirement_template_dictionary.json` | JTBD 模板字典。按 `jtbd_level2` 召回 requirement，并通过 `instantiated_requirement_name`、`instantiated_success_criteria`、`required`、`activation_condition` 控制实例化与激活。 |
| `core_skill/dictionaries/category_purchase_criteria_dictionary.json` | 品类购买判断字典。按 `category_group` 定义品类决策标准、证据要求、风险点，以及映射到 active requirement 的 `derived_requirement_id`。 |
| `core_skill/dictionaries/category_group_routing_dictionary.json` | 类目路由字典。按 `leaf_category` 路由到 `category_group`，同时输出 `routing_confidence`；未命中时回落 `unknown`。 |

## 4. 约束

1. `CandidateSet` 不改。P0 只新增 `persuasion_requirement_profile` 旁路字段，不迁移、不替换、不改写既有 `candidate_set` 输入/输出接口。
2. action gap 受 `content_goal` 控制。`conversion`、`purchase`、`add_to_cart`、`coupon_claim`、`shop_entry` 为转化目标，可激活 action gap；`seeding`、`education`、`brand_awareness`、`unknown` 为非转化目标，action gap 必须进入 `not_applicable_requirements`。
3. `content_goal` 由 caller 传入，不由 LLM 推断。引擎只做闭集枚举校验；越界直接报错。
4. `persuasion_requirement_profile` 不得包含旧版 `persuasion_profile`、`required_persuasion_tasks`，也不得包含 HEC 或动作映射字段。
5. 字典 active 集合、JTBD 模板引用、品类 derived requirement 必须全部命中 active MVP 白名单；不一致时启动期 Crash Early。

## 5. 测试基准

P0 基准为 45 条 persuasion 测试，测试文件与关键用例如下：

| 测试文件 | 条数 | 关键用例名 |
|---|---:|---|
| `tests/test_persuasion_requirement_module.py` | 37 | `test_TC_PR_002_profile_top_fields_exactly_11`、`test_TC_PR_005_no_hec_or_action_keys`、`test_TC_CG_003_out_of_enum_raises`、`test_TC_AG_003_non_conversion_goals_force_not_applicable`、`test_TC_CR_004_four_groups_coverage`、`test_TC_JT_001_four_templates_matched`、`test_TC_LG_003_to_protocol_dict_has_new_no_legacy`、`test_TC_CS_001_candidate_set_field_unchanged` |
| `tests/test_persuasion_requirement_p0_2_binding.py` | 5 | `test_dictionary_has_no_legacy_instantiation_template`、`test_jtbd_instantiated_fields_bind_to_output`、`test_required_false_without_evidence_is_not_activated`、`test_required_false_with_evidence_is_activated`、`test_required_true_item_remains_required` |
| `tests/test_persuasion_requirement_official_path_smoke.py` | 3 | `test_import_persuasion_requirement_engine`、`test_official_engine_loads_all_four_v3_1_dictionaries`、`test_legacy_shim_is_alias_of_official_implementation` |

基准命令：

```bash
pytest tests/test_persuasion_requirement_module.py \
       tests/test_persuasion_requirement_p0_2_binding.py \
       tests/test_persuasion_requirement_official_path_smoke.py \
       -v
```

当前基准结果：`45 passed`。

## 不可随意改动的契约

- `ProductDiagnosisOutput.persuasion_requirement_profile`：Optional 旁路字段，不可删；类型为 `PersuasionRequirementProfile | None`。
- 正式 engine 路径：`commerce_video_diagnosis/understanding/engines/persuasion_requirement_engine.py`。
- 字典路径：`core_skill/dictionaries/`，四个 V3.1 字典同目录，分别为：
  - `persuasion_requirement_dictionary.json`
  - `JTBD_requirement_template_dictionary.json`
  - `category_purchase_criteria_dictionary.json`
  - `category_group_routing_dictionary.json`
- CandidateSet 输入接口：P0 不改；模块 4 仍只消费已经裁决完成的 `candidate_set`。
- `content_goal`：caller 传入，必须通过枚举校验；不允许 LLM 推断、补默认业务目标或基于文本反推。
