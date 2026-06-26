# PRD: video_persuasion_requirement_diagnoser v1（Step 1 协议层）

> 版本：v1.0 (Step 1 协议草稿)
> 范围：仅定义协议字段，不实现诊断逻辑
> 上游依赖：商品侧 `persuasion_requirement_profile`（说服要求建模 V3.1 P0-fix，已通过 /JG 独立验收）
> 下游消费：脚本/分镜重写引擎、人审兜底、效果归因

---

## 1. 模块定位

`video_persuasion_requirement_diagnoser` 是一个**视频侧执行落点诊断模块**，唯一职责是：

> 在已经存在商品侧 `persuasion_requirement_profile` 的前提下，判断**视频内容是否完成了商品侧已确认的说服要求**。

### 1.1 明确不做的事

为防止职责蔓延，本模块**不**承担以下职责：

1. **不重新判断商品该怎么卖**
   - 不重算 requirement、不重算优先级、不重算 required 标记；
   - 商品侧 `persuasion_requirement_profile` 是绝对 SSOT，本模块只读不写。

2. **不重做 HEC 一致性诊断**
   - HEC 一致性属于 `video-hec-consistency-diagnoser` 的职责；
   - 本模块只消费 `recommended_hec` 和 `video_hec_analysis` 作为辅助参考，不输出 HEC 是否一致的结论。

3. **不重新抽取视频事实**
   - 视频证据已由上游（FactPack / AssetIngest 链路）提供为 `video_evidence_spans`；
   - 本模块只在已有 spans 上做对齐，不调用 VLM/ASR/OCR。

### 1.2 唯一关注问题

> 商品侧已经说"这条视频必须打透 R1/R3/R7"，本模块判断这条视频在 R1/R3/R7 上是 **completed / weak / missing**，并指出哪段证据支撑、哪些缺失需要补、哪些是过度承诺。

---

## 2. 协议字段定义

> 所有协议优先采用 Pydantic v2 BaseModel，且强制 `model_config = ConfigDict(extra="forbid")`，防止协议污染。
> 时间字段单位统一为秒（float），与 FactPack `segments[*].start_sec / end_sec` 对齐。

### 2.1 `VideoEvidenceSpan`

视频中可追溯的最小证据片段。

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `span_id` | `str` | 是 | 全局唯一；建议 `{video_id}::{span_type}::{idx}`；与 FactPack segment 关联时保留可溯源前缀 |
| `video_id` | `str` | 是 | 与 Input.video_id 一致；不一致必须 Crash Early |
| `start_time` | `float` | 是 | 单位秒，`>= 0`，且 `< end_time` |
| `end_time` | `float` | 是 | 单位秒，`<= video_meta.duration_sec` |
| `span_type` | `str` (Enum) | 是 | 枚举值见 §2.1.1 |
| `content_text` | `str` | 是 | 该片段的可读文本：口播转写 / 字幕 OCR / 画面动作描述 / 商品展示描述等。**不允许为空字符串**；纯空镜头允许，必须填如 `"[empty_shot]"` 占位语义 |
| `confidence` | `float` | 是 | `[0.0, 1.0]`，来自上游 ASR/OCR/VLM；缺失必须 Crash Early，不允许默认 1.0 |
| `source_segment_ids` | `list[str]` | 否 | 关联的 FactPack `segment_id` 列表；用于双向溯源 |
| `metadata` | `dict[str, Any]` | 否 | 仅承载上游辅助元信息，不允许承载 `completion_status`、`requirement_id`、`risk_type` 等诊断结论字段 |

#### 2.1.1 `span_type` 枚举（覆盖六类）

| 枚举值 | 含义 |
|---|---|
| `voiceover` | 口播（ASR） |
| `subtitle` | 字幕 / 花字（OCR） |
| `visual_scene` | 画面 / 场景 / 动作（VLM） |
| `product_showcase` | 商品展示（特写、上身、对比、试用等） |
| `kol_endorsement` | 达人背书（出镜身份、口头背书、资质露出） |
| `price_mechanism` | 价格机制（划线价、券、满减、限时、赠品等画面或口播） |

> **约束**：六类必须穷举覆盖；未来扩展必须先改本协议再发布，禁止直接落代码。

---

### 2.2 `VideoPersuasionRequirementDiagnosisInput`

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `product_id` | `str` | 是 | 与 `persuasion_requirement_profile.product_id` 一致；不一致必须 Crash Early |
| `video_id` | `str` | 是 | 与 `video_evidence_spans[*].video_id` 一致；不一致必须 Crash Early |
| `persuasion_requirement_profile` | `PersuasionRequirementProfile` | 是 | 来自商品诊断模块；**只读**，本模块绝对禁止重算或修改 |
| `recommended_hec` | `Optional[RecommendedHEC]` | 否 | 推荐策略的 HEC 三元组；用于判断"推荐策略 vs 视频实际表达"是否偏离，不承担一致性主诊断 |
| `video_hec_analysis` | `VideoHECAnalysis` | 是 | 视频侧 H/E/C 结构识别结果，作为证据对齐时的辅助锚点 |
| `video_evidence_spans` | `list[VideoEvidenceSpan]` | 是 | 视频可追溯证据集合；空列表允许，但必须显式给出（不允许 None） |

> **协议级强约束**：
> - `extra="forbid"`：禁止任何调用方夹带答案字段（如 `requirement_completion_results`）。
> - `profile_version` 不在 Input 显式字段中，由 `persuasion_requirement_profile.profile_version` 透传到 Result。

---

### 2.3 `RequirementCompletionResult`

单条 requirement 的诊断结果。

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `requirement_id` | `str` | 是 | 与 `persuasion_requirement_profile.requirements[*].requirement_id` 完全一致 |
| `requirement_name` | `str` | 是 | 透传商品侧名称，禁止改写 |
| `required` | `bool` | 是 | 透传商品侧 `required` 字段，禁止本模块重算 |
| `priority` | `str` (Enum) | 是 | 透传商品侧优先级，枚举：`high / medium / low` |
| `completion_status` | `str` (Enum) | 是 | 枚举：`completed / weak / missing / not_applicable` |
| `matched_evidence_spans` | `list[str]` | 是 | 元素为 `VideoEvidenceSpan.span_id`；`completed / weak` 时**不得为空**；`missing / not_applicable` 时必须为空列表 |
| `missing_reason` | `Optional[str]` | 视情况 | 约束见 §2.3.2 状态字段约束表 |
| `repair_direction` | `Optional[str]` | 视情况 | 约束见 §2.3.2 状态字段约束表 |

#### 2.3.1 `completion_status` 语义

| 值 | 含义 |
|---|---|
| `completed` | 视频中存在充分证据完成该 requirement |
| `weak` | 有相关表达但强度/清晰度/位置不足以构成有效说服 |
| `missing` | 视频中无证据触达该 requirement |
| `not_applicable` | 该 requirement 在当前视频形态/类目下不适用（必须给出 missing_reason 说明，不能用作"无法判断"的兜底） |

#### 2.3.2 状态字段约束表

| 状态 | matched_evidence_spans | missing_reason | repair_direction |
|---|---|---|---|
| completed | 非空 | None | None |
| weak | 非空 | None | 建议非空 |
| missing | 空列表 | 必填 | 必填 |
| not_applicable | 空列表 | 必填，说明不适用原因 | None |

---

### 2.4 `OverclaimRisk`

视频说了，但商品事实/要求不支持的内容。

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `risk_id` | `str` | 是 | 全局唯一 |
| `claim_text` | `str` | 是 | 视频中实际说出/展示的过度承诺原文 |
| `evidence_span_ids` | `list[str]` | 是 | 引发该过度承诺的视频证据 span_id；不得为空 |
| `risk_type` | `str` (Enum) | 是 | 枚举：`unsupported_efficacy`（无依据功效）/ `unsupported_quantification`（无依据量化）/ `unsupported_endorsement`（无依据背书）/ `regulated_term`（极限词/医疗承诺）/ `category_mismatch`（与商品事实/品类不符） |
| `severity` | `str` (Enum) | 是 | 枚举：`high / medium / low`；`regulated_term` 默认 `high` |
| `conflicting_fact_ref` | `Optional[str]` | 否 | 引用商品侧具体事实/字段路径（如 `product_fact_vector.endorsement_attribute`） |
| `repair_direction` | `Optional[str]` | 否 | 建议修复方向（删除 / 弱化 / 替换为有依据表达） |

> **强约束**：`overclaim` 必须独立输出，**绝对不允许**把 overclaim 混入 `RequirementCompletionResult.completion_status = missing`。

---

### 2.5 `RepairSuggestion`

修复建议。

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `suggestion_id` | `str` | 是 | 全局唯一 |
| `target_type` | `str` (Enum) | 是 | 枚举：`requirement / overclaim`；指明本建议是修哪类问题 |
| `target_ref` | `str` | 是 | 当 `target_type=requirement` 时填 `requirement_id`；当 `target_type=overclaim` 时填 `risk_id` |
| `action` | `str` (Enum) | 是 | 枚举：`add / strengthen / replace / remove / reposition` |
| `description` | `str` | 是 | 具体修复说明（人类可读） |
| `suggested_span_type` | `Optional[str]` | 否 | 建议落到的 span 类型，取自 §2.1.1 |
| `suggested_position` | `Optional[str]` | 否 | 建议出现的位置：`hook / effect / cta` 之一，或时间区间字符串 |

---

### 2.6 `profile_match` 前端输出 schema

`profile_match` 用于表达商品目标画像与视频目标画像之间的匹配关系。该对象直接面向前端消费，不再依赖 `available_for_frontend_mapping`。

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `status` | `str` (Enum) | 是 | `completed / needs_review / insufficient_evidence` |
| `product_audience.primary` | `str` | 是 | 来自 Product FactPack / 商品侧诊断，不得读取视频侧字段；`completed / needs_review` 下不得为空 |
| `product_audience.scene` | `str` | 建议必填 | 来自 Product FactPack / 商品侧诊断 |
| `product_audience.core_need` | `str` | 是 | 来自 Product FactPack / 商品侧诊断；`completed / needs_review` 下不得为空 |
| `video_audience.primary` | `str` | 是 | 来自 Video FactPack / 视频侧诊断，不得读取商品侧字段；`completed / needs_review` 下不得为空 |
| `video_audience.scene` | `str` | 建议必填 | 来自 Video FactPack / 视频侧诊断 |
| `video_audience.core_need` | `str` | 是 | 来自 Video FactPack / 视频侧诊断；`completed / needs_review` 下不得为空 |
| `gap.level` | `str` (Enum) | 是 | `high / medium / low` |
| `gap.description` | `str` | 是 | `completed / needs_review` 下不得为空 |
| `match_result` | `str` (Enum) | 是 | `high_match / partial / mismatch` |
| `evidence[]` | `list[object]` | 是 | 每条包含 `source / field / value`；`source` 仅允许 `product_factpack / video_factpack` |
| `summary` | `str` | 建议必填 | `insufficient_evidence` 下必须说明缺失原因 |

**后置断言**：

1. `available_for_frontend_mapping` 已废弃，任意输出命中该字段必须 Crash Early。
2. `product_audience` 只能来自 Product FactPack，`video_audience` 只能来自 Video FactPack，两侧不得混用。
3. `insufficient_evidence` 状态下必填 string 字段允许为空，但 `summary` 必须说明缺失原因。
4. `completed / needs_review` 状态下必填 string 字段不得为空。
5. `completed` 状态下 `evidence` 必须同时覆盖 `product_factpack` 和 `video_factpack`。

---

### 2.7 `VideoPersuasionRequirementDiagnosisResult`

| 字段 | 类型 | 必填 | 说明 / 约束 |
|---|---|---|---|
| `product_id` | `str` | 是 | 与 Input 一致 |
| `video_id` | `str` | 是 | 与 Input 一致 |
| `profile_version` | `str` | 是 | 透传自 `persuasion_requirement_profile.profile_version`，用于版本溯源 |
| `overall_completion_status` | `str` (Enum) | 是 | 枚举：`fully_completed / partially_completed / failed`；推导规则见 §2.6.1 |
| `requirement_completion_results` | `list[RequirementCompletionResult]` | 是 | 必须逐条覆盖 `persuasion_requirement_profile.persuasion_requirements` 中每条 active requirement；`not_applicable_requirements` 不进入 `requirement_completion_results`，除非后续协议显式要求诊断不适用项 |
| `missing_required_requirements` | `list[str]` | 是 | 元素为 `requirement_id`；仅收录 `required=true 且 completion_status=missing` 的项；**必须独立汇总** |
| `weak_requirements` | `list[str]` | 是 | 元素为 `requirement_id`；收录所有 `completion_status=weak` 的项 |
| `overclaim_risks` | `list[OverclaimRisk]` | 是 | 独立列表，禁止与 missing 合并 |
| `repair_suggestions` | `list[RepairSuggestion]` | 是 | 可为空列表（如全部 completed 时） |

#### 2.6.1 overall_completion_status 推导规则

- 所有 `required=true` 的 requirement 均为 completed，且无 high severity overclaim → `fully_completed`
- 存在 `required=true` 且 `completion_status=missing` 的 requirement → `failed`
- 存在 high severity overclaim → `failed`
- 其他情况 → `partially_completed`

---

## 3. 验收标准（6 条硬约束）

| # | 验收标准 | 校验方式（Step 2 实装时落地） |
|---|---|---|
| 1 | 输入 `persuasion_requirement_profile` 后，诊断模块**不得重算 requirement** | 代码级断言：禁止本模块对 `requirement_id / required / priority / requirement_name` 写操作；diff 后不得变化 |
| 2 | 每个 `required=true` 的 requirement **必须出现在诊断结果中** | 后置断言：`{r.requirement_id for r in profile if r.required} ⊆ {r.requirement_id for r in result.requirement_completion_results}` |
| 3 | `missing_required_requirements` **必须单独汇总** | 后置断言：等价于 `{r.requirement_id for r in result.requirement_completion_results if r.required and r.completion_status == "missing"}` |
| 4 | `completed / weak` 判断**必须有** `matched_evidence_spans` | 后置断言：状态为 `completed/weak` 时 `len(matched_evidence_spans) >= 1`，且每个 span_id 必须能在 Input.video_evidence_spans 中找到 |
| 5 | `missing` **必须给出** `missing_reason` 和 `repair_direction` | 后置断言：状态为 `missing` 时两字段非空；其他状态时两字段必须为 None |
| 6 | `overclaim` 风险**必须独立输出**，不能混在 `missing` 里 | 后置断言：`overclaim_risks` 非空时不允许把同一文案再以 `missing` 形式出现；状态机互斥校验 |

> 6 条全部为**硬断言**，违反任意一条必须 Crash Early，禁止静默放行。

---

## 4. P1 诊断规则层 + LLM 层边界（Step 3 / Step 4 实装时遵守）

> 本节先**定边界**，不写规则细节。

### 4.1 规则层（硬约束，不依赖 LLM）

规则层负责**确定性、可验证、可回归**的判断：

1. **结构性覆盖校验**：每条 `required=true` requirement 是否都出现在结果中（验收 #2）。
2. **互斥与字段一致性**：completion_status 与 matched_evidence_spans / missing_reason / repair_direction 的强关联（验收 #4 #5）。
3. **span_id 完整性**：所有引用的 span_id 必须存在于 Input.video_evidence_spans。
4. **价格/极限词/合规红线检测**：基于词表与正则识别 `regulated_term` 类 overclaim（对齐 Gate 4 三层检查清单 — 脚本级 / 分镜级 / CTA 级）。
5. **品牌资产白名单路由**：涉及"信任存量""品牌资产"判断时，强制走 `memory/topics/brand_whitelist.csv`，禁止 LLM 主观判断。
6. **状态汇总与一致性**：missing_required_requirements / weak_requirements 的汇总等价性。
7. **overall_completion_status 推导**（见 §2.6.1）：按协议推导规则执行。

### 4.2 LLM 层（受限判断）

LLM 层只允许做**不能用规则枚举的语义判断**：

1. **证据语义对齐**：判断某 span 的 content_text 是否在语义上完成了某 requirement（输出 completed / weak / missing）。
2. **过度承诺识别**：判断 claim_text 与商品事实是否存在语义不符（生成 `OverclaimRisk.risk_type` 候选）。
3. **修复方向生成**：生成 `repair_direction` 与 `RepairSuggestion.description` 的人类可读文本。

#### 4.2.1 LLM 层强制约束

- **禁止改写商品侧字段**：requirement_id / required / priority / profile_version 等只读。
- **禁止生成无证据结论**：所有 `completed / weak` 判断必须挂上 `matched_evidence_spans`，否则规则层后置断言会拦截（对齐"差异化卖点 evidence → conclusion 语义支撑校验"铁律）。
- **禁止反向放行**：不允许把 LLM 自己生成的 conclusion 拼回 evidence 做反向支撑。
- **LLM-as-Judge 二次质检**：在结果产出后，必须走独立 Judge 节点核验"evidence → conclusion 是否真支撑"，命中即降级为 `weak` 或转 `missing`。

---

## 5. 开发顺序

| Step | 目标 | 交付物 | 是否本次范围 |
|---|---|---|---|
| **Step 1** | **协议定字段**（本 PRD） | 本文档 + `schema/video_persuasion_requirement_diagnosis.py` 的 Pydantic 模型骨架（`extra="forbid"`，仅字段，无逻辑） | ✅ 本次仅 PRD |
| Step 2 | 协议级断言与契约测试 | `tests/test_protocol_video_persuasion_requirement_diagnoser.py`：6 条验收标准的契约用例（含反例 / Crash Early 场景） | ❌ 后续 |
| Step 3 | 规则层实装 | `engines/video_persuasion_requirement_diagnoser/rules.py`：§4.1 全部规则 + 后置断言；不接 LLM | ❌ 后续 |
| Step 4 | LLM 层实装 + Judge 闭环 | LLM 调用层 + LLM-as-Judge 独立质检节点；与规则层做最终一致性校验后输出 Result | ❌ 后续 |

> **顺序铁律**：严格遵循"先更新 PRD → 再改代码 → 最后双向对比 PRD 与代码"。Step 2 之前不允许写任何业务逻辑代码；Step 3 完成前不允许接入 LLM。

---

## 6. 与既有模块的协议关系

```text
[商品诊断模块]
    └─> persuasion_requirement_profile (SSOT, 只读)
          │
          ▼
[video-understanding / FactPack 链路]
    └─> video_hec_analysis + video_evidence_spans
          │
          ▼
[本模块 video_persuasion_requirement_diagnoser]
    └─> VideoPersuasionRequirementDiagnosisResult
          │
          ▼
[下游：脚本/分镜重写、人审、效果归因]
```

- 上游污染（profile 被改写、spans 缺字段、HEC 字段越界）→ Crash Early。
- 下游消费方只能消费 Result，不允许反向修改 Input。

---

## 7. 变更记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| 2026-06-19 | v1.0 | Step 1 协议字段初版 | 师诗 |
