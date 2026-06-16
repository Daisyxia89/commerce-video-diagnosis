# commerce-video-diagnosis Input / Output Schema

## 1. 文档目标

本文档定义 `commerce-video-diagnosis` 的统一输入输出协议，供以下场景使用：
- GitHub 对外 README / Quick Start 引用
- SDK / CLI / API 接入方开发参考
- 抽取层与理解层的统一协议约束
- 开发者模式、回放模式与调试模式的结构化输出标准

本文档只定义：
- 输入对象结构（Input Schema）
- 输出对象结构（Output Schema）
- 必填字段、字段语义、模式边界

本文档不包含：
- provider 厂商的详细配置说明
- 内部实现逻辑
- 完整业务标签词典全文

相关文档：
- HEC 字典：`docs/hec_dictionary.md`
- 商品诊断字典：`docs/product_diagnosis_dictionary.md`

---

## 2. 总体设计原则

### 2.1 统一入口
系统对外只提供一个统一入口：`commerce-video-diagnosis`。

### 2.2 双输入模式
输入只允许两种模式：
- `video`
- `factpack`

### 2.3 商品信息外部显式输入
调用方必须显式传入 `product_info`。  
系统不依赖内部商品库，不依赖 `source_product_id` 自动查表，也不自动补齐商品标题、价格或店铺名。

### 2.4 输出分层
输出拆成两层：
- **默认层**：面向业务阅读
- **开发者层**：面向回放、调试、二次开发

### 2.5 HEC 输出规范
`diagnosis` 层属于业务阅读层，因此 HEC 字段默认同时输出：
- `label code`
- `label_name`（中文名）

例如：
- `hook_label: "H5"`
- `hook_label_name: "反常识与悬念"`

---

## 3. Input Schema

## 3.1 顶层结构

```json
{
  "request_id": "REQ_xxx",
  "input_mode": "video | factpack",
  "video": {
    "video_path": "local/path/to/video.mp4",
    "video_url": "https://example.com/video.mp4"
  },
  "fact_pack": {},
  "product_info": {
    "basic_product_info": {
      "leaf_category": "蓬松喷雾",
      "shop_name": "示例店铺",
      "item_name": "示例商品标题",
      "price": 79.9,
      "core_selling_points": ["快速蓬松定型", "无胶感不粘腻"]
    },
    "full_product_info": {
      "target_people": ["细软塌发人群"],
      "differentiator": "一喷一吹即可快速蓬松，且不粘腻"
    },
    "product_fact_vector": {
      "product_task": "降本增效/懒人替代",
      "cognitive_attribute": "低认知门槛",
      "frequency_attribute": "高频",
      "trust_attribute": "中",
      "price_attribute": "中",
      "endorsement_attribute": "无明确背书",
      "channel_risk_attribute": "低"
    }
  },
  "providers": {
    "asr": {},
    "vlm": {},
    "ocr": {}
  },
  "options": {
    "include_factpack": true,
    "include_blueprint": true,
    "include_trace": false,
    "include_provenance": false
  }
}
```

---

## 3.2 顶层字段定义

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 必填 | 请求唯一标识 |
| `input_mode` | string | 必填 | 输入模式，只允许 `video` 或 `factpack` |
| `video` | object | 条件必填 | 当 `input_mode=video` 时必填 |
| `fact_pack` | object | 条件必填 | 当 `input_mode=factpack` 时必填 |
| `product_info` | object | **必填** | 商品信息，必须由调用方显式提供 |
| `providers` | object | `video` 模式必填 | provider 配置 |
| `options` | object | 选填 | 控制输出层级 |

---

## 3.3 `input_mode` 枚举定义

### 3.3.1 `video`
表示调用方输入的是原始视频或视频 URL。  
系统需要执行：
- preprocess
- extract
- factpack build
- understanding
- diagnosis output

### 3.3.2 `factpack`
表示调用方已经准备好 FactPack。  
系统跳过上游抽取，直接执行：
- understanding
- diagnosis output

---

## 3.4 `video` 对象定义

```json
{
  "video_path": "local/path/to/video.mp4",
  "video_url": "https://example.com/video.mp4"
}
```

### 字段说明

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `video_path` | string | 二选一 | 本地视频路径 |
| `video_url` | string | 二选一 | 公网视频 URL |

### 约束
- `video_path` 与 `video_url` 至少提供一个；
- 两者同时提供时，优先级应在实现层明确；
- `input_mode=factpack` 时，不允许依赖 `video` 作为主输入。

---

## 3.5 `fact_pack` 对象定义

```json
{
  "video_meta": {
    "source_platform": "douyin",
    "duration_sec": 18.5,
    "fps": 25,
    "resolution": "1080x1920"
  },
  "segments": []
}
```

### 最低字段要求

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `video_meta` | object | 必填 | 视频元信息 |
| `segments` | array | 必填 | 分镜级结构化事实 |

### `video_meta` 子字段

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `source_platform` | string | 必填 | 视频来源平台 |
| `duration_sec` | number | 必填 | 视频时长 |
| `fps` | number | 必填 | 帧率 |
| `resolution` | string | 必填 | 分辨率 |

### `segments[*]` 最低字段

```json
{
  "segment_id": "SEG01",
  "start_sec": 0.0,
  "end_sec": 3.5,
  "visual_facts": {},
  "audio_facts": {},
  "ocr_facts": [],
  "rhythm_facts": {}
}
```

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `segment_id` | string | 必填 | 分镜 ID |
| `start_sec` | number | 必填 | 起始时间 |
| `end_sec` | number | 必填 | 结束时间 |
| `visual_facts` | object | 必填 | 视觉事实 |
| `audio_facts` | object | 必填 | 音频事实 |
| `ocr_facts` | array | 必填 | OCR 事实 |
| `rhythm_facts` | object | 必填 | 节奏与转场事实 |

---

## 3.6 `product_info` 对象定义（强制外部输入）

```json
{
  "basic_product_info": {
    "leaf_category": "蓬松喷雾",
    "shop_name": "示例店铺",
    "item_name": "示例商品标题",
    "price": 79.9,
    "core_selling_points": ["快速蓬松定型", "无胶感不粘腻"]
  },
  "full_product_info": {
    "target_people": ["细软塌发人群", "需要快速打理发型的人群"],
    "differentiator": "一喷一吹即可快速蓬松，且无胶感不粘腻"
  },
  "product_fact_vector": {
    "product_task": "降本增效/懒人替代",
    "cognitive_attribute": "低认知门槛",
    "frequency_attribute": "高频",
    "trust_attribute": "中",
    "price_attribute": "中",
    "endorsement_attribute": "无明确背书",
    "channel_risk_attribute": "低"
  }
}
```

### 字段分层

`product_info` 是**输入层商品事实协议**，只允许承载前 3 层：

1. **基础商品信息** `basic_product_info`
2. **完整商品信息** `full_product_info`
3. **商品事实向量** `product_fact_vector`

#### 1）`basic_product_info`

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `leaf_category` | string | **必填** | 叶子类目 / 最终商品类目 |
| `shop_name` | string | **必填** | 店铺名 |
| `item_name` | string | **必填** | 商品名 / 标题主名 |
| `price` | number | **必填** | 商品价格 |
| `core_selling_points` | array[string] | **必填** | 核心卖点列表 |

#### 2）`full_product_info`

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `target_people` | array[string] | **必填** | 目标人群 |
| `differentiator` | string | **必填** | 商品差异化卖点总结 |

#### 3）`product_fact_vector`

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `product_task` | string | **必填** | 商品核心任务 |
| `cognitive_attribute` | string | **必填** | 认知门槛属性 |
| `frequency_attribute` | string | **必填** | 购买 / 使用频次属性 |
| `trust_attribute` | string | **必填** | 信任存量属性 |
| `price_attribute` | string | **必填** | 价格带属性 |
| `endorsement_attribute` | string | **必填** | 背书属性 |
| `channel_risk_attribute` | string | **必填** | 渠道风险属性 |

### 强约束
1. `product_info` 是统一 skill 的**强制输入对象**；
2. 无论 `input_mode=video` 还是 `factpack`，都必须传；
3. 不允许系统内部自动查商品库补齐；
4. 不允许仅凭 `source_product_id` 自动反查商品标题、类目、价格、店铺名；
5. 若 `product_info` 缺失、前 3 层不完整或字段为空，系统必须直接报错；
6. `CandidateSet` **不是输入字段**，不得提前混入 `product_info`。

### 模块边界说明
- 模块 1 / 2 / 3 可消费 `product_info` 的前三层商品事实；
- `CandidateSet` 属于**模块 3 输出给模块 4 的候选表达协议层**，不属于调用方输入；
- 调用方若把 `category_strategy_intent`、`product_strategy_intent`、`intent_coordinates`、`modifiers` 等中间变量透传进输入，系统必须视为协议污染并 Crash Early。

---

## 3.7 `providers` 对象定义

```json
{
  "asr": {
    "provider": "aliyun_asr",
    "endpoint": "https://...",
    "api_key": "<YOUR_API_KEY>",
    "model": "paraformer-v2"
  },
  "vlm": {
    "provider": "openai_compatible_vlm",
    "endpoint": "https://...",
    "api_key": "<YOUR_API_KEY>",
    "model": "qwen-vl-max"
  },
  "ocr": {
    "provider": "openai_compatible_vlm",
    "endpoint": "https://...",
    "api_key": "<YOUR_API_KEY>",
    "model": "qwen-vl-max"
  }
}
```

### 说明
- `video` 模式下，`providers` 必须完整可用；
- `factpack` 模式下，可不要求 `providers`，除非启用了额外运行能力；
- provider 的详细 schema 可拆到独立 provider 文档中说明。

---

## 3.8 `options` 对象定义

```json
{
  "include_factpack": true,
  "include_blueprint": true,
  "include_trace": false,
  "include_provenance": false
}
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `include_factpack` | boolean | `true` | 是否输出 FactPack |
| `include_blueprint` | boolean | `true` | 是否输出 VideoBlueprint |
| `include_trace` | boolean | `false` | 是否输出运行 trace |
| `include_provenance` | boolean | `false` | 是否输出 provenance |

---

## 4. Input Validation Rules

### 4.1 模式互斥规则
- `input_mode=video` 时：
  - 必须有 `video`
  - 可以没有 `fact_pack`
- `input_mode=factpack` 时：
  - 必须有 `fact_pack`
  - 不应依赖 `video` 作为主输入

### 4.2 商品信息规则
- `product_info` 不能为空；
- `basic_product_info` / `full_product_info` / `product_fact_vector` 缺任一层时直接报错；
- `leaf_category` / `shop_name` / `item_name` / `price` / `core_selling_points` / `target_people` / `differentiator` / `product_task` 等核心字段缺失时直接报错；
- `CandidateSet` 不得出现在输入侧 `product_info` 中；
- 系统不做内部商品库兜底。

### 4.3 provider 规则
- `video` 模式必须配置可用 provider；
- 若未配置 provider，按 fallback 策略处理；
- 外部环境未配置 provider 时必须明确报错。
- 若调用 extractor runtime 配置对象，统一使用 `runtime.provider_fallback_mode` 表达 fallback 协议意图；默认值为 `force_off`。
- `runtime.provider_fallback_mode` 只允许 `auto / force_on / force_off`；其中 `auto` 与 `force_on` 为协议保留位，当前公开版无内置 fallback 实现，仅用于控制报错语义与 trace 记录。

---

## 5. Output Schema

## 5.1 顶层结构

```json
{
  "diagnosis": {
    "video_summary": {},
    "primary_hec": {},
    "secondary_effects": [],
    "persuasion_chain": "",
    "signal_scores": {},
    "segment_summary": [],
    "risk_notes": [],
    "product_diagnosis": {
      "target_people": [],
      "difference_type": "functional_result",
      "differentiator": "",
      "candidate_set": {
        "product_task": "降本增效/懒人替代",
        "main_persuasion_route": "先建立即时效果感知，再承接教程型使用方法",
        "r_rule": "优先走功能结果 + 使用演示路线",
        "p_rule": "面向细软塌发且追求快速打理的人群",
        "candidate_h_pool": ["H5"],
        "core_e_list": ["E1", "E5"],
        "core_c_list": ["C1"]
      }
    }
  },
  "blueprint": {},
  "fact_pack": {},
  "workflow_report": {},
  "provenance_report": [],
  "trace": {}
}
```

---

## 5.2 输出分层说明

### 5.2.1 默认层：`diagnosis`
面向业务同学、运营、内容策略、投放分析。

### 5.2.2 开发者层
包括：
- `blueprint`
- `fact_pack`
- `workflow_report`
- `provenance_report`
- `trace`

面向：
- 调试
- 回放
- 自动化接入
- 二次开发

---

## 5.3 `diagnosis` 对象定义

```json
{
  "video_summary": {
    "platform": "douyin",
    "duration_sec": 71.7,
    "segment_count": 10,
    "bundle_count": 7
  },
  "primary_hec": {
    "hook_label": "H5",
    "hook_label_name": "反常识与悬念",
    "effect_label": "E5",
    "effect_label_name": "保姆级教程",
    "cta_label": "C1",
    "cta_label_name": "利益/价格逼单",
    "reason": "..."
  },
  "secondary_effects": [
    {
      "effect_label": "E1",
      "effect_label_name": "效果测评",
      "reason": "..."
    }
  ],
  "persuasion_chain": "...",
  "signal_scores": {
    "visual": {
      "score": 2,
      "interpretation": "..."
    },
    "audio": {
      "score": 8,
      "interpretation": "..."
    },
    "proof": {
      "score": 4,
      "interpretation": "..."
    },
    "cta": {
      "score": 8,
      "interpretation": "..."
    }
  },
  "segment_summary": [],
  "risk_notes": []
}
```

### 子字段说明

#### `video_summary`
| 字段 | 类型 | 说明 |
|---|---|---|
| `platform` | string | 视频平台 |
| `duration_sec` | number | 视频时长 |
| `segment_count` | integer | 分镜数 |
| `bundle_count` | integer | bundle 数 |

#### `primary_hec`
| 字段 | 类型 | 说明 |
|---|---|---|
| `hook_label` | string | 主 Hook 标签代码 |
| `hook_label_name` | string | 主 Hook 标签中文名 |
| `effect_label` | string | 主 Effect 标签代码 |
| `effect_label_name` | string | 主 Effect 标签中文名 |
| `cta_label` | string | 主 CTA 标签代码 |
| `cta_label_name` | string | 主 CTA 标签中文名 |
| `reason` | string | 判定理由 |

#### `secondary_effects[*]`
| 字段 | 类型 | 说明 |
|---|---|---|
| `effect_label` | string | 副 Effect 标签代码 |
| `effect_label_name` | string | 副 Effect 标签中文名 |
| `reason` | string | 判定理由 |

#### `persuasion_chain`
- 类型：`string`
- 含义：视频主要说服链路的人类可读摘要

#### `signal_scores`
每个维度结构统一为：

```json
{
  "score": 8,
  "interpretation": "..."
}
```

支持维度：
- `visual`
- `audio`
- `proof`
- `cta`

#### `segment_summary[*]`

```json
{
  "segment_id": "SEG01",
  "start_sec": 0.0,
  "end_sec": 3.5,
  "hec_tag": "H5",
  "hec_tag_name": "反常识与悬念",
  "persuasion_function": "通过错误吹发方式制造认知冲突，引发观众继续看",
  "asr_summary": "不要再这样吹头发..."
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `segment_id` | string | 分镜 ID |
| `start_sec` | number | 起始时间 |
| `end_sec` | number | 结束时间 |
| `hec_tag` | string | 本地 H/E/C 标签代码 |
| `hec_tag_name` | string | 本地 H/E/C 标签中文名 |
| `persuasion_function` | string | 说服功能摘要 |
| `asr_summary` | string | 口播摘要 |

#### `risk_notes`
- 类型：`array[string]`
- 含义：风险、限制、备注说明

#### `product_diagnosis`
- 类型：`object`
- 含义：商品诊断层输出，完整字段以 `docs/product_diagnosis_dictionary.md` 与 `docs/schema/product-diagnosis.schema.json` 为准。
- 关键约束：
  - 前 3 层商品事实来自输入侧 `product_info`；
  - `candidate_set` 是**模块 3 输出给模块 4** 的候选表达协议层；
  - 模块 4 只允许消费已经裁决完成的 `candidate_set`，不得依赖 `category_strategy_intent`、`product_strategy_intent`、`intent_coordinates`、`modifiers` 等中间变量；
  - 若输出中混入上述中间变量，应视为协议越界并直接报错。

---

## 5.4 HEC 简版枚举表（摘要）

> 完整定义见 `docs/hec_dictionary.md`。

### 5.4.1 Hook 示例
| Code | 中文名 | 说明 |
|---|---|---|
| `H5` | 反常识与悬念 | 通过行为猎奇、反常规观点、认知冲突或质疑验证制造信息差 |

### 5.4.2 Effect 示例
| Code | 中文名 | 说明 |
|---|---|---|
| `E1` | 效果测评 | 围绕明确验证目标进行常规、可理解的效果实测 |
| `E5` | 保姆级教程 | 以步骤教学、方法说明或纠错讲解承担主要说服 |
| `E6` | 成分/参数科普 | 以成分、参数、机理或专有名词完成理性论证 |

### 5.4.3 CTA 示例
| Code | 中文名 | 说明 |
|---|---|---|
| `C1` | 利益/价格逼单 | 通过绝对低价、价格落差或限时优惠推动转化 |

---

## 5.5 `blueprint` 对象定义

`blueprint` 面向开发者与下游系统，承接标准化视频理解结果。

最低字段建议如下：

```json
{
  "blueprint_id": "BP_xxx",
  "video_id": "VID_xxx",
  "primary_hec": {
    "hook_label": "H5",
    "effect_label": "E5",
    "cta_label": "C1"
  },
  "secondary_effects": [],
  "slider_signature": {},
  "storyboard_source": "segments",
  "semantic_bundles": [],
  "segment_to_bundle_map": {},
  "bundle_to_segment_range": {},
  "storyboard_segments": []
}
```

### 说明
`blueprint` 是结构化资产，不要求业务用户直接阅读。  
它主要为：
- 回放
- 素材管理
- 二次分析
- 下游脚本系统
提供统一协议。

---

## 5.6 `fact_pack` 对象定义

`fact_pack` 输出可选返回，用于：
- 抽取结果回放
- 抽取层排障
- 二次开发

其结构应与输入 `fact_pack` 协议保持一致。

---

## 5.7 `workflow_report` 对象定义

```json
{
  "workflow_version": "v1",
  "request_id": "REQ_xxx",
  "video_id": "VID_xxx",
  "blueprint_id": "BP_xxx",
  "gate_checks": [],
  "stage_sequence": []
}
```

### 字段说明
| 字段 | 类型 | 说明 |
|---|---|---|
| `workflow_version` | string | 工作流版本 |
| `request_id` | string | 请求 ID |
| `video_id` | string | 视频 ID |
| `blueprint_id` | string | 蓝图 ID |
| `gate_checks` | array | Gate 校验结果 |
| `stage_sequence` | array | 执行阶段序列 |

---

## 5.8 `provenance_report` 对象定义

- 类型：`array[object]`
- 含义：记录关键输出字段的来源链路

示例：

```json
[
  {
    "field_path": "blueprint.primary_hec",
    "producer_type": "system_native_inference",
    "source_type": "fact_pack.segments[*]",
    "source_refs": ["SEG01", "SEG02"]
  }
]
```

### 子字段说明
| 字段 | 类型 | 说明 |
|---|---|---|
| `field_path` | string | 输出字段路径 |
| `producer_type` | string | 生产方式 |
| `source_type` | string | 来源类型 |
| `source_refs` | array | 来源引用 |

---

## 5.9 `trace` 对象定义

`trace` 面向调试与排障，不作为默认业务阅读内容。

建议最低包含：

```json
{
  "provider_resolution_trace": {
    "environment_mode": "external_public",
    "fallback_protocol_mode": "force_off",
    "asr": {
      "selected_provider_mode": "byok",
      "provider_name": "aliyun_asr",
      "fallback_used": false,
      "fallback_reason": ""
    },
    "vlm": {
      "selected_provider_mode": "byok",
      "provider_name": "openai_compatible_vlm",
      "fallback_used": false,
      "fallback_reason": ""
    },
    "ocr": {
      "selected_provider_mode": "byok",
      "provider_name": "openai_compatible_vlm",
      "fallback_used": false,
      "fallback_reason": ""
    }
  }
}
```

---

## 6. Output Validation Rules

### 6.1 默认层必须存在
无论调用方是否要求开发者层输出，默认层 `diagnosis` 必须存在。

### 6.2 开发者层按 options 控制
- `include_factpack=true` 时返回 `fact_pack`
- `include_blueprint=true` 时返回 `blueprint`
- `include_trace=true` 时返回 `trace`
- `include_provenance=true` 时返回 `provenance_report`

### 6.3 业务输出与结构化输出语义必须一致
- `diagnosis.primary_hec` 应与 `blueprint.primary_hec` 保持一致；
- `diagnosis.segment_summary` 应与 `blueprint.storyboard_segments` 的语义一致；
- 不允许“业务摘要”和结构化标签互相冲突。

---

## 7. 标准错误对象建议

建议统一错误结构：

```json
{
  "error": {
    "code": "PROVIDER_NOT_CONFIGURED",
    "message": "ASR provider 未配置，且公开仓库不再内置 fallback 执行路径。请在 providers.asr 中配置 BYOK provider。",
    "details": {
      "field": "providers.asr",
      "input_mode": "video"
    }
  }
}
```

### 推荐错误码
- `INVALID_INPUT_MODE`
- `MISSING_VIDEO_INPUT`
- `MISSING_FACTPACK_INPUT`
- `MISSING_PRODUCT_INFO`
- `INCOMPLETE_PRODUCT_INFO`
- `PROVIDER_NOT_CONFIGURED`
- `INTERNAL_FALLBACK_DISABLED`
- `FACTPACK_SCHEMA_INVALID`
- `UNDERSTANDING_FAILED`

---

## 8. 一句话结论
这套 Schema 的核心边界是：
1. **输入只允许 `video` / `factpack` 两种模式**；
2. **`product_info` 必须由调用方显式提供**，系统不做内部商品查表；
3. **输出分默认业务层和开发者结构层**；
4. **默认层保证好读，开发者层保证可回放、可追踪、可二次开发。**
