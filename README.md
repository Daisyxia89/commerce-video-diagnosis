# Commerce Video Diagnosis

`commerce-video-diagnosis` 是面向电商短视频分析场景的统一入口。它既可以从原始视频或外部抽取结果出发，先构建合法 `FactPack`，再继续完成视频理解与商品诊断；也可以在你已经持有 `FactPack` 时，直接产出结构化诊断结果。

如果你需要判断一条电商视频是如何起钩、如何完成说服、如何收口转化，或者希望把视频理解结果与商品事实放进同一套协议里统一分析，这个仓库就是公开版入口。

## 这个仓库解决什么问题

它主要覆盖两类任务：

1. **从原始视频出发完成两阶段诊断。**
   你可以把本地视频或视频 URL 输入给 Extractor 阶段，先得到结构化 `FactPack`，再进入 Understanding 阶段输出诊断结果。
2. **从已有 FactPack 直接进入诊断。**
   如果你已经有外部抽取结果，或者上游系统已经产出合法 `FactPack`，可以跳过抽取，直接执行视频理解与商品诊断。

统一输出围绕以下几类对象展开：

- `diagnosis`：默认业务阅读层，适合运营、内容策略、投放分析直接查看；
- `blueprint`：结构化内容蓝图，保留供下游生成或编排模块消费；
- `workflow_report`：链路级解释与过程摘要；
- `triad_assets`：供后续脚本、结构生成或分析模块消费的三元资产；
- `fact_pack / provenance_report / trace`：开发者与调试层产物。

## 什么时候适合使用

当你遇到以下场景时，适合加载这个 skill：

- 你只有一条电商视频，想判断它的结构、说服链路与主副 HEC 标签；
- 你已经有外部 ASR / OCR / VLM 结果，想把它们整理成统一 `FactPack` 后继续诊断；
- 你已经持有合法 `FactPack`，希望直接补齐视频理解、商品诊断与结构化输出；
- 你需要在统一协议下同时承接视频事实、商品事实与 HEC 结果，而不是只做抽取。

## 能力总览

### 1. 抽取层：原始视频或外部结果整理为 FactPack

公开版默认支持两类 provider 路径：

- `fixture_file`：用本地 JSON fixture 回放抽取结果；
- 外部 BYOK provider：显式配置 `endpoint + api_key + model` 后调用外部服务。

当前已提供的公开示例包括：

- OpenAI Whisper（ASR）；
- OpenAI GPT-4o vision（VLM / OCR）；
- 阿里云 `aliyun_asr`；
- 火山引擎 `volcengine_asr`。

当 provider 未配置完整时，公开版会直接报错，不做静默兜底。

### 2. 视频理解与诊断层

`FactPack` 进入 Understanding 阶段后，系统会继续完成视频结构理解与诊断，重点输出：

- 视频摘要，如平台、时长、分镜数、bundle 数；
- 主 HEC 标签 `primary_hec`，包括 `Hook / Effect / CTA` 的代码与中文名；
- 副 Effect 标签 `secondary_effects`；
- `persuasion_chain`，用于描述视频主要说服链路；
- `segment_summary`，用于说明关键分镜承担的作用；
- `risk_notes`，用于标记表达风险或诊断注意点。

默认业务阅读入口是 `diagnosis`，开发者层保留 `blueprint / fact_pack / workflow_report / provenance_report / trace` 等结构化产物。

### 3. 商品诊断协议层

如果链路涉及商品诊断，调用方必须显式提供 `product_info`。公开版不会依赖内部商品库自动补齐标题、类目、价格或店铺名。

`product_info` 只允许承载前三层商品事实：

1. **4.1 基础商品信息**：回答“这是什么商品”；
2. **4.2 完整商品信息**：回答“主要卖给谁、核心差异点是什么”；
3. **4.3 商品事实向量**：回答“这个商品在任务、门槛、风险等维度上的结构化事实”。

`CandidateSet` 属于 **4.4 候选表达协议层**，是模块 3 输出给模块 4 的中间结果，**不得由调用方预填进输入**。

### 4. 商品任务分类摘要版

`product_task` 是商品事实向量中的一级锚点，用来回答：**这个商品本质上是在帮助用户完成什么任务**。

它的作用不是描述卖点文案，而是给后续商品诊断、路线判断与 `CandidateSet` 提供稳定的任务域基线。

当前 README 展示的是 `product_task` 的一级任务标签示例，覆盖功能、情绪、社会三大类：

| `product_task` 示例 | 所属大类 | 判定锚点 |
|---|---|---|
| `降本增效/懒人替代` | 功能任务 | 用户核心诉求是更省时间、更省步骤、更省力，期待商品替自己完成原本繁琐的操作。 |
| `情绪安心/主观降险` | 情绪任务 | 用户核心诉求是降低主观担忧、获得安心感，而不是先追求显性功能提升。 |
| `礼赠与关系表达` | 社会任务 | 用户核心诉求是借商品完成送礼、表态、维系关系等社会表达。 |

如需查看完整一级任务标签集合，请以 `docs/product_diagnosis_dictionary.md` 为准。
边界约束如下：

- `product_task` 写的是**用户任务**，不是商品广告句；
- 不要把 `快速蓬松定型` 这类卖点句直接当作任务；
- 不要把 `H5`、`E5`、`C1` 这类 HEC 标签写进任务字段；
- 不要把 `适合走痛点开场`、`建议用教程承接` 这类策略判断写进任务字段。

如果你需要完整字段定义与上下游边界，请查看 `docs/product_diagnosis_dictionary.md`。

### 5. HEC 输出层

HEC 是视频诊断的核心阅读框架：

- **H = Hook**：视频如何抓住用户注意力；
- **E = Effect**：视频主要通过什么方式完成说服；
- **C = CTA**：视频如何完成收口与转化推动。

在 `diagnosis` 层中，主标签默认同时输出 `label code + label_name`。README 只保留摘要版，帮助你快速理解当前公开输出的标签范围。

#### HEC 标签概览表

| 段 | 标签 | 中文名 | 一句话说明 |
|---|---|---|---|
| H | `H1` | 痛点/焦虑直击 | 直接暴露症状、制造麻烦与焦虑。 |
| H | `H2` | 利益/价格前置 | 直接抛出价格数字、极高性价比或福利开场。 |
| H | `H3` | 反差结果前置 | 极具视觉落差的 Before & After 蜕变。 |
| H | `H4` | 即时操作展示 | 动作即看点，开局直接一镜到底顺滑操作，无情绪铺垫。 |
| H | `H5` | 反常识与悬念 | 包括行为猎奇/反常规、认知/观点冲突、质疑打假/亲测验证。 |
| H | `H6` | 场景/人群代入 | 包含剧情关系、节点时空、人群特征、热点话题代入。 |
| H | `H7` | 明星/权威同款 | 借势明星、大 V 的脸或权威身份留人，包含同款妆造等。 |
| E | `E0` | 单点演示 | 无门槛的普通状态、外观展示或常规使用步骤展示。 |
| E | `E1` | 效果测评 | 有明确验证目标的常规实测验证。 |
| E | `E2` | 暴力实测 | 高压、破坏性、反常规的极端物理测试。 |
| E | `E3` | 对比/拉踩 | 跟竞品、其他通用方案、旧方案或过去的自己进行明确优劣比较。 |
| E | `E4` | 感官实证 | 通过极致单感官输入触发大脑“通感”代偿，实现“以感代证”。 |
| E | `E5` | 保姆级教程 | 提供新方法、新认知或纠正旧方法的教程型说明。 |
| E | `E6` | 成分/参数科普 | 通过成分、参数、机理或专有名词完成理性论证。 |
| E | `E7` | 产地溯源/工厂实录 | 通过原产地、种植/养殖基地、工厂流水线等源头事实建立信任。 |
| C | `C1` | 利益/价格逼单 | 用绝对低价或强烈的价格落差进行逼单。 |
| C | `C2` | 福利/保障机制 | 提供超预期价值或售后兜底。 |
| C | `C3` | 指令行动 | 下达极简的物理操作指令。 |
| C | `C4` | 人群/场景总结 | 在尾部再次圈定人群或情境呼应。 |
| C | `C5` | 效果留白/情绪定格 | 用冲击力画面或治愈声音自然定格。 |

在 `diagnosis` 层中的结构示例如下：

```json
{
  "primary_hec": {
    "hook_label": "H5",
    "hook_label_name": "反常识与悬念",
    "effect_label": "E5",
    "effect_label_name": "保姆级教程",
    "cta_label": "C1",
    "cta_label_name": "利益/价格逼单"
  }
}
```

如果你需要查看完整标签定义、适用边界与正反例，请继续阅读 `docs/hec_dictionary.md`。

## README 与 SKILL.md 的分工

- `README.md`：公开发布说明，面向 GitHub 访问者与接入方，重点是适用场景、能力边界、快速开始与协议导航；
- `SKILL.md`：Skill 入口说明，面向调用链与维护者，重点是路由边界、目录结构、脚本入口与强约束。

如果你是首次接入，先读本文件；如果你要把它作为 Skill 接到自动化链路里，再看 `SKILL.md`。

## 环境要求

- Python 3.10+；
- `ffmpeg`；
- `ffprobe`；
- 可访问的 ASR / VLM / OCR provider（若使用 BYOK 模式）。

## 安装

### 1. 安装 Python 依赖

```bash
cd commerce-video-diagnosis
python3 -m pip install -r requirements.txt
```

> ⚠️ **pydantic 版本要求**：本仓库的数据模型基于 pydantic **v1** 语法（`@validator` / `class Config` 等），
> `requirements.txt` 已锁定 `pydantic>=1.10,<2`。若你的全局环境已是 pydantic 2.x 且不便降级，
> 可隔离安装后通过 `PYTHONPATH` 优先加载：
>
> ```bash
> pip install --target ./vendor_pydantic1 "pydantic<2"
> PYTHONPATH=./vendor_pydantic1 python3 scripts/run_extractor.py ...
> ```

### 2. 安装系统依赖

确保运行环境中可直接执行：

```bash
ffmpeg -version
ffprobe -version
```

如果命令不可用，请先按你的操作系统安装 FFmpeg。

## 快速开始

> 所有命令均假设你已 `cd` 进入仓库根目录（即包含 `SKILL.md` / `requirements.txt` 的目录）。
> 配置文件中的相对路径会自动相对仓库根目录解析，无需关心仓库被放在什么位置。

### 场景零：零配置 fixture demo（无需任何 API Key）

```bash
python3 scripts/run_extractor.py \
  --config fixtures/p0_fixture_config.json \
  --mode two-stage-run
```

这是最快验证链路是否打通的方式：它用仓库自带的 ASR / VLM / OCR fixture 回放，
**不需要任何 API Key**，即可跑出完整的 `diagnosis / blueprint / workflow_report / triad_assets`。
建议新装环境第一步先跑它。

> 注意：fixture 回放只验证**链路**。HEC 判定、CandidateSet 等"诊断大脑"在没有 LLM 时
> 会走 `rule_fallback`（关键词兜底），结论可能不准，仅用于打通流程。详见下方"公开版边界"。

### 场景一：只执行抽取

```bash
python3 scripts/run_extractor.py \
  --config fixtures/examples/openai_whisper_gpt4o_config.json \
  --mode extract-only
```

适用于你只想验证 provider、预处理和 `FactPack` 构建是否正常的场景。（需配置 BYOK key）

### 场景二：执行完整两阶段链路

```bash
python3 scripts/run_extractor.py \
  --config fixtures/examples/openai_whisper_gpt4o_config.json \
  --mode two-stage-run
```

适用于你从原始视频起步，希望直接得到 `diagnosis / blueprint / workflow_report / triad_assets` 的场景。（需配置 BYOK key）

### 场景三：已有 FactPack，直接跑理解阶段

```bash
python3 scripts/run_v2.py \
  --payload <payload.json> \
  --output <output.json>
```

适用于你已经有合法 `FactPack`，只想执行视频理解与商品诊断的场景。

## 示例配置

可直接参考以下文件：

- `fixtures/p0_fixture_config.json`：fixture 回放示例；
- `fixtures/raw_video_regression_config.json`：原始视频回归示例配置；
- `fixtures/examples/openai_whisper_gpt4o_config.json`：BYOK 最小可运行示例；
- `fixtures/examples/aliyun_asr_config.json`：阿里云 ASR 示例（显式传 `audio_url`）；
- `fixtures/examples/aliyun_asr_upload_config.json`：阿里云 ASR 示例（OSS 自动上传）；
- `fixtures/examples/volcengine_asr_config.json`：火山引擎 ASR 示例（显式传 `audio_url`）；
- `fixtures/examples/volcengine_asr_upload_config.json`：火山引擎 ASR 示例（TOS 自动上传）。

## 输入输出协议导航

如果你正在接入或校验协议，建议按下面顺序阅读：

1. `docs/input_output_schema.md`：统一输入输出 Schema，总览 `video / factpack / product_info / diagnosis` 等对象；
2. `docs/product_diagnosis_dictionary.md`：商品诊断输入口径，解释三层商品事实与 `CandidateSet` 边界；
3. `docs/hec_dictionary.md`：HEC 体系说明与标签字典承载方式。

## 配置规则

### Provider 路由规则

每个 provider（`vlm` / `asr` / `ocr`）独立判定：

1. `provider=fixture_file`：读取本地 fixture；
2. 存在 `endpoint + api_key`：调用外部 BYOK provider；
3. 否则：按公开仓库 fallback 协议位直接报错，要求显式补齐 provider 配置。

### 外部 provider 必填字段

- `provider`；
- `adapter`；
- `endpoint`；
- `api_key`；
- `model`；
- `timeout_sec`。

### 国内 ASR 的 `audio_url` / 自动上传规则

阿里云 `aliyun_asr` 与火山引擎 `volcengine_asr` 当前都按“提交公网音频 URL -> 异步轮询结果”实现。

因此 `providers.asr` 有两种可用方式：

1. **直接提供公网 URL。**
   - `providers.asr.extra.audio_url`。
2. **只提供本地音频路径，由运行时自动上传。**
   - 当 preprocess 已产出本地 `audio_path`，且没有显式传 `extra.audio_url` 时；
   - 可在 `providers.asr.extra` 下配置自动上传参数；
   - 上传成功后，runtime 会把上传后的公网 URL 自动回填给阿里云 / 火山引擎 ASR adapter。

支持两种上传后端：

- `upload_provider = "oss"`。
  - `upload_endpoint`；
  - `upload_bucket`；
  - `upload_access_key_id`；
  - `upload_access_key_secret`；
  - `upload_object_prefix`（可选）；
  - `upload_public_base_url`（可选）。
- `upload_provider = "tos"`。
  - `upload_endpoint`；
  - `upload_bucket`；
  - `upload_region`；
  - `upload_access_key_id`；
  - `upload_access_key_secret`；
  - `upload_object_prefix`（可选）；
  - `upload_public_base_url`（可选）。

## 输出说明

### `extract-only`

返回：

- `factpack`。

若配置了 `output.factpack_path`，同时会落盘为 JSON 文件。

### `build-request`

返回：

- `request`。

### `two-stage-run`

返回：

- `result.diagnosis`；
- `result.blueprint`；
- `result.triad_assets`；
- `result.workflow_report`。

### 运行时治理产物

当走外部 provider 或 BYOK provider 时，workspace 下会生成：

- `provider_runtime/<provider_name>/runtime_state.json`；
- `provider_runtime/<provider_name>/cache/*.json`。

这些文件用于请求去重缓存、失败重试和回放诊断。

## 公开版边界

### 功能限制说明

- Understanding阶段 - 已完全可用（需配置 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，支持 BYOK 模式）

- 公开版仅支持 `fixture_file` 与外部 BYOK provider；
- 公开仓库不内置任何私有执行路径；
- `runtime.provider_fallback_mode` 保留为协议字段，但当前公开版没有内置 fallback 实现；
- `product_info` 必须由调用方显式输入；
- 调用方不得在输入中混入 `CandidateSet` 或其他中间变量；
- provider 未配置时，公开版必须显式报错，不做 silent fallback。

### ⚠️ 关于 LLM 与诊断质量（必读）

本 skill 的"诊断大脑"（HEC 标签判定、CandidateSet 生成、四维评分等）**依赖一个 OpenAI 兼容的 LLM**（BYOK，通过环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` 注入）。

- **配了 LLM**：走 LLM-first 主路径，这是设计上的正确用法，诊断质量取决于你用的模型。
- **没配 LLM**：自动退化到 `rule_fallback`（关键词硬匹配）。它只保证**链路可跑通**，
  **不保证判定准确**。实测中，rule_fallback 可能把 Hook / Effect / CTA 标签判错（例如把"跟竞品对比"
  误判为"自比"）。因此 fixture demo / 无 LLM 模式的输出仅适合验证流程与集成，
  **不要直接拿 rule_fallback 的结论当作真实诊断依据**。

输出中的 `inference_mode` 字段（例如 `blueprint.risk_flags.inference_mode`、`workflow_report.inference_mode`）会标记本次裁决是来自 LLM 还是 rule_fallback；`risk_flags.hec_reason` 会给出可读理由。请据此判断输出可信度。

## 常见报错

### `外部 provider/BYOK 模式需要 preprocess 输出`

原因：启用了非 fixture provider，但没有提供 `input.video_path`，或调用链没有先跑 preprocess。

处理：补齐 `input.video_path`，并确保本机安装了 `ffmpeg` 与 `ffprobe`。

### `provider <name> 缺少 endpoint / api_key / model`

原因：外部 BYOK provider 必填项未填完整。

处理：检查示例 config，确保对应字段已填写。

### `该 ASR adapter 需要公网可访问的 audio_url`

原因：阿里云 / 火山引擎 ASR 需要公网音频 URL；当前既没有显式传 `providers.asr.extra.audio_url`，也没有配好 OSS / TOS 自动上传参数。

处理：

- 直接传 `providers.asr.extra.audio_url`；或
- 在 `providers.asr.extra` 下补齐 `upload_provider=oss/tos` 及对应上传配置。

### `HTTP 401` / `HTTP 403`

原因：API Key 无效、过期，或 endpoint 与账号不匹配。

处理：确认 `api_key`、endpoint、模型名称和服务商控制台一致。

### `VLM 返回内容不是 JSON 对象`

原因：外部视觉模型未按 JSON schema 返回。

处理：优先使用支持 JSON mode 的 endpoint；若是兼容 OpenAI Chat Completions 的网关，确认其支持 `response_format={"type":"json_object"}`。

## 相关文档

- `SKILL.md`；
- `docs/quickstart_byok.md`；
- `docs/input_output_schema.md`；
- `docs/product_diagnosis_dictionary.md`；
- `docs/hec_dictionary.md`；
- `references/two_stage_handoff.md`。
