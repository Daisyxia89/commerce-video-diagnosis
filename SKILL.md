---
name: commerce-video-diagnosis
description: 当你需要分析电商短视频的结构、说服链路与 HEC 标签，或需要把原始视频、外部抽取结果整理为 FactPack 后继续完成视频诊断、商品诊断与结构化输出时，加载这个 skill。也适用于已经有 FactPack、希望直接产出 diagnosis、blueprint、workflow_report 与 triad_assets 的场景。
---

# Commerce Video Diagnosis Skill

`commerce-video-diagnosis` 是统一入口 skill，对外收敛为一个产品入口，对内保留两阶段引擎：

1. **Extractor 阶段**：把原始视频或外部多模态结果整理成合法 FactPack；
2. **Understanding 阶段**：基于 FactPack 执行视频理解与诊断，输出 `blueprint / workflow_report / triad_assets / provenance_report`。

## 文档定位

- `README.md`：公开发布说明，面向 GitHub 访问者与接入方，重点是安装、依赖、示例配置与公开边界；
- `SKILL.md`：Skill 入口说明，面向调用链与维护者，重点是能力边界、入口脚本、目录结构与强约束。

## 1. 统一边界

本 skill 允许两种使用方式：

### 方式 A：原始视频 / 外部抽取结果 -> FactPack -> Diagnosis
适用于需要从原始视频起步，完成完整两阶段链路的场景。

### 方式 B：已有 FactPack -> Diagnosis
适用于调用方已经有合法 FactPack，只想执行理解与诊断阶段的场景。

## 2. 目录结构

根目录包含：

- `README.md`：公开版说明；
- `docs/`：协议、字典、Schema 与补充说明；
- `extractor/`：上游抽取与 provider 编排；
- `scripts/`：统一入口、smoke、下游 runner 等脚本；
- `tests/`：单测、smoke、协议断言；
- `fixtures/`：公开样例与回放素材；
- `references/`：补充参考资料。

## 3. 公开版运行策略

### 3.1 Provider
公开版仅支持：

- `fixture_file`
- 外部 BYOK provider

公开仓库不内置任何私有执行路径。

### 3.2 Fallback 协议位
保留 `runtime.provider_fallback_mode` 作为协议字段：

- 默认值：`force_off`
- `auto` / `force_on`：协议保留位

当前公开版没有内置 fallback 实现，因此 provider 缺失时会显式报错。

## 4. 商品信息协议

若链路涉及商品诊断，调用方必须显式提供 `product_info`，并遵守四层业务字典口径：

- 4.1 基础商品信息
- 4.2 完整商品信息
- 4.3 商品事实向量
- 4.4 CandidateSet（模块 3 输出给模块 4 的候选表达协议层）

输入侧只允许承载前三层商品事实；`candidate_set` 不得由调用方预填。

## 5. 入口脚本

### 5.1 Extractor 统一入口
```bash
cd <repo-root> && python3 user_skills/commerce-video-diagnosis/scripts/run_extractor.py \
  --config <config.json> \
  --mode validate-only|extract-only|build-request|two-stage-run
```

### 5.2 Understanding 最小入口
```bash
cd <repo-root> && python3 user_skills/commerce-video-diagnosis/scripts/run_v2.py \
  --payload <payload.json> \
  --output <output.json>
```

## 6. 输出理解

- Extractor 阶段可输出：`factpack / request / result`
- Understanding 阶段可输出：`blueprint / workflow_report / triad_assets / provenance_report`
- 业务阅读层默认以 `diagnosis` 为主；开发者层保留 `fact_pack / blueprint / trace / provenance`

## 7. 推荐工作流

1. 若用户只有视频：先走 Extractor 阶段，再进入 Understanding 阶段；
2. 若用户已有 FactPack：直接走 `scripts/run_v2.py`；
3. 若要验证公开版边界是否稳定，优先跑：
   - `pytest tests/test_public_release_smoke.py -v`

## 8. 强约束

- 不允许把原始视频直接喂给下游理解引擎当作合法 payload；
- 不允许在 FactPack 中夹带答案型字段；
- 不允许输入侧混入 `candidate_set` 或模块 3 中间变量；
- provider 未配置时，公开版必须显式报错，不做 silent fallback。
