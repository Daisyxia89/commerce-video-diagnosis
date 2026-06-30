## 背景
本次提交聚焦 JTBD Diagnosis MVP v1，补齐 scope gate、field provenance 与 anti-pollution guards，确保诊断链路在字段来源、污染拦截和任务域路由上具备更强的可解释性与稳定性。

## 主要改动
- 增强 `product_diagnoser` 的 JTBD 诊断与路由逻辑
- 补充 `product_variant_assembler`、`module3_intent_derivation` 的字段衔接与推导约束
- 新增/强化 schema 断言与 anti-pollution 校验
- 更新协议定义与输入 schema 文档
- 补充 audience taxonomy、video diagnoser 与相关测试文件
- 更新关键词规则与样例输出，便于端到端验证

## 影响范围
- `commerce_video_diagnosis/understanding/engines/*`
- `commerce_video_diagnosis/understanding/validators/*`
- `commerce_video_diagnosis/understanding/schemas/*`
- `docs/*`
- `tests/*`

## 验证建议
- 运行相关单测：`tests/test_product_target_audience.py`、`tests/test_video_diagnoser.py`
- 重点检查 scope gate、字段 provenance、污染字段拦截是否符合预期
- 核对样例输出与 schema 文档保持一致

## PR 标题
JTBD Diagnosis MVP v1: scope gate, field provenance, anti-pollution guards
