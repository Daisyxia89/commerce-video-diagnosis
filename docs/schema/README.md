# JSON Schema Drafts

本目录承载 B2 阶段的 machine-readable JSON Schema 草案。

当前文件：
- `input.schema.json`：统一输入协议
- `output.schema.json`：统一输出协议
- `error.schema.json`：标准错误对象
- `hec-dictionary.schema.json`：HEC 字典条目结构
- `product-diagnosis.schema.json`：商品诊断层结构

说明：
1. 当前 Schema 基于 `docs/input_output_schema.md`、`docs/hec_dictionary.md`、`docs/product_diagnosis_dictionary.md` 的现有草案生成；
2. HEC 标签枚举已同步到最新版字典：`H1-H7`、`E0-E7`、`C1-C5`；
3. 商品诊断中的 `difference_type`、阻力四象限、品类卡点等，按当前文档中的推荐枚举先落成程序化约束；
4. 若后续字典补齐，需要同步更新这些 JSON Schema，保持文档与程序约束一致；
5. `input.schema.json` 中 `basic_product_info.field_provenance` 已作为字段级 provenance 协议入口，其中 `core_selling_points` 只能来自 `caller_product_info` 或 `product_detail`；若标记为 `video_extracted_candidate`，系统必须 Crash Early。
