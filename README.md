# 视频诊断交付包验收说明

## 1. 本轮开发范围说明

本交付包覆盖本轮开发的 Block 1 + Block 2：

### Block 1：商品理解与目标人群判定

- 商品诊断核心逻辑：`commerce_video_diagnosis/understanding/engines/product_diagnoser.py`
- 商品目标人群分类与枚举口径：`commerce_video_diagnosis/understanding/engines/audience_taxonomy.py`
- Audience Slider 偏好字典：`commerce_video_diagnosis/understanding/engines/data/audience_slider_preference_dictionary.json`
- 商品目标人群测试：`tests/test_product_target_audience.py`

### Block 2：视频诊断与说服档案

- 视频诊断核心逻辑：`commerce_video_diagnosis/understanding/engines/video_diagnoser.py`
- 说服需求引擎：`commerce_video_diagnosis/understanding/engines/persuasion_requirement_engine.py`
- HEC 武器库快照：`commerce_video_diagnosis/understanding/data/hec_weapon_library_snapshot.json`
- 视频诊断测试：`tests/test_video_diagnoser.py`
- /JG 独立验收脚本：`jg_independent_acceptance.py`

## 2. 关键枚举口径

### 2.1 Profile Match & Audience Match

`profile_match`：用于判断视频是否覆盖商品侧 `persuasion_requirement_profile` 中 required/high requirements，即"该说的有没有说"。

枚举值：

- `matched`：视频已覆盖全部 required 且覆盖大部分 high requirements。
- `partially_matched`：视频覆盖部分 required/high requirements，但存在关键要求缺失。
- `mismatched`：视频未覆盖 required requirements，核心说服链路缺失。
- `unknown`：信息不足，无法稳定判断覆盖关系。

`audience_match`：用于判断视频实际吸引的人是否覆盖商品应该优先说服的人。

枚举值：

- `matched`：视频实际吸引的人群与商品目标人群一致。
- `partially_matched`：视频实际吸引的人群与商品目标人群存在交集，但有偏移。
- `mismatched`：视频实际吸引的人群与商品目标人群明显不一致。
- `unknown`：信息不足，无法稳定判断匹配关系。

### 2.2 Slider Match

`slider_match` 用于表达视频采用的说服滑块、偏好表达或内容侧重，是否命中目标人群的偏好要求。

核心口径：

- `matched`：视频表达与目标人群偏好一致，能支撑主要说服需求。
- `partially_matched`：视频表达覆盖部分偏好，但存在关键偏好缺失或表达不足。
- `mismatched`：视频表达与目标人群偏好明显不一致，或强化了非核心诉求。
- `unknown`：缺少可判定信息。

### 2.3 overall_status

`overall_status` 是验收层面汇总状态，综合商品理解、目标人群、视频诊断与说服档案输出。

核心口径：

- `PASS`：所有必验字段存在，字段值符合预期，关键诊断结论与 /JG 验收标准一致。
- `FAIL`：任一必验字段缺失、枚举不合法、关键结论不符合预期，或独立验收脚本断言未通过。

## 3. 润本关键验收结论

以下为润本样本验收时需要逐字段核对的预期结论。验收以 `outputs/runben_diagnosis/` 下的实际输出文件为准。

### 3.1 `runben_full_diagnosis.json`

- 文件必须存在，且为合法 JSON。
- 必须包含商品诊断、视频诊断、说服档案相关结果。
- 商品目标人群字段必须可追溯到商品理解结果。
- 视频目标人群字段必须可追溯到视频诊断结果。
- Profile Match 结果必须使用标准枚举：`matched`、`partially_matched`、`mismatched`、`unknown`。
- Slider Match 结果必须使用标准枚举：`matched`、`partially_matched`、`mismatched`、`unknown`。
- overall_status 必须使用标准枚举：`PASS` 或 `FAIL`。

### 3.2 `runben_video_diagnosis.json`

- 文件必须存在，且为合法 JSON。
- 必须包含视频目标人群判定结果。
- 必须包含视频说服链路相关诊断信息。
- 必须包含 Profile Match / Slider Match 相关字段或可被独立验收脚本稳定读取的等价字段。

### 3.3 `runben_persuasion_profile_report.md`

- 文件必须存在，且为 Markdown 文本。
- 必须包含润本样本的说服档案诊断结论。
- 必须包含 HEC 相关结论或可解释的说服动作拆解。
- 内容不得仅有结构占位，必须能支撑独立验收读取与人工复核。

## 4. 验收方法

在解压后的包根目录下运行：

```bash
cd <解压目录>
python3 jg_independent_acceptance.py
# 预期：11/11 Pass
```

若未达到 `11/11 Pass`，则本交付包不得判定为通过，需要根据脚本输出定位失败字段并修复后重新打包。

## 5. Pydantic V1 隔离安装说明

本项目的数据模型（`core_skill/schemas/protocols.py` 等）基于 **pydantic v1** 语法
（`@validator`、`class Config`、`root_validator`）。若全局环境已安装 pydantic v2 且不能降级，
请使用 `vendor_pydantic1` + `PYTHONPATH` 方式隔离：

```bash
# 1. 在项目根创建 vendor 目录并安装 pydantic v1
pip install --target ./vendor_pydantic1 "pydantic>=1.10,<2"

# 2. 运行时将 vendor 目录加到 PYTHONPATH 最前面（优先于全局 pydantic v2）
export PYTHONPATH="$(pwd)/vendor_pydantic1:$PYTHONPATH"

# 3. 验证版本
python -c "import pydantic; print(pydantic.VERSION)"  # 应输出 1.10.x

# 4. 正常运行测试
pytest tests/
```

> 提示：`vendor_pydantic1/` 已被 `.gitignore` 忽略，不会被意外提交。

## 6. Schema 备注

### `conversion_barriers`

`conversion_barriers` 为 `list[str]` 可读解释层，位于 `product_fact_vector` 内部，
用于以自然语言描述影响转化的障碍因素（如"价格水位偏高，需强化价值证明"）。

**不是结构化对象**，不替代六维结构化枚举（cognition_attribute / frequency_attribute /
trust_attribute / price_attribute / endorsement_attribute / channel_risk_attribute），
仅作补充说明供人工快速理解。

## 7. PRD 文档链接表

<table header-row="true" header-col="false" col-widths="220,780">
    <tr>
        <td>文档</td>
        <td>链接</td>
    </tr>
    <tr>
        <td>商品理解模块 PRD</td>
        <td>https://bytedance.larkoffice.com/docx/HSZ5dL4Jeo6xPRxIdMPcUYjSnTg</td>
    </tr>
    <tr>
        <td>商品目标人群 PRD</td>
        <td>https://bytedance.larkoffice.com/docx/IGkrdp3RyoCjsdxoMAiccd9Fnjf</td>
    </tr>
    <tr>
        <td>视频诊断模块 PRD</td>
        <td>https://bytedance.larkoffice.com/docx/ED7xdnpnXokpqKxbf6tcETZKnWs</td>
    </tr>
    <tr>
        <td>video_target_audience 判定逻辑 PRD</td>
        <td>https://bytedance.larkoffice.com/docx/BMzPdehIQoJVSqx00wlcEtiinJg</td>
    </tr>
    <tr>
        <td>说服档案 Wiki</td>
        <td>https://bytedance.larkoffice.com/wiki/X20uw9DRUiJDmPkyr3OcEBuln7g</td>
    </tr>
    <tr>
        <td>/JG 独立验收报告</td>
        <td>https://bytedance.larkoffice.com/docx/QMh7dNMQIoFRd2xe6KRcyDPdnfh</td>
    </tr>
    <tr>
        <td>/JG 独立测试用例</td>
        <td>https://bytedance.larkoffice.com/docx/KKjTdnPfcotaLUxj80tcnx0fnzb</td>
    </tr>
</table>
