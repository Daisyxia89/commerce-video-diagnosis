# 两段式链路交接说明

## 1. 目标

把“原始视频 -> 外部抽取 -> FactPack -> 视频理解”这条链路明确拆成两个阶段，避免下游引擎承担不属于自己的多模态抽取责任。

## 2. 阶段划分

### 阶段一：上游抽取器

输入：原始视频、外部模型能力、BYOK 配置。

输出：纯净 FactPack。

必须完成：
- 视频基础元数据采集；
- 分段与时间轴；
- 视觉事实抽取；
- 音频事实抽取；
- OCR 事实抽取；
- rhythm_facts 填充；
- 尾部分离与主内容区分。

绝对不能输出：
- HEC 标签；
- Slider 分值；
- JTBD、策略结论、商品诊断快照；
- 任何 blueprint 结构。

### 阶段二：commerce-video-diagnosis 下游诊断阶段

输入：标准 request JSON，其中 `fact_pack` 为阶段一输出。

输出：
- `blueprint`
- `workflow_report`
- `triad_assets`
- `provenance_report`

## 3. 前置能力检查

在安装或接入“上游抽取器 Skill”前，必须先确认调用方已具备以下能力：

1. `VLM provider`
   - 能从视频帧或图片中抽取主体、动作、镜头、场景、光影、关键物体等物理事实。

2. `ASR provider`
   - 能从音轨中抽取口播文本；
   - 最好返回时间戳。

3. `OCR provider`
   - 能从视频帧中抽取字幕、花字、UI 文本；
   - 最好返回 bbox、置信度、时间关联。

4. 本地媒体处理工具
   - `ffmpeg`
   - 元数据读取
   - 切片
   - 抽帧
   - 尾部分离

如果用户还没有这些能力，不应该直接宣称“安装本 Skill 即可从 raw video 起跑”。

正确做法是：
- 先去调用方自己的模型平台、云厂商或内部模型服务中安装 / 开通 `VLM / ASR / OCR` provider；
- 先在目标运行环境安装 `ffmpeg` 等本地媒体处理依赖；
- 若暂时只有 FactPack 样例，则先用样例验证两段式链路，不要伪装成已经具备真实抽取能力。

## 4. 交接契约

上游对下游的最小交付物：

1. 一个纯 FactPack 文件；
2. `video_id`；
3. `source_product_id`；
4. `provenance` 元信息（至少 producer_type / generator_version / generated_at）。

## 5. 边界判断

### 合法
- 外部抽取器输出真实物理事实；
- 用脚本把纯 FactPack 包装成 request JSON；
- 再调用下游推理。

### 非法
- 让下游自己看 raw video；
- 在上游就把 HEC 或结论写好；
- 用 LLM 猜缺失字段后冒充真实抽取结果。

## 6. 草案阶段的现实约束

如果当前仓库里没有真正可运行的 ASR/OCR/VLM 提取器实现，那么“上游抽取器 Skill”能交付的是：

- 边界定义；
- 配置契约；
- FactPack 交接脚本；
- 两段式 smoke test；
- 给未来真实抽取器的接入位。

不能虚构为“已经具备从 raw video 一键提取完整 FactPack 的真实能力”。

## 7. 推荐接入口径

推荐按能力层写接入说明，而不是按单一工具名写默认依赖：

1. `VLM provider`
   - 抽取画面物理事实；
   - 只输出主体、动作、镜头、场景、光影、关键物体等事实字段。

2. `ASR provider`
   - 抽取口播文本；
   - 最好返回时间戳。

3. `OCR provider`
   - 抽取字幕、花字、UI 文本；
   - 最好返回 bbox、置信度、时间关联。

4. 本地媒体处理工具
   - `ffmpeg`
   - 元数据读取
   - 切片
   - 抽帧
   - 尾部分离

现阶段仓库已具备研发样例级能力与两段式验证链路，但尚未形成通用 BYOK 配置解析与标准 provider adapter 工程。因此，推荐文案必须明确：当前能交付的是“接入规范 + 样例链路 + smoke test”，而不是对外即插即用的 raw video 抽取产品。