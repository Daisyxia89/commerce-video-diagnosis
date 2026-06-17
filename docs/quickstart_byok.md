# BYOK 快速上手

## 1. 目标

用外部 ASR / VLM / OCR provider 从原始视频生成 FactPack。当前仓库已经实装：

- ASR：OpenAI Whisper API 兼容路径
- ASR：阿里云录音文件识别 REST API
- ASR：火山引擎录音文件识别 API
- VLM / OCR：OpenAI Chat Completions Vision 兼容路径（以 GPT-4o vision 为例）

## 2. 最小配置思路

- `asr`：配置对应 ASR 服务的 `endpoint`、`api_key`、`model`、`adapter`
- `vlm`：配置 GPT-4o vision endpoint、api_key、model
- `ocr`：可与 `vlm` 复用同一视觉 endpoint；若不配置，则会按公开仓库 fallback 协议位显式报错
- `input.video_path`：必须提供原始视频路径
- `local_tools.workspace_dir`：建议单独设置，方便查看 runtime cache
- `providers.asr.extra`：国内云厂商的录音文件识别通常要求公网可访问音频 URL；可以直接传 `extra.audio_url`，也可以配置 OSS / TOS 自动上传，由运行时先上传本地音频再回填 `audio_url`

## 3. 示例配置

- OpenAI：`fixtures/examples/openai_whisper_gpt4o_config.json`
- 阿里云（显式传 `audio_url`）：`fixtures/examples/aliyun_asr_config.json`
- 阿里云（OSS 自动上传）：`fixtures/examples/aliyun_asr_upload_config.json`
- 火山引擎（显式传 `audio_url`）：`fixtures/examples/volcengine_asr_config.json`
- 火山引擎（TOS 自动上传）：`fixtures/examples/volcengine_asr_upload_config.json`

## 4. 国内 ASR 配置说明

### 4.1 阿里云 `aliyun_asr`

关键配置：

- `providers.asr.provider = "aliyun_asr"`
- `providers.asr.adapter = "aliyun_asr"`
- `providers.asr.endpoint` 指向阿里云录音文件识别提交接口
- `providers.asr.api_key` 使用 DashScope API Key
- `providers.asr.model` 例如 `paraformer-v2`
- `providers.asr.extra.audio_url` 可直接传公网可访问的音频 URL
- 若没有公网 URL，可改配 `providers.asr.extra.upload_provider = "oss"`，并补齐 `upload_endpoint`、`upload_bucket`、`upload_access_key_id`、`upload_access_key_secret`
- `providers.asr.extra.parameters` 可透传阿里云提交参数，如 `timestamp_alignment_enabled`

说明：当前 adapter 会先提交异步任务，再轮询任务结果，并把阿里云返回的句级 `begin_time / end_time / text` 归一化为仓库内部的 `segments[*].audio_facts.asr_text`。

### 4.2 火山引擎 `volcengine_asr`

关键配置：

- `providers.asr.provider = "volcengine_asr"`
- `providers.asr.adapter = "volcengine_asr"`
- `providers.asr.endpoint` 指向火山引擎 submit 接口
- `providers.asr.api_key` 默认作为 Access Key 使用
- `providers.asr.extra.app_key` 为 APP ID / App Key
- `providers.asr.extra.resource_id` 为资源 ID，例如 `volc.seedasr.auc`
- `providers.asr.extra.query_endpoint` 可显式指定 query 接口；不填时会基于 submit 地址自动推导
- `providers.asr.extra.audio_url` 可直接传公网可访问的音频 URL
- 若没有公网 URL，可改配 `providers.asr.extra.upload_provider = "tos"`，并补齐 `upload_endpoint`、`upload_bucket`、`upload_region`、`upload_access_key_id`、`upload_access_key_secret`
- `providers.asr.extra.request` 可透传 request 字段，例如 `enable_itn`、`show_utterances`

说明：当前 adapter 会先提交任务，再按 task_id 轮询 query 接口，并把火山返回的 `result.text` 与 `result.utterances[*].start_time/end_time/text` 归一化为统一 ASR 输出。

### 4.3 关于本地音频路径

当前这两条国内 ASR 适配链路，都按“提交公网音频 URL -> 轮询结果”实现。

这意味着：

- `input.video_path` 仍然用于本地预处理、切片、抽帧和下游 FactPack 组装
- `asr` 侧最终仍然要提交公网可访问 URL 给云厂商接口
- 若你已经有公网 URL，可直接填 `providers.asr.extra.audio_url`
- 若没有公网 URL，运行时现在支持把本地抽出的音频自动上传到 OSS / TOS，再回填 `audio_url`

自动上传配置放在 `providers.asr.extra` 下：

- OSS：`upload_provider = "oss"`
  - `upload_endpoint`
  - `upload_bucket`
  - `upload_access_key_id`
  - `upload_access_key_secret`
  - `upload_object_prefix`（可选）
  - `upload_public_base_url`（可选）
- TOS：`upload_provider = "tos"`
  - `upload_endpoint`
  - `upload_bucket`
  - `upload_region`
  - `upload_access_key_id`
  - `upload_access_key_secret`
  - `upload_object_prefix`（可选）
  - `upload_public_base_url`（可选）

### 4.4 自动上传配置排障说明

#### 4.4.1 Bucket 公网读配置

自动上传只是先把本地音频放到 OSS / TOS；阿里云 / 火山引擎 ASR 真正读取的，仍然是最终回填出来的公网 `audio_url`。

先检查：

- 最终 `audio_url` 能否在无登录浏览器环境直接访问
- 访问时是否返回 `200 OK`
- 返回内容是否真的是音频文件，而不是 403 页面、鉴权页或 CDN 错误页
- 响应头 `Content-Type` 是否合理

如果 bucket 默认不是公网读，要确保：

- `upload_public_base_url` 对应的是公网可访问域名；并且
- 这个域名背后的 CDN / 网关 / 代理能把对象真实回源出来

#### 4.4.2 域名 / CDN 设置

如果你用了自定义域名或 CDN，重点检查：

- 域名是否已正确解析到对象存储或 CDN
- CDN 是否已经正确回源到目标 bucket
- 回源 path 是否保留对象 key，没有被错误重写
- 新上传文件是否会被 CDN 缓存旧 404 / 403
- 是否限制了来源 IP、Referer、User-Agent，导致 ASR 服务端无法拉取

建议对最终 `audio_url` 额外做两步：

- 浏览器直开验证是否可播放 / 下载
- `curl -I <audio_url>` 确认是否返回 200

#### 4.4.3 ACL 权限排查

如果上传成功，但 ASR query 阶段失败，优先排查：

1. bucket 权限是否允许公网读，或经由公网代理域名可读
2. object ACL 是否被覆盖成私有
3. 上传账号权限是否只有写，没有对应读策略配套
4. 域名 / CDN 是否附加了访问控制，导致云厂商服务端拉取失败

#### 4.4.4 ASR query 失败排查路径

##### 阿里云 `aliyun_asr`

建议顺序：

1. 先确认 submit 是否返回 `task_id`
2. 再确认 submit payload 中 `input.file_urls` 使用的就是最终公网 URL
3. 若 query 长时间卡在 `PENDING` / `RUNNING`，优先检查音频 URL 公网可达性与音频格式
4. 若 query 直接失败，去 DashScope 控制台核对任务详情

常见报错与处理：

- `HTTP 401` / `HTTP 403`
  - 处理：核对 DashScope API Key、服务开通状态、endpoint 与模型名
- `提交任务失败，未返回 task_id`
  - 处理：核对 `endpoint`、`model`、`parameters` 是否匹配接口要求
- query 失败 / 状态异常
  - 处理：优先验证 `audio_url` 是否真实可下载；再检查音频编码、时长、采样率

##### 火山引擎 `volcengine_asr`

建议顺序：

1. 先确认 submit 是否返回 `id` / `task_id`
2. 再确认 submit payload 中 `audio.url` 是否为最终公网 URL
3. 若 query 失败，先检查 `extra.query_endpoint` 与 submit 返回 task id 是否匹配
4. 再检查 `resource_id`、`app_key`、`audio.url` 可达性

常见报错与处理：

- `HTTP 401` / `HTTP 403`
  - 处理：核对 Access Key、App Key、资源权限、接口地址
- `提交任务失败，未返回 task_id`
  - 处理：精简 `request` 字段，只保留必要参数后重试
- query 返回 `message` / `code` 异常
  - 处理：核对 `query_endpoint`、`X-Api-Request-Id`、`audio.url` 是否一致且可读

## 5. 运行命令

### 只抽取 FactPack

```bash
cd commerce-video-diagnosis && python3 scripts/run_extractor.py \
  --config fixtures/examples/openai_whisper_gpt4o_config.json \
  --mode extract-only
```

### 跑完整两段式链路

```bash
cd commerce-video-diagnosis && python3 scripts/run_extractor.py \
  --config fixtures/examples/openai_whisper_gpt4o_config.json \
  --mode two-stage-run
```

如需测试国内 ASR，只需要把 `--config` 切到对应示例文件。

## 6. 输出落盘

示例 config 默认会输出：

- `output/byok_openai/factpack.json`
- `output/byok_openai/request.json`
- `output/byok_openai/result.json`
- `output/byok_openai/runtime/provider_runtime/...`
- `output/byok_aliyun/...`
- `output/byok_volcengine/...`

## 7. 输出字段约束

### ASR

上游最终会被归一化为：

```json
{
  "segment_id": "SEG01",
  "start_sec": 0.0,
  "end_sec": 2.0,
  "audio_facts": {
    "asr_text": "口播文本",
    "sfx_events": [],
    "bgm_events": []
  }
}
```

### VLM

上游最终会被归一化为：

```json
{
  "segment_id": "SEG01",
  "start_sec": 0.0,
  "end_sec": 2.0,
  "visual_facts": {
    "shot_size": "close_up",
    "camera_movement": "static",
    "visual_subject": "主体描述",
    "lighting_tone": "bright_natural_daylight",
    "key_objects": ["产品", "手"],
    "actions": [{"action_name": "拿起", "physical_intensity": "low"}]
  },
  "rhythm_facts": {
    "transition_type": "hard_cut",
    "pace_marker": "normal"
  }
}
```

### OCR

OCR adapter 要求每条文字都必须提供：

- `text`
- `position.x/y/w/h`
- `color`
- `font_family`
- `font_weight`
- `font_size_level`
- `stroke_style`
- `text_effect_style`

缺任一字段都会被断言拦截。

## 8. 常见问题

### Q1：为什么我明明填了 provider 还是没走外部调用？

判定条件不是只有 `provider`。必须同时存在：

- `endpoint`
- `api_key`

否则会直接报错，并提示你显式补齐对应 provider 配置。

### Q2：Whisper 返回了文本，但 segment 对不齐怎么办？

当前 orchestrator 会优先使用 provider 返回的 segment 时间戳做重叠对齐；若 provider 没给有效分段，会退回到按句切分并均匀映射到 preprocess segments。

### Q3：阿里云 / 火山引擎为什么还要求 `audio_url`？

因为这两类录音文件识别接口当前按异步文件识别接入，云厂商消费的仍然是公网音频 URL。

现在有两种满足方式：

- 你自己直接提供 `providers.asr.extra.audio_url`
- 或让运行时基于 `providers.asr.extra.upload_provider = oss/tos` 先把本地音频上传，再自动回填 `audio_url`

如果两者都没有配置，adapter 会直接报错，而不会静默回退或伪造识别结果。

### Q4：GPT-4o vision 返回的 OCR 样式不全怎么办？

这类情况不会被静默放过。OCR adapter 会直接报错，要求 provider 返回完整样式字段。必要时需要增强 prompt 或更换 provider。
