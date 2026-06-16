# 火山引擎 ASR 自动上传排障说明

对应配置文件：`volcengine_asr_upload_config.json`

## 适用场景

当你使用：

- `provider = "volcengine_asr"`
- `upload_provider = "tos"`

并且走“本地音频自动上传 TOS -> 回填 `audio_url` -> 提交火山 ASR -> query 轮询结果”链路时，优先参考本说明排障。

## 1. 先确认公网 URL 是否真的可读

自动上传成功，不代表火山引擎一定能拉到音频。

优先检查最终生成的 `audio_url`：

- 浏览器无登录状态下能否直接打开
- `curl -I <audio_url>` 是否返回 `200 OK`
- 返回内容是否是音频文件，而不是 403 页面、鉴权页、CDN 错误页
- `Content-Type` 是否合理

## 2. TOS / 域名 / CDN 排查

重点检查：

- `upload_public_base_url` 是否为真实公网域名
- 域名是否已正确解析到 TOS 或 CDN
- CDN 是否已正确回源到 bucket
- 回源路径是否保留对象 key
- bucket / object ACL 是否允许公网读取，或是否通过代理域名实现公网可读

## 3. 火山 submit / query 排查路径

### submit 失败

常见现象：

- `HTTP 401` / `HTTP 403`
- `提交任务失败，未返回 task_id`

优先检查：

- `providers.asr.api_key`
- `providers.asr.extra.app_key`
- `providers.asr.extra.resource_id`
- `providers.asr.endpoint`
- `providers.asr.extra.request`

### query 失败

优先检查：

- `providers.asr.extra.query_endpoint`
- query 请求里的 `X-Api-Request-Id` 是否等于 submit 返回的 task id
- `audio.url` 是否就是最终公网 URL
- `resource_id` 是否与账号能力匹配

### query 返回业务错误

若返回 `message` / `code` 异常：

- 先验证 `audio_url` 是否真实可下载
- 再检查 query endpoint 是否填错
- 再检查 Access Key / App Key / resource_id 组合是否匹配

## 4. 推荐排查顺序

1. 先人工验证最终 `audio_url`
2. 再检查 TOS bucket / ACL / 域名 / CDN
3. 再检查火山 `api_key` / `app_key` / `resource_id` / `query_endpoint`
4. 最后再看 query 返回的 `code` / `message`

## 相关入口

- 配置文件：`volcengine_asr_upload_config.json`
- 总体说明：`../../README.md`
- 快速上手：`../../docs/quickstart_byok.md`
