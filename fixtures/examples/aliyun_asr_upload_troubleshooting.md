# 阿里云 ASR 自动上传排障说明

对应配置文件：`aliyun_asr_upload_config.json`

## 适用场景

当你使用：

- `provider = "aliyun_asr"`
- `upload_provider = "oss"`

并且走“本地音频自动上传 OSS -> 回填 `audio_url` -> 提交阿里云 ASR -> 异步轮询结果”链路时，优先参考本说明排障。

## 1. 先确认公网 URL 是否真的可读

自动上传成功，不代表阿里云一定能拉到音频。

优先检查最终生成的 `audio_url`：

- 浏览器无登录状态下能否直接打开
- `curl -I <audio_url>` 是否返回 `200 OK`
- 返回内容是否是音频文件，而不是 403 页面、鉴权页、CDN 错误页
- `Content-Type` 是否合理

## 2. OSS / 域名 / CDN 排查

重点检查：

- `upload_public_base_url` 是否为真实公网域名
- 域名是否已正确 CNAME 到 OSS 或 CDN
- CDN 是否已配置正确回源到 bucket
- 回源路径是否保留对象 key
- bucket / object ACL 是否允许公网读取，或是否通过代理域名实现公网可读

## 3. 阿里云 submit / query 排查路径

### submit 失败

常见现象：

- `HTTP 401` / `HTTP 403`
- `提交任务失败，未返回 task_id`

优先检查：

- `providers.asr.api_key`
- `providers.asr.endpoint`
- `providers.asr.model`
- `providers.asr.extra.parameters`

### query 长时间卡住

若状态长时间停留在 `PENDING` / `RUNNING`：

- 优先检查 `audio_url` 是否公网可达
- 再检查音频格式、时长、采样率是否符合服务要求

### query 失败

若 query 直接失败：

- 去 DashScope 控制台核对任务详情
- 核对 submit payload 中 `input.file_urls` 是否就是最终公网 URL
- 确认对象未被 CDN 缓存旧 403 / 404

## 4. 推荐排查顺序

1. 先人工验证最终 `audio_url`
2. 再检查 OSS bucket / ACL / 域名 / CDN
3. 再检查阿里云 `api_key` / `endpoint` / `model`
4. 最后再看 query 返回状态和控制台任务详情

## 相关入口

- 配置文件：`aliyun_asr_upload_config.json`
- 总体说明：`../../README.md`
- 快速上手：`../../docs/quickstart_byok.md`
