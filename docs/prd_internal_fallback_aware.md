# PRD 补丁：公开仓库 fallback 协议说明

## 1. 背景

公开发布版本不再携带任何私有执行脚本、私有路径或环境探测逻辑。

保留 `runtime.provider_fallback_mode` 的唯一目的，是让调用方在配置层显式表达：

- 是否希望保留一个“provider fallback 协议位”；
- 当 provider 未配置时，错误信息如何解释；
- trace 中如何记录当前运行意图。

## 2. 公开版约束

1. 公开仓库只支持两类 provider 路径：
   - `fixture_file`
   - 外部 BYOK provider
2. 当 `providers.asr / providers.vlm / providers.ocr` 未配置可执行的外部 provider 时：
   - 不再尝试任何内置兜底实现；
   - 必须直接报错；
   - 错误信息需明确提示调用方显式补齐 provider 配置。
3. `runtime.provider_fallback_mode` 作为协议字段保留，但不再触发任何私有执行路径。
4. `runtime.provider_fallback_mode` 默认值为 `force_off`；`auto` 与 `force_on` 为协议保留位，当前公开版无内置 fallback 实现。

## 3. Trace 要求

`provider_resolution_trace` 至少包含：

```json
{
  "environment_mode": "external_public | fallback_requested",
  "fallback_protocol_mode": "auto | force_on | force_off",
  "asr": {
    "selected_provider_mode": "disabled | fixture | byok | error | fallback_stub",
    "provider_name": "string",
    "fallback_used": false,
    "fallback_reason": "string"
  }
}
```

说明：

- `fallback_protocol_mode` 只表示 `runtime.provider_fallback_mode` 的配置意图；
- `fallback_stub` 仅表示协议位被请求，但公开仓库没有对应实现；
- 默认推荐公开调用方统一使用 `force_off`。

## 4. 验收标准

1. 仓库源码、示例配置、README、Schema、测试样例中，不再出现私有脚本名或私有路径。
2. provider 缺失时，系统报错文案统一指向“请显式配置 BYOK provider”。
3. 公开版运行结果中的 trace 字段与文档定义保持一致。
