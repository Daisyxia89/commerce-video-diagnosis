# Frontend API Wrapper

This wrapper exposes the Python `commerce-video-diagnosis` skill as a browser-callable HTTP API for the frontend MVP.

## Endpoint

```text
POST /api/diagnose
GET /health
```

## Run locally

The repository currently uses pydantic v1 syntax. If your environment has pydantic v2, install v1 into a local vendor directory first:

```bash
python3 -m pip install --target ./vendor_pydantic1 "pydantic>=1.10,<2"
python3 -m pip install fastapi uvicorn
PYTHONPATH=./vendor_pydantic1:. uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then configure the frontend page:

```text
Mode: Real Skill API
Skill Endpoint: http://localhost:8000/api/diagnose
```

## Provider configuration

`run_v2` may call LLM-backed product diagnosis modules. In a real environment, configure provider variables before starting the server:

```bash
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
```

If provider config is missing, the wrapper returns:

```json
{
  "status": "provider_not_configured",
  "error": {
    "code": "PROVIDER_NOT_CONFIGURED",
    "message": "缺少 LLM provider 配置..."
  }
}
```

## Frontend payload

```json
{
  "product_factpack": {
    "fields": {
      "product_name": "草本控油洗发水",
      "leaf_category": "洗发水",
      "shop_name": "示例店铺",
      "price": "99",
      "core_selling_points": ["改善头油", "清爽头皮", "蓬松发根"]
    },
    "field_provenance": {
      "core_selling_points": "product_detail"
    }
  },
  "video_factpack": {
    "fields": {
      "video_metadata": {"video_id": "VID_001", "duration_sec": 15, "source_platform": "douyin"},
      "text_stream": {"asr_segments": [{"segment_id": "s1", "text": "头发塌扁怎么办", "confidence": 0.9}]},
      "visual_stream": [{"summary": "展示头发塌扁和使用后蓬松对比"}]
    },
    "field_provenance": {
      "text_stream.asr_segments": "asr_model"
    }
  },
  "options": {"mvp_scope_gate_enabled": true}
}
```

## Adapter behavior

The wrapper converts frontend payload into the repository schema:

```text
product_factpack.fields -> top-level caller product fields / product_info equivalent
video_factpack.fields -> fact_pack
```

It blocks product contamination:

```json
{
  "product_factpack": {
    "field_provenance": {
      "core_selling_points": "video_extracted_candidate"
    }
  }
}
```

returns:

```json
{
  "status": "provenance_violation",
  "error": {
    "code": "PROVENANCE_VIOLATION"
  }
}
```

## Sanity check

```bash
PYTHONPATH=./vendor_pydantic1:. python3 scripts/run_api_sanity.py
```

This writes:

```text
output/api_sanity_result.json
```

If provider variables are not configured, the adapter conversion still passes and the output records `provider_not_configured`.
