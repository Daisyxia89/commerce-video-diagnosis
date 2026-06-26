from __future__ import annotations

import json
from pathlib import Path

from api.adapter import build_repo_payload, normalize_result
from commerce_video_diagnosis.understanding.core import handle_request


def sample_payload() -> dict:
    return {
        "product_factpack": {
            "fields": {
                "product_name": "草本控油洗发水",
                "leaf_category": "洗发水",
                "shop_name": "示例店铺",
                "price": "中",
                "core_selling_points": ["改善头油", "清爽头皮", "蓬松发根"],
            },
            "field_provenance": {"core_selling_points": "product_detail"},
        },
        "video_factpack": {
            "fields": {
                "video_metadata": {"video_id": "VID_SMOKE_SHAMPOO", "duration_sec": 15, "source_platform": "douyin"},
                "text_stream": {"asr_segments": [{"segment_id": "s1", "text": "头发塌扁怎么办，用完发根更蓬松", "confidence": 0.9}]},
                "visual_stream": [{"summary": "展示头发塌扁和使用后蓬松对比"}],
                "diagnostic_derivations": {"video_primary_claim": "改善头油塌扁"},
            },
            "field_provenance": {"text_stream.asr_segments": "asr_model"},
        },
        "options": {"mvp_scope_gate_enabled": True},
    }


def main() -> None:
    repo_payload = build_repo_payload(sample_payload())
    try:
        raw = handle_request(repo_payload).dict()
        normalized = normalize_result(raw)
    except RuntimeError as exc:
        raw = {"error": {"code": "PROVIDER_NOT_CONFIGURED", "message": str(exc)}}
        normalized = {"status": "provider_not_configured", "error": raw["error"]}
    out = {"repo_payload": repo_payload, "normalized": normalized}
    path = Path("output/api_sanity_result.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS", path)


if __name__ == "__main__":
    main()
