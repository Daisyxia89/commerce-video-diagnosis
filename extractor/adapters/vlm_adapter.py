from __future__ import annotations

from typing import Any

from ..errors import AdapterViolation



def adapt_vlm(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterViolation("vlm_raw 必须为 list")
    segments: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise AdapterViolation("vlm_raw[*] 必须为对象")
        required = ["segment_id", "start_sec", "end_sec", "visual_facts", "rhythm_facts"]
        missing = [field for field in required if field not in item]
        if missing:
            raise AdapterViolation(f"vlm_raw 缺少字段: {missing}")
        segments.append(item)
    return segments
