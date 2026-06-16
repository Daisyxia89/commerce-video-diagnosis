from __future__ import annotations

from typing import Any

from ..errors import AdapterViolation



def adapt_ocr(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterViolation("ocr_raw 必须为 list")
    segments: list[dict] = []
    style_fields = ["font_family", "font_weight", "font_size_level", "stroke_style", "text_effect_style"]
    for item in raw:
        if not isinstance(item, dict):
            raise AdapterViolation("ocr_raw[*] 必须为对象")
        required = ["segment_id", "start_sec", "end_sec", "ocr_facts"]
        missing = [field for field in required if field not in item]
        if missing:
            raise AdapterViolation(f"ocr_raw 缺少字段: {missing}")
        ocr_facts = item.get("ocr_facts")
        if not isinstance(ocr_facts, list):
            raise AdapterViolation("ocr_raw[*].ocr_facts 必须为 list")
        for idx, ocr in enumerate(ocr_facts):
            if not isinstance(ocr, dict):
                raise AdapterViolation(f"ocr_raw[*].ocr_facts[{idx}] 必须为对象")
            ocr_required = ["text", "position", "color", *style_fields]
            ocr_missing = [field for field in ocr_required if field not in ocr]
            if ocr_missing:
                raise AdapterViolation(f"ocr_raw[*].ocr_facts[{idx}] 缺少字段: {ocr_missing}")
            for field in ["text", "color", *style_fields]:
                if not str(ocr.get(field) or "").strip():
                    raise AdapterViolation(f"ocr_raw[*].ocr_facts[{idx}].{field} 不能为空")
        segments.append(item)
    return segments
