from __future__ import annotations

from typing import Any

from ..errors import AdapterViolation



def adapt_asr(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterViolation("asr_raw 必须为 list")
    segments: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise AdapterViolation("asr_raw[*] 必须为对象")
        required = ["segment_id", "start_sec", "end_sec", "audio_facts"]
        missing = [field for field in required if field not in item]
        if missing:
            raise AdapterViolation(f"asr_raw 缺少字段: {missing}")
        audio_facts = item.get("audio_facts")
        if not isinstance(audio_facts, dict):
            raise AdapterViolation("asr_raw[*].audio_facts 必须为对象")
        asr_text = str(audio_facts.get("asr_text") or "")
        normalized_audio_facts = {
            "asr_text": asr_text,
            "sfx_events": audio_facts.get("sfx_events") if isinstance(audio_facts.get("sfx_events"), list) else [],
            "bgm_events": audio_facts.get("bgm_events") if isinstance(audio_facts.get("bgm_events"), list) else [],
        }
        segments.append({**item, "audio_facts": normalized_audio_facts})
    return segments
