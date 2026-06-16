from __future__ import annotations

import json



def extract_first_json(raw: str):
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("未找到合法 JSON")
