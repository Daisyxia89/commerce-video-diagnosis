from __future__ import annotations

from typing import Any

from ..errors import ContaminationViolation

FORBIDDEN_KEYS = {
    "primary_hec",
    "slider_signature",
    "hook_label",
    "effect_label",
    "cta_label",
    "jtbd",
    "original_jtbd",
    "category_strategy_intent",
    "product_strategy_intent",
    "intent_coordinates",
    "modifiers",
    "weapon_tags",
}



def _walk(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_KEYS:
                hits.append(path)
            hits.extend(_walk(nested, path))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_walk(nested, path))
    return hits



def assert_no_contamination(payload: dict[str, Any]) -> None:
    hits = _walk(payload)
    if hits:
        raise ContaminationViolation(f"FactPack 含答案型污染字段: {hits}")
