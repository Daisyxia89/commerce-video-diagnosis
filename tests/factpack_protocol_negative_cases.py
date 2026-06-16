from __future__ import annotations

import copy
from typing import Any


Mutation = dict[str, Any]
NegativeCase = dict[str, Any]


def _set_nested(payload: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    cursor: Any = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _delete_nested(payload: dict[str, Any], path: tuple[str | int, ...]) -> None:
    cursor: Any = payload
    for key in path[:-1]:
        cursor = cursor[key]
    del cursor[path[-1]]


def apply_mutation(factpack: dict[str, Any], mutation: Mutation) -> dict[str, Any]:
    mutated = copy.deepcopy(factpack)
    mode = mutation["mode"]
    path = mutation["path"]
    if mode == "set":
        _set_nested(mutated, path, mutation["value"])
        return mutated
    if mode == "delete":
        _delete_nested(mutated, path)
        return mutated
    raise ValueError(f"unsupported mutation mode: {mode}")


def build_main_skeleton_parse_negative_cases(factpack: dict[str, Any]) -> list[NegativeCase]:
    bundle_id = str((factpack.get("semantic_bundles") or [{}])[0].get("bundle_id") or "")
    if not bundle_id:
        raise ValueError("factpack.semantic_bundles[0].bundle_id is required for negative cases")

    return [
        {
            "label": "semantic_bundle_missing_segment_ids",
            "mutation": {"mode": "delete", "path": ("semantic_bundles", 0, "segment_ids")},
            "expected_path": "semantic_bundles -> 0 -> segment_ids",
        },
        {
            "label": "semantic_bundle_extra_field",
            "mutation": {"mode": "set", "path": ("semantic_bundles", 0, "unexpected_marker"), "value": True},
            "expected_path": "semantic_bundles -> 0 -> unexpected_marker",
        },
        {
            "label": "semantic_bundle_segment_ids_type_error",
            "mutation": {"mode": "set", "path": ("semantic_bundles", 0, "segment_ids"), "value": [101]},
            "expected_path": "semantic_bundles -> 0 -> segment_ids",
        },
        {
            "label": "bundle_range_missing_start_segment_id",
            "mutation": {"mode": "delete", "path": ("bundle_to_segment_range", bundle_id, "start_segment_id")},
            "expected_path": f"bundle_to_segment_range -> {bundle_id} -> start_segment_id",
        },
        {
            "label": "bundle_range_extra_field",
            "mutation": {"mode": "set", "path": ("bundle_to_segment_range", bundle_id, "rogue_field"), "value": "polluted"},
            "expected_path": f"bundle_to_segment_range -> {bundle_id} -> rogue_field",
        },
        {
            "label": "bundle_range_start_index_type_error",
            "mutation": {"mode": "set", "path": ("bundle_to_segment_range", bundle_id, "start_segment_index"), "value": 1.5},
            "expected_path": f"bundle_to_segment_range -> {bundle_id} -> start_segment_index",
        },
    ]


def build_main_skeleton_assertion_negative_cases(factpack: dict[str, Any]) -> list[NegativeCase]:
    bundle_id = str((factpack.get("semantic_bundles") or [{}])[0].get("bundle_id") or "")
    bundle_segment_ids = list((factpack.get("semantic_bundles") or [{}])[0].get("segment_ids") or [])
    if not bundle_id or not bundle_segment_ids:
        raise ValueError("factpack.semantic_bundles[0] with segment_ids is required for negative cases")

    return [
        {
            "label": "bundle_range_start_index_mismatch",
            "mutation": {"mode": "set", "path": ("bundle_to_segment_range", bundle_id, "start_segment_index"), "value": 999},
            "expected_error": f"bundle_to_segment_range[{bundle_id}].start_segment_index 不正确",
        },
        {
            "label": "bundle_range_end_segment_id_mismatch",
            "mutation": {
                "mode": "set",
                "path": ("bundle_to_segment_range", bundle_id, "end_segment_id"),
                "value": f"{bundle_segment_ids[-1]}_MISMATCH",
            },
            "expected_error": f"bundle_to_segment_range[{bundle_id}].end_segment_id 不正确",
        },
    ]
