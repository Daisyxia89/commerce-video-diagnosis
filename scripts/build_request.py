from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap a pure FactPack into a commerce-video-diagnosis downstream request JSON.")
    parser.add_argument("--factpack", required=True, help="Path to pure FactPack JSON file.")
    parser.add_argument("--video-id", required=True, help="Video ID passed to downstream request.")
    parser.add_argument("--source-product-id", required=True, help="Source product ID for SSOT routing.")
    parser.add_argument("--output", required=True, help="Output request JSON path.")
    parser.add_argument("--request-id", default="", help="Optional request ID. Auto-generated when empty.")
    parser.add_argument("--producer-type", default="external_vlm", help="Request provenance producer_type.")
    parser.add_argument("--generator-version", default="commerce_video_diagnosis_draft_v1", help="Request provenance generator_version.")
    parser.add_argument("--video-url", default="", help="Optional caller-provided video url.")
    parser.add_argument("--item-name", default="", help="Optional caller-provided item name.")
    parser.add_argument("--shop-name", default="", help="Optional caller-provided shop name.")
    parser.add_argument("--leaf-category", default="", help="Optional caller-provided leaf category.")
    parser.add_argument("--price", default="", help="Optional caller-provided price.")
    parser.add_argument(
        "--core-selling-point",
        action="append",
        default=[],
        help="Optional caller-provided core selling point. Repeat this flag to pass multiple values.",
    )
    return parser.parse_args()


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(nested, path))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.extend(_flatten_keys(nested, path))
    return keys


def load_factpack(path: Path) -> dict[str, Any]:
    factpack = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(factpack, dict):
        raise ValueError("FactPack 顶层必须是 JSON 对象")
    if "video_meta" not in factpack or "segments" not in factpack:
        raise ValueError("FactPack 缺少 video_meta 或 segments")
    hits = [path for path in _flatten_keys(factpack) if path.split(".")[-1] in FORBIDDEN_KEYS]
    if hits:
        raise ValueError(f"FactPack 含答案型污染字段: {hits}")
    return factpack


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    factpack = load_factpack(Path(args.factpack))
    payload = {
        "request_id": args.request_id or f"REQ_{uuid.uuid4().hex[:12]}",
        "video_id": args.video_id,
        "source_product_id": args.source_product_id,
        "fact_pack": factpack,
        "provenance": {
            "producer_type": args.producer_type,
            "generator_version": args.generator_version,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        },
        "options": {"include_segment_tags": True},
    }
    if args.item_name:
        payload.update(
            {
                "video_url": args.video_url,
                "item_name": args.item_name,
                "shop_name": args.shop_name,
                "leaf_category": args.leaf_category,
                "price": args.price,
                "core_selling_points": list(args.core_selling_point or []),
            }
        )
    return payload


def main() -> int:
    args = parse_args()
    request = build_request(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
