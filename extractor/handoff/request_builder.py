from __future__ import annotations

import json
from pathlib import Path

from ..assembly.provenance_builder import build_provenance



def build_request(
    factpack: dict,
    video_id: str,
    source_product_id: str,
    request_id: str = "",
    *,
    video_url: str = "",
    item_name: str = "",
    shop_name: str = "",
    leaf_category: str = "",
    price: str = "",
    core_selling_points: list[str] | None = None,
) -> dict:
    payload = {
        "request_id": request_id or f"REQ_{video_id}",
        "video_id": video_id,
        "source_product_id": source_product_id,
        "fact_pack": factpack,
        "provenance": build_provenance(),
        "options": {"include_segment_tags": True},
    }
    if item_name:
        payload.update(
            {
                "video_url": video_url,
                "item_name": item_name,
                "shop_name": shop_name,
                "leaf_category": leaf_category,
                "price": price,
                "core_selling_points": list(core_selling_points or []),
            }
        )
    return payload



def write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
