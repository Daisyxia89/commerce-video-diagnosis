from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


ALLOWED_PRODUCT_SOURCES = {"caller_product_info", "product_detail", "manual_edit"}
BLOCKED_PRODUCT_SOURCES = {"video_extracted_candidate"}


class AdapterError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for sep in ["；", ";", "、", ",", "\n"]:
        text = text.replace(sep, "|")
    return [item.strip() for item in text.split("|") if item.strip()]


def validate_frontend_payload(payload: dict[str, Any]) -> None:
    product = payload.get("product_factpack") or {}
    provenance = product.get("field_provenance") or {}
    source = provenance.get("core_selling_points") or provenance.get("fields.core_selling_points")
    if source in BLOCKED_PRODUCT_SOURCES:
        raise AdapterError(
            "PROVENANCE_VIOLATION",
            "core_selling_points 不得来自 video_extracted_candidate",
            {"field": "product_factpack.fields.core_selling_points", "source": source},
        )
    if source and source not in ALLOWED_PRODUCT_SOURCES:
        raise AdapterError(
            "PROVENANCE_UNKNOWN_SOURCE",
            f"core_selling_points 来源不在允许集合：{source}",
            {"allowed_sources": sorted(ALLOWED_PRODUCT_SOURCES)},
        )


def build_repo_payload(frontend_payload: dict[str, Any]) -> dict[str, Any]:
    validate_frontend_payload(frontend_payload)
    product = frontend_payload.get("product_factpack") or {}
    pfields = product.get("fields") or {}
    video = frontend_payload.get("video_factpack") or {}
    vfields = video.get("fields") or video
    metadata = vfields.get("video_metadata") or vfields.get("video_meta") or {}
    text_stream = vfields.get("text_stream") or {}
    visual_stream = vfields.get("visual_stream") or []
    derivations = vfields.get("diagnostic_derivations") or {}

    item_name = str(pfields.get("product_name") or pfields.get("item_name") or "").strip()
    leaf_category = str(pfields.get("leaf_category") or pfields.get("category") or "").strip()
    shop_name = str(pfields.get("shop_name") or pfields.get("brand_name") or "前端输入").strip()
    raw_price = str(pfields.get("price") or pfields.get("price_band") or "").strip()
    price_map = {"低": "39", "中": "99", "高": "399", "低价": "39", "中价": "99", "高价": "399", "未知": "99"}
    price = price_map.get(raw_price, raw_price or "99")
    core_selling_points = _as_list(pfields.get("core_selling_points") or pfields.get("selling_points"))
    missing = [name for name, value in {
        "item_name": item_name,
        "leaf_category": leaf_category,
        "shop_name": shop_name,
        "price": price,
        "core_selling_points": core_selling_points,
    }.items() if not value]
    if missing:
        raise AdapterError("SCHEMA_ERROR", f"商品字段缺失：{', '.join(missing)}", {"missing": missing})

    asr_segments = text_stream.get("asr_segments") or []
    asr_text = " ".join(str(seg.get("text") or "") for seg in asr_segments if isinstance(seg, dict)).strip()
    if not asr_text:
        asr_text = str(vfields.get("spoken_script") or derivations.get("video_primary_claim") or "视频口播缺失").strip()
    visual_summary = "；".join(
        str(item.get("summary") or item.get("visual_subject") or item) for item in visual_stream[:3]
    ) if isinstance(visual_stream, list) else str(visual_stream or "")
    if not visual_summary:
        visual_summary = str(vfields.get("visual_scenes") or "画面事实缺失").strip()
    ocr_texts = text_stream.get("ocr_texts") or []
    ocr_facts = []
    for item in ocr_texts[:5] if isinstance(ocr_texts, list) else []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            ocr_facts.append({
                "text": str(item.get("text")),
                "position": item.get("position") or {"x": 0, "y": 0, "w": 1, "h": 1},
                "color": item.get("color") or "unknown",
                "font_family": item.get("font_family") or "unknown",
                "font_weight": item.get("font_weight") or "unknown",
                "font_size_level": item.get("font_size_level") or "unknown",
                "stroke_style": item.get("stroke_style") or "none",
                "text_effect_style": item.get("text_effect_style") or "none",
            })

    duration = float(metadata.get("duration_sec") or 15)
    fact_pack = {
        "video_meta": {
            "source_platform": metadata.get("source_platform") or "frontend",
            "duration_sec": duration,
            "fps": float(metadata.get("fps") or 25),
            "resolution": metadata.get("resolution") or "unknown",
        },
        "segments": [
            {
                "segment_id": "SEG01",
                "start_sec": 0.0,
                "end_sec": duration,
                "visual_facts": {
                    "shot_size": "unknown",
                    "camera_movement": "unknown",
                    "visual_subject": visual_summary,
                    "lighting_tone": "unknown",
                    "key_objects": _as_list(vfields.get("product_mentions")) or [item_name],
                    "actions": [{"action_name": "展示/演示", "physical_intensity": "low", "summary": visual_summary}],
                },
                "audio_facts": {"asr_text": asr_text, "sfx_events": [], "bgm_events": []},
                "ocr_facts": ocr_facts,
                "rhythm_facts": {"transition_type": "unknown", "pace_marker": "unknown"},
            }
        ],
        "semantic_bundles": [
            {
                "bundle_id": "BUNDLE01",
                "start_sec": 0.0,
                "end_sec": duration,
                "segment_ids": ["SEG01"],
                "bundle_role": "narrative_unit",
                "aggregation_reason": ["frontend_factpack_adapter"],
                "coverage_frame_refs": ["SEG01"],
            }
        ],
        "segment_to_bundle_map": {"SEG01": "BUNDLE01"},
        "bundle_to_segment_range": {
            "BUNDLE01": {"start_segment_index": 0, "end_segment_index": 0, "start_segment_id": "SEG01", "end_segment_id": "SEG01"}
        },
        "storyboard_source": "semantic_bundles",
    }

    return {
        "request_id": frontend_payload.get("request_id") or f"REQ_{uuid.uuid4().hex[:12]}",
        "video_id": metadata.get("video_id") or f"VID_{uuid.uuid4().hex[:8]}",
        "source_product_id": str(pfields.get("source_product_id") or item_name),
        "video_url": metadata.get("source_url") or vfields.get("video_url") or "",
        "item_name": item_name,
        "shop_name": shop_name,
        "leaf_category": leaf_category,
        "price": price,
        "core_selling_points": core_selling_points,
        "fact_pack": fact_pack,
        "provenance": {"producer_type": "external_pipeline", "generator_version": "frontend_api_adapter_v1", "generated_at": datetime.now(timezone.utc).isoformat()},
        "options": {
            "include_factpack": True,
            "include_blueprint": True,
            "include_trace": False,
            "include_provenance": True,
            **(frontend_payload.get("options") or {}),
        },
    }


def normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    diagnosis = raw.get("diagnosis") or {}
    primary_hec = diagnosis.get("primary_hec") or raw.get("blueprint", {}).get("primary_hec") or {}
    product_diag = (raw.get("triad_assets") or {}).get("product_diagnosis_result") or {}
    product_jtbd = product_diag.get("jtbd") or product_diag.get("primary_task") or product_diag.get("product_task") or ""
    return {
        "status": "diagnosis_completed",
        "diagnosis": diagnosis,
        "product_jtbd": {"jtbd_level1": product_jtbd},
        "video_jtbd": {"jtbd_level1": diagnosis.get("persuasion_chain") or ""},
        "profile_match": {"status": "available_for_frontend_mapping"},
        "hec_detail": {
            "hook": f"{primary_hec.get('hook_label', '')} {primary_hec.get('hook_label_name', '')}".strip(),
            "effect": f"{primary_hec.get('effect_label', '')} {primary_hec.get('effect_label_name', '')}".strip(),
            "cta": f"{primary_hec.get('cta_label', '')} {primary_hec.get('cta_label_name', '')}".strip(),
            "verdict": primary_hec.get("reason") or "HEC generated",
        },
        "audience_fit": {"product": "", "video": "", "status": "待前端业务映射", "gap": ""},
        "suggestions": diagnosis.get("risk_notes") or [],
        "raw_output": raw,
    }
