from __future__ import annotations

from ..adapters.vlm_adapter import adapt_vlm
from ..adapters.asr_adapter import adapt_asr
from ..adapters.ocr_adapter import adapt_ocr



def normalize_provider_outputs(raw_bundle: dict) -> dict:
    return {
        "vlm": adapt_vlm(raw_bundle.get("vlm_raw")),
        "asr": adapt_asr(raw_bundle.get("asr_raw")),
        "ocr": adapt_ocr(raw_bundle.get("ocr_raw")),
    }
