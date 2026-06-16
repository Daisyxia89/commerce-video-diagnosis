from __future__ import annotations

from datetime import datetime, timezone



def build_provenance(generator_version: str = "commerce_video_diagnosis_p0") -> dict:
    return {
        "producer_type": "external_vlm",
        "generator_version": generator_version,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
