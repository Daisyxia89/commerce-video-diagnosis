from __future__ import annotations

import json
from pathlib import Path

from ..models.config_models import ExtractorConfig, extractor_config_from_dict



def load_config(path: str) -> ExtractorConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须为 JSON 对象")
    return extractor_config_from_dict(payload)
