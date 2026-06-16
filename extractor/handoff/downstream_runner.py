from __future__ import annotations

from commerce_video_diagnosis.understanding.core import ProtocolViolation, handle_request

from ..errors import HandoffViolation



def run_downstream(request: dict, ssot_path: str = "") -> dict:
    try:
        result = handle_request(request, ssot_path=ssot_path or None)
    except ProtocolViolation as exc:
        raise HandoffViolation(str(exc)) from exc
    return result.dict()
