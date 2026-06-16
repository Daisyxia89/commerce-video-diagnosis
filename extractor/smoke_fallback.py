from __future__ import annotations

from typing import Any

AUTH_DOWNGRADE_REASON_CODES = (
    "AUTH_MISSING_TOKEN",
    "AUTH_PERMISSION_DENIED",
    "AUTH_PROVIDER_UNAVAILABLE",
)



def classify_smoke_failure(output: str) -> dict[str, Any]:
    for reason_code in AUTH_DOWNGRADE_REASON_CODES:
        if reason_code in output:
            return {
                "degradable": True,
                "reason_code": reason_code,
                "matched_fragment": reason_code,
            }
    return {
        "degradable": False,
        "reason_code": "NON_DEGRADABLE_FAILURE",
        "matched_fragment": "",
    }



def build_smoke_gate_log(
    *,
    mode: str,
    status: str,
    exit_code: int,
    reason_code: str,
    command: list[str],
    output_path: str,
    decision_log_path: str,
    combined_output: str,
    matched_fragment: str = "",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": status,
        "exit_code": exit_code,
        "reason_code": reason_code,
        "matched_fragment": matched_fragment,
        "command": command,
        "output_path": output_path,
        "decision_log_path": decision_log_path,
        "combined_output_excerpt": combined_output[-4000:],
    }
