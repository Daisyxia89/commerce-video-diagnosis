from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(SKILL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.smoke_fallback import build_smoke_gate_log, classify_smoke_failure



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw video smoke first, and degrade to pytest-only only for provider auth failures."
    )
    parser.add_argument("--video", required=True, help="Path to local raw video.")
    parser.add_argument("--video-id", required=True, help="Video id for downstream request.")
    parser.add_argument("--source-product-id", required=True, help="Source product id for SSOT routing.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated artifacts.")
    parser.add_argument("--decision-log", required=True, help="Path to write smoke gate decision log JSON.")
    parser.add_argument("--source-platform", default="抖音", help="Source platform written into video_meta.")
    parser.add_argument("--output", default="", help="Optional final JSON output path.")
    parser.add_argument(
        "--include-ocr-regression-summary",
        action="store_true",
        help="Also emit OCR feedback regression summary in unified smoke output.",
    )
    return parser.parse_args()



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def run_smoke_gate(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    decision_log_path = Path(args.decision_log)

    command = [
        sys.executable,
        str(SKILL_ROOT / "scripts/run_raw_video_smoke.py"),
        "--video",
        args.video,
        "--video-id",
        args.video_id,
        "--source-product-id",
        args.source_product_id,
        "--workspace",
        args.workspace,
        "--source-platform",
        args.source_platform,
    ]
    if args.output:
        command.extend(["--output", args.output])
    if args.include_ocr_regression_summary:
        command.append("--include-ocr-regression-summary")

    proc = subprocess.run(command, text=True, capture_output=True)
    combined_output = f"{proc.stdout}\n{proc.stderr}".strip()

    if proc.returncode == 0:
        payload = build_smoke_gate_log(
            mode="full_smoke_executed",
            status="executed",
            exit_code=0,
            reason_code="FULL_SMOKE_EXECUTED",
            command=command,
            output_path=args.output,
            decision_log_path=str(decision_log_path),
            combined_output=combined_output,
        )
        _write_json(decision_log_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    classification = classify_smoke_failure(combined_output)
    if classification["degradable"]:
        payload = build_smoke_gate_log(
            mode="degraded_pytest_only",
            status="degraded",
            exit_code=proc.returncode,
            reason_code=classification["reason_code"],
            matched_fragment=classification["matched_fragment"],
            command=command,
            output_path=args.output,
            decision_log_path=str(decision_log_path),
            combined_output=combined_output,
        )
        _write_json(decision_log_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = build_smoke_gate_log(
        mode="failed",
        status="failed",
        exit_code=proc.returncode,
        reason_code="NON_DEGRADABLE_FAILURE",
        matched_fragment=classification["matched_fragment"],
        command=command,
        output_path=args.output,
        decision_log_path=str(decision_log_path),
        combined_output=combined_output,
    )
    _write_json(decision_log_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    return proc.returncode or 1



def main() -> int:
    args = parse_args()
    return run_smoke_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
