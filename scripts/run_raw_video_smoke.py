from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(SKILL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.entry import run_extractor



def build_ocr_regression_summary(*, workspace: Path, result: dict) -> dict:
    preprocess_payload = json.loads((workspace / "runtime" / "preprocess.json").read_text(encoding="utf-8"))
    decision_payload = json.loads((workspace / "runtime" / "decision_report.json").read_text(encoding="utf-8"))
    decision_report = preprocess_payload.get("decision_report", [])

    summary_entry = next(item for item in decision_report if item.get("reason_code") == "DECISION_SUMMARY")
    dropped_after_rescoring = [
        item for item in decision_report if item.get("reason_code") == "DROPPED_AFTER_OCR_RESCORING"
    ]
    return {
        "status": result["status"],
        "video_id": result["result"]["blueprint"]["video_id"],
        "blueprint_id": result["result"]["blueprint"]["blueprint_id"],
        "ocr_feedback_enabled": summary_entry["summary"]["ocr_feedback_enabled"],
        "ocr_hit_count": summary_entry["summary"]["ocr_hit_count"],
        "dropped_after_ocr_rescoring_count": len(dropped_after_rescoring),
        "decision_report_synced": decision_payload.get("decision_report") == decision_report,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw video smoke test for commerce-video-diagnosis.")
    parser.add_argument("--video", required=True, help="Path to local raw video.")
    parser.add_argument("--video-id", required=True, help="Video id for downstream request.")
    parser.add_argument("--source-product-id", required=True, help="Source product id for SSOT routing.")
    parser.add_argument("--workspace", default="output/raw_video_smoke", help="Workspace directory for generated artifacts.")
    parser.add_argument("--source-platform", default="抖音", help="Source platform written into video_meta.")
    parser.add_argument("--output", default="", help="Optional final JSON output path.")
    parser.add_argument(
        "--include-ocr-regression-summary",
        action="store_true",
        help="Also emit OCR feedback regression summary in unified smoke output.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config = {
        "runtime": {"source_platform": args.source_platform, "trace_artifacts": True, "enable_real_ocr_feedback": True, "ocr_feedback_top_k": 4},
        "local_tools": {"workspace_dir": str(workspace / "runtime")},
        "input": {
            "video_path": args.video,
            "video_id": args.video_id,
            "source_product_id": args.source_product_id,
            "request_id": f"REQ_{args.video_id}",
        },
        "providers": {
            "vlm": {
                "enabled": True,
                "provider": "openai_compatible_vlm",
                "adapter": "openai_chat_vision_json",
                "endpoint": os.environ.get("VIDEO_FACTPACK_VLM_ENDPOINT", "https://example.com/v1/chat/completions"),
                "api_key": os.environ.get("VIDEO_FACTPACK_VLM_API_KEY", "demo-key"),
                "model": os.environ.get("VIDEO_FACTPACK_VLM_MODEL", "gpt-4o-mini"),
                "timeout_sec": 180,
                "required": True,
            },
            "asr": {
                "enabled": True,
                "provider": "openai_compatible_asr",
                "adapter": "openai_audio_transcription",
                "endpoint": os.environ.get("VIDEO_FACTPACK_ASR_ENDPOINT", "https://example.com/v1/audio/transcriptions"),
                "api_key": os.environ.get("VIDEO_FACTPACK_ASR_API_KEY", "demo-key"),
                "model": os.environ.get("VIDEO_FACTPACK_ASR_MODEL", "whisper-1"),
                "timeout_sec": 180,
                "required": True,
            },
            "ocr": {
                "enabled": True,
                "provider": "openai_compatible_ocr",
                "adapter": "openai_chat_vision_json",
                "endpoint": os.environ.get("VIDEO_FACTPACK_OCR_ENDPOINT", os.environ.get("VIDEO_FACTPACK_VLM_ENDPOINT", "https://example.com/v1/chat/completions")),
                "api_key": os.environ.get("VIDEO_FACTPACK_OCR_API_KEY", os.environ.get("VIDEO_FACTPACK_VLM_API_KEY", "demo-key")),
                "model": os.environ.get("VIDEO_FACTPACK_OCR_MODEL", os.environ.get("VIDEO_FACTPACK_VLM_MODEL", "gpt-4o-mini")),
                "timeout_sec": 180,
                "required": True,
            },
        },
        "output": {
            "factpack_path": str(workspace / "factpack.json"),
            "request_path": str(workspace / "request.json"),
            "result_path": str(workspace / "result.json"),
        },
    }
    config_path = workspace / "smoke_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = run_extractor(str(config_path), mode="two-stage-run")
    payload = {
        "smoke_result": result,
    }
    if args.include_ocr_regression_summary:
        payload["ocr_feedback_regression"] = build_ocr_regression_summary(workspace=workspace, result=result)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
