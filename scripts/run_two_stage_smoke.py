from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from commerce_video_diagnosis.understanding.core import ProtocolViolation, handle_request  # noqa: E402

BUILD_REQUEST_SCRIPT = Path(__file__).with_name("build_request.py")
_build_request_ns: dict[str, object] = {"__name__": "build_request_module"}
exec(BUILD_REQUEST_SCRIPT.read_text(encoding="utf-8"), _build_request_ns)
build_request = _build_request_ns["build_request"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage smoke flow inside commerce-video-diagnosis: FactPack -> request JSON -> downstream diagnosis.")
    parser.add_argument("--factpack", required=True, help="Path to pure FactPack JSON file.")
    parser.add_argument("--video-id", required=True, help="Video ID passed to downstream request.")
    parser.add_argument("--source-product-id", required=True, help="Source product ID for SSOT routing.")
    parser.add_argument("--request-output", required=True, help="Generated request JSON path.")
    parser.add_argument("--result-output", required=True, help="Downstream result JSON path.")
    parser.add_argument("--ssot", default="", help="Optional override path to SSOT JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_args = argparse.Namespace(
        factpack=args.factpack,
        video_id=args.video_id,
        source_product_id=args.source_product_id,
        output=args.request_output,
        request_id="",
        producer_type="external_vlm",
        generator_version="commerce_video_diagnosis_draft_v1",
    )
    request = build_request(build_args)

    request_output = Path(args.request_output)
    request_output.parent.mkdir(parents=True, exist_ok=True)
    request_output.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        result = handle_request(request, ssot_path=args.ssot or None)
    except ProtocolViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result_output = Path(args.result_output)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(json.dumps(result.dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
