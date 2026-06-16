from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from commerce_video_diagnosis.understanding.core import ProtocolViolation, handle_request  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run commerce_video_diagnosis.understanding with a FactPack or AssetPackage payload."
    )
    parser.add_argument("--payload", required=True, help="Path to request JSON payload.")
    parser.add_argument("--ssot", default="", help="Optional override path to SSOT JSON.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload_path = Path(args.payload)
    raw_payload = json.loads(payload_path.read_text(encoding="utf-8"))

    try:
        result = handle_request(raw_payload, ssot_path=args.ssot or None)
    except ProtocolViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_text = json.dumps(result.dict(), ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
