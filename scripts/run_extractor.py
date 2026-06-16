from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(SKILL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.entry import run_extractor



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video factpack extractor unified entry.")
    parser.add_argument("--config", required=True, help="Path to extractor config json.")
    parser.add_argument("--mode", required=True, choices=["validate-only", "extract-only", "build-request", "two-stage-run"])
    parser.add_argument("--ssot", default="", help="Optional SSOT override path for downstream two-stage run.")
    parser.add_argument("--output", default="", help="Optional path to write final tool result JSON.")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    result = run_extractor(config_path=args.config, mode=args.mode, ssot_path=args.ssot)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
