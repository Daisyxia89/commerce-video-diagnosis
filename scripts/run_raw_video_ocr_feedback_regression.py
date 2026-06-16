from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(SKILL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.entry import run_extractor
from run_raw_video_smoke import build_ocr_regression_summary

DEFAULT_CONFIG = "user_skills/commerce-video-diagnosis/fixtures/raw_video_regression_config.json"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw video OCR feedback regression through extractor two-stage entry."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to raw video regression config.")
    parser.add_argument("--output", default="", help="Optional path to write regression summary JSON.")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    result = run_extractor(args.config, mode="two-stage-run")

    regression_summary = build_ocr_regression_summary(
        workspace=Path("output/raw_video_regression"),
        result=result,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(regression_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(regression_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
