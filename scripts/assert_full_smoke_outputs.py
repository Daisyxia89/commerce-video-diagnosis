from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = SKILL_ROOT / "tests"
for candidate in (str(REPO_ROOT), str(SKILL_ROOT), str(TESTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from full_smoke_assertions import assert_full_smoke_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert full smoke outputs for commerce-video-diagnosis.")
    parser.add_argument("--workspace", required=True, help="Workspace directory produced by run_raw_video_smoke.py")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    assert_full_smoke_workspace(Path(args.workspace))
    print("full smoke output assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
