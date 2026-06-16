from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "references" / "test_targets.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve SSOT test target list for extractor CI entrypoints.")
    parser.add_argument("--layer", required=True, choices=["unit", "integration", "smoke"])
    parser.add_argument("--format", choices=["shell", "json", "lines", "keyword"], default="shell")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_manifest()
    layer_cfg = payload[args.layer]
    targets = layer_cfg["files"]
    if args.format == "json":
        print(json.dumps(targets, ensure_ascii=False))
    elif args.format == "lines":
        print("\n".join(targets))
    elif args.format == "keyword":
        print(str(layer_cfg.get("keyword", "")))
    else:
        print(" ".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
