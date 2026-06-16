from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "references" / "test_targets.json"
RESOLVER = SKILL_ROOT / "scripts" / "resolve_test_targets.py"


def _run(layer: str, fmt: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RESOLVER), "--layer", layer, "--format", fmt],
        text=True,
    ).strip()


def test_resolver_json_matches_manifest_exactly() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {layer: cfg["files"] for layer, cfg in manifest.items()}
    actual = {
        "unit": json.loads(_run("unit", "json")),
        "integration": json.loads(_run("integration", "json")),
        "smoke": json.loads(_run("smoke", "json")),
    }
    assert actual == expected


def test_resolver_shell_and_keyword_outputs_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for layer in ("unit", "integration", "smoke"):
        expected_files = manifest[layer]["files"]
        assert _run(layer, "shell") == " ".join(expected_files)
        assert _run(layer, "lines") == "\n".join(expected_files)

    assert _run("smoke", "keyword") == manifest["smoke"]["keyword"]
    assert _run("unit", "keyword") == ""
    assert _run("integration", "keyword") == ""
