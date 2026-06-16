from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = SKILL_ROOT / "scripts" / "resolve_test_targets.py"
MANIFEST = SKILL_ROOT / "references" / "test_targets.json"



def _run(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RESOLVER), *args], text=True).strip()


@pytest.mark.unit
def test_manifest_has_expected_layers_and_fields() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"unit", "integration", "smoke"}
    assert set(payload["unit"].keys()) == {"files"}
    assert set(payload["integration"].keys()) == {"files"}
    assert set(payload["smoke"].keys()) == {"files", "keyword"}


@pytest.mark.unit
@pytest.mark.parametrize("layer", ["unit", "integration", "smoke"])
def test_resolver_json_and_lines_and_shell_formats_are_exact(layer: str) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = payload[layer]["files"]
    assert json.loads(_run("--layer", layer, "--format", "json")) == files
    assert _run("--layer", layer, "--format", "lines") == "\n".join(files)
    assert _run("--layer", layer, "--format", "shell") == " ".join(files)


@pytest.mark.unit
def test_resolver_keyword_format_contract() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert _run("--layer", "smoke", "--format", "keyword") == payload["smoke"]["keyword"]
    assert _run("--layer", "unit", "--format", "keyword") == ""
    assert _run("--layer", "integration", "--format", "keyword") == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "args, expected_fragment",
    [
        (["--layer", "bad-layer"], "invalid choice"),
        (["--layer", "unit", "--format", "bad-format"], "invalid choice"),
        ([], "the following arguments are required: --layer"),
    ],
)
def test_resolver_invalid_args_fail_explicitly(args: list[str], expected_fragment: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(RESOLVER), *args],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert expected_fragment in combined
