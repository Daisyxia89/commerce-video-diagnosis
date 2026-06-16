from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_CONTRACT_MANIFEST = SKILL_ROOT / "references" / "script_cli_contracts.json"


def _resolve_repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _load_manifest() -> list[dict[str, object]]:
    payload = json.loads(CLI_CONTRACT_MANIFEST.read_text(encoding="utf-8"))
    return payload["argparse_required_scripts"]


@pytest.mark.unit
def test_cli_contract_manifest_paths_exist() -> None:
    for entry in _load_manifest():
        script_path = _resolve_repo_path(str(entry["path"]))
        assert script_path.is_file(), f"missing script declared in cli contract manifest: {script_path}"


@pytest.mark.unit
@pytest.mark.parametrize("entry", _load_manifest())
def test_helper_scripts_fail_on_unknown_args(entry: dict[str, object]) -> None:
    script_path = _resolve_repo_path(str(entry["path"]))
    probe_args = list(entry.get("probe_args", []))
    proc = subprocess.run(
        [sys.executable, str(script_path), *probe_args, "--unexpected-arg"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "unrecognized arguments: --unexpected-arg" in combined


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_rel_path, args, expected_fragment",
    [
        (
            "user_skills/commerce-video-diagnosis/scripts/assert_full_smoke_outputs.py",
            [],
            "the following arguments are required: --workspace",
        ),
        (
            "user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py",
            [],
            "the following arguments are required: --layer",
        ),
        (
            "user_skills/commerce-video-diagnosis/scripts/run_extractor.py",
            [],
            "the following arguments are required: --config, --mode",
        ),
    ],
)
def test_helper_scripts_fail_when_required_args_missing(
    script_rel_path: str,
    args: list[str],
    expected_fragment: str,
) -> None:
    script_path = _resolve_repo_path(script_rel_path)
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert expected_fragment in combined
