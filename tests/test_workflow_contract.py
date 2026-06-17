from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    # 历史写法 user_skills/<skill-name>/... 自动剖离，改为相对 SKILL_ROOT 解析。
    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == "user_skills":
        return SKILL_ROOT / Path(*parts[2:])
    skill_candidate = SKILL_ROOT / candidate
    if skill_candidate.exists():
        return skill_candidate
    # 回退：优先落在仓库内（SKILL_ROOT），避免 clone 到浅目录时 REPO_ROOT 算成 "/"。
    return skill_candidate

from scripts.check_workflow_contract import scan_required_cli_args, validate_contracts
from scripts.run_smoke_gate import run_smoke_gate
from extractor.smoke_fallback import AUTH_DOWNGRADE_REASON_CODES, classify_smoke_failure

MANIFEST_PATH = SKILL_ROOT / "references" / "test_targets.json"
CLI_CONTRACT_MANIFEST_PATH = SKILL_ROOT / "references" / "script_cli_contracts.json"
RESOLVER_PATH = SKILL_ROOT / "scripts" / "resolve_test_targets.py"


@pytest.mark.unit
def test_ci_entrypoints_are_isomorphic() -> None:
    assert REPO_ROOT == EXPECTED_REPO_ROOT
    # 私有 CI 产物（.github/workflows/*.yml、ci/*.template.yaml）不随公开版发布；
    # 缺失时跳过该契约校验，避免公开版 pytest 误报红。
    ci_workflow = SKILL_ROOT / ".github" / "workflows" / "commerce-video-diagnosis-regression.yml"
    if not ci_workflow.is_file():
        pytest.skip("私有 CI workflow 未随公开版发布，跳过 CI 契约校验")
    validate_contracts()


def _resolve(layer: str, fmt: str) -> str:
    return subprocess.run(
        [sys.executable, str(RESOLVER_PATH), "--layer", layer, "--format", fmt],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer", "expect_keyword"),
    [
        ("unit", False),
        ("integration", False),
        ("smoke", True),
    ],
)
def test_resolver_snapshot_matches_manifest(layer: str, expect_keyword: bool) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    layer_cfg = manifest[layer]
    expected_files = layer_cfg["files"]

    files_json = _resolve(layer, "json")
    shell = _resolve(layer, "shell")
    lines = _resolve(layer, "lines")

    assert json.loads(files_json) == expected_files
    assert shell == " ".join(expected_files)
    assert lines == "\n".join(expected_files)

    keyword = _resolve(layer, "keyword")
    if expect_keyword:
        assert keyword == layer_cfg["keyword"]
    else:
        assert keyword == ""


@pytest.mark.unit
def test_cli_contract_manifest_snapshot() -> None:
    payload = json.loads(CLI_CONTRACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    scripts = payload["argparse_required_scripts"]

    expected_paths = [
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_extractor.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_two_stage_smoke.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_raw_video_smoke.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_smoke_gate.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_raw_video_ocr_feedback_regression.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/assert_full_smoke_outputs.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/build_request.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/run_v2.py")),
        str(_repo_path("user_skills/commerce-video-diagnosis/scripts/check_workflow_contract.py")),
    ]
    assert [_repo_path(entry["path"]) for entry in scripts] == [Path(path) for path in expected_paths]
    assert all(entry["must_fail_on_unknown_args"] is True for entry in scripts)
    assert all("probe_args" in entry for entry in scripts)


@pytest.mark.unit
def test_static_required_args_snapshot() -> None:
    expected = {
        "user_skills/commerce-video-diagnosis/scripts/run_extractor.py": ["--config", "--mode"],
        "user_skills/commerce-video-diagnosis/scripts/run_two_stage_smoke.py": [
            "--factpack",
            "--video-id",
            "--source-product-id",
            "--request-output",
            "--result-output",
        ],
        "user_skills/commerce-video-diagnosis/scripts/run_raw_video_smoke.py": [
            "--video",
            "--video-id",
            "--source-product-id",
        ],
        "user_skills/commerce-video-diagnosis/scripts/run_smoke_gate.py": [
            "--video",
            "--video-id",
            "--source-product-id",
            "--workspace",
            "--decision-log",
        ],
        "user_skills/commerce-video-diagnosis/scripts/run_raw_video_ocr_feedback_regression.py": [],
        "user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py": ["--layer"],
        "user_skills/commerce-video-diagnosis/scripts/assert_full_smoke_outputs.py": ["--workspace"],
        "user_skills/commerce-video-diagnosis/scripts/build_request.py": [
            "--factpack",
            "--video-id",
            "--source-product-id",
            "--output",
        ],
        "user_skills/commerce-video-diagnosis/scripts/run_v2.py": ["--payload"],
        "user_skills/commerce-video-diagnosis/scripts/check_workflow_contract.py": [],
    }
    actual = {
        path: scan_required_cli_args(_repo_path(path))
        for path in expected
    }
    assert actual == expected


@pytest.fixture
def smoke_gate_args(tmp_path: Path) -> Namespace:
    workspace = tmp_path / "smoke-workspace"
    return Namespace(
        video="dummy.mp4",
        video_id="v1",
        source_product_id="p1",
        workspace=str(workspace),
        decision_log=str(workspace / "smoke_gate_log.json"),
        source_platform="抖音",
        output=str(workspace / "combined_smoke_result.json"),
        include_ocr_regression_summary=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("combined_output", "degradable", "reason_code"),
    [
        ("provider failed: AUTH_MISSING_TOKEN", True, "AUTH_MISSING_TOKEN"),
        ("provider failed: AUTH_PERMISSION_DENIED", True, "AUTH_PERMISSION_DENIED"),
        ("provider failed: AUTH_PROVIDER_UNAVAILABLE", True, "AUTH_PROVIDER_UNAVAILABLE"),
        ("Traceback: assertion failed", False, "NON_DEGRADABLE_FAILURE"),
    ],
)
def test_smoke_failure_classifier(combined_output: str, degradable: bool, reason_code: str) -> None:
    payload = classify_smoke_failure(combined_output)
    assert payload["degradable"] is degradable
    assert payload["reason_code"] == reason_code


@pytest.mark.unit
@pytest.mark.parametrize("reason_code", AUTH_DOWNGRADE_REASON_CODES)
def test_run_smoke_gate_degrades_for_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
    smoke_gate_args: Namespace,
    capsys: pytest.CaptureFixture[str],
    reason_code: str,
) -> None:
    def _fake_run(command: list[str], text: bool, capture_output: bool) -> subprocess.CompletedProcess[str]:
        assert text is True
        assert capture_output is True
        return subprocess.CompletedProcess(command, 17, stdout="provider failed", stderr=f"fatal: {reason_code}")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    exit_code = run_smoke_gate(smoke_gate_args)

    assert exit_code == 0
    payload = json.loads(Path(smoke_gate_args.decision_log).read_text(encoding="utf-8"))
    assert payload["mode"] == "degraded_pytest_only"
    assert payload["status"] == "degraded"
    assert payload["reason_code"] == reason_code
    assert payload["matched_fragment"] == reason_code
    assert "run_raw_video_smoke.py" in " ".join(payload["command"])
    stdout = capsys.readouterr().out
    assert reason_code in stdout


@pytest.mark.unit
def test_run_smoke_gate_does_not_degrade_non_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
    smoke_gate_args: Namespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fake_run(command: list[str], text: bool, capture_output: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 23, stdout="", stderr="Traceback: assertion failed")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    exit_code = run_smoke_gate(smoke_gate_args)

    assert exit_code == 23
    payload = json.loads(Path(smoke_gate_args.decision_log).read_text(encoding="utf-8"))
    assert payload["mode"] == "failed"
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "NON_DEGRADABLE_FAILURE"
    stderr = capsys.readouterr().err
    assert "NON_DEGRADABLE_FAILURE" in stderr

