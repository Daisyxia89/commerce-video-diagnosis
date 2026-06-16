from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.errors import ProviderExecutionViolation
from extractor.providers.runtime_governance import ProviderRuntimeGovernance


@pytest.mark.unit
def test_runtime_governance_uses_checkpoint_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run(command: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ocr_facts": [{"text": "首屏", "color": "white"}]}), stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    governance = ProviderRuntimeGovernance(workspace_dir=str(tmp_path), provider_name="vision_provider", max_retries=1, backoff_sec=0)
    payload = {"paths": [str(tmp_path / "frame.jpg")], "task": "ocr"}
    frame_path = Path(payload["paths"][0])
    frame_path.write_bytes(b"frame-bytes")

    first = governance.execute_json_command(
        operation_key="seg01:middle",
        script_path="fake_script.py",
        payload=payload,
        failure_label="vision_provider",
    )
    second = governance.execute_json_command(
        operation_key="seg01:middle",
        script_path="fake_script.py",
        payload=payload,
        failure_label="vision_provider",
    )
    third = governance.execute_json_command(
        operation_key="seg02:middle",
        script_path="fake_script.py",
        payload=payload,
        failure_label="vision_provider",
    )

    assert first == second == third
    assert len(calls) == 1
    assert governance.state.checkpoint_hits == 1
    assert governance.state.cache_hits == 1
    assert governance.state.operations["seg01:middle"].source == "checkpoint"
    assert governance.state.operations["seg02:middle"].source == "cache"


@pytest.mark.unit
def test_runtime_governance_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    sleeps: list[int] = []

    def _fake_run(command: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls["count"] += 1
        if calls["count"] == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="429 Resource Exhausted")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"segments": [], "asr_text": "测试"}), stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("extractor.providers.runtime_governance.time.sleep", lambda sec: sleeps.append(sec))

    governance = ProviderRuntimeGovernance(workspace_dir=str(tmp_path), provider_name="asr_provider", max_retries=2, backoff_sec=3)
    payload = {"path": str(tmp_path / "audio.wav"), "task": "asr"}
    Path(payload["path"]).write_bytes(b"audio-bytes")

    result = governance.execute_json_command(
        operation_key="audio:full",
        script_path="fake_audio.py",
        payload=payload,
        failure_label="asr_provider",
    )

    assert result["asr_text"] == "测试"
    assert calls["count"] == 2
    assert sleeps == [3]
    assert governance.state.retry_count == 1
    assert governance.state.operations["audio:full"].attempts == 2
    assert governance.state.operations["audio:full"].status == "success"


@pytest.mark.unit
def test_runtime_governance_blocks_when_budget_exceeded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(command: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ocr_facts": []}), stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    governance = ProviderRuntimeGovernance(
        workspace_dir=str(tmp_path),
        provider_name="vision_provider",
        max_retries=0,
        backoff_sec=0,
        max_requests_per_run=1,
    )
    frame_a = tmp_path / "a.jpg"
    frame_b = tmp_path / "b.jpg"
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")

    governance.execute_json_command(
        operation_key="seg01",
        script_path="fake_image.py",
        payload={"paths": [str(frame_a)], "task": "ocr"},
        failure_label="vision_provider",
    )

    with pytest.raises(ProviderExecutionViolation, match="超出运行时预算"):
        governance.execute_json_command(
            operation_key="seg02",
            script_path="fake_image.py",
            payload={"paths": [str(frame_b)], "task": "ocr"},
            failure_label="vision_provider",
        )
