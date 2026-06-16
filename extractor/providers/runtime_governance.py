from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ProviderExecutionViolation
from ..utils.json_utils import extract_first_json


@dataclass
class ProviderRuntimeOperationRecord:
    operation_key: str
    fingerprint: str
    status: str = "pending"
    source: str = ""
    attempts: int = 0
    last_error: str = ""
    cache_path: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_key": self.operation_key,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "source": self.source,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "cache_path": self.cache_path,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderRuntimeOperationRecord":
        return cls(
            operation_key=str(data.get("operation_key") or ""),
            fingerprint=str(data.get("fingerprint") or ""),
            status=str(data.get("status") or "pending"),
            source=str(data.get("source") or ""),
            attempts=int(data.get("attempts") or 0),
            last_error=str(data.get("last_error") or ""),
            cache_path=str(data.get("cache_path") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class ProviderRuntimeState:
    provider_name: str
    requests_dispatched: int = 0
    cache_hits: int = 0
    checkpoint_hits: int = 0
    retry_count: int = 0
    operations: dict[str, ProviderRuntimeOperationRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "requests_dispatched": self.requests_dispatched,
            "cache_hits": self.cache_hits,
            "checkpoint_hits": self.checkpoint_hits,
            "retry_count": self.retry_count,
            "operations": {key: value.to_dict() for key, value in self.operations.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], provider_name: str) -> "ProviderRuntimeState":
        operations_raw = data.get("operations") or {}
        operations = {
            str(key): ProviderRuntimeOperationRecord.from_dict(value)
            for key, value in operations_raw.items()
            if isinstance(value, dict)
        }
        return cls(
            provider_name=provider_name,
            requests_dispatched=int(data.get("requests_dispatched") or 0),
            cache_hits=int(data.get("cache_hits") or 0),
            checkpoint_hits=int(data.get("checkpoint_hits") or 0),
            retry_count=int(data.get("retry_count") or 0),
            operations=operations,
        )


class ProviderRuntimeGovernance:
    RETRYABLE_FAILURE_FRAGMENTS = (
        "429",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "quota",
    )

    def __init__(
        self,
        *,
        workspace_dir: str,
        provider_name: str,
        max_retries: int = 2,
        backoff_sec: int = 2,
        max_requests_per_run: int = 0,
    ) -> None:
        self.provider_name = provider_name
        self.max_retries = max(0, int(max_retries))
        self.backoff_sec = max(0, int(backoff_sec))
        self.max_requests_per_run = max(0, int(max_requests_per_run))
        self.runtime_dir = Path(workspace_dir) / "provider_runtime" / provider_name
        self.cache_dir = self.runtime_dir / "cache"
        self.state_path = self.runtime_dir / "runtime_state.json"
        self.requests_dispatched_this_run = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def execute_json_command(
        self,
        *,
        operation_key: str,
        script_path: str | Path,
        payload: dict[str, Any],
        failure_label: str,
    ) -> dict[str, Any]:
        fingerprint = self._payload_fingerprint(payload)
        operation = self.state.operations.get(operation_key)
        if operation and operation.status == "success" and operation.fingerprint == fingerprint:
            cached_payload = self._load_cached_payload(fingerprint)
            if cached_payload is not None:
                self.state.checkpoint_hits += 1
                self._record_operation(
                    operation_key=operation_key,
                    fingerprint=fingerprint,
                    status="success",
                    source="checkpoint",
                    attempts=operation.attempts,
                    cache_path=str(self._cache_path_for_fingerprint(fingerprint)),
                )
                return cached_payload

        cached_payload = self._load_cached_payload(fingerprint)
        if cached_payload is not None:
            self.state.cache_hits += 1
            self._record_operation(
                operation_key=operation_key,
                fingerprint=fingerprint,
                status="success",
                source="cache",
                attempts=operation.attempts if operation else 0,
                cache_path=str(self._cache_path_for_fingerprint(fingerprint)),
            )
            return cached_payload

        if self.max_requests_per_run and self.requests_dispatched_this_run >= self.max_requests_per_run:
            raise ProviderExecutionViolation(
                f"{failure_label} 超出运行时预算: provider={self.provider_name}, max_requests_per_run={self.max_requests_per_run}"
            )

        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            self.requests_dispatched_this_run += 1
            self.state.requests_dispatched += 1
            self._persist_state()
            proc = subprocess.run(
                ["python3", str(script_path), json.dumps(payload, ensure_ascii=False)],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                try:
                    data = extract_first_json(proc.stdout)
                except Exception as exc:
                    self._record_operation(
                        operation_key=operation_key,
                        fingerprint=fingerprint,
                        status="failed",
                        source="remote",
                        attempts=attempt,
                        last_error=f"{failure_label} 返回 JSON 解析失败: {exc}",
                    )
                    raise ProviderExecutionViolation(f"{failure_label} 返回 JSON 解析失败: {exc}") from exc
                if not isinstance(data, dict):
                    self._record_operation(
                        operation_key=operation_key,
                        fingerprint=fingerprint,
                        status="failed",
                        source="remote",
                        attempts=attempt,
                        last_error=f"{failure_label} 返回不是对象",
                    )
                    raise ProviderExecutionViolation(f"{failure_label} 返回不是对象")
                cache_path = self._write_cache_payload(fingerprint, data)
                self._record_operation(
                    operation_key=operation_key,
                    fingerprint=fingerprint,
                    status="success",
                    source="remote",
                    attempts=attempt,
                    cache_path=str(cache_path),
                )
                return data

            last_error = self._format_process_error(proc, failure_label)
            if attempt <= self.max_retries and self._is_retryable_failure(proc.stdout, proc.stderr):
                self.state.retry_count += 1
                self._record_operation(
                    operation_key=operation_key,
                    fingerprint=fingerprint,
                    status="retrying",
                    source="remote",
                    attempts=attempt,
                    last_error=last_error,
                )
                self._sleep_before_retry(attempt)
                continue

            self._record_operation(
                operation_key=operation_key,
                fingerprint=fingerprint,
                status="failed",
                source="remote",
                attempts=attempt,
                last_error=last_error,
            )
            raise ProviderExecutionViolation(last_error)

        raise ProviderExecutionViolation(last_error or f"{failure_label} 未知失败")

    def _load_state(self) -> ProviderRuntimeState:
        if not self.state_path.exists():
            return ProviderRuntimeState(provider_name=self.provider_name)
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return ProviderRuntimeState(provider_name=self.provider_name)
        if not isinstance(payload, dict):
            return ProviderRuntimeState(provider_name=self.provider_name)
        return ProviderRuntimeState.from_dict(payload, provider_name=self.provider_name)

    def _persist_state(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.state_path)

    def _record_operation(
        self,
        *,
        operation_key: str,
        fingerprint: str,
        status: str,
        source: str,
        attempts: int,
        last_error: str = "",
        cache_path: str = "",
    ) -> None:
        self.state.operations[operation_key] = ProviderRuntimeOperationRecord(
            operation_key=operation_key,
            fingerprint=fingerprint,
            status=status,
            source=source,
            attempts=attempts,
            last_error=last_error,
            cache_path=cache_path,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._persist_state()

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.backoff_sec <= 0:
            return
        sleep_sec = self.backoff_sec * (2 ** max(attempt - 1, 0))
        time.sleep(sleep_sec)

    def _is_retryable_failure(self, stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}".lower()
        return any(fragment in combined for fragment in self.RETRYABLE_FAILURE_FRAGMENTS)

    def _format_process_error(self, proc: subprocess.CompletedProcess[str], failure_label: str) -> str:
        tail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
        return f"{failure_label} 执行失败: {tail[:300]}"

    def _cache_path_for_fingerprint(self, fingerprint: str) -> Path:
        return self.cache_dir / f"{fingerprint}.json"

    def _write_cache_payload(self, fingerprint: str, payload: dict[str, Any]) -> Path:
        cache_path = self._cache_path_for_fingerprint(fingerprint)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(cache_path)
        return cache_path

    def _load_cached_payload(self, fingerprint: str) -> dict[str, Any] | None:
        cache_path = self._cache_path_for_fingerprint(fingerprint)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _payload_fingerprint(self, payload: dict[str, Any]) -> str:
        normalized_payload = dict(payload)
        input_hashes: dict[str, str] = {}
        for key in ("path", "paths"):
            value = normalized_payload.get(key)
            if isinstance(value, str) and value:
                input_hashes[value] = self._hash_file(Path(value))
            elif isinstance(value, list):
                for item in value:
                    text = str(item or "").strip()
                    if text:
                        input_hashes[text] = self._hash_file(Path(text))
        normalized_payload["_input_hashes"] = input_hashes
        digest = hashlib.sha256(json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _hash_file(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8192)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
