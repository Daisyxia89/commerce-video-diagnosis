from __future__ import annotations

from pathlib import Path
import io
import json
import sys

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(TESTS_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.errors import ConfigViolation
from extractor.models.config_models import (
    ExtractorConfig,
    InputConfig,
    LocalToolConfig,
    OutputConfig,
    ProviderConfig,
    ProvidersConfig,
    RuntimeConfig,
)
from extractor.providers import byok_provider
from extractor.providers.orchestrator import ProviderOrchestrator
from extractor.validators.config_assertions import assert_config_valid



def _base_config(tmp_path: Path) -> ExtractorConfig:
    return ExtractorConfig(
        runtime=RuntimeConfig(),
        local_tools=LocalToolConfig(workspace_dir=str(tmp_path)),
        input=InputConfig(video_path="demo.mp4", video_id="vid-demo", source_product_id="prod-demo"),
        providers=ProvidersConfig(),
        output=OutputConfig(),
    )


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _header(request, name: str) -> str | None:
    target = name.lower()
    for key, value in request.header_items():
        if key.lower() == target:
            return value
    return request.headers.get(name) or request.headers.get(name.lower())


@pytest.mark.unit
def test_resolve_remote_audio_url_prefers_existing_audio_url() -> None:
    payload = {
        "path": "/tmp/local.wav",
        "timeout_sec": 30,
        "extra": {"audio_url": "https://example.com/audio.wav", "upload_provider": "oss"},
    }

    assert byok_provider._resolve_remote_audio_url(payload) == "https://example.com/audio.wav"


@pytest.mark.unit
def test_external_asr_provider_uses_byok_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.asr = ProviderConfig(
        enabled=True,
        provider="openai_whisper",
        endpoint="https://api.openai.com/v1",
        api_key="sk-demo",
        model="whisper-1",
        timeout_sec=120,
        adapter="default_asr_adapter",
    )
    orchestrator = ProviderOrchestrator(config)

    calls: list[dict] = []

    def _fake_execute_json_command(**kwargs):
        calls.append(kwargs)
        return {
            "text": "第一句。第二句。",
            "segments": [
                {"start_sec": 0.0, "end_sec": 1.5, "text": "第一句。"},
                {"start_sec": 1.5, "end_sec": 3.0, "text": "第二句。"},
            ],
        }

    monkeypatch.setattr(orchestrator.asr_byok_runtime, "execute_json_command", _fake_execute_json_command)

    preproc = {
        "audio_path": str(tmp_path / "audio.wav"),
        "segments": [
            {"segment_id": "SEG01", "start_sec": 0.0, "end_sec": 1.5},
            {"segment_id": "SEG02", "start_sec": 1.5, "end_sec": 3.0},
        ],
    }
    Path(preproc["audio_path"]).write_bytes(b"audio")

    rows = orchestrator._run_asr_provider(preproc)

    assert calls
    assert calls[0]["payload"]["request_mode"] == "openai_audio_transcription"
    assert calls[0]["payload"]["model"] == "whisper-1"
    assert [row["audio_facts"]["asr_text"] for row in rows] == ["第一句。", "第二句。"]


@pytest.mark.unit
def test_external_vlm_provider_uses_byok_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.vlm = ProviderConfig(
        enabled=True,
        provider="openai_gpt4o_vision",
        endpoint="https://api.openai.com/v1",
        api_key="sk-demo",
        model="gpt-4o",
        timeout_sec=120,
        adapter="default_vlm_adapter",
    )
    orchestrator = ProviderOrchestrator(config)

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")

    calls: list[tuple[str, str, str]] = []

    def _fake_analyze(frame_path_arg: str, segment_id: str, role: str) -> dict:
        calls.append((frame_path_arg, segment_id, role))
        return {
            "visual_subject": "一只手拿着产品",
            "shot_size": "close-up",
            "camera_movement": "static",
            "lighting_tone": "bright natural daylight",
            "key_objects": ["手", "产品"],
            "actions": [{"action_name": "拿起", "physical_intensity": "low"}],
        }

    monkeypatch.setattr(orchestrator, "_analyze_single_frame_vlm", _fake_analyze)

    preproc = {
        "segments": [
            {
                "segment_id": "SEG01",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "frame_path": str(frame_path),
                "frames": [{"frame_path": str(frame_path), "sampling_role": "middle"}],
            }
        ]
    }

    rows = orchestrator._run_vlm_provider(preproc)

    assert calls == [(str(frame_path), "SEG01", "middle")]
    assert rows[0]["visual_facts"]["visual_subject"] == "一只手拿着产品"
    assert rows[0]["visual_facts"]["shot_size"] == "close_up"
    assert rows[0]["rhythm_facts"]["transition_type"] == "hard_cut"


@pytest.mark.unit
def test_external_ocr_provider_uses_byok_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.ocr = ProviderConfig(
        enabled=True,
        provider="openai_gpt4o_vision",
        endpoint="https://api.openai.com/v1",
        api_key="sk-demo",
        model="gpt-4o",
        timeout_sec=120,
        adapter="default_ocr_adapter",
    )
    orchestrator = ProviderOrchestrator(config)

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")

    def _fake_ocr(frame_path_arg: str, segment_id: str, role: str) -> dict:
        assert frame_path_arg == str(frame_path)
        assert segment_id == "SEG01"
        assert role == "middle"
        return {
            "ocr_facts": [
                {
                    "text": "真实字幕",
                    "position": {"x": 0.2, "y": 0.7, "w": 0.4, "h": 0.05},
                    "color": "#FFFFFF",
                    "font_family": "Source Han Sans",
                    "font_weight": "bold",
                    "font_size_level": "large",
                    "stroke_style": "none",
                    "text_effect_style": "solid_fill",
                }
            ]
        }

    monkeypatch.setattr(orchestrator, "_analyze_single_frame_ocr", _fake_ocr)

    preproc = {
        "segments": [
            {
                "segment_id": "SEG01",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "frame_path": str(frame_path),
                "frames": [{"frame_path": str(frame_path), "sampling_role": "middle"}],
                "frame_plan": {"upsampling_triggers": [], "metrics": {}},
            }
        ]
    }

    rows = orchestrator._run_ocr_provider(preproc)

    assert rows[0]["ocr_facts"][0]["text"] == "真实字幕"
    assert rows[0]["ocr_facts"][0]["position"]["x"] == 0.2


@pytest.mark.unit
def test_external_provider_requires_model_and_api_key(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.vlm = ProviderConfig(
        enabled=True,
        provider="openai_gpt4o_vision",
        endpoint="https://api.openai.com/v1",
        timeout_sec=120,
        adapter="default_vlm_adapter",
    )

    with pytest.raises(ConfigViolation) as exc_info:
        assert_config_valid(config)

    assert "provider vlm 缺少字段" in str(exc_info.value) or "provider vlm 缺少 api_key" in str(exc_info.value)


@pytest.mark.unit
def test_upload_local_audio_to_oss_and_return_public_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"demo-audio")
    captured: dict[str, object] = {}

    monkeypatch.setattr(byok_provider.uuid, "uuid4", lambda: type("_FixedUUID", (), {"hex": "ossfixed"})())
    monkeypatch.setattr(byok_provider, "_http_date_now", lambda: "Tue, 16 Jun 2026 01:23:45 GMT")

    def _fake_urlopen(request, timeout=0):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHttpResponse({})

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)

    result = byok_provider._resolve_remote_audio_url(
        {
            "path": str(audio_path),
            "timeout_sec": 12,
            "extra": {
                "upload_provider": "oss",
                "upload_endpoint": "oss-cn-shanghai.aliyuncs.com",
                "upload_bucket": "demo-bucket",
                "upload_access_key_id": "ak-demo",
                "upload_access_key_secret": "sk-demo",
                "upload_object_prefix": "demo/audio",
            },
        }
    )

    request = captured["request"]
    assert request.full_url == "https://demo-bucket.oss-cn-shanghai.aliyuncs.com/demo/audio/ossfixed.wav"
    assert _header(request, "Authorization").startswith("OSS ak-demo:")
    assert _header(request, "Date") == "Tue, 16 Jun 2026 01:23:45 GMT"
    assert captured["timeout"] == 12
    assert result == "https://demo-bucket.oss-cn-shanghai.aliyuncs.com/demo/audio/ossfixed.wav"


@pytest.mark.unit
def test_upload_local_audio_to_tos_and_return_public_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"demo-audio")
    captured: dict[str, object] = {}

    monkeypatch.setattr(byok_provider.uuid, "uuid4", lambda: type("_FixedUUID", (), {"hex": "tosfixed"})())
    monkeypatch.setattr(byok_provider, "_iso8601_basic_now", lambda: "20260616T012345Z")

    def _fake_urlopen(request, timeout=0):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHttpResponse({})

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)

    result = byok_provider._resolve_remote_audio_url(
        {
            "path": str(audio_path),
            "timeout_sec": 15,
            "extra": {
                "upload_provider": "tos",
                "upload_endpoint": "tos-cn-beijing.volces.com",
                "upload_bucket": "demo-bucket",
                "upload_region": "cn-beijing",
                "upload_access_key_id": "ak-demo",
                "upload_access_key_secret": "sk-demo",
                "upload_object_prefix": "demo/audio",
            },
        }
    )

    request = captured["request"]
    assert request.full_url == "https://demo-bucket.tos-cn-beijing.volces.com/demo/audio/tosfixed.wav"
    assert _header(request, "Authorization").startswith("TOS4-HMAC-SHA256 Credential=ak-demo/20260616/cn-beijing/tos/request")
    assert _header(request, "X-Tos-Date") == "20260616T012345Z"
    assert _header(request, "X-Tos-Content-Sha256")
    assert captured["timeout"] == 15
    assert result == "https://demo-bucket.tos-cn-beijing.volces.com/demo/audio/tosfixed.wav"


@pytest.mark.unit
def test_domestic_asr_without_audio_url_or_upload_config_crashes_early(tmp_path: Path) -> None:
    audio_path = tmp_path / "local.wav"
    audio_path.write_bytes(b"audio")

    with pytest.raises(Exception) as exc_info:
        byok_provider._resolve_remote_audio_url({"path": str(audio_path), "timeout_sec": 30, "extra": {}})

    assert "upload_provider=oss/tos" in str(exc_info.value)


@pytest.mark.unit
def test_external_asr_provider_uses_aliyun_request_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.asr = ProviderConfig(
        enabled=True,
        provider="aliyun_asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        api_key="sk-demo",
        model="paraformer-v2",
        timeout_sec=120,
        adapter="aliyun_asr",
        extra={"audio_url": "https://example.com/audio.wav"},
    )
    orchestrator = ProviderOrchestrator(config)

    calls: list[dict] = []

    def _fake_execute_json_command(**kwargs):
        calls.append(kwargs)
        return {
            "text": "第一句。第二句。",
            "segments": [
                {"start_sec": 0.0, "end_sec": 1.5, "text": "第一句。"},
                {"start_sec": 1.5, "end_sec": 3.0, "text": "第二句。"},
            ],
        }

    monkeypatch.setattr(orchestrator.asr_byok_runtime, "execute_json_command", _fake_execute_json_command)

    preproc = {
        "audio_path": str(tmp_path / "audio.wav"),
        "segments": [
            {"segment_id": "SEG01", "start_sec": 0.0, "end_sec": 1.5},
            {"segment_id": "SEG02", "start_sec": 1.5, "end_sec": 3.0},
        ],
    }
    Path(preproc["audio_path"]).write_bytes(b"audio")

    orchestrator._run_asr_provider(preproc)

    assert calls[0]["payload"]["request_mode"] == "aliyun_asr"
    assert calls[0]["payload"]["extra"]["audio_url"] == "https://example.com/audio.wav"


@pytest.mark.unit
def test_external_asr_provider_uses_volcengine_request_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.providers.asr = ProviderConfig(
        enabled=True,
        provider="volcengine_asr",
        endpoint="https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit",
        api_key="ak-demo",
        model="bigmodel",
        timeout_sec=120,
        adapter="volcengine_asr",
        extra={"audio_url": "https://example.com/audio.wav", "app_key": "app-demo"},
    )
    orchestrator = ProviderOrchestrator(config)

    calls: list[dict] = []

    def _fake_execute_json_command(**kwargs):
        calls.append(kwargs)
        return {
            "text": "第一句。第二句。",
            "segments": [
                {"start_sec": 0.0, "end_sec": 1.5, "text": "第一句。"},
                {"start_sec": 1.5, "end_sec": 3.0, "text": "第二句。"},
            ],
        }

    monkeypatch.setattr(orchestrator.asr_byok_runtime, "execute_json_command", _fake_execute_json_command)

    preproc = {
        "audio_path": str(tmp_path / "audio.wav"),
        "segments": [
            {"segment_id": "SEG01", "start_sec": 0.0, "end_sec": 1.5},
            {"segment_id": "SEG02", "start_sec": 1.5, "end_sec": 3.0},
        ],
    }
    Path(preproc["audio_path"]).write_bytes(b"audio")

    orchestrator._run_asr_provider(preproc)

    assert calls[0]["payload"]["request_mode"] == "volcengine_asr"
    assert calls[0]["payload"]["extra"]["app_key"] == "app-demo"


@pytest.mark.unit
def test_aliyun_asr_adapter_normalizes_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {"output": {"task_id": "task-123"}},
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "transcripts": [
                                {
                                    "text": "你好世界",
                                    "sentences": [
                                        {"begin_time": 0, "end_time": 1200, "text": "你好"},
                                        {"begin_time": 1200, "end_time": 2500, "text": "世界"},
                                    ],
                                }
                            ]
                        }
                    ],
                }
            },
        ]
    )

    def _fake_urlopen(request, timeout=0):
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(byok_provider.time, "sleep", lambda *_args, **_kwargs: None)

    result = byok_provider._call_aliyun_asr(
        {
            "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            "api_key": "sk-demo",
            "model": "paraformer-v2",
            "timeout_sec": 30,
            "extra": {"audio_url": "https://example.com/audio.wav", "poll_interval_sec": 0},
        }
    )

    assert result["text"] == "你好世界"
    assert result["segments"] == [
        {"start_sec": 0.0, "end_sec": 1.2, "text": "你好"},
        {"start_sec": 1.2, "end_sec": 2.5, "text": "世界"},
    ]


@pytest.mark.unit
def test_aliyun_asr_adapter_reads_transcription_url_when_query_response_has_no_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {"output": {"task_id": "task-123"}},
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "output": {
                                "results": [
                                    {
                                        "transcription_url": "https://example.com/transcription.json",
                                        "subtask_status": "SUCCEEDED",
                                    }
                                ],
                                "transcription_url": "https://example.com/transcription.json",
                            }
                        }
                    ],
                }
            },
            {
                "file_url": "https://example.com/audio.wav",
                "transcripts": [
                    {
                        "text": "你好世界",
                        "sentences": [
                            {"begin_time": 0, "end_time": 1200, "text": "你好"},
                            {"begin_time": 1200, "end_time": 2500, "text": "世界"},
                        ],
                    }
                ],
            },
        ]
    )

    def _fake_urlopen(request, timeout=0):
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(byok_provider.time, "sleep", lambda *_args, **_kwargs: None)

    result = byok_provider._call_aliyun_asr(
        {
            "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            "api_key": "sk-demo",
            "model": "paraformer-v2",
            "timeout_sec": 30,
            "extra": {"audio_url": "https://example.com/audio.wav", "poll_interval_sec": 0},
        }
    )

    assert result["text"] == "你好世界"
    assert result["segments"] == [
        {"start_sec": 0.0, "end_sec": 1.2, "text": "你好"},
        {"start_sec": 1.2, "end_sec": 2.5, "text": "世界"},
    ]


@pytest.mark.unit
def test_aliyun_asr_adapter_uploads_local_audio_then_submits_and_polls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "aliyun_local.wav"
    audio_path.write_bytes(b"aliyun-audio")
    captured_requests: list[dict[str, object]] = []
    responses = iter(
        [
            {},
            {"output": {"task_id": "task-upload-aliyun"}},
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "transcripts": [
                                {
                                    "text": "自动上传成功",
                                    "sentences": [
                                        {"begin_time": 0, "end_time": 1800, "text": "自动上传成功"}
                                    ],
                                }
                            ]
                        }
                    ],
                }
            },
        ]
    )

    monkeypatch.setattr(byok_provider.uuid, "uuid4", lambda: type("_FixedUUID", (), {"hex": "aliyunupload"})())
    monkeypatch.setattr(byok_provider, "_http_date_now", lambda: "Tue, 16 Jun 2026 02:00:00 GMT")
    monkeypatch.setattr(byok_provider.time, "sleep", lambda *_args, **_kwargs: None)

    def _fake_urlopen(request, timeout=0):
        captured_requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "body": request.data,
                "timeout": timeout,
            }
        )
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)

    result = byok_provider._call_aliyun_asr(
        {
            "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            "api_key": "sk-demo",
            "model": "paraformer-v2",
            "timeout_sec": 30,
            "path": str(audio_path),
            "extra": {
                "upload_provider": "oss",
                "upload_endpoint": "oss-cn-shanghai.aliyuncs.com",
                "upload_bucket": "demo-bucket",
                "upload_access_key_id": "upload-ak",
                "upload_access_key_secret": "upload-sk",
                "upload_object_prefix": "demo/audio",
                "poll_interval_sec": 0,
            },
        }
    )

    assert len(captured_requests) == 3
    upload_request, submit_request, poll_request = captured_requests
    assert upload_request["url"] == "https://demo-bucket.oss-cn-shanghai.aliyuncs.com/demo/audio/aliyunupload.wav"
    assert upload_request["method"] == "PUT"
    assert upload_request["headers"]["Authorization"].startswith("OSS upload-ak:")
    submit_payload = json.loads((submit_request["body"] or b"{}").decode("utf-8"))
    assert submit_request["method"] == "POST"
    assert submit_request["headers"]["X-dashscope-async"] == "enable"
    assert submit_payload["input"]["file_urls"] == [
        "https://demo-bucket.oss-cn-shanghai.aliyuncs.com/demo/audio/aliyunupload.wav"
    ]
    assert poll_request["method"] == "GET"
    assert poll_request["url"] == "https://dashscope.aliyuncs.com/api/v1/tasks/task-upload-aliyun"
    assert result["text"] == "自动上传成功"
    assert result["segments"] == [{"start_sec": 0.0, "end_sec": 1.8, "text": "自动上传成功"}]


@pytest.mark.unit
def test_volcengine_asr_adapter_normalizes_utterances(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {"id": "task-456"},
            {
                "result": {
                    "task_status": "SUCCESS",
                    "text": "这是字节跳动",
                    "utterances": [
                        {"start_time": 0, "end_time": 1705, "text": "这是字节跳动"}
                    ],
                }
            },
        ]
    )

    def _fake_urlopen(request, timeout=0):
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(byok_provider.time, "sleep", lambda *_args, **_kwargs: None)

    result = byok_provider._call_volcengine_asr(
        {
            "endpoint": "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit",
            "api_key": "ak-demo",
            "model": "bigmodel",
            "timeout_sec": 30,
            "extra": {
                "audio_url": "https://example.com/audio.wav",
                "app_key": "app-demo",
                "resource_id": "volc.seedasr.auc",
                "poll_interval_sec": 0,
            },
        }
    )

    assert result["text"] == "这是字节跳动"
    assert result["segments"] == [{"start_sec": 0.0, "end_sec": 1.705, "text": "这是字节跳动"}]


@pytest.mark.unit
def test_volcengine_asr_adapter_uploads_local_audio_then_submits_and_polls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "volc_local.wav"
    audio_path.write_bytes(b"volc-audio")
    captured_requests: list[dict[str, object]] = []
    responses = iter(
        [
            {},
            {"id": "task-upload-volc"},
            {
                "result": {
                    "task_status": "SUCCESS",
                    "text": "火山上传成功",
                    "utterances": [
                        {"start_time": 0, "end_time": 2000, "text": "火山上传成功"}
                    ],
                }
            },
        ]
    )

    monkeypatch.setattr(byok_provider.uuid, "uuid4", lambda: type("_FixedUUID", (), {"hex": "volcupload"})())
    monkeypatch.setattr(byok_provider, "_iso8601_basic_now", lambda: "20260616T031500Z")
    monkeypatch.setattr(byok_provider.time, "sleep", lambda *_args, **_kwargs: None)

    def _fake_urlopen(request, timeout=0):
        captured_requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "body": request.data,
                "timeout": timeout,
            }
        )
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(byok_provider.urllib.request, "urlopen", _fake_urlopen)

    result = byok_provider._call_volcengine_asr(
        {
            "endpoint": "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit",
            "api_key": "ak-demo",
            "model": "bigmodel",
            "timeout_sec": 30,
            "path": str(audio_path),
            "extra": {
                "upload_provider": "tos",
                "upload_endpoint": "tos-cn-beijing.volces.com",
                "upload_bucket": "demo-bucket",
                "upload_region": "cn-beijing",
                "upload_access_key_id": "upload-ak",
                "upload_access_key_secret": "upload-sk",
                "upload_object_prefix": "demo/audio",
                "app_key": "app-demo",
                "resource_id": "volc.seedasr.auc",
                "poll_interval_sec": 0,
            },
        }
    )

    assert len(captured_requests) == 3
    upload_request, submit_request, poll_request = captured_requests
    assert upload_request["url"] == "https://demo-bucket.tos-cn-beijing.volces.com/demo/audio/volcupload.wav"
    assert upload_request["method"] == "PUT"
    assert upload_request["headers"]["Authorization"].startswith(
        "TOS4-HMAC-SHA256 Credential=upload-ak/20260616/cn-beijing/tos/request"
    )
    submit_payload = json.loads((submit_request["body"] or b"{}").decode("utf-8"))
    assert submit_request["method"] == "POST"
    assert submit_request["headers"]["X-api-app-key"] == "app-demo"
    assert submit_request["headers"]["X-api-access-key"] == "ak-demo"
    assert submit_payload["audio"]["url"] == "https://demo-bucket.tos-cn-beijing.volces.com/demo/audio/volcupload.wav"
    assert poll_request["method"] == "POST"
    assert poll_request["url"] == "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
    assert poll_request["headers"]["X-api-request-id"] == "task-upload-volc"
    assert poll_request["headers"]["X-api-sequence"] == "-1"
    assert result["text"] == "火山上传成功"
    assert result["segments"] == [{"start_sec": 0.0, "end_sec": 2.0, "text": "火山上传成功"}]


@pytest.mark.unit
def test_domestic_asr_requires_remote_audio_url() -> None:
    with pytest.raises(Exception) as exc_info:
        byok_provider._call_aliyun_asr(
            {
                "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
                "api_key": "sk-demo",
                "model": "paraformer-v2",
                "timeout_sec": 30,
                "path": "/tmp/local.wav",
            }
        )

    assert "audio_url" in str(exc_info.value)
