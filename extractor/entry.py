from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .adapters.normalize import normalize_provider_outputs
from .assembly.factpack_builder import build_factpack
from .config.loader import load_config
from .handoff.downstream_runner import run_downstream
from .handoff.request_builder import build_request, write_json
from .preprocess.pipeline import run_preprocess
from .providers.orchestrator import ProviderOrchestrator
from .validators.config_assertions import assert_config_valid, assert_local_dependencies
from .validators.contamination_assertions import assert_no_contamination
from .validators.factpack_assertions import assert_factpack_schema
from .validators.preprocess_assertions import assert_preprocess_output



def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def _probe_video_meta(video_path: str, ffprobe_path: str) -> dict[str, Any]:
    args = [ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-show_entries", "format=duration", "-of", "json", video_path]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessViolation(f"ffprobe 失败: {proc.stderr[:300]}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise PreprocessViolation("视频缺少 video stream")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    rate = str(stream.get("r_frame_rate") or "0/1")
    num, den = rate.split("/")
    fps = round(float(num) / float(den), 3) if float(den) else 0.0
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    return {"duration_sec": duration, "fps": fps, "resolution": f"{width}x{height}"}



def _derive_video_meta(config) -> dict:
    if config.input.factpack_path:
        factpack = _read_json(config.input.factpack_path)
        return factpack["video_meta"]
    if config.input.video_path:
        ffprobe = config.local_tools.ffprobe_path or "ffprobe"
        meta = _probe_video_meta(config.input.video_path, ffprobe)
        meta["source_platform"] = config.runtime.source_platform or "unknown"
        return meta
    return {"source_platform": config.runtime.source_platform or "unknown", "duration_sec": 0.0, "fps": 0.0, "resolution": "0x0"}



def _need_real_preprocess(config) -> bool:
    providers = [config.providers.vlm, config.providers.asr, config.providers.ocr]
    return any(provider.enabled and provider.provider != "fixture_file" for provider in providers)



def run_extractor(config_path: str, mode: str, ssot_path: str = "") -> dict:
    config = load_config(config_path)
    assert_config_valid(config)
    assert_local_dependencies(config)

    if mode == "validate-only":
        return {"status": "validated", "config_path": config_path}

    if config.input.factpack_path:
        factpack = _read_json(config.input.factpack_path)
    else:
        preproc = None
        if _need_real_preprocess(config):
            ffmpeg = config.local_tools.ffmpeg_path or "ffmpeg"
            ffprobe = config.local_tools.ffprobe_path or "ffprobe"
            preproc = run_preprocess(
                video_path=config.input.video_path,
                workspace_dir=config.local_tools.workspace_dir or "output/commerce_video_diagnosis_runtime",
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                source_platform=config.runtime.source_platform or "unknown",
                enable_real_ocr_feedback=config.runtime.enable_real_ocr_feedback,
                ocr_feedback_top_k=config.runtime.ocr_feedback_top_k,
                provider_runtime_max_retries=config.runtime.provider_runtime_max_retries,
                provider_runtime_backoff_sec=config.runtime.provider_runtime_backoff_sec,
                provider_runtime_max_requests_per_run=config.runtime.provider_runtime_max_requests_per_run,
                provider_fallback_mode=config.runtime.provider_fallback_mode,
            )
            assert_preprocess_output(preproc)
            video_meta = preproc["video_meta"]
        else:
            video_meta = _derive_video_meta(config)
        raw_bundle = ProviderOrchestrator(config).run(preproc=preproc)
        normalized = normalize_provider_outputs(raw_bundle)
        factpack = build_factpack(normalized, video_meta, preproc=preproc)

    assert_no_contamination(factpack)
    assert_factpack_schema(factpack)

    if config.output.factpack_path:
        write_json(config.output.factpack_path, factpack)

    if mode == "extract-only":
        return {"status": "extracted", "factpack": factpack}

    request = build_request(
        factpack=factpack,
        video_id=config.input.video_id,
        source_product_id=config.input.source_product_id,
        request_id=config.input.request_id,
    )
    if config.output.request_path:
        write_json(config.output.request_path, request)

    if mode == "build-request":
        return {"status": "request_built", "request": request}

    if mode != "two-stage-run":
        raise ValueError(f"不支持的 mode: {mode}")
    result = run_downstream(request, ssot_path=ssot_path)
    if config.output.result_path:
        write_json(config.output.result_path, result)
    return {"status": "two_stage_done", "result": result}
