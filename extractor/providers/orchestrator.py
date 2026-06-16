from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from ..errors import ProviderExecutionViolation
from ..models.config_models import ExtractorConfig, ProviderConfig
from .runtime_governance import ProviderRuntimeGovernance

BYOK_PROVIDER = Path(__file__).resolve().with_name("byok_provider.py")


class ProviderOrchestrator:
    OCR_BACKFILL_TRIGGER_BOXES = 12
    OCR_BACKFILL_FRAME_LIMIT = 3
    OCR_BACKFILL_ROLE_PRIORITY = {
        "dense_text": 0,
        "result_state": 1,
        "front": 2,
        "back": 3,
        "action_peak": 4,
        "middle": 5,
    }

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config
        workspace_dir = config.local_tools.workspace_dir or "output/commerce_video_diagnosis_runtime"
        self.vlm_byok_runtime = self._build_byok_runtime(workspace_dir, "vlm", config.providers.vlm)
        self.asr_byok_runtime = self._build_byok_runtime(workspace_dir, "asr", config.providers.asr)
        self.ocr_byok_runtime = self._build_byok_runtime(workspace_dir, "ocr", config.providers.ocr)
        self.provider_resolution_trace = {
            "environment_mode": self._environment_mode(),
            "fallback_protocol_mode": self.config.runtime.provider_fallback_mode,
            "asr": {},
            "vlm": {},
            "ocr": {},
        }

    def _provider_fallback_mode(self) -> str:
        mode = str(self.config.runtime.provider_fallback_mode or "force_off").strip().lower()
        return mode or "force_off"

    def _environment_mode(self) -> str:
        if self._provider_fallback_mode() == "force_on":
            return "fallback_requested"
        return "external_public"

    def _allow_provider_fallback(self, script_path: Path | None = None) -> bool:
        _ = script_path
        return False

    def _record_provider_resolution(
        self,
        capability: str,
        *,
        selected_provider_mode: str,
        provider_name: str,
        fallback_used: bool,
        fallback_reason: str = "",
    ) -> None:
        self.provider_resolution_trace[capability] = {
            "selected_provider_mode": selected_provider_mode,
            "provider_name": provider_name,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }

    def _warn_provider_fallback(self, capability: str, provider_name: str) -> None:
        print(
            f"WARNING: {capability.upper()} provider 未配置。公开仓库仅保留 fallback 协议位，当前不会自动执行 {provider_name}。请显式配置 BYOK provider。",
            file=sys.stderr,
        )

    def _raise_provider_fallback_error(self, capability: str, provider_name: str) -> None:
        mode = self._provider_fallback_mode()
        if mode == "force_on":
            raise ProviderExecutionViolation(
                f"{capability.upper()} fallback 协议位已被显式请求，但公开仓库未内置 {provider_name} 实现。请改为配置 BYOK provider。"
            )
        provider_field = f"providers.{capability}"
        if capability == "ocr_feedback":
            raise ProviderExecutionViolation(
                "OCR feedback 已启用，但当前未配置可用 OCR provider。公开仓库仅保留 fallback 协议位；请配置 providers.ocr，或关闭 enable_real_ocr_feedback。"
            )
        capability_name = capability.upper()
        guidance = {
            "ASR": "请在 providers.asr 中配置 BYOK provider（如 Aliyun DashScope / Volcengine / OpenAI-compatible ASR）。",
            "VLM": "请在 providers.vlm 中配置可用的视觉 provider。",
            "OCR": "请在 providers.ocr 中配置可用的 OCR / Vision provider。",
        }.get(capability_name, f"请在 {provider_field} 中配置可用 provider。")
        raise ProviderExecutionViolation(
            f"{capability_name} provider 未配置，且公开仓库不再内置 fallback 执行路径。{guidance}"
        )

    def _build_byok_runtime(
        self,
        workspace_dir: str,
        logical_name: str,
        provider: ProviderConfig,
    ) -> ProviderRuntimeGovernance:
        provider_name = provider.provider or f"{logical_name}_byok"
        max_retries = (
            provider.retry_policy.max_retries
            if provider.retry_policy.max_retries > 0
            else self.config.runtime.provider_runtime_max_retries
        )
        backoff_sec = (
            provider.retry_policy.backoff_sec
            if provider.retry_policy.backoff_sec > 0
            else self.config.runtime.provider_runtime_backoff_sec
        )
        return ProviderRuntimeGovernance(
            workspace_dir=workspace_dir,
            provider_name=f"{logical_name}_{provider_name}",
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            max_requests_per_run=self.config.runtime.provider_runtime_max_requests_per_run,
        )

    @staticmethod
    def _has_external_provider(provider: ProviderConfig) -> bool:
        return bool(provider.enabled and provider.endpoint and provider.api_key)

    def _run_fixture_provider(self, name: str, provider: ProviderConfig) -> Any:
        path = Path(provider.path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProviderExecutionViolation(f"读取 provider {name} fixture 失败: {exc}") from exc

    def _split_asr_by_segments(self, text: str, segments: list[dict]) -> list[str]:
        cleaned = re.sub(r"\s+", "", text or "")
        if not cleaned:
            return [""] * len(segments)
        parts = [chunk for chunk in re.split(r"(?<=[。！？!?])", cleaned) if chunk]
        if not parts:
            parts = [cleaned]
        buckets = [""] * len(segments)
        for idx, part in enumerate(parts):
            buckets[min(idx * len(segments) // len(parts), len(segments) - 1)] += part
        return [bucket or cleaned for bucket in buckets]

    @staticmethod
    def _resolve_asr_request_mode(provider: ProviderConfig) -> str:
        adapter = str(provider.adapter or "").strip().lower()
        provider_name = str(provider.provider or "").strip().lower()
        if adapter == "aliyun_asr" or provider_name == "aliyun_asr":
            return "aliyun_asr"
        if adapter == "volcengine_asr" or provider_name == "volcengine_asr":
            return "volcengine_asr"
        return "openai_audio_transcription"

    def _align_asr_segments(self, data: dict, segments: list[dict]) -> list[str]:
        audio_segments = data.get("segments")
        if not isinstance(audio_segments, list) or not audio_segments:
            return self._split_asr_by_segments(str(data.get("asr_text") or data.get("text") or ""), segments)
        buckets = [""] * len(segments)
        for item in audio_segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start_sec = float(item.get("start_sec", item.get("start", 0.0)) or 0.0)
            end_sec = float(item.get("end_sec", item.get("end", start_sec)) or start_sec)
            best_idx = 0
            best_overlap = -1.0
            for idx, seg in enumerate(segments):
                overlap = min(end_sec, float(seg["end_sec"])) - max(start_sec, float(seg["start_sec"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            buckets[best_idx] += text
        fallback = self._split_asr_by_segments(str(data.get("asr_text") or data.get("text") or ""), segments)
        return [bucket or fallback[idx] for idx, bucket in enumerate(buckets)]

    def _run_public_fallback_audio_stub(self, preproc: dict) -> list[dict]:
        _ = preproc
        raise ProviderExecutionViolation("Public ASR fallback stub is not executable in the open-source repository.")

    def _normalize_asr_rows(self, data: dict, segments: list[dict]) -> list[dict]:
        text_chunks = self._align_asr_segments(data, segments)
        rows = []
        for seg, text in zip(segments, text_chunks):
            rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "segment_type": seg.get("segment_type", "main"),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "audio_facts": {"asr_text": text, "sfx_events": [], "bgm_events": []},
                }
            )
        return rows

    def _run_external_asr(self, preproc: dict, provider: ProviderConfig) -> list[dict]:
        payload = {
            "request_mode": self._resolve_asr_request_mode(provider),
            "endpoint": provider.endpoint,
            "api_key": provider.api_key,
            "model": provider.model,
            "timeout_sec": provider.timeout_sec,
            "path": preproc["audio_path"],
            "extra": provider.extra,
        }
        data = self.asr_byok_runtime.execute_json_command(
            operation_key=f"asr:external:{provider.provider}:{Path(preproc['audio_path']).name}",
            script_path=BYOK_PROVIDER,
            payload=payload,
            failure_label=f"external_asr[{provider.provider}]",
        )
        return self._normalize_asr_rows(data, preproc["segments"])

    def _run_asr_provider(self, preproc: dict) -> list[dict] | None:
        provider = self.config.providers.asr
        if not provider.enabled:
            self._record_provider_resolution(
                "asr",
                selected_provider_mode="disabled",
                provider_name="disabled",
                fallback_used=False,
            )
            return None
        if provider.provider == "fixture_file":
            self._record_provider_resolution(
                "asr",
                selected_provider_mode="fixture",
                provider_name="fixture_file",
                fallback_used=False,
            )
            return self._run_fixture_provider("asr", provider)
        if self._has_external_provider(provider):
            self._record_provider_resolution(
                "asr",
                selected_provider_mode="byok",
                provider_name=provider.provider or "external_asr",
                fallback_used=False,
            )
            return self._run_external_asr(preproc, provider)
        if not self._allow_provider_fallback():
            self._record_provider_resolution(
                "asr",
                selected_provider_mode="error",
                provider_name="unconfigured",
                fallback_used=False,
                fallback_reason="provider_not_configured",
            )
            self._raise_provider_fallback_error("asr", "public_fallback_stub")
        self._warn_provider_fallback("asr", "public_fallback_stub")
        self._record_provider_resolution(
            "asr",
            selected_provider_mode="fallback_stub",
            provider_name="public_fallback_stub",
            fallback_used=True,
            fallback_reason="provider_not_configured",
        )
        raise ProviderExecutionViolation("ASR fallback stub reached unexpectedly.")

    def _build_public_fallback_image_task(self) -> str:
        return '请分析这张视频帧，返回严格 JSON：{"visual_subject": string, "shot_size": string, "camera_movement": string, "lighting_tone": string, "key_objects": string[], "actions": [{"action_name": string, "physical_intensity": string}], "ocr_facts": [{"text": string, "color": string, "font_family": string, "font_weight": string, "font_size_level": string, "stroke_style": string, "text_effect_style": string}] }。其中 OCR 字体样式字段必须基于画面真实识别并逐条返回，不允许省略；若画面存在文字但个别样式难以完全确认，也必须给出最接近的视觉判断。若无法判断 camera_movement，可返回 static；如果没有 OCR 则返回空数组。不要输出 markdown。'

    def _build_external_vlm_task(self) -> str:
        return (
            "你是视频 FactPack 上游抽取器的视觉事实节点。"
            "请只输出严格 JSON 对象，不要 markdown，不要解释。"
            "返回 schema: "
            '{"visual_subject": string, "shot_size": string, "camera_movement": string, '
            '"lighting_tone": string, "key_objects": string[], '
            '"actions": [{"action_name": string, "physical_intensity": string}]}. '
            "只允许描述可见物理事实，禁止输出 HEC、策略、营销结论或猜测。"
            "若 camera_movement 无法确认，返回 static；若没有动作，actions 返回空数组。"
        )

    def _build_external_ocr_task(self) -> str:
        return (
            "你是视频 FactPack 上游抽取器的 OCR 节点。"
            "请只输出严格 JSON 对象，不要 markdown，不要解释。"
            "返回 schema: "
            '{"ocr_facts": ['
            '{"text": string, "position": {"x": number, "y": number, "w": number, "h": number}, '
            '"color": string, "font_family": string, "font_weight": string, '
            '"font_size_level": string, "stroke_style": string, "text_effect_style": string}]}. '
            "position 使用 0-1 归一化坐标；若无文字，返回空数组。"
            "样式字段必须逐条返回；无法完全确认时给出最接近的视觉判断。"
        )

    def _run_external_image_call(
        self,
        *,
        runtime: ProviderRuntimeGovernance,
        provider: ProviderConfig,
        frame_path: str,
        segment_id: str,
        role: str,
        task: str,
        failure_label: str,
    ) -> dict:
        payload = {
            "request_mode": "openai_chat_vision_json",
            "endpoint": provider.endpoint,
            "api_key": provider.api_key,
            "model": provider.model,
            "timeout_sec": provider.timeout_sec,
            "paths": [frame_path],
            "task": task,
        }
        return runtime.execute_json_command(
            operation_key=f"{failure_label}:{segment_id}:{role}:{Path(frame_path).name}",
            script_path=BYOK_PROVIDER,
            payload=payload,
            failure_label=f"{failure_label}[{provider.provider}]",
        )

    def _run_public_fallback_frame_stub(self, frame_path: str, segment_id: str, role: str) -> dict:
        _ = frame_path
        _ = segment_id
        _ = role
        raise ProviderExecutionViolation("Public image fallback stub is not executable in the open-source repository.")

    def _analyze_single_frame_vlm(self, frame_path: str, segment_id: str, role: str) -> dict:
        provider = self.config.providers.vlm
        if self._has_external_provider(provider):
            self._record_provider_resolution(
                "vlm",
                selected_provider_mode="byok",
                provider_name=provider.provider or "external_vlm",
                fallback_used=False,
            )
            return self._run_external_image_call(
                runtime=self.vlm_byok_runtime,
                provider=provider,
                frame_path=frame_path,
                segment_id=segment_id,
                role=role,
                task=self._build_external_vlm_task(),
                failure_label="external_vlm",
            )
        if not self._allow_provider_fallback():
            self._record_provider_resolution(
                "vlm",
                selected_provider_mode="error",
                provider_name="unconfigured",
                fallback_used=False,
                fallback_reason="provider_not_configured",
            )
            self._raise_provider_fallback_error("vlm", "public_fallback_stub")
        self._warn_provider_fallback("vlm", "public_fallback_stub")
        self._record_provider_resolution(
            "vlm",
            selected_provider_mode="fallback_stub",
            provider_name="public_fallback_stub",
            fallback_used=True,
            fallback_reason="provider_not_configured",
        )
        raise ProviderExecutionViolation("VLM fallback stub reached unexpectedly.")

    def _analyze_single_frame_ocr(self, frame_path: str, segment_id: str, role: str) -> dict:
        provider = self.config.providers.ocr
        if self._has_external_provider(provider):
            self._record_provider_resolution(
                "ocr",
                selected_provider_mode="byok",
                provider_name=provider.provider or "external_ocr",
                fallback_used=False,
            )
            return self._run_external_image_call(
                runtime=self.ocr_byok_runtime,
                provider=provider,
                frame_path=frame_path,
                segment_id=segment_id,
                role=role,
                task=self._build_external_ocr_task(),
                failure_label="external_ocr",
            )
        if not self._allow_provider_fallback():
            self._record_provider_resolution(
                "ocr",
                selected_provider_mode="error",
                provider_name="unconfigured",
                fallback_used=False,
                fallback_reason="provider_not_configured",
            )
            self._raise_provider_fallback_error("ocr", "public_fallback_stub")
        self._warn_provider_fallback("ocr", "public_fallback_stub")
        self._record_provider_resolution(
            "ocr",
            selected_provider_mode="fallback_stub",
            provider_name="public_fallback_stub",
            fallback_used=True,
            fallback_reason="provider_not_configured",
        )
        raise ProviderExecutionViolation("OCR fallback stub reached unexpectedly.")

    @staticmethod
    def _extract_ocr_rows(data: dict) -> list[dict]:
        rows: list[dict] = []
        for item in data.get("ocr_facts") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            position = item.get("position") if isinstance(item.get("position"), dict) else {}
            rows.append(
                {
                    "text": text,
                    "position": {
                        "x": float(position.get("x", 0.1) or 0.1),
                        "y": float(position.get("y", 0.8) or 0.8),
                        "w": float(position.get("w", 0.3) or 0.3),
                        "h": float(position.get("h", 0.05) or 0.05),
                    },
                    "color": str(item.get("color") or "").strip(),
                    "font_family": str(item.get("font_family") or "").strip(),
                    "font_weight": str(item.get("font_weight") or "").strip(),
                    "font_size_level": str(item.get("font_size_level") or "").strip(),
                    "stroke_style": str(item.get("stroke_style") or "").strip(),
                    "text_effect_style": str(item.get("text_effect_style") or "").strip(),
                }
            )
        return rows

    @staticmethod
    def _dedupe_ocr_rows(rows: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in rows:
            text = str(item.get("text") or "").strip()
            normalized = re.sub(r"\s+", "", text).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
        return deduped

    def _should_backfill_segment_ocr(self, segment: dict, primary_ocr_rows: list[dict]) -> bool:
        if primary_ocr_rows:
            return False
        frames = segment.get("frames") or []
        if len(frames) <= 1:
            return False
        frame_plan = segment.get("frame_plan") or {}
        triggers = set(frame_plan.get("upsampling_triggers") or [])
        metrics = frame_plan.get("metrics") or {}
        return (
            "TEXT_CHANGE_DENSE" in triggers
            or float(metrics.get("avg_text_density", 0.0)) >= 0.08
            or int(metrics.get("max_text_boxes", 0) or 0) >= self.OCR_BACKFILL_TRIGGER_BOXES
        )

    def _select_ocr_backfill_frames(self, segment: dict) -> list[dict]:
        primary_frame_path = str(segment.get("frame_path") or "")
        ranked_frames = sorted(
            [
                frame
                for frame in (segment.get("frames") or [])
                if str(frame.get("frame_path") or "") and str(frame.get("frame_path") or "") != primary_frame_path
            ],
            key=lambda frame: (
                self.OCR_BACKFILL_ROLE_PRIORITY.get(str(frame.get("sampling_role") or ""), 99),
                -float(frame.get("text_density", 0.0) or 0.0),
                -float(frame.get("representative_score", 0.0) or 0.0),
                float(frame.get("frame_second", 0.0) or 0.0),
            ),
        )
        return ranked_frames[: self.OCR_BACKFILL_FRAME_LIMIT]

    def _collect_segment_ocr_rows(
        self,
        segment: dict,
        primary_data: dict,
        analyze_frame: Callable[[str, str, str], dict],
    ) -> list[dict]:
        ocr_rows = self._extract_ocr_rows(primary_data)
        if not self._should_backfill_segment_ocr(segment, ocr_rows):
            return ocr_rows

        merged_rows = list(ocr_rows)
        for frame in self._select_ocr_backfill_frames(segment):
            frame_path = str(frame.get("frame_path") or "").strip()
            if not frame_path:
                continue
            frame_role = str(frame.get("sampling_role") or "backfill")
            supplemental = analyze_frame(frame_path, str(segment.get("segment_id") or ""), frame_role)
            supplemental_rows = self._extract_ocr_rows(supplemental)
            if not supplemental_rows:
                continue
            merged_rows.extend(supplemental_rows)
            break
        return self._dedupe_ocr_rows(merged_rows)

    def _run_public_fallback_image_stub(self, preproc: dict) -> tuple[list[dict], list[dict]]:
        visual_rows: list[dict] = []
        ocr_rows: list[dict] = []
        for seg in preproc["segments"]:
            primary_frame = next(
                (frame for frame in seg.get("frames") or [] if frame.get("frame_path") == seg.get("frame_path")),
                None,
            )
            if primary_frame is None:
                primary_frame = {"frame_path": seg["frame_path"], "sampling_role": "middle"}
            data = self._run_public_fallback_frame_stub(
                str(primary_frame.get("frame_path") or seg["frame_path"]),
                str(seg["segment_id"]),
                str(primary_frame.get("sampling_role") or "middle"),
            )
            segment_ocr_rows = self._collect_segment_ocr_rows(seg, data, self._run_public_fallback_frame_stub)
            actions = []
            for item in data.get("actions") or []:
                if not isinstance(item, dict):
                    continue
                actions.append(
                    {
                        "action_name": str(item.get("action_name") or "unknown_action"),
                        "physical_intensity": str(item.get("physical_intensity") or "low").strip().lower().replace(" ", "_"),
                    }
                )
            visual_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "segment_type": seg.get("segment_type", "main"),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "visual_facts": {
                        "shot_size": str(data.get("shot_size") or "medium_close_up").strip().lower().replace("-", "_").replace(" ", "_"),
                        "camera_movement": str(data.get("camera_movement") or "static").strip().lower().replace("-", "_").replace(" ", "_"),
                        "visual_subject": str(data.get("visual_subject") or "unknown subject"),
                        "lighting_tone": str(data.get("lighting_tone") or "neutral").strip().lower().replace(",", "_").replace("-", "_").replace(" ", "_"),
                        "key_objects": [str(x) for x in data.get("key_objects") or [] if str(x).strip()],
                        "actions": actions,
                    },
                    "rhythm_facts": {"transition_type": "hard_cut", "pace_marker": "normal"},
                }
            )
            ocr_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "segment_type": seg.get("segment_type", "main"),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "ocr_facts": segment_ocr_rows,
                }
            )
        return visual_rows, ocr_rows

    def _run_vlm_provider(self, preproc: dict) -> list[dict] | None:
        provider = self.config.providers.vlm
        if not provider.enabled:
            self._record_provider_resolution(
                "vlm",
                selected_provider_mode="disabled",
                provider_name="disabled",
                fallback_used=False,
            )
            return None
        if provider.provider == "fixture_file":
            self._record_provider_resolution(
                "vlm",
                selected_provider_mode="fixture",
                provider_name="fixture_file",
                fallback_used=False,
            )
            return self._run_fixture_provider("vlm", provider)
        if not self._has_external_provider(provider):
            self._record_provider_resolution(
                "vlm",
                selected_provider_mode="error",
                provider_name="unconfigured",
                fallback_used=False,
                fallback_reason="provider_not_configured",
            )
            self._raise_provider_fallback_error("vlm", "public_fallback_stub")

        visual_rows: list[dict] = []
        for seg in preproc["segments"]:
            primary_frame = next(
                (frame for frame in seg.get("frames") or [] if frame.get("frame_path") == seg.get("frame_path")),
                None,
            )
            if primary_frame is None:
                primary_frame = {"frame_path": seg["frame_path"], "sampling_role": "middle"}
            data = self._analyze_single_frame_vlm(
                str(primary_frame.get("frame_path") or seg["frame_path"]),
                str(seg["segment_id"]),
                str(primary_frame.get("sampling_role") or "middle"),
            )
            actions = []
            for item in data.get("actions") or []:
                if not isinstance(item, dict):
                    continue
                actions.append(
                    {
                        "action_name": str(item.get("action_name") or "unknown_action"),
                        "physical_intensity": str(item.get("physical_intensity") or "low").strip().lower().replace(" ", "_"),
                    }
                )
            visual_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "segment_type": seg.get("segment_type", "main"),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "visual_facts": {
                        "shot_size": str(data.get("shot_size") or "medium_close_up").strip().lower().replace("-", "_").replace(" ", "_"),
                        "camera_movement": str(data.get("camera_movement") or "static").strip().lower().replace("-", "_").replace(" ", "_"),
                        "visual_subject": str(data.get("visual_subject") or "unknown subject"),
                        "lighting_tone": str(data.get("lighting_tone") or "neutral").strip().lower().replace(",", "_").replace("-", "_").replace(" ", "_"),
                        "key_objects": [str(x) for x in data.get("key_objects") or [] if str(x).strip()],
                        "actions": actions,
                    },
                    "rhythm_facts": {"transition_type": "hard_cut", "pace_marker": "normal"},
                }
            )
        return visual_rows

    def _run_ocr_provider(self, preproc: dict) -> list[dict] | None:
        provider = self.config.providers.ocr
        if not provider.enabled:
            self._record_provider_resolution(
                "ocr",
                selected_provider_mode="disabled",
                provider_name="disabled",
                fallback_used=False,
            )
            return None
        if provider.provider == "fixture_file":
            self._record_provider_resolution(
                "ocr",
                selected_provider_mode="fixture",
                provider_name="fixture_file",
                fallback_used=False,
            )
            return self._run_fixture_provider("ocr", provider)
        if not self._has_external_provider(provider):
            self._record_provider_resolution(
                "ocr",
                selected_provider_mode="error",
                provider_name="unconfigured",
                fallback_used=False,
                fallback_reason="provider_not_configured",
            )
            self._raise_provider_fallback_error("ocr", "public_fallback_stub")

        ocr_rows: list[dict] = []
        for seg in preproc["segments"]:
            primary_frame = next(
                (frame for frame in seg.get("frames") or [] if frame.get("frame_path") == seg.get("frame_path")),
                None,
            )
            if primary_frame is None:
                primary_frame = {"frame_path": seg["frame_path"], "sampling_role": "middle"}
            data = self._analyze_single_frame_ocr(
                str(primary_frame.get("frame_path") or seg["frame_path"]),
                str(seg["segment_id"]),
                str(primary_frame.get("sampling_role") or "middle"),
            )
            segment_ocr_rows = self._collect_segment_ocr_rows(seg, data, self._analyze_single_frame_ocr)
            ocr_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "segment_type": seg.get("segment_type", "main"),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "ocr_facts": segment_ocr_rows,
                }
            )
        return ocr_rows

    def _runtime_states(self) -> dict[str, Any]:
        states = {}
        if self.vlm_byok_runtime.state.requests_dispatched or self.vlm_byok_runtime.state.cache_hits or self.vlm_byok_runtime.state.checkpoint_hits:
            states["vlm_byok"] = self.vlm_byok_runtime.state.to_dict()
        if self.asr_byok_runtime.state.requests_dispatched or self.asr_byok_runtime.state.cache_hits or self.asr_byok_runtime.state.checkpoint_hits:
            states["asr_byok"] = self.asr_byok_runtime.state.to_dict()
        if self.ocr_byok_runtime.state.requests_dispatched or self.ocr_byok_runtime.state.cache_hits or self.ocr_byok_runtime.state.checkpoint_hits:
            states["ocr_byok"] = self.ocr_byok_runtime.state.to_dict()
        return states

    def run(self, preproc: dict | None = None) -> dict[str, Any]:
        need_preproc = any(
            provider.enabled and provider.provider != "fixture_file"
            for provider in [self.config.providers.vlm, self.config.providers.asr, self.config.providers.ocr]
        )
        if need_preproc and preproc is None:
            raise ProviderExecutionViolation("外部 provider/BYOK 模式需要 preprocess 输出")

        if preproc is None:
            bundle = {
                "vlm_raw": self._run_fixture_provider("vlm", self.config.providers.vlm) if self.config.providers.vlm.enabled else None,
                "asr_raw": self._run_fixture_provider("asr", self.config.providers.asr) if self.config.providers.asr.enabled else None,
                "ocr_raw": self._run_fixture_provider("ocr", self.config.providers.ocr) if self.config.providers.ocr.enabled else None,
            }
            bundle["provider_resolution_trace"] = self.provider_resolution_trace
            return bundle

        bundle = {
            "vlm_raw": self._run_vlm_provider(preproc),
            "asr_raw": self._run_asr_provider(preproc),
            "ocr_raw": self._run_ocr_provider(preproc),
        }
        runtime_states = self._runtime_states()
        if runtime_states:
            bundle["runtime_governance"] = runtime_states
        bundle["provider_resolution_trace"] = self.provider_resolution_trace
        return bundle
