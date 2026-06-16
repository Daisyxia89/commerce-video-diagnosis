from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.errors import ProviderExecutionViolation
from extractor.utils.json_utils import extract_first_json


JSON_HEADERS = {
    "Content-Type": "application/json",
}


def _normalize_endpoint(endpoint: str, suffix: str) -> str:
    stripped = (endpoint or "").rstrip("/")
    if stripped.endswith(suffix):
        return stripped
    return f"{stripped}{suffix}"


def _auth_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if extra:
        headers.update(extra)
    return headers


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: int,
    payload: dict[str, Any] | None = None,
    method: str = "POST",
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderExecutionViolation(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderExecutionViolation(f"网络请求失败: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionViolation(f"返回非 JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise ProviderExecutionViolation("返回 payload 不是 JSON 对象")
    return data


def _http_json(url: str, *, api_key: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    return _request_json(
        url,
        headers=_auth_headers(api_key, JSON_HEADERS),
        payload=payload,
        timeout_sec=timeout_sec,
        method="POST",
    )


def _build_multipart_form(fields: dict[str, str], file_field: str, file_path: str) -> tuple[bytes, str]:
    boundary = "----AimeBYOKBoundary7MA4YWxkTrZu0gW"
    path = Path(file_path)
    filename = path.name
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def _image_to_data_url(path: str) -> str:
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_openai_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderExecutionViolation("chat completions 返回缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ProviderExecutionViolation("chat completions 返回缺少 message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text") or ""))
        joined = "\n".join(texts).strip()
        if joined:
            return joined
    raise ProviderExecutionViolation("chat completions 返回缺少可解析 content")


def _normalize_asr_output(text: str, language: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_segments: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        seg_text = str(item.get("text") or "").strip()
        if not seg_text:
            continue
        normalized_segments.append(
            {
                "start_sec": float(item.get("start_sec", item.get("start", 0.0)) or 0.0),
                "end_sec": float(item.get("end_sec", item.get("end", item.get("start_sec", item.get("start", 0.0))) or 0.0)),
                "text": seg_text,
            }
        )
    normalized_text = str(text or "").strip()
    if not normalized_text and normalized_segments:
        normalized_text = "".join(item["text"] for item in normalized_segments)
    return {
        "text": normalized_text,
        "asr_text": normalized_text,
        "language": str(language or ""),
        "segments": normalized_segments,
    }


def _extra_payload(payload: dict[str, Any]) -> dict[str, Any]:
    extra = payload.get("extra")
    return dict(extra) if isinstance(extra, dict) else {}


def _http_date_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


def _iso8601_basic_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_binary_response(request: urllib.request.Request, *, timeout_sec: int) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderExecutionViolation(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderExecutionViolation(f"网络请求失败: {exc}") from exc


def _guess_upload_content_type(path: str) -> str:
    return mimetypes.guess_type(Path(path).name)[0] or "application/octet-stream"


def _normalize_upload_object_key(path: str, extra: dict[str, Any]) -> str:
    prefix = str(extra.get("upload_object_prefix") or "commerce-video-diagnosis/audio").strip().strip("/")
    suffix = Path(path).suffix or ".bin"
    filename = f"{uuid.uuid4().hex}{suffix}"
    return f"{prefix}/{filename}" if prefix else filename


def _build_public_object_url(*, object_key: str, bucket: str, endpoint: str, extra: dict[str, Any]) -> str:
    public_base_url = str(extra.get("upload_public_base_url") or "").strip().rstrip("/")
    encoded_key = urllib.parse.quote(object_key, safe="/-_.~")
    if public_base_url:
        return f"{public_base_url}/{encoded_key}"
    normalized_endpoint = endpoint.strip().rstrip("/")
    if not normalized_endpoint.startswith(("http://", "https://")):
        normalized_endpoint = f"https://{normalized_endpoint}"
    parsed = urllib.parse.urlparse(normalized_endpoint)
    host = parsed.netloc or parsed.path
    scheme = parsed.scheme or "https"
    if host.startswith(f"{bucket}."):
        base_host = host
    else:
        base_host = f"{bucket}.{host}"
    return f"{scheme}://{base_host}/{encoded_key}"


def _upload_via_oss_signature_v1(path: str, extra: dict[str, Any], *, timeout_sec: int) -> str:
    access_key_id = str(extra.get("upload_access_key_id") or "").strip()
    access_key_secret = str(extra.get("upload_access_key_secret") or "").strip()
    endpoint = str(extra.get("upload_endpoint") or "").strip()
    bucket = str(extra.get("upload_bucket") or "").strip()
    if not all([access_key_id, access_key_secret, endpoint, bucket]):
        raise ProviderExecutionViolation(
            "OSS 自动上传缺少必要配置：upload_access_key_id / upload_access_key_secret / upload_endpoint / upload_bucket"
        )
    object_key = _normalize_upload_object_key(path, extra)
    content_type = _guess_upload_content_type(path)
    date_value = _http_date_now()
    canonical_resource = f"/{bucket}/{object_key}"
    string_to_sign = f"PUT\n\n{content_type}\n{date_value}\n{canonical_resource}"
    signature = base64.b64encode(
        hmac.new(access_key_secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    normalized_endpoint = endpoint.rstrip("/")
    if not normalized_endpoint.startswith(("http://", "https://")):
        normalized_endpoint = f"https://{normalized_endpoint}"
    parsed = urllib.parse.urlparse(normalized_endpoint)
    host = parsed.netloc or parsed.path
    scheme = parsed.scheme or "https"
    upload_url = f"{scheme}://{bucket}.{host}/{urllib.parse.quote(object_key, safe='/-_.~')}"
    request = urllib.request.Request(
        upload_url,
        data=Path(path).read_bytes(),
        headers={
            "Content-Type": content_type,
            "Date": date_value,
            "Authorization": f"OSS {access_key_id}:{signature}",
        },
        method="PUT",
    )
    _read_binary_response(request, timeout_sec=timeout_sec)
    return _build_public_object_url(object_key=object_key, bucket=bucket, endpoint=endpoint, extra=extra)


def _sign_tos_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = hmac.new(("TOS4" + secret).encode("utf-8"), date_stamp.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"request", hashlib.sha256).digest()


def _upload_via_tos_sigv4(path: str, extra: dict[str, Any], *, timeout_sec: int) -> str:
    access_key_id = str(extra.get("upload_access_key_id") or "").strip()
    access_key_secret = str(extra.get("upload_access_key_secret") or "").strip()
    endpoint = str(extra.get("upload_endpoint") or "").strip()
    bucket = str(extra.get("upload_bucket") or "").strip()
    region = str(extra.get("upload_region") or "").strip()
    if not all([access_key_id, access_key_secret, endpoint, bucket, region]):
        raise ProviderExecutionViolation(
            "TOS 自动上传缺少必要配置：upload_access_key_id / upload_access_key_secret / upload_endpoint / upload_bucket / upload_region"
        )
    object_key = _normalize_upload_object_key(path, extra)
    content_type = _guess_upload_content_type(path)
    body = Path(path).read_bytes()
    payload_hash = hashlib.sha256(body).hexdigest()
    amz_date = _iso8601_basic_now()
    date_stamp = amz_date[:8]
    normalized_endpoint = endpoint.rstrip("/")
    if not normalized_endpoint.startswith(("http://", "https://")):
        normalized_endpoint = f"https://{normalized_endpoint}"
    parsed = urllib.parse.urlparse(normalized_endpoint)
    host = parsed.netloc or parsed.path
    scheme = parsed.scheme or "https"
    canonical_uri = "/" + urllib.parse.quote(object_key, safe="/-_.~")
    query_string = ""
    canonical_headers = (
        f"host:{bucket}.{host}\n"
        f"x-tos-content-sha256:{payload_hash}\n"
        f"x-tos-date:{amz_date}\n"
    )
    signed_headers = "host;x-tos-content-sha256;x-tos-date"
    canonical_request = "\n".join(
        [
            "PUT",
            canonical_uri,
            query_string,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    credential_scope = f"{date_stamp}/{region}/tos/request"
    string_to_sign = "\n".join(
        [
            "TOS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _sign_tos_key(access_key_secret, date_stamp, region, "tos")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"TOS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    upload_url = f"{scheme}://{bucket}.{host}{canonical_uri}"
    request = urllib.request.Request(
        upload_url,
        data=body,
        headers={
            "Content-Type": content_type,
            "X-Tos-Content-Sha256": payload_hash,
            "X-Tos-Date": amz_date,
            "Authorization": authorization,
        },
        method="PUT",
    )
    _read_binary_response(request, timeout_sec=timeout_sec)
    return _build_public_object_url(object_key=object_key, bucket=bucket, endpoint=endpoint, extra=extra)


def _upload_local_audio(path: str, extra: dict[str, Any], *, timeout_sec: int) -> str:
    upload_provider = str(extra.get("upload_provider") or "").strip().lower()
    if upload_provider == "oss":
        return _upload_via_oss_signature_v1(path, extra, timeout_sec=timeout_sec)
    if upload_provider == "tos":
        return _upload_via_tos_sigv4(path, extra, timeout_sec=timeout_sec)
    raise ProviderExecutionViolation(
        "该 ASR adapter 需要公网可访问的 audio_url；若只提供本地 path，请在 provider.extra 配置 upload_provider=oss/tos 及对应上传参数。"
    )


def _resolve_remote_audio_url(payload: dict[str, Any]) -> str:
    extra = _extra_payload(payload)
    candidates = [
        payload.get("audio_url"),
        payload.get("remote_url"),
        extra.get("audio_url"),
        extra.get("remote_url"),
        payload.get("path"),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
    local_path = str(payload.get("path") or "").strip()
    if local_path and Path(local_path).exists():
        return _upload_local_audio(local_path, extra, timeout_sec=int(payload.get("timeout_sec") or 180))
    raise ProviderExecutionViolation(
        "该 ASR adapter 需要公网可访问的 audio_url；当前 payload 只有本地 path。请在 provider.extra.audio_url 中传入可访问 URL，或配置 OSS/TOS 自动上传。"
    )


def _poll_until_complete(
    poller: Any,
    *,
    timeout_sec: int,
    poll_interval_sec: float,
    pending_status: set[str],
    success_status: set[str],
    failure_hint: str,
) -> dict[str, Any]:
    deadline = time.time() + max(timeout_sec, 1)
    last_status = ""
    while time.time() <= deadline:
        data = poller()
        status = str(data.get("task_status") or data.get("status") or "").strip().upper()
        if not status and data.get("result") is not None:
            return data
        last_status = status
        if status in success_status:
            return data
        if status in pending_status or not status:
            time.sleep(max(poll_interval_sec, 0.2))
            continue
        raise ProviderExecutionViolation(f"{failure_hint}: status={status or 'UNKNOWN'} payload={json.dumps(data, ensure_ascii=False)[:500]}")
    raise ProviderExecutionViolation(f"{failure_hint}: 轮询超时，最后状态={last_status or 'UNKNOWN'}")


def _call_openai_audio_transcription(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = _normalize_endpoint(str(payload.get("endpoint") or ""), "/audio/transcriptions")
    body, boundary = _build_multipart_form(
        {
            "model": str(payload.get("model") or ""),
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
        "file",
        str(payload.get("path") or ""),
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=_auth_headers(
            str(payload.get("api_key") or ""),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ),
        method="POST",
    )
    timeout_sec = int(payload.get("timeout_sec") or 60)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderExecutionViolation(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderExecutionViolation(f"网络请求失败: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionViolation(f"ASR 返回非 JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise ProviderExecutionViolation("ASR 返回 payload 不是 JSON 对象")

    segments: list[dict[str, Any]] = []
    for item in data.get("segments") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start_sec": float(item.get("start", 0.0) or 0.0),
                "end_sec": float(item.get("end", item.get("start", 0.0)) or 0.0),
                "text": text,
            }
        )
    return _normalize_asr_output(str(data.get("text") or ""), str(data.get("language") or ""), segments)


def _call_openai_chat_vision_json(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = _normalize_endpoint(str(payload.get("endpoint") or ""), "/chat/completions")
    image_paths = [str(item or "").strip() for item in payload.get("paths") or [] if str(item or "").strip()]
    if not image_paths:
        raise ProviderExecutionViolation("VLM 调用缺少 image paths")
    content: list[dict[str, Any]] = [{"type": "text", "text": str(payload.get("task") or "").strip()}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path)}})
    request_payload = {
        "model": str(payload.get("model") or ""),
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }
    data = _http_json(
        endpoint,
        api_key=str(payload.get("api_key") or ""),
        payload=request_payload,
        timeout_sec=int(payload.get("timeout_sec") or 60),
    )
    message_content = _extract_openai_message_content(data)
    parsed = extract_first_json(message_content)
    if not isinstance(parsed, dict):
        raise ProviderExecutionViolation("VLM 返回内容不是 JSON 对象")
    return parsed


def _extract_aliyun_transcripts_from_payload(data: Any) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    if isinstance(data, dict):
        current = data.get("transcripts")
        if isinstance(current, list):
            transcripts.extend(item for item in current if isinstance(item, dict))
        nested_candidates = [
            data.get("output"),
            data.get("result"),
            data.get("results"),
        ]
        for candidate in nested_candidates:
            transcripts.extend(_extract_aliyun_transcripts_from_payload(candidate))
    elif isinstance(data, list):
        for item in data:
            transcripts.extend(_extract_aliyun_transcripts_from_payload(item))
    return transcripts



def _load_aliyun_transcripts_from_url(url: str, *, timeout_sec: int) -> list[dict[str, Any]]:
    transcription_url = str(url or "").strip()
    if not transcription_url:
        return []
    data = _request_json(
        transcription_url,
        headers={},
        payload=None,
        timeout_sec=timeout_sec,
        method="GET",
    )
    return _extract_aliyun_transcripts_from_payload(data)



def _call_aliyun_asr(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(payload.get("endpoint") or "").strip() or "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    timeout_sec = int(payload.get("timeout_sec") or 180)
    extra = _extra_payload(payload)
    audio_url = _resolve_remote_audio_url(payload)
    submit_payload: dict[str, Any] = {
        "model": str(payload.get("model") or "paraformer-v2"),
        "input": {"file_urls": [audio_url]},
        "parameters": {"timestamp_alignment_enabled": True},
    }
    if isinstance(extra.get("parameters"), dict):
        submit_payload["parameters"].update(extra.get("parameters") or {})
    submit_response = _request_json(
        endpoint,
        headers=_auth_headers(
            str(payload.get("api_key") or ""),
            {"Content-Type": "application/json", "X-DashScope-Async": "enable"},
        ),
        payload=submit_payload,
        timeout_sec=timeout_sec,
        method="POST",
    )
    output = submit_response.get("output") if isinstance(submit_response.get("output"), dict) else {}
    task_id = str(output.get("task_id") or submit_response.get("task_id") or "").strip()
    if not task_id:
        raise ProviderExecutionViolation(f"阿里云 ASR 提交任务失败，未返回 task_id: {json.dumps(submit_response, ensure_ascii=False)[:500]}")
    task_url = str(output.get("task_url") or extra.get("task_url") or "").strip()
    if not task_url:
        parsed = urllib.parse.urlparse(endpoint)
        task_url = f"{parsed.scheme}://{parsed.netloc}/api/v1/tasks/{task_id}"

    def _poller() -> dict[str, Any]:
        task_data = _request_json(
            task_url,
            headers=_auth_headers(str(payload.get("api_key") or "")),
            payload=None,
            timeout_sec=timeout_sec,
            method="GET",
        )
        task_output = task_data.get("output") if isinstance(task_data.get("output"), dict) else {}
        return {
            "task_status": str(task_output.get("task_status") or task_data.get("task_status") or task_data.get("status") or ""),
            "result": task_output.get("results") or task_output.get("result") or task_data.get("results") or task_data.get("result") or task_data,
            "raw": task_data,
        }

    final_data = _poll_until_complete(
        _poller,
        timeout_sec=timeout_sec,
        poll_interval_sec=float(extra.get("poll_interval_sec") or 2),
        pending_status={"PENDING", "RUNNING", "QUEUING", "PROCESSING"},
        success_status={"SUCCEEDED", "SUCCESS", "COMPLETED"},
        failure_hint="阿里云 ASR 任务失败",
    )
    result_items = final_data.get("result")
    if isinstance(result_items, dict):
        result_items = [result_items]
    if not isinstance(result_items, list) or not result_items:
        raw = final_data.get("raw") if isinstance(final_data.get("raw"), dict) else {}
        task_output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
        result_items = task_output.get("results") or []

    transcripts = _extract_aliyun_transcripts_from_payload(result_items)
    if not transcripts:
        raw = final_data.get("raw") if isinstance(final_data.get("raw"), dict) else {}
        transcripts = _extract_aliyun_transcripts_from_payload(raw)

    if not transcripts:
        transcription_urls: list[str] = []
        for item in result_items or []:
            if not isinstance(item, dict):
                continue
            direct_url = str(item.get("transcription_url") or "").strip()
            if direct_url:
                transcription_urls.append(direct_url)
            nested_output = item.get("output") if isinstance(item.get("output"), dict) else {}
            nested_url = str(nested_output.get("transcription_url") or "").strip()
            if nested_url:
                transcription_urls.append(nested_url)
            for nested in item.get("results") or []:
                if not isinstance(nested, dict):
                    continue
                nested_result_url = str(nested.get("transcription_url") or "").strip()
                if nested_result_url:
                    transcription_urls.append(nested_result_url)
        for transcription_url in transcription_urls:
            transcripts.extend(_load_aliyun_transcripts_from_url(transcription_url, timeout_sec=timeout_sec))
            if transcripts:
                break

    segments: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    for transcript in transcripts:
        transcript_text = str(transcript.get("text") or "").strip()
        if transcript_text:
            full_text_parts.append(transcript_text)
        for sentence in transcript.get("sentences") or []:
            if not isinstance(sentence, dict):
                continue
            sentence_text = str(sentence.get("text") or "").strip()
            if not sentence_text:
                continue
            segments.append(
                {
                    "start_sec": float(sentence.get("begin_time", 0) or 0) / 1000.0,
                    "end_sec": float(sentence.get("end_time", sentence.get("begin_time", 0)) or 0) / 1000.0,
                    "text": sentence_text,
                }
            )
    return _normalize_asr_output("".join(full_text_parts), str(extra.get("language") or ""), segments)


def _call_volcengine_asr(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(payload.get("endpoint") or "").strip() or "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
    timeout_sec = int(payload.get("timeout_sec") or 180)
    extra = _extra_payload(payload)
    audio_url = _resolve_remote_audio_url(payload)
    submit_headers = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": str(extra.get("resource_id") or "volc.seedasr.auc"),
    }
    app_key = str(extra.get("app_key") or "").strip()
    if app_key:
        submit_headers["X-Api-App-Key"] = app_key
        submit_headers["X-Api-Access-Key"] = str(payload.get("api_key") or "")
    else:
        submit_headers["X-Api-Key"] = str(payload.get("api_key") or "")
    submit_payload: dict[str, Any] = {
        "user": {"uid": str(extra.get("uid") or "commerce-video-diagnosis")},
        "audio": {"url": audio_url},
        "request": {"model_name": str(payload.get("model") or extra.get("model_name") or "bigmodel")},
    }
    if isinstance(extra.get("audio"), dict):
        submit_payload["audio"].update(extra.get("audio") or {})
    if isinstance(extra.get("request"), dict):
        submit_payload["request"].update(extra.get("request") or {})
    submit_response = _request_json(
        endpoint,
        headers=submit_headers,
        payload=submit_payload,
        timeout_sec=timeout_sec,
        method="POST",
    )
    task_id = str(
        submit_response.get("id")
        or submit_response.get("task_id")
        or (submit_response.get("result") or {}).get("id")
        or (submit_response.get("result") or {}).get("task_id")
        or ""
    ).strip()
    if not task_id:
        raise ProviderExecutionViolation(f"火山引擎 ASR 提交任务失败，未返回 task_id: {json.dumps(submit_response, ensure_ascii=False)[:500]}")
    query_endpoint = str(extra.get("query_endpoint") or "").strip()
    if not query_endpoint:
        if endpoint.endswith("/submit"):
            query_endpoint = endpoint[: -len("/submit")] + "/query"
        else:
            query_endpoint = endpoint.rstrip("/") + "/query"

    def _poller() -> dict[str, Any]:
        query_headers = dict(submit_headers)
        query_headers["X-Api-Request-Id"] = task_id
        query_headers["X-Api-Sequence"] = "-1"
        query_response = _request_json(
            query_endpoint,
            headers=query_headers,
            payload={},
            timeout_sec=timeout_sec,
            method="POST",
        )
        result = query_response.get("result") if isinstance(query_response.get("result"), dict) else {}
        status = str(
            result.get("task_status")
            or query_response.get("task_status")
            or query_response.get("status")
            or (query_response.get("message") if query_response.get("code") not in (None, 0, "0") else "")
            or ""
        )
        return {
            "task_status": status,
            "result": result or query_response,
            "raw": query_response,
        }

    final_data = _poll_until_complete(
        _poller,
        timeout_sec=timeout_sec,
        poll_interval_sec=float(extra.get("poll_interval_sec") or 2),
        pending_status={"PENDING", "RUNNING", "QUEUEING", "PROCESSING"},
        success_status={"SUCCESS", "SUCCEEDED", "DONE", "COMPLETED", "FINISHED"},
        failure_hint="火山引擎 ASR 任务失败",
    )
    result = final_data.get("result") if isinstance(final_data.get("result"), dict) else {}
    utterances = result.get("utterances") if isinstance(result.get("utterances"), list) else []
    segments: list[dict[str, Any]] = []
    for item in utterances:
        if not isinstance(item, dict):
            continue
        seg_text = str(item.get("text") or "").strip()
        if not seg_text:
            continue
        segments.append(
            {
                "start_sec": float(item.get("start_time", 0) or 0) / 1000.0,
                "end_sec": float(item.get("end_time", item.get("start_time", 0)) or 0) / 1000.0,
                "text": seg_text,
            }
        )
    text = str(result.get("text") or "").strip()
    language = str(result.get("language") or extra.get("language") or "")
    return _normalize_asr_output(text, language, segments)


def main() -> None:
    if len(sys.argv) < 2:
        raise ProviderExecutionViolation("缺少 JSON payload 参数")
    payload = json.loads(sys.argv[1])
    if not isinstance(payload, dict):
        raise ProviderExecutionViolation("payload 必须为 JSON 对象")
    mode = str(payload.get("request_mode") or "").strip()
    if mode == "openai_audio_transcription":
        result = _call_openai_audio_transcription(payload)
    elif mode == "openai_chat_vision_json":
        result = _call_openai_chat_vision_json(payload)
    elif mode == "aliyun_asr":
        result = _call_aliyun_asr(payload)
    elif mode == "volcengine_asr":
        result = _call_volcengine_asr(payload)
    else:
        raise ProviderExecutionViolation(f"不支持的 request_mode: {mode}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
