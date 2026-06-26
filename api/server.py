from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.adapter import AdapterError, build_repo_payload, normalize_result
from commerce_video_diagnosis.understanding.core import ProtocolViolation, handle_request


class DiagnoseRequest(BaseModel):
    product_factpack: dict[str, Any]
    video_factpack: dict[str, Any]
    options: dict[str, Any] = {}
    request_id: str | None = None


app = FastAPI(title="commerce-video-diagnosis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/diagnose")
def diagnose(request: DiagnoseRequest) -> dict[str, Any]:
    frontend_payload = request.dict()
    try:
        repo_payload = build_repo_payload(frontend_payload)
        raw = handle_request(repo_payload).dict()
        normalized = normalize_result(raw)
        normalized["repo_payload"] = repo_payload
        return normalized
    except AdapterError as exc:
        return {"status": exc.code.lower(), "error": {"code": exc.code, "message": exc.message, "details": exc.details}}
    except ProtocolViolation as exc:
        return {"status": "schema_error", "error": {"code": "PROTOCOL_VIOLATION", "message": str(exc)}}
    except RuntimeError as exc:
        message = str(exc)
        code = "PROVIDER_NOT_CONFIGURED" if "缺少 LLM provider 配置" in message else exc.__class__.__name__
        return {"status": "provider_not_configured" if code == "PROVIDER_NOT_CONFIGURED" else "api_error", "error": {"code": code, "message": message}}
    except Exception as exc:  # pragma: no cover
        return {"status": "api_error", "error": {"code": exc.__class__.__name__, "message": str(exc)}}
