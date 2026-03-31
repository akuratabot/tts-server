"""
app.py — FastAPI application exposing an OpenAI-compatible TTS API.

Endpoints:
  POST /v1/audio/speech   — synthesise text, return WAV audio
  GET  /v1/models         — list available models
  GET  /v1/voices         — list registered voice names
  POST /v1/voices/refresh — re-sync external voices and rebuild the index

The model is loaded at worker startup (see model.py).  All inference requests
are serialised through model.inference_lock so only one generation runs at a
time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import model as _model
import jobs as _jobs_module  # bare import — app/ is on sys.path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#  API key authentication
# ---------------------------------------------------------------------------- #

_API_KEY: str = os.environ.get("TTS_API_KEY", "")
if not _API_KEY:
    raise RuntimeError(
        "TTS_API_KEY environment variable is not set. "
        "Set it to a secret value before starting the server."
    )

_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Depends(_api_key_header),
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: validates auth via X-Api-Key or Authorization: Bearer (timing-safe).

    Accepts either:
      - X-Api-Key: <key>
      - Authorization: Bearer <key>

    X-Api-Key takes precedence. auto_error=False means FastAPI will NOT
    auto-reject missing headers — this function is solely responsible for
    enforcing auth and returning the unified 401 response.
    """
    candidate: str | None = api_key

    # Fall back to Bearer token if X-Api-Key was not provided.
    if candidate is None and authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            candidate = token

    if candidate is None or not secrets.compare_digest(candidate, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the job cleanup loop on startup; cancel it on shutdown."""
    cleanup_task = asyncio.create_task(_jobs_module._cleanup_loop())
    logger.info("Job cleanup loop scheduled.")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Job cleanup loop stopped.")


app = FastAPI(
    title="VibeServer",
    description="OpenAI-compatible TTS server backed by VibeVoice-7B.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------- #
#  Request / response schemas
# ---------------------------------------------------------------------------- #

class SpeechRequest(BaseModel):
    model: str = Field(..., description="Must be 'vibevoice-7b'.")
    input: str = Field(..., min_length=1, description="Text to synthesise.")
    voice: str = Field(
        default="",
        description=(
            "Voice name — must match the stem of a WAV file in app/voices/ "
            "(case-insensitive). Use GET /v1/voices to list available voices. "
            "Unknown names fall back to the first available voice alphabetically."
        ),
    )
    response_format: Optional[str] = Field(
        default="opus",
        description="Audio format.  Only OGG/Opus is supported; this field is accepted for OpenAI compatibility but ignored.",
    )
    speed: Optional[float] = Field(
        default=None,
        description="Playback speed.  Not supported; ignored.",
    )


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str
    created_at: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------- #
#  Endpoints
# ---------------------------------------------------------------------------- #

@app.post(
    "/v1/audio/speech",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/ogg": {}},
            "description": "OGG/Opus audio of the synthesised speech.",
        },
        422: {"description": "Validation error (e.g. missing `input`)."},
        500: {"description": "Inference error."},
    },
    summary="Create speech",
    description=(
        "Synthesise text into speech and return WAV audio.  "
        "Compatible with the OpenAI `POST /v1/audio/speech` endpoint."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def create_speech(request: SpeechRequest) -> Response:
    if request.model != "vibevoice-7b":
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported model '{request.model}'. Use 'vibevoice-7b'.",
        )

    start = time.perf_counter()

    # Acquire the inference lock — queues concurrent requests rather than
    # rejecting them.
    async with _model.inference_lock:
        try:
            # Run the blocking inference in a thread so the event loop stays
            # responsive (allows health checks etc. to complete while waiting).
            audio_bytes: bytes = await asyncio.to_thread(
                _model.generate_speech,
                text=request.input,
                voice=request.voice,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Inference error: {exc}",
            ) from exc

    elapsed = time.perf_counter() - start
    logger.info(
        "Speech generated in %.1f s — voice=%s, chars=%d, audio_bytes=%d",
        elapsed,
        request.voice,
        len(request.input),
        len(audio_bytes),
    )

    return Response(
        content=audio_bytes,
        media_type="audio/ogg",
        headers={
            "Content-Disposition": "attachment; filename=speech.ogg",
            "X-Generation-Time": f"{elapsed:.3f}",
        },
    )


@app.post(
    "/v1/audio/jobs",
    response_model=JobSubmittedResponse,
    status_code=200,
    summary="Submit background speech job",
    description=(
        "Enqueue a TTS generation job and return immediately with a job_id. "
        "Poll GET /v1/audio/jobs/{job_id} for status or to retrieve the audio."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def submit_speech_job(request: SpeechRequest) -> JobSubmittedResponse:
    if request.model != "vibevoice-7b":
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported model '{request.model}'. Use 'vibevoice-7b'.",
        )
    job = _jobs_module.submit_job(
        model=request.model,
        input=request.input,
        voice=request.voice,
    )
    return JobSubmittedResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )


@app.get(
    "/v1/audio/jobs/{job_id}",
    summary="Get job status or audio result",
    description=(
        "Poll the status of a background job. "
        "Returns 202 while queued/running, 200 audio/ogg when done (result file deleted after serving), "
        "500 on failure, 410 if expired, 404 if not found."
    ),
    responses={
        200: {"content": {"audio/ogg": {}}, "description": "Completed audio"},
        202: {"description": "Job is queued or running"},
        404: {"description": "Job not found"},
        410: {"description": "Job result expired"},
        500: {"description": "Job failed"},
    },
    dependencies=[Depends(verify_api_key)],
)
async def get_speech_job(job_id: str) -> Response:
    job = _jobs_module.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "queued":
        return Response(
            content=JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
            ).model_dump_json(),
            status_code=202,
            media_type="application/json",
        )

    if job.status == "running":
        return Response(
            content=JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,  # required field — must always be passed
                started_at=job.started_at,
            ).model_dump_json(),
            status_code=202,
            media_type="application/json",
        )

    if job.status == "done":
        if job.result_path is None or not job.result_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Result file missing — job may have been cleaned up prematurely",
            )
        audio_bytes = job.result_path.read_bytes()
        elapsed = (job.completed_at or 0) - (job.started_at or 0)
        # Delete the temp file immediately after reading — the client now has the data.
        try:
            job.result_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete result file %s: %s", job.result_path, exc)
        job.result_path = None
        # Mark expired so the cleanup loop doesn't try to delete it again.
        job.status = "expired"
        return Response(
            content=audio_bytes,
            media_type="audio/ogg",
            headers={
                "Content-Disposition": "attachment; filename=speech.ogg",
                "X-Generation-Time": f"{elapsed:.3f}",
            },
        )

    if job.status == "failed":
        return Response(
            content=JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
                error=job.error,
            ).model_dump_json(),
            status_code=500,
            media_type="application/json",
        )

    if job.status == "expired":
        return Response(
            content=JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
            ).model_dump_json(),
            status_code=410,
            media_type="application/json",
        )

    # Should never reach here.
    raise HTTPException(status_code=500, detail=f"Unknown job status: {job.status}")


@app.get(
    "/v1/models",
    summary="List models",
    description="Returns the list of available models in OpenAI format.",
    dependencies=[Depends(verify_api_key)],
)
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "vibevoice-7b",
                "object": "model",
                "created": 1724630400,  # 2025-08-26 — VibeVoice-7B release date
                "owned_by": "vibevoice-community",
            }
        ],
    }


@app.get(
    "/v1/voices",
    summary="List voices",
    description=(
        "Returns the currently registered voice names — the stems of audio files "
        "in app/voices/. Use POST /v1/voices/refresh to pick up new files without "
        "restarting the server."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def list_voices() -> dict:
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in _model.available_voices()],
    }


@app.post(
    "/v1/voices/refresh",
    summary="Refresh voice index",
    description=(
        "Re-syncs audio files from VIBEVOICE_EXTRA_VOICES_DIR into app/voices/ "
        "and rebuilds the voice index. Use this after adding new files to the "
        "mounted external directory (e.g. a Cloudflare R2 volume) without "
        "restarting the server. Returns the updated list of registered voices."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def refresh_voices() -> dict:
    try:
        voices = await asyncio.to_thread(_model.refresh_voices)
    except Exception as exc:
        logger.error("Voice refresh failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Voice refresh error: {exc}",
        ) from exc

    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in voices],
    }


# ---------------------------------------------------------------------------- #
#  Health / liveness probe (for Kubernetes)
# ---------------------------------------------------------------------------- #

@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
