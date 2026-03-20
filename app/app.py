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
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import model as _model

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


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency: validates the X-Api-Key header (timing-safe).

    auto_error=False means FastAPI will NOT auto-reject missing headers —
    this function is solely responsible for enforcing auth and returning the
    unified 401 response. Do not change auto_error to True.
    """
    if api_key is None or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(
    title="VibeServer",
    description="OpenAI-compatible TTS server backed by VibeVoice-7B.",
    version="1.0.0",
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
