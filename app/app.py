"""
app.py — FastAPI application exposing an OpenAI-compatible TTS API.

Endpoints:
  POST /v1/audio/speech  — synthesise text, return WAV audio
  GET  /v1/models        — list available models

The model is loaded at worker startup (see model.py).  All inference requests
are serialised through model.inference_lock so only one generation runs at a
time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import model as _model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VibeServer",
    description="OpenAI-compatible TTS server backed by VibeVoice-7B.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------- #
#  Request / response schemas
# ---------------------------------------------------------------------------- #

# Valid OpenAI voice names that we map to VibeVoice presets.
VoiceName = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class SpeechRequest(BaseModel):
    model: str = Field(..., description="Must be 'vibevoice-7b'.")
    input: str = Field(..., min_length=1, description="Text to synthesise.")
    voice: VoiceName = Field(
        default="alloy",
        description="Voice preset.  Maps to a bundled WAV sample.",
    )
    response_format: Optional[str] = Field(
        default="wav",
        description="Audio format.  Only 'wav' is supported; other values are ignored.",
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
            "content": {"audio/wav": {}},
            "description": "WAV audio of the synthesised speech.",
        },
        422: {"description": "Validation error (e.g. missing `input`)."},
        500: {"description": "Inference error."},
    },
    summary="Create speech",
    description=(
        "Synthesise text into speech and return WAV audio.  "
        "Compatible with the OpenAI `POST /v1/audio/speech` endpoint."
    ),
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
            wav_bytes: bytes = await asyncio.to_thread(
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
        "Speech generated in %.1f s — voice=%s, chars=%d, wav_bytes=%d",
        elapsed,
        request.voice,
        len(request.input),
        len(wav_bytes),
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
            "X-Generation-Time": f"{elapsed:.3f}",
        },
    )


@app.get(
    "/v1/models",
    summary="List models",
    description="Returns the list of available models in OpenAI format.",
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


# ---------------------------------------------------------------------------- #
#  Health / liveness probe (for Kubernetes)
# ---------------------------------------------------------------------------- #

@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
