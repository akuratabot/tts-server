# app/jobs.py
"""
jobs.py — Background TTS job store and execution logic.

Runs with app/ on sys.path (same as app.py). Import as:
    import jobs as _jobs_module        # from app.py
    import model as _model             # from _run_job (inside function body)

Manages the lifecycle of async TTS generation jobs:
  - Job submission (submit_job)
  - Background execution with timeout (_run_job)
  - Status/result retrieval (get_job)
  - Periodic cleanup of expired jobs and temp files (_cleanup_loop / _cleanup_once)
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#  Configuration
# ---------------------------------------------------------------------------- #

# Seconds before a queued-or-running job is considered timed out.
JOB_TIMEOUT: float = float(os.getenv("TTS_JOB_TIMEOUT", "300"))

# Seconds a completed job's audio file is kept before expiry.
RESULT_TTL: float = float(os.getenv("TTS_RESULT_TTL", "3600"))

# How often the cleanup coroutine runs (seconds).
CLEANUP_INTERVAL: float = float(os.getenv("TTS_CLEANUP_INTERVAL", "3600"))


# ---------------------------------------------------------------------------- #
#  Job dataclass
# ---------------------------------------------------------------------------- #

# NOTE: Job stores model/input/voice as individual fields rather than a
# SpeechRequest object to avoid importing app.py (which would create a circular
# dependency — app.py imports jobs.py).
@dataclass
class Job:
    job_id: str
    model: str
    input: str
    voice: str
    status: Literal["queued", "running", "done", "failed", "expired"]
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_path: Optional[Path] = None
    # repr=False: asyncio.Task is not comparable/serialisable; exclude from repr.
    task: Optional[asyncio.Task] = field(default=None, repr=False)


# ---------------------------------------------------------------------------- #
#  In-process store
# ---------------------------------------------------------------------------- #

_jobs: dict[str, Job] = {}


def submit_job(*, model: str, input: str, voice: str) -> Job:
    """
    Create a Job record, register it in the store, and fire a background task.

    Must be called from within a running asyncio event loop (e.g. from a FastAPI
    route handler or an async test). Returns the Job immediately.
    """
    job_id = secrets.token_hex(16)
    job = Job(
        job_id=job_id,
        model=model,
        input=input,
        voice=voice,
        status="queued",
        created_at=time.time(),
    )
    _jobs[job_id] = job
    # asyncio.ensure_future works whether or not we have a running loop reference,
    # and is safe to call from sync code that is itself inside a running loop
    # (e.g. called from an async FastAPI route via a sync helper).
    job.task = asyncio.ensure_future(_run_job(job))
    return job


def get_job(job_id: str) -> Optional[Job]:
    """Return the Job for *job_id*, or None if not found."""
    return _jobs.get(job_id)


async def _run_job(job: Job) -> None:
    """
    Background coroutine: acquires the inference lock, runs generate_speech,
    and writes the result to a temp file.

    Timeout logic:
    - Checks elapsed time before acquiring the lock (fails fast for stale jobs).
    - Checks again after acquiring the lock (handles long queue wait).
    - Uses asyncio.wait_for with the remaining timeout during inference.

    model is imported inside the function body so that test mocks inserted into
    sys.modules['model'] are picked up at call time, not at module load time.
    """
    import model as _model  # late import — allows sys.modules mock in tests

    def _elapsed() -> float:
        return time.time() - job.created_at

    def _remaining() -> float:
        return JOB_TIMEOUT - _elapsed()

    # Pre-lock timeout check — fail fast for stale queued jobs.
    if _remaining() <= 0:
        job.status = "failed"
        job.error = "Job timed out before inference could start"
        logger.warning("Job %s timed out before acquiring lock", job.job_id)
        return

    async with _model.inference_lock:
        # Post-lock timeout check — may have waited a long time for the lock.
        remaining = _remaining()
        if remaining <= 0:
            job.status = "failed"
            job.error = "Job timed out while waiting for inference slot"
            logger.warning("Job %s timed out after acquiring lock", job.job_id)
            return

        job.status = "running"
        job.started_at = time.time()
        logger.info("Job %s: inference starting (%.1fs remaining)", job.job_id, remaining)

        try:
            audio_bytes: bytes = await asyncio.wait_for(
                asyncio.to_thread(
                    _model.generate_speech,
                    text=job.input,
                    voice=job.voice,
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "Inference timed out"
            logger.error("Job %s: inference timed out", job.job_id)
            return
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            logger.error("Job %s: inference failed: %s", job.job_id, exc, exc_info=True)
            return

    # Write result to a temp file (outside the lock — IO doesn't need the GPU lock).
    fd, tmp_path = tempfile.mkstemp(suffix=".ogg", prefix=f"tts_job_{job.job_id}_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(audio_bytes)
        job.result_path = Path(tmp_path)
        job.status = "done"
        job.completed_at = time.time()
        logger.info(
            "Job %s: done in %.1fs — %d bytes at %s",
            job.job_id,
            job.completed_at - job.started_at,
            len(audio_bytes),
            tmp_path,
        )
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"Failed to write result: {exc}"
        logger.error("Job %s: failed to write temp file: %s", job.job_id, exc, exc_info=True)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _cleanup_once() -> None:
    """
    Single pass of the cleanup logic.

    - Expires done jobs whose result file has outlived RESULT_TTL.
    - Marks timed-out queued/running jobs as failed and cancels their tasks.
    - Removes old failed/expired jobs from the store to prevent unbounded growth.

    Factored out from _cleanup_loop so tests can call it directly without waiting
    for the sleep interval.

    Note on age_reference for failed/expired removal: jobs that fail before
    ever running have no completed_at; their TTL clock starts from created_at,
    which is intentional — they are removed after RESULT_TTL from creation.
    """
    now = time.time()
    # Snapshot keys to allow safe mutation during iteration.
    for job_id in list(_jobs):
        job = _jobs.get(job_id)
        if job is None:
            continue

        if job.status in ("queued", "running"):
            if now - job.created_at > JOB_TIMEOUT:
                if job.task is not None and not job.task.done():
                    job.task.cancel()
                job.status = "failed"
                job.error = "Job timed out"
                logger.warning("Job %s timed out during cleanup", job_id)

        elif job.status == "done":
            if now - (job.completed_at or job.created_at) > RESULT_TTL:
                if job.result_path is not None:
                    try:
                        job.result_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning("Could not delete %s: %s", job.result_path, exc)
                job.status = "expired"
                logger.info("Job %s expired — result deleted", job_id)

        if job.status in ("failed", "expired"):
            age_reference = job.completed_at or job.created_at
            if now - age_reference > RESULT_TTL:
                del _jobs[job_id]
                logger.debug("Job %s removed from store", job_id)


async def _cleanup_loop() -> None:
    """
    Periodic cleanup coroutine. Runs every CLEANUP_INTERVAL seconds.
    Started as an asyncio task from the FastAPI lifespan handler.
    """
    logger.info(
        "Job cleanup loop started (interval=%.0fs, job_timeout=%.0fs, result_ttl=%.0fs)",
        CLEANUP_INTERVAL, JOB_TIMEOUT, RESULT_TTL,
    )
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            await _cleanup_once()
        except Exception as exc:  # noqa: BLE001
            logger.error("Cleanup loop error: %s", exc, exc_info=True)
