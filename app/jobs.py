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
