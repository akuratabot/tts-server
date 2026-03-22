# Background TTS Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/audio/jobs` (submit background TTS job) and `GET /v1/audio/jobs/{job_id}` (poll status / retrieve audio) endpoints to VibeServer, with configurable job timeout and result TTL.

**Architecture:** A new `app/jobs.py` module owns the `Job` dataclass, in-process `_jobs` dict, background task coroutine `_run_job`, and cleanup loop `_cleanup_loop`. `app/app.py` gains a FastAPI lifespan handler that starts the cleanup loop and registers the two new routes. Completed audio is stored in temp files; the cleanup loop deletes them on expiry, and they are also deleted immediately when served to the client.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, `tempfile`, `dataclasses`, `secrets`, existing `model.py` (inference lock + `generate_speech`).

**Spec:** `docs/superpowers/specs/2026-03-22-background-tts-jobs-design.md`

---

## Important: Module Import Conventions

The server runs with `app/` on `sys.path` (see the existing `import model as _model` in `app/app.py`). This means:
- `app/app.py` uses `import jobs as _jobs_module` (bare name)
- `app/jobs.py` uses `import model as _model` (bare name, inside function body)

Tests, however, run from the repo root and use `import app.jobs` and `import app.app` (package-qualified). **These are two different module objects.** To keep them in sync, `make_client()` in the test file reloads `app.app` first (which re-executes `import jobs as _jobs_module`, binding to the `jobs` bare-name module), then reloads `app.jobs`. After this sequence the `_jobs` dict inside the running app and the `jobs_module` in the test are the same object, so test-injected jobs are visible to route handlers.

The exact sequence in `make_client` is critical — do not change the reload order.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/jobs.py` | **Create** | Job dataclass, store, submit, run, cleanup |
| `app/app.py` | **Modify** | Lifespan handler, two new routes, response schemas |
| `tests/test_jobs.py` | **Create** | All job endpoint and unit tests |
| `docs/README.md` | **Modify** | Document new endpoints + env vars |

---

## Task 1: Create `app/jobs.py` — config, dataclass, store

**Files:**
- Create: `app/jobs.py`

- [ ] **Step 1: Write the file with config constants and Job dataclass**

```python
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
```

- [ ] **Step 2: No test needed for pure config/dataclass — commit the skeleton**

```bash
git add app/jobs.py
git commit -m "feat: add jobs.py skeleton with config and Job dataclass"
```

---

## Task 2: Implement `submit_job` and `get_job` in `app/jobs.py`

**Files:**
- Modify: `app/jobs.py`
- Create: `tests/test_jobs.py` (partial — store/retrieval tests only)

- [ ] **Step 1: Write failing tests for submit_job and get_job**

```python
# tests/test_jobs.py
"""tests/test_jobs.py — Background TTS job endpoint tests.

Import conventions:
- Tests use `import app.jobs as jobs_module` (package-qualified, from repo root).
- app/app.py uses `import jobs as _jobs_module` (bare name, app/ is on sys.path).
- make_client() reloads app.app first (rebinds its _jobs_module to the bare-name
  'jobs' module), then reloads app.jobs. After this sequence both the running app
  and the test reference the same _jobs dict. Reload order matters — do not swap.
"""
import asyncio
import os
import importlib
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-secret-key"


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """Prevent model.py from loading ML weights during tests."""
    fake_model = MagicMock()
    fake_model.inference_lock = asyncio.Lock()
    fake_model.generate_speech.return_value = b"\x00" * 64
    fake_model.available_voices.return_value = ["test_voice"]
    fake_model.refresh_voices.return_value = ["test_voice"]
    # Patch both the bare name and the package-qualified name so that
    # whichever import path is used at runtime, the mock is found.
    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "app.model", fake_model)


def make_client(api_key: str = VALID_KEY) -> tuple[TestClient, object]:
    """Reload app with TTS_API_KEY set and return (TestClient, jobs_module).

    Reload order is critical:
    1. Reload app.app  — re-executes `import jobs as _jobs_module` inside app.py,
       binding it to the 'jobs' module object (bare name, loaded via app/ sys.path).
    2. Reload app.jobs — resets the _jobs dict and all state for a clean test.
    After both reloads, app._jobs_module and the returned jobs_module share the
    same underlying module object (sys.modules['jobs'] == sys.modules['app.jobs']).
    """
    with patch.dict(os.environ, {"TTS_API_KEY": api_key}):
        import app.app as app_module
        importlib.reload(app_module)
        import app.jobs as jobs_module
        importlib.reload(jobs_module)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        return client, jobs_module


# --------------------------------------------------------------------------- #
# submit_job / get_job unit tests (no HTTP layer yet)
# --------------------------------------------------------------------------- #

def test_submit_job_creates_job_record():
    """submit_job returns a Job with status=queued and a non-empty job_id."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)

    # submit_job schedules an asyncio task — run inside an event loop.
    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="hello", voice="")
        assert job.job_id
        assert job.status == "queued"
        assert job.model == "vibevoice-7b"
        assert job.input == "hello"
        assert job.created_at > 0
        if job.task:
            job.task.cancel()

    asyncio.run(_go())


def test_get_job_returns_none_for_unknown():
    import app.jobs as jobs_module
    importlib.reload(jobs_module)
    assert jobs_module.get_job("nonexistent") is None


def test_get_job_returns_submitted_job():
    import app.jobs as jobs_module
    importlib.reload(jobs_module)

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="hi", voice="")
        fetched = jobs_module.get_job(job.job_id)
        assert fetched is job
        if job.task:
            job.task.cancel()

    asyncio.run(_go())
```

- [ ] **Step 2: Run the tests — expect AttributeError (functions don't exist yet)**

```bash
pytest tests/test_jobs.py::test_submit_job_creates_job_record tests/test_jobs.py::test_get_job_returns_none_for_unknown tests/test_jobs.py::test_get_job_returns_submitted_job -v
```

Expected: FAIL (`AttributeError: module 'app.jobs' has no attribute 'submit_job'`)

- [ ] **Step 3: Implement `submit_job` and `get_job` in `app/jobs.py`**

Append after the `_jobs` dict:

```python
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
```

Add a stub for `_run_job` (implemented in Task 3):

```python
async def _run_job(job: Job) -> None:
    """Background coroutine — implemented in Task 3."""
    pass
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
pytest tests/test_jobs.py::test_submit_job_creates_job_record tests/test_jobs.py::test_get_job_returns_none_for_unknown tests/test_jobs.py::test_get_job_returns_submitted_job -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: add submit_job, get_job, and store unit tests"
```

---

## Task 3: Implement `_run_job` (background inference + timeout)

**Files:**
- Modify: `app/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write failing tests for `_run_job` behaviour**

Add to `tests/test_jobs.py`:

```python
# --------------------------------------------------------------------------- #
# _run_job behaviour
# --------------------------------------------------------------------------- #

def test_run_job_sets_done_on_success():
    """_run_job transitions queued → running → done and writes a temp file."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)
    import model as fake_model  # mocked by patch_model fixture
    fake_model.generate_speech.return_value = b"FAKEAUDIO"

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="hello", voice="")
        if job.task:
            job.task.cancel()
        # Wait one tick for cancellation to propagate.
        await asyncio.sleep(0)

        job.status = "queued"
        job.task = None
        await jobs_module._run_job(job)

        assert job.status == "done"
        assert job.result_path is not None
        assert job.result_path.exists()
        assert job.result_path.read_bytes() == b"FAKEAUDIO"
        assert job.started_at is not None
        assert job.completed_at is not None
        job.result_path.unlink(missing_ok=True)

    asyncio.run(_go())


def test_run_job_fails_on_inference_error():
    """_run_job marks status=failed when generate_speech raises."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)
    import model as fake_model
    fake_model.generate_speech.side_effect = RuntimeError("GPU exploded")

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="hello", voice="")
        if job.task:
            job.task.cancel()
        await asyncio.sleep(0)

        job.status = "queued"
        job.task = None
        await jobs_module._run_job(job)

        assert job.status == "failed"
        assert "GPU exploded" in job.error

    asyncio.run(_go())
    fake_model.generate_speech.side_effect = None  # reset


def test_run_job_fails_when_already_timed_out():
    """_run_job marks status=failed immediately if job creation is beyond JOB_TIMEOUT."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="hello", voice="")
        if job.task:
            job.task.cancel()
        await asyncio.sleep(0)

        job.status = "queued"
        job.task = None
        # Backdate creation so the job is already expired.
        job.created_at = time.time() - jobs_module.JOB_TIMEOUT - 1
        await jobs_module._run_job(job)

        assert job.status == "failed"
        assert job.error is not None

    asyncio.run(_go())
```

- [ ] **Step 2: Run the tests — expect FAIL (stub `_run_job` does nothing)**

```bash
pytest tests/test_jobs.py::test_run_job_sets_done_on_success tests/test_jobs.py::test_run_job_fails_on_inference_error tests/test_jobs.py::test_run_job_fails_when_already_timed_out -v
```

Expected: 3 FAIL

- [ ] **Step 3: Implement `_run_job` in `app/jobs.py`**

Replace the stub with:

```python
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
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
pytest tests/test_jobs.py::test_run_job_sets_done_on_success tests/test_jobs.py::test_run_job_fails_on_inference_error tests/test_jobs.py::test_run_job_fails_when_already_timed_out -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: implement _run_job with timeout and temp-file result storage"
```

---

## Task 4: Implement `_cleanup_once` and `_cleanup_loop`

**Files:**
- Modify: `app/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write failing tests for cleanup behaviour**

Add to `tests/test_jobs.py`:

```python
# --------------------------------------------------------------------------- #
# _cleanup_loop
# --------------------------------------------------------------------------- #

def test_cleanup_expires_done_job_and_deletes_temp_file():
    """_cleanup_once deletes result file and marks job expired after RESULT_TTL."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)
    import model as fake_model
    fake_model.generate_speech.return_value = b"AUDIO"

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="test", voice="")
        if job.task:
            job.task.cancel()
        await asyncio.sleep(0)
        job.status = "queued"
        job.task = None
        await jobs_module._run_job(job)

        assert job.status == "done"
        tmp = job.result_path
        assert tmp.exists()

        # Backdate completed_at so cleanup considers the result expired.
        job.completed_at = time.time() - jobs_module.RESULT_TTL - 1
        await jobs_module._cleanup_once()

        assert job.status == "expired"
        assert not tmp.exists()

    asyncio.run(_go())


def test_cleanup_removes_old_failed_job_from_store():
    """_cleanup_once removes failed jobs older than RESULT_TTL from the dict."""
    import app.jobs as jobs_module
    importlib.reload(jobs_module)

    async def _go():
        job = jobs_module.submit_job(model="vibevoice-7b", input="test", voice="")
        if job.task:
            job.task.cancel()
        await asyncio.sleep(0)

        job.status = "failed"
        job.error = "test failure"
        # Backdate so cleanup removes it from the store entirely.
        job.created_at = time.time() - jobs_module.RESULT_TTL - 1

        await jobs_module._cleanup_once()
        assert jobs_module.get_job(job.job_id) is None

    asyncio.run(_go())
```

- [ ] **Step 2: Run the tests — expect FAIL**

```bash
pytest tests/test_jobs.py::test_cleanup_expires_done_job_and_deletes_temp_file tests/test_jobs.py::test_cleanup_removes_old_failed_job_from_store -v
```

Expected: 2 FAIL (`AttributeError: module 'app.jobs' has no attribute '_cleanup_once'`)

- [ ] **Step 3: Implement `_cleanup_once` and `_cleanup_loop` in `app/jobs.py`**

Append to `app/jobs.py`:

```python
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
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
pytest tests/test_jobs.py::test_cleanup_expires_done_job_and_deletes_temp_file tests/test_jobs.py::test_cleanup_removes_old_failed_job_from_store -v
```

Expected: 2 passed

- [ ] **Step 5: Run the full jobs test suite so far**

```bash
pytest tests/test_jobs.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: implement cleanup loop with expiry and store pruning"
```

---

## Task 5: Wire up FastAPI routes and lifespan in `app/app.py`

**Files:**
- Modify: `app/app.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write failing HTTP-layer tests**

Add to `tests/test_jobs.py`:

```python
# --------------------------------------------------------------------------- #
# HTTP endpoint tests
# --------------------------------------------------------------------------- #

def test_submit_job_endpoint_returns_queued():
    """POST /v1/audio/jobs returns 200 with job_id and status=queued."""
    client, _ = make_client()
    resp = client.post(
        "/v1/audio/jobs",
        json={"model": "vibevoice-7b", "input": "Hello"},
        headers={"X-Api-Key": VALID_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "created_at" in body


def test_submit_job_requires_auth():
    client, _ = make_client()
    resp = client.post("/v1/audio/jobs", json={"model": "vibevoice-7b", "input": "Hi"})
    assert resp.status_code == 401


def test_get_job_unknown_returns_404():
    client, _ = make_client()
    resp = client.get("/v1/audio/jobs/doesnotexist", headers={"X-Api-Key": VALID_KEY})
    assert resp.status_code == 404


def test_get_job_requires_auth():
    client, _ = make_client()
    resp = client.get("/v1/audio/jobs/someid")
    assert resp.status_code == 401


def test_get_job_queued_returns_202():
    """GET /v1/audio/jobs/{id} returns 202 while job is queued.

    Job is manually injected into the store (via the jobs_module returned by
    make_client) rather than submitted via the endpoint, so there is no race
    between the background task completing and the GET request.
    """
    client, jobs_module = make_client()

    job = jobs_module.Job(
        job_id="test-queued-job",
        model="vibevoice-7b",
        input="hello",
        voice="",
        status="queued",
        created_at=time.time(),
    )
    jobs_module._jobs["test-queued-job"] = job

    resp = client.get("/v1/audio/jobs/test-queued-job", headers={"X-Api-Key": VALID_KEY})
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"


def test_get_job_done_returns_audio_and_deletes_file():
    """GET /v1/audio/jobs/{id} returns 200 audio/ogg and deletes the temp file."""
    import tempfile as _tempfile
    client, jobs_module = make_client()

    fd, tmp = _tempfile.mkstemp(suffix=".ogg")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"FAKEAUDIO")

        job = jobs_module.Job(
            job_id="test-done-job",
            model="vibevoice-7b",
            input="hello",
            voice="",
            status="done",
            created_at=time.time(),
            completed_at=time.time(),
            result_path=Path(tmp),
        )
        jobs_module._jobs["test-done-job"] = job

        resp = client.get("/v1/audio/jobs/test-done-job", headers={"X-Api-Key": VALID_KEY})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/ogg"
        assert resp.content == b"FAKEAUDIO"
        # Spec: result file is deleted on serve.
        assert not Path(tmp).exists()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def test_get_job_failed_returns_500():
    """GET /v1/audio/jobs/{id} returns 500 when job failed."""
    client, jobs_module = make_client()

    job = jobs_module.Job(
        job_id="test-failed-job",
        model="vibevoice-7b",
        input="hello",
        voice="",
        status="failed",
        created_at=time.time(),
        error="something broke",
    )
    jobs_module._jobs["test-failed-job"] = job

    resp = client.get("/v1/audio/jobs/test-failed-job", headers={"X-Api-Key": VALID_KEY})
    assert resp.status_code == 500
    assert resp.json()["status"] == "failed"
    assert "something broke" in resp.json()["error"]


def test_get_job_expired_returns_410():
    """GET /v1/audio/jobs/{id} returns 410 when job is expired."""
    client, jobs_module = make_client()

    job = jobs_module.Job(
        job_id="test-expired-job",
        model="vibevoice-7b",
        input="hello",
        voice="",
        status="expired",
        created_at=time.time(),
    )
    jobs_module._jobs["test-expired-job"] = job

    resp = client.get("/v1/audio/jobs/test-expired-job", headers={"X-Api-Key": VALID_KEY})
    assert resp.status_code == 410
    assert resp.json()["status"] == "expired"
```

- [ ] **Step 2: Run the new tests — expect FAIL (routes don't exist yet)**

```bash
pytest tests/test_jobs.py::test_submit_job_endpoint_returns_queued tests/test_jobs.py::test_get_job_unknown_returns_404 -v
```

Expected: FAIL (`404 Not Found` — no route registered)

- [ ] **Step 3: Modify `app/app.py` — add lifespan, imports, schemas, and two routes**

**3a. Add imports** at the top of `app/app.py` (after the existing import block):

```python
import contextlib

import jobs as _jobs_module  # bare import — app/ is on sys.path
```

**3b. Replace the `app = FastAPI(...)` instantiation** with a lifespan-enabled version:

```python
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
```

**3c. Add response schemas** (after the existing `SpeechRequest` class):

```python
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
```

**3d. Add the two new endpoints** after the existing `/v1/audio/speech` route:

```python
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
```

- [ ] **Step 4: Run all HTTP-layer tests**

```bash
pytest tests/test_jobs.py -k "endpoint or auth or queued or done or failed or expired or 404" -v
```

Expected: all pass

- [ ] **Step 5: Fix `tests/test_auth.py` — add jobs mock to patch_model fixture**

After Task 5, `app/app.py` imports `jobs` at module level. The existing `test_auth.py`
reloads `app.app` without importing `jobs` first, causing `ModuleNotFoundError`.

Edit `tests/test_auth.py` — update the `patch_model` fixture to also mock the `jobs` module:

```python
@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """Prevent model.py and jobs.py from loading during tests."""
    fake_model = MagicMock()
    fake_model.inference_lock = __import__("asyncio").Lock()
    fake_model.generate_speech.return_value = b"\x00" * 16
    fake_model.available_voices.return_value = ["test_voice"]
    fake_model.refresh_voices.return_value = ["test_voice"]
    monkeypatch.setitem(sys.modules, "model", fake_model)

    # jobs module is imported at app.py module level; mock it so test_auth.py
    # reloads don't fail with ModuleNotFoundError.
    fake_jobs = MagicMock()
    # _cleanup_loop is called as asyncio.create_task(_cleanup_loop()) in the lifespan
    # handler — it must return a coroutine, not an async generator.
    async def _noop_loop():
        pass
    fake_jobs._cleanup_loop = _noop_loop
    monkeypatch.setitem(sys.modules, "jobs", fake_jobs)
    monkeypatch.setitem(sys.modules, "app.jobs", fake_jobs)
```

- [ ] **Step 6: Run the entire test suite (including existing auth tests)**

```bash
pytest tests/ -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add app/app.py tests/test_auth.py tests/test_jobs.py
git commit -m "feat: add POST /v1/audio/jobs and GET /v1/audio/jobs/{job_id} endpoints"
```

---

## Task 6: Update `docs/README.md`

**Files:**
- Modify: `docs/README.md`

- [ ] **Step 1: Add new env vars to the Environment Variables table**

In the `## Environment Variables` section, append three rows to the existing table:

```markdown
| `TTS_JOB_TIMEOUT` | `300` | Seconds before a background job (queued or running) is considered timed out and marked failed. |
| `TTS_RESULT_TTL` | `3600` | Seconds to keep a completed job's audio file before expiring it. Cleanup runs every `TTS_CLEANUP_INTERVAL` seconds; the file is also deleted immediately when served. |
| `TTS_CLEANUP_INTERVAL` | `3600` | How often the background cleanup sweep runs (seconds). |
```

- [ ] **Step 2: Add new endpoint docs to the API Reference section**

After the `### POST /v1/audio/speech` section and before `### GET /v1/voices`, insert:

````markdown
---

### `POST /v1/audio/jobs`

Submit a background TTS generation job. Accepts the same request body as `POST /v1/audio/speech` and returns immediately with a job record. The audio is generated in the background; poll `GET /v1/audio/jobs/{job_id}` for results.

**Request body (JSON):**

```json
{
  "model": "vibevoice-7b",
  "input": "Hello, world!",
  "voice": "alice"
}
```

**Response — `200 OK`:**

```json
{
  "job_id": "abc123def456abc123def456abc123de",
  "status": "queued",
  "created_at": 1711234567.89
}
```

**curl example:**

```bash
JOB=$(curl -s -X POST http://localhost:8000/v1/audio/jobs \
  -H "X-Api-Key: $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"vibevoice-7b","input":"Hello from VibeVoice!","voice":"nova"}')
echo $JOB
# {"job_id":"abc123...","status":"queued","created_at":...}
```

---

### `GET /v1/audio/jobs/{job_id}`

Poll the status of a submitted job or retrieve its audio.

| Job status | HTTP status | Response body |
|---|---|---|
| `queued` | `202 Accepted` | JSON `{"job_id":...,"status":"queued","created_at":...}` |
| `running` | `202 Accepted` | JSON `{"job_id":...,"status":"running","started_at":...}` |
| `done` | `200 OK` | `audio/ogg` binary — result file is deleted after serving |
| `failed` | `500 Internal Server Error` | JSON `{"job_id":...,"status":"failed","error":"..."}` |
| `expired` | `410 Gone` | JSON `{"job_id":...,"status":"expired"}` |
| Unknown ID | `404 Not Found` | JSON `{"detail":"Job not found"}` |

**curl example (poll until done):**

```bash
JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-Api-Key: $TTS_API_KEY" \
    http://localhost:8000/v1/audio/jobs/$JOB_ID)
  if [ "$STATUS" = "200" ]; then
    curl -s -H "X-Api-Key: $TTS_API_KEY" \
      http://localhost:8000/v1/audio/jobs/$JOB_ID \
      --output speech.ogg
    echo "Downloaded speech.ogg"
    break
  elif [ "$STATUS" = "202" ]; then
    echo "Still processing… (HTTP $STATUS)"
    sleep 5
  else
    echo "Error: HTTP $STATUS"
    break
  fi
done
```
````

- [ ] **Step 3: Update the Architecture diagram in `## Architecture`**

The current diagram in `docs/README.md` looks like:

```
│  uvicorn :8000  (FastAPI)                                │
│    ├── POST /v1/audio/speech  ──► asyncio.Lock           │
│    │                                └► model.generate() │
│    ├── GET  /v1/models                                   │
│    └── GET  /health                                      │
```

Update it to:

```
│  uvicorn :8000  (FastAPI)                                │
│    ├── POST /v1/audio/speech  ──► asyncio.Lock           │
│    │                                └► model.generate() │
│    ├── POST /v1/audio/jobs    ──► jobs.submit_job()      │
│    │                                └► asyncio.Task     │
│    ├── GET  /v1/audio/jobs/{id} ──► jobs.get_job()       │
│    ├── GET  /v1/models                                   │
│    └── GET  /health                                      │
```

- [ ] **Step 4: Commit**

```bash
git add docs/README.md
git commit -m "docs: document background job endpoints and new env vars"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all pass, no warnings about unrelated failures.

- [ ] **Step 2: Verify no import errors in app/jobs.py**

```bash
python -c "
import sys
from unittest.mock import MagicMock
sys.modules['model'] = MagicMock()
sys.path.insert(0, 'app')
import jobs
print('jobs.py OK — JOB_TIMEOUT=%s RESULT_TTL=%s' % (jobs.JOB_TIMEOUT, jobs.RESULT_TTL))
"
```

Expected: `jobs.py OK — JOB_TIMEOUT=300.0 RESULT_TTL=3600.0`

- [ ] **Step 3: Check git log looks clean**

```bash
git log --oneline -8
```

Expected: 6 feature commits on top of the design doc commit.

- [ ] **Step 4: Create the PR**

```bash
git push origin HEAD
gh pr create \
  --title "feat: add background TTS job endpoints" \
  --body "$(cat <<'EOF'
## Summary

- Adds `POST /v1/audio/jobs` to submit TTS generation as a background job, returning immediately with a `job_id`.
- Adds `GET /v1/audio/jobs/{job_id}` to poll status or retrieve the completed `audio/ogg` result (result file is deleted on serve).
- New `app/jobs.py` module owns the job store, background execution, timeout logic, and periodic cleanup.
- Configurable via `TTS_JOB_TIMEOUT` (default 300s), `TTS_RESULT_TTL` (default 3600s), `TTS_CLEANUP_INTERVAL` (default 3600s).
- Jobs timed out before acquiring the inference lock are marked `failed`; if already running, `asyncio.wait_for` enforces the remaining timeout.
- 13 new tests in `tests/test_jobs.py` covering all job states, auth, cleanup, and file deletion.

Spec: `docs/superpowers/specs/2026-03-22-background-tts-jobs-design.md`
EOF
)"
```
