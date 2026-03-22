# Design: Background TTS Jobs

**Date:** 2026-03-22  
**Status:** Approved

## Overview

Add two endpoints to VibeServer that allow callers to submit TTS generation as a background job and poll for its completion. This decouples the HTTP response from the (potentially long) inference time, which is useful for long inputs or clients that cannot hold open an HTTP connection for the duration of inference.

---

## API

### `POST /v1/audio/jobs`

Submit a background TTS generation job. Accepts the same request body as `POST /v1/audio/speech`. Returns immediately with a job record.

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
  "job_id": "abc123def456",
  "status": "queued",
  "created_at": 1711234567.89
}
```

---

### `GET /v1/audio/jobs/{job_id}`

Poll the status of a submitted job, or retrieve its audio result.

**Status responses:**

| Job status | HTTP status | Body |
|---|---|---|
| `queued` | `202 Accepted` | `{"job_id": ..., "status": "queued", "created_at": ...}` |
| `running` | `202 Accepted` | `{"job_id": ..., "status": "running", "created_at": ..., "started_at": ...}` |
| `done` | `200 OK` | `audio/ogg` binary (same format as `POST /v1/audio/speech`) |
| `failed` | `500 Internal Server Error` | `{"job_id": ..., "status": "failed", "error": "..."}` |
| `expired` | `410 Gone` | `{"job_id": ..., "status": "expired"}` |
| Unknown ID | `404 Not Found` | `{"detail": "Job not found"}` |

When status is `done`, the response includes:
- `Content-Type: audio/ogg`
- `Content-Disposition: attachment; filename=speech.ogg`
- `X-Generation-Time: <seconds>` (time from job start to completion)

---

## Data Model

### Job state machine

```
queued ──► running ──► done ──► expired (after result_ttl elapses)
  │                  ↗
  └──► failed (timeout before lock, timeout during inference, inference error)
         └──► (removed from store after result_ttl elapses)
```

### `Job` dataclass (in `app/jobs.py`)

```python
@dataclass
class Job:
    job_id: str
    request: SpeechRequest
    status: Literal["queued", "running", "done", "failed", "expired"]
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result_path: Path | None = None  # temp file path for audio result
    task: asyncio.Task | None = None  # background asyncio task
```

### In-process job store

A `dict[str, Job]` keyed by `job_id`. Lives in `app/jobs.py` as a module-level variable. Job IDs are `secrets.token_hex(16)` (32 hex chars).

### Result storage

Audio bytes for completed jobs are written to a temp file (`tempfile.mkstemp(suffix=".ogg")`). The path is stored in `job.result_path`. The file is read and deleted when served to the client on GET, or deleted by the cleanup sweep when the job expires.

---

## Configuration

All timeouts are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `TTS_JOB_TIMEOUT` | `300` | Seconds. A job that is still `queued` or `running` after this time is marked `failed`. |
| `TTS_RESULT_TTL` | `3600` | Seconds. A `done` job's audio file is kept for this long before being deleted and the job marked `expired`. |
| `TTS_CLEANUP_INTERVAL` | `3600` | Seconds. How often the background cleanup coroutine runs. |

---

## Execution Logic

### Job submission (`POST /v1/audio/jobs`)

1. Validate request (same validation as the sync endpoint).
2. Create `Job` with `status="queued"`, `created_at=time.time()`.
3. Insert into `_jobs` dict.
4. `asyncio.create_task(_run_job(job))` — fire and forget.
5. Return `{job_id, status, created_at}` immediately.

### Background task (`_run_job` in `app/jobs.py`)

1. Check `time.time() - job.created_at > JOB_TIMEOUT`. If so: set `status="failed"`, `error="Job timed out before inference could start"`, return.
2. Acquire `inference_lock` (from `model.py`).
3. Re-check timeout (time may have elapsed waiting for the lock). If expired: set `status="failed"`, return.
4. Set `status="running"`, `started_at=time.time()`.
5. Compute `remaining = JOB_TIMEOUT - (time.time() - job.created_at)`.
6. `await asyncio.wait_for(asyncio.to_thread(_model.generate_speech, text=..., voice=...), timeout=remaining)`.
   - On `asyncio.TimeoutError`: set `status="failed"`, `error="Inference timed out"`.
   - On other exception: set `status="failed"`, `error=str(exc)`.
7. On success: write bytes to temp file, set `status="done"`, `completed_at=time.time()`, `result_path=<path>`.

### Cleanup coroutine (`_cleanup_loop` in `app/jobs.py`)

Runs on FastAPI lifespan startup, loops every `CLEANUP_INTERVAL` seconds.

For each job in `_jobs`:
- If `status in ("queued", "running")` and `time.time() - job.created_at > JOB_TIMEOUT`:
  - Cancel `job.task` if it exists and is not done.
  - Set `status="failed"`, `error="Timed out"`.
- If `status == "done"` and `time.time() - job.completed_at > RESULT_TTL`:
  - Delete `job.result_path` if it exists.
  - Set `status="expired"`.
- If `status in ("failed", "expired")` and `time.time() - (job.completed_at or job.created_at) > RESULT_TTL`:
  - Remove from `_jobs` dict entirely (prevents unbounded memory growth).

---

## Code Organisation

| File | Change |
|---|---|
| `app/jobs.py` | **New.** Contains `Job` dataclass, `_jobs` dict, `submit_job()`, `get_job()`, `_run_job()`, `_cleanup_loop()`, env-var config constants. |
| `app/app.py` | **Modified.** Add FastAPI lifespan handler to start `_cleanup_loop`. Register `POST /v1/audio/jobs` and `GET /v1/audio/jobs/{job_id}` routes. Import from `jobs.py`. |
| `tests/test_jobs.py` | **New.** Unit tests for the job endpoints (see Testing section). |
| `docs/README.md` | **Modified.** Document the two new endpoints and the three new env vars. |

---

## Testing

File: `tests/test_jobs.py`

Uses the same `patch_model` fixture pattern as `tests/test_auth.py` to avoid loading ML weights.

Test cases:
1. **Submit job** — `POST /v1/audio/jobs` with valid body and key → `200`, response has `job_id`, `status="queued"`, `created_at`.
2. **Poll queued** — `GET /v1/audio/jobs/{id}` immediately after submit → `202`, `status="queued"`.
3. **Poll completed** — poll after the background task completes → `200`, `Content-Type: audio/ogg`.
4. **Poll failed** — mock inference to raise an exception → `500`, `status="failed"`, `error` present.
5. **Poll expired** — manually set `status="expired"` on a job → `410`.
6. **Poll unknown** — `GET /v1/audio/jobs/nonexistent` → `404`.
7. **Auth: submit without key** → `401`.
8. **Auth: poll without key** → `401`.
9. **Timeout before lock** — set `created_at` far in the past → job fails with timeout error.
10. **Cleanup removes expired jobs** — trigger cleanup, verify dict cleanup and file deletion.

---

## Error Handling

- Inference errors are caught and stored in `job.error`. The job is marked `failed`.
- Timeout during queueing (before lock) → `failed` with descriptive error.
- Timeout during inference (`asyncio.TimeoutError`) → `failed` with descriptive error.
- Temp file missing when serving a `done` job (unexpected) → `500` with descriptive error.
- All errors are logged with `logger.error`.
