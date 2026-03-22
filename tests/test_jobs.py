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
