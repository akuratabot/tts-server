# API Key Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect all business endpoints with `X-Api-Key` header authentication backed by an `API_KEY` environment variable, leaving `/health` open.

**Architecture:** A single `APIKeyHeader` security scheme and `verify_api_key` FastAPI dependency are added to `app/app.py`. All four business endpoints declare this dependency. The server refuses to start if `API_KEY` is unset.

**Tech Stack:** Python 3.11+, FastAPI, `secrets` (stdlib), `pytest`, `httpx` (for test client)

---

### Task 1: Add failing tests for API key auth

**Files:**
- Create: `tests/test_auth.py`

The tests use FastAPI's `TestClient` (via `httpx`). Because `model.py` imports heavy ML libraries at startup, we override the app's model dependency and startup behaviour using `unittest.mock.patch` so tests run without GPU/model weights.

- [ ] **Step 1: Create the test file**

```python
"""tests/test_auth.py — API key authentication tests."""
import os
import importlib
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-secret-key"


def make_client(api_key: str = VALID_KEY) -> TestClient:
    """Import app with API_KEY set, return a TestClient."""
    # Patch env before importing so the startup guard sees the key.
    with patch.dict(os.environ, {"API_KEY": api_key}):
        # Reload to pick up the patched env at module level.
        import app.app as app_module
        importlib.reload(app_module)
        return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """Prevent model.py from loading ML weights during tests."""
    fake_model = MagicMock()
    fake_model.inference_lock = __import__("asyncio").Lock()
    fake_model.generate_speech.return_value = b"\x00" * 16
    fake_model.available_voices.return_value = ["test_voice"]
    fake_model.refresh_voices.return_value = ["test_voice"]
    monkeypatch.setitem(__import__("sys").modules, "model", fake_model)


# --------------------------------------------------------------------------- #
# Startup guard
# --------------------------------------------------------------------------- #

def test_missing_api_key_raises_at_import():
    """Server must refuse to start when API_KEY is absent."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("API_KEY", None)
        import app.app as app_module
        with pytest.raises((RuntimeError, Exception)):
            importlib.reload(app_module)


# --------------------------------------------------------------------------- #
# /health — always open
# --------------------------------------------------------------------------- #

def test_health_no_key():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_wrong_key():
    client = make_client()
    resp = client.get("/health", headers={"X-Api-Key": "wrong"})
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Protected endpoints — missing key → 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("POST", "/v1/audio/speech", {"model": "vibevoice-7b", "input": "hello"}),
    ("GET",  "/v1/models",       None),
    ("GET",  "/v1/voices",       None),
    ("POST", "/v1/voices/refresh", None),
])
def test_missing_key_returns_401(method, path, body):
    client = make_client()
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Protected endpoints — wrong key → 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("POST", "/v1/audio/speech", {"model": "vibevoice-7b", "input": "hello"}),
    ("GET",  "/v1/models",       None),
    ("GET",  "/v1/voices",       None),
    ("POST", "/v1/voices/refresh", None),
])
def test_wrong_key_returns_401(method, path, body):
    client = make_client()
    resp = client.request(method, path, json=body, headers={"X-Api-Key": "bad-key"})
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Protected endpoints — valid key → not 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("GET",  "/v1/models",  None),
    ("GET",  "/v1/voices",  None),
    ("POST", "/v1/voices/refresh", None),
])
def test_valid_key_passes_auth(method, path, body):
    client = make_client()
    resp = client.request(method, path, json=body, headers={"X-Api-Key": VALID_KEY})
    assert resp.status_code != 401
```

- [ ] **Step 2: Verify tests can be discovered (they will fail — that's expected)**

```bash
cd /workspace/tts-server && python -m pytest tests/test_auth.py -v --no-header 2>&1 | head -40
```

Expected: collection succeeds, tests fail with import errors or assertion errors (auth not yet implemented).

---

### Task 2: Implement API key authentication in app.py

**Files:**
- Modify: `app/app.py`

- [ ] **Step 1: Add imports at the top of app.py**

In `app/app.py`, add to the existing import block:

```python
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
```

(Replace the existing `from fastapi import FastAPI, HTTPException` line.)

- [ ] **Step 2: Add startup guard and security dependency after the imports, before `app = FastAPI(...)`**

```python
# ---------------------------------------------------------------------------- #
#  API key authentication
# ---------------------------------------------------------------------------- #

_API_KEY: str = os.environ.get("API_KEY", "")
if not _API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. "
        "Set it to a secret value before starting the server."
    )

_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency: validates the X-Api-Key header (timing-safe)."""
    if api_key is None or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

- [ ] **Step 3: Add the dependency to each protected endpoint decorator**

For each of the four route decorators, add `dependencies=[Depends(verify_api_key)]`:

- `@app.post("/v1/audio/speech", ...)` → add `dependencies=[Depends(verify_api_key)]`
- `@app.get("/v1/models", ...)` → add `dependencies=[Depends(verify_api_key)]`
- `@app.get("/v1/voices", ...)` → add `dependencies=[Depends(verify_api_key)]`
- `@app.post("/v1/voices/refresh", ...)` → add `dependencies=[Depends(verify_api_key)]`

Leave `@app.get("/health", ...)` unchanged.

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd /workspace/tts-server && API_KEY=test-secret-key python -m pytest tests/test_auth.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/app.py tests/test_auth.py
git commit -m "feat: add X-Api-Key header authentication backed by API_KEY env var"
```

---

### Task 3: Update Dockerfile comment to document API_KEY

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Add API_KEY to the example run command in the Dockerfile comment**

In `Dockerfile`, update the run example comment (around line 13) to include `-e API_KEY=your-secret-key`:

```dockerfile
# Run (example):
#   docker run --gpus all -p 8000:8000 \
#     -v /data/hf-cache:/data/hf-cache \
#     -v /mnt/r2-voices:/samples \
#     -e API_KEY=your-secret-key \
#     -e HF_HOME=/data/hf-cache \
#     -e HF_TOKEN=hf_... \
#     vibeserver:latest
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "docs: document API_KEY env var in Dockerfile run example"
```
