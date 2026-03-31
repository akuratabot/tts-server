"""tests/test_auth.py — API key authentication tests.

Note: `patch_model` (autouse) patches sys.modules["model"] *before* make_client()
calls importlib.reload(). This ordering is critical: the reloaded app module must
bind its `_model` name against the mock, not the real ML library. Do not change
the fixture/helper ordering without understanding this dependency.
"""
import os
import importlib
import sys
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-secret-key"


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

    # Pre-load app.app with a valid key so that `import app.app` in
    # test_missing_api_key_raises_at_import is always a no-op (module cached),
    # letting the reload-without-key inside pytest.raises actually trigger the guard.
    with patch.dict(os.environ, {"TTS_API_KEY": "fixture-preload-key"}):
        import app.app as _app_module  # noqa: PLC0415
        importlib.reload(_app_module)


def make_client(api_key: str = VALID_KEY) -> TestClient:
    """Import app with API_KEY set and return a TestClient."""
    # Patch env before reloading so the module-level startup guard sees the key.
    with patch.dict(os.environ, {"TTS_API_KEY": api_key}):
        import app.app as app_module  # noqa: PLC0415
        importlib.reload(app_module)
        return TestClient(app_module.app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Startup guard
# --------------------------------------------------------------------------- #

def test_missing_api_key_raises_at_import():
    """Server must refuse to start when API_KEY is absent."""
    env_without_key = {k: v for k, v in os.environ.items() if k != "TTS_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        import app.app as app_module  # noqa: PLC0415
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


# --------------------------------------------------------------------------- #
# Authorization: Bearer — valid token → not 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("GET",  "/v1/models",  None),
    ("GET",  "/v1/voices",  None),
    ("POST", "/v1/voices/refresh", None),
])
def test_bearer_token_passes_auth(method, path, body):
    client = make_client()
    resp = client.request(
        method, path, json=body,
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert resp.status_code != 401


# --------------------------------------------------------------------------- #
# Authorization: Bearer — wrong token → 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("GET",  "/v1/models",  None),
    ("GET",  "/v1/voices",  None),
    ("POST", "/v1/voices/refresh", None),
])
def test_wrong_bearer_token_returns_401(method, path, body):
    client = make_client()
    resp = client.request(
        method, path, json=body,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Authorization: Bearer — malformed (no token after scheme) → 401
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,body", [
    ("GET",  "/v1/models",  None),
])
def test_malformed_bearer_returns_401(method, path, body):
    client = make_client()
    # "Bearer " with no token, and "NotBearer key" (wrong scheme)
    for auth_value in ("Bearer ", "Token some-key", "notascheme"):
        resp = client.request(
            method, path, json=body,
            headers={"Authorization": auth_value},
        )
        assert resp.status_code == 401, f"Expected 401 for Authorization: {auth_value!r}"


# --------------------------------------------------------------------------- #
# X-Api-Key takes precedence over Authorization: Bearer
# --------------------------------------------------------------------------- #

def test_x_api_key_takes_precedence_over_bearer():
    """When both headers are present, X-Api-Key is used (valid key wins)."""
    client = make_client()
    resp = client.get(
        "/v1/models",
        headers={
            "X-Api-Key": VALID_KEY,
            "Authorization": "Bearer wrong-token",
        },
    )
    assert resp.status_code != 401
