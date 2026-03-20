# API Key Authentication — Design Spec

**Date:** 2026-03-20  
**Status:** Approved  

---

## Overview

Add `X-Api-Key` header authentication to the VibeServer FastAPI application. All business endpoints require a valid API key; the `/health` liveness probe remains unauthenticated.

---

## Scope

Changes are confined to a single file: `app/app.py`.  
No new files, no new dependencies beyond the Python standard library and FastAPI (already in use).

---

## Environment Variable

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | The secret key value clients must send in the `X-Api-Key` header. |

The server **must refuse to start** if `API_KEY` is absent or empty. This is enforced at module load time with a `RuntimeError`, producing a clear error message before any network traffic is accepted.

---

## Security Dependency

A module-level `APIKeyHeader(name="X-Api-Key", auto_error=False)` scheme is registered. This causes the header to appear in the OpenAPI spec so that the Swagger UI renders an "Authorize" button — users can enter the key and test endpoints directly from the docs.

An async `verify_api_key` function serves as the FastAPI dependency:

- Accepts the extracted header value (or `None` if absent).
- Compares it to the loaded `API_KEY` using `secrets.compare_digest` to prevent timing attacks.
- Raises `HTTPException(status_code=401, detail="Invalid or missing API key")` on failure.
- Returns `None` on success (no return value needed by callers).

---

## Protected Endpoints

The following 4 endpoints get `dependencies=[Depends(verify_api_key)]` added to their route decorators:

| Method | Path |
|---|---|
| `POST` | `/v1/audio/speech` |
| `GET` | `/v1/models` |
| `GET` | `/v1/voices` |
| `POST` | `/v1/voices/refresh` |

---

## Unprotected Endpoint

| Method | Path | Reason |
|---|---|---|
| `GET` | `/health` | Kubernetes liveness/readiness probe — must remain unauthenticated. |

---

## Error Response

On missing or invalid key:

```
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{"detail": "Invalid or missing API key"}
```

---

## Dockerfile / Deployment Notes

The `API_KEY` env var must be supplied at runtime alongside existing vars (`HF_HOME`, `HF_TOKEN`, etc.):

```bash
docker run --gpus all -p 8000:8000 \
  -e API_KEY=your-secret-key \
  -e HF_HOME=/data/hf-cache \
  -e HF_TOKEN=hf_... \
  vibeserver:latest
```

---

## Out of Scope

- Multiple API keys / key rotation
- Key hashing / storage in a database
- Rate limiting per key
- JWT or OAuth flows
