# VibeServer Design Spec

**Date:** 2026-03-19  
**Status:** Approved  
**Author:** OpenCode

---

## 1. Purpose

Build a containerized OpenAI-compatible TTS HTTP server that wraps the `vibevoice/VibeVoice-7B` model. The server will be deployed on an ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell Superchip, ARM64, 128 GB unified memory) as a container in a Kubernetes cluster.

The primary consumer is **OpenClaw**, which expects OpenAI-compatible TTS endpoints.

---

## 2. Scope

### In scope
- `POST /v1/audio/speech` — OpenAI-compatible TTS endpoint returning WAV audio
- `GET /v1/models` — Returns the model list in OpenAI format
- Dockerfile targeting `nvcr.io/nvidia/pytorch:26.02-py3` (ARM64/aarch64)
- `requirements.txt` listing all Python dependencies
- 6 bundled voice preset WAV files mapped to OpenAI voice names
- Full documentation (`README.md`)

### Out of scope
- docker-compose / Kubernetes manifests (handled outside the repo)
- Streaming/chunked audio responses
- Concurrent request batching
- Fine-tuning or ASR functionality
- Real-time / low-latency (Streaming 0.5B) model variant
- Voice upload / custom voice cloning at runtime

---

## 3. Hardware Target

| Property | Value |
|---|---|
| Device | ASUS Ascent GX10 |
| Chip | NVIDIA GB10 Grace Blackwell Superchip |
| CPU arch | ARM64 (aarch64) |
| GPU arch | Blackwell (sm_100) |
| Unified memory | 128 GB |
| CUDA | 12.8+ |

The 128 GB unified memory pool means the 7B model in BF16 (~18 GB) fits comfortably with large headroom for long-context generation.

---

## 4. Model

| Property | Value |
|---|---|
| HuggingFace ID | `vibevoice/VibeVoice-7B` |
| Parameters | ~9B (7B LLM + tokenizer/diffusion head) |
| Precision | BF16 |
| Context length | 32K tokens |
| Max generation | ~45 minutes of audio |
| Speakers | Up to 4 simultaneous |
| Languages | English, Chinese |
| License | MIT |

**Inference library:** `vibevoice-community/VibeVoice` community fork, installed via pip at image build time from `git+https://github.com/vibevoice-community/VibeVoice.git`. This is the only available inference code; the official Microsoft repo removed TTS code in Sept 2025.

**Model weights:** Downloaded at first container startup from HuggingFace Hub into a host-mounted cache volume (`/root/.cache/huggingface`). Not baked into the image.

---

## 5. Architecture

### 5.1 Single-process FastAPI + uvicorn

One uvicorn process hosts the FastAPI application. The model is loaded once at startup. An `asyncio.Lock` ensures only one inference job runs at a time (VibeVoice does not support batching).

```
┌─────────────────────────────────────────────────────┐
│  Container                                           │
│                                                      │
│  uvicorn :8000                                       │
│    └── FastAPI app                                   │
│          ├── POST /v1/audio/speech                   │
│          │     └── asyncio.Lock → model.generate()  │
│          └── GET  /v1/models                         │
│                                                      │
│  GPU memory (BF16)                                   │
│    └── VibeVoiceForConditionalGenerationInference    │
│          + VibeVoiceProcessor                        │
│                                                      │
│  /app/voices/   (bundled preset WAVs)                │
└─────────────────────────────────────────────────────┘
```

### 5.2 Request lifecycle

1. Client sends `POST /v1/audio/speech`
2. Endpoint acquires `asyncio.Lock` (queues if another request is running)
3. `input` text is wrapped as `Speaker 1: <text>\n`
4. `voice` is mapped to a WAV file path from `/app/voices/`
5. `VibeVoiceProcessor` encodes text + voice sample
6. `model.generate()` runs on GPU (BF16, `flash_attention_2` preferred with automatic `sdpa` fallback if unavailable, cfg_scale=1.3)
7. `processor.save_audio()` writes to an in-memory `BytesIO` buffer
8. `StreamingResponse(audio/wav)` is returned to client
9. Lock is released

---

## 6. File Structure

```
vibeserver/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── app.py          # FastAPI application, endpoint definitions
│   ├── model.py        # Model loading, inference wrapper, voice mapping
│   └── voices/
│       ├── Alice.wav   # alloy
│       ├── Echo.wav    # echo
│       ├── Frank.wav   # fable
│       ├── Onyx.wav    # onyx
│       ├── Nova.wav    # nova
│       └── Shimmer.wav # shimmer
└── docs/
    └── README.md
```

---

## 7. API Specification

### 7.1 POST /v1/audio/speech

**Request body (JSON):**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Must be `"vibevoice-7b"` (validated, 422 if wrong) |
| `input` | string | yes | Text to synthesize |
| `voice` | string | no | One of: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`. Defaults to `alloy`. Unknown values fall back to `alloy` with a warning log. |
| `response_format` | string | no | Only `"wav"` is supported. Ignored if provided. |
| `speed` | float | no | Ignored (not supported by VibeVoice). |

**Response:**
- Status: `200 OK`
- Content-Type: `audio/wav`
- Body: WAV audio binary (24 kHz, mono)

**Error responses:**
- `422 Unprocessable Entity` — missing `input`
- `500 Internal Server Error` — JSON body `{ "detail": "<message>" }`

### 7.2 GET /v1/models

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "vibevoice-7b",
      "object": "model",
      "created": 1724630400,
      "owned_by": "vibevoice-community"
    }
  ]
}
```

---

## 8. Voice Mapping

| OpenAI voice name | Bundled WAV file | Character |
|---|---|---|
| `alloy` | `Alice.wav` | Female, neutral American English |
| `echo` | `Echo.wav` | Male, neutral American English |
| `fable` | `Frank.wav` | Male, British English |
| `onyx` | `Onyx.wav` | Male, deep American English |
| `nova` | `Nova.wav` | Female, warm American English |
| `shimmer` | `Shimmer.wav` | Female, soft American English |

WAV files must be clean speech samples (no background music) at 24 kHz, mono or stereo, minimum 5 seconds. The repository includes placeholder silent WAV files; the operator must replace them with real voice samples before deployment. Instructions are in `docs/README.md`.

---

## 9. Dockerfile

```
Base image:  nvcr.io/nvidia/pytorch:26.02-py3  (aarch64)
Platform:    linux/arm64

Steps:
1. Set WORKDIR /app
2. Set env: PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1
3. Install system deps: git, libsndfile1
4. COPY requirements.txt → pip install
5. Install vibevoice library: pip install git+https://github.com/vibevoice-community/VibeVoice.git
6. COPY app/ → /app/
7. EXPOSE 8000
8. ENTRYPOINT uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Environment variables (runtime)

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | If model is gated | — | HuggingFace API token for model download |
| `HF_HUB_CACHE` | No | `/root/.cache/huggingface` | Override HF cache path |
| `VIBEVOICE_CFG_SCALE` | No | `1.3` | Classifier-Free Guidance scale |
| `VIBEVOICE_DDPM_STEPS` | No | `10` | DDPM inference steps (quality vs speed) |
| `PORT` | No | `8000` | Listen port. The uvicorn entrypoint uses `${PORT:-8000}` so this variable is actively consumed. |

---

## 10. Dependencies (requirements.txt)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
soundfile>=0.12.1
scipy>=1.11.0
huggingface-hub>=0.23.0
python-multipart>=0.0.9
```

Note: `torch`, `transformers`, `accelerate`, `diffusers`, `tqdm`, `numpy`, `librosa`, `scipy` etc. come from either the NGC base image or the `vibevoice` package's own dependencies.

---

## 11. Error Handling

| Scenario | Behavior |
|---|---|
| Unknown `voice` value | Log warning, fall back to `alloy` (Alice.wav) |
| Missing `input` field | 422 from FastAPI Pydantic validation |
| Inference exception | Catch, log traceback, return 500 with `{"detail": "..."}` |
| Model not loaded (startup failure) | Container exits non-zero; Kubernetes restarts it |
| Another request in progress | Second request waits on `asyncio.Lock`, no 503 |
| Voice WAV file missing | Log error, attempt with `disable_prefill=True` as fallback |

---

## 12. Documentation

`docs/README.md` will cover:
1. Overview and architecture
2. Hardware requirements (GB10, ARM64, CUDA 12.8+)
3. Building the Docker image
4. Running locally (with `docker run` example, GPU passthrough flags)
5. Environment variables reference
6. How to replace voice preset WAV files
7. API reference (both endpoints with curl examples)
8. Voice mapping table
9. Known limitations (no concurrency, English/Chinese only, no live streaming)
10. Model card / responsible use notice

---

## 13. Known Limitations

- **No concurrency:** Only one inference at a time. Additional requests queue in uvicorn's event loop.
- **Language:** English and Chinese only. Other languages produce undefined output.
- **Long inputs:** Very long `input` text may exceed the 32K context limit; the model will truncate.
- **No streaming audio:** Full WAV is buffered before response.
- **Voice WAV placeholders:** Shipped WAV files are silent placeholders; must be replaced by operator.
- **VibeVoice TTS code removed by Microsoft:** We depend on the community fork. If that fork becomes unavailable, pin to a specific commit in the Dockerfile.
- **Blackwell (sm_100) compatibility:** `flash_attention_2` support for sm_100 in the PyTorch 26.02 NGC image is assumed but should be verified on first deployment. The code falls back to `sdpa` automatically if `flash_attention_2` fails.
