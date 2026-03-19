# VibeServer

An OpenAI-compatible Text-to-Speech (TTS) HTTP server that wraps the
[VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) model.  Drop-in
replacement for the OpenAI TTS API (`/v1/audio/speech`).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Hardware Requirements](#hardware-requirements)
3. [Building the Docker Image](#building-the-docker-image)
4. [Running the Server](#running-the-server)
5. [Environment Variables](#environment-variables)
6. [Voice Presets](#voice-presets)
7. [API Reference](#api-reference)
8. [Integrating with OpenClaw](#integrating-with-openclaw)
9. [Known Limitations](#known-limitations)
10. [Responsible Use](#responsible-use)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Docker Container                                        │
│                                                          │
│  uvicorn :8000  (FastAPI)                                │
│    ├── POST /v1/audio/speech  ──► asyncio.Lock           │
│    │                                └► model.generate() │
│    ├── GET  /v1/models                                   │
│    └── GET  /health                                      │
│                                                          │
│  GPU (BF16 — ~18 GB VRAM)                                │
│    └── VibeVoice-7B (LLM + diffusion head)              │
│                                                          │
│  /app/voices/   (bundled WAV presets)                    │
└─────────────────────────────────────────────────────────┘
```

A single uvicorn process owns the model.  Requests are serialised via an
`asyncio.Lock` — only one TTS inference runs at a time.  Concurrent requests
queue in the event loop rather than being rejected.

The model weights are **not** baked into the image.  They are downloaded from
HuggingFace Hub on first startup and cached at `/root/.cache/huggingface`.
Mount a persistent volume at that path to avoid re-downloading on every
container start.

---

## Hardware Requirements

| Property | Required |
|---|---|
| GPU architecture | NVIDIA Blackwell (GB10 / sm_100) or later |
| GPU memory | ≥ 20 GB (BF16 weights ~18 GB + activations) |
| CPU architecture | ARM64 (aarch64) |
| CUDA | 12.8+ |
| Host OS | Linux with NVIDIA Container Toolkit |

The server is built and tested on the **ASUS Ascent GX10** (NVIDIA GB10 Grace
Blackwell Superchip, 128 GB unified memory).

---

## Building the Docker Image

```bash
# On an ARM64 host or via buildx cross-compilation:
docker buildx build \
  --platform linux/arm64 \
  -t vibeserver:latest \
  .
```

> **Note:** The build pulls the `nvcr.io/nvidia/pytorch:26.02-py3` base image
> (~10 GB) and installs the `vibevoice` Python library from GitHub.  Expect
> the first build to take 15–30 minutes depending on network speed.

---

## Running the Server

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v /data/hf-cache:/data/hf-cache \
  -e HF_HOME=/data/hf-cache \
  -e HF_TOKEN=hf_YOUR_TOKEN_HERE \
  vibeserver:latest
```

On first startup the model weights (~18 GB) are downloaded from HuggingFace
Hub.  Subsequent starts use the cached weights.  Watch the logs:

```
INFO  Initialising VibeVoice model (this may take a few minutes) …
INFO  Loading VibeVoiceProcessor from vibevoice/VibeVoice-7B …
INFO  Loading VibeVoiceForConditionalGenerationInference (BF16, CUDA) …
INFO  Loaded with attn_implementation=flash_attention_2
INFO  Model ready.
INFO  Application startup complete.
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(none)* | HuggingFace API token. Required if the model repo is gated. |
| `HF_HOME` | `~/.cache/huggingface` | Base directory for all HuggingFace data (cache, tokens, etc.). **Recommended** — mount a persistent volume at this path. Model weights land at `$HF_HOME/hub`. |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | Override only the model/dataset cache path. Takes precedence over `HF_HOME` if both are set. |
| `VIBEVOICE_MODEL_ID` | `vibevoice/VibeVoice-7B` | HuggingFace model ID to load. |
| `VIBEVOICE_CFG_SCALE` | `1.3` | Classifier-Free Guidance scale. Higher = more faithful to voice prompt; lower = more varied. |
| `VIBEVOICE_DDPM_STEPS` | `10` | Diffusion inference steps. More steps = higher quality, slower generation. |
| `PORT` | `8000` | Port uvicorn listens on. |

---

## Voice Presets

VibeVoice uses short WAV audio samples to clone a speaker's voice.  Six presets
are bundled in the image under `/app/voices/`, mapped to OpenAI voice names:

| OpenAI voice | WAV file | Character |
|---|---|---|
| `alloy` | `Alice.wav` | Female, neutral American English |
| `echo` | `Echo.wav` | Male, neutral American English |
| `fable` | `Frank.wav` | Male, British English |
| `onyx` | `Onyx.wav` | Male, deep American English |
| `nova` | `Nova.wav` | Female, warm American English |
| `shimmer` | `Shimmer.wav` | Female, soft American English |

### Replacing Voice Presets

The repository ships **silent placeholder WAV files**.  For best quality,
replace them with real 5–30 second clean speech samples before building the
image:

1. Record or source a clean WAV sample (24 kHz, mono or stereo, no background
   noise or music).
2. Name it after the voice you want to replace (e.g. `Alice.wav`).
3. Place it in `app/voices/`.
4. Rebuild the Docker image.

> **Important:** Only use voice samples for which you have the right to use the
> speaker's voice.  Do not use recordings of real people without their explicit
> consent.  See [Responsible Use](#responsible-use).

---

## API Reference

### `POST /v1/audio/speech`

Synthesise text into speech.  Compatible with the
[OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body (JSON):**

```json
{
  "model": "vibevoice-7b",
  "input": "Hello, world! This is VibeVoice speaking.",
  "voice": "alloy"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | Must be `"vibevoice-7b"` |
| `input` | string | yes | Text to synthesise |
| `voice` | string | no | `alloy` (default), `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `response_format` | string | no | Only `"wav"` supported; other values accepted but ignored |
| `speed` | float | no | Not supported; ignored |

**Response:**

- `200 OK` — `audio/wav` binary body
- `422 Unprocessable Entity` — validation error (missing `input`, wrong `model`)
- `500 Internal Server Error` — `{"detail": "..."}` on inference failure

**curl example:**

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"vibevoice-7b","input":"Hello from VibeVoice!","voice":"nova"}' \
  --output speech.wav
```

---

### `GET /v1/models`

Returns the available models in OpenAI list format.

```bash
curl http://localhost:8000/v1/models
```

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

### `GET /health`

Liveness probe for Kubernetes / load balancers.

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Integrating with OpenClaw

In your OpenClaw configuration, point the TTS base URL at the server:

```
TTS Base URL:  http://<server-ip>:8000
TTS Model:     vibevoice-7b
TTS Voice:     alloy   # or any supported voice name
```

OpenClaw will call `POST /v1/audio/speech` using the standard OpenAI request
format, which VibeServer handles natively.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **No concurrent inference** | Only one TTS request runs at a time. Additional requests queue; they do not receive 503 errors. |
| **Language** | English and Chinese only. Other languages produce undefined/poor output. |
| **No streaming audio** | The full WAV file is buffered before the response is sent. Long inputs take proportionally longer. |
| **Max context** | VibeVoice-7B supports up to 32K tokens (~45 minutes of generated audio). Very long `input` strings may be truncated by the model. |
| **Voice placeholders** | Shipped WAV files are silent placeholders. Replace them with real voice samples before deployment. |
| **Inference library** | Depends on the community fork [`vibevoice-community/VibeVoice`](https://github.com/vibevoice-community/VibeVoice). The official Microsoft repo removed TTS code in September 2025. |
| **Blackwell sm_100** | `flash_attention_2` support for the GB10 Blackwell GPU is assumed in the NGC 26.02 image. The server falls back to `sdpa` automatically if it fails. |

---

## Responsible Use

VibeVoice is licensed under MIT, but the upstream maintainers (Microsoft
Research) recommend **research use only**.  When deploying this server:

- Do **not** use it to impersonate real people without their explicit, recorded
  consent.
- Do **not** use it to generate disinformation or fake audio presented as
  genuine recordings.
- Do **not** expose the server publicly without authentication.
- Disclose when audio was AI-generated.
- Comply with all applicable laws and regulations in your jurisdiction.

See the [VibeVoice model card](https://huggingface.co/vibevoice/VibeVoice-7B)
and [Microsoft's responsible AI principles](https://www.microsoft.com/en-us/ai/responsible-ai)
for further guidance.
