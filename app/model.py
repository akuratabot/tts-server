"""
model.py — VibeVoice-7B model loading and inference wrapper.

Responsibilities:
  - Sync external voice files from VIBEVOICE_EXTRA_VOICES_DIR into the bundled
    app/voices/ directory at startup, so all voices live on local disk.
  - Load VibeVoiceForConditionalGenerationInference + VibeVoiceProcessor once
    at startup.
  - Discover available voices by scanning app/voices/ after the sync.
  - Expose a single generate_speech() function that takes plain-text input and
    returns raw WAV bytes.
  - Provide an asyncio.Lock so the FastAPI layer can serialise requests.

Startup order (all synchronous, before the model loads):
  1. sync_voices()   — copy from VIBEVOICE_EXTRA_VOICES_DIR → VOICES_DIR
  2. _build_voice_index() — scan VOICES_DIR to build the name → Path index

Any startup error will cause the process to exit non-zero (Kubernetes restarts).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import traceback
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#  Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------- #

MODEL_ID: str = os.getenv("VIBEVOICE_MODEL_ID", "vibevoice/VibeVoice-7B")
CFG_SCALE: float = float(os.getenv("VIBEVOICE_CFG_SCALE", "1.3"))
DDPM_STEPS: int = int(os.getenv("VIBEVOICE_DDPM_STEPS", "10"))

# ---------------------------------------------------------------------------- #
#  Voice sync + mapping
# ---------------------------------------------------------------------------- #

VOICES_DIR = Path(__file__).parent / "voices"

# Extra voices directory — mount a volume here (e.g. backed by Cloudflare R2).
# Files found here are copied into VOICES_DIR at startup.
# Mounted files overwrite bundled files of the same name.
EXTRA_VOICES_DIR = Path(os.getenv("VIBEVOICE_EXTRA_VOICES_DIR", "/samples"))

# Supported audio extensions (what VibeVoice's audio processor can load).
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def sync_voices() -> None:
    """
    Copy all audio files from EXTRA_VOICES_DIR into VOICES_DIR.

    - Runs once at module import, before the voice index is built.
    - EXTRA_VOICES_DIR not existing is silently ignored (mount is optional).
    - Files in EXTRA_VOICES_DIR overwrite same-named files in VOICES_DIR so
      that operator-supplied voices replace bundled placeholders.
    - Non-audio files (e.g. PLACEHOLDER.md) are skipped.
    """
    if not EXTRA_VOICES_DIR.exists():
        logger.info(
            "Extra voices directory %s not found — skipping external voice sync.",
            EXTRA_VOICES_DIR,
        )
        return

    if not EXTRA_VOICES_DIR.is_dir():
        logger.warning(
            "VIBEVOICE_EXTRA_VOICES_DIR=%s exists but is not a directory — skipping.",
            EXTRA_VOICES_DIR,
        )
        return

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for src in sorted(EXTRA_VOICES_DIR.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() not in _AUDIO_EXTS:
            logger.debug("Skipping non-audio file: %s", src.name)
            skipped += 1
            continue

        dst = VOICES_DIR / src.name
        if dst.exists():
            logger.info(
                "Overwriting bundled voice %s with external file %s", dst.name, src
            )
        else:
            logger.info("Copying external voice %s → %s", src.name, dst)

        shutil.copy2(src, dst)
        copied += 1

    logger.info(
        "Voice sync complete: %d copied, %d non-audio skipped (source: %s)",
        copied,
        skipped,
        EXTRA_VOICES_DIR,
    )


def _build_voice_index() -> dict[str, Path]:
    """
    Scan VOICES_DIR and return a mapping of lowercase voice name → Path.

    The voice name is the filename stem, lowercased.  For example:
        Alice.wav    → "alice"
        my-voice.wav → "my-voice"

    Called once at module import, after sync_voices().
    """
    index: dict[str, Path] = {}
    if not VOICES_DIR.is_dir():
        logger.warning("Voices directory not found: %s", VOICES_DIR)
        return index
    for p in sorted(VOICES_DIR.iterdir()):
        if p.suffix.lower() in _AUDIO_EXTS:
            index[p.stem.lower()] = p
    logger.info("Registered %d voice(s): %s", len(index), sorted(index))
    return index


# --- Startup sequence ---
logger.info("Starting voice sync …")
sync_voices()

# Built once at startup after sync; all voices are on local disk at this point.
_VOICE_INDEX: dict[str, Path] = _build_voice_index()


def available_voices() -> list[str]:
    """Return a sorted list of available voice names (lowercase stems)."""
    return sorted(_VOICE_INDEX)


def resolve_voice_path(voice: str) -> Path | None:
    """
    Return the absolute path to the WAV file for *voice*.

    Matching is case-insensitive against the filename stem.  If the requested
    voice is not found, falls back to the first available voice alphabetically.
    Returns None only if the voices directory is completely empty (triggers
    prefill-less generation as a last resort).
    """
    name = (voice or "").strip().lower()
    path = _VOICE_INDEX.get(name)

    if path is None:
        fallback = next(iter(sorted(_VOICE_INDEX)), None)
        if fallback is None:
            logger.error(
                "No voice WAV files found in %s — running without voice cloning.", VOICES_DIR
            )
            return None
        logger.warning(
            "Voice %r not found — falling back to %r.", voice, fallback
        )
        path = _VOICE_INDEX[fallback]

    return path


# ---------------------------------------------------------------------------- #
#  Model loading
# ---------------------------------------------------------------------------- #

def _load_model():
    """Load processor and model onto GPU (BF16).  Called once at module import."""
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    logger.info("Loading VibeVoiceProcessor from %s …", MODEL_ID)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)

    logger.info(
        "Loading VibeVoiceForConditionalGenerationInference (BF16, CUDA) …"
    )

    # Try flash_attention_2 first (optimal on Blackwell); fall back to sdpa.
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
            logger.info("Loaded with attn_implementation=%s", attn_impl)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "attn_implementation=%s failed (%s: %s); trying next …",
                attn_impl,
                type(exc).__name__,
                exc,
            )
    else:
        raise RuntimeError(
            "Could not load VibeVoice model with any supported attention implementation."
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
    logger.info("Model ready.")
    return processor, model


logger.info("Initialising VibeVoice model (this may take a few minutes) …")
_processor, _model = _load_model()

# One inference at a time — VibeVoice does not support batching and holding the
# GPU across concurrent requests would corrupt outputs.
inference_lock = asyncio.Lock()


# ---------------------------------------------------------------------------- #
#  Public API
# ---------------------------------------------------------------------------- #

def generate_speech(text: str, voice: str = "") -> bytes:
    """
    Synthesise *text* with the requested *voice* and return raw WAV bytes.

    This is a synchronous, blocking function.  Callers must hold
    *inference_lock* before calling it and release it afterwards — or use
    asyncio.to_thread() with the lock held in the async layer.

    Args:
        text:  Plain-text input to synthesise.  Single-speaker only.
        voice: OpenAI voice name ('alloy', 'echo', 'fable', 'onyx', 'nova',
               'shimmer').  Defaults to 'alloy'.

    Returns:
        Raw WAV audio bytes (24 kHz).

    Raises:
        RuntimeError: if the model produces no speech output.
    """
    voice_path = resolve_voice_path(voice)
    disable_prefill = voice_path is None

    # VibeVoice expects the "Speaker N: text" format.
    script = f"Speaker 1: {text}\n"

    voice_samples = [str(voice_path)] if not disable_prefill else []

    inputs = _processor(
        text=[script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move all tensors to GPU.
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    logger.info(
        "Generating speech: voice=%r, disable_prefill=%s, input_len=%d chars",
        voice,
        disable_prefill,
        len(text),
    )

    outputs = _model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=CFG_SCALE,
        tokenizer=_processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=not disable_prefill,
    )

    speech = outputs.speech_outputs[0] if outputs.speech_outputs else None
    if speech is None:
        raise RuntimeError("VibeVoice returned no speech output for the given input.")

    # Serialise to WAV in memory.
    #
    # processor.save_audio() delegates to soundfile.write(output_path, ...) without
    # ever passing format=, so soundfile infers the format from the file extension.
    # A BytesIO object has no extension → TypeError.  We write the WAV ourselves,
    # passing format='WAV' explicitly so soundfile works correctly with a BytesIO.
    import soundfile as sf
    import numpy as np

    # Normalise to a 1-D float32 numpy array regardless of what the model returns.
    if isinstance(speech, torch.Tensor):
        audio_np = speech.float().detach().cpu().numpy()
    else:
        audio_np = np.array(speech, dtype=np.float32)

    # Squeeze any leading batch / channel dimensions down to (T,).
    audio_np = audio_np.squeeze()
    if audio_np.ndim != 1:
        raise RuntimeError(
            f"Unexpected audio shape after squeeze: {audio_np.shape}. "
            "Expected a 1-D array."
        )

    sample_rate = getattr(_processor.audio_processor, "sampling_rate", 24000)

    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
