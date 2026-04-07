# Design: Dynamic Weight Quantization via torchao

**Date:** 2026-04-07  
**Status:** Approved  
**Revision:** 2 — CPU-staged loading for memory-constrained GPU nodes

## Summary

Add support for loading VibeVoice-7B weights in different numeric precisions, controlled by a single environment variable. The default behaviour (bfloat16) is unchanged. An `fp8` mode is added using torchao's post-load `quantize_()` API.

When `fp8` is selected, the model is loaded to CPU memory first, quantized on CPU, then moved to GPU. This allows nodes with limited GPU memory (e.g. 12 GB) to run a 7B-parameter model that would otherwise require ~14 GB in bfloat16. The quantized fp8 weights occupy ~7 GB on the GPU.

## Configuration

A new environment variable `VIBEVOICE_DTYPE` is read in `model.py` alongside the existing config constants:

| Value | Behaviour |
|-------|-----------|
| `bfloat16` | Default. Load weights directly to GPU as bfloat16. No quantization. |
| `fp8` | Load weights to CPU as bfloat16, apply torchao float8 weight-only quantization on CPU, then move the quantized model to GPU. |

**Validation:** Any value other than `bfloat16` or `fp8` causes the process to exit at startup with a clear error message. This is consistent with how startup errors are handled elsewhere (Kubernetes will restart the pod).

**Logging:** The resolved dtype and load strategy (CPU-staged vs direct GPU) are logged at startup so the chosen path is always visible in container logs.

## Architecture

The change is isolated entirely to `model.py`. No other files (`app.py`, `jobs.py`) are touched.

### New constant

```python
DTYPE: str = os.getenv("VIBEVOICE_DTYPE", "bfloat16")
```

Validated at module import time:

```python
_SUPPORTED_DTYPES = {"bfloat16", "fp8"}
if DTYPE not in _SUPPORTED_DTYPES:
    raise RuntimeError(
        f"Unsupported VIBEVOICE_DTYPE={DTYPE!r}. "
        f"Supported values: {sorted(_SUPPORTED_DTYPES)}"
    )
```

### New helper: `_apply_quantization(model)`

A private function called from `_load_model()` after `from_pretrained()` succeeds:

```python
def _apply_quantization(model) -> None:
    """Apply torchao post-load quantization based on the DTYPE env var."""
    if DTYPE == "bfloat16":
        return  # nothing to do

    from torchao.quantization import quantize_, Float8WeightOnlyConfig

    logger.info("Applying torchao quantization: dtype=%s …", DTYPE)
    t0 = time.perf_counter()
    if DTYPE == "fp8":
        quantize_(model, Float8WeightOnlyConfig())
    elapsed = time.perf_counter() - t0
    logger.info("Quantization complete in %.1f s.", elapsed)
```

### Changes to `_load_model()`

The `device_map` argument to `from_pretrained()` depends on DTYPE:

- `bfloat16`: `device_map="cuda"` (unchanged — loads directly to GPU).
- `fp8`: `device_map="cpu"` (loads to CPU for quantization staging).

After `_apply_quantization(model)` returns, when DTYPE is `fp8` the model is moved to GPU with `model.to("cuda")`.

```python
_load_device = "cpu" if DTYPE == "fp8" else "cuda"

# inside _load_model():
model = ...from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=_load_device,
    attn_implementation=attn_impl,
)

_apply_quantization(model)

if _load_device == "cpu":
    logger.info("Moving quantized model to CUDA …")
    t0 = time.perf_counter()
    model.to("cuda")
    elapsed = time.perf_counter() - t0
    logger.info("Model moved to CUDA in %.1f s.", elapsed)

model.eval()
model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
```

## Data Flow

```
bfloat16 path (unchanged):
  from_pretrained(device_map="cuda", torch_dtype=bfloat16)
         │
         ▼
  model.eval() → ready on GPU

fp8 path (CPU-staged):
  from_pretrained(device_map="cpu", torch_dtype=bfloat16)   ~14 GB CPU RAM
         │
         ▼
  quantize_(model, Float8WeightOnlyConfig())                 ~7 GB CPU RAM (in-place)
         │
         ▼
  model.to("cuda")                                           ~7 GB GPU VRAM
         │
         ▼
  model.eval() → ready on GPU
```

## Error Handling

- **Invalid DTYPE at startup:** `RuntimeError` raised at module import → process exits non-zero → Kubernetes restarts.
- **torchao unavailable (fp8 mode):** `ImportError` from the late import inside `_apply_quantization()` → propagates as a startup error with a clear traceback.
- **quantize_() failure:** Exception propagates out of `_load_model()` → process exits non-zero.
- **model.to("cuda") failure (e.g. OOM):** Exception propagates out of `_load_model()` → process exits non-zero.

## Testing

Existing tests mock `model._load_model` so they are unaffected by default. New unit tests cover:

1. `VIBEVOICE_DTYPE=bfloat16` → `_apply_quantization` is a no-op (torchao never imported).
2. `VIBEVOICE_DTYPE=fp8` → `quantize_` is called with `Float8WeightOnlyConfig()`.
3. Invalid dtype → `RuntimeError` raised at validation.
4. `_load_device` is `"cpu"` when DTYPE is `fp8`, `"cuda"` when `bfloat16`.
5. `model.to("cuda")` is called after quantization in the fp8 path.

Tests mock `torchao.quantization.quantize_` to avoid requiring GPU or torchao in CI.

## Out of Scope

- int8 / int4 / other formats (can be added later by extending `_SUPPORTED_DTYPES` and `_apply_quantization`).
- Per-request dtype switching (model is loaded once; switching requires a process restart).
- Hot-reload endpoint.
- Dockerfile changes (torchao ships with PyTorch 2.6 in the base image; no new pip install needed).
- Configurable load device (fp8 always stages through CPU; there's no use case for direct-GPU fp8 on nodes that already have enough VRAM, since CPU staging has negligible one-time startup cost).
