# Design: Dynamic Weight Quantization via torchao

**Date:** 2026-04-07  
**Status:** Approved

## Summary

Add support for loading VibeVoice-7B weights in different numeric precisions, controlled by a single environment variable. The default behaviour (bfloat16) is unchanged. An `fp8` mode is added using torchao's post-load `quantize_()` API.

## Configuration

A new environment variable `VIBEVOICE_DTYPE` is read in `model.py` alongside the existing config constants:

| Value | Behaviour |
|-------|-----------|
| `bfloat16` | Default. Load weights as bfloat16. No quantization applied. |
| `fp8` | Load weights as bfloat16, then apply torchao float8 weight-only quantization in-place. |

**Validation:** Any value other than `bfloat16` or `fp8` causes the process to exit at startup with a clear error message. This is consistent with how startup errors are handled elsewhere (Kubernetes will restart the pod).

**Logging:** The resolved dtype is logged at startup alongside the existing model-loading messages so the chosen precision is always visible in container logs.

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

After the existing attention-implementation fallback loop succeeds and before `model.eval()`:

```python
_apply_quantization(model)
model.eval()
model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
```

The `from_pretrained()` call itself is unchanged — weights are always loaded as bfloat16 first.

## Data Flow

```
VIBEVOICE_DTYPE env var
        │
        ▼
model.py module import
  ├── validate DTYPE
  └── _load_model()
        ├── from_pretrained(..., torch_dtype=bfloat16)  [unchanged]
        ├── _apply_quantization(model)
        │     ├── bfloat16 → no-op
        │     └── fp8 → torchao.quantization.quantize_(model, Float8WeightOnlyConfig())
        ├── model.eval()
        └── model.set_ddpm_inference_steps()
```

## Error Handling

- **Invalid DTYPE at startup:** `RuntimeError` raised at module import → process exits non-zero → Kubernetes restarts.
- **torchao unavailable (fp8 mode):** `ImportError` from the late import inside `_apply_quantization()` → propagates as a startup error with a clear traceback.
- **quantize_() failure:** Exception propagates out of `_load_model()` → process exits non-zero.

## Testing

Existing tests mock `model._load_model` so they are unaffected by default. New unit tests cover:

1. `VIBEVOICE_DTYPE=bfloat16` → `_apply_quantization` is a no-op (torchao never imported).
2. `VIBEVOICE_DTYPE=fp8` → `quantize_` is called with `Float8WeightOnlyConfig()`.
3. Invalid dtype → `RuntimeError` raised at validation.

Tests mock `torchao.quantization.quantize_` to avoid requiring GPU or torchao in CI.

## Out of Scope

- int8 / int4 / other formats (can be added later by extending `_SUPPORTED_DTYPES` and `_apply_quantization`).
- Per-request dtype switching (model is loaded once; switching requires a process restart).
- Hot-reload endpoint.
- Dockerfile changes (torchao ships with PyTorch 2.6 in the base image; no new pip install needed).
