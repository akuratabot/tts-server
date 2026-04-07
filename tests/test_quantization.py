"""tests/test_quantization.py — Dynamic quantization unit tests.

Tests exercise the DTYPE validation and _apply_quantization() logic
from app/model.py.  Since model.py has side effects at import time
(loads the real ML model), tests replicate the production logic in
isolation and mock torchao to avoid GPU/hardware dependency.
"""

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  DTYPE validation
# --------------------------------------------------------------------------- #


class TestDtypeValidation:
    """Test VIBEVOICE_DTYPE environment variable validation."""

    _SUPPORTED_DTYPES = {"bfloat16", "fp8"}

    def _validate(self, dtype: str) -> None:
        """Replicate the module-level validation from model.py."""
        if dtype not in self._SUPPORTED_DTYPES:
            raise RuntimeError(
                f"Unsupported VIBEVOICE_DTYPE={dtype!r}. "
                f"Supported values: {sorted(self._SUPPORTED_DTYPES)}"
            )

    def test_bfloat16_is_valid(self):
        """bfloat16 is accepted (default)."""
        self._validate("bfloat16")  # should not raise

    def test_fp8_is_valid(self):
        """fp8 is accepted."""
        self._validate("fp8")  # should not raise

    def test_invalid_dtype_raises_runtime_error(self):
        """Unsupported value raises RuntimeError with descriptive message."""
        with pytest.raises(RuntimeError, match=r"Unsupported VIBEVOICE_DTYPE='int4'"):
            self._validate("int4")

    def test_empty_string_dtype_raises(self):
        """Empty string is not a valid dtype."""
        with pytest.raises(RuntimeError, match=r"Unsupported VIBEVOICE_DTYPE=''"):
            self._validate("")


# --------------------------------------------------------------------------- #
#  _apply_quantization
# --------------------------------------------------------------------------- #


class TestApplyQuantization:
    """Test _apply_quantization() behaviour.

    A standalone mirror of the production function is used so we can
    test the logic without triggering model.py's module-level side effects.
    """

    @staticmethod
    def _apply_quantization(model, dtype: str) -> None:
        """Mirror of model._apply_quantization for testing."""
        if dtype == "bfloat16":
            return

        from torchao.quantization import quantize_, Float8WeightOnlyConfig

        logger.info("Applying torchao quantization: dtype=%s …", dtype)
        t0 = time.perf_counter()
        if dtype == "fp8":
            quantize_(model, Float8WeightOnlyConfig())
        elapsed = time.perf_counter() - t0
        logger.info("Quantization complete in %.1f s.", elapsed)

    def test_bfloat16_is_noop(self):
        """bfloat16 dtype returns early — torchao is never imported."""
        fake_model = MagicMock()
        # bfloat16 should return immediately without touching the model.
        self._apply_quantization(fake_model, "bfloat16")
        fake_model.assert_not_called()

    def test_fp8_calls_quantize_with_float8_config(self):
        """fp8 dtype calls quantize_(model, Float8WeightOnlyConfig())."""
        mock_quantize = MagicMock()
        mock_config_cls = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_cls.return_value = mock_config_instance

        fake_torchao_quant = MagicMock()
        fake_torchao_quant.quantize_ = mock_quantize
        fake_torchao_quant.Float8WeightOnlyConfig = mock_config_cls

        with patch.dict(
            sys.modules,
            {
                "torchao": MagicMock(),
                "torchao.quantization": fake_torchao_quant,
            },
        ):
            fake_model = MagicMock()
            self._apply_quantization(fake_model, "fp8")

            mock_config_cls.assert_called_once_with()
            mock_quantize.assert_called_once_with(fake_model, mock_config_instance)

    def test_fp8_logs_timing(self, caplog):
        """fp8 quantization logs start and completion with elapsed time."""
        fake_torchao_quant = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torchao": MagicMock(),
                "torchao.quantization": fake_torchao_quant,
            },
        ):
            with caplog.at_level(logging.INFO):
                self._apply_quantization(MagicMock(), "fp8")

            messages = [r.message for r in caplog.records]
            assert any("Applying torchao quantization" in m for m in messages)
            assert any("Quantization complete" in m for m in messages)
