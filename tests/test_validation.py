"""Tests for the guards added during review: config validation, device handling,
left-censoring helpers, duplicate-pair removal and SMILES robustness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_ranking.chemistry import (  # noqa: E402
    apply_censor_floor,
    censor_floor,
    cyclize_polymer_with_cp_marking,
)
from polymer_ranking.config import (  # noqa: E402
    DEVICE_MODES,
    SPLIT_MODES,
    ModelConfig,
    TrainingConfig,
    get_default_device,
    resolve_device,
    set_default_device,
)
from polymer_ranking.training import _drop_duplicate_pairs  # noqa: E402


class TestConfigValidation:
    """Structurally invalid values must be refused instead of silently used.

    Values that are merely *inadvisable* (a very small lr, no weight decay, ...)
    stay legal — those are warnings belonging to
    ``hyperparam_search.py --stage check``.
    """

    @pytest.mark.parametrize("field,value", [
        ("lr", 0.0),
        ("lr", -1e-3),
        ("batch_size", 0),
        ("epochs", 0),
        ("patience", 0),
        ("max_repeats", 0),
        ("weight_decay", -1e-3),
        ("rank_weight", -0.1),
        ("reg_weight", -0.1),
        ("censored_weight", 1.5),
        ("censored_weight", -0.1),
        ("delta_scale", 0.0),
        ("delta_scale", -1.0),
        ("device", "tpu"),
        ("split_by", "scaffold"),
        ("scheduler", "step"),
        ("early_stop_metric", "acc"),
    ])
    def test_training_rejects_invalid(self, field, value):
        with pytest.raises(ValueError):
            TrainingConfig(**{field: value})

    def test_both_loss_terms_off_is_rejected(self):
        with pytest.raises(ValueError, match="switched off"):
            TrainingConfig(rank_weight=0.0, reg_weight=0.0)

    @pytest.mark.parametrize("field,value", [
        ("lr", 1e-8),          # tiny but representable -> only a check warning
        ("weight_decay", 0.0),
        ("epochs", 5000),
        ("patience", 3),
        ("delta_scale", 50.0),
        ("censored_weight", 0.0),
    ])
    def test_inadvisable_values_stay_legal(self, field, value):
        assert getattr(TrainingConfig(**{field: value}), field) == value

    @pytest.mark.parametrize("field,value", [
        ("hidden_size", 0),
        ("ffn_hidden", 0),
        ("depth", 0),
        ("num_tasks", 0),
        ("dropout", 1.0),
        ("dropout", -0.1),
        ("aggregation", "max"),
    ])
    def test_model_rejects_invalid(self, field, value):
        with pytest.raises(ValueError):
            ModelConfig(**{field: value})

    def test_to_dict_covers_every_field(self):
        cfg = TrainingConfig()
        assert cfg.to_dict()["seed"] == 42
        assert set(cfg.to_dict()) == set(cfg.__dataclass_fields__)

    def test_valid_splits_and_devices_registered(self):
        assert set(SPLIT_MODES) == {"pair", "material", "group"}
        assert set(DEVICE_MODES) == {"auto", "cpu", "cuda"}


class TestDeviceResolution:
    def test_cpu_is_explicit(self):
        assert resolve_device("cpu") == pytest.importorskip("torch").device("cpu")

    def test_auto_falls_back_to_cpu_without_a_gpu(self):
        import torch
        dev = resolve_device("auto")
        assert dev.type == ("cuda" if torch.cuda.is_available() else "cpu")

    def test_requesting_cuda_without_a_gpu_is_an_error(self):
        import torch
        if torch.cuda.is_available():
            pytest.skip("a GPU is present")
        with pytest.raises(ValueError, match="cuda"):
            resolve_device("cuda")

    def test_runtime_switch_takes_effect_after_import(self):
        """The old module-level DEVICE froze at import time; this proves it does not."""
        set_default_device("cpu")
        assert get_default_device().type == "cpu"
        set_default_device("auto")
        assert get_default_device().type in ("cpu", "cuda")


class TestCensorFloorIsSplitLocal:
    """A held-out split must not influence another split's sentinel."""

    def _frame(self):
        raw = {
            "mu_e_1": [1e-2, 1e-3, 1e-1, 5e-2],
            "mu_e_2": [2e-3, 2e-2, 2e-2, 0.0],
            "mu_h_1": [1e-2, 1e-3, 1e-1, 5e-2],
            "mu_h_2": [2e-3, 2e-2, 2e-2, 0.0],
        }
        return pd.DataFrame(raw)

    def test_floor_depends_only_on_the_passed_frame(self):
        whole = self._frame()
        subset = whole.iloc[[2, 3]]            # smallest measured value is 1e-2
        assert censor_floor(subset, "mu_e") > censor_floor(whole, "mu_e")

    def test_apply_rewrites_log_columns_per_frame(self):
        whole = self._frame()
        recomputed = apply_censor_floor(whole.iloc[[2, 3]])
        original = apply_censor_floor(whole).iloc[[2, 3]]
        whole_vals = original["log_mu_e_2"].values
        subset_vals = recomputed["log_mu_e_2"].values
        # row 3 has a censored mu_e_2 -> its sentinel follows the frame it is in
        assert not np.allclose(whole_vals, subset_vals)
        assert not recomputed["ok_mu_e_2"].iloc[1]

    def test_measured_values_are_untouched(self):
        df = apply_censor_floor(self._frame())
        assert np.isclose(df["log_mu_e_1"].iloc[0], np.log10(1e-2))

    def test_missing_task_column_raises_clearly(self):
        with pytest.raises(KeyError):
            censor_floor(pd.DataFrame({"mu_e_1": [1.0]}), "mu_h")


class TestDuplicatePairRemoval:
    def test_second_occurrence_is_dropped(self):
        df = pd.DataFrame({
            "Materials_1": ["A", "B", "A"],
            "Materials_2": ["B", "A", "B"],
        })
        out = _drop_duplicate_pairs(df)
        assert len(out) == 1

    def test_distinct_pairs_survive(self):
        df = pd.DataFrame({
            "Materials_1": ["A", "B", "C"],
            "Materials_2": ["B", "C", "D"],
        })
        assert len(_drop_duplicate_pairs(df)) == 3


class TestSmilesRobustness:
    @pytest.mark.parametrize("bad", [float("nan"), None, 3.5, b"*c1ccccc1*"])
    def test_non_string_input_is_skipped_not_fatal(self, bad):
        """A single empty CSV cell used to raise inside RDKit and abort the run."""
        assert cyclize_polymer_with_cp_marking(bad) == (None, None)

    def test_empty_string_is_not_fatal(self):
        assert cyclize_polymer_with_cp_marking("") == (None, None)

    def test_valid_input_still_works(self):
        smiles, mol = cyclize_polymer_with_cp_marking("*c1ccccc1*")
        assert smiles is not None and mol is not None
