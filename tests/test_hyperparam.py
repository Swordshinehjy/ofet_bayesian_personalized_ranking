"""Tests for hyperparam_search: rule-based sanity check + auto-fix."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymer_ranking.config import ModelConfig, TrainingConfig
from hyperparam_search import check_config, auto_fix


def _levels(findings):
    return {f["item"]: f["level"] for f in findings}


class TestCheckConfig:
    def test_default_config_passes(self):
        findings = check_config(ModelConfig(), TrainingConfig(), n_train=1100,
                                delta_std=1.1)
        assert all(f["level"] == "OK" for f in findings)

    def test_oversized_model_warns(self):
        m = ModelConfig(hidden_size=1024, ffn_hidden=1024)
        lvl = _levels(check_config(m, TrainingConfig(), n_train=200, delta_std=1.0))
        assert lvl["model_capacity"] == "WARN"

    def test_tiny_learning_rate_warns(self):
        t = TrainingConfig(lr=1e-8)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["lr"] == "WARN"

    def test_lr_inconsistent_with_batch_size_warns(self):
        # batch_size=128 needs lr ~4e-3 under linear scaling; 1e-4 is far below
        t = TrainingConfig(lr=1e-4, batch_size=128)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["lr_vs_batch_size"] == "WARN"

    def test_lr_scaled_with_batch_size_is_ok(self):
        t = TrainingConfig(lr=4e-3, batch_size=128)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["lr_vs_batch_size"] == "OK"

    def test_cosine_with_large_epoch_budget_warns(self):
        t = TrainingConfig(scheduler="cosine", epochs=400, patience=30)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["scheduler"] == "WARN"

    def test_regression_dominant_loss_warns(self):
        t = TrainingConfig(rank_weight=0.2, reg_weight=0.8)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["rank/reg weights"] == "WARN"

    def test_wrong_delta_scale_warns(self):
        t = TrainingConfig(delta_scale=20.0)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["delta_scale"] == "WARN"

    def test_auto_delta_scale_is_ok(self):
        t = TrainingConfig(delta_scale=None)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["delta_scale"] == "OK"

    def test_zero_censored_weight_warns(self):
        t = TrainingConfig(censored_weight=0.0)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["censored_weight"] == "WARN"

    def test_loss_monitoring_warns(self):
        t = TrainingConfig(early_stop_metric="loss")
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["early_stop_metric"] == "WARN"

    def test_short_patience_warns(self):
        t = TrainingConfig(patience=3)
        lvl = _levels(check_config(ModelConfig(), t, n_train=1100, delta_std=1.0))
        assert lvl["patience"] == "WARN"

    def test_no_dropout_and_no_weight_decay_warn(self):
        m, t = ModelConfig(dropout=0.0), TrainingConfig(weight_decay=0.0)
        lvl = _levels(check_config(m, t, n_train=1100, delta_std=1.0))
        assert lvl["dropout"] == "WARN"
        assert lvl["weight_decay"] == "WARN"


class TestAutoFix:
    def test_fixes_every_warning(self):
        m = ModelConfig(hidden_size=1024, ffn_hidden=1024, dropout=0.0)
        t = TrainingConfig(lr=1e-8, weight_decay=0.0, epochs=5000, patience=3,
                           scheduler="cosine", early_stop_metric="loss",
                           rank_weight=0.2, reg_weight=0.8, censored_weight=0.0,
                           delta_scale=50.0)
        findings = check_config(m, t, n_train=1100, delta_std=1.0)
        assert any(f["level"] != "OK" for f in findings)

        m2, t2, applied = auto_fix(m, t, findings, delta_std=1.0)
        assert applied, "auto_fix should report what it changed"
        after = check_config(m2, t2, n_train=1100, delta_std=1.0)
        assert all(f["level"] == "OK" for f in after), [
            f for f in after if f["level"] != "OK"]

    def test_leaves_a_clean_config_untouched(self):
        m, t = ModelConfig(), TrainingConfig()
        findings = check_config(m, t, n_train=1100, delta_std=1.1)
        _, _, applied = auto_fix(m, t, findings, delta_std=1.1)
        assert applied == []
