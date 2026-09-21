import numpy as np
import torch
import torch.nn as nn
import pytest

from polymer_ranking.training import (
    EarlyStopping,
    compute_delta_scale,
    compute_metrics,
    leakage_report,
    _material_split,
    _monitor_value,
    _stopper_mode,
)
from polymer_ranking.config import TrainingConfig


class TestEarlyStopping:
    def test_init_defaults(self):
        es = EarlyStopping()
        assert es.patience == 20
        assert es.delta == 1e-4
        assert es.mode == "min"
        assert es.best_score == float("inf")
        assert es.counter == 0
        assert es.best_state is None

    def test_init_custom(self):
        es = EarlyStopping(patience=10, delta=0.01)
        assert es.patience == 10
        assert es.delta == 0.01

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="auto")

    def test_first_step_saves_state(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        should_stop = es.step(5.0, model)
        assert should_stop is False
        assert es.best_score == 5.0
        assert es.counter == 0
        assert es.best_state is not None

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        es.step(6.0, model)
        assert es.counter == 1
        es.step(4.0, model)
        assert es.counter == 0
        assert es.best_score == 4.0

    def test_no_improvement_increments_counter(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        es.step(5.1, model)
        assert es.counter == 1
        es.step(5.2, model)
        assert es.counter == 2

    def test_triggers_stop_at_patience(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        assert es.step(6.0, model) is False
        assert es.step(6.0, model) is False
        assert es.step(6.0, model) is True

    def test_delta_threshold(self):
        es = EarlyStopping(patience=5, delta=0.1)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        assert es.step(4.95, model) is False
        assert es.counter == 1

    def test_max_mode_monitors_increasing_values(self):
        es = EarlyStopping(patience=2, mode="max")
        model = nn.Linear(10, 2)
        es.step(0.5, model)
        assert es.counter == 0
        es.step(0.4, model)
        assert es.counter == 1
        es.step(0.6, model)
        assert es.counter == 0
        assert es.best_score == 0.6

    def test_best_state_is_cpu_copy(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        for key, val in es.best_state.items():
            assert val.device == torch.device("cpu")


class TestMonitorValue:
    """The monitor is always 'higher is better'; otherwise early stopping picks
    the *worst* model when it watches the loss."""

    def test_pair_acc_is_used_directly(self):
        cfg = TrainingConfig(early_stop_metric="pair_acc")
        assert _monitor_value({"mean_pair_acc": 0.7, "loss": 1.2}, cfg) == 0.7

    def test_loss_is_negated(self):
        cfg = TrainingConfig(early_stop_metric="loss")
        assert _monitor_value({"mean_pair_acc": 0.7, "loss": 1.2}, cfg) == -1.2

    def test_lower_loss_means_better_monitor(self):
        cfg = TrainingConfig(early_stop_metric="loss")
        worse = _monitor_value({"loss": 2.0}, cfg)
        better = _monitor_value({"loss": 1.0}, cfg)
        assert better > worse

    def test_stopper_always_maximises(self):
        for metric in ("pair_acc", "loss"):
            assert _stopper_mode(TrainingConfig(early_stop_metric=metric)) == "max"


class TestComputeDeltaScale:
    def test_uses_only_measured_pairs(self):
        y1 = np.array([[1.0, 1.0], [2.0, 2.0], [-7.0, -7.0]], dtype=np.float32)
        y2 = np.array([[0.0, 0.0], [0.0, 0.0], [-7.0, -7.0]], dtype=np.float32)
        ok = np.array([[True, True], [True, True], [False, False]])
        # differences of the two measured pairs: 1.0 and 2.0 -> std 0.5
        assert compute_delta_scale(y1, y2, ok) == pytest.approx(0.5)

    def test_fallback_when_nothing_measured(self):
        y1 = np.zeros((2, 2), dtype=np.float32)
        y2 = np.zeros((2, 2), dtype=np.float32)
        assert compute_delta_scale(y1, y2, np.zeros((2, 2), dtype=bool)) == 1.0

    def test_fallback_when_constant(self):
        y1 = np.ones((3, 2), dtype=np.float32)
        y2 = np.ones((3, 2), dtype=np.float32)
        assert compute_delta_scale(y1, y2, np.ones((3, 2), dtype=bool)) == 1.0


class TestComputeMetrics:
    def test_perfect_ranking(self):
        s1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        s2 = np.array([[1.0, 1.0], [2.0, 3.0]])
        y1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        y2 = np.array([[1.0, 1.0], [2.0, 3.0]])
        metrics = compute_metrics(s1, s2, y1, y2)
        assert metrics["mu_e_pair_acc"] == pytest.approx(1.0)
        assert metrics["mu_h_pair_acc"] == pytest.approx(1.0)
        assert metrics["mean_pair_acc"] == pytest.approx(1.0)

    def test_random_ranking(self):
        np.random.seed(42)
        N = 100
        metrics = compute_metrics(np.random.randn(N, 2), np.random.randn(N, 2),
                                  np.random.randn(N, 2), np.random.randn(N, 2))
        assert 0.0 <= metrics["mu_e_pair_acc"] <= 1.0
        assert 0.0 <= metrics["mu_h_pair_acc"] <= 1.0

    def test_output_keys(self):
        metrics = compute_metrics(np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]),
                                  np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]))
        for k in ("mu_e_pair_acc", "mu_h_pair_acc", "mu_e_spearman", "mu_h_spearman",
                  "mu_e_avg_prob", "mean_pair_acc"):
            assert k in metrics

    def test_spearman_range(self):
        np.random.seed(0)
        N = 50
        metrics = compute_metrics(np.random.randn(N, 2), np.random.randn(N, 2),
                                  np.random.randn(N, 2), np.random.randn(N, 2))
        assert -1.0 <= metrics["mu_e_spearman"] <= 1.0
        assert -1.0 <= metrics["mu_h_spearman"] <= 1.0

    def test_equal_targets_pair_acc(self):
        metrics = compute_metrics(np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]),
                                  np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))
        assert metrics["mu_e_pair_acc"] == 0.0
        assert metrics["mu_h_pair_acc"] == 0.0

    def test_avg_prob_rewards_confident_correct_direction(self):
        """avg_prob is the probability of the *correct* direction, not |Δs|."""
        y1 = np.array([[1.0, 1.0], [1.0, 1.0]])
        y2 = np.array([[0.0, 0.0], [0.0, 0.0]])
        # row 0: correctly ordered but barely; row 1: wrongly ordered but confident
        s1 = np.array([[0.01, 0.01], [-8.0, -8.0]])
        s2 = np.array([[0.0, 0.0], [0.0, 0.0]])
        metrics = compute_metrics(s1, s2, y1, y2)
        assert metrics["mu_e_pair_acc"] == pytest.approx(0.5)
        assert metrics["mu_e_avg_prob"] == pytest.approx(
            (1 / (1 + np.exp(-0.01)) + 1 / (1 + np.exp(8.0))) / 2)
        assert metrics["mu_e_avg_prob"] < 0.6

    def test_censored_pairs_count_towards_accuracy_not_spearman(self):
        y1 = np.array([[0.0, 0.0], [0.0, 0.0]])
        y2 = np.array([[-7.0, -7.0], [-7.0, -7.0]])   # side 2 censored
        s1 = np.array([[3.0, 3.0], [-3.0, -3.0]])     # first right, second wrong
        s2 = np.zeros((2, 2))
        ok1 = np.array([[True, True], [True, True]])
        ok2 = np.array([[False, False], [False, False]])
        valid = ok1 & ok2
        rank_valid = (ok1 | ok2) & (y1 != y2)
        metrics = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        assert metrics["mu_e_pair_acc"] == pytest.approx(0.5)
        assert metrics["mu_e_n_rank"] == 2
        assert metrics["mu_e_n_reg"] == 0
        assert metrics["mu_e_spearman"] == 0.0

    def test_both_censored_pairs_are_ignored(self):
        y1 = np.array([[-7.0, -7.0]])
        y2 = np.array([[-7.0, -7.0]])
        s1 = np.array([[3.0, 3.0]])
        s2 = np.zeros((1, 2))
        ok = np.zeros((1, 2), dtype=bool)
        metrics = compute_metrics(s1, s2, y1, y2, ok, ok)
        assert metrics["mu_e_n_rank"] == 0
        assert metrics["mu_e_pair_acc"] == 0.0


class TestSplitting:
    @pytest.fixture
    def df(self):
        import pandas as pd
        # A-B, A-C, B-D, C-D, E-F  (E/F only occur in the last pair)
        return pd.DataFrame({
            "Materials_1": ["A", "A", "B", "C", "E"],
            "Materials_2": ["B", "C", "D", "D", "F"],
        })

    def test_material_split_is_disjoint(self, df):
        tr, va, te = _material_split(df, 0.4, 0.2, seed=0)
        # straddling pairs are dropped, so the splits cannot exceed the data
        assert len(tr) + len(va) + len(te) <= len(df)
        assert len(te) > 0

        def mats(idx):
            return set(df["Materials_1"].iloc[idx]) | set(df["Materials_2"].iloc[idx])

        assert not (mats(tr) & mats(te))
        assert not (mats(tr) & mats(va))
        assert not (mats(va) & mats(te))

    def test_leakage_report_detects_overlap(self, df):
        # identical split on both sides -> full material overlap
        rep = leakage_report(df, np.arange(5), np.array([]), np.arange(5))
        assert rep["test_material_overlap"] == pytest.approx(1.0)
        assert rep["test_pair_overlap"] == pytest.approx(1.0)

    def test_leakage_report_zero_for_disjoint_split(self, df):
        tr, va, te = _material_split(df, 0.4, 0.2, seed=0)
        rep = leakage_report(df, tr, va, te)
        assert rep["test_material_overlap"] == 0.0
