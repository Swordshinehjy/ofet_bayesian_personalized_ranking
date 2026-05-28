import numpy as np
import torch
import torch.nn as nn
import pytest

from polymer_ranking.training import EarlyStopping, compute_metrics


class TestEarlyStopping:
    def test_init_defaults(self):
        es = EarlyStopping()
        assert es.patience == 20
        assert es.delta == 1e-4
        assert es.best_loss == float("inf")
        assert es.counter == 0
        assert es.best_state is None

    def test_init_custom(self):
        es = EarlyStopping(patience=10, delta=0.01)
        assert es.patience == 10
        assert es.delta == 0.01

    def test_first_step_saves_state(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        should_stop = es.step(5.0, model)
        assert should_stop is False
        assert es.best_loss == 5.0
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
        assert es.best_loss == 4.0

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
        assert es.step(4.94, model) is False
        assert es.counter == 2

    def test_best_state_is_cpu_copy(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(10, 2)
        es.step(5.0, model)
        for key, val in es.best_state.items():
            assert val.device == torch.device("cpu")


class TestComputeMetrics:
    def test_perfect_ranking(self):
        s1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        s2 = np.array([[1.0, 1.0], [2.0, 3.0]])
        y1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        y2 = np.array([[1.0, 1.0], [2.0, 3.0]])
        metrics = compute_metrics(s1, s2, y1, y2)
        assert metrics["mu_e_pair_acc"] == pytest.approx(1.0)
        assert metrics["mu_h_pair_acc"] == pytest.approx(1.0)

    def test_random_ranking(self):
        np.random.seed(42)
        N = 100
        s1 = np.random.randn(N, 2)
        s2 = np.random.randn(N, 2)
        y1 = np.random.randn(N, 2)
        y2 = np.random.randn(N, 2)
        metrics = compute_metrics(s1, s2, y1, y2)
        assert 0.0 <= metrics["mu_e_pair_acc"] <= 1.0
        assert 0.0 <= metrics["mu_h_pair_acc"] <= 1.0

    def test_output_keys(self):
        s1 = np.array([[1.0, 2.0]])
        s2 = np.array([[0.0, 0.0]])
        y1 = np.array([[1.0, 2.0]])
        y2 = np.array([[0.0, 0.0]])
        metrics = compute_metrics(s1, s2, y1, y2)
        assert "mu_e_pair_acc" in metrics
        assert "mu_h_pair_acc" in metrics
        assert "mu_e_spearman" in metrics
        assert "mu_h_spearman" in metrics

    def test_spearman_range(self):
        np.random.seed(0)
        N = 50
        s1 = np.random.randn(N, 2)
        s2 = np.random.randn(N, 2)
        y1 = np.random.randn(N, 2)
        y2 = np.random.randn(N, 2)
        metrics = compute_metrics(s1, s2, y1, y2)
        assert -1.0 <= metrics["mu_e_spearman"] <= 1.0
        assert -1.0 <= metrics["mu_h_spearman"] <= 1.0

    def test_equal_targets_pair_acc(self):
        s1 = np.array([[1.0, 2.0]])
        s2 = np.array([[0.0, 0.0]])
        y1 = np.array([[1.0, 1.0]])
        y2 = np.array([[1.0, 1.0]])
        metrics = compute_metrics(s1, s2, y1, y2)
        assert metrics["mu_e_pair_acc"] == 0.0
        assert metrics["mu_h_pair_acc"] == 0.0
