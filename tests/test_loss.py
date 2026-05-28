import torch
import pytest

from polymer_ranking.loss import MultiTaskBayesianRankingLoss


class TestMultiTaskBayesianRankingLoss:
    @pytest.fixture
    def loss_fn(self):
        return MultiTaskBayesianRankingLoss()

    def test_init_defaults(self):
        fn = MultiTaskBayesianRankingLoss()
        assert fn.alpha == 0.6
        assert fn.beta == 0.4
        assert fn.task_w == [1.0, 1.0]

    def test_init_custom(self):
        fn = MultiTaskBayesianRankingLoss(
            rank_weight=0.7,
            reg_weight=0.3,
            task_weights=[2.0, 1.0],
        )
        assert fn.alpha == 0.7
        assert fn.beta == 0.3
        assert fn.task_w == [2.0, 1.0]

    def test_output_is_tuple(self, loss_fn):
        B, T = 4, 2
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        result = loss_fn(s1, s2, y1, y2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_total_loss_is_tensor(self, loss_fn):
        B, T = 4, 2
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        total, log = loss_fn(s1, s2, y1, y2)
        assert isinstance(total, torch.Tensor)
        assert total.dim() == 0

    def test_log_contains_expected_keys(self, loss_fn):
        B, T = 4, 2
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        _, log = loss_fn(s1, s2, y1, y2)
        assert "mu_e_bpr" in log
        assert "mu_e_reg" in log
        assert "mu_h_bpr" in log
        assert "mu_h_reg" in log
        assert "total" in log

    def test_loss_nonnegative(self, loss_fn):
        B, T = 8, 2
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        total, _ = loss_fn(s1, s2, y1, y2)
        assert total.item() >= 0.0

    def test_perfect_ranking_low_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[1.0, 2.0]] * B)
        y2 = torch.tensor([[0.0, 0.0]] * B)
        s1 = y1.clone() + 10.0
        s2 = y2.clone()
        total, log = loss_fn(s1, s2, y1, y2)
        assert log["mu_e_bpr"] == pytest.approx(0.0, abs=1e-4)
        assert log["mu_h_bpr"] == pytest.approx(0.0, abs=1e-4)

    def test_equal_targets_zero_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.ones(B, T)
        y2 = torch.ones(B, T)
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        _, log = loss_fn(s1, s2, y1, y2)
        assert log["mu_e_bpr"] == pytest.approx(0.0, abs=1e-6)
        assert log["mu_h_bpr"] == pytest.approx(0.0, abs=1e-6)

    def test_wrong_ranking_high_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[1.0, 2.0]] * B)
        y2 = torch.tensor([[0.0, 0.0]] * B)
        s1 = y2.clone() - 10.0
        s2 = y1.clone() + 10.0
        _, log = loss_fn(s1, s2, y1, y2)
        assert log["mu_e_bpr"] > 0.0
        assert log["mu_h_bpr"] > 0.0

    def test_reg_loss_zero_when_diffs_match(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[2.0, 3.0]] * B)
        y2 = torch.tensor([[1.0, 1.0]] * B)
        s1 = y1.clone()
        s2 = y2.clone()
        _, log = loss_fn(s1, s2, y1, y2)
        assert log["mu_e_reg"] == pytest.approx(0.0, abs=1e-6)
        assert log["mu_h_reg"] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self, loss_fn):
        B, T = 4, 2
        s1 = torch.randn(B, T, requires_grad=True)
        s2 = torch.randn(B, T, requires_grad=True)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        total, _ = loss_fn(s1, s2, y1, y2)
        total.backward()
        assert s1.grad is not None
        assert s2.grad is not None

    def test_task_weights_affect_loss(self):
        fn_equal = MultiTaskBayesianRankingLoss(task_weights=[1.0, 1.0])
        fn_weighted = MultiTaskBayesianRankingLoss(task_weights=[2.0, 1.0])
        B, T = 4, 2
        s1 = torch.randn(B, T)
        s2 = torch.randn(B, T)
        y1 = torch.randn(B, T)
        y2 = torch.randn(B, T)
        total_equal, _ = fn_equal(s1, s2, y1, y2)
        total_weighted, _ = fn_weighted(s1, s2, y1, y2)
        assert total_equal.item() != total_weighted.item()

    def test_bpr_loss_monotonic_with_score_diff(self, loss_fn):
        y1 = torch.tensor([[1.0, 1.0]])
        y2 = torch.tensor([[0.0, 0.0]])
        s_diff_small = torch.tensor([[0.1, 0.1]])
        s_diff_large = torch.tensor([[5.0, 5.0]])
        s2_base = torch.tensor([[0.0, 0.0]])
        _, log_small = loss_fn(s_diff_small, s2_base, y1, y2)
        _, log_large = loss_fn(s_diff_large, s2_base, y1, y2)
        assert log_small["mu_e_bpr"] > log_large["mu_e_bpr"]
