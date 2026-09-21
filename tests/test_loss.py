import torch
import pytest

from polymer_ranking.loss import MultiTaskBayesianRankingLoss


def _rank(y1, y2, s1, s2, **kw):
    return MultiTaskBayesianRankingLoss(**kw)(s1, s2, y1, y2)


class TestMultiTaskBayesianRankingLoss:
    @pytest.fixture
    def loss_fn(self):
        return MultiTaskBayesianRankingLoss()

    def test_init_defaults(self):
        fn = MultiTaskBayesianRankingLoss()
        assert fn.alpha == 0.9
        assert fn.beta == 0.1
        assert fn.task_w == [1.0, 1.0]
        assert fn.delta_scale == 1.0
        assert fn.censored_weight == 1.0

    def test_init_custom(self):
        fn = MultiTaskBayesianRankingLoss(
            rank_weight=0.7,
            reg_weight=0.3,
            task_weights=[2.0, 1.0],
            delta_scale=1.5,
            censored_weight=0.25,
        )
        assert fn.alpha == 0.7
        assert fn.beta == 0.3
        assert fn.task_w == [2.0, 1.0]
        assert fn.delta_scale == 1.5
        assert fn.censored_weight == 0.25

    def test_output_is_tuple(self, loss_fn):
        B, T = 4, 2
        result = loss_fn(torch.randn(B, T), torch.randn(B, T),
                         torch.randn(B, T), torch.randn(B, T))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_total_loss_is_tensor(self, loss_fn):
        B, T = 4, 2
        total, _ = loss_fn(torch.randn(B, T), torch.randn(B, T),
                           torch.randn(B, T), torch.randn(B, T))
        assert isinstance(total, torch.Tensor)
        assert total.dim() == 0

    def test_log_contains_expected_keys(self, loss_fn):
        B, T = 4, 2
        _, log = loss_fn(torch.randn(B, T), torch.randn(B, T),
                         torch.randn(B, T), torch.randn(B, T))
        for k in ("mu_e_bpr", "mu_e_reg", "mu_h_bpr", "mu_h_reg",
                  "mu_e_n_rank", "mu_e_n_reg", "total"):
            assert k in log

    def test_loss_nonnegative(self, loss_fn):
        B, T = 8, 2
        total, _ = loss_fn(torch.randn(B, T), torch.randn(B, T),
                           torch.randn(B, T), torch.randn(B, T))
        assert total.item() >= 0.0

    def test_perfect_ranking_low_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[1.0, 2.0]] * B)
        y2 = torch.tensor([[0.0, 0.0]] * B)
        s1 = y1.clone() + 10.0
        s2 = y2.clone()
        _, log = loss_fn(s1, s2, y1, y2)
        assert log["mu_e_bpr"] == pytest.approx(0.0, abs=1e-4)
        assert log["mu_h_bpr"] == pytest.approx(0.0, abs=1e-4)

    def test_equal_targets_zero_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.ones(B, T)
        y2 = torch.ones(B, T)
        _, log = loss_fn(torch.randn(B, T), torch.randn(B, T), y1, y2)
        assert log["mu_e_bpr"] == pytest.approx(0.0, abs=1e-6)
        assert log["mu_h_bpr"] == pytest.approx(0.0, abs=1e-6)

    def test_wrong_ranking_high_bpr_loss(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[1.0, 2.0]] * B)
        y2 = torch.tensor([[0.0, 0.0]] * B)
        _, log = loss_fn(y2.clone() - 10.0, y1.clone() + 10.0, y1, y2)
        assert log["mu_e_bpr"] > 0.0
        assert log["mu_h_bpr"] > 0.0

    def test_reg_loss_zero_when_diffs_match(self, loss_fn):
        B, T = 4, 2
        y1 = torch.tensor([[2.0, 3.0]] * B)
        y2 = torch.tensor([[1.0, 1.0]] * B)
        _, log = loss_fn(y1.clone(), y2.clone(), y1, y2)
        assert log["mu_e_reg"] == pytest.approx(0.0, abs=1e-6)
        assert log["mu_h_reg"] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self, loss_fn):
        B, T = 4, 2
        s1 = torch.randn(B, T, requires_grad=True)
        s2 = torch.randn(B, T, requires_grad=True)
        total, _ = loss_fn(s1, s2, torch.randn(B, T), torch.randn(B, T))
        total.backward()
        assert s1.grad is not None
        assert s2.grad is not None

    def test_task_weights_affect_loss(self):
        s1, s2, y1, y2 = (torch.randn(4, 2) for _ in range(4))
        a, _ = MultiTaskBayesianRankingLoss(task_weights=[1.0, 1.0])(s1, s2, y1, y2)
        b, _ = MultiTaskBayesianRankingLoss(task_weights=[2.0, 1.0])(s1, s2, y1, y2)
        assert a.item() != b.item()

    def test_bpr_loss_monotonic_with_score_diff(self, loss_fn):
        y1 = torch.tensor([[1.0, 1.0]])
        y2 = torch.tensor([[0.0, 0.0]])
        s2_base = torch.tensor([[0.0, 0.0]])
        _, log_small = loss_fn(torch.tensor([[0.1, 0.1]]), s2_base, y1, y2)
        _, log_large = loss_fn(torch.tensor([[5.0, 5.0]]), s2_base, y1, y2)
        assert log_small["mu_e_bpr"] > log_large["mu_e_bpr"]

    def test_delta_scale_normalizes_regression(self):
        """A larger delta_scale shrinks the regression term (target / delta_scale)."""
        y1 = torch.tensor([[3.0, 3.0]])
        y2 = torch.tensor([[1.0, 1.0]])
        s1 = torch.tensor([[0.0, 0.0]])
        s2 = torch.tensor([[0.0, 0.0]])
        _, small = MultiTaskBayesianRankingLoss(delta_scale=1.0)(s1, s2, y1, y2)
        _, large = MultiTaskBayesianRankingLoss(delta_scale=2.0)(s1, s2, y1, y2)
        assert large["mu_e_reg"] == pytest.approx(small["mu_e_reg"] / 4.0, rel=1e-5)


class TestCensoredLabels:
    """A mobility of 0 is left-censored: ordering known, magnitude unknown."""

    def test_censored_pair_excluded_from_regression(self):
        # y1 measured (0.0 in log space == 1 cm2/Vs), y2 censored -> sentinel -7
        y1 = torch.tensor([[0.0, 0.0]])
        y2 = torch.tensor([[-7.0, -7.0]])
        s1 = torch.tensor([[5.0, 5.0]])   # deliberately "wrong" magnitude
        s2 = torch.tensor([[0.0, 0.0]])
        ok1 = torch.tensor([[True, True]])
        ok2 = torch.tensor([[False, False]])
        total, log = MultiTaskBayesianRankingLoss()(
            s1, s2, y1, y2, valid=ok1 & ok2, rank_valid=(ok1 | ok2) & (y1 != y2))
        assert log["mu_e_reg"] == pytest.approx(0.0)
        assert log["mu_e_n_reg"] == 0
        assert log["mu_e_n_rank"] == 1
        # ranking still supervises: the measured side must score higher
        assert total.item() == pytest.approx(0.9 * log["mu_e_bpr"] +
                                             0.9 * log["mu_h_bpr"], rel=1e-5)

    def test_both_censored_pair_contributes_nothing(self):
        y1 = torch.tensor([[-7.0, -7.0]])
        y2 = torch.tensor([[-7.0, -7.0]])
        s1 = torch.tensor([[3.0, 3.0]])
        s2 = torch.tensor([[0.0, 0.0]])
        ok1 = torch.tensor([[False, False]])
        ok2 = torch.tensor([[False, False]])
        total, log = MultiTaskBayesianRankingLoss()(
            s1, s2, y1, y2, valid=ok1 & ok2, rank_valid=(ok1 | ok2) & (y1 != y2))
        assert log["mu_e_n_rank"] == 0
        assert log["mu_e_n_reg"] == 0
        assert total.item() == pytest.approx(0.0)

    def test_censored_weight_downweights_censored_pairs(self):
        """``censored_weight`` is a *relative* weight inside the weighted mean.

        A batch holding one measured pair (wrongly ordered, large BPR term) and
        one censored pair (correctly ordered, ~0 BPR term) shows the effect:
        halving the censored weight increases the weighted mean towards the
        measured term.
        """
        import torch.nn.functional as F
        # pair 0: measured on both sides, model orders it wrongly
        # pair 1: side 2 censored (sentinel -7), model orders it correctly
        y1 = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
        y2 = torch.tensor([[0.0, 0.0], [-7.0, -7.0]])
        s1 = torch.tensor([[-4.0, -4.0], [4.0, 4.0]])
        s2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        ok1 = torch.tensor([[True, True], [True, True]])
        ok2 = torch.tensor([[True, True], [False, False]])
        valid = ok1 & ok2
        rank_valid = (ok1 | ok2) & (y1 != y2)
        assert int(rank_valid[:, 0].sum()) == 2
        assert int(valid[:, 0].sum()) == 1

        sp_measured = float(F.softplus(torch.tensor(4.0)))   # wrongly ordered
        sp_censored = float(F.softplus(torch.tensor(-4.0)))  # correctly ordered

        _, full = MultiTaskBayesianRankingLoss(censored_weight=1.0)(
            s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)
        _, half = MultiTaskBayesianRankingLoss(censored_weight=0.5)(
            s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)

        assert full["mu_e_bpr"] == pytest.approx(
            (sp_measured + sp_censored) / 2.0, rel=1e-5)
        assert half["mu_e_bpr"] == pytest.approx(
            (sp_measured + 0.5 * sp_censored) / 1.5, rel=1e-5)
        assert half["mu_e_bpr"] > full["mu_e_bpr"]

    def test_measured_pairs_keep_full_weight(self):
        y1 = torch.tensor([[2.0, 2.0]])
        y2 = torch.tensor([[0.0, 0.0]])
        s1 = torch.tensor([[-4.0, -4.0]])
        s2 = torch.tensor([[0.0, 0.0]])
        ok = torch.tensor([[True, True]])
        _, full = MultiTaskBayesianRankingLoss(censored_weight=1.0)(
            s1, s2, y1, y2, valid=ok, rank_valid=ok)
        _, half = MultiTaskBayesianRankingLoss(censored_weight=0.5)(
            s1, s2, y1, y2, valid=ok, rank_valid=ok)
        assert half["mu_e_bpr"] == pytest.approx(full["mu_e_bpr"], rel=1e-5)
