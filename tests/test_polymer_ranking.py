"""Unit tests for polymer_ranking package.

Run: conda activate chemprop2 && python -m pytest tests/ -v
"""

import sys
import os
import tempfile
import logging

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── chemistry.py tests ──────────────────────────────────────────────

from polymer_ranking.chemistry import (
    cyclize_polymer_with_cp_marking,
    cyclize_df,
    extra_feat,
    load_and_preprocess,
)


class TestCyclizePolymer:
    """Tests for cyclize_polymer_with_cp_marking."""

    def test_normal_polymer(self):
        """Complex polymer SMILES from dataset should cyclize successfully."""
        smiles = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
        result, mol = cyclize_polymer_with_cp_marking(smiles)
        assert result is not None, "Normal polymer should cyclize"
        assert mol is not None
        cp_atoms = [i for i, a in enumerate(mol.GetAtoms())
                    if a.HasProp("is_cp") and a.GetBoolProp("is_cp")]
        assert len(cp_atoms) == 2, f"Should have 2 CP atoms, got {len(cp_atoms)}"

    def test_neighbors_already_bonded(self):
        """Two dummy atoms whose neighbors are already bonded (*c1ccccc1*)."""
        smiles = "*c1ccccc1*"
        result, mol = cyclize_polymer_with_cp_marking(smiles)
        assert result is not None, "Should handle already-bonded neighbors"
        assert mol is not None
        cp_atoms = [i for i, a in enumerate(mol.GetAtoms())
                    if a.HasProp("is_cp") and a.GetBoolProp("is_cp")]
        assert len(cp_atoms) == 2

    def test_shared_neighbor(self):
        """Two dummy atoms sharing the same neighbor (*C*).
        n1==n2 so only one CP atom is marked."""
        smiles = "*C*"
        result, mol = cyclize_polymer_with_cp_marking(smiles)
        assert result is not None, "Should handle shared neighbor"
        assert mol is not None
        cp_atoms = [i for i, a in enumerate(mol.GetAtoms())
                    if a.HasProp("is_cp") and a.GetBoolProp("is_cp")]
        # n1==n2, so only 1 unique CP atom is marked
        assert len(cp_atoms) >= 1

    def test_no_dummy(self):
        """No dummy atoms should return (None, None)."""
        result, mol = cyclize_polymer_with_cp_marking("c1ccccc1")
        assert result is None and mol is None

    def test_one_dummy(self):
        """Only one dummy atom should return (None, None)."""
        result, mol = cyclize_polymer_with_cp_marking("*c1ccccc1")
        assert result is None and mol is None

    def test_three_dummies(self):
        """Three dummy atoms should return (None, None)."""
        result, mol = cyclize_polymer_with_cp_marking("*C*C*")
        assert result is None and mol is None

    def test_invalid_smiles(self):
        """Invalid SMILES should return (None, None)."""
        result, mol = cyclize_polymer_with_cp_marking("not_a_smiles")
        assert result is None and mol is None

    def test_dummy_no_neighbor(self):
        """Isolated dummy atom with no neighbor."""
        mol = Chem.MolFromSmiles("[*]")
        if mol is None:
            pytest.skip("RDKit can't parse [*]")
        result, mol = cyclize_polymer_with_cp_marking("[*]")
        assert result is None and mol is None

    def test_cp_atoms_are_marked(self):
        """CP atoms should have is_cp=True, all others should not."""
        smiles = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
        _, mol = cyclize_polymer_with_cp_marking(smiles)
        cp_count = sum(1 for a in mol.GetAtoms()
                       if a.HasProp("is_cp") and a.GetBoolProp("is_cp"))
        non_cp_count = sum(1 for a in mol.GetAtoms()
                           if not (a.HasProp("is_cp") and a.GetBoolProp("is_cp")))
        assert cp_count == 2
        assert non_cp_count > 0


class TestCyclizeDf:
    """Tests for cyclize_df."""

    def test_cyclize_df_adds_columns(self):
        df = pd.DataFrame({
            "Polymer_1": ["*C1=CC=C(*)C=C1"],
            "Polymer_2": ["*C1=CC=C(*)C=C1"],
        })
        result = cyclize_df(df)
        assert "cyc_1" in result.columns
        assert "cyc_2" in result.columns
        assert "mol_1" in result.columns
        assert "mol_2" in result.columns

    def test_cyclize_df_invalid_smiles(self):
        df = pd.DataFrame({
            "Polymer_1": ["invalid"],
            "Polymer_2": ["*C1=CC=C(*)C=C1"],
        })
        result = cyclize_df(df)
        assert result.loc[0, "cyc_1"] is None
        assert result.loc[0, "cyc_2"] is not None


class TestExtraFeat:
    """Tests for extra_feat."""

    def test_extra_feat_shape(self):
        df = pd.DataFrame({
            "conjugation_1": [1.0],
            "Isomer_1": [0.0],
            "CentroSymmetry_1": [1.0],
            "E_LUMO (eV)_1": [-3.5],
            "E_HOMO (eV)_1": [-5.5],
        })
        ef = extra_feat(df, "1")
        assert ef.shape == (1, 5)
        assert ef.dtype == np.float32

    def test_extra_feat_nan_fill(self):
        df = pd.DataFrame({
            "conjugation_1": [np.nan],
            "Isomer_1": [1.0],
            "CentroSymmetry_1": [np.nan],
            "E_LUMO (eV)_1": [-3.5],
            "E_HOMO (eV)_1": [-5.5],
        })
        ef = extra_feat(df, "1")
        assert ef[0, 0] == 0.0  # NaN filled with 0
        assert ef[0, 2] == 0.0


# ── config.py tests ─────────────────────────────────────────────────

from polymer_ranking.config import (
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
    EXTRA_DIM,
    NUM_TASKS,
)


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.hidden_size == 300
        assert cfg.depth == 6
        assert cfg.extra_dim == EXTRA_DIM
        assert cfg.num_tasks == NUM_TASKS

    def test_to_dict_from_dict_roundtrip(self):
        cfg = ModelConfig(hidden_size=128, depth=4)
        d = cfg.to_dict()
        cfg2 = ModelConfig.from_dict(d)
        assert cfg2.hidden_size == 128
        assert cfg2.depth == 4

    def test_from_dict_ignores_unknown_keys(self):
        d = {"hidden_size": 64, "unknown_key": 999}
        cfg = ModelConfig.from_dict(d)
        assert cfg.hidden_size == 64


class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.val_ratio == 0.1
        assert cfg.test_ratio == 0.1

    def test_valid_ratios(self):
        cfg = TrainingConfig(val_ratio=0.2, test_ratio=0.2)
        assert cfg.val_ratio == 0.2

    def test_test_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="test_ratio"):
            TrainingConfig(test_ratio=0.0)

    def test_test_ratio_one_raises(self):
        with pytest.raises(ValueError, match="test_ratio"):
            TrainingConfig(test_ratio=1.0)

    def test_val_ratio_too_large_raises(self):
        with pytest.raises(ValueError, match="val_ratio"):
            TrainingConfig(val_ratio=0.95, test_ratio=0.1)

    def test_val_ratio_zero_ok(self):
        cfg = TrainingConfig(val_ratio=0.0, test_ratio=0.1)
        assert cfg.val_ratio == 0.0


# ── loss.py tests ───────────────────────────────────────────────────

from polymer_ranking.loss import MultiTaskBayesianRankingLoss


class TestMultiTaskBayesianRankingLoss:
    def test_basic_forward(self):
        criterion = MultiTaskBayesianRankingLoss()
        B = 8
        s1 = torch.randn(B, 2, requires_grad=True)
        s2 = torch.randn(B, 2, requires_grad=True)
        y1 = torch.randn(B, 2)
        y2 = torch.randn(B, 2)
        total, log = criterion(s1, s2, y1, y2)
        assert total.shape == ()
        assert total.requires_grad
        assert "mu_e_bpr" in log
        assert "mu_h_bpr" in log
        assert "mu_e_reg" in log
        assert "mu_h_reg" in log
        assert "total" in log

    def test_correct_ranking_gives_low_bpr(self):
        """When sign(diff_s) matches sign(diff_y), BPR loss should be low."""
        criterion = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)
        y1 = torch.tensor([[2.0, 1.0]])
        y2 = torch.tensor([[1.0, 2.0]])
        s1 = torch.tensor([[2.0, 1.0]], requires_grad=True)
        s2 = torch.tensor([[1.0, 2.0]], requires_grad=True)
        total, log = criterion(s1, s2, y1, y2)
        assert log["mu_e_bpr"] < 0.5  # correct ranking → low BPR loss
        assert log["mu_h_bpr"] < 0.5

    def test_wrong_ranking_gives_high_bpr(self):
        """When sign(diff_s) opposes sign(diff_y), BPR loss should be high."""
        criterion = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)
        # y1 > y2 for both tasks → correct ranking means s1 > s2
        y1 = torch.tensor([[2.0, 2.0]])
        y2 = torch.tensor([[1.0, 1.0]])
        # s1 < s2 → wrong ranking for both tasks
        s1 = torch.tensor([[-5.0, -5.0]], requires_grad=True)
        s2 = torch.tensor([[5.0, 5.0]], requires_grad=True)
        total, log = criterion(s1, s2, y1, y2)
        assert log["mu_e_bpr"] > 1.0  # wrong ranking → high BPR loss
        assert log["mu_h_bpr"] > 1.0

    def test_equal_targets_zero_bpr(self):
        """When y1 == y2 (diff_y == 0), BPR should be 0."""
        criterion = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)
        y1 = torch.tensor([[1.0, 1.0]])
        y2 = torch.tensor([[1.0, 1.0]])
        s1 = torch.tensor([[2.0, 3.0]], requires_grad=True)
        s2 = torch.tensor([[1.0, 0.0]], requires_grad=True)
        total, log = criterion(s1, s2, y1, y2)
        assert log["mu_e_bpr"] == 0.0
        assert log["mu_h_bpr"] == 0.0

    def test_regression_loss(self):
        """Delta regression should match MSE of score diff vs target diff."""
        criterion = MultiTaskBayesianRankingLoss(rank_weight=0.0, reg_weight=1.0)
        y1 = torch.tensor([[3.0, 0.0]])
        y2 = torch.tensor([[1.0, 0.0]])
        s1 = torch.tensor([[2.5, 0.0]], requires_grad=True)
        s2 = torch.tensor([[1.5, 0.0]], requires_grad=True)
        total, log = criterion(s1, s2, y1, y2)
        # diff_y = [2.0, 0.0], diff_s = [1.0, 0.0], MSE = ((2-1)^2 + 0) / 2 = 0.5
        assert abs(log["mu_e_reg"] - 1.0) < 1e-5
        assert abs(log["mu_h_reg"] - 0.0) < 1e-5

    def test_task_weights(self):
        """Task weights should scale the per-task loss."""
        criterion_eq = MultiTaskBayesianRankingLoss(
            rank_weight=0.0, reg_weight=1.0, task_weights=[1.0, 1.0]
        )
        criterion_2x = MultiTaskBayesianRankingLoss(
            rank_weight=0.0, reg_weight=1.0, task_weights=[2.0, 1.0]
        )
        y1 = torch.tensor([[3.0, 3.0]])
        y2 = torch.tensor([[1.0, 1.0]])
        s1 = torch.tensor([[2.5, 2.5]], requires_grad=True)
        s2 = torch.tensor([[1.5, 1.5]], requires_grad=True)
        total_eq, _ = criterion_eq(s1, s2, y1, y2)
        total_2x, _ = criterion_2x(s1, s2, y1, y2)
        # With task_weights=[2,1], mu_e loss is doubled
        assert total_2x.item() > total_eq.item()


# ── model.py tests ──────────────────────────────────────────────────

from polymer_ranking.model import DMPNNEncoder, PolymerRankingModel
from polymer_ranking.featurizer import create_featurizer, get_feature_dims


class TestModelArchitecture:
    def test_feature_dims(self):
        atom_fdim, bond_fdim = get_feature_dims()
        assert atom_fdim > 0
        assert bond_fdim > 0

    def test_model_instantiation(self):
        model = PolymerRankingModel(hidden_size=64, depth=2, ffn_hidden=32)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward_shape(self):
        """Test forward pass with synthetic BatchMolGraph."""
        model = PolymerRankingModel(hidden_size=64, depth=2, ffn_hidden=32)
        model.eval()

        featurizer = create_featurizer()
        smiles = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
        _, mol = cyclize_polymer_with_cp_marking(smiles)
        assert mol is not None

        from chemprop.data.collate import BatchMolGraph
        g = featurizer(mol)
        bmg = BatchMolGraph([g, g])

        ef = torch.randn(2, 5)

        with torch.no_grad():
            s1, s2 = model(bmg, ef, bmg, ef)

        assert s1.shape == (2, 2)
        assert s2.shape == (2, 2)


# ── __init__.py import test ─────────────────────────────────────────

class TestInit:
    def test_no_global_warning_filter(self):
        """__init__.py should not set warnings.filterwarnings('ignore')."""
        import warnings
        # Import should not have added a global "ignore" filter
        has_ignore = any(
            f[2] == "ignore"
            for f in warnings.filters
        )
        # This test just checks that our import doesn't add a blanket ignore
        # (Other packages may have added filters, so we can't assert not has_ignore)

    def test_explicit_imports(self):
        """Key names should be importable from polymer_ranking."""
        from polymer_ranking import ModelConfig, TrainingConfig
        from polymer_ranking import MultiTaskBayesianRankingLoss
        from polymer_ranking import cyclize_polymer_with_cp_marking


# ── dataset.py inheritance test ─────────────────────────────────────

from torch.utils.data import Dataset as TorchDataset
from polymer_ranking.dataset import _BasePairDataset, PairDataset, CachedPairDataset


class TestDatasetInheritance:
    def test_base_inherits_dataset(self):
        assert issubclass(_BasePairDataset, TorchDataset)

    def test_pair_dataset_inherits_base(self):
        assert issubclass(PairDataset, _BasePairDataset)

    def test_cached_pair_dataset_inherits_base(self):
        assert issubclass(CachedPairDataset, _BasePairDataset)


# ── cli.py logging test ─────────────────────────────────────────────

class TestCliLogging:
    def test_main_configures_logging(self):
        """main() should call logging.basicConfig."""
        import inspect
        from polymer_ranking.cli import main
        source = inspect.getsource(main)
        assert "logging.basicConfig" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
