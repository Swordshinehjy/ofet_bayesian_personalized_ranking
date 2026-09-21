"""Tests for left-censored mobility labels (mu = 0 -> below the detection limit)."""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymer_ranking.chemistry import load_and_preprocess
from polymer_ranking.config import CENSOR_LOG_MARGIN, TASK_NAMES
from polymer_ranking.dataset import (PairDataset, CachedPairDataset, _ok_flags,
                                     collate_cached_batch)

POLY_A = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
POLY_B = "*c1ccc(-c2ccc(-c3ccc(-c4cc5c(s4)-c4cc6c(cc4[Si]5(C)C)-c4sc(*)cc4[Si]6(C)C)s3)c3nsnc23)s1"


def _write_csv(tmp_path: Path, mu_e_1, mu_e_2, mu_h_1, mu_h_2) -> str:
    df = pd.DataFrame({
        "Materials_1": ["A", "B", "C"][:len(mu_e_1)],
        "Materials_2": ["B", "C", "A"][:len(mu_e_1)],
        "Polymer_1": [POLY_A] * len(mu_e_1),
        "Polymer_2": [POLY_B] * len(mu_e_1),
        "conjugation_1": [1] * len(mu_e_1),
        "Isomer_1": [0.0] * len(mu_e_1),
        "CentroSymmetry_1": [0] * len(mu_e_1),
        "E_LUMO (eV)_1": [-3.1] * len(mu_e_1),
        "E_HOMO (eV)_1": [-5.3] * len(mu_e_1),
        "mu_e_1": mu_e_1,
        "mu_h_1": mu_h_1,
        "conjugation_2": [1] * len(mu_e_1),
        "Isomer_2": [0.0] * len(mu_e_1),
        "CentroSymmetry_2": [0] * len(mu_e_1),
        "E_LUMO (eV)_2": [-3.4] * len(mu_e_1),
        "E_HOMO (eV)_2": [-5.2] * len(mu_e_1),
        "mu_e_2": mu_e_2,
        "mu_h_2": mu_h_2,
    })
    path = tmp_path / "pairs.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadAndPreprocessCensoring:
    def test_ok_flags_and_log_columns(self, tmp_path):
        path = _write_csv(tmp_path, [0.1, 0.0, 0.2], [0.01, 0.3, 0.0],
                                    [0.1, 0.0, 0.2], [0.01, 0.3, 0.0])
        df = load_and_preprocess(path)
        for t in TASK_NAMES:
            for s in ("1", "2"):
                assert f"ok_{t}_{s}" in df.columns
                assert f"log_{t}_{s}" in df.columns
        # row 0: both measured; row 1: side 1 censored; row 2: side 2 censored
        assert df["ok_mu_e_1"].tolist() == [True, False, True]
        assert df["ok_mu_e_2"].tolist() == [True, True, False]

    def test_censored_log_lies_below_every_measured_value(self, tmp_path):
        path = _write_csv(tmp_path, [0.1, 0.0, 0.0], [0.01, 0.3, 0.0],
                                    [0.1, 0.0, 0.0], [0.01, 0.3, 0.0])
        df = load_and_preprocess(path)
        floor = float(np.log10(1e-2)) - CENSOR_LOG_MARGIN   # min measured = 0.01
        for t in TASK_NAMES:
            censored = df.loc[~df[f"ok_{t}_1"], f"log_{t}_1"]
            assert len(censored) > 0
            assert censored.eq(floor).all()
            measured = df.loc[df[f"ok_{t}_2"], f"log_{t}_2"]
            assert censored.max() < measured.min()

    def test_ordering_of_censored_vs_measured_is_correct(self, tmp_path):
        """sign(log_measured - log_censored) decides the direction used by BPR."""
        path = _write_csv(tmp_path, [0.0, 0.5], [0.5, 0.0],
                                    [0.0, 0.5], [0.5, 0.0])
        df = load_and_preprocess(path)
        dy = df["log_mu_e_1"] - df["log_mu_e_2"]
        assert dy.iloc[0] < 0   # side 1 censored -> side 2 ranks higher
        assert dy.iloc[1] > 0   # side 2 censored -> side 1 ranks higher

    def test_both_censored_gives_zero_difference(self, tmp_path):
        path = _write_csv(tmp_path, [0.0], [0.0], [0.0], [0.0])
        df = load_and_preprocess(path)
        assert float((df["log_mu_e_1"] - df["log_mu_e_2"]).iloc[0]) == 0.0


class TestDatasetOkFlags:
    def _df(self, tmp_path):
        path = _write_csv(tmp_path, [0.1, 0.0, 0.2], [0.01, 0.3, 0.0],
                                    [0.1, 0.0, 0.2], [0.01, 0.3, 0.0])
        return load_and_preprocess(path)

    def test_pair_dataset_returns_ok_tensors(self, tmp_path):
        ds = PairDataset(self._df(tmp_path), fit_scaler=True)
        item = ds[0]
        assert len(item) == 8
        ok1, ok2 = item[-2], item[-1]
        assert ok1.dtype == torch.bool
        assert ok2.dtype == torch.bool
        assert bool(ok1[0]) is True

    def test_ok_flags_match_the_dataframe(self, tmp_path):
        df = self._df(tmp_path)
        ds = PairDataset(df, fit_scaler=True)
        assert ds.ok1.shape == (len(df), 2)
        assert ds.ok1[:, 0].tolist() == df["ok_mu_e_1"].tolist()
        assert ds.ok2[:, 1].tolist() == df["ok_mu_h_2"].tolist()

    def test_cached_dataset_keeps_ok_flags(self, tmp_path):
        ds = CachedPairDataset(self._df(tmp_path), batch_size=2, fit_scaler=True)
        *_, ok1, ok2 = ds[0]
        assert ok1.shape == (2, 2)
        assert ok2.shape == (2, 2)
        assert ok1.dtype == torch.bool

    def test_missing_ok_columns_default_to_all_measured(self):
        flags = _ok_flags(pd.DataFrame({"a": [1, 2]}), "1")
        assert flags.shape == (2, 2)
        assert flags.all()


class TestCollateCachedBatch:
    def test_rejects_batches_larger_than_one(self, tmp_path):
        path = _write_csv(tmp_path, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                                    [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        ds = CachedPairDataset(load_and_preprocess(path), batch_size=2,
                               fit_scaler=True)
        with pytest.raises(ValueError, match="batch_size=1"):
            collate_cached_batch([ds[0], ds[1]])
