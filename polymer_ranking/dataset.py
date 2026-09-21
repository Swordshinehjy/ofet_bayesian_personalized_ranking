"""Dataset classes and collate functions."""

from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

from .featurizer import create_featurizer
from .chemistry import extra_feat
from .config import TASK_NAMES

OK_COLS = [f"ok_{t}_{{s}}" for t in TASK_NAMES]


def _ok_flags(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Validity flags [N, T]: True when that mobility was measured (> 0).

    Older / hand-built DataFrames may lack the ``ok_*`` columns produced by
    :func:`load_and_preprocess`; in that case everything is treated as measured
    so the behaviour matches the pre-censoring code path.
    """
    cols = [c.format(s=suffix) for c in OK_COLS]
    if not all(c in df.columns for c in cols):
        return np.ones((len(df), len(TASK_NAMES)), dtype=bool)
    return df[cols].fillna(False).values.astype(bool)


class _BasePairDataset(Dataset):
    """Base class encapsulating featurization / standardization / target extraction logic."""

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.featurizer = create_featurizer()

        self.graphs1: List[MolGraph] = []
        self.graphs2: List[MolGraph] = []
        valid_indices: List[int] = []
        for idx in range(len(self.df)):
            mol1 = self.df.loc[idx, "mol_1"]
            mol2 = self.df.loc[idx, "mol_2"]
            g1 = self.featurizer(mol1) if mol1 else None
            g2 = self.featurizer(mol2) if mol2 else None
            if g1 is not None and g2 is not None:
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                valid_indices.append(idx)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        ef1 = extra_feat(self.df, "1")
        ef2 = extra_feat(self.df, "2")

        if fit_scaler:
            ef_all = np.vstack([ef1, ef2])
            self.scaler = StandardScaler().fit(ef_all)
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2

        self.y1 = self.df[[f"log_{t}_1" for t in TASK_NAMES]].values.astype(np.float32)
        self.y2 = self.df[[f"log_{t}_2" for t in TASK_NAMES]].values.astype(np.float32)

        # A mobility of 0 is left-censored (below the detection limit): the
        # ordering is still decidable, the numeric difference is not.
        self.ok1 = _ok_flags(self.df, "1")
        self.ok2 = _ok_flags(self.df, "2")

    def __len__(self):
        return len(self.df)


class PairDataset(_BasePairDataset):
    """Each sample = a pair of polymers (mol1, mol2)."""

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        super().__init__(df, scaler, fit_scaler)

    def __getitem__(self, idx):
        return (
            self.graphs1[idx],
            self.graphs2[idx],
            torch.tensor(self.ef1[idx]),
            torch.tensor(self.ef2[idx]),
            torch.tensor(self.y1[idx]),
            torch.tensor(self.y2[idx]),
            torch.tensor(self.ok1[idx]),
            torch.tensor(self.ok2[idx]),
        )


def collate_fn(batch):
    g1s, g2s, ef1s, ef2s, y1s, y2s, ok1s, ok2s = zip(*batch)
    g1s, g2s = list(g1s), list(g2s)
    if any(g is None for g in g1s) or any(g is None for g in g2s):
        raise ValueError("Found None in graphs")
    return (
        BatchMolGraph(g1s),
        BatchMolGraph(g2s),
        torch.stack(ef1s),
        torch.stack(ef2s),
        torch.stack(y1s),
        torch.stack(y2s),
        torch.stack(ok1s),
        torch.stack(ok2s),
    )


class CachedPairDataset(_BasePairDataset):
    """Dataset with pre-built BatchMolGraph, suitable for shuffle=False scenarios."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.batch_size = batch_size
        super().__init__(df, scaler, fit_scaler)
        self._build_cached_batches()

    def _build_cached_batches(self):
        self._cached_batches = []
        n = len(self.df)
        for i in range(0, n, self.batch_size):
            end_idx = min(i + self.batch_size, n)
            batch_indices = list(range(i, end_idx))

            g1s = [self.graphs1[idx] for idx in batch_indices]
            g2s = [self.graphs2[idx] for idx in batch_indices]

            bmg1 = BatchMolGraph(g1s)
            bmg2 = BatchMolGraph(g2s)

            ef1_batch = torch.tensor(self.ef1[batch_indices])
            ef2_batch = torch.tensor(self.ef2[batch_indices])
            y1_batch = torch.tensor(self.y1[batch_indices])
            y2_batch = torch.tensor(self.y2[batch_indices])
            ok1_batch = torch.tensor(self.ok1[batch_indices])
            ok2_batch = torch.tensor(self.ok2[batch_indices])

            self._cached_batches.append(
                (bmg1, bmg2, ef1_batch, ef2_batch, y1_batch, y2_batch,
                 ok1_batch, ok2_batch))

        # Release individual MolGraph lists — already packed into BatchMolGraph
        del self.graphs1, self.graphs2

    def __len__(self):
        return len(self._cached_batches)

    def __getitem__(self, idx):
        return self._cached_batches[idx]


def collate_cached_batch(batch):
    """Collate function for CachedPairDataset, returns the pre-built batch directly.

    ``CachedPairDataset`` must be used with ``DataLoader(batch_size=1)``: a
    cached item is already a whole batch, and merging several of them would
    require re-building the ``BatchMolGraph`` (which the previous
    implementation silently faked by dropping all but the first graph).
    """
    if len(batch) != 1:
        raise ValueError(
            "CachedPairDataset must be used with DataLoader(batch_size=1); "
            f"got a collated batch of {len(batch)} items"
        )
    return batch[0]
