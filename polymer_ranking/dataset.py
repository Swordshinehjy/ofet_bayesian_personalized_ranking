"""Dataset classes and collate functions."""

from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

try:
    from chemprop.data.molgraph import MolGraph
    from chemprop.data.collate import BatchMolGraph
except ImportError:
    pass

from .featurizer import create_featurizer
from .chemistry import extra_feat
from .config import TASK_NAMES


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

        log_cols = [f"log_{t}_{s}" for t in TASK_NAMES for s in ("1", "2")]
        self.y1 = self.df[[f"log_{t}_1" for t in TASK_NAMES]].values.astype(np.float32)
        self.y2 = self.df[[f"log_{t}_2" for t in TASK_NAMES]].values.astype(np.float32)

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
        )


def collate_fn(batch):
    g1s, g2s, ef1s, ef2s, y1s, y2s = zip(*batch)
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

            self._cached_batches.append(
                (bmg1, bmg2, ef1_batch, ef2_batch, y1_batch, y2_batch))

        # Release individual MolGraph lists — already packed into BatchMolGraph
        del self.graphs1, self.graphs2

    def __len__(self):
        return len(self._cached_batches)

    def __getitem__(self, idx):
        return self._cached_batches[idx]


def collate_cached_batch(batch):
    """Collate function for CachedPairDataset, returns pre-built batch directly."""
    if len(batch) == 1:
        return batch[0]
    bmg1s, bmg2s, ef1s, ef2s, y1s, y2s = zip(*batch)
    return (
        bmg1s[0] if len(bmg1s) == 1 else bmg1s,
        bmg2s[0] if len(bmg2s) == 1 else bmg2s,
        ef1s[0] if len(ef1s) == 1 else torch.cat(ef1s),
        ef2s[0] if len(ef2s) == 1 else torch.cat(ef2s),
        y1s[0] if len(y1s) == 1 else torch.cat(y1s),
        y2s[0] if len(y2s) == 1 else torch.cat(y2s),
    )
