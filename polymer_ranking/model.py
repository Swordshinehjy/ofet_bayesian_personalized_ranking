"""Model definitions: DMPNNEncoder + PolymerRankingModel."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from chemprop.data.collate import BatchMolGraph
    from chemprop.nn import BondMessagePassing, MeanAggregation, SumAggregation, NormAggregation
except ImportError:
    pass

from .config import EXTRA_DIM, NUM_TASKS
from .featurizer import get_feature_dims


class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) encoder.
    Uses chemprop v2 BondMessagePassing module.
    Output: [B, hidden_size]
    """

    def __init__(self,
                 hidden_size: int = 300,
                 depth: int = 6,
                 dropout: float = 0.1,
                 aggregation: str = "mean",
                 d_v: Optional[int] = None,
                 d_e: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation

        if d_v is None or d_e is None:
            d_v, d_e = get_feature_dims()

        self.mpnn = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=hidden_size,
            depth=depth,
            dropout=dropout,
        )
        if aggregation == "mean":
            self.agg = MeanAggregation()
        elif aggregation == "sum":
            self.agg = SumAggregation()
        else:
            self.agg = NormAggregation()

    def forward(self, batch: BatchMolGraph) -> torch.Tensor:
        device = next(self.parameters()).device
        batch.to(device)
        H = self.mpnn(batch)
        mol_vecs = self.agg(H, batch.batch)
        return mol_vecs


class PolymerRankingModel(nn.Module):
    """
    Architecture:
        SMILES ──► D-MPNN ──► mol_emb [H]
                                       ├─ cat ──► FFN ──► [score_e, score_h]
        extra_features [5] ────────────┘

    Two polymers share the same parameters (siamese network).
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 6,
        dropout: float = 0.1,
        ffn_hidden: int = 256,
        extra_dim: int = EXTRA_DIM,
        num_tasks: int = NUM_TASKS,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.mpnn = DMPNNEncoder(hidden_size, depth, dropout, aggregation)

        ffn_in = hidden_size + extra_dim
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, ffn_hidden),
            nn.LayerNorm(ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, ffn_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(ffn_hidden // 2, num_tasks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        mol_graphs: BatchMolGraph,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, num_tasks] score vector."""
        emb = self.mpnn(mol_graphs)
        x = torch.cat([emb, extra], dim=-1)
        return self.ffn(x)

    def forward(
        self,
        mg1: BatchMolGraph,
        ef1: torch.Tensor,
        mg2: BatchMolGraph,
        ef2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores_1, scores_2), each [B, T]."""
        return self.encode(mg1, ef1), self.encode(mg2, ef2)
