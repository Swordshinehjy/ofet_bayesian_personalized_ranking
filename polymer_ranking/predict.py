"""Prediction logic: checkpoint loading, single pair prediction, batch prediction."""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from chemprop.data.collate import BatchMolGraph
except ImportError:
    pass

from .config import ModelConfig, TASK_NAMES
from .model import PolymerRankingModel
from .featurizer import create_featurizer
from .chemistry import cyclize_polymer_with_cp_marking, cyclize_df, extra_feat

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[ModelConfig, Any, PolymerRankingModel]:
    """
    Load checkpoint, returns (model_config, scaler, model).
    Eliminates duplicate loading logic in predict_pair / predict_batch / finetune.
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_config = ModelConfig.from_dict(ckpt["config"])
    scaler = ckpt["scaler"]

    model = PolymerRankingModel(**model_config.to_dict()).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model_config, scaler, model


@torch.no_grad()
def predict_pair(
    smiles_1: str,
    smiles_2: str,
    extra_raw_1: np.ndarray,
    extra_raw_2: np.ndarray,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """
    Predict electron/hole mobility ranking for a single new structure pair.

    Parameters
    ----------
    smiles_1/2 : str  — Polymer repeat unit SMILES with * markers
    extra_raw_1/2 : np.ndarray shape [5]
        [conjugation, isomer, centrosymmetry, E_LUMO(eV), E_HOMO(eV)]
    """
    model_config, scaler, model = load_checkpoint(checkpoint_path)
    featurizer = create_featurizer()

    cyc1, mol1 = cyclize_polymer_with_cp_marking(smiles_1)
    cyc2, mol2 = cyclize_polymer_with_cp_marking(smiles_2)
    if mol1 is None or mol2 is None:
        raise ValueError("Cyclization failed, please check SMILES format and * count")

    mg1 = BatchMolGraph([featurizer(mol1)])
    mg2 = BatchMolGraph([featurizer(mol2)])

    ef1 = torch.tensor(scaler.transform(extra_raw_1.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    ef2 = torch.tensor(scaler.transform(extra_raw_2.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)

    s1, s2 = model(mg1, ef1, mg2, ef2)
    s1 = s1.cpu().numpy()[0]
    s2 = s2.cpu().numpy()[0]

    scores_1 = {n: float(s1[i]) for i, n in enumerate(TASK_NAMES)}
    scores_2 = {n: float(s2[i]) for i, n in enumerate(TASK_NAMES)}
    ranking = {
        n: ("polymer_1" if s1[i] > s2[i] else "polymer_2")
        for i, n in enumerate(TASK_NAMES)
    }
    probability = {
        n: float(1.0 / (1.0 + np.exp(-abs(s1[i] - s2[i]))))
        for i, n in enumerate(TASK_NAMES)
    }

    return {
        "scores_1": scores_1,
        "scores_2": scores_2,
        "ranking": ranking,
        "probability": probability,
        "cyc_smiles_1": cyc1,
        "cyc_smiles_2": cyc2,
    }


@torch.no_grad()
def predict_batch(
    df_new: pd.DataFrame,
    checkpoint_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch predict all new polymer pairs in DataFrame.
    df_new must contain Polymer_1, Polymer_2 and 5 extra feature columns (same format as training set).
    """
    model_config, scaler, model = load_checkpoint(checkpoint_path)
    featurizer = create_featurizer()

    df = cyclize_df(df_new)
    valid = df.dropna(subset=["cyc_1", "cyc_2"]).copy()
    logger.info(f"Valid pairs: {len(valid)}/{len(df)}")

    ef1 = scaler.transform(extra_feat(valid, "1"))
    ef2 = scaler.transform(extra_feat(valid, "2"))

    all_s1, all_s2 = [], []
    batch_size = 32
    for i in range(0, len(valid), batch_size):
        end_idx = min(i + batch_size, len(valid))
        batch_df = valid.iloc[i:end_idx]

        g1s = [featurizer(m) for m in batch_df["mol_1"]]
        g2s = [featurizer(m) for m in batch_df["mol_2"]]
        mg1 = BatchMolGraph(g1s)
        mg2 = BatchMolGraph(g2s)

        t1 = torch.tensor(ef1[i:end_idx], dtype=torch.float32).to(DEVICE)
        t2 = torch.tensor(ef2[i:end_idx], dtype=torch.float32).to(DEVICE)
        s1, s2 = model(mg1, t1, mg2, t2)
        all_s1.append(s1.cpu().numpy())
        all_s2.append(s2.cpu().numpy())

    S1 = np.concatenate(all_s1, axis=0)
    S2 = np.concatenate(all_s2, axis=0)

    for i, n in enumerate(TASK_NAMES):
        valid[f"score_{n}_1"] = S1[:, i]
        valid[f"score_{n}_2"] = S2[:, i]
        valid[f"prob_{n}"] = 1.0 / (1.0 + np.exp(-np.abs(S1[:, i] - S2[:, i])))
        valid[f"preferred_{n}"] = np.where(S1[:, i] > S2[:, i],
                                           valid["Materials_1"],
                                           valid["Materials_2"])

    output_cols = [c for c in valid.columns if c not in ("mol_1", "mol_2")]
    if output_path:
        valid[output_cols].to_csv(output_path, index=False)
        logger.info(f"Predictions saved → {output_path}")
    return valid[output_cols]
