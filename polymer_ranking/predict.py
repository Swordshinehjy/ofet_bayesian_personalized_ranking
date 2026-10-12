"""Prediction logic: checkpoint loading, single pair prediction, batch prediction."""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from chemprop.data.collate import BatchMolGraph

from .config import (
    ModelConfig,
    TASK_NAMES,
    OLIGOMER_MAX_REPEATS,
    get_default_device,
    set_default_device,
)
from .model import PolymerRankingModel
from .featurizer import create_featurizer
from .chemistry import cyclize_polymer_with_cp_marking, cyclize_df, extra_feat
from .checkpoint import load_checkpoint_dict

logger = logging.getLogger(__name__)

# columns ``predict_batch`` needs besides the two SMILES columns
REQUIRED_COLUMNS = ("Polymer_1", "Polymer_2")
# columns used only to label the winner; missing ones degrade gracefully
LABEL_COLUMNS = ("Materials_1", "Materials_2")


def _require(ckpt: Dict[str, Any], key: str, path: str):
    """Fetch a required checkpoint field with an actionable error message."""
    if key not in ckpt:
        raise ValueError(
            f"{path}: checkpoint metadata is missing {key!r}. The file may have "
            f"been truncated or hand-edited; re-export it from training "
            f"(available keys: {sorted(ckpt)})")
    return ckpt[key]


def load_checkpoint(
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[ModelConfig, Any, PolymerRankingModel, Dict[str, Any]]:
    """
    Load checkpoint, returns (model_config, scaler, model, meta).
    Eliminates duplicate loading logic in predict_pair / predict_batch / finetune.

    ``meta`` carries the training-time settings the loss depends on
    (``delta_scale``, ``loss_config``) so fine-tuning resumes with exactly the
    same objective instead of silently falling back to the defaults.

    Reads safetensors + JSON metadata (no pickle).
    """
    set_default_device(device)
    dev = get_default_device()
    ckpt = load_checkpoint_dict(checkpoint_path, device=dev)
    model_config = ModelConfig.from_dict(_require(ckpt, "config", checkpoint_path))
    scaler = _require(ckpt, "scaler", checkpoint_path)

    model = PolymerRankingModel(**model_config.to_dict()).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model_config, scaler, model, ckpt


@torch.no_grad()
def predict_pair(
    smiles_1: str,
    smiles_2: str,
    extra_raw_1: np.ndarray,
    extra_raw_2: np.ndarray,
    checkpoint_path: str,
    max_repeats: int = OLIGOMER_MAX_REPEATS,
    bundle: Optional[Tuple[ModelConfig, Any, PolymerRankingModel, Dict]] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Predict electron/hole mobility ranking for a single new structure pair.

    Parameters
    ----------
    smiles_1/2 : str  — Polymer repeat unit SMILES with * markers
    extra_raw_1/2 : np.ndarray shape [5]
        [conjugation, isomer, centrosymmetry, E_LUMO(eV), E_HOMO(eV)]
    bundle : optional
        Result of a previous :func:`load_checkpoint` call. Pass it when scoring
        many pairs in a loop — otherwise every call re-reads the checkpoint from
        disk and rebuilds the model.
    device : str
        ``auto`` / ``cpu`` / ``cuda``.
    """
    if bundle is None:
        bundle = load_checkpoint(checkpoint_path, device=device)
    model_config, scaler, model, _ = bundle
    if not isinstance(model_config, ModelConfig):  # pragma: no cover - defensive
        raise TypeError("bundle must come from load_checkpoint()")
    dev = get_default_device()
    featurizer = create_featurizer()

    cyc1, mol1 = cyclize_polymer_with_cp_marking(smiles_1,
                                                 depth=model_config.depth,
                                                 max_repeats=max_repeats)
    cyc2, mol2 = cyclize_polymer_with_cp_marking(smiles_2,
                                                 depth=model_config.depth,
                                                 max_repeats=max_repeats)
    if mol1 is None or mol2 is None:
        raise ValueError(
            f"Cyclization failed for {smiles_1!r} / {smiles_2!r}; check the "
            f"SMILES format and that each unit carries exactly two '*'")

    mg1 = BatchMolGraph([featurizer(mol1)])
    mg2 = BatchMolGraph([featurizer(mol2)])

    ef1 = torch.tensor(scaler.transform(extra_raw_1.reshape(1, -1)),
                       dtype=torch.float32).to(dev)
    ef2 = torch.tensor(scaler.transform(extra_raw_2.reshape(1, -1)),
                       dtype=torch.float32).to(dev)

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
    max_repeats: int = OLIGOMER_MAX_REPEATS,
    batch_size: int = 32,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Batch predict all new polymer pairs in DataFrame.

    ``df_new`` must contain ``Polymer_1`` / ``Polymer_2`` and the 5 extra feature
    columns (same format as the training set). ``Materials_1`` / ``Materials_2``
    are used to label the preferred material; when they are absent the labels
    fall back to ``polymer_1`` / ``polymer_2`` instead of failing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df_new.columns]
    if missing:
        raise ValueError(
            f"{missing} column(s) required for prediction are missing from the "
            f"input; available columns: {list(df_new.columns)}")

    bundle = load_checkpoint(checkpoint_path, device=device)
    model_config, scaler, model, _ = bundle
    dev = get_default_device()
    featurizer = create_featurizer()

    df = cyclize_df(df_new, depth=model_config.depth, max_repeats=max_repeats)
    valid = df.dropna(subset=["cyc_1", "cyc_2"]).copy()
    logger.info(f"Valid pairs: {len(valid)}/{len(df)}")
    if valid.empty:
        raise ValueError(
            "No valid polymer pairs after cyclization; "
            "check Polymer_1/Polymer_2 SMILES (each must contain exactly two '*')"
        )
    if not all(c in valid.columns for c in LABEL_COLUMNS):
        logger.warning(
            f"{list(LABEL_COLUMNS)} not found in the input; 'preferred_*' will be "
            f"reported as polymer_1/polymer_2")

    ef1 = scaler.transform(extra_feat(valid, "1"))
    ef2 = scaler.transform(extra_feat(valid, "2"))

    all_s1, all_s2 = [], []
    for i in range(0, len(valid), batch_size):
        end_idx = min(i + batch_size, len(valid))
        batch_df = valid.iloc[i:end_idx]

        g1s = [featurizer(m) for m in batch_df["mol_1"]]
        g2s = [featurizer(m) for m in batch_df["mol_2"]]
        mg1 = BatchMolGraph(g1s)
        mg2 = BatchMolGraph(g2s)

        t1 = torch.tensor(ef1[i:end_idx], dtype=torch.float32).to(dev)
        t2 = torch.tensor(ef2[i:end_idx], dtype=torch.float32).to(dev)
        s1, s2 = model(mg1, t1, mg2, t2)
        all_s1.append(s1.cpu().numpy())
        all_s2.append(s2.cpu().numpy())

    S1 = np.concatenate(all_s1, axis=0)
    S2 = np.concatenate(all_s2, axis=0)

    labels = (valid["Materials_1"].astype(str), valid["Materials_2"].astype(str)) \
        if all(c in valid.columns for c in LABEL_COLUMNS) else ("polymer_1", "polymer_2")

    for i, n in enumerate(TASK_NAMES):
        valid[f"score_{n}_1"] = S1[:, i]
        valid[f"score_{n}_2"] = S2[:, i]
        valid[f"prob_{n}"] = 1.0 / (1.0 + np.exp(-np.abs(S1[:, i] - S2[:, i])))
        valid[f"preferred_{n}"] = np.where(S1[:, i] > S2[:, i], labels[0], labels[1])

    output_cols = [c for c in valid.columns if c not in ("mol_1", "mol_2")]
    if output_path:
        valid[output_cols].to_csv(output_path, index=False)
        logger.info(f"Predictions saved -> {output_path}")
    return valid[output_cols]
