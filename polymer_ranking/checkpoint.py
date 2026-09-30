"""Safe checkpoint I/O.

Model weights are stored in `.safetensors` (no pickle execution on load).
All non-tensor state is stored as JSON metadata inside the same file:

- tensors ``model.<name>``   : model state_dict entries
- metadata ``config``        : ModelConfig dict (JSON)
- metadata ``extra.<key>``   : misc training state, e.g. delta_scale, history
- metadata ``scaler.<attr>`` : StandardScaler state as JSON — ``mean_`` /
  ``var_`` / ``scale_`` arrays as JSON lists (float64 exact round-trip),
  plus ``n_features_in_`` / ``n_samples_seen_`` integer attributes

Why this replaces `torch.save`
------------------------------
The old format pickled the fitted ``StandardScaler`` together with the tensors,
so loading required ``torch.load(..., weights_only=False)`` — i.e. executing
arbitrary code from the file. Nothing here can execute code when read.

Legacy `.pt` checkpoints are still readable, but only when the caller opts in
with ``allow_unsafe_legacy=True``.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from safetensors import safe_open
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "model."
_SCALER_PREFIX = "scaler."
_EXTRA_PREFIX = "extra."
_SCALER_ARRAYS = ("mean_", "var_", "scale_")
CHECKPOINT_FORMAT = "ofet-polymer-ranking"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(obj)!r}")


def save_checkpoint(
    path,
    model_state: Dict[str, torch.Tensor],
    scaler: Optional[StandardScaler] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a checkpoint as a single `.safetensors` file (no pickle)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {
        f"{_MODEL_PREFIX}{k}": v.detach().cpu().contiguous()
        for k, v in model_state.items()
    }
    meta: Dict[str, str] = {"format": CHECKPOINT_FORMAT}
    if config is not None:
        meta["config"] = json.dumps(config, default=_json_default)
    for key, value in (extra or {}).items():
        meta[f"{_EXTRA_PREFIX}{key}"] = json.dumps(value, default=_json_default)
    if scaler is not None:
        for attr in _SCALER_ARRAYS:
            meta[f"{_SCALER_PREFIX}{attr}"] = json.dumps(
                np.asarray(getattr(scaler, attr)).tolist())
        meta[f"{_SCALER_PREFIX}n_features_in_"] = json.dumps(
            int(scaler.n_features_in_))
        meta[f"{_SCALER_PREFIX}n_samples_seen_"] = json.dumps(
            scaler.n_samples_seen_, default=_json_default)

    save_path = p.with_suffix(".safetensors")
    from safetensors.torch import save_file
    save_file(tensors, str(save_path), metadata=meta)
    logger.info(f"Checkpoint saved (safetensors) -> {save_path}")
    return save_path


def load_checkpoint_dict(
    checkpoint_path: str,
    device=None,
    allow_unsafe_legacy: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint into a plain dict.

    Returns ``{"model_state": ..., "config": ..., "scaler": ..., <extra keys>}``
    so existing callers keep working unchanged.
    """
    p = Path(checkpoint_path)
    if p.suffix == ".pt":  # legacy pickle checkpoint
        if not allow_unsafe_legacy:
            raise RuntimeError(
                f"{checkpoint_path} is a legacy pickle-based checkpoint and "
                "cannot be read safely. Convert it with "
                "`python convert_checkpoints.py` (or pass "
                "allow_unsafe_legacy=True only for files you authored: "
                "torch.load(weights_only=False) executes arbitrary code).")
        logger.warning(
            "Loading %s with weights_only=False: this executes the pickle "
            "embedded in the file.", checkpoint_path)
        return torch.load(p, map_location=device or "cpu", weights_only=False)

    device = str(device) if device is not None else "cpu"
    model_state: Dict[str, torch.Tensor] = {}
    with safe_open(str(p), framework="pt", device=device) as f:
        meta = f.metadata() or {}
        for key in f.keys():
            if key.startswith(_MODEL_PREFIX):
                model_state[key[len(_MODEL_PREFIX):]] = f.get_tensor(key)

    ckpt: Dict[str, Any] = {"model_state": model_state}
    if "config" in meta:
        ckpt["config"] = json.loads(meta["config"])
    for mk, mv in meta.items():
        if mk.startswith(_EXTRA_PREFIX):
            ckpt[mk[len(_EXTRA_PREFIX):]] = json.loads(mv)

    if f"{_SCALER_PREFIX}mean_" in meta:
        scaler = StandardScaler()
        for attr in _SCALER_ARRAYS:
            setattr(scaler, attr,
                    np.asarray(json.loads(meta[f"{_SCALER_PREFIX}{attr}"]),
                               dtype=np.float64))
        scaler.n_features_in_ = int(
            json.loads(meta[f"{_SCALER_PREFIX}n_features_in_"]))
        n_seen = json.loads(meta[f"{_SCALER_PREFIX}n_samples_seen_"])
        scaler.n_samples_seen_ = (np.asarray(n_seen) if isinstance(n_seen, list)
                                  else n_seen)
        ckpt["scaler"] = scaler
    return ckpt


def is_safe_format(checkpoint_path: str) -> bool:
    """True when `checkpoint_path` is a safetensors checkpoint."""
    return Path(checkpoint_path).suffix == ".safetensors"
