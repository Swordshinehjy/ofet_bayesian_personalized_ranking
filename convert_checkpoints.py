"""One-time conversion of legacy .pt checkpoints to .safetensors.

Loads each existing checkpoint with torch.load, re-writes it as a single
safetensors file (weights + JSON metadata), then verifies the result matches
the original exactly. No retraining involved.

Usage:
    python convert_checkpoints.py            # convert checkpoints/*.pt
    python convert_checkpoints.py path.pt    # convert specific file(s)
"""

import sys
from pathlib import Path

import numpy as np
import torch

from polymer_ranking.checkpoint import save_checkpoint, load_checkpoint_dict

# Everything that is not one of these is carried over verbatim as JSON
# metadata (delta_scale / loss_config / split_by / extra_dim / history ...).
RESERVED = ("model_state", "scaler", "config")


def convert(pt_path: Path) -> Path:
    print(f"=== Converting {pt_path} ===")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    extra = {k: v for k, v in ckpt.items() if k not in RESERVED}
    out_path = save_checkpoint(
        pt_path.with_suffix(".safetensors"),
        model_state=ckpt["model_state"],
        scaler=ckpt.get("scaler"),
        config=ckpt.get("config"),
        extra=extra,
    )

    # --- verify round-trip ---
    loaded = load_checkpoint_dict(out_path)
    assert set(loaded["model_state"]) == set(ckpt["model_state"]), \
        "state_dict keys differ"
    n_params = 0
    for k, v0 in ckpt["model_state"].items():
        v1 = loaded["model_state"][k]
        assert v0.shape == v1.shape and v0.dtype == v1.dtype, \
            f"{k}: shape/dtype differ"
        assert torch.equal(v0, v1), f"{k}: tensor values differ"
        n_params += v0.numel()
    assert loaded["config"] == ckpt["config"], "config differs"
    for k, v0 in extra.items():
        assert loaded[k] == v0, f"extra[{k}] differs"
    if ckpt.get("scaler") is not None:
        s0, s1 = ckpt["scaler"], loaded["scaler"]
        for attr in ("mean_", "var_", "scale_"):
            assert np.array_equal(getattr(s0, attr), getattr(s1, attr)), \
                f"scaler.{attr} differs"
        X = np.random.default_rng(0).normal(size=(20, s0.n_features_in_))
        assert np.allclose(s0.transform(X), s1.transform(X)), \
            "scaler transform differs"

    print(f"OK: {out_path} | {len(ckpt['model_state'])} tensors, "
          f"{n_params:,} params | extra keys: {sorted(extra)}")
    return out_path


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        Path("checkpoints").glob("*.pt"))
    if not targets:
        print("No .pt checkpoints found.")
        return
    for t in targets:
        convert(t)
    print("\nDone. Original .pt files kept as backup; safe to delete manually.")


if __name__ == "__main__":
    main()
