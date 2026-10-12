"""Constants and configuration dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

# Constants
EXTRA_COLS = [
    "conjugation_{s}",
    "Isomer_{s}",
    "CentroSymmetry_{s}",
    "E_LUMO (eV)_{s}",
    "E_HOMO (eV)_{s}",
]
EXTRA_DIM = len(EXTRA_COLS)
TASK_NAMES = ["mu_e", "mu_h"]
NUM_TASKS = len(TASK_NAMES)
CP_FEATURE_DIM = 1

# D-MPNN message passing depth (mirrors ModelConfig.depth)
DEFAULT_DEPTH = 6
# Upper bound on how many repeat units are chained when a unit is too short
OLIGOMER_MAX_REPEATS = 8

# mobilities are stored on a log10 scale; a mobility of 0 is left-censored
# (below the detection limit), so its log value is filled with a sentinel that
# lies strictly below every measured value. The sentinel is only used to get the
# *direction* of the comparison right; the numeric difference is never used as a
# regression target (see ``ok_{task}_{side}``).
CENSOR_LOG_MARGIN = 1.0

SPLIT_MODES = ("pair", "material", "group")
DEVICE_MODES = ("auto", "cpu", "cuda")


def min_span_for_depth(depth: int = DEFAULT_DEPTH) -> int:
    """
    Minimum backbone span L required between the two attachment atoms.

    A D-MPNN with ``depth`` message passing steps spreads information at most
    ``depth`` bonds per step. Cyclization collapses the span L of a repeat unit
    to a single bond, so that artificial shortcut is only harmless when the real
    span satisfies ``L >= 2 * depth + 1`` — otherwise the unit is expanded into
    an oligomer first.
    """
    return 2 * depth + 1


# Configuration dataclasses
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 300
    depth: int = DEFAULT_DEPTH
    dropout: float = 0.1
    ffn_hidden: int = 256
    extra_dim: int = EXTRA_DIM
    num_tasks: int = NUM_TASKS
    aggregation: str = "mean"

    def __post_init__(self):
        valid = {"mean", "sum", "norm"}
        if self.aggregation not in valid:
            raise ValueError(
                f"aggregation must be one of {sorted(valid)}, got {self.aggregation!r}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.ffn_hidden < 1:
            raise ValueError(f"ffn_hidden must be >= 1, got {self.ffn_hidden}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {self.num_tasks}")
        if self.extra_dim < 0:
            raise ValueError(f"extra_dim must be >= 0, got {self.extra_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{
            k: v
            for k, v in d.items() if k in cls.__dataclass_fields__
        })


@dataclass
class TrainingConfig:
    """Training configuration"""
    csv_path: str = "contrastive_paired.csv"
    save_dir: str = "checkpoints"
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 30
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    # Max repeat units used when a unit is too short for cyclization
    max_repeats: int = OLIGOMER_MAX_REPEATS
    # loss weighting, measured on this dataset with
    # ``hyperparam_search.py --stage ablation`` (2 splits per setting):
    #   rank/reg  1.0/0.0 -> 0.756 | 0.9/0.1 -> 0.776 | 0.8/0.2 -> 0.755
    #   censored  0.0 -> 0.713 | 0.5 -> 0.752 | 1.0 -> 0.773   (validation pair acc)
    # the ranking term is the objective; a small regression term only anchors
    # the scale of the scores
    rank_weight: float = 0.9
    reg_weight: float = 0.1
    # BPR weight of pairs that contain a censored value (mobility = 0, i.e.
    # below the detection limit). The direction of such a pair is *certain*
    # (measured > below-detection-limit), only the magnitude is unknown, so the
    # full weight is justified — and measured to be the best.
    censored_weight: float = 1.0
    # scale of the regression target; None -> estimated from the training set
    delta_scale: Optional[float] = None
    # what early stopping / LR scheduling monitor
    early_stop_metric: str = "pair_acc"   # "pair_acc" | "loss"
    scheduler: str = "plateau"            # "plateau" | "cosine"
    # how train/val/test are carved out:
    #   "pair"     -> random split over pairs (materials re-occur across splits)
    #   "material" -> material-disjoint split (no material shared between
    #                 train and val/test); a stricter, extrapolation-flavoured
    #                 estimate that is useful as a leakage diagnostic
    #   "group"    -> paper-disjoint split by doi (all pairs of one paper stay
    #                 in a single split); guards against paper-level bias such
    #                 as shared device fab / measurement setup
    split_by: str = "pair"
    # device used for training ("auto" = cuda when available)
    device: str = "auto"
    # Drop rows whose (unordered) material pair already appeared earlier in the
    # file. Duplicate comparisons can land on both sides of a random split and
    # inflate the reported pairwise accuracy.
    drop_duplicate_pairs: bool = False

    def __post_init__(self):
        if not 0 < self.test_ratio < 1:
            raise ValueError(f"test_ratio must be in (0, 1), got {self.test_ratio}")
        if not 0 <= self.val_ratio < 1 - self.test_ratio:
            raise ValueError(
                f"val_ratio must be in [0, {1 - self.test_ratio:.2f}), got {self.val_ratio}"
            )
        if self.early_stop_metric not in ("pair_acc", "loss"):
            raise ValueError(
                f"early_stop_metric must be 'pair_acc' or 'loss', "
                f"got {self.early_stop_metric!r}"
            )
        if self.scheduler not in ("plateau", "cosine"):
            raise ValueError(
                f"scheduler must be 'plateau' or 'cosine', got {self.scheduler!r}")
        if self.split_by not in SPLIT_MODES:
            raise ValueError(
                f"split_by must be one of {list(SPLIT_MODES)}, got {self.split_by!r}")
        if self.device not in DEVICE_MODES:
            raise ValueError(
                f"device must be one of {list(DEVICE_MODES)}, got {self.device!r}")
        # structural validation: anything *representable* (a very small lr, no
        # weight decay, ...) is left to ``hyperparam_search.py --stage check``,
        # which turns it into warning + suggestion instead of refusing to run.
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.max_repeats < 1:
            raise ValueError(f"max_repeats must be >= 1, got {self.max_repeats}")
        if not self.lr > 0:
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if self.rank_weight < 0 or self.reg_weight < 0:
            raise ValueError(
                f"rank_weight/reg_weight must be >= 0, got "
                f"{self.rank_weight!r}/{self.reg_weight!r}")
        if self.rank_weight + self.reg_weight <= 0:
            raise ValueError(
                "rank_weight + reg_weight must be > 0; otherwise both loss terms "
                "are switched off")
        if not 0.0 <= self.censored_weight <= 1.0:
            raise ValueError(
                f"censored_weight must be in [0, 1], got {self.censored_weight!r}")
        if self.delta_scale is not None and not self.delta_scale > 0:
            raise ValueError(
                f"delta_scale must be > 0 or None (auto), got {self.delta_scale!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration"""
    csv_path: str = "contrastive_paired.csv"
    checkpoint_path: str = "checkpoints/best_model.safetensors"
    save_dir: str = "checkpoints"
    finetune_epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-6
    seed: int = 42
    max_repeats: int = OLIGOMER_MAX_REPEATS
    # hold out a validation split so fine-tuning is monitored instead of blind
    val_ratio: float = 0.1
    patience: int = 10
    device: str = "auto"
    drop_duplicate_pairs: bool = False

    def __post_init__(self):
        if not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if self.finetune_epochs < 1:
            raise ValueError(
                f"finetune_epochs must be >= 1, got {self.finetune_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.max_repeats < 1:
            raise ValueError(f"max_repeats must be >= 1, got {self.max_repeats}")
        if not self.lr > 0:
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if self.device not in DEVICE_MODES:
            raise ValueError(
                f"device must be one of {list(DEVICE_MODES)}, got {self.device!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictConfig:
    """Prediction configuration"""
    predict_csv: str = ""
    checkpoint_path: str = "checkpoints/best_model.safetensors"
    output_path: str = "predictions.csv"
    max_repeats: int = OLIGOMER_MAX_REPEATS
    batch_size: int = 32
    device: str = "auto"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_repeats < 1:
            raise ValueError(f"max_repeats must be >= 1, got {self.max_repeats}")
        if self.device not in DEVICE_MODES:
            raise ValueError(
                f"device must be one of {list(DEVICE_MODES)}, got {self.device!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_device(device: str = "auto") -> "torch.device":  # noqa: F821
    """Turn a ``DEVICE_MODES`` string into a ``torch.device``.

    ``auto`` picks CUDA when it is available; asking for CUDA explicitly on a
    machine without a GPU is an error rather than a silent fallback, so a typo
    in the config cannot hide a CPU-only run.
    """
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "device='cuda' was requested but torch.cuda.is_available() is False")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


_DEFAULT_DEVICE: Optional["torch.device"] = None  # noqa: F821


def set_default_device(device: str = "auto") -> "torch.device":  # noqa: F821
    """Set the process-wide device used by training / fine-tuning / prediction.

    Modules must read :func:`get_default_device` instead of capturing a
    ``torch.device`` at import time, otherwise ``--device`` could not take
    effect after the module graph is loaded.
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = resolve_device(device)
    return _DEFAULT_DEVICE


def get_default_device() -> "torch.device":  # noqa: F821
    """The device configured via :func:`set_default_device` (auto-detected once)."""
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        _DEFAULT_DEVICE = resolve_device("auto")
    return _DEFAULT_DEVICE
