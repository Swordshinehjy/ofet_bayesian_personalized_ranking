"""Training logic: EarlyStopping, epoch runner, metric computation, train, finetune."""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from .config import ModelConfig, TrainingConfig, FinetuneConfig, TASK_NAMES
from .model import PolymerRankingModel
from .loss import MultiTaskBayesianRankingLoss
from .dataset import PairDataset, CachedPairDataset, collate_fn, collate_cached_batch
from .chemistry import load_and_preprocess

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Stop training when the monitored value stops improving.

    mode="min" -> lower is better (e.g. validation loss)
    mode="max" -> higher is better (e.g. pairwise accuracy)
    """

    def __init__(self, patience: int = 20, delta: float = 1e-4, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_loss = self.best_score
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, value: float, model: nn.Module) -> bool:
        if self.mode == "min":
            improved = value < self.best_score - self.delta
        else:
            improved = value > self.best_score + self.delta

        if improved:
            self.best_score = value
            self.best_loss = value
            self.counter = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_delta_scale(y1: np.ndarray, y2: np.ndarray, ok: np.ndarray) -> float:
    """Std of the measured log10 mobility differences.

    Normalizes the delta-regression target so that it stays in the same order of
    magnitude as the BPR term. Pairs with a censored mobility contribute nothing.
    """
    diffs = []
    for t in range(ok.shape[1]):
        m = ok[:, t]
        if m.any():
            diffs.append(y1[m, t] - y2[m, t])
    if not diffs:
        return 1.0
    s = float(np.std(np.concatenate(diffs)))
    return s if s > 1e-6 else 1.0


# ── data splitting ──────────────────────────────────────────────────────────

def _pair_keys(df: pd.DataFrame) -> np.ndarray:
    """Order-insensitive key of a row: the frozenset of its two materials."""
    m1 = df["Materials_1"].astype(str).values if "Materials_1" in df.columns else None
    m2 = df["Materials_2"].astype(str).values if "Materials_2" in df.columns else None
    if m1 is None or m2 is None:
        return np.arange(len(df)).astype(str)
    return np.array([ "||".join(sorted([a, b])) for a, b in zip(m1, m2)])


def _material_split(df, test_ratio: float, val_ratio: float, seed: int):
    """Material-disjoint split: no material occurs in two different splits.

    The *materials* are partitioned first — greedily, in a seeded random order,
    growing the test set until enough pairs are fully covered, then the
    validation set. A pair belongs to a split only when **both** of its
    materials belong to it, so no material (hence no polymer) is ever shared
    between train and val/test.

    Pairs whose two materials land in different splits are dropped; that is the
    price of a strict material-disjoint split, and :func:`prepare_splits` logs
    how many were lost.
    """
    n = len(df)
    m1 = df["Materials_1"].astype(str).values
    m2 = df["Materials_2"].astype(str).values
    rng = np.random.default_rng(seed)
    materials = np.unique(np.concatenate([m1, m2]))
    rng.shuffle(materials)

    def inside(mats: set) -> np.ndarray:
        arr = np.array(sorted(mats), dtype=object)
        if len(arr) == 0:
            return np.zeros(n, dtype=bool)
        return np.isin(m1, arr) & np.isin(m2, arr)

    def grow(target: int, forbidden: set) -> set:
        chosen: set = set()
        for m in materials:
            if str(m) in forbidden:
                continue
            if int(inside(chosen).sum()) >= target:
                break
            chosen.add(str(m))
        return chosen

    te_mats = grow(round(test_ratio * n), forbidden=set())
    va_mats = grow(round(val_ratio * n), forbidden=te_mats)
    tr_mats = {str(m) for m in materials} - te_mats - va_mats

    return (np.flatnonzero(inside(tr_mats)),
            np.flatnonzero(inside(va_mats)),
            np.flatnonzero(inside(te_mats)))

def _random_split(n: int, test_ratio: float, val_ratio: float, seed: int):
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=test_ratio, random_state=seed)
    if val_ratio > 0:
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    else:
        va_idx = np.array([], dtype=int)
    return np.sort(tr_idx), np.sort(va_idx), np.sort(te_idx)


def leakage_report(df, tr_idx, va_idx, te_idx) -> Dict[str, float]:
    """How much information the test split shares with the training split.

    ``material_overlap``
        fraction of test materials that also appear in train.
    ``pair_overlap``
        fraction of test pairs whose (unordered) material pair also occurs in
        train — exact-duplicate leakage.

    The dataset is small and every material is a variant of a handful of known
    backbones, so some overlap is unavoidable; the numbers are diagnostics, not
    pass/fail criteria.
    """
    out: Dict[str, float] = {}
    if "Materials_1" not in df.columns or len(te_idx) == 0:
        return out
    m1 = df["Materials_1"].astype(str).values
    m2 = df["Materials_2"].astype(str).values

    def mats(idx):
        return set(m1[idx]) | set(m2[idx]) if len(idx) else set()

    tr_m, te_m = mats(tr_idx), mats(te_idx)
    out["test_material_overlap"] = round(
        len(te_m & tr_m) / max(len(te_m), 1), 3)

    keys = _pair_keys(df)
    tr_k, te_k = set(keys[tr_idx]), set(keys[te_idx])
    out["test_pair_overlap"] = round(len(te_k & tr_k) / max(len(te_k), 1), 3)
    return out


def prepare_splits(df, cfg: TrainingConfig) -> Dict[str, Any]:
    """Build train/val/test datasets and loaders for a preprocessed DataFrame."""
    if cfg.split_by == "material":
        tr_idx, va_idx, te_idx = _material_split(
            df, cfg.test_ratio, cfg.val_ratio, cfg.seed)
        n_dropped = len(df) - (len(tr_idx) + len(va_idx) + len(te_idx))
        if n_dropped:
            logger.info(f"Material-disjoint split dropped {n_dropped} pairs whose "
                        f"two materials fall in different splits")
    else:
        tr_idx, va_idx, te_idx = _random_split(
            len(df), cfg.test_ratio, cfg.val_ratio, cfg.seed)

    tr_ds = PairDataset(df.iloc[tr_idx], fit_scaler=True)
    has_val = len(va_idx) > 0
    va_ds = (CachedPairDataset(df.iloc[va_idx], cfg.batch_size, scaler=tr_ds.scaler)
             if has_val else None)
    te_ds = CachedPairDataset(df.iloc[te_idx], cfg.batch_size, scaler=tr_ds.scaler)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    va_loader = (DataLoader(va_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_cached_batch)
                 if has_val else None)
    te_loader = DataLoader(te_ds, batch_size=1, shuffle=False,
                           collate_fn=collate_cached_batch)

    ok = tr_ds.ok1 & tr_ds.ok2
    delta_scale = cfg.delta_scale or compute_delta_scale(tr_ds.y1, tr_ds.y2, ok)

    return {
        "tr_ds": tr_ds, "va_ds": va_ds, "te_ds": te_ds,
        "tr_loader": tr_loader, "va_loader": va_loader, "te_loader": te_loader,
        "scaler": tr_ds.scaler, "delta_scale": float(delta_scale),
        "sizes": (len(tr_ds), len(va_ds.df) if has_val else 0, len(te_ds.df)),
        "leakage": leakage_report(df, tr_idx, va_idx, te_idx),
        "indices": (tr_idx, va_idx, te_idx),
    }


def _run_epoch(
    model: PolymerRankingModel,
    loader: DataLoader,
    criterion: MultiTaskBayesianRankingLoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute one epoch (training or validation).

    Returns ``(loss, s1, s2, y1, y2, valid, rank_valid)``:

    * ``valid`` [N, T] — both polymers measured (``> 0``), so the numeric
      difference is known and delta regression applies.
    * ``rank_valid`` [N, T] — the ordering is decidable. A mobility of 0 is
      left-censored (below the detection limit), hence ``measured > censored``
      is decidable, while two censored values are not comparable.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_s1, all_s2, all_y1, all_y2, all_valid, all_rank = [], [], [], [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for mg1, mg2, ef1, ef2, y1, y2, ok1, ok2 in loader:
            ef1, ef2 = ef1.to(DEVICE), ef2.to(DEVICE)
            y1, y2 = y1.to(DEVICE), y2.to(DEVICE)
            ok1, ok2 = ok1.to(DEVICE), ok2.to(DEVICE)
            valid = ok1 & ok2                       # both measured -> regression
            rank_valid = (ok1 | ok2) & (y1 != y2)   # ordering decidable -> BPR

            s1, s2 = model(mg1, ef1, mg2, ef2)
            loss, _ = criterion(s1, s2, y1, y2,
                                valid=valid, rank_valid=rank_valid)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = y1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_s1.append(s1.detach().cpu())
            all_s2.append(s2.detach().cpu())
            all_y1.append(y1.cpu())
            all_y2.append(y2.cpu())
            all_valid.append(valid.cpu())
            all_rank.append(rank_valid.cpu())

    def _cat(lst):
        return torch.cat(lst).numpy() if lst else np.empty((0, len(TASK_NAMES)))

    if total_samples == 0:
        logger.warning("Epoch ran on an empty data loader; returning zero loss / empty arrays")
        empty = np.empty((0, len(TASK_NAMES)), dtype=np.float32)
        empty_b = np.empty((0, len(TASK_NAMES)), dtype=bool)
        return 0.0, empty, empty.copy(), empty.copy(), empty.copy(), empty_b, empty_b.copy()

    return (
        total_loss / total_samples,
        _cat(all_s1),
        _cat(all_s2),
        _cat(all_y1),
        _cat(all_y2),
        _cat(all_valid).astype(bool),
        _cat(all_rank).astype(bool),
    )


def compute_metrics(s1, s2, y1, y2, valid=None, rank_valid=None) -> Dict[str, float]:
    """Compute pairwise accuracy, Spearman rho and correct-direction probability.

    ``rank_valid`` [N, T] (optional) marks pairs whose ordering is decidable —
    accuracy and ``avg_prob`` are computed on those. This includes pairs with one
    censored value, because a measured mobility is always above the detection
    limit.

    ``valid`` [N, T] (optional) marks pairs where both mobilities were measured;
    only those carry a numeric difference, so Spearman rho is computed on them.

    ``avg_prob`` is the mean probability assigned to the **correct** direction
    (:math:`\\sigma(\\text{sign}(y_1-y_2)(s_1-s_2))`), so it is high only when
    the model is both right and confident.
    """
    out: Dict[str, float] = {}
    n_tasks = s1.shape[1]
    for t in range(n_tasks):
        name = TASK_NAMES[t] if t < len(TASK_NAMES) else f"task{t}"
        dp = s1[:, t] - s2[:, t]
        dy = y1[:, t] - y2[:, t]

        # ordering-decidable pairs (accuracy / probability)
        mask = (dy != 0) if rank_valid is None else (
            rank_valid[:, t].astype(bool) & (dy != 0))
        # pairs with two measured values (spearman)
        v = (np.ones_like(dp, dtype=bool) if valid is None
             else valid[:, t].astype(bool))

        acc = float((np.sign(dp[mask]) == np.sign(dy[mask])).mean()) if mask.any() else 0.0

        if v.sum() >= 3 and np.ptp(np.concatenate([s1[v, t], s2[v, t]])) > 0:
            scores = np.concatenate([s1[v, t], s2[v, t]])
            targets = np.concatenate([y1[v, t], y2[v, t]])
            rho, _ = spearmanr(scores, targets)
            rho = 0.0 if np.isnan(rho) else float(rho)
        else:
            rho = 0.0

        prob = float((1.0 / (1.0 + np.exp(-np.sign(dy[mask]) * dp[mask]))).mean()) \
            if mask.any() else 0.0

        out[f"{name}_pair_acc"] = acc
        out[f"{name}_spearman"] = rho
        out[f"{name}_avg_prob"] = prob
        out[f"{name}_n_rank"] = float(mask.sum())
        out[f"{name}_n_reg"] = float(v.sum())

    out["mean_pair_acc"] = float(np.mean(
        [out[f"{TASK_NAMES[t]}_pair_acc"] for t in range(min(n_tasks, len(TASK_NAMES)))]))
    return out


def _monitor_value(metrics: Dict[str, float], cfg) -> float:
    """Value used for early stopping / LR scheduling — **higher is always better**.

    ``pair_acc`` -> the mean pairwise accuracy itself;
    ``loss``     -> the *negated* validation loss, so that "maximise the monitor"
    stays the single rule for both the stopper and the scheduler.
    """
    if getattr(cfg, "early_stop_metric", "pair_acc") == "pair_acc":
        return metrics["mean_pair_acc"]
    return -metrics.get("loss", 0.0)


def _stopper_mode(cfg) -> str:
    """``_monitor_value`` is always maximised, whatever is being monitored."""
    return "max"


def _build_model(mcfg: ModelConfig) -> PolymerRankingModel:
    return PolymerRankingModel(
        hidden_size=mcfg.hidden_size,
        depth=mcfg.depth,
        dropout=mcfg.dropout,
        ffn_hidden=mcfg.ffn_hidden,
        extra_dim=mcfg.extra_dim,
        num_tasks=mcfg.num_tasks,
        aggregation=mcfg.aggregation,
    ).to(DEVICE)


def _make_scheduler(cfg, optimizer):
    if getattr(cfg, "scheduler", "plateau") == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2)
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                             patience=10, min_lr=1e-6)


def _run_training_loop(model, criterion, optimizer, scheduler, stopper, cfg,
                       tr_loader, va_loader, log_every: int = 10):
    """Shared epoch loop for :func:`train` and :func:`run_experiment`."""
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_metrics: Dict[str, float] = {}
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, *_ = _run_epoch(model, tr_loader, criterion, optimizer)
        history["train_loss"].append(tr_loss)

        if va_loader is None:
            history["val_loss"].append(None)
            history["val_metrics"].append({})
            if epoch % log_every == 0 or epoch == 1:
                logger.info(f"Ep {epoch:4d} | tr={tr_loss:.4f}")
            best_epoch = epoch
            continue

        va_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(
            model, va_loader, criterion)
        va_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        va_met["loss"] = va_loss
        history["val_loss"].append(va_loss)
        history["val_metrics"].append(va_met)

        monitor = _monitor_value(va_met, cfg)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(monitor)
        elif scheduler is not None:
            scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                f"Ep {epoch:4d} | tr={tr_loss:.4f}  va={va_loss:.4f} | "
                f"mu_e acc={va_met['mu_e_pair_acc']:.3f} "
                f"rho={va_met['mu_e_spearman']:.3f} "
                f"n={int(va_met['mu_e_n_rank'])} | "
                f"mu_h acc={va_met['mu_h_pair_acc']:.3f} "
                f"rho={va_met['mu_h_spearman']:.3f} "
                f"n={int(va_met['mu_h_n_rank'])}"
            )

        if stopper.step(monitor, model):
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(best {cfg.early_stop_metric}={stopper.best_score:.4f})")
            break
        if stopper.counter == 0:
            best_epoch = epoch
            best_metrics = va_met

    if stopper.best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in stopper.best_state.items()})
    if not best_metrics:
        best_metrics = history["val_metrics"][-1] if history["val_metrics"] else {}
        best_metrics = dict(best_metrics or {})
    return {"history": history, "best_metrics": best_metrics, "best_epoch": best_epoch}


def run_experiment(
    df,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    max_epochs: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train one configuration and return its validation metrics.

    Lightweight variant of :func:`train` used for hyper-parameter search: no
    checkpoint is written and only the best validation metrics are kept.
    """
    cfg = copy.deepcopy(train_config)
    if max_epochs:
        cfg.epochs = max_epochs

    splits = prepare_splits(df, cfg)
    model = _build_model(model_config)
    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=cfg.rank_weight,
        reg_weight=cfg.reg_weight,
        delta_scale=splits["delta_scale"],
        censored_weight=cfg.censored_weight,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)
    stopper = EarlyStopping(patience=cfg.patience, mode=_stopper_mode(cfg))

    out = _run_training_loop(model, criterion, optimizer, scheduler, stopper, cfg,
                             splits["tr_loader"], splits["va_loader"],
                             log_every=10 if verbose else 10 ** 9)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model

    return {
        "val_metrics": out["best_metrics"],
        "best_epoch": out["best_epoch"],
        "delta_scale": splits["delta_scale"],
        "n_params": n_params,
        "sizes": splits["sizes"],
    }


def train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> Dict[str, Any]:
    """Complete training pipeline, returns dict with test_metrics / history / model / scalers."""
    cfg = train_config
    mcfg = model_config

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    df = load_and_preprocess(cfg.csv_path,
                             depth=mcfg.depth,
                             max_repeats=cfg.max_repeats)
    splits = prepare_splits(df, cfg)
    tr_ds, tr_loader = splits["tr_ds"], splits["tr_loader"]
    va_loader, te_loader = splits["va_loader"], splits["te_loader"]

    logger.info(f"Split by '{cfg.split_by}': "
                f"Train/Val/Test: {splits['sizes'][0]}/"
                f"{splits['sizes'][1]}/{splits['sizes'][2]}")
    logger.info(f"delta_scale (std of measured log10 differences): "
                f"{splits['delta_scale']:.3f}")
    if splits["leakage"]:
        logger.info("Leakage diagnostics: " + ", ".join(
            f"{k}={v}" for k, v in splits["leakage"].items()))

    model = _build_model(mcfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=cfg.rank_weight,
        reg_weight=cfg.reg_weight,
        delta_scale=splits["delta_scale"],
        censored_weight=cfg.censored_weight,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)
    stopper = EarlyStopping(patience=cfg.patience, mode=_stopper_mode(cfg))

    if va_loader is None:
        logger.info("Validation set is empty (val_ratio=0); early stopping disabled")

    out = _run_training_loop(model, criterion, optimizer, scheduler, stopper, cfg,
                             tr_loader, va_loader)
    history = out["history"]

    te_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(
        model, te_loader, criterion)
    te_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
    logger.info("\n========== Test Results ==========")
    for k, v in te_met.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    meta = {
        "model_state": model.state_dict(),
        "scaler": splits["scaler"],
        "config": mcfg.to_dict(),
        "delta_scale": splits["delta_scale"],
        "loss_config": {
            "rank_weight": cfg.rank_weight,
            "reg_weight": cfg.reg_weight,
            "censored_weight": cfg.censored_weight,
        },
        "split_by": cfg.split_by,
    }
    ckpt_path = Path(cfg.save_dir) / "best_model.pt"
    torch.save(meta, ckpt_path)
    logger.info(f"Checkpoint saved -> {ckpt_path}")

    final_path = Path(cfg.save_dir) / "final_model.pt"
    torch.save({**meta, "history": history}, final_path)
    logger.info(f"Checkpoint saved -> {final_path}")

    return {
        "test_metrics": te_met,
        "history": history,
        "model": model,
        "scaler": splits["scaler"],
        "checkpoint": ckpt_path,
        "final_checkpoint": final_path,
        "delta_scale": splits["delta_scale"],
        "leakage": splits["leakage"],
    }


def finetune(config: FinetuneConfig) -> Dict[str, Any]:
    """Fine-tuning mode: load best weights, continue on the full data with monitoring."""
    from .predict import load_checkpoint

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    model_config, scaler, model, ckpt_meta = load_checkpoint(config.checkpoint_path)
    # keep the regression target on the same scale as during pre-training
    delta_scale = (ckpt_meta or {}).get("delta_scale")
    loss_cfg = (ckpt_meta or {}).get("loss_config") or {}

    df = load_and_preprocess(config.csv_path,
                             depth=model_config.depth,
                             max_repeats=config.max_repeats)
    logger.info(f"Full dataset size: {len(df)} pairs")

    idx = np.arange(len(df))
    if config.val_ratio > 0:
        tr_idx, va_idx = train_test_split(idx, test_size=config.val_ratio,
                                          random_state=config.seed)
    else:
        tr_idx, va_idx = idx, np.array([], dtype=int)

    full_ds = PairDataset(df.iloc[tr_idx], scaler=scaler, fit_scaler=False)
    full_loader = DataLoader(full_ds, batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)

    va_loader = None
    if len(va_idx):
        va_ds = CachedPairDataset(df.iloc[va_idx], config.batch_size,
                                  scaler=scaler, fit_scaler=False)
        va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                               collate_fn=collate_cached_batch)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    if delta_scale is None:
        delta_scale = compute_delta_scale(full_ds.y1, full_ds.y2,
                                          full_ds.ok1 & full_ds.ok2)
    logger.info(f"delta_scale: {delta_scale:.3f}")

    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=loss_cfg.get("rank_weight", 0.9),
        reg_weight=loss_cfg.get("reg_weight", 0.1),
        delta_scale=delta_scale,
        censored_weight=loss_cfg.get("censored_weight", 1.0),
    )
    optimizer = AdamW(model.parameters(), lr=config.lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.finetune_epochs,
                                  eta_min=config.lr * 0.1)
    stopper = EarlyStopping(patience=config.patience, mode="max")

    history: Dict[str, Any] = {"train_loss": [], "train_metrics": [], "val_metrics": []}

    logger.info(
        f"Starting fine-tuning for {config.finetune_epochs} epochs with lr={config.lr}")
    for epoch in range(1, config.finetune_epochs + 1):
        tr_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(
            model, full_loader, criterion, optimizer)
        tr_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_metrics"].append(tr_met)

        msg = (f"Ep {epoch:4d} | loss={tr_loss:.4f} | "
               f"mu_e acc={tr_met['mu_e_pair_acc']:.3f} "
               f"rho={tr_met['mu_e_spearman']:.3f} | "
               f"mu_h acc={tr_met['mu_h_pair_acc']:.3f} "
               f"rho={tr_met['mu_h_spearman']:.3f}")

        monitor = tr_met["mean_pair_acc"]
        if va_loader is not None:
            _, v1, v2, vy1, vy2, vvalid, vrank = _run_epoch(
                model, va_loader, criterion)
            va_met = compute_metrics(v1, v2, vy1, vy2, vvalid, vrank)
            history["val_metrics"].append(va_met)
            monitor = va_met["mean_pair_acc"]
            msg += (f" | val acc_e={va_met['mu_e_pair_acc']:.3f} "
                    f"acc_h={va_met['mu_h_pair_acc']:.3f}")
            if stopper.step(monitor, model):
                logger.info(msg)
                logger.info(f"Fine-tuning early stop at epoch {epoch}")
                break

        logger.info(msg)

    if stopper.best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in stopper.best_state.items()})

    final_ckpt_path = Path(config.save_dir) / "final_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": model_config.to_dict(),
            "delta_scale": delta_scale,
            "loss_config": loss_cfg,
            "finetune_history": history,
        },
        final_ckpt_path,
    )
    logger.info(f"Final model saved -> {final_ckpt_path}")

    return {
        "final_checkpoint": final_ckpt_path,
        "history": history,
        "model": model,
    }
