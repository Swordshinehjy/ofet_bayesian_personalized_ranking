"""Hyper-parameter analysis and search for the polymer BPR ranking model.

Stages
------
``analyze``
    Data health check: how many pairs actually carry a *measured* mobility, how
    many are left-censored (mu = 0), the scale of the log10 differences,
    class balance, material reuse / duplicate pairs (leakage indicators), plus
    reference baselines (a LUMO/HOMO heuristic and a descriptor-only logistic
    regression).
``check``
    Rule-based sanity check of a configuration — is ``lr`` compatible with
    ``batch_size``, is the model oversized for the number of training pairs, is
    ``delta_scale`` consistent with the data, is ``patience``/``scheduler``
    compatible with ``epochs``, ... Every finding carries a suggested fix.
``search``
    Random search over lr / hidden_size / depth / dropout / weight_decay /
    batch_size / rank-reg weighting / censored weighting. Each trial is trained
    with early stopping on the validation pairwise accuracy and — because
    literature mobility data is noisy — repeated on ``--n_repeats`` splits and
    averaged.
``all``
    analyze -> check -> search, then print the recommended configuration.

Usage
-----
::

    D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage analyze
    D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage check
    D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage all --n_trials 12 --max_epochs 40

Results are written to ``hyperparam_search.csv``, ``hyperparam_ablation.csv``
and ``best_hyperparams.json``.

Note
----
Cyclization depends on ``depth`` (the span requirement is ``2 * depth + 1``), but
re-running it per trial is expensive. The molecules are therefore built **once**
at the default ``depth`` (6), the strictest requirement in the search space;
trials with a smaller ``depth`` simply see a slightly more expanded oligomer than
necessary, which is valid input but not exactly what production would build.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from polymer_ranking.chemistry import load_and_preprocess
from polymer_ranking.config import ModelConfig, TrainingConfig, SPLIT_MODES
from polymer_ranking.model import PolymerRankingModel
from polymer_ranking.training import run_experiment

EXTRA_RAW = [
    "conjugation_{s}",
    "Isomer_{s}",
    "CentroSymmetry_{s}",
    "E_LUMO (eV)_{s}",
    "E_HOMO (eV)_{s}",
]
TASKS = ["mu_e", "mu_h"]


# ── stage 1: data analysis ──────────────────────────────────────────────────

def _rank_mask(df: pd.DataFrame, t: str) -> np.ndarray:
    """Pairs whose ordering is decidable: at least one measured mobility and no tie.

    A mobility of 0 is left-censored (below the detection limit), so a measured
    value always ranks above a censored one; two censored values are incomparable.
    """
    ok1 = df[f"ok_{t}_1"].values.astype(bool)
    ok2 = df[f"ok_{t}_2"].values.astype(bool)
    dy = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
    return (ok1 | ok2) & (dy != 0)


def analyze(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Data health check + descriptor-only baselines."""
    report: Dict[str, Any] = {"n_pairs": len(df)}

    if "Materials_1" in df.columns:
        mats = pd.concat([df["Materials_1"], df["Materials_2"]]).astype(str)
        report["n_materials"] = int(mats.nunique())
        report["material_reuse_mean"] = round(float(mats.value_counts().mean()), 2)
        keys = ["||".join(sorted([a, b]))
                for a, b in zip(df["Materials_1"].astype(str),
                                df["Materials_2"].astype(str))]
        report["n_duplicate_pairs"] = int(len(keys) - len(set(keys)))

    for t in TASKS:
        ok1, ok2 = df[f"ok_{t}_1"], df[f"ok_{t}_2"]
        both = (ok1 & ok2).values
        dy_all = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
        report[f"{t}_both_measured"] = int(both.sum())
        report[f"{t}_both_measured_frac"] = round(float(both.mean()), 3)
        report[f"{t}_rank_decidable"] = int(_rank_mask(df, t).sum())
        report[f"{t}_one_censored"] = int((ok1 ^ ok2).sum())
        report[f"{t}_both_censored"] = int((~ok1 & ~ok2).sum())
        dy = dy_all[both]
        report[f"{t}_delta_std"] = round(float(dy.std()), 3) if len(dy) > 1 else 0.0
        report[f"{t}_delta_abs_median"] = round(float(np.median(np.abs(dy))), 3) if len(dy) else 0.0
        report[f"{t}_pos_frac"] = round(float((dy > 0).mean()), 3) if len(dy) else 0.0

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=seed)
    report.update(_baselines(df, tr_idx, te_idx))
    return report


def _baselines(df: pd.DataFrame, tr_idx, te_idx) -> Dict[str, float]:
    """Reference scores: heuristic rules and a descriptor-only classifier.

    Anything a trained D-MPNN cannot clearly beat is not worth the compute.
    """
    out: Dict[str, float] = {}

    # heuristic: lower LUMO -> higher mu_e ; higher HOMO -> higher mu_h
    for t, sign in (("mu_e", -1.0), ("mu_h", 1.0)):
        col = "E_LUMO (eV)" if t == "mu_e" else "E_HOMO (eV)"
        if f"{col}_1" not in df.columns:
            continue
        d = (df[f"{col}_1"] - df[f"{col}_2"]).values
        dy = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
        m = _rank_mask(df, t)
        te = np.isin(np.arange(len(df)), te_idx) & m
        if te.sum():
            pred = sign * np.sign(d[te])
            out[f"{t}_baseline_heuristic_acc"] = round(
                float((pred == np.sign(dy[te])).mean()), 3)

    # descriptor-only logistic regression on the 5 extra features
    for t in TASKS:
        cols1 = [c.format(s="1") for c in EXTRA_RAW]
        cols2 = [c.format(s="2") for c in EXTRA_RAW]
        if not all(c in df.columns for c in cols1 + cols2):
            continue
        X = (df[cols1].values - df[cols2].values).astype(float)
        dy = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
        m = _rank_mask(df, t)
        if m.sum() < 50:
            continue
        y = np.sign(dy[m])
        Xm = X[m]
        pos = np.where(m)[0]
        tr_m, te_m = np.isin(pos, tr_idx), np.isin(pos, te_idx)
        if tr_m.sum() < 20 or te_m.sum() < 10:
            continue
        sc = StandardScaler().fit(Xm[tr_m])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xm[tr_m]), y[tr_m])
        pred = clf.predict(sc.transform(Xm[te_m]))
        out[f"{t}_baseline_logreg_acc"] = round(float((pred == y[te_m]).mean()), 3)
    return out


def print_report(report: Dict[str, Any]) -> None:
    print("\n========== Data Analysis ==========")
    for k, v in report.items():
        print(f"  {k:34s}: {v}")

    print("\n--- Interpretation ---")
    for t in TASKS:
        frac = report.get(f"{t}_both_measured_frac")
        if frac is None:
            continue
        print(f"  {t}: {frac:.1%} of pairs have two measured mobilities (regression "
              f"target known); {report.get(f'{t}_rank_decidable')} pairs have a decidable "
              f"ordering (ranking target known, including {report.get(f'{t}_one_censored')} "
              f"with one censored side). {report.get(f'{t}_both_censored')} pairs are "
              f"censored on both sides and carry no ordering information.")
        std = report.get(f"{t}_delta_std")
        if std:
            print(f"  {t}: std of the log10 difference = {std} -> delta_scale should be "
                  f"close to this so the regression term stays comparable to the BPR term.")
    reuse = report.get("material_reuse_mean")
    if reuse:
        print(f"  Each material appears in ~{reuse} pairs; a random pair-level split "
              f"therefore re-uses materials across train/test (interpolation setting). "
              f"Use --split_by material for a stricter, material-disjoint estimate.")
    if report.get("n_duplicate_pairs"):
        print(f"  {report['n_duplicate_pairs']} duplicate (unordered) material pairs: "
              f"identical comparisons appear in the data.")


# ── stage 2: sanity check of a configuration ────────────────────────────────

def check_config(
    mcfg: ModelConfig,
    tcfg: TrainingConfig,
    n_train: int,
    delta_std: float,
    n_params: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Rule-based sanity check; returns a list of findings.

    Each finding is ``{level, item, value, issue, suggestion}`` with
    ``level`` in ``{"OK", "WARN", "ERROR"}``.
    """
    if n_params is None:
        n_params = sum(p.numel() for p in PolymerRankingModel(
            **mcfg.to_dict()).parameters() if p.requires_grad)

    findings: List[Dict[str, str]] = []

    def add(level, item, value, issue, suggestion):
        findings.append({"level": level, "item": item, "value": str(value),
                         "issue": issue, "suggestion": suggestion})

    # ---- capacity vs data ------------------------------------------------
    ratio = n_params / max(n_train, 1)
    if ratio > 500:
        add("WARN", "model_capacity", f"{n_params:,} params / {n_train} pairs = {ratio:.0f}",
            "heavily over-parameterised for this dataset",
            "reduce hidden_size (<=256) / ffn_hidden (<=128), or raise dropout & weight_decay")
    elif ratio > 150:
        add("OK", "model_capacity", f"{n_params:,} params / {n_train} pairs = {ratio:.0f}",
            "large but manageable with regularisation", "keep dropout >= 0.2")
    else:
        add("OK", "model_capacity", f"{n_params:,} params / {n_train} pairs = {ratio:.0f}",
            "capacity is proportionate to the data size", "-")

    # ---- optimizer --------------------------------------------------------
    lo, hi = 1e-5, 5e-3
    if tcfg.lr < lo:
        add("WARN", "lr", tcfg.lr, "too small, training will barely move",
            f"use {max(lo, 1e-4):g} - 1e-3")
    elif tcfg.lr > hi:
        add("WARN", "lr", tcfg.lr, "too large for a siamese D-MPNN with grad clipping",
            "use <= 3e-3")
    else:
        add("OK", "lr", tcfg.lr, "in the usable AdamW range", "-")

    # linear-scaling reference for the batch size
    ref_lr = 1e-3 * (tcfg.batch_size / 32.0)
    if not (0.25 * ref_lr <= tcfg.lr <= 4 * ref_lr):
        add("WARN", "lr_vs_batch_size", f"lr={tcfg.lr:g}, batch_size={tcfg.batch_size}",
            "learning rate does not follow the batch size",
            f"for batch_size={tcfg.batch_size} a linear-scaling reference is lr≈{ref_lr:g}")
    else:
        add("OK", "lr_vs_batch_size", f"lr={tcfg.lr:g}, batch_size={tcfg.batch_size}",
            "consistent with linear scaling", "-")

    if not 8 <= tcfg.batch_size <= 128:
        add("WARN", "batch_size", tcfg.batch_size, "outside the sane 8-128 range",
            "use 32 (64 if you also scale the learning rate up)")
    else:
        add("OK", "batch_size", tcfg.batch_size, "in the sane range", "-")

    # ---- schedule ---------------------------------------------------------
    if tcfg.epochs < 30:
        add("WARN", "epochs", tcfg.epochs, "too few epochs to converge", "use >= 100")
    elif tcfg.epochs > 500:
        add("WARN", "epochs", tcfg.epochs, "early stopping will never see these epochs; "
            "the budget is wasted", "use 200-300")
    else:
        add("OK", "epochs", tcfg.epochs, "reasonable budget for this data size", "-")

    if tcfg.patience < 10:
        add("WARN", "patience", tcfg.patience,
            "too short; pair_acc on ~150 val pairs is noisy and will trigger early",
            "use >= 20")
    elif tcfg.patience > max(30, tcfg.epochs // 3):
        add("WARN", "patience", tcfg.patience,
            "patience is a large fraction of the epoch budget",
            f"use ~{max(20, tcfg.epochs // 6)}")
    else:
        add("OK", "patience", tcfg.patience, "tolerates validation noise", "-")

    if tcfg.scheduler == "cosine" and tcfg.epochs > 3 * tcfg.patience:
        add("WARN", "scheduler", "cosine",
            f"cosine annealing is spread over {tcfg.epochs} epochs but early stopping "
            f"usually fires around epoch {3 * tcfg.patience}, so the LR never decays",
            "use 'plateau'")
    else:
        add("OK", "scheduler", tcfg.scheduler, "compatible with the epoch budget", "-")

    if tcfg.early_stop_metric != "pair_acc":
        add("WARN", "early_stop_metric", tcfg.early_stop_metric,
            "the objective is the pairwise ranking accuracy, not the loss",
            "use 'pair_acc'")
    else:
        add("OK", "early_stop_metric", tcfg.early_stop_metric,
            "monitors the actual objective", "-")

    # ---- regularisation ---------------------------------------------------
    if mcfg.dropout < 0.05:
        add("WARN", "dropout", mcfg.dropout, "almost no dropout on an over-parameterised "
            "model", "use 0.1-0.3")
    elif mcfg.dropout > 0.5:
        add("WARN", "dropout", mcfg.dropout, "very high dropout slows convergence",
            "use <= 0.3")
    else:
        add("OK", "dropout", mcfg.dropout, "in the sane range", "-")

    wd = tcfg.weight_decay
    if wd < 1e-6:
        add("WARN", "weight_decay", wd, "effectively no weight decay",
            "use 1e-4 - 1e-3 when params >> pairs")
    elif wd > 1e-2:
        add("WARN", "weight_decay", wd, "too strong, will underfit", "use <= 1e-2")
    else:
        add("OK", "weight_decay", wd, "in the sane range", "-")

    # ---- loss weighting ---------------------------------------------------
    if tcfg.reg_weight > tcfg.rank_weight:
        add("WARN", "rank/reg weights", f"{tcfg.rank_weight}/{tcfg.reg_weight}",
            "the regression term outweighs the ranking term, but ranking is the objective",
            "use rank_weight >= 2 * reg_weight (measured optimum here: 0.9 / 0.1)")
    elif tcfg.rank_weight <= 0:
        add("ERROR", "rank_weight", tcfg.rank_weight, "ranking supervision is switched off",
            "use >= 0.5")
    else:
        add("OK", "rank/reg weights", f"{tcfg.rank_weight}/{tcfg.reg_weight}",
            "ranking dominates, regression only fixes the score scale", "-")

    if not 0.0 <= tcfg.censored_weight <= 1.0:
        add("ERROR", "censored_weight", tcfg.censored_weight, "must lie in [0, 1]",
            "use 1.0")
    elif tcfg.censored_weight == 0.0:
        add("WARN", "censored_weight", tcfg.censored_weight,
            "pairs with a censored mobility (mu = 0) are dropped from the ranking loss, "
            "although their ordering is known", "use 1.0 (measured optimum here)")
    else:
        add("OK", "censored_weight", tcfg.censored_weight,
            "censored pairs contribute with a reduced weight", "-")

    # ---- delta_scale ------------------------------------------------------
    ds = tcfg.delta_scale
    if ds is None:
        add("OK", "delta_scale", "auto", "estimated from the training split",
            f"measured std is ~{delta_std:.2f}")
    elif delta_std > 0 and not (0.5 * delta_std <= ds <= 2 * delta_std):
        add("WARN", "delta_scale", ds,
            f"differs from the std of the measured log10 differences (~{delta_std:.2f})",
            f"use {delta_std:.2f}")
    else:
        add("OK", "delta_scale", ds, "matches the scale of the targets", "-")

    # ---- split ------------------------------------------------------------
    if tcfg.split_by not in SPLIT_MODES:
        add("ERROR", "split_by", tcfg.split_by, f"must be one of {list(SPLIT_MODES)}",
            "use 'pair'")
    else:
        add("OK", "split_by", tcfg.split_by,
            "'pair' = interpolation estimate; re-run with 'material' to quantify "
            "the leakage-induced optimism", "-")

    return findings


def print_findings(findings: List[Dict[str, str]]) -> None:
    print("\n========== Hyper-parameter Sanity Check ==========")
    for f in findings:
        if f["level"] == "OK":
            print(f"  [ OK ] {f['item']:18s} = {f['value']:28s} {f['issue']}")
        else:
            print(f"  [{f['level']:4s}] {f['item']:18s} = {f['value']:28s} {f['issue']}")
            print(f"          -> fix: {f['suggestion']}")
    n_warn = sum(1 for f in findings if f["level"] != "OK")
    print(f"\n  {len(findings) - n_warn} OK, {n_warn} finding(s) need attention")


def auto_fix(mcfg: ModelConfig, tcfg: TrainingConfig, findings: List[Dict[str, str]],
             delta_std: float) -> Tuple[ModelConfig, TrainingConfig, List[str]]:
    """Apply the suggested fixes; returns (model_cfg, train_cfg, applied)."""
    applied: List[str] = []
    for f in findings:
        if f["level"] == "OK":
            continue
        item = f["item"]
        if item == "model_capacity":
            mcfg.hidden_size = min(mcfg.hidden_size, 256)
            mcfg.ffn_hidden = min(mcfg.ffn_hidden, 128)
            applied.append("hidden_size/ffn_hidden reduced to 256/128")
        elif item == "lr":
            tcfg.lr = min(max(tcfg.lr, 1e-4), 3e-3)
            applied.append(f"lr clipped to {tcfg.lr:g}")
        elif item == "lr_vs_batch_size":
            tcfg.lr = 1e-3 * (tcfg.batch_size / 32.0)
            applied.append(f"lr rescaled to {tcfg.lr:g} for batch_size={tcfg.batch_size}")
        elif item == "batch_size":
            tcfg.batch_size = 32
            applied.append("batch_size set to 32")
        elif item == "epochs":
            tcfg.epochs = min(max(tcfg.epochs, 100), 300)
            applied.append(f"epochs set to {tcfg.epochs}")
        elif item == "patience":
            tcfg.patience = max(20, min(tcfg.patience, tcfg.epochs // 6))
            applied.append(f"patience set to {tcfg.patience}")
        elif item == "scheduler":
            tcfg.scheduler = "plateau"
            applied.append("scheduler set to 'plateau'")
        elif item == "early_stop_metric":
            tcfg.early_stop_metric = "pair_acc"
            applied.append("early_stop_metric set to 'pair_acc'")
        elif item == "dropout":
            mcfg.dropout = min(max(mcfg.dropout, 0.1), 0.3)
            applied.append(f"dropout clipped to {mcfg.dropout}")
        elif item == "weight_decay":
            tcfg.weight_decay = min(max(tcfg.weight_decay, 1e-4), 1e-2)
            applied.append(f"weight_decay clipped to {tcfg.weight_decay:g}")
        elif item == "rank/reg weights":
            tcfg.rank_weight, tcfg.reg_weight = 0.9, 0.1
            applied.append("rank_weight/reg_weight set to 0.9/0.1")
        elif item == "censored_weight":
            tcfg.censored_weight = 1.0
            applied.append("censored_weight set to 1.0")
        elif item == "delta_scale":
            tcfg.delta_scale = round(float(delta_std), 3) if delta_std > 0 else None
            applied.append(f"delta_scale set to {tcfg.delta_scale}")
        elif item == "split_by":
            tcfg.split_by = "pair"
            applied.append("split_by set to 'pair'")
    return mcfg, tcfg, applied


# ── stage 3: random search ──────────────────────────────────────────────────

SEARCH_SPACE: Dict[str, List[Any]] = {
    "lr": [1e-4, 3e-4, 1e-3],
    "hidden_size": [128, 256, 300],
    "depth": [3, 4, 6],
    "dropout": [0.1, 0.2, 0.3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64],
    "loss_weights": [(0.8, 0.2), (0.6, 0.4), (1.0, 0.0)],
    "censored_weight": [0.3, 0.5, 1.0],
}


def _sample_config(rng: random.Random) -> Dict[str, Any]:
    cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    cfg["ffn_hidden"] = max(64, cfg["hidden_size"] // 2)
    return cfg


def _evaluate(
    df: pd.DataFrame,
    cand: Dict[str, Any],
    max_epochs: int,
    patience: int,
    seeds: List[int],
) -> Dict[str, Any]:
    """Train one candidate on several splits and average the validation metrics."""
    alpha, beta = cand["loss_weights"]
    mcfg = ModelConfig(
        hidden_size=cand["hidden_size"],
        depth=cand["depth"],
        dropout=cand["dropout"],
        ffn_hidden=cand["ffn_hidden"],
    )
    accs, accs_e, accs_h, rhos_e, rhos_h, epochs = [], [], [], [], [], []
    delta_scale = 1.0
    n_params = 0
    for s in seeds:
        tcfg = TrainingConfig(
            batch_size=cand["batch_size"],
            lr=cand["lr"],
            weight_decay=cand["weight_decay"],
            patience=patience,
            epochs=max_epochs,
            seed=s,
            rank_weight=alpha,
            reg_weight=beta,
            censored_weight=cand["censored_weight"],
            early_stop_metric="pair_acc",
            scheduler="plateau",
        )
        out = run_experiment(df, mcfg, tcfg, max_epochs=max_epochs)
        vm = out["val_metrics"]
        accs.append(vm.get("mean_pair_acc", 0.0))
        accs_e.append(vm.get("mu_e_pair_acc", 0.0))
        accs_h.append(vm.get("mu_h_pair_acc", 0.0))
        rhos_e.append(vm.get("mu_e_spearman", 0.0))
        rhos_h.append(vm.get("mu_h_spearman", 0.0))
        epochs.append(out["best_epoch"])
        delta_scale = out["delta_scale"]
        n_params = out["n_params"]

    return {
        "val_mean_acc": round(float(np.mean(accs)), 4),
        "val_acc_std": round(float(np.std(accs)), 4),
        "val_mu_e_acc": round(float(np.mean(accs_e)), 4),
        "val_mu_h_acc": round(float(np.mean(accs_h)), 4),
        "val_mu_e_rho": round(float(np.mean(rhos_e)), 4),
        "val_mu_h_rho": round(float(np.mean(rhos_h)), 4),
        "best_epoch": int(np.mean(epochs)),
        "n_params": n_params,
        "delta_scale": round(float(delta_scale), 3),
    }


def search(
    df: pd.DataFrame,
    n_trials: int = 12,
    max_epochs: int = 40,
    patience: int = 15,
    seed: int = 42,
    n_repeats: int = 2,
) -> List[Dict[str, Any]]:
    """Random search; returns one record per trial sorted by validation accuracy."""
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    for i in range(1, n_trials + 1):
        cand = _sample_config(rng)
        seeds = [seed + 100 * r for r in range(n_repeats)]
        scores = _evaluate(df, cand, max_epochs, patience, seeds)
        alpha, beta = cand["loss_weights"]
        rec = {
            "trial": i,
            "lr": cand["lr"],
            "hidden_size": cand["hidden_size"],
            "depth": cand["depth"],
            "dropout": cand["dropout"],
            "ffn_hidden": cand["ffn_hidden"],
            "weight_decay": cand["weight_decay"],
            "batch_size": cand["batch_size"],
            "rank_weight": alpha,
            "reg_weight": beta,
            "censored_weight": cand["censored_weight"],
            **scores,
        }
        results.append(rec)
        print(f"[trial {i:2d}/{n_trials}] val_acc={rec['val_mean_acc']:.3f} "
              f"±{rec['val_acc_std']:.3f} (e={rec['val_mu_e_acc']:.3f} "
              f"h={rec['val_mu_h_acc']:.3f}) ep={rec['best_epoch']:5.1f} | "
              f"lr={rec['lr']:g} h={rec['hidden_size']} d={rec['depth']} "
              f"do={rec['dropout']} wd={rec['weight_decay']:g} bs={rec['batch_size']} "
              f"a/b={alpha}/{beta} cw={rec['censored_weight']}")

    results.sort(key=lambda r: r["val_mean_acc"], reverse=True)
    return results


# ── stage 4: loss-weighting ablation ────────────────────────────────────────

# The ranking/regression trade-off is dataset specific: it depends on how noisy
# the absolute log10 mobilities are and on how many pairs are left-censored, so
# it must be re-measured here rather than copied from another project.
LOSS_WEIGHT_GRID: List[Tuple[float, float]] = [
    (1.0, 0.0),   # pure BPR
    (0.9, 0.1),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),   # ranking and regression equally weighted
]
CENSORED_WEIGHT_GRID: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0]


def ablation(
    df: pd.DataFrame,
    base: Dict[str, Any],
    max_epochs: int = 40,
    patience: int = 15,
    seed: int = 42,
    n_repeats: int = 2,
    loss_weights: Optional[List[Tuple[float, float]]] = None,
    censored_weights: Optional[List[float]] = None,
    only: str = "both",
) -> List[Dict[str, Any]]:
    """Sweep the loss weighting with the architecture and optimizer fixed.

    Two sweeps are run:

    ``rank_weight``/``reg_weight``
        how much the numeric delta regression is allowed to steer the score
        scale relative to the BPR ranking term.
    ``censored_weight``
        how much weight a pair with one left-censored mobility (mu = 0) gets in
        the BPR term. ``0`` throws that ordering information away, ``1`` treats
        it like a fully measured pair.
    """
    out: List[Dict[str, Any]] = []
    seeds = [seed + 100 * r for r in range(n_repeats)]

    run_loss = only in ("loss", "both")
    run_cens = only in ("censored", "both")

    for alpha, beta in ((loss_weights or LOSS_WEIGHT_GRID) if run_loss else []):
        cand = {**base, "loss_weights": (alpha, beta)}
        scores = _evaluate(df, cand, max_epochs, patience, seeds)
        rec = {"sweep": "loss", "rank_weight": alpha, "reg_weight": beta,
               "censored_weight": base.get("censored_weight", 0.5), **scores}
        out.append(rec)
        print(f"[rank/reg {alpha}/{beta}] val_acc={rec['val_mean_acc']:.3f} "
              f"±{rec['val_acc_std']:.3f} (e={rec['val_mu_e_acc']:.3f} "
              f"h={rec['val_mu_h_acc']:.3f}) ep={rec['best_epoch']}")

    loss_rows = [r for r in out if r["sweep"] == "loss"]
    if loss_rows:
        best_loss = max(loss_rows, key=lambda r: r["val_mean_acc"])
        base_cens = {**base, "loss_weights": (best_loss["rank_weight"],
                                              best_loss["reg_weight"])}
    else:
        # the censored sweep needs a base weighting; use the one from `base`
        base_cens = dict(base)
    for cw in ((censored_weights or CENSORED_WEIGHT_GRID) if run_cens else []):
        cand = {**base_cens, "censored_weight": cw}
        scores = _evaluate(df, cand, max_epochs, patience, seeds)
        base_rank, base_reg = base_cens.get("loss_weights", (0.8, 0.2))
        rec = {"sweep": "censored", "rank_weight": base_rank,
               "reg_weight": base_reg, "censored_weight": cw, **scores}
        out.append(rec)
        print(f"[censored_w {cw}] val_acc={rec['val_mean_acc']:.3f} "
              f"±{rec['val_acc_std']:.3f} (e={rec['val_mu_e_acc']:.3f} "
              f"h={rec['val_mu_h_acc']:.3f}) ep={rec['best_epoch']}")

    return out


# ── recommendation ──────────────────────────────────────────────────────────

def recommend(
    report: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    fallback: TrainingConfig,
    ablation_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Combine the search / ablation output into one recommended configuration."""
    best = dict(search_results[0]) if search_results else {}
    delta_scale = max(report.get("mu_e_delta_std", 1.0),
                      report.get("mu_h_delta_std", 1.0), 1e-3)

    rank_w, reg_w = best.get("rank_weight", 0.8), best.get("reg_weight", 0.2)
    cens_w = best.get("censored_weight", 0.5)
    if ablation_results:
        loss_rows = [r for r in ablation_results if r["sweep"] == "loss"]
        cens_rows = [r for r in ablation_results if r["sweep"] == "censored"]
        if loss_rows:
            top = max(loss_rows, key=lambda r: r["val_mean_acc"])
            rank_w, reg_w = top["rank_weight"], top["reg_weight"]
        if cens_rows:
            cens_w = max(cens_rows, key=lambda r: r["val_mean_acc"])["censored_weight"]

    return {
        "hidden_size": best.get("hidden_size", 256),
        "depth": best.get("depth", 4),
        "dropout": best.get("dropout", 0.2),
        "ffn_hidden": best.get("ffn_hidden", 128),
        "lr": best.get("lr", 3e-4),
        "weight_decay": best.get("weight_decay", 1e-4),
        "batch_size": best.get("batch_size", 32),
        # measured on this dataset by the ablation, not copied from elsewhere
        "rank_weight": rank_w,
        "reg_weight": reg_w,
        "censored_weight": cens_w,
        "delta_scale": round(float(delta_scale), 3),
        "epochs": 200,
        "patience": 30,
        "early_stop_metric": "pair_acc",
        "scheduler": "plateau",
        "split_by": fallback.split_by,
        "val_mean_acc_from_search": best.get("val_mean_acc"),
        "n_trials": len(search_results),
    }


def print_recommendation(rec: Dict[str, Any]) -> None:
    print("\n========== Recommended Hyper-parameters ==========")
    for k, v in rec.items():
        print(f"  {k:28s}: {v}")
    print("\nTrain with:")
    print(f"  python polymer_ranking.py --mode train --csv <data>.csv \\")
    print(f"      --hidden_size {rec['hidden_size']} --depth {rec['depth']} "
          f"--dropout {rec['dropout']} --ffn_hidden {rec['ffn_hidden']} \\")
    print(f"      --lr {rec['lr']} --weight_decay {rec['weight_decay']} "
          f"--batch_size {rec['batch_size']} \\")
    print(f"      --rank_weight {rec['rank_weight']} --reg_weight {rec['reg_weight']} "
          f"--censored_weight {rec['censored_weight']} \\")
    print(f"      --delta_scale {rec['delta_scale']} --epochs {rec['epochs']} "
          f"--patience {rec['patience']} \\")
    print(f"      --early_stop_metric {rec['early_stop_metric']} "
          f"--scheduler {rec['scheduler']} --split_by {rec['split_by']}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyper-parameter analysis and search")
    p.add_argument("--csv", type=str, default="contrastive_paired.csv")
    p.add_argument("--stage", choices=["analyze", "check", "search", "ablation", "all"],
                   default="all")
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--n_repeats", type=int, default=2,
                   help="Splits per trial; validation accuracy is averaged over them")
    p.add_argument("--max_epochs", type=int, default=40,
                   help="Epoch budget per trial (early stopping may stop earlier)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--apply_fix", action="store_true",
                   help="Apply the suggested fixes to the configuration before searching")
    p.add_argument("--no_ablation", action="store_true",
                   help="Skip the rank/reg + censored weight sweep")
    p.add_argument("--only", choices=["loss", "censored", "both"], default="both",
                   help="Which ablation sweep to run (both by default)")
    p.add_argument("--loss_weight_pairs", type=float, nargs="*", default=None,
                   help="Custom rank/reg pairs, e.g. --loss_weight_pairs 1 0 0.8 0.2")
    p.add_argument("--censored_weights", type=float, nargs="*", default=None,
                   help="Custom censored weights, e.g. --censored_weights 0.3 0.5 1.0")
    p.add_argument("--out_csv", type=str, default="hyperparam_search.csv")
    p.add_argument("--out_json", type=str, default="best_hyperparams.json")
    return p.parse_args()


DEFAULT_BASE: Dict[str, Any] = {
    "hidden_size": 256,
    "depth": 6,
    "dropout": 0.2,
    "ffn_hidden": 128,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "batch_size": 32,
    "censored_weight": 0.5,
    "loss_weights": (0.8, 0.2),
}


def _base_from_json(path: str) -> Dict[str, Any]:
    """Reuse a previously recommended configuration as the ablation base."""
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_BASE)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return dict(DEFAULT_BASE)
    base = dict(DEFAULT_BASE)
    for k in ("hidden_size", "depth", "dropout", "ffn_hidden", "lr",
              "weight_decay", "batch_size", "censored_weight",
              "rank_weight", "reg_weight"):
        if k in d:
            base[k] = d[k]
    base["loss_weights"] = (base.pop("rank_weight"), base.pop("reg_weight"))
    return base


def _base_from_search(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not search_results:
        return dict(DEFAULT_BASE)
    b = dict(search_results[0])
    return {
        "hidden_size": b["hidden_size"],
        "depth": b["depth"],
        "dropout": b["dropout"],
        "ffn_hidden": b["ffn_hidden"],
        "lr": b["lr"],
        "weight_decay": b["weight_decay"],
        "batch_size": b["batch_size"],
        "censored_weight": b["censored_weight"],
        "loss_weights": (b["rank_weight"], b["reg_weight"]),
    }


def main() -> int:
    args = parse_args()

    print("Preprocessing data (cyclization + oligomer expansion)...")
    mcfg = ModelConfig()
    tcfg = TrainingConfig(csv_path=args.csv)
    df = load_and_preprocess(args.csv, depth=mcfg.depth, max_repeats=tcfg.max_repeats)
    print(f"Loaded {len(df)} valid pairs\n")

    report = analyze(df, seed=args.seed)
    print_report(report)

    n_train = int(round(len(df) * (1 - tcfg.val_ratio - tcfg.test_ratio)))
    delta_std = max(report.get("mu_e_delta_std", 0.0), report.get("mu_h_delta_std", 0.0))

    findings = check_config(mcfg, tcfg, n_train, delta_std)
    print_findings(findings)

    if args.stage in ("check", "all") and args.apply_fix:
        mcfg, tcfg, applied = auto_fix(mcfg, tcfg, findings, delta_std)
        print("\n  Applied fixes:")
        for a in applied:
            print(f"    - {a}")
        findings = check_config(mcfg, tcfg, n_train, delta_std)
        print_findings(findings)

    if args.stage == "analyze" or args.stage == "check":
        return 0

    search_results: List[Dict[str, Any]] = []
    if args.stage in ("search", "all"):
        print(f"\n========== Random search ({args.n_trials} trials "
              f"x {args.n_repeats} splits) ==========")
        search_results = search(
            df, n_trials=args.n_trials, max_epochs=args.max_epochs,
            patience=args.patience, seed=args.seed, n_repeats=args.n_repeats,
        )
        pd.DataFrame(search_results).to_csv(args.out_csv, index=False)
        print(f"\nSearch results -> {args.out_csv}")
        print(pd.DataFrame(search_results).head().to_string(index=False))

    ablation_results: List[Dict[str, Any]] = []
    if args.stage in ("ablation", "all") and not args.no_ablation:
        base = (_base_from_search(search_results) if search_results
                else _base_from_json(args.out_json))
        print(f"\n========== Loss-weight ablation ({args.only}) ==========")
        print(f"  base: h={base['hidden_size']} d={base['depth']} "
              f"do={base['dropout']} lr={base['lr']:g} bs={base['batch_size']}")
        def _pairs(values):
            return [(values[i], values[i + 1])
                    for i in range(0, len(values) - 1, 2)] if values else None

        ablation_results = ablation(
            df, base, max_epochs=args.max_epochs, patience=args.patience,
            seed=args.seed, n_repeats=args.n_repeats, only=args.only,
            loss_weights=_pairs(args.loss_weight_pairs),
            censored_weights=args.censored_weights,
        )
        pd.DataFrame(ablation_results).to_csv(
            Path(args.out_csv).with_name("hyperparam_ablation.csv"), index=False)
        print(pd.DataFrame(ablation_results).to_string(index=False))

    rec = recommend(report, search_results, tcfg, ablation_results)
    Path(args.out_json).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print_recommendation(rec)
    print(f"\nSaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
