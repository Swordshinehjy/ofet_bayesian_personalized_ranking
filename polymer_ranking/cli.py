"""CLI entry point."""

import logging
import argparse

import pandas as pd

from .config import ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from .training import train, finetune
from .predict import predict_batch

logger = logging.getLogger(__name__)


def merge_args(config_cls, args, arg_mapping: dict):
    """Merge non-None CLI arguments into config dataclass."""
    kwargs = {}
    for config_field, arg_name in arg_mapping.items():
        arg_val = getattr(args, arg_name)
        if arg_val is not None:
            kwargs[config_field] = arg_val
    return config_cls(**kwargs)


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--csv", type=str, default=None, help="Training CSV path")
    p.add_argument("--predict_csv", type=str, default=None, help="Prediction CSV path")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--ffn_hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_repeats", type=int, default=None,
                   help="Max repeat units when a unit is too short for cyclization")
    p.add_argument("--rank_weight", type=float, default=None,
                   help="Weight of the BPR ranking term")
    p.add_argument("--reg_weight", type=float, default=None,
                   help="Weight of the delta regression term")
    p.add_argument("--censored_weight", type=float, default=None,
                   help="BPR weight of pairs containing a censored mobility (mu=0)")
    p.add_argument("--delta_scale", type=float, default=None,
                   help="Scale of the regression target (default: std of the "
                        "measured log10 differences)")
    p.add_argument("--early_stop_metric", choices=["pair_acc", "loss"], default=None,
                   help="Quantity monitored by early stopping / LR scheduling")
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default=None,
                   help="LR scheduler")
    p.add_argument("--split_by", choices=["pair", "material"], default=None,
                   help="pair: random split over pairs (default); "
                        "material: material-disjoint split (leakage diagnostic)")
    p.add_argument("--finetune_epochs",
                   type=int,
                   default=None,
                   help="Fine-tuning epoch count")
    p.add_argument("--finetune_lr", type=float, default=None, help="Fine-tuning learning rate")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        encoding="utf-8",
        handlers=[
            logging.FileHandler("result.log"), 
            logging.StreamHandler()
        ]
    )

    p = argparse.ArgumentParser(
        description="Polymer Mobility Bayesian Personalized Ranking (BPR) via D-MPNN")
    p.add_argument("--mode",
                   choices=["train", "predict", "finetune"],
                   default="train")
    add_common_args(p)
    args = p.parse_args()

    model_arg_map = {
        "hidden_size": "hidden_size",
        "depth": "depth",
        "dropout": "dropout",
        "ffn_hidden": "ffn_hidden",
    }
    train_arg_map = {
        "csv_path": "csv",
        "save_dir": "save_dir",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "patience": "patience",
        "val_ratio": "val_ratio",
        "test_ratio": "test_ratio",
        "seed": "seed",
        "max_repeats": "max_repeats",
        "rank_weight": "rank_weight",
        "reg_weight": "reg_weight",
        "censored_weight": "censored_weight",
        "delta_scale": "delta_scale",
        "early_stop_metric": "early_stop_metric",
        "scheduler": "scheduler",
        "split_by": "split_by",
    }

    if args.mode == "train":
        model_config = merge_args(ModelConfig, args, model_arg_map)
        train_config = merge_args(TrainingConfig, args, train_arg_map)
        results = train(model_config, train_config)
        logger.info("\n===== Final Test Metrics =====")
        for k, v in results["test_metrics"].items():
            logger.info(f"  {k:30s}: {v:.4f}")

    elif args.mode == "finetune":
        finetune_config = merge_args(
            FinetuneConfig, args, {
                "csv_path": "csv",
                "checkpoint_path": "checkpoint",
                "save_dir": "save_dir",
                "finetune_epochs": "finetune_epochs",
                "batch_size": "batch_size",
                "lr": "finetune_lr",
                "weight_decay": "weight_decay",
                "seed": "seed",
                "max_repeats": "max_repeats",
                "val_ratio": "val_ratio",
                "patience": "patience",
            })
        results = finetune(finetune_config)
        logger.info(
            f"\n===== Final model saved to {results['final_checkpoint']} ====="
        )

    elif args.mode == "predict":
        if not args.predict_csv:
            p.error("--predict_csv is required for prediction mode")
        predict_config = merge_args(
            PredictConfig, args, {
                "predict_csv": "predict_csv",
                "checkpoint_path": "checkpoint",
                "output_path": "output",
                "max_repeats": "max_repeats",
                "batch_size": "batch_size",
            })
        df_new = pd.read_csv(predict_config.predict_csv)
        result = predict_batch(
            df_new=df_new,
            checkpoint_path=predict_config.checkpoint_path,
            output_path=predict_config.output_path,
            max_repeats=predict_config.max_repeats,
            batch_size=predict_config.batch_size,
        )
        logger.info(result[["Materials_1", "Materials_2", "preferred_mu_e", "prob_mu_e", "preferred_mu_h", "prob_mu_h"]])
