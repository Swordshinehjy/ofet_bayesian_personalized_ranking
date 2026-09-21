import argparse

from polymer_ranking.cli import merge_args
from polymer_ranking.config import ModelConfig, TrainingConfig


class TestMergeArgs:
    def test_none_args_use_defaults(self):
        args = argparse.Namespace(
            hidden_size=None,
            depth=None,
            dropout=None,
            ffn_hidden=None,
        )
        mapping = {
            "hidden_size": "hidden_size",
            "depth": "depth",
            "dropout": "dropout",
            "ffn_hidden": "ffn_hidden",
        }
        cfg = merge_args(ModelConfig, args, mapping)
        assert cfg.hidden_size == 300
        assert cfg.depth == 6

    def test_non_none_args_override(self):
        args = argparse.Namespace(
            hidden_size=128,
            depth=3,
            dropout=None,
            ffn_hidden=None,
        )
        mapping = {
            "hidden_size": "hidden_size",
            "depth": "depth",
            "dropout": "dropout",
            "ffn_hidden": "ffn_hidden",
        }
        cfg = merge_args(ModelConfig, args, mapping)
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.dropout == 0.1

    def test_different_arg_names(self):
        args = argparse.Namespace(
            csv="my_data.csv",
            epochs=50,
            lr=None,
        )
        mapping = {
            "csv_path": "csv",
            "epochs": "epochs",
            "lr": "lr",
        }
        cfg = merge_args(TrainingConfig, args, mapping)
        assert cfg.csv_path == "my_data.csv"
        assert cfg.epochs == 50
        assert cfg.lr == 1e-3

    def test_empty_mapping(self):
        args = argparse.Namespace()
        cfg = merge_args(ModelConfig, args, {})
        assert cfg.hidden_size == 300

    def test_all_args_provided(self):
        args = argparse.Namespace(
            hidden_size=64,
            depth=2,
            dropout=0.5,
            ffn_hidden=32,
        )
        mapping = {
            "hidden_size": "hidden_size",
            "depth": "depth",
            "dropout": "dropout",
            "ffn_hidden": "ffn_hidden",
        }
        cfg = merge_args(ModelConfig, args, mapping)
        assert cfg.hidden_size == 64
        assert cfg.depth == 2
        assert cfg.dropout == 0.5
        assert cfg.ffn_hidden == 32
