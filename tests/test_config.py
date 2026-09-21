import pytest

from polymer_ranking.config import (
    EXTRA_COLS,
    EXTRA_DIM,
    TASK_NAMES,
    NUM_TASKS,
    CP_FEATURE_DIM,
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
)


class TestConstants:
    def test_extra_cols_count(self):
        assert len(EXTRA_COLS) == 5

    def test_extra_dim_matches_cols(self):
        assert EXTRA_DIM == len(EXTRA_COLS)

    def test_task_names(self):
        assert TASK_NAMES == ["mu_e", "mu_h"]

    def test_num_tasks(self):
        assert NUM_TASKS == 2

    def test_cp_feature_dim(self):
        assert CP_FEATURE_DIM == 1

    def test_extra_cols_format_string(self):
        for col in EXTRA_COLS:
            assert "{s}" in col


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.hidden_size == 300
        assert cfg.depth == 6
        assert cfg.dropout == 0.1
        assert cfg.ffn_hidden == 256
        assert cfg.extra_dim == EXTRA_DIM
        assert cfg.num_tasks == NUM_TASKS
        assert cfg.aggregation == "mean"

    def test_custom_values(self):
        cfg = ModelConfig(hidden_size=128, depth=3, dropout=0.2)
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.dropout == 0.2

    def test_to_dict(self):
        cfg = ModelConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_size"] == 300
        assert d["depth"] == 6
        assert "extra_dim" in d

    def test_from_dict(self):
        d = {"hidden_size": 128, "depth": 3, "dropout": 0.2, "ffn_hidden": 64}
        cfg = ModelConfig.from_dict(d)
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.ffn_hidden == 64

    def test_from_dict_ignores_unknown_keys(self):
        d = {"hidden_size": 128, "unknown_key": 999}
        cfg = ModelConfig.from_dict(d)
        assert cfg.hidden_size == 128
        assert not hasattr(cfg, "unknown_key")

    def test_roundtrip(self):
        cfg = ModelConfig(hidden_size=200, depth=4)
        d = cfg.to_dict()
        cfg2 = ModelConfig.from_dict(d)
        assert cfg == cfg2


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 200
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 1e-3
        assert cfg.seed == 42
        assert cfg.patience == 30
        assert cfg.rank_weight == 0.9
        assert cfg.reg_weight == 0.1
        assert cfg.censored_weight == 1.0
        assert cfg.delta_scale is None
        assert cfg.early_stop_metric == "pair_acc"
        assert cfg.scheduler == "plateau"
        assert cfg.split_by == "pair"

    def test_custom_values(self):
        cfg = TrainingConfig(epochs=50, lr=0.01)
        assert cfg.epochs == 50
        assert cfg.lr == 0.01

    def test_invalid_test_ratio(self):
        with pytest.raises(ValueError):
            TrainingConfig(test_ratio=0.0)

    def test_invalid_val_ratio(self):
        with pytest.raises(ValueError):
            TrainingConfig(val_ratio=0.95, test_ratio=0.1)

    def test_invalid_early_stop_metric(self):
        with pytest.raises(ValueError):
            TrainingConfig(early_stop_metric="accuracy")

    def test_invalid_scheduler(self):
        with pytest.raises(ValueError):
            TrainingConfig(scheduler="step")

    def test_invalid_split_by(self):
        with pytest.raises(ValueError):
            TrainingConfig(split_by="scaffold")

    def test_larger_batch_allows_larger_val_ratio(self):
        cfg = TrainingConfig(val_ratio=0.2, test_ratio=0.1)
        assert cfg.val_ratio == 0.2


class TestFinetuneConfig:
    def test_defaults(self):
        cfg = FinetuneConfig()
        assert cfg.finetune_epochs == 20
        assert cfg.lr == 1e-5
        assert cfg.checkpoint_path == "checkpoints/best_model.pt"
        assert cfg.val_ratio == 0.1
        assert cfg.patience == 10

    def test_custom_values(self):
        cfg = FinetuneConfig(finetune_epochs=20, lr=5e-6)
        assert cfg.finetune_epochs == 20
        assert cfg.lr == 5e-6


class TestPredictConfig:
    def test_defaults(self):
        cfg = PredictConfig()
        assert cfg.predict_csv == ""
        assert cfg.checkpoint_path == "checkpoints/best_model.pt"
        assert cfg.output_path == "predictions.csv"
        assert cfg.batch_size == 32

    def test_custom_values(self):
        cfg = PredictConfig(predict_csv="test.csv", output_path="out.csv")
        assert cfg.predict_csv == "test.csv"
        assert cfg.output_path == "out.csv"
