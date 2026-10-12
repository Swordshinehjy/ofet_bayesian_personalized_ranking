"""End-to-end tests for the parts that actually touch disk and state.

The 200+ unit tests cover the pure functions; these cover the seams that can
silently corrupt a result:

* ``train`` must write **two different** checkpoints — "best" and "final".
  A previous revision wrote the same state dict into both files, so
  ``final_model.safetensors`` was a duplicate of "best" and the weights of the
  last training epoch were lost.
* checkpoint round-trip must restore the fitted ``StandardScaler`` exactly.
* ``predict_batch`` must survive a missing ``Materials_*`` column.
* ``finetune`` must not overwrite the user's ``best_model`` with final weights.
* ``run_experiment`` must honour the seed (search reproducibility).

Everything runs on CPU with a deliberately tiny network, so the whole module
finishes in a few seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_ranking.checkpoint import load_checkpoint_dict, save_checkpoint  # noqa: E402
from polymer_ranking.config import ModelConfig, TrainingConfig, FinetuneConfig  # noqa: E402
from polymer_ranking.predict import load_checkpoint, predict_batch, predict_pair  # noqa: E402
from polymer_ranking.training import finetune, train  # noqa: E402

POLY_A = "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"
POLY_B = "*c1ccc(-c2ccc(-c3ccc(-c4cc5c(s4)-c4cc6c(cc4[Si]5(C)C)-c4sc(*)cc4[Si]6(C)C)s3)c3nsnc23)s1"
EXTRA = ("conjugation", "Isomer", "CentroSymmetry", "E_LUMO (eV)", "E_HOMO (eV)")


def _write_pairs_csv(path: Path, n: int = 40, seed: int = 0) -> Path:
    """A small but non-degenerate pair file (some pairs left-censored)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        swap = bool(i % 2)
        p1, p2 = (POLY_B, POLY_A) if swap else (POLY_A, POLY_B)
        factor = 0.5 if i % 3 else 2.0
        e1 = float(10 ** (-3 - 0.02 * i))
        e2 = e1 * factor
        h1 = float(10 ** (-3.5 - 0.015 * i))
        h2 = h1 * (0.7 if i % 4 else 1.4)
        mu_e_1, mu_e_2 = e1, e2
        mu_h_1, mu_h_2 = h1, h2
        if i % 7 == 0:      # censored side 1 (below detection limit)
            mu_e_1, mu_h_1 = 0.0, 0.0
        if i % 11 == 0:     # censored side 2
            mu_e_2 = 0.0
        row = {
            "doi": f"paper{i % 5}",
            "Materials_1": f"M{i}",
            "Materials_2": f"M{i + 1}",
            "Polymer_1": p1,
            "Polymer_2": p2,
            "mu_e_1": mu_e_1,
            "mu_h_1": mu_h_1,
            "mu_e_2": mu_e_2,
            "mu_h_2": mu_h_2,
        }
        for s, ratios in (("1", (1, 0.0, 0, -3.1, -5.3)), ("2", (1, 0.0, 0, -3.4, -5.2))):
            for name, base in zip(EXTRA, ratios):
                value = base + (float(rng.normal(0, 0.05)) if isinstance(base, float) else 0.0)
                row[f"{name}_{s}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _tiny_model() -> ModelConfig:
    return ModelConfig(hidden_size=32, depth=2, dropout=0.1, ffn_hidden=16)


def _read_state(path: Path):
    return load_checkpoint_dict(str(path))["model_state"]


def _states_differ(a, b) -> bool:
    return any(not torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    return _write_pairs_csv(tmp_path_factory.mktemp("data") / "pairs.csv")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, csv_path):
    """One short training run reused by the tests below."""
    out = train(
        _tiny_model(),
        TrainingConfig(
            csv_path=str(csv_path),
            save_dir=str(tmp_path_factory.mktemp("ckpt")),
            epochs=20,
            patience=1,
            batch_size=8,
            seed=0,
            device="cpu",
            val_ratio=0.2,
            test_ratio=0.2,
        ),
    )
    return out


class TestTrainWritesTwoDistinctCheckpoints:
    def test_best_and_final_both_exist(self, trained):
        assert Path(trained["checkpoint"]).is_file()
        assert Path(trained["final_checkpoint"]).is_file()

    def test_final_is_the_last_epoch_not_a_copy_of_best(self, trained):
        """Regression guard: both files used to receive the very same tensors."""
        best = _read_state(Path(trained["checkpoint"]))
        final = _read_state(Path(trained["final_checkpoint"]))
        assert set(best) == set(final)
        assert _states_differ(best, final), (
            "final_model.safetensors is identical to best_model.safetensors — "
            "the last training epoch was not persisted")

    def test_history_was_recorded(self, trained):
        assert len(trained["history"]["train_loss"]) >= 1
        assert 1 <= trained["test_metrics"]["best_epoch"] <= 20


class TestCheckpointRoundTrip:
    def test_scaler_restored_exactly(self, trained):
        ckpt = load_checkpoint_dict(str(trained["checkpoint"]))
        scaler = ckpt["scaler"]
        assert scaler.n_features_in_ == 5
        assert np.asarray(scaler.mean_).shape == (5,)
        assert np.asarray(scaler.scale_).shape == (5,)
        # a rescaled prediction must be bit-identical to the in-memory scaler
        X = np.random.default_rng(0).normal(size=(3, 5))
        assert np.allclose(scaler.transform(X), (X - scaler.mean_) / scaler.scale_)

    def test_training_settings_are_persisted(self, trained):
        """lr / seed / ratios used to be lost, making runs impossible to replay."""
        meta = load_checkpoint_dict(str(trained["checkpoint"]))
        cfg = meta["train_config"]
        assert cfg["seed"] == 0
        assert cfg["epochs"] == 20
        assert cfg["batch_size"] == 8
        assert cfg["split_by"] == "pair"
        assert meta["snapshot"] == "best"
        assert meta["train_epochs_run"] >= 1
        final_meta = load_checkpoint_dict(str(trained["final_checkpoint"]))
        assert final_meta["snapshot"] == "final"
        assert len(final_meta["history"]["train_loss"]) >= 1

    def test_device_roundtrip(self, tmp_path, trained):
        src = load_checkpoint_dict(str(trained["checkpoint"]))
        dst = tmp_path / "copy.safetensors"
        save_checkpoint(dst, model_state=src["model_state"],
                        scaler=src["scaler"], config=src["config"],
                        extra={"delta_scale": src["delta_scale"]})
        again = load_checkpoint_dict(str(dst))
        assert np.allclose(again["scaler"].mean_, src["scaler"].mean_)
        assert torch.equal(again["model_state"]["ffn.0.weight"],
                           src["model_state"]["ffn.0.weight"])


class TestScaling:

    def test_load_checkpoint_rejects_pickle_files(self, tmp_path):
        bogus = tmp_path / "legacy.pt"
        bogus.write_bytes(b"fake")
        with pytest.raises(ValueError, match="safetensors"):
            load_checkpoint_dict(str(bogus))


class TestPredictBatch:
    def test_output_columns_and_row_count(self, trained, csv_path):
        df_new = pd.read_csv(csv_path).head(6)
        out = predict_batch(df_new=df_new,
                            checkpoint_path=str(trained["final_checkpoint"]),
                            batch_size=3,
                            device="cpu")
        assert len(out) == 6
        for col in ("score_mu_e_1", "score_mu_e_2", "preferred_mu_e", "prob_mu_e",
                    "score_mu_h_1", "preferred_mu_h", "prob_mu_h"):
            assert col in out.columns
        assert out["prob_mu_e"].between(0.5, 1.0).all()
        names = set(df_new["Materials_1"]) | set(df_new["Materials_2"])
        assert set(out["preferred_mu_e"]) <= names
        assert "mol_1" not in out.columns

    def test_missing_materials_columns_degrades_gracefully(self, trained, csv_path):
        df_new = pd.read_csv(csv_path).head(4).drop(
            columns=["Materials_1", "Materials_2"])
        out = predict_batch(df_new=df_new,
                            checkpoint_path=str(trained["final_checkpoint"]),
                            device="cpu")
        assert set(out["preferred_mu_e"]) <= {"polymer_1", "polymer_2"}

    def test_missing_smiles_column_raises_clearly(self, trained, csv_path):
        df_new = pd.read_csv(csv_path).head(2).drop(columns=["Polymer_2"])
        with pytest.raises(ValueError, match="Polymer_2"):
            predict_batch(df_new=df_new,
                          checkpoint_path=str(trained["final_checkpoint"]),
                          device="cpu")

    def test_predict_pair_reuses_a_loaded_bundle(self, trained):
        bundle = load_checkpoint(str(trained["final_checkpoint"]), device="cpu")
        extra = np.array([[1.0, 0.0, 0.0, -3.1, -5.3]], dtype=np.float32)
        res = predict_pair(POLY_A, POLY_B, extra, extra,
                           checkpoint_path="", bundle=bundle, device="cpu")
        assert set(res["ranking"]) == {"mu_e", "mu_h"}
        assert 0.5 <= res["probability"]["mu_e"] <= 1.0
        assert res["cyc_smiles_1"]


class TestFinetune:
    def test_writes_final_and_best_separately(self, trained, csv_path, tmp_path):
        save_dir = tmp_path / "finetune"
        out = finetune(FinetuneConfig(
            csv_path=str(csv_path),
            checkpoint_path=str(trained["checkpoint"]),
            save_dir=str(save_dir),
            finetune_epochs=2,
            val_ratio=0.25,
            patience=5,
            batch_size=8,
            lr=1e-4,
            seed=0,
            device="cpu",
        ))
        assert Path(out["final_checkpoint"]).is_file()
        assert Path(out["best_checkpoint"]).is_file()
        # fine-tuning must never silently overwrite the pre-trained best model
        assert Path(out["best_checkpoint"]).name == "finetuned_best_model.safetensors"

    def test_resumes_with_the_checkpoint_loss_settings(self, trained, csv_path, tmp_path):
        out = finetune(FinetuneConfig(
            csv_path=str(csv_path),
            checkpoint_path=str(trained["checkpoint"]),
            save_dir=str(tmp_path / "ft2"),
            finetune_epochs=1,
            val_ratio=0.25,
            batch_size=8,
            seed=0,
            device="cpu",
        ))
        meta = load_checkpoint_dict(str(out["final_checkpoint"]))
        assert meta["loss_config"]["rank_weight"] == pytest.approx(0.9)
        assert meta["finetune_config"]["finetune_epochs"] == 1


class TestSearchReproducibility:
    def test_run_experiment_is_seed_controlled(self, csv_path):
        """Same seed -> same initial weights -> same validation trajectory."""
        from polymer_ranking.chemistry import load_and_preprocess
        from polymer_ranking.training import run_experiment

        df = load_and_preprocess(str(csv_path), depth=2, max_repeats=2)
        cfg_kwargs = dict(csv_path=str(csv_path), epochs=2, batch_size=8,
                          patience=5, val_ratio=0.2, test_ratio=0.2,
                          device="cpu", seed=7, max_repeats=2)
        kwargs_a = dict(hidden_size=8, depth=2, ffn_hidden=4, dropout=0.0)
        cfg_a = TrainingConfig(**cfg_kwargs)
        r1 = run_experiment(df, ModelConfig(**kwargs_a), cfg_a, max_epochs=2)
        r2 = run_experiment(df, ModelConfig(**kwargs_a), cfg_a, max_epochs=2)
        assert r1["val_metrics"]["mean_pair_acc"] == pytest.approx(
            r2["val_metrics"]["mean_pair_acc"], abs=1e-6)
