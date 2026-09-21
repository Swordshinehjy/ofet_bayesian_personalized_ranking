# Polymer Carrier Mobility Bayesian Personalized Ranking (BPR)

A Bayesian Personalized Ranking (BPR) model for polymer carrier mobility based on a D-MPNN
(Directed Message Passing Neural Network). The model is trained on **pairs** of polymers and
jointly ranks electron mobility (μ\_e) and hole mobility (μ\_h).

## Pipeline

```
repeat unit (with *) ──► cyclization ──► cyclic model compound ──► D-MPNN ──► FFN ──► score
                          (short units are first expanded into an oligomer)
                                                                     │
                     extra features (conjugation, isomer, symmetry, LUMO, HOMO) ──┘
```

1. **Polymer cyclization**: the repeat unit (two `*` attachment points) is closed into a cyclic
   model compound. When the backbone span `L` between the two attachment atoms is shorter than
   `2 * depth + 1` (13 for the default `depth=6`), the unit is first expanded head-to-tail into an
   oligomer (at most `--max_repeats` units, default 8) so that the artificial ring-closure bond
   stays outside the receptive field of the D-MPNN.
2. **Molecular encoding**: chemprop v2 D-MPNN, plus a connection-point (CP) bit on the atom
   features.
3. **Loss**: `MultiTaskBayesianRankingLoss` = BPR ranking term + delta-regression term, with
   explicit handling of left-censored labels (see below).
4. **Multi-task prediction**: μ\_e and μ\_h are predicted at the same time by a siamese network
   with shared weights.

## Censored labels: mobility = 0 is *below the detection limit*

A mobility of `0` in this dataset is **not** a missing measurement and **not** a real zero — it is
**left-censored**: the true value is below what the experiment could resolve. Such a value still
carries ordering information — it is lower than every measured mobility — but it carries no numeric
information.

How many pairs fall into each case depends on the data file; `hyperparam_search.py --stage analyze`
prints the current counts.

The two parts of the loss therefore treat it differently:

| Situation | Ranking (BPR) | Delta regression |
|-----------|---------------|------------------|
| both values measured (`> 0`) | yes | yes |
| one value censored (`= 0`) | yes — a measured value is always above the detection limit | **no** — the numeric difference is unknown |
| both censored | no — the ordering is undecidable | no |

Implementation details:

* `load_and_preprocess` adds `ok_{task}_{side}` (bool, "was measured") and fills the `log_`
  column of a censored entry with a sentinel placed one decade *below* the smallest measured
  value of that task. The sentinel only exists so that `sign(y1 - y2)` points the right way; it is
  never used as a regression target.
* Pairwise accuracy is computed on all pairs with a decidable ordering (including the
  one-censored ones); Spearman rho only on pairs with two measured values.
* `avg_prob` is the mean probability the model assigns to the **correct** direction,
  `σ(sign(y1−y2)·(s1−s2))`, so it is high only when the model is both right and confident.

> Previously `mu = 0` was clipped to `1e-12` before `log10`, i.e. turned into `-12`. Every pair with
> two censored values thereby became a spurious `"tie"` regression target, and every pair with one
> censored value got a fake 12-decade difference, which dominated the MSE term and inflated all
> reported metrics.

## Delta-regression scale

The BPR term lives in `[0, log 2]`, while a raw log10 mobility difference is of order one (its
std is printed by `--stage analyze` as `*_delta_std`). The regression target is therefore divided by
`delta_scale` — the std of the measured log10 differences, estimated from the training split, or set
explicitly with `--delta_scale` — so the two terms stay comparable.

## Hyper-parameter analysis

`hyperparam_search.py` checks whether the current hyper-parameters are reasonable, measures the
loss weighting on *this* dataset, and searches for a good configuration.

```bash
python hyperparam_search.py --stage analyze
python hyperparam_search.py --stage check
python hyperparam_search.py --stage all --n_trials 12 --max_epochs 45
```

| Stage | What it does |
|-------|--------------|
| `analyze` | Label availability (measured / one-censored / both-censored), scale of the log10 differences, class balance, material reuse, duplicate pairs, plus descriptor-only baselines (LUMO/HOMO heuristic and a logistic regression on the 5 extra features) |
| `check` | Rule-based sanity check of a configuration: model capacity vs. number of pairs, `lr` vs. `batch_size` (linear scaling), `epochs`/`patience`/`scheduler` consistency, loss weighting, `delta_scale` vs. the measured std, split strategy. Every finding carries a suggested fix; `--apply_fix` applies them |
| `search` | Random search over `lr`, `hidden_size`, `depth`, `dropout`, `weight_decay`, `batch_size`, `rank_weight`/`reg_weight`, `censored_weight`; each trial is early-stopped on validation pairwise accuracy and repeated on `--n_repeats` splits (the data is noisy, so a single split is not trustworthy) |
| `ablation` | Sweeps the loss weighting with everything else fixed: `rank_weight`/`reg_weight` and `censored_weight` |
| `all` | All of the above, then writes the recommended configuration |

Outputs: `hyperparam_search.csv`, `hyperparam_ablation.csv`, `best_hyperparams.json`.

### Re-measure after every data update

The optimal hyper-parameters depend on the data — in particular on the censoring rate and on how
noisy the reported mobilities are. **No result is hard-coded in this README**; treat
`best_hyperparams.json` as a snapshot of the last run and regenerate it whenever the dataset
changes:

```bash
D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage all \
    --csv <new_data>.csv --n_trials 12 --n_repeats 2 --max_epochs 60
```

What to look at:

* `--stage analyze` prints `*_both_measured_frac` (how much regression supervision is available),
  `*_delta_std` (→ `delta_scale`) and the descriptor-only baselines
  (`*_baseline_heuristic_acc`, `*_baseline_logreg_acc`). A D-MPNN that cannot clearly beat those is
  not worth the compute.
* `--stage ablation` measures the loss weighting on the current data. Reasoning to check it
  against:
  * a **small but non-zero** regression term usually wins — it anchors the scale of the scores
    without letting the noisy absolute mobilities dominate;
  * `censored_weight` close to **1** is expected, because the direction of a censored pair is
    certain (a measured mobility is definitely above the detection limit) and only the magnitude is
    unknown — and BPR does not use magnitudes. A much lower optimum means the censoring flag
    should be re-checked.
* The `TrainingConfig` defaults are the values measured on the bundled dataset; override them (or
  re-run the search with a larger `--n_trials` / `--n_repeats`) after a data update.

## Installation

```bash
pip install torch pandas numpy scikit-learn scipy rdkit chemprop
```

## Data Format

### Training Data (CSV)

| Column               | Description                                    |
| -------------------- | ---------------------------------------------- |
| `Materials_1`        | Material 1 name                                |
| `Polymer_1`          | Material 1 SMILES (with `*` connection points) |
| `Materials_2`        | Material 2 name                                |
| `Polymer_2`          | Material 2 SMILES                              |
| `conjugation_1/2`    | Conjugation features                           |
| `Isomer_1/2`         | Isomer features                                |
| `CentroSymmetry_1/2` | Centrosymmetry                                 |
| `E_LUMO (eV)_1/2`    | LUMO energy level                              |
| `E_HOMO (eV)_1/2`    | HOMO energy level                              |
| `mu_e_1/2`           | Electron mobility; `0` = below detection limit  |
| `mu_h_1/2`           | Hole mobility; `0` = below detection limit      |

### Prediction Data (CSV)

For prediction, `mu_e` / `mu_h` are not required; the other columns follow the same format.

## Usage

### Training

```bash
python polymer_ranking.py --mode train --csv contrastive_paired.csv
```

Optional arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 200 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 1e-3 | Learning rate |
| `--weight_decay` | float | 1e-3 | AdamW weight decay |
| `--hidden_size` | int | 300 | Hidden dimension |
| `--depth` | int | 6 | D-MPNN depth; also sets the cyclization span requirement `2 * depth + 1` |
| `--dropout` | float | 0.1 | Dropout rate |
| `--ffn_hidden` | int | 256 | FFN hidden dimension |
| `--max_repeats` | int | 8 | Max repeat units used when a unit is too short for cyclization |
| `--rank_weight` | float | 0.9 | Weight α of the BPR ranking term |
| `--reg_weight` | float | 0.1 | Weight β of the delta-regression term |
| `--censored_weight` | float | 1.0 | BPR weight of pairs containing a censored mobility |
| `--delta_scale` | float | auto | Scale of the regression target (std of the measured log10 differences) |
| `--early_stop_metric` | str | pair_acc | Quantity monitored by early stopping / LR scheduling (`pair_acc` / `loss`) |
| `--scheduler` | str | plateau | LR scheduler (`plateau` / `cosine`) |
| `--patience` | int | 30 | Early stopping patience |
| `--val_ratio` | float | 0.1 | Validation ratio |
| `--test_ratio` | float | 0.1 | Test ratio |
| `--split_by` | str | pair | `pair` = random split; `material` = material-disjoint split |
| `--seed` | int | 42 | Random seed |
| `--save_dir` | str | checkpoints | Checkpoint directory |

### Fine-tuning

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 20 \
    --finetune_lr 1e-5
```

Fine-tuning loads `delta_scale` and the loss weights from the checkpoint, so it continues with
exactly the same objective. It holds out `--val_ratio` of the data and early-stops on validation
pairwise accuracy instead of training blind on everything.

### Prediction

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

## Train/test leakage

Random pair-level splitting re-uses materials across the splits: the same polymer appears in several
pairs, so most test materials also occur in the training split. **This is expected and, for this
task, acceptable** — the candidate space is a set of alkyl-chain / substituent / functional-group
variants of known backbones, so the intended use is interpolation inside a known chemical family,
not extrapolation to unseen scaffolds.

What the code provides instead of pretending the leak does not exist:

* `--split_by material` builds a **material-disjoint** split (no material is shared between train
  and val/test; pairs whose two materials fall in different splits are dropped). Use it to obtain a
  pessimistic lower bound on the generalization performance.
* Every training run logs the leakage diagnostics
  (`test_material_overlap`, `test_pair_overlap`), and `hyperparam_search.py --stage analyze`
  reports material reuse (`material_reuse_mean`) and duplicate pairs (`n_duplicate_pairs`).

How to read the two numbers:

* `--split_by pair` is the **operating point** — the setting that matches how the model is meant to
  be used (ranking variants of known backbones).
* `--split_by material` is the **honest lower bound**. It is expected to be far worse, and a large
  part of the drop is not leakage but data loss: enforcing material-disjointness discards many
  pairs, so the model trains on much less data *and* is asked to extrapolate to polymers it has
  never seen a relative of. Do not use it as the headline metric.

## Notes

- **The candidate space is derived from the known chemical space, not entirely new chemistry.** The
  structures to be evaluated are variants of alkyl chains, substituents and functional groups on
  known skeletons. For a molecule from a completely different family the performance will drop.
- **The training data are literature values with high noise.** Repeat the hold-out several times
  (different `--seed`, or `--n_repeats` in the search) and average — a single split moves the
  pairwise accuracy by several points.
- Early stopping monitors **pairwise accuracy**, not the loss: the loss is a weighted mixture of two
  terms, so a lower loss does not necessarily mean a better ranking.

## Output

| Column | Description |
|--------|-------------|
| `score_mu_e_1` / `score_mu_e_2` | Electron-mobility scores of material 1 / 2 |
| `preferred_mu_e` | Material with the higher electron mobility |
| `prob_mu_e` | σ(|s1 − s2|), confidence of that preference |
| `score_mu_h_1` / `score_mu_h_2` | Hole-mobility scores |
| `preferred_mu_h` / `prob_mu_h` | Same for hole mobility |

Reported metrics: `*_pair_acc` (accuracy on pairs with a decidable ordering), `*_spearman`
(only on pairs with two measured mobilities), `*_avg_prob`, `*_n_rank`, `*_n_reg` (how many pairs
actually supervised each term), and `mean_pair_acc`.

## Checkpoints

Training writes two checkpoints:

- `checkpoints/best_model.pt` — best model on the validation split
- `checkpoints/final_model.pt` — model at the end of training

Fine-tuning overwrites `checkpoints/final_model.pt`.

Each checkpoint holds `model_state`, `scaler`, `config`, `delta_scale`, `loss_config` and
`split_by`.

## Tests

```bash
D:/anaconda3/envs/chemprop2/python.exe -m pytest tests -q
```
