# Polymer Carrier Mobility Bayesian Personalized Ranking (BPR)

A Bayesian Personalized Ranking (BPR) prediction model for polymer carrier mobility based on D-MPNN.

## Overview

This project uses a deep learning model to predict electron mobility (μ_e) and hole mobility (μ_h) of polymer materials. It employs a Bayesian Personalized Ranking (BPR) approach by comparing the mobility rankings of two polymers.

### Pipeline

1. **Polymer Cyclization**: Convert polymer repeat units (with `*` markers) into cyclic model compounds
2. **Molecular Encoding**: Use chemprop v2 D-MPNN for molecular feature encoding + additional feature concatenation
3. **Bayesian Personalized Ranking (BPR) Loss**: BPR loss (margin ranking + delta regression)
4. **Multi-task Prediction**: Simultaneously predict electron mobility (μ_e) and hole mobility (μ_h)

## Installation

```bash
pip install torch pandas numpy scikit-learn rdkit chemprop
```

## Data Format

### Training Data (CSV)

| Column | Description |
|--------|-------------|
| `Materials_1` | Material 1 name |
| `Polymer_1` | Material 1 SMILES (with `*` connection points) |
| `Materials_2` | Material 2 name |
| `Polymer_2` | Material 2 SMILES |
| `conjugation_1/2` | Conjugation features |
| `Isomer_1/2` | Isomer features |
| `CentroSymmetry_1/2` | Centrosymmetry |
| `E_LUMO (eV)_1/2` | LUMO energy level |
| `E_HOMO (eV)_1/2` | HOMO energy level |
| `mu_e_1/2` | Electron mobility (training label) |
| `mu_h_1/2` | Hole mobility (training label) |

### Prediction Data (CSV)

For prediction, `mu_e` and `mu_h` columns are not required; other columns follow the same format.

## Usage

### Training Mode

```bash
python polymer_ranking.py --mode train --csv contrastive_paired.csv
```

Optional parameters:
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_size`: Hidden layer dimension (default: 300)
- `--depth`: MPNN depth (default: 6)
- `--dropout`: Dropout rate (default: 0.1)
- `--patience`: Early stopping patience (default: 25)
- `--val_ratio`: Validation set ratio (default: 0.1)
- `--test_ratio`: Test set ratio (default: 0.1)

### Fine-tuning Mode

Load a pre-trained model for fine-tuning:

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### Prediction Mode

Predict rankings for new polymer pairs:

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

## Model Configuration

Default configurations are defined in `ModelConfig`, `TrainingConfig`, `FinetuneConfig`, and `PredictConfig` classes:

```python
@dataclass
class ModelConfig:
    hidden_size: int = 300    # Hidden layer dimension
    depth: int = 6            # Number of MPNN message passing layers
    dropout: float = 0.1      # Dropout rate
    ffn_hidden: int = 256     # FFN hidden layer dimension

@dataclass
class TrainingConfig:
    epochs: int = 1000        # Number of training epochs
    batch_size: int = 32      # Batch size
    lr: float = 1e-3          # Learning rate
    patience: int = 25        # Early stopping patience
    val_ratio: float = 0.1    # Validation set ratio
    test_ratio: float = 0.1   # Test set ratio
    margin: float = 0.2       # Ranking loss margin
```

## Output Description

The prediction result CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `Materials_1` | Material 1 name |
| `Materials_2` | Material 2 name |
| `score_mu_e_1` | Material 1 electron mobility score |
| `score_mu_e_2` | Material 2 electron mobility score |
| `preferred_mu_e` | Material with higher electron mobility |
| `score_mu_h_1` | Material 1 hole mobility score |
| `score_mu_h_2` | Material 2 hole mobility score |
| `preferred_mu_h` | Material with higher hole mobility |

## Checkpoint Files

Two checkpoints are saved during training:

- `best_model.pt`: Best model on validation set
- `final_model.pt`: Final model after training completes

Checkpoint contains:
- `model_state`: Model weights
- `config`: Model configuration
- `scaler`: Feature normalizer
