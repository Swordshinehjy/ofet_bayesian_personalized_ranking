"""
Polymer Carrier Mobility Prediction
=====================================
Pipeline:
  1. Polymer repeat unit (with *) cyclization → cyclic model compound
  2. chemprop v2 D-MPNN encoding + additional feature concatenation
  3. Bayesian Personalized Ranking (BPR) Loss + Delta Regression
  4. Multi-task simultaneous prediction of electron mobility (mu_e) and hole mobility (mu_h)
"""

import logging

from .config import EXTRA_COLS, EXTRA_DIM, TASK_NAMES, NUM_TASKS, CP_FEATURE_DIM, DEFAULT_DEPTH, OLIGOMER_MAX_REPEATS, CENSOR_LOG_MARGIN, SPLIT_MODES, min_span_for_depth, ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from .featurizer import CustomMultiHotAtomFeaturizer, create_featurizer, get_feature_dims
from .chemistry import (
    cyclize_polymer_with_cp_marking,
    cyclize_df,
    extra_feat,
    load_and_preprocess,
    get_attachment_atoms,
    attachment_topological_distance,
    get_cp_atoms,
    cp_topological_distance,
    build_linear_oligomer,
    choose_oligomer_repeats,
)
from .dataset import PairDataset, CachedPairDataset, collate_fn, collate_cached_batch
from .model import DMPNNEncoder, PolymerRankingModel
from .loss import MultiTaskBayesianRankingLoss
from .training import EarlyStopping, compute_delta_scale, leakage_report, prepare_splits, run_experiment, train, finetune
from .predict import load_checkpoint, predict_pair, predict_batch
from .cli import main

logger = logging.getLogger(__name__)

try:
    from chemprop.data.molgraph import MolGraph
    from chemprop.data.collate import BatchMolGraph, collate_batch
    logger.info("chemprop v2 loaded.")
except ImportError:
    logger.warning("chemprop not installed. Run: pip install chemprop>=2.0.0")
