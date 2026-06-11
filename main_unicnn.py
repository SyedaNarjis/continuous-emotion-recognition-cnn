#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_unicnn.py
==============
Unimodal (and multimodal) CNN for continuous emotion recognition using the
CreativeIT dataset with IS17 feature sets.

Predicts Activation (A), Valence (V), or Dominance (D) from audio and/or
motion features using a leave-one-session-out cross-validation scheme.

Performance on the test set (CCC, PCC, RMSE) is appended to ``results.txt``.

Usage
-----
    python main_unicnn.py [OPTIONS]

    # Audio only
    python main_unicnn.py --audio --no-motion

    # Motion only
    python main_unicnn.py --no-audio --motion

    # Audio + motion (default)
    python main_unicnn.py --audio --motion

    # Full options
    python main_unicnn.py \\
        --audio-path  /path/to/audio_features/ \\
        --motion-path /path/to/motion_features/ \\
        --label-path  /path/to/labels/ \\
        --results     results.txt \\
        --epochs      25 \\
        --batch-size  200

Dependencies
------------
    pip install tensorflow keras numpy pandas matplotlib

References
----------
    CreativeIT dataset: https://sail.usc.edu/CreativeIT/ImprovDatabase.htm
    IS17 feature set:   http://www.compare.openaudio.eu/

Author  : SyedaNarjisFatima
License : MIT
"""

import argparse
import fnmatch
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional GPU memory configuration
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    # Limit per-process GPU memory to avoid starving other users on shared
    # machines. Set to None (or remove) for exclusive GPU access.
    GPU_MEMORY_FRACTION = 0.45
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass  # CPU-only environment; no GPU config needed

from tensorflow.keras.callbacks import EarlyStopping

from calc_scores import calc_scores
from load_features import d_2d_to_3d, load_all, win_mean
from modelcnn_tester import get_model
from write_predictions import write_predictions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
TEMPORAL_BATCH_SIZE   = 120   # temporal window size (frames)
TEMPORAL_BATCH_STRIDE = 20    # hop between windows (frames)
EPOCHS                = 25
BATCH_SIZE            = 200
RANDOM_SEED           = 7

SESSIONS = ["April23*", "April30*", "March5*", "May13*"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a unimodal/multimodal CNN for emotion recognition "
                    "on the CreativeIT dataset (IS17 feature set).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data paths ----------------------------------------------------------
    parser.add_argument("--audio-path",  default="features/audio/",
                        help="Directory containing audio feature CSV files.")
    parser.add_argument("--motion-path", default="features/motion/",
                        help="Directory containing motion feature CSV files.")
    parser.add_argument("--label-path",  default="features/labels/",
                        help="Directory containing label CSV files.")
    parser.add_argument("--pred-path",   default="test_predictions/",
                        help="Directory for writing test predictions.")
    parser.add_argument("--results",     default="results.txt",
                        help="Output file for session-wise and overall scores.")

    # -- Modality flags -------------------------------------------------------
    modality = parser.add_argument_group("modality flags")
    modality.add_argument("--audio",    dest="audio",    action="store_true",  default=True)
    modality.add_argument("--no-audio", dest="audio",    action="store_false")
    modality.add_argument("--motion",   dest="motion",   action="store_true",  default=True)
    modality.add_argument("--no-motion",dest="motion",   action="store_false")
    modality.add_argument("--type",     dest="use_type", action="store_true",  default=True,
                          help="Include type feature column.")
    modality.add_argument("--no-type",  dest="use_type", action="store_false")
    modality.add_argument("--act",      dest="use_act",  action="store_true",  default=False,
                          help="Include cross-subject activation feature column.")
    modality.add_argument("--no-act",   dest="use_act",  action="store_false")

    # -- Training hyperparameters ---------------------------------------------
    train = parser.add_argument_group("training")
    train.add_argument("--epochs",      type=int, default=EPOCHS)
    train.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    train.add_argument("--window-size", type=int, default=TEMPORAL_BATCH_SIZE,
                       help="Temporal window size (frames).")
    train.add_argument("--hop",         type=int, default=TEMPORAL_BATCH_STRIDE,
                       help="Stride between temporal windows (frames).")
    train.add_argument("--seed",        type=int, default=RANDOM_SEED)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    np.random.seed(args.seed)

    if not args.audio and not args.motion:
        log.error("At least one modality (--audio or --motion) must be enabled.")
        return 1

    # Collect feature paths in order (audio first, then motion)
    path_features = []
    if args.audio:
        path_features.append(args.audio_path)
    if args.motion:
        path_features.append(args.motion_path)

    Path(args.pred_path).mkdir(parents=True, exist_ok=True)

    # -- Log run metadata ----------------------------------------------------
    exp_name   = "Unimodal CNN — CreativeIT IS17"
    model_name = "SingleModelCNN"
    feats_desc = "CreativeIT IS17 feature set"
    pred_dim   = "Activation prediction"
    run_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        f"\n{'=' * 60}\n"
        f"  {exp_name}\n"
        f"  Model      : {model_name}\n"
        f"  Features   : {feats_desc}\n"
        f"  Prediction : {pred_dim}\n"
        f"  Modalities : audio={args.audio}, motion={args.motion}\n"
        f"  Run time   : {run_time}\n"
        f"{'=' * 60}\n"
    )
    log.info(header)
    with open(args.results, "a") as f:
        f.write(header)

    # -- Leave-one-session-out cross-validation ------------------------------
    cumulative_scores = np.zeros(3)   # CCC, PCC, RMSE

    for sess in SESSIONS:
        log.info("Test session: %s", sess)

        # Split into train / test sessions
        train_sessions = [s for s in SESSIONS if s != sess]

        files_test  = fnmatch.filter(os.listdir(path_features[0]), sess)
        files_train = [
            f
            for t_sess in train_sessions
            for f in fnmatch.filter(os.listdir(path_features[0]), t_sess)
        ]

        # Load features and labels
        train_data   = load_all(files_train, path_features, args.use_type, args.use_act)
        test_data    = load_all(files_test,  path_features, args.use_type, args.use_act)
        train_labels = load_all(files_train, [args.label_path], True, True)
        test_labels  = load_all(files_test,  [args.label_path], True, True)

        # Reshape into temporal windows
        train_feats  = d_2d_to_3d(train_data, args.window_size, args.hop)[..., np.newaxis]
        test_feats   = d_2d_to_3d(test_data,  args.window_size, args.hop)[..., np.newaxis]
        train_labels = win_mean(train_labels, args.window_size, args.hop)
        test_labels  = win_mean(test_labels,  args.window_size, args.hop)

        log.info("Train shape: %s   Test shape: %s",
                 train_feats.shape, test_feats.shape)

        # Build and train model
        model = get_model(args.window_size, train_feats.shape[2], class_num=1)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="auto"
        )
        model.fit(
            train_feats, train_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stop],
        )

        # Evaluate
        predictions = model.predict(test_feats)
        scores = calc_scores(test_labels[:, 0], predictions[:, 0])
        cumulative_scores += np.abs(scores)

        log.info("Session %s — CCC: %.4f  PCC: %.4f  RMSE: %.4f",
                 sess, scores[0], scores[1], scores[2])

        with open(args.results, "a") as f:
            f.write(f"Test session : {sess}\n")
            f.write(f"CCC / PCC / RMSE : {scores}\n")

    # -- Overall results -----------------------------------------------------
    mean_scores = cumulative_scores / len(SESSIONS)
    log.info("Overall weighted average — CCC: %.4f  PCC: %.4f  RMSE: %.4f",
             *mean_scores)

    with open(args.results, "a") as f:
        f.write(f"\nWeighted average over all sessions:\n{mean_scores}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
