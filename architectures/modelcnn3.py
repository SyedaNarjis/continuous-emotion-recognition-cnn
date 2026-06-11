#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
architectures/modelcnn3.py
==========================
Experimental architecture 3 — shallow single-block CNN with sigmoid activation.

Architecture
------------
    Conv2D(8, 39×39) → MaxPool → BatchNorm
    → Flatten
    → Dense(1024, Sigmoid) → Dropout(0.25) → BatchNorm
    → Dense(output)

Notes
-----
- Single convolutional block with a large 39×39 kernel.
- Uses sigmoid (rather than ReLU) in the dense layer — an intentional
  experiment to observe the effect on regression performance.
- Loss: MSE | Optimizer: Adam
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: F401
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def get_model(temporal_batch_size: int, n_features: int, class_num: int = 1) -> Sequential:
    """Build and compile experimental CNN architecture 3.

    Parameters
    ----------
    temporal_batch_size:
        Number of time frames per input window.
    n_features:
        Number of features per frame.
    class_num:
        Number of output dimensions.

    Returns
    -------
    keras.models.Sequential
        Compiled Keras model.
    """
    model = Sequential(name="ModelCNN3")

    # -- Single convolutional block with large kernel ------------------------
    model.add(Conv2D(8, kernel_size=(39, 39), padding="same", activation="relu",
                     input_shape=(temporal_batch_size, n_features, 1), name="conv1"))
    model.add(MaxPooling2D(name="pool1"))
    model.add(BatchNormalization(name="bn_conv1"))

    # -- Fully-connected head (sigmoid activation) ---------------------------
    model.add(Flatten(name="flatten"))

    model.add(Dense(1024, activation="sigmoid", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))

    # -- Output --------------------------------------------------------------
    model.add(Dense(class_num, name="output"))

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


# Backward-compatible alias
getModel = get_model
