#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelcnn_tester.py
==================
Defines the CNN architecture used for emotion regression.

Architecture
------------
    Conv2D(16, 1×13) → MaxPool → BatchNorm
    → Flatten
    → Dense(512) → Dropout(0.25) → BatchNorm
    → Dense(256) → Dropout(0.50) → BatchNorm
    → Dense(64)  → Dropout(0.50) → BatchNorm
    → Dense(output)

Loss: Mean Squared Error | Optimizer: Adam
"""

from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
)
from tensorflow.keras.models import Sequential


def get_model(temporal_batch_size: int, n_features: int, class_num: int = 1) -> Sequential:
    """Build and compile the CNN regression model.

    Parameters
    ----------
    temporal_batch_size:
        Number of time frames per input window (height of the 2-D input).
    n_features:
        Number of features per frame (width of the 2-D input).
    class_num:
        Number of output dimensions (1 for single-dimension regression).

    Returns
    -------
    keras.models.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential(name="UnimodalCNN")

    # -- Convolutional block -------------------------------------------------
    model.add(Conv2D(
        16, kernel_size=(1, 13), padding="same", activation="relu",
        input_shape=(temporal_batch_size, n_features, 1),
        name="conv1",
    ))
    model.add(MaxPooling2D(name="pool1"))
    model.add(BatchNormalization(name="bn_conv1"))

    # -- Fully-connected head ------------------------------------------------
    model.add(Flatten(name="flatten"))

    model.add(Dense(512, activation="relu", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))

    model.add(Dense(256, activation="relu", name="fc2"))
    model.add(Dropout(0.50, name="drop2"))
    model.add(BatchNormalization(name="bn_fc2"))

    model.add(Dense(64, activation="relu", name="fc3"))
    model.add(Dropout(0.50, name="drop3"))
    model.add(BatchNormalization(name="bn_fc3"))

    # -- Output layer --------------------------------------------------------
    model.add(Dense(class_num, name="output"))

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


# Backward-compatible alias used in legacy scripts
getModel = get_model
