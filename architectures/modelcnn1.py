#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
architectures/modelcnn1.py
==========================
Experimental architecture 1 — deep 4-block CNN with shrinking kernels.

Architecture
------------
    Conv2D(16, 32×32) → MaxPool(2×1, stride 1×3)
    Conv2D(32, 16×16) → MaxPool(2×1, stride 1×3)
    Conv2D(64,  8×8)  → MaxPool(2×1, stride 1×3)
    Conv2D(92,  4×4)  → MaxPool(2×1, stride 1×3)
    → Flatten
    → Dense(1024, ReLU) → Dropout(0.5)
    → Dense(1024, ReLU) → Dropout(0.5)
    → Dense(output)

Notes
-----
- Four convolutional blocks with progressively smaller kernels and more
  filters (pyramid style).
- No BatchNormalization (commented out during experiments).
- Loss: MSE | Optimizer: Adam
"""

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def get_model(temporal_batch_size: int, n_features: int, class_num: int = 1) -> Sequential:
    """Build and compile experimental CNN architecture 1.

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
    model = Sequential(name="ModelCNN1")

    # -- Convolutional blocks (shrinking kernels, growing filters) -----------
    model.add(Conv2D(16, kernel_size=(32, 32), padding="same", activation="relu",
                     input_shape=(temporal_batch_size, n_features, 1), name="conv1"))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 3), name="pool1"))

    model.add(Conv2D(32, kernel_size=(16, 16), padding="same", activation="relu", name="conv2"))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 3), name="pool2"))

    model.add(Conv2D(64, kernel_size=(8, 8), padding="same", activation="relu", name="conv3"))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 3), name="pool3"))

    model.add(Conv2D(92, kernel_size=(4, 4), padding="same", activation="relu", name="conv4"))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 3), name="pool4"))

    # -- Fully-connected head ------------------------------------------------
    model.add(Flatten(name="flatten"))

    model.add(Dense(1024, activation="relu", name="fc1"))
    model.add(Dropout(0.5, name="drop1"))

    model.add(Dense(1024, activation="relu", name="fc2"))
    model.add(Dropout(0.5, name="drop2"))

    # -- Output --------------------------------------------------------------
    model.add(Dense(class_num, name="output"))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


# Backward-compatible alias
getModel = get_model
