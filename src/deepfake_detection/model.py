"""EfficientNetV2-B0 classifier and the warm-up cosine learning-rate schedule."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetV2B0

from .data import IMG_SIZE


def build_model(img_size: int = IMG_SIZE):
    """ImageNet EfficientNetV2-B0 backbone (frozen) with a small dense head.

    The backbone includes its own rescaling, so it takes float pixels in [0, 255]. The
    output layer is float32 to keep the loss stable under mixed precision.
    """
    base = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        include_preprocessing=True,
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return Model(inputs, outputs), base


def unfreeze_top(base, fraction: float = 0.30) -> int:
    """Unfreeze the top ``fraction`` of backbone layers; returns how many are trainable."""
    n = len(base.layers)
    cut = int(n * (1 - fraction))
    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers[cut:]:
        layer.trainable = True
    return n - cut


@tf.keras.utils.register_keras_serializable(package="deepfake_detection")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to ``base_lr`` then cosine decay to zero at ``total_steps``."""

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        progress = (step - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cos = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}
