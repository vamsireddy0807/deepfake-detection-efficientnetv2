"""Dataset indexing and the tf.data input pipeline.

Expected layout (xhlulu/140k-real-and-fake-faces on Kaggle)::

    <root>/{train,valid,test}/{real,fake}/*.jpg
"""

from __future__ import annotations

import glob
import os

import pandas as pd

IMG_SIZE = 160
BATCH_SIZE = 128
LABELS = {"real": 1, "fake": 0}


def index_split(root: str, split: str, seed: int = 42) -> pd.DataFrame:
    """(path, label) rows for one split, shuffled. label: 1 = real, 0 = fake."""
    rows = []
    for cls, lbl in LABELS.items():
        paths = glob.glob(os.path.join(root, split, cls, "*.jpg"))
        rows.extend((p, lbl) for p in paths)
    df = pd.DataFrame(rows, columns=["path", "label"])
    if df.empty:
        raise FileNotFoundError(f"no images under {os.path.join(root, split)}")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_ds(df: pd.DataFrame, training: bool = False, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Parallel JPEG decode and resize, light photometric augmentation for training, prefetch."""
    import tensorflow as tf

    autotune = tf.data.AUTOTUNE

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size], method="bilinear")
        return tf.cast(img, tf.float32), label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        # tight photometric ranges: deepfake datasets dislike heavy distortion
        img = tf.image.random_brightness(img, max_delta=15.0)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        return tf.clip_by_value(img, 0.0, 255.0), label

    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values))
    if training:
        ds = ds.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=autotune)
    if training:
        ds = ds.map(augment, num_parallel_calls=autotune)
    return ds.batch(batch_size).prefetch(autotune)


def make_flip_ds(df: pd.DataFrame, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Horizontally flipped copy of an evaluation set, for test-time augmentation."""
    import tensorflow as tf

    ds = make_ds(df, training=False, img_size=img_size, batch_size=batch_size).unbatch()
    ds = ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
