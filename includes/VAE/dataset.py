import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow_datasets.core.utils import gcs_utils

# Fix tfds load
gcs_utils._is_gcs_disabled = True


def cast_img(example):
    example["b"] = tf.cast(example["b"], tf.float32)
    max_val = 255.0

    example["x"] = tf.cast(example["x"], tf.float32) / max_val
    example["x_zero_imp"] = tf.cast(example["x_zero_imp"], tf.float32) / max_val
    example["x_mean_imp"] = tf.cast(example["x_mean_imp"], tf.float32) / max_val
    return example


def generate_dataset(ds_name: str, X: np.ndarray, y: np.ndarray, batch_size: int, infer=False):
    """
    Description:
        X:corrupted input (with missing value), assuming shape = (N x W x H x C)
        y:label vector
    Adapting MA's dataset to VAE's dataset

    Returns:
        b: mask (1: observed data, 0: missing)
    """
    images = []
    masks = []
    zero_imp_images = []
    mean_imp_images = []
    # labels = []

    # Mean across all example, assuming shape of X = (N x 28 x 28 x 1)
    x_mean = np.nanmean(X, axis=0)
    # NOTE: In the original paper, the author used true mean on the original
    # data. In this experiment, we don't assume to have uncorrupted data, so we
    # compute x-mean on corrupted data

    # for example, label in zip(X, y):
    for example in X:
        b = (~np.isnan(example)) * 1.0

        # Constructing dataset
        images.append(example)
        masks.append(b)

        mask = np.isnan(example)
        example[mask] = 0
        zero_imp_images.append(b * example)
        mean_imp_images.append(b * example + (1 - b) * x_mean)
        # labels.append(label)

    dset = tf.data.Dataset.from_tensor_slices(
        {
            "x": images,
            "x_zero_imp": zero_imp_images,
            "x_mean_imp": mean_imp_images,
            "b": masks,
            # "y": labels,
        }
    )

    if not infer:
        dset = (
            dset.map(cast_img)
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        dset = (
            dset.map(cast_img)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return dset


if __name__ == "__main__":
    X = np.arange(120).reshape((-1, 3))
    X = X.astype(float)
    X[1, 1] = np.nan
    y = np.arange(40)
    dset = generate_dataset("mnist", X, y, 10)

    for example in dset:
        print(example)
