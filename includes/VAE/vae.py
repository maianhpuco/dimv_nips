import argparse
import logging
import os
import pickle
import random
import shutil

from matplotlib import pyplot as plt

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tf.enable_v2_behavior()

BATCH_SIZE = 256
TEST_BATCH_SIZE = 1
NUM_IMPORTANCE_SAMPLES = 256  # for test set marginal likelihood estimation
MISSINGNESS_TYPE = "MNAR"  # MCAR or MNAR
MISSINGNESS_COMPLEXITY = "COMPLEX"  # SIMPLE or COMPLEX
MARGINAL_LL_MC_SAMPLES = 100
DATASET = "MNIST"  # MNIST or SVHN
LIKELIHOOD = "BERNOULLI"  # BERNOULLI or LOGISTIC_MIXTURE
# DATASET = 'SVHN' # MNIST or SVHN
# LIKELIHOOD = 'LOGISTIC_MIXTURE' # BERNOULLI or LOGISTIC_MIXTURE
LOGISTIC_MIXTURE_COMPONENTS = 1
if DATASET == "MNIST":
    Z_DIM = 50
    img_dim = 28
else:
    Z_DIM = 200
    img_dim = 32

NUM_RUNS = 5
VISUALIZE = False  # plot reconstructions
VERBOSE = False  # print verbose updates after each training epoch
