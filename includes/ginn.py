import os
import numpy as np
import sys

sys.path.append("includes/GINN")
from ginn.core import GINN

import argparse
import csv
import numpy as np
from sklearn import model_selection, preprocessing
from ginn.utils import degrade_dataset, data2onehot

import time
import pandas as pd
import sys

import dgl
import logging

# Set logging level to error
logging.basicConfig(level=logging.ERROR)


def ginn_run(missing_x_train, y_train, kwargs):
    num_cols = range(missing_x_train.shape[1])
    cat_cols = []

    x_train = missing_x_train

    cx_train, cx_train_mask = missing_x_train, 1 - np.isnan(missing_x_train) * 1

    cx_tr = np.c_[cx_train, y_train]

    mask_tr = np.c_[cx_train_mask, np.ones(y_train.shape)]

    [oh_x, oh_mask, oh_num_mask, oh_cat_mask,
     oh_cat_cols] = data2onehot(cx_tr, mask_tr, num_cols, cat_cols)

    oh_x_tr = oh_x[:x_train.shape[0], :]

    oh_mask_tr = oh_mask[:x_train.shape[0], :]
    oh_num_mask_tr = oh_mask[:x_train.shape[0], :]
    oh_cat_mask_tr = oh_mask[:x_train.shape[0], :]

    imputer = GINN(oh_x_tr, oh_mask_tr, oh_num_mask_tr, oh_cat_mask_tr,
                   oh_cat_cols, num_cols, cat_cols)

    imputer.fit()
    imputed_tr = imputer.transform()

    print("done imputation, result's shape", imputed_tr.shape)
    return imputed_tr
