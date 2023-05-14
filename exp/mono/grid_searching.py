import os
import sys

sys.path.append("")
import numpy as np
from src.utils import get_directory
from src.clf import classification, grid_search, train_test_splitting
from src.load_data import load_data
import argparse
import json
import re


def get_save_path(ds_name, exp_num, create_new_exp):
    if not os.path.exists("data/exp/mono"):
        os.makedirs("data/exp/mono")
    files = os.listdir("data/exp/mono")

    print(files)
    pattern = "^\d+$"
    regex = re.compile(pattern)
    integer_files = [f for f in files if regex.match(f)]

    if integer_files == []:
        exp_num = 0
    elif exp_num is not None:
        exp_num = exp_num
    elif create_new_exp == True:
        exp_num = np.amax(np.array([int(i) for i in integer_files])) + 1
    else:
        exp_num = np.amax(np.array([int(i) for i in integer_files]))

    exp_dir = get_directory(stage='exp',
                            mono_or_rand='mono',
                            dataset_name=ds_name,
                            mrate=0,
                            exp_num=exp_num)
    _dir = "/".join(exp_dir.split("/")[:-1])

    return _dir


def grid_searching(ds_name, dryrun=0, exp_num=None, create_new_exp=False):

    Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = \
            train_test_splitting(ds_name)

    print("grid search shape {} and test set shape {}", Xtrain.shape,
          Xtest.shape)
    if dryrun == 1:
        Xtrain, ytrain = Xtrain[
            :1000,
        ], ytrain[:1000]
        print(Xtrain.shape)
        print(ytrain.shape)

    gtruth_path = get_save_path(ds_name, exp_num, create_new_exp)
    print("Ground Truth Path: ", gtruth_path)

    hparams, acc = grid_search(Xtrain, ytrain, n_splits=5)

    with open(os.path.join(gtruth_path, "acc_ground_truth.json"), 'w') as f:
        json.dump({"acc": acc}, f)

    with open(os.path.join(gtruth_path, "hyperparameters.json"), 'w') as f:
        json.dump(hparams, f)

    with open(os.path.join(gtruth_path, "indices.json"), 'w') as f:
        json.dump(
            {
                'train_indices': list(train_indices),
                "test_indices": list(test_indices)
            }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after monotone missing created")

    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--dryrun", type=int, default=0)
    parser.add_argument("--exp_num", type=int, default=None)
    parser.add_argument("--create_new_exp", type=int, default=False)

    args = parser.parse_args()

    #r = get_exp_num("mnist", .5)

    grid_searching(args.ds,
                   dryrun=args.dryrun,
                   exp_num=args.exp_num,
                   create_new_exp=args.create_new_exp)
