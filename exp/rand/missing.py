import os
import sys

sys.path.append("")  # run from the root directory

from src.load_data import load_data
from src.utils import create_randomly_missing, get_directory
import argparse
import numpy as np
import re
import yaml

with open("exp/cfg.yml", "r") as f:
    cfg_loaded = yaml.safe_load(f)


def get_save_path(ds_name, mrate, exp_num, stage):
    if not os.path.exists("data/{}/rand".format(stage)):
        os.makedirs("data/{}/rand".format(stage))
    files = os.listdir("data/{}/rand".format(stage))

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
    _dir = get_directory(stage=stage,
                         mono_or_rand='rand',
                         dataset_name=ds_name,
                         mrate=mrate,
                         exp_num=exp_num)

    return _dir


def create_missing(
    dataset_name,
    exp_nums,
    missing_rates,
):
    X, y = load_data(dataset_name)

    for exp_num in exp_nums:
        for mrate in missing_rates:

            save_folder = get_save_path(dataset_name, mrate, exp_num, "missing")
            print(save_folder)
            Xmiss = create_randomly_missing(X, mrate)

            file_name = "Xmiss.npz"
            file_path = os.path.join(save_folder, file_name)

            np.savez(file_path, Xmiss)
            print("save at path {}".format(file_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create missing data (randtone)")
    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--exp_nums", type=list, default=[i for i in range(10)])
    parser.add_argument("--missing_rates",
                        type=list,
                        default=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
    args = parser.parse_args()
    create_missing(args.ds, args.exp_nums, args.missing_rates)
