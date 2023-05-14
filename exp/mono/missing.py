import os
import sys

sys.path.append("")  # run from the root directory

from src.load_data import load_data
from src.utils import create_image_monotone_missing, get_directory
import argparse
import numpy as np
import re
import yaml

with open("exp/cfg.yml", "r") as f:
    cfg_loaded = yaml.safe_load(f)


def get_save_path(ds_name, mrate, exp_num):
    if not os.path.exists("data/missing/mono"):
        os.makedirs("data/missing/mono")
    files = os.listdir("data/missing/mono")

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
    _dir = get_directory(stage='missing',
                         mono_or_rand='mono',
                         dataset_name=ds_name,
                         mrate=mrate,
                         exp_num=exp_num)

    print("saved folder check", _dir)
    return _dir


def create_missing(
    dataset_name,
    exp_nums,
    missing_rates,
):
    X, y = load_data(dataset_name)

    w = cfg_loaded["data_meta"][dataset_name]["im_width"]
    h = cfg_loaded["data_meta"][dataset_name]["im_height"]

    for exp_num in exp_nums:
        for mrate in missing_rates:

            save_folder = get_save_path(dataset_name, mrate, exp_num)
            print(save_folder)
            Xmiss, _ = create_image_monotone_missing(X, 0.5, mrate, mrate, w, h)

            file_name = "Xmiss.npz"
            file_path = os.path.join(save_folder, file_name)

            #np.savez(file_path, Xmiss)
            print("save at path {}".format(file_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create missing data (monotone)")
    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--exp_nums", type=list, default=[0])
    parser.add_argument("--missing_rates", type=list, default=[0.6, 0.5, 0.4])

    args = parser.parse_args()

    create_missing(args.ds, args.exp_nums, args.missing_rates)
