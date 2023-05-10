import os
import sys

sys.path.append("")  # run from the root directory

from src.load_data import load_data
from src.utils import create_image_monotone_missing, get_directory
import argparse
import numpy as np

import yaml

with open("exp/cfg.yml", "r") as f:
    cfg_loaded = yaml.safe_load(f)


def create_missing(dataset_name, missing_rates):
    X, y = load_data(dataset_name)

    w = cfg_loaded["data_meta"][dataset_name]["im_width"]
    h = cfg_loaded["data_meta"][dataset_name]["im_height"]

    for mrate in missing_rates:
        missing_directory = get_directory(
            stage="missing",
            mono_or_rand="mono",
            dataset_name=dataset_name,
            mrate=mrate,
        )

        Xmiss, _ = create_image_monotone_missing(
            X, 0.5, mrate, mrate, w, h
        )

        file_name = "Xmiss.npz"
        file_path = os.path.join(missing_directory, file_name)

        np.savez(file_path, Xmiss)
        print("save at path {}".format(file_path)) 
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create missing data (monotone)")
    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--missing_rates", type=list, default=[0.6, 0.5, 0.4])

    args = parser.parse_args()

    create_missing(args.ds, args.missing_rates)
