import argparse
import sys
import os

import numpy as np
import yaml

root = os.environ.get("ROOT")
sys.path.append(root)  # run from the root directory

from src.load_data import load_data
from src.utils import create_randomly_missing
from src.utils import get_directory


with open("exp/cfg.yml", "r") as f:
    cfg_loaded = yaml.safe_load(f)


def create_missing(dataset_name, missing_rates):
    X, y = load_data(dataset_name)

    for mrate in missing_rates:
        missing_directory = get_directory(
            stage="missing",
            mono_or_rand="rand",
            dataset_name=dataset_name,
            mrate=mrate
        )

        Xmiss = create_randomly_missing(
            X, mrate
        )  # create_randomly_missing return only 1 value

        # file_name:  remove, doesn't get used later
        np.savez_compressed(missing_directory + "Xmiss.npz", Xmiss)  # typo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create missing data (randomly)")

    parser.add_argument("--ds", type=str, default=None)
    missing_rates = [i / 10.0 for i in range(1, 10)]

    args = parser.parse_args()
    create_missing(args.ds, missing_rates)
