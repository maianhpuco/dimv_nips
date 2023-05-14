import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.environ.get("ROOT")
sys.path.append(ROOT)
from src.utils import get_directory


# hyperparameters = lambda algo: cfg["hyper"]["rand"][algo]
def hyperparameters(algo):
    with open("exp/cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)

    return cfg["hyper"]["rand"][algo]


sys.path.append("")

from src.imputer import *
from src.load_data import load_data
from src.utils import get_directory, rmse_calc
import json
import re

with open("exp/cfg.yml", "r") as f:
    cfg = yaml.safe_load(f)

get_hparams = lambda algo: cfg["hyper"]["rand"][algo]
get_hparam_mnist = lambda algo: cfg["hyper"]["rand_mnist"][algo]


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
    else:
        exp_num = np.amax(np.array([int(i) for i in integer_files]))
    _dir = get_directory(stage=stage,
                         mono_or_rand='rand',
                         dataset_name=ds_name,
                         mrate=mrate,
                         exp_num=exp_num)

    return _dir


def impute(
    algo,
    ds_name,
    missing_rates=None,
    dryrun=None,
    exp_num=None,
):
    # get_data
    if dryrun:
        # missing_rates = [missing_rates[0]]
        pass

    for mrate in missing_rates:
        print("--------------------------------------")
        print("Missing rate: ", mrate)

        missing_dir = get_save_path(ds_name, mrate, exp_num, "missing")

        X_gtruth, y_gtruth = load_data(ds_name)
        print("X shape", X_gtruth.shape)
        if dryrun == 1:
            X_gtruth, y_gtruth = X_gtruth[
                :1000,
            ], y_gtruth[:1000]

        X_miss_path = os.path.join(missing_dir, "Xmiss.npz")

        Xmiss = np.load(X_miss_path)['arr_0']
        print("X_miss.shape", Xmiss.shape)
        if dryrun == 1:
            Xmiss = Xmiss[
                :1000,
            ]
        print("X_miss.shape", Xmiss.shape)

        try:
            hyperparams = get_hparams(algo)
            if ds_name in ('mnist', 'fashion_mnist'):
                hyperparams = get_hparams_mnist(algo)

        except Exception as e:
            print(e)
            hyperparams = {}

        if algo == "mean":
            from src.imputer import mean_imputer
            Ximp, duration = mean_imputer(Xmiss, **hyperparams)

        elif algo == "softimpute":
            from src.imputer import softimpute_imputer
            Ximp, duration = softimpute_imputer(Xmiss, **hyperparams)

        elif algo == "mice":
            from src.imputer import mice_imputer
            Ximp, duration = mice_imputer(Xmiss, **hyperparams)

        elif algo == "imputepca":
            from src.imputer import imputepca_imputer
            Ximp, duration = imputepca_imputer(Xmiss, **hyperparams)

        elif algo == "em":
            from src.imputer import em_imputer
            Ximp, duration = em_imputer(Xmiss, **hyperparams)

        elif algo == "missforest":
            from src.imputer import missforest_imputer
            Ximp, duration = missforest_imputer(Xmiss, **hyperparams)

        elif algo == "knn":
            from src.imputer import knn_imputer
            Ximp, duration = knn_imputer(Xmiss, **hyperparams)

        elif algo == "gain":
            from src.imputer import gain_imputer
            Ximp, duration = gain_imputer(Xmiss, **hyperparams)

        elif algo == "ginn":
            from src.imputer import ginn_imputer
            Ximp, duration = ginn_imputer(Xmiss, **hyperparams)

        elif algo == "dimv":
            from src.imputer import dimv_imputer
            Ximp, duration = dimv_imputer(Xmiss, **hyperparams)

        elif algo == "vae":
            from src.imputer import vae_imputer
            Ximp, duration = vae_imputer(Xmiss)
            # TODO: adding hyperparams

        else:
            raise NotImplementedError(f"{algo} is not implemented")

        print(">> Complete imputation {} with shape {}".format(
            algo, Ximp.shape))

        save_folder = get_save_path(ds_name, mrate, exp_num, "exp")

        mmask = np.isnan(Xmiss)
        rmse = rmse_calc(X_gtruth, Ximp, mmask)

        print("Algorithm: {}, Total time: {}, RMSE: {} ".format(
            algo, duration, rmse))

        assert np.sum(
            np.isnan(Ximp)) == 0, "imputed data Ximp still contain missing"

        np.savez(os.path.join(save_folder, "X_imp_{}.npz".format(algo)), Ximp)

        with open(os.path.join(save_folder, "rmse_{}.json".format(algo)),
                  'w') as f:
            json.dump({"rmse": rmse, "time": duration}, f)

        print(">> Complete save {} with at path {}".format(algo, save_folder))
        print("--------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after randtone missing created")

    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--missing_rates",
                        type=list,
                        default=[i * .1 for i in range(1, 10)])
    parser.add_argument("--dryrun", type=int, default=0)
    parser.add_argument("--exp_num", type=int, default=None)
    args = parser.parse_args()

    print(args.missing_rates)
    if isinstance(args.missing_rates, str):
        mrates = [float(i) for i in args.missing_rates.split(" ")]
        print(mrates)
    else:
        mrates = args.missing_rates

    impute(args.algo,
           args.ds,
           missing_rates=mrates,
           dryrun=args.dryrun,
           exp_num=args.exp_num)
