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


def impute(algo, ds_name, missing_rates=None, dryrun=False):

    # get_data
    if dryrun:
        missing_rates = [missing_rates[0]]

    for mrate in missing_rates:
        missing_dir = get_directory(
            stage="missing",
            mono_or_rand="rand",
            dataset_name=ds_name,
            mrate=mrate
        )

        Xmiss_path  = os.path.join(missing_dir, "Xmiss.npz")
        Xmiss = np.load(Xmiss_path)["arr_0"]


        if dryrun:
            Xmiss = Xmiss[:1000,]

        try:
            hyperparams = hyperparameters(algo)

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
            from src.vae_imputer import vae_imputer
            Ximp, duration = vae_imputer(Xmiss)
            # TODO: adding hyperparams

        else:
            raise NotImplementedError(f"{algo} is not implemented")

        print(
                ">> Complete imputation {} with shape {}"
                .format(algo, Ximp.shape)
            )

        print("Total time: {}".format(duration))

        assert np.sum(np.isnan(Ximp)) == 0, "imputed data Ximp still contain missing"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after monotone missing created"
    )

    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--missing_rates", type=list, default=[0.6, 0.5, 0.4])
    parser.add_argument("--dryrun", type=bool, default=False)

    args = parser.parse_args()

    impute(args.algo, args.ds, args.missing_rates, args.dryrun)
