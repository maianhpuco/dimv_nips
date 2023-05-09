import argparse
import os
# import sys
# import time

import numpy as np
import yaml

from src.imputer import *
from src.utils import get_directory
# sys.path.append("")

with open("exp/cfg.yml", "r") as f:
    cfg = yaml.safe_load(f)


hyperparameters = lambda algo: cfg["hyper"]["rand"][algo]


def impute(algo, ds_name, missing_rates=None, dryrun=False):
    # get_data
    if dryrun:
        missing_rates = [missing_rates[0]]

    for mrate in missing_rates:
        missing_dir = get_directory(
            stage="missing",
            mono_or_rand="mono",
            dataset_name=ds_name, mrate=mrate
        )

        X_mtrain_path = os.path.join(missing_dir, "Xtrain.npz")
        X_mtest_path = os.path.join(missing_dir, "Xtest.npz")

        print(X_mtest_path)

        Xmtrain = np.load(X_mtrain_path)["arr_0"]
        Xmtest = np.load(X_mtest_path)["arr_0"]

        if dryrun:
            Xmtrain = Xmtrain[:1000,]
            Xmtest = Xmtest[:300,]

        # imputing and save the result
        _func_name = "{}_{}".format(algo, "imputer")
        print("func: ", _func_name)
        _func = eval(_func_name)

        try:
            hyperparams = hyperparameters(algo)
        except Exception as e:
            print(e)
            hyperparams = {}

        Ximp, duration = _func(Xmtrain, **hyperparams)

        print(">> Complete imputation {} with shape {}".format(algo, Ximp.shape))
        print("Total time: {}".format(duration))

        # asser no missing data left in the imputed (algorithm is converged)
        # assert np.sum(np.isnan(train_imp)) == 0, "train_imp still contain missing"
        # assert np.sum(np.isnan(test_imp)) == 0 , "test_imp still contain missing"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after monotone missing created"
    )

    parser.add_argument("ds", type=str, default=None)
    parser.add_argument("algo", type=str)
    parser.add_argument("--missing_rates", type=list, default=[0.6, 0.5, 0.4])
    parser.add_argument("--dryrun", type=bool, default=False)

    args = parser.parse_args()

    impute(args.algo, args.ds, args.missing_rates, args.dryrun)
