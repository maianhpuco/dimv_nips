import argparse
import os
# import sys
# import time

import numpy as np
import yaml

# from src.imputer import *
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
        # _func_name = "{}_{}".format(algo, "imputer")
        # print("func: ", _func_name)
        # _func = eval(_func_name)

        try:
            hyperparams = hyperparameters(algo)
        except Exception as e:
            print(e)
            hyperparams = {}

        if algo == "mean":
            from src.imputer import mean_imputer
            Ximp, duration = mean_imputer(Xmtrain, **hyperparams)

        if algo == "softimpute":
            from src.imputer import softimpute_imputer
            Ximp, duration = softimpute_imputer(Xmtrain, **hyperparams)

        if algo == "mice":
            from src.imputer import mice_imputer
            Ximp, duration = mice_imputer(Xmtrain, **hyperparams)

        if algo == "imputepca":
            from src.imputer import imputepca_imputer
            Ximp, duration = imputepca_imputer(Xmtrain, **hyperparams)

        if algo == "em":
            from src.imputer import em_imputer
            Ximp, duration = em_imputer(Xmtrain, **hyperparams)

        if algo == "missforest":
            from src.imputer import missforest_imputer
            Ximp, duration = missforest_imputer(Xmtrain, **hyperparams)

        if algo == "knn":
            from src.imputer import knn_imputer
            Ximp, duration = knn_imputer(Xmtrain, **hyperparams)

        if algo == "gain":
            from src.imputer import gain_imputer
            Ximp, duration = gain_imputer(Xmtrain, **hyperparams)

        if algo == "ginn":
            from src.imputer import ginn_imputer
            Ximp, duration = ginn_imputer(Xmtrain, **hyperparams)

        if algo == "dimv":
            from src.imputer import dimv_imputer
            Ximp, duration = dimv_imputer(Xmtrain, **hyperparams)

        print(">> Complete imputation {} with shape {}"
              .format(algo, Ximp.shape))

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
