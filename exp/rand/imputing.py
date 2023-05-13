import argparse
import os
import sys
# import time

import numpy as np
import yaml

sys.path.append("")

from src.imputer import *
from src.load_data import load_data 
from src.utils import get_directory, rmse_calc
import json 
import re 

with open("exp/cfg.yml", "r") as f:
    cfg = yaml.safe_load(f)


hyperparameters = lambda algo: cfg["hyper"]["rand"][algo] 
hyperparameters_mnist =  lambda algo: cfg["hyper"]["rand_mnist"][algo] 

def get_save_path(ds_name, mrate, exp_num, create_new_exp):
    if not os.path.exists("data/exp/rand"):
        os.makedirs("data/exp/rand")
    files  = os.listdir("data/exp/rand")
     
    print(files)
    pattern = "^\d+$"
    regex = re.compile(pattern)
    integer_files = [f for f in files if regex.match(f)]
     
    
    if integer_files == []: 
        exp_num = 0 
    elif exp_num is not None:
        exp_num = exp_num 
    elif create_new_exp == True:
        exp_num =  np.amax(np.array([int(i) for i in integer_files])) + 1
    else: 
        exp_num =  np.amax(np.array([int(i) for i in integer_files]))
    _dir = get_directory(
            stage = 'exp', 
            mono_or_rand  = 'rand', 
            dataset_name = ds_name, 
            mrate = mrate,
            exp_num = exp_num 
            )

    print("saved folder ", _dir)
    return _dir 


def impute(algo, ds_name, missing_rates=None, dryrun=None, exp_num=None, create_new_exp=False):

    print("dryrun mode (1 is dryrun, 0 is actuall running): ", dryrun)
    # get_data
    if dryrun == 1:
        missing_rates = [missing_rates[0]]

    for mrate in missing_rates:
        print("--------------------------------------")
        print("Missing rate: ", mrate) 
        missing_dir = get_directory(
            stage="missing",
            mono_or_rand="rand",
            dataset_name=ds_name,
            mrate=mrate
        )

        X_gtruth, y_gtruth = load_data(ds_name)
        print("X shape", X_gtruth.shape) 
        if dryrun == 1:
            X_gtruth, y_gtruth = X_gtruth[:1000, ], y_gtruth[:1000]  
        X_miss_path  = os.path.join(missing_dir, "Xmiss.npz")

        X_miss = np.load(X_miss_path)["arr_0"]

        if dryrun == 1:
            X_miss = X_miss[:1000,]

        try:
            
            hyperparams = hyperparameters(algo)
            if ds_name in ('mnist', 'fashion_mnist'):
                hyperparams = hyperparameters_mnist(algo)

        except Exception as e:
            print(e)
            hyperparams = {}

        if algo == "mean":
            from src.imputer import mean_imputer
            Ximp, duration = mean_imputer(X_miss, **hyperparams)

        elif algo == "softimpute":
            from src.imputer import softimpute_imputer
            Ximp, duration = softimpute_imputer(X_miss, **hyperparams)

        elif algo == "mice":
            from src.imputer import mice_imputer
            Ximp, duration = mice_imputer(X_miss, **hyperparams)

        elif algo == "imputepca":
            from src.imputer import imputepca_imputer
            Ximp, duration = imputepca_imputer(X_miss, **hyperparams)

        elif algo == "em":
            from src.imputer import em_imputer
            Ximp, duration = em_imputer(X_miss, **hyperparams)

        elif algo == "missforest":
            from src.imputer import missforest_imputer
            Ximp, duration = missforest_imputer(X_miss, **hyperparams)

        elif algo == "knn":
            from src.imputer import knn_imputer
            Ximp, duration = knn_imputer(X_miss, **hyperparams)

        elif algo == "gain":
            from src.imputer import gain_imputer
            Ximp, duration = gain_imputer(X_miss, **hyperparams)

        elif algo == "ginn":
            from src.imputer import ginn_imputer
            Ximp, duration = ginn_imputer(X_miss, y_gtruth, **hyperparams)

        elif algo == 'vae':
            from src.imputer import vae_imputer
            Ximp, duration = vae_imputer(X_miss, **hyperparams)

        elif algo == "dimv":
            from src.imputer import dimv_imputer
            Ximp, duration = dimv_imputer(X_miss, **hyperparams)

        else:
            raise NotImplementedError(f"{algo} is not implemented")


        print(
                ">> Complete imputation {} with shape {}"
                .format(algo, Ximp.shape)
            )
        
        save_folder = get_save_path(ds_name, mrate, exp_num, create_new_exp)

        mmask = np.isnan(X_miss)
        rmse = rmse_calc(X_gtruth, Ximp, mmask)
        print("Algorithm: {}, Total time: {}, RMSE: {} ".format(algo, duration, rmse))

        assert np.sum(np.isnan(Ximp)) == 0, "imputed data Ximp still contain missing"
        

        np.savez(os.path.join(save_folder, "X_imp_{}.npz".format(algo)), Ximp)

        with open(os.path.join(save_folder, "rmse_{}.json".format(algo)), 'w') as f:
            json.dump({"rmse": rmse, "time": duration}, f) 

        print(
                ">> Complete save {} with at path {}"
                .format(algo, save_folder)
            )
        print("--------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after randtone missing created"
    )

    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--missing_rates", type=list, default=[.1, .2,.3,.4,.5,.6,.7,.8,.9])
    parser.add_argument("--dryrun", type=int, default=0)
    parser.add_argument("--exp_num", type=int, default=None)
    parser.add_argument("--create_new_exp", type=int, default=None)

    args = parser.parse_args()

    #r = get_exp_num("mnist", .5)


    impute(
            args.algo, 
            args.ds, 
            missing_rates = args.missing_rates, 
            dryrun = args.dryrun, 
            exp_num = args.exp_num, 
            create_new_exp = args.create_new_exp
            )
