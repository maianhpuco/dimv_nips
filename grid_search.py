import os 
import sys
sys.path.append("")
import numpy as np
from src.utils import get_directory
from src.clf import classification, grid_search
from src.load_data import load_data 
import argparse
import json
import re


def get_save_path(ds_name, mrate, exp_num, create_new_exp):
    if not os.path.exists("data/exp/mono"):
        os.makedirs("data/exp/mono")
    files  = os.listdir("data/exp/mono")
    
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
    _dir = os.path.join(
            [i for i get_directory(
                stage = 'exp', 
                mono_or_rand  = 'mono', 
                dataset_name = ds_name, 
                mrate = mrate,
                exp_num = exp_num 
                ).split("_")]) 
    

    print("saved folder ", _dir)
    return _dir 


def grid_search(
        algo, 
        ds_name, 
        missing_rates=None, 
        dryrun=None, 
        exp_num=None, 
        create_new_exp=False):

    if dryrun == 1:
        missing_rates = [missing_rates[0]] 


    print("Ground Truth Path: ", get_save_path(ds_name, 0, exp_num, create_new_exp))
    

    hparams, acc = grid_search(X, y)

    with open(os.path.join(ground_truth_path, "acc_ground_truth.json"), 'w') as f:
        json.dump({"acc": acc}, f) 

    with open(os.path.join(ground_truth_path, "hyperparamete.json"), 'w') as f:
        json.dump(hparams, f) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputing data after monotone missing created"
    )

    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--missing_rates", type=list, default=[0.6, 0.5, 0.4])
    parser.add_argument("--dryrun", type=int, default=0)
    parser.add_argument("--exp_num", type=int, default=None)
    parser.add_argument("--create_new_exp", type=int, default=None)

    args = parser.parse_args()

    #r = get_exp_num("mnist", .5)


    grid_search(args.algo, 
            args.ds, 
            missing_rates = args.missing_rates, 
            dryrun = args.dryrun, 
            exp_num = args.exp_num, 
            create_new_exp = args.create_new_exp
            )
