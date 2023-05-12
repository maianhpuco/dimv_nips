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

    _dir = get_directory(
            stage = 'exp', 
            mono_or_rand  = 'mono', 
            dataset_name = ds_name, 
            mrate = mrate,
            exp_num = exp_num 
            )

    print("saved folder ", _dir)
    return _dir 


def classify(
        algo, 
        ds_name, 
        missing_rates=None, 
        dryrun=None, 
        exp_num=None, 
        create_new_exp=False):

    if dryrun==1:
        missing_rates = [missing_rates[0]] 
        print("this is dryrun mode")
    
    exp_dir = get_save_path(ds_name, 0, exp_num, create_new_exp)
    

    ground_truth_path = "/".join(exp_dir.split("/")[:-1])
    print("Ground Truth Path: ", ground_truth_path)
    
        
    acc_f        = open(os.path.join(ground_truth_path, "acc_ground_truth.json"))
    hparams_f    = open(os.path.join(ground_truth_path, "hyperparameters.json"))
    indices_f =  open(os.path.join(ground_truth_path, "indices.json"))
        
    hparams = json.load(hparams_f)
    acc_gtruth = json.load(acc_f)
    indices_dict = json.load(indices_f)
    train_indices = indices_dict["train_indices"]
    test_indices  = indices_dict["test_indices"]
    
    print("train, test indice {} {} ".format(len(train_indices), len(test_indices)))
    print("Best acc ", acc_gtruth) 
    print("Best params ", hparams)
    
    X, y = load_data(ds_name)

    for mrate in missing_rates:
        save_folder = get_save_path(
            ds_name, mrate, exp_num, create_new_exp)

        X_imp_path = os.path.join(save_folder, "X_imp_{}.npz".format(algo))     

        Ximp_train = np.load(X_imp_path)["arr_0"][train_indices]
        y_train  = y[train_indices]

        Xgt_test  = X[test_indices]
        y_test    = y[test_indices]

        if dryrun == 1:
            print("dry runing")
            Ximp_train, y_train = Ximp_train[:1000], y_train[:1000]
            Xgt_test, y_test    = Xgt_test[:200], y_test[:200]
        
        acc = classification(Ximp_train, y_train, Xgt_test, y_test, hparams)

        
        with open(os.path.join(save_folder, "acc_{}.json".format(algo)), 'w') as f:
            json.dump({"acc": acc}, f) 
        print(
                ">> -------Complete save --- {} ----with acc {}  with at path {}"
                .format(algo, acc, save_folder)
            )



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


    classify(args.algo, 
            args.ds, 
            missing_rates = args.missing_rates, 
            dryrun = args.dryrun, 
            exp_num = args.exp_num, 
            create_new_exp = args.create_new_exp
            )
