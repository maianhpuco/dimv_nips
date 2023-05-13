import os 
import sys
sys.path.append("")

import re
import pandas as pd
from src.utils import get_directory
import json 

def get_save_path(ds_name, mrate, exp_num):
    if not os.path.exists("data/exp/mono"):
        os.makedirs("data/exp/mono")
    files  = os.listdir("data/exp/mono")
    
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



def build_one_report(dataset_name, missing_rates, exp_num):

    rmse_df = pd.DataFrame(columns=["rmse", 'method', 'missing_rate', 'time'])
    acc_df  = pd.DataFrame(columns=['acc', 'method', 'missing_rate'])

    for mrate in missing_rates:
        
        dir_path = get_save_path(dataset_name, mrate, exp_num)
        print(dir_path)
        files = os.listdir(dir_path)
      
        rmse_pattern = r"rmse_(.*)\.json"
        acc_pattern = r"acc_(.*)\.json"


        for fname in files: 
            if re.match(rmse_pattern, fname):
                frmse = open(os.path.join(dir_path, fname))
                rmse_json = json.load(frmse)
            
                rmse_json.update(
                        {
                            'method': re.match(rmse_pattern, fname).group(1), 
                            'missing_rate': mrate
                            }
                        )
                   
                
                rmse_df = pd.concat([rmse_df, pd.DataFrame([rmse_json])], ignore_index=True)
            if re.match(acc_pattern, fname):
                facc = open(os.path.join(dir_path, fname))
                acc_json = json.load(facc)
                acc_json.update(
                        {
                            'method': re.match(acc_pattern, fname).group(1), 
                            'missing_rate': mrate
                            }
                        )
               
                acc_df = pd.concat([acc_df, pd.DataFrame([acc_json])], ignore_index=True)
    
    rmse_df.loc[rmse_df["method"] == "em", "time"] *= 3600
     
    acc_df['acc'] = acc_df['acc'] * 100
   
    return (rmse_df, acc_df)

def build_report(datasets, missing_rates, exp_num):
    rmse_dfs = pd.DataFrame(columns=["dataset", "rmse", 'method', 'missing_rate', 'time'])
    acc_dfs  = pd.DataFrame(columns=["dataset", 'acc', 'method', 'missing_rate'])
    
    for ds in datasets:
        rmse_df, acc_df = build_one_report(ds, missing_rates, exp_num)
        rmse_df['dataset'] = ds
        acc_df['dataset']  = ds 

        acc_dfs = pd.concat([acc_dfs, acc_df], ignore_index=True)
        rmse_dfs = pd.concat([rmse_dfs, rmse_df], ignore_index=True)

        print(acc_dfs)
        #print(rmse_dfs)

    
    acc_report = acc_dfs.pivot(
            index=['dataset','missing_rate'], columns='method', values='acc')
    print(acc_report)

    acc_latex = acc_report.to_latex(
            index=True, escape=False)
    print(acc_latex)


    rmse_report = rmse_dfs.pivot(
            index=['dataset','missing_rate'], columns='method', values='rmse')
            
    rmse_latex = rmse_report.to_latex(
            index=True, escape=False)
    print(rmse_latex)

if __name__ == "__main__":
    build_report(["mnist", "fashion_mnist"], [.6, .5, .4], 0)
