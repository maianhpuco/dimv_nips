import numpy as np
import json
import pandas as pd
import re
import os
import sys

sys.path.append("")

from src.utils import get_directory


def get_save_path(ds_name, mrate, exp_num):
    if not os.path.exists("data/exp/mono"):
        os.makedirs("data/exp/mono")
    files = os.listdir("data/exp/mono")

    pattern = "^\d+$"
    regex = re.compile(pattern)
    integer_files = [f for f in files if regex.match(f)]

    if integer_files == []:
        exp_num = 0
    elif exp_num is not None:
        exp_num = exp_num
    elif create_new_exp == True:
        exp_num = np.amax(np.array([int(i) for i in integer_files])) + 1
    else:
        exp_num = np.amax(np.array([int(i) for i in integer_files]))

    _dir = get_directory(stage='exp',
                         mono_or_rand='mono',
                         dataset_name=ds_name,
                         mrate=mrate,
                         exp_num=exp_num)

    print("saved folder ", _dir)
    return _dir


def build_one_report(dataset_name, missing_rates, exp_num):

    rmse_df = pd.DataFrame(columns=["rmse", 'method', 'missing_rate', 'time'])
    acc_df = pd.DataFrame(columns=['acc', 'method', 'missing_rate'])

    for mrate in missing_rates:

        dir_path = get_save_path(dataset_name, mrate, exp_num)
        print(dir_path)
        files = os.listdir(dir_path)

        pattern = r"X_imp_(.*)\.json"

        for fname in files:
            if re.match(rmse_pattern, fname):
                imp = np.load(os.path.join(dir_path, fname))
                
                

    rmse_df.loc[rmse_df["method"] == "em", "time"] *= 3600

    acc_df['acc'] = acc_df['acc'] * 100

    return (rmse_df, acc_df)


def build_report(datasets, missing_rates, exps=None, mono_or_rand="mono"):
    if exps is None:
        files = os.listdir("data/exp/mono/")
        pattern = "^\d+$"
        regex = re.compile(pattern)
        int_files = [f for f in files if regex.match(f)]
    else:
        int_files = [f for f in exps]

    print("experiment files:", int_files)

    rmse_dfs = pd.DataFrame(columns=[
        "dataset", "rmse", 'method', 'missing_rate', 'time', 'exp_num'
    ])

    acc_dfs = pd.DataFrame(
        columns=["dataset", 'acc', 'method', 'missing_rate', 'exp_num'])

    for exp_num in int_files:
        print(exp_num)
        for ds in datasets:
            rmse_df, acc_df = build_one_report(ds, missing_rates, exp_num)
            rmse_df['dataset'] = ds
            acc_df['dataset'] = ds
            rmse_df['exp_num'] = exp_num
            acc_df['exp_num'] = exp_num

            acc_dfs = pd.concat([acc_dfs, acc_df], ignore_index=True)
            rmse_dfs = pd.concat([rmse_dfs, rmse_df], ignore_index=True)

    agg_acc = pd.DataFrame(
        acc_dfs.groupby(['dataset', 'method',
                         'missing_rate'])['acc'].agg([np.mean,
                                                      np.std])).reset_index()
    agg_rmse = pd.DataFrame(
        rmse_dfs.groupby(['dataset', 'method',
                          'missing_rate'])['rmse'].agg([np.mean,
                                                        np.std])).reset_index()

    agg_time = pd.DataFrame(
        rmse_dfs.groupby(['dataset', 'method',
                          'missing_rate'])['time'].agg([np.mean,
                                                        np.std])).reset_index()

    # return agg_table at the end of report
    # print(agg_acc)
    # print(agg_rmse)
    # print(agg_time)
    #
    pv_acc = convert_to_latex(agg_acc, "acc", mono_or_rand)
    pv_rmse = convert_to_latex(agg_rmse, "rmse", mono_or_rand)
    pv_time = convert_to_latex(agg_time, "time", mono_or_rand)
    print("acc-----------------------")
    print(pv_acc)
    print("rmse-----------------------")
    print(pv_rmse)
    print("time----------------------")
    print(pv_time)


def convert_to_latex(agg_table, metric_name, mono_or_rand):
    asc = True
    if metric_name == 'acc':
        asc = False
 
rmse--------   agg_table['rank'] = (agg_table.groupby(['dataset', 'missing_rate'
                                           ])['mean'].rank(ascending=asc,
                                                           method='dense'))
    if mono_or_rand == 'rand':
        agg_table['mean_std'] = agg_table.apply(
            lambda x: "$\\textbf{{ {:.1f} }} \\pm \\textbf{{ {:.1f} }}$".format(
                x['mean'], x['std'])
            if x['rank'] == 1 else "${:.1f} \\pm {:.1f}$".format(
                x['mean'], x['std']),
            axis=1)
    else:
        agg_table['mean_std'] = agg_table.apply(
            lambda x: "$\\textbf{{ {:.1f} }}$".format(x['mean'], x['std'])
            if x['rank'] == 1 else "{:.1f}".format(x['mean'], x['std']),
            axis=1)

    print(agg_table)
    pivot_table = pd.pivot_table(agg_table,
                                 index=['dataset', 'missing_rate'],
                                 columns='method',
                                 values='mean_std',
                                 aggfunc='first')

    _latex = pivot_table.to_latex(escape=False)

    return _latex


if __name__ == "__main__":
    build_report(["mnist", "fashion_mnist"], [.6, .5, .4], mono_or_rand='mono')
