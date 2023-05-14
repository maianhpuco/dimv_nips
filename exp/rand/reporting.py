import numpy as np
import json
import pandas as pd
import re
import os
import sys

sys.path.append("")

from src.utils import get_directory, get_path_int_file
import re


def build_one_report(dataset_name, missing_rates, exp_num):

    rmse_df = pd.DataFrame(columns=["rmse", 'method', 'missing_rate', 'time'])
    acc_df = pd.DataFrame(columns=['acc', 'method', 'missing_rate'])

    for mrate in missing_rates:

        dir_path = get_path_int_file(dataset_name, mrate, exp_num, "exp",
                                     "rand")
        print(dir_path)
        files = os.listdir(dir_path)

        rmse_pattern = r"rmse_(.*)\.json"
        acc_pattern = r"acc_(.*)\.json"

        for fname in files:
            if re.match(rmse_pattern, fname):
                frmse = open(os.path.join(dir_path, fname))
                rmse_json = json.load(frmse)

                rmse_json.update({
                    'method': re.match(rmse_pattern, fname).group(1),
                    'missing_rate': mrate
                })

                rmse_df = pd.concat(
                    [rmse_df, pd.DataFrame([rmse_json])], ignore_index=True)
            if re.match(acc_pattern, fname):
                facc = open(os.path.join(dir_path, fname))
                acc_json = json.load(facc)
                acc_json.update({
                    'method': re.match(acc_pattern, fname).group(1),
                    'missing_rate': mrate
                })

                acc_df = pd.concat([acc_df, pd.DataFrame([acc_json])],
                                   ignore_index=True)

    rmse_df.loc[rmse_df["method"] == "em", "time"] *= 3600

    acc_df['acc'] = acc_df['acc'] * 100

    return (rmse_df, acc_df)


def build_report(datasets, missing_rates, exps=None, missing_type="rand"):
    if exps is None:
        files = os.listdir("data/exp/{}/".format(missing_type))
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
    print("---- RAW ACC-----------------")
    print(agg_acc)
    convert_to_md(agg_acc, "acc", missing_type)

    print("---- raw rmse-----------------")
    print(agg_rmse)
    convert_to_md(agg_rmse, "rmse", missing_type)

    print("---- raw time-----------------")
    print(agg_time)
    convert_to_md(agg_time, "time", missing_type)

    pv_acc = convert_to_latex(agg_acc, "acc", missing_type)
    pv_rmse = convert_to_latex(agg_rmse, "rmse", missing_type)
    pv_time = convert_to_latex(agg_time, "time", missing_type)
    print("acc-----------------------")

    print(pv_acc)
    print("rmse-----------------------")
    print(pv_rmse)
    print("time----------------------")
    print(pv_time)


def convert_to_md(agg_table, metric_name, missing_type):
    agg_table.sort_values(by=['dataset', 'missing_rate', 'mean'], inplace=True)
    with open("report/{}/{}.md".format(missing_type, metric_name), 'w') as f:
        f.write(agg_table.to_markdown())


def convert_to_latex(agg_table, metric_name, missing_type):
    asc = True
    if metric_name == 'acc':
        asc = False

    agg_table['rank'] = (agg_table.groupby(['dataset', 'missing_rate'
                                           ])['mean'].rank(ascending=asc,
                                                           method='dense'))
    if missing_type == 'rand':
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

    small_datas = ['iris', 'new_thyroid', 'yeast', 'ionosphere', 'seeds']
    build_report(small_datas, [i / 10 for i in range(1, 10)],
                 missing_type='rand')
