# -*- coding: utf-8 -*-
# @Time    : 2024/8/27
# @Author  : Eric
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tscm.overall_sensation.models import smoothed_low_level_warm, low_level_warm, modified_low_level_warm, smoothed_modified_low_level_warm


def get_smoothed_whole_ts(local_ts_df):
    whole_ts_ser = pd.Series().reindex_like(local_ts_df)
    for i, local_ts in local_ts_df.iterrows():
        whole_ts = smoothed_low_level_warm(local_ts)
        whole_ts_ser.loc[i] = whole_ts
    return whole_ts_ser


def get_whole_ts(local_ts_df):
    whole_ts_ser = pd.Series().reindex_like(local_ts_df)
    for i, local_ts in local_ts_df.iterrows():
        whole_ts = low_level_warm(local_ts)
        whole_ts_ser.loc[i] = whole_ts
    return whole_ts_ser


def get_modified_whole_ts(local_ts_df):
    whole_ts_ser = pd.Series().reindex_like(local_ts_df)
    for i, local_ts in local_ts_df.iterrows():
        whole_ts = modified_low_level_warm(local_ts)
        whole_ts_ser.loc[i] = whole_ts
    return whole_ts_ser

def get_smoothed_modified_whole_ts(local_ts_df):
    whole_ts_ser = pd.Series().reindex_like(local_ts_df)
    for i, local_ts in local_ts_df.iterrows():
        whole_ts = smoothed_modified_low_level_warm(local_ts)
        whole_ts_ser.loc[i] = whole_ts
    return whole_ts_ser


if __name__ == '__main__':
    tsv_file = r"C:\Users\user\OneDrive - The Hong Kong Polytechnic University\Journal Paper\CBE model smoothing\jump sample\within model 3.xlsx"
    local_ts_df = pd.read_excel(tsv_file, index_col=0)

    smoothed_whole_ts_df = get_smoothed_whole_ts(local_ts_df)
    # modified_whole_ts_df = get_modified_whole_ts(local_ts_df)
    # smoothed_modified_whole_ts_df = get_smoothed_modified_whole_ts(local_ts_df)
    original_whole_ts_df = get_whole_ts(local_ts_df)

    pd.concat([smoothed_whole_ts_df, original_whole_ts_df], axis=1, keys=['smoothed', 'original']).to_csv('temp.csv')

    # plt.figure(layout='constrained')
    # sns.lineplot(smoothed_whole_ts_df, lw=3)
    # sns.lineplot(modified_whole_ts_df, lw=3)
    # sns.lineplot(smoothed_modified_whole_ts_df, lw=3)
    # sns.lineplot(whole_ts_df, lw=3)
    # sns.lineplot(local_ts_df)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.show()