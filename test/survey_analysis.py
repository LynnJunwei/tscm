# -*- coding: utf-8 -*-
# @Time    : 2024/11/25
# @Author  : Eric
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba.cuda import local

from tscm.overall_sensation.core import OverallSensationCalculator
from tscm.config import OverallSensationConfig


def calculate_overall_ts_origin(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation


def calculate_overall_ts_smoothed1(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=True, model_type='origin', internal_smooth=False, smooth_adjusted=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def calculate_overall_ts_modified(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='modified', internal_smooth=False, smooth_adjusted=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def calculate_overall_ts_smoothed2(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=True, model_type='modified', internal_smooth=True, smooth_adjusted=True, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation


def cal_diff_second(df):
    tsv_df = df
    diff_seconds = []
    for i in range(1, len(tsv_df)-1):
        tsv_ser = tsv_df.iloc[i, :]
        tsv_ser_before = tsv_df.iloc[i-1, :]
        tsv_ser_after = tsv_df.iloc[i+1, :]

        delta_t2 = pd.Timedelta(tsv_df.index[i]-tsv_df.index[i-1]).seconds / 60
        delta_t1 = pd.Timedelta(tsv_df.index[i+1] - tsv_df.index[i]).seconds / 60

        diff_second = (
                2 / (delta_t1 + delta_t2) *
                (
                (tsv_ser_after - tsv_ser) / delta_t1 - (tsv_ser - tsv_ser_before) / delta_t2
                )
        )
        diff_seconds.append(diff_second)
    diff_second_sum = (pd.concat(diff_seconds, axis=1).T ** 2).sum()
    return diff_second_sum.to_frame().T



if __name__ == '__main__':
    local_ts_file = "form_info_20240906_151955.csv"
    local_ts_dfs = list(pd.read_csv(local_ts_file, chunksize=11, index_col=3, parse_dates=True, encoding='utf-8'))

    local_ts_df = pd.concat([df.iloc[2:, :] for df in local_ts_dfs])
    local_ts_df = local_ts_df.loc[:, [col.endswith('tsv') for col in local_ts_df.columns]]
    # local_ts_df = np.round(local_ts_df, 2)
    overall_ts_mea = local_ts_df.iloc[:, 0]
    local_ts_df = local_ts_df.iloc[:, 1:]
    local_ts_df.columns = ['Head', 'Chest', 'Back', 'Pelvis', 'LUpperArm', 'LLowerArm', 'LHand', 'LThigh', 'LLeg', 'LFoot']

    overall_ts_origin_df = local_ts_df.apply(calculate_overall_ts_origin, axis=1)
    overall_ts_smoothed1_df = local_ts_df.apply(calculate_overall_ts_smoothed1, axis=1)
    overall_ts_modified_df = local_ts_df.apply(calculate_overall_ts_modified, axis=1)
    overall_ts_smoothed2_df = local_ts_df.apply(calculate_overall_ts_smoothed2, axis=1)

    overall_ts_df = pd.concat([overall_ts_mea, overall_ts_origin_df, overall_ts_smoothed1_df, overall_ts_modified_df, overall_ts_smoothed2_df],
                              axis=1, keys=['measure', 'origin', 'smoothed1', 'modified', 'smoothed2'])
    # overall_ts_df.to_csv('temp.csv')
    ts_df = pd.concat([local_ts_df, overall_ts_df], axis=1)
    # ts_df.to_csv('ts_df.csv', encoding='utf_8_sig')

    overall_ts_dfs = np.array_split(overall_ts_df, 72)
    diff_second_sum = pd.concat([cal_diff_second(df) for df in overall_ts_dfs])
    diff_second_sum.to_csv('diff_second_sum.csv', encoding='utf_8_sig')
