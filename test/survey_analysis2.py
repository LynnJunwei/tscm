# -*- coding: utf-8 -*-
# @Time    : 2024/11/26
# @Author  : Eric
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tscm.overall_sensation.core import OverallSensationCalculator
from tscm.config import OverallSensationConfig
# calculate overall ts


def calculate_overall_ts_origin(local_ts_ser):
    if local_ts_ser.isnull().any():
        return np.nan
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation


def calculate_overall_ts_smoothed1(local_ts_ser):
    if local_ts_ser.isnull().any():
        return np.nan
    overall_sensation_config = OverallSensationConfig(smooth=True, model_type='origin', internal_smooth=False, smooth_adjusted=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def calculate_overall_ts_modified(local_ts_ser):
    if local_ts_ser.isnull().any():
        return np.nan
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='modified', internal_smooth=False, smooth_adjusted=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def calculate_overall_ts_smoothed2(local_ts_ser):
    if local_ts_ser.isnull().any():
        return np.nan
    overall_sensation_config = OverallSensationConfig(smooth=True, model_type='modified', internal_smooth=True, smooth_adjusted=True, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

if __name__ == '__main__':
    local_ts_file = "form_info_20240906_151955.csv"
    local_ts_df = pd.read_csv(local_ts_file, index_col=[0, 5], encoding='utf-8')
    local_ts_df = local_ts_df
    overall_ts_mea = local_ts_df['overall_tsv']

    exp_index = local_ts_df['exp_index']
    local_ts_df = local_ts_df.loc[:, [col.endswith('tsv') for col in local_ts_df.columns]]
    local_ts_df = local_ts_df.iloc[:, 1:]
    local_ts_df.columns = ['Head', 'Chest', 'Back', 'Pelvis', 'LUpperArm', 'LLowerArm', 'LHand', 'LThigh', 'LLeg','LFoot']

    overall_ts_origin_df = local_ts_df.apply(calculate_overall_ts_origin, axis=1)
    overall_ts_smoothed1_df = local_ts_df.apply(calculate_overall_ts_smoothed1, axis=1)
    overall_ts_modified_df = local_ts_df.apply(calculate_overall_ts_modified, axis=1)
    overall_ts_smoothed2_df = local_ts_df.apply(calculate_overall_ts_smoothed2, axis=1)

    overall_ts_df = pd.concat([exp_index, overall_ts_mea, overall_ts_origin_df, overall_ts_smoothed1_df,
                               overall_ts_modified_df, overall_ts_smoothed2_df],
                              axis=1, keys=['exp_index', 'measure', 'origin', 'smoothed1', 'modified', 'smoothed2'])
    overall_ts_df = pd.concat([overall_ts_df, local_ts_df], axis=1)
    overall_ts_df['local_max'] = local_ts_df.max(axis=1)
    print(overall_ts_df)
    # print(overall_ts_df)
    # selected_ts_df = overall_ts_df[overall_ts_df['exp_index'] == 5]
    for i in range(1, 8):
        selected_ts_df = overall_ts_df[overall_ts_df['exp_index'] == i]
        selected_ts_df.to_csv(f'exp_{i}.csv', encoding='utf-8-sig')