# -*- coding: utf-8 -*-
# @Time    : 2024/11/26
# @Author  : Eric
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def get_model_number(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.model_num


if __name__ == '__main__':
    local_ts_file = "form_info_20240906_151955.csv"
    local_ts_df = pd.read_csv(local_ts_file, index_col=[0, 5], encoding='utf-8')
    local_ts_df = local_ts_df.loc[:, list(range(3, 12)), :]
    overall_ts_mea = local_ts_df['overall_tsv']

    exp_index = local_ts_df['exp_index']
    local_ts_df = local_ts_df.loc[:, [col.endswith('tsv') for col in local_ts_df.columns]]
    local_ts_df = local_ts_df.iloc[:, 1:]
    local_ts_df.columns = ['Head', 'Chest', 'Back', 'Pelvis', 'LUpperArm', 'LLowerArm', 'LHand', 'LThigh', 'LLeg','LFoot']

    model_num_df = local_ts_df.apply(get_model_number, axis=1)
    overall_ts_origin_df = local_ts_df.apply(calculate_overall_ts_origin, axis=1)
    overall_ts_smoothed1_df = local_ts_df.apply(calculate_overall_ts_smoothed1, axis=1)
    overall_ts_modified_df = local_ts_df.apply(calculate_overall_ts_modified, axis=1)
    overall_ts_smoothed2_df = local_ts_df.apply(calculate_overall_ts_smoothed2, axis=1)

    overall_ts_df = pd.concat([exp_index, model_num_df, overall_ts_mea, overall_ts_origin_df, overall_ts_smoothed1_df,
                               overall_ts_modified_df, overall_ts_smoothed2_df],
                              axis=1, keys=['exp_index', 'model_num', 'measure', 'origin', 'smoothed1', 'modified', 'smoothed2'])
    # print(overall_ts_df)
    selected_ts_df = overall_ts_df.melt(id_vars=['model_num'], value_vars=['measure', 'origin', 'modified'])
    print(selected_ts_df)

    sns.boxplot(x=selected_ts_df['model_num'], y=selected_ts_df['value'], hue=selected_ts_df['variable'])
    '''sns.boxplot(selected_ts_df[['measure','origin', 'modified']], color='white', showmeans=True,
                meanprops={'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '4'})'''
    # plt.xlim(-1, 9)
    plt.ylim(-4, 4)
    plt.show()