# -*- coding: utf-8 -*-
# @Time    : 2025/4/14
# @Author  : Eric
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tscm.overall_sensation.core import OverallSensationCalculator
from tscm.config import OverallSensationConfig


def calculate_overall_ts_origin(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(external_smooth=False, original_model=True, internal_smooth=False)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def calculate_overall_ts_modified(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(external_smooth=False, original_model=False, internal_smooth=False)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def get_model_number(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(external_smooth=False, original_model=True, internal_smooth=False)
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
    overall_ts_modified_df = local_ts_df.apply(calculate_overall_ts_modified, axis=1)

    overall_ts_df = pd.concat([exp_index, model_num_df, overall_ts_mea, overall_ts_origin_df, overall_ts_modified_df],
                              axis=1, keys=['exp_index', 'model_num', 'measure', 'origin', 'modified'])
    overall_ts_df = pd.concat([overall_ts_df, local_ts_df], axis=1)
    print(overall_ts_df)
    model_5_df = overall_ts_df[overall_ts_df['model_num'] == 5]
    print(model_5_df)
    # model_5_df.to_csv("model_5.csv", index=False)
    print(np.corrcoef(model_5_df['measure'], model_5_df['origin'])[0, 1]**2)
    print(np.corrcoef(model_5_df['measure'], model_5_df['modified'])[0, 1]**2)