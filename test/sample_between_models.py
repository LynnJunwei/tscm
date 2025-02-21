# -*- coding: utf-8 -*-
# @Time    : 2024/10/30
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

def calculate_overall_ts_smoothed2(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=True, model_type='origin', internal_smooth=False, smooth_adjusted=True, smooth_alpha=15)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation


if __name__ == '__main__':
    local_ts_file = r"C:\Users\user\OneDrive - The Hong Kong Polytechnic University\Journal Paper\CBE model smoothing\jump sample\between models 1-5.xlsx"
    local_ts_df = pd.read_excel(local_ts_file, index_col=0)
    overall_ts_origin_df = local_ts_df.apply(calculate_overall_ts_origin, axis=1)
    overall_ts_smoothed1_df = local_ts_df.apply(calculate_overall_ts_smoothed1, axis=1)
    overall_ts_smoothed2_df = local_ts_df.apply(calculate_overall_ts_smoothed2, axis=1)

    overall_ts_df = pd.concat([overall_ts_origin_df, overall_ts_smoothed1_df, overall_ts_smoothed2_df], axis=1, keys=['origin', 'smoothed1', 'smoothed2'])
    overall_ts_df.to_csv('temp.csv')
