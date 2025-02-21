# -*- coding: utf-8 -*-
# @Time    : 2024/11/1
# @Author  : Eric
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tscm.overall_sensation.core import OverallSensationCalculator
from tscm.config import OverallSensationConfig


def calculate_overall_ts_origin(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation


def calculate_overall_ts_modified(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='modified', internal_smooth=False, smooth_adjusted=False)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.overall_sensation

def get_model_index(local_ts_ser):
    overall_sensation_config = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False, smooth_adjusted=False)
    overall_ts_model = OverallSensationCalculator(local_sensation=local_ts_ser, overall_sensation_config=overall_sensation_config)
    return overall_ts_model.model_num


if __name__ == '__main__':
    local_ts_file = r"C:\Users\user\OneDrive - The Hong Kong Polytechnic University\Journal Paper\CBE model smoothing\jump sample\model 3-6_1.xlsx"
    local_ts_df = pd.read_excel(local_ts_file, index_col=0)
    overall_ts_origin_df = local_ts_df.apply(calculate_overall_ts_origin, axis=1)
    overall_ts_modified_df = local_ts_df.apply(calculate_overall_ts_modified, axis=1)
    model_num_df = local_ts_df.apply(get_model_index, axis=1)

    overall_ts_df = pd.concat([overall_ts_origin_df, overall_ts_modified_df], axis=1, keys=['origin', 'modified'])
    overall_ts_df.to_csv('temp.csv')
    print(model_num_df)
    sns.lineplot(overall_ts_df, lw=3)
    plt.show()