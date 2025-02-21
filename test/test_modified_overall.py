# -*- coding: utf-8 -*-
# @Time    : 2024/7/19
# @Author  : Eric
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tscm.overall_sensation import OverallSensationCalculator
from tscm.config import HumanConfig, OverallSensationConfig


def get_whole_ts(local_ts_df, overall_sensation_config):
    whole_ts_ser = pd.Series().reindex_like(local_ts_df)
    model_num_ser = pd.Series().reindex_like(local_ts_df)
    for i, local_ts in local_ts_df.iterrows():
        overall_ts_model = OverallSensationCalculator(local_sensation=local_ts,
                                                      overall_sensation_config=overall_sensation_config)
        whole_ts = overall_ts_model.overall_sensation
        print(i, whole_ts, overall_ts_model.bigger_group, overall_ts_model.model_num,
              overall_ts_model.overall_sensations_dict)
        whole_ts_ser.loc[i] = whole_ts
        model_num_ser.loc[i] = overall_ts_model.model_num
    return whole_ts_ser, model_num_ser


if __name__ == '__main__':
    local_ts_df = pd.read_csv('test_case/TSV_7-5-6_1.csv', index_col=0)

    human_config = HumanConfig(sex='male', met=1, clo=0.5)
    overall_sensation_config_origin = OverallSensationConfig(smooth=False, model_type='origin', internal_smooth=False)
    overall_sensation_config_modified = OverallSensationConfig(smooth=False, model_type='modified',
                                                               smooth_adjusted=True, smooth_alpha=10, internal_smooth=False)

    whole_ts_origin, model_num = get_whole_ts(local_ts_df, overall_sensation_config_origin)
    whole_ts_modified, _ = get_whole_ts(local_ts_df, overall_sensation_config_modified)
    whole_ts_df = pd.concat([whole_ts_origin, whole_ts_modified], axis=1, keys=['origin', 'modified'])

    plt.figure(layout='constrained')
    sns.lineplot(whole_ts_df, lw=3)
    sns.lineplot(local_ts_df)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    sns.lineplot(model_num, lw=3)
    # plt.show()


