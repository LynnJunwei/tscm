# -*- coding: utf-8 -*-
# @Time    : 2024/7/19
# @Author  : Eric
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tscm.overall_sensation import OverallSensationCalculator
from tscm.config import HumanConfig, OverallSensationConfig
from tscm.core import TSCMObject


if __name__ == '__main__':
    local_ts_df = pd.read_csv('test_case/TSV_3.csv', index_col=0)

    human_config = HumanConfig(sex='male', met=1, clo=0.5)
    overall_sensation_config_origin = OverallSensationConfig(external_smooth=True, original_model=True,
                                                             external_smooth_adjusted=True,
                                                             internal_smooth=False)
    overall_sensation_config_modified = OverallSensationConfig(external_smooth=True, original_model=False,
                                                               external_smooth_adjusted=True,
                                                               external_smooth_alpha=10,
                                                               external_smooth_simplified=True,
                                                               internal_smooth=True)

    whole_ts_origin_model = TSCMObject(local_sensation=local_ts_df,
                                       overall_sensation_config=overall_sensation_config_origin,
                                       output='os')
    whole_ts_origin_model.run(5)
    whole_ts_origin = whole_ts_origin_model.overall_sensation
    model_num = whole_ts_origin_model.model_num

    whole_ts_modified_model = TSCMObject(local_sensation=local_ts_df,
                                         overall_sensation_config=overall_sensation_config_modified,
                                         output='os')
    whole_ts_modified_model.run(5)
    whole_ts_modified = whole_ts_modified_model.overall_sensation

    # whole_ts_df = pd.concat([whole_ts_origin, whole_ts_modified], axis=1, keys=['origin', 'modified'])
    print(whole_ts_origin)
    plt.figure(layout='constrained')
    # sns.lineplot(whole_ts_df, lw=3)
    sns.lineplot(whole_ts_origin, lw=4, label='overall', color='red')
    sns.lineplot(local_ts_df)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    sns.lineplot(model_num, lw=3)
    # plt.show()


