# -*- coding: utf-8 -*-
# @Time    : 2025/4/3
# @Author  : Eric
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tscm.config import HumanConfig, OverallSensationConfig
from tscm.core import TSCMObject

if __name__ == '__main__':
    file_name = r"C:\Users\user\OneDrive - The Hong Kong Polytechnic University\Journal Paper\CBE model smoothing\Data2.csv"
    local_ts_df = pd.read_csv(file_name, index_col=0).dropna()
    overall_ts_mea = local_ts_df.iloc[:, 0]
    local_ts_df_mea = local_ts_df.iloc[:, 1:]

    overall_sensation_config_modified = OverallSensationConfig(external_smooth=True, original_model=True,
                                                               external_smooth_adjusted=True,
                                                               external_smooth_alpha=10,
                                                               external_smooth_simplified=True,
                                                               internal_smooth=True)
    whole_ts_modified_model = TSCMObject(local_sensation=local_ts_df_mea,
                                         overall_sensation_config=overall_sensation_config_modified,
                                         output='os')
    whole_ts_modified_model.run(1)
    whole_ts_modified = whole_ts_modified_model.overall_sensation
    print(whole_ts_modified)

