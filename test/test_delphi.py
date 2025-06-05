# -*- coding: utf-8 -*-
# @Time    : 2025/4/2
# @Author  : Eric
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tscm.config import HumanConfig, OverallSensationConfig
from tscm.core import TSCMObject

if __name__ == '__main__':
    file_name = r"C:\Users\user\OneDrive - The Hong Kong Polytechnic University\Journal Paper\CBE model smoothing\Delphi_data.csv"
    local_ts_df = pd.read_csv(file_name, index_col=0).dropna().iloc[:, 2:]
    local_ts_df.index  = local_ts_df.index.astype(int)
    overall_ts_mea = local_ts_df.iloc[:, 0]
    local_ts_df_mea = local_ts_df.iloc[:, 1:]

    overall_sensation_config_modified = OverallSensationConfig(external_smooth=False, original_model=False,
                                                               external_smooth_adjusted=True,
                                                               external_smooth_alpha=10,
                                                               external_smooth_simplified=True,
                                                               internal_smooth=False)
    whole_ts_modified_model = TSCMObject(local_sensation=local_ts_df_mea,
                                         overall_sensation_config=overall_sensation_config_modified,
                                         output='os')
    whole_ts_modified_model.run(5)
    whole_ts_modified = whole_ts_modified_model.overall_sensation
    print(whole_ts_modified)

    # ax = sns.pointplot(x=overall_ts_mea.values, y=whole_ts_modified.values, dodge=False)
    plt.figure(layout='constrained', figsize=(5, 4.5))
    sns.scatterplot(x=overall_ts_mea.values, y=whole_ts_modified.values)
    sns.lineplot(x=[-4, 4], y=[-4, 4], color='grey', linestyle='--', legend=None)
    plt.xlabel('Actual Overall Sensation')
    plt.ylabel('Predicted Overall Sensation')
    plt.ylim(-4.1, 4.1)
    plt.xlim(-4.1, 4.1)


    res = overall_ts_mea.sub(whole_ts_modified.values).pow(2).sum()
    tot = overall_ts_mea.sub(overall_ts_mea.mean()).pow(2).sum()
    r2 = 1 - res / tot
    print(f"R2: {r2:.4f}")
    mae = overall_ts_mea.sub(whole_ts_modified.values).abs().mean()
    print(f"MAE: {mae:.4f}")

    plt.text(-3.5, 3.5, 'R2: {:.4f}'.format(r2), fontsize=12)
    plt.text(-3.5, 3, 'MAE: {:.4f}'.format(mae), fontsize=12)

    plt.show()
