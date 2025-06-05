# -*- coding: utf-8 -*-
# @Time    : 2024/11/27
# @Author  : Eric
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # selected_ts_df = overall_ts_df[overall_ts_df['exp_index'] == 5]
    plt.figure(figsize=(7, 7), dpi=300)
    # plt.figure(figsize=(10, 5), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    for i in range(1, 8):
        selected_ts_df = pd.read_csv('exp_{}.csv'.format(i), index_col=[0, 1])

        plt.subplot(3, 3, i)

        '''sns.boxplot(selected_ts_df['measure'].unstack(), color='white', showmeans=True,
                    meanprops={'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '4'})
        sns.boxplot(selected_ts_df['modified'].unstack(), color='gray', showmeans=True,
                    meanprops={'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '4'})'''

        sns.pointplot(selected_ts_df['measure'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Surveyed')
        sns.pointplot(selected_ts_df['origin'].unstack(),  errorbar=("pi", 50), capsize=.2, lw=1.2, label='Predicted')
        # sns.pointplot(selected_ts_df['modified'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Predicted (modified)')
        '''if i == 4 or i == 6:
            sns.pointplot(selected_ts_df['Back'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Back', color='gray')
            sns.pointplot(selected_ts_df['local_max'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Maximum', color='darkred')
            pass'''
        selected_ts_df.to_csv('exp_{}.csv'.format(i))
        plt.xlim(-0.5, 10.5)
        plt.ylim(-4, 4)
        plt.xticks(list(range(11)))
        plt.yticks(list(range(-4, 5)))
        plt.xlabel('Questionnaire Number')
        plt.ylabel('Overall Thermal Sensation')
        plt.title('Case {}'.format(i), x=0.8, y=0)

        if i in [2, 3, 5, 6]:
            plt.ylabel('')

        print(i)
        print(selected_ts_df['measure'].unstack().mean()-selected_ts_df['origin'].unstack().mean())

    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)

    des = ('Case description'
           '\n1: Indoor (cool) / Indoor (gradually warmer)'
           '\n2: Indoor (cool) / Local cooling'
           '\n3: Indoor (cool) / Local heating'
           '\n4: Indoor (warm) / Local cooling'
           '\n5: Indoor (warm) / Local heating'
           '\n6: Outdoor (fully shaded) / Local cooling'
           '\n7: Outdoor (partly shaded) / Local cooling')

    plt.text(27.5, 3.5, des, va='top', ha='left', fontsize=9, linespacing=1.5)

    plt.savefig('exp_result-1.png')
    # plt.savefig('exp_result-3.png')
    plt.show()
