# -*- coding: utf-8 -*-
# @Time    : 2024/11/27
# @Author  : Eric
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# draw overall ts for 4 and 6

if __name__ == '__main__':

    # selected_ts_df = overall_ts_df[overall_ts_df['exp_index'] == 5]
    plt.figure(figsize=(7, 7), dpi=300)
    # plt.figure(figsize=(10, 5), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    # for i in [1, 2, 3, 4]:
    for i in [1, 2]:
        plt.subplot(3, 3, i)

        i = 4 if i == 1 else 6
        '''if i == 1:
            i = 2
        elif i == 2:
            i = 4
        elif i == 3:
            i = 6
        elif i == 4:
            i = 7'''
        selected_ts_df = pd.read_csv('exp_{}.csv'.format(i), index_col=[0, 1], encoding='utf-8-sig')


        '''sns.boxplot(selected_ts_df['measure'].unstack(), color='white', showmeans=True,
                    meanprops={'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '4'})
        sns.boxplot(selected_ts_df['modified'].unstack(), color='gray', showmeans=True,
                    meanprops={'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '4'})'''

        sns.pointplot(selected_ts_df['measure'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Surveyed Overall')
        sns.pointplot(selected_ts_df['origin'].unstack(),  errorbar=("pi", 50), capsize=.2, lw=1.2, label='Predicted Overall')
        # sns.pointplot(selected_ts_df['modified'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Predicted (modified)')
        sns.pointplot(selected_ts_df['Back'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Surveyed Back', color='gray')
        sns.pointplot(selected_ts_df['local_max'].unstack(), errorbar=("pi", 50), capsize=.2, lw=1.2, label='Surveyed Maximum', color='darkred')
        selected_ts_df.to_csv('exp_{}.csv'.format(i))
        plt.xlim(-0.5, 10.5)
        plt.ylim(-4, 4)
        plt.xticks(list(range(11)))
        plt.yticks(list(range(-4, 5)))
        plt.xlabel('Questionnaire Number')
        plt.ylabel('Thermal Sensation')
        plt.title('Case {}'.format(i), x=0.8, y=0)

        if i in [6]:
            plt.ylabel('')
        print(i)
        print(selected_ts_df['origin'].unstack().mean())

    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)

    plt.savefig('exp_result-2.png')
    plt.show()
