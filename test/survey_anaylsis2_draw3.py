# -*- coding: utf-8 -*-
# @Time    : 2025/2/18
# @Author  : Eric

# draw local and overall sensation

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

        sns.pointplot(selected_ts_df['measure'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Overall', zorder=20, color='black')
        sns.pointplot(selected_ts_df['Chest'].unstack(),  errorbar=None, capsize=.2, lw=1.2, label='Chest', alpha=0.5)
        sns.pointplot(selected_ts_df['Back'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Back', alpha=0.5)
        sns.pointplot(selected_ts_df['Pelvis'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Pelvis', alpha=0.5)
        sns.pointplot(selected_ts_df['LUpperArm'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Upper Arm', alpha=0.5)
        sns.pointplot(selected_ts_df['LLowerArm'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Lower Arm', alpha=0.5)
        sns.pointplot(selected_ts_df['LHand'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Hand', alpha=0.5)
        sns.pointplot(selected_ts_df['LThigh'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Thigh', alpha=0.5)
        sns.pointplot(selected_ts_df['LLeg'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Leg', alpha=0.5)
        sns.pointplot(selected_ts_df['LFoot'].unstack(), errorbar=None, capsize=.2, lw=1.2, label='Foot', alpha=0.5)

        selected_ts_df.to_csv('exp_{}.csv'.format(i))
        plt.xlim(-0.5, 10.5)
        plt.ylim(-4, 4)
        plt.xticks(list(range(11)))
        plt.yticks(list(range(-4, 5)))
        plt.xlabel('Questionnaire Number')
        plt.ylabel('Thermal Sensation')
        plt.title('Case {}'.format(i), x=0.8, y=0)

        if i in [2, 3, 5, 6]:
            plt.ylabel('')

    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False, fontsize=10)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)

    des = ('Case description'
           '\n1: Indoor (cool) / Indoor (warm)'
           '\n2: Indoor (cool) / Local cooling'
           '\n3: Indoor (cool) / Local heating'
           '\n4: Indoor (warm) / Local cooling'
           '\n5: Indoor (warm) / Local heating'
           '\n6: Outdoor (fully shaded) / Local cooling'
           '\n7: Outdoor (partly shaded) / Local cooling')

    plt.text(28.5, 3.5, des, va='top', ha='left', fontsize=9, linespacing=1.5)

    plt.savefig('exp_result-4.png')
    plt.show()