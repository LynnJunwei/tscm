# -*- coding: utf-8 -*-
# @Time    : 2025/2/20
# @Author  : Eric
# calculate MAE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.proj3d import transform


def show_error():
    plt.figure(figsize=(7, 7), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    for i in range(1, 8):
        df = pd.read_csv(f"exp_{i}.csv", index_col=0)
        df_select = df[df['form_times'] > 2]

        e_1 = (df_select['origin'] - df_select['measure'])
        e_3 = (df_select['modified'] - df_select['measure'])

        e_1.index = df_select['form_times']
        e_3.index = df_select['form_times']

        plt.subplot(3, 3, i)
        # sns.boxplot(pd.concat([e_1, e_3], axis=1, keys=['origin', 'modified']), showmeans=True, color='white')
        sns.lineplot(data=pd.concat([e_1, e_3], axis=1, keys=['origin', 'modified']), palette=['black', 'gray'])
        plt.xlim(0.5, 11.5)
        plt.ylim(-2, 2)
        plt.xticks(list(range(1, 12)))
        plt.yticks(list(np.arange(-2, 2.1, 1)))
        plt.xlabel('Questionnaire Number')
        plt.ylabel('Error')
        plt.title('Case {}'.format(i), x=0.8, y=0)
        plt.legend('', frameon=False)

        if i in [2, 3, 5, 6]:
            plt.ylabel('')

    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)

    des = ('Case description'
           '\n1: Indoor (cool) / Indoor (warm)'
           '\n2: Indoor (cool) / Local cooling'
           '\n3: Indoor (cool) / Local heating'
           '\n4: Indoor (warm) / Local cooling'
           '\n5: Indoor (warm) / Local heating'
           '\n6: Outdoor (fully shaded) / Local cooling'
           '\n7: Outdoor (partly shaded) / Local cooling')

    plt.annotate(des, xy=(0.68, 0.14), xycoords='figure fraction', fontsize=9, linespacing=1.5)
    plt.show()


def show_abs_error():
    plt.figure(figsize=(7, 7), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    for i in range(1, 8):
        df = pd.read_csv(f"exp_{i}.csv", index_col=0)
        df_select = df[df['form_times'] > 2]

        ae_1 = np.abs(df_select['origin'] - df_select['measure'])
        ae_3 = np.abs(df_select['modified'] - df_select['measure'])

        ae_1.index = df_select['form_times']
        ae_3.index = df_select['form_times']

        plt.subplot(3, 3, i)
        # sns.boxplot(pd.concat([e_1, e_3], axis=1, keys=['origin', 'modified']), showmeans=True, color='white')
        sns.lineplot(data=pd.concat([ae_1, ae_3], axis=1, keys=['origin', 'modified']), palette=['black', 'gray'])
        plt.xlim(0.5, 11.5)
        plt.ylim(-0.5, 2.5)
        plt.xticks(list(range(1, 12)))
        plt.yticks(list(np.arange(0, 3, 1)))
        plt.xlabel('Questionnaire Number')
        plt.ylabel('Absolute Error')
        plt.title('Case {}'.format(i), x=0.8, y=0)
        plt.legend('', frameon=False)

        if i in [2, 3, 5, 6]:
            plt.ylabel('')

    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)

    des = ('Case description'
           '\n1: Indoor (cool) / Indoor (warm)'
           '\n2: Indoor (cool) / Local cooling'
           '\n3: Indoor (cool) / Local heating'
           '\n4: Indoor (warm) / Local cooling'
           '\n5: Indoor (warm) / Local heating'
           '\n6: Outdoor (fully shaded) / Local cooling'
           '\n7: Outdoor (partly shaded) / Local cooling')

    plt.annotate(des, xy=(0.68, 0.14), xycoords='figure fraction', fontsize=9, linespacing=1.5)
    plt.show()


def cal_error():
    for i in range(1, 8):
        df = pd.read_csv(f"exp_{i}.csv", index_col=0)
        df_select = df[df['form_times'] > 2]
        mae_1 = np.abs(df_select['measure'] - df_select['origin']).mean()
        mae_2 = np.abs(df_select['measure'] - df_select['smoothed1']).median()
        mae_3 = np.abs(df_select['measure'] - df_select['modified']).mean()
        mae_4 = np.abs(df_select['measure'] - df_select['smoothed2']).mean()
        print(f"exp{i} MAE origin: {mae_1}")
        # print(f"exp{i} MAE smoothed1: {mae_2}")
        print(f"exp{i} MAE modified: {mae_3}")
        # print(f"exp{i} MAE smoothed2: {mae_4}")

        mse_1 = np.square(df_select['measure'] - df_select['origin']).mean()
        mse_2 = np.square(df_select['measure'] - df_select['smoothed1']).mean()
        mse_3 = np.square(df_select['measure'] - df_select['modified']).mean()
        mse_4 = np.square(df_select['measure'] - df_select['smoothed2']).mean()
        print(f"exp{i} MSE origin: {mse_1}")
        # print(f"exp{i} MSE smoothed1: {mse_2}")
        print(f"exp{i} MSE modified: {mse_3}")
        # print(f"exp{i} MSE smoothed2: {mse_4}")

        me_1 = (df_select['measure'] - df_select['origin']).mean()
        me_2 = (df_select['measure'] - df_select['smoothed1']).mean()
        me_3 = (df_select['measure'] - df_select['modified']).mean()
        me_4 = (df_select['measure'] - df_select['smoothed2']).mean()
        print(f"exp{i} ME origin: {me_1}")
        # print(f"exp{i} ME smoothed1: {me_2}")
        print(f"exp{i} ME modified: {me_3}")
        # print(f"exp{i} ME smoothed2: {me_4}")

        print()


if __name__ == '__main__':
    # cal_error()
    show_error()
    show_abs_error()


