# -*- coding: utf-8 -*-
# @Time    : 2024/8/20
# @Author  : Eric
import time
import numpy as np

def a():
    def a_b(num):
        return num + 1

    return {1: a_b}


def b(num):
    a_b = a()[1]
    print(a_b(num))

def c():
    time.sleep(2)
    print('c')


if __name__ == '__main__':
    b(5)
    b(7)
    d = c
    print(np.prod([]))