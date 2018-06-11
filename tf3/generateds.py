import numpy as np
import matplotlib.pyplot as plt

seed = 2


def generateds():
    # 基于seed生成随机数
    rdm = np.random.RandomState(seed)
    # 用随机数拼成300行2列的矩阵数据
    X = rdm.randn(300, 2)
    # 从矩阵X中取出一行，如果两个数的平方和小于2，给Y赋值1，否则赋值0。把它的结果当做输入数据集的标签
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 遍历Y_数据集，1为red，0为blue
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对输入数据集X和标签Y_进行形状整理。第一个元素表示，其为-1表示其跟随第二列计算，第二个元素表示有多少列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c

