#!/usr/bin/env python
# encoding=utf-8

"""
Logistic回归
===
根据现有数据对分类边界线建立回归公式, 以此进行分类.
"回归"来源于最佳拟合, 训练分类器即使用最优化算法寻找最佳拟合参数.
"""

from __future__ import print_function

import numpy
import matplotlib.pyplot as plt


def getDataset(filename='testSet.txt'):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split()
            dataset.append([
                1.0,
                float(datas[0]),
                float(datas[1]),
            ])
            labels.append(int(datas[2]))
        return numpy.array(dataset), labels


def sigmoid(inX):
    """ 海维赛德阶跃函数 """
    return 1.0 / (1 + numpy.exp(-inX))

"""
梯度上升算法
沿着函数f的梯度方向, 寻找f的最大值
因为梯度算子总是指向函数值增长最快的方向
"""


def getGradientAsecent(dataset, labels):
    """使用梯度上升算法计算最佳回归系数

    Parameters
    ----------
    dataset
    labels

    Returns
    -------
    list of floats : 回归系数
    """
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    m, n = dataset.shape
    alpha = 0.001  # 向目标移动的步长
    numCycles = 500  # 迭代次数
    weights = numpy.ones((n, 1))
    # 计算真实类别与预测类别的差值, 按照该差值的方向调整回归系数
    # FIXME: 这里使用了大量的矩阵运算, 导致计算效率低下
    for k in range(numCycles):
        h = sigmoid(dataset * weights)  # 矩阵相乘
        error = labels - h  # 向量相减
        weights += alpha * dataset.T * error  # 矩阵相乘
    return weights.T[0]


def getStochasticGradientAsecent_0(dataset, labels):
    """使用随机梯度上升算法计算最佳回归系数
    一次只用一个样本点来更新回归系数, 能对数据进行增量更新, 是一个"在线学习"算法
    但因为数据集可能不是线性可分, 在迭代的时候可能导致回归系数抖动, 收敛速度慢

    Parameters
    ----------
    dataset
    labels

    Returns
    -------
    list of floats : 回归系数
    """
    m, n = dataset.shape
    alpha = 0.01
    weights = numpy.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataset[i]*weights))
        error = labels[i] - h
        weights += alpha * error * dataset[i]
    return weights


def getStochasticGradientAsecent_1(
        dataset, labels, numIter=150):
    """使用改进的随机梯度上升算法计算最佳回归系数
    步长alpha每次都会调整
    通过随机选取样本来更新回归系数, 减少周期型抖动, 增加收敛速度

    Parameters
    ----------
    dataset
    labels
    numIter : int default 150
        迭代次数

    Returns
    -------
    list of floats : 回归系数
    """
    m, n = dataset.shape
    weights = numpy.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # 步长每次迭代都会减少 1/(j+i)
            # j 为迭代次数, i 为样本点的下标
            alpha = 4/(1.0+j+i) + 0.0001  # 常数使得 alpha 永远不会减少到 0
            # 通过随机选择来更新回归系数
            randIndex = int(numpy.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataset[randIndex]*weights))
            error = labels[randIndex] - h
            weights += alpha * error * dataset[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(dataset, labels, weights):
    """绘制数据分界线

    Parameters
    ----------
    weights : list of floats
        系数

    """
    m, _n = dataset.shape
    # 收集绘制的数据
    cord = {
        '1': {
            'x': [],
            'y': [],
        },
        '2': {
            'x': [],
            'y': [],
        },
    }
    for i in range(m):
        if labels[i] == 1:
            cord['1']['x'].append(dataset[i, 1])
            cord['1']['y'].append(dataset[i, 2])
        else:
            cord['2']['x'].append(dataset[i, 1])
            cord['2']['y'].append(dataset[i, 2])
    # 绘制图形
    figure = plt.figure()
    subplot = figure.add_subplot(111)
    # 绘制散点
    subplot.scatter(
        cord['1']['x'], cord['1']['y'],
        s=30, c='red', marker='s'
    )
    subplot.scatter(
        cord['2']['x'], cord['2']['y'],
        s=30, c='green'
    )
    # 绘制直线
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/ weights[2]
    subplot.plot(x, y)
    # 标签
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

""" 利用logistic回归来进行分类 -- 从疝气病症状预测病马的死亡率 """


def predict(inX, weights):
    probability = sigmoid(sum(numpy.array(inX)*weights))
    if probability > 0.5:
        return 1
    else:
        return 0


def loadDatasetFromFile(filename):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split('\t')
            row = list(map(lambda x: float(x), datas[:21]))
            dataset.append(row)
            labels.append(float(datas[21]))
    return numpy.array(dataset), numpy.array(labels)


def testColicPredict(num_iter=1000):
    """在马的疝气病数据上训练 logistic 回归模型

    Parameters
    ----------
    num_iter

    Returns
    -------

    """
    # 训练模型
    train = {}
    train['dataset'], train['labels'] = loadDatasetFromFile(
        'horseColicTraining.txt'
    )
    train['weights'] = getStochasticGradientAsecent_1(
        train['dataset'],
        train['labels'],
        numIter=num_iter
    )
    # 测试
    errorCount = 0
    test = {}
    test['dataset'], test['labels'] = loadDatasetFromFile(
        'horseColicTest.txt'
    )
    m, _n = test['dataset'].shape
    for rowno, row in enumerate(test['dataset']):
        if predict(row, train['weights']) != test['labels'][rowno]:
            errorCount += 1
    errorRate = 1.0*errorCount / m
    print("Error rate: {:.4f}".format(errorRate))
    return errorRate


def multiTestColicPredict(numTests=10):
    errorSum = 0.0
    # 多次运行结果可能不同, 因为使用随机选取的向量来更新回归系数
    for k in range(numTests):
        errorSum += testColicPredict()
    print('after %d iterations the average error rate is: %f'
          % (numTests, errorSum/float(numTests))
    )

if __name__ == '__main__':
    dataset, labels = getDataset()
    weights = {
        0: getGradientAsecent(dataset, labels),
        1: getStochasticGradientAsecent_0(dataset, labels),
        2: getStochasticGradientAsecent_1(dataset, labels),
    }
    # plotBestFit(dataset, labels, weights[0])
    # plotBestFit(dataset, labels, weights[1])
    # plotBestFit(dataset, labels, weights[2])

    multiTestColicPredict(10)
