#!/usr/bin/env python
# encoding=utf-8

"""
k-近邻算法
===
存在训练样本集, 且样本集中每个数据都存在标签, 即已知样本集中每一组数据与所属分类的对应关系.

当输入没有标签的新数据后, 将新数据的每个特征与样本集中数据对应的特征进行比较,
算法提取样本集中特征最相似的k组数据(最近邻)的分类标签, (一般k<20),
取k个最相似的数据中出现次数最多的分类, 作为新数据的分类.
"""

from __future__ import print_function

import os
import operator

import numpy
import matplotlib
import matplotlib.pyplot as plt

""" 获取数据源的函数 """


def GetFakeDataset():
    """创建数据集及其分类
    
    Returns
    -------
    numpy.array, labels : 数据集, 数据集元素对应的标签
    """
    groups = numpy.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.1],
    ])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels


def GetFileDataset(filename):
    """把文本中的数据转换为数据集, labels返回

    Parameters
    ----------
    filename : string
        文本名
    
    Returns
    -------
    numpy.array, list : 文本中的数据矩阵, 数据对应的标签列表
    """
    with open(filename) as infile:
        lines = infile.readlines()
        numberOflines = len(lines)
        dataset = numpy.zeros((numberOflines, 3))
        dataLabels = []
        for index, line in enumerate(lines):
            listFromLine = line.strip().split()
            dataset[index,:] = listFromLine[0:3]
            dataLabels.append(int(listFromLine[-1]))
        return dataset, dataLabels

""" kNN 分类器 """


class KNNModel(object):
    """ kNN分类器 """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def predict(self, inX, k):
        if k <= 0:
            raise ValueError('K > 0')

        m, n = self.dataset.shape
        # 利用矩阵运算, 每个dataset的分量都减去inX
        diffMat = numpy.tile(inX, (m, 1)) - self.dataset
        # 计算欧式距离 sqrt(sum())
        distances = ((diffMat**2).sum(axis=1))**0.5
        # 对数据从小到大次序排列，确定前k个距离最小元素所在的主要分类
        sortedDistInd = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = self.labels[sortedDistInd[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 返回最相近的类
        sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True
        )
        return sortedClassCount[0][0]


class KNNModelWithNormalize(object):
    """ 带归一化的kNN分类器 """

    def __init__(self, dataset, labels):
        self.normDataset, self.ranges, self.minVals = self.normalize(dataset)
        self.labels = labels

    def normalize(self, dataset):
        """ 对dataset进行归一化处理, 使得输入的特征权重一致 """
        minVals = dataset.min(0)  # 获取每一列的最小值
        maxVals = dataset.max(0)  # 获取每一列的最大值
        ranges = maxVals - minVals  # 每一列的范围
        m, n = dataset.shape
        # 归一化 (Xi - Xmin) / (Xmax - Xmin)
        normDataset = (dataset - numpy.tile(minVals, (m, 1))) / numpy.tile(ranges, (m, 1))
        return normDataset, ranges, minVals

    def predict(self, inX, k):
        if k <= 0:
            raise ValueError('K > 0')
        
        # 先对输入特征进行归一化处理
        inX = (inX - self.minVals) / self.ranges

        datasetSize = self.normDataset.shape[0]
        # 利用矩阵运算, 每个 dataset 的分量都减去inX
        diffMat = numpy.tile(inX, (self.normDataset.shape[0],1)) - self.normDataset
        # 计算欧式距离 sqrt(sum())
        distances = ((diffMat**2).sum(axis=1))**0.5
        # 对数据从小到大次序排列，确定前k个距离最小元素所在的主要分类
        sortedDistInd = distances.argsort()
        classCount={}
        for i in range(k):
            voteIlabel = self.labels[sortedDistInd[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 返回最相近的类
        sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True
        )
        return sortedClassCount[0][0]

    @classmethod
    def test(cls, testfile, k=3, ratio=0.10):
        dataset, labels = GetFileDataset(testfile)
        m, n = dataset.shape
        numTestVectors = int(m * ratio)
        numError = 0

        model = cls(dataset[numTestVectors:m, :], labels[numTestVectors:m])
        for i in range(numTestVectors):
            result = model.predict(dataset[i, :], k)
            if result != labels[i]:
                numError += 1
                print('× Predict/Real {0}/{1}'.format(result, labels[i]))
            else:
                print('√ Predict/Real {0}/{1}'.format(result, labels[i]))
        print('Total error rate: {0:.1%}'.format(1.0*numError / numTestVectors))


def TestClassifyPerson(dataset_filename):
    result2str = {
        1: '完全不感兴趣',
        2: '可能喜欢',
        3: '很有可能喜欢',
    }
    print('请输入该人的相关信息:')
    percentageTimeOfPlayGames = float(
        input('消耗在玩游戏上的时间百分比?\n： ')
    )
    flyMiles = float(
        input('每年搭乘飞机的飞行里程数?\n： ')
    )
    iceCream = float(
        input('每周消费的冰淇淋公升数?\n： ')
    )

    dataset, labels = GetFileDataset(dataset_filename)
    DrawPlot(dataset, labels)
    model = KNNModelWithNormalize(dataset, labels)
    inVector = numpy.array([flyMiles, percentageTimeOfPlayGames, iceCream])
    classifierResult = model.predict(inVector, k=3)
    print(
        '预测你对这个人:', result2str[classifierResult]
    )


""" 使用 Matplotlib绘制散点图 """


def DrawPlot(dataset, labels):
    """绘制散点图

    Parameters
    ----------
    dataset : numpy.array
        数据集
    labels : list of int
        标签值
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _ = ax.scatter(
        dataset[:, 1], dataset[:, 2],
        s=15.0*numpy.array(labels),   # 大小
        c=15.0*numpy.array(labels)    # 颜色
    )
    plt.show()

""" 手写识别系统 """


def VectorDebugPrint(vector):
    for i in range(32):
        print(''.join(
            list(map(
                lambda x: str(int(x)),
                vector[i*32:(i+1)*32]
            ))
        ))


def TranslateImg2Vector(filename):
    """ 把'图像文件'转换为1024维的向量 """
    vector = numpy.zeros((1, 1024))
    with open(filename, 'r') as infile:
        for lineno, line in enumerate(infile):
            for rowno in range(32):
                vector[0, 32*lineno+rowno] = int(line[rowno])
    return vector


def GetDigitsDatasetFromDir(dirname):
    """从文件夹中获取数据集, labels

    Parameters
    ----------
    dirname 文件夹名称

    Returns
    -------
    numpy.array, labels : 数据集, 数据集元素对应的标签
    """
    filenames = os.listdir(dirname)

    labels = [None] * len(filenames)
    dataset = numpy.zeros((len(filenames), 1024))

    for i, filename in enumerate(filenames):
        fileclass = filename.split('.')[0].split('_')[0]
        filepath = os.path.join(dirname, filename)
        dataset[i, :], labels[i] = TranslateImg2Vector(filepath), fileclass
    return dataset, labels


def TestHandwritingNumber(trainDir, testDir, k=3):
    dataset, labels = GetDigitsDatasetFromDir(trainDir)
    model = KNNModel(dataset, labels)

    dataset, labels = GetDigitsDatasetFromDir(testDir)
    numError = 0
    numTestVectors = len(labels)
    for testVec, label in zip(dataset, labels):
        result = model.predict(testVec, k)
        if result != label:
            numError += 1
            print('× Predict/Real {0}/{1}'.format(result, label))
        else:
            print('√ Predict/Real {0}/{1}'.format(result, label))
    print('Total error rate: {0:.1%}'.format(1.0*numError / numTestVectors))


if __name__ == '__main__':
    dataset, labels = GetFakeDataset()
    model = KNNModel(dataset, labels)
    inX = [0, 0]
    print('{} should be {}'.format(inX, model.predict(inX, k=3)))

    # TestClassifyPerson('datingTestSet2.txt')

    TestHandwritingNumber('trainingDigits', 'testDigits', k=3)
