#!/usr/bin/env python
# encoding=utf-8

"""
决策树
===
决策树的数据形式非常容易理解, 决策树很多任务都是为了数据中所蕴含的知识信息,
因此决策树可以使用不熟悉的数据集合, 并从中提取出一系列规则, 机器学习算法最终使用机器从数据集中创造的规则.
专家系统中经常使用决策树.
"""

from __future__ import print_function

import math
import operator
import pickle
from collections import defaultdict

import numpy as np

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

class Dataset(object):
    """ 对数据集以及相关操作的封装 """

    def __init__(self, rawDataset):
        self.rawDataset = np.array(rawDataset)

    @property
    def shape(self):
        return self.rawDataset.shape

    @property
    def classList(self):
        return self.rawDataset[:, -1].tolist()

    @property
    def shannonEntropy(self):
        """获取数据集的香农熵
        熵越大代表混乱程度越高, 即混合的数据越多
        
        Returns
        -------
        float : 数据集的香农熵
        """
        # 统计每个 label 出现的次数
        labelCounts = defaultdict(int)
        for featVec in self.rawDataset:
            label = featVec[-1]
            labelCounts[label] += 1
        # 计算熵
        # H = - ∑(n, i=1) ( p(xi) * log(2)(p(xi)) )
        entropy = 0.0
        numEntries = len(self.rawDataset)
        for label in labelCounts:
            probability = 1.0*labelCounts[label] / numEntries
            entropy -= probability * math.log(probability, 2)  # 底数为2
        return entropy

    def split(self, axis):
        """ 对数据集按照 axis 指定的特征进行划分

        Parameters
        ----------
        axis : int
            指定用于进行划分的特征

        Returns
        -------
        按照指定特征划分后的 (值, 子数据集) 对
        """
        subDatasets = defaultdict(list)
        for featureVector in self.rawDataset:
            value = featureVector[axis]
            subFeatureVector = (
                featureVector[:axis].tolist()
                + featureVector[axis+1:].tolist()
            )  # 去除已经用于划分的特征
            subDatasets[value].append(subFeatureVector)
        return (
            list(subDatasets.keys()),
            list(map(self.__class__, subDatasets.values())),
        )

    def ChooseBestSplitFeature(self):
        """通过遍历数据集, 计算香农熵, 选择最好的特征进行数据划分

        Returns
        -------
        int : 对数据熵增益最大的划分特征的index
        """
        m, n = self.shape
        numFeatures = n - 1
        baseEntropy = self.shannonEntropy  # 当前整个数据集的熵
        best = {
            'gain': 0.0,  # 记录最好的信息增益
            'feature': -1,  # 记录最好的特征index
        }
        # 按照不同的特征遍历进行划分
        for featureIndex in range(numFeatures):
            _labels, subDatasets = self.split(featureIndex)
            # 计算按照此特征进行划分后的熵
            newEntropy = 0.0
            for subDataset in subDatasets:
                sub_m, _sub_n = subDataset.shape
                probability = 1.0*sub_m / m
                newEntropy += probability * subDataset.shannonEntropy
            # 计算信息增益, 更新最好的特征
            infoGain = baseEntropy - newEntropy
            if infoGain > best['gain']:
                best['gain'] = infoGain
                best['feature'] = featureIndex
        return best['feature']


def GetFakeDataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return Dataset(dataset), labels


class DicisionTree(object):
    """ 决策树 """

    def __init__(self, dataset, labels):
        if dataset and labels:
            self.labels = labels
            self.tree = self.BuildTree(Dataset(dataset), self.labels)

    def SaveToFile(self, filename):
        with open(filename, 'w') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def LoadFromFile(filename):
        with open(filename, 'r') as infile:
            tree = pickle.load(infile)
        return tree

    @staticmethod
    def GetMajorityClass(classList):
        classCount = defaultdict(int)
        for vote in classList:
            classCount[vote] += 1
        sortedClassCount = sorted(
            classCount.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
        return sortedClassCount[0][0]

    def BuildTree(self, dataset, labels):
        labels = labels[:]  # 复制防止破坏原来的 labels 列表

        classList = dataset.classList
        # 当子集中所有项都为同一 label , 直接返回
        if classList.count(classList[0]) == len(classList): 
            return classList[0]

        # 当所有 feature 都用完, 返回出现次数最多的
        _m, n = dataset.shape
        if n == 1:
            return self.GetMajorityClass(classList)

        # 选择信息增益最大的进行划分
        bestFeatureIndex = dataset.ChooseBestSplitFeature()
        bestFeatureLabel = labels[bestFeatureIndex]
        del(labels[bestFeatureIndex])
        logging.info('Spliting by Feature {0}({1})'.format(
            bestFeatureLabel,
            bestFeatureIndex
        ))

        dicisionTree = {
            bestFeatureLabel: {},
        }

        # 对特征下每个值进行递归划分
        subLabels, subDatasets = dataset.split(bestFeatureIndex)
        logging.info('labels:{0} for Feature {1}'.format(subLabels, bestFeatureLabel))
        for subLabel, subDataset in zip(subLabels, subDatasets):
            logging.info('Building subtree of value `{0}`'.format(subLabel))
            dicisionTree[bestFeatureLabel][subLabel] = self.BuildTree(
                subDataset,
                labels
            )
            logging.info('Subtree `{0}` built'.format(subLabel))
        return dicisionTree

    def predict(self, inputVector):
        return self.GetClassOfVector(self.tree, self.labels, inputVector)

    def GetClassOfVector(self, dicisionTree, featureLabels, inputVector):
        featureLabel = dicisionTree.keys()[0]
        subDicisionTree = dicisionTree[featureLabel]
        featureIndex = featureLabels.index(featureLabel)

        downKey = inputVector[featureIndex]
        downNode = subDicisionTree[downKey]

        if isinstance(downNode, dict):
            # 递归在子树中查找所属类别
            classLabel = self.GetClassOfVector(
                downNode, featureLabels,
                inputVector
            )
        else:
            classLabel = downNode
        return classLabel

    @property
    def depth(self):
        return self.GetTreeDepth(self.tree)

    @classmethod
    def GetTreeDepth(cls, tree):
        max_depth = 0
        featureLabel = tree.keys()[0]
        subDicisionTree = tree[featureLabel]
        for featureValue in subDicisionTree:
            if isinstance(subDicisionTree[featureValue], dict):
                depth = 1 + cls.GetTreeDepth(subDicisionTree[featureValue])
            else:
                depth = 1

            max_depth = max(depth, max_depth)
        return max_depth

    @property
    def num_leaves(self):
        return self.GetNumLeaves(self.tree)

    @classmethod
    def GetNumLeaves(cls, tree):
        num = 0
        featureLabel = tree.keys()[0]
        subDicisionTree = tree[featureLabel]
        for featureValue in subDicisionTree:
            if isinstance(subDicisionTree[featureValue], dict):
                num += cls.GetNumLeaves(subDicisionTree[featureValue])
            else:
                num += 1
        return num

    @property
    def feature_label(self):
        return self.tree.keys()[0]

    def GetSubTree(self, feature_value):
        tree = self.__class__(None, None)
        tree.tree = self.tree[self.feature_label][feature_value]
        return tree

    @classmethod
    def GetRetrieveTree(cls, index):
        trees = (
            {'no surfacing': {
                0: 'no',
                1: {'flippers':
                    {0: 'no', 1: 'yes'}}
            }},
            {'no surfacing': {
                0: 'no',
                1: {'flippers':
                    {0: {'head':
                             {0: 'no', 1: 'yes'}},
                     1:'no'}
            }}},
        )
        tree = cls(None, None)
        tree.tree = trees[index]
        return tree


def LoadLensesData(filename):
    with open(filename) as infile:
        lensesDataset = []
        for line in infile:
            trainVector = line.strip().split('\t')
            lensesDataset.append(trainVector)
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', ]
    lenseTree = DicisionTree(lensesDataset, lensesLabels)
    return lenseTree


""" 绘制树形图 """
import matplotlib.pyplot as plt


class DicisionTreePlotter(object):

    DECISION_NODE = {
        'boxstyle': 'sawtooth',
        'fc': '0.8',
    }
    LEAF_NODE = {
        'boxstyle': 'round4',
        'fc': '0.8',
    }
    ARROW_ARGS = {
        'arrowstyle': '<-',
    }

    def __init__(self, tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        self.ax1 = plt.subplot(111, frameon=False, xticks=[], yticks=[])
        self.width = 1.0*tree.num_leaves
        self.depth = 1.0*tree.depth
        self.offset = {
            'x': -0.5/self.width,
            'y': 1.0
        }
        self.plot_tree(tree, (0.5, 1.0), '')
        plt.show()

    def plot_mid_text(self, text, centerPoint, parentPoint):
        xMid = (parentPoint[0] - centerPoint[0]) / 2.0 + centerPoint[0]
        yMid = (parentPoint[1] - centerPoint[1]) / 2.0 + centerPoint[1]
        self.ax1.text(xMid, yMid, text)

    def plot_node(self, text, centerPoint, parentPoint, node_type):
        self.ax1.annotate(
            text,
            xy=parentPoint, xycoords='axes fraction',
            xytext=centerPoint, textcoords='axes fraction',
            va='center', ha='center',
            bbox=node_type, arrowprops=DicisionTreePlotter.ARROW_ARGS
        )

    def plot_tree(self, tree, parentPoint, text):
        num_leaves = tree.num_leaves
        featureLabel = tree.feature_label
        centerPoint = (
            self.offset['x'] + (1.0 + num_leaves) / 2.0 / self.width,
            self.offset['y']
        )
        self.plot_mid_text(text, centerPoint, parentPoint)
        self.plot_node(
            featureLabel,
            centerPoint, parentPoint,
            DicisionTreePlotter.DECISION_NODE
        )
        subDicisionTree = tree.tree[featureLabel]
        self.offset['y'] -= 1.0/self.depth
        for featureValue in subDicisionTree:
            if isinstance(subDicisionTree[featureValue], dict):
                self.plot_tree(
                    tree.GetSubTree(featureValue),
                    centerPoint,
                    str(featureValue)
                )
            else:
                self.offset['x'] += 1.0 / self.width
                self.plot_node(
                    subDicisionTree[featureValue],
                    (self.offset['x'], self.offset['y']),
                    centerPoint,
                    DicisionTreePlotter.LEAF_NODE
                )
                self.plot_mid_text(
                    str(featureValue),
                    (self.offset['x'], self.offset['y']),
                    centerPoint
                )
        self.offset['y'] += 1.0 / self.depth


if __name__ == '__main__':
    tree = LoadLensesData('lenses.txt')
    print(tree.depth)
    t = DicisionTree.GetRetrieveTree(0)
    print(t.depth, t.num_leaves)
    plotter = DicisionTreePlotter(t)
