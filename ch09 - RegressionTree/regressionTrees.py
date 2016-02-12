#!/usr/bin/env python
# encoding=utf-8

from __future__ import print_function

import copy
import logging

import numpy

logging.basicConfig(
        level=logging.DEBUG,
        # level=logging.INFO,
        format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1


def load_dataset_from_file(filename):
    dataset = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split('\t')
            dataset.append(list(map(float, line)))
        return dataset


class Dataset(object):
    def __init__(self, dataset):
        self.rawDataset = numpy.mat(dataset)

    @property
    def shape(self):
        return self.rawDataset.shape

    @property
    def leaf_val(self):
        """因变量均值"""
        return numpy.mean(self.rawDataset[:, -1])

    @property
    def leaf_error(self):
        """因变量均方差和"""
        m, _n = self.rawDataset.shape
        return m * numpy.var(self.rawDataset[:, -1])

    def split(self, feature, value):
        row_indexs = numpy.nonzero(self.rawDataset[:, feature] > value)[0]
        m0 = self.rawDataset[row_indexs, :]
        row_indexs = numpy.nonzero(self.rawDataset[:, feature] <= value)[0]
        m1 = self.rawDataset[row_indexs, :]
        return Dataset(m0), Dataset(m1)

    def choose_best_split(self, total_s=1.0, total_n=4):
        """

        Parameters
        ----------
        total_s : float
            分裂叶节点时, 数据集方差和下降值最小值
        total_n : int
            叶节点中最少包含的样本数

        Returns
        -------
        (int, float) : 对数据集划分的最好特征的index, 划分值
        """
        # 如果所有值都相等, 生成一个叶节点
        if len(set(self.rawDataset[:, -1].T.A1)) == 1:
            return None, self.leaf_val

        _m, n = self.rawDataset.shape
        best_info = {
            's': numpy.inf,
            'index': 0,
            'value': 0,
        }
        for feature_index in range(n - 1):
            values = set(self.rawDataset[:, feature_index].A1)
            for split_val in values:
                d0, d1 = self.split(feature_index, split_val)
                # 如果切分出来的数据集很小, 跳过?
                if d0.shape[0] < total_n or d1.shape[0] < total_n:
                    continue
                new_s = d0.leaf_error + d1.leaf_error
                if new_s < best_info['s']:
                    best_info['s'] = new_s
                    best_info['index'] = feature_index
                    best_info['value'] = split_val

        # 如果误差减少不大, 则生成一个叶节点
        if self.leaf_error - best_info['s'] < total_s:
            return None, self.leaf_val

        # 如果切分出来的数据集很小, 则生成一个叶节点
        d0, d1 = self.split(best_info['index'], best_info['value'])
        if d0.shape[0] < total_n or d1.shape[0] < total_n:
            return None, self.leaf_val

        return best_info['index'], best_info['value']


class RegressionTree(object):
    def __init__(self, dataset):
        self.dataset = Dataset(dataset)
        self.tree = self.__build_tree(self.dataset)

    @classmethod
    def __build_tree(cls, dataset):
        feature_index, value = dataset.choose_best_split()
        if feature_index is None:
            return value

        d0, d1 = dataset.split(feature_index, value)
        tree = {
            'index': feature_index,
            'value': value,
            'left': cls.__build_tree(d0),
            'right': cls.__build_tree(d1),
        }
        return tree

    @staticmethod
    def is_tree(node):
        return isinstance(node, dict)

    @classmethod
    def mean(cls, tree):
        if cls.is_tree(tree['right']):
            tree['right'] = cls.mean(tree['right'])
        if cls.is_tree(tree['left']):
            tree['left'] = cls.mean(tree['left'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, test_dataset):
        return self.__do_prune(copy.deepcopy(self.tree), Dataset(test_dataset))

    @classmethod
    def __do_prune(cls, tree, test_dataset):
        m, _n = test_dataset.shape
        if m == 0:
            return cls.mean(tree)

        if cls.is_tree(tree['right']) or cls.is_tree(tree['left']):
            d0, d1 = test_dataset.split(tree['index'], tree['value'])
            if cls.is_tree(tree['left']):
                tree['left'] = cls.__do_prune(tree['left'], d0)
            if cls.is_tree(tree['right']):
                tree['right'] = cls.__do_prune(tree['right'], d1)

        if cls.is_tree(tree['left']) or cls.is_tree(tree['right']):
            return tree
        else:
            # 如果两个子节点都已经不是树, 则对子节点尝试合并
            # 比较合并前后的误差, 如果误差能得到提升则进行合并
            d0, d1 = test_dataset.split(tree['index'], tree['value'])
            errorNoMerge = sum(numpy.power(
                d0.rawDataset[:, -1] - tree['left'],
                2
            )) + sum(numpy.power(
                d1.rawDataset[:, -1] - tree['right'],
                2
            ))

            tree_mean = (tree['left'] + tree['right']) / 2.0
            errorMerge = sum(numpy.power(test_dataset.rawDataset[:, -1], 2))

            if errorMerge < errorNoMerge:
                logging.debug('merging...')
                return tree_mean
            else:
                return tree


def main():
    '''
    m = numpy.mat(numpy.eye(4))
    print(binSplitDataset(m, 1, 0.5))
    m = Dataset(m)
    m0, m1 = m.split(1, 0.5)
    print((m0.rawDataset, m1.rawDataset))
    '''
    import pprint

    filename = 'ex00.txt'
    dataset = load_dataset_from_file(filename)
    tree = RegressionTree(dataset)
    logging.info('`{0}` -> 回归树:\n{1}'.format(
        filename,
        pprint.pformat(tree.tree)
    ))

    filename = 'ex0.txt'
    dataset = load_dataset_from_file(filename)
    tree = RegressionTree(dataset)
    logging.info('`{0}` -> 回归树:\n{1}'.format(
        filename,
        pprint.pformat(tree.tree)
    ))

    filename = 'ex2.txt'
    dataset = load_dataset_from_file(filename)
    tree = RegressionTree(dataset)
    logging.info('`{0}` -> 回归树:\n{1}'.format(
        filename,
        pprint.pformat(tree.tree)
    ))
    filename = 'ex2test.txt'
    test_dataset = load_dataset_from_file(filename)
    pruned_tree = tree.prune(test_dataset)
    logging.info('利用`{0}`进行后剪支 -> 回归树:\n{1}'.format(
        filename,
        pprint.pformat(pruned_tree)
    ))

if __name__ == '__main__':
    main()


def linearSolve(dataset):  # helper function used in two places
    m, n = numpy.shape(dataset)
    X = numpy.mat(numpy.ones((m, n)))
    Y = numpy.mat(numpy.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataset[:, 0:n - 1]
    Y = dataset[:, -1]  # and strip out Y
    xTx = X.T * X
    if numpy.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n'
                        'try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataset):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataset)
    return ws


def modelErr(dataset):
    ws, X, Y = linearSolve(dataset)
    yHat = X * ws
    return sum(numpy.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = numpy.shape(inDat)[1]
    X = numpy.mat(numpy.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = numpy.mat(numpy.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, numpy.mat(testData[i]), modelEval)
    return yHat
