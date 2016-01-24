#!/usr/bin/env python
# encoding=utf-8

"""
朴素贝叶斯
===
朴素贝叶斯是贝叶斯决策理论的一部分.
## 贝叶斯决策
* 核心思想 -> 选择具有最高概率的决策

## 朴素贝叶斯分类器
朴素贝叶斯分类器是用于文档分类的常用算法
* 把每个次的出现或者不出现作为一个特征
* 假设特征之间相互独立, 即一个单词出现的可能性和其他相邻单词没有关系
* 每个特征同等重要

"""

from __future__ import print_function

import numpy as np
from numpy import random
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def getFakeDataset():
    posts = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]
    classes = [0, 1, 0, 1, 0, 1]  # 1表示为侮辱性句子, 0为普通句子
    return posts, classes


def getVocabulary(dataset):
    """创建一个包含所有在文档中出现的不重复词的词典

    Parameters
    ----------
    dataset : list of documents

    Returns
    -------
    list : 词典
    """
    vocabulary = set([])
    for document in dataset:
        vocabulary |= set(document)  # 取并集
    return list(vocabulary)


def getSetOfWords2Vec(vocabulary, inputSet):
    """`词集模型`
    词汇表中的单词在输入文档中是否出现

    Parameters
    ----------
    vocabulary : 词典
    inputSet : 文档

    Returns
    -------
    list : 文档向量, 0/1表示词汇表中的单词在输入文档中是否出现
    """
    appearVector = [0]*len(vocabulary)
    for word in inputSet:
        if word in vocabulary:
            appearVector[vocabulary.index(word)] = 1
    return appearVector


def getBagOfWords2Vec(vocabulary, inputSet):
    """`词袋模型`
    词汇表中的单词在输入文档中出现的次数

    Parameters
    ----------
    vocabulary : 词典
    inputSet : 文档

    Returns
    -------
    list : 文档向量, 每一维表示词汇表中的单词在输入文档中出现的次数
    """
    appearCountVector = [0]*len(vocabulary)
    for word in inputSet:
        if word in vocabulary:
            appearCountVector[vocabulary.index(word)] += 1
    return appearCountVector


class NaiveBayesModel(object):
    """ 朴素贝叶斯模型 """

    def __init__(self, matrix, categories):
        self.trainMatrix = np.array(matrix)
        self.trainCategory = np.array(categories)

        # m 为有多少个样例, n 为每个样例的词向量长度
        m, n = self.trainMatrix.shape
        # 样例中为 class1 的概率
        self.pClass1 = 1.0*sum(self.trainCategory) / m
        # 防止概率相乘为 0, 把所有词出现次数初始化为 1, 总词数初始化为 2
        wordsCountVector = {
            'class0': np.ones(n),  # 属于 Class0 的各个词数
            'class1': np.ones(n),  # 属于 Class1 的各个词数
        }
        for rowno in range(m):
            if self.trainCategory[rowno] == 0:
                wordsCountVector['class0'] += self.trainMatrix[rowno]
            else:
                wordsCountVector['class1'] += self.trainMatrix[rowno]
        # 防止太多小的浮点数相乘导致下溢出, 对乘积取自然对数
        self.pWordsVector = {
            'class0': np.log(
                wordsCountVector['class0']
                / (1 + wordsCountVector['class0'].sum())
            ),
            'class1': np.log(
                wordsCountVector['class1']
                / (1 + wordsCountVector['class1'].sum())
            ),
        }

    def predict(self, inputVector):
        inputVector = np.array(inputVector)
        p0 = (
            sum(inputVector * self.pWordsVector['class0'])
            + np.log(1.0 - self.pClass1)
        )
        p1 = (
            sum(inputVector * self.pWordsVector['class1'])
            + np.log(self.pClass1)
        )
        # print('{:.3f}/{:.3f} for Class {}/{}'.format(
        #     np.exp(p0)*100, np.exp(p1)*100, 0, 1
        # ))
        if p1 > p0:
            return 1
        else:
            return 0


def testingNaiveBayes():
    postsToken, postsClass = getFakeDataset()
    vocabulary = getVocabulary(postsToken)
    trainMatrix = [
        getSetOfWords2Vec(vocabulary, post) for post in postsToken
        ]
    model = NaiveBayesModel(trainMatrix, postsClass)

    testEntry = ['love', 'my', 'dalmation']
    testPost = getSetOfWords2Vec(vocabulary, testEntry)
    print(testEntry, 'classified as: ', model.predict(testPost))

    testEntry = ['stupid', 'garbage']
    testPost = getSetOfWords2Vec(vocabulary, testEntry)
    print(testEntry, 'classified as: ', model.predict(testPost))

""" 使用朴素贝叶斯对电子邮件进行分类 """


def getContentTokens(content):
    """ 简单切分英语文本 """
    import re
    tokens = re.split(r'\W*', content)
    return [token.lower() for token in tokens if len(token) > 2]


def testNaiveBayesToSpamEmail():
    """ 使用朴素贝叶斯进行电子邮件分类的测试 """
    emails = []
    emails_class = []

    for i in range(1, 26):
        # 垃圾邮件样本
        words = getContentTokens(open('email/spam/%d.txt' % i).read())
        emails.append(words)
        emails_class.append(1)
        # 正常邮件样本
        words = getContentTokens(open('email/ham/%d.txt' % i).read())
        emails.append(words)
        emails_class.append(0)

    # `留存交叉验证` -- 随机选择数据一部分作为训练集, 剩余部分作为测试集
    # 生成测试集, 训练集
    random_order = random.permutation(50)
    testIndexs, trainIndexs = random_order[:10], random_order[10:]

    # 生成词典
    vocabulary = getVocabulary(emails)
    # 训练朴素贝叶斯分类器
    trainMatrix = []
    trainCategories = []
    for docIndex in trainIndexs:
        trainMatrix.append(
            getBagOfWords2Vec(vocabulary, emails[docIndex])  # 使用词袋模型
        )
        trainCategories.append(emails_class[docIndex])
    logging.info('Train dataset is ready.')
    model = NaiveBayesModel(trainMatrix, trainCategories)
    logging.info('NaiveBayes model is trained.')

    # 进行分类测试
    errorCount = 0
    for docIndex in testIndexs:
        wordVector = getBagOfWords2Vec(vocabulary, emails[docIndex])
        result = model.predict(wordVector)
        if result != emails_class[docIndex]:
            errorCount += 1
            logging.warning('classification error. Predict/Actual: {}/{}\n{}'.format(
                result,
                emails_class[docIndex],
                emails[docIndex]
            ))
    logging.info('the error rate is: {:.2%}'.format(1.0*errorCount/len(testIndexs)))

if __name__ == '__main__':
    testingNaiveBayes()
    testNaiveBayesToSpamEmail()
