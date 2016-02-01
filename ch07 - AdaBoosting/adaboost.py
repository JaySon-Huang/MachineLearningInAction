#!/usr/bin/env python
# encoding=utf-8

"""
AdaBoost -- Adaptive boosting
===
通过串行训练多个分类器, 每一个分类器根据已训练出来的分类器的性能来进行训练,
每个新的分类器集中关注被已有分类器错分的那些数据来获得新的分类器.
最终把所有分类器的结果加权求和.
"""
import logging

import numpy

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1


def load_fake_dataset():
    dataset = numpy.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataset, labels


def load_dataset_from_file(filename):
    dataset = []
    labels = []
    num_features = None
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split('\t')
            if num_features is None:
                num_features = len(line)
            dataset.append(list(map(float, line[:-1])))
            labels.append(float(line[-1]))
        return dataset, labels


class DicisionStump(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def predict(self, dimension, threshold_val, inequal):
        m, _n = self.dataset.shape
        predict = numpy.ones((m, 1))
        if inequal == 'lt':
            predict[self.dataset[:, dimension] <= threshold_val] = -1.0
        elif inequal == 'gt':
            predict[self.dataset[:, dimension] > threshold_val] = -1.0
        return predict


class AdaBoostDicisionStump(object):
    def __init__(self, dataset, labels, max_iter=40):
        self.dataset = numpy.mat(dataset)
        self.labels = numpy.mat(labels).T
        self.m, self.n = self.dataset.shape
        self.train(max_iter=max_iter)

    def build_stump(self, D):
        stump = DicisionStump(self.dataset)
        num_steps = 10.0  # 在特征的可能值上通过递增步长遍历的次数
        best_stump_info = {}  # 记录对于给定权重向量D, 最佳的单层决策树
        best_predict_values = numpy.mat(numpy.zeros((self.m, 1)))
        min_error = 0x3f3f3f3f  # init error sum, to +infinity
        # 遍历所有特征
        for i in range(self.n):
            # 计算遍历该特征的步长
            feature_min = self.dataset[:, i].min()
            feature_max = self.dataset[:, i].max()
            step = (feature_max - feature_min) / num_steps
            # 对于该特征, 遍历所有可能的值
            for j in range(-1, int(num_steps) + 1):  # loop over all range in current dimension
                for inequal in ['lt', 'gt']:  # 在 >/< 之间进行切换
                    threshold_val = feature_min + float(j) * step
                    predicted_values = stump.predict(i, threshold_val, inequal)
                    # 记录预测值与实际分类不同
                    errors = numpy.mat(numpy.ones((self.m, 1)))
                    errors[predicted_values == self.labels] = 0
                    # 计算在给定权重下的总错误权重
                    weighted_errors = D.T * errors
                    logging.log(TRACE, '[Split] dimension {:d}, threshold {:.2f} threshold inequal: {:s}'.format(
                        i, threshold_val, inequal
                    ))
                    logging.log(TRACE, '[Split] Weighted errors is {:.3f}'.format(weighted_errors[0, 0]))
                    # 根据总错误权重来更新最好的单层决策树信息
                    if weighted_errors < min_error:
                        min_error = weighted_errors
                        best_predict_values = predicted_values.copy()
                        best_stump_info['dimension'] = i
                        best_stump_info['threshold'] = threshold_val
                        best_stump_info['inequal'] = inequal
        return best_stump_info, min_error, best_predict_values

    def train(self, max_iter):
        weak_classifiers = []
        D = numpy.mat(numpy.ones((self.m, 1)) / self.m)
        aggregated_predict = numpy.mat(numpy.zeros((self.m, 1)))
        for i in range(max_iter):
            stump_info, error, predict = self.build_stump(D)
            logging.debug('D: {}'.format(D.T))
            # 计算本次单层决策树输出结果的权重, `max(error, 1e-16)` 保证不会出现除0错误
            alpha = float(0.5 * numpy.log((1.0 - error) / max(error, 1e-16)))
            stump_info['alpha'] = alpha
            weak_classifiers.append(stump_info)  # store Stump Params in Array
            logging.debug('predict: {}'.format(predict.T))
            # 更新权重D
            exponent = numpy.multiply(-1 * alpha * self.labels, predict)
            D = numpy.multiply(D, numpy.exp(exponent))
            D = D / D.sum()  # 保证 D 各维度总和为 1
            # 计算应用所有分类器后的分类结果
            aggregated_predict += alpha * predict
            logging.debug('aggregated predict: {}'.format(aggregated_predict.T))
            aggregated_errors = numpy.multiply(
                numpy.sign(aggregated_predict) != self.labels,
                numpy.ones((self.m, 1))
            )
            errorRate = aggregated_errors.sum() / self.m
            logging.info('Total error: {}'.format(errorRate))
            if errorRate == 0.0:
                break
        self.classifiers = weak_classifiers
        self.aggregated_predict = aggregated_predict

    def predict(self, dataset):
        dataset = numpy.mat(dataset)
        stump = DicisionStump(dataset)
        m, _n = dataset.shape
        aggregated_estimate = numpy.mat(numpy.zeros((m, 1)))
        for classifier in self.classifiers:
            logging.info('Applying stumb: {}'.format(classifier))
            estimate = stump.predict(
                classifier['dimension'],
                classifier['threshold'],
                classifier['inequal']
            )
            aggregated_estimate += classifier['alpha'] * estimate
            logging.info(aggregated_estimate)
        return numpy.sign(aggregated_estimate)


def plotROCCurve(predStrengths, labels):
    """
    ROC曲线(Receiver Operating Characteristic curve)
    ROC曲线给出当阈值变化时假阳率和真阳率的变化情况
    """
    import matplotlib.pyplot as plt
    cursor = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # variable to calculate AUC
    numPositiveClass = sum(numpy.array(labels) == 1.0)  # 正例的数目
    step = {
        'x': 1.0 / numPositiveClass,
        'y': 1.0 / (len(labels) - numPositiveClass),
    }
    sortedIndicies = predStrengths.A1.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies:
        if labels[index] == 1.0:
            deltaX = 0
            deltaY = step['x']
        else:
            deltaX = step['y']
            deltaY = 0
            ySum += cursor[1]
        # draw line from cursor to (cursor[0]-deltaX, cursor[1]-deltaY)
        logging.debug('Drawing line from {} -> {}'.format(
            cursor, (cursor[0]-deltaX, cursor[1]-deltaY)
        ))
        ax.plot(
            [cursor[0], cursor[0]-deltaX],
            [cursor[1], cursor[1]-deltaY],
            c='b'
        )
        cursor = (cursor[0] - deltaX, cursor[1] - deltaY)
    ax.plot([0, 1], [0, 1], 'b--')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    logging.info('曲线下面积AUC(Area Under the Curve): {}'.format(ySum * step['y']))


def main():
    import pprint
    dataset, labels = load_fake_dataset()
    # D = numpy.mat(numpy.ones((5, 1)) / 5)
    # build_stump(dataset, labels, D)
    model = AdaBoostDicisionStump(dataset, labels)
    logging.info('Classifiers: {}'.format(pprint.pformat(model.classifiers)))
    logging.info('结果对比 (预测/真实):\n{}'.format(zip(
        model.predict(dataset).A1.tolist(),
        labels
    )))

    plotROCCurve(model.aggregated_predict, labels)

    dataset, labels = load_dataset_from_file('horseColicTraining2.txt')
    model = AdaBoostDicisionStump(dataset, labels)
    plotROCCurve(model.aggregated_predict, labels)


if __name__ == '__main__':
    main()

